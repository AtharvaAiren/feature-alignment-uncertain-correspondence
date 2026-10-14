from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError, seal_manifest
from marginal_alignment.campaign.maps import AffineMap
from marginal_alignment.iclr_extension.optimization_certificate import (
    FULL_MAHALANOBIS,
    CertificateGeometry,
    OptimizationCertificateProtocol,
    PreparedCertificateProblem,
    _linear_assignment,
    construct_optimization_certificate_from_problem,
    solve_exact_budgeted_disagreement_milp,
)
from marginal_alignment.iclr_extension.pair_structure import (
    random_cost_moments_from_cost_matrix,
)
from marginal_alignment.iclr_extension.witness_scaling import (
    WitnessSizeProtocol,
    build_nested_natural_selection,
    build_nested_pair_selection,
    class_coverage,
    construct_scaling_certificate_from_problem,
    cross_endpoint_class_coverage,
    materialize_prefix,
    uniform_random_cost_moments_blockwise,
    validate_nested_selection,
    validate_nested_selection_manifest,
)


def _pair_inputs(rows: int = 28) -> dict:
    reference = np.asarray([f"clean-{index:03d}" for index in range(rows)])
    target = np.asarray([f"corrupt-{index:03d}" for index in range(rows)])
    return {
        "reference_ids": reference,
        "target_ids": target,
        "paired_reference_ids": reference,
        "paired_target_ids": target,
        "head_fit_reference_ids": reference[:2],
    }


def _protocol() -> WitnessSizeProtocol:
    return WitnessSizeProtocol(calibration_rows=4, witness_sizes=(2, 4, 6))


def test_pair_selection_uses_one_maximum_disjoint_block_and_literal_prefixes() -> None:
    selected = build_nested_pair_selection(
        **_pair_inputs(),
        seed=201,
        selection_salt="nested-unit-test",
        protocol=_protocol(),
    )
    smallest = selected.prefix(2)
    middle = selected.prefix(4)
    largest = selected.prefix(6)

    assert np.array_equal(
        smallest.reference_calibration_rows, largest.reference_calibration_rows
    )
    assert np.array_equal(
        smallest.target_calibration_rows, largest.target_calibration_rows
    )
    assert np.array_equal(
        smallest.reference_witness_rows, largest.reference_witness_rows[:2]
    )
    assert np.array_equal(middle.target_witness_rows, largest.target_witness_rows[:4])
    reference_to_target = dict(
        zip(
            _pair_inputs()["paired_reference_ids"],
            _pair_inputs()["paired_target_ids"],
        )
    )
    paired_targets = {
        reference_to_target[value] for value in largest.reference_ids.tolist()
    }
    assert paired_targets.isdisjoint(set(largest.target_ids.tolist()))
    assert selected.manifest["parent_selection"]["eligible_pair_count"] == 26
    assert (
        selected.manifest["parent_selection"]["required_disjoint_identity_rows"] == 20
    )
    validate_nested_selection_manifest(selected.manifest)


def test_pair_selection_fails_when_maximum_size_exceeds_identity_pool() -> None:
    with pytest.raises(IntegrityError, match="needs 30 eligible identities"):
        build_nested_pair_selection(
            **_pair_inputs(rows=28),
            seed=201,
            selection_salt="nested-unit-test",
            protocol=WitnessSizeProtocol(calibration_rows=4, witness_sizes=(2, 4, 11)),
        )


def test_natural_selection_is_nested_and_materializes_locked_rows() -> None:
    rows = 24
    reference = np.arange(rows * 3, dtype=np.float64).reshape(rows, 3)
    target = reference[::-1] + 0.25
    reference_ids = np.arange(100, 100 + rows, dtype=np.int64)
    target_ids = np.arange(200, 200 + rows, dtype=np.int64)
    selected = build_nested_natural_selection(
        reference_features=reference,
        target_features=target,
        reference_ids=reference_ids,
        target_ids=target_ids,
        dataset="imagenet_sketch",
        seed=201,
        protocol=_protocol(),
    )
    small = selected.prefix(2)
    large = selected.prefix(6)
    assert np.array_equal(
        small.reference_witness_rows, large.reference_witness_rows[:2]
    )
    materialized = materialize_prefix(
        selected,
        reference_features=reference,
        target_features=target,
        witness_rows=4,
    )
    assert np.array_equal(
        materialized["reference_features"],
        reference[materialized["reference_rows"]],
    )
    assert materialized["target_features"].shape == (8, 3)


def test_nested_manifest_rejects_resealed_protocol_tampering() -> None:
    selected = build_nested_pair_selection(
        **_pair_inputs(),
        seed=201,
        selection_salt="nested-unit-test",
        protocol=_protocol(),
    )
    tampered = copy.deepcopy(dict(selected.manifest))
    tampered.pop("manifest_sha256")
    tampered["witness_endpoints_are_literal_prefixes_of_one_maximum_selection"] = False
    with pytest.raises(IntegrityError, match="leakage contract"):
        validate_nested_selection_manifest(seal_manifest(tampered))


def test_materialization_rejects_live_array_mutation_against_sealed_hashes() -> None:
    selected = build_nested_pair_selection(
        **_pair_inputs(),
        seed=201,
        selection_salt="nested-unit-test",
        protocol=_protocol(),
    )
    changed_rows = np.array(selected.reference_rows, copy=True)
    changed_rows[0], changed_rows[1] = changed_rows[1], changed_rows[0]
    tampered = replace(selected, reference_rows=changed_rows)
    with pytest.raises(IntegrityError, match="sealed prefix hashes"):
        validate_nested_selection(tampered)
    features = np.arange(28 * 2, dtype=np.float64).reshape(28, 2)
    with pytest.raises(IntegrityError, match="sealed prefix hashes"):
        materialize_prefix(
            tampered,
            reference_features=features,
            target_features=features,
            witness_rows=2,
        )


def test_blockwise_random_moments_match_dense_reference() -> None:
    rng = np.random.default_rng(7)
    cost = rng.uniform(0.0, 5.0, size=(19, 19))
    expected = random_cost_moments_from_cost_matrix(cost)
    observed = uniform_random_cost_moments_blockwise(cost, block_rows=4)
    for key in (
        "expected_total_cost",
        "variance_total_cost",
        "standard_deviation_total_cost",
        "coefficient_of_variation",
    ):
        assert observed[key] == pytest.approx(expected[key], rel=2e-14, abs=2e-14)
    assert observed["evaluation"]["additional_dense_n_by_n_copy_created"] is False


def test_class_coverage_is_postfreeze_descriptive_summary() -> None:
    summary = class_coverage(np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64))
    assert summary["rows"] == 6
    assert summary["unique_classes"] == 3
    assert summary["singleton_classes"] == 1
    assert summary["classes_with_at_least_two_rows"] == 2
    assert summary["rows_per_observed_class"]["maximum"] == 3


def test_cross_endpoint_coverage_reports_matching_mass_not_only_pair_count() -> None:
    observed = cross_endpoint_class_coverage(
        np.asarray([0, 0, 1, 2], dtype=np.int64),
        np.asarray([0, 1, 1, 3], dtype=np.int64),
    )
    assert observed["target_rows_with_at_least_one_reference_same_class_partner"] == 3
    assert (
        observed["target_fraction_with_at_least_one_reference_same_class_partner"]
        == 0.75
    )
    assert observed["same_class_candidate_pairs"] == 4
    assert observed["maximum_class_preserving_matching_rows"] == 2
    assert observed["maximum_class_preserving_matching_fraction"] == 0.5
    assert observed["minimum_rows_forced_to_cross_class_in_any_bijection"] == 2
    assert observed["minimum_fraction_forced_to_cross_class_in_any_bijection"] == 0.5


def _random_assignment_problem(seed: int, *, gap_target: float):
    rng = np.random.default_rng(seed)
    rows = 10
    cost = rng.uniform(0.8, 2.0, size=(rows, rows))
    cost[np.arange(rows), np.arange(rows)] = rng.uniform(0.02, 0.12, rows)
    assignment_zero = _linear_assignment(cost)
    reference_prediction = np.asarray([0, 1, 0, 1, 2, 0, 2, 1, 2, 0], dtype=np.int64)
    prediction_zero = reference_prediction[assignment_zero]
    protocol = OptimizationCertificateProtocol(
        cost_geometry=FULL_MAHALANOBIS,
        cost_budget_relative=5.0,
        covariance_estimator="oas",
        maximum_witness_rows=rows,
        greedy_neighbors=rows,
        dual_bisection_iterations=48,
        primal_dual_gap_target=gap_target,
    )
    geometry = CertificateGeometry(
        bures_map=AffineMap(np.eye(2), np.zeros(2), "test"),
        reference_mean=np.zeros(2),
        reference_covariance=np.eye(2),
        cost_transform=np.eye(2),
        readout_weight=np.eye(2),
        readout_bias=np.zeros(2),
        protocol=protocol,
        diagnostics={},
    )
    identifiers = {"sample_identifiers_supplied": False}
    return (
        PreparedCertificateProblem(
            geometry=geometry,
            cost=cost,
            assignment_world_zero=assignment_zero,
            reference_prediction=reference_prediction,
            prediction_world_zero=prediction_zero,
            target_canonical_order=np.arange(rows, dtype=np.int64),
            reference_canonical_order=np.arange(rows, dtype=np.int64),
            baseline_cost=float(cost[np.arange(rows), assignment_zero].sum()),
            target_indexed_atom_sha256="target-test",
            reference_indexed_atom_sha256="reference-test",
            target_indexed_atom_diagnostics=identifiers,
            reference_indexed_atom_diagnostics=identifiers,
        ),
        protocol,
    )


@pytest.mark.parametrize("seed", [3, 11, 23])
def test_no_greedy_solver_and_historical_solver_both_contain_exact_milp(
    seed: int,
) -> None:
    problem, protocol = _random_assignment_problem(seed, gap_target=0.30)
    fresh = construct_scaling_certificate_from_problem(problem, protocol=protocol)
    historical = construct_optimization_certificate_from_problem(
        problem, protocol=protocol
    )
    _, exact = solve_exact_budgeted_disagreement_milp(
        problem.cost,
        problem.reference_prediction,
        problem.prediction_world_zero,
        allowed_cost=problem.baseline_cost * (1.0 + protocol.cost_budget_relative),
        time_limit_seconds=30.0,
    )
    optimum = exact["maximum_disagreement_exact"]
    for result in (fresh, historical):
        assert result.maximum_disagreement_primal_lower_bound <= optimum + 1e-10
        assert optimum <= result.maximum_disagreement_dual_upper_bound + 1e-10
        assert result.primal_cost <= result.allowed_cost
    assert fresh.diagnostics["python_greedy_pair_set_constructed"] is False


def test_early_stop_preserves_exact_containment_before_iteration_ceiling() -> None:
    problem, protocol = _random_assignment_problem(11, gap_target=0.01)
    fresh = construct_scaling_certificate_from_problem(problem, protocol=protocol)
    _, exact = solve_exact_budgeted_disagreement_milp(
        problem.cost,
        problem.reference_prediction,
        problem.prediction_world_zero,
        allowed_cost=problem.baseline_cost * (1.0 + protocol.cost_budget_relative),
        time_limit_seconds=30.0,
    )
    optimum = exact["maximum_disagreement_exact"]
    diagnostics = fresh.diagnostics["optimization"]
    assert diagnostics["early_stop_reached"] is True
    assert diagnostics["evaluations"] < protocol.dual_bisection_iterations + 2
    assert fresh.maximum_disagreement_primal_lower_bound <= optimum + 1e-10
    assert optimum <= fresh.maximum_disagreement_dual_upper_bound + 1e-10
