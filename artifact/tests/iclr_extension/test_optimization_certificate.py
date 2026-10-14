from __future__ import annotations

import itertools

import numpy as np
import pytest

from marginal_alignment.iclr_extension.optimization_certificate import (
    COSINE_UNCENTERED,
    DIAGONAL_STANDARDIZED,
    FULL_MAHALANOBIS,
    OptimizationCertificateProtocol,
    _linear_assignment,
    _primal_dual_search,
    construct_optimization_certificate,
    fit_certificate_geometry,
    solve_exact_budgeted_disagreement_milp,
    verify_result_against_inputs,
)


def _calibration_and_readout() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(919)
    target = rng.normal(size=(48, 3)) @ np.asarray([
        [1.2, 0.4, 0.1],
        [0.0, 0.7, -0.2],
        [0.0, 0.0, 1.8],
    ])
    reference = rng.normal(size=(51, 3)) @ np.asarray([
        [0.8, -0.3, 0.2],
        [0.0, 1.5, 0.5],
        [0.0, 0.0, 0.6],
    ])
    weight = np.asarray([
        [1.0, 0.0, -0.2],
        [-0.5, 0.8, 0.1],
        [-0.5, -0.8, 0.1],
    ])
    bias = np.asarray([0.1, -0.1, 0.0])
    return target, reference, weight, bias


def _brute_optimum(
    cost: np.ndarray,
    reference_prediction: np.ndarray,
    prediction_zero: np.ndarray,
    allowed_cost: float,
) -> float:
    row = np.arange(len(cost))
    best = 0.0
    for values in itertools.permutations(range(len(cost))):
        assignment = np.asarray(values, dtype=np.int64)
        if float(cost[row, assignment].sum()) <= allowed_cost + 1e-12:
            best = max(
                best,
                float(np.mean(
                    reference_prediction[assignment] != prediction_zero
                )),
            )
    return best


def test_primal_dual_bounds_cover_exact_small_instance_optima() -> None:
    rng = np.random.default_rng(20260903)
    protocol = OptimizationCertificateProtocol(
        cost_geometry=FULL_MAHALANOBIS,
        cost_budget_relative=0.2,
        dual_bisection_iterations=48,
    )
    for _ in range(12):
        cost = rng.uniform(0.1, 3.0, size=(5, 5))
        baseline = _linear_assignment(cost)
        row = np.arange(5)
        baseline_cost = float(cost[row, baseline].sum())
        allowed_cost = baseline_cost * 1.2
        reference_prediction = rng.integers(0, 3, size=5)
        prediction_zero = reference_prediction[baseline]
        primal_assignment, primal, dual, gap, _ = _primal_dual_search(
            cost,
            reference_prediction,
            prediction_zero,
            baseline_cost=baseline_cost,
            allowed_cost=allowed_cost,
            initial_primal=baseline,
            protocol=protocol,
        )
        exact = _brute_optimum(
            cost, reference_prediction, prediction_zero, allowed_cost
        )
        assert float(cost[row, primal_assignment].sum()) <= allowed_cost + 1e-10
        assert primal <= exact + 1e-12
        assert exact <= dual + 1e-10
        assert gap == pytest.approx(dual - primal, abs=1e-12)


def test_dynamic_range_never_relaxes_primal_budget_feasibility() -> None:
    cost = np.asarray([
        [
            2.1333966066910814e-06,
            0.019633748466504754,
            0.18399275644889576,
            49026470.906169794,
            0.00021536188483454338,
        ],
        [
            2.6870609994649826e-10,
            3345545.4590740367,
            1.7255943394964118e-09,
            7.241688724525688e-09,
            1834992674.315035,
        ],
        [
            116763.83823086933,
            0.0005715892542874347,
            4.4012580497017305e-08,
            0.005427645361700925,
            0.0006079919870776038,
        ],
        [
            30819389.308617767,
            0.023717032367338773,
            4.3742149474683036e-09,
            343640266.5031035,
            279.42629626191365,
        ],
        [
            2.158724078147431e-08,
            1.8343542105550266e-08,
            0.20828827501457425,
            6.946712315370064e-09,
            431.5626428918988,
        ],
    ], dtype=np.float64)
    rows = len(cost)
    baseline = np.asarray([0, 3, 4, 2, 1], dtype=np.int64)
    baseline_cost = 0.0006101553431300725
    allowed_cost = 0.0006406631102865761
    reference_prediction = np.asarray([1, 0, 0, 1, 2])
    prediction_zero = np.asarray([1, 1, 2, 0, 0])
    protocol = OptimizationCertificateProtocol(
        cost_geometry=FULL_MAHALANOBIS,
        cost_budget_relative=0.05,
        dual_bisection_iterations=48,
    )
    assignment, primal, dual, gap, diagnostics = _primal_dual_search(
        cost,
        reference_prediction,
        prediction_zero,
        baseline_cost=baseline_cost,
        allowed_cost=allowed_cost,
        initial_primal=baseline,
        protocol=protocol,
    )
    exact = _brute_optimum(
        cost, reference_prediction, prediction_zero, allowed_cost
    )
    assert exact == 0.0
    assert primal == exact
    assert float(cost[np.arange(rows), assignment].sum()) <= allowed_cost
    assert diagnostics["feasibility_tolerance"] == 0.0
    assert primal <= dual
    assert gap == pytest.approx(dual - primal)


def test_exact_milp_matches_enumeration_and_proves_optimality() -> None:
    cost = np.asarray([
        [0.0, 1.0, 3.0, 2.0],
        [1.0, 0.0, 2.0, 3.0],
        [3.0, 2.0, 0.0, 1.0],
        [2.0, 3.0, 1.0, 0.0],
    ])
    reference_prediction = np.asarray([0, 0, 1, 1])
    prediction_zero = reference_prediction.copy()
    allowed_cost = 2.0
    assignment, report = solve_exact_budgeted_disagreement_milp(
        cost,
        reference_prediction,
        prediction_zero,
        allowed_cost=allowed_cost,
        time_limit_seconds=30.0,
    )
    exact = _brute_optimum(
        cost, reference_prediction, prediction_zero, allowed_cost
    )
    assert report["status"] == "PROVEN_OPTIMAL"
    assert report["maximum_disagreement_exact"] == exact
    assert report["reported_mip_gap"] == 0.0
    assert sorted(assignment.tolist()) == list(range(4))


def test_three_geometries_have_explicit_distinct_normalizations() -> None:
    target, reference, weight, bias = _calibration_and_readout()
    normalizations = set()
    transforms = []
    for name in (
        FULL_MAHALANOBIS,
        DIAGONAL_STANDARDIZED,
        COSINE_UNCENTERED,
    ):
        protocol = OptimizationCertificateProtocol(
            cost_geometry=name,
            cost_budget_relative=0.05,
        )
        geometry = fit_certificate_geometry(
            target, reference, weight, bias, protocol=protocol
        )
        assert geometry.diagnostics["cost_geometry"] == name
        assert geometry.diagnostics["discarded_feature_dimensions"] == 0
        normalizations.add(geometry.diagnostics["cost_normalization"])
        transforms.append(geometry.cost_transform)
    assert len(normalizations) == 3
    assert transforms[0] is not None
    assert transforms[1] is not None
    assert transforms[2] is None
    assert not np.allclose(transforms[0], transforms[1])


def test_certificate_serialization_and_full_reconstruction() -> None:
    calibration_target, calibration_reference, weight, bias = (
        _calibration_and_readout()
    )
    protocol = OptimizationCertificateProtocol(
        cost_geometry=COSINE_UNCENTERED,
        cost_budget_relative=0.20,
        maximum_witness_rows=8,
        greedy_neighbors=8,
        dual_bisection_iterations=32,
    )
    geometry = fit_certificate_geometry(
        calibration_target,
        calibration_reference,
        weight,
        bias,
        protocol=protocol,
    )
    rng = np.random.default_rng(77)
    target = rng.normal(size=(8, 3))
    reference = rng.normal(size=(8, 3))
    result = construct_optimization_certificate(geometry, target, reference)
    summary = result.summary()
    assert summary["selected_two_world_minimax_recovery"] == pytest.approx(
        1.0 - summary["selected_world_disagreement"] / 2.0
    )
    assert summary["maximum_disagreement_primal_lower_bound"] <= (
        summary["maximum_disagreement_dual_upper_bound"] + 1e-12
    )
    assert sorted(result.assignment_world_zero.tolist()) == list(range(8))
    assert sorted(result.assignment_world_one.tolist()) == list(range(8))
    verification = verify_result_against_inputs(
        result, geometry, target, reference
    )
    assert verification["status"] == "PASS"
    assert verification["costs_recomputed_from_endpoint_features"] is True


def test_duplicate_witness_rows_preserve_indexed_atoms_and_joint_invariance() -> None:
    calibration_target, calibration_reference, weight, bias = (
        _calibration_and_readout()
    )
    protocol = OptimizationCertificateProtocol(
        cost_geometry=FULL_MAHALANOBIS,
        cost_budget_relative=0.05,
        maximum_witness_rows=4,
    )
    geometry = fit_certificate_geometry(
        calibration_target,
        calibration_reference,
        weight,
        bias,
        protocol=protocol,
    )
    target = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reference = np.asarray([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    with pytest.raises(ValueError, match="unique immutable sample IDs"):
        construct_optimization_certificate(geometry, target, reference)
    target_ids = np.asarray(["target-b", "target-a"])
    reference_ids = np.asarray(["reference-b", "reference-a"])
    first = construct_optimization_certificate(
        geometry,
        target,
        reference,
        target_sample_ids=target_ids,
        reference_sample_ids=reference_ids,
    )
    second = construct_optimization_certificate(
        geometry,
        target[[1, 0]],
        reference[[1, 0]],
        target_sample_ids=target_ids[[1, 0]],
        reference_sample_ids=reference_ids[[1, 0]],
    )
    assert first.selected_world_disagreement == second.selected_world_disagreement
    assert first.maximum_disagreement_primal_lower_bound == (
        second.maximum_disagreement_primal_lower_bound
    )
    assert first.maximum_disagreement_dual_upper_bound == (
        second.maximum_disagreement_dual_upper_bound
    )
    assert np.array_equal(first.assignment_world_zero, second.assignment_world_zero)
    assert np.array_equal(first.assignment_world_one, second.assignment_world_one)
    assert first.diagnostics["target_witness_indexed_atom_sha256"] == (
        second.diagnostics["target_witness_indexed_atom_sha256"]
    )
    assert first.diagnostics["target_witness_indexed_atoms"][
        "exact_duplicate_feature_groups"
    ] == 1
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        construct_optimization_certificate(
            geometry,
            target,
            reference,
            target_sample_ids=np.asarray(["same", "same"]),
            reference_sample_ids=reference_ids,
        )


def test_exact_small_two_world_value_is_not_called_the_global_optimum() -> None:
    calibration_target, calibration_reference, weight, bias = (
        _calibration_and_readout()
    )
    protocol = OptimizationCertificateProtocol(
        cost_geometry=DIAGONAL_STANDARDIZED,
        cost_budget_relative=0.05,
        maximum_witness_rows=6,
        greedy_neighbors=6,
    )
    geometry = fit_certificate_geometry(
        calibration_target,
        calibration_reference,
        weight,
        bias,
        protocol=protocol,
    )
    rng = np.random.default_rng(13)
    result = construct_optimization_certificate(
        geometry,
        rng.normal(size=(6, 3)),
        rng.normal(size=(6, 3)),
    )
    interpretation = result.summary()["scientific_interpretation"]
    assert interpretation["selected_two_world_minimax_value"] == (
        "exact_for_the_two_serialized_assignments"
    )
    assert interpretation["maximum_disagreement_primal"] == (
        "integer_feasible_lower_bound_on_maximum_ambiguity"
    )
