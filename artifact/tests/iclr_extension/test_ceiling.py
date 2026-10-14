from __future__ import annotations

import inspect

import numpy as np

from marginal_alignment.campaign.maps import AffineMap
from marginal_alignment.iclr_extension.ceiling import (
    CorrespondenceCeilingProtocol,
    WitnessGeometry,
    construct_two_world_witness,
    estimate_two_world_ceiling,
    fit_witness_geometry,
)
from marginal_alignment.iclr_extension.frontier import (
    MARGINAL_FRONTIER_METHODS,
    fit_marginal_frontier,
    frozen_frontier_predictions,
)


def _one_dimensional_geometry(*, invariant_readout: bool) -> WitnessGeometry:
    protocol = CorrespondenceCeilingProtocol(
        cost_budget_relative=0.0,
        maximum_witness_rows=8,
        swap_neighbors=8,
    )
    weight = (
        np.zeros((2, 1), dtype=np.float64)
        if invariant_readout
        else np.asarray([[-1.0], [1.0]])
    )
    bias = np.asarray([1.0, 0.0]) if invariant_readout else np.zeros(2)
    return WitnessGeometry(
        bures_map=AffineMap(
            matrix=np.eye(1), offset=np.zeros(1), method="fixture"
        ),
        reference_mean=np.zeros(1),
        reference_inverse_sqrt=np.eye(1),
        readout_weight=weight,
        readout_bias=bias,
        protocol=protocol,
        calibration_diagnostics={"fixture": True},
    )


def test_public_ceiling_inputs_cannot_receive_labels_ids_or_pairs() -> None:
    for function in (
        fit_witness_geometry,
        construct_two_world_witness,
        estimate_two_world_ceiling,
        fit_marginal_frontier,
        frozen_frontier_predictions,
    ):
        names = set(inspect.signature(function).parameters)
        assert not any(
            token in name
            for name in names
            for token in ("label", "sample_id", "pairing", "pair_")
        )


def test_marginal_frontier_is_exact_six_methods_and_label_free() -> None:
    target = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    reference = np.asarray([[-3.0], [-0.5], [0.5], [3.0]])
    frontier = fit_marginal_frontier(target, reference)
    assert tuple(frontier.maps) == MARGINAL_FRONTIER_METHODS
    assert frontier.diagnostics["labels_used"] is False
    predictions = frozen_frontier_predictions(
        frontier,
        target,
        np.asarray([[-1.0], [1.0]]),
        np.zeros(2),
    )
    assert tuple(predictions) == MARGINAL_FRONTIER_METHODS
    assert all(value.shape == (4,) for value in predictions.values())


def test_reversal_world_has_exact_bijections_and_exact_ceiling() -> None:
    geometry = _one_dimensional_geometry(invariant_readout=False)
    # The two target points are observationally identical.  Swapping the two
    # differently predicted reference points therefore costs exactly zero.
    witness = construct_two_world_witness(
        geometry,
        target_features=np.asarray([[0.0], [0.0]]),
        reference_features=np.asarray([[-1.0], [1.0]]),
    )
    assert sorted(witness.assignment_world_zero.tolist()) == [0, 1]
    assert sorted(witness.assignment_world_one.tolist()) == [0, 1]
    assert not np.array_equal(
        witness.assignment_world_zero, witness.assignment_world_one
    )
    assert witness.alternative_cost <= witness.allowed_cost + 1e-12
    assert witness.disagreement == 1.0
    assert witness.prediction_recovery_ceiling == 0.5
    assert witness.diagnostics["labels_used"] is False
    assert witness.diagnostics["pairing_used"] is False


def test_prediction_invariant_world_has_no_artificial_ambiguity() -> None:
    witness = construct_two_world_witness(
        _one_dimensional_geometry(invariant_readout=True),
        target_features=np.asarray([[0.0], [0.0]]),
        reference_features=np.asarray([[-1.0], [1.0]]),
    )
    assert witness.disagreement == 0.0
    assert witness.prediction_recovery_ceiling == 1.0
    assert len(witness.swapped_target_rows) == 0


def test_exact_ties_are_deterministic_and_row_order_invariant() -> None:
    geometry = _one_dimensional_geometry(invariant_readout=False)
    target = np.asarray([[0.0], [0.0]])
    reference = np.asarray([[-1.0], [1.0]])
    first = construct_two_world_witness(geometry, target, reference)
    second = construct_two_world_witness(
        geometry, target[::-1].copy(), reference[::-1].copy()
    )
    assert np.array_equal(
        first.assignment_world_zero, second.assignment_world_zero
    )
    assert np.array_equal(
        first.assignment_world_one, second.assignment_world_one
    )
    assert first.as_dict() == second.as_dict()


def test_fitted_geometry_is_unpaired_and_cost_budget_is_enforced() -> None:
    rng = np.random.default_rng(91)
    target_calibration = rng.normal(size=(24, 3)) @ np.diag([1.8, 0.7, 0.3])
    reference_calibration = rng.normal(size=(27, 3)) @ np.diag([0.6, 1.2, 2.0])
    target = rng.normal(size=(8, 3))
    reference = rng.normal(size=(8, 3))
    protocol = CorrespondenceCeilingProtocol(
        cost_budget_relative=0.10,
        maximum_witness_rows=8,
        swap_neighbors=8,
    )
    fitted = fit_witness_geometry(
        target_calibration,
        reference_calibration,
        np.asarray([[1.0, 0.0, -0.5], [-1.0, 0.2, 0.5]]),
        protocol=protocol,
    )
    assert fitted.calibration_diagnostics["labels_used"] is False
    assert fitted.calibration_diagnostics["sample_identifiers_used"] is False
    assert fitted.calibration_diagnostics["pairing_used"] is False
    assert fitted.calibration_diagnostics["discarded_feature_dimensions"] == 0
    assert all(
        len(fitted.calibration_diagnostics[key]) == 64
        for key in (
            "bures_matrix_sha256",
            "bures_offset_sha256",
            "reference_mean_sha256",
            "reference_inverse_sqrt_sha256",
            "full_geometry_sha256",
        )
    )
    witness = construct_two_world_witness(fitted, target, reference)
    assert sorted(witness.assignment_world_zero.tolist()) == list(range(8))
    assert sorted(witness.assignment_world_one.tolist()) == list(range(8))
    tolerance = witness.diagnostics["cost_budget_absolute_tolerance"]
    assert witness.alternative_cost <= witness.allowed_cost + tolerance
    assert witness.prediction_recovery_ceiling == 1.0 - witness.disagreement / 2.0
    assert witness.as_dict()["not_a_realized_world_performance_bound"] is True
    assert witness.diagnostics["swap_neighbors"] == 8


def test_full_rank_cost_cannot_discard_a_classifier_relevant_direction() -> None:
    """A classifier-changing direction must have positive locality cost."""

    geometry = WitnessGeometry(
        bures_map=AffineMap(np.eye(2), np.zeros(2), "fixture"),
        reference_mean=np.zeros(2),
        reference_inverse_sqrt=np.eye(2),
        readout_weight=np.asarray([[0.0, -1.0], [0.0, 1.0]]),
        readout_bias=np.zeros(2),
        protocol=CorrespondenceCeilingProtocol(
            cost_budget_relative=0.0,
            maximum_witness_rows=4,
            swap_neighbors=2,
        ),
        calibration_diagnostics={"fixture": True},
    )
    endpoints = np.asarray([[0.0, -10.0], [0.0, 10.0]])
    witness = construct_two_world_witness(geometry, endpoints, endpoints)
    assert witness.baseline_cost == 0.0
    assert witness.alternative_cost == 0.0
    assert witness.disagreement == 0.0
    assert witness.diagnostics["discarded_feature_dimensions"] == 0


def test_assignment_does_not_claim_unimplemented_lexicographic_ties() -> None:
    geometry = WitnessGeometry(
        bures_map=AffineMap(np.eye(2), np.zeros(2), "fixture"),
        reference_mean=np.zeros(2),
        reference_inverse_sqrt=np.eye(2),
        readout_weight=np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        readout_bias=np.zeros(2),
        protocol=CorrespondenceCeilingProtocol(
            maximum_witness_rows=4, swap_neighbors=4
        ),
        calibration_diagnostics={"fixture": True},
    )
    target = np.asarray([[2.0, -4.0], [4.0, -1.0], [-1.0, 3.0], [0.0, -1.0]])
    reference = np.asarray([[1.0, 3.0], [4.0, -4.0], [4.0, 0.0], [4.0, -5.0]])
    witness = construct_two_world_witness(geometry, target, reference)
    assert witness.diagnostics["assignment"]["tie_break"] == (
        "pinned_scipy_on_content_canonical_order"
    )
