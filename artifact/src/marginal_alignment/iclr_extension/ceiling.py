"""Finite-batch correspondence ceilings for genuinely unpaired endpoints.

The routines in this module are deliberately separate from the locked campaign.
They construct two hidden correspondence worlds from exactly the same observed
reference and target feature multisets.  The only task information available to
the construction is a frozen linear readout.  In particular, the public API has
no arguments for labels, sample identifiers, or pairing tables.

The returned value is the equal-prior randomized-minimax ceiling on recovering
the frozen readout's *reference prediction* across the two constructed
finite-batch worlds.  It is not a ceiling on external target-label accuracy, a
realized-world performance bound, or a population claim about every transport
compatible with a natural shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

from marginal_alignment.campaign.maps import (
    AffineMap,
    estimate_covariance,
    fit_covariance_gauge,
)
from marginal_alignment.ceiling import task_disagreement, two_world_ceiling
from marginal_alignment.linalg import as_samples, symmetric_power


SCHEMA_VERSION = "marginal_alignment.iclr_extension.correspondence_ceiling.v2"
_FLOAT_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class CorrespondenceCeilingProtocol:
    """Prespecified numerical choices for a two-world witness."""

    cost_budget_relative: float = 0.05
    covariance_estimator: str | float = "oas"
    covariance_relative_floor: float = 1e-6
    maximum_witness_rows: int = 2048
    swap_neighbors: int = 64
    swap_neighbor_sensitivity: tuple[int, ...] = ()
    assignment_tie_relative_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost_budget_relative) or self.cost_budget_relative < 0.0:
            raise ValueError("cost_budget_relative must be finite and nonnegative")
        if not np.isfinite(self.covariance_relative_floor) or self.covariance_relative_floor <= 0.0:
            raise ValueError("covariance_relative_floor must be finite and positive")
        if int(self.maximum_witness_rows) < 2:
            raise ValueError("maximum_witness_rows must be at least two")
        if int(self.swap_neighbors) < 1:
            raise ValueError("swap_neighbors must be positive")
        sensitivity = tuple(int(value) for value in self.swap_neighbor_sensitivity)
        if (
            any(value < 1 for value in sensitivity)
            or len(set(sensitivity)) != len(sensitivity)
            or int(self.swap_neighbors) in sensitivity
        ):
            raise ValueError(
                "swap_neighbor_sensitivity must be unique, positive, and exclude the primary"
            )
        if (
            not np.isfinite(self.assignment_tie_relative_tolerance)
            or self.assignment_tie_relative_tolerance < 0.0
        ):
            raise ValueError(
                "assignment_tie_relative_tolerance must be finite and nonnegative"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost_geometry": (
                "full_rank_reference_oas_mahalanobis_after_bures"
            ),
            "cost_normalization": (
                "squared_distance_divided_by_full_feature_dimension"
            ),
            "discarded_feature_dimensions": 0,
            "cost_budget_relative": float(self.cost_budget_relative),
            "covariance_estimator": self.covariance_estimator,
            "covariance_relative_floor": float(self.covariance_relative_floor),
            "maximum_witness_rows": int(self.maximum_witness_rows),
            "swap_neighbors": int(self.swap_neighbors),
            "swap_neighbor_sensitivity": [
                int(value) for value in self.swap_neighbor_sensitivity
            ],
            "assignment_tie_relative_tolerance": float(
                self.assignment_tie_relative_tolerance
            ),
        }


@dataclass(frozen=True)
class WitnessGeometry:
    """Calibration-only geometry frozen before constructing either world."""

    bures_map: AffineMap
    reference_mean: np.ndarray
    reference_inverse_sqrt: np.ndarray
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    protocol: CorrespondenceCeilingProtocol
    calibration_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TwoWorldCeilingWitness:
    """Two exact bijections and their fixed-readout prediction ceiling."""

    assignment_world_zero: np.ndarray
    assignment_world_one: np.ndarray
    prediction_world_zero: np.ndarray
    prediction_world_one: np.ndarray
    swapped_target_rows: np.ndarray
    disagreement: float
    prediction_recovery_ceiling: float
    baseline_cost: float
    alternative_cost: float
    allowed_cost: float
    diagnostics: dict[str, Any]

    def as_dict(self, *, include_assignments: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "interpretation": (
                "equal-prior randomized-minimax frozen-reference-prediction "
                "recovery ceiling across the two constructed cost-constrained "
                "finite-batch correspondence worlds; conservative witness "
                "derived because the alternative-world search is greedy"
            ),
            "not_an_accuracy_ceiling": True,
            "not_a_realized_world_performance_bound": True,
            "disagreement": float(self.disagreement),
            "prediction_recovery_ceiling": float(self.prediction_recovery_ceiling),
            "baseline_cost": float(self.baseline_cost),
            "alternative_cost": float(self.alternative_cost),
            "allowed_cost": float(self.allowed_cost),
            "swapped_rows": int(len(self.swapped_target_rows)),
            "diagnostics": dict(self.diagnostics),
        }
        if include_assignments:
            result.update({
                "assignment_world_zero": self.assignment_world_zero.tolist(),
                "assignment_world_one": self.assignment_world_one.tolist(),
                "prediction_world_zero": self.prediction_world_zero.tolist(),
                "prediction_world_one": self.prediction_world_one.tolist(),
                "swapped_target_rows": self.swapped_target_rows.tolist(),
            })
        return result


def _canonical_rows(
    features: np.ndarray, *, name: str, salt: bytes,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return a content-canonical row order independent of endpoint row order."""

    value = as_samples(features, name=name)
    canonical = np.ascontiguousarray(value, dtype="<f8")
    keyed: list[tuple[bytes, bytes, int]] = []
    for index, row in enumerate(canonical):
        raw = row.tobytes()
        keyed.append((hashlib.sha256(salt + raw).digest(), raw, index))
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    order = np.asarray([item[2] for item in keyed], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(salt)
    digest.update(str(canonical.shape).encode("ascii"))
    for row_hash, raw, _ in keyed:
        digest.update(row_hash)
        digest.update(raw)
    return canonical[order], order, digest.hexdigest()


def _validate_readout(
    readout_weight: np.ndarray,
    readout_bias: np.ndarray | None,
    *,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(readout_weight, dtype=np.float64)
    if (
        weight.ndim != 2
        or weight.shape[0] < 2
        or weight.shape[1] != dimension
        or not np.all(np.isfinite(weight))
    ):
        raise ValueError(
            "readout_weight must be finite with shape "
            "[at_least_two_classes,feature_dimension]"
        )
    if readout_bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float64)
    else:
        bias = np.asarray(readout_bias, dtype=np.float64)
        if bias.shape != (weight.shape[0],) or not np.all(np.isfinite(bias)):
            raise ValueError("readout_bias must have one finite value per class")
    return np.array(weight, copy=True), np.array(bias, copy=True)


def fit_witness_geometry(
    target_calibration_features: np.ndarray,
    reference_calibration_features: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray | None = None,
    *,
    protocol: CorrespondenceCeilingProtocol | None = None,
) -> WitnessGeometry:
    """Fit Bures and full-rank reference-Mahalanobis geometry.

    There are intentionally no label, identifier, or pair arguments.  Target
    and reference rows are content-canonicalized independently before fitting,
    preventing a shared input order from becoming an accidental pairing cue.
    """

    declared = CorrespondenceCeilingProtocol() if protocol is None else protocol
    target, _, target_digest = _canonical_rows(
        target_calibration_features,
        name="target_calibration_features",
        salt=b"play3-target-calibration-v1",
    )
    reference, _, reference_digest = _canonical_rows(
        reference_calibration_features,
        name="reference_calibration_features",
        salt=b"play3-reference-calibration-v1",
    )
    if target.shape[1] != reference.shape[1]:
        raise ValueError("target and reference calibration dimensions differ")
    if min(len(target), len(reference)) < 2:
        raise ValueError("each calibration marginal needs at least two rows")
    weight, bias = _validate_readout(
        readout_weight, readout_bias, dimension=reference.shape[1]
    )
    fitted = fit_covariance_gauge(
        target,
        reference,
        gauge="bures",
        estimator=declared.covariance_estimator,
        relative_floor=float(declared.covariance_relative_floor),
    )

    reference_mean, reference_covariance, reference_covariance_diagnostics = (
        estimate_covariance(
            reference,
            estimator=declared.covariance_estimator,
            relative_floor=float(declared.covariance_relative_floor),
        )
    )
    reference_inverse_sqrt = symmetric_power(reference_covariance, -0.5)
    identity = np.eye(reference.shape[1], dtype=np.float64)
    whitening_error = float(
        np.linalg.norm(
            reference_inverse_sqrt
            @ reference_covariance
            @ reference_inverse_sqrt
            - identity,
            ord="fro",
        )
        / max(float(np.linalg.norm(identity, ord="fro")), _FLOAT_EPSILON)
    )
    if not np.isfinite(whitening_error) or whitening_error > 1e-8:
        raise np.linalg.LinAlgError(
            "full-rank reference covariance whitening is numerically invalid"
        )
    return WitnessGeometry(
        bures_map=fitted,
        reference_mean=np.array(reference_mean, copy=True),
        reference_inverse_sqrt=np.array(reference_inverse_sqrt, copy=True),
        readout_weight=weight,
        readout_bias=bias,
        protocol=declared,
        calibration_diagnostics={
            "schema_version": SCHEMA_VERSION,
            "information_scope": "unpaired_features_and_frozen_readout",
            "labels_used": False,
            "sample_identifiers_used": False,
            "pairing_used": False,
            "target_calibration_rows": int(len(target)),
            "reference_calibration_rows": int(len(reference)),
            "feature_dimension": int(reference.shape[1]),
            "classes": int(weight.shape[0]),
            "target_calibration_content_sha256": target_digest,
            "reference_calibration_content_sha256": reference_digest,
            "cost_geometry": (
                "full_rank_reference_oas_mahalanobis_after_bures"
            ),
            "cost_dimension": int(reference.shape[1]),
            "discarded_feature_dimensions": 0,
            "distance_normalization": (
                "squared_distance_divided_by_full_feature_dimension"
            ),
            "reference_centering": "calibration_reference_mean",
            "reference_covariance": reference_covariance_diagnostics,
            "reference_whitening_relative_error": whitening_error,
            "bures_matrix_sha256": _array_sha256(fitted.matrix),
            "bures_offset_sha256": _array_sha256(fitted.offset),
            "reference_mean_sha256": _array_sha256(reference_mean),
            "reference_inverse_sqrt_sha256": _array_sha256(
                reference_inverse_sqrt
            ),
            "full_geometry_sha256": hashlib.sha256(
                b"".join([
                    bytes.fromhex(_array_sha256(fitted.matrix)),
                    bytes.fromhex(_array_sha256(fitted.offset)),
                    bytes.fromhex(_array_sha256(reference_mean)),
                    bytes.fromhex(_array_sha256(reference_inverse_sqrt)),
                ])
            ).hexdigest(),
            "bures": dict(fitted.diagnostics),
            "protocol": declared.as_dict(),
        },
    )


def _squared_distance_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(np.square(left), axis=1)[:, None]
    right_norm = np.sum(np.square(right), axis=1)[None, :]
    cost = left_norm + right_norm - 2.0 * (left @ right.T)
    return np.maximum(cost / left.shape[1], 0.0)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _deterministic_linear_assignment(
    cost: np.ndarray, *, relative_tolerance: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - declared full dependency
        raise RuntimeError("the correspondence witness requires scipy") from error
    rows, columns = cost.shape
    if rows != columns or rows < 2 or not np.all(np.isfinite(cost)):
        raise ValueError("assignment cost must be a finite square matrix")
    scale = max(float(np.max(np.abs(cost))), 1.0)
    tolerance = float(relative_tolerance * scale)
    # Rows and columns have already been sorted by content.  We pin SciPy and
    # pass this canonical content order without perturbing scientific costs.
    # This is deterministic under the locked runtime; it is deliberately not
    # described as the lexicographically smallest optimal assignment.
    row_indices, column_indices = linear_sum_assignment(cost)
    if not np.array_equal(row_indices, np.arange(rows)):
        raise RuntimeError("linear assignment did not return every target row")
    return np.asarray(column_indices, dtype=np.int64), {
        "algorithm": "scipy_linear_sum_assignment",
        "tie_break": "pinned_scipy_on_content_canonical_order",
        "tie_relative_tolerance": float(relative_tolerance),
        "tie_absolute_tolerance": tolerance,
    }


def _candidate_swap_pairs(
    cost: np.ndarray, assignment: np.ndarray, *, neighbors: int,
) -> list[tuple[int, int]]:
    rows = len(assignment)
    inverse = np.empty(rows, dtype=np.int64)
    inverse[assignment] = np.arange(rows, dtype=np.int64)
    pairs: set[tuple[int, int]] = set()
    width = min(int(neighbors), rows)
    columns = np.arange(rows, dtype=np.int64)
    for target_row in range(rows):
        # lexsort makes equal-cost neighbor selection independent of quickselect
        # implementation details.
        nearest = np.lexsort((columns, cost[target_row]))[:width]
        for reference_column in nearest:
            other = int(inverse[int(reference_column)])
            if other != target_row:
                pairs.add(tuple(sorted((target_row, other))))
    return sorted(pairs)


def construct_two_world_witness(
    geometry: WitnessGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
) -> TwoWorldCeilingWitness:
    """Construct two cost-constrained bijections on held-out endpoint sets."""

    target, _, target_digest = _canonical_rows(
        target_features,
        name="target_features",
        salt=b"play3-target-witness-v1",
    )
    reference, _, reference_digest = _canonical_rows(
        reference_features,
        name="reference_features",
        salt=b"play3-reference-witness-v1",
    )
    if target.shape != reference.shape:
        raise ValueError(
            "the finite correspondence witness requires equal endpoint shapes"
        )
    rows, dimension = target.shape
    if rows > int(geometry.protocol.maximum_witness_rows):
        raise ValueError(
            f"witness has {rows} rows, exceeding the declared dense-assignment "
            f"limit {geometry.protocol.maximum_witness_rows}"
        )
    if geometry.reference_mean.shape != (dimension,):
        raise ValueError("witness feature dimension differs from fitted geometry")
    if geometry.reference_inverse_sqrt.shape != (dimension, dimension):
        raise ValueError("reference whitener differs from fitted feature dimension")
    mapped_target = geometry.bures_map.apply(target)
    target_whitened = (
        mapped_target - geometry.reference_mean
    ) @ geometry.reference_inverse_sqrt
    reference_whitened = (
        reference - geometry.reference_mean
    ) @ geometry.reference_inverse_sqrt
    cost = _squared_distance_cost(target_whitened, reference_whitened)
    assignment_zero, assignment_diagnostics = _deterministic_linear_assignment(
        cost,
        relative_tolerance=float(
            geometry.protocol.assignment_tie_relative_tolerance
        ),
    )
    if len(np.unique(assignment_zero)) != rows:
        raise RuntimeError("baseline assignment is not a bijection")

    logits = reference @ geometry.readout_weight.T + geometry.readout_bias
    reference_prediction = np.argmax(logits, axis=1).astype(np.int64)
    prediction_zero = reference_prediction[assignment_zero]
    baseline_cost = float(cost[np.arange(rows), assignment_zero].sum())
    allowed_cost = float(
        baseline_cost * (1.0 + geometry.protocol.cost_budget_relative)
    )
    absolute_budget_tolerance = float(
        256.0 * rows * _FLOAT_EPSILON * max(allowed_cost, 1.0)
    )

    candidates: list[tuple[float, int, int]] = []
    for first, second in _candidate_swap_pairs(
        cost,
        assignment_zero,
        neighbors=int(geometry.protocol.swap_neighbors),
    ):
        if prediction_zero[first] == prediction_zero[second]:
            continue
        first_reference = int(assignment_zero[first])
        second_reference = int(assignment_zero[second])
        increment = float(
            cost[first, second_reference]
            + cost[second, first_reference]
            - cost[first, first_reference]
            - cost[second, second_reference]
        )
        if increment < -absolute_budget_tolerance:
            raise RuntimeError(
                "a swap improved the declared baseline assignment beyond numerical tolerance"
            )
        candidates.append((max(increment, 0.0), first, second))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    assignment_one = np.array(assignment_zero, copy=True)
    used = np.zeros(rows, dtype=bool)
    running_cost = baseline_cost
    accepted = 0
    for increment, first, second in candidates:
        if used[first] or used[second]:
            continue
        if running_cost + increment > allowed_cost + absolute_budget_tolerance:
            continue
        assignment_one[first], assignment_one[second] = (
            assignment_one[second], assignment_one[first]
        )
        used[first] = True
        used[second] = True
        running_cost += increment
        accepted += 1
    if len(np.unique(assignment_one)) != rows:
        raise RuntimeError("alternative assignment is not a bijection")
    alternative_cost = float(cost[np.arange(rows), assignment_one].sum())
    if alternative_cost > allowed_cost + absolute_budget_tolerance:
        raise RuntimeError("alternative assignment violates the cost budget")

    prediction_one = reference_prediction[assignment_one]
    disagreement = task_disagreement(prediction_zero, prediction_one)
    ceiling = two_world_ceiling(disagreement)
    swapped_rows = np.flatnonzero(assignment_zero != assignment_one).astype(np.int64)
    return TwoWorldCeilingWitness(
        assignment_world_zero=assignment_zero,
        assignment_world_one=assignment_one,
        prediction_world_zero=prediction_zero,
        prediction_world_one=prediction_one,
        swapped_target_rows=swapped_rows,
        disagreement=disagreement,
        prediction_recovery_ceiling=ceiling,
        baseline_cost=baseline_cost,
        alternative_cost=alternative_cost,
        allowed_cost=allowed_cost,
        diagnostics={
            "schema_version": SCHEMA_VERSION,
            "information_scope": "unpaired_features_and_frozen_readout",
            "labels_used": False,
            "sample_identifiers_used": False,
            "pairing_used": False,
            "witness_rows": int(rows),
            "feature_dimension": int(dimension),
            "cost_geometry": (
                "full_rank_reference_oas_mahalanobis_after_bures"
            ),
            "cost_dimension": int(dimension),
            "discarded_feature_dimensions": 0,
            "target_content_sha256": target_digest,
            "reference_content_sha256": reference_digest,
            "candidate_swap_pairs": int(len(candidates)),
            "accepted_disjoint_swaps": int(accepted),
            "swap_neighbors": int(geometry.protocol.swap_neighbors),
            "swap_search": (
                "deterministic_greedy_all_pairs"
                if int(geometry.protocol.swap_neighbors) >= rows
                else "deterministic_greedy_sparse_neighborhood"
            ),
            "swap_search_optimality_claimed": False,
            "ceiling_tightness": "conservative_witness_not_global_assignment_optimum",
            "quantity_interpretation": (
                "equal_prior_randomized_minimax_across_two_constructed_"
                "finite_batch_worlds"
            ),
            "not_a_realized_world_performance_bound": True,
            "cost_budget_relative": float(
                geometry.protocol.cost_budget_relative
            ),
            "cost_budget_absolute_tolerance": absolute_budget_tolerance,
            "actual_cost_ratio": (
                float(alternative_cost / baseline_cost)
                if baseline_cost > 0.0 else (1.0 if alternative_cost == 0.0 else None)
            ),
            "assignment": assignment_diagnostics,
            "assignment_world_zero_sha256": _array_sha256(assignment_zero),
            "assignment_world_one_sha256": _array_sha256(assignment_one),
            "prediction_world_zero_sha256": _array_sha256(prediction_zero),
            "prediction_world_one_sha256": _array_sha256(prediction_one),
            "calibration": dict(geometry.calibration_diagnostics),
        },
    )


def estimate_two_world_ceiling(
    target_calibration_features: np.ndarray,
    reference_calibration_features: np.ndarray,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray | None = None,
    *,
    protocol: CorrespondenceCeilingProtocol | None = None,
) -> TwoWorldCeilingWitness:
    """Fit calibration geometry and return the held-out two-world witness."""

    geometry = fit_witness_geometry(
        target_calibration_features,
        reference_calibration_features,
        readout_weight,
        readout_bias,
        protocol=protocol,
    )
    return construct_two_world_witness(
        geometry, target_features, reference_features
    )
