"""Additive Play-3 v2 optimization certificates.

This module deliberately does not alter the completed Play-3 v1 protocol.
It reconstructs fresh finite-batch witnesses under a separately frozen v2
lock and reports both sides of the budgeted maximum-disagreement problem:

* an integer-feasible assignment (a primal lower bound on ambiguity); and
* a sampled Lagrangian upper bound (also valid for the LP relaxation under the
  recorded floating-point solver model) obtained from linear-assignment
  subproblems.

The gap is a computational certificate for the maximum-disagreement search,
not a statistical confidence interval.  The equal-prior minimax value remains
exact only for the two assignments actually serialized in an output unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping

import numpy as np

from marginal_alignment.campaign.maps import (
    AffineMap,
    estimate_covariance,
    fit_covariance_gauge,
)
from marginal_alignment.linalg import as_samples, symmetric_power


CERTIFICATE_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_certificate.v1"
)
INFORMATION_SCOPE = (
    "unpaired_features_frozen_readout_and_prespecified_public_wnid_support;_"
    "authenticated_sample_ids_used_only_for_indexed_atom_administration_and_"
    "exact_duplicate_feature_ties;_"
    "row_level_wnids_withheld_from_assignment_optimization"
)
MAX_EXACT_VALIDATION_ROWS = 64
FULL_MAHALANOBIS = "full_reference_oas_mahalanobis_after_bures"
DIAGONAL_STANDARDIZED = "diagonal_reference_oas_standardized_after_bures"
COSINE_UNCENTERED = "uncentered_l2_normalized_cosine_after_bures"
SUPPORTED_COST_GEOMETRIES = (
    FULL_MAHALANOBIS,
    DIAGONAL_STANDARDIZED,
    COSINE_UNCENTERED,
)
_COST_NORMALIZATIONS = {
    FULL_MAHALANOBIS: (
        "squared_reference_oas_mahalanobis_distance_divided_by_"
        "full_feature_dimension"
    ),
    DIAGONAL_STANDARDIZED: (
        "squared_euclidean_distance_after_reference_oas_diagonal_"
        "standardization_divided_by_full_feature_dimension"
    ),
    COSINE_UNCENTERED: (
        "one_minus_dot_product_after_independent_row_l2_normalization_"
        "of_uncentered_bures_aligned_features"
    ),
}
_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class OptimizationCertificateProtocol:
    """Numerical contract for one additive v2 certificate unit."""

    cost_geometry: str
    cost_budget_relative: float
    covariance_estimator: str | float = "oas"
    covariance_relative_floor: float = 1e-6
    maximum_witness_rows: int = 2048
    greedy_neighbors: int = 2048
    dual_bisection_iterations: int = 48
    primal_dual_gap_target: float = 0.05
    assignment_tie_relative_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.cost_geometry not in SUPPORTED_COST_GEOMETRIES:
            raise ValueError(
                f"unsupported cost geometry {self.cost_geometry!r}: "
                f"{SUPPORTED_COST_GEOMETRIES}"
            )
        if (
            not np.isfinite(self.cost_budget_relative)
            or self.cost_budget_relative < 0.0
        ):
            raise ValueError("cost_budget_relative must be finite and nonnegative")
        if (
            not np.isfinite(self.covariance_relative_floor)
            or self.covariance_relative_floor <= 0.0
        ):
            raise ValueError("covariance_relative_floor must be finite and positive")
        if int(self.maximum_witness_rows) < 2:
            raise ValueError("maximum_witness_rows must be at least two")
        if int(self.greedy_neighbors) < 1:
            raise ValueError("greedy_neighbors must be positive")
        if int(self.dual_bisection_iterations) < 1:
            raise ValueError("dual_bisection_iterations must be positive")
        if (
            not np.isfinite(self.primal_dual_gap_target)
            or not 0.0 <= self.primal_dual_gap_target <= 1.0
        ):
            raise ValueError("primal_dual_gap_target must lie in [0,1]")
        if (
            not np.isfinite(self.assignment_tie_relative_tolerance)
            or self.assignment_tie_relative_tolerance < 0.0
        ):
            raise ValueError(
                "assignment_tie_relative_tolerance must be finite and nonnegative"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost_geometry": self.cost_geometry,
            "cost_normalization": _COST_NORMALIZATIONS[self.cost_geometry],
            "cost_budget_relative": float(self.cost_budget_relative),
            "covariance_estimator": self.covariance_estimator,
            "covariance_relative_floor": float(self.covariance_relative_floor),
            "maximum_witness_rows": int(self.maximum_witness_rows),
            "greedy_neighbors": int(self.greedy_neighbors),
            "dual_bisection_iterations": int(self.dual_bisection_iterations),
            "primal_dual_gap_target": float(self.primal_dual_gap_target),
            "assignment_tie_relative_tolerance": float(
                self.assignment_tie_relative_tolerance
            ),
            "discarded_feature_dimensions": 0,
        }


@dataclass(frozen=True)
class CertificateGeometry:
    bures_map: AffineMap
    reference_mean: np.ndarray
    reference_covariance: np.ndarray
    cost_transform: np.ndarray | None
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    protocol: OptimizationCertificateProtocol
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedCertificateProblem:
    """Budget-independent dense assignment inputs for one locked geometry."""

    geometry: CertificateGeometry
    cost: np.ndarray
    assignment_world_zero: np.ndarray
    reference_prediction: np.ndarray
    prediction_world_zero: np.ndarray
    target_canonical_order: np.ndarray
    reference_canonical_order: np.ndarray
    baseline_cost: float
    target_indexed_atom_sha256: str
    reference_indexed_atom_sha256: str
    target_indexed_atom_diagnostics: Mapping[str, Any]
    reference_indexed_atom_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class OptimizationCertificateResult:
    assignment_world_zero: np.ndarray
    assignment_world_one: np.ndarray
    prediction_world_zero: np.ndarray
    prediction_world_one: np.ndarray
    target_canonical_order: np.ndarray
    reference_canonical_order: np.ndarray
    baseline_cost: float
    primal_cost: float
    allowed_cost: float
    selected_world_disagreement: float
    selected_two_world_minimax_recovery: float
    maximum_disagreement_primal_lower_bound: float
    maximum_disagreement_dual_upper_bound: float
    primal_dual_gap: float
    diagnostics: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        target = float(self.diagnostics["protocol"]["primal_dual_gap_target"])
        return {
            "schema_version": CERTIFICATE_SCHEMA,
            "status": "PASS",
            "scientific_interpretation": {
                "selected_two_world_minimax_value": (
                    "exact_for_the_two_serialized_assignments"
                ),
                "maximum_disagreement_primal": (
                    "integer_feasible_lower_bound_on_maximum_ambiguity"
                ),
                "maximum_disagreement_dual": (
                    "sampled_lagrangian_computational_upper_bound_under_the_"
                    "recorded_floating_point_solver_model_also_valid_for_the_"
                    "lp_relaxation"
                ),
                "gap": (
                    "optimization_gap_not_a_statistical_confidence_interval"
                ),
                "not_external_target_label_accuracy": True,
                "not_a_realized_world_performance_bound": True,
            },
            "baseline_cost": float(self.baseline_cost),
            "primal_cost": float(self.primal_cost),
            "allowed_cost": float(self.allowed_cost),
            "selected_world_disagreement": float(
                self.selected_world_disagreement
            ),
            "selected_two_world_minimax_recovery": float(
                self.selected_two_world_minimax_recovery
            ),
            "maximum_disagreement_primal_lower_bound": float(
                self.maximum_disagreement_primal_lower_bound
            ),
            "maximum_disagreement_dual_upper_bound": float(
                self.maximum_disagreement_dual_upper_bound
            ),
            "maximum_disagreement_identification_interval": [
                float(self.maximum_disagreement_primal_lower_bound),
                float(self.maximum_disagreement_dual_upper_bound),
            ],
            "strongest_two_world_recovery_ceiling_interval": [
                float(1.0 - self.maximum_disagreement_dual_upper_bound / 2.0),
                float(1.0 - self.maximum_disagreement_primal_lower_bound / 2.0),
            ],
            "primal_dual_gap": float(self.primal_dual_gap),
            "primal_dual_gap_percentage_points": float(
                100.0 * self.primal_dual_gap
            ),
            "gap_target": target,
            "gap_target_percentage_points": float(100.0 * target),
            "meets_gap_target": bool(self.primal_dual_gap <= target),
            "assignment_world_zero_sha256": _array_sha256(
                self.assignment_world_zero
            ),
            "assignment_world_one_sha256": _array_sha256(
                self.assignment_world_one
            ),
            "prediction_world_zero_sha256": _array_sha256(
                self.prediction_world_zero
            ),
            "prediction_world_one_sha256": _array_sha256(
                self.prediction_world_one
            ),
            "target_canonical_order_sha256": _array_sha256(
                self.target_canonical_order
            ),
            "reference_canonical_order_sha256": _array_sha256(
                self.reference_canonical_order
            ),
            "diagnostics": dict(self.diagnostics),
        }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atom_id_bytes(value: Any) -> bytes:
    if isinstance(value, (str, np.str_)):
        return b"text:" + str(value).encode("utf-8")
    if isinstance(value, (bytes, np.bytes_)):
        return b"bytes:" + bytes(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return b"integer:" + str(int(value)).encode("ascii")
    raise ValueError(
        "sample IDs must be scalar UTF-8 text, bytes, or integers"
    )


def _unique_canonical_rows(
    features: np.ndarray,
    *,
    name: str,
    salt: bytes,
    sample_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Canonicalize indexed atoms without letting IDs enter costs or rewards.

    Feature bytes determine the primary order. Authenticated immutable sample
    IDs distinguish only atoms with identical feature bytes; original row
    position is never a tie key. Thus the result is invariant to a joint
    permutation of feature rows and their IDs. Duplicate IDs fail closed.
    """

    value = as_samples(features, name=name)
    canonical = np.ascontiguousarray(value, dtype="<f8")
    identifiers: list[bytes] | None = None
    if sample_ids is not None:
        raw_ids = np.asarray(sample_ids)
        if raw_ids.ndim != 1 or len(raw_ids) != len(canonical):
            raise ValueError(f"{name} must have one sample ID per row")
        identifiers = [_atom_id_bytes(item) for item in raw_ids.tolist()]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name} sample IDs must be unique")
    raw_rows = [row.tobytes() for row in canonical]
    counts: dict[bytes, int] = {}
    for raw in raw_rows:
        counts[raw] = counts.get(raw, 0) + 1
    duplicate_groups = sum(count > 1 for count in counts.values())
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    if duplicate_groups and identifiers is None:
        raise ValueError(
            f"{name} contains exact duplicate feature rows; unique immutable "
            "sample IDs are required to distinguish indexed atoms"
        )
    keyed: list[tuple[bytes, bytes, bytes, bytes, int]] = []
    for index, raw in enumerate(raw_rows):
        atom_id = b"" if identifiers is None else identifiers[index]
        identifier_hash = hashlib.sha256(
            salt + b"\x00indexed-atom-id\x00" + atom_id
        ).digest()
        keyed.append((
            hashlib.sha256(salt + b"\x00feature\x00" + raw).digest(),
            raw,
            identifier_hash,
            atom_id,
            index,
        ))
    keyed.sort(key=lambda item: item[:4])
    order = np.asarray([item[4] for item in keyed], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(salt)
    digest.update(str(canonical.shape).encode("ascii"))
    for row_hash, raw, identifier_hash, atom_id, _ in keyed:
        digest.update(row_hash)
        digest.update(raw)
        digest.update(identifier_hash)
        digest.update(atom_id)
    return canonical[order], order, digest.hexdigest(), {
        "indexed_atom_rows": int(len(canonical)),
        "exact_duplicate_feature_groups": int(duplicate_groups),
        "exact_duplicate_feature_rows_beyond_first": int(duplicate_rows),
        "sample_identifiers_supplied": identifiers is not None,
        "sample_identifiers_unique": identifiers is not None,
        "sample_identifiers_used_only_for_exact_duplicate_feature_ties": True,
        "sample_identifier_bytes_included_in_indexed_atom_integrity_digest": (
            identifiers is not None
        ),
        "sample_identifier_semantic_parsing": False,
        "original_row_position_used_as_tie_key": False,
        "cost_and_reward_depend_only_on_features_and_frozen_readout": True,
        "joint_row_and_id_permutation_invariant": True,
    }


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
        raise ValueError("readout_weight has an invalid shape or value")
    bias = (
        np.zeros(weight.shape[0], dtype=np.float64)
        if readout_bias is None
        else np.asarray(readout_bias, dtype=np.float64)
    )
    if bias.shape != (weight.shape[0],) or not np.all(np.isfinite(bias)):
        raise ValueError("readout_bias has an invalid shape or value")
    return np.array(weight, copy=True), np.array(bias, copy=True)


def fit_certificate_geometry(
    target_calibration_features: np.ndarray,
    reference_calibration_features: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray | None = None,
    *,
    protocol: OptimizationCertificateProtocol,
    target_calibration_ids: np.ndarray | None = None,
    reference_calibration_ids: np.ndarray | None = None,
) -> CertificateGeometry:
    """Fit the Bures map and one of three explicitly serialized geometries."""

    target, _, target_digest, target_atom_diagnostics = _unique_canonical_rows(
        target_calibration_features,
        name="target_calibration_features",
        salt=b"play3-v2-target-calibration-v1",
        sample_ids=target_calibration_ids,
    )
    reference, _, reference_digest, reference_atom_diagnostics = _unique_canonical_rows(
        reference_calibration_features,
        name="reference_calibration_features",
        salt=b"play3-v2-reference-calibration-v1",
        sample_ids=reference_calibration_ids,
    )
    if target.shape[1] != reference.shape[1]:
        raise ValueError("target and reference calibration dimensions differ")
    weight, bias = _validate_readout(
        readout_weight, readout_bias, dimension=reference.shape[1]
    )
    fitted = fit_covariance_gauge(
        target,
        reference,
        gauge="bures",
        estimator=protocol.covariance_estimator,
        relative_floor=float(protocol.covariance_relative_floor),
    )
    reference_mean, reference_covariance, covariance_diagnostics = (
        estimate_covariance(
            reference,
            estimator=protocol.covariance_estimator,
            relative_floor=float(protocol.covariance_relative_floor),
        )
    )
    if protocol.cost_geometry == FULL_MAHALANOBIS:
        cost_transform: np.ndarray | None = symmetric_power(
            reference_covariance, -0.5
        )
        cost_covariance_structure = "full_rank"
    elif protocol.cost_geometry == DIAGONAL_STANDARDIZED:
        diagonal = np.diag(reference_covariance)
        if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
            raise np.linalg.LinAlgError("reference covariance diagonal is invalid")
        cost_transform = np.diag(np.reciprocal(np.sqrt(diagonal)))
        cost_covariance_structure = "diagonal"
    else:
        cost_transform = None
        cost_covariance_structure = "rowwise_cosine_no_centering"

    diagnostics = {
        "schema_version": CERTIFICATE_SCHEMA,
        "information_scope": INFORMATION_SCOPE,
        "external_ground_truth_labels_used_for_optimization": False,
        "frozen_readout_predictions_used_for_optimization": True,
        "sample_identifiers_used_for_cost_or_reward": False,
        "sample_identifiers_used_for_indexed_atom_administration": bool(
            target_calibration_ids is not None
            and reference_calibration_ids is not None
        ),
        "pairing_used": False,
        "indexed_atom_policy": (
            "preserve_all_rows;_authenticated_unique_sample_ids_break_only_"
            "exact_feature_ties"
        ),
        "target_calibration_indexed_atom_sha256": target_digest,
        "reference_calibration_indexed_atom_sha256": reference_digest,
        "target_calibration_indexed_atoms": target_atom_diagnostics,
        "reference_calibration_indexed_atoms": reference_atom_diagnostics,
        "feature_dimension": int(reference.shape[1]),
        "classes": int(weight.shape[0]),
        "cost_geometry": protocol.cost_geometry,
        "cost_normalization": _COST_NORMALIZATIONS[protocol.cost_geometry],
        "cost_covariance_structure": cost_covariance_structure,
        "discarded_feature_dimensions": 0,
        "reference_covariance": covariance_diagnostics,
        "bures_matrix_sha256": _array_sha256(fitted.matrix),
        "bures_offset_sha256": _array_sha256(fitted.offset),
        "reference_mean_sha256": _array_sha256(reference_mean),
        "cost_transform_sha256": (
            _array_sha256(cost_transform) if cost_transform is not None else None
        ),
        "protocol": protocol.as_dict(),
    }
    return CertificateGeometry(
        bures_map=fitted,
        reference_mean=np.array(reference_mean, copy=True),
        reference_covariance=np.array(reference_covariance, copy=True),
        cost_transform=(
            None if cost_transform is None else np.array(cost_transform, copy=True)
        ),
        readout_weight=weight,
        readout_bias=bias,
        protocol=protocol,
        diagnostics=diagnostics,
    )


def derive_certificate_geometry(
    base: CertificateGeometry,
    *,
    protocol: OptimizationCertificateProtocol,
) -> CertificateGeometry:
    """Reuse a fitted Bures/reference covariance under another locked cost.

    Only the geometry-specific cost representation and budget-independent
    diagnostics are changed. Calibration features are not revisited.
    """

    if (
        protocol.covariance_estimator != base.protocol.covariance_estimator
        or float(protocol.covariance_relative_floor)
        != float(base.protocol.covariance_relative_floor)
    ):
        raise ValueError("derived geometry changed the fitted covariance protocol")
    if protocol.cost_geometry == FULL_MAHALANOBIS:
        transform: np.ndarray | None = symmetric_power(
            base.reference_covariance, -0.5
        )
        structure = "full_rank"
    elif protocol.cost_geometry == DIAGONAL_STANDARDIZED:
        diagonal = np.diag(base.reference_covariance)
        if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
            raise np.linalg.LinAlgError("reference covariance diagonal is invalid")
        transform = np.diag(np.reciprocal(np.sqrt(diagonal)))
        structure = "diagonal"
    else:
        transform = None
        structure = "rowwise_cosine_no_centering"
    diagnostics = dict(base.diagnostics)
    diagnostics.update({
        "cost_geometry": protocol.cost_geometry,
        "cost_normalization": _COST_NORMALIZATIONS[protocol.cost_geometry],
        "cost_covariance_structure": structure,
        "cost_transform_sha256": (
            _array_sha256(transform) if transform is not None else None
        ),
        "protocol": protocol.as_dict(),
        "shared_bures_calibration_sha256": hashlib.sha256(
            (
                str(base.diagnostics["bures_matrix_sha256"])
                + str(base.diagnostics["bures_offset_sha256"])
                + str(base.diagnostics["reference_mean_sha256"])
            ).encode("ascii")
        ).hexdigest(),
    })
    return CertificateGeometry(
        bures_map=base.bures_map,
        reference_mean=np.array(base.reference_mean, copy=True),
        reference_covariance=np.array(base.reference_covariance, copy=True),
        cost_transform=None if transform is None else np.array(transform, copy=True),
        readout_weight=np.array(base.readout_weight, copy=True),
        readout_bias=np.array(base.readout_bias, copy=True),
        protocol=protocol,
        diagnostics=diagnostics,
    )


def _cost_matrix(
    geometry: CertificateGeometry,
    target: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    mapped_target = geometry.bures_map.apply(target)
    if geometry.protocol.cost_geometry == COSINE_UNCENTERED:
        target_norm = np.linalg.norm(mapped_target, axis=1)
        reference_norm = np.linalg.norm(reference, axis=1)
        norm_scale = max(
            float(np.max(target_norm)), float(np.max(reference_norm)), 1.0
        )
        minimum_norm = 1024.0 * _EPSILON * norm_scale
        if np.any(target_norm <= minimum_norm) or np.any(
            reference_norm <= minimum_norm
        ):
            raise np.linalg.LinAlgError(
                "cosine geometry encountered a zero or numerically zero row"
            )
        left = mapped_target / target_norm[:, None]
        right = reference / reference_norm[:, None]
        return np.clip(1.0 - left @ right.T, 0.0, 2.0)
    if geometry.cost_transform is None:
        raise RuntimeError("non-cosine geometry lacks its cost transform")
    left = (mapped_target - geometry.reference_mean) @ geometry.cost_transform
    right = (reference - geometry.reference_mean) @ geometry.cost_transform
    left_norm = np.sum(np.square(left), axis=1)[:, None]
    right_norm = np.sum(np.square(right), axis=1)[None, :]
    return np.maximum(
        (left_norm + right_norm - 2.0 * (left @ right.T)) / left.shape[1],
        0.0,
    )


def _linear_assignment(cost: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - full environment contract
        raise RuntimeError("optimization certificates require scipy") from error
    row, column = linear_sum_assignment(cost)
    if not np.array_equal(row, np.arange(len(cost))):
        raise RuntimeError("linear assignment omitted a target row")
    return np.asarray(column, dtype=np.int64)


def _greedy_primal(
    cost: np.ndarray,
    assignment_zero: np.ndarray,
    prediction_zero: np.ndarray,
    *,
    allowed_cost: float,
    neighbors: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = len(assignment_zero)
    inverse = np.empty(rows, dtype=np.int64)
    inverse[assignment_zero] = np.arange(rows, dtype=np.int64)
    columns = np.arange(rows, dtype=np.int64)
    pairs: set[tuple[int, int]] = set()
    width = min(int(neighbors), rows)
    for target_row in range(rows):
        nearest = np.lexsort((columns, cost[target_row]))[:width]
        for reference_column in nearest:
            other = int(inverse[int(reference_column)])
            if other != target_row:
                pairs.add(tuple(sorted((target_row, other))))
    candidates: list[tuple[float, int, int]] = []
    for first, second in sorted(pairs):
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
        if increment < -tolerance:
            raise RuntimeError("a greedy swap improved the minimum-cost baseline")
        candidates.append((max(increment, 0.0), first, second))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    assignment = np.array(assignment_zero, copy=True)
    used = np.zeros(rows, dtype=bool)
    running = float(cost[np.arange(rows), assignment].sum())
    accepted = 0
    for increment, first, second in candidates:
        if used[first] or used[second]:
            continue
        # A positive numerical tolerance must never turn an over-budget
        # assignment into a claimed integer-feasible lower bound.
        if running + increment > allowed_cost:
            continue
        assignment[first], assignment[second] = (
            assignment[second], assignment[first]
        )
        exact_running = float(
            cost[np.arange(rows, dtype=np.int64), assignment].sum()
        )
        if exact_running > allowed_cost:
            assignment[first], assignment[second] = (
                assignment[second], assignment[first]
            )
            continue
        used[first] = True
        used[second] = True
        running = exact_running
        accepted += 1
    return assignment, {
        "algorithm": "deterministic_greedy_disjoint_swaps",
        "neighbors": width,
        "candidate_pairs": len(candidates),
        "accepted_swaps": accepted,
    }


def _primal_dual_search(
    cost: np.ndarray,
    reference_prediction: np.ndarray,
    prediction_zero: np.ndarray,
    *,
    baseline_cost: float,
    allowed_cost: float,
    initial_primal: np.ndarray,
    protocol: OptimizationCertificateProtocol,
) -> tuple[np.ndarray, float, float, float, dict[str, Any]]:
    rows = len(reference_prediction)
    row = np.arange(rows, dtype=np.int64)
    reward = (
        prediction_zero[:, None] != reference_prediction[None, :]
    ).astype(np.float64)
    tolerance = float(
        512.0
        * rows
        * _EPSILON
        * max(allowed_cost, baseline_cost, float(np.max(cost)), 1.0)
    )
    initial_cost = float(cost[row, initial_primal].sum())
    if initial_cost > allowed_cost:
        raise RuntimeError("initial primal violates the cost budget")
    if baseline_cost > tolerance:
        scale = baseline_cost
    else:
        positive = cost[cost > tolerance]
        scale = float(np.min(positive)) if positive.size else 1.0
    normalized_cost = cost / scale
    normalized_budget = allowed_cost / scale

    def values(assignment: np.ndarray) -> tuple[float, float]:
        return (
            float(reward[row, assignment].mean()),
            float(cost[row, assignment].sum()),
        )

    best_assignment = np.array(initial_primal, copy=True)
    best_primal, best_primal_cost = values(best_assignment)
    best_dual = 1.0
    best_multiplier = 0.0
    trace: list[dict[str, Any]] = []

    def evaluate(multiplier: float) -> tuple[float, float]:
        nonlocal best_assignment, best_primal, best_primal_cost
        nonlocal best_dual, best_multiplier
        assignment = _linear_assignment(
            multiplier * normalized_cost - reward / float(rows)
        )
        primal, assignment_cost = values(assignment)
        normalized_assignment_cost = assignment_cost / scale
        raw_dual = float(
            primal
            + multiplier * (normalized_budget - normalized_assignment_cost)
        )
        cushion = float(
            2048.0
            * rows
            * _EPSILON
            * max(
                1.0,
                abs(raw_dual),
                abs(multiplier)
                * (abs(normalized_budget) + abs(normalized_assignment_cost)),
            )
        )
        dual = min(1.0, raw_dual + cushion)
        if dual < best_dual:
            best_dual = dual
            best_multiplier = float(multiplier)
        feasible = assignment_cost <= allowed_cost
        if feasible and (
            primal > best_primal + 8.0 * _EPSILON
            or (
                abs(primal - best_primal) <= 8.0 * _EPSILON
                and (
                    assignment_cost < best_primal_cost - max(
                        tolerance,
                        float(protocol.assignment_tie_relative_tolerance)
                        * max(abs(best_primal_cost), abs(assignment_cost), 1.0),
                    )
                    or (
                        abs(assignment_cost - best_primal_cost) <= max(
                            tolerance,
                            float(protocol.assignment_tie_relative_tolerance)
                            * max(
                                abs(best_primal_cost), abs(assignment_cost), 1.0
                            ),
                        )
                        and tuple(assignment.tolist())
                        < tuple(best_assignment.tolist())
                    )
                )
            )
        ):
            best_assignment = np.array(assignment, copy=True)
            best_primal = primal
            best_primal_cost = assignment_cost
        trace.append({
            "multiplier": float(multiplier),
            "assignment_cost": assignment_cost,
            "assignment_cost_ratio_to_baseline": (
                float(assignment_cost / baseline_cost)
                if baseline_cost > 0.0 else None
            ),
            "disagreement": primal,
            "dual_upper_bound": dual,
            "budget_feasible": bool(feasible),
            "assignment_sha256": _array_sha256(assignment),
        })
        return assignment_cost, dual

    zero_cost, _ = evaluate(0.0)
    if zero_cost <= allowed_cost:
        bracket = {"lower": 0.0, "upper": 0.0, "doublings": 0}
    else:
        lower = 0.0
        upper = 1.0
        doublings = 0
        while True:
            upper_cost, _ = evaluate(upper)
            if upper_cost <= allowed_cost:
                break
            lower = upper
            upper *= 2.0
            doublings += 1
            if doublings > 64 or not np.isfinite(upper):
                raise RuntimeError("could not bracket a feasible dual assignment")
        for _ in range(int(protocol.dual_bisection_iterations)):
            midpoint = float(lower + (upper - lower) / 2.0)
            midpoint_cost, _ = evaluate(midpoint)
            if midpoint_cost > allowed_cost:
                lower = midpoint
            else:
                upper = midpoint
        bracket = {
            "lower": float(lower),
            "upper": float(upper),
            "doublings": int(doublings),
        }
    dual_upper = float(min(1.0, max(best_primal, best_dual)))
    gap = float(max(dual_upper - best_primal, 0.0))
    return best_assignment, best_primal, dual_upper, gap, {
        "primal_solver": "best_of_greedy_and_feasible_lagrangian_assignments",
        "dual_solver": "scipy_linear_sum_assignment_lagrangian_bisection",
        "dual_formula": (
            "min_lambda_ge_0 lambda*B + max_bijection(mean_reward-lambda*cost)"
        ),
        "upper_bounds_lp_relaxation_optimum": True,
        "optimized_lagrangian_dual_value_claimed": False,
        "formal_real_arithmetic_certificate_claimed": False,
        "floating_point_solver_not_formally_verified": True,
        "roundoff_upper_cushion_applied": True,
        "cost_scale": float(scale),
        "absolute_cost_tolerance": tolerance,
        "feasibility_tolerance": 0.0,
        "assignment_tie_relative_tolerance": float(
            protocol.assignment_tie_relative_tolerance
        ),
        "best_dual_multiplier": best_multiplier,
        "bracket": bracket,
        "evaluations": len(trace),
        "trace": trace,
    }


def solve_exact_budgeted_disagreement_milp(
    cost: np.ndarray,
    reference_prediction: np.ndarray,
    prediction_zero: np.ndarray,
    *,
    allowed_cost: float,
    time_limit_seconds: float = 900.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a small validation problem to proven MILP optimality.

    This check is intended for the separately locked 64-row diagnostic
    subsets.  It is not used to describe the 1,024-row solution as exact.
    """

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:  # pragma: no cover - full environment contract
        raise RuntimeError("exact validation requires scipy.optimize.milp") from error
    matrix = np.asarray(cost, dtype=np.float64)
    reference = np.asarray(reference_prediction)
    world_zero = np.asarray(prediction_zero)
    rows = len(reference)
    if (
        matrix.shape != (rows, rows)
        or world_zero.shape != (rows,)
        or rows < 2
        or not np.all(np.isfinite(matrix))
        or not np.isfinite(allowed_cost)
        or allowed_cost < 0.0
    ):
        raise ValueError("exact-validation inputs are malformed")
    reward = (world_zero[:, None] != reference[None, :]).astype(np.float64)
    variable = np.arange(rows * rows, dtype=np.int64).reshape(rows, rows)
    constraint_indices = np.concatenate([
        np.repeat(np.arange(rows, dtype=np.int64), rows),
        np.repeat(np.arange(rows, 2 * rows, dtype=np.int64), rows),
        np.full(rows * rows, 2 * rows, dtype=np.int64),
    ])
    variable_indices = np.concatenate([
        variable.reshape(-1),
        variable.T.reshape(-1),
        variable.reshape(-1),
    ])
    values = np.concatenate([
        np.ones(rows * rows, dtype=np.float64),
        np.ones(rows * rows, dtype=np.float64),
        matrix.reshape(-1),
    ])
    constraints = coo_matrix(
        (values, (constraint_indices, variable_indices)),
        shape=(2 * rows + 1, rows * rows),
    ).tocsr()
    lower = np.concatenate([
        np.ones(2 * rows, dtype=np.float64),
        np.asarray([-np.inf]),
    ])
    upper = np.concatenate([
        np.ones(2 * rows, dtype=np.float64),
        np.asarray([float(allowed_cost)]),
    ])
    result = milp(
        c=-reward.reshape(-1) / float(rows),
        integrality=np.ones(rows * rows, dtype=np.uint8),
        bounds=Bounds(
            np.zeros(rows * rows, dtype=np.float64),
            np.ones(rows * rows, dtype=np.float64),
        ),
        constraints=LinearConstraint(constraints, lower, upper),
        options={
            "presolve": True,
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": 0.0,
        },
    )
    if not result.success or result.x is None or int(result.status) != 0:
        raise RuntimeError(
            "exact MILP validation did not prove optimality: "
            f"status={result.status}, message={result.message!s}, "
            f"mip_gap={getattr(result, 'mip_gap', None)}"
        )
    solution = np.asarray(result.x, dtype=np.float64).reshape(rows, rows)
    assignment = np.argmax(solution, axis=1).astype(np.int64)
    row = np.arange(rows, dtype=np.int64)
    if (
        len(np.unique(assignment)) != rows
        or not np.all(solution[row, assignment] >= 1.0 - 1e-7)
        or float(matrix[row, assignment].sum()) > allowed_cost
    ):
        raise RuntimeError("exact MILP returned a nonintegral or infeasible assignment")
    optimum = float(reward[row, assignment].mean())
    solver_objective = float(-result.fun)
    if abs(optimum - solver_objective) > 1e-7:
        raise RuntimeError("exact MILP objective disagrees with reconstructed reward")
    return assignment, {
        "status": "PROVEN_OPTIMAL",
        "solver": "scipy_optimize_milp_highs",
        "rows": rows,
        "binary_variables": rows * rows,
        "relative_mip_gap_requested": 0.0,
        "reported_mip_gap": (
            0.0
            if getattr(result, "mip_gap", None) is None
            else float(result.mip_gap)
        ),
        "maximum_disagreement_exact": optimum,
        "assignment_cost": float(matrix[row, assignment].sum()),
        "allowed_cost": float(allowed_cost),
        "assignment_sha256": _array_sha256(assignment),
        "solver_message": str(result.message),
    }


def run_exact_subset_validation(
    geometry: CertificateGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    target_sample_ids: np.ndarray | None = None,
    reference_sample_ids: np.ndarray | None = None,
    validation_rows: int = MAX_EXACT_VALIDATION_ROWS,
    time_limit_seconds: float = 900.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compare the certificate with a proven exact deterministic subset."""

    target, target_order, target_digest, target_atom_diagnostics = (
        _unique_canonical_rows(
        target_features,
        name="exact_validation_target_features",
        salt=b"play3-v2-target-witness-v1",
        sample_ids=target_sample_ids,
    ))
    reference, reference_order, reference_digest, reference_atom_diagnostics = (
        _unique_canonical_rows(
        reference_features,
        name="exact_validation_reference_features",
        salt=b"play3-v2-reference-witness-v1",
        sample_ids=reference_sample_ids,
    ))
    count = int(validation_rows)
    if (
        count < 2
        or count > MAX_EXACT_VALIDATION_ROWS
        or count > min(len(target), len(reference))
    ):
        raise ValueError("exact validation row count is invalid")
    target_ids = (
        None
        if target_sample_ids is None
        else np.asarray(target_sample_ids)[target_order][:count]
    )
    reference_ids = (
        None
        if reference_sample_ids is None
        else np.asarray(reference_sample_ids)[reference_order][:count]
    )
    target = target[:count]
    reference = reference[:count]
    cost = _cost_matrix(geometry, target, reference)
    assignment_zero = _linear_assignment(cost)
    row = np.arange(count, dtype=np.int64)
    baseline_cost = float(cost[row, assignment_zero].sum())
    allowed_cost = float(
        baseline_cost * (1.0 + geometry.protocol.cost_budget_relative)
    )
    reference_prediction = np.argmax(
        reference @ geometry.readout_weight.T + geometry.readout_bias,
        axis=1,
    ).astype(np.int64)
    prediction_zero = reference_prediction[assignment_zero]
    exact_assignment, exact = solve_exact_budgeted_disagreement_milp(
        cost,
        reference_prediction,
        prediction_zero,
        allowed_cost=allowed_cost,
        time_limit_seconds=time_limit_seconds,
    )
    approximate = construct_optimization_certificate(
        geometry,
        target,
        reference,
        target_sample_ids=target_ids,
        reference_sample_ids=reference_ids,
    ).summary()
    exact_value = float(exact["maximum_disagreement_exact"])
    if not (
        float(approximate["maximum_disagreement_primal_lower_bound"])
        <= exact_value + 1e-10
        and exact_value
        <= float(approximate["maximum_disagreement_dual_upper_bound"]) + 1e-10
    ):
        raise RuntimeError("exact subset optimum falls outside certificate bounds")
    return exact_assignment, {
        **exact,
        "selection": (
            "first_indexed_atoms_after_independent_feature_primary_and_"
            "sample_id_duplicate_tie_canonicalization_of_each_endpoint"
        ),
        "target_parent_indexed_atom_sha256": target_digest,
        "reference_parent_indexed_atom_sha256": reference_digest,
        "target_parent_indexed_atoms": target_atom_diagnostics,
        "reference_parent_indexed_atoms": reference_atom_diagnostics,
        "certificate_primal_lower_bound": approximate[
            "maximum_disagreement_primal_lower_bound"
        ],
        "certificate_dual_upper_bound": approximate[
            "maximum_disagreement_dual_upper_bound"
        ],
        "certificate_contains_exact_optimum": True,
    }


def prepare_certificate_problem(
    geometry: CertificateGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    target_sample_ids: np.ndarray | None = None,
    reference_sample_ids: np.ndarray | None = None,
) -> PreparedCertificateProblem:
    """Prepare the budget-independent cost/reward problem exactly once."""

    target, target_order, target_digest, target_atom_diagnostics = (
        _unique_canonical_rows(
        target_features,
        name="target_witness_features",
        salt=b"play3-v2-target-witness-v1",
        sample_ids=target_sample_ids,
    ))
    reference, reference_order, reference_digest, reference_atom_diagnostics = (
        _unique_canonical_rows(
        reference_features,
        name="reference_witness_features",
        salt=b"play3-v2-reference-witness-v1",
        sample_ids=reference_sample_ids,
    ))
    if target.shape != reference.shape:
        raise ValueError("certificate endpoints must have equal shapes")
    rows, _ = target.shape
    if rows > int(geometry.protocol.maximum_witness_rows):
        raise ValueError("witness exceeds the declared dense-assignment limit")
    cost = _cost_matrix(geometry, target, reference)
    assignment_zero = _linear_assignment(cost)
    if len(np.unique(assignment_zero)) != rows:
        raise RuntimeError("world-zero assignment is not a bijection")
    reference_prediction = np.argmax(
        reference @ geometry.readout_weight.T + geometry.readout_bias,
        axis=1,
    ).astype(np.int64)
    prediction_zero = reference_prediction[assignment_zero]
    row = np.arange(rows, dtype=np.int64)
    baseline_cost = float(cost[row, assignment_zero].sum())
    return PreparedCertificateProblem(
        geometry=geometry,
        cost=cost,
        assignment_world_zero=assignment_zero,
        reference_prediction=reference_prediction,
        prediction_world_zero=prediction_zero,
        target_canonical_order=target_order,
        reference_canonical_order=reference_order,
        baseline_cost=baseline_cost,
        target_indexed_atom_sha256=target_digest,
        reference_indexed_atom_sha256=reference_digest,
        target_indexed_atom_diagnostics=target_atom_diagnostics,
        reference_indexed_atom_diagnostics=reference_atom_diagnostics,
    )


def construct_optimization_certificate_from_problem(
    problem: PreparedCertificateProblem,
    *,
    protocol: OptimizationCertificateProtocol | None = None,
) -> OptimizationCertificateResult:
    """Solve one budget while reusing the immutable dense assignment problem."""

    geometry = problem.geometry
    declared = geometry.protocol if protocol is None else protocol
    if (
        declared.cost_geometry != geometry.protocol.cost_geometry
        or declared.covariance_estimator != geometry.protocol.covariance_estimator
        or float(declared.covariance_relative_floor)
        != float(geometry.protocol.covariance_relative_floor)
        or int(declared.maximum_witness_rows)
        != int(geometry.protocol.maximum_witness_rows)
    ):
        raise ValueError("budget solve changed the prepared geometry contract")
    cost = problem.cost
    assignment_zero = problem.assignment_world_zero
    reference_prediction = problem.reference_prediction
    prediction_zero = problem.prediction_world_zero
    rows = len(assignment_zero)
    dimension = int(geometry.readout_weight.shape[1])
    row = np.arange(rows, dtype=np.int64)
    baseline_cost = float(problem.baseline_cost)
    allowed_cost = float(
        baseline_cost * (1.0 + declared.cost_budget_relative)
    )
    tolerance = float(
        512.0
        * rows
        * _EPSILON
        * max(allowed_cost, baseline_cost, float(np.max(cost)), 1.0)
    )
    greedy, greedy_diagnostics = _greedy_primal(
        cost,
        assignment_zero,
        prediction_zero,
        allowed_cost=allowed_cost,
        neighbors=int(declared.greedy_neighbors),
        tolerance=tolerance,
    )
    assignment_one, primal, dual, gap, optimization_diagnostics = (
        _primal_dual_search(
            cost,
            reference_prediction,
            prediction_zero,
            baseline_cost=baseline_cost,
            allowed_cost=allowed_cost,
            initial_primal=greedy,
            protocol=declared,
        )
    )
    if len(np.unique(assignment_one)) != rows:
        raise RuntimeError("world-one assignment is not a bijection")
    primal_cost = float(cost[row, assignment_one].sum())
    if primal_cost > allowed_cost:
        raise RuntimeError("world-one assignment violates the cost budget")
    prediction_one = reference_prediction[assignment_one]
    selected_disagreement = float(np.mean(prediction_zero != prediction_one))
    if abs(selected_disagreement - primal) > 8.0 * _EPSILON:
        raise RuntimeError("serialized primal disagreement is inconsistent")
    return OptimizationCertificateResult(
        assignment_world_zero=assignment_zero,
        assignment_world_one=assignment_one,
        prediction_world_zero=prediction_zero,
        prediction_world_one=prediction_one,
        target_canonical_order=problem.target_canonical_order,
        reference_canonical_order=problem.reference_canonical_order,
        baseline_cost=baseline_cost,
        primal_cost=primal_cost,
        allowed_cost=allowed_cost,
        selected_world_disagreement=selected_disagreement,
        selected_two_world_minimax_recovery=float(
            1.0 - selected_disagreement / 2.0
        ),
        maximum_disagreement_primal_lower_bound=primal,
        maximum_disagreement_dual_upper_bound=dual,
        primal_dual_gap=gap,
        diagnostics={
            "information_scope": INFORMATION_SCOPE,
            "external_ground_truth_labels_used_for_optimization": False,
            "frozen_readout_predictions_used_for_optimization": True,
            "sample_identifiers_used_for_cost_or_reward": False,
            "sample_identifiers_used_for_indexed_atom_administration": bool(
                problem.target_indexed_atom_diagnostics[
                    "sample_identifiers_supplied"
                ]
                and problem.reference_indexed_atom_diagnostics[
                    "sample_identifiers_supplied"
                ]
            ),
            "pairing_used": False,
            "witness_rows": rows,
            "feature_dimension": dimension,
            "indexed_atom_policy": (
                "preserve_all_rows;_authenticated_unique_sample_ids_break_only_"
                "exact_feature_ties"
            ),
            "target_witness_indexed_atom_sha256": (
                problem.target_indexed_atom_sha256
            ),
            "reference_witness_indexed_atom_sha256": (
                problem.reference_indexed_atom_sha256
            ),
            "target_witness_indexed_atoms": dict(
                problem.target_indexed_atom_diagnostics
            ),
            "reference_witness_indexed_atoms": dict(
                problem.reference_indexed_atom_diagnostics
            ),
            "cost_matrix_sha256": _array_sha256(cost),
            "reference_prediction_sha256": _array_sha256(reference_prediction),
            "protocol": declared.as_dict(),
            "geometry": {
                **dict(geometry.diagnostics),
                "protocol": declared.as_dict(),
            },
            "budget_independent_problem_reused": protocol is not None,
            "greedy": greedy_diagnostics,
            "optimization": optimization_diagnostics,
        },
    )


def construct_optimization_certificate(
    geometry: CertificateGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    target_sample_ids: np.ndarray | None = None,
    reference_sample_ids: np.ndarray | None = None,
) -> OptimizationCertificateResult:
    """Construct and bound a maximum-disagreement finite-batch witness."""

    problem = prepare_certificate_problem(
        geometry,
        target_features,
        reference_features,
        target_sample_ids=target_sample_ids,
        reference_sample_ids=reference_sample_ids,
    )
    return construct_optimization_certificate_from_problem(problem)


def verify_result_against_inputs(
    result: OptimizationCertificateResult,
    geometry: CertificateGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    target_sample_ids: np.ndarray | None = None,
    reference_sample_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recompute geometry costs, rewards, bijections, and all result hashes."""

    recomputed = construct_optimization_certificate(
        geometry,
        target_features,
        reference_features,
        target_sample_ids=target_sample_ids,
        reference_sample_ids=reference_sample_ids,
    )
    arrays = (
        "assignment_world_zero",
        "assignment_world_one",
        "prediction_world_zero",
        "prediction_world_one",
        "target_canonical_order",
        "reference_canonical_order",
    )
    for name in arrays:
        if not np.array_equal(getattr(result, name), getattr(recomputed, name)):
            raise RuntimeError(f"certificate reconstruction mismatch: {name}")
    if result.summary() != recomputed.summary():
        raise RuntimeError("certificate summary reconstruction mismatch")
    return {
        "status": "PASS",
        "assignments_recomputed": True,
        "costs_recomputed_from_endpoint_features": True,
        "rewards_recomputed_from_frozen_readout": True,
        "bijections_revalidated": True,
        "duplicate_row_admissibility_revalidated": True,
    }
