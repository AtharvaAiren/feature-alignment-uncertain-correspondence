"""Locked correction maps and the one shared correspondence-cost geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from marginal_alignment.campaign.maps import (
    AffineMap,
    estimate_covariance,
    fit_covariance_gauge,
    fit_diagonal,
    fit_identity,
)
from marginal_alignment.linalg import as_samples, symmetric_power


COMMON_COST_GEOMETRY = "full_reference_oas_mahalanobis_after_bures"
METHODS = ("identity", "diagonal", "bures")
_EPSILON = np.finfo(np.float64).eps


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FittedRecoveryGeometry:
    corrections: Mapping[str, AffineMap]
    cost_bures_map: AffineMap
    reference_mean: np.ndarray
    reference_covariance: np.ndarray
    reference_inverse_sqrt: np.ndarray
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class BatchCostProblem:
    cost: np.ndarray
    minimum_assignment: np.ndarray
    minimum_cost: float
    random_expected_cost: float
    random_minus_minimum: float
    true_assignment: np.ndarray
    true_cost: float
    true_normalized_score: float
    diagnostics: Mapping[str, Any]


def _score_assignment(
    cost: np.ndarray,
    assignment: np.ndarray,
    *,
    minimum_cost: float,
    random_minus_minimum: float,
    tolerance: float,
) -> tuple[float, float]:
    rows = len(cost)
    candidate = np.asarray(assignment)
    if (
        candidate.shape != (rows,)
        or not np.issubdtype(candidate.dtype, np.integer)
        or not np.array_equal(
            np.sort(candidate.astype(np.int64, copy=False)), np.arange(rows)
        )
    ):
        raise ValueError("correspondence must be a permutation")
    candidate = candidate.astype(np.int64, copy=False)
    value = float(
        cost[np.arange(rows, dtype=np.int64), candidate].sum(dtype=np.float64)
    )
    if value < minimum_cost - tolerance:
        raise RuntimeError("correspondence cost is below the LAP minimum")
    score = float(max((value - minimum_cost) / random_minus_minimum, 0.0))
    return value, score


def _readout(
    weight: np.ndarray, bias: np.ndarray, *, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if (
        w.ndim != 2
        or w.shape[0] < 2
        or w.shape[1] != dimension
        or b.shape != (w.shape[0],)
        or not np.all(np.isfinite(w))
        or not np.all(np.isfinite(b))
    ):
        raise ValueError("frozen readout has an invalid shape or nonfinite value")
    return np.array(w, copy=True), np.array(b, copy=True)


def fit_recovery_geometry(
    target_map_fit: np.ndarray,
    reference_map_fit: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
    *,
    covariance_estimator: str = "oas",
    covariance_relative_floor: float = 1e-6,
    diagonal_standard_deviation_floor: float = 1e-6,
) -> FittedRecoveryGeometry:
    """Fit three correction maps and one method-independent feasible-set cost."""

    target = as_samples(target_map_fit, name="target_map_fit")
    reference = as_samples(reference_map_fit, name="reference_map_fit")
    if target.shape != reference.shape or len(target) < 2:
        raise ValueError("map-fit endpoints must be equal-shape paired-set matrices")
    weight, bias = _readout(
        readout_weight, readout_bias, dimension=reference.shape[1]
    )
    identity = fit_identity(target, reference)
    diagonal = fit_diagonal(
        target,
        reference,
        standard_deviation_floor=float(diagonal_standard_deviation_floor),
    )
    bures = fit_covariance_gauge(
        target,
        reference,
        gauge="bures",
        estimator=covariance_estimator,
        relative_floor=float(covariance_relative_floor),
    )
    reference_mean, reference_covariance, covariance_diagnostics = estimate_covariance(
        reference,
        estimator=covariance_estimator,
        relative_floor=float(covariance_relative_floor),
    )
    inverse_sqrt = symmetric_power(reference_covariance, -0.5)
    corrections = {"identity": identity, "diagonal": diagonal, "bures": bures}
    diagnostics = {
        "cost_geometry": COMMON_COST_GEOMETRY,
        "cost_normalization": "total_dimension_normalized_squared_reference_oas_mahalanobis",
        "cost_map_fit": "bures_oas_fitted_once_on_assignment_erased_paired_endpoint_set",
        "correction_methods": list(METHODS),
        "correction_fit_rows": int(len(target)),
        "feature_dimension": int(target.shape[1]),
        "classes": int(weight.shape[0]),
        "covariance_estimator": covariance_estimator,
        "covariance_relative_floor": float(covariance_relative_floor),
        "diagonal_standard_deviation_floor": float(
            diagonal_standard_deviation_floor
        ),
        "reference_covariance": covariance_diagnostics,
        "map_fit_pair_assignment_used_by_any_correction": False,
        "map_fit_endpoint_membership_is_paired": True,
        "maps": {
            name: {
                "matrix_sha256": array_sha256(fitted.matrix),
                "offset_sha256": array_sha256(fitted.offset),
                "diagnostics": dict(fitted.diagnostics),
            }
            for name, fitted in corrections.items()
        },
        "cost_bures_matrix_sha256": array_sha256(bures.matrix),
        "cost_bures_offset_sha256": array_sha256(bures.offset),
        "reference_mean_sha256": array_sha256(reference_mean),
        "reference_covariance_sha256": array_sha256(reference_covariance),
        "reference_inverse_sqrt_sha256": array_sha256(inverse_sqrt),
        "readout_weight_sha256": array_sha256(weight),
        "readout_bias_sha256": array_sha256(bias),
    }
    return FittedRecoveryGeometry(
        corrections=corrections,
        cost_bures_map=bures,
        reference_mean=np.array(reference_mean, copy=True),
        reference_covariance=np.array(reference_covariance, copy=True),
        reference_inverse_sqrt=np.array(inverse_sqrt, copy=True),
        readout_weight=weight,
        readout_bias=bias,
        diagnostics=diagnostics,
    )


def correspondence_cost_matrix(
    fitted: FittedRecoveryGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
) -> np.ndarray:
    target = as_samples(target_features, name="target_batch")
    reference = as_samples(reference_features, name="reference_batch")
    if target.shape != reference.shape:
        raise ValueError("batch endpoints must have equal shape")
    mapped = fitted.cost_bures_map.apply(target)
    left = (mapped - fitted.reference_mean) @ fitted.reference_inverse_sqrt
    right = (reference - fitted.reference_mean) @ fitted.reference_inverse_sqrt
    left_norm = np.sum(np.square(left), axis=1, dtype=np.float64)[:, None]
    right_norm = np.sum(np.square(right), axis=1, dtype=np.float64)[None, :]
    cost = (left_norm + right_norm - 2.0 * left @ right.T) / float(left.shape[1])
    return np.maximum(np.asarray(cost, dtype=np.float64), 0.0)


def _linear_assignment(cost: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - server dependency contract
        raise RuntimeError("paired recovery bounds require scipy") from error
    rows, columns = linear_sum_assignment(np.asarray(cost, dtype=np.float64))
    if not np.array_equal(rows, np.arange(len(cost))):
        raise RuntimeError("linear assignment omitted a target row")
    return np.asarray(columns, dtype=np.int64)


def batch_cost_problem(
    fitted: FittedRecoveryGeometry,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    true_assignment: np.ndarray | None = None,
) -> BatchCostProblem:
    """Build one cost problem and score its separately supplied true pairing."""

    cost = correspondence_cost_matrix(fitted, target_features, reference_features)
    rows = len(cost)
    row = np.arange(rows, dtype=np.int64)
    minimum_assignment = _linear_assignment(cost)
    minimum = float(cost[row, minimum_assignment].sum(dtype=np.float64))
    random_expected = float(cost.sum(dtype=np.float64) / rows)
    gap = float(random_expected - minimum)
    tolerance = float(
        4096.0
        * rows
        * _EPSILON
        * max(abs(random_expected), abs(minimum), float(np.max(cost)), 1.0)
    )
    if not math.isfinite(gap) or gap <= tolerance:
        raise np.linalg.LinAlgError(
            "uniform-random minus minimum assignment-cost gap is not positive"
        )
    assignment = (
        np.arange(rows, dtype=np.int64)
        if true_assignment is None
        else np.asarray(true_assignment, dtype=np.int64)
    )
    true_cost, score = _score_assignment(
        cost,
        assignment,
        minimum_cost=minimum,
        random_minus_minimum=gap,
        tolerance=tolerance,
    )
    centered = (
        cost
        - cost.mean(axis=1, keepdims=True)
        - cost.mean(axis=0, keepdims=True)
        + cost.mean()
    )
    variance = float(np.sum(np.square(centered), dtype=np.float64) / (rows - 1))
    return BatchCostProblem(
        cost=cost,
        minimum_assignment=minimum_assignment,
        minimum_cost=minimum,
        random_expected_cost=random_expected,
        random_minus_minimum=gap,
        true_assignment=assignment,
        true_cost=true_cost,
        true_normalized_score=score,
        diagnostics={
            "rows": rows,
            "cost_matrix_sha256": array_sha256(cost),
            "minimum_assignment_sha256": array_sha256(minimum_assignment),
            "normalization": "(C_true-C_min)/(E_uniform_bijection_C-C_min)",
            "uniform_random_cost_variance": max(variance, 0.0),
            "numerical_tolerance": tolerance,
        },
    )


def score_correspondence(
    problem: BatchCostProblem, assignment: np.ndarray
) -> tuple[float, float]:
    """Score a held-out correspondence after constructing its feasible set."""

    return _score_assignment(
        problem.cost,
        assignment,
        minimum_cost=problem.minimum_cost,
        random_minus_minimum=problem.random_minus_minimum,
        tolerance=float(problem.diagnostics["numerical_tolerance"]),
    )


def allowed_cost(problem: BatchCostProblem, normalized_threshold: float) -> float:
    threshold = float(normalized_threshold)
    if math.isnan(threshold) or threshold < 0.0:
        raise ValueError("normalized threshold must be nonnegative or positive infinity")
    if math.isinf(threshold):
        return math.inf
    return float(problem.minimum_cost + threshold * problem.random_minus_minimum)


def frozen_predictions(
    fitted: FittedRecoveryGeometry,
    *,
    method: str,
    target_features: np.ndarray,
) -> np.ndarray:
    if method not in fitted.corrections:
        raise ValueError(f"unknown recovery method: {method}")
    mapped = fitted.corrections[method].apply(target_features)
    logits = mapped @ fitted.readout_weight.T + fitted.readout_bias
    return np.argmax(logits, axis=1).astype(np.int64)


def clean_predictions(
    fitted: FittedRecoveryGeometry, reference_features: np.ndarray
) -> np.ndarray:
    reference = as_samples(reference_features, name="reference_batch")
    logits = reference @ fitted.readout_weight.T + fitted.readout_bias
    return np.argmax(logits, axis=1).astype(np.int64)
