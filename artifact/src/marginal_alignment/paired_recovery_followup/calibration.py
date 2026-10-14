"""Same-raw-data direct recovery-error conformal baseline.

The baseline uses a single max-standardized score across the predeclared
methods.  Centers and scales are fixed from the disjoint map-fit role only;
the calibration role then supplies the ordinary split-conformal rank.  This
gives simultaneous method coverage without a methodwise Bonferroni factor.
It uses true fit-role pairing for its centers and scales, while geometric map
fitting erases that assignment; the raw identity budget, not usable
information, is matched.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_alignment.paired_recovery_bounds.conformal import conformal_rank


DIRECT_METHODS = ("identity", "diagonal", "bures")


def _method_matrix(
    values: Mapping[str, Sequence[float] | np.ndarray],
    *,
    methods: Sequence[str],
    name: str,
) -> np.ndarray:
    ordered = tuple(str(method) for method in methods)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("methods must be a nonempty ordered set")
    if set(values) != set(ordered):
        raise ValueError(f"{name} must contain exactly the predeclared methods")
    columns = []
    length: int | None = None
    for method in ordered:
        column = np.asarray(values[method], dtype=np.float64)
        if column.ndim != 1 or not len(column) or not np.all(np.isfinite(column)):
            raise ValueError(f"{name}[{method}] must be a nonempty finite vector")
        if np.any(column < 0.0) or np.any(column > 1.0):
            raise ValueError(f"{name}[{method}] must lie in [0,1]")
        if length is None:
            length = len(column)
        elif len(column) != length:
            raise ValueError(f"{name} vectors must have equal length")
        columns.append(column)
    return np.column_stack(columns)


@dataclass(frozen=True)
class DirectStandardizer:
    methods: tuple[str, ...]
    centers: np.ndarray
    scales: np.ndarray
    scale_floor: float
    fitting_batch_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "methods": list(self.methods),
            "centers": {
                method: float(self.centers[index])
                for index, method in enumerate(self.methods)
            },
            "scales": {
                method: float(self.scales[index])
                for index, method in enumerate(self.methods)
            },
            "center_rule": "arithmetic_mean_of_map_fit_role_batch_errors",
            "scale_rule": "sample_standard_deviation_ddof_1_floored",
            "scale_floor": float(self.scale_floor),
            "fitting_batch_count": int(self.fitting_batch_count),
            "pairing_used_only_after_all_correction_maps_were_frozen": True,
        }


@dataclass(frozen=True)
class DirectCalibration:
    standardizer: DirectStandardizer
    threshold: float
    rank_one_indexed: int
    calibration_scores: np.ndarray
    upper_bounds: np.ndarray
    delta: float

    def upper_by_method(self) -> dict[str, float]:
        return {
            method: float(self.upper_bounds[index])
            for index, method in enumerate(self.standardizer.methods)
        }

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "shared_max_standardized_direct_recovery_error",
            "methods": list(self.standardizer.methods),
            "delta": float(self.delta),
            "rank_one_indexed": int(self.rank_one_indexed),
            "threshold": float(self.threshold),
            "calibration_count": int(len(self.calibration_scores)),
            "calibration_scores": [float(value) for value in self.calibration_scores],
            "upper_bounds": self.upper_by_method(),
            "simultaneous_method_coverage_from_one_shared_score": True,
            "standardizer": self.standardizer.summary(),
        }


def fit_direct_standardizer(
    fitting_batch_errors: Mapping[str, Sequence[float] | np.ndarray],
    *,
    methods: Sequence[str] = DIRECT_METHODS,
    scale_floor: float = 1.0 / 256.0,
) -> DirectStandardizer:
    """Fix method centers/scales using only disjoint fitting-role errors."""

    matrix = _method_matrix(
        fitting_batch_errors, methods=methods, name="fitting_batch_errors"
    )
    if len(matrix) < 2:
        raise ValueError("sample-standard-deviation scaling needs at least two batches")
    floor = float(scale_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    centers = matrix.mean(axis=0, dtype=np.float64)
    raw_scales = matrix.std(axis=0, ddof=1, dtype=np.float64)
    scales = np.maximum(raw_scales, floor)
    return DirectStandardizer(
        methods=tuple(str(method) for method in methods),
        centers=np.asarray(centers, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        scale_floor=floor,
        fitting_batch_count=int(len(matrix)),
    )


def simultaneous_direct_score(
    errors: Mapping[str, float], standardizer: DirectStandardizer
) -> float:
    """Return max_a (e_a-c_a)/s_a for one paired batch."""

    if set(errors) != set(standardizer.methods):
        raise ValueError("errors must contain exactly the standardized methods")
    values = np.asarray(
        [float(errors[method]) for method in standardizer.methods],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("direct recovery errors must lie in [0,1]")
    return float(np.max((values - standardizer.centers) / standardizer.scales))


def calibrate_direct_recovery_bounds(
    fitting_batch_errors: Mapping[str, Sequence[float] | np.ndarray],
    calibration_batch_errors: Sequence[Mapping[str, float]],
    *,
    delta: float,
    methods: Sequence[str] = DIRECT_METHODS,
    scale_floor: float = 1.0 / 256.0,
) -> DirectCalibration:
    """Calibrate simultaneous future-batch upper bounds for all methods."""

    standardizer = fit_direct_standardizer(
        fitting_batch_errors, methods=methods, scale_floor=scale_floor
    )
    if not calibration_batch_errors:
        raise ValueError("at least one calibration batch is required")
    scores = np.asarray(
        [
            simultaneous_direct_score(errors, standardizer)
            for errors in calibration_batch_errors
        ],
        dtype=np.float64,
    )
    rank = conformal_rank(len(scores), float(delta))
    if rank > len(scores):
        threshold = math.inf
        upper = np.ones(len(standardizer.methods), dtype=np.float64)
    else:
        threshold = float(np.sort(scores, kind="stable")[rank - 1])
        upper = np.clip(
            standardizer.centers + standardizer.scales * threshold,
            0.0,
            1.0,
        )
    return DirectCalibration(
        standardizer=standardizer,
        threshold=threshold,
        rank_one_indexed=int(rank),
        calibration_scores=scores,
        upper_bounds=np.asarray(upper, dtype=np.float64),
        delta=float(delta),
    )
