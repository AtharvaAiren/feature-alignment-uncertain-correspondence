"""Finite-sample split-conformal calibration for batch correspondence costs."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def conformal_rank(calibration_count: int, delta: float) -> int:
    """Return the one-indexed split-conformal order-statistic rank.

    The rank is ``ceil((m + 1) * (1 - delta))``.  It is intentionally not
    clipped to ``m``: callers must use an infinite threshold when ``k > m``.
    """

    count = int(calibration_count)
    if count < 1:
        raise ValueError("split conformal needs at least one calibration score")
    level = float(delta)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    return int(math.ceil((count + 1) * (1.0 - level)))


def split_conformal_quantile(
    scores: Iterable[float], *, delta: float
) -> tuple[float, dict[str, int | float | str | bool]]:
    """Calibrate a conservative threshold, including the exact infinite case."""

    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be a nonempty finite vector")
    if np.any(values < 0.0):
        raise ValueError("normalized correspondence-cost scores must be nonnegative")
    rank = conformal_rank(len(values), delta)
    if rank > len(values):
        threshold = math.inf
        rule = "infinite_threshold_because_order_statistic_rank_exceeds_m"
    else:
        threshold = float(np.sort(values, kind="stable")[rank - 1])
        rule = "one_indexed_kth_smallest_with_k_ceil_m_plus_1_times_1_minus_delta"
    return threshold, {
        "calibration_count": int(len(values)),
        "delta": float(delta),
        "rank_one_indexed": int(rank),
        "rank_exceeds_calibration_count": bool(rank > len(values)),
        "quantile_rule": rule,
        "finite_threshold": bool(math.isfinite(threshold)),
    }
