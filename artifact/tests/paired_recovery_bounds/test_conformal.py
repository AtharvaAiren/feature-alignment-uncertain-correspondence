from __future__ import annotations

import math

import pytest

from marginal_alignment.paired_recovery_bounds.conformal import (
    conformal_rank,
    split_conformal_quantile,
)


def test_locked_rank_is_twenty_seven() -> None:
    assert conformal_rank(29, 0.1) == 27
    qhat, diagnostics = split_conformal_quantile(range(29), delta=0.1)
    assert qhat == 26.0
    assert diagnostics["rank_one_indexed"] == 27
    assert diagnostics["finite_threshold"] is True


def test_rank_beyond_m_is_infinite_not_clipped() -> None:
    qhat, diagnostics = split_conformal_quantile([0.1, 0.2, 0.3, 0.4], delta=0.01)
    assert math.isinf(qhat)
    assert diagnostics["rank_one_indexed"] == 5
    assert diagnostics["rank_exceeds_calibration_count"] is True


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, float("nan")])
def test_invalid_delta_fails(delta: float) -> None:
    with pytest.raises(ValueError, match="delta"):
        split_conformal_quantile([1.0], delta=delta)
