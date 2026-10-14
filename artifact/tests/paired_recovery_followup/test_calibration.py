from __future__ import annotations

import numpy as np

from marginal_alignment.paired_recovery_followup.calibration import (
    calibrate_direct_recovery_bounds,
    fit_direct_standardizer,
    simultaneous_direct_score,
)


METHODS = ("identity", "diagonal", "bures")


def test_direct_baseline_uses_corrected_rank_and_shared_max_score() -> None:
    fitting = {
        method: np.linspace(0.20 + 0.02 * index, 0.35 + 0.02 * index, 16)
        for index, method in enumerate(METHODS)
    }
    calibration = []
    for batch in range(29):
        calibration.append({
            method: float(0.22 + 0.003 * batch + 0.02 * index)
            for index, method in enumerate(METHODS)
        })
    result = calibrate_direct_recovery_bounds(
        fitting,
        calibration,
        delta=0.1,
        methods=METHODS,
        scale_floor=1.0 / 256.0,
    )
    assert result.rank_one_indexed == 27
    assert result.threshold == np.sort(result.calibration_scores)[26]
    assert tuple(result.upper_by_method()) == METHODS
    assert np.all((0.0 <= result.upper_bounds) & (result.upper_bounds <= 1.0))
    for batch, score in zip(calibration, result.calibration_scores):
        assert simultaneous_direct_score(batch, result.standardizer) == score


def test_scale_floor_handles_constant_fitting_errors() -> None:
    fitting = {method: np.full(16, 0.25) for method in METHODS}
    fitted = fit_direct_standardizer(fitting, scale_floor=1.0 / 256.0)
    np.testing.assert_allclose(fitted.centers, 0.25)
    np.testing.assert_allclose(fitted.scales, 1.0 / 256.0)


def test_shared_score_event_implies_all_method_bounds() -> None:
    fitting = {
        "identity": np.linspace(0.50, 0.65, 16),
        "diagonal": np.linspace(0.40, 0.55, 16),
        "bures": np.linspace(0.39, 0.54, 16),
    }
    calibration = [
        {"identity": 0.55 + j / 1000, "diagonal": 0.47 + j / 1000, "bures": 0.46 + j / 1000}
        for j in range(29)
    ]
    result = calibrate_direct_recovery_bounds(
        fitting, calibration, delta=0.1, methods=METHODS
    )
    evaluation = {"identity": 0.56, "diagonal": 0.48, "bures": 0.47}
    assert simultaneous_direct_score(evaluation, result.standardizer) <= result.threshold
    for index, method in enumerate(METHODS):
        assert evaluation[method] <= result.upper_bounds[index]
