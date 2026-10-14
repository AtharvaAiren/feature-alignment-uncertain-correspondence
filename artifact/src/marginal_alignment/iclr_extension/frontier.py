"""Prespecified label-free marginal-map frontier for Play 3.

Every map is fitted only to independently canonicalized target and reference
calibration feature multisets.  The public fitting and prediction APIs expose
no label, identifier, or pairing argument.  Labels or verified pairs may be
used later only to score the already-frozen predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from marginal_alignment.campaign.maps import (
    FittedMap,
    fit_coordinate_quantile,
    fit_covariance_gauge,
    fit_diagonal,
    fit_identity,
    fit_mean,
)
from marginal_alignment.iclr_extension.ceiling import _canonical_rows


MARGINAL_FRONTIER_METHODS = (
    "none",
    "mean",
    "diagonal",
    "coordinate_quantile_257",
    "coral_oas",
    "bures_oas",
)


@dataclass(frozen=True)
class FrozenMarginalFrontier:
    maps: Mapping[str, FittedMap]
    diagnostics: Mapping[str, Any]


def fit_marginal_frontier(
    target_calibration_features: np.ndarray,
    reference_calibration_features: np.ndarray,
    *,
    prefit_bures_map: FittedMap | None = None,
    quantile_knots: int = 257,
    covariance_relative_floor: float = 1e-6,
    diagonal_standard_deviation_floor: float = 1e-6,
) -> FrozenMarginalFrontier:
    """Fit the locked six-map panel without labels or correspondence."""

    target, _, target_sha = _canonical_rows(
        target_calibration_features,
        name="frontier_target_calibration",
        salt=b"play3-target-calibration-v1",
    )
    reference, _, reference_sha = _canonical_rows(
        reference_calibration_features,
        name="frontier_reference_calibration",
        salt=b"play3-reference-calibration-v1",
    )
    if target.shape[1] != reference.shape[1] or min(len(target), len(reference)) < 2:
        raise ValueError("frontier calibration marginals have an invalid shape")
    knots = int(quantile_knots)
    if knots != 257:
        raise ValueError("Play-3 coordinate quantile is locked to 257 knots")
    relative_floor = float(covariance_relative_floor)
    standard_deviation_floor = float(diagonal_standard_deviation_floor)
    if relative_floor <= 0.0 or standard_deviation_floor <= 0.0:
        raise ValueError("frontier numerical floors must be positive")

    maps: dict[str, FittedMap] = {
        "none": fit_identity(target, reference),
        "mean": fit_mean(target, reference),
        "diagonal": fit_diagonal(
            target,
            reference,
            standard_deviation_floor=standard_deviation_floor,
        ),
        "coordinate_quantile_257": fit_coordinate_quantile(
            target, reference, knots=knots
        ),
        "coral_oas": fit_covariance_gauge(
            target,
            reference,
            gauge="coral",
            estimator="oas",
            relative_floor=relative_floor,
        ),
        "bures_oas": (
            fit_covariance_gauge(
                target,
                reference,
                gauge="bures",
                estimator="oas",
                relative_floor=relative_floor,
            )
            if prefit_bures_map is None
            else prefit_bures_map
        ),
    }
    if tuple(maps) != MARGINAL_FRONTIER_METHODS:
        raise RuntimeError("marginal frontier method order drifted")
    return FrozenMarginalFrontier(
        maps=maps,
        diagnostics={
            "schema_version": "marginal_alignment.iclr_extension.frontier_fit.v1",
            "information_scope": "unpaired_calibration_features_only",
            "labels_used": False,
            "sample_identifiers_used": False,
            "pairing_used": False,
            "methods": list(MARGINAL_FRONTIER_METHODS),
            "quantile_knots": knots,
            "covariance_estimator": "oas",
            "covariance_relative_floor": relative_floor,
            "diagonal_standard_deviation_floor": standard_deviation_floor,
            "target_calibration_content_sha256": target_sha,
            "reference_calibration_content_sha256": reference_sha,
        },
    )


def frozen_frontier_predictions(
    frontier: FrozenMarginalFrontier,
    target_features: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply all frozen maps and readout, still without labels or pairing."""

    target = np.asarray(target_features, dtype=np.float64)
    weight = np.asarray(readout_weight, dtype=np.float64)
    bias = np.asarray(readout_bias, dtype=np.float64)
    if target.ndim != 2 or weight.ndim != 2 or weight.shape[1] != target.shape[1]:
        raise ValueError("frontier prediction dimensions differ")
    if bias.shape != (weight.shape[0],):
        raise ValueError("frontier readout bias has the wrong shape")
    predictions: dict[str, np.ndarray] = {}
    for method in MARGINAL_FRONTIER_METHODS:
        transformed = np.asarray(frontier.maps[method].apply(target), dtype=np.float64)
        if transformed.shape != target.shape or not np.all(np.isfinite(transformed)):
            raise FloatingPointError(f"nonfinite transformed features for {method}")
        predictions[method] = np.argmax(
            transformed @ weight.T + bias, axis=1
        ).astype(np.int64)
    return predictions
