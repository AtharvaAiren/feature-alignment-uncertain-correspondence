"""Bures-first orientation recovery from unpaired, soft pseudo-label anchors.

This module is intentionally outside the locked ICLR campaign.  It implements
the constructive complement to the covariance-ambiguity result without
changing any prespecified result: Bures first fixes the fitted first and second
moments, then a low-rank orthogonal orientation is estimated in reference-
whitened coordinates.  The correction uses only the frozen classifier's soft
predictions on two *unpaired* calibration marginals.

The estimator is conditional, not a universal escape from the impossibility
result.  It can resolve an orientation only when the frozen classifier supplies
several reliable semantic anchors.  All rank and regularization choices are
made on held-out calibration rows, and every numerical or identification
failure returns the exact Bures map (an identity orthogonal orientation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable

import numpy as np

from marginal_alignment.campaign.maps import (
    AffineMap,
    estimate_covariance,
    fit_covariance_gauge,
)
from marginal_alignment.linalg import as_samples, symmetric_power


_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class BuresPseudoOrientationMap:
    """An affine Bures map followed by a covariance-preserving orientation.

    ``orientation`` acts in reference-whitened coordinates and may be a
    rotation or a reflection.  When calibration
    does not identify a useful correction it is exactly the identity, and
    ``matrix``/``offset`` are copied from ``base_map`` without recomposition.
    """

    base_map: AffineMap
    matrix: np.ndarray
    offset: np.ndarray
    orientation: np.ndarray
    basis: np.ndarray
    subspace_rotation: np.ndarray
    method: str = "bures_pseudo_orientation"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(self, features: np.ndarray) -> np.ndarray:
        features = as_samples(features)
        if features.shape[1] != self.matrix.shape[0]:
            raise ValueError("feature dimension does not match fitted map")
        return features @ self.matrix + self.offset


@dataclass(frozen=True)
class _AnchorSummary:
    centroids: np.ndarray
    mass: np.ndarray
    retained_rows: int
    mean_confidence: float


def _validate_hyperparameters(
    *,
    candidate_ranks: Iterable[int],
    procrustes_shrinkages: Iterable[float],
    confidence_threshold: float,
    temperature: float,
    validation_fraction: float,
    minimum_fit_rows: int,
    minimum_validation_rows: int,
    minimum_class_mass: float,
    minimum_validation_class_mass: float,
    minimum_active_classes: int,
    minimum_effective_class_mass: float,
    minimum_explained_discrepancy_energy: float,
    minimum_subspace_stability: float,
    null_permutations: int,
    null_quantile: float,
    minimum_relative_improvement: float,
    orthogonality_tolerance: float,
    covariance_tolerance: float,
    relative_floor: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    raw_ranks = tuple(candidate_ranks)
    if any(
        isinstance(value, (bool, np.bool_))
        or not np.isfinite(float(value))
        or float(value) != np.floor(float(value))
        for value in raw_ranks
    ):
        raise ValueError("candidate_ranks must contain positive integers")
    ranks = tuple(sorted(set(int(value) for value in raw_ranks)))
    shrinkages = tuple(
        sorted(set(float(value) for value in procrustes_shrinkages))
    )
    if not ranks or any(value < 1 for value in ranks):
        raise ValueError("candidate_ranks must contain positive integers")
    if not shrinkages or any(
        value < 0.0 or not np.isfinite(value) for value in shrinkages
    ):
        raise ValueError("procrustes_shrinkages must contain finite nonnegative values")
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must lie in [0,1)")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if min(minimum_fit_rows, minimum_validation_rows, minimum_active_classes) < 1:
        raise ValueError("minimum row and active-class counts must be positive")
    if (
        isinstance(null_permutations, (bool, np.bool_))
        or int(null_permutations) != null_permutations
        or int(null_permutations) < 1
    ):
        raise ValueError("null_permutations must be a positive integer")
    if not np.all(np.isfinite([
        minimum_class_mass,
        minimum_validation_class_mass,
        minimum_effective_class_mass,
        minimum_explained_discrepancy_energy,
        minimum_subspace_stability,
        null_quantile,
        minimum_relative_improvement,
        orthogonality_tolerance,
        covariance_tolerance,
        relative_floor,
    ])):
        raise ValueError("mass, improvement, floor, and tolerance values must be finite")
    if min(minimum_class_mass, minimum_validation_class_mass) <= 0.0:
        raise ValueError("minimum class masses must be positive")
    if minimum_effective_class_mass < 1.0:
        raise ValueError("minimum_effective_class_mass must be at least one")
    if not 0.0 < minimum_explained_discrepancy_energy <= 1.0:
        raise ValueError("minimum_explained_discrepancy_energy must lie in (0,1]")
    if not 0.0 <= minimum_subspace_stability <= 1.0:
        raise ValueError("minimum_subspace_stability must lie in [0,1]")
    if not 0.0 < null_quantile < 1.0:
        raise ValueError("null_quantile must lie strictly between zero and one")
    if minimum_relative_improvement < 0.0:
        raise ValueError("minimum_relative_improvement must be nonnegative")
    if min(orthogonality_tolerance, covariance_tolerance) <= 0.0:
        raise ValueError("numerical tolerances must be positive")
    if relative_floor <= 0.0:
        raise ValueError("relative_floor must be positive")
    return ranks, shrinkages


def _content_partition(
    features: np.ndarray,
    *,
    validation_fraction: float,
    salt: bytes,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split rows deterministically without using their input positions.

    Source and reference use different salts, so coincident row ordering cannot
    manufacture calibration pairs.  Hash collisions are broken by the row
    bytes themselves; exact duplicate rows are statistically interchangeable.
    """

    canonical = np.ascontiguousarray(features, dtype="<f8")
    keys: list[tuple[bytes, bytes, int]] = []
    for index, row in enumerate(canonical):
        raw = row.tobytes()
        keys.append((hashlib.sha256(salt + raw).digest(), raw, index))
    keys.sort(key=lambda item: (item[0], item[1]))
    validation_rows = int(round(validation_fraction * len(features)))
    validation_rows = min(max(validation_rows, 1), len(features) - 1)
    fit_indices = np.asarray(
        [item[2] for item in keys[:-validation_rows]], dtype=np.int64
    )
    validation_indices = np.asarray(
        [item[2] for item in keys[-validation_rows:]], dtype=np.int64
    )
    digest = hashlib.sha256()
    digest.update(salt)
    for key, raw, _ in keys:
        digest.update(key)
        digest.update(raw)
    return fit_indices, validation_indices, digest.hexdigest()


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    if scaled.ndim != 2 or not np.all(np.isfinite(scaled)):
        raise FloatingPointError("classifier logits are not a finite matrix")
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    denominator = exponentiated.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("softmax normalization is non-finite")
    return exponentiated / denominator


def _anchor_summary(
    whitened_features: np.ndarray,
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
) -> _AnchorSummary:
    confidence = probabilities.max(axis=1)
    retained = confidence >= confidence_threshold
    # Soft responsibilities avoid the discontinuity of hard pseudo-labels;
    # confidence weighting keeps boundary points from dominating an anchor.
    weights = probabilities * (confidence * retained)[:, None]
    mass = weights.sum(axis=0)
    centroids = np.zeros((probabilities.shape[1], whitened_features.shape[1]))
    positive = mass > 0.0
    centroids[positive] = (
        weights[:, positive].T @ whitened_features
    ) / mass[positive, None]
    mean_confidence = (
        float(confidence[retained].mean()) if np.any(retained) else 0.0
    )
    return _AnchorSummary(
        centroids=centroids,
        mass=mass,
        retained_rows=int(np.sum(retained)),
        mean_confidence=mean_confidence,
    )


def _common_anchor_geometry(
    source: _AnchorSummary,
    reference: _AnchorSummary,
    *,
    minimum_mass: float,
    minimum_effective_mass: float,
    minimum_active_classes: int,
    class_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = (source.mass >= minimum_mass) & (reference.mass >= minimum_mass)
    if class_ids is not None:
        requested = np.zeros_like(active)
        requested[np.asarray(class_ids, dtype=np.int64)] = True
        active &= requested
    source_mass = source.mass
    reference_mass = reference.mass
    effective_mass = np.zeros_like(source_mass)
    positive = (source_mass > 0.0) & (reference_mass > 0.0)
    effective_mass[positive] = (
        2.0 * source_mass[positive] * reference_mass[positive]
        / (source_mass[positive] + reference_mass[positive])
    )
    active &= effective_mass >= minimum_effective_mass
    indices = np.flatnonzero(active)
    if len(indices) < minimum_active_classes:
        raise ValueError(
            f"only {len(indices)} common confident classes; "
            f"at least {minimum_active_classes} are required"
        )
    harmonic_mass = effective_mass[indices]
    weights = harmonic_mass / harmonic_mass.sum()
    source_centroids = source.centroids[indices]
    reference_centroids = reference.centroids[indices]
    source_centroids = source_centroids - weights @ source_centroids
    reference_centroids = reference_centroids - weights @ reference_centroids
    return indices, source_centroids, reference_centroids, weights, harmonic_mass


def _weighted_discrepancy(
    source_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    weights: np.ndarray,
    orientation: np.ndarray,
) -> float:
    residual = source_centroids @ orientation - reference_centroids
    numerator = float(np.sum(weights[:, None] * np.square(residual)))
    scale = max(
        float(np.sum(weights[:, None] * np.square(reference_centroids))),
        _EPSILON,
    )
    return numerator / scale


def _discrepancy_subspace(
    source_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the pseudo-anchor discrepancy spectrum and cumulative energy."""

    weighted = np.sqrt(weights)[:, None] * (
        source_centroids - reference_centroids
    )
    _, spectrum, right_t = np.linalg.svd(weighted, full_matrices=False)
    squared = np.square(spectrum)
    total = float(np.sum(squared))
    cumulative = (
        np.cumsum(squared) / total
        if total > np.finfo(np.float64).tiny
        else np.zeros_like(squared)
    )
    return spectrum, cumulative, right_t, total


def _subspace_stability(
    fit_right_t: np.ndarray,
    validation_right_t: np.ndarray,
    rank: int,
) -> float:
    if rank > min(len(fit_right_t), len(validation_right_t)):
        return 0.0
    overlap = fit_right_t[:rank] @ validation_right_t[:rank].T
    value = float(np.sum(np.square(overlap)) / rank)
    return float(np.clip(value, 0.0, 1.0))


def _fit_orthogonal_orientation(
    source_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    weights: np.ndarray,
    *,
    basis: np.ndarray,
    shrinkage: float,
    orthogonality_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rank = basis.shape[1]
    projected_source = source_centroids @ basis
    projected_reference = reference_centroids @ basis
    root_weights = np.sqrt(weights)[:, None]
    cross = (root_weights * projected_source).T @ (
        root_weights * projected_reference
    )
    cross_scale = max(
        float(np.linalg.norm(cross, ord="fro") / np.sqrt(rank)),
        np.finfo(np.float64).tiny,
    )
    regularized_cross = cross + shrinkage * cross_scale * np.eye(rank)
    left, singular_values, right_t = np.linalg.svd(
        regularized_cross, full_matrices=False
    )
    subspace_orientation = left @ right_t
    dimension = basis.shape[0]
    identity = np.eye(dimension)
    orientation = identity + basis @ (
        subspace_orientation - np.eye(rank)
    ) @ basis.T
    orthogonality_error = float(
        np.linalg.norm(orientation.T @ orientation - identity, ord="fro")
    )
    if not np.isfinite(orthogonality_error):
        raise FloatingPointError("orientation orthogonality error is non-finite")
    if orthogonality_error > orthogonality_tolerance:
        raise FloatingPointError(
            "orientation violates the declared orthogonality tolerance"
        )
    return orientation, subspace_orientation, singular_values, orthogonality_error


def _deterministic_permutation(length: int, *, replicate: int, salt: bytes) -> np.ndarray:
    digest = hashlib.sha256(
        salt + int(length).to_bytes(8, "little") + int(replicate).to_bytes(8, "little")
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed).permutation(length)


def _higher_quantile(values: Iterable[float], probability: float) -> float:
    ordered = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if ordered.ndim != 1 or not len(ordered) or not np.all(np.isfinite(ordered)):
        raise FloatingPointError("null improvements must be finite and nonempty")
    index = int(np.ceil(probability * (len(ordered) - 1)))
    return float(ordered[index])


def _empirical_covariance(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0)
    return centered.T @ centered / len(centered)


def _identity_fallback(
    base_map: AffineMap,
    *,
    reason: str,
    diagnostics: dict[str, Any],
) -> BuresPseudoOrientationMap:
    dimension = base_map.matrix.shape[0]
    complete = {
        **diagnostics,
        "status": "FALLBACK",
        "fallback_used": True,
        "fallback_reason": reason,
        "selected_rank": 0,
        "selected_shrinkage": None,
        "orientation_orthogonality_error": 0.0,
        "orientation_determinant": 1.0,
        "orientation_includes_reflection": False,
        "orthogonal_orientation_refit_on_all_calibration": False,
        "fitted_covariance_relative_error": float(
            base_map.diagnostics["fitted_covariance_relative_error"]
        ),
        "fitted_regularized_covariance_relative_error": float(
            base_map.diagnostics["fitted_covariance_relative_error"]
        ),
    }
    return BuresPseudoOrientationMap(
        base_map=base_map,
        matrix=np.array(base_map.matrix, copy=True),
        offset=np.array(base_map.offset, copy=True),
        orientation=np.eye(dimension),
        basis=np.zeros((dimension, 0)),
        subspace_rotation=np.zeros((0, 0)),
        diagnostics=complete,
    )


def fit_bures_pseudo_orientation(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray | None = None,
    covariance_estimator: str | float = "oas",
    relative_floor: float = 1e-6,
    candidate_ranks: Iterable[int] = (2, 4, 8, 16, 32),
    procrustes_shrinkages: Iterable[float] = (0.1, 1.0, 10.0),
    confidence_threshold: float = 0.8,
    temperature: float = 1.0,
    validation_fraction: float = 0.25,
    minimum_fit_rows: int = 64,
    minimum_validation_rows: int = 32,
    minimum_class_mass: float = 4.0,
    minimum_validation_class_mass: float = 1.0,
    minimum_active_classes: int = 8,
    minimum_effective_class_mass: float = 1.0,
    minimum_explained_discrepancy_energy: float = 0.5,
    minimum_subspace_stability: float = 0.05,
    null_permutations: int = 19,
    null_quantile: float = 0.95,
    minimum_relative_improvement: float = 0.01,
    orthogonality_tolerance: float = 1e-8,
    covariance_tolerance: float = 1e-6,
) -> BuresPseudoOrientationMap:
    """Fit Bures plus a held-out-selected pseudo-anchor orientation.

    Parameters contain calibration marginals and a frozen readout only.  There
    is deliberately no interface for labels, sample identifiers, row pairs, or
    evaluation data.  Bures whitening and candidate orientations are fit only
    on the orientation-fit partition. Rank and regularization are selected on
    the disjoint validation partition against deterministic responsibility-
    permutation nulls. The selected specification is then refit on all
    calibration rows before evaluation.
    """

    ranks, shrinkages = _validate_hyperparameters(
        candidate_ranks=candidate_ranks,
        procrustes_shrinkages=procrustes_shrinkages,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        validation_fraction=validation_fraction,
        minimum_fit_rows=minimum_fit_rows,
        minimum_validation_rows=minimum_validation_rows,
        minimum_class_mass=minimum_class_mass,
        minimum_validation_class_mass=minimum_validation_class_mass,
        minimum_active_classes=minimum_active_classes,
        minimum_effective_class_mass=minimum_effective_class_mass,
        minimum_explained_discrepancy_energy=minimum_explained_discrepancy_energy,
        minimum_subspace_stability=minimum_subspace_stability,
        null_permutations=null_permutations,
        null_quantile=null_quantile,
        minimum_relative_improvement=minimum_relative_improvement,
        orthogonality_tolerance=orthogonality_tolerance,
        covariance_tolerance=covariance_tolerance,
        relative_floor=relative_floor,
    )
    source = as_samples(source, name="source")
    reference = as_samples(reference, name="reference")
    if source.shape[1] != reference.shape[1]:
        raise ValueError("source and reference feature dimensions differ")
    dimension = source.shape[1]
    weight = np.asarray(readout_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[1] != dimension or weight.shape[0] < 2:
        raise ValueError(
            "readout_weight must have shape [at_least_two_classes,feature_dimension]"
        )
    if readout_bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float64)
    else:
        bias = np.asarray(readout_bias, dtype=np.float64)
    if bias.shape != (weight.shape[0],):
        raise ValueError("readout_bias must have one value per classifier output")

    # The all-calibration Bures map is the exact fail-closed baseline and is
    # used only after candidate selection. Selection gets its own fit-only map.
    base_map = fit_covariance_gauge(
        source,
        reference,
        gauge="bures",
        estimator=covariance_estimator,
        relative_floor=relative_floor,
    )
    reference_mean, reference_covariance, _ = estimate_covariance(
        reference, covariance_estimator, relative_floor
    )
    _, source_covariance, _ = estimate_covariance(
        source, covariance_estimator, relative_floor
    )
    empirical_source_covariance = _empirical_covariance(source)
    empirical_reference_covariance = _empirical_covariance(reference)
    base_empirical_covariance_error = float(
        np.linalg.norm(
            base_map.matrix.T @ empirical_source_covariance @ base_map.matrix
            - empirical_reference_covariance,
            ord="fro",
        )
        / max(
            float(np.linalg.norm(empirical_reference_covariance, ord="fro")),
            _EPSILON,
        )
    )
    common_diagnostics: dict[str, Any] = {
        "schema_version": "beyond_marginals.pseudo_orientation.v2",
        "information_tier": "unpaired_marginals_plus_frozen_predictions",
        "base_method": "bures",
        "covariance_estimator": str(covariance_estimator),
        "correction_type": "low_rank_pseudo_anchor_orthogonal_orientation",
        "reflection_permitted": True,
        "labels_used": False,
        "pairing_used": False,
        "evaluation_data_used_for_selection": False,
        "selection_data": "orientation_fit_and_disjoint_validation_unpaired_calibration",
        "selection_bures_and_whitening_fit_rows_only": True,
        "validation_used_for_selection_only": True,
        "selected_specification_refit_policy": "all_calibration_rows_before_evaluation",
        "source_rows": int(len(source)),
        "reference_rows": int(len(reference)),
        "feature_dimension": int(dimension),
        "classes": int(weight.shape[0]),
        "confidence_threshold": float(confidence_threshold),
        "temperature": float(temperature),
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "minimum_effective_class_mass": float(minimum_effective_class_mass),
        "rank_support_rule": "active_classes_at_least_max_8_2r_plus_1",
        "subspace_name": "low_rank_pseudo_anchor_discrepancy_subspace",
        "minimum_explained_discrepancy_energy": float(
            minimum_explained_discrepancy_energy
        ),
        "minimum_fit_validation_subspace_stability": float(
            minimum_subspace_stability
        ),
        "null_calibration": {
            "kind": "deterministic_independent_fit_responsibility_row_permutations_with_fixed_validation",
            "permutations": int(null_permutations),
            "quantile": float(null_quantile),
            "acceptance_rule": "observed_gain_strictly_exceeds_max_1pct_and_null_quantile_plus_1pct",
            "interpretation": (
                "conservative_heuristic_selection_gate_not_a_significance_test_"
                "or_identification_claim"
            ),
        },
        "candidate_ranks": list(ranks),
        "candidate_shrinkages": list(shrinkages),
        "base_diagnostics": base_map.diagnostics,
        "covariance_preservation_scope": "fitted_regularized_covariances",
        "fitted_oas_covariance_relative_error": float(
            base_map.diagnostics["fitted_covariance_relative_error"]
        ) if str(covariance_estimator) == "oas" else None,
        "empirical_covariance_relative_error": base_empirical_covariance_error,
        "candidate_records": [],
    }

    if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
        return _identity_fallback(
            base_map,
            reason="nonfinite_frozen_readout",
            diagnostics=common_diagnostics,
        )
    if len(source) < minimum_fit_rows + minimum_validation_rows:
        return _identity_fallback(
            base_map,
            reason="insufficient_source_calibration_rows",
            diagnostics=common_diagnostics,
        )
    if len(reference) < minimum_fit_rows + minimum_validation_rows:
        return _identity_fallback(
            base_map,
            reason="insufficient_reference_calibration_rows",
            diagnostics=common_diagnostics,
        )

    try:
        source_fit, source_validation, source_digest = _content_partition(
            source,
            validation_fraction=validation_fraction,
            salt=b"pseudo-orientation-source-v1",
        )
        reference_fit, reference_validation, reference_digest = _content_partition(
            reference,
            validation_fraction=validation_fraction,
            salt=b"pseudo-orientation-reference-v1",
        )
        if (
            len(source_fit) < minimum_fit_rows
            or len(source_validation) < minimum_validation_rows
        ):
            raise ValueError("source content split violates minimum row counts")
        if (
            len(reference_fit) < minimum_fit_rows
            or len(reference_validation) < minimum_validation_rows
        ):
            raise ValueError("reference content split violates minimum row counts")

        selection_base_map = fit_covariance_gauge(
            source[source_fit],
            reference[reference_fit],
            gauge="bures",
            estimator=covariance_estimator,
            relative_floor=relative_floor,
        )
        selection_reference_mean, selection_reference_covariance, _ = (
            estimate_covariance(
                reference[reference_fit], covariance_estimator, relative_floor
            )
        )
        selection_reference_inverse_sqrt = symmetric_power(
            selection_reference_covariance, -0.5
        )
        mapped_source_fit = selection_base_map.apply(source[source_fit])
        mapped_source_validation = selection_base_map.apply(source[source_validation])
        whitened_source_fit = (
            mapped_source_fit - selection_reference_mean
        ) @ selection_reference_inverse_sqrt
        whitened_source_validation = (
            mapped_source_validation - selection_reference_mean
        ) @ selection_reference_inverse_sqrt
        whitened_reference_fit = (
            reference[reference_fit] - selection_reference_mean
        ) @ selection_reference_inverse_sqrt
        whitened_reference_validation = (
            reference[reference_validation] - selection_reference_mean
        ) @ selection_reference_inverse_sqrt
        source_fit_probability = _softmax(
            mapped_source_fit @ weight.T + bias, temperature
        )
        source_validation_probability = _softmax(
            mapped_source_validation @ weight.T + bias, temperature
        )
        reference_fit_probability = _softmax(
            reference[reference_fit] @ weight.T + bias, temperature
        )
        reference_validation_probability = _softmax(
            reference[reference_validation] @ weight.T + bias, temperature
        )

        def candidate_search(
            fit_source_probability: np.ndarray,
            fit_reference_probability: np.ndarray,
            validation_source_probability: np.ndarray,
            validation_reference_probability: np.ndarray,
            *,
            capture_records: bool,
        ) -> dict[str, Any]:
            source_fit_summary = _anchor_summary(
                whitened_source_fit,
                fit_source_probability,
                confidence_threshold=confidence_threshold,
            )
            reference_fit_summary = _anchor_summary(
                whitened_reference_fit,
                fit_reference_probability,
                confidence_threshold=confidence_threshold,
            )
            (
                active_fit,
                fit_source_centroids,
                fit_reference_centroids,
                fit_weights,
                fit_effective_mass,
            ) = _common_anchor_geometry(
                source_fit_summary,
                reference_fit_summary,
                minimum_mass=minimum_class_mass,
                minimum_effective_mass=minimum_effective_class_mass,
                minimum_active_classes=max(8, minimum_active_classes),
            )
            source_validation_summary = _anchor_summary(
                whitened_source_validation,
                validation_source_probability,
                confidence_threshold=confidence_threshold,
            )
            reference_validation_summary = _anchor_summary(
                whitened_reference_validation,
                validation_reference_probability,
                confidence_threshold=confidence_threshold,
            )
            (
                active_validation,
                validation_source_centroids,
                validation_reference_centroids,
                validation_weights,
                validation_effective_mass,
            ) = _common_anchor_geometry(
                source_validation_summary,
                reference_validation_summary,
                minimum_mass=minimum_validation_class_mass,
                minimum_effective_mass=minimum_effective_class_mass,
                minimum_active_classes=max(8, minimum_active_classes),
                class_ids=active_fit,
            )
            fit_spectrum, fit_energy, fit_right_t, fit_total = (
                _discrepancy_subspace(
                    fit_source_centroids, fit_reference_centroids, fit_weights
                )
            )
            validation_spectrum, validation_energy, validation_right_t, validation_total = (
                _discrepancy_subspace(
                    validation_source_centroids,
                    validation_reference_centroids,
                    validation_weights,
                )
            )
            if fit_total <= 0.0 or validation_total <= 0.0:
                raise ValueError("pseudo-anchor discrepancy energy is exactly zero")
            identity = np.eye(dimension)
            baseline = _weighted_discrepancy(
                validation_source_centroids,
                validation_reference_centroids,
                validation_weights,
                identity,
            )
            if not np.isfinite(baseline):
                raise FloatingPointError("baseline validation objective is non-finite")
            best: tuple[float, int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
            records: list[dict[str, Any]] = []
            for rank in ranks:
                support_required = max(8, minimum_active_classes, 2 * rank + 1)
                rejection: str | None = None
                if min(len(active_fit), len(active_validation)) < support_required:
                    rejection = "rank_dependent_anchor_support_not_met"
                elif rank > min(len(fit_spectrum), len(validation_spectrum), dimension):
                    rejection = "rank_exceeds_discrepancy_subspace_dimension"
                elif float(fit_energy[rank - 1]) < minimum_explained_discrepancy_energy:
                    rejection = "fit_explained_discrepancy_energy_below_threshold"
                stability = _subspace_stability(
                    fit_right_t, validation_right_t, rank
                )
                if rejection is None and stability < minimum_subspace_stability:
                    rejection = "fit_validation_discrepancy_subspace_unstable"
                if rejection is not None:
                    if capture_records:
                        records.append({
                            "rank": int(rank),
                            "status": "REJECTED",
                            "reason": rejection,
                            "required_active_classes": int(support_required),
                            "fit_active_classes": int(len(active_fit)),
                            "validation_active_classes": int(len(active_validation)),
                            "fit_explained_discrepancy_energy": (
                                float(fit_energy[rank - 1])
                                if rank <= len(fit_energy) else None
                            ),
                            "validation_explained_discrepancy_energy": (
                                float(validation_energy[rank - 1])
                                if rank <= len(validation_energy) else None
                            ),
                            "fit_validation_subspace_stability": stability,
                        })
                    continue
                basis = fit_right_t[:rank].T
                for shrinkage in shrinkages:
                    try:
                        orientation, subspace_orientation, singular_values, error = (
                            _fit_orthogonal_orientation(
                                fit_source_centroids,
                                fit_reference_centroids,
                                fit_weights,
                                basis=basis,
                                shrinkage=shrinkage,
                                orthogonality_tolerance=orthogonality_tolerance,
                            )
                        )
                        objective = _weighted_discrepancy(
                            validation_source_centroids,
                            validation_reference_centroids,
                            validation_weights,
                            orientation,
                        )
                        if not np.isfinite(objective):
                            raise FloatingPointError(
                                "candidate validation objective is non-finite"
                            )
                        gain = float(
                            (baseline - objective) / max(baseline, _EPSILON)
                        )
                        if capture_records:
                            records.append({
                                "rank": int(rank),
                                "shrinkage": float(shrinkage),
                                "status": "PASS",
                                "validation_objective": objective,
                                "sign_normalized_validation_improvement": gain,
                                "orientation_harms_validation_objective": bool(gain < 0.0),
                                "required_active_classes": int(support_required),
                                "fit_explained_discrepancy_energy": float(
                                    fit_energy[rank - 1]
                                ),
                                "validation_explained_discrepancy_energy": float(
                                    validation_energy[rank - 1]
                                ),
                                "fit_validation_subspace_stability": stability,
                                "orthogonality_error": error,
                                "regularized_cross_singular_values": (
                                    singular_values.tolist()
                                ),
                                "orientation_determinant": float(
                                    np.linalg.det(orientation)
                                ),
                            })
                        candidate = (
                            objective,
                            int(rank),
                            float(shrinkage),
                            basis,
                            subspace_orientation,
                            orientation,
                        )
                        if best is None or candidate[:3] < best[:3]:
                            best = candidate
                    except (FloatingPointError, np.linalg.LinAlgError) as error:
                        if capture_records:
                            records.append({
                                "rank": int(rank),
                                "shrinkage": float(shrinkage),
                                "status": "REJECTED",
                                "reason": f"{type(error).__name__}: {error}",
                            })
            return {
                "best": best,
                "baseline": baseline,
                "records": records,
                "active_fit": active_fit,
                "active_validation": active_validation,
                "fit_effective_mass": fit_effective_mass,
                "validation_effective_mass": validation_effective_mass,
                "fit_summary": source_fit_summary,
                "reference_fit_summary": reference_fit_summary,
                "validation_summary": source_validation_summary,
                "reference_validation_summary": reference_validation_summary,
                "fit_spectrum": fit_spectrum,
                "fit_energy": fit_energy,
                "validation_spectrum": validation_spectrum,
                "validation_energy": validation_energy,
            }

        observed = candidate_search(
            source_fit_probability,
            reference_fit_probability,
            source_validation_probability,
            reference_validation_probability,
            capture_records=True,
        )
        common_diagnostics.update({
            "source_partition_sha256": source_digest,
            "reference_partition_sha256": reference_digest,
            "source_fit_rows": int(len(source_fit)),
            "source_validation_rows": int(len(source_validation)),
            "reference_fit_rows": int(len(reference_fit)),
            "reference_validation_rows": int(len(reference_validation)),
            "selection_base_map_fit_source_rows": int(len(source_fit)),
            "selection_base_map_fit_reference_rows": int(len(reference_fit)),
            "selection_base_diagnostics": selection_base_map.diagnostics,
            "source_fit_confident_rows": observed["fit_summary"].retained_rows,
            "reference_fit_confident_rows": observed["reference_fit_summary"].retained_rows,
            "source_validation_confident_rows": observed["validation_summary"].retained_rows,
            "reference_validation_confident_rows": (
                observed["reference_validation_summary"].retained_rows
            ),
            "source_fit_mean_confidence": observed["fit_summary"].mean_confidence,
            "reference_fit_mean_confidence": observed["reference_fit_summary"].mean_confidence,
            "active_fit_classes": observed["active_fit"].tolist(),
            "active_validation_classes": observed["active_validation"].tolist(),
            "fit_effective_anchor_mass_min": float(
                np.min(observed["fit_effective_mass"])
            ),
            "fit_effective_anchor_mass_median": float(
                np.median(observed["fit_effective_mass"])
            ),
            "validation_effective_anchor_mass_min": float(
                np.min(observed["validation_effective_mass"])
            ),
            "validation_effective_anchor_mass_median": float(
                np.median(observed["validation_effective_mass"])
            ),
            "fit_anchor_discrepancy_singular_values": observed["fit_spectrum"].tolist(),
            "fit_anchor_discrepancy_cumulative_energy": observed["fit_energy"].tolist(),
            "validation_anchor_discrepancy_singular_values": (
                observed["validation_spectrum"].tolist()
            ),
            "validation_anchor_discrepancy_cumulative_energy": (
                observed["validation_energy"].tolist()
            ),
            "baseline_validation_objective": float(observed["baseline"]),
            "candidate_records": observed["records"],
        })
        if observed["best"] is None:
            raise ValueError("no supported stable orthogonal-orientation candidate")

        objective, rank, shrinkage, _, _, _ = observed["best"]
        relative_improvement = float(
            (observed["baseline"] - objective)
            / max(float(observed["baseline"]), _EPSILON)
        )
        null_improvements: list[float] = []
        for replicate in range(int(null_permutations)):
            null = candidate_search(
                source_fit_probability[_deterministic_permutation(
                    len(source_fit_probability),
                    replicate=replicate,
                    salt=b"play1-null-source-fit-v1",
                )],
                reference_fit_probability[_deterministic_permutation(
                    len(reference_fit_probability),
                    replicate=replicate,
                    salt=b"play1-null-reference-fit-v1",
                )],
                source_validation_probability,
                reference_validation_probability,
                capture_records=False,
            )
            if null["best"] is None:
                null_improvements.append(0.0)
            else:
                null_objective = float(null["best"][0])
                null_improvements.append(float(
                    (null["baseline"] - null_objective)
                    / max(float(null["baseline"]), _EPSILON)
                ))
        null_threshold = _higher_quantile(null_improvements, null_quantile)
        required_improvement = max(
            float(minimum_relative_improvement),
            float(null_threshold + minimum_relative_improvement),
        )
        common_diagnostics.update({
            "selected_validation_objective": float(objective),
            "validation_relative_improvement": relative_improvement,
            "validation_sign_normalized_improvement": relative_improvement,
            "orientation_harms_validation_objective": bool(
                relative_improvement < 0.0
            ),
            "null_relative_improvements": null_improvements,
            "null_improvement_quantile": null_threshold,
            "required_null_calibrated_relative_improvement": required_improvement,
            "null_calibrated_acceptance_margin": float(
                relative_improvement - required_improvement
            ),
        })
        if not relative_improvement > required_improvement:
            return _identity_fallback(
                base_map,
                reason="held_out_gain_does_not_exceed_permutation_null_plus_margin",
                diagnostics=common_diagnostics,
            )

        # Refit the selected rank and shrinkage on all calibration rows. No
        # candidate is re-selected here.
        reference_inverse_sqrt = symmetric_power(reference_covariance, -0.5)
        reference_sqrt = symmetric_power(reference_covariance, 0.5)
        mapped_source = base_map.apply(source)
        whitened_source = (
            mapped_source - reference_mean
        ) @ reference_inverse_sqrt
        whitened_reference = (
            reference - reference_mean
        ) @ reference_inverse_sqrt
        source_probability = _softmax(
            mapped_source @ weight.T + bias, temperature
        )
        reference_probability = _softmax(
            reference @ weight.T + bias, temperature
        )
        source_all_summary = _anchor_summary(
            whitened_source,
            source_probability,
            confidence_threshold=confidence_threshold,
        )
        reference_all_summary = _anchor_summary(
            whitened_reference,
            reference_probability,
            confidence_threshold=confidence_threshold,
        )
        (
            refit_active_classes,
            refit_source_centroids,
            refit_reference_centroids,
            refit_weights,
            refit_effective_mass,
        ) = _common_anchor_geometry(
            source_all_summary,
            reference_all_summary,
            minimum_mass=minimum_class_mass,
            minimum_effective_mass=minimum_effective_class_mass,
            minimum_active_classes=max(8, minimum_active_classes, 2 * rank + 1),
        )
        refit_spectrum, refit_energy, refit_right_t, refit_total = (
            _discrepancy_subspace(
                refit_source_centroids, refit_reference_centroids, refit_weights
            )
        )
        if refit_total <= 0.0 or rank > len(refit_spectrum):
            raise ValueError("selected discrepancy rank is unavailable on refit")
        if float(refit_energy[rank - 1]) < minimum_explained_discrepancy_energy:
            raise ValueError("selected rank loses explained discrepancy energy on refit")
        basis = refit_right_t[:rank].T
        orientation, subspace_rotation, refit_cross_spectrum, orthogonality_error = (
            _fit_orthogonal_orientation(
                refit_source_centroids,
                refit_reference_centroids,
                refit_weights,
                basis=basis,
                shrinkage=shrinkage,
                orthogonality_tolerance=orthogonality_tolerance,
            )
        )
        whitened_correction = reference_inverse_sqrt @ orientation @ reference_sqrt
        matrix = base_map.matrix @ whitened_correction
        offset = (base_map.offset - reference_mean) @ whitened_correction + reference_mean
        mapped_covariance = matrix.T @ source_covariance @ matrix
        fitted_covariance_error = float(
            np.linalg.norm(mapped_covariance - reference_covariance, ord="fro")
            / max(float(np.linalg.norm(reference_covariance, ord="fro")), _EPSILON)
        )
        mapped_empirical_covariance = (
            matrix.T @ empirical_source_covariance @ matrix
        )
        empirical_covariance_error = float(
            np.linalg.norm(
                mapped_empirical_covariance - empirical_reference_covariance,
                ord="fro",
            )
            / max(
                float(np.linalg.norm(empirical_reference_covariance, ord="fro")),
                _EPSILON,
            )
        )
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
            raise FloatingPointError("composed affine map is non-finite")
        if fitted_covariance_error > covariance_tolerance:
            raise FloatingPointError(
                "composed map violates fitted-covariance preservation tolerance"
            )
        if orthogonality_error > orthogonality_tolerance:
            raise FloatingPointError(
                "composed orientation violates the declared orthogonality tolerance"
            )
        common_diagnostics.update({
            "status": "PASS",
            "fallback_used": False,
            "fallback_reason": None,
            "selected_rank": int(rank),
            "selected_shrinkage": float(shrinkage),
            "selected_rank_required_active_classes": int(
                max(8, minimum_active_classes, 2 * rank + 1)
            ),
            "orthogonal_orientation_refit_on_all_calibration": True,
            "refit_source_rows": int(len(source)),
            "refit_reference_rows": int(len(reference)),
            "refit_active_classes": refit_active_classes.tolist(),
            "refit_effective_anchor_mass_min": float(np.min(refit_effective_mass)),
            "refit_effective_anchor_mass_median": float(
                np.median(refit_effective_mass)
            ),
            "refit_anchor_discrepancy_singular_values": refit_spectrum.tolist(),
            "refit_anchor_discrepancy_cumulative_energy": refit_energy.tolist(),
            "refit_regularized_cross_singular_values": refit_cross_spectrum.tolist(),
            "orientation_orthogonality_error": orthogonality_error,
            "orientation_determinant": float(np.linalg.det(orientation)),
            "subspace_orientation_determinant": float(
                np.linalg.det(subspace_rotation)
            ),
            "orientation_includes_reflection": bool(
                np.linalg.det(orientation) < 0.0
            ),
            "fitted_covariance_relative_error": fitted_covariance_error,
            "fitted_regularized_covariance_relative_error": fitted_covariance_error,
            "fitted_oas_covariance_relative_error": (
                fitted_covariance_error
                if str(covariance_estimator) == "oas" else None
            ),
            "empirical_covariance_relative_error": empirical_covariance_error,
            "orientation_correction_frobenius_norm": float(
                np.linalg.norm(orientation - np.eye(dimension), ord="fro")
            ),
        })
        return BuresPseudoOrientationMap(
            base_map=base_map,
            matrix=matrix,
            offset=offset,
            orientation=orientation,
            basis=basis,
            subspace_rotation=subspace_rotation,
            diagnostics=common_diagnostics,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        return _identity_fallback(
            base_map,
            reason=f"{type(error).__name__}: {error}",
            diagnostics=common_diagnostics,
        )
