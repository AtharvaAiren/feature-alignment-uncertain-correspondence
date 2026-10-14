"""Out-of-sample feature maps for the alignment information audit.

Rows are observations throughout.  Every fitted object stores only calibration
statistics and implements :meth:`apply` on untouched observations.  In
particular, the empirical quantile maps never keep a row assignment between
the source and reference calibration sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from marginal_alignment.linalg import as_samples, gram_square_root, sym, symmetric_power


class FittedMap(Protocol):
    """Minimal contract required by the V2 runner."""

    method: str
    diagnostics: dict[str, Any]

    def apply(self, features: np.ndarray) -> np.ndarray: ...


def _same_dimension(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = as_samples(source, name="source")
    reference = as_samples(reference, name="reference")
    if source.shape[1] != reference.shape[1]:
        raise ValueError("source and reference feature dimensions differ")
    return source, reference


@dataclass(frozen=True)
class AffineMap:
    matrix: np.ndarray
    offset: np.ndarray
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(self, features: np.ndarray) -> np.ndarray:
        features = as_samples(features)
        if features.shape[1] != self.matrix.shape[0]:
            raise ValueError("feature dimension does not match fitted map")
        return features @ self.matrix + self.offset


def _affine(matrix: np.ndarray, source_mean: np.ndarray, reference_mean: np.ndarray,
            method: str, diagnostics: dict[str, Any] | None = None) -> AffineMap:
    matrix = np.asarray(matrix, dtype=np.float64)
    return AffineMap(
        matrix=matrix,
        offset=np.asarray(reference_mean, dtype=np.float64)
        - np.asarray(source_mean, dtype=np.float64) @ matrix,
        method=method,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def fit_identity(source: np.ndarray, reference: np.ndarray) -> AffineMap:
    source, _ = _same_dimension(source, reference)
    return AffineMap(np.eye(source.shape[1]), np.zeros(source.shape[1]), "none")


def fit_mean(source: np.ndarray, reference: np.ndarray) -> AffineMap:
    source, reference = _same_dimension(source, reference)
    return _affine(np.eye(source.shape[1]), source.mean(0), reference.mean(0), "mean")


def fit_diagonal(source: np.ndarray, reference: np.ndarray,
                 standard_deviation_floor: float = 1e-6) -> AffineMap:
    source, reference = _same_dimension(source, reference)
    source_std_raw = source.std(0)
    reference_std_raw = reference.std(0)
    source_std = np.maximum(source_std_raw, standard_deviation_floor)
    reference_std = np.maximum(reference_std_raw, standard_deviation_floor)
    return _affine(
        np.diag(reference_std / source_std), source.mean(0), reference.mean(0), "diagonal",
        {
            "standard_deviation_floor": float(standard_deviation_floor),
            "floored_source_dimensions": int(np.sum(source_std_raw < standard_deviation_floor)),
            "floored_reference_dimensions": int(
                np.sum(reference_std_raw < standard_deviation_floor)
            ),
        },
    )


def _unique_quantile_knots(values: np.ndarray, knots: int | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return monotone value/CDF knots, optionally on a bounded grid.

    ``knots=None`` retains the complete empirical marginal.  A finite value
    records at most that many mid-quantiles, which makes the configuration an
    actual estimator choice rather than inert provenance.  Repeated quantiles
    are collapsed because :func:`numpy.interp` requires an increasing value
    axis; their probability is the mean of the tied grid locations.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("quantile values must be a nonempty vector")
    if knots is None or int(knots) >= len(values):
        ordered = np.sort(values)
        unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
        probabilities = (starts + 0.5 * counts) / len(ordered)
        return unique, probabilities
    knots = int(knots)
    if knots < 2:
        raise ValueError("quantile knots must be at least two")
    probabilities = (np.arange(knots, dtype=np.float64) + 0.5) / knots
    quantiles = np.quantile(values, probabilities, method="linear")
    unique, inverse = np.unique(quantiles, return_inverse=True)
    probability_sums = np.bincount(inverse, weights=probabilities)
    counts = np.bincount(inverse)
    return unique, probability_sums / counts


def _reference_quantile_knots(values: np.ndarray, knots: int | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Return values parameterized by a strictly increasing probability grid."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("quantile values must be a nonempty vector")
    if knots is None or int(knots) >= len(values):
        ordered = np.sort(values)
        probabilities = (np.arange(len(ordered), dtype=np.float64) + 0.5) / len(ordered)
        return ordered, probabilities
    knots = int(knots)
    if knots < 2:
        raise ValueError("quantile knots must be at least two")
    probabilities = (np.arange(knots, dtype=np.float64) + 0.5) / knots
    return np.quantile(values, probabilities, method="linear"), probabilities


@dataclass(frozen=True)
class CoordinateQuantileMap:
    source_knots: tuple[np.ndarray, ...]
    source_probabilities: tuple[np.ndarray, ...]
    reference_values: tuple[np.ndarray, ...]
    reference_probabilities: tuple[np.ndarray, ...]
    method: str = "coordinate_quantile"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(self, features: np.ndarray) -> np.ndarray:
        features = as_samples(features)
        if features.shape[1] != len(self.source_knots):
            raise ValueError("feature dimension does not match fitted quantile map")
        result = np.empty_like(features, dtype=np.float64)
        for coordinate in range(features.shape[1]):
            probabilities = np.interp(
                features[:, coordinate],
                self.source_knots[coordinate],
                self.source_probabilities[coordinate],
                left=self.source_probabilities[coordinate][0],
                right=self.source_probabilities[coordinate][-1],
            )
            result[:, coordinate] = np.interp(
                probabilities,
                self.reference_probabilities[coordinate],
                self.reference_values[coordinate],
            )
        return result


def fit_coordinate_quantile(source: np.ndarray, reference: np.ndarray, *,
                            knots: int | None = None) -> CoordinateQuantileMap:
    """Match every one-dimensional empirical marginal without using row pairing.

    This map is a strict extension of diagonal moment matching, but remains
    coordinatewise monotone and therefore cannot alter the empirical copula.
    """

    source, reference = _same_dimension(source, reference)
    source_knots: list[np.ndarray] = []
    source_probabilities: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    reference_probabilities: list[np.ndarray] = []
    for coordinate in range(source.shape[1]):
        coordinate_knots, probabilities = _unique_quantile_knots(
            source[:, coordinate], knots
        )
        ref, ref_probability = _reference_quantile_knots(
            reference[:, coordinate], knots
        )
        source_knots.append(coordinate_knots)
        source_probabilities.append(probabilities)
        reference_values.append(ref)
        reference_probabilities.append(ref_probability)
    return CoordinateQuantileMap(
        tuple(source_knots), tuple(source_probabilities),
        tuple(reference_values), tuple(reference_probabilities),
        diagnostics={
            "source_rows": len(source), "reference_rows": len(reference),
            "copula_changing": False,
            "requested_knots": knots,
            "maximum_source_knots": max(len(value) for value in source_knots),
            "maximum_reference_knots": max(len(value) for value in reference_values),
        },
    )


def _empirical_covariance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(0)
    centered = features - mean
    return mean, sym(centered.T @ centered / len(centered))


def _oas_shrinkage(covariance: np.ndarray, n_samples: int) -> tuple[np.ndarray, float]:
    dimension = covariance.shape[0]
    mu = float(np.trace(covariance) / dimension)
    alpha = float(np.mean(np.square(covariance)))
    denominator = (n_samples + 1.0) * (alpha - (mu * mu) / dimension)
    shrinkage = 1.0 if denominator <= 0.0 else min((alpha + mu * mu) / denominator, 1.0)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * mu * np.eye(dimension)
    return sym(shrunk), float(shrinkage)


def estimate_covariance(features: np.ndarray, estimator: str | float = "oas",
                        relative_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a positive-definite covariance with an explicit estimator."""

    features = as_samples(features)
    mean, covariance = _empirical_covariance(features)
    if isinstance(estimator, (float, int)):
        shrinkage = float(estimator)
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("fixed covariance shrinkage must lie in [0,1]")
        target = float(np.trace(covariance) / covariance.shape[0]) * np.eye(covariance.shape[0])
        covariance = (1.0 - shrinkage) * covariance + shrinkage * target
        estimator_name = "fixed"
    else:
        estimator_name = str(estimator).lower()
        if estimator_name == "empirical":
            shrinkage = 0.0
        elif estimator_name == "oas":
            covariance, shrinkage = _oas_shrinkage(covariance, len(features))
        elif estimator_name in {"ledoit_wolf", "ledoitwolf", "lw"}:
            try:
                from sklearn.covariance import ledoit_wolf
            except ImportError as error:
                raise RuntimeError("Ledoit-Wolf covariance requires scikit-learn") from error
            covariance, shrinkage = ledoit_wolf(np.asarray(features, dtype=np.float64),
                                                assume_centered=False)
            covariance = sym(covariance)
            estimator_name = "ledoit_wolf"
        else:
            raise ValueError(f"unknown covariance estimator: {estimator}")
    eigenvalues, eigenvectors = np.linalg.eigh(sym(covariance))
    scale = max(float(np.trace(covariance) / covariance.shape[0]), np.finfo(float).eps)
    floor = max(relative_floor * scale, np.finfo(float).eps)
    clipped = np.maximum(eigenvalues, floor)
    covariance = sym((eigenvectors * clipped) @ eigenvectors.T)
    return mean, covariance, {
        "covariance_estimator": estimator_name,
        "shrinkage": float(shrinkage),
        "eigenvalue_floor": float(floor),
        "clipped_eigenvalues": int(np.sum(eigenvalues < floor)),
        "effective_rank": float(np.square(clipped.sum()) / np.square(clipped).sum()),
    }


def _rank_restrict(
    matrix: np.ndarray, rank: int | None, *, symmetric_correction: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    dimension = matrix.shape[0]
    if rank is None or int(rank) >= dimension:
        return matrix, 1.0, {
            "algorithm": "none",
            "requested_rank": rank,
            "effective_rank": dimension,
        }
    rank = int(rank)
    if rank < 0:
        raise ValueError("rank must be non-negative")
    correction_raw = np.asarray(matrix, dtype=np.float64) - np.eye(dimension)
    if not np.all(np.isfinite(correction_raw)):
        raise np.linalg.LinAlgError("identity correction contains non-finite values")
    correction_norm = float(np.linalg.norm(correction_raw, ord="fro"))
    symmetry_error = float(
        np.linalg.norm(correction_raw - correction_raw.T, ord="fro")
        / max(correction_norm, np.finfo(np.float64).eps)
    )
    if symmetric_correction:
        # Bures is symmetric positive definite, so A-I is symmetric.  Its
        # Frobenius-optimal rank-r approximation keeps the eigenpairs with the
        # largest |lambda|.  A generic SVD is algebraically redundant here and
        # LAPACK's divide-and-conquer SVD can fail on the extremely clustered
        # spectra arising at n << d.
        correction = sym(correction_raw)
        eigenvalues, eigenvectors = np.linalg.eigh(correction)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        spectrum = np.abs(eigenvalues[order])
        spectral_scale = float(spectrum[0]) if len(spectrum) else 0.0
        spectral_tolerance = float(
            64.0 * dimension * np.finfo(np.float64).eps
            * max(spectral_scale, np.finfo(np.float64).tiny)
        )
        cutoff = float(spectrum[rank - 1]) if rank else None
        next_value = float(spectrum[rank]) if rank < dimension else None
        cutoff_gap = (
            None if cutoff is None or next_value is None
            else float(cutoff - next_value)
        )
        boundary_tie_split = bool(
            cutoff is not None and next_value is not None
            and cutoff > spectral_tolerance
            and cutoff_gap is not None and cutoff_gap <= spectral_tolerance
        )
        if boundary_tie_split:
            raise np.linalg.LinAlgError(
                "rank boundary splits a materially nonzero repeated Bures "
                f"eigenspace: rank={rank}, cutoff={cutoff:.17g}, "
                f"next={next_value:.17g}, tolerance={spectral_tolerance:.17g}"
            )
        keep = order[:rank]
        if rank:
            restricted = np.eye(dimension) + (
                eigenvectors[:, keep] * eigenvalues[keep]
            ) @ eigenvectors[:, keep].T
            restricted = sym(restricted)
        else:
            restricted = np.eye(dimension)
        algorithm = "symmetric_eigen_largest_absolute"
        numerical_input_rank = int(np.sum(spectrum > spectral_tolerance))
        numerical_retained_rank = int(np.sum(spectrum[:rank] > spectral_tolerance))
        decomposition_diagnostics: dict[str, Any] = {
            "spectral_tolerance": spectral_tolerance,
            "numerical_input_correction_rank": numerical_input_rank,
            "numerical_retained_correction_rank": numerical_retained_rank,
            "cutoff_abs_eigenvalue": cutoff,
            "next_abs_eigenvalue": next_value,
            "cutoff_absolute_gap": cutoff_gap,
            "boundary_tie_split": boundary_tie_split,
            "full_map_minimum_eigenvalue": float(1.0 + np.min(eigenvalues)),
        }
    else:
        left, spectrum, right_t = np.linalg.svd(
            correction_raw, full_matrices=False
        )
        if rank:
            restricted = np.eye(dimension) + (
                left[:, :rank] * spectrum[:rank]
            ) @ right_t[:rank]
        else:
            restricted = np.eye(dimension)
        algorithm = "general_svd"
        spectral_tolerance = float(
            64.0 * dimension * np.finfo(np.float64).eps
            * max(float(spectrum[0]) if len(spectrum) else 0.0,
                  np.finfo(np.float64).tiny)
        )
        decomposition_diagnostics = {
            "spectral_tolerance": spectral_tolerance,
            "numerical_input_correction_rank": int(
                np.sum(spectrum > spectral_tolerance)
            ),
            "numerical_retained_correction_rank": int(
                np.sum(spectrum[:rank] > spectral_tolerance)
            ),
        }
    retained = 0.0
    total = float(np.square(spectrum).sum())
    if rank:
        retained = 1.0 if total == 0.0 else float(
            np.square(spectrum[:rank]).sum() / total
        )
    else:
        retained = 1.0 if total == 0.0 else 0.0
    restricted_minimum_eigenvalue = (
        float(np.min(np.linalg.eigvalsh(restricted)))
        if symmetric_correction else None
    )
    if symmetric_correction and restricted_minimum_eigenvalue <= 0.0:
        raise np.linalg.LinAlgError(
            "rank-restricted Bures map is not positive definite"
        )
    return restricted, retained, {
        "algorithm": algorithm,
        "requested_rank": rank,
        "effective_rank": decomposition_diagnostics[
            "numerical_retained_correction_rank"
        ],
        "input_correction_symmetry_relative_error": symmetry_error,
        "input_correction_frobenius_norm": correction_norm,
        "retained_frobenius_energy": retained,
        "restricted_map_minimum_eigenvalue": restricted_minimum_eigenvalue,
        **decomposition_diagnostics,
    }


def fit_covariance_gauge(source: np.ndarray, reference: np.ndarray, *, gauge: str,
                         estimator: str | float = "oas", relative_floor: float = 1e-6,
                         rank: int | None = None) -> AffineMap:
    source, reference = _same_dimension(source, reference)
    source_mean, source_covariance, source_diagnostics = estimate_covariance(
        source, estimator, relative_floor
    )
    reference_mean, reference_covariance, reference_diagnostics = estimate_covariance(
        reference, estimator, relative_floor
    )
    gauge = gauge.lower()
    if gauge == "coral":
        matrix = symmetric_power(source_covariance, -0.5) @ symmetric_power(
            reference_covariance, 0.5
        )
    elif gauge == "bures":
        source_sqrt = symmetric_power(source_covariance, 0.5)
        source_inverse_sqrt = symmetric_power(source_covariance, -0.5)
        reference_sqrt = symmetric_power(reference_covariance, 0.5)
        # The old ``eigh(source_sqrt @ reference_covariance @ source_sqrt)``
        # route squared the endpoint condition numbers and could manufacture a
        # tiny negative eigenvalue.  This SVD computes the identical principal
        # square root from B^T B with B=reference_sqrt @ source_sqrt.
        middle_sqrt, bures_sqrt_diagnostics = gram_square_root(
            reference_sqrt @ source_sqrt
        )
        matrix = sym(
            source_inverse_sqrt @ middle_sqrt @ source_inverse_sqrt
        )
    else:
        raise ValueError("covariance gauge must be 'coral' or 'bures'")
    matrix, retained, rank_diagnostics = _rank_restrict(
        matrix, rank, symmetric_correction=(gauge == "bures")
    )
    mapped = matrix.T @ source_covariance @ matrix
    denominator = max(float(np.linalg.norm(reference_covariance, ord="fro")), 1e-15)
    method = gauge if rank is None else f"{gauge}_rank{int(rank)}"
    return _affine(matrix, source_mean, reference_mean, method, {
        "source_covariance": source_diagnostics,
        "reference_covariance": reference_diagnostics,
        **({"bures_middle_square_root": bures_sqrt_diagnostics}
           if gauge == "bures" else {}),
        "rank": rank,
        "rank_restriction": rank_diagnostics,
        "identity_correction_energy_retained": retained,
        "fitted_covariance_relative_error": float(
            np.linalg.norm(mapped - reference_covariance, ord="fro") / denominator
        ),
    })


@dataclass(frozen=True)
class ScalarQuantileStep:
    direction: np.ndarray
    source_knots: np.ndarray
    source_probabilities: np.ndarray
    reference_values: np.ndarray
    reference_probabilities: np.ndarray

    def apply(self, features: np.ndarray) -> np.ndarray:
        projection = features @ self.direction
        probabilities = np.interp(
            projection, self.source_knots, self.source_probabilities,
            left=self.source_probabilities[0], right=self.source_probabilities[-1],
        )
        mapped = np.interp(
            probabilities, self.reference_probabilities, self.reference_values
        )
        return features + (mapped - projection)[:, None] * self.direction[None, :]


@dataclass(frozen=True)
class SlicedQuantileFlow:
    steps: tuple[ScalarQuantileStep, ...]
    basis: np.ndarray | None = None
    origin: np.ndarray | None = None
    method: str = "sliced_quantile"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(self, features: np.ndarray) -> np.ndarray:
        features = as_samples(features).astype(np.float64, copy=False)
        if self.basis is None:
            result = features.copy()
            if self.steps and result.shape[1] != len(self.steps[0].direction):
                raise ValueError("feature dimension does not match fitted sliced map")
        else:
            if self.origin is None or features.shape[1] != self.basis.shape[0]:
                raise ValueError("feature dimension does not match fitted PCA sliced map")
            result = (features - self.origin) @ self.basis
        for step in self.steps:
            result = step.apply(result)
        if self.basis is None:
            return result
        original_projection = (features - self.origin) @ self.basis
        # Only the fitted PCA component is changed.  The orthogonal residual
        # of every out-of-sample feature is preserved exactly (up to roundoff).
        return features + (result - original_projection) @ self.basis.T


def _pooled_pca_basis(source: np.ndarray, reference: np.ndarray, *, rank: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a common randomized PCA subspace from unpaired pooled endpoints."""

    dimension = source.shape[1]
    requested_rank = int(rank)
    if requested_rank < 1:
        raise ValueError("sliced quantile rank must be positive")
    effective_rank = min(requested_rank, dimension, len(source) + len(reference) - 1)
    pooled = np.concatenate([source, reference], axis=0)
    origin = pooled.mean(0)
    centered = pooled - origin
    if effective_rank == dimension:
        basis = np.eye(dimension)
        retained_energy = 1.0
    else:
        try:
            from sklearn.utils.extmath import randomized_svd
        except ImportError as error:
            raise RuntimeError("rank-restricted sliced quantiles require scikit-learn") from error
        _, singular_values, right_t = randomized_svd(
            centered,
            n_components=effective_rank,
            n_oversamples=min(10, max(dimension - effective_rank, 1)),
            n_iter=2,
            random_state=int(rng.integers(0, 2**31 - 1)),
            flip_sign=True,
        )
        basis = right_t.T
        total_energy = float(np.square(centered).sum())
        retained_energy = (
            1.0 if total_energy == 0.0
            else float(np.square(singular_values).sum() / total_energy)
        )
    return origin, basis, {
        "requested_rank": requested_rank,
        "effective_rank": effective_rank,
        "pca_energy_retained": retained_energy,
        "pca_basis_orthogonality_error": float(np.linalg.norm(
            basis.T @ basis - np.eye(effective_rank), ord="fro"
        )),
        "orthogonal_residual_preserved": True,
        "pca_fit_information": "pooled_unpaired_endpoint_marginals",
    }


def fit_sliced_quantile(source: np.ndarray, reference: np.ndarray, *,
                        rng: np.random.Generator, projections: int = 64,
                        sweeps: int = 1, rank: int | None = None,
                        knots: int | None = None) -> SlicedQuantileFlow:
    """Fit a finite random-projection sliced distribution-transfer flow.

    Each stage is a one-dimensional empirical transport followed by an
    out-of-sample ridge displacement along its declared projection.  This is a
    richer joint-marginal baseline, not an OT/correspondence oracle.
    """

    source, reference = _same_dimension(source, reference)
    if projections < 1 or sweeps < 1:
        raise ValueError("projections and sweeps must be positive")
    basis = None
    origin = None
    rank_diagnostics: dict[str, Any] = {
        "requested_rank": None,
        "effective_rank": source.shape[1],
        "orthogonal_residual_preserved": False,
    }
    fit_source = source
    fit_reference = reference
    if rank is not None:
        origin, basis, rank_diagnostics = _pooled_pca_basis(
            source, reference, rank=int(rank), rng=rng
        )
        fit_source = (source - origin) @ basis
        fit_reference = (reference - origin) @ basis
    current = fit_source.astype(np.float64, copy=True)
    steps: list[ScalarQuantileStep] = []
    for _ in range(int(sweeps)):
        directions = rng.normal(size=(int(projections), fit_source.shape[1]))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
        for direction in directions:
            source_projection = current @ direction
            reference_projection = fit_reference @ direction
            source_knots, probabilities = _unique_quantile_knots(
                source_projection, knots
            )
            reference_values, reference_probabilities = _reference_quantile_knots(
                reference_projection, knots
            )
            step = ScalarQuantileStep(
                direction.copy(), source_knots, probabilities,
                reference_values, reference_probabilities,
            )
            current = step.apply(current)
            steps.append(step)
    method = "sliced_quantile" if rank is None else f"sliced_quantile_rank{int(rank)}"
    return SlicedQuantileFlow(tuple(steps), basis=basis, origin=origin, method=method,
                              diagnostics={
        "projections": int(projections), "sweeps": int(sweeps),
        "stages": len(steps), "changes_joint_dependence": True,
        "requested_knots": knots,
        "maximum_source_knots": max(len(step.source_knots) for step in steps),
        "maximum_reference_knots": max(len(step.reference_values) for step in steps),
        **rank_diagnostics,
    })


def _svd_with_robust_driver(
    matrix: np.ndarray, *, full_matrices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute the same SVD with a conservative LAPACK fallback.

    NumPy normally dispatches the divide-and-conquer driver.  On a finite,
    identified matrix, falling back to SciPy's QR-based ``gesvd`` changes no
    estimator; it only avoids turning a driver convergence failure into a
    missing experimental cell.  Rank-deficient orthogonal completions are
    handled by explicit identification gates, not by this fallback.
    """

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("SVD input must be a nonempty matrix")
    if not np.all(np.isfinite(value)):
        raise np.linalg.LinAlgError("SVD input contains non-finite values")
    try:
        left, singular_values, right_t = np.linalg.svd(
            value, full_matrices=full_matrices
        )
        return left, singular_values, right_t, "numpy_default"
    except np.linalg.LinAlgError as first_error:
        try:
            from scipy.linalg import svd as scipy_svd
            left, singular_values, right_t = scipy_svd(
                value, full_matrices=full_matrices,
                check_finite=False, lapack_driver="gesvd",
            )
        except Exception as second_error:
            raise np.linalg.LinAlgError(
                "both NumPy SVD and SciPy gesvd failed"
            ) from second_error
        if not all(np.all(np.isfinite(part)) for part in (
            left, singular_values, right_t,
        )):
            raise np.linalg.LinAlgError(
                "SciPy gesvd fallback returned non-finite values"
            ) from first_error
        return left, singular_values, right_t, "scipy_gesvd_fallback"


def fit_paired_procrustes(
    source: np.ndarray, reference: np.ndarray, *,
    require_full_centered_rank: bool = False,
) -> AffineMap:
    source, reference = _same_dimension(source, reference)
    if len(source) != len(reference):
        raise ValueError("paired Procrustes requires equal row counts")
    source_mean, reference_mean = source.mean(0), reference.mean(0)
    dimension = source.shape[1]
    if require_full_centered_rank and len(source) - 1 < dimension:
        raise np.linalg.LinAlgError(
            "full paired Procrustes is not identified: centered pair count "
            f"{len(source) - 1} is smaller than feature dimension {dimension}"
        )
    cross = (source - source_mean).T @ (reference - reference_mean)
    left, singular_values, right_t, svd_driver = _svd_with_robust_driver(
        cross, full_matrices=False
    )
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    rank_tolerance = float(
        max(cross.shape) * np.finfo(np.float64).eps * largest
    )
    centered_cross_rank = int(np.sum(singular_values > rank_tolerance))
    if require_full_centered_rank and centered_cross_rank < dimension:
        raise np.linalg.LinAlgError(
            "full paired Procrustes is not numerically identified: "
            f"centered cross rank {centered_cross_rank} < {dimension} "
            f"at tolerance {rank_tolerance:.17g}"
        )
    matrix = left @ right_t
    return _affine(matrix, source_mean, reference_mean, "paired_procrustes", {
        "orthogonality_error": float(
            np.linalg.norm(matrix.T @ matrix - np.eye(matrix.shape[0]), ord="fro")
        ),
        "svd_driver": svd_driver,
        "centered_cross_rank": centered_cross_rank,
        "centered_cross_rank_tolerance": rank_tolerance,
        "full_orthogonal_gauge_identified": bool(
            centered_cross_rank == dimension
        ),
        "full_centered_rank_required": bool(require_full_centered_rank),
    })


def fit_paired_ridge(source: np.ndarray, reference: np.ndarray, *, ridge: float = 1e-3,
                     rank: int | None = None) -> AffineMap:
    source, reference = _same_dimension(source, reference)
    if len(source) != len(reference):
        raise ValueError("paired regression requires equal row counts")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    source_mean, reference_mean = source.mean(0), reference.mean(0)
    centered_source = source - source_mean
    centered_reference = reference - reference_mean
    scale = max(float(np.square(centered_source).sum() / source.shape[1]), 1e-15)
    regularization = float(ridge) * scale
    if ridge == 0.0:
        # The Moore--Penrose minimum-norm OLS solution remains well-defined
        # when m < d; a square normal-equation solve does not.
        matrix, _, numerical_rank, singular_values = np.linalg.lstsq(
            centered_source, centered_reference, rcond=None
        )
        solve_form = "minimum_norm_lstsq"
        linear_system_dimension = min(centered_source.shape)
    elif len(source) < source.shape[1]:
        # Exact primal/dual ridge identity:
        # (X^T X + lambda I)^-1 X^T Y
        #     = X^T (X X^T + lambda I)^-1 Y.
        # Pair-budget experiments commonly have m << d, so the dual form
        # replaces a d x d solve by an m x m solve without changing the
        # estimator or its regularization convention.
        dual_gram = centered_source @ centered_source.T
        matrix = centered_source.T @ np.linalg.solve(
            dual_gram + regularization * np.eye(len(source)),
            centered_reference,
        )
        solve_form = "dual"
        linear_system_dimension = len(source)
    else:
        gram = centered_source.T @ centered_source
        matrix = np.linalg.solve(
            gram + regularization * np.eye(source.shape[1]),
            centered_source.T @ centered_reference,
        )
        solve_form = "primal"
        linear_system_dimension = source.shape[1]
    method = "paired_ridge"
    retained = 1.0
    requested_rank = rank
    effective_rank = None
    if rank is not None:
        rank = int(rank)
        if rank < 1:
            raise ValueError("reduced-rank regression rank must be positive")
        predicted = centered_source @ matrix
        _, predicted_singular_values, right_t, rrr_svd_driver = (
            _svd_with_robust_driver(predicted, full_matrices=False)
        )
        effective_rank = min(rank, len(right_t), max(len(source) - 1, 1))
        largest_predicted = (
            float(predicted_singular_values[0])
            if len(predicted_singular_values) else 0.0
        )
        rank_tolerance = float(
            max(predicted.shape) * np.finfo(np.float64).eps * largest_predicted
        )
        cutoff = (
            float(predicted_singular_values[effective_rank - 1])
            if effective_rank else None
        )
        next_value = (
            float(predicted_singular_values[effective_rank])
            if effective_rank < len(predicted_singular_values) else None
        )
        boundary_tie = bool(
            cutoff is not None and next_value is not None
            and cutoff > rank_tolerance
            and cutoff - next_value <= rank_tolerance
        )
        if boundary_tie:
            raise np.linalg.LinAlgError(
                "reduced-rank ridge boundary splits a materially nonzero "
                f"repeated singular subspace: rank={effective_rank}, "
                f"cutoff={cutoff:.17g}, next={next_value:.17g}, "
                f"tolerance={rank_tolerance:.17g}"
            )
        output_subspace = right_t[:effective_rank].T
        full_energy = float(np.square(predicted).sum())
        matrix = matrix @ output_subspace @ output_subspace.T
        reduced_energy = float(np.square(centered_source @ matrix).sum())
        retained = 1.0 if full_energy == 0.0 else reduced_energy / full_energy
        method = f"paired_reduced_rank{rank}"
    else:
        rrr_svd_driver = None
        rank_tolerance = None
        cutoff = None
        next_value = None
        boundary_tie = False
    fitted_error = float(np.mean(np.square(centered_source @ matrix - centered_reference)))
    return _affine(matrix, source_mean, reference_mean, method, {
        "ridge": float(ridge), "rank": requested_rank,
        "effective_fitted_rank": effective_rank,
        "prediction_energy_retained": float(retained),
        "fitted_paired_mse": fitted_error,
        "solve_form": solve_form,
        "linear_system_dimension": int(linear_system_dimension),
        "regularization_scale": float(scale),
        "rrr_svd_driver": rrr_svd_driver,
        "rrr_rank_tolerance": rank_tolerance,
        "rrr_cutoff_singular_value": cutoff,
        "rrr_next_singular_value": next_value,
        "rrr_boundary_tie_split": boundary_tie,
    })


def fit_paired_readout_gauge(source: np.ndarray, reference: np.ndarray,
                             readout_weight: np.ndarray, *,
                             estimator: str | float = "oas",
                             relative_floor: float = 1e-6) -> AffineMap:
    """Choose the covariance gauge using true pairs and a frozen readout."""

    source, reference = _same_dimension(source, reference)
    if len(source) != len(reference):
        raise ValueError("readout gauge requires equal paired row counts")
    weight = np.asarray(readout_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[1] != source.shape[1]:
        raise ValueError("readout_weight must have shape [outputs,feature_dimension]")
    source_mean, source_covariance, source_diag = estimate_covariance(
        source, estimator, relative_floor
    )
    reference_mean, reference_covariance, reference_diag = estimate_covariance(
        reference, estimator, relative_floor
    )
    source_inverse_sqrt = symmetric_power(source_covariance, -0.5)
    reference_inverse_sqrt = symmetric_power(reference_covariance, -0.5)
    reference_sqrt = symmetric_power(reference_covariance, 0.5)
    whitened_source = (source - source_mean) @ source_inverse_sqrt
    whitened_reference = (reference - reference_mean) @ reference_inverse_sqrt
    cross = whitened_source.T @ whitened_reference / len(source)
    centered_weight = weight - weight.mean(axis=0, keepdims=True)
    task_metric = reference_sqrt @ centered_weight.T @ centered_weight @ reference_sqrt
    left, singular_values, right_t = np.linalg.svd(cross @ task_metric, full_matrices=False)
    orthogonal_gauge = left @ right_t
    matrix = source_inverse_sqrt @ orthogonal_gauge @ reference_sqrt
    transformed = (source - source_mean) @ matrix + reference_mean
    centered_logit_error = (transformed - reference) @ centered_weight.T
    return _affine(matrix, source_mean, reference_mean, "paired_readout_gauge", {
        "source_covariance": source_diag,
        "reference_covariance": reference_diag,
        "gauge_orthogonality_error": float(np.linalg.norm(
            orthogonal_gauge.T @ orthogonal_gauge - np.eye(source.shape[1]), ord="fro"
        )),
        "fitted_centered_logit_mse": float(np.mean(np.square(centered_logit_error))),
        "task_weighted_cross_nuclear_norm": float(singular_values.sum()),
    })


def fit_third_moment_ica(source: np.ndarray, reference: np.ndarray, *,
                         rng: np.random.Generator, rank: int = 64,
                         max_iterations: int = 1000) -> AffineMap:
    """Higher-order marginal map via independently fitted ICA coordinates.

    Component assignment uses only signed component skewness and never sample
    pairs. Distinct non-Gaussian component signatures can break a covariance
    gauge; Gaussian/equal-skew components intentionally remain ambiguous.
    """

    source, reference = _same_dimension(source, reference)
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.decomposition import FastICA
    except ImportError as error:
        raise RuntimeError("third-moment ICA requires scipy and scikit-learn") from error
    components = min(int(rank), source.shape[1], len(source) - 1, len(reference) - 1)
    if components < 2:
        raise ValueError("ICA rank must permit at least two components")
    random_seed = int(rng.integers(0, 2**31 - 1))
    source_ica = FastICA(n_components=components, whiten="unit-variance",
                         max_iter=max_iterations, random_state=random_seed)
    reference_ica = FastICA(n_components=components, whiten="unit-variance",
                            max_iter=max_iterations, random_state=random_seed + 1)
    source_codes = source_ica.fit_transform(source)
    reference_codes = reference_ica.fit_transform(reference)

    def signature(codes: np.ndarray) -> np.ndarray:
        centered = codes - codes.mean(0)
        std = np.maximum(centered.std(0), 1e-12)
        standardized = centered / std
        return np.column_stack([
            np.mean(standardized ** 3, axis=0),
            np.mean(standardized ** 4, axis=0) - 3.0,
            np.mean(np.abs(standardized), axis=0),
        ])

    source_signature = signature(source_codes)
    reference_signature = signature(reference_codes)
    # A component sign is arbitrary; compare both signs and record the chosen
    # orientation. Kurtosis/absolute moments are sign invariant.
    direct = np.square(source_signature[:, None, :] - reference_signature[None, :, :]).sum(2)
    flipped_signature = source_signature.copy()
    flipped_signature[:, 0] *= -1.0
    flipped = np.square(flipped_signature[:, None, :] - reference_signature[None, :, :]).sum(2)
    cost = np.minimum(direct, flipped)
    source_component, reference_component = linear_sum_assignment(cost)
    signs = np.where(
        direct[source_component, reference_component]
        <= flipped[source_component, reference_component], 1.0, -1.0
    )
    permutation = np.zeros((components, components), dtype=np.float64)
    permutation[source_component, reference_component] = signs
    # sklearn transform: (x-mean) @ components_.T; inverse reference codes:
    # codes @ mixing_.T + reference mean.
    matrix = source_ica.components_.T @ permutation @ reference_ica.mixing_.T
    fitted = _affine(matrix, source_ica.mean_, reference_ica.mean_, "third_moment_ica", {
        "rank": components,
        "assignment_cost": float(cost[source_component, reference_component].sum()),
        "source_component_skew": source_signature[:, 0].tolist(),
        "reference_component_skew": reference_signature[:, 0].tolist(),
        "converged_source_iterations": int(source_ica.n_iter_),
        "converged_reference_iterations": int(reference_ica.n_iter_),
        "pairing_used": False,
    })
    return fitted
