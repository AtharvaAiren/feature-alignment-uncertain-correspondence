"""Numerically explicit linear-algebra helpers under a row-vector convention."""

from __future__ import annotations

import numpy as np


def as_samples(x: np.ndarray, *, name: str = "features") -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must have non-empty shape [n,d], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def sym(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    return 0.5 * (value + value.T)


def population_mean_cov(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = as_samples(x)
    mean = value.mean(axis=0)
    centered = value - mean
    return mean, sym(centered.T @ centered / value.shape[0])


def regularize_covariance(
    covariance: np.ndarray,
    *,
    shrinkage: float = 1e-4,
    relative_floor: float = 1e-7,
    absolute_floor: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0,1]")
    if min(relative_floor, absolute_floor) < 0.0:
        raise ValueError("eigenvalue floors must be non-negative")
    cov = sym(np.asarray(covariance, dtype=np.float64))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    d = cov.shape[0]
    scale = max(float(np.trace(cov) / d), np.finfo(float).tiny)
    shrunk = (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(d)
    values, vectors = np.linalg.eigh(sym(shrunk))
    floor = max(absolute_floor, relative_floor * max(float(np.max(np.abs(values))), scale))
    clipped = np.maximum(values, floor)
    result = sym((vectors * clipped) @ vectors.T)
    return result, {
        "raw_min_eigenvalue": float(values.min()),
        "raw_max_eigenvalue": float(values.max()),
        "floor": float(floor),
        "condition_number": float(clipped.max() / clipped.min()),
        "shrinkage": float(shrinkage),
    }


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(sym(matrix))
    if np.any(values <= 0.0):
        raise np.linalg.LinAlgError("matrix must be positive definite")
    return sym((vectors * np.power(values, power)) @ vectors.T)


def gram_square_root(factor: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | str | None]]:
    """Return the principal square root of ``factor.T @ factor`` via an SVD.

    Forming the Gram matrix and then diagonalizing it squares the condition
    number.  In the Bures map this can turn an analytically positive-definite
    middle matrix into one with tiny negative computed eigenvalues, even after
    both endpoint covariances have been explicitly floored.  If ``B=U S V^T``,
    then ``(B^T B)^(1/2)=V S V^T``; evaluating that identity directly preserves
    the already-declared covariance estimator without adding another jitter or
    changing the mathematical map.
    """

    value = np.asarray(factor, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("Gram factor must be a nonempty matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("Gram factor contains non-finite values")
    _, singular_values, right_t = np.linalg.svd(value, full_matrices=False)
    root = sym((right_t.T * singular_values) @ right_t)
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    smallest = float(singular_values[-1]) if len(singular_values) else 0.0
    tolerance = float(
        max(value.shape) * np.finfo(np.float64).eps * largest
    )
    effective_rank = int(np.sum(singular_values > tolerance))
    condition_number = (
        float(largest / smallest) if smallest > tolerance else None
    )
    return root, {
        "algorithm": "svd_of_factor_without_forming_gram",
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "svd_rank_tolerance": tolerance,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
    }


def relative_frobenius(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected), ord="fro"))
    denominator = max(float(np.linalg.norm(expected, ord="fro")), np.finfo(float).eps)
    return numerator / denominator


def stable_hash_arrays(*arrays: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()
