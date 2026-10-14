"""Affine information-ladder transforms.

All feature matrices use rows as samples.  A fitted map acts as ``X @ matrix +
offset``.  The unpaired methods receive the two feature multisets but never
their row correspondence.  Paired methods are explicit diagnostic oracles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .linalg import (
    as_samples,
    gram_square_root,
    population_mean_cov,
    regularize_covariance,
    relative_frobenius,
    sym,
    symmetric_power,
)


@dataclass(frozen=True)
class AffineTransform:
    matrix: np.ndarray
    offset: np.ndarray
    method: str
    fit_information: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(self, features: np.ndarray) -> np.ndarray:
        x = as_samples(features)
        if x.shape[1] != self.matrix.shape[0]:
            raise ValueError("feature dimension does not match fitted transform")
        return x @ self.matrix + self.offset


def _same_dimension(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = as_samples(source, name="source")
    y = as_samples(reference, name="reference")
    if x.shape[1] != y.shape[1]:
        raise ValueError("source and reference feature dimensions differ")
    return x, y


def _affine(matrix: np.ndarray, source_mean: np.ndarray, reference_mean: np.ndarray,
            method: str, information: str, **diagnostics: Any) -> AffineTransform:
    matrix = np.asarray(matrix, dtype=np.float64)
    return AffineTransform(
        matrix=matrix,
        offset=np.asarray(reference_mean) - np.asarray(source_mean) @ matrix,
        method=method,
        fit_information=information,
        diagnostics=diagnostics,
    )


def fit_identity(source: np.ndarray, reference: np.ndarray) -> AffineTransform:
    x, y = _same_dimension(source, reference)
    del y
    d = x.shape[1]
    return AffineTransform(np.eye(d), np.zeros(d), "none", "none")


def fit_mean(source: np.ndarray, reference: np.ndarray) -> AffineTransform:
    x, y = _same_dimension(source, reference)
    return _affine(np.eye(x.shape[1]), x.mean(0), y.mean(0), "mean", "unpaired_means")


def fit_diagonal(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    standard_deviation_floor: float = 1e-6,
) -> AffineTransform:
    x, y = _same_dimension(source, reference)
    mx, my = x.mean(0), y.mean(0)
    sx = np.maximum(x.std(0), standard_deviation_floor)
    sy = np.maximum(y.std(0), standard_deviation_floor)
    # Row convention: the source-to-reference multiplier is sigma_r/sigma_t.
    matrix = np.diag(sy / sx)
    return _affine(
        matrix, mx, my, "diagonal", "unpaired_coordinate_moments",
        floored_source_dimensions=int(np.sum(x.std(0) < standard_deviation_floor)),
        floored_reference_dimensions=int(np.sum(y.std(0) < standard_deviation_floor)),
    )


def _moments(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    shrinkage: float,
    relative_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x, y = _same_dimension(source, reference)
    mx, cx0 = population_mean_cov(x)
    my, cy0 = population_mean_cov(y)
    cx, dx = regularize_covariance(cx0, shrinkage=shrinkage, relative_floor=relative_floor)
    cy, dy = regularize_covariance(cy0, shrinkage=shrinkage, relative_floor=relative_floor)
    return mx, cx, my, cy, {"source_stability": dx, "reference_stability": dy}


def fit_coral(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    shrinkage: float = 1e-4,
    relative_floor: float = 1e-7,
) -> AffineTransform:
    mx, cx, my, cy, diagnostics = _moments(
        source, reference, shrinkage=shrinkage, relative_floor=relative_floor
    )
    matrix = symmetric_power(cx, -0.5) @ symmetric_power(cy, 0.5)
    mapped = matrix.T @ cx @ matrix
    return _affine(
        matrix, mx, my, "coral", "unpaired_full_covariance",
        **diagnostics,
        regularized_covariance_residual=relative_frobenius(mapped, cy),
    )


def fit_bures(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    shrinkage: float = 1e-4,
    relative_floor: float = 1e-7,
    rank: int | None = None,
) -> AffineTransform:
    mx, cx, my, cy, diagnostics = _moments(
        source, reference, shrinkage=shrinkage, relative_floor=relative_floor
    )
    d = cx.shape[0]
    if rank is not None and not 0 <= int(rank) <= d:
        raise ValueError(f"rank must be in [0,{d}]")
    sqrt_x = symmetric_power(cx, 0.5)
    inverse_sqrt_x = symmetric_power(cx, -0.5)
    sqrt_y = symmetric_power(cy, 0.5)
    middle_sqrt, bures_sqrt_diagnostics = gram_square_root(sqrt_y @ sqrt_x)
    full = sym(inverse_sqrt_x @ middle_sqrt @ inverse_sqrt_x)
    retained = 1.0
    if rank is not None and int(rank) < d:
        values, vectors = np.linalg.eigh(sym(full - np.eye(d)))
        order = np.argsort(np.abs(values))[::-1]
        keep = order[: int(rank)]
        matrix = np.eye(d)
        if len(keep):
            matrix += (vectors[:, keep] * values[keep]) @ vectors[:, keep].T
        total = float(np.square(values).sum())
        retained = 1.0 if total == 0.0 else float(np.square(values[keep]).sum() / total)
        matrix = sym(matrix)
    else:
        matrix = full
    mapped = matrix.T @ cx @ matrix
    return _affine(
        matrix, mx, my, "bures" if rank is None else f"bures_rank{int(rank)}",
        "unpaired_full_covariance",
        **diagnostics,
        bures_middle_square_root=bures_sqrt_diagnostics,
        regularized_covariance_residual=relative_frobenius(mapped, cy),
        identity_correction_energy_retained=retained,
        rank=rank,
    )


def _orthogonal_procrustes(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(source.T @ reference, full_matrices=False)
    return left @ right_t


def fit_paired_procrustes(source: np.ndarray, reference: np.ndarray) -> AffineTransform:
    x, y = _same_dimension(source, reference)
    if x.shape[0] != y.shape[0]:
        raise ValueError("paired Procrustes requires equal sample counts")
    mx, my = x.mean(0), y.mean(0)
    q = _orthogonal_procrustes(x - mx, y - my)
    before = float(np.sqrt(np.mean(np.square(x - y))))
    after = float(np.sqrt(np.mean(np.square((x - mx) @ q - (y - my)))))
    return _affine(
        q, mx, my, "paired_procrustes", "true_sample_correspondence",
        paired_rmse_before=before, paired_rmse_after=after,
        orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(q.shape[0]), ord="fro")),
    )


def fit_paired_gauge(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    shrinkage: float = 1e-4,
    relative_floor: float = 1e-7,
) -> AffineTransform:
    """Bures marginal repair followed by a paired covariance-preserving gauge."""

    x, y = _same_dimension(source, reference)
    if x.shape[0] != y.shape[0]:
        raise ValueError("paired gauge requires equal sample counts")
    bures = fit_bures(x, y, shrinkage=shrinkage, relative_floor=relative_floor)
    xb = bures.apply(x)
    my, cy0 = population_mean_cov(y)
    cy, _ = regularize_covariance(cy0, shrinkage=shrinkage, relative_floor=relative_floor)
    sqrt_y = symmetric_power(cy, 0.5)
    inverse_sqrt_y = symmetric_power(cy, -0.5)
    u = (xb - my) @ inverse_sqrt_y
    v = (y - my) @ inverse_sqrt_y
    q = _orthogonal_procrustes(u, v)
    residual = inverse_sqrt_y @ q @ sqrt_y
    residual_offset = my - my @ residual
    matrix = bures.matrix @ residual
    offset = bures.offset @ residual + residual_offset
    transformed = x @ matrix + offset
    return AffineTransform(
        matrix, offset, "paired_oracle", "unpaired_moments_plus_true_correspondence",
        {
            "paired_rmse_after_bures": float(np.sqrt(np.mean(np.square(xb - y)))),
            "paired_rmse_after": float(np.sqrt(np.mean(np.square(transformed - y)))),
            "gauge_orthogonality_error": float(
                np.linalg.norm(q.T @ q - np.eye(q.shape[0]), ord="fro")
            ),
        },
    )


def fit_readout_weighted_gauge(
    source: np.ndarray,
    reference: np.ndarray,
    readout_weight: np.ndarray,
    *,
    shrinkage: float = 1e-4,
    relative_floor: float = 1e-7,
) -> AffineTransform:
    """Paired covariance gauge minimizing centered frozen-logit MSE.

    This is an evaluation oracle: it uses both true sample correspondence and
    the frozen readout.  It remains in the covariance-matching affine family.
    """

    y, x = _same_dimension(source, reference)
    if y.shape[0] != x.shape[0]:
        raise ValueError("readout-weighted gauge requires equal paired sample counts")
    weight = np.asarray(readout_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[1] != x.shape[1]:
        raise ValueError("readout_weight must have shape [outputs,d]")
    mt, ct0 = population_mean_cov(y)
    mr, cr0 = population_mean_cov(x)
    ct, _ = regularize_covariance(ct0, shrinkage=shrinkage, relative_floor=relative_floor)
    cr, _ = regularize_covariance(cr0, shrinkage=shrinkage, relative_floor=relative_floor)
    inv_sqrt_t = symmetric_power(ct, -0.5)
    inv_sqrt_r = symmetric_power(cr, -0.5)
    sqrt_r = symmetric_power(cr, 0.5)
    v = (y - mt) @ inv_sqrt_t
    u = (x - mr) @ inv_sqrt_r
    cross = v.T @ u / len(v)
    centering = np.eye(weight.shape[0]) - np.ones((weight.shape[0], weight.shape[0])) / weight.shape[0]
    centered_weight = centering @ weight
    task_metric = sqrt_r @ centered_weight.T @ centered_weight @ sqrt_r
    left, singular_values, right_t = np.linalg.svd(cross @ task_metric, full_matrices=False)
    gauge = right_t.T @ left.T
    # Column map A=Sr^(1/2) R St^(-1/2); transpose for row samples.
    matrix = inv_sqrt_t @ gauge.T @ sqrt_r
    transformed = (y - mt) @ matrix + mr
    logit_error = (transformed - x) @ centered_weight.T
    # The theorem defines E||centered-logit error||_2^2.  Paper-facing metrics
    # report a per-logit MSE, so divide the trace formulas by the output count.
    outputs = weight.shape[0]
    empirical_loss = float(np.mean(np.square(logit_error)))
    covariance_v = v.T @ v / len(v)
    covariance_u = u.T @ u / len(u)
    # This identity evaluates the selected gauge under the *empirical*
    # whitened covariances and remains exact under shrinkage/clipping.
    empirical_formula_total = float(
        np.trace(gauge.T @ task_metric @ gauge @ covariance_v)
        + np.trace(task_metric @ covariance_u)
        - 2.0 * np.trace(gauge @ cross @ task_metric)
    )
    predicted_loss = empirical_formula_total / outputs
    # The SVD is the exact optimum only for the population/surrogate problem in
    # which both whitened covariances are identity.  With regularization this
    # value must not be presented as the exact raw empirical optimum.
    surrogate_optimum = float(
        (2.0 * np.trace(task_metric) - 2.0 * singular_values.sum()) / outputs
    )
    return _affine(
        matrix, mt, mr, "paired_readout_oracle", "true_correspondence_plus_frozen_readout",
        centered_logit_mse_empirical=empirical_loss,
        centered_logit_mse_formula=predicted_loss,
        surrogate_centered_logit_mse_optimum=surrogate_optimum,
        formula_absolute_error=abs(empirical_loss - predicted_loss),
        whitened_target_covariance_residual=relative_frobenius(
            covariance_v, np.eye(covariance_v.shape[0])
        ),
        whitened_reference_covariance_residual=relative_frobenius(
            covariance_u, np.eye(covariance_u.shape[0])
        ),
        covariance_residual=relative_frobenius(matrix.T @ ct @ matrix, cr),
        gauge_orthogonality_error=float(
            np.linalg.norm(gauge.T @ gauge - np.eye(gauge.shape[0]), ord="fro")
        ),
    )


def fit_transform(
    method: str,
    source: np.ndarray,
    reference: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    rank: int | None = None,
    shrinkage: float = 1e-4,
    relative_floor: float = 1e-7,
    readout_weight: np.ndarray | None = None,
) -> AffineTransform:
    """Fit one named information-ladder transform."""

    name = method.lower().replace("-", "_")
    if name in {"none", "identity"}:
        return fit_identity(source, reference)
    if name in {"mean", "mean_only"}:
        return fit_mean(source, reference)
    if name in {"diagonal", "bn", "bn_style"}:
        return fit_diagonal(source, reference)
    if name == "coral":
        return fit_coral(source, reference, shrinkage=shrinkage, relative_floor=relative_floor)
    if name == "bures":
        return fit_bures(
            source, reference, shrinkage=shrinkage, relative_floor=relative_floor, rank=rank
        )
    if name in {"paired_procrustes", "procrustes"}:
        return fit_paired_procrustes(source, reference)
    if name in {"paired_oracle", "paired_gauge"}:
        return fit_paired_gauge(
            source, reference, shrinkage=shrinkage, relative_floor=relative_floor
        )
    if name in {"paired_readout_oracle", "readout_weighted_gauge"}:
        if readout_weight is None:
            raise ValueError("paired_readout_oracle requires the frozen readout weight")
        return fit_readout_weighted_gauge(
            source, reference, readout_weight, shrinkage=shrinkage,
            relative_floor=relative_floor,
        )
    if name in {"shuffled_pairs", "shuffled_procrustes", "shuffled_oracle"}:
        if rng is None:
            raise ValueError("shuffled-pair control requires an explicit RNG")
        permutation = rng.permutation(reference.shape[0])
        if np.array_equal(permutation, np.arange(reference.shape[0])) and len(permutation) > 1:
            permutation = np.roll(permutation, 1)
        if name == "shuffled_oracle":
            fitted = fit_paired_gauge(
                source, reference[permutation], shrinkage=shrinkage,
                relative_floor=relative_floor,
            )
        else:
            fitted = fit_paired_procrustes(source, reference[permutation])
        return AffineTransform(
            fitted.matrix, fitted.offset, "shuffled_pairs", "shuffled_correspondence_control",
            {**fitted.diagnostics, "permutation_changed": True},
        )
    raise KeyError(f"unknown transform method: {method}")
