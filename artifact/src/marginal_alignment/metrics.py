"""Separate marginal repair, sample recovery, and frozen-task recovery."""

from __future__ import annotations

import numpy as np

from .linalg import (
    as_samples, gram_square_root, population_mean_cov, relative_frobenius,
    symmetric_power,
)


def logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    x = as_samples(features)
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != x.shape[1]:
        raise ValueError("weight must have shape [classes,d]")
    b = np.zeros(w.shape[0]) if bias is None else np.asarray(bias, dtype=np.float64)
    if b.shape != (w.shape[0],):
        raise ValueError("bias must have one value per class")
    return x @ w.T + b


def predictions(features: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    return np.argmax(logits(features, weight, bias), axis=1)


def accuracy(features: np.ndarray, labels: np.ndarray, weight: np.ndarray,
             bias: np.ndarray | None = None) -> float:
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(features),):
        raise ValueError("labels must have shape [n]")
    return float(np.mean(predictions(features, weight, bias) == y))


def gaussian_w2_squared(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    sqrt_a = symmetric_power(covariance_a, 0.5)
    sqrt_b = symmetric_power(covariance_b, 0.5)
    middle_sqrt, _ = gram_square_root(sqrt_b @ sqrt_a)
    covariance_term = np.trace(covariance_a + covariance_b - 2.0 * middle_sqrt)
    return float(np.square(mean_a - mean_b).sum() + max(float(covariance_term), 0.0))


def evaluate_alignment(
    reference: np.ndarray,
    source: np.ndarray,
    aligned: np.ndarray,
    *,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    compute_gaussian_w2: bool = True,
) -> dict[str, float]:
    """Evaluate untouched paired rows; fitting is handled by the caller."""

    r = as_samples(reference, name="reference")
    s = as_samples(source, name="source")
    a = as_samples(aligned, name="aligned")
    if r.shape != s.shape or r.shape != a.shape:
        raise ValueError("reference, source, and aligned arrays must be paired and same-shaped")
    mr, cr = population_mean_cov(r)
    ms, cs = population_mean_cov(s)
    ma, ca = population_mean_cov(a)
    pred_ref = predictions(r, weight, bias)
    pred_aligned = predictions(a, weight, bias)
    pred_source = predictions(s, weight, bias)
    reference_acc = accuracy(r, labels, weight, bias)
    source_acc = accuracy(s, labels, weight, bias)
    aligned_acc = accuracy(a, labels, weight, bias)
    denominator = reference_acc - source_acc
    recovery = float((aligned_acc - source_acc) / denominator) if abs(denominator) > 1e-12 else float("nan")
    centered = np.eye(weight.shape[0]) - np.ones((weight.shape[0], weight.shape[0])) / weight.shape[0]
    centered_weight = centered @ np.asarray(weight, dtype=np.float64)
    logit_error = (a - r) @ centered_weight.T
    return {
        "source_mean_error": float(np.linalg.norm(ms - mr)),
        "aligned_mean_error": float(np.linalg.norm(ma - mr)),
        "source_covariance_relative_error": relative_frobenius(cs, cr),
        "aligned_covariance_relative_error": relative_frobenius(ca, cr),
        "source_gaussian_w2_squared": (
            gaussian_w2_squared(ms, cs, mr, cr) if compute_gaussian_w2 else float("nan")
        ),
        "aligned_gaussian_w2_squared": (
            gaussian_w2_squared(ma, ca, mr, cr) if compute_gaussian_w2 else float("nan")
        ),
        "paired_feature_mse": float(np.mean(np.square(a - r))),
        "centered_logit_mse": float(np.mean(np.square(logit_error))),
        "reference_prediction_agreement": float(np.mean(pred_aligned == pred_ref)),
        "source_prediction_agreement": float(np.mean(pred_source == pred_ref)),
        "reference_accuracy": reference_acc,
        "source_accuracy": source_acc,
        "aligned_accuracy": aligned_acc,
        "accuracy_recovery_fraction": recovery,
    }
