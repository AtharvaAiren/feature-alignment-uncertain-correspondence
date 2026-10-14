"""Distribution, samplewise, and frozen-task metrics for V2 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marginal_alignment.linalg import as_samples
from marginal_alignment.metrics import logits


def _mean_covariance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(0)
    centered = features - mean
    return mean, centered.T @ centered / len(centered)


def _correlation(features: np.ndarray) -> np.ndarray:
    _, covariance = _mean_covariance(features)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    return covariance / scale[:, None] / scale[None, :]


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-15))


def _standardized_moments(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features - features.mean(0)
    standardized = centered / np.maximum(features.std(0), 1e-12)
    return np.mean(standardized ** 3, axis=0), np.mean(standardized ** 4, axis=0) - 3.0


def _standardized_vectors(features: np.ndarray) -> np.ndarray:
    """Coordinate-standardize a sample with the population convention.

    The diagonal-alignment theorem is a statement about these vectors, not
    merely about whether the aligned correlation happens to be close to the
    reference correlation.  Keeping this as a separate diagnostic lets the
    experiment distinguish exact preservation of source dependence from
    successful repair of reference statistics.
    """

    centered = features - features.mean(0)
    return centered / np.maximum(features.std(0), 1e-12)


def _ordinal_coordinate_ranks(features: np.ndarray) -> np.ndarray:
    """Return stable, normalized within-coordinate ordinal ranks.

    A coordinatewise strictly increasing map preserves these ranks exactly.
    Stable ordering gives a deterministic convention when an empirical
    quantile map introduces ties; the resulting number remains a diagnostic,
    not a claim that finite tied samples define a unique copula.
    """

    features = as_samples(features)
    order = np.argsort(features, axis=0, kind="stable")
    ranks = np.empty_like(order)
    ordinal = np.broadcast_to(np.arange(len(features))[:, None], order.shape)
    np.put_along_axis(ranks, order, ordinal, axis=0)
    return ranks.astype(np.float64) / max(len(features) - 1, 1)


def coordinate_w2_squared(left: np.ndarray, right: np.ndarray,
                          max_coordinates: int | None = None) -> float:
    """Mean empirical one-dimensional W2 squared over coordinates."""

    left, right = as_samples(left), as_samples(right)
    dimension = left.shape[1]
    coordinates = np.arange(dimension)
    if max_coordinates is not None and dimension > max_coordinates:
        coordinates = np.linspace(0, dimension - 1, max_coordinates, dtype=int)
    probabilities = (np.arange(max(len(left), len(right))) + 0.5) / max(len(left), len(right))
    left_grid = (np.arange(len(left)) + 0.5) / len(left)
    right_grid = (np.arange(len(right)) + 0.5) / len(right)
    values = []
    for coordinate in coordinates:
        a = np.interp(probabilities, left_grid, np.sort(left[:, coordinate]))
        b = np.interp(probabilities, right_grid, np.sort(right[:, coordinate]))
        values.append(float(np.mean(np.square(a - b))))
    return float(np.mean(values))


def sliced_w2_squared(left: np.ndarray, right: np.ndarray, *,
                      rng: np.random.Generator, projections: int = 64) -> float:
    left, right = as_samples(left), as_samples(right)
    if left.shape[1] != right.shape[1]:
        raise ValueError("sliced W2 inputs have different dimensions")
    directions = rng.normal(size=(projections, left.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
    probabilities = (np.arange(max(len(left), len(right))) + 0.5) / max(len(left), len(right))
    left_grid = (np.arange(len(left)) + 0.5) / len(left)
    right_grid = (np.arange(len(right)) + 0.5) / len(right)
    values = []
    for direction in directions:
        a = np.interp(probabilities, left_grid, np.sort(left @ direction))
        b = np.interp(probabilities, right_grid, np.sort(right @ direction))
        values.append(float(np.mean(np.square(a - b))))
    return float(np.mean(values))


def _median_bandwidth_squared(left: np.ndarray, right: np.ndarray, *,
                              rng: np.random.Generator,
                              bandwidth_sample: int = 512) -> float:
    pooled = np.concatenate([left, right])
    selection = rng.choice(len(pooled), size=min(bandwidth_sample, len(pooled)), replace=False)
    probe = pooled[selection]
    if len(probe) <= 1:
        return 1.0
    pairs = rng.integers(0, len(probe), size=(min(4096, len(probe) ** 2), 2))
    squared = np.sum(np.square(probe[pairs[:, 0]] - probe[pairs[:, 1]]), axis=1)
    positive = squared[squared > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def rff_mmd_squared(left: np.ndarray, right: np.ndarray, *, rng: np.random.Generator,
                    features: int = 512, bandwidth_sample: int = 512,
                    bandwidth_squared: float | None = None) -> float:
    """Deterministic-seeded linear-memory Gaussian-kernel MMD approximation."""

    left, right = as_samples(left), as_samples(right)
    if bandwidth_squared is None:
        bandwidth_squared = _median_bandwidth_squared(
            left, right, rng=rng, bandwidth_sample=bandwidth_sample
        )
    weights = rng.normal(
        scale=1.0 / np.sqrt(max(bandwidth_squared, 1e-15)),
        size=(left.shape[1], features),
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=features)
    scale = np.sqrt(2.0 / features)
    left_embedding = scale * np.cos(left @ weights + phase).mean(0)
    right_embedding = scale * np.cos(right @ weights + phase).mean(0)
    return float(np.square(left_embedding - right_embedding).sum())


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    predictions = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = predictions == labels
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(mask):
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def classification_metrics(logit_values: np.ndarray, labels: np.ndarray, *,
                           reference_logits: np.ndarray | None = None,
                           ece_bins: int = 15) -> dict[str, float]:
    """Compute every task metric from one declared set of logits.

    This helper is shared by frozen-map evaluation and consumer-changing
    labeled comparators.  It prevents a comparator from accidentally reporting
    adapted accuracy/NLL while retaining stale frozen-head Brier or ECE values.
    """

    logit_values = np.asarray(logit_values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logit_values.ndim != 2 or labels.shape != (len(logit_values),):
        raise ValueError("classification logits/labels have incompatible shapes")
    if np.any(labels < 0) or np.any(labels >= logit_values.shape[1]):
        raise ValueError("classification labels fall outside the logit class range")
    probabilities = _softmax(logit_values)
    predictions = logit_values.argmax(1)
    true_probability = probabilities[np.arange(len(labels)), labels]
    result = {
        "accuracy": float(np.mean(predictions == labels)),
        "negative_log_likelihood": float(-np.log(np.maximum(
            true_probability, 1e-15
        )).mean()),
        # ||p-e_y||^2 = ||p||^2 - 2 p_y + 1.  This avoids materializing an
        # evaluation_rows x classes one-hot matrix (80 MB at ImageNet scale).
        "brier_score": float(np.mean(
            np.sum(np.square(probabilities), axis=1) - 2.0 * true_probability + 1.0
        )),
        "expected_calibration_error": _ece(probabilities, labels, bins=ece_bins),
    }
    if reference_logits is not None:
        reference_logits = np.asarray(reference_logits, dtype=np.float64)
        if reference_logits.shape != logit_values.shape:
            raise ValueError("reference logits must match candidate logits")
        result["reference_prediction_agreement"] = float(np.mean(
            predictions == reference_logits.argmax(1)
        ))
    return result


def distribution_metrics(candidate: np.ndarray, reference: np.ndarray, *,
                         rng: np.random.Generator, projections: int = 64,
                         mmd_features: int = 512,
                         mmd_bandwidth_squared: float | None = None) -> dict[str, float]:
    candidate, reference = as_samples(candidate), as_samples(reference)
    candidate_mean, candidate_covariance = _mean_covariance(candidate)
    reference_mean, reference_covariance = _mean_covariance(reference)
    candidate_skew, candidate_kurtosis = _standardized_moments(candidate)
    reference_skew, reference_kurtosis = _standardized_moments(reference)
    return {
        "mean_l2": float(np.linalg.norm(candidate_mean - reference_mean)),
        "coordinate_variance_relative_error": _relative(
            np.diag(candidate_covariance), np.diag(reference_covariance)
        ),
        "covariance_relative_error": _relative(candidate_covariance, reference_covariance),
        "correlation_relative_error": _relative(
            _correlation(candidate), _correlation(reference)
        ),
        "standardized_skew_relative_error": _relative(candidate_skew, reference_skew),
        "standardized_skew_rms_difference": float(
            np.sqrt(np.mean(np.square(candidate_skew - reference_skew)))
        ),
        "standardized_kurtosis_relative_error": _relative(
            candidate_kurtosis, reference_kurtosis
        ),
        "standardized_kurtosis_rms_difference": float(
            np.sqrt(np.mean(np.square(candidate_kurtosis - reference_kurtosis)))
        ),
        "coordinate_w2_squared": coordinate_w2_squared(candidate, reference),
        "sliced_w2_squared": sliced_w2_squared(
            candidate, reference, rng=rng, projections=projections
        ),
        "rff_mmd_squared": rff_mmd_squared(
            candidate, reference, rng=rng, features=mmd_features,
            bandwidth_squared=mmd_bandwidth_squared,
        ),
    }


@dataclass(frozen=True)
class EvaluationContext:
    """Invariant work for every method on one fixed evaluation split."""

    reference: np.ndarray
    source: np.ndarray
    labels: np.ndarray
    readout_weight: np.ndarray
    readout_bias: np.ndarray | None
    distribution_reference: np.ndarray
    distribution_source: np.ndarray
    diagnostic_rows: int
    projections: int
    mmd_features: int
    metric_seed: int
    mmd_bandwidth_squared: float
    source_distribution_metrics: dict[str, float]
    source_standardized: np.ndarray
    source_correlation: np.ndarray
    source_coordinate_ranks: np.ndarray
    reference_logits: np.ndarray
    source_logits: np.ndarray
    reference_task: dict[str, float]
    source_task: dict[str, float]


def prepare_evaluation_context(
    reference: np.ndarray, source: np.ndarray, *, labels: np.ndarray,
    readout_weight: np.ndarray, readout_bias: np.ndarray | None = None,
    rng: np.random.Generator, projections: int = 64, mmd_features: int = 512,
    distribution_limit: int | None = None,
) -> EvaluationContext:
    """Precompute reference/source statistics once for a fixed split.

    The RNG draw order is intentionally identical to the historical evaluator,
    so using a context changes runtime, not the persisted metric definition.
    """

    reference, source = map(as_samples, (reference, source))
    if reference.shape != source.shape:
        raise ValueError("evaluation arrays must be naturally paired and same-shaped")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(reference),):
        raise ValueError("labels must have one value per evaluation row")
    if distribution_limit is not None:
        distribution_limit = int(distribution_limit)
        if distribution_limit < 2:
            raise ValueError("distribution_limit must be at least two")
        diagnostic_rows = min(distribution_limit, len(reference))
    else:
        diagnostic_rows = len(reference)
    distribution_reference = reference[:diagnostic_rows]
    distribution_source = source[:diagnostic_rows]
    metric_seed = int(rng.integers(0, 2**63 - 1))
    bandwidth_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    bandwidth = _median_bandwidth_squared(
        distribution_source, distribution_reference, rng=bandwidth_rng
    )
    source_distribution = distribution_metrics(
        distribution_source, distribution_reference,
        rng=np.random.default_rng(metric_seed), projections=int(projections),
        mmd_features=int(mmd_features), mmd_bandwidth_squared=bandwidth,
    )
    weight = np.asarray(readout_weight, dtype=np.float64)
    bias = None if readout_bias is None else np.asarray(readout_bias, dtype=np.float64)
    reference_logits = logits(reference, weight, bias)
    source_logits = logits(source, weight, bias)
    return EvaluationContext(
        reference=reference, source=source, labels=labels,
        readout_weight=weight, readout_bias=bias,
        distribution_reference=distribution_reference,
        distribution_source=distribution_source,
        diagnostic_rows=diagnostic_rows, projections=int(projections),
        mmd_features=int(mmd_features), metric_seed=metric_seed,
        mmd_bandwidth_squared=float(bandwidth),
        source_distribution_metrics=source_distribution,
        source_standardized=_standardized_vectors(distribution_source),
        source_correlation=_correlation(distribution_source),
        source_coordinate_ranks=_ordinal_coordinate_ranks(distribution_source),
        reference_logits=reference_logits, source_logits=source_logits,
        reference_task=classification_metrics(reference_logits, labels),
        source_task=classification_metrics(
            source_logits, labels, reference_logits=reference_logits
        ),
    )


def evaluate_alignment_with_context(
    aligned: np.ndarray, context: EvaluationContext,
) -> dict[str, float]:
    """Evaluate only method-dependent quantities against a prepared split."""

    aligned = as_samples(aligned)
    if aligned.shape != context.reference.shape:
        raise ValueError("aligned evaluation array differs from prepared split shape")
    distribution_aligned = aligned[:context.diagnostic_rows]
    metrics = {
        f"source_{key}": value
        for key, value in context.source_distribution_metrics.items()
    }
    metrics.update({
        f"aligned_{key}": value for key, value in distribution_metrics(
            distribution_aligned, context.distribution_reference,
            rng=np.random.default_rng(context.metric_seed),
            projections=context.projections, mmd_features=context.mmd_features,
            mmd_bandwidth_squared=context.mmd_bandwidth_squared,
        ).items()
    })
    metrics["mmd_bandwidth_squared"] = context.mmd_bandwidth_squared
    metrics["distribution_evaluation_rows"] = float(context.diagnostic_rows)
    aligned_standardized = _standardized_vectors(distribution_aligned)
    metrics["source_to_aligned_standardized_vector_mse"] = float(
        np.mean(np.square(context.source_standardized - aligned_standardized))
    )
    metrics["source_to_aligned_correlation_relative_change"] = _relative(
        _correlation(distribution_aligned), context.source_correlation
    )
    metrics["source_to_aligned_coordinate_rank_mse"] = float(np.mean(np.square(
        _ordinal_coordinate_ranks(distribution_aligned)
        - context.source_coordinate_ranks
    )))
    aligned_logits = logits(
        aligned, context.readout_weight, context.readout_bias
    )
    aligned_task = classification_metrics(
        aligned_logits, context.labels, reference_logits=context.reference_logits
    )
    aligned_accuracy = aligned_task["accuracy"]
    source_accuracy = context.source_task["accuracy"]
    reference_accuracy = context.reference_task["accuracy"]
    denominator = reference_accuracy - source_accuracy
    logit_delta = aligned_logits - context.reference_logits
    metrics.update({
        "paired_feature_mse": float(np.mean(np.square(
            aligned - context.reference
        ))),
        "centered_logit_mse": float(np.mean(np.square(
            logit_delta - logit_delta.mean(axis=1, keepdims=True)
        ))),
        "reference_prediction_agreement": aligned_task[
            "reference_prediction_agreement"
        ],
        "source_prediction_agreement": context.source_task[
            "reference_prediction_agreement"
        ],
        "reference_accuracy": reference_accuracy,
        "source_accuracy": source_accuracy,
        "aligned_accuracy": aligned_accuracy,
        "accuracy_recovery_fraction": float(
            (aligned_accuracy - source_accuracy) / denominator
        ) if abs(denominator) > 1e-12 else float("nan"),
        "negative_log_likelihood": aligned_task["negative_log_likelihood"],
        "source_negative_log_likelihood": context.source_task[
            "negative_log_likelihood"
        ],
        "reference_negative_log_likelihood": context.reference_task[
            "negative_log_likelihood"
        ],
        "brier_score": aligned_task["brier_score"],
        "source_brier_score": context.source_task["brier_score"],
        "reference_brier_score": context.reference_task["brier_score"],
        "expected_calibration_error": aligned_task["expected_calibration_error"],
        "source_expected_calibration_error": context.source_task[
            "expected_calibration_error"
        ],
        "reference_expected_calibration_error": context.reference_task[
            "expected_calibration_error"
        ],
    })
    return metrics


def evaluate_alignment(reference: np.ndarray, source: np.ndarray, aligned: np.ndarray, *,
                            labels: np.ndarray, readout_weight: np.ndarray,
                            readout_bias: np.ndarray | None = None,
                            rng: np.random.Generator,
                            projections: int = 64,
                            mmd_features: int = 512,
                            distribution_limit: int | None = None) -> dict[str, float]:
    """Keep distribution repair, paired recovery, and task recovery separate."""

    context = prepare_evaluation_context(
        reference, source, labels=labels, readout_weight=readout_weight,
        readout_bias=readout_bias, rng=rng, projections=projections,
        mmd_features=mmd_features, distribution_limit=distribution_limit,
    )
    return evaluate_alignment_with_context(aligned, context)
