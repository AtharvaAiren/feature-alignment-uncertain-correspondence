"""Monte Carlo validation of the exact two-world and finite-group ceilings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ceiling import (
    bayes_orbit_ceiling,
    binary_gaussian_two_world_ceiling,
    candidate_mixture_world_accuracies,
    cyclic_permutation_group,
    task_disagreement,
)


def _rotation(dimension: int, angle: float) -> np.ndarray:
    matrix = np.eye(dimension)
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix[:2, :2] = np.asarray([[cosine, -sine], [sine, cosine]])
    return matrix


def run_ceiling_validation(
    output_path: str | Path,
    *,
    samples: int = 250_000,
    seeds: tuple[int, ...] = tuple(range(20)),
) -> dict[str, object]:
    records: list[dict[str, float | int | str]] = []
    # Binary halfspace: the exact ceiling varies continuously with readout/drift angle.
    for degrees in (0, 30, 60, 90, 120, 180):
        angle = np.deg2rad(degrees)
        action = _rotation(4, angle)
        theoretical_delta = float(angle / np.pi)
        theoretical_ceiling = binary_gaussian_two_world_ceiling(np.cos(angle))
        for seed in seeds:
            rng = np.random.default_rng(10_000 + seed)
            x = rng.normal(size=(samples, 4))
            labels_identity = (x[:, 0] >= 0.0).astype(np.int64)
            labels_rotated_world = ((x @ action)[:, 0] >= 0.0).astype(np.int64)
            delta = task_disagreement(labels_identity, labels_rotated_world)
            labels = np.stack([labels_identity, labels_rotated_world])
            empirical_bayes_ceiling = bayes_orbit_ceiling(labels)
            equalizer_accuracies = candidate_mixture_world_accuracies(labels)
            fixed_identity_accuracies = np.asarray([1.0, 1.0 - delta])
            equalizer_error = float(
                np.max(np.abs(equalizer_accuracies - theoretical_ceiling))
            )
            records.append({
                "family": "binary_gaussian_rotation",
                "seed": seed,
                "angle_degrees": degrees,
                "theoretical_disagreement": theoretical_delta,
                "empirical_disagreement": delta,
                "theoretical_ceiling": theoretical_ceiling,
                "empirical_bayes_ceiling": empirical_bayes_ceiling,
                "fixed_identity_bayes_accuracy": float(fixed_identity_accuracies.mean()),
                "fixed_identity_worst_world_accuracy": float(fixed_identity_accuracies.min()),
                "equalizer_min_world_accuracy": float(equalizer_accuracies.min()),
                "equalizer_max_world_accuracy": float(equalizer_accuracies.max()),
                "equalizer_world_gap": float(equalizer_accuracies.max() - equalizer_accuracies.min()),
                "equalizer_absolute_ceiling_error": equalizer_error,
                "absolute_ceiling_error": max(
                    abs(empirical_bayes_ceiling - theoretical_ceiling), equalizer_error
                ),
            })
    # Multiclass cyclic ambiguity: the exact ceiling is 1/K.
    for classes in (2, 4, 8):
        dimension = max(classes, 8)
        group = cyclic_permutation_group(dimension, classes)
        for seed in seeds:
            rng = np.random.default_rng(20_000 + 100 * classes + seed)
            x = rng.normal(size=(samples, dimension))
            label_matrix = np.stack([np.argmax((x @ action)[:, :classes], axis=1) for action in group])
            empirical_bayes_ceiling = bayes_orbit_ceiling(label_matrix)
            equalizer_accuracies = candidate_mixture_world_accuracies(label_matrix)
            fixed_identity_accuracies = np.mean(label_matrix == label_matrix[0], axis=1)
            equalizer_error = float(
                np.max(np.abs(equalizer_accuracies - 1.0 / classes))
            )
            records.append({
                "family": "multiclass_cyclic",
                "seed": seed,
                "classes": classes,
                "worlds": classes,
                "theoretical_ceiling": 1.0 / classes,
                "empirical_bayes_ceiling": empirical_bayes_ceiling,
                "fixed_identity_bayes_accuracy": float(fixed_identity_accuracies.mean()),
                "fixed_identity_worst_world_accuracy": float(fixed_identity_accuracies.min()),
                "equalizer_min_world_accuracy": float(equalizer_accuracies.min()),
                "equalizer_max_world_accuracy": float(equalizer_accuracies.max()),
                "equalizer_world_gap": float(equalizer_accuracies.max() - equalizer_accuracies.min()),
                "equalizer_absolute_ceiling_error": equalizer_error,
                "absolute_ceiling_error": max(
                    abs(empirical_bayes_ceiling - 1.0 / classes), equalizer_error
                ),
            })
    maximum_error = float(max(float(row["absolute_ceiling_error"]) for row in records))
    result = {
        "schema_version": "marginal_alignment.ceiling_validation.v2",
        "samples_per_seed": samples,
        "seeds": list(seeds),
        "records": records,
        "maximum_absolute_ceiling_error": maximum_error,
        "status": "PASS" if maximum_error < 0.005 else "FAIL",
        "interpretation": (
            "The posterior-mode calculation validates the declared uniform-world Bayes ceiling, "
            "and the randomized inverse equalizer attains the same value in every fixed world. "
            "Fixed unpaired representatives need not attain the minimax value: their worst-world "
            "accuracy is reported separately. These are restricted, not universal real-data bounds."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return result
