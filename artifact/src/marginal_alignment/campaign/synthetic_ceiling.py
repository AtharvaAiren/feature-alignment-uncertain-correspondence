"""Controlled experiments for approximate ambiguity and its resolution.

This module adds two experiments without changing the sealed V1 suite.

``approximate_two_world``
    A hidden orientation is weakly encoded by ``n`` Bernoulli calibration
    observations.  The exact TV of the *joint count law* and the exact
    overlap-disagreement ceiling are computable.  Monte Carlo Bayes decisions
    validate the ceiling as distinguishability grows with sample size.

``identifiable_nongaussian``
    Two feature coordinates have the same population mean and variance, but
    one coordinate has controlled skewness.  The target either preserves or
    swaps the coordinates.  Second moments cannot identify that orientation;
    a third-moment marginal estimator can as calibration grows, while true
    pairs recover the permutation directly.  At zero skew the worlds become
    exactly indistinguishable and the third-moment estimator returns to the
    two-world ceiling.

Every public runner returns raw rows plus a summary.  The suite runner also
writes atomic JSONL/JSON artifacts and a content-hashed manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..ceiling import (
    bayes_orbit_ceiling,
    binary_gaussian_two_world_ceiling,
    candidate_mixture_world_accuracies,
    cyclic_permutation_group,
    task_disagreement,
)
from .approximate_ceiling import discrete_two_world_ceiling


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def binomial_count_mass(samples: int, probability: float) -> np.ndarray:
    """Stable mass function of the sufficient count for Bernoulli products."""

    n = int(samples)
    p = float(probability)
    if n < 0:
        raise ValueError("samples must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    if n == 0:
        return np.ones(1, dtype=np.float64)
    result = np.zeros(n + 1, dtype=np.float64)
    if p == 0.0:
        result[0] = 1.0
        return result
    if p == 1.0:
        result[-1] = 1.0
        return result
    indices = np.arange(n + 1, dtype=np.float64)
    log_mass = np.asarray([
        math.lgamma(n + 1) - math.lgamma(int(k) + 1) - math.lgamma(n - int(k) + 1)
        for k in indices
    ])
    log_mass += indices * math.log(p) + (n - indices) * math.log1p(-p)
    log_mass -= float(log_mass.max())
    result = np.exp(log_mass)
    return result / result.sum()


def approximate_orientation_ceiling(
    calibration_samples: int,
    epsilon: float,
    task_disagreement: float,
) -> dict[str, Any]:
    """Exact ceiling for the count-plus-query joint observation.

    In world zero the calibration bits are Bernoulli ``1/2-epsilon`` and in
    world one they are Bernoulli ``1/2+epsilon``.  An independent observed
    query-region bit equals one with probability ``task_disagreement``.  The
    desired labels agree off that region and equal the hidden world on it.
    Thus the query specifies *where* orientation matters, while the complete
    calibration sample supplies evidence about *which* orientation is true.
    """

    n = int(calibration_samples)
    eps = float(epsilon)
    delta = float(task_disagreement)
    if not 0.0 <= eps < 0.5:
        raise ValueError("epsilon must lie in [0,0.5)")
    if not 0.0 <= delta <= 1.0:
        raise ValueError("task_disagreement must lie in [0,1]")
    count0 = binomial_count_mass(n, 0.5 - eps)
    count1 = binomial_count_mass(n, 0.5 + eps)
    # Common support order: (count k, region r=0), (count k, region r=1).
    region_mass = np.asarray([1.0 - delta, delta], dtype=np.float64)
    probability0 = np.repeat(count0, 2) * np.tile(region_mass, n + 1)
    probability1 = np.repeat(count1, 2) * np.tile(region_mass, n + 1)
    labels0 = np.zeros(2 * (n + 1), dtype=np.int64)
    labels1 = np.tile(np.asarray([0, 1], dtype=np.int64), n + 1)
    result = discrete_two_world_ceiling(probability0, probability1, labels0, labels1)
    result.update({
        "calibration_samples": n,
        "epsilon": eps,
        "task_disagreement": delta,
        "calibration_count_total_variation": float(0.5 * np.abs(count0 - count1).sum()),
    })
    expected_overlap = delta * (1.0 - result["calibration_count_total_variation"])
    if abs(float(result["overlap_disagreement"]) - expected_overlap) > 2e-11:
        raise AssertionError("joint overlap did not factor in the independent query region")
    return result


def _simulate_approximate_accuracy(
    rng: np.random.Generator,
    *,
    calibration_samples: int,
    epsilon: float,
    task_disagreement: float,
    trials: int,
) -> float:
    n = int(calibration_samples)
    world = rng.integers(0, 2, size=trials)
    probabilities = 0.5 + (2.0 * world - 1.0) * float(epsilon)
    counts = rng.binomial(n, probabilities) if n else np.zeros(trials, dtype=np.int64)
    count0 = binomial_count_mass(n, 0.5 - epsilon)
    count1 = binomial_count_mass(n, 0.5 + epsilon)
    difference = count1[counts] - count0[counts]
    inferred_world = (difference > 0.0).astype(np.int64)
    ties = difference == 0.0
    inferred_world[ties] = rng.integers(0, 2, size=int(ties.sum()))
    disagreement_region = rng.random(trials) < task_disagreement
    desired = np.where(disagreement_region, world, 0)
    predicted = np.where(disagreement_region, inferred_world, 0)
    return float(np.mean(predicted == desired))


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    groups: list[dict[str, Any]] = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {name: value for name, value in zip(keys, key)}
        record["records"] = len(values)
        for metric in metrics:
            vector = np.asarray([row[metric] for row in values], dtype=np.float64)
            record[f"{metric}_mean"] = float(vector.mean())
            record[f"{metric}_std"] = float(vector.std(ddof=1)) if len(vector) > 1 else 0.0
        groups.append(record)
    return groups


def _rotation(dimension: int, angle: float) -> np.ndarray:
    if dimension < 2:
        raise ValueError("rotation dimension must be at least two")
    matrix = np.eye(dimension)
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix[:2, :2] = np.asarray([[cosine, -sine], [sine, cosine]])
    return matrix


def run_exact_indistinguishable_experiment(
    *,
    readout_angles_degrees: tuple[int, ...] = (0, 30, 60, 90, 120, 150, 180),
    orbit_sizes: tuple[int, ...] = (2, 4, 8),
    seeds: tuple[int, ...] = tuple(range(20)),
    samples_per_seed: int = 250_000,
) -> dict[str, Any]:
    """Validate exact Gaussian and finite-orbit indistinguishable-world values.

    This is the V2-owned equivalent of the sealed V1 ceiling validation.  It
    writes raw records into the V2 campaign rather than merely relying on an
    earlier result.  ``samples_per_seed`` concerns Monte Carlo evaluation, not
    the number of endpoint observations supplied to an aligner: these endpoint
    worlds are exactly indistinguishable by construction.
    """

    angles = tuple(int(value) for value in readout_angles_degrees)
    orbits = tuple(int(value) for value in orbit_sizes)
    if samples_per_seed < 1_000:
        raise ValueError("samples_per_seed must be at least 1000")
    if not seeds:
        raise ValueError("seeds must be nonempty")
    if not angles or any(value < 0 or value > 180 for value in angles):
        raise ValueError("readout angles must be nonempty and lie in [0,180]")
    if not orbits or any(value < 2 for value in orbits):
        raise ValueError("orbit sizes must be at least two")
    rows: list[dict[str, Any]] = []
    for degrees in angles:
        angle = np.deg2rad(degrees)
        action = _rotation(4, angle)
        theoretical_disagreement = float(angle / np.pi)
        theoretical_ceiling = binary_gaussian_two_world_ceiling(float(np.cos(angle)))
        for seed in seeds:
            rng = np.random.default_rng(10_000 + int(seed))
            features = rng.normal(size=(samples_per_seed, 4))
            identity_labels = (features[:, 0] >= 0.0).astype(np.int64)
            alternative_labels = ((features @ action)[:, 0] >= 0.0).astype(np.int64)
            disagreement = task_disagreement(identity_labels, alternative_labels)
            label_matrix = np.stack([identity_labels, alternative_labels])
            bayes = bayes_orbit_ceiling(label_matrix)
            equalizer = candidate_mixture_world_accuracies(label_matrix)
            error = max(
                abs(disagreement - theoretical_disagreement),
                abs(bayes - theoretical_ceiling),
                float(np.max(np.abs(equalizer - theoretical_ceiling))),
            )
            rows.append({
                "schema_version": "marginal_alignment.campaign.exact_ceiling_row.v1",
                "family": "binary_gaussian_rotation",
                "seed": int(seed),
                "samples": int(samples_per_seed),
                "angle_degrees": int(degrees),
                "theoretical_disagreement": theoretical_disagreement,
                "empirical_disagreement": disagreement,
                "theoretical_equal_prior_bayes_accuracy": theoretical_ceiling,
                "empirical_equal_prior_bayes_accuracy": bayes,
                "uniform_inverse_min_world_accuracy": float(equalizer.min()),
                "uniform_inverse_max_world_accuracy": float(equalizer.max()),
                "uniform_inverse_world_gap": float(equalizer.max() - equalizer.min()),
                "absolute_error": float(error),
            })
    for classes in orbits:
        dimension = max(classes, 8)
        group = cyclic_permutation_group(dimension, classes)
        for seed in seeds:
            rng = np.random.default_rng(20_000 + 100 * classes + int(seed))
            features = rng.normal(size=(samples_per_seed, dimension))
            label_matrix = np.stack([
                np.argmax((features @ action)[:, :classes], axis=1) for action in group
            ])
            bayes = bayes_orbit_ceiling(label_matrix)
            equalizer = candidate_mixture_world_accuracies(label_matrix)
            error = max(
                abs(bayes - 1.0 / classes),
                float(np.max(np.abs(equalizer - 1.0 / classes))),
            )
            rows.append({
                "schema_version": "marginal_alignment.campaign.exact_ceiling_row.v1",
                "family": "multiclass_cyclic_orbit",
                "seed": int(seed),
                "samples": int(samples_per_seed),
                "classes": int(classes),
                "worlds": int(classes),
                "theoretical_equal_prior_bayes_accuracy": float(1.0 / classes),
                "empirical_equal_prior_bayes_accuracy": bayes,
                "uniform_inverse_min_world_accuracy": float(equalizer.min()),
                "uniform_inverse_max_world_accuracy": float(equalizer.max()),
                "uniform_inverse_world_gap": float(equalizer.max() - equalizer.min()),
                "absolute_error": float(error),
            })
    groups = _aggregate(
        rows,
        keys=("family",),
        metrics=(
            "theoretical_equal_prior_bayes_accuracy",
            "empirical_equal_prior_bayes_accuracy",
            "uniform_inverse_world_gap",
            "absolute_error",
        ),
    )
    maximum_error = float(max(row["absolute_error"] for row in rows))
    maximum_world_gap = float(max(row["uniform_inverse_world_gap"] for row in rows))
    expected_cells = {
        ("binary_gaussian_rotation", int(value), int(seed))
        for value in angles for seed in seeds
    } | {
        ("multiclass_cyclic_orbit", int(value), int(seed))
        for value in orbits for seed in seeds
    }
    observed_cells = {
        (
            str(row["family"]),
            int(row["angle_degrees"] if row["family"] == "binary_gaussian_rotation"
                else row["classes"]),
            int(row["seed"]),
        )
        for row in rows
    }
    gates = {
        "requested_grid_complete": observed_cells == expected_cells,
        "closed_form_endpoint_angles_present": {0, 180}.issubset(set(angles)),
        "binary_and_multiclass_ambiguity_present": bool(angles and orbits),
        "all_requested_angles_executed": {
            int(row["angle_degrees"]) for row in rows
            if row["family"] == "binary_gaussian_rotation"
        } == set(angles),
        "all_requested_orbits_executed": {
            int(row["classes"]) for row in rows
            if row["family"] == "multiclass_cyclic_orbit"
        } == set(orbits),
        "closed_forms_match_monte_carlo": maximum_error < 0.01,
        "uniform_inverse_is_equalizer": maximum_world_gap < 0.01,
    }
    return {
        "schema_version": "marginal_alignment.campaign.exact_ceiling_experiment.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "rows": rows,
        "groups": groups,
        "scientific_gates": gates,
        "maximum_absolute_error": maximum_error,
        "maximum_uniform_inverse_world_gap": maximum_world_gap,
        "interpretation": (
            "These exact numerical values require declared exactly "
            "indistinguishable Gaussian/group worlds. They are not universal "
            "ceilings for natural corruptions."
        ),
    }


def run_approximate_symmetry_experiment(
    *,
    calibration_sizes: tuple[int, ...] = (0, 1, 4, 16, 64, 256, 1024),
    epsilons: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1),
    task_disagreements: tuple[float, ...] = (0.5, 1.0),
    seeds: tuple[int, ...] = tuple(range(20)),
    trials_per_seed: int = 50_000,
) -> dict[str, Any]:
    """Validate exact/coarse approximate-world ceilings over a locked grid."""

    if trials_per_seed < 100:
        raise ValueError("trials_per_seed must be at least 100")
    if not calibration_sizes or any(int(value) < 0 for value in calibration_sizes):
        raise ValueError("calibration_sizes must be nonempty and non-negative")
    if not epsilons or not task_disagreements or not seeds:
        raise ValueError("epsilon, disagreement, and seed axes must be nonempty")
    rows: list[dict[str, Any]] = []
    for delta in task_disagreements:
        for epsilon in epsilons:
            prior_exact = -np.inf
            for sample_size in calibration_sizes:
                theory = approximate_orientation_ceiling(sample_size, epsilon, delta)
                exact = float(theory["exact_equal_prior_bayes_accuracy"])
                if exact + 2e-12 < prior_exact:
                    raise AssertionError("exact Bayes accuracy decreased after observing more samples")
                prior_exact = exact
                for seed in seeds:
                    rng = np.random.default_rng(
                        3_100_000 + 100_003 * int(seed) + 997 * int(sample_size)
                        + int(round(100_000 * epsilon)) + int(round(10_000 * delta))
                    )
                    empirical = _simulate_approximate_accuracy(
                        rng,
                        calibration_samples=sample_size,
                        epsilon=epsilon,
                        task_disagreement=delta,
                        trials=trials_per_seed,
                    )
                    rows.append({
                        "schema_version": "marginal_alignment.campaign.approximate_row.v1",
                        "family": "approximate_two_world",
                        "seed": int(seed),
                        "calibration_samples": int(sample_size),
                        "epsilon": float(epsilon),
                        "task_disagreement": float(delta),
                        "joint_observation_total_variation": float(
                            theory["joint_observation_total_variation"]
                        ),
                        "overlap_disagreement": float(theory["overlap_disagreement"]),
                        "exact_equal_prior_bayes_accuracy": exact,
                        "coarse_joint_tv_accuracy_upper_bound": float(
                            theory["coarse_joint_tv_accuracy_upper_bound"]
                        ),
                        "empirical_bayes_accuracy": empirical,
                        "absolute_error": abs(empirical - exact),
                        "trials": int(trials_per_seed),
                    })
    groups = _aggregate(
        rows,
        keys=("calibration_samples", "epsilon", "task_disagreement"),
        metrics=(
            "joint_observation_total_variation",
            "exact_equal_prior_bayes_accuracy",
            "coarse_joint_tv_accuracy_upper_bound",
            "empirical_bayes_accuracy",
            "absolute_error",
        ),
    )
    maximum_z = 0.0
    for group in groups:
        exact = float(group["exact_equal_prior_bayes_accuracy_mean"])
        standard_error = math.sqrt(
            max(exact * (1.0 - exact), 1e-6) / (trials_per_seed * max(len(seeds), 1))
        )
        maximum_z = max(
            maximum_z,
            abs(float(group["empirical_bayes_accuracy_mean"]) - exact) / standard_error,
        )
    expected_cells = {
        (int(sample_size), float(epsilon), float(delta), int(seed))
        for delta in task_disagreements
        for epsilon in epsilons
        for sample_size in calibration_sizes
        for seed in seeds
    }
    observed_cells = {
        (
            int(row["calibration_samples"]), float(row["epsilon"]),
            float(row["task_disagreement"]), int(row["seed"]),
        )
        for row in rows
    }
    monotone = True
    group_lookup = {
        (
            int(group["calibration_samples"]), float(group["epsilon"]),
            float(group["task_disagreement"]),
        ): float(group["exact_equal_prior_bayes_accuracy_mean"])
        for group in groups
    }
    ordered_sizes = sorted({int(value) for value in calibration_sizes})
    for epsilon in epsilons:
        for delta in task_disagreements:
            values = [group_lookup[(size, float(epsilon), float(delta))]
                      for size in ordered_sizes]
            monotone &= bool(np.all(np.diff(values) >= -2e-12))
    gates = {
        "requested_grid_complete": observed_cells == expected_cells,
        "exact_and_approximate_regimes_present": (
            0.0 in {float(value) for value in epsilons}
            and any(float(value) > 0.0 for value in epsilons)
            and 0 in {int(value) for value in calibration_sizes}
            and any(int(value) > 0 for value in calibration_sizes)
            and any(float(value) > 0.0 for value in task_disagreements)
        ),
        "exact_ceiling_is_monotone_with_calibration_size": bool(monotone),
        "aggregate_monte_carlo_matches_exact": maximum_z <= 6.0,
        "exact_does_not_exceed_coarse_joint_tv_bound": all(
            float(row["exact_equal_prior_bayes_accuracy"])
            <= float(row["coarse_joint_tv_accuracy_upper_bound"]) + 2e-12
            for row in rows
        ),
    }
    return {
        "schema_version": "marginal_alignment.campaign.approximate_experiment.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "rows": rows,
        "groups": groups,
        "scientific_gates": gates,
        "maximum_aggregate_monte_carlo_z": float(maximum_z),
        "interpretation": (
            "As the complete calibration observation becomes distinguishable, "
            "the exact task ceiling rises. The one-sample endpoint TV is not "
            "substituted for the joint product-law TV."
        ),
    }


def asymmetric_population_skewness(asymmetry: float) -> float:
    """Population skew of the standardized Gaussian-mixture coordinate."""

    amplitude = float(asymmetry)
    if amplitude < 0.0 or not np.isfinite(amplitude):
        raise ValueError("asymmetry must be finite and non-negative")
    variance = 1.0 + 0.5 * amplitude * amplitude
    return float((0.25 * amplitude ** 3) / variance ** 1.5)


def _sample_reference_features(
    rng: np.random.Generator,
    samples: int,
    asymmetry: float,
) -> np.ndarray:
    amplitude = float(asymmetry)
    component = np.where(rng.random(samples) < (1.0 / 3.0), amplitude, -0.5 * amplitude)
    first = (component + rng.normal(size=samples)) / math.sqrt(1.0 + 0.5 * amplitude ** 2)
    second = rng.normal(size=samples)
    return np.column_stack([first, second])


def _swap(features: np.ndarray) -> np.ndarray:
    return np.asarray(features)[:, [1, 0]]


def _mean_cov_signature(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    covariance = np.cov(x, rowvar=False, bias=True)
    return np.concatenate([x.mean(axis=0), covariance.reshape(-1)])


def _third_moment_signature(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    centered = x - x.mean(axis=0)
    scale = np.maximum(np.sqrt(np.mean(np.square(centered), axis=0)), 1e-12)
    standardized_third = np.mean(np.power(centered / scale, 3), axis=0)
    return np.concatenate([_mean_cov_signature(x), standardized_third])


def _choose_signature_orientation(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    include_third_moment: bool,
    rng: np.random.Generator,
) -> int:
    signature = _third_moment_signature if include_third_moment else _mean_cov_signature
    reference_signature = signature(reference)
    distances = np.asarray([
        np.square(signature(target) - reference_signature).sum(),
        np.square(signature(_swap(target)) - reference_signature).sum(),
    ])
    if abs(float(distances[0] - distances[1])) <= 1e-14:
        return int(rng.integers(0, 2))
    return int(np.argmin(distances))


def _normal_log_density(values: np.ndarray) -> np.ndarray:
    return -0.5 * np.square(values) - 0.5 * math.log(2.0 * math.pi)


def _mixture_log_density(values: np.ndarray, asymmetry: float) -> np.ndarray:
    amplitude = float(asymmetry)
    scale = math.sqrt(1.0 + 0.5 * amplitude ** 2)
    unscaled = scale * np.asarray(values, dtype=np.float64)
    first = (
        math.log(1.0 / 3.0)
        - 0.5 * np.square(unscaled - amplitude)
        - 0.5 * math.log(2.0 * math.pi)
    )
    second = (
        math.log(2.0 / 3.0)
        - 0.5 * np.square(unscaled + 0.5 * amplitude)
        - 0.5 * math.log(2.0 * math.pi)
    )
    return np.logaddexp(first, second) + math.log(scale)


def _choose_likelihood_orientation(
    target: np.ndarray,
    asymmetry: float,
    *,
    rng: np.random.Generator,
) -> int:
    x = np.asarray(target, dtype=np.float64)
    log_identity = float(
        np.sum(_mixture_log_density(x[:, 0], asymmetry) + _normal_log_density(x[:, 1]))
    )
    log_swap = float(
        np.sum(_normal_log_density(x[:, 0]) + _mixture_log_density(x[:, 1], asymmetry))
    )
    if abs(log_identity - log_swap) <= 1e-12:
        return int(rng.integers(0, 2))
    # If target coordinate zero is the asymmetric coordinate, hidden world is
    # identity. Otherwise the inverse swap is selected.
    return int(log_swap > log_identity)


def _choose_paired_orientation(reference: np.ndarray, target: np.ndarray) -> int:
    losses = np.asarray([
        np.mean(np.square(target - reference)),
        np.mean(np.square(_swap(target) - reference)),
    ])
    return int(np.argmin(losses))


def _relative_covariance_error(first: np.ndarray, second: np.ndarray) -> float:
    covariance_first = np.cov(first, rowvar=False, bias=True)
    covariance_second = np.cov(second, rowvar=False, bias=True)
    denominator = max(float(np.linalg.norm(covariance_second, ord="fro")), 1e-12)
    return float(np.linalg.norm(covariance_first - covariance_second, ord="fro") / denominator)


def run_identifiable_nongaussian_experiment(
    *,
    regime: str = "standardized_asymmetric_mixture_swap",
    calibration_sizes: tuple[int, ...] = (16, 64, 256, 1024, 4096),
    asymmetries: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    seeds: tuple[int, ...] = tuple(range(20)),
    evaluation_size: int = 4096,
) -> dict[str, Any]:
    """Show the exact boundary between symmetric and identifiable marginals."""

    if regime != "standardized_asymmetric_mixture_swap":
        raise ValueError(
            "the only implemented identifiable regime is "
            "'standardized_asymmetric_mixture_swap'"
        )
    if evaluation_size < 100:
        raise ValueError("evaluation_size must be at least 100")
    if not calibration_sizes or any(int(value) < 2 for value in calibration_sizes):
        raise ValueError("calibration_sizes must be nonempty and at least two")
    if not asymmetries or not seeds:
        raise ValueError("asymmetry and seed axes must be nonempty")
    methods = (
        "identity_canonical",
        "second_moment_permutation",
        "third_moment_permutation",
        "likelihood_marginal_oracle",
        "paired_permutation",
    )
    rows: list[dict[str, Any]] = []
    for asymmetry in asymmetries:
        population_skew = asymmetric_population_skewness(asymmetry)
        for sample_size in calibration_sizes:
            for seed in seeds:
                for world in (0, 1):
                    # Independent endpoint samples for marginal estimators.
                    base_seed = (
                        7_000_000 + 100_003 * int(seed) + 1009 * int(sample_size)
                        + int(round(10_000 * asymmetry)) + 53 * world
                    )
                    rng = np.random.default_rng(base_seed)
                    reference_calibration = _sample_reference_features(
                        rng, sample_size, asymmetry
                    )
                    target_unpaired_base = _sample_reference_features(
                        rng, sample_size, asymmetry
                    )
                    target_calibration = (
                        target_unpaired_base if world == 0 else _swap(target_unpaired_base)
                    )
                    paired_reference = _sample_reference_features(rng, sample_size, asymmetry)
                    paired_target = paired_reference if world == 0 else _swap(paired_reference)
                    evaluation_reference = _sample_reference_features(
                        rng, evaluation_size, asymmetry
                    )
                    evaluation_target = (
                        evaluation_reference if world == 0 else _swap(evaluation_reference)
                    )
                    fitted = {
                        "identity_canonical": 0,
                        "second_moment_permutation": _choose_signature_orientation(
                            reference_calibration,
                            target_calibration,
                            include_third_moment=False,
                            rng=rng,
                        ),
                        "third_moment_permutation": _choose_signature_orientation(
                            reference_calibration,
                            target_calibration,
                            include_third_moment=True,
                            rng=rng,
                        ),
                        "likelihood_marginal_oracle": _choose_likelihood_orientation(
                            target_calibration,
                            asymmetry,
                            rng=rng,
                        ),
                        "paired_permutation": _choose_paired_orientation(
                            paired_reference, paired_target
                        ),
                    }
                    labels = (evaluation_reference[:, 0] >= 0.0).astype(np.int64)
                    for method in methods:
                        selected = int(fitted[method])
                        aligned = evaluation_target if selected == 0 else _swap(evaluation_target)
                        predictions = (aligned[:, 0] >= 0.0).astype(np.int64)
                        rows.append({
                            "schema_version": (
                                "marginal_alignment.campaign.identifiable_nongaussian_row.v1"
                            ),
                            "family": "identifiable_nongaussian_swap",
                            "regime": regime,
                            "seed": int(seed),
                            "hidden_world": int(world),
                            "calibration_samples": int(sample_size),
                            "evaluation_samples": int(evaluation_size),
                            "asymmetry": float(asymmetry),
                            "population_standardized_skewness": population_skew,
                            "method": method,
                            "fit_information": (
                                "true_pairs" if method == "paired_permutation"
                                else "full_density_family" if method == "likelihood_marginal_oracle"
                                else "third_marginal_moment" if method == "third_moment_permutation"
                                else "first_two_marginal_moments" if method == "second_moment_permutation"
                                else "fixed_representative"
                            ),
                            "selected_inverse": selected,
                            "orientation_correct": float(selected == world),
                            "task_agreement": float(np.mean(predictions == labels)),
                            "paired_feature_mse": float(
                                np.mean(np.square(aligned - evaluation_reference))
                            ),
                            "heldout_mean_error": float(
                                np.linalg.norm(aligned.mean(0) - evaluation_reference.mean(0))
                            ),
                            "heldout_covariance_relative_error": _relative_covariance_error(
                                aligned, evaluation_reference
                            ),
                        })
    groups = _aggregate(
        rows,
        keys=("calibration_samples", "asymmetry", "method"),
        metrics=(
            "orientation_correct",
            "task_agreement",
            "paired_feature_mse",
            "heldout_mean_error",
            "heldout_covariance_relative_error",
        ),
    )
    lookup = {
        (int(group["calibration_samples"]), float(group["asymmetry"]), str(group["method"])): group
        for group in groups
    }
    largest_n = max(int(value) for value in calibration_sizes)
    strongest = max(float(value) for value in asymmetries)
    zero_key = (largest_n, 0.0, "third_moment_permutation")
    strong_third_key = (largest_n, strongest, "third_moment_permutation")
    strong_second_key = (largest_n, strongest, "second_moment_permutation")
    expected_cells = {
        (
            int(sample_size), float(asymmetry), int(seed), int(world), method,
        )
        for sample_size in calibration_sizes
        for asymmetry in asymmetries
        for seed in seeds
        for world in (0, 1)
        for method in methods
    }
    observed_cells = {
        (
            int(row["calibration_samples"]), float(row["asymmetry"]),
            int(row["seed"]), int(row["hidden_world"]), str(row["method"]),
        )
        for row in rows
    }
    gates: dict[str, bool] = {
        "requested_grid_complete": observed_cells == expected_cells,
        "symmetric_and_identifiable_regimes_present": (
            0.0 in {float(value) for value in asymmetries}
            and any(float(value) > 0.0 for value in asymmetries)
            and len({int(value) for value in calibration_sizes}) >= 2
        ),
        "zero_asymmetry_is_unidentified": (
            0.0 not in {float(value) for value in asymmetries}
            or (
                zero_key in lookup
                and abs(float(lookup[zero_key]["orientation_correct_mean"]) - 0.5) <= 0.25
            )
        ),
        "third_moment_identifies_strong_asymmetry": (
            strongest <= 0.0
            or (
                strong_third_key in lookup
                and float(lookup[strong_third_key]["orientation_correct_mean"]) >= 0.9
            )
        ),
        "second_moment_does_not_identify_strong_asymmetry": (
            strongest <= 0.0
            or (
                strong_second_key in lookup
                and float(lookup[strong_second_key]["orientation_correct_mean"]) <= 0.75
            )
        ),
    }
    paired_values = [
        float(group["task_agreement_mean"])
        for group in groups if group["method"] == "paired_permutation"
    ]
    gates["paired_oracle_recovers"] = bool(paired_values and min(paired_values) >= 0.999)
    return {
        "schema_version": "marginal_alignment.campaign.identifiable_nongaussian_experiment.v1",
        "regime": regime,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "rows": rows,
        "groups": groups,
        "scientific_gates": gates,
        "interpretation": (
            "Equal mean and covariance do not reveal the swap. Nonzero "
            "asymmetry breaks the full-law symmetry, so third-order/full-law "
            "marginal information can identify orientation with enough "
            "samples. At zero asymmetry only correspondence resolves it."
        ),
    }


def run_synthetic_ceiling_suite(
    output_dir: str | Path,
    *,
    exact_kwargs: dict[str, Any] | None = None,
    approximate_kwargs: dict[str, Any] | None = None,
    identifiable_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all three arms and write raw, summarized, content-hashed output."""

    started = time.time()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exact = run_exact_indistinguishable_experiment(**(exact_kwargs or {}))
    approximate = run_approximate_symmetry_experiment(**(approximate_kwargs or {}))
    identifiable = run_identifiable_nongaussian_experiment(**(identifiable_kwargs or {}))
    exact_rows = output / "exact_indistinguishable_rows.jsonl"
    approximate_rows = output / "approximate_two_world_rows.jsonl"
    identifiable_rows = output / "identifiable_nongaussian_rows.jsonl"
    summary_path = output / "summary.json"
    _json_rows(exact_rows, exact["rows"])
    _json_rows(approximate_rows, approximate["rows"])
    _json_rows(identifiable_rows, identifiable["rows"])
    summary = {
        "schema_version": "marginal_alignment.campaign.synthetic_ceiling_suite.v1",
        "status": (
            "PASS" if exact["status"] == approximate["status"]
            == identifiable["status"] == "PASS" else "FAIL"
        ),
        "exact_indistinguishable": {
            key: value for key, value in exact.items() if key != "rows"
        },
        "approximate_two_world": {
            key: value for key, value in approximate.items() if key != "rows"
        },
        "identifiable_nongaussian": {
            key: value for key, value in identifiable.items() if key != "rows"
        },
    }
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    manifest = {
        "schema_version": "marginal_alignment.campaign.synthetic_manifest.v1",
        "status": summary["status"],
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (exact_rows, approximate_rows, identifiable_rows, summary_path)
        ],
        "row_counts": {
            "exact_indistinguishable": len(exact["rows"]),
            "approximate_two_world": len(approximate["rows"]),
            "identifiable_nongaussian": len(identifiable["rows"]),
        },
    }
    manifest_path = output / "manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return {"summary": summary, "manifest": manifest, "output_dir": str(output)}
