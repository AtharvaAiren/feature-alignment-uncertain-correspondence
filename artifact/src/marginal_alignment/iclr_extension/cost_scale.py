"""Cost-scale diagnostics for finite correspondence assignments.

The correspondence witness constrains total assignment cost relative to the
minimum-cost bijection.  A relative percentage is difficult to interpret on
its own: in a concentrated high-dimensional geometry, even a uniformly random
bijection might lie close to the minimum.  This module supplies exact moments
for the cost of a uniformly random bijection and normalizes declared budgets by
the random-minus-minimum gap.

These routines are descriptive.  They do not alter a fitted map, construct a
new witness, or use labels.  An optional identity assignment is scored only for
paired positive controls after the geometry and witness split are frozen.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "marginal_alignment.correspondence_cost_scale.v1"


def _endpoint_pair(
    target: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(target, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
        or left.shape[0] < 2
        or left.shape[1] < 1
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError(
            "target and reference must be finite, equal-shape matrices with "
            "at least two rows"
        )
    return left, right


def assignment_cost(
    target: np.ndarray,
    reference: np.ndarray,
    assignment: np.ndarray,
) -> float:
    """Return the witness's dimension-normalized squared-distance cost."""

    left, right = _endpoint_pair(target, reference)
    permutation = np.asarray(assignment)
    rows, dimension = left.shape
    if (
        permutation.shape != (rows,)
        or not np.issubdtype(permutation.dtype, np.integer)
        or not np.array_equal(np.sort(permutation.astype(np.int64)), np.arange(rows))
    ):
        raise ValueError("assignment must be a permutation of all reference rows")
    residual = left - right[permutation.astype(np.int64, copy=False)]
    return float(np.sum(np.square(residual), dtype=np.float64) / dimension)


def uniform_random_bijection_cost_moments(
    target: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Return exact mean and variance over every uniformly random bijection.

    For a square cost matrix ``C`` and uniform random permutation ``pi``,
    ``E sum_i C[i,pi(i)] = sum_ij C[i,j] / n``.  If ``A`` is the doubly
    centered cost matrix, the exact permutation variance is
    ``sum_ij A[i,j]^2 / (n-1)``.  For squared Euclidean cost divided by feature
    dimension, ``A[i,j] = -2 <x_i-xbar, y_j-ybar> / d``.  The implementation
    therefore avoids materializing the full assignment-cost matrix.
    """

    left, right = _endpoint_pair(target, reference)
    rows, dimension = left.shape
    left_mean = np.mean(left, axis=0, dtype=np.float64)
    right_mean = np.mean(right, axis=0, dtype=np.float64)
    left_centered = left - left_mean
    right_centered = right - right_mean

    expected = float(
        (
            np.sum(np.square(left_centered), dtype=np.float64)
            + np.sum(np.square(right_centered), dtype=np.float64)
            + rows * np.sum(np.square(left_mean - right_mean), dtype=np.float64)
        )
        / dimension
    )
    # trace((X'X)(Y'Y)) equals ||X Y'||_F^2.  Forming d-by-d scatters is
    # generally cheaper here than retaining the n-by-n cost matrix, and the
    # formula remains exact up to ordinary floating-point evaluation.
    left_scatter = left_centered.T @ left_centered
    right_scatter = right_centered.T @ right_centered
    centered_cost_energy = float(
        (4.0 / (dimension * dimension))
        * np.einsum("ij,ji->", left_scatter, right_scatter, optimize=True)
    )
    variance = max(centered_cost_energy / (rows - 1), 0.0)
    return {
        "expected_total_cost": expected,
        "variance_total_cost": float(variance),
        "standard_deviation_total_cost": float(math.sqrt(variance)),
        "coefficient_of_variation": (
            float(math.sqrt(variance) / expected) if expected > 0.0 else 0.0
        ),
    }


def summarize_cost_scale(
    *,
    minimum_cost: float,
    random_moments: dict[str, float],
    budgets: Iterable[float],
    identity_cost: float | None = None,
) -> dict[str, Any]:
    """Normalize budgets and an optional true pairing against random cost."""

    minimum = float(minimum_cost)
    random_mean = float(random_moments["expected_total_cost"])
    random_sd = float(random_moments["standard_deviation_total_cost"])
    if (
        not math.isfinite(minimum)
        or minimum < 0.0
        or not math.isfinite(random_mean)
        or random_mean <= minimum
        or not math.isfinite(random_sd)
        or random_sd < 0.0
    ):
        raise ValueError("minimum and random-assignment costs are inconsistent")
    random_excess = random_mean - minimum
    normalized_budgets = []
    for raw_budget in budgets:
        budget = float(raw_budget)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("cost budgets must be finite and nonnegative")
        allowed = minimum * (1.0 + budget)
        normalized_budgets.append(
            {
                "cost_budget_relative": budget,
                "allowed_cost": allowed,
                "allowed_to_random_mean_ratio": allowed / random_mean,
                "allowed_excess_fraction_of_random_minus_minimum": (
                    (allowed - minimum) / random_excess
                ),
                "random_mean_minus_allowed_in_random_sd": (
                    (random_mean - allowed) / random_sd if random_sd > 0.0 else None
                ),
            }
        )

    identity = None
    if identity_cost is not None:
        true_cost = float(identity_cost)
        tolerance = 1024.0 * np.finfo(np.float64).eps * max(minimum, true_cost, 1.0)
        if not math.isfinite(true_cost) or true_cost < minimum - tolerance:
            raise ValueError("identity cost is invalid or below the recorded minimum")
        identity = {
            "cost": true_cost,
            "cost_to_minimum_ratio": true_cost / minimum if minimum > 0.0 else None,
            "excess_fraction_of_random_minus_minimum": (
                (true_cost - minimum) / random_excess
            ),
            "random_mean_minus_identity_in_random_sd": (
                (random_mean - true_cost) / random_sd if random_sd > 0.0 else None
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "minimum_cost": minimum,
        "random_bijection": {
            **{key: float(value) for key, value in random_moments.items()},
            "expected_to_minimum_ratio": (
                random_mean / minimum if minimum > 0.0 else None
            ),
            "expected_excess_relative_to_minimum": (
                random_excess / minimum if minimum > 0.0 else None
            ),
        },
        "identity_pairing": identity,
        "budgets": normalized_budgets,
    }
