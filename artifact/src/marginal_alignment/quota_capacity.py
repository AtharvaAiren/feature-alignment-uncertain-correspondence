"""Finite indexed-atom controls for exact-alignment output-mass capacity.

The population theorem in the paper uses nonatomic standard-Borel laws.  This
module implements its finite, integer-quota analogue: every admissible output
labeling must have the same label counts as the reference readout.  It is a
small diagnostic and matrix-game check, not an implementation of the natural
feature witness optimizer.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Iterator, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog


class QuotaCapacityError(ValueError):
    """Raised when a finite correspondence-world control is malformed."""


def _validated_problem(
    desired_labels: np.ndarray,
    label_counts: Sequence[int],
    prior: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(desired_labels)
    if labels.ndim != 2 or labels.shape[0] < 1 or labels.shape[1] < 1:
        raise QuotaCapacityError("desired_labels must have shape (worlds, atoms)")
    if labels.dtype.kind not in "iu":
        raise QuotaCapacityError("desired_labels must contain integer label indices")
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(label_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size < 2 or np.any(counts <= 0):
        raise QuotaCapacityError("label_counts must contain at least two positive counts")
    if int(counts.sum()) != labels.shape[1]:
        raise QuotaCapacityError("label_counts must sum to the number of atoms")
    if np.any(labels < 0) or np.any(labels >= counts.size):
        raise QuotaCapacityError("desired_labels contains an out-of-range label")
    expected = counts.tolist()
    for world_index, world in enumerate(labels):
        observed = np.bincount(world, minlength=counts.size).tolist()
        if observed != expected:
            raise QuotaCapacityError(
                "every correspondence world must preserve the reference label "
                f"counts; world {world_index} has {observed}, expected {expected}"
            )

    if prior is None:
        weights = np.full(labels.shape[0], 1.0 / labels.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(prior, dtype=np.float64)
        if (
            weights.shape != (labels.shape[0],)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise QuotaCapacityError(
                "prior must be a finite nonnegative world vector summing to one"
            )
    return labels, counts, weights


def posterior_label_probabilities(
    desired_labels: np.ndarray,
    label_counts: Sequence[int],
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``q[x, y]`` after validating exact label-count conservation."""

    labels, counts, weights = _validated_problem(desired_labels, label_counts, prior)
    indicators = labels[:, :, None] == np.arange(counts.size)[None, None, :]
    return np.sum(weights[:, None, None] * indicators, axis=0)


def prior_average_capacities(
    desired_labels: np.ndarray,
    label_counts: Sequence[int],
    prior: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute unrestricted and exact-quota prior-average decision values."""

    labels, counts, weights = _validated_problem(desired_labels, label_counts, prior)
    q = posterior_label_probabilities(labels, counts, weights)
    atom_count = labels.shape[1]
    unrestricted_assignment = np.argmax(q, axis=1).astype(np.int64)
    unrestricted_value = float(np.max(q, axis=1).mean())

    slot_labels = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    rows, columns = linear_sum_assignment(-q[:, slot_labels])
    quota_assignment = np.empty(atom_count, dtype=np.int64)
    quota_assignment[rows] = slot_labels[columns]
    quota_value = float(q[np.arange(atom_count), quota_assignment].mean())

    return {
        "posterior_label_probabilities": q,
        "unrestricted_assignment": unrestricted_assignment,
        "quota_assignment": quota_assignment,
        "unrestricted_prior_average_value": unrestricted_value,
        "quota_prior_average_value": quota_value,
        "quota_penalty": unrestricted_value - quota_value,
    }


def _quota_action_count(counts: np.ndarray) -> int:
    total = int(counts.sum())
    result = math.factorial(total)
    for count in counts:
        result //= math.factorial(int(count))
    return result


def _enumerate_quota_assignments_recursive(
    remaining: tuple[int, ...],
    counts: tuple[int, ...],
    label: int,
    assignment: np.ndarray,
) -> Iterator[np.ndarray]:
    if label == len(counts) - 1:
        completed = assignment.copy()
        completed[list(remaining)] = label
        yield completed
        return
    for selected in combinations(remaining, counts[label]):
        selected_set = set(selected)
        next_assignment = assignment.copy()
        next_assignment[list(selected)] = label
        next_remaining = tuple(index for index in remaining if index not in selected_set)
        yield from _enumerate_quota_assignments_recursive(
            next_remaining,
            counts,
            label + 1,
            next_assignment,
        )


def enumerate_quota_assignments(
    label_counts: Sequence[int], *, max_actions: int = 100_000
) -> np.ndarray:
    """Enumerate every indexed labeling with the requested integer quotas."""

    counts = np.asarray(label_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size < 2 or np.any(counts <= 0):
        raise QuotaCapacityError("label_counts must contain at least two positive counts")
    action_count = _quota_action_count(counts)
    if action_count > max_actions:
        raise QuotaCapacityError(
            f"quota action class has {action_count} members, exceeding {max_actions}"
        )
    atom_count = int(counts.sum())
    assignments = list(
        _enumerate_quota_assignments_recursive(
            tuple(range(atom_count)),
            tuple(int(value) for value in counts),
            0,
            np.full(atom_count, -1, dtype=np.int64),
        )
    )
    return np.stack(assignments, axis=0)


def finite_quota_minimax(
    desired_labels: np.ndarray,
    label_counts: Sequence[int],
    *,
    max_actions: int = 100_000,
) -> dict[str, object]:
    """Solve both LPs for the randomized finite exact-quota matrix game."""

    labels, counts, _ = _validated_problem(desired_labels, label_counts, None)
    actions = enumerate_quota_assignments(counts, max_actions=max_actions)
    payoff = np.mean(actions[:, None, :] == labels[None, :, :], axis=2)
    action_count, world_count = payoff.shape

    primal = linprog(
        np.r_[np.zeros(action_count), -1.0],
        A_ub=np.c_[-payoff.T, np.ones(world_count)],
        b_ub=np.zeros(world_count),
        A_eq=np.r_[np.ones(action_count), 0.0][None, :],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * action_count + [(None, None)],
        method="highs",
    )
    if not primal.success:
        raise RuntimeError(f"quota-game primal failed: {primal.message}")

    dual = linprog(
        np.r_[np.zeros(world_count), 1.0],
        A_ub=np.c_[payoff, -np.ones(action_count)],
        b_ub=np.zeros(action_count),
        A_eq=np.r_[np.ones(world_count), 0.0][None, :],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * world_count + [(None, None)],
        method="highs",
    )
    if not dual.success:
        raise RuntimeError(f"quota-game dual failed: {dual.message}")

    primal_value = float(primal.x[-1])
    dual_value = float(dual.x[-1])
    return {
        "actions": actions,
        "payoff_matrix": payoff,
        "primal_action_weights": np.asarray(primal.x[:-1]),
        "least_favorable_prior": np.asarray(dual.x[:-1]),
        "primal_value": primal_value,
        "dual_value": dual_value,
        "absolute_primal_dual_gap": abs(primal_value - dual_value),
    }


def three_world_quota_control() -> dict[str, object]:
    """Return the symmetric control used for the paper's strict-gap example."""

    desired = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    counts = (2, 2)
    capacities = prior_average_capacities(desired, counts)
    minimax = finite_quota_minimax(desired, counts)
    return {
        "desired_labels": desired,
        "label_counts": np.asarray(counts, dtype=np.int64),
        **capacities,
        **minimax,
    }
