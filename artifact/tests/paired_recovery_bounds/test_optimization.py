from __future__ import annotations

import itertools

import numpy as np
import pytest

from marginal_alignment.paired_recovery_bounds.optimization import (
    certify_recovery_error,
    unconstrained_label_count_bound,
    verify_certificate,
)


def _exact(cost, method, clean, budget):
    values = []
    for permutation in itertools.permutations(range(len(method))):
        assignment = np.asarray(permutation, dtype=np.int64)
        row = np.arange(len(method))
        if float(cost[row, assignment].sum()) <= budget + 1e-12:
            values.append(float(np.mean(method != clean[assignment])))
    return max(values)


def test_label_count_bound_equals_exhaustive_unconstrained_optimum() -> None:
    method = np.asarray([0, 0, 0, 1, 2])
    clean = np.asarray([0, 0, 1, 1, 2])
    observed = max(
        float(np.mean(method != clean[np.asarray(permutation)]))
        for permutation in itertools.permutations(range(len(method)))
    )
    assert unconstrained_label_count_bound(method, clean) == observed


def test_certificate_contains_exact_cost_feasible_supremum() -> None:
    cost = np.asarray(
        [
            [0.0, 0.6, 0.7, 1.2, 1.4],
            [0.5, 0.0, 0.9, 0.8, 1.1],
            [0.8, 0.7, 0.0, 0.6, 0.9],
            [1.1, 0.8, 0.5, 0.0, 0.7],
            [1.4, 1.1, 0.8, 0.6, 0.0],
        ]
    )
    method = np.asarray([0, 0, 1, 1, 2])
    clean = np.asarray([0, 1, 1, 2, 2])
    budget = 1.7
    certificate = certify_recovery_error(
        cost, method, clean, allowed_cost=budget, dual_bisection_iterations=40
    )
    exact = _exact(cost, method, clean, budget)
    assert certificate.lower_bound <= exact + 1e-12
    assert exact <= certificate.upper_bound + 1e-10
    assert certificate.lower_cost <= budget
    verify_certificate(
        certificate.summary(), cost, method, clean, allowed_cost=budget
    )


def test_infinite_threshold_reduces_exactly_to_label_count_bound() -> None:
    cost = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    method = np.asarray([0, 0, 1])
    clean = np.asarray([0, 1, 1])
    certificate = certify_recovery_error(
        cost, method, clean, allowed_cost=float("inf")
    )
    assert certificate.width == 0.0
    assert certificate.lower_bound == certificate.upper_bound
    assert certificate.upper_bound == unconstrained_label_count_bound(method, clean)


def test_empty_cost_feasible_set_fails_closed() -> None:
    cost = np.asarray([[1.0, 3.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="empty"):
        certify_recovery_error(
            cost,
            np.asarray([0, 1]),
            np.asarray([0, 1]),
            allowed_cost=1.0,
        )
