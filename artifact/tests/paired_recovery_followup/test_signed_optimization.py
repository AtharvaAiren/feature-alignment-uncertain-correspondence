from __future__ import annotations

import itertools

import numpy as np

from marginal_alignment.paired_recovery_followup.signed_optimization import (
    certify_linear_reward,
    signed_contrast_reward_matrix,
    solve_exact_linear_reward_milp,
    verify_linear_reward_certificate,
)


def _exact(cost: np.ndarray, reward: np.ndarray, budget: float) -> float:
    rows = len(cost)
    values = []
    for order in itertools.permutations(range(rows)):
        assignment = np.asarray(order, dtype=np.int64)
        if float(cost[np.arange(rows), assignment].sum()) <= budget + 1e-12:
            values.append(float(reward[np.arange(rows), assignment].mean()))
    return max(values)


def test_signed_reward_is_not_clipped_and_certificate_contains_optimum() -> None:
    cost = np.asarray(
        [
            [0.0, 0.7, 1.0, 1.4],
            [0.6, 0.0, 0.8, 1.1],
            [1.0, 0.7, 0.0, 0.6],
            [1.3, 1.1, 0.5, 0.0],
        ]
    )
    reward = np.asarray(
        [
            [-1.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],
        ]
    )
    budget = 0.8
    certificate = certify_linear_reward(
        cost, reward, allowed_cost=budget, dual_bisection_iterations=48
    )
    exact = _exact(cost, reward, budget)
    assert certificate.lower_bound <= exact + 1e-12
    assert exact <= certificate.upper_bound + 1e-10
    assert certificate.upper_bound < 0.0
    verify_linear_reward_certificate(
        certificate.summary(), cost, reward, allowed_cost=budget
    )


def test_contrast_reward_matches_common_correspondence_error_difference() -> None:
    first = np.asarray([0, 1, 2, 0, 1])
    second = np.asarray([0, 0, 2, 1, 1])
    clean = np.asarray([0, 1, 1, 2, 0])
    assignment = np.asarray([3, 0, 4, 1, 2])
    reward = signed_contrast_reward_matrix(first, second, clean)
    observed = float(reward[np.arange(5), assignment].mean())
    direct = float(
        np.mean(first != clean[assignment]) - np.mean(second != clean[assignment])
    )
    assert observed == direct
    assert set(np.unique(reward)).issubset({-1.0, 0.0, 1.0})


def test_infinite_budget_is_exact_even_when_optimum_is_negative() -> None:
    cost = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    reward = -np.ones((2, 2), dtype=np.float64)
    certificate = certify_linear_reward(
        cost, reward, allowed_cost=float("inf")
    )
    assert certificate.lower_bound == -1.0
    assert certificate.upper_bound == -1.0
    assert certificate.width == 0.0


def test_small_highs_diagnostic_matches_exhaustive_optimum() -> None:
    cost = np.asarray(
        [[0.0, 0.4, 1.0], [0.5, 0.0, 0.6], [1.0, 0.5, 0.0]]
    )
    reward = np.asarray(
        [[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]]
    )
    budget = 1.1
    _, diagnostic = solve_exact_linear_reward_milp(
        cost, reward, allowed_cost=budget, time_limit_seconds=30.0
    )
    assert diagnostic["status"] == "PROVEN_OPTIMAL"
    assert diagnostic["exact_supremum"] == _exact(cost, reward, budget)
