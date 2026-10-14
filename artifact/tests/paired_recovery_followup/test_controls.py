from __future__ import annotations

import itertools

import numpy as np

from marginal_alignment.paired_recovery_bounds.geometry import BatchCostProblem
from marginal_alignment.paired_recovery_followup.controls import (
    column_permutation,
    column_permuted_cost_problem,
)


def _problem(cost: np.ndarray) -> BatchCostProblem:
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(cost)
    assert np.array_equal(rows, np.arange(len(cost)))
    minimum = float(cost[rows, columns].sum())
    random_expected = float(cost.sum() / len(cost))
    return BatchCostProblem(
        cost=cost,
        minimum_assignment=np.asarray(columns, dtype=np.int64),
        minimum_cost=minimum,
        random_expected_cost=random_expected,
        random_minus_minimum=random_expected - minimum,
        true_assignment=np.arange(len(cost), dtype=np.int64),
        true_cost=float(np.trace(cost)),
        true_normalized_score=float(
            max((np.trace(cost) - minimum) / (random_expected - minimum), 0.0)
        ),
        diagnostics={"numerical_tolerance": 1e-12},
    )


def test_keyed_column_permutation_is_deterministic_and_nonidentity() -> None:
    first = column_permutation(rows=256, master_seed=3026091401, key="calibration/0")
    second = column_permutation(rows=256, master_seed=3026091401, key="calibration/0")
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, np.arange(256))
    assert np.array_equal(np.sort(first), np.arange(256))


def test_column_control_preserves_all_bijection_costs_as_a_multiset() -> None:
    cost = np.asarray(
        [
            [0.1, 0.7, 1.4, 0.8],
            [0.9, 0.2, 0.6, 1.1],
            [1.2, 0.5, 0.3, 0.9],
            [0.8, 1.0, 0.7, 0.4],
        ],
        dtype=np.float64,
    )
    original = _problem(cost)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    true_assignment = np.asarray([0, 1, 2, 3], dtype=np.int64)
    controlled = column_permuted_cost_problem(
        original, permutation, true_assignment=true_assignment
    )
    original_totals = sorted(
        float(cost[np.arange(4), np.asarray(order)].sum())
        for order in itertools.permutations(range(4))
    )
    controlled_totals = sorted(
        float(controlled.cost[np.arange(4), np.asarray(order)].sum())
        for order in itertools.permutations(range(4))
    )
    np.testing.assert_allclose(original_totals, controlled_totals)
    assert controlled.minimum_cost == original.minimum_cost
    assert controlled.random_expected_cost == original.random_expected_cost
    assert controlled.diagnostics["physical_reference_predictions_permuted"] is False
    assert controlled.diagnostics["true_assignment_permuted"] is False
