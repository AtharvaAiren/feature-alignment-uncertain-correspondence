from __future__ import annotations

import numpy as np

from marginal_alignment.paired_recovery_bounds.geometry import (
    allowed_cost,
    batch_cost_problem,
    fit_recovery_geometry,
    frozen_predictions,
    score_correspondence,
)


def test_one_cost_problem_is_shared_across_methods() -> None:
    rng = np.random.default_rng(711)
    reference_fit = rng.normal(size=(80, 5))
    target_fit = reference_fit @ np.diag([1.2, 0.7, 1.1, 0.9, 1.3]) + 0.2
    weight = rng.normal(size=(4, 5))
    bias = rng.normal(size=4)
    fitted = fit_recovery_geometry(target_fit, reference_fit, weight, bias)
    reference = rng.normal(size=(12, 5))
    target = reference @ np.diag([1.2, 0.7, 1.1, 0.9, 1.3]) + 0.2
    problem = batch_cost_problem(fitted, target, reference)
    assert problem.cost.shape == (12, 12)
    assert problem.true_normalized_score >= 0.0
    assert allowed_cost(problem, 0.2) >= problem.minimum_cost
    predictions = {
        method: frozen_predictions(fitted, method=method, target_features=target)
        for method in ("identity", "diagonal", "bures")
    }
    assert all(value.shape == (12,) for value in predictions.values())
    # Cost construction is method-free: only one matrix exists on the problem.
    assert "method" not in problem.diagnostics


def test_true_correspondence_can_be_scored_after_feasible_set_construction() -> None:
    rng = np.random.default_rng(19)
    reference_fit = rng.normal(size=(50, 4))
    target_fit = reference_fit + 0.1 * rng.normal(size=(50, 4))
    fitted = fit_recovery_geometry(
        target_fit,
        reference_fit,
        rng.normal(size=(3, 4)),
        rng.normal(size=3),
    )
    reference = rng.normal(size=(8, 4))
    target = reference + 0.1 * rng.normal(size=(8, 4))
    assignment = rng.permutation(8)
    unscored = batch_cost_problem(fitted, target, reference)
    scored = batch_cost_problem(
        fitted, target, reference, true_assignment=assignment
    )
    assert score_correspondence(unscored, assignment) == (
        scored.true_cost,
        scored.true_normalized_score,
    )
