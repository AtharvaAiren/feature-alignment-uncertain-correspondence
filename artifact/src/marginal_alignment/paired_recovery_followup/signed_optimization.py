"""Numerical certificates for signed common-correspondence contrasts.

For an arbitrary finite edge reward matrix ``W``, this module brackets

    max_pi mean_i W[i, pi(i)]

over bijections satisfying a common cost budget.  The signed method contrast
uses ``W = loss_a - loss_c`` with entries in ``{-1,0,1}``.  Rewards are never
clipped: a negative certified upper bound is precisely the useful conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from marginal_alignment.paired_recovery_bounds.geometry import array_sha256


CERTIFICATE_SCHEMA = "marginal_alignment.paired_recovery_followup.linear_reward_certificate.v1"
_EPSILON = np.finfo(np.float64).eps


def _assignment(value: np.ndarray, *, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.shape != (rows,)
        or not np.issubdtype(result.dtype, np.integer)
        or not np.array_equal(
            np.sort(result.astype(np.int64, copy=False)), np.arange(rows)
        )
    ):
        raise ValueError(f"{name} must be a permutation of all reference rows")
    return result.astype(np.int64, copy=False)


def _linear_assignment(objective: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("linear-reward certificates require scipy") from error
    row, column = linear_sum_assignment(np.asarray(objective, dtype=np.float64))
    if not np.array_equal(row, np.arange(len(objective), dtype=np.int64)):
        raise RuntimeError("linear assignment omitted a target row")
    return np.asarray(column, dtype=np.int64)


def _validate_reward(reward: np.ndarray) -> np.ndarray:
    result = np.asarray(reward, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or result.shape[0] < 2
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("reward matrix must be finite, square, and at least 2-by-2")
    return result


def _validate_prediction(value: np.ndarray, *, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (rows,) or result.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object vector with one value per row")
    return result


def signed_contrast_reward_matrix(
    method_a_prediction: np.ndarray,
    method_c_prediction: np.ndarray,
    clean_prediction: np.ndarray,
) -> np.ndarray:
    """Return the unmodified edge loss difference ``loss_a-loss_c``."""

    clean = np.asarray(clean_prediction)
    if clean.ndim != 1 or len(clean) < 2 or clean.dtype.hasobject:
        raise ValueError("clean prediction must be a non-object rank-one vector")
    rows = len(clean)
    first = _validate_prediction(
        method_a_prediction, rows=rows, name="method_a_prediction"
    )
    second = _validate_prediction(
        method_c_prediction, rows=rows, name="method_c_prediction"
    )
    first_loss = (first[:, None] != clean[None, :]).astype(np.float64)
    second_loss = (second[:, None] != clean[None, :]).astype(np.float64)
    reward = first_loss - second_loss
    if not np.all(np.isin(reward, (-1.0, 0.0, 1.0))):
        raise RuntimeError("signed 0-1 loss contrast left {-1,0,1}")
    return reward


def _values(
    cost: np.ndarray, reward: np.ndarray, assignment: np.ndarray
) -> tuple[float, float]:
    row = np.arange(len(assignment), dtype=np.int64)
    return (
        float(reward[row, assignment].mean(dtype=np.float64)),
        float(cost[row, assignment].sum(dtype=np.float64)),
    )


def _greedy_disjoint_swaps(
    cost: np.ndarray,
    reward: np.ndarray,
    baseline: np.ndarray,
    *,
    budget: float,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = len(baseline)
    candidates: list[tuple[float, float, float, int, int]] = []
    for first in range(rows):
        first_column = int(baseline[first])
        for second in range(first + 1, rows):
            second_column = int(baseline[second])
            gain = float(
                reward[first, second_column]
                + reward[second, first_column]
                - reward[first, first_column]
                - reward[second, second_column]
            )
            if gain <= 0.0:
                continue
            increment = float(
                cost[first, second_column]
                + cost[second, first_column]
                - cost[first, first_column]
                - cost[second, second_column]
            )
            if increment < -tolerance:
                raise RuntimeError("a swap improves the asserted minimum-cost assignment")
            increment = max(increment, 0.0)
            candidates.append((increment / gain, -gain, increment, first, second))
    candidates.sort()
    result = np.array(baseline, copy=True)
    used = np.zeros(rows, dtype=bool)
    _, running_cost = _values(cost, reward, result)
    accepted = 0
    for _, _, _, first, second in candidates:
        if used[first] or used[second]:
            continue
        result[first], result[second] = result[second], result[first]
        _, proposed_cost = _values(cost, reward, result)
        if proposed_cost > budget + tolerance:
            result[first], result[second] = result[second], result[first]
            continue
        used[first] = True
        used[second] = True
        running_cost = proposed_cost
        accepted += 1
    return result, {
        "algorithm": "deterministic_cost_per_signed_reward_disjoint_swaps",
        "candidate_swaps": int(len(candidates)),
        "accepted_swaps": int(accepted),
        "final_cost": float(running_cost),
    }


@dataclass(frozen=True)
class LinearRewardCertificate:
    lower_assignment: np.ndarray
    lower_bound: float
    lower_cost: float
    upper_bound: float
    unconstrained_upper_bound: float
    unconstrained_assignment: np.ndarray
    dual_assignment: np.ndarray | None
    dual_multiplier: float | None
    dual_raw_upper_bound: float | None
    dual_roundoff_cushion: float | None
    width: float
    reward_sha256: str
    diagnostics: Mapping[str, Any]

    def summary(self, *, include_assignments: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": CERTIFICATE_SCHEMA,
            "status": "PASS",
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "optimization_interval": [float(self.lower_bound), float(self.upper_bound)],
            "optimization_width": float(self.width),
            "lower_assignment_cost": float(self.lower_cost),
            "unconstrained_upper_bound": float(self.unconstrained_upper_bound),
            "certified_tightening_from_unconstrained": float(
                self.unconstrained_upper_bound - self.upper_bound
            ),
            "lower_assignment_sha256": array_sha256(self.lower_assignment),
            "unconstrained_assignment_sha256": array_sha256(
                self.unconstrained_assignment
            ),
            "dual_multiplier": self.dual_multiplier,
            "dual_raw_upper_bound": self.dual_raw_upper_bound,
            "dual_roundoff_cushion": self.dual_roundoff_cushion,
            "dual_assignment_sha256": (
                None
                if self.dual_assignment is None
                else array_sha256(self.dual_assignment)
            ),
            "reward_matrix_sha256": self.reward_sha256,
            "diagnostics": dict(self.diagnostics),
        }
        if include_assignments:
            result["lower_assignment"] = [int(value) for value in self.lower_assignment]
            result["unconstrained_assignment"] = [
                int(value) for value in self.unconstrained_assignment
            ]
            result["dual_assignment"] = (
                None
                if self.dual_assignment is None
                else [int(value) for value in self.dual_assignment]
            )
        return result


def certify_linear_reward(
    cost: np.ndarray,
    reward: np.ndarray,
    *,
    allowed_cost: float,
    minimum_assignment: np.ndarray | None = None,
    dual_bisection_iterations: int = 56,
) -> LinearRewardCertificate:
    """Bracket a cost-feasible mean edge reward, including signed rewards."""

    matrix = np.asarray(cost, dtype=np.float64)
    values = _validate_reward(reward)
    rows = len(values)
    if (
        matrix.shape != (rows, rows)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
    ):
        raise ValueError("cost matrix must be finite, nonnegative, and square")
    budget = float(allowed_cost)
    if math.isnan(budget) or budget < 0.0:
        raise ValueError("allowed cost must be nonnegative or positive infinity")
    iterations = int(dual_bisection_iterations)
    if iterations < 1:
        raise ValueError("dual_bisection_iterations must be positive")
    baseline = (
        _linear_assignment(matrix)
        if minimum_assignment is None
        else _assignment(minimum_assignment, rows=rows, name="minimum_assignment")
    )
    baseline_reward, baseline_cost = _values(matrix, values, baseline)
    tolerance = float(
        4096.0
        * rows
        * _EPSILON
        * max(float(np.max(matrix)), abs(baseline_cost), 1.0)
    )
    if baseline_cost > budget + tolerance:
        raise ValueError("cost-feasible set is empty because budget is below C_min")
    unconstrained_assignment = _linear_assignment(-values)
    unconstrained, unconstrained_cost = _values(
        matrix, values, unconstrained_assignment
    )
    reward_digest = array_sha256(values)
    if math.isinf(budget):
        return LinearRewardCertificate(
            lower_assignment=unconstrained_assignment,
            lower_bound=unconstrained,
            lower_cost=unconstrained_cost,
            upper_bound=unconstrained,
            unconstrained_upper_bound=unconstrained,
            unconstrained_assignment=unconstrained_assignment,
            dual_assignment=None,
            dual_multiplier=None,
            dual_raw_upper_bound=None,
            dual_roundoff_cushion=None,
            width=0.0,
            reward_sha256=reward_digest,
            diagnostics={
                "formal_real_arithmetic_certificate_claimed": False,
                "floating_point_solver_model": "scipy_linear_sum_assignment",
                "feasible_set": "all_bijections_infinite_conformal_threshold",
                "upper_bound_source": "exact_unconstrained_linear_assignment",
                "signed_rewards_retained_without_clipping": True,
                "dual_evaluations": 0,
                "numerical_tolerance": tolerance,
            },
        )

    greedy, greedy_diagnostics = _greedy_disjoint_swaps(
        matrix,
        values,
        baseline,
        budget=budget,
        tolerance=tolerance,
    )
    greedy_reward, greedy_cost = _values(matrix, values, greedy)
    best_assignment = greedy
    best_lower = greedy_reward
    best_lower_cost = greedy_cost
    scale = max(baseline_cost, float(matrix.sum(dtype=np.float64) / rows), 1e-15)
    normalized_cost = matrix / scale
    normalized_budget = budget / scale
    best_dual = math.inf
    best_dual_assignment: np.ndarray | None = None
    best_multiplier: float | None = None
    best_raw: float | None = None
    best_cushion: float | None = None
    evaluations = 0

    def evaluate(multiplier: float) -> float:
        nonlocal best_assignment, best_lower, best_lower_cost
        nonlocal best_dual, best_dual_assignment, best_multiplier, best_raw
        nonlocal best_cushion, evaluations
        assignment = _linear_assignment(
            multiplier * normalized_cost - values / float(rows)
        )
        reward_value, assignment_cost = _values(matrix, values, assignment)
        normalized_assignment_cost = assignment_cost / scale
        raw = float(
            reward_value
            + multiplier * (normalized_budget - normalized_assignment_cost)
        )
        cushion = float(
            8192.0
            * rows
            * _EPSILON
            * max(
                1.0,
                abs(raw),
                abs(multiplier)
                * (abs(normalized_budget) + abs(normalized_assignment_cost)),
                float(np.max(np.abs(values))),
            )
        )
        upper = raw + cushion
        if upper < best_dual:
            best_dual = upper
            best_dual_assignment = np.array(assignment, copy=True)
            best_multiplier = float(multiplier)
            best_raw = raw
            best_cushion = cushion
        if assignment_cost <= budget + tolerance and (
            reward_value > best_lower + 8.0 * _EPSILON
            or (
                abs(reward_value - best_lower) <= 8.0 * _EPSILON
                and assignment_cost < best_lower_cost - tolerance
            )
        ):
            best_assignment = np.array(assignment, copy=True)
            best_lower = reward_value
            best_lower_cost = assignment_cost
        evaluations += 1
        return assignment_cost

    zero_cost = evaluate(0.0)
    if zero_cost > budget + tolerance:
        lower_multiplier = 0.0
        upper_multiplier = 1.0
        doublings = 0
        while evaluate(upper_multiplier) > budget + tolerance:
            lower_multiplier = upper_multiplier
            upper_multiplier *= 2.0
            doublings += 1
            if doublings > 64 or not math.isfinite(upper_multiplier):
                raise RuntimeError("could not bracket a cost-feasible Lagrange assignment")
        for _ in range(iterations):
            midpoint = float(
                lower_multiplier + (upper_multiplier - lower_multiplier) / 2.0
            )
            if evaluate(midpoint) > budget + tolerance:
                lower_multiplier = midpoint
            else:
                upper_multiplier = midpoint
        bracket: dict[str, float | int] = {
            "lower": float(lower_multiplier),
            "upper": float(upper_multiplier),
            "doublings": int(doublings),
        }
    else:
        bracket = {"lower": 0.0, "upper": 0.0, "doublings": 0}

    if best_dual_assignment is None or best_multiplier is None:
        raise RuntimeError("dual search produced no valid upper-bound witness")
    if best_dual + tolerance < best_lower:
        raise RuntimeError("numerical dual bound fell below an integer-feasible lower bound")
    upper = float(min(unconstrained, max(best_dual, best_lower)))
    width = float(max(upper - best_lower, 0.0))
    source = (
        "exact_unconstrained_linear_assignment"
        if unconstrained <= best_dual
        else "sampled_lagrangian_assignment_upper_bound_with_roundoff_cushion"
    )
    return LinearRewardCertificate(
        lower_assignment=best_assignment,
        lower_bound=best_lower,
        lower_cost=best_lower_cost,
        upper_bound=upper,
        unconstrained_upper_bound=unconstrained,
        unconstrained_assignment=unconstrained_assignment,
        dual_assignment=best_dual_assignment,
        dual_multiplier=best_multiplier,
        dual_raw_upper_bound=best_raw,
        dual_roundoff_cushion=best_cushion,
        width=width,
        reward_sha256=reward_digest,
        diagnostics={
            "formal_real_arithmetic_certificate_claimed": False,
            "floating_point_solver_model": "scipy_linear_sum_assignment",
            "integer_feasible_lower_bound": True,
            "lagrangian_upper_bound_valid_for_integer_and_lp_assignment_optima": True,
            "signed_rewards_retained_without_clipping": True,
            "dual_formula": (
                "lambda*B+max_bijection(mean_edge_reward-lambda*cost),lambda>=0"
            ),
            "upper_bound_source": source,
            "dual_evaluations": int(evaluations),
            "dual_bisection_iterations": iterations,
            "cost_scale": float(scale),
            "bracket": bracket,
            "numerical_tolerance": tolerance,
            "greedy": greedy_diagnostics,
        },
    )


def verify_linear_reward_certificate(
    payload: Mapping[str, Any],
    cost: np.ndarray,
    reward: np.ndarray,
    *,
    allowed_cost: float,
) -> dict[str, Any]:
    """Revalidate serialized primal and Lagrangian witnesses."""

    matrix = np.asarray(cost, dtype=np.float64)
    values = _validate_reward(reward)
    rows = len(values)
    if matrix.shape != (rows, rows):
        raise ValueError("cost and reward matrices must have equal square shape")
    if payload.get("schema_version") != CERTIFICATE_SCHEMA:
        raise RuntimeError("linear-reward certificate schema changed")
    if payload.get("reward_matrix_sha256") != array_sha256(values):
        raise RuntimeError("serialized reward matrix identity changed")
    tolerance = float(payload["diagnostics"]["numerical_tolerance"])
    lower_assignment = _assignment(
        np.asarray(payload["lower_assignment"], dtype=np.int64),
        rows=rows,
        name="lower_assignment",
    )
    lower_reward, lower_cost = _values(matrix, values, lower_assignment)
    if lower_cost > float(allowed_cost) + tolerance:
        raise RuntimeError("serialized lower certificate is cost-infeasible")
    if abs(lower_reward - float(payload["lower_bound"])) > 1e-12:
        raise RuntimeError("serialized lower reward changed")
    if abs(lower_cost - float(payload["lower_assignment_cost"])) > max(tolerance, 1e-12):
        raise RuntimeError("serialized lower cost changed")
    unconstrained_assignment = _assignment(
        np.asarray(payload["unconstrained_assignment"], dtype=np.int64),
        rows=rows,
        name="unconstrained_assignment",
    )
    optimum_assignment = _linear_assignment(-values)
    row = np.arange(rows, dtype=np.int64)
    serialized_unconstrained = float(
        values[row, unconstrained_assignment].mean(dtype=np.float64)
    )
    optimum_unconstrained = float(
        values[row, optimum_assignment].mean(dtype=np.float64)
    )
    if abs(serialized_unconstrained - optimum_unconstrained) > 1e-12:
        raise RuntimeError("serialized unconstrained assignment is not optimal")
    if abs(optimum_unconstrained - float(payload["unconstrained_upper_bound"])) > 1e-12:
        raise RuntimeError("serialized unconstrained upper bound changed")

    source = str(payload["diagnostics"]["upper_bound_source"])
    if source == "exact_unconstrained_linear_assignment":
        recomputed_upper = optimum_unconstrained
    else:
        multiplier = float(payload["dual_multiplier"])
        scale = float(payload["diagnostics"]["cost_scale"])
        objective = multiplier * matrix / scale - values / float(rows)
        optimum_dual = _linear_assignment(objective)
        serialized_dual = _assignment(
            np.asarray(payload["dual_assignment"], dtype=np.int64),
            rows=rows,
            name="dual_assignment",
        )
        optimum_objective = float(objective[row, optimum_dual].sum(dtype=np.float64))
        serialized_objective = float(
            objective[row, serialized_dual].sum(dtype=np.float64)
        )
        if serialized_objective > optimum_objective + max(tolerance, 1e-10):
            raise RuntimeError("serialized dual assignment no longer solves its LAP")
        dual_reward, dual_cost = _values(matrix, values, serialized_dual)
        raw = float(
            dual_reward
            + multiplier * (float(allowed_cost) / scale - dual_cost / scale)
        )
        if abs(raw - float(payload["dual_raw_upper_bound"])) > max(tolerance, 1e-10):
            raise RuntimeError("serialized Lagrange upper-bound value changed")
        recomputed_upper = min(
            optimum_unconstrained,
            raw + float(payload["dual_roundoff_cushion"]),
        )
    claimed = float(payload["upper_bound"])
    if abs(claimed - max(lower_reward, recomputed_upper)) > max(tolerance, 1e-10):
        raise RuntimeError("serialized linear-reward upper bound changed")
    return {
        "status": "PASS",
        "lower_assignment_revalidated": True,
        "lower_cost_feasible": True,
        "signed_upper_bound_revalidated": True,
        "unconstrained_assignment_revalidated": True,
    }


def solve_exact_linear_reward_milp(
    cost: np.ndarray,
    reward: np.ndarray,
    *,
    allowed_cost: float,
    time_limit_seconds: float = 300.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prove a small signed-reward assignment diagnostic optimal with HiGHS."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("exact linear-reward diagnostics require scipy") from error
    matrix = np.asarray(cost, dtype=np.float64)
    values = _validate_reward(reward)
    rows = len(values)
    if matrix.shape != (rows, rows) or rows > 64:
        raise ValueError("exact diagnostic requires equal n-by-n matrices with n<=64")
    variable = np.arange(rows * rows, dtype=np.int64).reshape(rows, rows)
    constraint_row = np.concatenate([
        np.repeat(np.arange(rows, dtype=np.int64), rows),
        np.repeat(np.arange(rows, 2 * rows, dtype=np.int64), rows),
        np.full(rows * rows, 2 * rows, dtype=np.int64),
    ])
    variable_column = np.concatenate([
        variable.reshape(-1),
        variable.T.reshape(-1),
        variable.reshape(-1),
    ])
    entries = np.concatenate([
        np.ones(rows * rows),
        np.ones(rows * rows),
        matrix.reshape(-1),
    ])
    constraints = coo_matrix(
        (entries, (constraint_row, variable_column)),
        shape=(2 * rows + 1, rows * rows),
    ).tocsr()
    lower = np.concatenate([np.ones(2 * rows), np.asarray([-np.inf])])
    upper = np.concatenate([
        np.ones(2 * rows), np.asarray([float(allowed_cost)])
    ])
    result = milp(
        c=-values.reshape(-1) / float(rows),
        integrality=np.ones(rows * rows, dtype=np.uint8),
        bounds=Bounds(np.zeros(rows * rows), np.ones(rows * rows)),
        constraints=LinearConstraint(constraints, lower, upper),
        options={
            "presolve": True,
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": 0.0,
        },
    )
    if not result.success or result.x is None or int(result.status) != 0:
        raise RuntimeError(
            "exact linear-reward MILP did not prove optimality: "
            f"status={result.status}, message={result.message}, "
            f"mip_gap={getattr(result, 'mip_gap', None)}"
        )
    solution = np.asarray(result.x).reshape(rows, rows)
    assignment = _assignment(
        np.argmax(solution, axis=1).astype(np.int64),
        rows=rows,
        name="exact assignment",
    )
    optimum, assignment_cost = _values(matrix, values, assignment)
    if assignment_cost > float(allowed_cost) + 1e-9:
        raise RuntimeError("exact linear-reward assignment is cost-infeasible")
    if abs(optimum + float(result.fun)) > 1e-7:
        raise RuntimeError("exact linear-reward objective is inconsistent")
    return assignment, {
        "status": "PROVEN_OPTIMAL",
        "solver": "scipy_optimize_milp_highs",
        "rows": rows,
        "binary_variables": rows * rows,
        "relative_mip_gap_requested": 0.0,
        "reported_mip_gap": (
            0.0
            if getattr(result, "mip_gap", None) is None
            else float(result.mip_gap)
        ),
        "exact_supremum": optimum,
        "assignment_cost": assignment_cost,
        "allowed_cost": float(allowed_cost),
        "assignment_sha256": array_sha256(assignment),
        "reward_matrix_sha256": array_sha256(values),
        "solver_message": str(result.message),
    }
