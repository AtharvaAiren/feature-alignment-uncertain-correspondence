"""Lower/upper certificates for method-specific recovery error.

For fixed method predictions ``p_i``, clean-reference predictions ``r_j``, a
cost matrix ``C``, and budget ``B``, the target functional is

    sup_pi n^-1 sum_i 1[p_i != r_{pi(i)}]

over bijections satisfying ``sum_i C[i,pi(i)] <= B``.  Integer-feasible
assignments give lower bounds.  Every evaluated nonnegative Lagrange
multiplier gives an upper bound; the exact unconstrained label-count optimum
is an additional upper bound.  These are numerical certificates under the
recorded SciPy/HiGHS floating-point model, not formally verified real-arithmetic
proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .geometry import array_sha256


CERTIFICATE_SCHEMA = "marginal_alignment.paired_recovery_bounds.certificate.v1"
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
    except ImportError as error:  # pragma: no cover - server dependency contract
        raise RuntimeError("recovery certificates require scipy") from error
    row, column = linear_sum_assignment(np.asarray(objective, dtype=np.float64))
    if not np.array_equal(row, np.arange(len(objective))):
        raise RuntimeError("linear assignment omitted a target row")
    return np.asarray(column, dtype=np.int64)


def _validate_predictions(
    method_prediction: np.ndarray, clean_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    method = np.asarray(method_prediction)
    clean = np.asarray(clean_prediction)
    if (
        method.ndim != 1
        or clean.ndim != 1
        or method.shape != clean.shape
        or len(method) < 2
        or method.dtype.hasobject
        or clean.dtype.hasobject
    ):
        raise ValueError("prediction vectors must be equal, non-object rank-one arrays")
    return method, clean


def recovery_reward_matrix(
    method_prediction: np.ndarray, clean_prediction: np.ndarray
) -> np.ndarray:
    method, clean = _validate_predictions(method_prediction, clean_prediction)
    return (method[:, None] != clean[None, :]).astype(np.float64)


def unconstrained_label_count_bound(
    method_prediction: np.ndarray, clean_prediction: np.ndarray
) -> float:
    """Return the exact worst error over every label-count-compatible bijection.

    If ``a_c`` and ``b_c`` are the two prediction counts, at least
    ``max(a_c+b_c-n, 0)`` pairs of class ``c`` are unavoidable.  At most one
    class can have a positive term, and the resulting lower bound on matches
    is attainable by a transportation table, so this is exact.
    """

    method, clean = _validate_predictions(method_prediction, clean_prediction)
    labels = np.union1d(np.unique(method), np.unique(clean))
    rows = len(method)
    unavoidable_matches = 0
    for label in labels:
        left = int(np.sum(method == label))
        right = int(np.sum(clean == label))
        unavoidable_matches += max(left + right - rows, 0)
    return float(1.0 - unavoidable_matches / rows)


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
    """Produce a deterministic integer-feasible improvement over ``baseline``."""

    rows = len(baseline)
    candidates: list[tuple[float, int, float, int, int]] = []
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
            ratio = increment / gain
            candidates.append((ratio, -int(gain), increment, first, second))
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
        if proposed_cost > budget:
            result[first], result[second] = result[second], result[first]
            continue
        used[first] = True
        used[second] = True
        running_cost = proposed_cost
        accepted += 1
    return result, {
        "algorithm": "deterministic_cost_per_reward_disjoint_swaps",
        "candidate_swaps": int(len(candidates)),
        "accepted_swaps": int(accepted),
        "final_cost": float(running_cost),
    }


@dataclass(frozen=True)
class RecoveryCertificate:
    lower_assignment: np.ndarray
    lower_bound: float
    lower_cost: float
    upper_bound: float
    unconstrained_label_count_bound: float
    dual_assignment: np.ndarray | None
    dual_multiplier: float | None
    dual_raw_upper_bound: float | None
    dual_roundoff_cushion: float | None
    width: float
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
            "unconstrained_label_count_bound": float(
                self.unconstrained_label_count_bound
            ),
            "certified_tightening_over_unconstrained": float(
                self.unconstrained_label_count_bound - self.upper_bound
            ),
            "possible_tightening_over_unconstrained": float(
                self.unconstrained_label_count_bound - self.lower_bound
            ),
            "lower_assignment_sha256": array_sha256(self.lower_assignment),
            "dual_multiplier": self.dual_multiplier,
            "dual_raw_upper_bound": self.dual_raw_upper_bound,
            "dual_roundoff_cushion": self.dual_roundoff_cushion,
            "dual_assignment_sha256": (
                None
                if self.dual_assignment is None
                else array_sha256(self.dual_assignment)
            ),
            "diagnostics": dict(self.diagnostics),
        }
        if include_assignments:
            result["lower_assignment"] = [
                int(value) for value in self.lower_assignment
            ]
            result["dual_assignment"] = (
                None
                if self.dual_assignment is None
                else [int(value) for value in self.dual_assignment]
            )
        return result


def certify_recovery_error(
    cost: np.ndarray,
    method_prediction: np.ndarray,
    clean_prediction: np.ndarray,
    *,
    allowed_cost: float,
    minimum_assignment: np.ndarray | None = None,
    dual_bisection_iterations: int = 56,
) -> RecoveryCertificate:
    """Numerically bracket the cost-feasible method-specific error supremum."""

    matrix = np.asarray(cost, dtype=np.float64)
    method, clean = _validate_predictions(method_prediction, clean_prediction)
    rows = len(method)
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
    reward = recovery_reward_matrix(method, clean)
    unconstrained = unconstrained_label_count_bound(method, clean)

    baseline = (
        _linear_assignment(matrix)
        if minimum_assignment is None
        else _assignment(minimum_assignment, rows=rows, name="minimum_assignment")
    )
    baseline_error, baseline_cost = _values(matrix, reward, baseline)
    tolerance = float(
        4096.0
        * rows
        * _EPSILON
        * max(float(np.max(matrix)), abs(baseline_cost), 1.0)
    )
    if baseline_cost > budget:
        raise ValueError("cost-feasible set is empty because budget is below C_min")

    if math.isinf(budget):
        assignment = _linear_assignment(-reward)
        exact_error, exact_cost = _values(matrix, reward, assignment)
        if abs(exact_error - unconstrained) > 64.0 * _EPSILON:
            raise RuntimeError("label-count formula disagrees with unconstrained assignment")
        return RecoveryCertificate(
            lower_assignment=assignment,
            lower_bound=unconstrained,
            lower_cost=exact_cost,
            upper_bound=unconstrained,
            unconstrained_label_count_bound=unconstrained,
            dual_assignment=None,
            dual_multiplier=None,
            dual_raw_upper_bound=None,
            dual_roundoff_cushion=None,
            width=0.0,
            diagnostics={
                "formal_real_arithmetic_certificate_claimed": False,
                "floating_point_solver_model": "scipy_linear_sum_assignment",
                "feasible_set": "all_bijections_infinite_conformal_threshold",
                "upper_bound_source": "exact_unconstrained_label_count_transport_bound",
                "dual_evaluations": 0,
                "numerical_tolerance": tolerance,
            },
        )

    greedy, greedy_diagnostics = _greedy_disjoint_swaps(
        matrix,
        reward,
        baseline,
        budget=budget,
        tolerance=tolerance,
    )
    greedy_error, greedy_cost = _values(matrix, reward, greedy)
    best_assignment = greedy
    best_lower = greedy_error
    best_lower_cost = greedy_cost

    # A scale change leaves every assignment subproblem and Lagrange bound
    # algebraically unchanged while keeping multipliers in a stable range.
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
            multiplier * normalized_cost - reward / float(rows)
        )
        error, assignment_cost = _values(matrix, reward, assignment)
        normalized_assignment_cost = assignment_cost / scale
        raw = float(
            error
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
            )
        )
        upper = min(1.0, raw + cushion)
        if upper < best_dual:
            best_dual = upper
            best_dual_assignment = np.array(assignment, copy=True)
            best_multiplier = float(multiplier)
            best_raw = raw
            best_cushion = cushion
        if assignment_cost <= budget and (
            error > best_lower + 8.0 * _EPSILON
            or (
                abs(error - best_lower) <= 8.0 * _EPSILON
                and assignment_cost < best_lower_cost - tolerance
            )
        ):
            best_assignment = np.array(assignment, copy=True)
            best_lower = error
            best_lower_cost = assignment_cost
        evaluations += 1
        return assignment_cost

    zero_cost = evaluate(0.0)
    if zero_cost > budget:
        lower_multiplier = 0.0
        upper_multiplier = 1.0
        doublings = 0
        while evaluate(upper_multiplier) > budget:
            lower_multiplier = upper_multiplier
            upper_multiplier *= 2.0
            doublings += 1
            if doublings > 64 or not math.isfinite(upper_multiplier):
                raise RuntimeError("could not bracket a cost-feasible Lagrange assignment")
        for _ in range(iterations):
            midpoint = float(
                lower_multiplier + (upper_multiplier - lower_multiplier) / 2.0
            )
            if evaluate(midpoint) > budget:
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
        "exact_unconstrained_label_count_transport_bound"
        if unconstrained <= best_dual
        else "sampled_lagrangian_assignment_upper_bound_with_roundoff_cushion"
    )
    return RecoveryCertificate(
        lower_assignment=best_assignment,
        lower_bound=best_lower,
        lower_cost=best_lower_cost,
        upper_bound=upper,
        unconstrained_label_count_bound=unconstrained,
        dual_assignment=best_dual_assignment,
        dual_multiplier=best_multiplier,
        dual_raw_upper_bound=best_raw,
        dual_roundoff_cushion=best_cushion,
        width=width,
        diagnostics={
            "formal_real_arithmetic_certificate_claimed": False,
            "floating_point_solver_model": "scipy_linear_sum_assignment",
            "integer_feasible_lower_bound": True,
            "lagrangian_upper_bound_valid_for_integer_and_lp_assignment_optima": True,
            "dual_formula": (
                "lambda*B+max_bijection(mean_error_reward-lambda*cost),lambda>=0"
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


def verify_certificate(
    payload: Mapping[str, Any],
    cost: np.ndarray,
    method_prediction: np.ndarray,
    clean_prediction: np.ndarray,
    *,
    allowed_cost: float,
) -> dict[str, Any]:
    """Revalidate serialized primal feasibility and the claimed dual objective."""

    matrix = np.asarray(cost, dtype=np.float64)
    method, clean = _validate_predictions(method_prediction, clean_prediction)
    rows = len(method)
    reward = recovery_reward_matrix(method, clean)
    lower_assignment = _assignment(
        np.asarray(payload["lower_assignment"], dtype=np.int64),
        rows=rows,
        name="lower_assignment",
    )
    lower_error, lower_cost = _values(matrix, reward, lower_assignment)
    tolerance = float(payload["diagnostics"]["numerical_tolerance"])
    if lower_cost > float(allowed_cost) or abs(lower_error - float(payload["lower_bound"])) > 1e-12:
        raise RuntimeError("serialized lower certificate is infeasible or inconsistent")
    if abs(lower_cost - float(payload["lower_assignment_cost"])) > max(tolerance, 1e-12):
        raise RuntimeError("serialized lower assignment cost changed")
    unconstrained = unconstrained_label_count_bound(method, clean)
    if abs(unconstrained - float(payload["unconstrained_label_count_bound"])) > 1e-12:
        raise RuntimeError("unconstrained label-count bound changed")

    source = str(payload["diagnostics"]["upper_bound_source"])
    if source == "exact_unconstrained_label_count_transport_bound":
        recomputed_upper = unconstrained
    else:
        multiplier = float(payload["dual_multiplier"])
        scale = float(payload["diagnostics"]["cost_scale"])
        objective = multiplier * matrix / scale - reward / float(rows)
        optimum_assignment = _linear_assignment(objective)
        serialized_dual = _assignment(
            np.asarray(payload["dual_assignment"], dtype=np.int64),
            rows=rows,
            name="dual_assignment",
        )
        row = np.arange(rows, dtype=np.int64)
        optimum_objective = float(objective[row, optimum_assignment].sum())
        serialized_objective = float(objective[row, serialized_dual].sum())
        if serialized_objective > optimum_objective + max(tolerance, 1e-10):
            raise RuntimeError("serialized dual assignment no longer solves its LAP")
        dual_error, dual_cost = _values(matrix, reward, serialized_dual)
        raw = float(
            dual_error
            + multiplier * (float(allowed_cost) / scale - dual_cost / scale)
        )
        if abs(raw - float(payload["dual_raw_upper_bound"])) > max(tolerance, 1e-10):
            raise RuntimeError("serialized Lagrange upper-bound value changed")
        recomputed_upper = min(
            unconstrained,
            raw + float(payload["dual_roundoff_cushion"]),
        )
    claimed_upper = float(payload["upper_bound"])
    if abs(claimed_upper - max(lower_error, recomputed_upper)) > max(tolerance, 1e-10):
        raise RuntimeError("serialized upper certificate changed")
    return {
        "status": "PASS",
        "lower_assignment_revalidated": True,
        "lower_cost_feasible": True,
        "upper_bound_revalidated": True,
        "label_count_bound_revalidated": True,
    }


def solve_exact_recovery_milp(
    cost: np.ndarray,
    method_prediction: np.ndarray,
    clean_prediction: np.ndarray,
    *,
    allowed_cost: float,
    time_limit_seconds: float = 300.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prove a small diagnostic problem optimal with SciPy/HiGHS MILP."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:  # pragma: no cover - server dependency contract
        raise RuntimeError("exact recovery diagnostics require scipy.optimize.milp") from error
    matrix = np.asarray(cost, dtype=np.float64)
    method, clean = _validate_predictions(method_prediction, clean_prediction)
    rows = len(method)
    if matrix.shape != (rows, rows) or rows > 64:
        raise ValueError("exact diagnostic requires an n-by-n cost with 2<=n<=64")
    reward = recovery_reward_matrix(method, clean)
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
    values = np.concatenate([
        np.ones(rows * rows),
        np.ones(rows * rows),
        matrix.reshape(-1),
    ])
    constraints = coo_matrix(
        (values, (constraint_row, variable_column)),
        shape=(2 * rows + 1, rows * rows),
    ).tocsr()
    lower = np.concatenate([np.ones(2 * rows), np.asarray([-np.inf])])
    upper = np.concatenate([
        np.ones(2 * rows), np.asarray([float(allowed_cost)])
    ])
    result = milp(
        c=-reward.reshape(-1) / float(rows),
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
            "exact recovery MILP did not prove optimality: "
            f"status={result.status}, message={result.message}, "
            f"mip_gap={getattr(result, 'mip_gap', None)}"
        )
    solution = np.asarray(result.x).reshape(rows, rows)
    assignment = np.argmax(solution, axis=1).astype(np.int64)
    assignment = _assignment(assignment, rows=rows, name="exact assignment")
    optimum, assignment_cost = _values(matrix, reward, assignment)
    if assignment_cost > float(allowed_cost) or abs(optimum + float(result.fun)) > 1e-7:
        raise RuntimeError("exact recovery MILP output is inconsistent")
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
        "solver_message": str(result.message),
    }
