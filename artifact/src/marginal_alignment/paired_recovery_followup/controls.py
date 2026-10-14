"""Outcome-blind column-permuted correspondence-geometry negative control."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from marginal_alignment.paired_recovery_bounds.geometry import (
    BatchCostProblem,
    array_sha256,
)


_EPSILON = np.finfo(np.float64).eps


def column_permutation(*, rows: int, master_seed: int, key: str) -> np.ndarray:
    """Create a deterministic keyed PCG64 permutation independent of outcomes."""

    size = int(rows)
    if size < 2:
        raise ValueError("column control needs at least two rows")
    if not isinstance(key, str) or not key:
        raise ValueError("permutation key must be nonempty text")
    payload = f"{int(master_seed)}\0{key}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(size)
    permutation = np.asarray(permutation, dtype=np.int64)
    if np.array_equal(permutation, np.arange(size, dtype=np.int64)):
        raise RuntimeError("locked geometry-control draw was the identity permutation")
    return permutation


def _linear_assignment(cost: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("column control requires scipy") from error
    rows, columns = linear_sum_assignment(np.asarray(cost, dtype=np.float64))
    if not np.array_equal(rows, np.arange(len(cost), dtype=np.int64)):
        raise RuntimeError("linear assignment omitted a target row")
    return np.asarray(columns, dtype=np.int64)


def column_permuted_cost_problem(
    original: BatchCostProblem,
    permutation: np.ndarray,
    *,
    true_assignment: np.ndarray | None = None,
) -> BatchCostProblem:
    """Permute only cost columns while retaining physical rewards/pairing.

    ``permutation[j]`` selects the original cost column installed at physical
    reference column ``j``.  Clean predictions and the true assignment are not
    transformed; doing so would merely relabel the original problem.
    """

    cost = np.asarray(original.cost, dtype=np.float64)
    rows = len(cost)
    columns = np.asarray(permutation)
    if (
        cost.shape != (rows, rows)
        or columns.shape != (rows,)
        or not np.issubdtype(columns.dtype, np.integer)
        or not np.array_equal(np.sort(columns.astype(np.int64)), np.arange(rows))
    ):
        raise ValueError("permutation must contain every cost column exactly once")
    columns = columns.astype(np.int64, copy=False)
    permuted = np.ascontiguousarray(cost[:, columns], dtype=np.float64)
    minimum_assignment = _linear_assignment(permuted)
    row = np.arange(rows, dtype=np.int64)
    minimum = float(permuted[row, minimum_assignment].sum(dtype=np.float64))
    random_expected = float(permuted.sum(dtype=np.float64) / rows)
    gap = float(random_expected - minimum)
    tolerance = float(
        4096.0
        * rows
        * _EPSILON
        * max(abs(random_expected), abs(minimum), float(np.max(permuted)), 1.0)
    )
    invariant_tolerance = max(
        tolerance,
        float(original.diagnostics["numerical_tolerance"]),
        1e-12,
    )
    if abs(minimum - original.minimum_cost) > invariant_tolerance:
        raise RuntimeError("column permutation changed the minimum bijection cost")
    if abs(random_expected - original.random_expected_cost) > invariant_tolerance:
        raise RuntimeError("column permutation changed expected random-bijection cost")
    if not math.isfinite(gap) or gap <= tolerance:
        raise np.linalg.LinAlgError("column-control random-minus-minimum gap is not positive")

    assignment = (
        np.arange(rows, dtype=np.int64)
        if true_assignment is None
        else np.asarray(true_assignment, dtype=np.int64)
    )
    if (
        assignment.shape != (rows,)
        or not np.issubdtype(assignment.dtype, np.integer)
        or not np.array_equal(np.sort(assignment.astype(np.int64)), np.arange(rows))
    ):
        raise ValueError("true assignment must be a permutation")
    assignment = assignment.astype(np.int64, copy=False)
    true_cost = float(permuted[row, assignment].sum(dtype=np.float64))
    if true_cost < minimum - tolerance:
        raise RuntimeError("true column-control cost fell below its LAP minimum")
    score = float(max((true_cost - minimum) / gap, 0.0))
    centered = (
        permuted
        - permuted.mean(axis=1, keepdims=True)
        - permuted.mean(axis=0, keepdims=True)
        + permuted.mean()
    )
    variance = float(np.sum(np.square(centered), dtype=np.float64) / (rows - 1))
    diagnostics: dict[str, Any] = {
        "rows": rows,
        "cost_matrix_sha256": array_sha256(permuted),
        "minimum_assignment_sha256": array_sha256(minimum_assignment),
        "normalization": "(C_true-C_min)/(E_uniform_bijection_C-C_min)",
        "uniform_random_cost_variance": max(variance, 0.0),
        "numerical_tolerance": tolerance,
        "negative_control": "permute_only_cost_columns",
        "column_permutation_sha256": array_sha256(columns),
        "physical_reference_predictions_permuted": False,
        "true_assignment_permuted": False,
        "minimum_cost_invariant_checked": True,
        "random_expected_cost_invariant_checked": True,
    }
    return BatchCostProblem(
        cost=permuted,
        minimum_assignment=minimum_assignment,
        minimum_cost=minimum,
        random_expected_cost=random_expected,
        random_minus_minimum=gap,
        true_assignment=assignment,
        true_cost=true_cost,
        true_normalized_score=score,
        diagnostics=diagnostics,
    )
