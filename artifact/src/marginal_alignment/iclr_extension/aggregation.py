"""Fail-closed complete-panel aggregation for the Play-3 extension."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    describe_file,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.iclr_extension.frontier import MARGINAL_FRONTIER_METHODS
from marginal_alignment.iclr_extension.natural_shift import (
    NATURAL_SHIFT_RESULT_SCHEMA,
    load_locked_play3_config,
)


AGGREGATE_SCHEMA = "marginal_alignment.iclr_extension.play3_aggregate.v2"
POSITIVE_SCHEMA = "marginal_alignment.iclr_extension.positive_control.v2"
WITNESS_INTERPRETATION = (
    "equal-prior randomized-minimax frozen-reference-prediction recovery ceiling "
    "across the two constructed cost-constrained finite-batch correspondence "
    "worlds; conservative witness derived because the alternative-world search "
    "is greedy"
)


def _probability(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise IntegrityError(f"{name} must be a finite probability")
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise IntegrityError(f"{name} must be finite and nonnegative")
    return result


def _budget_token(value: Any) -> int:
    budget = float(value)
    if not math.isfinite(budget) or budget < 0.0:
        raise IntegrityError("unit cost budget is invalid")
    token = int(round(10_000 * budget))
    if abs(token / 10_000.0 - budget) > 1e-15:
        raise IntegrityError("unit cost budget is not exactly representable in basis points")
    return token


def _load_unit_directory(root: str | Path, *, schema: str) -> list[tuple[Path, dict[str, Any]]]:
    supplied = Path(root).expanduser()
    if supplied.is_symlink():
        raise IntegrityError(f"unit root cannot be a symlink: {supplied}")
    directory = supplied.resolve()
    if not directory.is_dir():
        raise IntegrityError(f"unit root is absent: {directory}")
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise IntegrityError(f"unit root contains no JSON results: {directory}")
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"unit result cannot be a symlink: {path}")
        value = read_json(path)
        if not isinstance(value, Mapping):
            raise IntegrityError(f"unit result is not a JSON object: {path}")
        verify_manifest_checksum(value)
        if value.get("schema_version") != schema:
            raise IntegrityError(f"unexpected JSON artifact in unit root: {path}")
        loaded.append((path.resolve(), dict(value)))
    return loaded


def _verify_external_file(path_value: Any, expected_sha: Any, *, name: str) -> Path:
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise IntegrityError(f"{name} is absent: {path}")
    expected = str(expected_sha)
    if len(expected) != 64 or sha256_file(path) != expected:
        raise IntegrityError(f"{name} hash drifted: {path}")
    return path


def _expected_protocol(config: Mapping[str, Any], budget: float) -> dict[str, Any]:
    declared = config["ceiling_protocol"]
    return {
        "cost_geometry": declared["cost_geometry"],
        "cost_normalization": declared["cost_normalization"],
        "discarded_feature_dimensions": int(
            declared["discarded_feature_dimensions"]
        ),
        "cost_budget_relative": float(budget),
        "covariance_estimator": declared["covariance_estimator"],
        "covariance_relative_floor": float(declared["covariance_relative_floor"]),
        "maximum_witness_rows": int(declared["maximum_witness_rows"]),
        "swap_neighbors": int(declared["swap_neighbors"]),
        "swap_neighbor_sensitivity": [
            int(value) for value in declared["swap_neighbor_sensitivity"]
        ],
        "assignment_tie_relative_tolerance": float(
            declared["assignment_tie_relative_tolerance"]
        ),
    }


def _validate_swap_neighborhood_sensitivity(
    value: Any,
    *,
    config: Mapping[str, Any],
    budget_token: int,
    primary_witness: Mapping[str, Any],
) -> dict[int, dict[str, float]]:
    """Validate locked search sensitivity on the exact frozen primary geometry."""

    expected = [
        int(item)
        for item in config["ceiling_protocol"]["swap_neighbor_sensitivity"]
    ]
    if not isinstance(value, list) or len(value) != len(expected):
        raise IntegrityError("unit swap-neighborhood sensitivity panel is incomplete")
    primary_diagnostics = primary_witness["diagnostics"]
    primary_calibration = primary_diagnostics["calibration"]
    primary_baseline = _finite_nonnegative(
        primary_witness.get("baseline_cost"), name="primary baseline cost"
    )
    invariant_diagnostic_keys = (
        "target_content_sha256",
        "reference_content_sha256",
        "assignment_world_zero_sha256",
        "prediction_world_zero_sha256",
    )
    invariant_geometry_keys = (
        "target_calibration_content_sha256",
        "reference_calibration_content_sha256",
        "cost_geometry",
        "cost_dimension",
        "discarded_feature_dimensions",
        "distance_normalization",
        "reference_centering",
        "bures_matrix_sha256",
        "bures_offset_sha256",
        "reference_mean_sha256",
        "reference_inverse_sqrt_sha256",
        "full_geometry_sha256",
    )
    results: dict[int, dict[str, float]] = {}
    for position, expected_neighbors in enumerate(expected):
        record = value[position]
        if (
            not isinstance(record, Mapping)
            or int(record.get("swap_neighbors", -1)) != expected_neighbors
            or expected_neighbors in results
        ):
            raise IntegrityError(
                "unit swap-neighborhood sensitivity order or identity drifted"
            )
        witness = record.get("ceiling_witness")
        if not isinstance(witness, Mapping):
            raise IntegrityError("swap-neighborhood sensitivity lacks a witness")
        disagreement = _probability(
            witness.get("disagreement"),
            name=f"swap-neighborhood-{expected_neighbors} disagreement",
        )
        ceiling = _probability(
            witness.get("prediction_recovery_ceiling"),
            name=f"swap-neighborhood-{expected_neighbors} ceiling",
        )
        if abs(ceiling - (1.0 - disagreement / 2.0)) > 1e-12:
            raise IntegrityError(
                "swap-neighborhood ceiling does not equal one minus disagreement over two"
            )
        baseline = _finite_nonnegative(
            witness.get("baseline_cost"), name="sensitivity baseline cost"
        )
        alternative = _finite_nonnegative(
            witness.get("alternative_cost"), name="sensitivity alternative cost"
        )
        allowed = _finite_nonnegative(
            witness.get("allowed_cost"), name="sensitivity allowed cost"
        )
        tolerance = 1e-10 * max(1.0, allowed, primary_baseline)
        if abs(baseline - primary_baseline) > tolerance:
            raise IntegrityError(
                "swap-neighborhood sensitivity changed the frozen baseline assignment"
            )
        if alternative > allowed + tolerance:
            raise IntegrityError(
                "swap-neighborhood sensitivity violates its locked cost budget"
            )
        diagnostics = witness.get("diagnostics")
        calibration = diagnostics.get("calibration") if isinstance(
            diagnostics, Mapping
        ) else None
        if (
            witness.get("interpretation") != WITNESS_INTERPRETATION
            or witness.get("not_a_realized_world_performance_bound") is not True
            or witness.get("not_an_accuracy_ceiling") is not True
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("quantity_interpretation")
            != "equal_prior_randomized_minimax_across_two_constructed_finite_batch_worlds"
            or diagnostics.get("not_a_realized_world_performance_bound") is not True
            or diagnostics.get("ceiling_tightness")
            != "conservative_witness_not_global_assignment_optimum"
            or diagnostics.get("swap_search_optimality_claimed") is not False
            or int(diagnostics.get("swap_neighbors", -1)) != expected_neighbors
            or diagnostics.get("swap_search")
            != "deterministic_greedy_sparse_neighborhood"
            or diagnostics.get("cost_geometry")
            != config["ceiling_protocol"]["cost_geometry"]
            or int(diagnostics.get("cost_dimension", -1))
            != int(diagnostics.get("feature_dimension", -1))
            or int(diagnostics.get("discarded_feature_dimensions", -1)) != 0
            or _budget_token(diagnostics.get("cost_budget_relative")) != budget_token
            or not isinstance(calibration, Mapping)
            or calibration.get("distance_normalization")
            != config["ceiling_protocol"]["cost_normalization"]
            or calibration.get("reference_centering")
            != "calibration_reference_mean"
            or int(calibration.get("discarded_feature_dimensions", -1)) != 0
        ):
            raise IntegrityError(
                "swap-neighborhood sensitivity geometry or interpretation drifted"
            )
        if any(
            diagnostics.get(key) != primary_diagnostics.get(key)
            for key in invariant_diagnostic_keys
        ) or any(
            calibration.get(key) != primary_calibration.get(key)
            for key in invariant_geometry_keys
        ):
            raise IntegrityError(
                "swap-neighborhood sensitivity did not reuse the frozen primary geometry"
            )
        for key in ("assignment_world_one_sha256", "prediction_world_one_sha256"):
            if not isinstance(diagnostics.get(key), str) or len(diagnostics[key]) != 64:
                raise IntegrityError(
                    "swap-neighborhood sensitivity lacks a sealed alternative world"
                )
        results[expected_neighbors] = {
            "disagreement": disagreement,
            "conditional_ceiling": ceiling,
            "baseline_cost": baseline,
            "alternative_cost": alternative,
            "allowed_cost": allowed,
        }
    return results


def _validate_common(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha: str,
    lock: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int, int, dict[str, float]]:
    if (
        payload.get("status") != "PASS"
        or payload.get("scientific_result") is not True
        or payload.get("immutable_fresh_output") is not True
    ):
        raise IntegrityError("unit is not an immutable passing scientific result")
    if (
        payload.get("primary_quantity")
        != config["reporting"]["primary_quantity"]
        or payload.get("not_a_realized_world_performance_bound") is not True
    ):
        raise IntegrityError("unit primary quantity or scope drifted")
    plan = payload.get("unit_plan")
    if not isinstance(plan, Mapping):
        raise IntegrityError("unit lacks its sealed plan")
    if (
        plan.get("status") != "PASS"
        or plan.get("scientific_result") is not False
        or plan.get("compute_started") is not False
        or plan.get("config_sha256") != config_sha
    ):
        raise IntegrityError("unit plan eligibility or config identity drifted")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise IntegrityError("unit plan lacks runtime identity")
    seed = int(plan.get("split_seed", -1))
    budget_token = _budget_token(plan.get("cost_budget_relative"))
    budget = budget_token / 10_000.0
    if plan.get("protocol") != _expected_protocol(config, budget):
        raise IntegrityError("unit protocol drifted from the locked configuration")
    if int(plan.get("calibration_rows", -1)) != int(
        config["ceiling_protocol"]["calibration_rows"]
    ) or int(plan.get("witness_rows", -1)) != int(
        config["ceiling_protocol"]["witness_rows"]
    ):
        raise IntegrityError("unit split sizes drifted")
    witness = payload.get("ceiling_witness")
    if not isinstance(witness, Mapping):
        raise IntegrityError("unit lacks a ceiling witness")
    if (
        witness.get("interpretation") != WITNESS_INTERPRETATION
        or witness.get("not_an_accuracy_ceiling") is not True
        or witness.get("not_a_realized_world_performance_bound") is not True
    ):
        raise IntegrityError("unit witness interpretation drifted")
    disagreement = _probability(witness.get("disagreement"), name="disagreement")
    ceiling = _probability(
        witness.get("prediction_recovery_ceiling"), name="conditional ceiling"
    )
    if abs(ceiling - (1.0 - disagreement / 2.0)) > 1e-12:
        raise IntegrityError("unit ceiling does not equal one minus disagreement over two")
    baseline = _finite_nonnegative(witness.get("baseline_cost"), name="baseline cost")
    alternative = _finite_nonnegative(
        witness.get("alternative_cost"), name="alternative cost"
    )
    allowed = _finite_nonnegative(witness.get("allowed_cost"), name="allowed cost")
    if alternative > allowed + 1e-10 * max(1.0, allowed):
        raise IntegrityError("unit alternative world violates its cost budget")
    diagnostics = witness.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise IntegrityError("unit witness diagnostics are absent")
    if _budget_token(diagnostics.get("cost_budget_relative")) != budget_token:
        raise IntegrityError("witness and unit-plan budgets disagree")
    expected_geometry = config["ceiling_protocol"]["cost_geometry"]
    expected_normalization = config["ceiling_protocol"]["cost_normalization"]
    calibration_diagnostics = diagnostics.get("calibration")
    assignment_diagnostics = diagnostics.get("assignment")
    if (
        diagnostics.get("cost_geometry") != expected_geometry
        or int(diagnostics.get("discarded_feature_dimensions", -1)) != 0
        or diagnostics.get("quantity_interpretation")
        != "equal_prior_randomized_minimax_across_two_constructed_finite_batch_worlds"
        or diagnostics.get("not_a_realized_world_performance_bound") is not True
        or diagnostics.get("ceiling_tightness")
        != "conservative_witness_not_global_assignment_optimum"
        or diagnostics.get("swap_search_optimality_claimed") is not False
        or int(diagnostics.get("swap_neighbors", -1))
        != int(config["ceiling_protocol"]["swap_neighbors"])
        or diagnostics.get("swap_search") != "deterministic_greedy_all_pairs"
        or not isinstance(calibration_diagnostics, Mapping)
        or calibration_diagnostics.get("cost_geometry") != expected_geometry
        or calibration_diagnostics.get("distance_normalization")
        != expected_normalization
        or calibration_diagnostics.get("reference_centering")
        != "calibration_reference_mean"
        or int(calibration_diagnostics.get("discarded_feature_dimensions", -1)) != 0
        or int(diagnostics.get("feature_dimension", -1)) < 1
        or int(diagnostics.get("cost_dimension", -1))
        != int(diagnostics.get("feature_dimension", -1))
        or int(calibration_diagnostics.get("feature_dimension", -1))
        != int(diagnostics.get("feature_dimension", -1))
        or int(calibration_diagnostics.get("cost_dimension", -1))
        != int(diagnostics.get("feature_dimension", -1))
        or not isinstance(assignment_diagnostics, Mapping)
        or assignment_diagnostics.get("tie_break")
        != "pinned_scipy_on_content_canonical_order"
    ):
        raise IntegrityError("unit witness geometry or search semantics drifted")
    required_hashes = (
        "target_content_sha256",
        "reference_content_sha256",
        "assignment_world_zero_sha256",
        "assignment_world_one_sha256",
        "prediction_world_zero_sha256",
        "prediction_world_one_sha256",
    )
    if any(
        not isinstance(diagnostics.get(key), str)
        or len(str(diagnostics.get(key))) != 64
        for key in required_hashes
    ):
        raise IntegrityError("unit witness lacks sealed content/assignment hashes")
    geometry_hashes = (
        "bures_matrix_sha256",
        "bures_offset_sha256",
        "reference_mean_sha256",
        "reference_inverse_sqrt_sha256",
        "full_geometry_sha256",
    )
    if any(
        not isinstance(calibration_diagnostics.get(key), str)
        or len(str(calibration_diagnostics.get(key))) != 64
        for key in geometry_hashes
    ):
        raise IntegrityError("unit witness lacks sealed fitted-geometry hashes")
    frontier_fit = payload.get("marginal_frontier_fit")
    if not isinstance(frontier_fit, Mapping) or frontier_fit.get("methods") != list(
        MARGINAL_FRONTIER_METHODS
    ):
        raise IntegrityError("unit lacks the locked six-map marginal frontier fit")
    if any(frontier_fit.get(key) is not False for key in (
        "labels_used", "sample_identifiers_used", "pairing_used"
    )):
        raise IntegrityError("marginal frontier fit exceeded its information scope")
    _verify_external_file(
        plan.get("store_manifest"), plan.get("store_manifest_sha256"),
        name="feature-store manifest",
    )
    search_sensitivity = _validate_swap_neighborhood_sensitivity(
        payload.get("swap_neighborhood_sensitivity"),
        config=config,
        budget_token=budget_token,
        primary_witness=witness,
    )
    return plan, seed, budget_token, {
        "disagreement": disagreement,
        "conditional_ceiling": ceiling,
        "baseline_cost": baseline,
        "alternative_cost": alternative,
        "allowed_cost": allowed,
        "swap_neighborhood_sensitivity": search_sensitivity,
    }


def _validate_cluster_statistics(
    value: Any,
    *,
    rows: int,
    disagreement: float,
    method_scores: Mapping[str, float],
    method_count_key: str,
    class_count: int,
    name: str,
) -> dict[str, np.ndarray]:
    """Validate sparse per-class sufficient statistics and fill absent classes."""

    if not isinstance(value, list) or not value:
        raise IntegrityError(f"{name} lacks per-class sufficient statistics")
    row_counts = np.zeros(int(class_count), dtype=np.int64)
    disagreement_counts = np.zeros(int(class_count), dtype=np.int64)
    method_counts = {
        method: np.zeros(int(class_count), dtype=np.int64)
        for method in MARGINAL_FRONTIER_METHODS
    }
    seen: set[int] = set()
    for record in value:
        if not isinstance(record, Mapping):
            raise IntegrityError(f"{name} has a malformed class record")
        class_index = int(record.get("class_index", -1))
        if class_index < 0 or class_index >= int(class_count) or class_index in seen:
            raise IntegrityError(f"{name} has a duplicate or invalid class index")
        seen.add(class_index)
        cluster_rows = int(record.get("rows", -1))
        cluster_disagreement = int(record.get("prediction_disagreement_count", -1))
        counts = record.get(method_count_key)
        if (
            cluster_rows < 1
            or cluster_disagreement < 0
            or cluster_disagreement > cluster_rows
            or not isinstance(counts, Mapping)
            or set(counts) != set(MARGINAL_FRONTIER_METHODS)
        ):
            raise IntegrityError(f"{name} has invalid class counts")
        row_counts[class_index] = cluster_rows
        disagreement_counts[class_index] = cluster_disagreement
        for method in MARGINAL_FRONTIER_METHODS:
            count = int(counts[method])
            if count < 0 or count > cluster_rows:
                raise IntegrityError(f"{name} method count is outside class rows")
            method_counts[method][class_index] = count
    if int(np.sum(row_counts)) != int(rows):
        raise IntegrityError(f"{name} class rows do not sum to the witness rows")
    disagreement_total = int(np.sum(disagreement_counts))
    if abs(disagreement_total / rows - disagreement) > 1e-12:
        raise IntegrityError(f"{name} disagreement counts do not reproduce the witness")
    for method in MARGINAL_FRONTIER_METHODS:
        if abs(float(np.sum(method_counts[method])) / rows - method_scores[method]) > 1e-12:
            raise IntegrityError(f"{name} method counts do not reproduce its score")
    return {
        "rows": row_counts,
        "disagreement": disagreement_counts,
        **{f"method:{method}": value for method, value in method_counts.items()},
    }


def _validate_method_scores(value: Any, *, name: str) -> dict[str, float]:
    if (
        not isinstance(value, Mapping)
        or len(value) != len(MARGINAL_FRONTIER_METHODS)
        or set(value) != set(MARGINAL_FRONTIER_METHODS)
    ):
        raise IntegrityError(f"{name} does not contain the exact locked method panel")
    return {
        method: _probability(value[method], name=f"{name}:{method}")
        for method in MARGINAL_FRONTIER_METHODS
    }


def _argmax_method(scores: Mapping[str, float]) -> str:
    return max(
        MARGINAL_FRONTIER_METHODS,
        key=lambda method: (scores[method], -MARGINAL_FRONTIER_METHODS.index(method)),
    )


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise IntegrityError("cannot aggregate an empty or nonfinite metric")
    return float(np.mean(array))


def _average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right) or len(left) < 2:
        raise IntegrityError("Spearman correlation needs at least two paired cells")
    x = _average_ranks(left)
    y = _average_ranks(right)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {
            "defined": False,
            "value": None,
            "cells": len(left),
            "reason": "constant_rank_vector",
        }
    value = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(value):
        raise IntegrityError("Spearman correlation became nonfinite")
    return {"defined": True, "value": value, "cells": len(left), "reason": None}


def _condition_summary(
    records: list[dict[str, Any]], *, identity: Mapping[str, Any]
) -> dict[str, Any]:
    method_means = {
        method: _mean(record["method_scores"][method] for record in records)
        for method in MARGINAL_FRONTIER_METHODS
    }
    best = _argmax_method(method_means)
    sensitivity_neighbors = sorted(
        int(value) for value in records[0]["swap_neighborhood_sensitivity"]
    )
    if any(
        sorted(int(value) for value in record["swap_neighborhood_sensitivity"])
        != sensitivity_neighbors
        for record in records
    ):
        raise IntegrityError("condition has inconsistent swap-neighborhood sensitivity")
    search_sensitivity = [
        {
            "swap_neighbors": neighbors,
            "prediction_disagreement_fraction": _mean(
                record["swap_neighborhood_sensitivity"][neighbors]["disagreement"]
                for record in records
            ),
            "equal_prior_randomized_minimax_ceiling": _mean(
                record["swap_neighborhood_sensitivity"][neighbors][
                    "conditional_ceiling"
                ]
                for record in records
            ),
            "interpretation": (
                "conservative witness-derived search sensitivity on the same "
                "frozen full-rank geometry"
            ),
        }
        for neighbors in sensitivity_neighbors
    ]
    return {
        **identity,
        "split_seeds": sorted(int(record["seed"]) for record in records),
        "prediction_disagreement_fraction": _mean(
            record["disagreement"] for record in records
        ),
        "equal_prior_randomized_minimax_ceiling": _mean(
            record["conditional_ceiling"] for record in records
        ),
        "per_method_mean": method_means,
        "posthoc_oracle_envelope_method": best,
        "posthoc_oracle_envelope_value": method_means[best],
        "posthoc_oracle_envelope_deployable": False,
        "all_six_fixed_methods_are_prespecified_deployable_references": True,
        "source_value": method_means["none"],
        "bures_oas_value": method_means["bures_oas"],
        "swap_neighborhood_sensitivity": search_sensitivity,
        "seed_nesting": "method_mean_over_split_seeds_before_frontier_and_condition_macro",
    }


def _macro_swap_neighborhood_sensitivity(
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate prespecified search sensitivity over condition cells."""

    if not conditions:
        raise IntegrityError("cannot aggregate empty search-sensitivity conditions")
    expected = [
        int(row["swap_neighbors"])
        for row in conditions[0]["swap_neighborhood_sensitivity"]
    ]
    output: list[dict[str, Any]] = []
    for index, neighbors in enumerate(expected):
        entries = []
        for condition in conditions:
            sensitivity = condition["swap_neighborhood_sensitivity"]
            if (
                len(sensitivity) != len(expected)
                or int(sensitivity[index]["swap_neighbors"]) != neighbors
            ):
                raise IntegrityError(
                    "condition search-sensitivity order drifted during macro aggregation"
                )
            entries.append(sensitivity[index])
        output.append({
            "swap_neighbors": neighbors,
            "prediction_disagreement_fraction_macro_mean": _mean(
                entry["prediction_disagreement_fraction"] for entry in entries
            ),
            "equal_prior_randomized_minimax_ceiling_macro_mean": _mean(
                entry["equal_prior_randomized_minimax_ceiling"] for entry in entries
            ),
            "descriptive_sensitivity_only": True,
        })
    return output


def _percentile_summary(
    values: np.ndarray, *, point_estimate: float, replicates: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (int(replicates),)
        or not np.all(np.isfinite(array))
        or not math.isfinite(float(point_estimate))
    ):
        raise IntegrityError("bootstrap metric has missing or nonfinite replicates")
    lower, upper = np.quantile(array, [0.025, 0.975])
    return {
        "point_estimate": float(point_estimate),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": 0.95,
        "interval": "percentile",
        "replicates": int(replicates),
        "dropped_replicates": 0,
    }


def _conditional_spearman_summary(
    values: Iterable[float],
    *,
    point: Mapping[str, Any],
    attempted_replicates: int,
    undefined_reasons: Mapping[str, int],
) -> dict[str, Any]:
    """Report a secondary rank-correlation interval without hiding degeneracy.

    A six-cluster bootstrap can legitimately resample a constant rank vector.
    Such a draw has no Spearman coefficient; it is not a failed scientific
    unit and must not abort the primary panel.  We therefore report the exact
    number and reasons for undefined draws and condition the secondary
    percentile interval on the nonconstant draws.  No undefined draw is
    silently converted to zero or omitted from the accounting.
    """

    attempted = int(attempted_replicates)
    array = np.asarray(tuple(values), dtype=np.float64)
    if attempted < 1 or array.ndim != 1 or not np.all(np.isfinite(array)):
        raise IntegrityError("conditional Spearman bootstrap values are malformed")
    reasons = {str(key): int(value) for key, value in undefined_reasons.items()}
    if any(value < 0 for value in reasons.values()):
        raise IntegrityError("conditional Spearman undefined counts are negative")
    undefined = int(sum(reasons.values()))
    defined = int(len(array))
    if defined + undefined != attempted:
        raise IntegrityError("conditional Spearman replicate accounting is incomplete")
    if not isinstance(point, Mapping) or point.get("defined") not in {True, False}:
        raise IntegrityError("conditional Spearman point estimate is malformed")
    point_value = point.get("value")
    if point.get("defined") is True:
        if point_value is None or not math.isfinite(float(point_value)):
            raise IntegrityError("defined Spearman point estimate is nonfinite")
        point_value = float(point_value)
    elif point_value is not None:
        raise IntegrityError("undefined Spearman point estimate must be null")
    lower: float | None = None
    upper: float | None = None
    if defined:
        lower_value, upper_value = np.quantile(array, [0.025, 0.975])
        lower, upper = float(lower_value), float(upper_value)
    return {
        "point_estimate": point_value,
        "point_defined": bool(point["defined"]),
        "point_undefined_reason": point.get("reason"),
        "lower": lower,
        "upper": upper,
        "confidence_level": 0.95,
        "interval": (
            "percentile_conditional_on_defined_nonconstant_resamples"
            if defined else None
        ),
        "attempted_replicates": attempted,
        "defined_replicates": defined,
        "undefined_replicates": undefined,
        "defined_fraction": defined / attempted,
        "undefined_reason_counts": reasons,
        "undefined_replicates_explicitly_reported": True,
        "secondary_diagnostic_only": True,
    }


def _bootstrap_play3_panel(
    *,
    config: Mapping[str, Any],
    natural_records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    positive_records: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    natural_conditions: list[dict[str, Any]],
    positive_conditions: list[dict[str, Any]],
    encoders: tuple[str, ...],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    corruptions: tuple[str, ...],
) -> dict[str, Any]:
    """Run the exact prespecified nested cluster bootstrap without row drops."""

    policy = config["reporting"]["uncertainty"]
    replicates = int(policy["replicates"])
    seed = int(policy["seed"])
    class_count = int(config["dataset"]["class_count"])
    if (
        replicates != 10_000
        or seed != 20_260_827
        or policy.get("rng") != "numpy_pcg64"
        or policy.get("interval") != "percentile_95"
        or policy.get("natural_cluster") != "imagenet_wnid"
        or policy.get("positive_cluster") != "corruption"
        or policy.get("encoders_nested_within_cluster") is not True
        or policy.get("zero_silent_drops") is not True
        or policy.get("secondary_spearman_undefined_policy")
        != "report_point_undefined_and_defined_fraction_plus_conditional_percentile_interval_over_nonconstant_resamples"
    ):
        raise IntegrityError("Play-3 uncertainty policy drifted")

    # Split seeds are overlapping repeats, not bootstrap clusters.  Average
    # their per-WNID sufficient counts first, then draw WNIDs while retaining
    # every encoder and method under the same bootstrap weights.
    natural_rng = np.random.Generator(np.random.PCG64(seed))
    natural_weights = natural_rng.multinomial(
        class_count,
        np.full(class_count, 1.0 / class_count, dtype=np.float64),
        size=replicates,
    ).astype(np.float64, copy=False)
    natural_by_budget: list[dict[str, Any]] = []
    for budget in budgets:
        disagreement_by_encoder: list[np.ndarray] = []
        method_by_encoder: list[np.ndarray] = []
        for encoder in encoders:
            rows_by_seed = np.stack([
                np.asarray(natural_records[(encoder, split_seed, budget)][
                    "cluster_statistics"
                ]["rows"], dtype=np.float64)
                for split_seed in seeds
            ])
            disagreement_by_seed = np.stack([
                np.asarray(natural_records[(encoder, split_seed, budget)][
                    "cluster_statistics"
                ]["disagreement"], dtype=np.float64)
                for split_seed in seeds
            ])
            method_by_seed = np.stack([
                np.stack([
                    np.asarray(natural_records[(encoder, split_seed, budget)][
                        "cluster_statistics"
                    ][f"method:{method}"], dtype=np.float64)
                    for method in MARGINAL_FRONTIER_METHODS
                ], axis=1)
                for split_seed in seeds
            ])
            rows = np.mean(rows_by_seed, axis=0)
            disagreement_counts = np.mean(disagreement_by_seed, axis=0)
            method_counts = np.mean(method_by_seed, axis=0)
            denominators = natural_weights @ rows
            if np.any(denominators <= 0.0):
                raise IntegrityError(
                    "natural WNID bootstrap produced an empty resample; no drops allowed"
                )
            disagreement_by_encoder.append(
                (natural_weights @ disagreement_counts) / denominators
            )
            method_by_encoder.append(
                (natural_weights @ method_counts) / denominators[:, None]
            )
        disagreement_samples = np.mean(
            np.stack(disagreement_by_encoder, axis=1), axis=1
        )
        method_samples = np.mean(
            np.stack(method_by_encoder, axis=1), axis=1
        )
        oracle_samples = np.mean(
            np.max(np.stack(method_by_encoder, axis=1), axis=2), axis=1
        )
        ceiling_samples = 1.0 - disagreement_samples / 2.0
        conditions = [
            row for row in natural_conditions
            if _budget_token(row["cost_budget_relative"]) == budget
        ]
        point_disagreement = _mean(
            row["prediction_disagreement_fraction"] for row in conditions
        )
        point_methods = {
            method: _mean(row["per_method_mean"][method] for row in conditions)
            for method in MARGINAL_FRONTIER_METHODS
        }
        point_oracle = _mean(
            row["posthoc_oracle_envelope_value"] for row in conditions
        )
        natural_by_budget.append({
            "cost_budget_relative": budget / 10_000.0,
            "prediction_disagreement_fraction": _percentile_summary(
                disagreement_samples,
                point_estimate=point_disagreement,
                replicates=replicates,
            ),
            "equal_prior_randomized_minimax_ceiling": _percentile_summary(
                ceiling_samples,
                point_estimate=1.0 - point_disagreement / 2.0,
                replicates=replicates,
            ),
            "posthoc_oracle_envelope": _percentile_summary(
                oracle_samples,
                point_estimate=point_oracle,
                replicates=replicates,
            ),
            "per_fixed_method": {
                method: _percentile_summary(
                    method_samples[:, method_index],
                    point_estimate=point_methods[method],
                    replicates=replicates,
                )
                for method_index, method in enumerate(MARGINAL_FRONTIER_METHODS)
            },
        })

    # Corruptions are the positive-control clusters.  Split seeds are averaged
    # first and all encoders/methods remain paired inside each sampled cluster.
    positive_rng = np.random.Generator(np.random.PCG64(seed))
    sampled_corruptions = positive_rng.integers(
        0, len(corruptions), size=(replicates, len(corruptions))
    )
    positive_by_budget: list[dict[str, Any]] = []
    for budget in budgets:
        disagreement = np.empty((len(corruptions), len(encoders)), dtype=np.float64)
        method_values = np.empty(
            (len(corruptions), len(encoders), len(MARGINAL_FRONTIER_METHODS)),
            dtype=np.float64,
        )
        for corruption_index, corruption in enumerate(corruptions):
            for encoder_index, encoder in enumerate(encoders):
                condition_records = [
                    positive_records[(corruption, encoder, split_seed, budget)]
                    for split_seed in seeds
                ]
                disagreement[corruption_index, encoder_index] = _mean(
                    record["disagreement"] for record in condition_records
                )
                for method_index, method in enumerate(MARGINAL_FRONTIER_METHODS):
                    method_values[corruption_index, encoder_index, method_index] = (
                        _mean(
                            record["method_scores"][method]
                            for record in condition_records
                        )
                    )
        sampled_disagreement = disagreement[sampled_corruptions]
        sampled_methods = method_values[sampled_corruptions]
        disagreement_samples = np.mean(sampled_disagreement, axis=(1, 2))
        method_samples = np.mean(sampled_methods, axis=(1, 2))
        oracle_by_condition = np.max(sampled_methods, axis=3)
        oracle_samples = np.mean(oracle_by_condition, axis=(1, 2))
        ceiling_samples = 1.0 - disagreement_samples / 2.0
        spearman_ceiling: list[float] = []
        spearman_disagreement: list[float] = []
        ceiling_undefined: Counter[str] = Counter()
        disagreement_undefined: Counter[str] = Counter()
        for replicate in range(replicates):
            ceiling_cells = (
                1.0 - sampled_disagreement[replicate].reshape(-1) / 2.0
            ).tolist()
            disagreement_cells = sampled_disagreement[replicate].reshape(-1).tolist()
            oracle_cells = oracle_by_condition[replicate].reshape(-1).tolist()
            first = _spearman(ceiling_cells, oracle_cells)
            second = _spearman(disagreement_cells, oracle_cells)
            if first["defined"]:
                spearman_ceiling.append(float(first["value"]))
            else:
                ceiling_undefined[str(first["reason"])] += 1
            if second["defined"]:
                spearman_disagreement.append(float(second["value"]))
            else:
                disagreement_undefined[str(second["reason"])] += 1
        conditions = [
            row for row in positive_conditions
            if _budget_token(row["cost_budget_relative"]) == budget
        ]
        point_disagreement = _mean(
            row["prediction_disagreement_fraction"] for row in conditions
        )
        point_methods = {
            method: _mean(row["per_method_mean"][method] for row in conditions)
            for method in MARGINAL_FRONTIER_METHODS
        }
        point_oracle = _mean(
            row["posthoc_oracle_envelope_value"] for row in conditions
        )
        point_ceiling_spearman = _spearman(
            [row["equal_prior_randomized_minimax_ceiling"] for row in conditions],
            [row["posthoc_oracle_envelope_value"] for row in conditions],
        )
        point_disagreement_spearman = _spearman(
            [row["prediction_disagreement_fraction"] for row in conditions],
            [row["posthoc_oracle_envelope_value"] for row in conditions],
        )
        positive_by_budget.append({
            "cost_budget_relative": budget / 10_000.0,
            "prediction_disagreement_fraction": _percentile_summary(
                disagreement_samples,
                point_estimate=point_disagreement,
                replicates=replicates,
            ),
            "equal_prior_randomized_minimax_ceiling": _percentile_summary(
                ceiling_samples,
                point_estimate=1.0 - point_disagreement / 2.0,
                replicates=replicates,
            ),
            "posthoc_oracle_envelope": _percentile_summary(
                oracle_samples,
                point_estimate=point_oracle,
                replicates=replicates,
            ),
            "per_fixed_method": {
                method: _percentile_summary(
                    method_samples[:, method_index],
                    point_estimate=point_methods[method],
                    replicates=replicates,
                )
                for method_index, method in enumerate(MARGINAL_FRONTIER_METHODS)
            },
            "spearman_ceiling_vs_posthoc_oracle_envelope": _conditional_spearman_summary(
                spearman_ceiling,
                point=point_ceiling_spearman,
                attempted_replicates=replicates,
                undefined_reasons=ceiling_undefined,
            ),
            "spearman_disagreement_vs_posthoc_oracle_envelope": _conditional_spearman_summary(
                spearman_disagreement,
                point=point_disagreement_spearman,
                attempted_replicates=replicates,
                undefined_reasons=disagreement_undefined,
            ),
        })
    return {
        "status": "PASS",
        "rng": "numpy.random.Generator(PCG64)",
        "seed": seed,
        "replicates": replicates,
        "confidence_level": 0.95,
        "interval": "percentile",
        "zero_silent_drops": True,
        "secondary_spearman_undefined_policy": policy[
            "secondary_spearman_undefined_policy"
        ],
        "split_seed_policy": (
            "average_over_overlapping_repeats_before_cluster_resampling"
        ),
        "natural": {
            "cluster": "imagenet_wnid",
            "cluster_count": class_count,
            "encoders_and_methods_nested": True,
            "budget_rows": natural_by_budget,
        },
        "positive_control": {
            "cluster": "corruption",
            "cluster_count": len(corruptions),
            "encoders_and_methods_nested": True,
            "six_cluster_inference_is_weak": len(corruptions) == 6,
            "budget_rows": positive_by_budget,
        },
    }
def aggregate_play3_panel(
    *,
    config_path: str | Path,
    natural_units_root: str | Path,
    positive_units_root: str | Path,
    output: str | Path,
) -> Path:
    """Validate and aggregate the exact locked Play-3 panel without omissions."""

    config, resolved_config, lock, _ = load_locked_play3_config(config_path)
    config_sha = sha256_file(resolved_config)
    lock_sha = sha256_json(lock)
    natural_files = _load_unit_directory(
        natural_units_root, schema=NATURAL_SHIFT_RESULT_SCHEMA
    )
    positive_files = _load_unit_directory(positive_units_root, schema=POSITIVE_SCHEMA)
    encoders = tuple(str(value) for value in config["encoders"])
    seeds = tuple(int(value) for value in config["ceiling_protocol"]["split_seeds"])
    budgets = tuple(sorted({
        _budget_token(config["ceiling_protocol"]["primary_cost_budget_relative"]),
        *(
            _budget_token(value)
            for value in config["ceiling_protocol"]["cost_budget_relative_sensitivity"]
        ),
    }))
    corruptions = tuple(str(value) for value in config["positive_control"][
        "heldout_corruptions"
    ])
    severity = int(config["positive_control"]["severity"])
    expected_natural = {
        (encoder, seed, budget)
        for encoder in encoders for seed in seeds for budget in budgets
    }
    expected_positive = {
        (corruption, encoder, seed, budget)
        for corruption in corruptions for encoder in encoders
        for seed in seeds for budget in budgets
    }

    natural_records: dict[tuple[str, int, int], dict[str, Any]] = {}
    positive_records: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    runtime_hashes: set[str] = set()
    natural_store_binding: dict[str, tuple[str, str]] = {}
    positive_store_binding: dict[tuple[str, str, int], tuple[str, str]] = {}
    provenance_binding: tuple[str, str] | None = None

    for path, payload in natural_files:
        plan, seed, budget, metrics = _validate_common(
            payload, config=config, config_sha=config_sha, lock=lock
        )
        if plan.get("lock_manifest_sha256") != lock_sha:
            raise IntegrityError("natural unit extension-lock identity drifted")
        encoder = str(payload.get("encoder"))
        if (
            payload.get("dataset") != config["dataset"]["name"]
            or plan.get("dataset") != payload.get("dataset")
            or plan.get("encoder") != encoder
            or int(payload.get("split_seed", -1)) != seed
        ):
            raise IntegrityError("natural unit identity fields disagree")
        expected_id = (
            f"imagenet_sketch__{encoder}__seed_{seed}__budget_{budget:04d}bp"
        )
        if plan.get("unit_id") != expected_id:
            raise IntegrityError("natural unit ID drifted")
        key = (encoder, seed, budget)
        if key in natural_records:
            raise IntegrityError(f"duplicate natural unit: {key}")
        posthoc = payload.get("posthoc_evaluation")
        if (
            not isinstance(posthoc, Mapping)
            or posthoc.get("used_for_geometry_or_map_fitting") is not False
            or posthoc.get("used_for_posthoc_oracle_envelope") is not True
            or posthoc.get("posthoc_oracle_envelope_deployable") is not False
            or posthoc.get("deployable_primary_reference_method") != "bures_oas"
        ):
            raise IntegrityError("natural unit lacks separate posthoc label evaluation")
        method_scores = _validate_method_scores(
            posthoc.get("per_method_target_accuracy"),
            name="natural target accuracy",
        )
        oracle_method = _argmax_method(method_scores)
        if (
            posthoc.get("posthoc_oracle_envelope_method") != oracle_method
            or abs(float(posthoc.get("posthoc_oracle_envelope_target_accuracy", -1.0))
                   - method_scores[oracle_method]) > 1e-12
        ):
            raise IntegrityError("natural posthoc-oracle declaration drifted")
        witness_diagnostics = payload["ceiling_witness"]["diagnostics"]
        if posthoc.get("target_content_sha256") != witness_diagnostics.get(
            "target_content_sha256"
        ):
            raise IntegrityError("natural posthoc labels do not bind the witness target")
        cluster_statistics = _validate_cluster_statistics(
            posthoc.get("per_wnid_sufficient_statistics"),
            rows=int(plan["witness_rows"]),
            disagreement=metrics["disagreement"],
            method_scores=method_scores,
            method_count_key="method_correct_counts",
            class_count=int(config["dataset"]["class_count"]),
            name="natural WNID statistics",
        )
        provenance_path = _verify_external_file(
            plan.get("dataset_provenance_manifest"),
            plan.get("dataset_provenance_manifest_sha256"),
            name="ImageNet-Sketch provenance manifest",
        )
        current_provenance = (
            str(provenance_path), str(plan["dataset_provenance_manifest_sha256"])
        )
        if provenance_binding is None:
            provenance_binding = current_provenance
        elif provenance_binding != current_provenance:
            raise IntegrityError("natural units bind different dataset provenance")
        store_binding = (str(plan["store_manifest"]), str(plan["store_manifest_sha256"]))
        if encoder in natural_store_binding and natural_store_binding[encoder] != store_binding:
            raise IntegrityError("natural units bind different stores for one encoder")
        natural_store_binding[encoder] = store_binding
        runtime_hashes.add(sha256_json(plan["runtime"]))
        invariant = {
            "reference_split_content_sha256": payload.get(
                "reference_split_content_sha256"
            ),
            "target_split_content_sha256": payload.get("target_split_content_sha256"),
            "target_content_sha256": witness_diagnostics.get("target_content_sha256"),
            "reference_content_sha256": witness_diagnostics.get(
                "reference_content_sha256"
            ),
            "target_calibration_content_sha256": witness_diagnostics[
                "calibration"
            ].get("target_calibration_content_sha256"),
            "reference_calibration_content_sha256": witness_diagnostics[
                "calibration"
            ].get("reference_calibration_content_sha256"),
            "assignment_world_zero_sha256": witness_diagnostics.get(
                "assignment_world_zero_sha256"
            ),
            "prediction_world_zero_sha256": witness_diagnostics.get(
                "prediction_world_zero_sha256"
            ),
            "baseline_cost": metrics["baseline_cost"],
            "marginal_frontier_fit": payload["marginal_frontier_fit"],
            "method_scores": method_scores,
            "method_cluster_counts": {
                method: cluster_statistics[f"method:{method}"].tolist()
                for method in MARGINAL_FRONTIER_METHODS
            },
        }
        natural_records[key] = {
            **metrics, "seed": seed, "budget": budget,
            "method_scores": method_scores, "path": path,
            "cluster_statistics": cluster_statistics,
            "budget_invariant_sha256": sha256_json(invariant),
        }

    for path, payload in positive_files:
        plan, seed, budget, metrics = _validate_common(
            payload, config=config, config_sha=config_sha, lock=lock
        )
        if plan.get("extension_lock") != lock:
            raise IntegrityError("positive-control extension-lock identity drifted")
        encoder = str(payload.get("encoder"))
        corruption = str(payload.get("corruption"))
        unit_severity = int(payload.get("severity", -1))
        if (
            payload.get("dataset") != config["positive_control"]["dataset"]
            or plan.get("dataset") != payload.get("dataset")
            or plan.get("encoder") != encoder
            or plan.get("corruption") != corruption
            or int(plan.get("severity", -1)) != unit_severity
            or int(payload.get("split_seed", -1)) != seed
            or plan.get("pairing_used_for_split_administration_before_fit") is not True
            or plan.get(
                "labels_used_for_administrative_pair_validation_before_fit"
            ) is not True
            or plan.get("pairing_unavailable_to_estimator") is not True
            or plan.get("labels_unavailable_to_estimator") is not True
            or plan.get("labels_used_by_estimator") is not False
        ):
            raise IntegrityError("positive-control unit identity fields disagree")
        expected_id = (
            f"positive__{corruption}__{encoder}__seed_{seed}__budget_{budget:04d}bp"
        )
        if plan.get("unit_id") != expected_id:
            raise IntegrityError("positive-control unit ID drifted")
        key = (corruption, encoder, seed, budget)
        if key in positive_records:
            raise IntegrityError(f"duplicate positive-control unit: {key}")
        posthoc = payload.get("posthoc_validation")
        if (
            not isinstance(posthoc, Mapping)
            or payload.get("pairing_used_for_split_administration_before_fit") is not True
            or payload.get(
                "labels_used_for_administrative_pair_validation_before_fit"
            ) is not True
            or payload.get("pairing_unavailable_to_estimator") is not True
            or payload.get("labels_unavailable_to_estimator") is not True
            or payload.get("true_assignment_scored_only_after_witness_freeze") is not True
            or posthoc.get("posthoc_oracle_envelope_deployable") is not False
            or posthoc.get("deployable_primary_reference_method") != "bures_oas"
        ):
            raise IntegrityError("positive-control unit lacks posthoc pairing evaluation")
        pairing_content_sha = posthoc.get("pairing_content_sha256")
        if not isinstance(pairing_content_sha, str) or len(pairing_content_sha) != 64:
            raise IntegrityError("positive-control unit lacks a sealed pairing table")
        true_assignment_cost = _finite_nonnegative(
            posthoc.get("true_assignment_cost"), name="true assignment cost"
        )
        for metric_key in (
            "world_zero_assignment_recovery",
            "world_one_assignment_recovery",
        ):
            _probability(posthoc.get(metric_key), name=metric_key)
        method_scores = _validate_method_scores(
            posthoc.get("per_method_reference_prediction_agreement"),
            name="positive-control prediction agreement",
        )
        oracle_method = _argmax_method(method_scores)
        if (
            posthoc.get("posthoc_oracle_envelope_method") != oracle_method
            or abs(float(posthoc.get(
                "posthoc_oracle_envelope_reference_prediction_agreement", -1.0
            )) - method_scores[oracle_method]) > 1e-12
        ):
            raise IntegrityError("positive-control posthoc-oracle declaration drifted")
        cluster_statistics = _validate_cluster_statistics(
            posthoc.get("per_wnid_sufficient_statistics"),
            rows=int(plan["witness_rows"]),
            disagreement=metrics["disagreement"],
            method_scores=method_scores,
            method_count_key="method_agreement_counts",
            class_count=int(config["dataset"]["class_count"]),
            name="positive-control WNID statistics",
        )
        store_binding = (str(plan["store_manifest"]), str(plan["store_manifest_sha256"]))
        condition = (corruption, encoder, unit_severity)
        if condition in positive_store_binding and positive_store_binding[condition] != store_binding:
            raise IntegrityError("positive-control units bind different condition stores")
        positive_store_binding[condition] = store_binding
        runtime_hashes.add(sha256_json(plan["runtime"]))
        witness_diagnostics = payload["ceiling_witness"]["diagnostics"]
        selected_pair_sha = payload.get("selected_pair_split_sha256")
        if not isinstance(selected_pair_sha, str) or len(selected_pair_sha) != 64:
            raise IntegrityError("positive-control unit lacks its administrative split hash")
        invariant = {
            "selected_pair_split_sha256": selected_pair_sha,
            "pairing_content_sha256": pairing_content_sha,
            "target_content_sha256": witness_diagnostics.get("target_content_sha256"),
            "reference_content_sha256": witness_diagnostics.get(
                "reference_content_sha256"
            ),
            "target_calibration_content_sha256": witness_diagnostics[
                "calibration"
            ].get("target_calibration_content_sha256"),
            "reference_calibration_content_sha256": witness_diagnostics[
                "calibration"
            ].get("reference_calibration_content_sha256"),
            "assignment_world_zero_sha256": witness_diagnostics.get(
                "assignment_world_zero_sha256"
            ),
            "prediction_world_zero_sha256": witness_diagnostics.get(
                "prediction_world_zero_sha256"
            ),
            "baseline_cost": metrics["baseline_cost"],
            "true_assignment_cost": true_assignment_cost,
            "marginal_frontier_fit": payload["marginal_frontier_fit"],
            "method_scores": method_scores,
            "method_cluster_counts": {
                method: cluster_statistics[f"method:{method}"].tolist()
                for method in MARGINAL_FRONTIER_METHODS
            },
        }
        positive_records[key] = {
            **metrics, "seed": seed, "budget": budget,
            "method_scores": method_scores, "path": path,
            "budget_invariant_sha256": sha256_json(invariant),
        }

    missing_natural = sorted(expected_natural.difference(natural_records))
    extra_natural = sorted(set(natural_records).difference(expected_natural))
    missing_positive = sorted(expected_positive.difference(positive_records))
    extra_positive = sorted(set(positive_records).difference(expected_positive))
    if missing_natural or extra_natural or missing_positive or extra_positive:
        raise IntegrityError(
            "Play-3 panel is incomplete or contains unlocked units: "
            f"natural_missing={missing_natural[:3]}, natural_extra={extra_natural[:3]}, "
            f"positive_missing={missing_positive[:3]}, positive_extra={extra_positive[:3]}"
        )
    for encoder in encoders:
        for seed in seeds:
            hashes = {
                natural_records[(encoder, seed, budget)][
                    "budget_invariant_sha256"
                ]
                for budget in budgets
            }
            if len(hashes) != 1:
                raise IntegrityError(
                    "natural split/geometry/frontier identity drifted across budgets"
                )
    for corruption in corruptions:
        for encoder in encoders:
            for seed in seeds:
                hashes = {
                    positive_records[(corruption, encoder, seed, budget)][
                        "budget_invariant_sha256"
                    ]
                    for budget in budgets
                }
                if len(hashes) != 1:
                    raise IntegrityError(
                        "positive split/geometry/frontier identity drifted across budgets"
                    )
    if len(runtime_hashes) != 1:
        raise IntegrityError("Play-3 unit runtime identities drift across the panel")

    natural_conditions: list[dict[str, Any]] = []
    for budget in budgets:
        for encoder in encoders:
            records = [natural_records[(encoder, seed, budget)] for seed in seeds]
            natural_conditions.append(_condition_summary(
                records,
                identity={
                    "dataset": config["dataset"]["name"],
                    "encoder": encoder,
                    "cost_budget_relative": budget / 10_000.0,
                    "metric": "target_accuracy",
                },
            ))

    positive_conditions: list[dict[str, Any]] = []
    for budget in budgets:
        for corruption in corruptions:
            for encoder in encoders:
                records = [
                    positive_records[(corruption, encoder, seed, budget)]
                    for seed in seeds
                ]
                positive_conditions.append(_condition_summary(
                    records,
                    identity={
                        "dataset": config["positive_control"]["dataset"],
                        "corruption": corruption,
                        "severity": severity,
                        "encoder": encoder,
                        "cost_budget_relative": budget / 10_000.0,
                        "metric": "reference_prediction_agreement",
                    },
                ))

    positive_macro: list[dict[str, Any]] = []
    natural_macro: list[dict[str, Any]] = []
    for budget in budgets:
        relative_budget = budget / 10_000.0
        positive_cells = [
            row for row in positive_conditions
            if row["cost_budget_relative"] == relative_budget
        ]
        natural_cells = [
            row for row in natural_conditions
            if row["cost_budget_relative"] == relative_budget
        ]
        positive_macro.append({
            "cost_budget_relative": relative_budget,
            "condition_cells": len(positive_cells),
            "equal_prior_randomized_minimax_ceiling_macro_mean": _mean(
                row["equal_prior_randomized_minimax_ceiling"] for row in positive_cells
            ),
            "posthoc_oracle_envelope_macro_mean": _mean(
                row["posthoc_oracle_envelope_value"] for row in positive_cells
            ),
            "posthoc_oracle_envelope_deployable": False,
            "source_macro_mean": _mean(row["source_value"] for row in positive_cells),
            "bures_oas_macro_mean": _mean(
                row["bures_oas_value"] for row in positive_cells
            ),
            "swap_neighborhood_sensitivity": (
                _macro_swap_neighborhood_sensitivity(positive_cells)
            ),
            "spearman_ceiling_vs_posthoc_oracle_envelope": _spearman(
                [row["equal_prior_randomized_minimax_ceiling"] for row in positive_cells],
                [row["posthoc_oracle_envelope_value"] for row in positive_cells],
            ),
            "spearman_disagreement_vs_posthoc_oracle_envelope": _spearman(
                [row["prediction_disagreement_fraction"] for row in positive_cells],
                [row["posthoc_oracle_envelope_value"] for row in positive_cells],
            ),
        })
        natural_macro.append({
            "cost_budget_relative": relative_budget,
            "condition_cells": len(natural_cells),
            "equal_prior_randomized_minimax_ceiling_macro_mean": _mean(
                row["equal_prior_randomized_minimax_ceiling"] for row in natural_cells
            ),
            "posthoc_oracle_envelope_target_accuracy_macro_mean": _mean(
                row["posthoc_oracle_envelope_value"] for row in natural_cells
            ),
            "posthoc_oracle_envelope_deployable": False,
            "source_target_accuracy_macro_mean": _mean(
                row["source_value"] for row in natural_cells
            ),
            "bures_oas_target_accuracy_macro_mean": _mean(
                row["bures_oas_value"] for row in natural_cells
            ),
            "swap_neighborhood_sensitivity": (
                _macro_swap_neighborhood_sensitivity(natural_cells)
            ),
        })

    uncertainty = _bootstrap_play3_panel(
        config=config,
        natural_records=natural_records,
        positive_records=positive_records,
        natural_conditions=natural_conditions,
        positive_conditions=positive_conditions,
        encoders=encoders,
        seeds=seeds,
        budgets=budgets,
        corruptions=corruptions,
    )

    destination = Path(output).expanduser().resolve()
    natural_root = Path(natural_units_root).expanduser().resolve()
    positive_root = Path(positive_units_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite aggregate output: {destination}")
    if destination in {resolved_config, natural_root, positive_root} or any(
        root in destination.parents for root in (natural_root, positive_root)
    ):
        raise IntegrityError("aggregate output overlaps a protected input tree")
    payload = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "complete_panel": True,
        "config": str(resolved_config),
        "config_sha256": config_sha,
        "extension_lock": dict(lock),
        "extension_lock_content_sha256": lock_sha,
        "runtime_identity_sha256": next(iter(runtime_hashes)),
        "panel_scope": {
            "methods": list(MARGINAL_FRONTIER_METHODS),
            "posthoc_oracle_envelope": (
                "non-deployable maximum externally scored condition-level "
                "method mean after split-seed averaging"
            ),
            "posthoc_oracle_envelope_deployable": False,
            "deployable_prespecified_references": list(MARGINAL_FRONTIER_METHODS),
            "all_six_prespecified_marginal_methods_present": True,
            "adaptive_parameter_updating_tta_included": False,
            "ceiling_interpretation": (
                "equal-prior randomized minimax across the two constructed "
                "finite-batch worlds; conservative witness derived because "
                "the alternative-world search is greedy"
            ),
            "not_a_realized_world_performance_bound": True,
            "swap_neighborhood_sensitivity": [
                int(value)
                for value in config["ceiling_protocol"][
                    "swap_neighbor_sensitivity"
                ]
            ],
            "swap_neighborhood_sensitivity_is_descriptive": True,
        },
        "unit_counts": {
            "natural": len(natural_records),
            "positive_control": len(positive_records),
            "total": len(natural_records) + len(positive_records),
        },
        "aggregation_order": (
            "split-seed means within condition, non-deployable posthoc oracle "
            "envelope within condition, then unweighted condition macro"
        ),
        "natural_condition_rows": natural_conditions,
        "natural_budget_macro": natural_macro,
        "positive_control_condition_rows": positive_conditions,
        "positive_control_budget_macro": positive_macro,
        "uncertainty": uncertainty,
        "input_files": {
            "natural_root": str(natural_root),
            "positive_control_root": str(positive_root),
            "natural": [
                describe_file(path, relative_to=natural_root)
                for path, _ in natural_files
            ],
            "positive_control": [
                describe_file(path, relative_to=positive_root)
                for path, _ in positive_files
            ],
        },
    }
    write_json_atomic(destination, seal_manifest(payload))
    stored = read_json(destination)
    if not isinstance(stored, Mapping):
        raise IntegrityError("published aggregate is malformed")
    verify_manifest_checksum(stored)
    return destination
