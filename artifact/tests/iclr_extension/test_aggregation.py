from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    canonical_json_bytes,
    seal_manifest,
    sha256_file,
    sha256_json,
)
from marginal_alignment.iclr_extension.aggregation import (
    _conditional_spearman_summary,
    aggregate_play3_panel,
)
from marginal_alignment.iclr_extension.frontier import MARGINAL_FRONTIER_METHODS
from marginal_alignment.iclr_extension.natural_shift import (
    NATURAL_SHIFT_RESULT_SCHEMA,
)


WITNESS_INTERPRETATION = (
    "equal-prior randomized-minimax frozen-reference-prediction recovery ceiling "
    "across the two constructed cost-constrained finite-batch correspondence "
    "worlds; conservative witness derived because the alternative-world search "
    "is greedy"
)


def test_secondary_spearman_reports_all_undefined_resamples_without_abort() -> None:
    summary = _conditional_spearman_summary(
        [],
        point={
            "defined": False,
            "value": None,
            "cells": 6,
            "reason": "constant_rank_vector",
        },
        attempted_replicates=10_000,
        undefined_reasons={"constant_rank_vector": 10_000},
    )
    assert summary["point_estimate"] is None
    assert summary["lower"] is None and summary["upper"] is None
    assert summary["defined_replicates"] == 0
    assert summary["undefined_replicates"] == 10_000
    assert summary["defined_fraction"] == 0.0
    assert summary["undefined_replicates_explicitly_reported"] is True


def _protocol(config: dict[str, Any], budget: float) -> dict[str, Any]:
    value = config["ceiling_protocol"]
    return {
        "cost_geometry": value["cost_geometry"],
        "cost_normalization": value["cost_normalization"],
        "discarded_feature_dimensions": value["discarded_feature_dimensions"],
        "cost_budget_relative": budget,
        "covariance_estimator": value["covariance_estimator"],
        "covariance_relative_floor": value["covariance_relative_floor"],
        "maximum_witness_rows": value["maximum_witness_rows"],
        "swap_neighbors": value["swap_neighbors"],
        "swap_neighbor_sensitivity": value["swap_neighbor_sensitivity"],
        "assignment_tie_relative_tolerance": value[
            "assignment_tie_relative_tolerance"
        ],
    }


def _write_sealed(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(seal_manifest(payload)))


def _sha(value: Any) -> str:
    return sha256_json(value)


def _witness_diagnostics(
    *,
    config: dict[str, Any],
    budget: float,
    identity: str,
    neighbors: int | None = None,
) -> dict[str, Any]:
    fixed = f"fixed:{identity}"
    search_neighbors = int(
        config["ceiling_protocol"]["swap_neighbors"]
        if neighbors is None else neighbors
    )
    return {
        "cost_budget_relative": budget,
        "cost_geometry": config["ceiling_protocol"]["cost_geometry"],
        "feature_dimension": 4,
        "cost_dimension": 4,
        "discarded_feature_dimensions": 0,
        "ceiling_tightness": "conservative_witness_not_global_assignment_optimum",
        "swap_search_optimality_claimed": False,
        "quantity_interpretation": (
            "equal_prior_randomized_minimax_across_two_constructed_finite_batch_worlds"
        ),
        "not_a_realized_world_performance_bound": True,
        "target_content_sha256": _sha(f"target:{fixed}"),
        "reference_content_sha256": _sha(f"reference:{fixed}"),
        "assignment_world_zero_sha256": _sha(f"assignment0:{fixed}"),
        "assignment_world_one_sha256": _sha(
            f"assignment1:{fixed}:{budget}:{search_neighbors}"
        ),
        "prediction_world_zero_sha256": _sha(f"prediction0:{fixed}"),
        "prediction_world_one_sha256": _sha(
            f"prediction1:{fixed}:{budget}:{search_neighbors}"
        ),
        "swap_neighbors": search_neighbors,
        "swap_search": (
            "deterministic_greedy_all_pairs"
            if search_neighbors >= int(config["ceiling_protocol"]["witness_rows"])
            else "deterministic_greedy_sparse_neighborhood"
        ),
        "assignment": {"tie_break": "pinned_scipy_on_content_canonical_order"},
        "calibration": {
            "cost_geometry": config["ceiling_protocol"]["cost_geometry"],
            "feature_dimension": 4,
            "cost_dimension": 4,
            "discarded_feature_dimensions": 0,
            "distance_normalization": config["ceiling_protocol"][
                "cost_normalization"
            ],
            "reference_centering": "calibration_reference_mean",
            "target_calibration_content_sha256": _sha(f"target-cal:{fixed}"),
            "reference_calibration_content_sha256": _sha(f"reference-cal:{fixed}"),
            "bures_matrix_sha256": _sha(f"bures-matrix:{fixed}"),
            "bures_offset_sha256": _sha(f"bures-offset:{fixed}"),
            "reference_mean_sha256": _sha(f"reference-mean:{fixed}"),
            "reference_inverse_sqrt_sha256": _sha(f"whitener:{fixed}"),
            "full_geometry_sha256": _sha(f"full-geometry:{fixed}"),
        },
    }


def _swap_sensitivity(
    *, config: dict[str, Any], budget: float, identity: str
) -> list[dict[str, Any]]:
    result = []
    for index, neighbors in enumerate(
        config["ceiling_protocol"]["swap_neighbor_sensitivity"]
    ):
        disagreement = 0.03 + 0.005 * index
        result.append({
            "swap_neighbors": int(neighbors),
            "ceiling_witness": {
                "interpretation": WITNESS_INTERPRETATION,
                "not_an_accuracy_ceiling": True,
                "not_a_realized_world_performance_bound": True,
                "disagreement": disagreement,
                "prediction_recovery_ceiling": 1.0 - disagreement / 2.0,
                "baseline_cost": 10.0,
                "alternative_cost": 10.0 * (1.0 + 0.4 * budget),
                "allowed_cost": 10.0 * (1.0 + budget),
                "diagnostics": _witness_diagnostics(
                    config=config,
                    budget=budget,
                    identity=identity,
                    neighbors=int(neighbors),
                ),
            },
        })
    return result


def _allocate_count(total: int, capacities: list[int]) -> list[int]:
    result = [0 for _ in capacities]
    remaining = int(total)
    while remaining:
        progressed = False
        for index, capacity in enumerate(capacities):
            if remaining and result[index] < capacity:
                result[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("count exceeds cluster capacity")
    return result


def _natural_cluster_statistics(
    *, rows: int, disagreement_count: int, method_counts: dict[str, int]
) -> list[dict[str, Any]]:
    clusters = 100
    base, remainder = divmod(rows, clusters)
    capacities = [base + (index < remainder) for index in range(clusters)]
    disagreement = _allocate_count(disagreement_count, capacities)
    by_method = {
        method: _allocate_count(count, capacities)
        for method, count in method_counts.items()
    }
    return [
        {
            "class_index": index,
            "rows": capacity,
            "prediction_disagreement_count": disagreement[index],
            "method_correct_counts": {
                method: values[index] for method, values in by_method.items()
            },
        }
        for index, capacity in enumerate(capacities)
    ]


def _positive_cluster_statistics(
    *, rows: int, disagreement_count: int, method_counts: dict[str, int]
) -> list[dict[str, Any]]:
    records = _natural_cluster_statistics(
        rows=rows,
        disagreement_count=disagreement_count,
        method_counts=method_counts,
    )
    for record in records:
        record["method_agreement_counts"] = record.pop("method_correct_counts")
    return records


def _build_panel(root: Path, *, failure: str | None = None) -> tuple[Path, Path, Path]:
    repository = Path(__file__).resolve().parents[2]
    config_path = repository / "configs/iclr_extension/play3_natural.json"
    config = json.loads(config_path.read_text())
    lock = json.loads((repository / config["lock_path"]).read_text())
    config_sha = sha256_file(config_path)
    natural_root = root / "natural"
    positive_root = root / "positive"
    bindings = root / "bindings"
    bindings.mkdir(parents=True)
    provenance = bindings / "sketch_provenance.json"
    provenance.write_text("fixture provenance", encoding="utf-8")
    natural_stores: dict[str, Path] = {}
    positive_stores: dict[tuple[str, str], Path] = {}
    for encoder in config["encoders"]:
        store = bindings / f"natural_{encoder}.json"
        store.write_text(f"natural store {encoder}", encoding="utf-8")
        natural_stores[encoder] = store
    for corruption in config["positive_control"]["heldout_corruptions"]:
        for encoder in config["encoders"]:
            store = bindings / f"positive_{corruption}_{encoder}.json"
            store.write_text(f"positive store {corruption} {encoder}", encoding="utf-8")
            positive_stores[(corruption, encoder)] = store
    budgets = sorted({
        config["ceiling_protocol"]["primary_cost_budget_relative"],
        *config["ceiling_protocol"]["cost_budget_relative_sensitivity"],
    })
    runtime = {"fixture_runtime": "locked"}
    natural_index = 0
    for encoder_index, encoder in enumerate(config["encoders"]):
        for seed in config["ceiling_protocol"]["split_seeds"]:
            for budget in budgets:
                if failure == "missing" and natural_index == 0:
                    natural_index += 1
                    continue
                token = int(round(10_000 * budget))
                unit_id = f"imagenet_sketch__{encoder}__seed_{seed}__budget_{token:04d}bp"
                store = natural_stores[encoder]
                witness_rows = int(config["ceiling_protocol"]["witness_rows"])
                method_counts = {
                    method: 460 + 15 * index + 10 * encoder_index + (seed - 201)
                    for index, method in enumerate(MARGINAL_FRONTIER_METHODS)
                }
                method_scores = {
                    method: count / witness_rows
                    for method, count in method_counts.items()
                }
                if failure == "nonfinite" and natural_index == 0:
                    method_scores["none"] = "nan"  # type: ignore[assignment]
                best = max(
                    MARGINAL_FRONTIER_METHODS,
                    key=lambda method: (
                        float(method_scores[method]),
                        -MARGINAL_FRONTIER_METHODS.index(method),
                    ),
                )
                unit_runtime = (
                    {"fixture_runtime": "drift"}
                    if failure == "runtime_drift" and natural_index == 0
                    else runtime
                )
                disagreement_count = 205 + 10 * encoder_index + (seed - 201)
                disagreement = disagreement_count / witness_rows
                identity = f"natural:{encoder}:{seed}"
                diagnostics = _witness_diagnostics(
                    config=config, budget=budget, identity=identity
                )
                cluster_statistics = _natural_cluster_statistics(
                    rows=witness_rows,
                    disagreement_count=disagreement_count,
                    method_counts=method_counts,
                )
                if failure == "cluster_counts" and natural_index == 0:
                    cluster_statistics[0]["method_correct_counts"]["none"] += 1
                payload = {
                    "schema_version": NATURAL_SHIFT_RESULT_SCHEMA,
                    "status": "PASS",
                    "scientific_result": True,
                    "immutable_fresh_output": True,
                    "dataset": "imagenet_sketch",
                    "encoder": encoder,
                    "split_seed": seed,
                    "primary_quantity": config["reporting"]["primary_quantity"],
                    "not_a_realized_world_performance_bound": True,
                    "marginal_frontier_fit": {
                        "methods": list(MARGINAL_FRONTIER_METHODS),
                        "labels_used": False,
                        "sample_identifiers_used": False,
                        "pairing_used": False,
                    },
                    "ceiling_witness": {
                        "interpretation": WITNESS_INTERPRETATION,
                        "not_an_accuracy_ceiling": True,
                        "not_a_realized_world_performance_bound": True,
                        "disagreement": disagreement,
                        "prediction_recovery_ceiling": 1.0 - disagreement / 2.0,
                        "baseline_cost": 10.0,
                        "alternative_cost": 10.0 * (1.0 + 0.5 * budget),
                        "allowed_cost": 10.0 * (1.0 + budget),
                        "diagnostics": diagnostics,
                    },
                    "swap_neighborhood_sensitivity": _swap_sensitivity(
                        config=config, budget=budget, identity=identity
                    ),
                    "posthoc_evaluation": {
                        "used_for_geometry_or_map_fitting": False,
                        "used_for_posthoc_oracle_envelope": True,
                        "posthoc_oracle_envelope_deployable": False,
                        "deployable_primary_reference_method": "bures_oas",
                        "target_content_sha256": diagnostics[
                            "target_content_sha256"
                        ],
                        "per_method_target_accuracy": method_scores,
                        "posthoc_oracle_envelope_method": best,
                        "posthoc_oracle_envelope_target_accuracy": method_scores[best],
                        "per_wnid_sufficient_statistics": cluster_statistics,
                    },
                    "reference_split_content_sha256": _sha(
                        f"reference-split:{identity}"
                    ),
                    "target_split_content_sha256": _sha(
                        f"target-split:{identity}:"
                        f"{budget if failure == 'budget_drift' and natural_index == 0 else 'fixed'}"
                    ),
                    "unit_plan": {
                        "status": "PASS",
                        "scientific_result": False,
                        "compute_started": False,
                        "unit_id": unit_id,
                        "dataset": "imagenet_sketch",
                        "encoder": encoder,
                        "split_seed": seed,
                        "cost_budget_relative": budget,
                        "calibration_rows": config["ceiling_protocol"]["calibration_rows"],
                        "witness_rows": config["ceiling_protocol"]["witness_rows"],
                        "store_manifest": str(store),
                        "store_manifest_sha256": sha256_file(store),
                        "config_sha256": config_sha,
                        "lock_manifest_sha256": sha256_json(lock),
                        "dataset_provenance_manifest": str(provenance),
                        "dataset_provenance_manifest_sha256": sha256_file(provenance),
                        "protocol": _protocol(config, budget),
                        "runtime": unit_runtime,
                    },
                }
                _write_sealed(natural_root / f"{unit_id}.json", payload)
                natural_index += 1

    positive_index = 0
    severity = config["positive_control"]["severity"]
    for corruption_index, corruption in enumerate(
        config["positive_control"]["heldout_corruptions"]
    ):
        for encoder_index, encoder in enumerate(config["encoders"]):
            for seed in config["ceiling_protocol"]["split_seeds"]:
                for budget in budgets:
                    token = int(round(10_000 * budget))
                    unit_id = (
                        f"positive__{corruption}__{encoder}__seed_{seed}__"
                        f"budget_{token:04d}bp"
                    )
                    store = positive_stores[(corruption, encoder)]
                    witness_rows = int(config["ceiling_protocol"]["witness_rows"])
                    method_counts = {
                        method: (
                            500 + 12 * index + 8 * corruption_index
                            + 3 * encoder_index + (seed - 201)
                        )
                        for index, method in enumerate(MARGINAL_FRONTIER_METHODS)
                    }
                    method_scores = {
                        method: count / witness_rows
                        for method, count in method_counts.items()
                    }
                    best = max(
                        MARGINAL_FRONTIER_METHODS,
                        key=lambda method: (
                            method_scores[method],
                            -MARGINAL_FRONTIER_METHODS.index(method),
                        ),
                    )
                    disagreement_count = (
                        80 + 15 * corruption_index + 5 * encoder_index
                    )
                    disagreement = disagreement_count / witness_rows
                    identity = f"positive:{corruption}:{encoder}:{seed}"
                    diagnostics = _witness_diagnostics(
                        config=config, budget=budget, identity=identity
                    )
                    payload = {
                        "schema_version": "marginal_alignment.iclr_extension.positive_control.v2",
                        "status": "PASS",
                        "scientific_result": True,
                        "immutable_fresh_output": True,
                        "dataset": "imagenet_c",
                        "encoder": encoder,
                        "corruption": corruption,
                        "severity": severity,
                        "split_seed": seed,
                        "primary_quantity": config["reporting"]["primary_quantity"],
                        "not_a_realized_world_performance_bound": True,
                        "pairing_used_for_split_administration_before_fit": True,
                        "labels_used_for_administrative_pair_validation_before_fit": True,
                        "pairing_unavailable_to_estimator": True,
                        "labels_unavailable_to_estimator": True,
                        "true_assignment_scored_only_after_witness_freeze": True,
                        "selected_pair_split_sha256": _sha(f"pair-split:{identity}"),
                        "marginal_frontier_fit": {
                            "methods": list(MARGINAL_FRONTIER_METHODS),
                            "labels_used": False,
                            "sample_identifiers_used": False,
                            "pairing_used": False,
                        },
                        "ceiling_witness": {
                            "interpretation": WITNESS_INTERPRETATION,
                            "not_an_accuracy_ceiling": True,
                            "not_a_realized_world_performance_bound": True,
                            "disagreement": disagreement,
                            "prediction_recovery_ceiling": 1.0 - disagreement / 2.0,
                            "baseline_cost": 10.0,
                            "alternative_cost": 10.0 * (1.0 + 0.5 * budget),
                            "allowed_cost": 10.0 * (1.0 + budget),
                            "diagnostics": diagnostics,
                        },
                        "swap_neighborhood_sensitivity": _swap_sensitivity(
                            config=config, budget=budget, identity=identity
                        ),
                        "posthoc_validation": {
                            "pairing_content_sha256": _sha(
                                f"pairing:{corruption}:{encoder}"
                            ),
                            "true_assignment_cost": 9.0 + 0.01 * corruption_index,
                            "world_zero_assignment_recovery": 0.6,
                            "world_one_assignment_recovery": 0.4,
                            "per_method_reference_prediction_agreement": method_scores,
                            "deployable_primary_reference_method": "bures_oas",
                            "posthoc_oracle_envelope_deployable": False,
                            "posthoc_oracle_envelope_method": best,
                            "posthoc_oracle_envelope_reference_prediction_agreement": (
                                method_scores[best]
                            ),
                            "per_wnid_sufficient_statistics": (
                                _positive_cluster_statistics(
                                    rows=witness_rows,
                                    disagreement_count=disagreement_count,
                                    method_counts=method_counts,
                                )
                            ),
                        },
                        "unit_plan": {
                            "status": "PASS",
                            "scientific_result": False,
                            "compute_started": False,
                            "unit_id": unit_id,
                            "dataset": "imagenet_c",
                            "encoder": encoder,
                            "corruption": corruption,
                            "severity": severity,
                            "split_seed": seed,
                            "cost_budget_relative": budget,
                            "calibration_rows": config["ceiling_protocol"]["calibration_rows"],
                            "witness_rows": config["ceiling_protocol"]["witness_rows"],
                            "store_manifest": str(store),
                            "store_manifest_sha256": sha256_file(store),
                            "config_sha256": config_sha,
                            "extension_lock": lock,
                            "pairing_used_for_split_administration_before_fit": True,
                            "labels_used_for_administrative_pair_validation_before_fit": True,
                            "pairing_unavailable_to_estimator": True,
                            "labels_unavailable_to_estimator": True,
                            "labels_used_by_estimator": False,
                            "protocol": _protocol(config, budget),
                            "runtime": runtime,
                        },
                    }
                    _write_sealed(positive_root / f"{unit_id}.json", payload)
                    positive_index += 1
    if failure == "duplicate":
        first = sorted(natural_root.glob("*.json"))[0]
        (natural_root / "duplicate.json").write_bytes(first.read_bytes())
    if failure == "sensitivity_drift":
        first = sorted(natural_root.glob("*.json"))[0]
        payload = json.loads(first.read_text())
        payload.pop("manifest_sha256", None)
        payload["swap_neighborhood_sensitivity"].pop()
        _write_sealed(first, payload)
    return config_path, natural_root, positive_root


def test_complete_panel_aggregation_is_seed_nested_and_immutable(tmp_path: Path) -> None:
    config, natural, positive = _build_panel(tmp_path / "panel")
    output = tmp_path / "aggregate.json"
    aggregate_play3_panel(
        config_path=config,
        natural_units_root=natural,
        positive_units_root=positive,
        output=output,
    )
    result = json.loads(output.read_text())
    assert result["complete_panel"] is True
    assert result["unit_counts"] == {
        "natural": 45,
        "positive_control": 270,
        "total": 315,
    }
    assert result["panel_scope"]["methods"] == list(MARGINAL_FRONTIER_METHODS)
    assert len(result["natural_condition_rows"]) == 9
    assert len(result["positive_control_condition_rows"]) == 54
    assert all(len(row["split_seeds"]) == 5 for row in result[
        "positive_control_condition_rows"
    ])
    assert result["panel_scope"]["posthoc_oracle_envelope_deployable"] is False
    assert result["panel_scope"]["swap_neighborhood_sensitivity"] == [64, 256]
    assert all(
        len(row["swap_neighborhood_sensitivity"]) == 2
        for row in result["positive_control_condition_rows"]
    )
    assert result["uncertainty"]["replicates"] == 10_000
    assert result["uncertainty"]["zero_silent_drops"] is True
    assert result["uncertainty"]["positive_control"][
        "six_cluster_inference_is_weak"
    ] is True
    assert all(
        row["equal_prior_randomized_minimax_ceiling"]["dropped_replicates"] == 0
        for row in result["uncertainty"]["natural"]["budget_rows"]
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregate_play3_panel(
            config_path=config,
            natural_units_root=natural,
            positive_units_root=positive,
            output=output,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "incomplete"),
        ("duplicate", "duplicate natural unit"),
        ("nonfinite", "finite probability"),
        ("runtime_drift", "runtime identities drift"),
        ("budget_drift", "identity drifted across budgets"),
        ("cluster_counts", "do not reproduce its score"),
        ("sensitivity_drift", "sensitivity panel is incomplete"),
    ],
)
def test_aggregation_rejects_incomplete_duplicate_nonfinite_or_drifted_panels(
    tmp_path: Path, failure: str, message: str
) -> None:
    config, natural, positive = _build_panel(tmp_path / failure, failure=failure)
    with pytest.raises(IntegrityError, match=message):
        aggregate_play3_panel(
            config_path=config,
            natural_units_root=natural,
            positive_units_root=positive,
            output=tmp_path / f"{failure}.json",
        )
