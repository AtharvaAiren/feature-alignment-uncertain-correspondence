from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / "configs/iclr_extension/witness_size_scaling_20260910.json"
RUNNER = REPOSITORY / "scripts/iclr_extension/run_witness_size_scaling.py"
DEPLOYMENT = REPOSITORY / "deployment/witness_size_scaling_20260910_v2"


def _load_runner():
    spec = importlib.util.spec_from_file_location("tested_witness_scaling", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_locked_grid_is_96_groups_and_288_fresh_certificates() -> None:
    config = runner._load_config(CONFIG)
    groups = runner.enumerate_groups(config)
    assert len(groups) == 96
    assert runner.WITNESS_SIZES == (1024, 4096, 8192)
    assert 16384 not in runner.WITNESS_SIZES
    assert sum(group.panel == runner.NATURAL_PANEL for group in groups) == 6
    assert sum(group.panel == runner.CONTROL_PANEL for group in groups) == 90
    assert len(groups) * len(runner.WITNESS_SIZES) == 288


def test_two_shards_keep_each_store_and_its_three_seeds_together() -> None:
    config = runner._load_config(CONFIG)
    shards = runner.shard_groups(runner.enumerate_groups(config), 2)
    assert sorted(map(len, shards)) == [48, 48]
    locations = {}
    for shard_index, shard in enumerate(shards):
        for group in shard:
            key = (group.dataset, group.corruption, group.encoder)
            locations.setdefault(key, set()).add(shard_index)
    assert all(len(value) == 1 for value in locations.values())
    with pytest.raises(ValueError, match="requires two shards"):
        runner.shard_groups(runner.enumerate_groups(config), 3)


@pytest.mark.parametrize(
    ("brackets", "expected"),
    [
        ((0.70, 0.71, 0.60, 0.61), "certified_decrease"),
        ((0.60, 0.61, 0.70, 0.71), "certified_increase"),
        ((0.60, 0.72, 0.70, 0.80), "inconclusive"),
    ],
)
def test_interval_direction_uses_worst_case_bracket_endpoints(
    brackets, expected
) -> None:
    lower, upper, direction = runner.interval_certified_direction(
        left_lower=brackets[0],
        left_upper=brackets[1],
        right_lower=brackets[2],
        right_upper=brackets[3],
    )
    assert lower == pytest.approx(brackets[2] - brackets[1])
    assert upper == pytest.approx(brackets[3] - brackets[0])
    assert direction == expected


def test_config_removes_historical_greedy_and_prespecifies_interval_rule() -> None:
    config = runner._load_config(CONFIG)
    common = config["common_protocol"]
    reporting = config["reporting"]
    assert "greedy_neighbors" not in common
    assert "dual_bisection_iterations" not in common
    assert common["maximum_dual_bisection_iterations"] == 48
    assert reporting["point_estimate_trend_claimed"] is False
    assert "otherwise_inconclusive" in reporting["trend_classification"]
    assert (
        config["expected_counts"][
            "external_completed_1024_certificates_used_as_baseline"
        ]
        == 0
    )


def test_config_capacity_tampering_is_rejected(tmp_path: Path) -> None:
    config = copy.deepcopy(runner._load_config(CONFIG))
    config["panels"][runner.CONTROL_PANEL]["mathematical_maximum_witness_rows"] = 16384
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Exception, match="protocol changed"):
        runner._load_config(path)


def test_source_inventory_includes_prior_runner_transitive_closure() -> None:
    assert (
        "src/marginal_alignment/iclr_extension/realized_profile.py"
        in runner.SOURCE_FILES
    )
    assert "src/marginal_alignment/campaign/raw_extract.py" in runner.SOURCE_FILES
    assert (
        "src/marginal_alignment/iclr_extension/witness_scaling.py"
        in runner.SOURCE_FILES
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("common_protocol", "covariance_estimator", "empirical"),
        ("common_protocol", "covariance_relative_floor", 1e-5),
        ("common_protocol", "exact_validation_rows", 63),
        ("common_protocol", "exact_milp_time_limit_seconds", 899.0),
        ("common_protocol", "exact_reported_mip_gap_tolerance", 1e-9),
        ("common_protocol", "assignment_tie_relative_tolerance", 1e-9),
        ("common_protocol", "random_cost_moment_block_rows", 128),
        ("common_protocol", "maximum_dual_bisection_iterations", 47),
        ("common_protocol", "maximum_primal_dual_gap", 0.02),
        ("common_protocol", "primal_initializer", "greedy"),
        ("resource_contract", "parallel_workers", 3),
        ("resource_contract", "maximum_concurrent_campaigns", 2),
        ("resource_contract", "conservative_peak_gib_per_worker", 10),
        ("resource_contract", "required_memavailable_gib_before_launch", 24),
    ],
)
def test_every_material_numerical_and_resource_setting_is_hard_locked(
    tmp_path: Path, section: str, field: str, replacement
) -> None:
    config = copy.deepcopy(runner._load_config(CONFIG))
    config[section][field] = replacement
    path = tmp_path / f"tampered_{section}_{field}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Exception, match="protocol changed"):
        runner._load_config(path)


def _synthetic_traced_result(
    *,
    size: int,
    baseline: float,
    feasible_cost: float,
    lower: float,
    first_disagreement: float,
    gap_target: float,
) -> dict:
    relative = 0.5
    allowed = baseline * (1.0 + relative)
    scale = baseline
    normalized_budget = allowed / scale
    epsilon = np.finfo(np.float64).eps
    records = []
    best_primal = 0.0
    best_dual = 1.0
    for scaled, cost, disagreement, feasible, assignment_hash in (
        (0.0, allowed + 1.0, first_disagreement, False, "1" * 64),
        (float(size), feasible_cost, lower, True, "2" * 64),
    ):
        multiplier = scaled / size
        normalized_cost = cost / scale
        raw_dual = disagreement + multiplier * (normalized_budget - normalized_cost)
        cushion = (
            2048.0
            * size
            * epsilon
            * max(
                1.0,
                abs(raw_dual),
                abs(multiplier) * (abs(normalized_budget) + abs(normalized_cost)),
            )
        )
        dual = min(1.0, raw_dual + cushion)
        best_dual = min(best_dual, dual)
        if feasible:
            best_primal = max(best_primal, disagreement)
        upper = min(1.0, max(best_primal, best_dual))
        records.append(
            {
                "scaled_multiplier": scaled,
                "mean_reward_multiplier": multiplier,
                "assignment_cost": cost,
                "disagreement": disagreement,
                "dual_upper_bound": dual,
                "budget_feasible": feasible,
                "assignment_sha256": assignment_hash,
                "best_gap_after_evaluation": upper - best_primal,
            }
        )
    upper = min(1.0, max(best_primal, best_dual))
    gap = upper - lower
    return {
        "status": "PASS",
        "baseline_cost": baseline,
        "allowed_cost": allowed,
        "primal_cost": feasible_cost,
        "selected_world_disagreement": lower,
        "selected_two_world_minimax_recovery": 1.0 - lower / 2.0,
        "maximum_disagreement_primal_lower_bound": lower,
        "maximum_disagreement_dual_upper_bound": upper,
        "maximum_disagreement_identification_interval": [lower, upper],
        "strongest_two_world_recovery_ceiling_interval": [
            1.0 - upper / 2.0,
            1.0 - lower / 2.0,
        ],
        "primal_dual_gap": gap,
        "primal_dual_gap_percentage_points": 100.0 * gap,
        "gap_target": gap_target,
        "gap_target_percentage_points": 100.0 * gap_target,
        "meets_gap_target": bool(gap <= gap_target),
        "assignment_world_zero_sha256": "0" * 64,
        "assignment_world_one_sha256": "2" * 64,
        "diagnostics": {
            "witness_rows": size,
            "protocol": {
                "cost_budget_relative": relative,
                "primal_dual_gap_target": gap_target,
                "dual_bisection_iterations": 48,
                "assignment_tie_relative_tolerance": 1e-12,
            },
            "optimization": {
                "primal_solver": "best_feasible_lagrangian_assignment",
                "dual_solver": "scipy_linear_sum_assignment_lagrangian_search",
                "cost_normalization_scale": scale,
                "normalized_budget": normalized_budget,
                "feasibility_tolerance": 1e-10,
                "early_stop_target": gap_target,
                "early_stop_reached": bool(gap <= gap_target),
                "maximum_bisection_iterations": 48,
                "evaluations": len(records),
                "best_dual_multiplier": 0.0,
                "trace": records,
                "objective_buffer_reused": True,
                "dense_float_reward_matrix_created": False,
                "upper_bounds_lp_relaxation_optimum": True,
                "optimized_lagrangian_dual_value_claimed": False,
                "formal_real_arithmetic_certificate_claimed": False,
                "floating_point_solver_not_formally_verified": True,
                "roundoff_upper_cushion_applied": True,
            },
        },
    }


def _valid_certificate_row() -> dict:
    result = _synthetic_traced_result(
        size=1024,
        baseline=10.0,
        feasible_cost=14.0,
        lower=0.4,
        first_disagreement=0.405,
        gap_target=0.01,
    )
    scaling = _synthetic_traced_result(
        size=64,
        baseline=4.0,
        feasible_cost=5.0,
        lower=0.2,
        first_disagreement=0.3,
        gap_target=0.0,
    )
    exact_lower = scaling["maximum_disagreement_primal_lower_bound"]
    exact_upper = scaling["maximum_disagreement_dual_upper_bound"]
    exact_optimum = 0.25
    return {
        "witness_rows": 1024,
        "shared_random_gap_q": 0.05,
        "derived_cost_budget_relative_to_minimum": 0.5,
        "random_bijection_cost": {"expected_total_cost": 110.0},
        "result": result,
        "exact_small_instance_validation": {
            "status": "PROVEN_OPTIMAL",
            "certificate_solver": "witness_scaling_no_greedy_early_stop_v1",
            "rows": 64,
            "reported_mip_gap": 0.0,
            "maximum_disagreement_exact": exact_optimum,
            "assignment_cost": 5.0,
            "allowed_cost": 6.0,
            "certificate_primal_lower_bound": exact_lower,
            "certificate_dual_upper_bound": exact_upper,
            "certificate_primal_dual_gap": exact_upper - exact_lower,
            "certificate_contains_exact_optimum": True,
            "scaling_certificate": scaling,
        },
    }


def _set_nested(value: dict, path: tuple[str, ...], replacement) -> None:
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("result", "primal_dual_gap"), 0.006),
        (("result", "maximum_disagreement_identification_interval"), [0.3, 0.405]),
        (("result", "strongest_two_world_recovery_ceiling_interval"), [0.0, 1.0]),
        (("result", "selected_two_world_minimax_recovery"), 0.7),
        (("result", "allowed_cost"), 14.9),
        (("result", "primal_cost"), 15.1),
        (("result", "baseline_cost"), -10.0),
        (("exact_small_instance_validation", "reported_mip_gap"), -1e-15),
        (("exact_small_instance_validation", "certificate_primal_dual_gap"), 0.11),
        (("exact_small_instance_validation", "maximum_disagreement_exact"), 0.31),
        (("exact_small_instance_validation", "assignment_cost"), 6.1),
    ],
)
def test_resealed_numerically_tampered_certificate_is_rejected(
    path: tuple[str, ...], replacement
) -> None:
    row = _valid_certificate_row()
    runner._validate_certificate_numbers(row, size=1024)
    _set_nested(row, path, replacement)
    # A dishonest producer can recompute the outer checksum.  Re-derivation,
    # not checksum validity, must reject the altered scientific semantics.
    resealed = runner.seal_manifest({"row": row})
    runner.verify_manifest_checksum(resealed)
    with pytest.raises(Exception, match="numerical identities"):
        runner._validate_certificate_numbers(resealed["row"], size=1024)


def test_coordinated_upper_gap_interval_reseal_is_rejected_by_trace() -> None:
    row = _valid_certificate_row()
    result = row["result"]
    changed_upper = result["maximum_disagreement_dual_upper_bound"] + 0.001
    changed_gap = changed_upper - result["maximum_disagreement_primal_lower_bound"]
    result["maximum_disagreement_dual_upper_bound"] = changed_upper
    result["primal_dual_gap"] = changed_gap
    result["primal_dual_gap_percentage_points"] = 100.0 * changed_gap
    result["maximum_disagreement_identification_interval"][1] = changed_upper
    result["strongest_two_world_recovery_ceiling_interval"][0] = (
        1.0 - changed_upper / 2.0
    )
    resealed = runner.seal_manifest({"row": row})
    runner.verify_manifest_checksum(resealed)
    with pytest.raises(Exception, match="trace"):
        runner._validate_certificate_numbers(resealed["row"], size=1024)


def test_fabricated_nonselected_trace_assignment_is_rejected_by_frozen_rerun() -> None:
    row = _valid_certificate_row()
    authentic = copy.deepcopy(row["result"])
    row["result"]["diagnostics"]["optimization"]["trace"][0]["assignment_sha256"] = (
        "f" * 64
    )
    # Trace algebra alone cannot authenticate an unselected LAP assignment;
    # final verification therefore compares the entire trace with a fresh
    # solve from the locked endpoints.
    runner._validate_certificate_numbers(row, size=1024)
    independently_recomputed = SimpleNamespace(summary=lambda: authentic)
    with pytest.raises(Exception, match="LAP trace"):
        runner._validate_recomputed_certificate(
            row["result"], independently_recomputed, context="full scaling"
        )


def _exact_array_fixture() -> tuple[dict, dict[str, np.ndarray], str]:
    size = 64
    key = "n_01024"
    row = np.arange(size, dtype=np.int64)
    shifted = np.roll(row, -1)
    cost = np.ones((size, size), dtype=np.float64)
    cost[row, row] = 0.1
    reference_prediction = row.copy()
    prediction_zero = reference_prediction.copy()
    prediction_one = reference_prediction[shifted]
    arrays = {
        f"{key}__exact_subset_assignment": shifted.copy(),
        f"{key}__exact_subset_cost": cost,
        f"{key}__exact_subset_reference_prediction": reference_prediction,
        f"{key}__exact_subset_prediction_world_zero": prediction_zero,
        f"{key}__exact_subset_scaling_assignment_world_zero": row.copy(),
        f"{key}__exact_subset_scaling_assignment_world_one": shifted.copy(),
        f"{key}__exact_subset_scaling_prediction_world_zero": prediction_zero.copy(),
        f"{key}__exact_subset_scaling_prediction_world_one": prediction_one,
        f"{key}__exact_subset_scaling_target_canonical_order": row.copy(),
        f"{key}__exact_subset_scaling_reference_canonical_order": row.copy(),
    }
    scaling = {
        "baseline_cost": float(cost[row, row].sum()),
        "primal_cost": float(cost[row, shifted].sum()),
        "allowed_cost": float(cost[row, shifted].sum()),
        "selected_world_disagreement": 1.0,
        "assignment_world_zero_sha256": runner._array_sha256(row),
        "assignment_world_one_sha256": runner._array_sha256(shifted),
        "prediction_world_zero_sha256": runner._array_sha256(prediction_zero),
        "prediction_world_one_sha256": runner._array_sha256(prediction_one),
        "target_canonical_order_sha256": runner._array_sha256(row),
        "reference_canonical_order_sha256": runner._array_sha256(row),
        "diagnostics": {
            "cost_matrix_sha256": runner._array_sha256(cost),
            "reference_prediction_sha256": runner._array_sha256(reference_prediction),
        },
    }
    exact = {
        "allowed_cost": scaling["allowed_cost"],
        "assignment_cost": scaling["primal_cost"],
        "maximum_disagreement_exact": 1.0,
        "cost_matrix_sha256": runner._array_sha256(cost),
        "reference_prediction_sha256": runner._array_sha256(reference_prediction),
        "prediction_world_zero_sha256": runner._array_sha256(prediction_zero),
        "assignment_sha256": runner._array_sha256(shifted),
        "scaling_certificate": scaling,
    }
    return exact, arrays, key


def test_coordinated_nonoptimal_exact_assignment_reseal_is_rejected() -> None:
    exact, arrays, key = _exact_array_fixture()
    runner._validate_exact_diagnostic_arrays(
        exact, arrays, key=key, rerun_exact_solver=True
    )
    identity = np.arange(64, dtype=np.int64)
    arrays[f"{key}__exact_subset_assignment"] = identity
    exact["assignment_sha256"] = runner._array_sha256(identity)
    exact["assignment_cost"] = 6.4
    exact["maximum_disagreement_exact"] = 0.0
    resealed = runner.seal_manifest({"exact": exact})
    runner.verify_manifest_checksum(resealed)
    with pytest.raises(Exception, match="independently proven optimal"):
        runner._validate_exact_diagnostic_arrays(
            resealed["exact"], arrays, key=key, rerun_exact_solver=True
        )


def _fake_group_manifests() -> tuple[list, dict]:
    specs = runner.enumerate_groups(runner._load_config(CONFIG))
    manifests = {}
    for index, spec in enumerate(specs):
        certificates = []
        for size in runner.WITNESS_SIZES:
            lower = 0.2 + index / 10000.0 + size / 1e7
            upper = lower + 0.005
            certificates.append(
                {
                    "witness_rows": size,
                    "result": {
                        "maximum_disagreement_primal_lower_bound": lower,
                        "maximum_disagreement_dual_upper_bound": upper,
                        "primal_dual_gap": upper - lower,
                    },
                    "class_coverage": {
                        "target": {"unique_classes": min(size, 1000)},
                        "reference": {"unique_classes": min(size, 1000)},
                        "cross_endpoint": {
                            "common_classes": min(size, 1000),
                            "target_fraction_with_at_least_one_reference_same_class_partner": 0.5,
                            "same_class_candidate_pair_fraction": 0.001,
                            "maximum_class_preserving_matching_fraction": 0.4,
                        },
                    },
                }
            )
        manifests[spec.group_id] = {"scientific": {"certificates": certificates}}
    return specs, manifests


@pytest.mark.parametrize(
    "filename",
    [
        "WITNESS_SIZE_ROWS.jsonl",
        "WITNESS_SIZE_DELTAS.jsonl",
        "WITNESS_SIZE_SUMMARIES.jsonl",
    ],
)
def test_resealed_aggregate_table_content_is_rejected(filename: str) -> None:
    specs, manifests = _fake_group_manifests()
    rows, deltas, summaries = runner._derive_aggregate_tables(specs, manifests)
    expected = {
        "WITNESS_SIZE_ROWS.jsonl": rows,
        "WITNESS_SIZE_DELTAS.jsonl": deltas,
        "WITNESS_SIZE_SUMMARIES.jsonl": summaries,
    }
    observed = copy.deepcopy(expected)
    observed[filename][0][next(iter(observed[filename][0]))] = "coordinated-reseal"
    with pytest.raises(Exception, match="authenticated group manifests"):
        runner._validate_aggregate_table_contents(observed, expected)


def test_frozen_external_inputs_are_reread_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r5 = tmp_path / "r5"
    historical = tmp_path / "historical"
    r5.mkdir()
    historical.mkdir()
    r5_aggregate = r5 / "PLAY3_OPTIMIZATION_AGGREGATE.json"
    historical_aggregate = historical / "REVIEWER_FOLLOWUP_AGGREGATE.json"
    historical_done = historical / "DONE.json"
    runner.write_json_atomic(
        r5_aggregate,
        runner.seal_manifest({"status": "PASS", "all_required_gates_pass": True}),
    )
    runner.write_json_atomic(
        historical_aggregate, runner.seal_manifest({"status": "PASS"})
    )
    frozen_historical = runner.read_json(historical_aggregate)
    runner.write_json_atomic(
        historical_done,
        runner.seal_manifest(
            {
                "schema_version": "marginal_alignment.reviewer_followup_done.v1",
                "status": "PASS",
                "aggregate_file_sha256": runner.sha256_file(historical_aggregate),
                "aggregate_manifest_sha256": frozen_historical["manifest_sha256"],
            }
        ),
    )
    external = tmp_path / "external.json"
    class_index = tmp_path / "class_index.json"
    external.write_text("{}\n", encoding="utf-8")
    class_index.write_text("{}\n", encoding="utf-8")
    completed = runner._validate_completed_inputs(r5, historical)
    lock = {
        **completed,
        "external_input_manifest": {
            "path": str(external),
            "file_sha256": runner.sha256_file(external),
        },
        "class_index": {
            "path": str(class_index),
            "file_sha256": runner.sha256_file(class_index),
            "mapping_sha256": "mapping-test",
        },
    }
    monkeypatch.setattr(
        runner.prior,
        "_external_imagenet_c_hashes",
        lambda _: {("brightness", "clip"): "store-test"},
    )
    monkeypatch.setattr(
        runner,
        "load_official_class_index",
        lambda _: SimpleNamespace(
            file_sha256=runner.sha256_file(class_index),
            canonical_mapping_sha256="mapping-test",
        ),
    )
    assert runner._reauthenticate_frozen_inputs(lock) == {
        ("brightness", "clip"): "store-test"
    }
    external.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(Exception, match="external input manifest changed"):
        runner._reauthenticate_frozen_inputs(lock)

