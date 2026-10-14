from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

import marginal_alignment.campaign.real_runner as real_runner_module
from marginal_alignment.campaign.artifacts import IntegrityError, sha256_file
from marginal_alignment.campaign.campaign import (
    _configured_real_workers,
    _load_real_unit,
    _ordered_bounded_map,
    _split_artifact_records,
    _write_real_unit,
)
from marginal_alignment.campaign.feature_store import build_native_manifest
from marginal_alignment.campaign.real_runner import (
    _fit_labeled_er,
    expected_unit_semantic_keys,
    finalize_rows,
    run_condition,
    validate_semantic_grid,
)


def _toy_store(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8)
    rows, dimension, classes = 320, 6, 3
    reference = rng.normal(size=(rows, dimension))
    rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    target = reference @ rotation.T + 0.2
    weight = rng.normal(size=(classes, dimension))
    labels = (reference @ weight.T).argmax(1)
    arrays = {
        "reference_features": reference.astype(np.float32),
        "target_features": target.astype(np.float32),
        "reference_ids": np.arange(rows, dtype=np.int64),
        "target_ids": np.arange(1000, 1000 + rows, dtype=np.int64),
        "reference_labels": labels.astype(np.int64),
        "target_labels": labels.astype(np.int64),
        "pair_reference_ids": np.arange(rows, dtype=np.int64),
        "pair_target_ids": np.arange(1000, 1000 + rows, dtype=np.int64),
        "head_weight": weight.astype(np.float32),
        "head_bias": np.zeros(classes, dtype=np.float32),
        "head_fit_ids": np.arange(10_000, 10_032, dtype=np.int64),
    }
    paths = {}
    for role, value in arrays.items():
        path = root / f"{role}.npy"
        np.save(path, value)
        paths[role] = path.name
    head_fit_hash = sha256_file(root / "head_fit_ids.npy")
    manifest = build_native_manifest(
        root, dataset="toy", encoder="tiny", arrays=paths,
        id_provenance={
            "reference": {"verified": True, "kind": "fixture", "source": "test",
                          "namespace": "toy:clean"},
            "target": {"verified": True, "kind": "fixture", "source": "test",
                       "namespace": "toy:corrupt"},
        },
        pairing={
            "status": "available", "reference_ids_role": "pair_reference_ids",
            "target_ids_role": "pair_target_ids",
            "provenance": {"verified": True, "kind": "fixture_pair", "source": "test"},
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata={
            "corruption": "rotation", "severity": 1,
            "model_provenance": {
                "encoder_id": "tiny", "checkpoint_sha256": "0" * 64,
                "preprocessing_sha256": "1" * 64, "feature_layer": "fixture",
                "pooling": "fixture", "head_protocol": "fixture",
                "head_fit_ids_sha256": head_fit_hash,
                "head_fit_ids_role": "head_fit_ids",
                "head_fit_id_namespace": "toy:head_fit",
                "reference_id_namespace": "toy:clean",
                "reference_split": "test",
                "head_fit_split": "train",
            },
        },
    )
    return manifest


def test_real_scheduler_is_bounded_and_preserves_parent_order():
    items = [0, 1, 2, 3]
    callback_order: list[int] = []

    def deliberately_out_of_order(value: int) -> int:
        time.sleep(0.004 * (len(items) - value))
        return value * value

    sequential = _ordered_bounded_map(
        deliberately_out_of_order, items, workers=1,
    )
    parallel = _ordered_bounded_map(
        deliberately_out_of_order, items, workers=2,
        on_result=lambda index, _result: callback_order.append(index),
    )
    assert sequential == parallel == [0, 1, 4, 9]
    assert callback_order == [0, 1, 2, 3]


def test_real_worker_configuration_prevents_nested_thread_oversubscription():
    pinned = {
        "MARGINAL_REAL_WORKERS": "2",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    assert _configured_real_workers(pinned, cpu_count=8) == 2
    assert _configured_real_workers(
        {**pinned, "MARGINAL_REAL_WORKERS": ""}, cpu_count=3,
    ) == 3
    with pytest.raises(RuntimeError, match="numerical thread pool"):
        _configured_real_workers(
            {**pinned, "OPENBLAS_NUM_THREADS": "8"}, cpu_count=8,
        )
    assert _configured_real_workers(
        {**pinned, "MARGINAL_REAL_WORKERS": "8"}, cpu_count=8,
    ) == 8
    with pytest.raises(ValueError, match="safe range"):
        _configured_real_workers(
            {**pinned, "MARGINAL_REAL_WORKERS": "9"}, cpu_count=16,
        )


def test_real_condition_smoke_has_fixed_eval_and_matched_pair_controls(
    tmp_path, monkeypatch,
):
    manifest = _toy_store(tmp_path / "store")
    suite = {
        "methods": [
            "none", "diagonal", "bures_pair_endpoints",
            "paired_ridge", "paired_ridge_shuffled", "labeled_er",
        ],
        "calibration_sizes": [32, 64], "pair_budgets": [16, 64],
        "pair_noise_rates": [0.0, 0.1], "split_seeds": [0],
    }
    methods = {
        "none": {"implementation": "none", "information": "none"},
        "diagonal": {"implementation": "diagonal", "information": "coordinate_moments"},
        "bures_pair_endpoints": {
            "implementation": "bures", "information": "pair_endpoint_marginals",
            "covariance": "oas", "fit_on_pair_endpoint_sets": True,
        },
        "paired_ridge": {"implementation": "paired_ridge", "information": "true_pairs",
                         "ridge": 1e-5, "noise_sweep": True},
        "paired_ridge_shuffled": {"implementation": "paired_ridge",
                         "information": "shuffled_pairs", "pair_control": "shuffled",
                         "ridge": 1e-5},
        "labeled_er": {"implementation": "labeled_er", "information": "target_labels",
                       "buffer_size": 16, "steps": 1, "learning_rate": 1e-3},
    }
    output = tmp_path / "results"
    original_prepare = real_runner_module.prepare_evaluation_context
    preparation_calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal preparation_calls
        preparation_calls += 1
        return original_prepare(*args, **kwargs)

    def reject_uncached_evaluation(*args, **kwargs):
        raise AssertionError("run_condition rebuilt an evaluation context per method")

    monkeypatch.setattr(
        real_runner_module, "prepare_evaluation_context", counted_prepare
    )
    monkeypatch.setattr(
        real_runner_module, "evaluate_alignment", reject_uncached_evaluation
    )
    rows = run_condition(
        store_path=manifest, suite_name="smoke", suite=suite, methods=methods,
        condition={
            "dataset": "toy", "encoder": "tiny", "corruption": "rotation",
            "severity": 1, "evaluation_size": 64,
        },
        output_dir=output, global_seed=9,
    )
    # The suite includes both frozen-map methods and labeled ER.  They must all
    # share one prepared context for this one fixed split.
    assert preparation_calls == 1
    assert len(rows) == 22
    assert len({row["evaluation_sha256"] for row in rows}) == 1
    # Source-only randomized metrics use the same projections/RFF bank for
    # every method and calibration budget on the fixed evaluation set.
    assert len({row["source_sliced_w2_squared"] for row in rows}) == 1
    assert len({row["source_rff_mmd_squared"] for row in rows}) == 1
    true = [row for row in rows if row["method"] == "paired_ridge" and row["pair_budget"] == 64]
    shuffled = [row for row in rows if row["method"] == "paired_ridge_shuffled"
                and row["pair_budget"] == 64]
    assert true[0]["reference_prediction_agreement"] > shuffled[0]["reference_prediction_agreement"]
    assert shuffled[0]["diagnostics"]["pair_assignment_materialized"] is True
    assert shuffled[0]["diagnostics"]["pair_control"] == "shuffled"
    noisy = [row for row in rows if row["method"] == "paired_ridge_noise0.1"
             and row["pair_budget"] == 16]
    assert noisy[0]["pair_noise_rate"] == 0.1
    assert noisy[0]["pair_noise_rate_requested"] == 0.1
    # Exactly one mismatch cannot preserve a 16-element endpoint multiset;
    # the split plan realizes the nearest feasible two-pair corruption.
    assert noisy[0]["pair_noise_rate_actual"] == 0.125
    assert noisy[0]["diagnostics"]["requested_pair_noise_fraction"] == 0.1
    assert noisy[0]["diagnostics"]["actual_pair_noise_fraction"] == 0.125
    endpoint_rows = [
        row for row in rows if row["method"] == "bures_pair_endpoints"
        and row["pair_budget"] == 64
    ]
    assert endpoint_rows[0]["pair_control"] == "assignment_erased"
    assert endpoint_rows[0]["diagnostics"]["pair_assignment_available_to_estimator"] is False
    assert endpoint_rows[0]["diagnostics"]["endpoint_set_budget"] == 64
    expected = expected_unit_semantic_keys(
        suite_name="smoke", suite=suite, methods=methods,
        condition={
            "dataset": "toy", "encoder": "tiny", "corruption": "rotation",
            "severity": 1, "evaluation_size": 64,
        },
    )
    grid = validate_semantic_grid(rows, expected)
    assert grid["expected_count"] == 22
    with pytest.raises(IntegrityError, match="semantic grid mismatch"):
        validate_semantic_grid(rows[:-1], expected)
    with pytest.raises(IntegrityError, match="duplicates=1"):
        validate_semantic_grid([*rows, rows[0]], expected)
    outputs = finalize_rows(rows, output, expected_semantic_keys=expected)
    assert all(path.is_file() for path in outputs)
    real_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert real_manifest["semantic_grid_validation"]["status"] == "PASS"
    assert real_manifest["semantic_grid_validation"]["scope"] == "prespecified_expected"


def test_method_specific_minimum_calibration_is_part_of_exact_grid():
    suite = {
        "methods": ["none", "ranked"],
        "calibration_sizes": [64, 256, 1024],
        "split_seeds": [0],
    }
    methods = {
        "none": {"implementation": "none", "information": "none"},
        "ranked": {
            "implementation": "bures", "information": "joint_second_moments",
            "covariance": "oas", "rank": 64, "minimum_calibration_size": 1024,
        },
    }
    expected = expected_unit_semantic_keys(
        suite_name="rank_fixture", suite=suite, methods=methods,
        condition={
            "dataset": "toy", "encoder": "enc", "corruption": "fog",
            "severity": 5, "evaluation_size": 128,
        },
    )
    decoded = [json.loads(value) for value in expected]
    ranked = [value for value in decoded if value["method"] == "ranked"]
    assert len(ranked) == 1
    assert ranked[0]["calibration_size"] == 1024


def test_real_unit_envelope_resumes_only_hash_bound_complete_unit(tmp_path):
    row = {
        "suite": "tiny", "dataset": "toy", "encoder": "enc",
        "corruption": "fog", "severity": 5, "split_seed": 1,
        "calibration_size": 64, "evaluation_size": 128, "method": "none",
        "pair_budget": None, "pair_control": None,
        "pair_noise_rate_requested": None, "pair_noise_rate_actual": None,
        "diagnostics": {"optional_value": float("nan")},
    }
    expected = {
        expected_unit_semantic_keys(
            suite_name="tiny",
            suite={"methods": ["none"], "calibration_sizes": [64], "split_seeds": [1]},
            methods={"none": {"implementation": "none", "information": "none"}},
            condition={
                "dataset": "toy", "encoder": "enc", "corruption": "fog",
                "severity": 5, "evaluation_size": 128,
            },
        ).pop()
    }
    real_dir = tmp_path / "real"
    split = real_dir / "splits" / "plan.json"
    split.parent.mkdir(parents=True)
    split.write_text('{"status":"PASS"}\n', encoding="utf-8")
    split_records = _split_artifact_records([split], real_dir=real_dir)
    identity = {
        "suite": "tiny", "dataset": "toy", "encoder": "enc",
        "corruption": "fog", "severity": 5,
    }
    envelope = real_dir / "units" / "unit.json"
    _write_real_unit(
        envelope, identity=identity, input_fingerprint="a" * 64,
        rows=[row], expected_keys=expected, split_artifacts=split_records,
    )
    resumed = _load_real_unit(
        envelope, identity=identity, input_fingerprint="a" * 64,
        expected_keys=expected, real_dir=real_dir,
    )
    assert resumed is not None and np.isnan(resumed[0]["diagnostics"]["optional_value"])
    assert _load_real_unit(
        envelope, identity=identity, input_fingerprint="b" * 64,
        expected_keys=expected, real_dir=real_dir,
    ) is None
    split.write_text('{"status":"CHANGED"}\n', encoding="utf-8")
    assert _load_real_unit(
        envelope, identity=identity, input_fingerprint="a" * 64,
        expected_keys=expected, real_dir=real_dir,
    ) is None


def test_labeled_er_starts_from_frozen_head_and_replays_clean_examples():
    rng = np.random.default_rng(44)

    def endpoint(rows: int, *, target: bool) -> tuple[np.ndarray, np.ndarray]:
        signal = rng.normal(size=rows)
        nuisance = rng.normal(size=rows)
        labels = (signal > 0).astype(np.int64)
        features = (np.column_stack([nuisance, signal]) if target
                    else np.column_stack([signal, nuisance]))
        return features, labels

    target_fit, target_labels = endpoint(600, target=True)
    clean_replay, replay_labels = endpoint(600, target=False)
    signal = rng.normal(size=400)
    nuisance = rng.normal(size=400)
    eval_labels = (signal > 0).astype(np.int64)
    reference_eval = np.column_stack([signal, nuisance])
    source_eval = np.column_stack([nuisance, signal])
    frozen_weight = np.asarray([[-3.0, 0.0], [3.0, 0.0]])
    frozen_bias = np.zeros(2)

    metrics, diagnostics = _fit_labeled_er(
        source_fit=target_fit, fit_labels=target_labels,
        reference_replay_fit=clean_replay,
        reference_replay_labels=replay_labels,
        source_eval=source_eval, reference_eval=reference_eval,
        eval_labels=eval_labels, weight=frozen_weight, bias=frozen_bias,
        specification={
            "buffer_size": 512, "replay_ratio": 1.0, "steps": 12,
            "batch_size": 64, "learning_rate": 0.1, "weight_decay": 1e-4,
        },
        fit_rng=np.random.default_rng(45), metric_rng=np.random.default_rng(46),
        metrics_projections=8, mmd_features=32,
    )

    assert diagnostics["initialization"] == "exact_frozen_reference_head_weight_and_bias"
    assert diagnostics["target_buffer_rows"] == 512
    assert diagnostics["clean_reference_replay_rows"] == 512
    assert diagnostics["feature_transform_applied"] is False
    assert diagnostics["uses_reference_replay_labels"] is True
    assert diagnostics["weight_update_frobenius"] > 0.0
    assert diagnostics["final_target_buffer_nll"] < diagnostics["initial_target_buffer_nll"]
    # The replay optimum must retain the clean x-axis task while learning the
    # target y-axis task, so this deliberately tests improvement rather than
    # an impossible target-only optimum.
    assert metrics["aligned_accuracy"] > metrics["source_accuracy"] + 0.15
    # No feature transform is hidden inside the labeled comparator.
    assert metrics["aligned_mean_l2"] == metrics["source_mean_l2"]
    assert metrics["aligned_sliced_w2_squared"] == metrics["source_sliced_w2_squared"]
    assert np.isclose(
        metrics["paired_feature_mse"], np.mean(np.square(source_eval - reference_eval))
    )
    for key in (
        "reference_prediction_agreement", "aligned_accuracy",
        "negative_log_likelihood", "brier_score", "expected_calibration_error",
    ):
        assert np.isfinite(metrics[key])
    # These adapted values must not be stale identity/frozen-head metrics.
    assert metrics["negative_log_likelihood"] != metrics["source_negative_log_likelihood"]
    assert metrics["brier_score"] != metrics["source_brier_score"]
    assert metrics["expected_calibration_error"] != metrics[
        "source_expected_calibration_error"
    ]
