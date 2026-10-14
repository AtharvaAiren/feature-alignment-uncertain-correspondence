from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    sha256_file,
)
from marginal_alignment.campaign.feature_store import build_native_manifest
from marginal_alignment.campaign.streaming import (
    STREAMING_WINDOWS,
    WelfordMoments,
    build_streaming_plan,
    run_prequential_stream,
    streaming_scientific_gates,
    write_streaming_results,
)


def _toy_store(root: Path, *, perturb_second_block: bool = False) -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(741)
    rows, dimension, classes = 720, 5, 3
    reference = rng.normal(size=(rows, dimension))
    target = reference * np.asarray([1.6, 0.7, 1.2, 0.8, 1.4]) + np.asarray(
        [0.5, -0.3, 0.2, 0.0, 0.1]
    )
    if perturb_second_block:
        # The locked second block is rows 64:128. It must not affect the map
        # used to predict itself, but it must affect block three onward.
        target[64:128] += np.asarray([8.0, -4.0, 3.0, 0.0, 2.0])
    weight = rng.normal(size=(classes, dimension))
    bias = np.asarray([0.2, -0.1, 0.05])
    labels = (reference @ weight.T + bias).argmax(1)
    arrays = {
        "reference_features": reference.astype(np.float32),
        "target_features": target.astype(np.float32),
        "reference_ids": np.asarray([f"r{index}" for index in range(rows)]),
        "target_ids": np.asarray([f"t{index}" for index in range(rows)]),
        "reference_labels": labels.astype(np.int64),
        "target_labels": labels.astype(np.int64),
        "pair_reference_ids": np.asarray([f"r{index}" for index in range(rows)]),
        "pair_target_ids": np.asarray([f"t{index}" for index in range(rows)]),
        "head_weight": weight.astype(np.float32),
        "head_bias": bias.astype(np.float32),
        "head_fit_ids": np.asarray([f"h{index}" for index in range(32)]),
    }
    paths = {}
    for role, value in arrays.items():
        path = root / f"{role}.npy"
        np.save(path, value)
        paths[role] = path.name
    return build_native_manifest(
        root,
        dataset="stream_toy",
        encoder="stream_encoder",
        arrays=paths,
        id_provenance={
            "reference": {
                "verified": True, "kind": "fixture_id", "source": "unit stream",
                "namespace": "stream:reference",
            },
            "target": {
                "verified": True, "kind": "fixture_id", "source": "unit stream",
                "namespace": "stream:target",
            },
        },
        pairing={
            "status": "available",
            "reference_ids_role": "pair_reference_ids",
            "target_ids_role": "pair_target_ids",
            "provenance": {
                "verified": True,
                "kind": "natural_clean_corruption_identity",
                "source": "unit stream construction",
            },
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata={
            "corruption": "affine_fixture",
            "severity": 3,
            "model_provenance": {
                "encoder_id": "stream_encoder",
                "checkpoint_sha256": "0" * 64,
                "preprocessing_sha256": "1" * 64,
                "feature_layer": "fixture",
                "pooling": "fixture",
                "head_protocol": "fixture",
                "head_fit_ids_sha256": sha256_file(root / "head_fit_ids.npy"),
                "head_fit_ids_role": "head_fit_ids",
                "head_fit_id_namespace": "stream:head_fit",
                "reference_id_namespace": "stream:reference",
                "reference_split": "test",
                "head_fit_split": "train",
            },
        },
    )


def _plan(manifest: Path, destination: Path | None = None):
    return build_streaming_plan(
        manifest,
        target_order_ids=[f"t{index}" for index in range(320)],
        reference_calibration_ids=[f"r{index}" for index in range(400, 656)],
        order_provenance={
            "verified": True,
            "kind": "prespecified_dataset_order",
            "source": "unit-test lock",
        },
        block_size=64,
        metric_seed=99,
        metric_projections=4,
        mmd_features=8,
        distribution_limit=32,
        destination=destination,
    )


def test_welford_batch_updates_match_population_moments():
    rng = np.random.default_rng(18)
    values = rng.normal(size=(137, 7))
    state = WelfordMoments.empty(values.shape[1])
    for batch in np.array_split(values, [13, 54, 101]):
        state.update(batch)
    assert state.count == len(values)
    assert np.allclose(state.mean, values.mean(0), atol=1e-13)
    assert np.allclose(state.covariance, np.cov(values, rowvar=False, bias=True), atol=1e-13)


def test_prequential_grid_is_strict_and_persisted(tmp_path):
    manifest = _toy_store(tmp_path / "store")
    plan_path = tmp_path / "stream_plan.json"
    plan = _plan(manifest, plan_path)
    result = run_prequential_stream(manifest, plan_path)

    assert result["status"] == "PASS"
    assert result["scientific_gates"]["all_passed"] is True
    # 1 static baseline + 3 adaptive methods x (cumulative + three windows).
    assert len(result["rows"]) == 5 * 13
    assert len(result["aggregates"]) == 13
    first_adaptive = next(
        row for row in result["rows"]
        if row["block_index"] == 0 and row["strategy"] == "diagonal__cumulative"
    )
    second_cumulative = next(
        row for row in result["rows"]
        if row["block_index"] == 1 and row["strategy"] == "diagonal__cumulative"
    )
    third_window = next(
        row for row in result["rows"]
        if row["block_index"] == 2 and row["strategy"] == "diagonal__window64"
    )
    assert first_adaptive["fit_status"] == "COLD_START_IDENTITY"
    assert first_adaptive["history_count_before"] == 0
    assert second_cumulative["fit_status"] == "FITTED"
    assert second_cumulative["history_count_before"] == 64
    assert third_window["history_count_before"] == 64
    assert third_window["window_size"] == 64
    assert third_window["prediction_used_current_block"] is False
    assert third_window["current_block_history_overlap_count"] == 0
    assert {row["window_size"] for row in result["rows"]
            if row["history_mode"] == "fixed_window"} == set(STREAMING_WINDOWS)
    assert all(row["frozen_head_value_sha256"] == plan["frozen_head_value_sha256"]
               for row in result["rows"])
    assert all(row["aligned_covariance_relative_error"] is not None
               and row["reference_prediction_agreement"] is not None
               for row in result["rows"])

    outputs = write_streaming_results(tmp_path / "streaming", result)
    assert all(path.is_file() for path in outputs)
    persisted = read_json(tmp_path / "streaming/manifest.json")
    assert persisted["status"] == "PASS"
    assert persisted["record_count"] == len(result["rows"])
    assert persisted["rows_sha256"] == sha256_file(tmp_path / "streaming/rows.jsonl")
    assert persisted["summary_sha256"] == sha256_file(tmp_path / "streaming/summary.json")


def test_current_block_perturbation_changes_only_future_fitted_state(tmp_path):
    first_manifest = _toy_store(tmp_path / "first", perturb_second_block=False)
    second_manifest = _toy_store(tmp_path / "second", perturb_second_block=True)
    first = run_prequential_stream(first_manifest, _plan(first_manifest))
    second = run_prequential_stream(second_manifest, _plan(second_manifest))

    def row(result, block):
        return next(
            value for value in result["rows"]
            if value["block_index"] == block
            and value["strategy"] == "diagonal__cumulative"
        )

    # Block one is zero-indexed: its map sees only block zero, which is
    # identical across stores. The perturbed block enters state afterward.
    assert row(first, 1)["source_state_sha256_before"] == row(
        second, 1
    )["source_state_sha256_before"]
    assert row(first, 1)["map_sha256"] == row(second, 1)["map_sha256"]
    assert row(first, 1)["aligned_mean_l2"] != row(second, 1)["aligned_mean_l2"]
    assert row(first, 2)["source_state_sha256_before"] != row(
        second, 2
    )["source_state_sha256_before"]
    assert row(first, 2)["map_sha256"] != row(second, 2)["map_sha256"]


def test_plan_and_scientific_gates_fail_closed(tmp_path):
    manifest = _toy_store(tmp_path / "store")
    with pytest.raises(IntegrityError, match="provenance"):
        build_streaming_plan(
            manifest,
            target_order_ids=[f"t{index}" for index in range(128)],
            reference_calibration_ids=[f"r{index}" for index in range(400, 500)],
            order_provenance={"verified": False, "kind": "fixture", "source": "test"},
            block_size=64,
        )
    with pytest.raises(IntegrityError, match="overlaps clean evaluation"):
        build_streaming_plan(
            manifest,
            target_order_ids=[f"t{index}" for index in range(128)],
            reference_calibration_ids=[f"r{index}" for index in range(100)],
            order_provenance={
                "verified": True, "kind": "fixture", "source": "test",
            },
            block_size=64,
        )

    plan = _plan(manifest)
    result = run_prequential_stream(manifest, plan)
    tampered = copy.deepcopy(result["rows"])
    target = next(row for row in tampered if row["block_index"] == 1
                  and row["strategy"] == "mean__window64")
    target["history_count_before"] += 1
    gates = streaming_scientific_gates(tampered, plan)
    assert gates["history_is_exact_strict_prefix"] is False
    assert gates["all_passed"] is False

    tampered = copy.deepcopy(result["rows"])
    target = next(row for row in tampered if row["strategy"] == "mean__window64")
    target["window_size"] = 256
    gates = streaming_scientific_gates(tampered, plan)
    assert gates["strategy_metadata_matches_lock"] is False
    assert gates["all_passed"] is False
