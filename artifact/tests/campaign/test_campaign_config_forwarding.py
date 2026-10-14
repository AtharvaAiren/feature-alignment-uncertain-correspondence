from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import marginal_alignment.campaign.campaign as campaign
from marginal_alignment.campaign.artifacts import IntegrityError, sha256_file, write_json_atomic
from marginal_alignment.campaign.stages import StageContext


def _context(tmp_path: Path, synthetic: dict) -> StageContext:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    return StageContext(
        project_root=tmp_path,
        run_dir=run_dir,
        config={
            "campaign_id": "forwarding_fixture",
            "global_seed": 700,
            "synthetic": synthetic,
        },
        config_hash="c" * 64,
        source_hash="s" * 64,
    )


def test_lock_status_accepts_only_original_or_documented_pre_result_amendment():
    assert campaign._validate_lock_status({
        "status": "PRESPECIFIED_BEFORE_CAMPAIGN_EXECUTION",
    }) == "PRESPECIFIED_BEFORE_CAMPAIGN_EXECUTION"
    amended = {
        "status": (
            "AMENDED_BEFORE_SCIENTIFIC_CAMPAIGN_EXECUTION_"
            "AFTER_READOUT_SOLVER_FAILURE"
        ),
        "date_utc": "2026-08-22",
        "amendment": (
            "No scientific campaign result existed when the failed SAGA solver was "
            "replaced by deterministic L-BFGS for every frozen clean readout."
        ),
    }
    assert campaign._validate_lock_status(amended) == amended["status"]
    numerical = {
        "status": "AMENDED_AFTER_NUMERICAL_FAILURES_BEFORE_OUTCOME_INSPECTION",
        "date_utc": "2026-08-22",
        "amendment": (
            "Only numerical failure traces were inspected. The Bures evaluation was "
            "made algebraically stable and a nonunique extra rank diagnostic removed."
        ),
    }
    assert campaign._validate_lock_status(numerical) == numerical["status"]
    with pytest.raises(IntegrityError, match="substantive amendment"):
        campaign._validate_lock_status({
            "status": amended["status"], "date_utc": "2026-08-22", "amendment": "short",
        })
    with pytest.raises(IntegrityError, match="invalid prespecification status"):
        campaign._validate_lock_status({"status": "PASS"})


def test_locked_correspondence_grid_avoids_nonidentified_orthogonal_completions():
    project = Path(__file__).resolve().parents[2]
    config = json.loads(
        (project / "configs/campaign/campaign_full.json").read_text(encoding="utf-8")
    )
    dose = config["suites"]["correspondence_dose"]
    confirmatory = config["suites"]["correspondence_confirmatory"]
    assert dose["methods"] == [
        "bures_pair_endpoints", "paired_ridge", "paired_ridge_shuffled",
        "paired_reduced_rank", "paired_reduced_rank_shuffled", "labeled_er",
    ]
    assert confirmatory["pair_budgets"] == [4096]
    assert confirmatory["methods"] == [
        "bures_pair_endpoints", "paired_reduced_rank",
        "paired_reduced_rank_shuffled",
    ]
    assert config["methods"]["paired_procrustes"][
        "require_full_centered_rank"
    ] is True
    assert config["methods"]["paired_procrustes_shuffled"][
        "require_full_centered_rank"
    ] is True
    assert all(
        "paired_readout_gauge" not in suite["methods"]
        for suite in config["suites"].values()
    )
    assert all(
        "paired_procrustes" not in suite["methods"]
        and "paired_procrustes_shuffled" not in suite["methods"]
        for suite in config["suites"].values()
    )
    correspondence = config["inference"]["primary_contrasts"][-2:]
    assert [(row["left"], row["right"]) for row in correspondence] == [
        ("paired_reduced_rank", "paired_reduced_rank_shuffled"),
        ("paired_reduced_rank", "bures_pair_endpoints"),
    ]


def test_synthetic_stage_forwards_every_nested_locked_axis(tmp_path, monkeypatch):
    synthetic = {
        "exact_indistinguishable": {
            "seeds": [31], "samples_per_seed": 1001,
            "readout_angles_degrees": [17], "orbit_sizes": [3],
        },
        "approximate_two_world": {
            "seeds": [32], "calibration_sizes": [7], "epsilons": [0.04],
            "task_disagreements": [0.3], "trials_per_seed": 101,
        },
        "identifiable_nongaussian": {
            "regime": "standardized_asymmetric_mixture_swap", "seeds": [33],
            "calibration_sizes": [9], "asymmetries": [1.7], "evaluation_size": 111,
        },
    }
    context = _context(tmp_path, synthetic)
    observed = {}

    def fake_suite(output_dir, **kwargs):
        observed["output_dir"] = output_dir
        observed.update(kwargs)
        output = Path(output_dir)
        output.mkdir(parents=True)
        write_json_atomic(output / "manifest.json", {"status": "PASS"})
        return {"summary": {"status": "PASS"}}

    monkeypatch.setattr(campaign, "run_synthetic_ceiling_suite", fake_suite)
    outputs = list(campaign._stage_synthetic(context))
    assert observed == {
        "output_dir": context.run_dir / "synthetic",
        "exact_kwargs": synthetic["exact_indistinguishable"],
        "approximate_kwargs": synthetic["approximate_two_world"],
        "identifiable_kwargs": synthetic["identifiable_nongaussian"],
    }
    assert outputs == [context.run_dir / "synthetic/manifest.json"]


class _Reference:
    features = np.ones((8, 3), dtype=np.float32)


class _Store:
    manifest = {"head": {"weight_role": "head_weight", "bias_role": "head_bias"}}

    def endpoint(self, name):
        assert name == "reference"
        return _Reference()

    def array(self, role):
        if role == "head_weight":
            return np.ones((2, 3), dtype=np.float32)
        assert role == "head_bias"
        return np.asarray([0.25, -0.5], dtype=np.float32)


def test_controlled_stage_forwards_axes_and_real_store_identity(tmp_path, monkeypatch):
    synthetic = {
        "pair_rank_phase": {
            "seeds": [41], "dimensions": [5], "classes": [3],
            "pair_ratios": [0.75], "threshold_offsets": [-2, 2],
            "pair_noise": [0.2], "evaluation_size": 123,
        },
        "real_embedding_orbit": {"rank": 2, "worlds": 3, "evaluation_rows": 7},
    }
    context = _context(tmp_path, synthetic)
    first_manifest = tmp_path / "first_store.json"
    second_manifest = tmp_path / "second_store.json"
    first_manifest.write_text("first\n", encoding="utf-8")
    second_manifest.write_text("second\n", encoding="utf-8")
    write_json_atomic(context.run_dir / "feature_store_index.json", {
        "records": [
            {"dataset": "a", "encoder": "e1", "manifest": str(first_manifest),
             "manifest_sha256": sha256_file(first_manifest)},
            # A repeated condition for a/e1 must not create a second orbit row.
            {"dataset": "a", "encoder": "e1", "manifest": str(first_manifest),
             "manifest_sha256": sha256_file(first_manifest)},
            {"dataset": "b", "encoder": "e2", "manifest": str(second_manifest),
             "manifest_sha256": sha256_file(second_manifest)},
        ]
    })
    observed = {"orbits": []}

    def fake_pair(**kwargs):
        observed["pair"] = kwargs
        return [{"fixture": "pair"}]

    def fake_orbit(features, weight, **kwargs):
        observed["orbits"].append({
            "features_shape": tuple(features.shape), "weight_shape": tuple(weight.shape),
            "bias": kwargs.pop("bias").tolist(),
            **kwargs,
        })
        return {"fixture": "orbit", **kwargs}

    def fake_write(output_dir, pair_rows, orbit_records):
        observed["written_pair"] = pair_rows
        observed["written_orbits"] = orbit_records
        output = Path(output_dir)
        output.mkdir(parents=True)
        manifest = output / "manifest.json"
        write_json_atomic(manifest, {"status": "PASS", "scientific_gates": {"all_passed": True}})
        return [manifest]

    monkeypatch.setattr(campaign, "run_pair_rank_phase_diagram", fake_pair)
    monkeypatch.setattr(campaign, "load_feature_store", lambda *_args, **_kwargs: _Store())
    monkeypatch.setattr(campaign, "run_real_embedding_orbit", fake_orbit)
    monkeypatch.setattr(campaign, "write_controlled_results", fake_write)

    outputs = list(campaign._stage_controlled(context))
    assert observed["pair"] == synthetic["pair_rank_phase"]
    assert observed["written_pair"] == [{"fixture": "pair"}]
    assert outputs == [context.run_dir / "controlled/manifest.json"]
    assert observed["orbits"] == [
        {
            "features_shape": (8, 3), "weight_shape": (2, 3),
            "bias": [0.25, -0.5],
            "seed": 701, "dataset": "a", "encoder": "e1",
            "input_manifest_sha256": sha256_file(first_manifest),
            **synthetic["real_embedding_orbit"],
        },
        {
            "features_shape": (8, 3), "weight_shape": (2, 3),
            "bias": [0.25, -0.5],
            "seed": 702, "dataset": "b", "encoder": "e2",
            "input_manifest_sha256": sha256_file(second_manifest),
            **synthetic["real_embedding_orbit"],
        },
    ]


def test_controlled_stage_fails_closed_on_scientific_gate_failure(tmp_path, monkeypatch):
    context = _context(tmp_path, {
        "pair_rank_phase": {},
        "real_embedding_orbit": {},
    })
    write_json_atomic(context.run_dir / "feature_store_index.json", {"records": []})
    monkeypatch.setattr(campaign, "run_pair_rank_phase_diagram", lambda **_kwargs: [])

    def fake_write(output_dir, _pair_rows, _orbit_records):
        output = Path(output_dir)
        output.mkdir(parents=True)
        manifest = output / "manifest.json"
        write_json_atomic(manifest, {
            "status": "FAIL", "scientific_gates": {"all_passed": False}
        })
        return [manifest]

    monkeypatch.setattr(campaign, "write_controlled_results", fake_write)
    with pytest.raises(RuntimeError, match="controlled scientific gates failed"):
        campaign._stage_controlled(context)


def _store_validation_context(tmp_path: Path) -> StageContext:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "experiment_lock.json", {
        "status": "PRESPECIFIED_BEFORE_CAMPAIGN_EXECUTION",
        "feature_build_config_canonical_sha256": "f" * 64,
    })
    return StageContext(
        project_root=tmp_path,
        run_dir=run_dir,
        config={
            "datasets": [{
                "name": "toy_c", "encoders": ["enc"],
                "corruptions": ["fog", "snow"], "severities": [5],
                "evaluation_size": 128,
                "reference_split": "test", "head_fit_split": "train",
                "head_protocol": {"kind": "fixture"},
            }],
            "suites": {
                "primary": {
                    "datasets": ["toy_c"], "condition_filter": "all",
                },
            },
        },
        config_hash="c" * 64,
        source_hash="s" * 64,
    )


def _store_record(corruption: str, *, checkpoint: str = "a" * 64,
                  build_hash: str = "f" * 64) -> dict:
    model = {
        "encoder_id": "enc", "model_id": "locked/model",
        "checkpoint_sha256": checkpoint,
        "preprocessing_sha256": "b" * 64,
        "feature_layer": "cls", "pooling": "class_token",
        "feature_build_config_sha256": build_hash,
        "head_protocol": {"kind": "fixture"},
        "reference_split": "test", "head_fit_split": "train",
        "reference_id_namespace": "toy:test",
        "head_fit_id_namespace": "toy:train",
    }
    return {
        "dataset": "toy_c", "encoder": "enc", "corruption": corruption,
        "severity": 5, "evaluation_size": 128,
        "manifest": f"/{corruption}/feature_store.json",
        "manifest_sha256": "0" * 64,
        "manifest_payload": {
            "metadata": {"model_provenance": model},
            "id_provenance": {
                "reference": {"namespace": "toy:test"},
            },
        },
    }


def test_store_validation_binds_build_protocol_and_encoder_identity(
    tmp_path, monkeypatch,
):
    context = _store_validation_context(tmp_path)
    records = [_store_record("fog"), _store_record("snow")]
    monkeypatch.setattr(campaign, "_index_native_stores", lambda _config: records)
    outputs = list(campaign._stage_validate_stores(context))
    index = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert index["locked_feature_build_config_canonical_sha256"] == "f" * 64
    assert index["representation_identity_by_encoder"]["enc"] == {
        "checkpoint_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "feature_layer": "cls", "pooling": "class_token",
        "model_id": "locked/model",
        "feature_build_config_sha256": "f" * 64,
    }

    records[1] = _store_record("snow", checkpoint="c" * 64)
    with pytest.raises(IntegrityError, match="mix representation identities"):
        campaign._stage_validate_stores(context)

    records[1] = _store_record("snow", build_hash="d" * 64)
    with pytest.raises(IntegrityError, match="differs from the prespecified lock"):
        campaign._stage_validate_stores(context)
