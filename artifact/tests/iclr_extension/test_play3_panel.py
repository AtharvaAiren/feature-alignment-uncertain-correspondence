from __future__ import annotations

import json
from pathlib import Path
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError
from marginal_alignment.campaign.artifacts import seal_manifest, sha256_file
from marginal_alignment.campaign.feature_store import NATIVE_SCHEMA
from scripts.iclr_extension import run_play3_panel as panel


def _locked_config() -> dict:
    return {
        "encoders": list(panel.EXPECTED_ENCODERS),
        "dataset": {"name": "imagenet_sketch"},
        "ceiling_protocol": {
            "primary_cost_budget_relative": 0.05,
            "cost_budget_relative_sensitivity": [0.01, 0.1],
            "split_seeds": list(panel.EXPECTED_SEEDS),
        },
        "positive_control": {
            "dataset": "imagenet_c",
            "severity": 5,
            "heldout_corruptions": list(panel.EXPECTED_CORRUPTIONS),
        },
    }


def _extension_lock() -> dict:
    return {
        "status": "LOCKED",
        "base_campaign_feature_stores": {
            "external_input_manifest_file_sha256": "a" * 64,
            "external_input_sha256": "b" * 64,
            "natural_reference_condition": {
                "dataset": "imagenet_c", "corruption": "brightness", "severity": 5,
            },
        },
    }


def _natural_source_binding(encoder: str) -> dict:
    return {
        "schema_version": (
            "marginal_alignment.iclr_extension.base_campaign_store_binding.v1"
        ),
        "status": "PASS",
        "condition": {
            "dataset": "imagenet_c", "encoder": encoder,
            "corruption": "brightness", "severity": 5,
        },
        "external_input_manifest_file_sha256": "a" * 64,
        "external_input_sha256": "b" * 64,
        "expected_manifest_sha256": "c" * 64,
        "actual_manifest_sha256": "c" * 64,
    }


def _store_fixture(tmp_path: Path, monkeypatch, *, wrong_identity: bool = False):
    natural_paths: dict[str, str] = {}

    def write_manifest(
        root: Path, *, dataset: str, encoder: str, pairing: str,
        corruption: str | None = None,
    ) -> Path:
        body = root / "unselected-body.npy"
        body.write_bytes(b"deliberately-not-an-npy-body")
        manifest = root / "FEATURE_STORE_MANIFEST.json"
        manifest.write_text(json.dumps(seal_manifest({
            "schema_version": NATIVE_SCHEMA,
            "dataset": dataset,
            "encoder": encoder,
            "arrays": {
                "stub": {
                    "path": body.name,
                    "sha256": sha256_file(body),
                    "size_bytes": body.stat().st_size,
                }
            },
            "pairing": {"status": pairing},
            "metadata": ({
                "corruption": corruption,
                "severity": 5,
            } if corruption is not None else {
                "source_campaign_reference_store": _natural_source_binding(encoder),
            }),
        })))
        return manifest

    for encoder in panel.EXPECTED_ENCODERS:
        root = tmp_path / "sketch" / encoder
        root.mkdir(parents=True)
        write_manifest(
            root, dataset="imagenet_sketch", encoder=encoder, pairing="unavailable",
        )
        natural_paths[encoder] = str(root.resolve())

    positive_root = (tmp_path / "positive").resolve()
    positive_root.mkdir()
    for encoder in panel.EXPECTED_ENCODERS:
        for corruption in panel.EXPECTED_CORRUPTIONS:
            root = positive_root / "imagenet_c" / encoder / corruption / "severity_5"
            root.mkdir(parents=True)
            metadata_corruption = (
                "wrong" if wrong_identity and encoder == "dinov3"
                and corruption == "brightness" else corruption
            )
            write_manifest(
                root, dataset="imagenet_c", encoder=encoder, pairing="available",
                corruption=metadata_corruption,
            )
    store_map = tmp_path / "stores.json"
    store_map.write_text(json.dumps({
        "schema_version": panel.STORE_MAP_SCHEMA,
        "natural_stores": natural_paths,
        "positive_control_root": str(positive_root),
    }))
    return store_map, positive_root


def _orchestration_inputs(tmp_path: Path, monkeypatch):
    config = _locked_config()
    config_path = tmp_path / "play3.json"
    config_path.write_text(json.dumps(config))
    store_map, _ = _store_fixture(tmp_path, monkeypatch)
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}\n")
    archive = tmp_path / "sketch.tar"
    archive.write_bytes(b"archive")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setattr(
        panel,
        "load_locked_play3_config",
        lambda value: (config, config_path.resolve(), _extension_lock(), tmp_path),
    )
    return {
        "config_path": config_path,
        "store_map_path": store_map,
        "provenance_path": provenance,
        "dataset_root": dataset_root,
        "archive_path": archive,
        "output_root": tmp_path / "outputs",
    }


def test_exact_locked_panel_is_complete_unique_and_deterministic():
    first = panel.enumerate_locked_panel(_locked_config())
    second = panel.enumerate_locked_panel(_locked_config())
    assert first == second
    assert len(first) == 315
    assert sum(unit.panel == "natural" for unit in first) == 45
    assert sum(unit.panel == "positive_control" for unit in first) == 270
    assert [unit.global_index for unit in first] == list(range(315))
    assert len({unit.unit_id for unit in first}) == 315
    assert first[0].unit_id == "imagenet_sketch__dinov3__seed_201__budget_0500bp"
    assert first[45].unit_id == (
        "positive__brightness__dinov3__seed_201__budget_0500bp"
    )


def test_deterministic_shards_are_disjoint_and_exactly_complete():
    units = panel.enumerate_locked_panel(_locked_config())
    shards = [
        panel.select_shard(units, shard_index=index, num_shards=8)
        for index in range(8)
    ]
    flattened = [unit for shard in shards for unit in shard]
    assert len(flattened) == len(units)
    assert {unit.unit_id for unit in flattened} == {unit.unit_id for unit in units}
    assert sum(len(shard) for shard in shards) == 315
    assert all(
        unit.global_index % 8 == index
        for index, shard in enumerate(shards) for unit in shard
    )
    with pytest.raises(IntegrityError):
        panel.select_shard(units, shard_index=-1, num_shards=8)
    with pytest.raises(IntegrityError):
        panel.select_shard(units, shard_index=8, num_shards=8)
    with pytest.raises(IntegrityError):
        panel.select_shard(units, shard_index=0, num_shards=316)


def test_store_map_requires_exact_keys_and_store_identities(tmp_path, monkeypatch):
    store_map, _ = _store_fixture(tmp_path, monkeypatch)
    resolved = panel.validate_store_map(store_map, _locked_config())
    assert set(resolved.natural) == set(panel.EXPECTED_ENCODERS)
    assert len(resolved.positive) == 18
    assert len(resolved.records) == 21
    assert {record["body_hash_verification"] for record in resolved.records} == {
        "delegated_lower_runner_for_selected_shard_only"
    }

    value = json.loads(store_map.read_text())
    value["natural_stores"].pop("clip")
    store_map.write_text(json.dumps(value))
    with pytest.raises(IntegrityError, match="exactly the locked encoders"):
        panel.validate_store_map(store_map, _locked_config())

    other = tmp_path / "wrong"
    other.mkdir()
    wrong_map, _ = _store_fixture(other, monkeypatch, wrong_identity=True)
    with pytest.raises(IntegrityError, match="store identity is invalid"):
        panel.validate_store_map(wrong_map, _locked_config())


def test_store_map_rejects_body_size_drift_without_hashing_all_bodies(
    tmp_path, monkeypatch,
):
    store_map, _ = _store_fixture(tmp_path, monkeypatch)
    value = json.loads(store_map.read_text())
    natural_root = Path(value["natural_stores"]["dinov3"])
    (natural_root / "unselected-body.npy").write_bytes(b"changed-size")
    with pytest.raises(IntegrityError, match="array size mismatch"):
        panel.validate_store_map(store_map, _locked_config())


def test_plan_only_is_outcome_blind_and_does_not_launch(tmp_path, monkeypatch):
    arguments = _orchestration_inputs(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    result = panel.orchestrate_panel(
        **arguments,
        shard_index=17,
        num_shards=315,
        plan_only=True,
        lower_main=lambda argv: calls.append(argv) or 0,
    )
    assert calls == []
    assert result["scientific_result"] is False
    assert result["compute_started"] is False
    assert result["outcomes_inspected"] is False
    assert result["complete_panel"] == {
        "natural_units": 45,
        "positive_control_units": 270,
        "total_units": 315,
    }
    assert result["shard"]["selected_units"] == 1
    assert not arguments["output_root"].exists()


def test_full_delegates_to_lower_runner_with_fresh_unit_output(
    tmp_path, monkeypatch
):
    arguments = _orchestration_inputs(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def lower(argv: list[str]) -> int:
        calls.append(argv)
        # Deliberately do not publish an output.  Success proves that the
        # orchestrator does not open or interpret lower-level outcomes.
        return 0

    result = panel.orchestrate_panel(
        **arguments,
        shard_index=46,
        num_shards=315,
        plan_only=False,
        lower_main=lower,
    )
    assert result["delegated_units"] == 1
    assert result["lower_level_outcomes_read_by_orchestrator"] is False
    assert len(calls) == 1
    assert calls[0][0] == "positive-control"
    assert calls[0][calls[0].index("--split-seed") + 1] == "201"
    output = Path(calls[0][calls[0].index("--output") + 1])
    assert output.name.endswith(".json")
    assert not output.exists()

    output.parent.mkdir(parents=True)
    output.write_text("already present\n")
    calls.clear()
    with pytest.raises(IntegrityError, match="not fresh"):
        panel.orchestrate_panel(
            **arguments,
            shard_index=46,
            num_shards=315,
            plan_only=False,
            lower_main=lower,
        )
    assert calls == []


def test_resume_skips_only_exact_completed_identity_and_rejects_tamper(
    tmp_path, monkeypatch
):
    arguments = _orchestration_inputs(tmp_path, monkeypatch)
    plan_value = panel.orchestrate_panel(
        **arguments,
        shard_index=46,
        num_shards=315,
        plan_only=True,
    )
    unit = plan_value["units"][0]
    destination = Path(unit["output"])
    unit_plan = {
        "schema_version": "marginal_alignment.iclr_extension.positive_control_plan.v1",
        "status": "PASS",
        "scientific_result": False,
        "compute_started": False,
        "unit_id": unit["unit_id"],
        "dataset": unit["dataset"],
        "encoder": unit["encoder"],
        "corruption": unit["corruption"],
        "severity": unit["severity"],
        "split_seed": unit["split_seed"],
        "cost_budget_relative": unit["cost_budget_relative"],
        "store_manifest": unit["store_manifest"],
        "store_manifest_sha256": unit["store_manifest_sha256"],
        "source_campaign_store": {
            "status": "PASS",
            "actual_manifest_sha256": unit["store_manifest_sha256"],
        },
        "config_sha256": plan_value["config_sha256"],
        "extension_lock": _extension_lock(),
        "runtime": plan_value["runtime"],
    }
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(seal_manifest({
        "schema_version": "marginal_alignment.iclr_extension.positive_control.v2",
        "status": "PASS",
        "scientific_result": True,
        "immutable_fresh_output": True,
        "unit_plan": unit_plan,
        "posthoc_validation": {"outcome": "must-not-be-read"},
    })))
    calls: list[list[str]] = []
    resumed = panel.orchestrate_panel(
        **arguments,
        shard_index=46,
        num_shards=315,
        plan_only=False,
        resume=True,
        lower_main=lambda argv: calls.append(argv) or 0,
    )
    assert calls == []
    assert resumed["resumed_units"] == 1
    assert resumed["delegated_units"] == 0
    assert resumed["lower_level_outcomes_read_by_orchestrator"] is False

    result = json.loads(destination.read_text())
    result.pop("manifest_sha256")
    result["unit_plan"]["config_sha256"] = "0" * 64
    destination.write_text(json.dumps(seal_manifest(result)))
    with pytest.raises(IntegrityError, match="identity drifted"):
        panel.orchestrate_panel(
            **arguments,
            shard_index=46,
            num_shards=315,
            plan_only=False,
            resume=True,
            lower_main=lambda argv: 0,
        )


def test_natural_resume_binds_generated_store_and_locked_base_reference_separately(
    tmp_path, monkeypatch,
):
    arguments = _orchestration_inputs(tmp_path, monkeypatch)
    plan_value = panel.orchestrate_panel(
        **arguments, shard_index=0, num_shards=315, plan_only=True,
    )
    unit = plan_value["units"][0]
    assert unit["panel"] == "natural"
    destination = Path(unit["output"])
    store_manifest = Path(unit["store_manifest"])
    store_payload = json.loads(store_manifest.read_text())
    source_binding = store_payload["metadata"]["source_campaign_reference_store"]
    unit_plan = {
        "schema_version": "marginal_alignment.iclr_extension.play3_unit_plan.v1",
        "status": "PASS", "scientific_result": False, "compute_started": False,
        "unit_id": unit["unit_id"], "dataset": unit["dataset"],
        "encoder": unit["encoder"], "split_seed": unit["split_seed"],
        "cost_budget_relative": unit["cost_budget_relative"],
        "store_manifest": unit["store_manifest"],
        "store_manifest_sha256": unit["store_manifest_sha256"],
        "config_sha256": plan_value["config_sha256"],
        "lock_manifest_sha256": panel.sha256_json(_extension_lock()),
        "dataset_provenance_manifest_sha256": sha256_file(
            arguments["provenance_path"]
        ),
        "source_campaign_reference_store": source_binding,
        "runtime": plan_value["runtime"],
    }
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(seal_manifest({
        "schema_version": panel.NATURAL_SHIFT_RESULT_SCHEMA,
        "status": "PASS", "scientific_result": True,
        "immutable_fresh_output": True, "unit_plan": unit_plan,
        "posthoc_validation": {"outcome": "must-not-be-read"},
    })))
    calls: list[list[str]] = []
    resumed = panel.orchestrate_panel(
        **arguments, shard_index=0, num_shards=315, plan_only=False, resume=True,
        lower_main=lambda argv: calls.append(argv) or 0,
    )
    assert calls == []
    assert resumed["resumed_units"] == 1

    changed = json.loads(destination.read_text())
    changed.pop("manifest_sha256")
    changed["unit_plan"]["source_campaign_reference_store"] = {
        **source_binding, "actual_manifest_sha256": "f" * 64,
    }
    destination.write_text(json.dumps(seal_manifest(changed)))
    with pytest.raises(IntegrityError, match="natural source-store provenance drifted"):
        panel.orchestrate_panel(
            **arguments, shard_index=0, num_shards=315, plan_only=False, resume=True,
            lower_main=lambda argv: 0,
        )


def test_locked_axis_drift_fails_before_enumeration():
    changed = _locked_config()
    changed["ceiling_protocol"]["split_seeds"] = [201]
    with pytest.raises(IntegrityError, match="split seeds changed"):
        panel.enumerate_locked_panel(changed)
