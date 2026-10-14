from __future__ import annotations

from pathlib import Path

from marginal_alignment.campaign.artifacts import read_json, sha256_file
from marginal_alignment.paired_recovery_followup.protocol import (
    LockedFollowupProtocol,
    _validate_config,
    load_locked_protocol,
    verify_external_input_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def test_prospective_configuration_has_exact_frozen_grid() -> None:
    config = read_json(
        ROOT / "configs" / "paired_recovery_followup" / "followup_v1.json"
    )
    _validate_config(config)
    assert len(config["cells"]) == 8
    assert config["unit_accounting"]["evaluation_batch_units"] == 160
    assert config["unit_accounting"]["total_assignment_certificates"] == 1920


def test_authenticated_external_manifest_contains_all_followup_stores() -> None:
    config_path = (
        ROOT / "configs" / "paired_recovery_followup" / "followup_v1.json"
    )
    config = read_json(config_path)
    manifest_path = ROOT / config["external_input_manifest"]["source_path"]
    if not manifest_path.is_file():
        # The development tree keeps this large generated inventory with the
        # campaign results; sealed deployment sources embed it at source_path.
        manifest_path = ROOT / "results" / "campaign" / "external_input_manifest.json"
    protocol = LockedFollowupProtocol(
        project_root=ROOT,
        config_path=config_path,
        lock_path=ROOT / "protocol" / "paired_recovery_followup" / "pending.json",
        config=config,
        lock={},
        config_file_sha256=sha256_file(config_path),
        lock_file_sha256="pending",
    )
    manifest = verify_external_input_manifest(protocol, manifest_path)
    assert manifest["native_feature_store_count"] == 306


def test_source_lock_authenticates_complete_local_implementation() -> None:
    protocol = load_locked_protocol(ROOT)
    assert protocol.config["study_id"] == "imagenet_c_paired_recovery_followup_20260914_v1"
    assert len(protocol.lock["source_files"]) == 24
