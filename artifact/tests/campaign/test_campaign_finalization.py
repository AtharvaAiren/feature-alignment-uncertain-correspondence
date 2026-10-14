from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from marginal_alignment.campaign.artifacts import sha256_file
from marginal_alignment.campaign.campaign import (
    SOURCE_PATTERNS,
    UPSTREAM_STAGE_NAMES,
    _publish_final_evidence,
    _seed,
    _stage_force_flags,
    _stage_verify,
)
from marginal_alignment.campaign.stages import StageContext, StageRunner, hash_tree


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_source_manifest_covers_complete_script_tree():
    _, records = hash_tree(PROJECT_ROOT, SOURCE_PATTERNS)
    observed = {record["path"] for record in records}
    expected = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    }
    assert expected <= observed
    assert "scripts/prepare_cifar_c.py" in observed


def _stage_outputs(stage: str) -> tuple[str, ...]:
    declared = {
        "preflight": ("preflight.json",),
        "tests": ("test_report.txt",),
        "gpu_smoke": ("gpu_smoke.json",),
        "synthetic": ("synthetic/manifest.json",),
        "inventory": ("cache_inventory.json",),
        "validate_stores": ("feature_store_index.json",),
        "real": ("real/rows.jsonl", "real/summary.json", "real/manifest.json"),
        "streaming": ("streaming/manifest.json",),
        "controlled": ("controlled/manifest.json",),
        "inference": ("inference.json",),
        "paper_outputs": ("paper_outputs/manifest.json",),
    }
    return declared[stage]


def _minimal_stage(stage: str):
    def run(context: StageContext):
        outputs = []
        for relative in _stage_outputs(stage):
            path = context.run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"stable fixture for {stage}: {relative}\n", encoding="utf-8")
            outputs.append(path)
        return outputs

    return run


def _run_local_verifier(run_dir: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/campaign/verify_campaign.py"), str(run_dir)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_minimal_campaign_finalization_has_no_mutable_self_hash(tmp_path):
    run_dir = tmp_path / "run"
    context = StageContext(
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        config={"campaign_id": "minimal_finalization_fixture"},
        config_hash="c" * 64,
        source_hash="s" * 64,
    )
    runner = StageRunner(context)
    dependencies = {
        "preflight": [],
        "tests": ["preflight"],
        "gpu_smoke": ["tests"],
        "synthetic": ["gpu_smoke"],
        "inventory": ["tests"],
        "validate_stores": ["inventory"],
        "real": ["validate_stores", "synthetic"],
        "streaming": ["validate_stores"],
        "controlled": ["validate_stores", "synthetic"],
        "inference": ["real"],
        "paper_outputs": ["streaming", "controlled", "inference"],
    }
    for stage in UPSTREAM_STAGE_NAMES:
        runner.run(stage, _minimal_stage(stage), dependencies=dependencies[stage])
    runner.run("verify", _stage_verify, dependencies=["paper_outputs"])
    _publish_final_evidence(context)

    manifest_path = run_dir / "CAMPAIGN_MANIFEST.json"
    done_path = run_dir / "DONE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    records = {record["path"]: record for record in manifest["files"]}
    assert "CAMPAIGN_MANIFEST.json" not in records
    assert "DONE" not in records
    # These files were RUNNING/open during _stage_verify, but finalization occurs
    # only after StageRunner has atomically published PASS and closed the log.
    for relative in ("stages/verify.json", "logs/verify.log"):
        assert records[relative]["sha256"] == sha256_file(run_dir / relative)
    assert manifest["stages"]["verify"] == "PASS"
    assert done["status"] == "PASS"
    assert done["campaign_id"] == manifest["campaign_id"]
    assert done["campaign_manifest_sha256"] == sha256_file(manifest_path)
    verified = _run_local_verifier(run_dir)
    assert verified.returncode == 0, verified.stdout
    assert json.loads(verified.stdout)["done_verified"] is True

    extra_path = run_dir / "stale_uninventoried_file.txt"
    extra_path.write_text("stale", encoding="utf-8")
    failed_extra = _run_local_verifier(run_dir)
    assert failed_extra.returncode != 0
    assert "EXTRA stale_uninventoried_file.txt" in failed_extra.stdout
    extra_path.unlink()

    original_done = done_path.read_bytes()
    done["status"] = "FAILED"
    done_path.write_text(json.dumps(done), encoding="utf-8")
    failed_done = _run_local_verifier(run_dir)
    assert failed_done.returncode != 0
    assert "DONE is not PASS" in failed_done.stdout
    done_path.write_bytes(original_done)

    original_manifest = manifest_path.read_bytes()
    manifest["status"] = "FAILED"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    failed_manifest = _run_local_verifier(run_dir)
    assert failed_manifest.returncode != 0
    assert "campaign manifest is not PASS" in failed_manifest.stdout
    manifest_path.write_bytes(original_manifest)

    done = json.loads(original_done.decode("utf-8"))
    done["campaign_id"] = "wrong_campaign"
    done_path.write_text(json.dumps(done), encoding="utf-8")
    failed_id = _run_local_verifier(run_dir)
    assert failed_id.returncode != 0
    assert "campaign_id does not match" in failed_id.stdout

    done = json.loads(original_done.decode("utf-8"))
    done["campaign_manifest_sha256"] = "0" * 64
    done_path.write_text(json.dumps(done), encoding="utf-8")
    failed_binding = _run_local_verifier(run_dir)
    assert failed_binding.returncode != 0
    assert "does not bind" in failed_binding.stdout
    done_path.write_bytes(original_done)


def test_campaign_shell_entrypoints_are_source_hashed():
    assert "scripts/**/*.sh" in SOURCE_PATTERNS


def test_lock_binds_campaign_and_feature_build_configs():
    lock = json.loads(
        (PROJECT_ROOT / "protocol" / "experiment_lock.json").read_text(
            encoding="utf-8"
        )
    )
    for path_key, hash_key in (
        ("config", "config_file_sha256"),
        ("feature_build_config", "feature_build_config_file_sha256"),
    ):
        source = PROJECT_ROOT / lock[path_key]
        assert source.is_file()
        assert sha256_file(source) == lock[hash_key]


def test_campaign_seed_is_stable_and_numpy_compatible():
    assert _seed("stream", "fog", 5) == _seed("stream", "fog", 5)
    assert _seed("stream", "fog", 5) != _seed("stream", "fog", 3)
    assert 0 <= _seed("stream", "fog", 5) < 2**63 - 1


def test_from_stage_forces_descendants_but_still_traverses_ancestors():
    stages = ["preflight", "tests", "real", "verify"]
    flags = _stage_force_flags(stages, from_stage="real", force_stage=None)
    assert flags == {
        "preflight": False, "tests": False, "real": True, "verify": True,
    }
    assert _stage_force_flags(
        stages, from_stage=None, force_stage="real"
    ) == {
        "preflight": False, "tests": False, "real": True, "verify": False,
    }
    with pytest.raises(ValueError, match="unknown from_stage"):
        _stage_force_flags(stages, from_stage="missing", force_stage=None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _stage_force_flags(stages, from_stage="real", force_stage="real")
