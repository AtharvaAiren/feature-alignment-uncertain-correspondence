from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    sha256_file,
)
from marginal_alignment.campaign.feature_store import build_native_manifest
from marginal_alignment.campaign.real_runner import (
    expected_unit_semantic_keys,
    run_condition,
    split_plan_path,
)
from marginal_alignment.iclr_extension.full_rank_confirmatory import (
    THREAD_VARIABLES,
    _build_inference,
    _parse_sha256sum_lock,
    _prepare_output_root,
    _resolve_fresh_output_root,
    _validate_original_campaign_chain,
    compute_unit,
    load_unit,
    record_failure,
    require_single_thread_environment,
    run_extension,
    validate_source_root,
    validate_lock,
    validate_rows_against_anchors,
)
from marginal_alignment.iclr_extension import full_rank_confirmatory as full_rank_module
from scripts.iclr_extension.run_full_rank_paired_confirmatory import _parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _lock() -> dict:
    return json.loads(
        (
            _project_root()
            / "configs/iclr_extension/full_rank_paired_confirmatory_lock.json"
        ).read_text(encoding="utf-8")
    )


def _toy_store(root: Path) -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(811)
    rows, dimension, classes = 4300, 4, 3
    reference = rng.normal(size=(rows, dimension))
    transform = np.asarray(
        [
            [1.1, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.1, 0.0, 0.0, 0.8],
        ],
        dtype=np.float64,
    )
    target = reference @ transform.T + 0.2
    weight = rng.normal(size=(classes, dimension))
    labels = (reference @ weight.T).argmax(1)
    arrays = {
        "reference_features": reference.astype(np.float32),
        "target_features": target.astype(np.float32),
        "reference_ids": np.arange(rows, dtype=np.int64),
        "target_ids": np.arange(10_000, 10_000 + rows, dtype=np.int64),
        "reference_labels": labels.astype(np.int64),
        "target_labels": labels.astype(np.int64),
        "pair_reference_ids": np.arange(rows, dtype=np.int64),
        "pair_target_ids": np.arange(10_000, 10_000 + rows, dtype=np.int64),
        "head_weight": weight.astype(np.float32),
        "head_bias": np.zeros(classes, dtype=np.float32),
        "head_fit_ids": np.arange(20_000, 20_032, dtype=np.int64),
    }
    paths = {}
    for role, value in arrays.items():
        path = root / f"{role}.npy"
        np.save(path, value)
        paths[role] = path.name
    return build_native_manifest(
        root,
        dataset="cifar100_c",
        encoder="clip",
        arrays=paths,
        id_provenance={
            "reference": {
                "verified": True,
                "kind": "fixture",
                "source": "test",
                "namespace": "toy:clean",
            },
            "target": {
                "verified": True,
                "kind": "fixture",
                "source": "test",
                "namespace": "toy:corrupt",
            },
        },
        pairing={
            "status": "available",
            "reference_ids_role": "pair_reference_ids",
            "target_ids_role": "pair_target_ids",
            "provenance": {"verified": True, "kind": "fixture", "source": "test"},
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata={
            "corruption": "brightness",
            "severity": 5,
            "model_provenance": {
                "encoder_id": "clip",
                "checkpoint_sha256": "0" * 64,
                "preprocessing_sha256": "1" * 64,
                "feature_layer": "fixture",
                "pooling": "fixture",
                "head_protocol": "fixture",
                "head_fit_ids_sha256": sha256_file(root / "head_fit_ids.npy"),
                "head_fit_ids_role": "head_fit_ids",
                "head_fit_id_namespace": "toy:head_fit",
                "reference_id_namespace": "toy:clean",
                "reference_split": "test",
                "head_fit_split": "train",
            },
        },
    )


def test_lock_and_thread_contract_are_fail_closed():
    lock = _lock()
    validate_lock(lock)
    drifted = json.loads(json.dumps(lock))
    drifted["original_protocol"]["split_seeds"] = [35, 36, 38]
    drifted.pop("manifest_sha256")
    drifted = seal_manifest(drifted)
    with pytest.raises(IntegrityError, match="split_seeds drifted"):
        validate_lock(drifted)
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        **{
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    require_single_thread_environment(environment)
    environment["OPENBLAS_NUM_THREADS"] = "8"
    with pytest.raises(RuntimeError, match="every numerical thread"):
        require_single_thread_environment(environment)


def test_cli_rejects_original_artifact_path_overrides():
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--project-root",
                "project",
                "--original-run-root",
                "original",
                "--output-root",
                "output",
                "--original-rows",
                "substitute/rows.jsonl",
            ]
        )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_synthetic_original_chain(root: Path, monkeypatch) -> dict[str, Path]:
    root.mkdir()
    (root / "real").mkdir()
    payloads = {
        "resolved_config.json": _canonical_bytes({"kind": "config"}),
        "feature_store_index.json": _canonical_bytes({"kind": "index"}),
        "real/rows.jsonl": _canonical_bytes({"kind": "row"}),
        "source_manifest.json": _canonical_bytes(
            {
                "files": [{"path": f"source/{index}.py"} for index in range(83)],
                "source_sha256": "5" * 64,
            }
        ),
    }
    expectations = {}
    paths = {}
    for relative, payload in payloads.items():
        path = root / relative
        path.write_bytes(payload)
        paths[relative] = path
        expectations[relative] = {"bytes": len(payload), "sha256": _digest(payload)}
    campaign_manifest = {
        "campaign_id": "toy_campaign",
        "files": [
            {"path": relative, **record} for relative, record in expectations.items()
        ],
        "schema_version": "toy.campaign_manifest.v1",
        "source_sha256": "5" * 64,
        "status": "PASS",
    }
    manifest_payload = _canonical_bytes(campaign_manifest)
    manifest_path = root / "CAMPAIGN_MANIFEST.json"
    manifest_path.write_bytes(manifest_payload)
    done = {
        "campaign_id": "toy_campaign",
        "campaign_manifest_sha256": _digest(manifest_payload),
        "schema_version": "toy.done.v1",
        "status": "PASS",
    }
    done_payload = _canonical_bytes(done)
    done_path = root / "DONE"
    done_path.write_bytes(done_payload)
    monkeypatch.setattr(full_rank_module, "ORIGINAL_CAMPAIGN_ID", "toy_campaign")
    monkeypatch.setattr(full_rank_module, "ORIGINAL_INPUT_EXPECTATIONS", expectations)
    monkeypatch.setattr(
        full_rank_module,
        "ORIGINAL_DONE_EXPECTATION",
        {
            "path": "DONE",
            "size_bytes": len(done_payload),
            "sha256": _digest(done_payload),
            "schema_version": "toy.done.v1",
        },
    )
    monkeypatch.setattr(
        full_rank_module,
        "ORIGINAL_MANIFEST_EXPECTATION",
        {
            "path": "CAMPAIGN_MANIFEST.json",
            "size_bytes": len(manifest_payload),
            "sha256": _digest(manifest_payload),
            "schema_version": "toy.campaign_manifest.v1",
            "file_count": 4,
            "source_sha256": "5" * 64,
        },
    )
    return paths


def test_canonical_original_done_manifest_file_chain_is_fail_closed(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    paths = _write_synthetic_original_chain(original, monkeypatch)
    chain = _validate_original_campaign_chain(original)
    assert (
        chain["canonical_paths"]["real/rows.jsonl"]
        == paths["real/rows.jsonl"].resolve()
    )

    paths["real/rows.jsonl"].write_bytes(b"changed\n")
    with pytest.raises(IntegrityError, match="canonical original real/rows.jsonl"):
        _validate_original_campaign_chain(original)


def test_original_done_manifest_binding_mismatch_rejects(tmp_path, monkeypatch):
    original = tmp_path / "original"
    _write_synthetic_original_chain(original, monkeypatch)
    done = json.loads((original / "DONE").read_text(encoding="utf-8"))
    done["campaign_manifest_sha256"] = "0" * 64
    payload = _canonical_bytes(done)
    (original / "DONE").write_bytes(payload)
    monkeypatch.setattr(
        full_rank_module,
        "ORIGINAL_DONE_EXPECTATION",
        {
            **full_rank_module.ORIGINAL_DONE_EXPECTATION,
            "size_bytes": len(payload),
            "sha256": _digest(payload),
        },
    )
    with pytest.raises(IntegrityError, match="DONE semantic payload drifted"):
        _validate_original_campaign_chain(original)


def test_canonical_original_chain_rejects_gzip_substitute_and_symbolic_real_dir(
    tmp_path,
    monkeypatch,
):
    original = tmp_path / "original"
    paths = _write_synthetic_original_chain(original, monkeypatch)
    paths["real/rows.jsonl"].unlink()
    (original / "real/rows.jsonl.gz").write_bytes(b"not accepted")
    with pytest.raises(IntegrityError, match="canonical original real/rows.jsonl"):
        _validate_original_campaign_chain(original)

    other = tmp_path / "other_real"
    other.mkdir()
    (other / "rows.jsonl").write_bytes(_canonical_bytes({"kind": "row"}))
    (original / "real/rows.jsonl.gz").unlink()
    (original / "real").rmdir()
    (original / "real").symlink_to(other, target_is_directory=True)
    with pytest.raises(IntegrityError, match="path component may not be symbolic"):
        _validate_original_campaign_chain(original)


def test_source_root_checksum_locks_are_pre_outcome_roots_of_trust(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    trust = project / "provenance/full_rank"
    trust.mkdir(parents=True)
    base = project / "base.py"
    overlay = project / "overlay.py"
    base.write_bytes(b"base\n")
    overlay.write_bytes(b"overlay\n")
    base_lock = trust / "BASE_DEPENDENCY_SHA256SUMS"
    base_lock.write_text(f"{sha256_file(base)}  base.py\n", encoding="utf-8")
    payload_lock = trust / "PAYLOAD_SHA256SUMS"
    payload_lock.write_text(
        f"{sha256_file(overlay)}  overlay.py\n"
        f"{sha256_file(base_lock)}  provenance/full_rank/"
        "BASE_DEPENDENCY_SHA256SUMS\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(full_rank_module, "BASE_DEPENDENCY_PATHS", ("base.py",))
    monkeypatch.setattr(
        full_rank_module,
        "FULL_RANK_PAYLOAD_PATHS",
        (
            "overlay.py",
            "provenance/full_rank/BASE_DEPENDENCY_SHA256SUMS",
        ),
    )
    trust_record = validate_source_root(project)
    assert trust_record["base_dependency_records"] == 1
    overlay.write_bytes(b"tampered\n")
    with pytest.raises(IntegrityError, match="handoff payload source SHA-256 changed"):
        validate_source_root(project)


def test_source_checksum_parser_rejects_duplicate_and_traversal_paths(tmp_path):
    duplicate = tmp_path / "duplicate.sha256"
    duplicate.write_text(
        f"{'0' * 64}  source.py\n{'1' * 64}  source.py\n",
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="duplicate or noncanonical"):
        _parse_sha256sum_lock(duplicate)
    traversal = tmp_path / "traversal.sha256"
    traversal.write_text(f"{'0' * 64}  ../source.py\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="unsafe checksum path"):
        _parse_sha256sum_lock(traversal)


def test_source_root_rejects_symlinked_dependency(tmp_path, monkeypatch):
    project = tmp_path / "project"
    trust = project / "provenance/full_rank"
    trust.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"base\n")
    (project / "base.py").symlink_to(outside)
    base_lock = trust / "BASE_DEPENDENCY_SHA256SUMS"
    base_lock.write_text(f"{sha256_file(outside)}  base.py\n", encoding="utf-8")
    payload_lock = trust / "PAYLOAD_SHA256SUMS"
    payload_lock.write_text(
        f"{sha256_file(base_lock)}  provenance/full_rank/"
        "BASE_DEPENDENCY_SHA256SUMS\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(full_rank_module, "BASE_DEPENDENCY_PATHS", ("base.py",))
    monkeypatch.setattr(
        full_rank_module,
        "FULL_RANK_PAYLOAD_PATHS",
        ("provenance/full_rank/BASE_DEPENDENCY_SHA256SUMS",),
    )
    with pytest.raises(IntegrityError, match="regular nonsymbolic"):
        validate_source_root(project)


def test_output_resolution_and_failure_logging_reject_symlink_ancestor(
    tmp_path,
    monkeypatch,
):
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    original = tmp_path / "original"
    original.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    unsafe = alias / "new_result"
    with pytest.raises(IntegrityError, match="resolve inside original results"):
        _resolve_fresh_output_root(unsafe, original)
    with pytest.raises(IntegrityError, match="resolve inside original results"):
        run_extension(
            inputs=SimpleNamespace(original_root=original),
            output_root=unsafe,
            workers=1,
        )
    with pytest.raises(IntegrityError, match="resolve inside original results"):
        record_failure(unsafe, RuntimeError("boom"), original_root=original)
    assert not (original / "new_result").exists()


def test_resume_rejects_symbolic_operational_subdirectories(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    binding = seal_manifest({"schema_version": "test.binding.v1", "status": "PASS"})
    (output / "INPUT_BINDING.json").write_bytes(_canonical_bytes(binding))
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "units").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError, match="output subdirectory may not be symbolic"):
        _prepare_output_root(output.resolve(), binding)

    (output / "units").unlink()
    (output / "failures").symlink_to(outside, target_is_directory=True)
    original = tmp_path / "original"
    original.mkdir()
    with pytest.raises(IntegrityError, match="output subdirectory may not be symbolic"):
        record_failure(output, RuntimeError("boom"), original_root=original)
    assert not list(outside.iterdir())


def test_condition_unit_reuses_exact_original_split_identities_and_resumes(
    tmp_path,
    monkeypatch,
):
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    manifest = _toy_store(tmp_path / "store")
    common = {
        "datasets": ["cifar100_c", "imagenet_c"],
        "calibration_sizes": [4096],
        "pair_budgets": [4096],
        "pair_noise_rates": [0.0],
        "split_seeds": [35, 36, 37],
        "condition_filter": {"severities": [5]},
    }
    methods = {
        "bures_pair_endpoints": {
            "implementation": "bures",
            "information": "joint_second_moments_on_paired_endpoint_sets_with_assignment_erased",
            "covariance": "oas",
            "fit_on_pair_endpoint_sets": True,
        },
        "paired_ridge": {
            "implementation": "paired_ridge",
            "information": "true_pairs",
            "ridge": 0.01,
        },
        "paired_ridge_shuffled": {
            "implementation": "paired_ridge",
            "information": "shuffled_pairs",
            "pair_control": "shuffled",
            "ridge": 0.01,
        },
    }
    condition = {
        "dataset": "cifar100_c",
        "encoder": "clip",
        "corruption": "brightness",
        "severity": 5,
        "evaluation_size": 64,
    }
    original_rows = run_condition(
        store_path=manifest,
        suite_name="correspondence_confirmatory",
        suite={**common, "methods": ["bures_pair_endpoints"]},
        methods=methods,
        condition=condition,
        output_dir=tmp_path / "original",
        global_seed=20260812,
        metrics_projections=4,
        mmd_features=8,
        distribution_evaluation_limit=32,
    )
    anchors = {
        str(row["split_seed"]): {
            field: row[field]
            for field in (
                "split_plan_sha256",
                "evaluation_sha256",
                "feature_store_manifest_sha256",
            )
        }
        for row in original_rows
    }
    output = tmp_path / "extension"
    output.mkdir()
    (output / "units").mkdir()
    new_suite = {**common, "methods": ["paired_ridge", "paired_ridge_shuffled"]}
    identity = {
        "suite": "correspondence_confirmatory",
        "dataset": "cifar100_c",
        "encoder": "clip",
        "corruption": "brightness",
        "severity": 5,
        "methods": ["paired_ridge", "paired_ridge_shuffled"],
    }
    plan = {
        "identity": identity,
        "input_fingerprint": "f" * 64,
        "store_path": str(manifest),
        "condition": condition,
        "suite": new_suite,
        "methods": {
            name: methods[name] for name in ("paired_ridge", "paired_ridge_shuffled")
        },
        "global_seed": 20260812,
        "metric_projections": 4,
        "mmd_features": 8,
        "distribution_evaluation_limit": 32,
        "expected_keys": sorted(
            expected_unit_semantic_keys(
                suite_name="correspondence_confirmatory",
                suite=new_suite,
                methods=methods,
                condition=condition,
            )
        ),
        "anchors": anchors,
        "unit_path": str(output / "units/unit.json"),
        "split_paths": [
            str(
                split_plan_path(
                    output,
                    suite_name="correspondence_confirmatory",
                    dataset="cifar100_c",
                    encoder="clip",
                    corruption="brightness",
                    severity=5,
                    split_seed=seed,
                )
            )
            for seed in (35, 36, 37)
        ],
        "output_root": str(output),
    }
    result = compute_unit(plan)
    assert result["record_count"] == 6
    resumed = load_unit(plan)
    assert resumed is not None and len(resumed) == 6
    assert {row["diagnostics"]["rank"] for row in resumed} == {None}
    assert {row["diagnostics"]["solve_form"] for row in resumed} == {"primal"}
    damaged = read_json(plan["unit_path"])
    damaged["input_fingerprint"] = "0" * 64
    Path(plan["unit_path"]).write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(IntegrityError, match="manifest checksum mismatch"):
        load_unit(plan)


def _analysis_row(
    *,
    dataset: str,
    encoder: str,
    corruption: str,
    seed: int,
    method: str,
    value: float,
) -> dict:
    return {
        "suite": "correspondence_confirmatory",
        "dataset": dataset,
        "encoder": encoder,
        "corruption": corruption,
        "severity": 5,
        "split_seed": seed,
        "calibration_size": 4096,
        "evaluation_size": 3000 if dataset == "cifar100_c" else 10000,
        "method": method,
        "pair_budget": 4096,
        "pair_control": {
            "paired_ridge": "true",
            "paired_ridge_shuffled": "shuffled",
            "bures_pair_endpoints": "assignment_erased",
        }[method],
        "pair_noise_rate_requested": (
            0.0
            if method == "paired_ridge"
            else 1.0 if method == "paired_ridge_shuffled" else None
        ),
        "pair_noise_rate_actual": (
            0.0
            if method == "paired_ridge"
            else 1.0 if method == "paired_ridge_shuffled" else None
        ),
        "reference_prediction_agreement": value,
        "aligned_accuracy": value - 0.05,
        "paired_feature_mse": 1.0 - value,
    }


def test_inference_is_one_separate_four_test_holm_family():
    lock = _lock()
    new_rows = []
    anchors = []
    for dataset, specification in lock["original_protocol"]["datasets"].items():
        for corruption_index, corruption in enumerate(specification["corruptions"]):
            base = 0.50 + 0.001 * corruption_index
            for encoder in specification["encoders"]:
                for seed in lock["original_protocol"]["split_seeds"]:
                    anchors.append(
                        _analysis_row(
                            dataset=dataset,
                            encoder=encoder,
                            corruption=corruption,
                            seed=seed,
                            method="bures_pair_endpoints",
                            value=base,
                        )
                    )
                    new_rows.extend(
                        [
                            _analysis_row(
                                dataset=dataset,
                                encoder=encoder,
                                corruption=corruption,
                                seed=seed,
                                method="paired_ridge",
                                value=base + 0.04,
                            ),
                            _analysis_row(
                                dataset=dataset,
                                encoder=encoder,
                                corruption=corruption,
                                seed=seed,
                                method="paired_ridge_shuffled",
                                value=base - 0.02,
                            ),
                        ]
                    )
    result = _build_inference(new_rows=new_rows, anchor_rows=anchors, lock=lock)
    assert result["status"] == "PASS"
    assert len(result["contrasts"]) == 4
    assert [row["test_id"] for row in result["contrasts"]] == lock["inference"]["tests"]
    assert {row["clusters"] for row in result["contrasts"]} == {15, 19}
    assert {row["paired_cells"] for row in result["contrasts"]} == {135, 171}
    assert all(
        row["effect"]
        == pytest.approx(0.04 if row["right"] == "bures_pair_endpoints" else 0.06)
        for row in result["contrasts"]
    )
    assert all(0.0 <= row["holm_p"] <= 1.0 for row in result["contrasts"])
    assert "exchangeability" in result["bootstrap_assumption"]
    assert "sign_symmetry" in result["sign_flip_assumption"]
    assert result["assumption_distinction"].startswith("Cluster exchangeability")
    for row in result["contrasts"]:
        assert row["bootstrap_draws"] == 20_000
        assert row["sign_flip_calibration"] == "exhaustive_all_cluster_sign_assignments"
        assert row["sign_flip_assignments"] == 2 ** row["clusters"]
        assert "sign symmetry" in row["inference_scope"]


def test_identity_validation_rejects_one_drifted_evaluation_hash():
    rows = []
    anchors = {}
    identity = {
        "dataset": "cifar100_c",
        "encoder": "clip",
        "corruption": "fog",
        "severity": 5,
    }
    for seed in (35, 36, 37):
        anchor = {
            "split_plan_sha256": f"{seed:064x}",
            "evaluation_sha256": f"{seed + 100:064x}",
            "feature_store_manifest_sha256": "a" * 64,
        }
        anchors[str(seed)] = anchor
        for method in ("paired_ridge", "paired_ridge_shuffled"):
            rows.append(
                {
                    **identity,
                    "suite": "correspondence_confirmatory",
                    "split_seed": seed,
                    "calibration_size": 4096,
                    "pair_budget": 4096,
                    "method": method,
                    "reference_prediction_agreement": 0.5,
                    **anchor,
                }
            )
    rows[-1]["evaluation_sha256"] = "b" * 64
    with pytest.raises(IntegrityError, match="not bound to the original"):
        validate_rows_against_anchors(rows, anchors, identity=identity)
