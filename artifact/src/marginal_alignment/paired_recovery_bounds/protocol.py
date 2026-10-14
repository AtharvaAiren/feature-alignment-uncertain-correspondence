"""Fail-closed prospective lock and authenticated input validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    safe_relative_path,
    sha256_file,
    verify_manifest_checksum,
)
from marginal_alignment.campaign.feature_store import FeatureStore, load_feature_store


CONFIG_SCHEMA = "marginal_alignment.paired_recovery_bounds.study.v1"
LOCK_SCHEMA = "marginal_alignment.paired_recovery_bounds.lock.v1"
EXPECTED_STUDY_ID = "imagenet_c_clip_paired_conformal_recovery_bounds_20260913_v1"
EXPECTED_CORRUPTIONS = ("gaussian_noise", "motion_blur")
EXPECTED_METHODS = ("identity", "diagonal", "bures")
REQUIRED_SOURCE_PATHS = frozenset({
    "src/marginal_alignment/linalg.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/campaign/split_plan.py",
    "src/marginal_alignment/paired_recovery_bounds/__init__.py",
    "src/marginal_alignment/paired_recovery_bounds/conformal.py",
    "src/marginal_alignment/paired_recovery_bounds/geometry.py",
    "src/marginal_alignment/paired_recovery_bounds/optimization.py",
    "src/marginal_alignment/paired_recovery_bounds/protocol.py",
    "src/marginal_alignment/paired_recovery_bounds/runner.py",
    "scripts/paired_recovery_bounds/run_study.py",
    "scripts/paired_recovery_bounds/verify_results.py",
})


@dataclass(frozen=True)
class LockedProtocol:
    project_root: Path
    config_path: Path
    lock_path: Path
    config: Mapping[str, Any]
    lock: Mapping[str, Any]
    config_file_sha256: str
    lock_file_sha256: str

    @property
    def corruption_by_name(self) -> dict[str, Mapping[str, Any]]:
        return {
            str(record["name"]): record for record in self.config["corruptions"]
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a JSON object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise IntegrityError("unsupported paired recovery configuration schema")
    if config.get("status") != "PRESPECIFIED_BEFORE_NEW_RECOVERY_BOUND_OUTCOMES":
        raise IntegrityError("paired recovery configuration is not prospectively locked")
    if config.get("study_id") != EXPECTED_STUDY_ID:
        raise IntegrityError("paired recovery study identity changed")
    if config.get("dataset") != "imagenet_c":
        raise IntegrityError("paired recovery dataset changed")
    encoder = _mapping(config.get("encoder"), "encoder")
    if encoder.get("id") != "clip" or int(encoder.get("feature_dimension", -1)) != 768:
        raise IntegrityError("locked encoder contract changed")
    corruptions = config.get("corruptions")
    if not isinstance(corruptions, list) or tuple(
        str(row.get("name")) for row in corruptions if isinstance(row, Mapping)
    ) != EXPECTED_CORRUPTIONS:
        raise IntegrityError("locked corruption order changed")
    if any(int(row.get("severity", -1)) != 5 for row in corruptions):
        raise IntegrityError("locked ImageNet-C severity changed")
    partition = _mapping(config.get("partition"), "partition")
    expected_partition = {
        "map_fit_rows": 4096,
        "calibration_batches": 29,
        "evaluation_batches": 20,
        "batch_rows": 256,
        "required_eligible_pairs": 16640,
        "unused_eligible_pairs": 8360,
    }
    for key, expected in expected_partition.items():
        if int(partition.get(key, -1)) != expected:
            raise IntegrityError(f"locked partition field changed: {key}")
    if partition.get("endpoint_order") != (
        "reference_and_target_orders_independently_permuted_with_locked_rng_"
        "true_assignment_withheld_until_post_bound_scoring"
    ):
        raise IntegrityError("locked independent endpoint-order contract changed")
    methods = _mapping(config.get("methods"), "methods")
    if tuple(methods.get("ordered", ())) != EXPECTED_METHODS:
        raise IntegrityError("locked method family changed")
    geometry = _mapping(config.get("common_cost_geometry"), "common cost geometry")
    if (
        geometry.get("name") != "full_reference_oas_mahalanobis_after_bures"
        or geometry.get(
            "one_cost_matrix_and_one_feasible_correspondence_set_shared_by_all_methods_in_a_batch"
        )
        is not True
    ):
        raise IntegrityError("locked shared cost geometry changed")
    conformal = _mapping(config.get("conformal"), "conformal protocol")
    if (
        float(conformal.get("delta", math.nan)) != 0.1
        or int(conformal.get("calibration_count", -1)) != 29
        or int(conformal.get("rank_one_indexed", -1)) != 27
        or conformal.get("when_k_exceeds_m") != "qhat_is_positive_infinity_exactly"
    ):
        raise IntegrityError("locked conformal protocol changed")
    execution = _mapping(config.get("execution"), "execution")
    if execution.get("cpu_only") is not True or int(
        execution.get("parallel_workers", -1)
    ) != 2:
        raise IntegrityError("locked CPU execution contract changed")


def load_locked_protocol(
    project_root: str | Path,
    *,
    config_path: str | Path = "configs/paired_recovery_bounds/imagenet_c_clip_v1.json",
    lock_path: str | Path = "protocol/paired_recovery_bounds/STUDY_LOCK_20260913.json",
) -> LockedProtocol:
    root = Path(project_root).expanduser().resolve()
    config_candidate = Path(config_path)
    lock_candidate = Path(lock_path)
    config_file = (
        config_candidate.resolve()
        if config_candidate.is_absolute()
        else (root / safe_relative_path(config_candidate)).resolve()
    )
    lock_file = (
        lock_candidate.resolve()
        if lock_candidate.is_absolute()
        else (root / safe_relative_path(lock_candidate)).resolve()
    )
    for path, name in ((config_file, "config"), (lock_file, "lock")):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"paired recovery {name} is outside project root") from error
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"paired recovery {name} is absent or symbolic: {path}")
    config = _mapping(read_json(config_file), "paired recovery configuration")
    _validate_config(config)
    lock = _mapping(read_json(lock_file), "paired recovery lock")
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise IntegrityError("unsupported paired recovery lock schema")
    if lock.get("status") != "FROZEN_BEFORE_NEW_RECOVERY_BOUND_OUTCOMES":
        raise IntegrityError("paired recovery study lock is not frozen")
    verify_manifest_checksum(lock)
    config_digest = sha256_file(config_file)
    if lock.get("config_file_sha256") != config_digest:
        raise IntegrityError("paired recovery lock does not bind the current config")
    source_files = _mapping(lock.get("source_files"), "locked source file inventory")
    if not REQUIRED_SOURCE_PATHS.issubset(source_files):
        missing = sorted(REQUIRED_SOURCE_PATHS.difference(source_files))
        raise IntegrityError(f"paired recovery lock omits required sources: {missing}")
    for raw_relative, expected in sorted(source_files.items()):
        relative = safe_relative_path(str(raw_relative))
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"locked source escapes project root: {relative}") from error
        if not source.is_file() or source.is_symlink():
            raise IntegrityError(f"locked source is absent or symbolic: {relative}")
        if sha256_file(source) != str(expected):
            raise IntegrityError(f"locked source bytes changed: {relative}")
    return LockedProtocol(
        project_root=root,
        config_path=config_file,
        lock_path=lock_file,
        config=config,
        lock=lock,
        config_file_sha256=config_digest,
        lock_file_sha256=sha256_file(lock_file),
    )


def verify_external_input_manifest(
    protocol: LockedProtocol, path: str | Path
) -> Mapping[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise IntegrityError("external input manifest is absent or symbolic")
    expected = _mapping(
        protocol.config.get("external_input_manifest"), "external input contract"
    )
    if sha256_file(candidate) != expected.get("file_sha256"):
        raise IntegrityError("external input manifest file hash changed")
    manifest = _mapping(read_json(candidate), "external input manifest")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("external_input_sha256") != expected.get("external_input_sha256")
        or int(manifest.get("native_feature_store_count", -1))
        != int(expected.get("native_feature_store_count", -2))
    ):
        raise IntegrityError("external input manifest identity changed")
    expected_entries = {
        ("imagenet_c", "clip", str(row["name"]), int(row["severity"])): str(
            row["store_manifest_file_sha256"]
        )
        for row in protocol.config["corruptions"]
    }
    observed: dict[tuple[str, str, str, int], str] = {}
    stores = manifest.get("native_feature_stores")
    if not isinstance(stores, list):
        raise IntegrityError("external input manifest store inventory is malformed")
    for row in stores:
        if not isinstance(row, Mapping):
            raise IntegrityError("external input manifest contains a malformed store")
        key = (
            str(row.get("dataset")),
            str(row.get("encoder")),
            str(row.get("corruption")),
            int(row.get("severity", -1)),
        )
        if key in expected_entries:
            observed[key] = str(row.get("manifest_sha256"))
    if observed != expected_entries:
        raise IntegrityError("external input manifest does not bind both locked stores")
    return manifest


def load_authenticated_store(
    protocol: LockedProtocol,
    *,
    feature_root: str | Path,
    corruption: str,
) -> FeatureStore:
    try:
        corruption_spec = protocol.corruption_by_name[str(corruption)]
    except KeyError as error:
        raise IntegrityError(f"corruption is outside the lock: {corruption}") from error
    severity = int(corruption_spec["severity"])
    manifest_path = (
        Path(feature_root).expanduser().resolve()
        / "imagenet_c"
        / "clip"
        / str(corruption)
        / f"severity_{severity}"
        / "FEATURE_STORE_MANIFEST.json"
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise IntegrityError(f"locked feature-store manifest is absent or symbolic: {manifest_path}")
    if sha256_file(manifest_path) != corruption_spec["store_manifest_file_sha256"]:
        raise IntegrityError(f"locked {corruption} feature-store manifest hash changed")
    store = load_feature_store(manifest_path, verify_hashes=True)
    if store.dataset != "imagenet_c" or store.encoder != "clip":
        raise IntegrityError("loaded feature store has the wrong dataset or encoder")
    metadata = _mapping(store.manifest.get("metadata"), "feature-store metadata")
    if metadata.get("corruption") != corruption or int(metadata.get("severity", -1)) != severity:
        raise IntegrityError("loaded feature-store condition identity changed")
    shared = _mapping(protocol.config.get("shared_store_contract"), "shared store contract")
    expected_arrays = {
        "reference_features": shared["reference_features_sha256"],
        "reference_ids": shared["endpoint_ids_sha256"],
        "target_ids": shared["endpoint_ids_sha256"],
        "pair_reference_ids": shared["endpoint_ids_sha256"],
        "pair_target_ids": shared["endpoint_ids_sha256"],
        "head_weight": shared["head_weight_sha256"],
        "head_bias": shared["head_bias_sha256"],
        "head_fit_ids": shared["head_fit_ids_sha256"],
        "target_features": corruption_spec["target_features_sha256"],
    }
    for role, expected_hash in expected_arrays.items():
        record = store.records.get(role)
        if not isinstance(record, Mapping) or record.get("sha256") != expected_hash:
            raise IntegrityError(f"locked feature-store array identity changed: {role}")
    reference_shape = tuple(int(value) for value in store.records["reference_features"]["shape"])
    target_shape = tuple(int(value) for value in store.records["target_features"]["shape"])
    if reference_shape != (50000, 768) or target_shape != reference_shape:
        raise IntegrityError("locked CLIP endpoint shape changed")
    pairing = store.pairing()
    if pairing.provenance.get("kind") != "natural_clean_corruption_identity":
        raise IntegrityError("locked clean-corrupt pairing provenance changed")
    return store


def runtime_identity() -> dict[str, Any]:
    try:
        import scipy
    except ImportError as error:  # pragma: no cover - server dependency contract
        raise RuntimeError("paired recovery bounds require scipy") from error
    thread_keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
    }
