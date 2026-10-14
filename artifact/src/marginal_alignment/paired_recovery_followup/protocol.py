"""Fail-closed validation for the 2026-09-14 paired-recovery follow-up."""

from __future__ import annotations

from dataclasses import dataclass
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


CONFIG_SCHEMA = "marginal_alignment.paired_recovery_followup.study.v1"
LOCK_SCHEMA = "marginal_alignment.paired_recovery_followup.lock.v1"
EXPECTED_STUDY_ID = "imagenet_c_paired_recovery_followup_20260914_v1"
EXPECTED_METHODS = ("identity", "diagonal", "bures")
EXPECTED_CORRUPTIONS = ("gaussian_noise", "motion_blur")
EXPECTED_CONTRASTS = (
    ("diagonal_minus_identity", "diagonal", "identity"),
    ("bures_minus_identity", "bures", "identity"),
    ("bures_minus_diagonal", "bures", "diagonal"),
)
EXPECTED_CELL_KEYS = (
    ("clip", "gaussian_noise", 1),
    ("clip", "motion_blur", 1),
    ("clip", "gaussian_noise", 2),
    ("clip", "motion_blur", 2),
    ("clip", "gaussian_noise", 3),
    ("clip", "motion_blur", 3),
    ("vitb16", "gaussian_noise", 1),
    ("vitb16", "motion_blur", 1),
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a JSON object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise IntegrityError("unsupported follow-up configuration schema")
    if config.get("status") != "PRESPECIFIED_BEFORE_FOLLOWUP_OUTCOMES":
        raise IntegrityError("follow-up configuration is not prospectively frozen")
    if config.get("study_id") != EXPECTED_STUDY_ID:
        raise IntegrityError("follow-up study identity changed")
    if config.get("dataset") != "imagenet_c" or int(config.get("severity", -1)) != 5:
        raise IntegrityError("locked dataset or severity changed")
    if tuple(config.get("corruptions", ())) != EXPECTED_CORRUPTIONS:
        raise IntegrityError("locked corruption panel changed")
    if tuple(config.get("methods", ())) != EXPECTED_METHODS:
        raise IntegrityError("locked method family changed")
    contrasts = config.get("signed_contrasts")
    if not isinstance(contrasts, list):
        raise IntegrityError("signed contrasts are malformed")
    observed_contrasts = tuple(
        (str(row.get("id")), str(row.get("method_a")), str(row.get("method_c")))
        for row in contrasts
        if isinstance(row, Mapping)
    )
    if observed_contrasts != EXPECTED_CONTRASTS:
        raise IntegrityError("signed contrast directions changed")
    cells = config.get("cells")
    if not isinstance(cells, list):
        raise IntegrityError("follow-up cell grid is malformed")
    cell_keys = tuple(
        (str(row.get("encoder")), str(row.get("corruption")), int(row.get("repetition", -1)))
        for row in cells
        if isinstance(row, Mapping)
    )
    if cell_keys != EXPECTED_CELL_KEYS:
        raise IntegrityError("locked cell grid or ordering changed")
    if any(int(row.get("split_seed", -1)) <= 0 for row in cells):
        raise IntegrityError("cell split seed is invalid")
    if any(int(row.get("control_seed", -1)) <= 0 for row in cells):
        raise IntegrityError("cell control seed is invalid")
    partition = _mapping(config.get("partition"), "partition")
    expected_partition = {
        "map_fit_rows": 4096,
        "direct_standardization_batches_within_map_fit_role": 16,
        "calibration_batches": 29,
        "evaluation_batches": 20,
        "batch_rows": 256,
        "required_eligible_pairs": 16640,
        "unused_eligible_pairs": 8360,
    }
    for key, expected in expected_partition.items():
        if int(partition.get(key, -1)) != expected:
            raise IntegrityError(f"locked partition field changed: {key}")
    direct = _mapping(config.get("direct_recovery_baseline"), "direct baseline")
    if (
        direct.get("kind") != "split_conformal_shared_max_standardized_method_error"
        or float(direct.get("scale_floor", math.nan)) != 1.0 / 256.0
    ):
        raise IntegrityError("direct recovery baseline changed")
    geometry = _mapping(config.get("common_cost_geometry"), "cost geometry")
    if geometry.get("name") != "full_reference_oas_mahalanobis_after_bures":
        raise IntegrityError("common cost geometry changed")
    control = _mapping(
        config.get("column_permuted_geometry_control"), "geometry control"
    )
    if (
        control.get("enabled") is not True
        or control.get("operation") != "permute_only_columns_of_each_cost_matrix"
        or control.get("physical_reference_features_predictions_and_rewards_permuted")
        is not False
        or control.get("true_correspondence_permuted") is not False
        or control.get("original_threshold_reused") is not False
    ):
        raise IntegrityError("column-permuted control contract changed")
    conformal = _mapping(config.get("conformal"), "conformal protocol")
    if (
        float(conformal.get("delta", math.nan)) != 0.1
        or int(conformal.get("calibration_count", -1)) != 29
        or int(conformal.get("rank_one_indexed", -1)) != 27
    ):
        raise IntegrityError("conformal protocol changed")
    optimization = _mapping(config.get("optimization"), "optimization")
    if (
        int(optimization.get("dual_bisection_iterations", -1)) != 56
        or optimization.get("signed_rewards_clipped") is not False
        or optimization.get("certified_improvement_rule")
        != "signed_upper_bound_strictly_less_than_zero"
    ):
        raise IntegrityError("signed optimization protocol changed")
    encoders = _mapping(config.get("encoders"), "encoders")
    if set(encoders) != {"clip", "vitb16"}:
        raise IntegrityError("encoder panel changed")
    if int(_mapping(encoders["clip"], "CLIP encoder").get("feature_dimension", -1)) != 768:
        raise IntegrityError("CLIP feature dimension changed")
    if int(_mapping(encoders["vitb16"], "ViT-B/16 encoder").get("feature_dimension", -1)) != 768:
        raise IntegrityError("ViT-B/16 feature dimension changed")
    accounting = _mapping(config.get("unit_accounting"), "unit accounting")
    expected_accounting = {
        "cells": 8,
        "primary_clip_cells": 6,
        "secondary_vitb16_cells": 2,
        "evaluation_batch_units": 160,
        "absolute_certificates": 960,
        "signed_contrast_certificates": 960,
        "total_assignment_certificates": 1920,
        "calibration_cost_problems_across_both_geometries": 464,
        "evaluation_cost_problems_across_both_geometries": 320,
        "exact_milp_diagnostics": 96,
    }
    for key, expected in expected_accounting.items():
        if int(accounting.get(key, -1)) != expected:
            raise IntegrityError(f"unit accounting changed: {key}")
    execution = _mapping(config.get("execution"), "execution")
    if execution.get("cpu_only") is not True or int(execution.get("parallel_workers", -1)) != 2:
        raise IntegrityError("locked CPU execution contract changed")


@dataclass(frozen=True)
class LockedFollowupProtocol:
    project_root: Path
    config_path: Path
    lock_path: Path
    config: Mapping[str, Any]
    lock: Mapping[str, Any]
    config_file_sha256: str
    lock_file_sha256: str

    @property
    def cells(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.config["cells"])

    def cell(self, *, encoder: str, corruption: str, repetition: int) -> Mapping[str, Any]:
        matches = [
            row
            for row in self.cells
            if row["encoder"] == encoder
            and row["corruption"] == corruption
            and int(row["repetition"]) == int(repetition)
        ]
        if len(matches) != 1:
            raise IntegrityError("requested cell is outside the prospective grid")
        return matches[0]


def load_locked_protocol(
    project_root: str | Path,
    *,
    config_path: str | Path = "configs/paired_recovery_followup/followup_v1.json",
    lock_path: str | Path = "protocol/paired_recovery_followup/STUDY_LOCK_20260914.json",
) -> LockedFollowupProtocol:
    root = Path(project_root).expanduser().resolve()

    def resolve(raw: str | Path, name: str) -> Path:
        candidate = Path(raw)
        result = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / safe_relative_path(candidate)).resolve()
        )
        try:
            result.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"follow-up {name} is outside project root") from error
        if not result.is_file() or result.is_symlink():
            raise IntegrityError(f"follow-up {name} is absent or symbolic: {result}")
        return result

    config_file = resolve(config_path, "config")
    lock_file = resolve(lock_path, "lock")
    config = _mapping(read_json(config_file), "follow-up configuration")
    _validate_config(config)
    lock = _mapping(read_json(lock_file), "follow-up lock")
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise IntegrityError("unsupported follow-up lock schema")
    if lock.get("status") != "FROZEN_BEFORE_FOLLOWUP_OUTCOMES":
        raise IntegrityError("follow-up source lock is not frozen")
    if lock.get("study_id") != EXPECTED_STUDY_ID:
        raise IntegrityError("follow-up lock study identity changed")
    verify_manifest_checksum(lock)
    config_digest = sha256_file(config_file)
    if lock.get("config_file_sha256") != config_digest:
        raise IntegrityError("follow-up lock does not bind the current config")
    sources = _mapping(lock.get("source_files"), "locked source inventory")
    required = {
        "configs/paired_recovery_followup/followup_v1.json",
        "protocol/paired_recovery_followup/FOLLOWUP_PROTOCOL_20260914.md",
        "src/marginal_alignment/paired_recovery_followup/__init__.py",
        "src/marginal_alignment/paired_recovery_followup/calibration.py",
        "src/marginal_alignment/paired_recovery_followup/controls.py",
        "src/marginal_alignment/paired_recovery_followup/protocol.py",
        "src/marginal_alignment/paired_recovery_followup/signed_optimization.py",
        "src/marginal_alignment/paired_recovery_bounds/conformal.py",
        "src/marginal_alignment/paired_recovery_bounds/geometry.py",
        "src/marginal_alignment/paired_recovery_bounds/optimization.py",
        "src/marginal_alignment/paired_recovery_bounds/splitting.py",
        "src/marginal_alignment/campaign/artifacts.py",
        "src/marginal_alignment/campaign/feature_store.py",
        "src/marginal_alignment/campaign/maps.py",
        "src/marginal_alignment/linalg.py",
    }
    if not required.issubset(sources):
        raise IntegrityError(f"follow-up lock omits sources: {sorted(required-set(sources))}")
    for raw_relative, expected_digest in sorted(sources.items()):
        relative = safe_relative_path(str(raw_relative))
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"locked source escapes project root: {relative}") from error
        if not source.is_file() or source.is_symlink():
            raise IntegrityError(f"locked source is absent or symbolic: {relative}")
        if sha256_file(source) != str(expected_digest):
            raise IntegrityError(f"locked source bytes changed: {relative}")
    return LockedFollowupProtocol(
        project_root=root,
        config_path=config_file,
        lock_path=lock_file,
        config=config,
        lock=lock,
        config_file_sha256=config_digest,
        lock_file_sha256=sha256_file(lock_file),
    )


def verify_external_input_manifest(
    protocol: LockedFollowupProtocol, path: str | Path
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
    expected_entries = {}
    for encoder, encoder_spec in protocol.config["encoders"].items():
        for corruption, store_spec in encoder_spec["stores"].items():
            expected_entries[("imagenet_c", encoder, corruption, 5)] = str(
                store_spec["manifest_file_sha256"]
            )
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
        raise IntegrityError("external manifest does not bind all four follow-up stores")
    return manifest


def load_authenticated_store(
    protocol: LockedFollowupProtocol,
    *,
    feature_root: str | Path,
    encoder: str,
    corruption: str,
) -> FeatureStore:
    try:
        encoder_spec = _mapping(protocol.config["encoders"][encoder], "encoder")
        store_spec = _mapping(encoder_spec["stores"][corruption], "store")
    except KeyError as error:
        raise IntegrityError("requested store is outside the follow-up lock") from error
    manifest_path = (
        Path(feature_root).expanduser().resolve()
        / "imagenet_c"
        / encoder
        / corruption
        / "severity_5"
        / "FEATURE_STORE_MANIFEST.json"
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise IntegrityError(f"locked feature-store manifest is absent or symbolic: {manifest_path}")
    if sha256_file(manifest_path) != store_spec["manifest_file_sha256"]:
        raise IntegrityError(f"locked {encoder}/{corruption} store manifest hash changed")
    store = load_feature_store(manifest_path, verify_hashes=True)
    if store.dataset != "imagenet_c" or store.encoder != encoder:
        raise IntegrityError("loaded feature store has the wrong dataset or encoder")
    metadata = _mapping(store.manifest.get("metadata"), "feature-store metadata")
    if metadata.get("corruption") != corruption or int(metadata.get("severity", -1)) != 5:
        raise IntegrityError("loaded feature-store condition identity changed")
    shared = _mapping(protocol.config["shared_store_contract"], "shared store contract")
    expected_arrays = {
        "reference_features": encoder_spec["reference_features_sha256"],
        "reference_ids": shared["endpoint_ids_sha256"],
        "target_ids": shared["endpoint_ids_sha256"],
        "pair_reference_ids": shared["endpoint_ids_sha256"],
        "pair_target_ids": shared["endpoint_ids_sha256"],
        "head_weight": encoder_spec["head_weight_sha256"],
        "head_bias": encoder_spec["head_bias_sha256"],
        "head_fit_ids": shared["head_fit_ids_sha256"],
        "target_features": store_spec["target_features_sha256"],
    }
    for role, expected_hash in expected_arrays.items():
        record = store.records.get(role)
        if not isinstance(record, Mapping) or record.get("sha256") != expected_hash:
            raise IntegrityError(f"locked feature-store array identity changed: {role}")
    dimension = int(encoder_spec["feature_dimension"])
    reference_shape = tuple(int(value) for value in store.records["reference_features"]["shape"])
    target_shape = tuple(int(value) for value in store.records["target_features"]["shape"])
    if reference_shape != (50000, dimension) or target_shape != reference_shape:
        raise IntegrityError("locked endpoint feature shape changed")
    pairing = store.pairing()
    if pairing.provenance.get("kind") != shared["pairing_provenance_kind"]:
        raise IntegrityError("locked pairing provenance changed")
    return store


def runtime_identity() -> dict[str, Any]:
    try:
        import scipy
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("paired-recovery follow-up requires scipy") from error
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
