"""Prepare converter-ready legacy roots from scientifically usable server caches.

The shared server cache is treated as read-only.  Large arrays are hard-linked
into a separate preparation root, while V2-only head, identity, block, and
metadata sidecars are generated there.  This avoids rewriting an old cache or
invalidating a seal created by another project.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import IntegrityError, sha256_file, sha256_json
from .cache_import import (
    CIFAR100_C_CORRUPTIONS,
    IMAGENET_C_CORRUPTIONS,
    seal_legacy_arrays,
)
from .feature_protocol import (
    MODEL_EVIDENCE_SCHEMA,
    PREPARED_SOURCE_SCHEMA,
    atomic_json,
    atomic_npy,
    build_model_provenance,
    canonical_block_index,
    fit_frozen_readout,
    head_fit_positions,
    is_sha256,
    load_feature_build_config,
    read_json_object,
    validate_declared_blocks,
    validate_model_evidence,
)


PREPARATION_MANIFEST = "PREPARED_SOURCE_MANIFEST.json"
MODEL_EVIDENCE_FILES = ("MODEL_EVIDENCE.json", "model_evidence.json")


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object(path)


def _source_metadata(root: Path) -> dict[str, Any]:
    return _load_optional_json(root / "metadata.json")


def _safe_relative_file(root: Path, value: Any, *, name: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise IntegrityError(f"{name} must be a safe path inside {root}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _choose_array(
    root: Path,
    metadata: Mapping[str, Any],
    metadata_key: str,
    candidates: Sequence[str],
) -> Path:
    declared = metadata.get(metadata_key)
    if declared:
        return _safe_relative_file(root, declared, name=metadata_key)
    found = [root / name for name in candidates if (root / name).is_file()]
    if len(found) != 1:
        raise IntegrityError(
            f"{root}: declare {metadata_key}; unambiguous candidates were not found "
            f"(found={[path.name for path in found]})"
        )
    return found[0]


def _npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _matrix(path: Path, *, rows: int | None = None, dimension: int | None = None) -> np.ndarray:
    value = _npy(path)
    if value.ndim != 2:
        raise IntegrityError(f"feature array must be rank two: {path} {value.shape}")
    if rows is not None and len(value) != int(rows):
        raise IntegrityError(f"feature row count mismatch: {path} {len(value)} != {rows}")
    if dimension is not None and value.shape[1] != int(dimension):
        raise IntegrityError(
            f"feature dimension mismatch: {path} {value.shape[1]} != {dimension}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise IntegrityError(f"feature array is not floating point: {path} {value.dtype}")
    return value


def _labels(path: Path, *, rows: int) -> np.ndarray:
    value = _npy(path)
    if value.shape != (int(rows),) or not np.issubdtype(value.dtype, np.integer):
        raise IntegrityError(f"label array has invalid shape/dtype: {path} {value.shape} {value.dtype}")
    return value


def _identity_array(path: Path, *, rows: int, unique: bool) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1 or len(value) != int(rows) or value.dtype == object:
        raise IntegrityError(f"identity array has invalid shape/dtype: {path}")
    if unique and len(np.unique(value)) != len(value):
        raise IntegrityError(f"identity array contains duplicates: {path}")
    return value


def _link_verified(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise IntegrityError(f"prepared destination conflicts with source: {destination}")
    else:
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno == errno.EXDEV:
                raise IntegrityError(
                    "source/prepared roots are on different filesystems; refusing a massive "
                    f"implicit copy ({source} -> {destination})"
                ) from error
            raise
    return {
        "source": str(source), "destination": str(destination),
        "sha256": source_hash, "bytes": source.stat().st_size,
        "hard_linked": source.stat().st_dev == destination.stat().st_dev
        and source.stat().st_ino == destination.stat().st_ino,
    }


def _model_evidence(
    reference_root: Path, metadata: Mapping[str, Any], *,
    encoder: str, encoder_spec: Mapping[str, Any], dimension: int,
) -> dict[str, Any]:
    raw: dict[str, Any] | None = None
    for name in MODEL_EVIDENCE_FILES:
        path = reference_root / name
        if path.is_file():
            raw = read_json_object(path)
            break
    if raw is None:
        candidate = metadata.get("model_provenance")
        if isinstance(candidate, Mapping):
            raw = dict(candidate)
            raw.setdefault("schema_version", MODEL_EVIDENCE_SCHEMA)
            raw.setdefault("expected_dimension", dimension)
    if raw is None:
        # Some shared-cache versions placed the exact same structured block in
        # manifest.json.  Accept only that block; a feature-array checksum or a
        # free-form model name is not checkpoint/preprocessing evidence.
        manifest = _load_optional_json(reference_root / "manifest.json")
        candidate = manifest.get("model_provenance")
        if isinstance(candidate, Mapping):
            raw = dict(candidate)
            raw.setdefault("schema_version", MODEL_EVIDENCE_SCHEMA)
            raw.setdefault("expected_dimension", dimension)
    if raw is None:
        raise IntegrityError(
            f"{reference_root}: no exact model evidence; provide MODEL_EVIDENCE.json "
            "or rebuild features from locally pinned weights"
        )
    evidence = validate_model_evidence(raw, encoder=encoder, encoder_spec=encoder_spec)
    if int(evidence["expected_dimension"]) != int(dimension):
        raise IntegrityError("model evidence dimension disagrees with cached features")
    return evidence


def _source_blocks(
    target_root: Path,
    metadata: Mapping[str, Any],
    *, dataset: str, total_rows: int,
) -> list[dict[str, Any]]:
    raw = metadata.get("blocks") or metadata.get("block_index")
    if not isinstance(raw, list):
        sidecar = target_root / "block_index.json"
        if sidecar.is_file():
            # Two historical cache writers used equivalent encodings: the V2
            # object wrapper {"blocks": [...]} and a bare JSON list carrying a
            # sequential integer ``block`` field.  Accept both, but normalize
            # the legacy form only after proving exact sequential order.  This
            # is a schema adapter, not an inference of corruption order.
            sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(sidecar_value, Mapping):
                raw = sidecar_value.get("blocks")
            elif isinstance(sidecar_value, list):
                raw = sidecar_value
            else:
                raise IntegrityError(f"{sidecar}: block index must be an object or list")
    if isinstance(raw, list) and raw and not {
        "start", "stop"
    }.issubset(raw[0]):
        rows_per_block = 10_000 if dataset == "cifar100_c" else 50_000
        normalized = []
        for expected_block, record in enumerate(raw):
            if not isinstance(record, Mapping):
                raise IntegrityError("legacy block index has a malformed record")
            if int(record.get("block", -1)) != expected_block:
                raise IntegrityError(
                    "legacy block index is not an exact sequential block ordering"
                )
            if "corruption" not in record or "severity" not in record:
                raise IntegrityError("legacy block index omits corruption or severity")
            normalized.append({
                "corruption": str(record["corruption"]),
                "severity": int(record["severity"]),
                "start": expected_block * rows_per_block,
                "stop": (expected_block + 1) * rows_per_block,
            })
        raw = normalized
    if not isinstance(raw, list):
        order = metadata.get("corruption_order")
        severity_order = metadata.get("severity_order")
        layout = metadata.get("block_layout")
        if not isinstance(order, list) or severity_order != [1, 2, 3, 4, 5]:
            raise IntegrityError(
                f"{target_root}: explicit blocks are absent; corruption_order and exact "
                "severity_order=[1,2,3,4,5] are required"
            )
        if layout not in {"corruption_major_severity_minor", "corruption_then_severity"}:
            raise IntegrityError(f"{target_root}: block_layout does not prove row order")
        raw = canonical_block_index(
            dataset, corruptions=[str(value) for value in order],
            severities=(1, 2, 3, 4, 5),
        )
    required_corruptions = (
        CIFAR100_C_CORRUPTIONS if dataset == "cifar100_c" else IMAGENET_C_CORRUPTIONS
    )
    required = [(name, severity) for name in required_corruptions for severity in (1, 3, 5)]
    return validate_declared_blocks(
        raw, dataset=dataset, total_rows=total_rows, require_conditions=required,
    )


def audit_reuse_pair(
    *, source_root: str | Path, dataset: str, encoder: str,
    config: Mapping[str, Any], hash_large_arrays: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable readiness report for one source-cache pair."""

    base = Path(source_root).expanduser().resolve()
    clean_name = "cifar100" if dataset == "cifar100_c" else "imagenet"
    target_name = "cifar100c" if dataset == "cifar100_c" else "imagenet_c"
    reference_root = base / clean_name / encoder
    target_root = base / target_name / encoder
    report: dict[str, Any] = {
        "dataset": dataset, "encoder": encoder,
        "reference_root": str(reference_root), "target_root": str(target_root),
        "status": "BLOCKED", "checks": {}, "blockers": [],
    }
    try:
        dataset_spec = config["datasets"][dataset]
        encoder_spec = config["encoders"][encoder]
        reference_metadata = _source_metadata(reference_root)
        target_metadata = _source_metadata(target_root)
        prepared_manifest_path = reference_root / PREPARATION_MANIFEST
        if prepared_manifest_path.is_file():
            prepared_manifest = read_json_object(prepared_manifest_path)
            if prepared_manifest.get("schema_version") != PREPARED_SOURCE_SCHEMA:
                raise IntegrityError("prepared source manifest schema mismatch")
            if prepared_manifest.get("status") != "PASS":
                raise IntegrityError("prepared source manifest is not PASS")
            if prepared_manifest.get("feature_build_config_sha256") != sha256_json(config):
                raise IntegrityError("prepared source was built from a different feature config")
        if dataset == "cifar100_c":
            fit_features = _choose_array(reference_root, reference_metadata, "head_fit_feature_file", ("train_features.npy",))
            fit_labels = _choose_array(reference_root, reference_metadata, "head_fit_label_file", ("train_labels.npy",))
            reference_features = _choose_array(reference_root, reference_metadata, "reference_feature_file", ("test_features.npy",))
            reference_labels = _choose_array(reference_root, reference_metadata, "reference_label_file", ("test_labels.npy",))
        else:
            fit_features = reference_features = _choose_array(
                reference_root, reference_metadata, "reference_feature_file",
                ("val_features.npy", "test_features.npy"),
            )
            fit_labels = reference_labels = _choose_array(
                reference_root, reference_metadata, "reference_label_file",
                ("val_labels.npy", "test_labels.npy"),
            )
        target_features = _choose_array(target_root, target_metadata, "target_feature_file", ("test_features.npy",))
        target_labels = _choose_array(target_root, target_metadata, "target_label_file", ("test_labels.npy",))
        expected_dimension = int(encoder_spec["expected_dimension"])
        reference_matrix = _matrix(
            reference_features, rows=int(dataset_spec["rows_per_condition"]),
            dimension=expected_dimension,
        )
        _labels(reference_labels, rows=len(reference_matrix))
        fit_matrix = _matrix(fit_features, dimension=expected_dimension)
        _labels(fit_labels, rows=len(fit_matrix))
        target_matrix = _matrix(target_features, dimension=expected_dimension)
        _labels(target_labels, rows=len(target_matrix))
        blocks = _source_blocks(
            target_root, target_metadata, dataset=dataset, total_rows=len(target_matrix),
        )
        _model_evidence(
            reference_root, reference_metadata, encoder=encoder,
            encoder_spec=encoder_spec, dimension=expected_dimension,
        )
        if dataset == "cifar100_c":
            if target_metadata.get("pairing_provenance") not in {
                "official_cifar100c_canonical_test_index", "cifar100c_canonical_test_index",
            }:
                raise IntegrityError("source metadata does not prove official CIFAR-100-C indexing")
        else:
            reference_ids_path = _safe_relative_file(
                reference_root, reference_metadata.get("sample_ids_file"), name="sample_ids_file"
            )
            target_ids_path = _safe_relative_file(
                target_root, target_metadata.get("sample_ids_file"), name="sample_ids_file"
            )
            _identity_array(reference_ids_path, rows=len(reference_matrix), unique=True)
            target_ids = _identity_array(target_ids_path, rows=len(target_matrix), unique=False)
            for block in blocks:
                values = target_ids[int(block["start"]):int(block["stop"])]
                if len(np.unique(values)) != len(values):
                    raise IntegrityError(f"target IDs duplicate inside block {block}")
            if target_metadata.get("sample_id_kind") not in {
                "wnid_basename", "relative_image_path",
            }:
                raise IntegrityError("ImageNet target sample_id_kind is unverified")
        report["checks"] = {
            "array_shapes": "PASS", "declared_blocks": "PASS",
            "model_evidence": "PASS", "identity_evidence": "PASS",
        }
        report["arrays"] = {
            "head_fit_features": str(fit_features), "head_fit_labels": str(fit_labels),
            "reference_features": str(reference_features),
            "reference_labels": str(reference_labels),
            "target_features": str(target_features), "target_labels": str(target_labels),
        }
        if hash_large_arrays:
            report["array_sha256"] = {
                key: sha256_file(path) for key, path in report["arrays"].items()
            }
        report["block_count"] = len(blocks)
        report["status"] = "READY"
    except (IntegrityError, FileNotFoundError, KeyError, OSError, ValueError) as error:
        report["blockers"].append(str(error))
    return report


def prepare_reused_pair(
    *, source_root: str | Path, destination_root: str | Path,
    dataset: str, encoder: str, config: Mapping[str, Any], seal: bool = False,
) -> dict[str, Any]:
    """Hard-link a qualifying source pair and fit the locked clean readout."""

    readiness = audit_reuse_pair(
        source_root=source_root, dataset=dataset, encoder=encoder,
        config=config, hash_large_arrays=False,
    )
    if readiness["status"] != "READY":
        raise IntegrityError(f"source cache cannot be reused: {readiness['blockers']}")
    source_base = Path(source_root).expanduser().resolve()
    destination_base = Path(destination_root).expanduser().resolve()
    if source_base == destination_base:
        raise IntegrityError("prepared root must differ from the read-only source cache root")
    clean_name = "cifar100" if dataset == "cifar100_c" else "imagenet"
    target_name = "cifar100c" if dataset == "cifar100_c" else "imagenet_c"
    source_reference = source_base / clean_name / encoder
    source_target = source_base / target_name / encoder
    destination_reference = destination_base / clean_name / encoder
    destination_target = destination_base / target_name / encoder
    destination_reference.mkdir(parents=True, exist_ok=True)
    destination_target.mkdir(parents=True, exist_ok=True)
    if (destination_reference / "SOURCE_ARRAY_SHA256.json").exists() or (
        destination_target / "SOURCE_ARRAY_SHA256.json"
    ).exists():
        raise IntegrityError(
            "prepared destination is already sealed; use a fresh destination root rather than mutate it"
        )

    dataset_spec = config["datasets"][dataset]
    encoder_spec = config["encoders"][encoder]
    reference_metadata = _source_metadata(source_reference)
    target_metadata = _source_metadata(source_target)
    arrays = {key: Path(value) for key, value in readiness["arrays"].items()}
    dimension = int(encoder_spec["expected_dimension"])
    evidence = _model_evidence(
        source_reference, reference_metadata, encoder=encoder,
        encoder_spec=encoder_spec, dimension=dimension,
    )
    config_sha = sha256_json(config)
    links = []
    if dataset == "cifar100_c":
        reference_feature_name, reference_label_name = "test_features.npy", "test_labels.npy"
        fit_feature_name, fit_label_name = "train_features.npy", "train_labels.npy"
    else:
        reference_feature_name, reference_label_name = "test_features.npy", "test_labels.npy"
        fit_feature_name, fit_label_name = reference_feature_name, reference_label_name
    links.append(_link_verified(arrays["reference_features"], destination_reference / reference_feature_name))
    links.append(_link_verified(arrays["reference_labels"], destination_reference / reference_label_name))
    if dataset == "cifar100_c":
        links.append(_link_verified(arrays["head_fit_features"], destination_reference / fit_feature_name))
        links.append(_link_verified(arrays["head_fit_labels"], destination_reference / fit_label_name))
    links.append(_link_verified(arrays["target_features"], destination_target / "test_features.npy"))
    links.append(_link_verified(arrays["target_labels"], destination_target / "test_labels.npy"))

    fit_features = _npy(destination_reference / fit_feature_name)
    fit_labels = _npy(destination_reference / fit_label_name)
    protocol = dict(dataset_spec["head_protocol"])
    positions = head_fit_positions(fit_labels, protocol)
    if dataset == "cifar100_c":
        head_ids = positions.astype(np.int64, copy=False)
        fit_namespace = "cifar100:train:canonical_index"
        reference_namespace = "cifar100:test:canonical_index"
    else:
        source_reference_ids = _safe_relative_file(
            source_reference, reference_metadata["sample_ids_file"], name="sample_ids_file"
        )
        links.append(_link_verified(source_reference_ids, destination_reference / "sample_ids.npy"))
        reference_ids = _identity_array(
            destination_reference / "sample_ids.npy", rows=len(fit_features), unique=True,
        )
        head_ids = np.asarray(reference_ids[positions])
        reference_namespace = f"imagenet:val:{target_metadata['sample_id_kind']}"
        fit_namespace = reference_namespace
        source_target_ids = _safe_relative_file(
            source_target, target_metadata["sample_ids_file"], name="sample_ids_file"
        )
        links.append(_link_verified(source_target_ids, destination_target / "sample_ids.npy"))

    weight, bias, readout_report = fit_frozen_readout(
        fit_features, fit_labels, positions,
        realization=config["readout_realization"], seed=int(protocol["seed"]),
    )
    atomic_npy(destination_reference / "head_weight.npy", weight)
    atomic_npy(destination_reference / "head_bias.npy", bias)
    atomic_npy(destination_reference / "head_fit_ids.npy", head_ids)
    atomic_json(destination_reference / "head_realization.json", readout_report)
    head_fit_sha = sha256_file(destination_reference / "head_fit_ids.npy")
    model_provenance = build_model_provenance(
        evidence=evidence, head_protocol=protocol,
        head_fit_ids_file="head_fit_ids.npy", head_fit_ids_sha256=head_fit_sha,
        head_fit_id_namespace=fit_namespace,
        reference_id_namespace=reference_namespace,
        reference_split=str(dataset_spec["reference_split"]),
        head_fit_split=str(dataset_spec["head_fit_split"]),
        readout_report_file="head_realization.json",
        feature_build_config_sha256=config_sha,
    )
    prepared_reference_metadata: dict[str, Any] = {
        "schema_version": PREPARED_SOURCE_SCHEMA,
        "dataset": clean_name, "encoder": encoder,
        "reference_feature_file": reference_feature_name,
        "reference_label_file": reference_label_name,
        "head_fit_feature_file": fit_feature_name,
        "head_fit_label_file": fit_label_name,
        "head_weight_file": "head_weight.npy", "head_bias_file": "head_bias.npy",
        "model_provenance": model_provenance,
        "source_cache_root": str(source_reference),
    }
    if dataset == "imagenet_c":
        prepared_reference_metadata.update({
            "sample_ids_file": "sample_ids.npy",
            "sample_id_kind": target_metadata["sample_id_kind"],
        })
    blocks = _source_blocks(
        source_target, target_metadata, dataset=dataset,
        total_rows=len(_npy(destination_target / "test_features.npy")),
    )
    prepared_target_metadata: dict[str, Any] = {
        "schema_version": PREPARED_SOURCE_SCHEMA,
        "dataset": dataset, "encoder": encoder,
        "target_feature_file": "test_features.npy",
        "target_label_file": "test_labels.npy",
        "blocks": blocks,
        "source_cache_root": str(source_target),
    }
    if dataset == "cifar100_c":
        prepared_target_metadata["pairing_provenance"] = (
            "official_cifar100c_canonical_test_index"
        )
    else:
        prepared_target_metadata.update({
            "sample_ids_file": "sample_ids.npy",
            "sample_id_kind": target_metadata["sample_id_kind"],
        })
    atomic_json(destination_reference / "metadata.json", prepared_reference_metadata)
    atomic_json(destination_target / "metadata.json", prepared_target_metadata)
    atomic_json(destination_target / "block_index.json", {"blocks": blocks})
    manifest = {
        "schema_version": PREPARED_SOURCE_SCHEMA,
        "status": "PASS", "dataset": dataset, "encoder": encoder,
        "source_reference": str(source_reference), "source_target": str(source_target),
        "destination_reference": str(destination_reference),
        "destination_target": str(destination_target),
        "feature_build_config_sha256": config_sha,
        "model_evidence": evidence,
        "head_fit_ids_sha256": head_fit_sha,
        "head_weight_sha256": sha256_file(destination_reference / "head_weight.npy"),
        "head_bias_sha256": sha256_file(destination_reference / "head_bias.npy"),
        "readout_report": readout_report,
        "source_links": links,
        "block_index_sha256": sha256_file(destination_target / "block_index.json"),
    }
    atomic_json(destination_reference / PREPARATION_MANIFEST, manifest)
    atomic_json(destination_target / PREPARATION_MANIFEST, manifest)
    if seal:
        seal_legacy_arrays(destination_reference)
        seal_legacy_arrays(destination_target)
    return manifest


def audit_all_reuse(
    *, source_root: str | Path, config_path: str | Path,
    datasets: Sequence[str] = ("cifar100_c", "imagenet_c"),
    encoders: Sequence[str] = ("dinov3", "clip", "vitb16"),
) -> dict[str, Any]:
    config = load_feature_build_config(config_path)
    rows = [
        audit_reuse_pair(
            source_root=source_root, dataset=dataset, encoder=encoder, config=config,
        )
        for dataset in datasets for encoder in encoders
    ]
    return {
        "schema_version": "marginal_alignment.campaign.cache_reuse_audit.v1",
        "status": "READY" if all(row["status"] == "READY" for row in rows) else "BLOCKED",
        "source_root": str(Path(source_root).expanduser().resolve()),
        "feature_build_config": str(Path(config_path).expanduser().resolve()),
        "feature_build_config_sha256": sha256_file(config_path),
        "rows": rows,
    }
