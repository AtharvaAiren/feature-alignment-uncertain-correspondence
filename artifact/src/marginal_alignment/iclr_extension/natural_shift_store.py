"""Fail-closed preparation of the six Play-3 v2 natural-shift stores.

This module has no optimization or reporting entry point.  It authenticates
raw ImageNet-V2/ImageNet-R acquisitions, constructs target features, restricts
the ImageNet validation endpoint to the declared WNID support after excluding
the frozen-head fit IDs, and upgrades the already-completed ImageNet-Sketch
stores without changing their scientific arrays.  The resulting stores expose
row-level WNIDs for post-hoc clustered summaries, but pairing remains
unavailable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    describe_npy,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_file_record,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import (
    FeatureStore,
    build_native_manifest,
    load_feature_store,
)
from marginal_alignment.campaign.raw_extract import (
    ImagePathDataset,
    extract_dataset,
    load_feature_build_config,
    load_frozen_encoder,
)
from marginal_alignment.iclr_extension.natural_shift import (
    IMAGE_SUFFIXES,
    _assert_encoder_identity,
    _block_sha256,
    _load_locked_base_campaign_feature_store_evidence,
    _save_npy_fsynced,
    _validate_reference_class_identity,
    _validate_store_provenance_binding,
    _write_build_state,
    load_locked_play3_config,
    validate_imagenet_sketch_provenance,
    verify_base_campaign_feature_store,
)


NATURAL_SHIFT_PROVENANCE_SCHEMA = (
    "marginal_alignment.iclr_extension.natural_shift_provenance.v1"
)
STORE_PREPARATION_SCHEMA = (
    "marginal_alignment.iclr_extension.optimization_store_preparation.v1"
)
BUILD_STATE_SCHEMA = (
    "marginal_alignment.iclr_extension.natural_shift_store_build_state.v1"
)
PANEL_INPUT_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_store_preparation_inputs.v1"
)
STORE_MAP_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_store_map.v1"
)
V2_CONFIG_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_gate.v1"
)
OFFICIAL_SOURCE_URLS = {
    "imagenet_v2": "https://github.com/modestyachts/ImageNetV2",
    "imagenet_r": "https://github.com/hendrycks/imagenet-r",
}
DATASET_CONTRACTS = {
    "imagenet_v2": {
        "variant": "matched-frequency-format-val",
        "rows": 10_000,
        "classes": 1_000,
        "rows_per_class": 10,
        "layout": "integer_class_index",
        "intersection": "all_official_imagenet_1k_wnids",
    },
    "imagenet_r": {
        "variant": "official_imagenet-r_renditions",
        "rows": 30_000,
        "classes": 200,
        "layout": "wnid",
        "intersection": "official_imagenet_r_200_wnid_intersection",
    },
}
EXPECTED_IMAGENET_CLASSES = 1_000
EXPECTED_IMAGENET_VAL_ROWS = 50_000
EXPECTED_HEAD_SHOTS_PER_CLASS = 25
EXPECTED_REFERENCE_ROWS = {"imagenet_v2": 25_000, "imagenet_r": 5_000}
EXPECTED_SKETCH_REFERENCE_ROWS = 25_000
EXPECTED_SKETCH_TARGET_ROWS = 50_000
EXPECTED_SKETCH_REFERENCE_ROWS_PER_CLASS = 25


@dataclass(frozen=True)
class OfficialClassIndex:
    path: Path
    file_sha256: str
    wnids_by_index: tuple[str, ...]
    canonical_mapping_sha256: str


@dataclass(frozen=True)
class NaturalShiftDiscovery:
    dataset: str
    root: Path
    paths: tuple[Path, ...]
    ids: np.ndarray
    labels: np.ndarray
    wnids: np.ndarray
    intersection_wnids: tuple[str, ...]
    file_records: tuple[dict[str, Any], ...]
    file_tree_sha256: str
    target_order_sha256: str
    class_counts: dict[str, int]


@dataclass(frozen=True)
class ValidatedNaturalShiftProvenance:
    path: Path
    payload: dict[str, Any]
    archive_path: Path
    class_index: OfficialClassIndex
    discovery: NaturalShiftDiscovery


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _scalar_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise IntegrityError("sample identifiers must be UTF-8") from error
    return str(value)


def _is_wnid(value: str) -> bool:
    return len(value) == 9 and value.startswith("n") and value[1:].isdigit()


def load_official_class_index(path: str | Path) -> OfficialClassIndex:
    """Load the canonical 0..999 ImageNet class-to-WNID JSON mapping."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise IntegrityError(f"official ImageNet class index is absent or unsafe: {source}")
    raw = read_json(source)
    if not isinstance(raw, Mapping) or set(raw) != {
        str(index) for index in range(EXPECTED_IMAGENET_CLASSES)
    }:
        raise IntegrityError("ImageNet class index must contain exactly string keys 0..999")
    wnids: list[str] = []
    canonical: list[dict[str, Any]] = []
    for index in range(EXPECTED_IMAGENET_CLASSES):
        record = raw[str(index)]
        if (
            not isinstance(record, Sequence)
            or isinstance(record, (str, bytes))
            or len(record) < 2
        ):
            raise IntegrityError(f"malformed ImageNet class-index record: {index}")
        wnid = str(record[0])
        label = str(record[1])
        if not _is_wnid(wnid) or not label:
            raise IntegrityError(f"malformed ImageNet class-index identity: {index}")
        wnids.append(wnid)
        canonical.append({"class_index": index, "wnid": wnid, "label": label})
    if len(set(wnids)) != EXPECTED_IMAGENET_CLASSES:
        raise IntegrityError("ImageNet class index contains duplicate WNIDs")
    if wnids != sorted(wnids):
        raise IntegrityError(
            "official ImageNet class-index WNID order is not lexicographic; "
            "the frozen Play-3 v2 intersection contract cannot be satisfied"
        )
    return OfficialClassIndex(
        path=source,
        file_sha256=sha256_file(source),
        wnids_by_index=tuple(wnids),
        canonical_mapping_sha256=sha256_json(canonical),
    )


def _safe_class_directories(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError(f"natural-shift target root is absent or unsafe: {root}")
    entries = list(root.iterdir())
    unexpected = sorted(path.name for path in entries if not path.is_dir())
    symlinks = sorted(path.name for path in entries if path.is_symlink())
    if unexpected or symlinks:
        raise IntegrityError(
            f"target root must contain only real class directories: "
            f"files={unexpected[:3]}, symlinks={symlinks[:3]}"
        )
    return {path.name: path for path in entries}


def _class_image_files(directory: Path) -> list[Path]:
    entries = list(directory.iterdir())
    invalid = sorted(
        path.name
        for path in entries
        if path.is_symlink()
        or not path.is_file()
        or path.suffix.lower() not in IMAGE_SUFFIXES
    )
    if invalid:
        raise IntegrityError(
            f"class directory contains a symlink/non-image/nested entry: "
            f"{directory}: {invalid[:3]}"
        )
    return sorted(entries, key=lambda value: value.name)


def discover_natural_shift(
    dataset: str | Path,
    root: str | Path | OfficialClassIndex,
    class_index: OfficialClassIndex | None = None,
) -> NaturalShiftDiscovery:
    """Discover one exact V2/R image tree in deterministic endpoint order.

    The three-argument signature is intentional.  A defensive type check gives
    a useful error if an older caller accidentally omits ``dataset``.
    """

    if class_index is None or not isinstance(dataset, str):
        raise TypeError("discover_natural_shift requires dataset, root, class_index")
    if dataset not in DATASET_CONTRACTS:
        raise IntegrityError(f"unsupported natural-shift dataset: {dataset}")
    contract = DATASET_CONTRACTS[dataset]
    base = Path(root).expanduser().resolve()
    directories = _safe_class_directories(base)
    wnid_to_index = {
        wnid: index for index, wnid in enumerate(class_index.wnids_by_index)
    }
    if dataset == "imagenet_v2":
        expected_names = {str(index) for index in range(EXPECTED_IMAGENET_CLASSES)}
        if set(directories) != expected_names:
            raise IntegrityError(
                "ImageNet-V2 matched-frequency root must contain exactly integer "
                "class directories 0..999"
            )
        class_rows = [
            (str(index), class_index.wnids_by_index[index], index)
            for index in range(EXPECTED_IMAGENET_CLASSES)
        ]
    else:
        names = set(directories)
        malformed = sorted(name for name in names if not _is_wnid(name))
        outside = sorted(names.difference(wnid_to_index))
        if malformed or outside or len(names) != int(contract["classes"]):
            raise IntegrityError(
                f"ImageNet-R class intersection is invalid: "
                f"malformed={malformed[:3]}, outside={outside[:3]}, classes={len(names)}"
            )
        class_rows = [
            (wnid, wnid, wnid_to_index[wnid]) for wnid in sorted(names)
        ]

    paths: list[Path] = []
    ids: list[str] = []
    labels: list[int] = []
    row_wnids: list[str] = []
    records: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for directory_name, wnid, label in class_rows:
        files = _class_image_files(directories[directory_name])
        expected_per_class = int(contract.get("rows_per_class", -1))
        if dataset == "imagenet_v2" and len(files) != expected_per_class:
            raise IntegrityError(
                f"ImageNet-V2 matched-frequency class {directory_name} has "
                f"{len(files)} rows, expected {expected_per_class}"
            )
        if not files:
            raise IntegrityError(f"natural-shift class is empty: {directory_name}")
        class_counts[wnid] = len(files)
        for path in files:
            relative = f"{directory_name}/{path.name}"
            record = {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            paths.append(path)
            ids.append(relative)
            labels.append(int(label))
            row_wnids.append(wnid)
            records.append(record)
    expected_rows = int(contract["rows"])
    if len(paths) != expected_rows or len(set(ids)) != expected_rows:
        raise IntegrityError(
            f"{dataset} needs {expected_rows} unique target rows; found {len(paths)}"
        )
    intersection = tuple(sorted(class_counts))
    if len(intersection) != int(contract["classes"]):
        raise IntegrityError(f"{dataset} class count changed")
    order_records = [
        {
            "sample_id": sample_id,
            "class_index": int(label),
            "wnid": wnid,
            "image_sha256": record["sha256"],
        }
        for sample_id, label, wnid, record in zip(ids, labels, row_wnids, records)
    ]
    return NaturalShiftDiscovery(
        dataset=dataset,
        root=base,
        paths=tuple(paths),
        ids=np.asarray(ids, dtype=f"<U{max(map(len, ids), default=1)}"),
        labels=np.asarray(labels, dtype=np.int64),
        wnids=np.asarray(row_wnids, dtype="<U9"),
        intersection_wnids=intersection,
        file_records=tuple(records),
        file_tree_sha256=sha256_json(records),
        target_order_sha256=sha256_json(order_records),
        class_counts=class_counts,
    )


def _archive_image_records(
    archive: Path,
    *,
    dataset: str,
    expected_class_names: set[str],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Hash and normalize image members from a ZIP or tar archive."""

    def normalized(member_name: str) -> str | None:
        parts = PurePosixPath(member_name).parts
        for index, part in enumerate(parts):
            if part not in expected_class_names:
                continue
            remainder = parts[index:]
            if len(remainder) != 2:
                continue
            if Path(remainder[-1]).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            return "/".join(remainder)
        return None

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    if zipfile.is_zipfile(archive):
        archive_format = "zip"
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                relative = normalized(info.filename)
                if relative is None:
                    continue
                if info.is_dir() or relative in seen:
                    raise IntegrityError("archive has a duplicate/invalid image member")
                data = handle.read(info)
                records.append({
                    "path": relative,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
                seen.add(relative)
    elif tarfile.is_tarfile(archive):
        archive_format = "tar"
        with tarfile.open(archive, mode="r:*") as handle:
            for info in handle:
                relative = normalized(info.name)
                if relative is None:
                    continue
                if not info.isfile() or relative in seen:
                    raise IntegrityError("archive has a duplicate/invalid image member")
                extracted = handle.extractfile(info)
                if extracted is None:
                    raise IntegrityError(f"cannot read archive member: {info.name}")
                data = extracted.read()
                records.append({
                    "path": relative,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
                seen.add(relative)
    else:
        raise IntegrityError(f"unsupported natural-shift archive format: {archive}")
    records.sort(key=lambda value: value["path"])
    return archive_format, tuple(records)


def freeze_natural_shift_provenance(
    *,
    dataset: str,
    dataset_root: str | Path,
    archive: str | Path,
    class_index_path: str | Path,
    output: str | Path,
    source_url: str | None = None,
) -> Path:
    """Freeze raw acquisition evidence without computing features or outcomes."""

    if dataset not in DATASET_CONTRACTS:
        raise IntegrityError(f"unsupported natural-shift dataset: {dataset}")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite provenance: {destination}")
    root = Path(dataset_root).expanduser().resolve()
    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.is_file() or archive_path.is_symlink():
        raise IntegrityError(f"natural-shift archive is absent or unsafe: {archive_path}")
    if destination == root or root in destination.parents or destination == archive_path:
        raise IntegrityError("provenance output cannot be inside/overwrite an input")
    index = load_official_class_index(class_index_path)
    discovery = discover_natural_shift(dataset, root, index)
    class_names = {
        str(label) if dataset == "imagenet_v2" else wnid
        for label, wnid in enumerate(index.wnids_by_index)
        if dataset == "imagenet_v2" or wnid in discovery.intersection_wnids
    }
    archive_format, archive_records = _archive_image_records(
        archive_path, dataset=dataset, expected_class_names=class_names
    )
    if archive_records != tuple(sorted(discovery.file_records, key=lambda value: value["path"])):
        raise IntegrityError(
            "extracted natural-shift image tree does not exactly match the supplied archive"
        )
    contract = DATASET_CONTRACTS[dataset]
    resolved_url = OFFICIAL_SOURCE_URLS[dataset] if source_url is None else str(source_url)
    if resolved_url != OFFICIAL_SOURCE_URLS[dataset]:
        raise IntegrityError(f"{dataset} provenance must cite its locked official source URL")
    payload = {
        "schema_version": NATURAL_SHIFT_PROVENANCE_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "outcomes_read": False,
        "dataset": dataset,
        "variant": contract["variant"],
        "source_url": resolved_url,
        "archive": {
            "path": str(archive_path),
            "sha256": sha256_file(archive_path),
            "size_bytes": int(archive_path.stat().st_size),
            "format": archive_format,
            "normalized_image_tree_sha256": sha256_json(archive_records),
        },
        "dataset_root": str(root),
        "official_class_index": {
            "path": str(index.path),
            "file_sha256": index.file_sha256,
            "canonical_mapping_sha256": index.canonical_mapping_sha256,
        },
        "layout": contract["layout"],
        "row_count": int(contract["rows"]),
        "class_count": int(contract["classes"]),
        "class_intersection_policy": contract["intersection"],
        "class_intersection_wnids": list(discovery.intersection_wnids),
        "class_index_sha256": sha256_json(list(discovery.intersection_wnids)),
        "class_counts": discovery.class_counts,
        "class_counts_sha256": sha256_json(discovery.class_counts),
        "allowed_image_suffixes": sorted(IMAGE_SUFFIXES),
        "discovery_order": "lexicographic_class_directory_then_lexicographic_basename",
        "file_tree_sha256": discovery.file_tree_sha256,
        "target_order_sha256": discovery.target_order_sha256,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, seal_manifest(payload))
    validate_natural_shift_provenance(destination)
    return destination


def validate_natural_shift_provenance(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
    class_index_path: str | Path | None = None,
) -> ValidatedNaturalShiftProvenance:
    """Rehash every raw byte and structural mapping before extraction."""

    source = Path(path).expanduser().resolve()
    payload = read_json(source)
    if not isinstance(payload, Mapping):
        raise IntegrityError("natural-shift provenance must be a JSON object")
    verify_manifest_checksum(payload)
    dataset = str(payload.get("dataset", ""))
    if dataset not in DATASET_CONTRACTS:
        raise IntegrityError("natural-shift provenance has an unsupported dataset")
    contract = DATASET_CONTRACTS[dataset]
    if (
        payload.get("schema_version") != NATURAL_SHIFT_PROVENANCE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("scientific_result") is not False
        or payload.get("outcomes_read") is not False
        or payload.get("variant") != contract["variant"]
        or payload.get("source_url") != OFFICIAL_SOURCE_URLS[dataset]
        or payload.get("layout") != contract["layout"]
        or int(payload.get("row_count", -1)) != int(contract["rows"])
        or int(payload.get("class_count", -1)) != int(contract["classes"])
        or payload.get("class_intersection_policy") != contract["intersection"]
        or payload.get("allowed_image_suffixes") != sorted(IMAGE_SUFFIXES)
        or payload.get("discovery_order")
        != "lexicographic_class_directory_then_lexicographic_basename"
    ):
        raise IntegrityError("natural-shift provenance contract changed")
    class_record = payload.get("official_class_index")
    archive_record = payload.get("archive")
    if not isinstance(class_record, Mapping) or not isinstance(archive_record, Mapping):
        raise IntegrityError("natural-shift provenance lacks archive/class-index records")
    resolved_class_index = Path(
        str(class_index_path if class_index_path is not None else class_record.get("path", ""))
    ).expanduser().resolve()
    index = load_official_class_index(resolved_class_index)
    if (
        index.file_sha256 != class_record.get("file_sha256")
        or index.canonical_mapping_sha256
        != class_record.get("canonical_mapping_sha256")
    ):
        raise IntegrityError("official ImageNet class-to-WNID mapping changed")
    root = Path(
        str(dataset_root if dataset_root is not None else payload.get("dataset_root", ""))
    ).expanduser().resolve()
    discovery = discover_natural_shift(dataset, root, index)
    intersection = list(discovery.intersection_wnids)
    if (
        payload.get("class_intersection_wnids") != intersection
        or payload.get("class_index_sha256") != sha256_json(intersection)
        or payload.get("class_counts") != discovery.class_counts
        or payload.get("class_counts_sha256") != sha256_json(discovery.class_counts)
        or payload.get("file_tree_sha256") != discovery.file_tree_sha256
        or payload.get("target_order_sha256") != discovery.target_order_sha256
    ):
        raise IntegrityError("natural-shift extracted tree/order changed")
    archive_path = Path(
        str(archive if archive is not None else archive_record.get("path", ""))
    ).expanduser().resolve()
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or not _valid_sha256(archive_record.get("sha256"))
        or sha256_file(archive_path) != archive_record.get("sha256")
        or int(archive_path.stat().st_size) != int(archive_record.get("size_bytes", -1))
    ):
        raise IntegrityError("natural-shift archive bytes changed")
    class_names = {
        str(label) if dataset == "imagenet_v2" else wnid
        for label, wnid in enumerate(index.wnids_by_index)
        if dataset == "imagenet_v2" or wnid in discovery.intersection_wnids
    }
    archive_format, archive_records = _archive_image_records(
        archive_path, dataset=dataset, expected_class_names=class_names
    )
    if (
        archive_record.get("format") != archive_format
        or archive_record.get("normalized_image_tree_sha256")
        != sha256_json(archive_records)
        or archive_records
        != tuple(sorted(discovery.file_records, key=lambda value: value["path"]))
    ):
        raise IntegrityError("natural-shift archive and extracted tree differ")
    return ValidatedNaturalShiftProvenance(
        path=source,
        payload=dict(payload),
        archive_path=archive_path,
        class_index=index,
        discovery=discovery,
    )


def _load_v2_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    value = read_json(source)
    if not isinstance(value, Mapping) or value.get("schema_version") != V2_CONFIG_SCHEMA:
        raise IntegrityError("unsupported Play-3 v2 config")
    if value.get("status") != "READY_TO_FREEZE_BEFORE_NEW_OUTCOMES":
        raise IntegrityError("Play-3 v2 store preparation requires the pre-outcome config")
    datasets = value.get("datasets")
    if not isinstance(datasets, Mapping):
        raise IntegrityError("Play-3 v2 lacks dataset contracts")
    for dataset, contract in DATASET_CONTRACTS.items():
        declaration = datasets.get(dataset)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("variant") != contract["variant"]
            or int(declaration.get("expected_target_rows", -1)) != int(contract["rows"])
            or int(declaration.get("expected_target_classes", -1))
            != int(contract["classes"])
            or declaration.get("class_intersection_policy")
            != contract["intersection"]
            or declaration.get("expected_provenance_schema")
            != NATURAL_SHIFT_PROVENANCE_SCHEMA
        ):
            raise IntegrityError(f"Play-3 v2 {dataset} preparation contract changed")
    return dict(value), source


def _authenticated_reference_store(
    *,
    v1_config_path: str | Path,
    reference_store_path: str | Path,
    encoder_id: str,
) -> tuple[FeatureStore, dict[str, Any], dict[str, Any], Path, Path]:
    config, config_path, lock, repository = load_locked_play3_config(v1_config_path)
    if encoder_id not in config["encoders"]:
        raise IntegrityError(f"encoder is outside completed Play-3 v1: {encoder_id}")
    store = load_feature_store(reference_store_path, verify_hashes=True)
    evidence = _load_locked_base_campaign_feature_store_evidence(
        config, lock, repository
    )
    condition = dict(config["base_campaign_feature_stores"]["natural_reference_condition"])
    binding = verify_base_campaign_feature_store(
        store,
        evidence,
        dataset=str(condition["dataset"]),
        encoder=encoder_id,
        corruption=str(condition["corruption"]),
        severity=int(condition["severity"]),
    )
    feature_config_path = Path(str(config["feature_build_config"]))
    if not feature_config_path.is_absolute():
        feature_config_path = repository / feature_config_path
    return store, binding, config, config_path, feature_config_path.resolve()


def _reference_arrays(
    store: FeatureStore,
    *,
    class_index: OfficialClassIndex,
    intersection_wnids: Sequence[str],
    expected_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    endpoint = store.endpoint("reference")
    if endpoint.labels is None or len(endpoint.features) != EXPECTED_IMAGENET_VAL_ROWS:
        raise IntegrityError("source store lacks the complete labeled ImageNet validation endpoint")
    metadata = store.manifest.get("metadata", {})
    model = metadata.get("model_provenance") if isinstance(metadata, Mapping) else None
    if not isinstance(model, Mapping):
        raise IntegrityError("source store lacks model provenance")
    fit_role = str(model.get("head_fit_ids_role", ""))
    if fit_role not in store.records:
        raise IntegrityError("source store lacks frozen-head fit IDs")
    head_fit_ids = np.asarray(store.array(fit_role))
    head_protocol = model.get("head_protocol")
    if not isinstance(head_protocol, Mapping):
        raise IntegrityError("source store lacks a structured frozen-head protocol")
    shots = int(head_protocol.get("shots_per_class", -1))
    if shots != EXPECTED_HEAD_SHOTS_PER_CLASS:
        raise IntegrityError("Play-3 v2 requires the locked 25-shot-per-class head")
    reference_ids = np.asarray(endpoint.ids)
    reference_labels = np.asarray(endpoint.labels)
    class_identity = _validate_reference_class_identity(
        reference_ids=reference_ids,
        reference_labels=reference_labels,
        head_fit_ids=head_fit_ids,
        class_index=class_index.wnids_by_index,
        shots_per_class=shots,
    )
    allowed = set(map(str, intersection_wnids))
    fit_set = set(head_fit_ids.tolist())
    row_wnids: list[str] = []
    keep: list[bool] = []
    for raw_id in reference_ids.tolist():
        identity = _scalar_text(raw_id)
        parts = PurePosixPath(identity).parts
        if len(parts) != 2 or not _is_wnid(parts[0]):
            raise IntegrityError("ImageNet reference IDs must be WNID/basename keys")
        row_wnids.append(parts[0])
        keep.append(parts[0] in allowed and raw_id not in fit_set)
    mask = np.asarray(keep, dtype=bool)
    if int(np.sum(mask)) != int(expected_rows):
        raise IntegrityError(
            f"reference intersection/head exclusion has {int(np.sum(mask))} rows, "
            f"expected {expected_rows}"
        )
    selected_wnids = np.asarray(row_wnids, dtype="<U9")[mask]
    counts = {wnid: int(np.sum(selected_wnids == wnid)) for wnid in sorted(allowed)}
    if set(counts.values()) != {25}:
        raise IntegrityError("reference endpoint does not leave exactly 25 rows per WNID")
    head = store.manifest.get("head", {})
    if not isinstance(head, Mapping):
        raise IntegrityError("source store lacks frozen-head roles")
    weight_role = str(head.get("weight_role", ""))
    bias_role = str(head.get("bias_role", ""))
    if weight_role not in store.records or bias_role not in store.records:
        raise IntegrityError("source store lacks frozen-head arrays")
    arrays = {
        "reference_features": np.asarray(endpoint.features[mask], dtype=np.float32),
        "reference_ids": np.asarray(reference_ids[mask]),
        "reference_wnids": selected_wnids,
        "head_fit_ids": np.asarray(head_fit_ids),
        "head_weight": np.asarray(store.array(weight_role), dtype=np.float32),
        "head_bias": np.asarray(store.array(bias_role), dtype=np.float32),
    }
    if set(arrays["reference_ids"].tolist()).intersection(
        set(arrays["head_fit_ids"].tolist())
    ):
        raise IntegrityError("head-fit exclusion failed")
    return arrays, {
        "class_identity": class_identity,
        "reference_counts_by_wnid": counts,
        "model_provenance": copy.deepcopy(dict(model)),
        "reference_id_namespace": endpoint.id_namespace,
    }


def _validate_completed_blocks(
    state: Mapping[str, Any],
    *,
    features: np.ndarray,
    labels: np.ndarray,
    expected_labels: np.ndarray,
    rows: int,
    block_rows: int,
) -> int:
    blocks = state.get("completed_blocks")
    if not isinstance(blocks, list):
        raise IntegrityError("natural-shift build state has malformed completed_blocks")
    cursor = 0
    for record in blocks:
        if not isinstance(record, Mapping):
            raise IntegrityError("natural-shift build state has a malformed block")
        stop = min(cursor + int(block_rows), int(rows))
        if int(record.get("start", -1)) != cursor or int(record.get("stop", -1)) != stop:
            raise IntegrityError("natural-shift completed blocks are not an exact prefix")
        if not np.array_equal(labels[cursor:stop], expected_labels[cursor:stop]):
            raise IntegrityError("completed target labels changed")
        if record.get("features_sha256") != _block_sha256(features[cursor:stop]):
            raise IntegrityError("completed target feature bytes changed")
        if record.get("labels_sha256") != _block_sha256(labels[cursor:stop]):
            raise IntegrityError("completed target label bytes changed")
        cursor = stop
    return cursor


def _read_state(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise IntegrityError("natural-shift build state is malformed")
    verify_manifest_checksum(value)
    if value.get("schema_version") != BUILD_STATE_SCHEMA:
        raise IntegrityError("natural-shift build-state schema changed")
    return dict(value)


def _build_state_matches_binding(
    state: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    fingerprint: str,
) -> bool:
    """Compare a persisted binding through its canonical JSON identity.

    Model configuration dictionaries may contain integer keys in memory. JSON
    necessarily restores those keys as strings, so direct Python-dictionary
    equality can reject the state immediately after it was written even though
    its sealed content is unchanged. The build fingerprint is already defined
    as the canonical JSON hash of the binding; use that identity on both sides.
    """

    persisted = state.get("build_binding")
    return (
        state.get("build_fingerprint") == fingerprint
        and isinstance(persisted, Mapping)
        and sha256_json(persisted) == fingerprint
        and sha256_json(binding) == fingerprint
    )


def _remove_structural_label_checkpoint(path: Path) -> None:
    """Ensure a preparation-only label array cannot enter a published store."""

    if not path.is_file():
        raise IntegrityError("structural label checkpoint is absent before removal")
    path.unlink()
    if path.exists():  # pragma: no cover - defensive filesystem check
        raise IntegrityError("structural label checkpoint survived removal")


def _store_metadata(
    *,
    dataset: str,
    encoder_id: str,
    provenance_binding: Mapping[str, Any],
    provenance_manifest_sha256: str,
    intersection_wnids: Sequence[str],
    reference_source_manifest_sha256: str,
    source_binding: Mapping[str, Any],
    model_provenance: Mapping[str, Any],
    class_identity: Mapping[str, Any],
    arrays_root: Path,
    target_extraction: Mapping[str, Any],
    v2_config_sha256: str,
) -> dict[str, Any]:
    wnids = list(map(str, intersection_wnids))
    intersection_hash = sha256_json(wnids)
    preparation = {
        "schema_version": STORE_PREPARATION_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "outcomes_read": False,
        "dataset": dataset,
        "encoder": encoder_id,
        "head_fit_disjoint_from_reference": True,
        "head_fit_ids_sha256": sha256_file(arrays_root / "head_fit_ids.npy"),
        "reference_ids_sha256": sha256_file(arrays_root / "reference_ids.npy"),
        "reference_wnids_sha256": sha256_file(arrays_root / "reference_wnids.npy"),
        "target_wnids_sha256": sha256_file(arrays_root / "target_wnids.npy"),
        "class_intersection_wnids_sha256": intersection_hash,
        "reference_source_manifest_sha256": reference_source_manifest_sha256,
        "target_extraction_manifest_sha256": sha256_json(target_extraction),
        "dataset_provenance_manifest_sha256": provenance_manifest_sha256,
        "play3_v2_config_sha256": v2_config_sha256,
    }
    model = copy.deepcopy(dict(model_provenance))
    model["head_fit_ids_sha256"] = preparation["head_fit_ids_sha256"]
    model["reference_split"] = (
        "imagenet_val_declared_wnid_intersection_excluding_frozen_head_fit_ids"
    )
    return {
        "natural_shift": True,
        "endpoint_labels_not_published": True,
        "model_provenance": model,
        "class_intersection": {
            "policy": DATASET_CONTRACTS.get(dataset, {}).get(
                "intersection", "all_official_imagenet_1k_wnids"
            ),
            "wnids": wnids,
            "wnids_sha256": intersection_hash,
            "reference_endpoint_restricted": True,
            "target_endpoint_restricted": True,
            "reference_wnids_role": "reference_wnids",
            "target_wnids_role": "target_wnids",
        },
        "optimization_gate_preparation": preparation,
        "source_campaign_reference_store": dict(source_binding),
        "reference_class_identity": dict(class_identity),
        "target_extraction_manifest": dict(target_extraction),
        "store_preparation_source_manifest_sha256": (
            reference_source_manifest_sha256
        ),
        **dict(provenance_binding),
    }


def build_natural_shift_feature_store(
    *,
    dataset: str,
    v2_config_path: str | Path,
    v1_config_path: str | Path,
    provenance_path: str | Path,
    reference_store_path: str | Path,
    destination: str | Path,
    encoder_id: str,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
    class_index_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int | None = None,
    workers: int | None = None,
    checkpoint_rows: int = 5_000,
    work_dir: str | Path | None = None,
) -> Path:
    """Build one authenticated V2/R feature store into a fresh destination."""

    if dataset not in DATASET_CONTRACTS:
        raise IntegrityError("fresh extraction is supported only for ImageNet-V2/R")
    v2_config, v2_path = _load_v2_config(v2_config_path)
    if encoder_id not in v2_config["encoders"]:
        raise IntegrityError(f"encoder is outside Play-3 v2: {encoder_id}")
    validated = validate_natural_shift_provenance(
        provenance_path,
        dataset_root=dataset_root,
        archive=archive,
        class_index_path=class_index_path,
    )
    if validated.discovery.dataset != dataset:
        raise IntegrityError("requested dataset and provenance dataset differ")
    reference_store, source_binding, _, _, feature_config_path = (
        _authenticated_reference_store(
            v1_config_path=v1_config_path,
            reference_store_path=reference_store_path,
            encoder_id=encoder_id,
        )
    )
    expected_reference_rows = int(v2_config["datasets"][dataset][
        "expected_reference_rows_after_intersection_and_head_fit_exclusion"
    ])
    if expected_reference_rows != EXPECTED_REFERENCE_ROWS[dataset]:
        raise IntegrityError("Play-3 v2 reference row contract changed")
    reference_arrays, reference_evidence = _reference_arrays(
        reference_store,
        class_index=validated.class_index,
        intersection_wnids=validated.discovery.intersection_wnids,
        expected_rows=expected_reference_rows,
    )
    feature_config = load_feature_build_config(feature_config_path)
    if encoder_id not in feature_config["encoders"]:
        raise IntegrityError("encoder is absent from the locked feature-build config")
    encoder = load_frozen_encoder(
        encoder_id, feature_config["encoders"][encoder_id], device=device
    )
    _assert_encoder_identity(reference_store=reference_store, encoder_evidence=encoder.evidence)
    contract = v2_config["encoder_contracts"][encoder_id]
    if (
        encoder.evidence.get("model_id") != contract["model_id"]
        or encoder.evidence.get("feature_layer") != contract["feature_layer"]
        or encoder.evidence.get("pooling") != contract["pooling"]
        or int(encoder.expected_dimension) != int(contract["expected_dimension"])
    ):
        raise IntegrityError("loaded encoder differs from Play-3 v2")
    resolved_batch = int(feature_config["batch_size"] if batch_size is None else batch_size)
    resolved_workers = int(feature_config["workers"] if workers is None else workers)
    block_rows = int(checkpoint_rows)
    if resolved_batch < 1 or resolved_workers < 0 or block_rows < 1:
        raise IntegrityError("invalid extraction worker/batch/checkpoint setting")
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature store: {output}")
    if any(output == protected or protected in output.parents for protected in (
        validated.discovery.root, reference_store.root.resolve()
    )):
        raise IntegrityError("feature-store output overlaps an input tree")
    staging = (
        output.with_name(f".{output.name}.work")
        if work_dir is None
        else Path(work_dir).expanduser().resolve()
    )
    if any(staging == protected or protected in staging.parents for protected in (
        output, validated.discovery.root, reference_store.root.resolve()
    )):
        raise IntegrityError("feature-store work directory overlaps an input/output tree")
    rows = int(DATASET_CONTRACTS[dataset]["rows"])
    binding = {
        "schema_version": BUILD_STATE_SCHEMA,
        "dataset": dataset,
        "encoder": encoder_id,
        "destination": str(output),
        "v2_config_sha256": sha256_file(v2_path),
        "v1_config_sha256": sha256_file(Path(v1_config_path).expanduser().resolve()),
        "provenance_manifest_sha256": sha256_file(validated.path),
        "reference_store_manifest_sha256": sha256_file(reference_store.manifest_path),
        "source_binding": source_binding,
        "encoder_evidence": encoder.evidence,
        "rows": rows,
        "dimension": int(encoder.expected_dimension),
        "batch_size": resolved_batch,
        "workers": resolved_workers,
        "checkpoint_rows": block_rows,
        "device": str(device),
    }
    fingerprint = sha256_json(binding)
    state_path = staging / "BUILD_STATE.json"
    target_partial = staging / "target_features.partial.npy"
    target_final = staging / "target_features.npy"
    labels_partial = staging / "target_labels.partial.npy"
    labels_final = staging / "target_labels.npy"
    static_roles = tuple(reference_arrays) + ("target_ids", "target_wnids")
    if not staging.exists():
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        for role, value in reference_arrays.items():
            _save_npy_fsynced(staging / f"{role}.npy", value)
        _save_npy_fsynced(staging / "target_ids.npy", validated.discovery.ids)
        _save_npy_fsynced(staging / "target_wnids.npy", validated.discovery.wnids)
        features = np.lib.format.open_memmap(
            target_partial,
            mode="w+",
            dtype=np.float32,
            shape=(rows, int(encoder.expected_dimension)),
        )
        features[:] = np.nan
        features.flush()
        del features
        labels = np.lib.format.open_memmap(
            labels_partial, mode="w+", dtype=np.int64, shape=(rows,)
        )
        labels[:] = -1
        labels.flush()
        del labels
        _write_build_state(state_path, {
            "schema_version": BUILD_STATE_SCHEMA,
            "status": "IN_PROGRESS",
            "scientific_result": False,
            "outcomes_read": False,
            "build_fingerprint": fingerprint,
            "build_binding": binding,
            "phase": "extracting",
            "static_files": {
                role: describe_npy(staging / f"{role}.npy", relative_to=staging)
                for role in static_roles
            },
            "completed_blocks": [],
        })
    elif not staging.is_dir() or not state_path.is_file():
        raise IntegrityError("existing work directory lacks a sealed build state")
    state = _read_state(state_path)
    if (
        not _build_state_matches_binding(
            state, binding=binding, fingerprint=fingerprint
        )
        or state.get("phase") not in {"extracting", "extracted", "published_ready"}
    ):
        raise IntegrityError("natural-shift checkpoint does not match locked inputs")
    records = state.get("static_files")
    if not isinstance(records, Mapping) or set(records) != set(static_roles):
        raise IntegrityError("natural-shift checkpoint static inventory changed")
    for record in records.values():
        if not isinstance(record, Mapping):
            raise IntegrityError("malformed static file record")
        verify_file_record(staging, record, verify_hash=True)
    active_features = target_partial if target_partial.is_file() else target_final
    active_labels = labels_partial if labels_partial.is_file() else labels_final
    if (
        state["phase"] in {"extracted", "published_ready"}
        and not active_labels.is_file()
    ):
        # A crash after preparation-only labels were removed but before the
        # directory rename must not force feature recomputation.  Recreate the
        # labels only from the revalidated raw class structure, verify their
        # recorded block digests below, and remove them again before publish.
        _save_npy_fsynced(labels_final, validated.discovery.labels)
        active_labels = labels_final
    features = np.load(active_features, mmap_mode="r+", allow_pickle=False)
    labels = np.load(active_labels, mmap_mode="r+", allow_pickle=False)
    if features.shape != (rows, int(encoder.expected_dimension)) or features.dtype != np.float32:
        raise IntegrityError("target feature checkpoint shape/dtype changed")
    if labels.shape != (rows,) or labels.dtype != np.int64:
        raise IntegrityError("target label checkpoint shape/dtype changed")
    cursor = _validate_completed_blocks(
        state,
        features=features,
        labels=labels,
        expected_labels=validated.discovery.labels,
        rows=rows,
        block_rows=block_rows,
    )
    if state["phase"] == "extracting":
        while cursor < rows:
            stop = min(cursor + block_rows, rows)
            extract_dataset(
                encoder,
                ImagePathDataset(
                    validated.discovery.paths[cursor:stop],
                    validated.discovery.labels[cursor:stop],
                ),
                features,
                destination_offset=cursor,
                batch_size=resolved_batch,
                workers=resolved_workers,
                label_destination=labels,
                label_destination_offset=cursor,
            )
            features.flush()
            labels.flush()
            if not np.array_equal(
                labels[cursor:stop], validated.discovery.labels[cursor:stop]
            ):
                raise IntegrityError("target labels changed during extraction")
            state["completed_blocks"] = [*state["completed_blocks"], {
                "start": cursor,
                "stop": stop,
                "features_sha256": _block_sha256(features[cursor:stop]),
                "labels_sha256": _block_sha256(labels[cursor:stop]),
            }]
            cursor = stop
            state["phase"] = "extracted" if cursor == rows else "extracting"
            _write_build_state(state_path, state)
    if cursor != rows or not np.array_equal(labels, validated.discovery.labels):
        raise IntegrityError("natural-shift extraction is incomplete")
    features.flush()
    labels.flush()
    del features
    del labels
    if target_partial.is_file():
        os.replace(target_partial, target_final)
    if labels_partial.is_file():
        os.replace(labels_partial, labels_final)
    target_extraction = {
        "schema_version": "marginal_alignment.iclr_extension.target_extraction.v1",
        "status": "PASS",
        "dataset": dataset,
        "encoder": encoder_id,
        "scientific_result": False,
        "outcomes_read": False,
        "provenance_manifest_sha256": sha256_file(validated.path),
        "target_features": describe_npy(target_final, relative_to=staging),
        "target_structural_labels_sha256": _block_sha256(
            validated.discovery.labels
        ),
        "target_ids": describe_npy(staging / "target_ids.npy", relative_to=staging),
        "target_wnids": describe_npy(staging / "target_wnids.npy", relative_to=staging),
        "target_order_sha256": validated.discovery.target_order_sha256,
        "encoder_evidence_sha256": sha256_json(encoder.evidence),
        "completed_blocks": list(state["completed_blocks"]),
    }
    provenance_binding = {
        "natural_shift_provenance": {
            "schema_version": NATURAL_SHIFT_PROVENANCE_SCHEMA,
            "status": "PASS",
            "dataset": dataset,
            "variant": DATASET_CONTRACTS[dataset]["variant"],
            "manifest_sha256": sha256_file(validated.path),
            "file_tree_sha256": validated.discovery.file_tree_sha256,
            "class_index_sha256": sha256_json(
                list(validated.discovery.intersection_wnids)
            ),
            "official_class_to_wnid_mapping_file_sha256": (
                validated.class_index.file_sha256
            ),
            "official_class_to_wnid_mapping_sha256": (
                validated.class_index.canonical_mapping_sha256
            ),
            "archive_sha256": validated.payload["archive"]["sha256"],
            "target_order_sha256": validated.discovery.target_order_sha256,
            "row_count": rows,
        }
    }
    metadata = _store_metadata(
        dataset=dataset,
        encoder_id=encoder_id,
        provenance_binding=provenance_binding,
        provenance_manifest_sha256=sha256_file(validated.path),
        intersection_wnids=validated.discovery.intersection_wnids,
        reference_source_manifest_sha256=sha256_file(reference_store.manifest_path),
        source_binding=source_binding,
        model_provenance=reference_evidence["model_provenance"],
        class_identity=reference_evidence["class_identity"],
        arrays_root=staging,
        target_extraction=target_extraction,
        v2_config_sha256=sha256_file(v2_path),
    )
    manifest = build_native_manifest(
        staging,
        dataset=dataset,
        encoder=encoder_id,
        arrays={
            role: f"{role}.npy"
            for role in (
                "reference_features", "reference_ids", "reference_wnids",
                "target_features",
                "target_ids", "target_wnids", "head_fit_ids", "head_weight",
                "head_bias",
            )
        },
        id_provenance={
            "reference": {
                "verified": True,
                "kind": "imagenet_validation_sample_key_declared_wnid_intersection_excluding_head_fit",
                "source": str(reference_store.manifest_path),
                "namespace": reference_evidence["reference_id_namespace"],
            },
            "target": {
                "verified": True,
                "kind": "authenticated_official_relative_image_path",
                "source": str(validated.path),
                "namespace": f"{dataset}:{DATASET_CONTRACTS[dataset]['variant']}:relative_path",
            },
        },
        pairing={
            "status": "unavailable",
            "reason": "independent natural-shift images have no sample correspondence to ImageNet validation",
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata=metadata,
    )
    # Structural labels were needed to verify/extract rows, but the scientific
    # gate consumes neither labels nor external accuracy.  Do not publish them
    # as endpoint roles, so FeatureStore.endpoint() cannot eagerly expose them
    # before assignment construction.
    _remove_structural_label_checkpoint(labels_final)
    load_feature_store(manifest, verify_hashes=True)
    state["phase"] = "published_ready"
    state["status"] = "PASS"
    state["feature_store_manifest_sha256"] = sha256_file(manifest)
    _write_build_state(state_path, state)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    published = output / "FEATURE_STORE_MANIFEST.json"
    load_feature_store(published, verify_hashes=True)
    return published


def upgrade_imagenet_sketch_feature_store(
    *,
    v2_config_path: str | Path,
    v1_config_path: str | Path,
    provenance_path: str | Path,
    source_store_path: str | Path,
    destination: str | Path,
    encoder_id: str,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
) -> Path:
    """Repackage one authenticated completed Sketch store for the v2 contract."""

    v2_config, v2_path = _load_v2_config(v2_config_path)
    if encoder_id not in v2_config["encoders"]:
        raise IntegrityError(f"encoder is outside Play-3 v2: {encoder_id}")
    provenance = validate_imagenet_sketch_provenance(
        provenance_path, dataset_root=dataset_root, archive=archive
    )
    source = load_feature_store(source_store_path, verify_hashes=True)
    if source.dataset != "imagenet_sketch" or source.encoder != encoder_id:
        raise IntegrityError("Sketch source store has the wrong dataset/encoder")
    encoder_contract = v2_config["encoder_contracts"][encoder_id]
    source_model = source.manifest.get("metadata", {}).get("model_provenance")
    if (
        not isinstance(source_model, Mapping)
        or source_model.get("model_id") != encoder_contract["model_id"]
        or source_model.get("feature_layer") != encoder_contract["feature_layer"]
        or source_model.get("pooling") != encoder_contract["pooling"]
        or source.endpoint("reference").features.shape[1]
        != int(encoder_contract["expected_dimension"])
        or source.endpoint("target").features.shape[1]
        != int(encoder_contract["expected_dimension"])
    ):
        raise IntegrityError("completed Sketch encoder identity differs from Play-3 v2")
    v1_config, _, v1_lock, repository = load_locked_play3_config(v1_config_path)
    evidence = _load_locked_base_campaign_feature_store_evidence(
        v1_config, v1_lock, repository
    )
    condition = v1_config["base_campaign_feature_stores"]["natural_reference_condition"]
    source_binding = _validate_store_provenance_binding(
        source,
        provenance,
        source_evidence=evidence,
        natural_reference_condition=condition,
    )
    reference = source.endpoint("reference")
    target = source.endpoint("target")
    if reference.labels is None or target.labels is None:
        raise IntegrityError("completed Sketch store lacks structural labels")
    if (
        len(reference.features) != EXPECTED_SKETCH_REFERENCE_ROWS
        or len(target.features) != EXPECTED_SKETCH_TARGET_ROWS
    ):
        raise IntegrityError("completed Sketch store has the wrong endpoint counts")
    target_ids = tuple(_scalar_text(value) for value in target.ids.tolist())
    expected_target_ids = tuple(map(str, provenance.discovery.sample_ids.tolist()))
    if target_ids != expected_target_ids:
        raise IntegrityError("completed Sketch target order differs from raw provenance")
    if not np.array_equal(np.asarray(target.labels), provenance.discovery.labels):
        raise IntegrityError("completed Sketch labels differ from raw provenance")
    wnids = tuple(sorted(provenance.discovery.wnids))
    if source.manifest["metadata"]["imagenet_sketch_provenance"].get(
        "class_index_sha256"
    ) != sha256_json(list(wnids)):
        raise IntegrityError("completed Sketch class index differs from v2 intersection")
    target_parts = [PurePosixPath(value).parts for value in target_ids]
    reference_parts = [
        PurePosixPath(_scalar_text(value)).parts for value in reference.ids.tolist()
    ]
    if any(len(parts) != 2 or not _is_wnid(parts[0]) for parts in (
        *target_parts, *reference_parts
    )):
        raise IntegrityError("completed Sketch IDs are not WNID/basename keys")
    target_wnids = np.asarray([parts[0] for parts in target_parts], dtype="<U9")
    reference_wnids = np.asarray([parts[0] for parts in reference_parts], dtype="<U9")
    if set(target_wnids.tolist()) != set(wnids) or set(reference_wnids.tolist()) != set(wnids):
        raise IntegrityError("completed Sketch store does not cover all official WNIDs")
    if any(
        int(np.sum(reference_wnids == wnid))
        != EXPECTED_SKETCH_REFERENCE_ROWS_PER_CLASS
        for wnid in wnids
    ):
        raise IntegrityError("completed Sketch reference does not have 25 rows per WNID")
    model = source.manifest["metadata"]["model_provenance"]
    fit_role = str(model["head_fit_ids_role"])
    head = source.manifest["head"]
    arrays = {
        "reference_features": np.asarray(reference.features),
        "reference_ids": np.asarray(reference.ids),
        "reference_wnids": reference_wnids,
        "target_features": np.asarray(target.features),
        "target_ids": np.asarray(target.ids),
        "target_wnids": target_wnids,
        "head_fit_ids": np.asarray(source.array(fit_role)),
        "head_weight": np.asarray(source.array(str(head["weight_role"]))),
        "head_bias": np.asarray(source.array(str(head["bias_role"]))),
    }
    if set(arrays["reference_ids"].tolist()).intersection(
        set(arrays["head_fit_ids"].tolist())
    ):
        raise IntegrityError("completed Sketch store contains head-fit reference IDs")
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Sketch v2 store: {output}")
    if any(output == protected or protected in output.parents for protected in (
        source.root.resolve(), provenance.dataset_root.resolve()
    )):
        raise IntegrityError("Sketch v2 destination overlaps an input tree")
    staging = output.with_name(f".{output.name}.work")
    if staging.exists():
        raise IntegrityError("Sketch repackaging requires a fresh work directory")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    for role, value in arrays.items():
        _save_npy_fsynced(staging / f"{role}.npy", value)
    target_extraction = {
        "schema_version": "marginal_alignment.iclr_extension.target_extraction.v1",
        "status": "PASS",
        "dataset": "imagenet_sketch",
        "encoder": encoder_id,
        "scientific_result": False,
        "outcomes_read": False,
        "mode": "authenticated_completed_v1_store_repackaging_without_feature_recompute",
        "source_store_manifest_sha256": sha256_file(source.manifest_path),
        "provenance_manifest_sha256": sha256_file(provenance.manifest_path),
        "target_features": describe_npy(staging / "target_features.npy", relative_to=staging),
        "target_structural_labels_sha256": _block_sha256(
            np.asarray(target.labels, dtype=np.int64)
        ),
        "target_ids": describe_npy(staging / "target_ids.npy", relative_to=staging),
        "target_wnids": describe_npy(staging / "target_wnids.npy", relative_to=staging),
    }
    source_sketch_binding = source.manifest["metadata"][
        "imagenet_sketch_provenance"
    ]
    if not isinstance(source_sketch_binding, Mapping):
        raise IntegrityError("completed Sketch store lacks provenance binding")
    # State the v2 raw-provenance contract explicitly.  The completed v1 store
    # is a separate, hash-bound extraction source; it is not relabeled as raw
    # provenance and its scientific result files are never read here.
    sketch_binding = {
        "schema_version": provenance.manifest["schema_version"],
        "manifest_sha256": sha256_file(provenance.manifest_path),
        "file_tree_sha256": provenance.discovery.file_tree_sha256,
        "class_index_sha256": provenance.manifest["class_index_sha256"],
        "row_count": len(provenance.discovery.paths),
        "completed_v1_source_store_manifest_sha256": sha256_file(
            source.manifest_path
        ),
        "completed_v1_source_binding_sha256": sha256_json(
            dict(source_sketch_binding)
        ),
        "duplicate_digest_groups": int(
            provenance.discovery.duplicate_digest_groups
        ),
        "duplicate_rows_beyond_first": int(
            provenance.discovery.duplicate_rows_beyond_first
        ),
        "indexed_atom_policy": "preserve_all_rows_and_unique_sample_ids",
    }
    metadata = _store_metadata(
        dataset="imagenet_sketch",
        encoder_id=encoder_id,
        provenance_binding={"imagenet_sketch_provenance": sketch_binding},
        provenance_manifest_sha256=sha256_file(provenance.manifest_path),
        intersection_wnids=wnids,
        reference_source_manifest_sha256=str(
            source_binding["actual_manifest_sha256"]
        ),
        source_binding=source_binding,
        model_provenance=model,
        class_identity=source.manifest["metadata"]["reference_class_identity"],
        arrays_root=staging,
        target_extraction=target_extraction,
        v2_config_sha256=sha256_file(v2_path),
    )
    manifest = build_native_manifest(
        staging,
        dataset="imagenet_sketch",
        encoder=encoder_id,
        arrays={role: f"{role}.npy" for role in arrays},
        id_provenance=copy.deepcopy(dict(source.manifest["id_provenance"])),
        pairing={
            "status": "unavailable",
            "reason": "ImageNet validation and ImageNet-Sketch are independent image collections",
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata=metadata,
    )
    load_feature_store(manifest, verify_hashes=True)
    staging.rename(output)
    published = output / "FEATURE_STORE_MANIFEST.json"
    load_feature_store(published, verify_hashes=True)
    return published


def write_optimization_store_map(
    *,
    v2_config_path: str | Path,
    output_root: str | Path,
    destination: str | Path,
) -> Path:
    """Write and semantically validate the exact six-store map."""

    config, _ = _load_v2_config(v2_config_path)
    root = Path(output_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite store map: {output}")
    records = []
    for dataset in config["datasets"]:
        for encoder in config["encoders"]:
            manifest = root / dataset / encoder / "FEATURE_STORE_MANIFEST.json"
            if not manifest.is_file():
                raise IntegrityError(f"prepared store is missing: {manifest}")
            records.append({
                "dataset": dataset,
                "encoder": encoder,
                "manifest": str(manifest.resolve()),
            })
    write_json_atomic(output, {"schema_version": STORE_MAP_SCHEMA, "stores": records})
    try:
        from scripts.iclr_extension.run_play3_optimization_gate import _validate_store_map

        _validate_store_map(output, config, verify_hashes=True)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def prepare_six_store_panel(input_map_path: str | Path) -> Path:
    """Validate all inputs first, then resumably prepare the complete panel."""

    source = Path(input_map_path).expanduser().resolve()
    value = read_json(source)
    if not isinstance(value, Mapping) or value.get("schema_version") != PANEL_INPUT_SCHEMA:
        raise IntegrityError("unsupported Play-3 v2 store-preparation input map")
    required = {
        "schema_version", "v2_config", "v1_config", "official_class_index",
        "reference_stores", "datasets", "output_root", "device", "batch_size",
        "workers", "checkpoint_rows",
    }
    if set(value) != required:
        raise IntegrityError("store-preparation input map has missing/undeclared fields")
    v2_config, _ = _load_v2_config(str(value["v2_config"]))
    encoders = tuple(map(str, v2_config["encoders"]))
    reference_stores = value["reference_stores"]
    datasets = value["datasets"]
    if not isinstance(reference_stores, Mapping) or set(reference_stores) != set(encoders):
        raise IntegrityError("input map needs one authenticated reference store per encoder")
    if not isinstance(datasets, Mapping) or set(datasets) != {
        "imagenet_sketch", "imagenet_v2", "imagenet_r"
    }:
        raise IntegrityError("input map needs exactly Sketch, V2, and R")
    # Preflight every raw input before loading a model or creating an output.
    sketch = datasets["imagenet_sketch"]
    if not isinstance(sketch, Mapping) or set(sketch) != {
        "provenance", "dataset_root", "archive", "source_stores"
    }:
        raise IntegrityError("Sketch input declaration is malformed")
    if not isinstance(sketch["source_stores"], Mapping) or set(sketch["source_stores"]) != set(encoders):
        raise IntegrityError("Sketch input needs one completed v1 store per encoder")
    validated_sketch = validate_imagenet_sketch_provenance(
        str(sketch["provenance"]),
        dataset_root=str(sketch["dataset_root"]),
        archive=str(sketch["archive"]),
    )
    for dataset in ("imagenet_v2", "imagenet_r"):
        declaration = datasets[dataset]
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "provenance", "dataset_root", "archive"
        }:
            raise IntegrityError(f"{dataset} input declaration is malformed")
        validated = validate_natural_shift_provenance(
            str(declaration["provenance"]),
            dataset_root=str(declaration["dataset_root"]),
            archive=str(declaration["archive"]),
            class_index_path=str(value["official_class_index"]),
        )
        if validated.discovery.dataset != dataset:
            raise IntegrityError(f"{dataset} declaration points to the wrong provenance")
    for encoder in encoders:
        _authenticated_reference_store(
            v1_config_path=str(value["v1_config"]),
            reference_store_path=str(reference_stores[encoder]),
            encoder_id=encoder,
        )
        source_store = load_feature_store(str(sketch["source_stores"][encoder]), verify_hashes=True)
        if source_store.dataset != "imagenet_sketch" or source_store.encoder != encoder:
            raise IntegrityError("Sketch completed-store declaration is wrong")
        v1_config, _, v1_lock, repository = load_locked_play3_config(
            str(value["v1_config"])
        )
        source_evidence = _load_locked_base_campaign_feature_store_evidence(
            v1_config, v1_lock, repository
        )
        _validate_store_provenance_binding(
            source_store,
            validated_sketch,
            source_evidence=source_evidence,
            natural_reference_condition=v1_config[
                "base_campaign_feature_stores"
            ]["natural_reference_condition"],
        )
    output_root = Path(str(value["output_root"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for encoder in encoders:
        sketch_output = output_root / "imagenet_sketch" / encoder
        if (sketch_output / "FEATURE_STORE_MANIFEST.json").is_file():
            existing = load_feature_store(sketch_output, verify_hashes=True)
            if existing.dataset != "imagenet_sketch" or existing.encoder != encoder:
                raise IntegrityError("completed Sketch v2 destination has wrong identity")
        else:
            upgrade_imagenet_sketch_feature_store(
                v2_config_path=str(value["v2_config"]),
                v1_config_path=str(value["v1_config"]),
                provenance_path=str(sketch["provenance"]),
                source_store_path=str(sketch["source_stores"][encoder]),
                destination=sketch_output,
                encoder_id=encoder,
                dataset_root=str(sketch["dataset_root"]),
                archive=str(sketch["archive"]),
            )
        for dataset in ("imagenet_v2", "imagenet_r"):
            declaration = datasets[dataset]
            destination = output_root / dataset / encoder
            if (destination / "FEATURE_STORE_MANIFEST.json").is_file():
                load_feature_store(destination, verify_hashes=True)
                continue
            build_natural_shift_feature_store(
                dataset=dataset,
                v2_config_path=str(value["v2_config"]),
                v1_config_path=str(value["v1_config"]),
                provenance_path=str(declaration["provenance"]),
                reference_store_path=str(reference_stores[encoder]),
                destination=destination,
                encoder_id=encoder,
                dataset_root=str(declaration["dataset_root"]),
                archive=str(declaration["archive"]),
                class_index_path=str(value["official_class_index"]),
                device=str(value["device"]),
                batch_size=int(value["batch_size"]),
                workers=int(value["workers"]),
                checkpoint_rows=int(value["checkpoint_rows"]),
            )
    store_map = output_root / "play3_optimization_store_map.json"
    if store_map.exists():
        raise FileExistsError(
            "all stores were prepared but an existing store map prevents publication"
        )
    return write_optimization_store_map(
        v2_config_path=str(value["v2_config"]),
        output_root=output_root,
        destination=store_map,
    )
