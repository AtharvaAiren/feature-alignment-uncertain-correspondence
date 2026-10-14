"""Fail-closed ImageNet-Sketch inputs for the Play-3 ceiling experiment.

Dataset acquisition is intentionally outside this module.  A scientific run
accepts only an already-present archive, extracted image tree, class index, and
manifest whose bytes all revalidate.  The estimator reads only the two feature
roles and frozen readout roles from a pairing-unavailable V2 store.  When
requested, a separate post-hoc evaluator loads target labels only after every
map and both correspondence worlds have been frozen; it never exposes labels
or sample-pair metadata to the estimator.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
import platform
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    canonical_json_bytes,
    describe_npy,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
    verify_file_record,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import FeatureStore
from marginal_alignment.campaign.feature_store import build_native_manifest, load_feature_store
from marginal_alignment.campaign.raw_extract import (
    ImagePathDataset,
    extract_dataset,
    load_feature_build_config,
    load_frozen_encoder,
)
from marginal_alignment.iclr_extension.ceiling import (
    CorrespondenceCeilingProtocol,
    _canonical_rows,
    _squared_distance_cost,
    construct_two_world_witness,
    fit_witness_geometry,
)
from marginal_alignment.iclr_extension.frontier import (
    MARGINAL_FRONTIER_METHODS,
    FrozenMarginalFrontier,
    fit_marginal_frontier,
    frozen_frontier_predictions,
)


IMAGENET_SKETCH_PROVENANCE_SCHEMA = (
    "marginal_alignment.iclr_extension.imagenet_sketch_provenance.v2"
)
NATURAL_SHIFT_RESULT_SCHEMA = (
    "marginal_alignment.iclr_extension.natural_shift_ceiling.v2"
)
OFFICIAL_IMAGENET_SKETCH_URL = "https://github.com/HaohanWang/ImageNet-Sketch"
IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})
EXPECTED_CLASSES = 1000
EXPECTED_IMAGES_PER_CLASS = 50
EXPECTED_ROWS = EXPECTED_CLASSES * EXPECTED_IMAGES_PER_CLASS
_WNID = re.compile(r"n[0-9]{8}")
_BUILD_STATE_SCHEMA = (
    "marginal_alignment.iclr_extension.imagenet_sketch_build_state.v1"
)
PLAY3_REQUIRED_IMPORTED_DEPENDENCIES = frozenset({
    # Importing a project submodule executes both package initializers and
    # their eager imports before resolving the requested module.
    "src/marginal_alignment/__init__.py",
    "src/marginal_alignment/ceiling.py",
    "src/marginal_alignment/transforms.py",
    "src/marginal_alignment/linalg.py",
    "src/marginal_alignment/campaign/__init__.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/cache_import.py",
    "src/marginal_alignment/campaign/feature_protocol.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/campaign/method_registry.py",
    "src/marginal_alignment/campaign/raw_extract.py",
    "src/marginal_alignment/iclr_extension/__init__.py",
    "src/marginal_alignment/iclr_extension/aggregation.py",
    "src/marginal_alignment/iclr_extension/ceiling.py",
    "src/marginal_alignment/iclr_extension/frontier.py",
    "src/marginal_alignment/iclr_extension/natural_shift.py",
    "scripts/iclr_extension/__init__.py",
    "scripts/iclr_extension/create_natural_shift_provenance.py",
    "scripts/iclr_extension/run_play3.py",
    "scripts/iclr_extension/run_play3_panel.py",
})
PLAY3_SOURCE_FILES = PLAY3_REQUIRED_IMPORTED_DEPENDENCIES | frozenset({
    # Tests are byte-bound with the implementation so the executed contract
    # cannot silently differ from the reviewed one.
    "tests/iclr_extension/__init__.py",
    "tests/iclr_extension/test_aggregation.py",
    "tests/iclr_extension/test_ceiling.py",
    "tests/iclr_extension/test_natural_shift.py",
    "tests/iclr_extension/test_play3_panel.py",
})


@dataclass(frozen=True)
class ImageNetSketchDiscovery:
    root: Path
    paths: tuple[Path, ...]
    labels: np.ndarray
    sample_ids: np.ndarray
    wnids: tuple[str, ...]
    class_counts: dict[str, int]
    source_class_counts: dict[str, int]
    source_rows: int
    source_file_tree_sha256: str
    file_tree_sha256: str
    duplicate_digest_groups: int
    duplicate_rows_beyond_first: int


@dataclass(frozen=True)
class ValidatedImageNetSketchProvenance:
    manifest_path: Path
    manifest: dict[str, Any]
    dataset_root: Path
    archive_path: Path
    discovery: ImageNetSketchDiscovery


@dataclass(frozen=True)
class ImageNetSketchBuildPlan:
    """Validated extraction inputs; this object performs no model compute."""

    provenance_manifest: Path
    provenance_manifest_sha256: str
    dataset_root: Path
    archive_path: Path
    target_paths: tuple[Path, ...]
    target_labels: np.ndarray
    target_ids: np.ndarray
    wnids: tuple[str, ...]
    feature_store_pairing: dict[str, str]
    feature_store_metadata_binding: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": "imagenet_sketch",
            "rows": int(len(self.target_paths)),
            "classes": int(len(self.wnids)),
            "provenance_manifest": str(self.provenance_manifest),
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "dataset_root": str(self.dataset_root),
            "archive_path": str(self.archive_path),
            "pairing": dict(self.feature_store_pairing),
            "metadata_binding": dict(self.feature_store_metadata_binding),
        }


@dataclass(frozen=True)
class BaseCampaignFeatureStoreEvidence:
    """Exact feature-store identities sealed by the completed base campaign."""

    manifest_path: Path
    manifest_file_sha256: str
    external_input_sha256: str
    feature_store_count: int
    manifest_sha256_by_key: Mapping[tuple[str, str, str, int], str]

    def expected_manifest_sha256(
        self, *, dataset: str, encoder: str, corruption: str, severity: int
    ) -> str:
        key = (str(dataset), str(encoder), str(corruption), int(severity))
        try:
            return str(self.manifest_sha256_by_key[key])
        except KeyError as error:
            raise IntegrityError(
                f"feature-store condition is absent from the base campaign: {key}"
            ) from error


@dataclass(frozen=True)
class UnpairedStoreInputs:
    reference_features: np.ndarray
    target_features: np.ndarray
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    dataset: str
    encoder: str


def _validated_wnids(values: Sequence[str]) -> tuple[str, ...]:
    wnids = tuple(str(value) for value in values)
    if len(wnids) != EXPECTED_CLASSES or len(set(wnids)) != len(wnids):
        raise IntegrityError(
            f"ImageNet class index needs {EXPECTED_CLASSES} unique WNIDs"
        )
    malformed = [value for value in wnids if _WNID.fullmatch(value) is None]
    if malformed:
        raise IntegrityError(f"malformed ImageNet WNIDs: {malformed[:3]}")
    return wnids


def discover_reference_wnids(reference_root: str | Path) -> tuple[str, ...]:
    """Recover the exact class order used by the existing ImageFolder protocol."""

    base = Path(reference_root).expanduser().resolve()
    if not base.is_dir():
        raise IntegrityError(f"ImageNet reference root is absent: {base}")
    directories = sorted(path for path in base.iterdir() if path.is_dir())
    if any(path.is_symlink() for path in directories):
        raise IntegrityError("ImageNet reference class directories cannot be symlinks")
    return _validated_wnids([path.name for path in directories])


def discover_imagenet_sketch(
    dataset_root: str | Path,
    *,
    expected_wnids: Sequence[str],
) -> ImageNetSketchDiscovery:
    """Hash the source tree and select the first 50 basenames per WNID.

    The official ImageNet-Sketch archive is not exactly balanced: some WNIDs
    contain more than 50 images.  The scientific endpoint is therefore the
    outcome-blind, deterministic lexicographic 50-per-WNID subset.  Both the
    complete extracted source tree and the selected endpoint are hashed.
    """

    base = Path(dataset_root).expanduser().resolve()
    if not base.is_dir():
        raise IntegrityError(f"ImageNet-Sketch root is absent: {base}")
    wnids = _validated_wnids(expected_wnids)
    directories = {path.name: path for path in base.iterdir() if path.is_dir()}
    if set(directories) != set(wnids):
        missing = sorted(set(wnids).difference(directories))[:5]
        extra = sorted(set(directories).difference(wnids))[:5]
        raise IntegrityError(
            f"ImageNet-Sketch WNID mismatch: missing={missing}, extra={extra}"
        )
    if any(path.is_symlink() for path in directories.values()):
        raise IntegrityError("ImageNet-Sketch class directories cannot be symlinks")

    paths: list[Path] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    class_counts: dict[str, int] = {}
    source_class_counts: dict[str, int] = {}
    file_records: list[dict[str, Any]] = []
    source_file_records: list[dict[str, Any]] = []
    content_counts: dict[str, int] = {}
    for class_index, wnid in enumerate(wnids):
        directory = directories[wnid]
        regular_files = sorted(path for path in directory.iterdir() if path.is_file())
        if any(path.is_symlink() for path in regular_files):
            raise IntegrityError(f"ImageNet-Sketch image files cannot be symlinks: {wnid}")
        unsupported = [
            path.name for path in regular_files
            if path.suffix.lower() not in IMAGE_SUFFIXES
        ]
        if unsupported:
            raise IntegrityError(
                f"unsupported files in ImageNet-Sketch class {wnid}: {unsupported[:3]}"
            )
        if len(regular_files) < EXPECTED_IMAGES_PER_CLASS:
            raise IntegrityError(
                f"ImageNet-Sketch class {wnid} needs "
                f"at least {EXPECTED_IMAGES_PER_CLASS} images; "
                f"found {len(regular_files)}"
            )
        source_class_counts[wnid] = len(regular_files)
        for path in regular_files:
            relative = f"{wnid}/{path.name}"
            source_file_records.append({
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            })
        selected_files = regular_files[:EXPECTED_IMAGES_PER_CLASS]
        class_counts[wnid] = len(selected_files)
        for path in selected_files:
            relative = f"{wnid}/{path.name}"
            digest = sha256_file(path)
            size = int(path.stat().st_size)
            paths.append(path.resolve())
            labels.append(class_index)
            sample_ids.append(relative)
            file_records.append({
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
            })
            content_counts[digest] = content_counts.get(digest, 0) + 1
    if len(paths) != EXPECTED_ROWS or len(set(sample_ids)) != EXPECTED_ROWS:
        raise IntegrityError(
            f"ImageNet-Sketch needs {EXPECTED_ROWS} unique paths; found {len(paths)}"
        )
    tree_digest = hashlib.sha256()
    for record in file_records:
        tree_digest.update(canonical_json_bytes(record))
    source_tree_digest = hashlib.sha256()
    for record in source_file_records:
        source_tree_digest.update(canonical_json_bytes(record))
    duplicate_counts = [count for count in content_counts.values() if count > 1]
    maximum_id_length = max(map(len, sample_ids), default=1)
    return ImageNetSketchDiscovery(
        root=base,
        paths=tuple(paths),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=f"<U{maximum_id_length}"),
        wnids=wnids,
        class_counts=class_counts,
        source_class_counts=source_class_counts,
        source_rows=len(source_file_records),
        source_file_tree_sha256=source_tree_digest.hexdigest(),
        file_tree_sha256=tree_digest.hexdigest(),
        duplicate_digest_groups=int(len(duplicate_counts)),
        duplicate_rows_beyond_first=int(sum(count - 1 for count in duplicate_counts)),
    )


def _archive_tree_fingerprint(
    archive: Path, *, expected_wnids: Sequence[str]
) -> dict[str, Any]:
    """Hash the complete archive tree and its deterministic 50-per-class subset."""

    ordered_wnids = _validated_wnids(expected_wnids)
    wnids = set(ordered_wnids)
    records: dict[str, dict[str, Any]] = {}

    def register(member_name: str, size: int, payload: bytes) -> None:
        path = PurePosixPath(member_name)
        parts = path.parts
        positions = [index for index, value in enumerate(parts) if value in wnids]
        if not positions:
            return
        if len(positions) != 1 or positions[0] != len(parts) - 2:
            raise IntegrityError(
                f"archive image is not a direct WNID child: {member_name}"
            )
        suffix = PurePosixPath(parts[-1]).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise IntegrityError(
                f"unsupported archive member under WNID: {member_name}"
            )
        relative = f"{parts[-2]}/{parts[-1]}"
        if relative in records:
            raise IntegrityError(f"duplicate normalized archive member: {relative}")
        if len(payload) != int(size):
            raise IntegrityError(f"archive member size changed while reading: {member_name}")
        records[relative] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": int(size),
        }

    if zipfile.is_zipfile(archive):
        archive_format = "zip"
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if member.is_dir():
                    continue
                mode = int(member.external_attr >> 16)
                if mode and stat.S_ISLNK(mode):
                    raise IntegrityError(f"archive contains a symbolic link: {member.filename}")
                with handle.open(member, "r") as stream:
                    payload = stream.read()
                register(member.filename, member.file_size, payload)
    elif tarfile.is_tarfile(archive):
        archive_format = "tar"
        with tarfile.open(archive, mode="r:*") as handle:
            for member in handle.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise IntegrityError(f"archive contains a non-regular member: {member.name}")
                stream = handle.extractfile(member)
                if stream is None:
                    raise IntegrityError(f"cannot read archive member: {member.name}")
                payload = stream.read()
                register(member.name, member.size, payload)
    else:
        raise IntegrityError("ImageNet-Sketch archive must be a readable zip or tar file")
    by_wnid: dict[str, list[str]] = {wnid: [] for wnid in ordered_wnids}
    for relative in records:
        by_wnid[PurePosixPath(relative).parts[0]].append(relative)
    source_class_counts = {
        wnid: len(by_wnid[wnid]) for wnid in ordered_wnids
    }
    undersized = {
        wnid: count for wnid, count in source_class_counts.items()
        if count < EXPECTED_IMAGES_PER_CLASS
    }
    if undersized:
        raise IntegrityError(
            "archive WNIDs have fewer than the required 50 images: "
            f"{dict(list(undersized.items())[:5])}"
        )
    selected_relatives = [
        relative
        for wnid in ordered_wnids
        for relative in sorted(by_wnid[wnid])[:EXPECTED_IMAGES_PER_CLASS]
    ]
    if len(selected_relatives) != EXPECTED_ROWS:
        raise IntegrityError("archive selection did not produce exactly 50,000 rows")
    source_digest = hashlib.sha256()
    for relative in sorted(records):
        source_digest.update(canonical_json_bytes(records[relative]))
    selected_digest = hashlib.sha256()
    for relative in selected_relatives:
        selected_digest.update(canonical_json_bytes(records[relative]))
    return {
        "format": archive_format,
        "source_file_tree_sha256": source_digest.hexdigest(),
        "source_rows": len(records),
        "source_class_counts": source_class_counts,
        "selected_file_tree_sha256": selected_digest.hexdigest(),
        "selected_rows": len(selected_relatives),
    }


def freeze_imagenet_sketch_provenance(
    *,
    dataset_root: str | Path,
    reference_root: str | Path,
    archive: str | Path,
    output: str | Path,
    source_url: str = OFFICIAL_IMAGENET_SKETCH_URL,
) -> Path:
    """Freeze local acquisition bytes; this is not a scientific result."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing provenance manifest: {destination}"
        )
    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.is_file() or archive_path.is_symlink():
        raise IntegrityError(f"ImageNet-Sketch archive is absent or unsafe: {archive_path}")
    root = Path(dataset_root).expanduser().resolve()
    reference = Path(reference_root).expanduser().resolve()
    for protected in (root, reference):
        if destination == protected or protected in destination.parents:
            raise IntegrityError("provenance output cannot be inside an input data tree")
    wnids = discover_reference_wnids(reference)
    discovery = discover_imagenet_sketch(root, expected_wnids=wnids)
    archive_fingerprint = _archive_tree_fingerprint(
        archive_path, expected_wnids=wnids
    )
    if (
        archive_fingerprint["source_file_tree_sha256"]
        != discovery.source_file_tree_sha256
        or archive_fingerprint["source_rows"] != discovery.source_rows
        or archive_fingerprint["source_class_counts"]
        != discovery.source_class_counts
        or archive_fingerprint["selected_file_tree_sha256"]
        != discovery.file_tree_sha256
        or archive_fingerprint["selected_rows"] != len(discovery.paths)
    ):
        raise IntegrityError(
            "extracted ImageNet-Sketch source tree or selected subset does not "
            "match the supplied archive"
        )
    payload = {
        "schema_version": IMAGENET_SKETCH_PROVENANCE_SCHEMA,
        "status": "FROZEN_LOCAL_INPUT",
        "scientific_result": False,
        "dataset": "ImageNet-Sketch",
        "source_url": str(source_url),
        "archive": {
            "path": str(archive_path),
            "sha256": sha256_file(archive_path),
            "size_bytes": int(archive_path.stat().st_size),
            "format": archive_fingerprint["format"],
            "normalized_source_file_tree_sha256": archive_fingerprint[
                "source_file_tree_sha256"
            ],
            "normalized_source_image_rows": archive_fingerprint["source_rows"],
            "normalized_selected_file_tree_sha256": archive_fingerprint[
                "selected_file_tree_sha256"
            ],
            "normalized_selected_image_rows": archive_fingerprint[
                "selected_rows"
            ],
        },
        "dataset_root": str(root),
        "reference_wnid_root": str(reference),
        "class_index": list(wnids),
        "class_index_sha256": sha256_json(list(wnids)),
        "class_count": EXPECTED_CLASSES,
        "images_per_class": EXPECTED_IMAGES_PER_CLASS,
        "row_count": EXPECTED_ROWS,
        "allowed_image_suffixes": sorted(IMAGE_SUFFIXES),
        "source_cardinality_policy": "at_least_50_images_per_wnid",
        "selection_policy": "lexicographically_first_50_basenames_per_wnid",
        "discovery_order": (
            "manifest_class_index_then_lexicographic_basename_first_50"
        ),
        "source_image_rows": discovery.source_rows,
        "source_class_counts": discovery.source_class_counts,
        "source_class_counts_sha256": sha256_json(discovery.source_class_counts),
        "source_file_tree_sha256": discovery.source_file_tree_sha256,
        "file_tree_sha256": discovery.file_tree_sha256,
        "duplicate_digest_groups": discovery.duplicate_digest_groups,
        "duplicate_rows_beyond_first": discovery.duplicate_rows_beyond_first,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, seal_manifest(payload))
    validate_imagenet_sketch_provenance(destination)
    return destination


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def load_base_campaign_feature_store_evidence(
    manifest_path: str | Path,
    *,
    expected_file_sha256: str,
    expected_external_input_sha256: str,
    expected_feature_store_count: int,
) -> BaseCampaignFeatureStoreEvidence:
    """Load the exact base-campaign store allowlist, failing on any byte drift."""

    path = Path(manifest_path).expanduser().resolve()
    if not _valid_sha256(expected_file_sha256):
        raise IntegrityError("locked external-input manifest file hash is invalid")
    if not path.is_file():
        raise IntegrityError("base campaign external-input manifest is missing")
    if sha256_file(path) != str(expected_file_sha256):
        raise IntegrityError("base campaign external-input manifest bytes changed")
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise IntegrityError("base campaign external-input manifest is not an object")
    if (
        payload.get("schema_version")
        != "marginal_alignment.iclr.external_inputs.v1"
        or payload.get("status") != "PASS"
    ):
        raise IntegrityError("base campaign external-input manifest is not passing V1")
    if (
        not _valid_sha256(expected_external_input_sha256)
        or payload.get("external_input_sha256")
        != str(expected_external_input_sha256)
    ):
        raise IntegrityError("base campaign external-input semantic identity changed")
    rows = payload.get("native_feature_stores")
    if not isinstance(rows, list):
        raise IntegrityError("base campaign feature-store records are malformed")
    expected_count = int(expected_feature_store_count)
    if (
        expected_count < 1
        or int(payload.get("native_feature_store_count", -1)) != len(rows)
        or len(rows) != expected_count
    ):
        raise IntegrityError("base campaign feature-store count changed")
    required = {
        "dataset", "encoder", "corruption", "severity", "manifest",
        "manifest_sha256",
    }
    index: dict[tuple[str, str, str, int], str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise IntegrityError("base campaign feature-store record is malformed")
        try:
            key = (
                str(row["dataset"]),
                str(row["encoder"]),
                str(row["corruption"]),
                int(row["severity"]),
            )
        except (TypeError, ValueError) as error:
            raise IntegrityError(
                "base campaign feature-store condition is malformed"
            ) from error
        if key in index:
            raise IntegrityError(f"duplicate base campaign feature-store key: {key}")
        digest = str(row["manifest_sha256"])
        if not _valid_sha256(digest):
            raise IntegrityError(
                f"base campaign feature-store hash is malformed: {key}"
            )
        declared_path = str(row["manifest"])
        if not declared_path.endswith("/FEATURE_STORE_MANIFEST.json"):
            raise IntegrityError(
                f"base campaign feature-store manifest path is malformed: {key}"
            )
        index[key] = digest
    return BaseCampaignFeatureStoreEvidence(
        manifest_path=path,
        manifest_file_sha256=str(expected_file_sha256),
        external_input_sha256=str(expected_external_input_sha256),
        feature_store_count=expected_count,
        manifest_sha256_by_key=index,
    )


def verify_base_campaign_feature_store(
    store: FeatureStore,
    evidence: BaseCampaignFeatureStoreEvidence,
    *,
    dataset: str,
    encoder: str,
    corruption: str,
    severity: int,
) -> dict[str, Any]:
    """Bind a supplied store to one exact condition in the base-campaign allowlist."""

    condition = {
        "dataset": str(dataset),
        "encoder": str(encoder),
        "corruption": str(corruption),
        "severity": int(severity),
    }
    metadata = store.manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise IntegrityError("supplied feature store metadata is malformed")
    try:
        store_severity = int(metadata.get("severity", -1))
    except (TypeError, ValueError) as error:
        raise IntegrityError("supplied feature store has malformed severity") from error
    if (
        not store.hashes_verified
        or store.dataset != condition["dataset"]
        or store.encoder != condition["encoder"]
        or str(metadata.get("corruption", "")) != condition["corruption"]
        or store_severity != condition["severity"]
    ):
        raise IntegrityError(
            "supplied feature store does not match the requested exact condition"
        )
    expected = evidence.expected_manifest_sha256(**condition)
    actual = sha256_file(store.manifest_path)
    if actual != expected:
        raise IntegrityError(
            "supplied feature-store manifest differs from the exact base-campaign "
            f"manifest for {tuple(condition.values())}"
        )
    return {
        "schema_version": (
            "marginal_alignment.iclr_extension.base_campaign_store_binding.v1"
        ),
        "status": "PASS",
        "external_input_manifest_file_sha256": evidence.manifest_file_sha256,
        "external_input_sha256": evidence.external_input_sha256,
        "condition": condition,
        "expected_manifest_sha256": expected,
        "actual_manifest_sha256": actual,
    }


def _load_locked_base_campaign_feature_store_evidence(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    repository: Path,
) -> BaseCampaignFeatureStoreEvidence:
    declaration = config.get("base_campaign_feature_stores")
    locked = lock.get("base_campaign_feature_stores")
    if not isinstance(declaration, Mapping) or dict(declaration) != locked:
        raise IntegrityError(
            "Play-3 config and lock disagree on base-campaign feature stores"
        )
    relative = declaration.get("external_input_manifest")
    if not isinstance(relative, str) or not relative:
        raise IntegrityError("Play-3 lacks a base external-input manifest path")
    candidate = repository / relative
    if candidate.is_symlink():
        raise IntegrityError("base external-input manifest must not be a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise IntegrityError(
            "base external-input manifest escapes the repository"
        ) from error
    if not path.is_file():
        raise IntegrityError("base external-input manifest is missing")
    return load_base_campaign_feature_store_evidence(
        path,
        expected_file_sha256=str(
            declaration.get("external_input_manifest_file_sha256", "")
        ),
        expected_external_input_sha256=str(
            declaration.get("external_input_sha256", "")
        ),
        expected_feature_store_count=int(
            declaration.get("feature_store_count", -1)
        ),
    )


def validate_imagenet_sketch_provenance(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
) -> ValidatedImageNetSketchProvenance:
    """Rehash every bound byte before feature extraction or scientific use."""

    path = Path(manifest_path).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise IntegrityError("ImageNet-Sketch provenance must be a JSON object")
    if payload.get("schema_version") != IMAGENET_SKETCH_PROVENANCE_SCHEMA:
        raise IntegrityError(
            f"provenance is not {IMAGENET_SKETCH_PROVENANCE_SCHEMA}"
        )
    verify_manifest_checksum(payload)
    if payload.get("status") != "FROZEN_LOCAL_INPUT" or payload.get("scientific_result") is not False:
        raise IntegrityError("dataset freeze cannot claim scientific-result status")
    if payload.get("dataset") != "ImageNet-Sketch":
        raise IntegrityError("provenance has the wrong dataset identity")
    if str(payload.get("source_url")) != OFFICIAL_IMAGENET_SKETCH_URL:
        raise IntegrityError("provenance must cite the official ImageNet-Sketch source")
    if (
        int(payload.get("class_count", -1)) != EXPECTED_CLASSES
        or int(payload.get("images_per_class", -1)) != EXPECTED_IMAGES_PER_CLASS
        or int(payload.get("row_count", -1)) != EXPECTED_ROWS
    ):
        raise IntegrityError("provenance has the wrong ImageNet-Sketch cardinality")
    if payload.get("allowed_image_suffixes") != sorted(IMAGE_SUFFIXES):
        raise IntegrityError("provenance image-suffix policy changed")
    if payload.get("source_cardinality_policy") != "at_least_50_images_per_wnid":
        raise IntegrityError("provenance source-cardinality policy changed")
    if payload.get("selection_policy") != (
        "lexicographically_first_50_basenames_per_wnid"
    ):
        raise IntegrityError("provenance selection policy changed")
    if payload.get("discovery_order") != (
        "manifest_class_index_then_lexicographic_basename_first_50"
    ):
        raise IntegrityError("provenance discovery order changed")
    wnids = _validated_wnids(payload.get("class_index", []))
    if payload.get("class_index_sha256") != sha256_json(list(wnids)):
        raise IntegrityError("ImageNet class-index hash mismatch")

    archive_record = payload.get("archive")
    if not isinstance(archive_record, Mapping):
        raise IntegrityError("provenance lacks the source archive record")
    archive_path = Path(
        str(archive if archive is not None else archive_record.get("path", ""))
    ).expanduser().resolve()
    expected_archive_sha = archive_record.get("sha256")
    if not _valid_sha256(expected_archive_sha):
        raise IntegrityError("provenance archive SHA-256 is malformed")
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or int(archive_path.stat().st_size) != int(archive_record.get("size_bytes", -1))
        or sha256_file(archive_path) != expected_archive_sha
    ):
        raise IntegrityError("ImageNet-Sketch archive bytes do not match provenance")

    root = Path(
        str(dataset_root if dataset_root is not None else payload.get("dataset_root", ""))
    ).expanduser().resolve()
    discovery = discover_imagenet_sketch(root, expected_wnids=wnids)
    source_counts = payload.get("source_class_counts")
    if (
        not isinstance(source_counts, Mapping)
        or dict(source_counts) != discovery.source_class_counts
        or payload.get("source_class_counts_sha256")
        != sha256_json(discovery.source_class_counts)
        or int(payload.get("source_image_rows", -1)) != discovery.source_rows
        or payload.get("source_file_tree_sha256")
        != discovery.source_file_tree_sha256
    ):
        raise IntegrityError(
            "ImageNet-Sketch source file-tree bytes or counts do not match provenance"
        )
    expected_tree = payload.get("file_tree_sha256")
    if not _valid_sha256(expected_tree) or discovery.file_tree_sha256 != expected_tree:
        raise IntegrityError("ImageNet-Sketch file-tree bytes do not match provenance")
    archive_fingerprint = _archive_tree_fingerprint(
        archive_path, expected_wnids=wnids
    )
    if (
        archive_record.get("format") != archive_fingerprint["format"]
        or int(archive_record.get("normalized_source_image_rows", -1))
        != archive_fingerprint["source_rows"]
        or archive_record.get("normalized_source_file_tree_sha256")
        != archive_fingerprint["source_file_tree_sha256"]
        or int(archive_record.get("normalized_selected_image_rows", -1))
        != archive_fingerprint["selected_rows"]
        or archive_record.get("normalized_selected_file_tree_sha256")
        != archive_fingerprint["selected_file_tree_sha256"]
        or archive_fingerprint["source_file_tree_sha256"]
        != discovery.source_file_tree_sha256
        or archive_fingerprint["source_class_counts"]
        != discovery.source_class_counts
        or archive_fingerprint["selected_file_tree_sha256"]
        != discovery.file_tree_sha256
    ):
        raise IntegrityError(
            "ImageNet-Sketch archive source members, extracted source tree, or "
            "deterministic selected subset are not identical"
        )
    if (
        discovery.duplicate_digest_groups != int(payload.get("duplicate_digest_groups", -1))
        or discovery.duplicate_rows_beyond_first
        != int(payload.get("duplicate_rows_beyond_first", -1))
    ):
        raise IntegrityError("ImageNet-Sketch duplicate audit changed")
    return ValidatedImageNetSketchProvenance(
        manifest_path=path,
        manifest=dict(payload),
        dataset_root=root,
        archive_path=archive_path,
        discovery=discovery,
    )


def create_imagenet_sketch_build_plan(
    provenance_manifest: str | Path,
    *,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
) -> ImageNetSketchBuildPlan:
    """Return the exact target extraction order and unpaired-store declaration."""

    validated = validate_imagenet_sketch_provenance(
        provenance_manifest, dataset_root=dataset_root, archive=archive
    )
    manifest_sha = sha256_file(validated.manifest_path)
    binding = {
        "schema_version": IMAGENET_SKETCH_PROVENANCE_SCHEMA,
        "manifest_sha256": manifest_sha,
        "file_tree_sha256": validated.discovery.file_tree_sha256,
        "class_index_sha256": validated.manifest["class_index_sha256"],
        "row_count": EXPECTED_ROWS,
    }
    return ImageNetSketchBuildPlan(
        provenance_manifest=validated.manifest_path,
        provenance_manifest_sha256=manifest_sha,
        dataset_root=validated.dataset_root,
        archive_path=validated.archive_path,
        target_paths=validated.discovery.paths,
        target_labels=np.array(validated.discovery.labels, copy=True),
        target_ids=np.array(validated.discovery.sample_ids, copy=True),
        wnids=validated.discovery.wnids,
        feature_store_pairing={
            "status": "unavailable",
            "reason": (
                "ImageNet validation and ImageNet-Sketch are independent image "
                "collections; class equality is not sample correspondence"
            ),
        },
        feature_store_metadata_binding={
            "imagenet_sketch_provenance": binding,
            "natural_shift": True,
            "target_labels_available_only_for_posthoc_evaluation": True,
        },
    )


def _save_npy_fsynced(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_existing_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _assert_encoder_identity(
    *,
    reference_store: FeatureStore,
    encoder_evidence: Mapping[str, Any],
) -> None:
    model = reference_store.manifest.get("metadata", {}).get("model_provenance")
    if not isinstance(model, Mapping):
        raise IntegrityError("reference store lacks model provenance")
    comparisons = {
        "encoder_id": encoder_evidence.get("encoder_id"),
        "model_id": encoder_evidence.get("model_id"),
        "checkpoint_sha256": encoder_evidence.get("checkpoint_sha256"),
        "architecture_config_sha256": encoder_evidence.get(
            "architecture_config_sha256"
        ),
        "preprocessing_sha256": encoder_evidence.get("preprocessing_sha256"),
        "feature_layer": encoder_evidence.get("feature_layer"),
        "pooling": encoder_evidence.get("pooling"),
    }
    if any(value in {None, ""} for value in comparisons.values()):
        raise IntegrityError("loaded encoder evidence is missing a model-identity field")
    mismatches = {
        key: {"reference_store": model.get(key), "loaded_encoder": value}
        for key, value in comparisons.items()
        if model.get(key) != value
    }
    if mismatches:
        raise IntegrityError(
            f"loaded encoder disagrees with the frozen reference store: {mismatches}"
        )
    expected_dimension = int(encoder_evidence.get("expected_dimension", -1))
    reference_features = reference_store.endpoint("reference").features
    if (
        expected_dimension < 1
        or reference_features.ndim != 2
        or reference_features.shape[1] != expected_dimension
    ):
        raise IntegrityError(
            "loaded encoder dimension disagrees with the frozen reference features"
        )


def _validate_reference_class_identity(
    *,
    reference_ids: np.ndarray,
    reference_labels: np.ndarray,
    head_fit_ids: np.ndarray,
    class_index: Sequence[str],
    shots_per_class: int,
) -> dict[str, Any]:
    """Prove that the source-store label index is the frozen WNID index."""

    identifiers = np.asarray(reference_ids)
    labels = np.asarray(reference_labels)
    fit_ids = np.asarray(head_fit_ids)
    if identifiers.ndim != 1 or labels.shape != identifiers.shape:
        raise IntegrityError("ImageNet reference IDs and labels have incompatible shapes")
    if not np.issubdtype(labels.dtype, np.integer):
        raise IntegrityError("ImageNet reference labels must be integer class indices")
    wnids = _validated_wnids(class_index)
    wnid_to_label = {wnid: index for index, wnid in enumerate(wnids)}
    id_to_label: dict[Any, int] = {}
    for raw_id, raw_label in zip(identifiers.tolist(), labels.tolist()):
        sample_id = str(raw_id)
        parts = PurePosixPath(sample_id).parts
        if len(parts) != 2 or parts[0] not in wnid_to_label:
            raise IntegrityError(
                "ImageNet reference IDs must be exact WNID/basename keys"
            )
        expected = wnid_to_label[parts[0]]
        label = int(raw_label)
        if label != expected:
            raise IntegrityError(
                "ImageNet reference ID-to-label mapping disagrees with the WNID index"
            )
        if raw_id in id_to_label:
            raise IntegrityError("ImageNet reference sample IDs are not unique")
        id_to_label[raw_id] = label
    full_counts = np.bincount(labels.astype(np.int64), minlength=len(wnids))
    if len(full_counts) != len(wnids) or not np.all(
        full_counts == EXPECTED_IMAGES_PER_CLASS
    ):
        raise IntegrityError(
            "ImageNet reference endpoint is not exactly balanced over the WNID index"
        )
    try:
        fit_labels = np.asarray(
            [id_to_label[value] for value in fit_ids.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise IntegrityError("head-fit ID is absent from the ImageNet reference endpoint") from error
    fit_counts = np.bincount(fit_labels, minlength=len(wnids))
    if len(fit_counts) != len(wnids) or not np.all(fit_counts == int(shots_per_class)):
        raise IntegrityError("head-fit IDs are not exactly stratified by WNID")
    mapping_records = [
        {"sample_id": str(value), "class_index": int(id_to_label[value])}
        for value in identifiers.tolist()
    ]
    return {
        "status": "PASS",
        "reference_id_kind": "wnid_basename",
        "class_index_sha256": sha256_json(list(wnids)),
        "reference_id_label_mapping_sha256": sha256_json(mapping_records),
        "reference_rows_per_class": int(EXPECTED_IMAGES_PER_CLASS),
        "head_fit_rows_per_class": int(shots_per_class),
        "remaining_rows_per_class": int(
            EXPECTED_IMAGES_PER_CLASS - int(shots_per_class)
        ),
    }


def _block_sha256(value: np.ndarray) -> str:
    """Hash one completed extraction block in a dtype/shape-sensitive form."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(canonical_json_bytes([int(item) for item in array.shape]))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_build_state(path: Path, payload: Mapping[str, Any]) -> None:
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    write_json_atomic(path, seal_manifest(unsigned))


def _read_build_state(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise IntegrityError("ImageNet-Sketch build state must be a JSON object")
    verify_manifest_checksum(value)
    if value.get("schema_version") != _BUILD_STATE_SCHEMA:
        raise IntegrityError("unsupported ImageNet-Sketch build-state schema")
    return dict(value)


def _validate_completed_blocks(
    state: Mapping[str, Any],
    *,
    target_matrix: np.ndarray,
    extracted_labels: np.ndarray,
    expected_labels: np.ndarray,
    checkpoint_rows: int,
) -> int:
    """Fail closed on a non-prefix checkpoint or changed completed bytes."""

    completed = state.get("completed_blocks")
    if not isinstance(completed, list):
        raise IntegrityError("build state completed_blocks must be a list")
    cursor = 0
    for raw_record in completed:
        if not isinstance(raw_record, Mapping):
            raise IntegrityError("malformed completed extraction block")
        start = int(raw_record.get("start", -1))
        stop = int(raw_record.get("stop", -1))
        expected_stop = min(cursor + int(checkpoint_rows), EXPECTED_ROWS)
        if start != cursor or stop != expected_stop:
            raise IntegrityError("completed extraction blocks are not an exact prefix")
        if not np.array_equal(extracted_labels[start:stop], expected_labels[start:stop]):
            raise IntegrityError("completed extraction labels changed")
        if _block_sha256(target_matrix[start:stop]) != raw_record.get(
            "features_sha256"
        ):
            raise IntegrityError("completed extraction feature bytes changed")
        if _block_sha256(extracted_labels[start:stop]) != raw_record.get(
            "labels_sha256"
        ):
            raise IntegrityError("completed extraction label bytes changed")
        cursor = stop
    return cursor


def _build_fingerprint(
    *,
    config_path: Path,
    provenance_manifest: Path,
    reference_store: FeatureStore,
    reference_store_binding: Mapping[str, Any],
    encoder_id: str,
    encoder_evidence: Mapping[str, Any],
    expected_dimension: int,
    checkpoint_rows: int,
    batch_size: int,
    workers: int,
    device: str,
) -> tuple[str, dict[str, Any]]:
    binding = {
        "schema_version": "marginal_alignment.iclr_extension.imagenet_sketch_build_binding.v1",
        "config_sha256": sha256_file(config_path),
        "provenance_manifest_sha256": sha256_file(provenance_manifest),
        "reference_store_manifest_sha256": sha256_file(
            reference_store.manifest_path
        ),
        "reference_store_binding": dict(reference_store_binding),
        "encoder_id": str(encoder_id),
        "encoder_evidence": dict(encoder_evidence),
        "expected_rows": int(EXPECTED_ROWS),
        "expected_dimension": int(expected_dimension),
        "checkpoint_rows": int(checkpoint_rows),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "device": str(device),
    }
    return sha256_json(binding), binding


def build_imagenet_sketch_feature_store(
    *,
    play3_config_path: str | Path,
    provenance_manifest: str | Path,
    reference_store_path: str | Path,
    destination: str | Path,
    encoder_id: str,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
    device: str = "cuda",
    batch_size: int | None = None,
    workers: int | None = None,
    work_dir: str | Path | None = None,
    checkpoint_rows: int = 5000,
) -> Path:
    """Extract one pinned encoder into a fresh, resumable unpaired store."""

    play3, config_path, lock, repository = load_locked_play3_config(play3_config_path)
    if encoder_id not in play3["encoders"]:
        raise IntegrityError(f"encoder is outside the Play-3 lock: {encoder_id}")
    plan = create_imagenet_sketch_build_plan(
        provenance_manifest, dataset_root=dataset_root, archive=archive
    )
    reference_store = load_feature_store(reference_store_path, verify_hashes=True)
    source_evidence = _load_locked_base_campaign_feature_store_evidence(
        play3, lock, repository
    )
    reference_condition = dict(
        play3["base_campaign_feature_stores"]["natural_reference_condition"]
    )
    reference_binding = verify_base_campaign_feature_store(
        reference_store,
        source_evidence,
        dataset=str(reference_condition["dataset"]),
        encoder=encoder_id,
        corruption=str(reference_condition["corruption"]),
        severity=int(reference_condition["severity"]),
    )
    reference_endpoint = reference_store.endpoint("reference")
    if reference_endpoint.labels is None or len(reference_endpoint.features) != EXPECTED_ROWS:
        raise IntegrityError("reference store lacks the complete labeled ImageNet endpoint")
    model = reference_store.manifest.get("metadata", {}).get("model_provenance")
    if not isinstance(model, Mapping):
        raise IntegrityError("reference store lacks model provenance")
    head_fit_role = str(model.get("head_fit_ids_role", ""))
    if not head_fit_role:
        raise IntegrityError("reference store lacks head-fit IDs")
    head_fit_ids = np.asarray(reference_store.array(head_fit_role))
    reference_ids = np.asarray(reference_endpoint.ids)
    fit_set = set(head_fit_ids.tolist())
    if not fit_set or not fit_set.issubset(set(reference_ids.tolist())):
        raise IntegrityError("head-fit IDs are not a nonempty subset of ImageNet reference IDs")
    keep = np.asarray([value not in fit_set for value in reference_ids.tolist()], dtype=bool)
    head_protocol = model.get("head_protocol")
    if not isinstance(head_protocol, Mapping):
        raise IntegrityError("reference store lacks a structured head protocol")
    shots_per_class = int(head_protocol.get("shots_per_class", -1))
    if shots_per_class < 1:
        raise IntegrityError("ImageNet reference head is not a stratified shot protocol")
    expected_fit_rows = EXPECTED_CLASSES * shots_per_class
    if len(head_fit_ids) != expected_fit_rows or int(np.sum(keep)) != EXPECTED_ROWS - expected_fit_rows:
        raise IntegrityError(
            "ImageNet-Sketch build requires the locked 25-shot-per-class head exclusion"
        )
    class_identity = _validate_reference_class_identity(
        reference_ids=reference_ids,
        reference_labels=np.asarray(reference_endpoint.labels),
        head_fit_ids=head_fit_ids,
        class_index=plan.wnids,
        shots_per_class=shots_per_class,
    )

    feature_config_value = Path(str(play3["feature_build_config"]))
    if not feature_config_value.is_absolute():
        feature_config_value = repository / feature_config_value
    feature_config = load_feature_build_config(feature_config_value)
    if encoder_id not in feature_config["encoders"]:
        raise IntegrityError("encoder is absent from the feature-build lock")
    encoder = load_frozen_encoder(
        encoder_id, feature_config["encoders"][encoder_id], device=device
    )
    _assert_encoder_identity(
        reference_store=reference_store, encoder_evidence=encoder.evidence
    )

    resolved_batch_size = int(
        feature_config["batch_size"] if batch_size is None else batch_size
    )
    resolved_workers = int(feature_config["workers"] if workers is None else workers)
    block_rows = int(checkpoint_rows)
    if resolved_batch_size < 1 or resolved_workers < 0 or block_rows < 1:
        raise IntegrityError("batch size, workers, and checkpoint rows are invalid")

    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing feature store: {output}")
    if (
        reference_store.root.resolve() == output
        or reference_store.root.resolve() in output.parents
        or plan.dataset_root == output
        or plan.dataset_root in output.parents
    ):
        raise IntegrityError("feature-store destination overlaps an input tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        output.with_name(f".{output.name}.work")
        if work_dir is None
        else Path(work_dir).expanduser().resolve()
    )
    if (
        staging == output
        or staging in output.parents
        or output in staging.parents
        or any(
        staging == protected or protected in staging.parents
        for protected in (reference_store.root.resolve(), plan.dataset_root.resolve())
        )
    ):
        raise IntegrityError("build work directory overlaps an input or output tree")
    fingerprint, fingerprint_binding = _build_fingerprint(
        config_path=config_path,
        provenance_manifest=plan.provenance_manifest,
        reference_store=reference_store,
        reference_store_binding=reference_binding,
        encoder_id=encoder_id,
        encoder_evidence=encoder.evidence,
        expected_dimension=int(encoder.expected_dimension),
        checkpoint_rows=block_rows,
        batch_size=resolved_batch_size,
        workers=resolved_workers,
        device=device,
    )
    state_path = staging / "BUILD_STATE.json"
    static_roles = (
        "reference_features", "reference_labels", "reference_ids",
        "target_ids", "head_fit_ids", "head_weight", "head_bias",
    )
    partial_target_path = staging / "target_features.partial.npy"
    final_target_path = staging / "target_features.npy"
    partial_label_path = staging / "target_labels.partial.npy"
    final_label_path = staging / "target_labels.npy"

    if not staging.exists():
        staging.mkdir(parents=True)
        reference_features = np.asarray(reference_endpoint.features[keep], dtype=np.float32)
        reference_labels = np.asarray(reference_endpoint.labels[keep], dtype=np.int64)
        kept_reference_ids = np.asarray(reference_ids[keep])
        _save_npy_fsynced(staging / "reference_features.npy", reference_features)
        _save_npy_fsynced(staging / "reference_labels.npy", reference_labels)
        _save_npy_fsynced(staging / "reference_ids.npy", kept_reference_ids)
        _save_npy_fsynced(staging / "target_ids.npy", plan.target_ids)
        _save_npy_fsynced(staging / "head_fit_ids.npy", head_fit_ids)
        head = reference_store.manifest.get("head", {})
        weight = np.asarray(reference_store.array(str(head.get("weight_role", ""))))
        bias = np.asarray(reference_store.array(str(head.get("bias_role", ""))))
        _save_npy_fsynced(staging / "head_weight.npy", weight.astype(np.float32, copy=False))
        _save_npy_fsynced(staging / "head_bias.npy", bias.astype(np.float32, copy=False))

        target_matrix = np.lib.format.open_memmap(
            partial_target_path, mode="w+", dtype=np.float32,
            shape=(EXPECTED_ROWS, int(encoder.expected_dimension)),
        )
        target_matrix[:] = np.nan
        target_matrix.flush()
        del target_matrix
        _fsync_existing_file(partial_target_path)
        extracted_labels = np.lib.format.open_memmap(
            partial_label_path, mode="w+", dtype=np.int64, shape=(EXPECTED_ROWS,)
        )
        extracted_labels[:] = -1
        extracted_labels.flush()
        del extracted_labels
        _fsync_existing_file(partial_label_path)
        state = {
            "schema_version": _BUILD_STATE_SCHEMA,
            "status": "IN_PROGRESS",
            "phase": "extracting",
            "scientific_result": False,
            "published_destination": str(output),
            "build_fingerprint": fingerprint,
            "build_binding": fingerprint_binding,
            "static_files": {
                role: describe_npy(staging / f"{role}.npy", relative_to=staging)
                for role in static_roles
            },
            "completed_blocks": [],
        }
        _write_build_state(state_path, state)
    elif not staging.is_dir() or not state_path.is_file():
        raise IntegrityError(
            "existing build work directory lacks a sealed BUILD_STATE.json"
        )

    state = _read_build_state(state_path)
    if (
        state.get("published_destination") != str(output)
        or state.get("build_fingerprint") != fingerprint
        or state.get("build_binding") != fingerprint_binding
    ):
        raise IntegrityError("build checkpoint does not match the current locked inputs")
    static_files = state.get("static_files")
    if not isinstance(static_files, Mapping) or set(static_files) != set(static_roles):
        raise IntegrityError("build checkpoint has an incomplete static-file inventory")
    for record in static_files.values():
        if not isinstance(record, Mapping):
            raise IntegrityError("build checkpoint contains a malformed file record")
        verify_file_record(staging, record, verify_hash=True)

    phase = str(state.get("phase", ""))
    if phase not in {"extracting", "extracted", "published_ready"}:
        raise IntegrityError(f"unsupported build checkpoint phase: {phase!r}")
    active_target_path = (
        partial_target_path if partial_target_path.is_file() else final_target_path
    )
    active_label_path = (
        partial_label_path if partial_label_path.is_file() else final_label_path
    )
    if not active_target_path.is_file() or not active_label_path.is_file():
        raise IntegrityError("build checkpoint is missing resumable extraction arrays")
    target_matrix = np.load(active_target_path, mmap_mode="r+", allow_pickle=False)
    extracted_labels = np.load(active_label_path, mmap_mode="r+", allow_pickle=False)
    if (
        target_matrix.shape != (EXPECTED_ROWS, int(encoder.expected_dimension))
        or target_matrix.dtype != np.float32
    ):
        raise IntegrityError("resumable target-feature array has the wrong contract")
    if extracted_labels.shape != (EXPECTED_ROWS,) or extracted_labels.dtype != np.int64:
        raise IntegrityError("resumable extracted-label array has the wrong contract")
    cursor = _validate_completed_blocks(
        state,
        target_matrix=target_matrix,
        extracted_labels=extracted_labels,
        expected_labels=plan.target_labels,
        checkpoint_rows=block_rows,
    )
    if phase == "extracting":
        while cursor < EXPECTED_ROWS:
            stop = min(cursor + block_rows, EXPECTED_ROWS)
            extract_dataset(
                encoder,
                ImagePathDataset(
                    plan.target_paths[cursor:stop], plan.target_labels[cursor:stop]
                ),
                target_matrix,
                destination_offset=cursor,
                batch_size=resolved_batch_size,
                workers=resolved_workers,
                label_destination=extracted_labels,
                label_destination_offset=cursor,
            )
            target_matrix.flush()
            extracted_labels.flush()
            _fsync_existing_file(active_target_path)
            _fsync_existing_file(active_label_path)
            if not np.array_equal(
                extracted_labels[cursor:stop], plan.target_labels[cursor:stop]
            ):
                raise IntegrityError("ImageNet-Sketch labels changed during extraction")
            completed = list(state["completed_blocks"])
            completed.append({
                "start": int(cursor),
                "stop": int(stop),
                "features_sha256": _block_sha256(target_matrix[cursor:stop]),
                "labels_sha256": _block_sha256(extracted_labels[cursor:stop]),
            })
            state["completed_blocks"] = completed
            cursor = stop
            if cursor == EXPECTED_ROWS:
                state["phase"] = "extracted"
            _write_build_state(state_path, state)
    if cursor != EXPECTED_ROWS or not np.array_equal(extracted_labels, plan.target_labels):
        raise IntegrityError("ImageNet-Sketch extraction checkpoint is incomplete")
    target_matrix.flush()
    extracted_labels.flush()
    del target_matrix
    del extracted_labels
    if partial_target_path.is_file():
        os.replace(partial_target_path, final_target_path)
        _fsync_existing_file(final_target_path)
    if partial_label_path.is_file():
        os.replace(partial_label_path, final_label_path)
        _fsync_existing_file(final_label_path)
    state["phase"] = "extracted"
    _write_build_state(state_path, state)

    model_provenance = copy.deepcopy(dict(model))
    model_provenance["head_fit_ids_sha256"] = sha256_file(staging / "head_fit_ids.npy")
    model_provenance["reference_split"] = "val_excluding_head_fit_ids"
    metadata = {
        **plan.feature_store_metadata_binding,
        "model_provenance": model_provenance,
        "reference_store_manifest_sha256": sha256_file(reference_store.manifest_path),
        "reference_store_dataset": reference_store.dataset,
        "source_campaign_reference_store": reference_binding,
        "reference_head_fit_rows_excluded": int(len(head_fit_ids)),
        "reference_class_identity": class_identity,
        "encoder_extraction_evidence": dict(encoder.evidence),
        "play3_config_sha256": sha256_file(config_path),
        "resumable_build": {
            "build_fingerprint": fingerprint,
            "checkpoint_rows": block_rows,
            "completed_blocks": len(state["completed_blocks"]),
            "provenance_sidecar": {
                "path": "BUILD_STATE.json",
                "schema_version": _BUILD_STATE_SCHEMA,
                "sealed_self_checksum": True,
                "feature_role": False,
            },
        },
    }
    arrays = {
        role: f"{role}.npy"
        for role in (
            "reference_features", "reference_labels", "reference_ids",
            "target_features", "target_labels", "target_ids",
            "head_weight", "head_bias", "head_fit_ids",
        )
    }
    manifest_path = build_native_manifest(
        staging,
        dataset="imagenet_sketch",
        encoder=encoder_id,
        arrays=arrays,
        id_provenance={
            "reference": {
                "verified": True,
                "kind": "dataset_sample_key_excluding_head_fit",
                "source": str(reference_store.manifest_path),
                "namespace": reference_endpoint.id_namespace,
            },
            "target": {
                "verified": True,
                "kind": "official_relative_image_path",
                "source": str(plan.provenance_manifest),
                "namespace": "imagenet_sketch:relative_image_path",
            },
        },
        pairing=plan.feature_store_pairing,
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata=metadata,
    )
    load_feature_store(manifest_path, verify_hashes=True)
    state["phase"] = "published_ready"
    state["status"] = "PASS"
    state["feature_store_manifest_sha256"] = sha256_file(manifest_path)
    _write_build_state(state_path, state)
    staging.rename(output)
    published = output / "FEATURE_STORE_MANIFEST.json"
    load_feature_store(published, verify_hashes=True)
    return published


def build_all_imagenet_sketch_feature_stores(
    *,
    play3_config_path: str | Path,
    provenance_manifest: str | Path,
    reference_stores: Mapping[str, str | Path],
    output_root: str | Path,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
    device: str = "cuda",
    batch_size: int | None = None,
    workers: int | None = None,
) -> list[Path]:
    """Build the complete three-encoder panel into a fresh output root."""

    config, _, lock, repository = load_locked_play3_config(play3_config_path)
    encoders = list(config["encoders"])
    if set(reference_stores) != set(encoders):
        raise IntegrityError(
            f"reference-store map must contain exactly the locked encoders: {encoders}"
        )
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse an existing output root: {root}")
    # Preflight every expensive external input before loading the first model.
    create_imagenet_sketch_build_plan(
        provenance_manifest, dataset_root=dataset_root, archive=archive
    )
    source_evidence = _load_locked_base_campaign_feature_store_evidence(
        config, lock, repository
    )
    reference_condition = dict(
        config["base_campaign_feature_stores"]["natural_reference_condition"]
    )
    for encoder_id in encoders:
        store = load_feature_store(reference_stores[encoder_id], verify_hashes=True)
        verify_base_campaign_feature_store(
            store,
            source_evidence,
            dataset=str(reference_condition["dataset"]),
            encoder=encoder_id,
            corruption=str(reference_condition["corruption"]),
            severity=int(reference_condition["severity"]),
        )
    root.mkdir(parents=True)
    outputs: list[Path] = []
    for encoder_id in encoders:
        outputs.append(build_imagenet_sketch_feature_store(
            play3_config_path=play3_config_path,
            provenance_manifest=provenance_manifest,
            reference_store_path=reference_stores[encoder_id],
            destination=root / encoder_id,
            encoder_id=encoder_id,
            dataset_root=dataset_root,
            archive=archive,
            device=device,
            batch_size=batch_size,
            workers=workers,
        ))
    return outputs


def load_unpaired_store_inputs(
    store: FeatureStore,
    *,
    expected_dataset: str = "imagenet_sketch",
) -> UnpairedStoreInputs:
    """Load feature/readout roles without touching labels, IDs, or pairing APIs."""

    if not store.hashes_verified:
        raise IntegrityError("natural-shift store hashes must be verified")
    if store.dataset != expected_dataset:
        raise IntegrityError(
            f"expected dataset {expected_dataset!r}, found {store.dataset!r}"
        )
    if store.pairing_spec.get("status") != "unavailable":
        raise IntegrityError("natural-shift ceiling requires pairing to be unavailable")
    try:
        reference_role = str(store.endpoint_roles["reference"]["features"])
        target_role = str(store.endpoint_roles["target"]["features"])
        weight_role = str(store.manifest["head"]["weight_role"])
        bias_role = str(store.manifest["head"]["bias_role"])
    except (KeyError, TypeError) as error:
        raise IntegrityError("natural-shift store roles are incomplete") from error
    # These are the only four array calls in this loader.  In particular, do
    # not call Endpoint(), because that convenience object also materializes a
    # label role when labels are present.
    reference = store.array(reference_role)
    target = store.array(target_role)
    weight = store.array(weight_role)
    bias = store.array(bias_role)
    if reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        raise IntegrityError("natural-shift endpoint feature dimensions differ")
    if weight.ndim != 2 or weight.shape[1] != reference.shape[1]:
        raise IntegrityError("frozen readout dimension differs from endpoint features")
    if bias.shape != (weight.shape[0],):
        raise IntegrityError("frozen readout bias has the wrong shape")
    return UnpairedStoreInputs(
        reference_features=reference,
        target_features=target,
        readout_weight=weight,
        readout_bias=bias,
        dataset=store.dataset,
        encoder=store.encoder,
    )


def _content_split(
    features: np.ndarray,
    *,
    rows: int,
    salt: bytes,
) -> tuple[np.ndarray, np.ndarray, str]:
    value = np.asarray(features)
    if value.ndim != 2 or rows < 1 or rows > len(value):
        raise IntegrityError("invalid content split request")
    canonical = np.ascontiguousarray(value, dtype="<f8")
    keys: list[tuple[bytes, bytes, int]] = []
    for index, row in enumerate(canonical):
        raw = row.tobytes()
        keys.append((hashlib.sha256(salt + raw).digest(), raw, index))
    keys.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = np.asarray([item[2] for item in keys[:rows]], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(salt)
    for row_hash, raw, _ in keys[:rows]:
        digest.update(row_hash)
        digest.update(raw)
    return canonical[selected], selected, digest.hexdigest()


def run_unpaired_store_ceiling(
    store: FeatureStore,
    provenance_manifest: str | Path,
    *,
    source_campaign_reference_store: Mapping[str, Any],
    split_seed: int,
    calibration_rows: int,
    witness_rows: int,
    protocol: CorrespondenceCeilingProtocol,
    dataset_root: str | Path | None = None,
    archive: str | Path | None = None,
    include_posthoc_accuracy: bool = False,
) -> dict[str, Any]:
    """Validate natural data first, then run a label-free ceiling witness."""

    provenance = validate_imagenet_sketch_provenance(
        provenance_manifest, dataset_root=dataset_root, archive=archive
    )
    _validate_store_provenance_binding(
        store,
        provenance,
        supplied_source_binding=source_campaign_reference_store,
    )

    inputs = load_unpaired_store_inputs(store)
    requested = int(calibration_rows) + int(witness_rows)
    if min(int(calibration_rows), int(witness_rows)) < 2:
        raise IntegrityError("calibration_rows and witness_rows must both be at least two")
    if requested > min(len(inputs.reference_features), len(inputs.target_features)):
        raise IntegrityError("natural-shift store does not contain the requested rows")
    seed = int(split_seed)
    reference, _, reference_split_sha = _content_split(
        inputs.reference_features,
        rows=requested,
        salt=f"play3-reference-split-v1:{seed}".encode("ascii"),
    )
    target, target_indices, target_split_sha = _content_split(
        inputs.target_features,
        rows=requested,
        salt=f"play3-target-split-v1:{seed}".encode("ascii"),
    )
    calibration = int(calibration_rows)
    geometry = fit_witness_geometry(
        target[:calibration],
        reference[:calibration],
        inputs.readout_weight,
        inputs.readout_bias,
        protocol=protocol,
    )
    frontier = fit_marginal_frontier(
        target[:calibration],
        reference[:calibration],
        prefit_bures_map=geometry.bures_map,
        covariance_relative_floor=float(protocol.covariance_relative_floor),
    )
    witness_target = target[calibration:]
    witness_reference = reference[calibration:]
    witness = construct_two_world_witness(
        geometry, witness_target, witness_reference
    )
    sensitivity_witnesses = _construct_swap_neighborhood_sensitivity(
        geometry, witness_target, witness_reference
    )
    result = {
        "schema_version": NATURAL_SHIFT_RESULT_SCHEMA,
        "status": "PASS",
        "dataset": inputs.dataset,
        "encoder": inputs.encoder,
        "split_seed": seed,
        "calibration_rows": calibration,
        "witness_rows": int(witness_rows),
        "primary_quantity": (
            "conservative_witness_derived_equal_prior_randomized_minimax_"
            "frozen_reference_prediction_recovery_ceiling_across_two_"
            "constructed_finite_batch_worlds"
        ),
        "not_a_realized_world_performance_bound": True,
        "information_scope": "unpaired_features_and_frozen_readout",
        "labels_used": False,
        "labels_used_by_estimator": False,
        "posthoc_target_labels_used": bool(include_posthoc_accuracy),
        "sample_identifiers_used": False,
        "pairing_used": False,
        "source_campaign_reference_store": dict(
            source_campaign_reference_store
        ),
        "reference_split_content_sha256": reference_split_sha,
        "target_split_content_sha256": target_split_sha,
        "dataset_provenance_manifest_sha256": sha256_file(
            provenance.manifest_path
        ),
        "dataset_provenance": dict(provenance.manifest),
        "marginal_frontier_fit": dict(frontier.diagnostics),
        "ceiling_witness": witness.as_dict(include_assignments=False),
        "swap_neighborhood_sensitivity": sensitivity_witnesses,
    }
    if include_posthoc_accuracy:
        target_label_role = store.endpoint_roles.get("target", {}).get("labels")
        if not target_label_role:
            raise IntegrityError(
                "posthoc natural-shift accuracy requires a sealed target-label role"
            )
        all_target_labels = np.asarray(store.array(str(target_label_role)))
        if all_target_labels.ndim != 1 or len(all_target_labels) != len(
            inputs.target_features
        ):
            raise IntegrityError("posthoc target-label role has the wrong shape")
        selected_labels = all_target_labels[target_indices][calibration:]
        result["posthoc_evaluation"] = evaluate_natural_shift_accuracy_posthoc(
            frontier,
            witness_target,
            selected_labels,
            inputs.readout_weight,
            inputs.readout_bias,
            prediction_world_zero=witness.prediction_world_zero,
            prediction_world_one=witness.prediction_world_one,
        )
    return result


def _construct_swap_neighborhood_sensitivity(
    geometry: Any,
    target_features: np.ndarray,
    reference_features: np.ndarray,
) -> list[dict[str, Any]]:
    """Run locked greedy-search neighborhoods on the same frozen geometry."""

    values: list[dict[str, Any]] = []
    for neighbors in geometry.protocol.swap_neighbor_sensitivity:
        sensitivity_protocol = replace(
            geometry.protocol,
            swap_neighbors=int(neighbors),
            swap_neighbor_sensitivity=(),
        )
        sensitivity_geometry = replace(
            geometry, protocol=sensitivity_protocol
        )
        witness = construct_two_world_witness(
            sensitivity_geometry, target_features, reference_features
        )
        values.append({
            "swap_neighbors": int(neighbors),
            "ceiling_witness": witness.as_dict(include_assignments=False),
        })
    return values


def evaluate_natural_shift_accuracy_posthoc(
    frontier: FrozenMarginalFrontier,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
    *,
    prediction_world_zero: np.ndarray,
    prediction_world_one: np.ndarray,
) -> dict[str, Any]:
    """Evaluate labels only after fitting and both witness worlds are frozen."""

    target, order, content_sha = _canonical_rows(
        target_features,
        name="posthoc_target_features",
        salt=b"play3-target-witness-v1",
    )
    labels = np.asarray(target_labels)
    if labels.ndim != 1 or len(labels) != len(target):
        raise IntegrityError("posthoc target labels must have one value per target row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise IntegrityError("posthoc target labels must be integer class indices")
    labels = labels[order].astype(np.int64, copy=False)
    weight = np.asarray(readout_weight, dtype=np.float64)
    bias = np.asarray(readout_bias, dtype=np.float64)
    if np.any(labels < 0) or np.any(labels >= weight.shape[0]):
        raise IntegrityError("posthoc target labels are outside the frozen head class space")
    predictions = frozen_frontier_predictions(
        frontier, target, weight, bias
    )
    world_zero = np.asarray(prediction_world_zero, dtype=np.int64)
    world_one = np.asarray(prediction_world_one, dtype=np.int64)
    if world_zero.shape != labels.shape or world_one.shape != labels.shape:
        raise IntegrityError("posthoc world predictions have the wrong shape")
    scores = {
        method: float(np.mean(prediction == labels))
        for method, prediction in predictions.items()
    }
    oracle_method = max(
        MARGINAL_FRONTIER_METHODS,
        key=lambda method: (scores[method], -MARGINAL_FRONTIER_METHODS.index(method)),
    )
    return {
        "schema_version": "marginal_alignment.iclr_extension.natural_accuracy.v2",
        "status": "PASS",
        "evaluation_phase": "after_geometry_and_two_world_witness_freeze",
        "used_for_geometry_or_map_fitting": False,
        "used_for_posthoc_oracle_envelope": True,
        "target_content_sha256": content_sha,
        "rows": int(len(target)),
        "per_method_target_accuracy": scores,
        "all_six_fixed_methods_reported": True,
        "deployable_primary_reference_method": "bures_oas",
        "deployable_primary_reference_target_accuracy": scores["bures_oas"],
        "posthoc_oracle_envelope_method": oracle_method,
        "posthoc_oracle_envelope_target_accuracy": scores[oracle_method],
        "posthoc_oracle_envelope_deployable": False,
        "source_target_accuracy": scores["none"],
        "bures_target_accuracy": scores["bures_oas"],
        "per_wnid_sufficient_statistics": [
            {
                "class_index": int(class_index),
                "rows": int(np.sum(labels == class_index)),
                "prediction_disagreement_count": int(np.sum(
                    (world_zero != world_one) & (labels == class_index)
                )),
                "method_correct_counts": {
                    method: int(np.sum(
                        (prediction == labels) & (labels == class_index)
                    ))
                    for method, prediction in predictions.items()
                },
            }
            for class_index in np.unique(labels).tolist()
        ],
    }


def _pair_hash_order(
    reference_ids: Sequence[Any],
    target_ids: Sequence[Any],
    *,
    seed: int,
) -> np.ndarray:
    records: list[tuple[bytes, bytes, int]] = []
    salt = f"play3-positive-control-pair-split-v1:{int(seed)}".encode("ascii")
    for index, (reference_id, target_id) in enumerate(zip(reference_ids, target_ids)):
        raw = canonical_json_bytes({
            "reference_id": reference_id,
            "target_id": target_id,
        })
        records.append((hashlib.sha256(salt + raw).digest(), raw, index))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return np.asarray([item[2] for item in records], dtype=np.int64)


def _independent_pair_permutation(rows: int, *, seed: int, endpoint: str) -> np.ndarray:
    keys = []
    salt = f"play3-positive-control-{endpoint}-order-v1:{int(seed)}".encode("ascii")
    for index in range(rows):
        raw = str(index).encode("ascii")
        keys.append((hashlib.sha256(salt + raw).digest(), index))
    keys.sort()
    return np.asarray([index for _, index in keys], dtype=np.int64)


def run_paired_positive_control(
    store: FeatureStore,
    *,
    source_campaign_store: Mapping[str, Any],
    split_seed: int,
    calibration_rows: int,
    witness_rows: int,
    protocol: CorrespondenceCeilingProtocol,
) -> dict[str, Any]:
    """Hide verified ImageNet-C pairing from fitting, then reveal it post hoc."""

    if not store.hashes_verified or store.dataset != "imagenet_c":
        raise IntegrityError("positive control requires a verified ImageNet-C store")
    metadata = store.manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise IntegrityError("positive-control store metadata is malformed")
    corruption = str(metadata.get("corruption", ""))
    try:
        severity = int(metadata.get("severity", -1))
    except (TypeError, ValueError) as error:
        raise IntegrityError("positive-control store severity is malformed") from error
    condition = source_campaign_store.get("condition")
    actual_manifest_sha256 = sha256_file(store.manifest_path)
    if (
        source_campaign_store.get("schema_version")
        != "marginal_alignment.iclr_extension.base_campaign_store_binding.v1"
        or source_campaign_store.get("status") != "PASS"
        or not isinstance(condition, Mapping)
        or dict(condition) != {
            "dataset": store.dataset,
            "encoder": store.encoder,
            "corruption": corruption,
            "severity": severity,
        }
        or source_campaign_store.get("expected_manifest_sha256")
        != actual_manifest_sha256
        or source_campaign_store.get("actual_manifest_sha256")
        != actual_manifest_sha256
        or not _valid_sha256(source_campaign_store.get(
            "external_input_manifest_file_sha256"
        ))
        or not _valid_sha256(source_campaign_store.get("external_input_sha256"))
    ):
        raise IntegrityError(
            "positive-control store lacks its exact locked base-campaign binding"
        )
    # Pairing is used administratively to choose a common paired pool.  It is
    # never passed to the estimator, whose public API accepts feature marginals
    # and the frozen readout only.  The true assignment is scored only after the
    # maps and the two hidden worlds have been frozen below.
    pairing = store.pairing()
    paired_reference, paired_target = pairing.validated()
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    reference_lookup = {value: index for index, value in enumerate(reference.ids.tolist())}
    target_lookup = {value: index for index, value in enumerate(target.ids.tolist())}
    model = metadata.get("model_provenance")
    if not isinstance(model, Mapping):
        raise IntegrityError("positive-control store lacks model provenance")
    head_fit_ids = set(store.array(str(model["head_fit_ids_role"])).tolist())
    eligible = [
        (reference_id, target_id)
        for reference_id, target_id in zip(paired_reference, paired_target)
        if reference_id not in head_fit_ids
    ]
    requested = int(calibration_rows) + int(witness_rows)
    if min(int(calibration_rows), int(witness_rows)) < 2 or requested > len(eligible):
        raise IntegrityError("positive-control split does not fit eligible verified pairs")
    eligible_reference = [item[0] for item in eligible]
    eligible_target = [item[1] for item in eligible]
    order = _pair_hash_order(
        eligible_reference, eligible_target, seed=int(split_seed)
    )[:requested]
    selected = [eligible[int(index)] for index in order]
    reference_rows = np.asarray(
        [reference_lookup[item[0]] for item in selected], dtype=np.int64
    )
    target_rows = np.asarray(
        [target_lookup[item[1]] for item in selected], dtype=np.int64
    )
    reference_features = np.asarray(reference.features[reference_rows], dtype=np.float64)
    target_features = np.asarray(target.features[target_rows], dtype=np.float64)
    if reference.labels is None or target.labels is None:
        raise IntegrityError("positive-control administrative pairs require endpoint labels")
    selected_reference_labels = np.asarray(reference.labels[reference_rows], dtype=np.int64)
    selected_target_labels = np.asarray(target.labels[target_rows], dtype=np.int64)
    if not np.array_equal(selected_reference_labels, selected_target_labels):
        raise IntegrityError("positive-control administrative pair labels disagree")
    calibration = int(calibration_rows)

    calibration_reference_order = _independent_pair_permutation(
        calibration, seed=int(split_seed), endpoint="reference-calibration"
    )
    calibration_target_order = _independent_pair_permutation(
        calibration, seed=int(split_seed), endpoint="target-calibration"
    )
    evaluation_count = int(witness_rows)
    evaluation_reference_order = _independent_pair_permutation(
        evaluation_count, seed=int(split_seed), endpoint="reference-witness"
    )
    evaluation_target_order = _independent_pair_permutation(
        evaluation_count, seed=int(split_seed), endpoint="target-witness"
    )
    reference_calibration = reference_features[:calibration][calibration_reference_order]
    target_calibration = target_features[:calibration][calibration_target_order]
    reference_witness = reference_features[calibration:][evaluation_reference_order]
    target_witness = target_features[calibration:][evaluation_target_order]
    head = store.manifest.get("head", {})
    weight = np.asarray(store.array(str(head.get("weight_role", ""))), dtype=np.float64)
    bias = np.asarray(store.array(str(head.get("bias_role", ""))), dtype=np.float64)

    # The estimator receives only independently ordered feature marginals and
    # the frozen readout.  The pairing-derived map below is not constructed
    # until both hidden worlds and their ceiling have been frozen.
    geometry = fit_witness_geometry(
        target_calibration,
        reference_calibration,
        weight,
        bias,
        protocol=protocol,
    )
    frontier = fit_marginal_frontier(
        target_calibration,
        reference_calibration,
        prefit_bures_map=geometry.bures_map,
        covariance_relative_floor=float(protocol.covariance_relative_floor),
    )
    witness = construct_two_world_witness(
        geometry, target_witness, reference_witness
    )
    sensitivity_witnesses = _construct_swap_neighborhood_sensitivity(
        geometry, target_witness, reference_witness
    )

    canonical_target, target_canonical_order, _ = _canonical_rows(
        target_witness,
        name="positive_control_target",
        salt=b"play3-target-witness-v1",
    )
    canonical_reference, reference_canonical_order, _ = _canonical_rows(
        reference_witness,
        name="positive_control_reference",
        salt=b"play3-reference-witness-v1",
    )
    marginal_predictions = frozen_frontier_predictions(
        frontier, canonical_target, weight, bias
    )
    target_pair_positions = evaluation_target_order[target_canonical_order]
    reference_pair_positions = evaluation_reference_order[reference_canonical_order]
    inverse_reference_position = np.empty(evaluation_count, dtype=np.int64)
    inverse_reference_position[reference_pair_positions] = np.arange(
        evaluation_count, dtype=np.int64
    )
    true_assignment = inverse_reference_position[target_pair_positions]
    if sorted(true_assignment.tolist()) != list(range(evaluation_count)):
        raise RuntimeError("posthoc true assignment is not a bijection")
    reference_prediction = np.argmax(
        canonical_reference @ weight.T + bias, axis=1
    )
    true_prediction = reference_prediction[true_assignment]
    canonical_pair_labels = selected_target_labels[calibration:][target_pair_positions]
    method_agreement = {
        method: float(np.mean(prediction == true_prediction))
        for method, prediction in marginal_predictions.items()
    }
    oracle_method = max(
        MARGINAL_FRONTIER_METHODS,
        key=lambda method: (
            method_agreement[method],
            -MARGINAL_FRONTIER_METHODS.index(method),
        ),
    )
    whitened_target = (
        geometry.bures_map.apply(canonical_target) - geometry.reference_mean
    ) @ geometry.reference_inverse_sqrt
    whitened_reference = (
        canonical_reference - geometry.reference_mean
    ) @ geometry.reference_inverse_sqrt
    geometry_cost = _squared_distance_cost(
        whitened_target, whitened_reference
    )
    true_cost = float(
        geometry_cost[np.arange(evaluation_count), true_assignment].sum()
    )
    result = {
        "schema_version": "marginal_alignment.iclr_extension.positive_control.v2",
        "status": "PASS",
        "dataset": store.dataset,
        "encoder": store.encoder,
        "corruption": corruption,
        "severity": severity,
        "source_campaign_store": dict(source_campaign_store),
        "split_seed": int(split_seed),
        "calibration_rows": calibration,
        "witness_rows": evaluation_count,
        "primary_quantity": (
            "conservative_witness_derived_equal_prior_randomized_minimax_"
            "frozen_reference_prediction_recovery_ceiling_across_two_"
            "constructed_finite_batch_worlds"
        ),
        "not_a_realized_world_performance_bound": True,
        "pairing_used_for_split_administration_before_fit": True,
        "labels_used_for_administrative_pair_validation_before_fit": True,
        "pairing_unavailable_to_estimator": True,
        "labels_unavailable_to_estimator": True,
        "true_assignment_scored_only_after_witness_freeze": True,
        "labels_used_by_estimator": False,
        "selected_pair_split_sha256": sha256_json([
            {"reference_id": str(reference_id), "target_id": str(target_id)}
            for reference_id, target_id in selected
        ]),
        "marginal_frontier_fit": dict(frontier.diagnostics),
        "ceiling_witness": witness.as_dict(include_assignments=False),
        "swap_neighborhood_sensitivity": sensitivity_witnesses,
        "posthoc_validation": {
            "pairing_content_sha256": pairing.content_hash,
            "true_assignment_cost": true_cost,
            "true_to_baseline_assignment_cost_ratio": (
                float(true_cost / witness.baseline_cost)
                if witness.baseline_cost > 0.0 else None
            ),
            "world_zero_assignment_recovery": float(
                np.mean(witness.assignment_world_zero == true_assignment)
            ),
            "world_one_assignment_recovery": float(
                np.mean(witness.assignment_world_one == true_assignment)
            ),
            "per_method_reference_prediction_agreement": method_agreement,
            "all_six_fixed_methods_reported": True,
            "deployable_primary_reference_method": "bures_oas",
            "deployable_primary_reference_prediction_agreement": (
                method_agreement["bures_oas"]
            ),
            "posthoc_oracle_envelope_method": oracle_method,
            "posthoc_oracle_envelope_reference_prediction_agreement": (
                method_agreement[oracle_method]
            ),
            "posthoc_oracle_envelope_deployable": False,
            "source_reference_prediction_agreement": method_agreement["none"],
            "bures_reference_prediction_agreement": method_agreement["bures_oas"],
            "per_wnid_sufficient_statistics": [
                {
                    "class_index": int(class_index),
                    "rows": int(np.sum(canonical_pair_labels == class_index)),
                    "prediction_disagreement_count": int(np.sum(
                        (witness.prediction_world_zero != witness.prediction_world_one)
                        & (canonical_pair_labels == class_index)
                    )),
                    "method_agreement_counts": {
                        method: int(np.sum(
                            (prediction == true_prediction)
                            & (canonical_pair_labels == class_index)
                        ))
                        for method, prediction in marginal_predictions.items()
                    },
                }
                for class_index in np.unique(canonical_pair_labels).tolist()
            ],
        },
    }
    return result


def _repository_root(config_path: Path) -> Path:
    try:
        root = config_path.parents[2]
    except IndexError as error:
        raise IntegrityError("Play-3 configuration is outside the expected repository layout") from error
    if not (root / "src" / "marginal_alignment").is_dir():
        raise IntegrityError("cannot resolve repository root from Play-3 configuration")
    return root.resolve()


def load_locked_play3_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    """Load the isolated extension config and verify every locked source byte."""

    source = Path(config_path).expanduser().resolve()
    config = read_json(source)
    if not isinstance(config, Mapping):
        raise IntegrityError("Play-3 configuration must be a JSON object")
    if config.get("schema_version") != "marginal_alignment.iclr_extension.play3_natural.v1":
        raise IntegrityError("unsupported Play-3 configuration schema")
    root = _repository_root(source)
    lock_value = config.get("lock_path")
    if not isinstance(lock_value, str) or not lock_value:
        raise IntegrityError("Play-3 configuration lacks lock_path")
    lock_path = Path(lock_value)
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock_path = lock_path.resolve()
    lock = read_json(lock_path)
    if not isinstance(lock, Mapping):
        raise IntegrityError("Play-3 lock must be a JSON object")
    if lock.get("schema_version") != "marginal_alignment.iclr_extension.play3_lock.v1":
        raise IntegrityError("unsupported Play-3 lock schema")
    if lock.get("status") != "PRESPECIFIED_BEFORE_NATURAL_SHIFT_OUTCOME_INSPECTION":
        raise IntegrityError("Play-3 lock is not in its prespecified state")
    if lock.get("config_file_sha256") != sha256_file(source):
        raise IntegrityError("Play-3 configuration hash disagrees with its lock")
    master_value = config.get("master_lock_path")
    if not isinstance(master_value, str) or not master_value:
        raise IntegrityError("Play-3 configuration lacks master_lock_path")
    master_path = Path(master_value)
    if not master_path.is_absolute():
        master_path = root / master_path
    master_path = master_path.resolve()
    master = read_json(master_path)
    if not isinstance(master, Mapping):
        raise IntegrityError("ICLR extension master lock must be a JSON object")
    if (
        master.get("schema_version") != "beyond_marginals.iclr_extension_lock.v1"
        or master.get("status") != "PRESPECIFIED_BEFORE_EXTENSION_RESULTS"
    ):
        raise IntegrityError("ICLR extension master lock is not prespecified")
    if lock.get("master_lock_file_sha256") != sha256_file(master_path):
        raise IntegrityError("Play-3 lock does not bind the current master lock")
    source_files = lock.get("source_files")
    if (
        not isinstance(source_files, Mapping)
        or set(map(str, source_files)) != PLAY3_SOURCE_FILES
    ):
        raise IntegrityError("Play-3 lock source-file inventory drifted")
    required_dependencies = lock.get("required_imported_dependencies")
    if (
        not isinstance(required_dependencies, list)
        or set(map(str, required_dependencies))
        != PLAY3_REQUIRED_IMPORTED_DEPENDENCIES
        or not set(map(str, required_dependencies)).issubset(
            set(map(str, source_files))
        )
    ):
        raise IntegrityError(
            "Play-3 lock does not hash every required imported dependency"
        )
    for relative, expected in source_files.items():
        candidate = (root / str(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"locked source escapes repository: {relative}") from error
        if not candidate.is_file() or not _valid_sha256(expected):
            raise IntegrityError(f"locked source declaration is invalid: {relative}")
        if sha256_file(candidate) != expected:
            raise IntegrityError(f"locked source hash mismatch: {relative}")

    ceiling = config.get("ceiling_protocol")
    if not isinstance(ceiling, Mapping):
        raise IntegrityError("Play-3 config lacks ceiling_protocol")
    if ceiling.get("information_scope") != "unpaired_features_and_frozen_readout":
        raise IntegrityError("Play-3 information scope changed")
    if config.get("dataset", {}).get("pairing") != "unavailable":
        raise IntegrityError("Play-3 dataset must declare unavailable pairing")
    dataset = config.get("dataset", {})
    if (
        dataset.get("source_cardinality_policy")
        != "at_least_50_images_per_wnid"
        or dataset.get("selection_policy")
        != "lexicographically_first_50_basenames_per_wnid"
    ):
        raise IntegrityError("Play-3 ImageNet-Sketch selection policy changed")
    if ceiling.get("selection_labels") != "never_available":
        raise IntegrityError("Play-3 selection-label policy changed")
    master_play3 = master.get("play3_empirical_ceiling_and_natural_shift")
    if not isinstance(master_play3, Mapping):
        raise IntegrityError("master lock lacks Play-3")
    base_store_declaration = config.get("base_campaign_feature_stores")
    if (
        not isinstance(base_store_declaration, Mapping)
        or dict(base_store_declaration)
        != lock.get("base_campaign_feature_stores")
        or dict(base_store_declaration)
        != master_play3.get("base_campaign_feature_stores")
    ):
        raise IntegrityError(
            "Play-3 config, child lock, and master lock disagree on base stores"
        )
    natural_reference = base_store_declaration.get(
        "natural_reference_condition"
    )
    if natural_reference != {
        "dataset": "imagenet_c",
        "corruption": "brightness",
        "severity": 5,
    }:
        raise IntegrityError("Play-3 natural reference condition changed")
    # This reads and byte-verifies the source campaign allowlist before any
    # dataset, model, feature array, or estimator compute can begin.
    _load_locked_base_campaign_feature_store_evidence(config, lock, root)
    master_witness = master_play3.get("witness")
    master_frontier = master_play3.get("marginal_frontier")
    master_natural = master_play3.get("natural_shift")
    master_control = master_play3.get("positive_control")
    if not all(isinstance(value, Mapping) for value in (
        master_witness, master_frontier, master_natural, master_control
    )):
        raise IntegrityError("master Play-3 declarations are malformed")
    if (
        master_witness.get("tie_breaking")
        != "pinned_scipy_on_content_canonical_order"
        or ceiling.get("world_zero") != "deterministic_minimum_cost_bijection"
        or ceiling.get("world_one")
        != "deterministic_greedy_disjoint_all_pair_swaps"
        or ceiling.get("world_one_optimality_claimed") is not False
        or ceiling.get("ceiling_tightness")
        != "conservative_witness_derived_equal_prior_randomized_minimax_ceiling"
        or master_witness.get("alternative_search_optimality_claimed") is not False
    ):
        raise IntegrityError("Play-3 assignment or tie-breaking rule changed")
    frontier = config.get("marginal_frontier")
    if not isinstance(frontier, Mapping):
        raise IntegrityError("Play-3 config lacks marginal_frontier")
    frontier_checks = {
        "methods": list(MARGINAL_FRONTIER_METHODS),
        "coordinate_quantile_knots": 257,
        "covariance_estimator": "oas",
        "covariance_relative_floor": 1e-6,
        "diagonal_standard_deviation_floor": 1e-6,
        "fit_information": "unpaired_calibration_features_only",
        "deployable_primary_reference": "bures_oas",
        "posthoc_oracle_envelope_deployable": False,
    }
    for key, expected in frontier_checks.items():
        if frontier.get(key) != expected or master_frontier.get(key) != expected:
            raise IntegrityError(f"Play-3 marginal-frontier declaration changed: {key}")
    expected_pairs = {
        "witness_rows": int(master_witness["batch_size"]),
        "cost_geometry": str(master_witness["cost_geometry"]),
        "cost_normalization": str(master_witness["cost_normalization"]),
        "discarded_feature_dimensions": int(
            master_witness["discarded_feature_dimensions"]
        ),
        "primary_cost_budget_relative": float(
            master_witness["primary_cost_slack_fraction"]
        ),
    }
    for key, expected in expected_pairs.items():
        if ceiling.get(key) != expected:
            raise IntegrityError(f"Play-3 config disagrees with master lock: {key}")
    if [float(value) for value in ceiling["cost_budget_relative_sensitivity"]] != [
        float(value) for value in master_witness["sensitivity_cost_slack_fractions"]
    ]:
        raise IntegrityError("Play-3 sensitivity budgets disagree with master lock")
    swap_sensitivity = [
        int(value) for value in ceiling.get("swap_neighbor_sensitivity", [])
    ]
    if (
        swap_sensitivity
        != [int(value) for value in master_witness.get(
            "swap_neighbor_sensitivity", []
        )]
        or len(set(swap_sensitivity)) != len(swap_sensitivity)
        or any(value < 1 or value >= int(ceiling["swap_neighbors"])
               for value in swap_sensitivity)
    ):
        raise IntegrityError(
            "Play-3 swap-neighborhood sensitivity disagrees with the master lock"
        )
    if [int(value) for value in ceiling["split_seeds"]] != [
        int(value) for value in master_natural["split_seeds"]
    ]:
        raise IntegrityError("Play-3 split seeds disagree with master lock")
    if int(ceiling.get("swap_neighbors", -1)) != int(ceiling["witness_rows"]):
        raise IntegrityError("primary Play-3 witness must search all target-row pairs")
    reporting = config.get("reporting")
    uncertainty = reporting.get("uncertainty") if isinstance(reporting, Mapping) else None
    expected_uncertainty = master_play3.get("uncertainty")
    if not isinstance(uncertainty, Mapping) or not isinstance(
        expected_uncertainty, Mapping
    ):
        raise IntegrityError("Play-3 uncertainty declaration is absent")
    if (
        reporting.get("primary_quantity") != master_play3.get("quantity_name")
        or reporting.get("not_a_realized_world_performance_bound") is not True
    ):
        raise IntegrityError("Play-3 quantity interpretation changed")
    uncertainty_checks = {
        "replicates": int(expected_uncertainty["replicates"]),
        "rng": str(expected_uncertainty["rng"]),
        "seed": int(expected_uncertainty["seed"]),
        "interval": str(expected_uncertainty["interval"]),
        "zero_silent_drops": bool(expected_uncertainty["zero_silent_drops"]),
        "secondary_spearman_undefined_policy": str(
            expected_uncertainty["secondary_spearman_undefined_policy"]
        ),
    }
    for key, expected in uncertainty_checks.items():
        if uncertainty.get(key) != expected:
            raise IntegrityError(f"Play-3 uncertainty declaration changed: {key}")
    if (
        uncertainty.get("natural_cluster") != "imagenet_wnid"
        or uncertainty.get("positive_cluster") != "corruption"
        or uncertainty.get("split_seed_policy")
        != "average_over_overlapping_repeats_not_independent_bootstrap_units"
        or uncertainty.get("encoders_nested_within_cluster") is not True
        or expected_uncertainty.get("natural")
        != "wnid_cluster_bootstrap_with_encoders_nested_and_overlapping_split_seeds_averaged"
        or expected_uncertainty.get("positive_control")
        != "corruption_cluster_bootstrap_with_encoders_nested_and_overlapping_split_seeds_averaged"
    ):
        raise IntegrityError("Play-3 cluster-bootstrap policy changed")
    if list(config.get("encoders", [])) != list(master_natural["encoders"]):
        raise IntegrityError("Play-3 encoders disagree with master lock")
    if (
        config.get("dataset", {}).get("name") != master_natural["dataset"]
        or int(config["dataset"]["class_count"]) != int(master_natural["classes"])
        or int(config["dataset"]["row_count"]) != int(master_natural["expected_images"])
    ):
        raise IntegrityError("Play-3 natural dataset disagrees with master lock")
    if list(config.get("positive_control", {}).get("heldout_corruptions", [])) != list(
        master_control["heldout_corruptions"]
    ):
        raise IntegrityError("Play-3 positive-control panel disagrees with master lock")
    positive = config["positive_control"]
    for key in (
        "pairing_used_for_split_administration_before_fit",
        "labels_used_for_administrative_pair_validation_before_fit",
        "pairing_unavailable_to_estimator",
        "labels_unavailable_to_estimator",
        "true_assignment_scored_only_after_witness_freeze",
    ):
        if positive.get(key) is not True or master_control.get(key) is not True:
            raise IntegrityError(f"Play-3 positive-control scope changed: {key}")
    return dict(config), source, dict(lock), root


def _runtime_manifest() -> dict[str, Any]:
    try:
        import scipy
    except ImportError as error:  # pragma: no cover - declared full dependency
        raise RuntimeError("Play-3 requires scipy") from error
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "thread_environment": thread_variables,
    }


def capture_play3_runtime() -> dict[str, Any]:
    """Return the runtime identity embedded by every Play-3 unit type."""

    return _runtime_manifest()


def _resolve_cli_input(
    explicit: Path | None,
    *,
    environment_name: str,
) -> Path:
    value = explicit if explicit is not None else (
        Path(os.environ[environment_name]) if os.environ.get(environment_name) else None
    )
    if value is None:
        raise IntegrityError(
            f"provide the path explicitly or set {environment_name}"
        )
    return Path(value).expanduser().resolve()


def _validate_store_provenance_binding(
    store: FeatureStore,
    provenance: ValidatedImageNetSketchProvenance,
    *,
    source_evidence: BaseCampaignFeatureStoreEvidence | None = None,
    natural_reference_condition: Mapping[str, Any] | None = None,
    supplied_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = store.manifest.get("metadata", {}).get("imagenet_sketch_provenance")
    if not isinstance(binding, Mapping):
        raise IntegrityError("feature store lacks ImageNet-Sketch provenance binding")
    expected = {
        "manifest_sha256": sha256_file(provenance.manifest_path),
        "file_tree_sha256": provenance.discovery.file_tree_sha256,
        "class_index_sha256": provenance.manifest["class_index_sha256"],
        "row_count": EXPECTED_ROWS,
    }
    mismatches = {
        key: {"expected": value, "actual": binding.get(key)}
        for key, value in expected.items()
        if binding.get(key) != value
    }
    if mismatches:
        raise IntegrityError(
            f"feature-store and dataset provenance bindings disagree: {mismatches}"
        )
    class_identity = store.manifest.get("metadata", {}).get(
        "reference_class_identity"
    )
    if (
        not isinstance(class_identity, Mapping)
        or class_identity.get("status") != "PASS"
        or class_identity.get("reference_id_kind") != "wnid_basename"
        or class_identity.get("class_index_sha256")
        != provenance.manifest["class_index_sha256"]
        or not _valid_sha256(class_identity.get("reference_id_label_mapping_sha256"))
    ):
        raise IntegrityError(
            "feature store lacks verified ImageNet reference ID-to-label parity"
        )
    source_binding = store.manifest.get("metadata", {}).get(
        "source_campaign_reference_store"
    )
    if not isinstance(source_binding, Mapping):
        raise IntegrityError(
            "feature store lacks its exact base-campaign reference-store binding"
        )
    condition = source_binding.get("condition")
    if (
        source_binding.get("schema_version")
        != "marginal_alignment.iclr_extension.base_campaign_store_binding.v1"
        or source_binding.get("status") != "PASS"
        or not isinstance(condition, Mapping)
        or set(condition) != {"dataset", "encoder", "corruption", "severity"}
        or condition.get("dataset") != "imagenet_c"
        or condition.get("encoder") != store.encoder
        or not _valid_sha256(source_binding.get(
            "external_input_manifest_file_sha256"
        ))
        or not _valid_sha256(source_binding.get("external_input_sha256"))
        or not _valid_sha256(source_binding.get("expected_manifest_sha256"))
        or source_binding.get("actual_manifest_sha256")
        != source_binding.get("expected_manifest_sha256")
    ):
        raise IntegrityError(
            "feature store has a malformed base-campaign reference-store binding"
        )
    if (
        supplied_source_binding is not None
        and dict(source_binding) != dict(supplied_source_binding)
    ):
        raise IntegrityError(
            "feature store reference binding changed after locked preflight"
        )
    if source_evidence is not None:
        if not isinstance(natural_reference_condition, Mapping):
            raise IntegrityError("natural reference condition is missing")
        expected_condition = {
            "dataset": str(natural_reference_condition.get("dataset", "")),
            "encoder": store.encoder,
            "corruption": str(natural_reference_condition.get("corruption", "")),
            "severity": int(natural_reference_condition.get("severity", -1)),
        }
        expected_sha = source_evidence.expected_manifest_sha256(
            **expected_condition
        )
        if (
            dict(condition) != expected_condition
            or source_binding.get("expected_manifest_sha256") != expected_sha
            or source_binding.get("actual_manifest_sha256") != expected_sha
            or source_binding.get("external_input_manifest_file_sha256")
            != source_evidence.manifest_file_sha256
            or source_binding.get("external_input_sha256")
            != source_evidence.external_input_sha256
        ):
            raise IntegrityError(
                "feature store reference binding disagrees with the locked base campaign"
            )
    return dict(source_binding)


def build_unit_plan(
    *,
    config_path: str | Path,
    store_path: str | Path,
    provenance_path: str | Path,
    dataset_root: str | Path | None,
    archive: str | Path | None,
    split_seed: int,
    cost_budget_relative: float,
) -> tuple[dict[str, Any], FeatureStore, ValidatedImageNetSketchProvenance,
           CorrespondenceCeilingProtocol]:
    """Run all hashes and eligibility gates without fitting an alignment."""

    config, resolved_config, lock, repository = load_locked_play3_config(config_path)
    dataset = config["dataset"]
    provenance = validate_imagenet_sketch_provenance(
        provenance_path, dataset_root=dataset_root, archive=archive
    )
    store = load_feature_store(store_path, verify_hashes=True)
    if store.dataset != dataset["name"]:
        raise IntegrityError(
            f"expected store dataset {dataset['name']!r}, found {store.dataset!r}"
        )
    if store.encoder not in config["encoders"]:
        raise IntegrityError(f"encoder is outside the Play-3 lock: {store.encoder}")
    if store.pairing_spec.get("status") != "unavailable":
        raise IntegrityError("Play-3 store exposes pairing and is therefore ineligible")
    source_evidence = _load_locked_base_campaign_feature_store_evidence(
        config, lock, repository
    )
    reference_condition = config["base_campaign_feature_stores"][
        "natural_reference_condition"
    ]
    source_binding = _validate_store_provenance_binding(
        store,
        provenance,
        source_evidence=source_evidence,
        natural_reference_condition=reference_condition,
    )
    declared = config["ceiling_protocol"]
    seeds = [int(value) for value in declared["split_seeds"]]
    seed = int(split_seed)
    if seed not in seeds:
        raise IntegrityError(f"split seed is outside the Play-3 lock: {seed}")
    allowed_budgets = [
        float(declared["primary_cost_budget_relative"]),
        *[
            float(value)
            for value in declared["cost_budget_relative_sensitivity"]
        ],
    ]
    budget = float(cost_budget_relative)
    if not any(abs(budget - value) <= 1e-15 for value in allowed_budgets):
        raise IntegrityError(f"cost budget is outside the Play-3 lock: {budget}")
    protocol = CorrespondenceCeilingProtocol(
        cost_budget_relative=budget,
        covariance_estimator=declared["covariance_estimator"],
        covariance_relative_floor=float(declared["covariance_relative_floor"]),
        maximum_witness_rows=int(declared["maximum_witness_rows"]),
        swap_neighbors=int(declared["swap_neighbors"]),
        swap_neighbor_sensitivity=tuple(
            int(value) for value in declared["swap_neighbor_sensitivity"]
        ),
        assignment_tie_relative_tolerance=float(
            declared["assignment_tie_relative_tolerance"]
        ),
    )
    unit_id = (
        f"imagenet_sketch__{store.encoder}__seed_{seed}__"
        f"budget_{int(round(10000 * budget)):04d}bp"
    )
    plan = {
        "schema_version": "marginal_alignment.iclr_extension.play3_unit_plan.v1",
        "status": "PASS",
        "scientific_result": False,
        "compute_started": False,
        "unit_id": unit_id,
        "dataset": store.dataset,
        "encoder": store.encoder,
        "split_seed": seed,
        "cost_budget_relative": budget,
        "calibration_rows": int(declared["calibration_rows"]),
        "witness_rows": int(declared["witness_rows"]),
        "config_path": str(resolved_config),
        "config_sha256": sha256_file(resolved_config),
        "lock_manifest_sha256": sha256_json(lock),
        "store_manifest": str(store.manifest_path),
        "store_manifest_sha256": sha256_file(store.manifest_path),
        "source_campaign_reference_store": source_binding,
        "dataset_provenance_manifest": str(provenance.manifest_path),
        "dataset_provenance_manifest_sha256": sha256_file(
            provenance.manifest_path
        ),
        "file_tree_sha256": provenance.discovery.file_tree_sha256,
        "information_scope": "unpaired_features_and_frozen_readout",
        "labels_used": False,
        "pairing_used": False,
        "protocol": protocol.as_dict(),
        "runtime": _runtime_manifest(),
    }
    return plan, store, provenance, protocol


def write_immutable_unit_result(
    destination: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Publish a sealed unit result without overwrite or resume semantics."""

    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing unit output: {output}")
    write_json_atomic(output, seal_manifest(dict(payload)))
    stored = read_json(output)
    if not isinstance(stored, Mapping):
        raise IntegrityError("published unit result is malformed")
    verify_manifest_checksum(stored)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run one immutable ImageNet-Sketch Play-3 unit"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--cost-budget-relative", required=True, type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args(argv)

    config, _, _, _ = load_locked_play3_config(arguments.config)
    dataset_config = config["dataset"]
    provenance_path = _resolve_cli_input(
        arguments.provenance,
        environment_name=str(dataset_config["provenance_environment"]),
    )
    dataset_root = _resolve_cli_input(
        arguments.dataset_root,
        environment_name=str(dataset_config["dataset_root_environment"]),
    )
    archive = _resolve_cli_input(
        arguments.archive,
        environment_name=str(dataset_config["source_archive_environment"]),
    )
    plan, store, _, protocol = build_unit_plan(
        config_path=arguments.config,
        store_path=arguments.store,
        provenance_path=provenance_path,
        dataset_root=dataset_root,
        archive=archive,
        split_seed=arguments.split_seed,
        cost_budget_relative=arguments.cost_budget_relative,
    )
    if arguments.plan_only:
        if arguments.output is not None:
            raise IntegrityError("--plan-only prints to stdout and does not accept --output")
        print(json.dumps(plan, sort_keys=True))
        return 0
    if arguments.output is None:
        raise IntegrityError("a full unit requires a fresh --output path")
    output = arguments.output.expanduser().resolve()
    protected = {
        arguments.config.expanduser().resolve(),
        store.manifest_path.resolve(),
        provenance_path,
        archive,
    }
    if output in protected or store.root.resolve() in output.parents or dataset_root in output.parents:
        raise IntegrityError("unit output overlaps a protected input path")

    declared = config["ceiling_protocol"]
    result = run_unpaired_store_ceiling(
        store,
        provenance_path,
        source_campaign_reference_store=plan[
            "source_campaign_reference_store"
        ],
        split_seed=arguments.split_seed,
        calibration_rows=int(declared["calibration_rows"]),
        witness_rows=int(declared["witness_rows"]),
        protocol=protocol,
        dataset_root=dataset_root,
        archive=archive,
        include_posthoc_accuracy=True,
    )
    complete = {
        **result,
        "unit_plan": plan,
        "scientific_result": True,
        "immutable_fresh_output": True,
    }
    destination = write_immutable_unit_result(output, complete)
    print(json.dumps({
        "status": "PASS",
        "unit_id": plan["unit_id"],
        "output": str(destination),
        "output_sha256": sha256_file(destination),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
