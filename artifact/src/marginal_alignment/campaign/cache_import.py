"""Import documented server feature arrays into strict per-condition V2 stores.

The importer never invents identity.  It needs an explicit block-index sidecar
or a dataset-specific canonical-index declaration whose generator inputs and
arrays are all hashed.  Its output is small: manifests, IDs, pairing sidecars,
and content-addressed, hard-linked shared endpoint artifacts.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import (
    IntegrityError, describe_npy, read_json, seal_manifest, sha256_file,
    sha256_json, verify_file_record, verify_manifest_checksum, write_json_atomic,
)
from .feature_store import NATIVE_MANIFEST, build_native_manifest, load_feature_store


CIFAR100_C_CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow",
    "spatter", "speckle_noise", "zoom_blur",
)
IMAGENET_C_CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
    "pixelate", "shot_noise", "snow", "zoom_blur",
)
CAMPAIGN_ENCODERS = ("dinov3", "clip", "vitb16")
CAMPAIGN_SEVERITIES = (1, 3, 5)
EXPECTED_STORE_COUNT = (
    len(CAMPAIGN_ENCODERS)
    * len(CAMPAIGN_SEVERITIES)
    * (len(CIFAR100_C_CORRUPTIONS) + len(IMAGENET_C_CORRUPTIONS))
)
SHARED_SCHEMA = "marginal_alignment.shared_feature_arrays.v1"
MATERIALIZATION_SCHEMA = "marginal_alignment.shared_hardlinks.v1"
SHARED_MANIFEST = "SHARED_ARRAYS_MANIFEST.json"
SHARED_ROLES = (
    "reference_features", "reference_labels", "reference_ids", "target_labels",
    "target_ids", "pair_reference_ids", "pair_target_ids", "head_weight", "head_bias",
    "head_fit_ids",
)


# Conversion invokes ``import_condition`` dozens of times in one process.  A seal
# is immutable input to that process, so hashing every multi-GB source array for
# every condition is pure duplicated I/O.  The cache key changes if the seal file
# itself is replaced.
_SOURCE_SEAL_CACHE: dict[
    tuple[str, int, int], tuple[dict[str, str], str, dict[str, tuple[int, int, int]]]
] = {}
_VERIFIED_INODE_HASHES: dict[tuple[int, int, int, int], str] = {}
_SHARED_ARRAY_CACHE: dict[tuple[str, str], dict[str, Path]] = {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"value": value}


def inspect_legacy_cache(root: str | Path) -> dict[str, Any]:
    """Inventory shapes/metadata without loading feature bodies into RAM."""

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        record: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            record.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        elif path.suffix == ".json":
            record["json"] = _load_json(path)
        files.append(record)
    recursive_sidecars = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.parent == directory:
            continue
        relative = path.relative_to(directory).as_posix()
        lowered = relative.lower()
        if not any(token in lowered for token in (
            "block", "index", "manifest", "metadata", "sample", "filename", "basename", "wnid"
        )):
            continue
        record: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size}
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            record.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        elif path.suffix == ".json" and path.stat().st_size <= 16 * 1024 * 1024:
            record["json"] = _load_json(path)
        recursive_sidecars.append(record)
    return {"root": str(directory), "files": files, "recursive_sidecars": recursive_sidecars}


def write_inventory(roots: list[str | Path], destination: str | Path) -> Path:
    result = []
    for root in roots:
        path = Path(root).expanduser()
        if path.is_dir():
            result.append(inspect_legacy_cache(path))
    output = Path(destination)
    write_json_atomic(output, {
        "schema_version": "marginal_alignment.campaign.cache_inventory.v2",
        "stores": result,
    })
    return output


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise IntegrityError(f"existing link points to a different artifact: {destination}")
        return
    os.symlink(os.path.relpath(source.resolve(), destination.parent.resolve()), destination)


def _require_sha_manifest(source_root: Path) -> dict[str, str]:
    manifest_path = source_root / "SOURCE_ARRAY_SHA256.json"
    if not manifest_path.is_file():
        raise IntegrityError(
            f"missing {manifest_path}; run scripts/campaign/seal_legacy_arrays.py after inventory"
        )
    stat = manifest_path.stat()
    cache_key = (str(manifest_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _SOURCE_SEAL_CACHE.get(cache_key)
    if cached is not None:
        unchanged = True
        for name, expected_stat in cached[2].items():
            path = source_root / name
            if not path.is_file():
                unchanged = False
                break
            current = path.stat()
            if (int(current.st_size), int(current.st_mtime_ns), int(current.st_ino)) != expected_stat:
                unchanged = False
                break
        if unchanged:
            return dict(cached[0])
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise IntegrityError("source array hash manifest is malformed")
    result = {str(name): str(value) for name, value in files.items()}
    print(f"[cache] verifying source seal: {source_root} ({len(result)} files)", flush=True)
    source_stats = {}
    for name, expected in result.items():
        path = source_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(f"source array hash mismatch: {path}")
        path_stat = path.stat()
        source_stats[name] = (
            int(path_stat.st_size), int(path_stat.st_mtime_ns), int(path_stat.st_ino),
        )
    _SOURCE_SEAL_CACHE[cache_key] = (
        dict(result), sha256_file(manifest_path), source_stats,
    )
    print(f"[cache] source seal PASS: {source_root}", flush=True)
    return result


def seal_legacy_arrays(source_root: str | Path) -> Path:
    """Hash all reusable NPY/JSON sidecars once; can take minutes on large caches."""

    root = Path(source_root).expanduser().resolve()
    destination = root / "SOURCE_ARRAY_SHA256.json"
    if destination.is_file():
        # A previous complete seal is reusable, but only after its content is
        # checked.  This makes an interrupted launcher safe without silently
        # trusting a stale manifest.
        _require_sha_manifest(root)
        print(f"[cache] source seal verified: {root}", flush=True)
        return destination
    files = {}
    source_stats = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "SOURCE_ARRAY_SHA256.json" and path.suffix in {".npy", ".json"}:
            files[path.name] = sha256_file(path)
            stat = path.stat()
            source_stats[path.name] = (
                int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino),
            )
    if not files:
        raise IntegrityError(f"no sealable arrays under {root}")
    write_json_atomic(destination, {
        "schema_version": "marginal_alignment.campaign.legacy_array_seal.v2",
        "root": str(root),
        "files": files,
    })
    # The bytes were just hashed above; populate the process cache without a
    # second full read of the same feature bodies.
    manifest_stat = destination.stat()
    cache_key = (
        str(destination.resolve()), int(manifest_stat.st_mtime_ns), int(manifest_stat.st_size),
    )
    _SOURCE_SEAL_CACHE[cache_key] = (
        dict(files), sha256_file(destination), source_stats,
    )
    print(f"[cache] source seal created: {root} ({len(files)} files)", flush=True)
    return destination


def _validated_model_provenance(
    metadata: Mapping[str, Any], *, encoder: str,
) -> dict[str, Any]:
    """Validate the source-cache representation and frozen-head provenance."""

    raw = metadata.get("model_provenance")
    if not isinstance(raw, Mapping):
        raise IntegrityError(
            "reference metadata needs model_provenance with encoder/checkpoint/"
            "preprocessing/layer/pooling/head-fit identity"
        )
    result = dict(raw)
    required = {
        "encoder_id", "checkpoint_sha256", "preprocessing_sha256",
        "feature_layer", "pooling", "head_protocol", "head_fit_ids_sha256",
        "head_fit_ids_file", "head_fit_id_namespace", "reference_id_namespace",
        "reference_split", "head_fit_split",
    }
    missing = sorted(key for key in required if not result.get(key))
    if missing:
        raise IntegrityError(f"source model_provenance is missing fields: {missing}")
    if str(result["encoder_id"]) != str(encoder):
        raise IntegrityError(
            f"source encoder identity {result['encoder_id']!r} != requested {encoder!r}"
        )
    for key in ("checkpoint_sha256", "preprocessing_sha256", "head_fit_ids_sha256"):
        value = str(result[key])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise IntegrityError(f"source model_provenance {key} is not a SHA-256")
    fit_ids_path = Path(str(result["head_fit_ids_file"]))
    if fit_ids_path.is_absolute() or ".." in fit_ids_path.parts:
        raise IntegrityError("head_fit_ids_file must be a safe path inside the reference cache")
    if not str(result["reference_id_namespace"]).strip():
        raise IntegrityError("reference_id_namespace must be nonempty")
    if not str(result["reference_split"]).strip() or not str(result["head_fit_split"]).strip():
        raise IntegrityError("reference_split and head_fit_split must be nonempty")
    return result


def _published_model_provenance(model_provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in model_provenance.items()
           if key != "head_fit_ids_file"},
        "head_fit_ids_role": "head_fit_ids",
    }


def _array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": np.dtype(array.dtype).str,
        "data_sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _save_npy_fsynced(path: Path, value: np.ndarray) -> None:
    """Write one NPY before publication without relying on suffix rewriting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_generated_path(path: Path) -> None:
    """Remove only a converter-owned staging/backup path."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _publish_directory(staging: Path, destination: Path) -> None:
    """Atomically swap a validated staging directory, with rollback."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and not destination.is_symlink():
        os.replace(staging, destination)
        return
    backup = destination.with_name(f".{destination.name}.previous.{os.getpid()}")
    if backup.exists() or backup.is_symlink():
        _remove_generated_path(backup)
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    _remove_generated_path(backup)


def _verified_manifest_store(manifest_path: Path):
    """Validate a store while hashing each hard-linked inode at most once."""

    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise IntegrityError(f"feature-store manifest is malformed: {manifest_path}")
    verify_manifest_checksum(manifest)
    records = manifest.get("arrays")
    if not isinstance(records, Mapping):
        raise IntegrityError("feature-store arrays are malformed")
    for role, record in records.items():
        if not isinstance(record, Mapping):
            raise IntegrityError(f"feature-store record is malformed: {role}")
        path = verify_file_record(manifest_path.parent, record, verify_hash=False)
        stat = path.stat()
        inode_key = (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
        digest = _VERIFIED_INODE_HASHES.get(inode_key)
        if digest is None:
            digest = sha256_file(path)
            _VERIFIED_INODE_HASHES[inode_key] = digest
        if digest != str(record.get("sha256")):
            raise IntegrityError(f"artifact SHA-256 mismatch for {path}")
    # Hashes were checked above with hard-link memoization.  The ordinary loader
    # still applies every shape, ID, pairing, model, and head semantic check.
    return load_feature_store(manifest_path, verify_hashes=False)


def _condition_reusable(
    output: Path, *, dataset: str, encoder: str,
    expected_metadata: Mapping[str, Any], pairing_provenance: Mapping[str, Any],
) -> tuple[bool, str]:
    manifest_path = output / NATIVE_MANIFEST
    if not manifest_path.is_file():
        return False, "manifest absent"
    try:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            return False, "manifest malformed"
        verify_manifest_checksum(manifest)
        if manifest.get("dataset") != dataset or manifest.get("encoder") != encoder:
            return False, "dataset/encoder mismatch"
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            return False, "metadata malformed"
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                return False, f"metadata mismatch: {key}"
        pairing = manifest.get("pairing")
        if not isinstance(pairing, Mapping) or pairing.get("status") != "available":
            return False, "verified pairing absent"
        if pairing.get("provenance") != dict(pairing_provenance):
            return False, "pairing provenance mismatch"
        records = manifest.get("arrays")
        if not isinstance(records, Mapping) or set(records) != {
            "reference_features", "target_features", "reference_labels", "target_labels",
            "reference_ids", "target_ids", "pair_reference_ids", "pair_target_ids",
            "head_weight", "head_bias", "head_fit_ids",
        }:
            return False, "array-role contract mismatch"
        for role in SHARED_ROLES:
            record = records[role]
            if not isinstance(record, Mapping):
                return False, f"array record malformed: {role}"
            path = verify_file_record(manifest_path.parent, record, verify_hash=False)
            if path.stat().st_nlink < 2:
                return False, f"shared role is not hard-linked: {role}"
        _verified_manifest_store(manifest_path)
    except (IntegrityError, FileNotFoundError, OSError, ValueError) as error:
        return False, f"validation failed: {error}"
    return True, "validated source/model/content match"


def _shared_signature(
    *, dataset: str, encoder: str, reference_root: Path, target_root: Path,
    reference_seal_sha256: str, target_seal_sha256: str,
    reference_feature_sha256: str, model_provenance: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    small_roles = [role for role in SHARED_ROLES if role not in {
        "reference_features", "head_weight", "head_bias", "head_fit_ids",
    }]
    return {
        "dataset": dataset,
        "encoder": encoder,
        "source_reference_root": str(reference_root),
        "source_target_root": str(target_root),
        "source_reference_seal_sha256": reference_seal_sha256,
        "source_target_seal_sha256": target_seal_sha256,
        "source_reference_feature_sha256": reference_feature_sha256,
        "model_provenance": _published_model_provenance(model_provenance),
        "small_array_fingerprints": {
            role: _array_fingerprint(arrays[role]) for role in small_roles
        },
    }


def _load_shared_arrays(shared: Path, signature: Mapping[str, Any]) -> dict[str, Path]:
    manifest_path = shared / SHARED_MANIFEST
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SHARED_SCHEMA:
        raise IntegrityError(f"not a {SHARED_SCHEMA} manifest: {manifest_path}")
    verify_manifest_checksum(manifest)
    if manifest.get("signature") != dict(signature):
        raise IntegrityError("shared-array source/model signature mismatch")
    records = manifest.get("arrays")
    if not isinstance(records, Mapping) or set(records) != set(SHARED_ROLES):
        raise IntegrityError("shared-array roles are incomplete")
    result = {}
    for role, record in records.items():
        if not isinstance(record, Mapping):
            raise IntegrityError(f"shared-array record is malformed: {role}")
        result[str(role)] = verify_file_record(shared, record, verify_hash=True)
    return result


def _ensure_shared_arrays(
    shared: Path, *, signature: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
) -> dict[str, Path]:
    signature_sha = sha256_json(signature)
    cache_key = (str(shared.resolve()), signature_sha)
    if cache_key in _SHARED_ARRAY_CACHE:
        return dict(_SHARED_ARRAY_CACHE[cache_key])
    try:
        result = _load_shared_arrays(shared, signature)
        print(f"[cache] shared arrays REUSE: {shared}", flush=True)
    except (IntegrityError, FileNotFoundError, OSError, ValueError) as error:
        shared.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{shared.name}.building.", dir=shared.parent))
        try:
            records = {}
            for role in SHARED_ROLES:
                path = staging / f"{role}.npy"
                _save_npy_fsynced(path, arrays[role])
                records[role] = describe_npy(path, relative_to=staging)
            write_json_atomic(staging / SHARED_MANIFEST, seal_manifest({
                "schema_version": SHARED_SCHEMA,
                "signature": dict(signature),
                "arrays": records,
            }))
            _load_shared_arrays(staging, signature)
            _publish_directory(staging, shared)
            result = _load_shared_arrays(shared, signature)
            print(f"[cache] shared arrays BUILD: {shared} ({error})", flush=True)
        finally:
            if staging.exists():
                _remove_generated_path(staging)
    _SHARED_ARRAY_CACHE[cache_key] = dict(result)
    return result


def _hardlink_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.link.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def import_condition(
    *, source_reference_root: str | Path, source_target_root: str | Path,
    destination: str | Path, dataset: str, encoder: str,
    reference_feature_file: str, reference_label_file: str,
    target_feature_file: str, target_label_file: str,
    reference_rows: slice, target_rows: slice,
    reference_ids: np.ndarray, target_ids: np.ndarray,
    pairing_reference_ids: np.ndarray, pairing_target_ids: np.ndarray,
    pairing_provenance: Mapping[str, Any], head_weight_file: str,
    head_bias_file: str, metadata: Mapping[str, Any],
    model_provenance: Mapping[str, Any],
    shared_root: str | Path | None = None,
) -> Path:
    """Idempotently publish one condition after verifying source and model seals.

    A condition is reusable only when the complete manifest and every array
    verify, and its dataset, condition, source seals, and model provenance equal
    the current request.  Rebuilds are staged and validated before an atomic
    directory swap.  Endpoint/head artifacts invariant across conditions are
    local hard links to one verified dataset/encoder shared copy.
    """

    reference_root = Path(source_reference_root).expanduser().resolve()
    target_root = Path(source_target_root).expanduser().resolve()
    reference_seal = _require_sha_manifest(reference_root)
    target_seal = _require_sha_manifest(target_root)
    output = Path(destination).expanduser().resolve()
    shared = (
        Path(shared_root).expanduser().resolve()
        if shared_root is not None
        else (output.parent / "_shared" / dataset / encoder).resolve()
    )

    required_reference_files = {
        reference_feature_file, reference_label_file, head_weight_file, head_bias_file,
        str(model_provenance["head_fit_ids_file"]),
    }
    required_target_files = {target_feature_file, target_label_file}
    missing_reference = sorted(required_reference_files.difference(reference_seal))
    missing_target = sorted(required_target_files.difference(target_seal))
    if missing_reference or missing_target:
        raise IntegrityError(
            f"condition sources are absent from their seals: "
            f"reference={missing_reference}, target={missing_target}"
        )
    reference_seal_sha = sha256_file(reference_root / "SOURCE_ARRAY_SHA256.json")
    target_seal_sha = sha256_file(target_root / "SOURCE_ARRAY_SHA256.json")
    published_metadata = {
        **dict(metadata),
        "model_provenance": _published_model_provenance(model_provenance),
        "source_reference_root": str(reference_root),
        "source_target_root": str(target_root),
        "source_reference_seal_sha256": reference_seal_sha,
        "source_target_seal_sha256": target_seal_sha,
        "source_reference_feature_sha256": reference_seal[reference_feature_file],
        "source_target_feature_sha256": target_seal[target_feature_file],
        "materialization": {
            "schema_version": MATERIALIZATION_SCHEMA,
            "shared_key": f"{dataset}/{encoder}",
            "shared_roles": list(SHARED_ROLES),
        },
    }
    reusable, reason = _condition_reusable(
        output, dataset=dataset, encoder=encoder,
        expected_metadata=published_metadata, pairing_provenance=pairing_provenance,
    )
    condition_name = (
        f"{dataset}/{encoder}/{metadata.get('corruption', '?')}/"
        f"severity_{metadata.get('severity', '?')}"
    )
    if reusable:
        print(f"[cache] condition REUSE: {condition_name}", flush=True)
        return output / NATIVE_MANIFEST
    print(f"[cache] condition BUILD: {condition_name} ({reason})", flush=True)

    reference_features_all = np.load(reference_root / reference_feature_file, mmap_mode="r")
    target_features_all = np.load(target_root / target_feature_file, mmap_mode="r")
    reference_labels_all = np.load(reference_root / reference_label_file, mmap_mode="r")
    target_labels_all = np.load(target_root / target_label_file, mmap_mode="r")
    head_fit_ids_file = str(model_provenance["head_fit_ids_file"])
    if head_fit_ids_file not in reference_seal:
        raise IntegrityError("head-fit IDs are absent from the sealed reference-cache manifest")
    if reference_seal[head_fit_ids_file] != str(model_provenance["head_fit_ids_sha256"]):
        raise IntegrityError("head-fit ID file does not match model_provenance")
    head_fit_ids = np.load(reference_root / head_fit_ids_file, allow_pickle=False)
    reference_features = np.asarray(reference_features_all[reference_rows])
    target_features = np.asarray(target_features_all[target_rows])
    reference_labels = np.asarray(reference_labels_all[reference_rows])
    target_labels = np.asarray(target_labels_all[target_rows])
    if reference_features.ndim != 2 or target_features.ndim != 2:
        raise IntegrityError("imported feature slices must be matrices")
    if reference_features.shape[1] != target_features.shape[1]:
        raise IntegrityError("reference/target dimensions differ")
    if len(reference_features) != len(reference_ids) or len(target_features) != len(target_ids):
        raise IntegrityError("ID counts do not match imported feature slices")
    if len(reference_labels) != len(reference_features) or len(target_labels) != len(target_features):
        raise IntegrityError("label counts do not match imported features")

    arrays: dict[str, np.ndarray] = {
        "reference_features": reference_features.astype(np.float32, copy=False),
        "target_features": target_features.astype(np.float32, copy=False),
        "reference_labels": reference_labels.astype(np.int64, copy=False),
        "target_labels": target_labels.astype(np.int64, copy=False),
        "reference_ids": np.asarray(reference_ids),
        "target_ids": np.asarray(target_ids),
        "pair_reference_ids": np.asarray(pairing_reference_ids),
        "pair_target_ids": np.asarray(pairing_target_ids),
        "head_weight": np.asarray(np.load(reference_root / head_weight_file), dtype=np.float32),
        "head_bias": np.asarray(np.load(reference_root / head_bias_file), dtype=np.float32),
        "head_fit_ids": np.asarray(head_fit_ids),
    }
    signature = _shared_signature(
        dataset=dataset, encoder=encoder, reference_root=reference_root,
        target_root=target_root, reference_seal_sha256=reference_seal_sha,
        target_seal_sha256=target_seal_sha,
        reference_feature_sha256=reference_seal[reference_feature_file],
        model_provenance=model_provenance, arrays=arrays,
    )
    shared_paths = _ensure_shared_arrays(shared, signature=signature, arrays=arrays)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building.", dir=output.parent))
    try:
        paths: dict[str, str] = {}
        for role in SHARED_ROLES:
            path = staging / f"{role}.npy"
            _hardlink_atomic(shared_paths[role], path)
            paths[role] = path.name
        target_path = staging / "target_features.npy"
        _save_npy_fsynced(target_path, arrays["target_features"])
        paths["target_features"] = target_path.name
        build_native_manifest(
            staging, dataset=dataset, encoder=encoder, arrays=paths,
            id_provenance={
                "reference": {
                    "verified": True, "kind": "dataset_sample_key",
                    "source": str(pairing_provenance["source"]),
                    "namespace": str(model_provenance["reference_id_namespace"]),
                },
                "target": {
                    "verified": True, "kind": "dataset_sample_key",
                    "source": str(pairing_provenance["source"]),
                    "namespace": f"{dataset}:target",
                },
            },
            pairing={
                "status": "available",
                "reference_ids_role": "pair_reference_ids",
                "target_ids_role": "pair_target_ids",
                "provenance": dict(pairing_provenance),
            },
            head={"weight_role": "head_weight", "bias_role": "head_bias"},
            metadata=published_metadata,
        )
        _publish_directory(staging, output)
        _verified_manifest_store(output / NATIVE_MANIFEST)
    finally:
        if staging.exists():
            _remove_generated_path(staging)
    print(f"[cache] condition PASS: {condition_name}", flush=True)
    return output / NATIVE_MANIFEST


def convert_canonical_cifar_cache(
    *, reference_root: str | Path, target_root: str | Path, destination_root: str | Path,
    encoder: str, corruptions: list[str], severities: list[int],
    rows_per_condition: int = 10_000,
) -> list[Path]:
    """Convert the documented concatenated CIFAR-100-C store.

    Required target metadata must declare exact block order.  Required
    reference metadata must identify the clean test array and frozen head.
    Canonical CIFAR-100-C identity is accepted only when the metadata records
    the official generation index and the clean/corrupt label arrays are equal
    for every condition after slicing.
    """

    reference = Path(reference_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    reference_metadata = _load_json(reference / "metadata.json")
    target_metadata = _load_json(target / "metadata.json")
    blocks = target_metadata.get("blocks") or target_metadata.get("block_index")
    blocks_from_sidecar = False
    if not isinstance(blocks, list):
        block_path = target / "block_index.json"
        blocks = _load_json(block_path).get("blocks") if block_path.is_file() else None
        blocks_from_sidecar = isinstance(blocks, list)
    if not isinstance(blocks, list):
        raise IntegrityError("CIFAR-C conversion needs explicit metadata blocks/block_index.json")
    if target_metadata.get("pairing_provenance") not in {
        "official_cifar100c_canonical_test_index", "cifar100c_canonical_test_index"
    }:
        raise IntegrityError("CIFAR-C metadata does not prove official canonical test indexing")
    reference_feature_file = str(reference_metadata.get("reference_feature_file", "test_features.npy"))
    reference_label_file = str(reference_metadata.get("reference_label_file", "test_labels.npy"))
    target_feature_file = str(target_metadata.get("target_feature_file", "test_features.npy"))
    target_label_file = str(target_metadata.get("target_label_file", "test_labels.npy"))
    head_weight_file = str(reference_metadata.get("head_weight_file", "head_weight.npy"))
    head_bias_file = str(reference_metadata.get("head_bias_file", "head_bias.npy"))
    model_provenance = _validated_model_provenance(reference_metadata, encoder=encoder)
    reference_seal = _require_sha_manifest(reference)
    target_seal = _require_sha_manifest(target)
    if "metadata.json" not in reference_seal or "metadata.json" not in target_seal:
        raise IntegrityError("CIFAR-C metadata must be included in the source seals")
    if blocks_from_sidecar and "block_index.json" not in target_seal:
        raise IntegrityError("CIFAR-C block_index.json must be included in the target seal")
    reference_features = np.load(reference / reference_feature_file, mmap_mode="r")
    reference_labels = np.load(reference / reference_label_file, mmap_mode="r")
    if len(reference_features) != rows_per_condition:
        raise IntegrityError("clean CIFAR test endpoint must have exactly one canonical 10k block")
    block_lookup = {
        (str(row["corruption"]), int(row["severity"])): row for row in blocks
        if isinstance(row, Mapping) and {"corruption", "severity", "start", "stop"}.issubset(row)
    }
    outputs = []
    clean_ids = np.arange(rows_per_condition, dtype=np.int64)
    for corruption in corruptions:
        for severity in severities:
            key = (str(corruption), int(severity))
            if key not in block_lookup:
                raise IntegrityError(f"missing CIFAR-C cache block: {key}")
            row = block_lookup[key]
            start, stop = int(row["start"]), int(row["stop"])
            if stop - start != rows_per_condition:
                raise IntegrityError(f"CIFAR-C block has wrong size: {key}")
            target_labels = np.load(target / target_label_file, mmap_mode="r")[start:stop]
            if not np.array_equal(reference_labels, target_labels):
                raise IntegrityError(f"CIFAR-C canonical label assertion failed: {key}")
            output = Path(destination_root) / "cifar100_c" / encoder / corruption / f"severity_{severity}"
            manifest = import_condition(
                source_reference_root=reference, source_target_root=target,
                destination=output, dataset="cifar100_c", encoder=encoder,
                reference_feature_file=reference_feature_file,
                reference_label_file=reference_label_file,
                target_feature_file=target_feature_file, target_label_file=target_label_file,
                reference_rows=slice(0, rows_per_condition), target_rows=slice(start, stop),
                reference_ids=clean_ids, target_ids=clean_ids + 10_000_000,
                pairing_reference_ids=clean_ids, pairing_target_ids=clean_ids + 10_000_000,
                pairing_provenance={
                    "verified": True, "kind": "natural_clean_corruption_identity",
                    "source": (
                        f"official CIFAR-100-C canonical test index; block_index={key}; "
                        "clean/corrupt labels asserted equal"
                    ),
                },
                head_weight_file=head_weight_file, head_bias_file=head_bias_file,
                metadata={"corruption": corruption, "severity": int(severity)},
                model_provenance=model_provenance,
                shared_root=(Path(destination_root) / "_shared" / "cifar100_c" / encoder),
            )
            outputs.append(manifest)
    return outputs


def convert_indexed_imagenet_cache(
    *, reference_root: str | Path, target_root: str | Path, destination_root: str | Path,
    encoder: str, corruptions: list[str], severities: list[int],
    rows_per_condition: int = 50_000,
) -> list[Path]:
    """Convert ImageNet-C only from explicit hashed WNID/basename identity keys."""

    reference = Path(reference_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    reference_metadata = _load_json(reference / "metadata.json")
    target_metadata = _load_json(target / "metadata.json")
    blocks = target_metadata.get("blocks")
    blocks_from_sidecar = not isinstance(blocks, list)
    if blocks_from_sidecar:
        blocks = _load_json(target / "block_index.json").get("blocks")
    if not isinstance(blocks, list):
        raise IntegrityError("ImageNet-C conversion needs explicit block_index.json")
    reference_ids_file = reference_metadata.get("sample_ids_file")
    target_ids_file = target_metadata.get("sample_ids_file")
    if not reference_ids_file or not target_ids_file:
        raise IntegrityError("ImageNet correspondence requires sample_ids_file on both endpoints")
    if target_metadata.get("sample_id_kind") not in {"wnid_basename", "relative_image_path"}:
        raise IntegrityError("ImageNet target IDs must be WNID/basename or relative image paths")
    reference_seal = _require_sha_manifest(reference)
    target_seal = _require_sha_manifest(target)
    required_reference_sidecars = {"metadata.json", str(reference_ids_file)}
    required_target_sidecars = {"metadata.json", str(target_ids_file)}
    if blocks_from_sidecar:
        required_target_sidecars.add("block_index.json")
    missing_reference = sorted(required_reference_sidecars.difference(reference_seal))
    missing_target = sorted(required_target_sidecars.difference(target_seal))
    if missing_reference or missing_target:
        raise IntegrityError(
            "ImageNet identity/metadata sidecars are absent from source seals: "
            f"reference={missing_reference}, target={missing_target}"
        )
    reference_feature_file = str(reference_metadata.get("reference_feature_file", "test_features.npy"))
    reference_label_file = str(reference_metadata.get("reference_label_file", "test_labels.npy"))
    target_feature_file = str(target_metadata.get("target_feature_file", "test_features.npy"))
    target_label_file = str(target_metadata.get("target_label_file", "test_labels.npy"))
    head_weight_file = str(reference_metadata.get("head_weight_file", "head_weight.npy"))
    head_bias_file = str(reference_metadata.get("head_bias_file", "head_bias.npy"))
    model_provenance = _validated_model_provenance(reference_metadata, encoder=encoder)
    reference_ids = np.load(reference / str(reference_ids_file), allow_pickle=False)
    target_ids_all = np.load(target / str(target_ids_file), allow_pickle=False)
    reference_labels = np.load(reference / reference_label_file, mmap_mode="r")
    target_labels_all = np.load(target / target_label_file, mmap_mode="r")
    if len(reference_ids) != rows_per_condition or len(np.unique(reference_ids)) != len(reference_ids):
        raise IntegrityError("ImageNet clean identity index is not a unique 50k block")
    block_lookup = {(str(row["corruption"]), int(row["severity"])): row for row in blocks}
    outputs = []
    for corruption in corruptions:
        for severity in severities:
            key = (str(corruption), int(severity))
            if key not in block_lookup:
                raise IntegrityError(f"missing ImageNet-C cache block: {key}")
            row = block_lookup[key]
            start, stop = int(row["start"]), int(row["stop"])
            target_ids = np.asarray(target_ids_all[start:stop])
            target_labels = np.asarray(target_labels_all[start:stop])
            if len(target_ids) != rows_per_condition or len(np.unique(target_ids)) != len(target_ids):
                raise IntegrityError(f"ImageNet-C target identity block invalid: {key}")
            target_lookup = {value.item() if isinstance(value, np.generic) else value: index
                             for index, value in enumerate(target_ids)}
            try:
                permutation = np.asarray([target_lookup[value.item() if isinstance(value, np.generic) else value]
                                          for value in reference_ids], dtype=np.int64)
            except KeyError as error:
                raise IntegrityError(f"ImageNet-C identity set mismatch for {key}: {error.args[0]}") from error
            if not np.array_equal(reference_labels, target_labels[permutation]):
                raise IntegrityError(f"ImageNet-C ID-aligned labels disagree: {key}")
            if not np.array_equal(permutation, np.arange(rows_per_condition)):
                raise IntegrityError(
                    f"ImageNet-C feature rows are not in reference-ID order for {key}; "
                    "reindexing must happen before import"
                )
            output = Path(destination_root) / "imagenet_c" / encoder / corruption / f"severity_{severity}"
            outputs.append(import_condition(
                source_reference_root=reference, source_target_root=target,
                destination=output, dataset="imagenet_c", encoder=encoder,
                reference_feature_file=reference_feature_file,
                reference_label_file=reference_label_file,
                target_feature_file=target_feature_file, target_label_file=target_label_file,
                reference_rows=slice(0, rows_per_condition), target_rows=slice(start, stop),
                # Preserve the sealed ImageNet identity keys at both endpoints.
                # In particular, the frozen-head fit IDs are WNID/basename keys
                # from this same reference namespace.  Replacing these keys with
                # row numbers makes a valid head-fit subset look disjoint and
                # prevents its paired target counterparts from being excluded.
                reference_ids=np.asarray(reference_ids),
                target_ids=np.asarray(target_ids),
                pairing_reference_ids=np.asarray(reference_ids),
                pairing_target_ids=np.asarray(target_ids),
                pairing_provenance={
                    "verified": True, "kind": "natural_clean_corruption_identity",
                    "source": f"hashed {target_metadata['sample_id_kind']} sidecars; block_index={key}",
                },
                head_weight_file=head_weight_file, head_bias_file=head_bias_file,
                metadata={
                    "corruption": corruption,
                    "severity": int(severity),
                    # This field is also a migration marker: stores made by the
                    # earlier row-index converter cannot be reused silently.
                    "endpoint_identity_encoding": "sealed_sample_identity_keys_v1",
                },
                model_provenance=model_provenance,
                shared_root=(Path(destination_root) / "_shared" / "imagenet_c" / encoder),
            ))
    return outputs


def expected_conditions(
    *, encoders: tuple[str, ...] = CAMPAIGN_ENCODERS,
    severities: tuple[int, ...] = CAMPAIGN_SEVERITIES,
) -> list[dict[str, Any]]:
    """Return the locked 306-condition store topology."""

    result = []
    for dataset, corruptions in (
        ("cifar100_c", CIFAR100_C_CORRUPTIONS),
        ("imagenet_c", IMAGENET_C_CORRUPTIONS),
    ):
        for encoder in encoders:
            for corruption in corruptions:
                for severity in severities:
                    result.append({
                        "dataset": dataset,
                        "encoder": encoder,
                        "corruption": corruption,
                        "severity": int(severity),
                    })
    return result


def _validate_complete_condition(manifest_path: Path, expected: Mapping[str, Any]) -> None:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise IntegrityError("condition manifest is malformed")
    verify_manifest_checksum(manifest)
    if manifest.get("dataset") != expected["dataset"] or manifest.get("encoder") != expected["encoder"]:
        raise IntegrityError("dataset/encoder mismatch")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise IntegrityError("condition metadata is malformed")
    if metadata.get("corruption") != expected["corruption"]:
        raise IntegrityError("corruption metadata mismatch")
    if int(metadata.get("severity", -1)) != int(expected["severity"]):
        raise IntegrityError("severity metadata mismatch")
    materialization = metadata.get("materialization")
    if not isinstance(materialization, Mapping):
        raise IntegrityError("shared materialization metadata is absent")
    if materialization.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise IntegrityError("shared materialization schema mismatch")
    if materialization.get("shared_key") != f"{expected['dataset']}/{expected['encoder']}":
        raise IntegrityError("shared materialization key mismatch")
    if materialization.get("shared_roles") != list(SHARED_ROLES):
        raise IntegrityError("shared materialization roles mismatch")
    model = metadata.get("model_provenance")
    if not isinstance(model, Mapping) or model.get("encoder_id") != expected["encoder"]:
        raise IntegrityError("model provenance encoder mismatch")
    for key in (
        "source_reference_seal_sha256", "source_target_seal_sha256",
        "source_reference_feature_sha256", "source_target_feature_sha256",
    ):
        value = str(metadata.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise IntegrityError(f"condition lacks a valid {key}")
    records = manifest.get("arrays")
    if not isinstance(records, Mapping):
        raise IntegrityError("condition array records are malformed")
    for role in SHARED_ROLES:
        if role not in records or not isinstance(records[role], Mapping):
            raise IntegrityError(f"condition lacks shared role: {role}")
        path = verify_file_record(manifest_path.parent, records[role], verify_hash=False)
        if path.stat().st_nlink < 2:
            raise IntegrityError(f"condition role is not hard-linked: {role}")
    _verified_manifest_store(manifest_path)


def validate_store_completeness(
    root: str | Path,
    *,
    encoders: tuple[str, ...] = CAMPAIGN_ENCODERS,
    severities: tuple[int, ...] = CAMPAIGN_SEVERITIES,
    max_reported_failures: int = 20,
    progress: bool = False,
) -> dict[str, Any]:
    """Validate all expected stores, memoizing hashes of shared hard links."""

    base = Path(root).expanduser().resolve()
    expected = expected_conditions(encoders=encoders, severities=severities)
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    valid = 0
    for index, condition in enumerate(expected, start=1):
        if progress and (index == 1 or index % 10 == 0 or index == len(expected)):
            print(
                f"[cache] validating expected store {index}/{len(expected)}: "
                f"{condition['dataset']}/{condition['encoder']}/"
                f"{condition['corruption']}/severity_{condition['severity']}",
                flush=True,
            )
        relative = Path(
            condition["dataset"], condition["encoder"], condition["corruption"],
            f"severity_{condition['severity']}", NATIVE_MANIFEST,
        )
        manifest_path = base / relative
        if not manifest_path.is_file():
            if len(missing) < max_reported_failures:
                missing.append(relative.as_posix())
            continue
        try:
            _validate_complete_condition(manifest_path, condition)
        except (IntegrityError, FileNotFoundError, OSError, ValueError) as error:
            if len(invalid) < max_reported_failures:
                invalid.append({"path": relative.as_posix(), "error": str(error)})
            continue
        valid += 1
    expected_count = len(expected)
    missing_count = sum(
        not (
            base / condition["dataset"] / condition["encoder"] /
            condition["corruption"] / f"severity_{condition['severity']}" /
            NATIVE_MANIFEST
        ).is_file()
        for condition in expected
    )
    invalid_count = expected_count - valid - missing_count
    return {
        "status": "PASS" if valid == expected_count else "INCOMPLETE",
        "root": str(base),
        "expected_count": expected_count,
        "valid_count": valid,
        "missing_count": missing_count,
        "invalid_count": invalid_count,
        "expected_locked_count": EXPECTED_STORE_COUNT,
        "missing_examples": missing,
        "invalid_examples": invalid,
    }


def estimate_v2_materialization_space(
    *, legacy_root: str | Path, destination_root: str | Path,
    encoders: tuple[str, ...] = CAMPAIGN_ENCODERS,
    severities: tuple[int, ...] = CAMPAIGN_SEVERITIES,
) -> dict[str, Any]:
    """Estimate peak additional bytes for an idempotent conversion campaign.

    The estimate charges every absent target slice, one shared clean/head copy,
    and the largest atomic rebuild staging directory.  Existing generated bytes
    are credited at most once per inode.  A 2 GiB plus 5% filesystem margin is
    added rather than relying on an arbitrary fixed free-space threshold.
    """

    legacy = Path(legacy_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    net_new = 0
    max_rebuild_staging = 0
    credited_inodes: set[tuple[int, int]] = set()
    details = []
    for dataset, legacy_reference_name, legacy_target_name, corruptions, row_count in (
        ("cifar100_c", "cifar100", "cifar100c", CIFAR100_C_CORRUPTIONS, 10_000),
        ("imagenet_c", "imagenet", "imagenet_c", IMAGENET_C_CORRUPTIONS, 50_000),
    ):
        for encoder in encoders:
            reference_root = legacy / legacy_reference_name / encoder
            target_root = legacy / legacy_target_name / encoder
            reference_metadata = _load_json(reference_root / "metadata.json")
            target_metadata = _load_json(target_root / "metadata.json")
            reference_feature_file = str(
                reference_metadata.get("reference_feature_file", "test_features.npy")
            )
            target_feature_file = str(
                target_metadata.get("target_feature_file", "test_features.npy")
            )
            reference_features = np.load(
                reference_root / reference_feature_file, mmap_mode="r", allow_pickle=False,
            )
            target_features = np.load(
                target_root / target_feature_file, mmap_mode="r", allow_pickle=False,
            )
            if reference_features.ndim != 2 or target_features.ndim != 2:
                raise IntegrityError("space preflight requires matrix feature arrays")
            if reference_features.shape[1] != target_features.shape[1]:
                raise IntegrityError("space preflight found different endpoint dimensions")
            dimension = int(target_features.shape[1])
            target_npy_bytes = row_count * dimension * np.dtype(np.float32).itemsize + 512

            # Shared files: clean features; two label arrays; four ID/pairing
            # arrays; head matrices; and head-fit IDs.  Use on-disk source sizes
            # for heads and fit IDs and float32 for converted features/heads.
            reference_label_file = str(
                reference_metadata.get("reference_label_file", "test_labels.npy")
            )
            head_weight_file = str(reference_metadata.get("head_weight_file", "head_weight.npy"))
            head_bias_file = str(reference_metadata.get("head_bias_file", "head_bias.npy"))
            model = _validated_model_provenance(reference_metadata, encoder=encoder)
            fit_ids_file = str(model["head_fit_ids_file"])
            head_weight = np.load(reference_root / head_weight_file, mmap_mode="r", allow_pickle=False)
            head_bias = np.load(reference_root / head_bias_file, mmap_mode="r", allow_pickle=False)
            head_fit = np.load(reference_root / fit_ids_file, mmap_mode="r", allow_pickle=False)
            _ = np.load(reference_root / reference_label_file, mmap_mode="r", allow_pickle=False)
            shared_expected = (
                row_count * dimension * 4
                + 6 * row_count * 8
                + int(head_weight.size + head_bias.size) * 4
                + int(head_fit.nbytes)
                + 16 * 1024
            )
            shared_dir = destination / "_shared" / dataset / encoder
            shared_existing = 0
            if shared_dir.is_dir():
                for path in shared_dir.glob("*.npy"):
                    if not path.is_file():
                        continue
                    stat = path.stat()
                    inode = (int(stat.st_dev), int(stat.st_ino))
                    if inode not in credited_inodes:
                        credited_inodes.add(inode)
                        shared_existing += int(stat.st_size)
            net_new += max(0, shared_expected - shared_existing)
            if shared_existing:
                max_rebuild_staging = max(max_rebuild_staging, shared_expected)

            missing_bytes = 0
            existing_conditions = 0
            for corruption in corruptions:
                for severity in severities:
                    condition_root = (
                        destination / dataset / encoder / corruption / f"severity_{severity}"
                    )
                    target_path = condition_root / "target_features.npy"
                    existing_credit = 0
                    if target_path.is_file():
                        existing_conditions += 1
                        stat = target_path.stat()
                        inode = (int(stat.st_dev), int(stat.st_ino))
                        if inode not in credited_inodes:
                            credited_inodes.add(inode)
                            existing_credit = int(stat.st_size)
                        max_rebuild_staging = max(max_rebuild_staging, target_npy_bytes)
                    delta = max(0, target_npy_bytes - existing_credit)
                    missing_bytes += delta
                    net_new += delta
            details.append({
                "dataset": dataset,
                "encoder": encoder,
                "condition_count": len(corruptions) * len(severities),
                "existing_condition_count": existing_conditions,
                "estimated_condition_net_new_bytes": missing_bytes,
                "estimated_shared_bytes": shared_expected,
            })
    safety_margin = max(2 * 1024**3, int(round(0.05 * (net_new + max_rebuild_staging))))
    required = int(net_new + max_rebuild_staging + safety_margin)
    free = int(shutil.disk_usage(destination).free)
    return {
        "status": "PASS" if free >= required else "INSUFFICIENT_SPACE",
        "destination_root": str(destination),
        "free_bytes": free,
        "estimated_net_new_bytes": int(net_new),
        "largest_atomic_rebuild_bytes": int(max_rebuild_staging),
        "safety_margin_bytes": int(safety_margin),
        "required_free_bytes": required,
        "free_gib": round(free / 1024**3, 3),
        "required_free_gib": round(required / 1024**3, 3),
        "details": details,
    }
