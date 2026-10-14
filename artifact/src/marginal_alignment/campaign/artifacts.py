"""Content-addressed artifacts and deterministic JSON for the evaluation campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# A campaign reopens hard-linked clean/head arrays hundreds of times.  Cache a
# digest only while the exact inode, size, and nanosecond mtime are unchanged;
# every first encounter is still a full cryptographic read.
_VERIFIED_FILE_DIGESTS: dict[tuple[int, int, int, int], str] = {}


class IntegrityError(RuntimeError):
    """Raised when persisted evidence does not match its declared provenance."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 representation used for locks and manifests."""

    return (json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str | Path) -> Path:
    """Validate a manifest path without resolving through the host filesystem."""

    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise IntegrityError(f"artifact path must be a normalized relative path: {value!r}")
    return path


def resolve_artifact(root: str | Path, relative: str | Path) -> Path:
    base = Path(root).expanduser().resolve()
    path = safe_relative_path(relative)
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise IntegrityError(f"artifact escapes store root: {relative!r}") from error
    return candidate


def describe_file(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    recorded_path = source.name if relative_to is None else str(source.relative_to(Path(relative_to).resolve()))
    return {
        "path": str(safe_relative_path(recorded_path)),
        "sha256": sha256_file(source),
        "size_bytes": int(source.stat().st_size),
    }


def describe_npy(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    record = describe_file(source, relative_to=relative_to)
    try:
        array = np.load(source, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise IntegrityError(f"not a safe NumPy array: {source}") from error
    record.update({
        "format": "npy",
        "shape": [int(value) for value in array.shape],
        "dtype": np.dtype(array.dtype).str,
    })
    return record


def verify_file_record(
    root: str | Path,
    record: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> Path:
    required = {"path", "sha256", "size_bytes"}
    missing = sorted(required.difference(record))
    if missing:
        raise IntegrityError(f"artifact record is missing fields: {missing}")
    path = resolve_artifact(root, str(record["path"]))
    if not path.is_file():
        raise IntegrityError(f"artifact is missing: {path}")
    size = int(path.stat().st_size)
    if size != int(record["size_bytes"]):
        raise IntegrityError(f"artifact size mismatch for {path}: {size} != {record['size_bytes']}")
    expected = str(record["sha256"])
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise IntegrityError(f"invalid SHA-256 in artifact record for {path}")
    if verify_hash:
        stat = path.stat()
        cache_key = (
            int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns)
        )
        actual = _VERIFIED_FILE_DIGESTS.get(cache_key)
        if actual is None:
            actual = sha256_file(path)
            _VERIFIED_FILE_DIGESTS[cache_key] = actual
        if actual != expected:
            raise IntegrityError(f"artifact SHA-256 mismatch for {path}: {actual} != {expected}")
    if record.get("format") == "npy":
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise IntegrityError(f"declared NPY artifact cannot be loaded safely: {path}") from error
        expected_shape = tuple(int(value) for value in record.get("shape", ()))
        if tuple(array.shape) != expected_shape:
            raise IntegrityError(
                f"array shape mismatch for {path}: {tuple(array.shape)} != {expected_shape}"
            )
        expected_dtype = np.dtype(str(record.get("dtype")))
        if np.dtype(array.dtype) != expected_dtype:
            raise IntegrityError(
                f"array dtype mismatch for {path}: {array.dtype} != {expected_dtype}"
            )
    return path


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Atomically persist canonical JSON and fsync it before publication."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    payload = canonical_json_bytes(value)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"invalid JSON artifact: {path}") from error


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add a checksum over every manifest field except the checksum itself."""

    if "manifest_sha256" in payload:
        raise ValueError("payload must not already contain manifest_sha256")
    result = dict(payload)
    result["manifest_sha256"] = sha256_json(result)
    return result


def verify_manifest_checksum(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        raise IntegrityError("manifest_sha256 is absent")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = sha256_json(unsigned)
    if actual != expected:
        raise IntegrityError(f"manifest checksum mismatch: {actual} != {expected}")
