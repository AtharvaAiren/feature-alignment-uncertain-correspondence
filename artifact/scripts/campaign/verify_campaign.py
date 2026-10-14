#!/usr/bin/env python3
"""Re-hash a fetched evaluation campaign and fail on any changed artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    resolve_artifact,
    safe_relative_path,
    sha256_file,
)


def _exact_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise IntegrityError(f"{name} is not an exact lowercase SHA-256")
    return value


def verify_campaign_evidence(run_dir: str | Path) -> dict[str, Any]:
    """Verify the final manifest, completion marker, and every fetched file."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "CAMPAIGN_MANIFEST.json"
    done_path = root / "DONE"
    if not manifest_path.is_file():
        raise IntegrityError(f"campaign manifest is missing: {manifest_path}")
    if not done_path.is_file():
        raise IntegrityError(f"completion marker is missing: {done_path}")
    manifest = read_json(manifest_path)
    done = read_json(done_path)
    if not isinstance(manifest, Mapping):
        raise IntegrityError("CAMPAIGN_MANIFEST.json must be a JSON object")
    if not isinstance(done, Mapping):
        raise IntegrityError("DONE must be a JSON object")
    if manifest.get("schema_version") != "marginal_alignment.campaign.campaign_manifest.v2":
        raise IntegrityError("unsupported campaign-manifest schema")
    if manifest.get("status") != "PASS":
        raise IntegrityError(f"campaign manifest is not PASS: {manifest.get('status')!r}")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise IntegrityError("campaign manifest has no campaign_id")
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping) or not stages or any(
        status != "PASS" for status in stages.values()
    ):
        raise IntegrityError(f"campaign manifest contains non-passing stages: {stages!r}")
    if done.get("schema_version") != "marginal_alignment.campaign.done.v2":
        raise IntegrityError("unsupported DONE schema")
    if done.get("status") != "PASS":
        raise IntegrityError(f"DONE is not PASS: {done.get('status')!r}")
    if done.get("campaign_id") != campaign_id:
        raise IntegrityError("DONE campaign_id does not match CAMPAIGN_MANIFEST.json")
    declared_manifest_hash = _exact_sha256(
        done.get("campaign_manifest_sha256"), name="DONE campaign_manifest_sha256"
    )
    actual_manifest_hash = sha256_file(manifest_path)
    if declared_manifest_hash != actual_manifest_hash:
        raise IntegrityError(
            "DONE does not bind the fetched CAMPAIGN_MANIFEST.json: "
            f"{declared_manifest_hash} != {actual_manifest_hash}"
        )

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise IntegrityError("campaign manifest has no file inventory")
    failures: list[str] = []
    seen: set[str] = set()
    forbidden = {"CAMPAIGN_MANIFEST.json", "DONE"}
    for record in records:
        if not isinstance(record, Mapping):
            raise IntegrityError("campaign file inventory contains a malformed record")
        label = record.get("path")
        if not isinstance(label, str):
            raise IntegrityError("campaign file record has no path")
        normalized = safe_relative_path(label).as_posix()
        if normalized != label:
            raise IntegrityError(f"campaign file path is not normalized: {label!r}")
        if label in forbidden:
            raise IntegrityError(f"campaign manifest may not self-inventory {label}")
        if label in seen:
            raise IntegrityError(f"duplicate campaign file record: {label}")
        seen.add(label)
        expected_hash = _exact_sha256(record.get("sha256"), name=f"{label} sha256")
        try:
            expected_bytes = int(record["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(f"invalid byte count for {label}") from error
        if expected_bytes < 0:
            raise IntegrityError(f"negative byte count for {label}")
        path = resolve_artifact(root, label)
        if not path.is_file():
            failures.append(f"MISSING {label}")
        elif path.stat().st_size != expected_bytes:
            failures.append(f"SIZE {label}")
        elif sha256_file(path) != expected_hash:
            failures.append(f"SHA256 {label}")
    if failures:
        raise IntegrityError("campaign file verification failed:\n" + "\n".join(failures[:20]))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    expected_files = seen | forbidden
    extra = sorted(actual_files - expected_files)
    absent = sorted(expected_files - actual_files)
    if extra or absent:
        details = [*(f"EXTRA {value}" for value in extra[:20]),
                   *(f"MISSING {value}" for value in absent[:20])]
        raise IntegrityError(
            "campaign file set differs from its exact inventory:\n" + "\n".join(details)
        )
    return {
        "status": "PASS",
        "files": len(records),
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": actual_manifest_hash,
        "done_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    arguments = parser.parse_args()
    try:
        result = verify_campaign_evidence(arguments.run_dir)
    except (IntegrityError, FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"campaign verification failed: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
