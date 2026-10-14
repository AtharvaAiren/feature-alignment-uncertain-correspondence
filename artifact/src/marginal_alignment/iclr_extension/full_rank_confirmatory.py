"""Additive untruncated paired-ridge follow-up for the confirmatory panel.

This module deliberately does not reopen or mutate the completed campaign.  It
authenticates the frozen confirmatory split identities, computes two additional
methods in a fresh result root, and analyzes the four reviewer-requested
dataset-by-comparator tests as a separate post-hoc Holm family.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    canonical_json_bytes,
    describe_file,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import load_feature_store
from marginal_alignment.campaign.inference import (
    holm_adjust,
    infer_contrast,
    macro_summary,
    paired_effect_rows,
)
from marginal_alignment.campaign.real_runner import (
    expected_unit_semantic_keys,
    run_condition,
    semantic_row_key,
    split_plan_path,
    validate_semantic_grid,
)


LOCK_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_lock.v1"
BINDING_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_binding.v1"
UNIT_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_unit.v1"
INFERENCE_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_inference.v1"
MANIFEST_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_manifest.v1"
DONE_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_done.v1"
STATE_SCHEMA = "marginal_alignment.full_rank_paired_confirmatory_state.v1"

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
IDENTITY_FIELDS = (
    "split_plan_sha256",
    "evaluation_sha256",
    "feature_store_manifest_sha256",
)
ORIGINAL_METHODS = (
    "bures_pair_endpoints",
    "paired_reduced_rank",
    "paired_reduced_rank_shuffled",
)
NEW_METHODS = ("paired_ridge", "paired_ridge_shuffled")
ANCHOR_METHOD = "bures_pair_endpoints"
METRIC = "reference_prediction_agreement"
DESCRIPTIVE_METHOD = "paired_reduced_rank"

ORIGINAL_CAMPAIGN_ID = "marginal_alignment_iclr_2026_v2"
ORIGINAL_DONE_EXPECTATION = {
    "path": "DONE",
    "size_bytes": 211,
    "sha256": "7f11f60b01f90d3faea4747aa1b724f8f86a143f9e515587522280320485de93",
    "schema_version": "marginal_alignment.iclr.done.v2",
}
ORIGINAL_MANIFEST_EXPECTATION = {
    "path": "CAMPAIGN_MANIFEST.json",
    "size_bytes": 646_049,
    "sha256": "788b1efeedd1c3a067bb8106d5e2d18868b9bc5e975e0f40fa45b372dc5928f4",
    "schema_version": "marginal_alignment.iclr.campaign_manifest.v2",
    "file_count": 3_572,
    "source_sha256": "078cf0dc2dbb183d897d7b42b3eb8d9c08aa9a7253b521b4243625834c1ec99c",
}
ORIGINAL_INPUT_EXPECTATIONS = {
    "resolved_config.json": {
        "bytes": 9_174,
        "sha256": "2b3858e96213b2b6c5a092394c1ddcdda5b681b2903184b6a52517e8ccd62af4",
    },
    "feature_store_index.json": {
        "bytes": 1_679_777,
        "sha256": "13dd2e18fa1ed3975059ca88c76357faa032df2dfc7587117ba9a69ac1ccfdad",
    },
    "real/rows.jsonl": {
        "bytes": 108_133_226,
        "sha256": "12fab3212723005b28b6e0abf4c93406739dc9c3f7461fe57f819ec3118cc062",
    },
    "source_manifest.json": {
        "bytes": 11_402,
        "sha256": "2e3d30a3bedd00db967ce292ea30fd251d08a59d88a7dc4ef51a8387ae986075",
    },
}

SOURCE_TRUST_DIRECTORY = Path("provenance/full_rank")
BASE_DEPENDENCY_LOCK = SOURCE_TRUST_DIRECTORY / "BASE_DEPENDENCY_SHA256SUMS"
PAYLOAD_LOCK = SOURCE_TRUST_DIRECTORY / "PAYLOAD_SHA256SUMS"
BASE_DEPENDENCY_PATHS = (
    "src/marginal_alignment/__init__.py",
    "src/marginal_alignment/ceiling.py",
    "src/marginal_alignment/linalg.py",
    "src/marginal_alignment/metrics.py",
    "src/marginal_alignment/transforms.py",
    "src/marginal_alignment/campaign/__init__.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/cache_import.py",
    "src/marginal_alignment/campaign/feature_protocol.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/inference.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/campaign/method_registry.py",
    "src/marginal_alignment/campaign/metrics.py",
    "src/marginal_alignment/campaign/real_runner.py",
    "src/marginal_alignment/campaign/raw_extract.py",
    "src/marginal_alignment/campaign/split_plan.py",
    "src/marginal_alignment/iclr_extension/__init__.py",
    "src/marginal_alignment/iclr_extension/ceiling.py",
    "src/marginal_alignment/iclr_extension/frontier.py",
    "src/marginal_alignment/iclr_extension/natural_shift.py",
)
FULL_RANK_PAYLOAD_PATHS = (
    "configs/iclr_extension/full_rank_paired_confirmatory_lock.json",
    "scripts/iclr_extension/run_full_rank_paired_confirmatory.py",
    "src/marginal_alignment/iclr_extension/full_rank_confirmatory.py",
    BASE_DEPENDENCY_LOCK.as_posix(),
)


@dataclass(frozen=True)
class ValidatedInputs:
    """Immutable projection of the authenticated completed campaign inputs."""

    project_root: Path
    original_root: Path
    done_path: Path
    campaign_manifest_path: Path
    lock_path: Path
    config_path: Path
    index_path: Path
    rows_path: Path
    lock: dict[str, Any]
    config: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    anchors: dict[tuple[str, str, str, int], dict[str, Any]]
    descriptive_paired_reduced_rank: dict[tuple[str, str, str, int], dict[str, Any]]
    binding_payload: dict[str, Any]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, name: str) -> str:
    if not _is_sha256(value):
        raise IntegrityError(f"{name} is not a lowercase SHA-256")
    return str(value)


def _require_regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{name} must be a regular nonsymbolic file: {path}")
    return path.resolve()


def _require_directory(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError(f"{name} must be a real directory: {path}")
    return path.resolve()


def _require_exact_file(
    path: Path,
    *,
    name: str,
    size_bytes: int,
    sha256: str,
) -> Path:
    path = _require_regular_file(path, name=name)
    observed_size = int(path.stat().st_size)
    if observed_size != int(size_bytes):
        raise IntegrityError(
            f"{name} byte count changed: {observed_size} != {int(size_bytes)}"
        )
    observed_sha256 = sha256_file(path)
    if observed_sha256 != sha256:
        raise IntegrityError(f"{name} SHA-256 changed: {observed_sha256} != {sha256}")
    return path


def _require_exact_canonical_child(
    root: Path,
    relative: str,
    *,
    name: str,
    size_bytes: int,
    sha256: str,
) -> Path:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or relative_path.as_posix() != relative
    ):
        raise IntegrityError(f"noncanonical input path: {relative!r}")
    raw = root / relative_path
    cursor = raw
    while cursor != root:
        if cursor.is_symlink():
            raise IntegrityError(f"{name} path component may not be symbolic: {cursor}")
        cursor = cursor.parent
    resolved = _require_exact_file(
        raw,
        name=name,
        size_bytes=size_bytes,
        sha256=sha256,
    )
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise IntegrityError(f"{name} escapes original campaign root") from error
    return resolved


def _parse_sha256sum_lock(path: Path) -> dict[str, str]:
    """Parse the deliberately small GNU sha256sum lock format fail closed."""

    path = _require_regular_file(path, name="source checksum lock")
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line or len(line) < 67 or line[64:66] != "  ":
            raise IntegrityError(f"malformed checksum lock line {path}:{line_number}")
        digest, relative = line[:64], line[66:]
        _require_sha256(digest, name=f"{path.name}:{line_number}")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise IntegrityError(
                f"unsafe checksum path at {path}:{line_number}: {relative!r}"
            )
        normalized = relative_path.as_posix()
        if normalized != relative or normalized in records:
            raise IntegrityError(
                f"duplicate or noncanonical checksum path at {path}:{line_number}: {relative!r}"
            )
        records[normalized] = digest
    if not records:
        raise IntegrityError(f"empty source checksum lock: {path}")
    return records


def _verify_sha256sum_records(
    project_root: Path,
    records: Mapping[str, str],
    *,
    label: str,
) -> None:
    for relative, expected in records.items():
        path = _require_regular_file(project_root / relative, name=f"{label} source")
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise IntegrityError(
                f"{label} source escapes project root: {relative}"
            ) from error
        observed = sha256_file(path)
        if observed != expected:
            raise IntegrityError(
                f"{label} source SHA-256 changed: {relative}: {observed} != {expected}"
            )


def validate_source_root(project_root: Path) -> dict[str, Any]:
    """Authenticate base dependencies and the externally sealed overlay payload."""

    project_root = _require_directory(project_root, name="project root")
    base_lock_path = project_root / BASE_DEPENDENCY_LOCK
    payload_lock_path = project_root / PAYLOAD_LOCK
    for lock_path in (base_lock_path, payload_lock_path):
        resolved_lock = _require_regular_file(lock_path, name="source checksum lock")
        try:
            resolved_lock.relative_to(project_root)
        except ValueError as error:
            raise IntegrityError(
                f"source checksum lock escapes project root: {lock_path}"
            ) from error
    base_records = _parse_sha256sum_lock(base_lock_path)
    payload_records = _parse_sha256sum_lock(payload_lock_path)
    if set(base_records) != set(BASE_DEPENDENCY_PATHS):
        raise IntegrityError(
            "base dependency checksum lock inventory drifted: "
            f"observed={sorted(base_records)} expected={sorted(BASE_DEPENDENCY_PATHS)}"
        )
    missing_payload = sorted(set(FULL_RANK_PAYLOAD_PATHS) - set(payload_records))
    if missing_payload:
        raise IntegrityError(
            f"handoff payload lock lacks paired-ridge source records: {missing_payload}"
        )
    _verify_sha256sum_records(project_root, base_records, label="base dependency")
    _verify_sha256sum_records(project_root, payload_records, label="handoff payload")
    return {
        "base_dependency_lock": describe_file(base_lock_path, relative_to=project_root),
        "payload_lock": describe_file(payload_lock_path, relative_to=project_root),
        "base_dependency_records": len(base_records),
        "payload_records": len(payload_records),
        "trust_chain": (
            "external deterministic handoff archive SHA-256 -> PAYLOAD_SHA256SUMS -> "
            "BASE_DEPENDENCY_SHA256SUMS and paired-ridge overlay -> imported sources"
        ),
    }


def require_single_thread_environment(
    environment: Mapping[str, str] | None = None
) -> None:
    """Reject ambiguous or oversubscribed numerical-library thread state."""

    values = os.environ if environment is None else environment
    observed = {name: values.get(name) for name in THREAD_VARIABLES}
    if any(value != "1" for value in observed.values()):
        raise RuntimeError(
            "untruncated paired-ridge follow-up requires every numerical thread environment "
            f"variable to equal '1': observed={observed}"
        )
    if values.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE must equal '1'")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise IntegrityError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise IntegrityError(
                    f"JSONL record is not an object at {path}:{line_number}"
                )
            yield value


def _validate_original_campaign_chain(original_root: Path) -> dict[str, Any]:
    """Authenticate the exact untransformed execution artifacts and their chain."""

    original_root = _require_directory(original_root, name="original result root")
    done_path = _require_exact_canonical_child(
        original_root,
        str(ORIGINAL_DONE_EXPECTATION["path"]),
        name="original campaign DONE",
        size_bytes=int(ORIGINAL_DONE_EXPECTATION["size_bytes"]),
        sha256=str(ORIGINAL_DONE_EXPECTATION["sha256"]),
    )
    done = read_json(done_path)
    expected_done = {
        "campaign_id": ORIGINAL_CAMPAIGN_ID,
        "campaign_manifest_sha256": ORIGINAL_MANIFEST_EXPECTATION["sha256"],
        "schema_version": ORIGINAL_DONE_EXPECTATION["schema_version"],
        "status": "PASS",
    }
    if done != expected_done:
        raise IntegrityError("original campaign DONE semantic payload drifted")

    campaign_manifest_path = _require_exact_canonical_child(
        original_root,
        str(ORIGINAL_MANIFEST_EXPECTATION["path"]),
        name="original CAMPAIGN_MANIFEST.json",
        size_bytes=int(ORIGINAL_MANIFEST_EXPECTATION["size_bytes"]),
        sha256=str(ORIGINAL_MANIFEST_EXPECTATION["sha256"]),
    )
    campaign_manifest = read_json(campaign_manifest_path)
    if not isinstance(campaign_manifest, dict):
        raise IntegrityError("original campaign manifest is not an object")
    if (
        campaign_manifest.get("schema_version")
        != ORIGINAL_MANIFEST_EXPECTATION["schema_version"]
        or campaign_manifest.get("campaign_id") != ORIGINAL_CAMPAIGN_ID
        or campaign_manifest.get("status") != "PASS"
        or campaign_manifest.get("source_sha256")
        != ORIGINAL_MANIFEST_EXPECTATION["source_sha256"]
    ):
        raise IntegrityError("original campaign manifest identity/status drifted")
    files = campaign_manifest.get("files")
    if not isinstance(files, list) or len(files) != int(
        ORIGINAL_MANIFEST_EXPECTATION["file_count"]
    ):
        raise IntegrityError("original campaign manifest file inventory count drifted")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in files:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise IntegrityError(
                "original campaign manifest has a malformed file record"
            )
        relative = str(record["path"])
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative_path.as_posix() != relative
        ):
            raise IntegrityError(
                f"original campaign manifest has a noncanonical path: {relative!r}"
            )
        if relative in by_path:
            raise IntegrityError(f"original campaign manifest repeats path: {relative}")
        by_path[relative] = record

    paths: dict[str, Path] = {}
    for relative, expected in ORIGINAL_INPUT_EXPECTATIONS.items():
        expected_record = {
            "path": relative,
            "bytes": int(expected["bytes"]),
            "sha256": str(expected["sha256"]),
        }
        if by_path.get(relative) != expected_record:
            raise IntegrityError(
                f"original campaign manifest record drifted for canonical input: {relative}"
            )
        paths[relative] = _require_exact_canonical_child(
            original_root,
            relative,
            name=f"canonical original {relative}",
            size_bytes=int(expected["bytes"]),
            sha256=str(expected["sha256"]),
        )
    source_manifest = read_json(paths["source_manifest.json"])
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("source_sha256")
        != ORIGINAL_MANIFEST_EXPECTATION["source_sha256"]
        or not isinstance(source_manifest.get("files"), list)
        or len(source_manifest["files"]) != 83
    ):
        raise IntegrityError("original source manifest identity/count drifted")
    return {
        "done_path": done_path,
        "campaign_manifest_path": campaign_manifest_path,
        "done": done,
        "campaign_manifest": campaign_manifest,
        "canonical_paths": paths,
    }


def _dataset_protocol(lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    protocol = lock.get("original_protocol")
    if not isinstance(protocol, Mapping):
        raise IntegrityError("extension lock lacks original_protocol")
    datasets = protocol.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {
        "cifar100_c",
        "imagenet_c",
    }:
        raise IntegrityError(
            "extension lock must contain exactly CIFAR-100-C and ImageNet-C"
        )
    return {str(key): value for key, value in datasets.items()}


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("status") != "LOCKED":
        raise IntegrityError(f"not a locked {LOCK_SCHEMA} protocol")
    verify_manifest_checksum(lock)
    if lock.get("scientific_result") is not False:
        raise IntegrityError("protocol lock must be explicitly pre-outcome")
    if lock.get("analysis_status") != "post_hoc_reviewer_requested":
        raise IntegrityError("follow-up must remain labeled post hoc")
    protocol = lock.get("original_protocol")
    if not isinstance(protocol, Mapping):
        raise IntegrityError("lock original_protocol is malformed")
    exact = {
        "suite_name": "correspondence_confirmatory",
        "global_seed": 20260812,
        "calibration_sizes": [4096],
        "pair_budgets": [4096],
        "pair_noise_rates": [0.0],
        "split_seeds": [35, 36, 37],
        "condition_filter": {"severities": [5]},
        "original_methods": list(ORIGINAL_METHODS),
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise IntegrityError(f"locked original_protocol.{key} drifted")
    datasets = _dataset_protocol(lock)
    if len(datasets["cifar100_c"].get("corruptions", [])) != 19:
        raise IntegrityError("CIFAR-100-C lock must contain 19 corruptions")
    if len(datasets["imagenet_c"].get("corruptions", [])) != 15:
        raise IntegrityError("ImageNet-C lock must contain 15 corruptions")
    for name, expected_evaluation in (("cifar100_c", 3000), ("imagenet_c", 10000)):
        specification = datasets[name]
        if specification.get("encoders") != ["dinov3", "clip", "vitb16"]:
            raise IntegrityError(f"{name} encoder panel drifted")
        if specification.get("evaluation_size") != expected_evaluation:
            raise IntegrityError(f"{name} evaluation size drifted")
        corruptions = specification.get("corruptions")
        if not isinstance(corruptions, list) or len(set(corruptions)) != len(
            corruptions
        ):
            raise IntegrityError(f"{name} corruption panel is malformed")
    methods = lock.get("methods")
    expected_methods = {
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
    if methods != expected_methods:
        raise IntegrityError("locked untruncated paired-ridge specifications drifted")
    inference = lock.get("inference")
    if not isinstance(inference, Mapping):
        raise IntegrityError("inference lock is malformed")
    if (
        inference.get("metric") != METRIC
        or inference.get("cluster_unit") != "corruption"
        or inference.get("bootstrap_repetitions") != 20_000
        or inference.get("confidence") != 0.95
        or inference.get("multiplicity") != "holm"
        or inference.get("bootstrap_assumption")
        != "corruption_cluster_exchangeability_within_each_fixed_dataset_panel"
        or inference.get("sign_flip_assumption")
        != "corruption_cluster_effect_sign_symmetry_under_the_zero_effect_null"
        or inference.get("sign_flip_calibration")
        != "exhaustive_all_cluster_sign_assignments"
        or len(inference.get("tests", [])) != 4
    ):
        raise IntegrityError("four-test post-hoc inference family drifted")
    if (
        lock.get("expected_condition_units") != 102
        or lock.get("expected_new_rows") != 612
    ):
        raise IntegrityError("locked grid counts drifted")


def _validate_original_config(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    protocol = lock["original_protocol"]
    if config.get("global_seed") != protocol["global_seed"]:
        raise IntegrityError(
            "completed campaign global seed does not match follow-up lock"
        )
    suites = config.get("suites")
    if not isinstance(suites, Mapping):
        raise IntegrityError("completed config lacks suites")
    suite = suites.get(protocol["suite_name"])
    if not isinstance(suite, Mapping):
        raise IntegrityError("completed config lacks correspondence_confirmatory")
    expected_suite = {
        "datasets": list(protocol["datasets"]),
        "methods": protocol["original_methods"],
        "calibration_sizes": protocol["calibration_sizes"],
        "pair_budgets": protocol["pair_budgets"],
        "pair_noise_rates": protocol["pair_noise_rates"],
        "split_seeds": protocol["split_seeds"],
        "condition_filter": protocol["condition_filter"],
    }
    if dict(suite) != expected_suite:
        raise IntegrityError("completed correspondence_confirmatory suite drifted")
    methods = config.get("methods")
    if not isinstance(methods, Mapping):
        raise IntegrityError("completed config lacks method registry")
    for method in NEW_METHODS:
        if methods.get(method) != lock["methods"][method]:
            raise IntegrityError(
                f"completed config method {method} differs from follow-up lock"
            )
    expected_bures = {
        "implementation": "bures",
        "information": "joint_second_moments_on_paired_endpoint_sets_with_assignment_erased",
        "covariance": "oas",
        "fit_on_pair_endpoint_sets": True,
    }
    if methods.get(ANCHOR_METHOD) != expected_bures:
        raise IntegrityError("completed Bures endpoint control specification drifted")
    configured_datasets = config.get("datasets")
    if not isinstance(configured_datasets, list):
        raise IntegrityError("completed config datasets are malformed")
    by_name = {
        row.get("name"): row for row in configured_datasets if isinstance(row, Mapping)
    }
    for dataset, expected in _dataset_protocol(lock).items():
        observed = by_name.get(dataset)
        if not isinstance(observed, Mapping):
            raise IntegrityError(f"completed config lacks dataset {dataset}")
        for key in ("corruptions", "encoders", "evaluation_size"):
            if observed.get(key) != expected[key]:
                raise IntegrityError(f"completed config {dataset}.{key} drifted")


def _expected_condition_keys(lock: Mapping[str, Any]) -> set[tuple[str, str, str, int]]:
    expected: set[tuple[str, str, str, int]] = set()
    for dataset, specification in _dataset_protocol(lock).items():
        for encoder in specification["encoders"]:
            for corruption in specification["corruptions"]:
                expected.add((dataset, str(encoder), str(corruption), 5))
    if len(expected) != int(lock["expected_condition_units"]):
        raise IntegrityError("condition product does not equal locked unit count")
    return expected


def _validate_index(
    index: Mapping[str, Any], lock: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    records = index.get("records")
    if index.get("status") != "PASS" or not isinstance(records, list):
        raise IntegrityError("feature-store index is not a PASS record list")
    expected = _expected_condition_keys(lock)
    selected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    all_keys: set[tuple[str, str, str, int]] = set()
    for row in records:
        if not isinstance(row, dict):
            raise IntegrityError("feature-store index contains a non-object record")
        key = (
            str(row.get("dataset")),
            str(row.get("encoder")),
            str(row.get("corruption")),
            int(row.get("severity", -1)),
        )
        if key in all_keys:
            raise IntegrityError(f"duplicate feature-store condition: {key}")
        all_keys.add(key)
        if key in expected:
            _require_sha256(row.get("manifest_sha256"), name=f"{key} manifest hash")
            if not isinstance(row.get("manifest"), str) or not row["manifest"]:
                raise IntegrityError(f"{key} has no feature-store manifest path")
            selected[key] = row
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise IntegrityError(
            f"feature-store confirmatory grid mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    return tuple(selected[key] for key in sorted(selected))


def _row_condition_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("dataset")),
        str(row.get("encoder")),
        str(row.get("corruption")),
        int(row.get("severity", -1)),
    )


def _anchor_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    condition = _row_condition_key(row)
    return (*condition[:3], int(row.get("split_seed", -1)))


def _validate_original_rows(
    rows_path: Path,
    records: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, str, int], dict[str, Any]],
    dict[tuple[str, str, str, int], dict[str, Any]],
]:
    expected_conditions = _expected_condition_keys(lock)
    expected_seeds = set(lock["original_protocol"]["split_seeds"])
    indexed_hash = {
        _row_condition_key(row): str(row["manifest_sha256"]) for row in records
    }
    rows_by_method: dict[str, dict[tuple[str, str, str, int], dict[str, Any]]] = {
        method: {} for method in ORIGINAL_METHODS
    }
    method_counts = {method: 0 for method in ORIGINAL_METHODS}
    unexpected_methods: set[str] = set()
    for row in _read_jsonl(rows_path):
        if row.get("suite") != lock["original_protocol"]["suite_name"]:
            continue
        method = str(row.get("method"))
        if method not in rows_by_method:
            unexpected_methods.add(method)
            continue
        method_counts[method] += 1
        condition = _row_condition_key(row)
        seed = int(row.get("split_seed", -1))
        key = _anchor_key(row)
        if condition not in expected_conditions or seed not in expected_seeds:
            raise IntegrityError(
                f"completed confirmatory row lies outside locked grid: {key}"
            )
        if row.get("calibration_size") != 4096 or row.get("pair_budget") != 4096:
            raise IntegrityError(f"completed confirmatory dose drifted: {key}/{method}")
        if (
            row.get("evaluation_size")
            != _dataset_protocol(lock)[condition[0]]["evaluation_size"]
        ):
            raise IntegrityError(
                f"completed confirmatory evaluation size drifted: {key}"
            )
        for field in IDENTITY_FIELDS:
            _require_sha256(row.get(field), name=f"{key}/{method}/{field}")
        if row["feature_store_manifest_sha256"] != indexed_hash[condition]:
            raise IntegrityError(
                f"row/index feature-store identity mismatch: {key}/{method}"
            )
        if key in rows_by_method[method]:
            raise IntegrityError(
                f"duplicate completed confirmatory row: {key}/{method}"
            )
        rows_by_method[method][key] = row
    expected_per_method = len(expected_conditions) * len(expected_seeds)
    if unexpected_methods:
        raise IntegrityError(
            f"completed confirmatory suite has unexpected methods: {sorted(unexpected_methods)}"
        )
    for method, values in rows_by_method.items():
        if (
            method_counts[method] != expected_per_method
            or len(values) != expected_per_method
        ):
            raise IntegrityError(
                f"completed {method} grid has {len(values)} rows; expected {expected_per_method}"
            )
    for key in rows_by_method[ANCHOR_METHOD]:
        expected_identity = {
            field: rows_by_method[ANCHOR_METHOD][key][field]
            for field in IDENTITY_FIELDS
        }
        for method in ORIGINAL_METHODS[1:]:
            observed = {
                field: rows_by_method[method][key][field] for field in IDENTITY_FIELDS
            }
            if observed != expected_identity:
                raise IntegrityError(
                    f"completed methods do not share split/evaluation/store identity: {key}"
                )
    return rows_by_method[ANCHOR_METHOD], rows_by_method[DESCRIPTIVE_METHOD]


def _critical_source_records(project_root: Path) -> list[dict[str, Any]]:
    relative_paths = (*BASE_DEPENDENCY_PATHS, *FULL_RANK_PAYLOAD_PATHS)
    return [
        describe_file(
            _require_regular_file(project_root / relative, name="critical source"),
            relative_to=project_root,
        )
        for relative in relative_paths
    ]


def load_and_validate_inputs(
    *,
    project_root: Path,
    original_root: Path,
    verify_store_files: bool = True,
) -> ValidatedInputs:
    """Authenticate the frozen campaign without reading any follow-up outcome."""

    project_root = _require_directory(project_root, name="project root")
    original_root = _require_directory(original_root, name="original result root")
    source_trust = validate_source_root(project_root)
    chain = _validate_original_campaign_chain(original_root)
    lock_path = _require_regular_file(
        project_root / "configs/iclr_extension/full_rank_paired_confirmatory_lock.json",
        name="canonical paired-ridge extension lock",
    )
    config_path = chain["canonical_paths"]["resolved_config.json"]
    index_path = chain["canonical_paths"]["feature_store_index.json"]
    rows_path = chain["canonical_paths"]["real/rows.jsonl"]
    lock = read_json(lock_path)
    config = read_json(config_path)
    index = read_json(index_path)
    if (
        not isinstance(lock, dict)
        or not isinstance(config, dict)
        or not isinstance(index, dict)
    ):
        raise IntegrityError("lock/config/index must each be a JSON object")
    validate_lock(lock)
    _validate_original_config(config, lock)
    records = _validate_index(index, lock)
    anchors, descriptive_paired_reduced_rank = _validate_original_rows(
        rows_path, records, lock
    )
    if verify_store_files:
        for count, record in enumerate(records, 1):
            manifest_path = Path(str(record["manifest"])).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = original_root / manifest_path
            manifest_path = _require_regular_file(
                manifest_path, name="indexed feature-store manifest"
            )
            observed = sha256_file(manifest_path)
            if observed != record["manifest_sha256"]:
                raise IntegrityError(
                    f"indexed feature-store manifest changed: {manifest_path}: "
                    f"{observed} != {record['manifest_sha256']}"
                )
            load_feature_store(manifest_path, verify_hashes=True)
            if count % 10 == 0 or count == len(records):
                print(
                    f"Authenticated feature stores: {count}/{len(records)}", flush=True
                )
    binding_payload = {
        "schema_version": BINDING_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "analysis_status": lock["analysis_status"],
        "project_root": str(project_root),
        "original_root": str(original_root),
        "inputs": {
            "extension_lock": describe_file(lock_path),
            "completed_done": describe_file(chain["done_path"]),
            "completed_campaign_manifest": describe_file(
                chain["campaign_manifest_path"]
            ),
            "completed_resolved_config": describe_file(config_path),
            "completed_feature_store_index": describe_file(index_path),
            "completed_real_rows": describe_file(rows_path),
            "completed_source_manifest": describe_file(
                chain["canonical_paths"]["source_manifest.json"]
            ),
        },
        "source_root_of_trust": source_trust,
        "critical_source": _critical_source_records(project_root),
        "validated_original_confirmatory_rows": len(anchors) * len(ORIGINAL_METHODS),
        "validated_anchor_rows": len(anchors),
        "validated_descriptive_paired_reduced_rank_rows": len(
            descriptive_paired_reduced_rank
        ),
        "selected_feature_stores": len(records),
        "expected_new_rows": lock["expected_new_rows"],
        "identity_fields": list(IDENTITY_FIELDS),
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_VARIABLES
            },
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    return ValidatedInputs(
        project_root=project_root,
        original_root=original_root,
        done_path=chain["done_path"],
        campaign_manifest_path=chain["campaign_manifest_path"],
        lock_path=lock_path,
        config_path=config_path,
        index_path=index_path,
        rows_path=rows_path,
        lock=lock,
        config=config,
        records=records,
        anchors=anchors,
        descriptive_paired_reduced_rank=descriptive_paired_reduced_rank,
        binding_payload=binding_payload,
    )


def _encode_nonfinite(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            label = "nan"
        elif value > 0:
            label = "+inf"
        else:
            label = "-inf"
        return {"__nonfinite_float__": label}
    if isinstance(value, Mapping):
        return {str(key): _encode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_nonfinite(item) for item in value]
    return value


def _decode_nonfinite(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__nonfinite_float__"}:
            label = str(value["__nonfinite_float__"])
            try:
                return {
                    "nan": float("nan"),
                    "+inf": float("inf"),
                    "-inf": float("-inf"),
                }[label]
            except KeyError as error:
                raise IntegrityError(f"unknown tagged float: {label}") from error
        return {str(key): _decode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nonfinite(item) for item in value]
    return value


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(_encode_nonfinite(dict(row))))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _unit_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": "correspondence_confirmatory",
        "dataset": str(record["dataset"]),
        "encoder": str(record["encoder"]),
        "corruption": str(record["corruption"]),
        "severity": int(record["severity"]),
        "methods": list(NEW_METHODS),
    }


def _anchor_projection(
    anchors: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    condition: tuple[str, str, str, int],
    split_seeds: Sequence[int],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for seed in split_seeds:
        key = (*condition[:3], int(seed))
        if key not in anchors:
            raise IntegrityError(f"missing pre-outcome anchor: {key}")
        result[str(seed)] = {
            field: str(anchors[key][field]) for field in IDENTITY_FIELDS
        }
    return result


def _condition_suite(lock: Mapping[str, Any]) -> dict[str, Any]:
    protocol = lock["original_protocol"]
    return {
        "datasets": list(protocol["datasets"]),
        "methods": list(NEW_METHODS),
        "calibration_sizes": list(protocol["calibration_sizes"]),
        "pair_budgets": list(protocol["pair_budgets"]),
        "pair_noise_rates": list(protocol["pair_noise_rates"]),
        "split_seeds": list(protocol["split_seeds"]),
        "condition_filter": dict(protocol["condition_filter"]),
    }


def build_unit_plans(
    inputs: ValidatedInputs,
    *,
    output_root: Path,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    suite = _condition_suite(inputs.lock)
    methods = {name: dict(inputs.lock["methods"][name]) for name in NEW_METHODS}
    dataset_protocol = _dataset_protocol(inputs.lock)
    plans = []
    for record in inputs.records:
        identity = _unit_identity(record)
        condition_key = _row_condition_key(record)
        condition = dict(record)
        condition["evaluation_size"] = dataset_protocol[condition_key[0]][
            "evaluation_size"
        ]
        expected_keys = expected_unit_semantic_keys(
            suite_name="correspondence_confirmatory",
            suite=suite,
            methods=methods,
            condition=condition,
        )
        if len(expected_keys) != 6:
            raise IntegrityError(
                f"condition does not expand to six rows: {condition_key}"
            )
        anchors = _anchor_projection(
            inputs.anchors,
            condition_key,
            inputs.lock["original_protocol"]["split_seeds"],
        )
        input_fingerprint = sha256_json(
            {
                "binding_sha256": binding_sha256,
                "identity": identity,
                "feature_store_manifest_sha256": record["manifest_sha256"],
                "anchors": anchors,
                "suite": suite,
                "methods": methods,
                "evaluation_size": condition["evaluation_size"],
            }
        )
        manifest_path = Path(str(record["manifest"])).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = inputs.original_root / manifest_path
        unit_path = output_root / "units" / f"{sha256_json(identity)}.json"
        split_paths = [
            split_plan_path(
                output_root,
                suite_name="correspondence_confirmatory",
                dataset=condition_key[0],
                encoder=condition_key[1],
                corruption=condition_key[2],
                severity=condition_key[3],
                split_seed=int(seed),
            )
            for seed in inputs.lock["original_protocol"]["split_seeds"]
        ]
        plans.append(
            {
                "identity": identity,
                "input_fingerprint": input_fingerprint,
                "record": dict(record),
                "store_path": str(manifest_path.resolve()),
                "condition": condition,
                "suite": suite,
                "methods": methods,
                "global_seed": inputs.lock["original_protocol"]["global_seed"],
                "metric_projections": int(inputs.config.get("metric_projections", 64)),
                "mmd_features": int(inputs.config.get("mmd_features", 512)),
                "distribution_evaluation_limit": inputs.config.get(
                    "distribution_evaluation_limit"
                ),
                "expected_keys": sorted(expected_keys),
                "anchors": anchors,
                "unit_path": str(unit_path),
                "split_paths": [str(path) for path in split_paths],
                "output_root": str(output_root),
            }
        )
    if len(plans) != int(inputs.lock["expected_condition_units"]):
        raise IntegrityError("unit plan count differs from lock")
    return plans


def validate_rows_against_anchors(
    rows: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Mapping[str, str]],
    *,
    identity: Mapping[str, Any],
) -> None:
    expected_methods = set(NEW_METHODS)
    expected_seeds = {int(seed) for seed in anchors}
    observed: set[tuple[str, int]] = set()
    for row in rows:
        method = str(row.get("method"))
        seed = int(row.get("split_seed", -1))
        if method not in expected_methods or seed not in expected_seeds:
            raise IntegrityError(
                f"new row falls outside locked method/seed grid: {method}/{seed}"
            )
        cell = (method, seed)
        if cell in observed:
            raise IntegrityError(f"duplicate new row: {cell}/{identity}")
        observed.add(cell)
        if _row_condition_key(row) != _row_condition_key(identity):
            raise IntegrityError(
                f"new row condition differs from unit identity: {cell}"
            )
        if (
            row.get("suite") != "correspondence_confirmatory"
            or row.get("calibration_size") != 4096
            or row.get("pair_budget") != 4096
        ):
            raise IntegrityError(f"new row protocol dose drifted: {cell}")
        expected = anchors[str(seed)]
        observed_identity = {field: row.get(field) for field in IDENTITY_FIELDS}
        if observed_identity != expected:
            raise IntegrityError(
                "new outcome is not bound to the original split/evaluation/store identity: "
                f"{identity}/{cell}: observed={observed_identity} expected={expected}"
            )
        value = row.get(METRIC)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise IntegrityError(f"new primary metric is non-finite: {identity}/{cell}")
    expected_cells = {
        (method, seed) for method in NEW_METHODS for seed in expected_seeds
    }
    if observed != expected_cells:
        raise IntegrityError(
            f"new unit cell grid mismatch: missing={sorted(expected_cells - observed)}"
        )


def _split_records(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(plan["output_root"])).resolve()
    records = []
    for path_value in plan["split_paths"]:
        path = _require_regular_file(Path(str(path_value)), name="generated split plan")
        relative = path.relative_to(root)
        split = read_json(path)
        if not isinstance(split, dict):
            raise IntegrityError(f"generated split plan is not an object: {path}")
        verify_manifest_checksum(split)
        seed = str(int(str(path.stem).rsplit("seed", 1)[1]))
        expected = plan["anchors"][seed]
        if split.get("manifest_sha256") != expected["split_plan_sha256"]:
            raise IntegrityError(f"generated split-plan identity drifted: {path}")
        if split.get("evaluation_sha256") != expected["evaluation_sha256"]:
            raise IntegrityError(f"generated evaluation identity drifted: {path}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "manifest_sha256": split["manifest_sha256"],
                "evaluation_sha256": split["evaluation_sha256"],
            }
        )
    return sorted(records, key=lambda value: value["path"])


def _write_unit(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    encoded = _encode_nonfinite(list(rows))
    envelope = seal_manifest(
        {
            "schema_version": UNIT_SCHEMA,
            "status": "PASS",
            "identity": dict(plan["identity"]),
            "input_fingerprint": str(plan["input_fingerprint"]),
            "rows_encoding": "strict_json_tagged_nonfinite_v1",
            "record_count": len(rows),
            "rows_sha256": sha256_json(encoded),
            "split_artifacts": _split_records(plan),
            "rows": encoded,
        }
    )
    write_json_atomic(Path(str(plan["unit_path"])), envelope)


def compute_unit(plan: Mapping[str, Any]) -> dict[str, Any]:
    # Recheck inside every child process; the parent-side gate alone would not
    # prove that a worker inherited the locked numerical runtime.
    require_single_thread_environment()
    rows = run_condition(
        store_path=Path(str(plan["store_path"])),
        suite_name="correspondence_confirmatory",
        suite=plan["suite"],
        methods=plan["methods"],
        condition=plan["condition"],
        output_dir=Path(str(plan["output_root"])),
        global_seed=int(plan["global_seed"]),
        metrics_projections=int(plan["metric_projections"]),
        mmd_features=int(plan["mmd_features"]),
        distribution_evaluation_limit=(
            int(plan["distribution_evaluation_limit"])
            if plan.get("distribution_evaluation_limit") is not None
            else None
        ),
    )
    validate_semantic_grid(rows, plan["expected_keys"])
    validate_rows_against_anchors(rows, plan["anchors"], identity=plan["identity"])
    _write_unit(plan, rows)
    return {"identity": dict(plan["identity"]), "record_count": len(rows)}


def load_unit(plan: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    path = Path(str(plan["unit_path"]))
    if not path.exists() and not path.is_symlink():
        return None
    path = _require_regular_file(path, name="unit envelope")
    envelope = read_json(path)
    if not isinstance(envelope, dict):
        raise IntegrityError(f"unit envelope is not an object: {path}")
    verify_manifest_checksum(envelope)
    if (
        envelope.get("schema_version") != UNIT_SCHEMA
        or envelope.get("status") != "PASS"
        or envelope.get("identity") != plan["identity"]
        or envelope.get("input_fingerprint") != plan["input_fingerprint"]
        or envelope.get("rows_encoding") != "strict_json_tagged_nonfinite_v1"
    ):
        raise IntegrityError(f"unit envelope binding drifted: {path}")
    encoded = envelope.get("rows")
    if not isinstance(encoded, list) or sha256_json(encoded) != envelope.get(
        "rows_sha256"
    ):
        raise IntegrityError(f"unit row payload hash mismatch: {path}")
    rows = _decode_nonfinite(encoded)
    if not isinstance(rows, list) or len(rows) != envelope.get("record_count"):
        raise IntegrityError(f"unit record count mismatch: {path}")
    validate_semantic_grid(rows, plan["expected_keys"])
    validate_rows_against_anchors(rows, plan["anchors"], identity=plan["identity"])
    expected_split_records = _split_records(plan)
    if envelope.get("split_artifacts") != expected_split_records:
        raise IntegrityError(f"unit split inventory mismatch: {path}")
    return rows


def _write_state(
    output_root: Path,
    *,
    stage: str,
    completed_units: int,
    expected_units: int,
    status: str = "RUNNING",
    details: Mapping[str, Any] | None = None,
) -> None:
    write_json_atomic(
        output_root / "STATE.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": status,
            "stage": stage,
            "completed_condition_units": int(completed_units),
            "expected_condition_units": int(expected_units),
            "updated_unix": time.time(),
            "details": dict(details or {}),
        },
    )


def _build_inference(
    *,
    new_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    combined = [*new_rows, *anchor_rows]
    inference_lock = lock["inference"]
    results = []
    expected_tests = []
    test_index = 0
    for dataset in ("cifar100_c", "imagenet_c"):
        expected_clusters = len(_dataset_protocol(lock)[dataset]["corruptions"])
        expected_cells = expected_clusters * 3 * 3
        for right in (ANCHOR_METHOD, "paired_ridge_shuffled"):
            effects = paired_effect_rows(
                combined,
                left="paired_ridge",
                right=right,
                metric=METRIC,
                filters={
                    "suite": "correspondence_confirmatory",
                    "dataset": dataset,
                    "calibration_size": 4096,
                    "pair_budget": 4096,
                },
            )
            if len(effects) != expected_cells:
                raise IntegrityError(
                    f"{dataset} paired cell count is {len(effects)}; expected {expected_cells}"
                )
            result = infer_contrast(
                effects,
                cluster_keys=("corruption",),
                bootstrap_repetitions=int(inference_lock["bootstrap_repetitions"]),
                confidence=float(inference_lock["confidence"]),
                seed=int(lock["original_protocol"]["global_seed"]) + test_index,
            )
            if result["clusters"] != expected_clusters:
                raise IntegrityError(f"{dataset} corruption cluster count drifted")
            result["dataset"] = dataset
            result["effect_percentage_points"] = 100.0 * float(result["effect"])
            result["ci_lower_percentage_points"] = 100.0 * float(result["ci_lower"])
            result["ci_upper_percentage_points"] = 100.0 * float(result["ci_upper"])
            result["analysis_status"] = "post_hoc_reviewer_requested"
            result["test_id"] = f"{dataset}:paired_ridge-minus-{right}"
            result["bootstrap_assumption"] = inference_lock["bootstrap_assumption"]
            result["sign_flip_assumption"] = inference_lock["sign_flip_assumption"]
            result["sign_flip_null"] = (
                "zero centered corruption-level paired effect under joint sign symmetry"
            )
            result["sign_flip_calibration"] = inference_lock["sign_flip_calibration"]
            result["sign_flip_assignments"] = 1 << int(result["clusters"])
            result["bootstrap_draws"] = int(inference_lock["bootstrap_repetitions"])
            result["inference_scope"] = (
                "The percentile interval resamples corruption clusters under "
                "within-panel cluster exchangeability. The paired sign-flip p-value "
                "is separately calibrated by sign symmetry of corruption-cluster "
                "effects under the zero-effect null. Encoders and split seeds are "
                "repeated measurements within corruption, not independent clusters."
            )
            expected_tests.append(result["test_id"])
            results.append(result)
            test_index += 1
    if expected_tests != inference_lock["tests"]:
        raise IntegrityError("computed test order differs from locked four-test family")
    adjusted = holm_adjust([float(result["raw_p"]) for result in results])
    for result, value in zip(results, adjusted):
        result["holm_p"] = value
        result["holm_significant_0_05"] = bool(value < 0.05)
    return seal_manifest(
        {
            "schema_version": INFERENCE_SCHEMA,
            "status": "PASS",
            "scientific_result": True,
            "analysis_status": "post_hoc_reviewer_requested",
            "family": inference_lock["family"],
            "multiplicity": (
                "Holm family-wise correction over these four post-hoc tests only; "
                "this family is separate from the prespecified primary family"
            ),
            "cluster_unit": "corruption",
            "cluster_scope": (
                "fixed corruption panels; encoders and split seeds are repeated "
                "measurements within corruption"
            ),
            "bootstrap_assumption": inference_lock["bootstrap_assumption"],
            "sign_flip_assumption": inference_lock["sign_flip_assumption"],
            "sign_flip_calibration": inference_lock["sign_flip_calibration"],
            "assumption_distinction": (
                "Cluster exchangeability motivates the corruption bootstrap interval; "
                "the sign-flip test additionally requires sign symmetry of cluster effects "
                "under its zero-effect null. These are distinct assumptions."
            ),
            "contrasts": results,
        }
    )


def _anchor_rows_sorted(inputs: ValidatedInputs) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in inputs.anchors.values()),
        key=semantic_row_key,
    )


def _descriptive_rows_sorted(inputs: ValidatedInputs) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in inputs.descriptive_paired_reduced_rank.values()),
        key=semantic_row_key,
    )


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return describe_file(path, relative_to=root)


def finalize(
    *,
    inputs: ValidatedInputs,
    output_root: Path,
    binding: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    rows: list[dict[str, Any]],
) -> Path:
    if len(rows) != int(inputs.lock["expected_new_rows"]):
        raise IntegrityError(f"new row count is {len(rows)}; expected 612")
    expected_global: set[str] = set()
    for plan in plans:
        keys = set(plan["expected_keys"])
        overlap = expected_global.intersection(keys)
        if overlap:
            raise IntegrityError(
                f"two units claim the same scientific cells: {sorted(overlap)[:2]}"
            )
        expected_global.update(keys)
    validate_semantic_grid(rows, expected_global, scope="post_hoc_locked_expected")
    rows = sorted(rows, key=semantic_row_key)
    anchor_rows = _anchor_rows_sorted(inputs)
    descriptive_rows = _descriptive_rows_sorted(inputs)
    inference = _build_inference(
        new_rows=rows, anchor_rows=anchor_rows, lock=inputs.lock
    )
    summary = {
        "schema_version": "marginal_alignment.full_rank_paired_confirmatory_summary.v1",
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "post_hoc_reviewer_requested",
        "estimand": "condition macro, then mean and SD across split seeds",
        "inferential_methods": [
            "paired_ridge",
            ANCHOR_METHOD,
            "paired_ridge_shuffled",
        ],
        "descriptive_only_methods": [DESCRIPTIVE_METHOD],
        "descriptive_rows_excluded_from_four_test_family": True,
        "inference_exclusion": (
            "Authenticated paired_reduced_rank rows are included only for a "
            "self-contained descriptive comparison and are outside the four-test "
            "post-hoc Holm family."
        ),
        "records": macro_summary(
            [*rows, *anchor_rows, *descriptive_rows],
            metrics=(METRIC, "aligned_accuracy", "paired_feature_mse"),
        ),
    }
    rows_path = output_root / "rows.jsonl"
    anchor_path = output_root / "authenticated_bures_anchor_rows.jsonl"
    descriptive_path = output_root / "authenticated_paired_reduced_rank_rows.jsonl"
    inference_path = output_root / "INFERENCE.json"
    summary_path = output_root / "SUMMARY.json"
    _atomic_jsonl(rows_path, rows)
    _atomic_jsonl(anchor_path, anchor_rows)
    _atomic_jsonl(descriptive_path, descriptive_rows)
    write_json_atomic(inference_path, inference)
    write_json_atomic(summary_path, summary)
    unit_records = []
    for plan in plans:
        unit_path = Path(str(plan["unit_path"])).resolve()
        unit_records.append(
            {
                "identity": dict(plan["identity"]),
                "input_fingerprint": plan["input_fingerprint"],
                "envelope": _artifact_record(unit_path, output_root),
            }
        )
    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "PASS",
            "scientific_result": True,
            "analysis_status": "post_hoc_reviewer_requested",
            "binding_manifest_sha256": binding["manifest_sha256"],
            "condition_units": len(plans),
            "new_rows": len(rows),
            "authenticated_bures_anchor_rows": len(anchor_rows),
            "authenticated_paired_reduced_rank_rows": len(descriptive_rows),
            "methods": list(NEW_METHODS),
            "suite_name_reused_for_split_identity": "correspondence_confirmatory",
            "artifacts": {
                "rows": _artifact_record(rows_path, output_root),
                "authenticated_bures_anchor_rows": _artifact_record(
                    anchor_path, output_root
                ),
                "authenticated_paired_reduced_rank_rows": _artifact_record(
                    descriptive_path, output_root
                ),
                "inference": _artifact_record(inference_path, output_root),
                "summary": _artifact_record(summary_path, output_root),
            },
            "units": unit_records,
        }
    )
    manifest_path = write_json_atomic(output_root / "MANIFEST.json", manifest)
    done = seal_manifest(
        {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "scientific_result": True,
            "analysis_status": "post_hoc_reviewer_requested",
            "manifest": _artifact_record(manifest_path, output_root),
            "inference": _artifact_record(inference_path, output_root),
            "condition_units": len(plans),
            "new_rows": len(rows),
            "authenticated_bures_anchor_rows": len(anchor_rows),
            "authenticated_paired_reduced_rank_rows": len(descriptive_rows),
        }
    )
    done_path = write_json_atomic(output_root / "DONE.json", done)
    _write_state(
        output_root,
        stage="COMPLETE",
        completed_units=len(plans),
        expected_units=len(plans),
        status="PASS",
        details={
            "done": str(done_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "inference_sha256": inference["manifest_sha256"],
        },
    )
    return done_path


def verify_complete(output_root: Path, binding: Mapping[str, Any]) -> Path | None:
    done_path = output_root / "DONE.json"
    if not done_path.exists() and not done_path.is_symlink():
        return None
    _require_regular_file(done_path, name="completion marker")
    done = read_json(done_path)
    if not isinstance(done, dict):
        raise IntegrityError("completion marker is not an object")
    verify_manifest_checksum(done)
    if done.get("schema_version") != DONE_SCHEMA or done.get("status") != "PASS":
        raise IntegrityError("completion marker schema/status mismatch")
    manifest_record = done.get("manifest")
    if not isinstance(manifest_record, Mapping):
        raise IntegrityError("completion marker lacks manifest record")
    manifest_path = output_root / str(manifest_record["path"])
    if _artifact_record(manifest_path, output_root) != manifest_record:
        raise IntegrityError("completed manifest artifact changed")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise IntegrityError("completed manifest is not an object")
    verify_manifest_checksum(manifest)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("binding_manifest_sha256") != binding["manifest_sha256"]
        or manifest.get("authenticated_bures_anchor_rows") != 306
        or manifest.get("authenticated_paired_reduced_rank_rows") != 306
    ):
        raise IntegrityError(
            "completed result binding differs from current authenticated inputs"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "rows",
        "authenticated_bures_anchor_rows",
        "authenticated_paired_reduced_rank_rows",
        "inference",
        "summary",
    }:
        raise IntegrityError("completed artifact inventory is malformed")
    for record in artifacts.values():
        path = output_root / str(record["path"])
        if _artifact_record(path, output_root) != record:
            raise IntegrityError(f"completed artifact changed: {path}")
    if done.get("inference") != artifacts["inference"]:
        raise IntegrityError(
            "completion marker and manifest disagree on inference artifact"
        )
    units = manifest.get("units")
    if not isinstance(units, list) or len(units) != 102:
        raise IntegrityError("completed unit inventory count drifted")
    seen_envelopes: set[str] = set()
    for unit in units:
        if not isinstance(unit, Mapping) or not isinstance(
            unit.get("envelope"), Mapping
        ):
            raise IntegrityError("completed unit inventory is malformed")
        record = unit["envelope"]
        unit_path = output_root / str(record["path"])
        if str(record["path"]) in seen_envelopes:
            raise IntegrityError("completed unit inventory contains a duplicate")
        seen_envelopes.add(str(record["path"]))
        if _artifact_record(unit_path, output_root) != record:
            raise IntegrityError(f"completed unit envelope changed: {unit_path}")
        envelope = read_json(unit_path)
        if not isinstance(envelope, dict):
            raise IntegrityError(f"completed unit envelope is malformed: {unit_path}")
        verify_manifest_checksum(envelope)
        if (
            envelope.get("schema_version") != UNIT_SCHEMA
            or envelope.get("status") != "PASS"
        ):
            raise IntegrityError(f"completed unit envelope status drifted: {unit_path}")
    if (
        done.get("condition_units") != 102
        or done.get("new_rows") != 612
        or done.get("authenticated_bures_anchor_rows") != 306
        or done.get("authenticated_paired_reduced_rank_rows") != 306
    ):
        raise IntegrityError("completion marker grid counts drifted")
    return done_path


def _prepare_output_root(output_root: Path, binding: Mapping[str, Any]) -> None:
    if output_root.is_symlink():
        raise IntegrityError(f"output root may not be symbolic: {output_root}")
    binding_path = output_root / "INPUT_BINDING.json"
    if not output_root.exists():
        output_root.mkdir(parents=True, mode=0o700)
        write_json_atomic(binding_path, dict(binding))
    else:
        _require_directory(output_root, name="resume output root")
        _require_regular_file(binding_path, name="resume input binding")
        observed = read_json(binding_path)
        if observed != dict(binding):
            raise IntegrityError(
                "existing output root is bound to different inputs/runtime/source"
            )
    for name in ("units", "splits", "failures"):
        _require_output_subdirectory(output_root, name)


def _require_output_subdirectory(output_root: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise IntegrityError(f"invalid output subdirectory name: {name!r}")
    path = output_root / name
    if path.is_symlink():
        raise IntegrityError(f"output subdirectory may not be symbolic: {path}")
    if path.exists():
        if not path.is_dir():
            raise IntegrityError(f"output subdirectory is not a directory: {path}")
    else:
        path.mkdir(mode=0o700)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(output_root)
    except ValueError as error:
        raise IntegrityError(
            f"output subdirectory escapes output root: {path}"
        ) from error
    return resolved


def _resolve_fresh_output_root(output_root: Path, original_root: Path) -> Path:
    """Canonicalize output through existing ancestors and reject input containment."""

    expanded = output_root.expanduser()
    if expanded.is_symlink():
        raise IntegrityError(f"output root may not be symbolic: {expanded}")
    resolved = expanded.resolve(strict=False)
    original = original_root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(original)
    except ValueError:
        return resolved
    raise IntegrityError(
        "fresh extension output may not equal or resolve inside original results"
    )


def run_extension(
    *,
    inputs: ValidatedInputs,
    output_root: Path,
    workers: int,
) -> Path:
    require_single_thread_environment()
    if not 1 <= int(workers) <= 8:
        raise ValueError("workers must lie in [1,8]")
    output_root = _resolve_fresh_output_root(output_root, inputs.original_root)
    binding = seal_manifest(inputs.binding_payload)
    _prepare_output_root(output_root, binding)
    complete = verify_complete(output_root, binding)
    if complete is not None:
        print(f"Existing authenticated completion preserved: {complete}", flush=True)
        return complete
    plans = build_unit_plans(
        inputs, output_root=output_root, binding_sha256=binding["manifest_sha256"]
    )
    expected_unit_paths = {Path(str(plan["unit_path"])).resolve() for plan in plans}
    observed_unit_paths = set((output_root / "units").glob("*.json"))
    unexpected = sorted(
        str(path)
        for path in observed_unit_paths
        if path.resolve() not in expected_unit_paths
    )
    if unexpected:
        raise IntegrityError(
            f"unexpected unit envelopes in fresh extension root: {unexpected[:3]}"
        )
    all_rows: dict[str, list[dict[str, Any]]] = {}
    pending = []
    for plan in plans:
        rows = load_unit(plan)
        key = sha256_json(plan["identity"])
        if rows is None:
            pending.append(plan)
        else:
            all_rows[key] = rows
    _write_state(
        output_root,
        stage="FULL_RANK_CONDITION_UNITS",
        completed_units=len(all_rows),
        expected_units=len(plans),
        details={"pending": len(pending), "workers": int(workers)},
    )
    print(
        f"Untruncated paired-ridge scheduler: units={len(plans)} resume={len(all_rows)} "
        f"compute={len(pending)} workers={workers}",
        flush=True,
    )
    if workers == 1:
        for plan in pending:
            compute_unit(plan)
            key = sha256_json(plan["identity"])
            rows = load_unit(plan)
            assert rows is not None
            all_rows[key] = rows
            _write_state(
                output_root,
                stage="FULL_RANK_CONDITION_UNITS",
                completed_units=len(all_rows),
                expected_units=len(plans),
                details={"pending": len(plans) - len(all_rows), "workers": 1},
            )
            print(f"Completed units: {len(all_rows)}/{len(plans)}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(compute_unit, plan): plan for plan in pending}
            for future in as_completed(futures):
                plan = futures[future]
                future.result()
                key = sha256_json(plan["identity"])
                rows = load_unit(plan)
                assert rows is not None
                all_rows[key] = rows
                _write_state(
                    output_root,
                    stage="FULL_RANK_CONDITION_UNITS",
                    completed_units=len(all_rows),
                    expected_units=len(plans),
                    details={
                        "pending": len(plans) - len(all_rows),
                        "workers": int(workers),
                    },
                )
                print(f"Completed units: {len(all_rows)}/{len(plans)}", flush=True)
    ordered_rows = [
        row for plan in plans for row in all_rows[sha256_json(plan["identity"])]
    ]
    _write_state(
        output_root,
        stage="POST_HOC_INFERENCE",
        completed_units=len(plans),
        expected_units=len(plans),
    )
    return finalize(
        inputs=inputs,
        output_root=output_root,
        binding=binding,
        plans=plans,
        rows=ordered_rows,
    )


def record_failure(
    output_root: Path,
    error: BaseException,
    *,
    original_root: Path,
) -> Path | None:
    output_root = _resolve_fresh_output_root(output_root, original_root)
    if not output_root.is_dir() or output_root.is_symlink():
        return None
    failure_dir = _require_output_subdirectory(output_root, "failures")
    destination = failure_dir / f"FAILURE.{int(time.time())}.{os.getpid()}.json"
    write_json_atomic(
        destination,
        {
            "schema_version": "marginal_alignment.full_rank_paired_confirmatory_failure.v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_unix": time.time(),
        },
    )
    try:
        state = read_json(output_root / "STATE.json")
        completed = int(state.get("completed_condition_units", 0))
        expected = int(state.get("expected_condition_units", 102))
    except Exception:
        completed, expected = 0, 102
    _write_state(
        output_root,
        stage="FAILED",
        completed_units=completed,
        expected_units=expected,
        status="FAILED",
        details={"failure": str(destination)},
    )
    return destination
