"""Top-level evaluation campaign orchestration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from .artifacts import IntegrityError, read_json, sha256_file, sha256_json, write_json_atomic
from .cache_import import write_inventory
from .config import load_config
from .controlled import run_pair_rank_phase_diagram, run_real_embedding_orbit, write_controlled_results
from .feature_store import discover_feature_stores, load_feature_store
from .inference import hierarchical_dataset_inference
from .paper_outputs import generate_paper_outputs
from .real_runner import (
    expected_unit_semantic_keys,
    finalize_rows,
    run_condition,
    semantic_row_key,
    split_plan_path,
    validate_semantic_grid,
)
from .stages import StageContext, StageRunner, environment_manifest, hash_tree
from .streaming import (
    build_streaming_plan,
    run_prequential_stream,
    write_streaming_results,
)
from .synthetic_ceiling import run_synthetic_ceiling_suite


SOURCE_PATTERNS = (
    "src/marginal_alignment/**/*.py", "scripts/**/*.py", "scripts/**/*.sh",
    "configs/campaign/*.json", "protocol/*.json", "tests/**/*.py",
    "pyproject.toml",
)


UPSTREAM_STAGE_NAMES = (
    "preflight", "tests", "gpu_smoke", "synthetic", "inventory",
    "validate_stores", "real", "streaming", "controlled", "inference", "paper_outputs",
)
ALL_STAGE_NAMES = (*UPSTREAM_STAGE_NAMES, "verify")


_T = TypeVar("_T")
_R = TypeVar("_R")
_REAL_WORKER_ENV = "MARGINAL_REAL_WORKERS"
_REAL_WORKER_LIMIT = 8
_ORIGINAL_LOCK_STATUS = "PRESPECIFIED_BEFORE_CAMPAIGN_EXECUTION"
_SOLVER_AMENDMENT_STATUS = (
    "AMENDED_BEFORE_SCIENTIFIC_CAMPAIGN_EXECUTION_AFTER_READOUT_SOLVER_FAILURE"
)
_NUMERICAL_AMENDMENT_STATUS = (
    "AMENDED_AFTER_NUMERICAL_FAILURES_BEFORE_OUTCOME_INSPECTION"
)
_THREAD_COUNT_ENVIRONMENT = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _validate_lock_status(lock: Mapping[str, Any]) -> str:
    """Accept only the original lock or an exact documented amendment state."""

    status = lock.get("status")
    if status == _ORIGINAL_LOCK_STATUS:
        return str(status)
    if status in {_SOLVER_AMENDMENT_STATUS, _NUMERICAL_AMENDMENT_STATUS}:
        amendment = lock.get("amendment")
        date = lock.get("date_utc")
        if not isinstance(amendment, str) or len(amendment.strip()) < 80:
            raise IntegrityError("amended experiment lock lacks a substantive amendment record")
        if not isinstance(date, str) or len(date) != 10:
            raise IntegrityError("amended experiment lock lacks an exact amendment date")
        return str(status)
    raise IntegrityError(f"experiment lock has invalid prespecification status: {status!r}")


def _configured_real_workers(
    environ: Mapping[str, str] | None = None, *, cpu_count: int | None = None,
) -> int:
    """Resolve a conservative condition-worker count and reject oversubscription.

    Each real condition can materialize several float64 feature/covariance
    workspaces, so the supported process-wide ceiling is intentionally eight.
    When more than one condition runs concurrently, every native numerical
    thread pool must be pinned to one thread; otherwise four Python workers can
    silently become dozens of BLAS/OpenMP workers.
    """

    environment = os.environ if environ is None else environ
    available = max(1, int(os.cpu_count() or 1) if cpu_count is None else int(cpu_count))
    maximum = min(_REAL_WORKER_LIMIT, available)
    raw = environment.get(_REAL_WORKER_ENV)
    if raw is None or not str(raw).strip():
        workers = maximum
    else:
        try:
            workers = int(str(raw))
        except ValueError as error:
            raise ValueError(
                f"{_REAL_WORKER_ENV} must be an integer in [1, {maximum}]"
            ) from error
    if not 1 <= workers <= maximum:
        raise ValueError(
            f"{_REAL_WORKER_ENV}={workers} is outside the safe range [1, {maximum}]"
        )
    if workers > 1:
        unsafe = {
            name: environment.get(name)
            for name in _THREAD_COUNT_ENVIRONMENT
            if environment.get(name) != "1"
        }
        if unsafe:
            rendered = ", ".join(
                f"{name}={value!r}" for name, value in sorted(unsafe.items())
            )
            raise RuntimeError(
                "parallel real conditions require every numerical thread pool "
                f"to equal 1; unsafe settings: {rendered}"
            )
    return workers


def _ordered_bounded_map(
    function: Callable[[_T], _R], items: Sequence[_T], *, workers: int,
    on_result: Callable[[int, _R], None] | None = None,
) -> list[_R]:
    """Evaluate independent items concurrently but return input-order results.

    Waiting on futures in submission order makes parent aggregation and parent
    logs deterministic.  If one item fails, pending (not yet running) work is
    cancelled; already-completed atomic unit envelopes remain valid resume
    points for the next invocation.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1 or len(items) <= 1:
        results = []
        for index, item in enumerate(items):
            result = function(item)
            results.append(result)
            if on_result is not None:
                on_result(index, result)
        return results
    executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="marginal-real-condition"
    )
    futures = [executor.submit(function, item) for item in items]
    try:
        results = []
        for index, future in enumerate(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(index, result)
        return results
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _stage_force_flags(
    stage_names: list[str], *, from_stage: str | None, force_stage: str | None,
) -> dict[str, bool]:
    """Validate resume controls and return which stages must be recomputed.

    Every stage is still traversed.  ``from_stage`` means "force this stage and
    every descendant"; it never means "trust earlier state without checking."
    """

    for option_name, option_value in (
        ("from_stage", from_stage), ("force_stage", force_stage)
    ):
        if option_value is not None and option_value not in stage_names:
            raise ValueError(
                f"unknown {option_name}={option_value!r}; expected one of {stage_names}"
            )
    if from_stage is not None and force_stage is not None:
        raise ValueError("--from-stage and --force-stage are mutually exclusive")
    force_from_index = (
        stage_names.index(from_stage) if from_stage is not None else None
    )
    return {
        name: name == force_stage or (
            force_from_index is not None and index >= force_from_index
        )
        for index, name in enumerate(stage_names)
    }


def _condition_allowed(condition: Mapping[str, Any], filter_spec: Any) -> bool:
    if filter_spec in (None, "all"):
        return True
    if not isinstance(filter_spec, Mapping):
        raise ValueError(f"invalid condition_filter: {filter_spec!r}")
    for key in ("corruptions", "severities", "encoders"):
        if key in filter_spec:
            singular = {"corruptions": "corruption", "severities": "severity", "encoders": "encoder"}[key]
            if condition.get(singular) not in filter_spec[key]:
                return False
    return True


def _seed(*values: Any) -> int:
    """Derive a stable NumPy-compatible seed from declared campaign values."""

    return int(sha256_json(list(values))[:16], 16) % (2**63 - 1)


def _index_native_stores(config: dict[str, Any]) -> list[dict[str, Any]]:
    roots = [Path(value) for value in config.get("cache_roots", []) if value]
    discovered = discover_feature_stores(roots)
    records = []
    for item in discovered:
        if item.kind != "native_v2":
            continue
        store = load_feature_store(item.manifest_path, verify_hashes=True)
        metadata = store.manifest.get("metadata", {})
        condition = {
            "dataset": store.dataset,
            "encoder": store.encoder,
            "corruption": metadata.get("corruption"),
            "severity": metadata.get("severity"),
            "evaluation_size": metadata.get("evaluation_size"),
            "manifest": str(item.manifest_path),
            "manifest_sha256": sha256_file(item.manifest_path),
            "manifest_payload": store.manifest,
        }
        records.append(condition)
    return records


def _current_feature_store_identity(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Verify and bind all current native stores before considering a resume.

    Each native manifest seals its array records.  Loading the store here
    verifies those bytes; hashing the sorted manifest set then ensures that a
    valid cache replacement invalidates every saved stage fingerprint.
    """

    records = sorted(({
        "dataset": row["dataset"],
        "encoder": row["encoder"],
        "corruption": row.get("corruption"),
        "severity": row.get("severity"),
        "manifest": str(Path(row["manifest"]).resolve()),
        "manifest_sha256": row["manifest_sha256"],
    } for row in _index_native_stores(config)), key=lambda row: (
        str(row["dataset"]), str(row["encoder"]), str(row["corruption"]),
        str(row["severity"]), str(row["manifest"]),
    ))
    payload = {
        "schema_version": "marginal_alignment.campaign.external_inputs.v1",
        "status": "PASS",
        "native_feature_store_count": len(records),
        "native_feature_stores": records,
    }
    return sha256_json(payload), payload


def _dataset_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    for item in config["datasets"]:
        if item["name"] == name:
            return item
    raise KeyError(name)


def _expected_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected = []
    for dataset in config["datasets"]:
        for encoder in dataset["encoders"]:
            for corruption in dataset["corruptions"]:
                for severity in dataset["severities"]:
                    expected.append({
                        "dataset": dataset["name"], "encoder": encoder,
                        "corruption": corruption, "severity": severity,
                        "evaluation_size": dataset["evaluation_size"],
                    })
    return expected


def _stage_preflight(context: StageContext) -> Iterable[Path]:
    output = context.run_dir / "preflight.json"
    environment = environment_manifest()
    if environment["packages"]["numpy"].startswith("UNAVAILABLE"):
        raise RuntimeError("NumPy is unavailable")
    if environment["packages"]["scipy"].startswith("UNAVAILABLE"):
        raise RuntimeError("SciPy is unavailable")
    if environment["packages"]["sklearn"].startswith("UNAVAILABLE"):
        raise RuntimeError("scikit-learn is unavailable")
    cache_roots = [Path(value) for value in context.config.get("cache_roots", []) if value]
    present_roots = [path for path in cache_roots if path.is_dir()]
    if not present_roots:
        raise FileNotFoundError(f"none of the configured cache roots exists: {cache_roots}")
    disk = shutil.disk_usage(context.run_dir)
    if disk.free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free at output root")
    write_json_atomic(output, {
        "schema_version": "marginal_alignment.campaign.preflight.v2",
        "status": "PASS",
        "environment": environment,
        "cache_roots": [str(path.resolve()) for path in present_roots],
        "free_bytes": disk.free,
        "config_sha256": context.config_hash,
        "source_sha256": context.source_hash,
        "runtime_sha256": context.runtime_hash,
    })
    return [output]


def _stage_tests(context: StageContext) -> Iterable[Path]:
    output = context.run_dir / "test_report.txt"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(context.project_root / "src")
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    result = subprocess.run(command, cwd=context.project_root, env=environment,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"test suite failed; see {output}")
    return [output]


def _stage_gpu_smoke(context: StageContext) -> Iterable[Path]:
    """Exercise the selected CUDA runtime before a long campaign."""

    output = context.run_dir / "gpu_smoke.json"
    environment = environment_manifest()
    cuda = environment.get("cuda", {})
    require_gpu = bool(context.config.get("require_gpu", True))
    if require_gpu and not cuda.get("available"):
        raise RuntimeError("campaign requires CUDA but torch.cuda.is_available() is false")
    result: dict[str, Any] = {"required": require_gpu, "cuda": cuda}
    if cuda.get("available"):
        import torch
        generator = torch.Generator(device="cuda").manual_seed(int(context.config["global_seed"]))
        left = torch.randn((512, 512), device="cuda", generator=generator)
        right = torch.randn((512, 512), device="cuda", generator=generator)
        product = left @ right
        torch.cuda.synchronize()
        if not bool(torch.isfinite(product).all()):
            raise RuntimeError("CUDA smoke product is non-finite")
        result.update({
            "status": "PASS", "device": torch.cuda.get_device_name(0),
            "checksum": float(product[:8, :8].sum().cpu()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        })
    else:
        result.update({"status": "PASS_CPU_ONLY"})
    write_json_atomic(output, result)
    return [output]


def _stage_synthetic(context: StageContext) -> Iterable[Path]:
    config = context.config["synthetic"]
    output = context.run_dir / "synthetic"
    result = run_synthetic_ceiling_suite(
        output,
        exact_kwargs=dict(config["exact_indistinguishable"]),
        approximate_kwargs=dict(config["approximate_two_world"]),
        identifiable_kwargs=dict(config["identifiable_nongaussian"]),
    )
    manifest = output / "manifest.json"
    if result["summary"].get("status") != "PASS" or read_json(manifest).get("status") != "PASS":
        raise RuntimeError("synthetic scientific gates failed")
    return sorted(path for path in output.iterdir() if path.is_file())


def _stage_inventory(context: StageContext) -> Iterable[Path]:
    output = context.run_dir / "cache_inventory.json"
    roots = [
        Path(value) for value in context.config.get("cache_roots", [])
        if Path(value).is_dir()
    ]
    native = []
    for item in discover_feature_stores(roots):
        if item.kind == "native_v2":
            native.append({
                "path": str(item.manifest_path),
                "sha256": sha256_file(item.manifest_path),
                "bytes": int(item.manifest_path.stat().st_size),
                "dataset": item.dataset,
                "encoder": item.encoder,
            })
    write_json_atomic(output, {
        "schema_version": "marginal_alignment.campaign.cache_inventory.v3",
        "status": "PASS",
        "cache_roots": [str(path.resolve()) for path in roots],
        "native_feature_store_manifests": sorted(native, key=lambda row: row["path"]),
        "manifest_set_sha256": sha256_json(sorted(native, key=lambda row: row["path"])),
    })
    return [output]


def _load_indexed_store(record: Mapping[str, Any]):
    """Load one store only if its current manifest is the indexed artifact."""

    manifest_path = Path(str(record["manifest"])).resolve()
    current = sha256_file(manifest_path)
    if current != record["manifest_sha256"]:
        raise IntegrityError(
            f"feature-store manifest changed after validation: {manifest_path}"
        )
    return load_feature_store(manifest_path, verify_hashes=True)


def _locked_feature_build_canonical_hash(context: StageContext) -> str:
    """Read the feature-build identity copied into this campaign evidence.

    The store-validation stage must bind every representation to the same
    prespecified upstream build protocol.  Reading the copied lock (rather
    than a mutable external path) also makes the resulting stage evidence
    self-contained.
    """

    copied_lock = context.run_dir / "experiment_lock.json"
    lock_path = copied_lock if copied_lock.is_file() else Path(
        context.config.get(
            "lock_path", context.project_root / "protocol" / "experiment_lock.json"
        )
    )
    if not lock_path.is_file():
        raise IntegrityError(f"experiment lock is missing during store validation: {lock_path}")
    value = read_json(lock_path).get("feature_build_config_canonical_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise IntegrityError(
            "experiment lock lacks an exact feature_build_config_canonical_sha256"
        )
    return value


def _stage_validate_stores(context: StageContext) -> Iterable[Path]:
    output = context.run_dir / "feature_store_index.json"
    stores = _index_native_stores(context.config)
    keys: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in stores:
        key = (row["dataset"], row["encoder"], row["corruption"], row["severity"])
        if key in keys:
            raise IntegrityError(
                f"duplicate native V2 condition store {key}: "
                f"{keys[key]['manifest']} and {row['manifest']}"
            )
        keys[key] = row
    missing = []
    required = []
    for expected in _expected_conditions(context.config):
        used = False
        probe = dict(expected)
        for suite in context.config["suites"].values():
            if expected["dataset"] in suite["datasets"] and _condition_allowed(
                probe, suite.get("condition_filter")
            ):
                used = True
                break
        if not used:
            continue
        required.append(expected)
        key = (expected["dataset"], expected["encoder"], expected["corruption"], expected["severity"])
        if key not in keys:
            missing.append(key)
    if missing:
        raise IntegrityError(
            f"{len(missing)} required native V2 condition stores are missing; first={missing[:5]}. "
            "Use cache inventory/import stage; conditions are never silently skipped."
        )
    required_keys = {
        (row["dataset"], row["encoder"], row["corruption"], row["severity"])
        for row in required
    }
    records = [keys[key] for key in sorted(required_keys, key=str)]
    locked_feature_build_hash = _locked_feature_build_canonical_hash(context)
    representation_identity_by_encoder: dict[str, dict[str, str | None]] = {}
    for record in records:
        dataset = _dataset_spec(context.config, str(record["dataset"]))
        manifest = record["manifest_payload"]
        model = manifest.get("metadata", {}).get("model_provenance", {})
        if str(model.get("encoder_id")) != str(record["encoder"]):
            raise IntegrityError(
                f"feature-store encoder provenance mismatch: {record['manifest']}"
            )
        if model.get("feature_build_config_sha256") != locked_feature_build_hash:
            raise IntegrityError(
                "feature-store build protocol differs from the prespecified lock: "
                f"{record['dataset']}/{record['encoder']}"
            )
        representation_identity = {
            "checkpoint_sha256": str(model.get("checkpoint_sha256", "")),
            "preprocessing_sha256": str(model.get("preprocessing_sha256", "")),
            "feature_layer": str(model.get("feature_layer", "")),
            "pooling": str(model.get("pooling", "")),
            # Older native-store schemas did not require model_id.  It remains
            # optional, but mixing a declared and undeclared/different model
            # under one encoder label is still rejected.
            "model_id": (
                str(model["model_id"]) if model.get("model_id") not in (None, "")
                else None
            ),
            "feature_build_config_sha256": locked_feature_build_hash,
        }
        encoder = str(record["encoder"])
        previous_identity = representation_identity_by_encoder.get(encoder)
        if previous_identity is None:
            representation_identity_by_encoder[encoder] = representation_identity
        elif representation_identity != previous_identity:
            changed = sorted(
                key for key in representation_identity
                if representation_identity[key] != previous_identity[key]
            )
            raise IntegrityError(
                "feature stores mix representation identities under encoder "
                f"{encoder!r}; differing fields={changed}; "
                f"store={record['dataset']}/{record['corruption']}/s{record['severity']}"
            )
        if model.get("head_protocol") != dataset["head_protocol"]:
            raise IntegrityError(
                "feature-store head protocol does not match the locked dataset protocol: "
                f"{record['dataset']}/{record['encoder']}"
            )
        if str(model.get("reference_split")) != str(dataset["reference_split"]):
            raise IntegrityError(
                "feature-store reference split does not match the locked dataset split: "
                f"{record['dataset']}/{record['encoder']}"
            )
        if str(model.get("head_fit_split")) != str(dataset["head_fit_split"]):
            raise IntegrityError(
                "feature-store head-fit split does not match the locked dataset split: "
                f"{record['dataset']}/{record['encoder']}"
            )
        reference_provenance = manifest.get("id_provenance", {}).get("reference", {})
        reference_namespace = str(reference_provenance.get("namespace", ""))
        if str(model.get("reference_id_namespace")) != reference_namespace:
            raise IntegrityError(
                "feature-store model/reference ID namespaces disagree: "
                f"{record['dataset']}/{record['encoder']}"
            )
        if dataset["head_fit_split"] == dataset["reference_split"]:
            if str(model.get("head_fit_id_namespace")) != reference_namespace:
                raise IntegrityError(
                    "head and endpoint share a split, but head-fit IDs are not in the "
                    f"reference namespace: {record['dataset']}/{record['encoder']}"
                )
            store = _load_indexed_store(record)
            fit_ids = set(np.asarray(store.array(str(model["head_fit_ids_role"]))).tolist())
            reference_ids = set(np.asarray(store.endpoint("reference").ids).tolist())
            if not fit_ids or not fit_ids.issubset(reference_ids):
                raise IntegrityError(
                    "head-fit IDs must be a nonempty subset of the shared reference split: "
                    f"{record['dataset']}/{record['encoder']}"
                )
    write_json_atomic(output, {
        "schema_version": "marginal_alignment.campaign.store_index.v2",
        "status": "PASS", "records": records,
        "locked_feature_build_config_canonical_sha256": locked_feature_build_hash,
        "representation_identity_by_encoder": representation_identity_by_encoder,
    })
    return [output]


def _unit_identity(suite_name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "dataset": record["dataset"],
        "encoder": record["encoder"],
        "corruption": record.get("corruption"),
        "severity": record.get("severity"),
    }


def _unit_key(identity: Mapping[str, Any]) -> str:
    readable = "__".join(str(identity.get(key)) for key in (
        "suite", "dataset", "encoder", "corruption", "severity"
    ))
    return f"{readable}__{sha256_json(dict(identity))[:16]}"


def _unit_input_fingerprint(
    context: StageContext, *, suite_name: str, suite: Mapping[str, Any],
    record: Mapping[str, Any], methods: Mapping[str, Mapping[str, Any]],
) -> str:
    selected_methods = {name: methods[name] for name in suite["methods"]}
    return sha256_json({
        "schema_version": "marginal_alignment.campaign.real_unit_input.v2",
        "config_sha256": context.config_hash,
        "source_sha256": context.source_hash,
        "runtime_sha256": context.runtime_hash,
        "external_input_sha256": context.external_input_hash,
        "feature_store_manifest_sha256": record["manifest_sha256"],
        "identity": _unit_identity(suite_name, record),
        "suite": dict(suite),
        "methods": selected_methods,
        "evaluation_size": int(record["evaluation_size"]),
        "global_seed": int(context.config["global_seed"]),
        "metric_projections": int(context.config.get("metric_projections", 64)),
        "mmd_features": int(context.config.get("mmd_features", 512)),
        "distribution_evaluation_limit": context.config.get(
            "distribution_evaluation_limit"
        ),
    })


def _encode_nonfinite(value: Any) -> Any:
    """Make result rows strict-JSON serializable without losing NaN semantics."""

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
            label = value["__nonfinite_float__"]
            return {"nan": float("nan"), "+inf": float("inf"),
                    "-inf": float("-inf")}[str(label)]
        return {key: _decode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nonfinite(item) for item in value]
    return value


def _unit_split_paths(
    real_dir: Path, *, suite_name: str, suite: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[Path]:
    return [
        split_plan_path(
            real_dir, suite_name=suite_name, dataset=str(record["dataset"]),
            encoder=str(record["encoder"]), corruption=record.get("corruption"),
            severity=record.get("severity"), split_seed=int(split_seed),
        )
        for split_seed in suite["split_seeds"]
    ]


def _split_artifact_records(paths: Iterable[Path], *, real_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"unit did not produce split plan: {path}")
        records.append({
            "path": path.resolve().relative_to(real_dir.resolve()).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return records


def _write_real_unit(
    path: Path, *, identity: Mapping[str, Any], input_fingerprint: str,
    rows: list[dict[str, Any]], expected_keys: set[str],
    split_artifacts: list[dict[str, Any]],
) -> None:
    grid = validate_semantic_grid(rows, expected_keys)
    encoded_rows = _encode_nonfinite(rows)
    write_json_atomic(path, {
        "schema_version": "marginal_alignment.campaign.real_unit.v2",
        "status": "PASS",
        "unit_key": _unit_key(identity),
        "identity": dict(identity),
        "input_fingerprint": input_fingerprint,
        "rows_encoding": "strict_json_tagged_nonfinite_v1",
        "record_count": len(rows),
        "rows_sha256": sha256_json(encoded_rows),
        "semantic_grid_validation": grid,
        "split_artifacts": split_artifacts,
        "rows": encoded_rows,
    })


def _load_real_unit(
    path: Path, *, identity: Mapping[str, Any], input_fingerprint: str,
    expected_keys: set[str], real_dir: Path,
) -> list[dict[str, Any]] | None:
    """Return verified resumable rows, or None so the unit is recomputed."""

    if not path.is_file():
        return None
    try:
        envelope = read_json(path)
        if envelope.get("schema_version") != "marginal_alignment.campaign.real_unit.v2":
            raise IntegrityError("unit schema changed")
        if envelope.get("status") != "PASS":
            raise IntegrityError("unit is not PASS")
        if envelope.get("identity") != dict(identity):
            raise IntegrityError("unit identity does not match its filename")
        if envelope.get("unit_key") != _unit_key(identity):
            raise IntegrityError("unit key mismatch")
        if envelope.get("input_fingerprint") != input_fingerprint:
            raise IntegrityError("unit input fingerprint is stale")
        encoded_rows = envelope.get("rows")
        if not isinstance(encoded_rows, list):
            raise IntegrityError("unit rows are malformed")
        if sha256_json(encoded_rows) != envelope.get("rows_sha256"):
            raise IntegrityError("unit row checksum mismatch")
        rows = _decode_nonfinite(encoded_rows)
        if not isinstance(rows, list) or len(rows) != int(envelope.get("record_count", -1)):
            raise IntegrityError("unit record count mismatch")
        grid = validate_semantic_grid(rows, expected_keys)
        if envelope.get("semantic_grid_validation") != grid:
            raise IntegrityError("unit semantic-grid report is stale")
        split_records = envelope.get("split_artifacts")
        if not isinstance(split_records, list) or not split_records:
            raise IntegrityError("unit split-artifact inventory is absent")
        root = real_dir.resolve()
        for record in split_records:
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise IntegrityError("unit split path is unsafe")
            split_path = (root / relative).resolve()
            try:
                split_path.relative_to(root)
            except ValueError as error:
                raise IntegrityError("unit split path escapes real directory") from error
            if not split_path.is_file():
                raise IntegrityError(f"unit split artifact is missing: {relative}")
            if split_path.stat().st_size != int(record.get("bytes", -1)):
                raise IntegrityError(f"unit split artifact size changed: {relative}")
            if sha256_file(split_path) != record.get("sha256"):
                raise IntegrityError(f"unit split artifact hash changed: {relative}")
        return rows
    except (IntegrityError, KeyError, TypeError, ValueError) as error:
        print(f"[unit recompute] {path.name}: {error}", flush=True)
        return None


def _compute_real_unit(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compute and atomically seal one independent real-condition unit."""

    try:
        unit_rows = run_condition(
            store_path=Path(plan["record"]["manifest"]),
            suite_name=str(plan["suite_name"]), suite=plan["suite"],
            methods=plan["methods"], condition=plan["condition"],
            output_dir=Path(plan["real_dir"]), global_seed=int(plan["global_seed"]),
            metrics_projections=int(plan["metrics_projections"]),
            mmd_features=int(plan["mmd_features"]),
            distribution_evaluation_limit=plan["distribution_evaluation_limit"],
        )
    except Exception as error:
        raise RuntimeError(
            "real-condition unit failed: "
            + json.dumps(dict(plan["identity"]), sort_keys=True)
        ) from error
    expected_unit = set(plan["expected_unit"])
    validate_semantic_grid(unit_rows, expected_unit)
    real_dir = Path(plan["real_dir"])
    split_records = _split_artifact_records(
        [Path(path) for path in plan["split_paths"]], real_dir=real_dir,
    )
    _write_real_unit(
        Path(plan["unit_path"]), identity=plan["identity"],
        input_fingerprint=str(plan["input_fingerprint"]), rows=unit_rows,
        expected_keys=expected_unit, split_artifacts=split_records,
    )
    return {
        "rows": unit_rows,
        "split_artifacts": split_records,
        "resume_status": "computed",
    }


def _stage_real(context: StageContext) -> Iterable[Path]:
    index = read_json(context.run_dir / "feature_store_index.json")["records"]
    # Verify the complete indexed input set before resuming even one condition;
    # otherwise an interrupted run could mix old completed units with a newly
    # replaced feature store in the remaining units.
    for record in index:
        _load_indexed_store(record)
    methods = {**context.config["methods"], **context.config.get("diagnostic_methods", {})}
    real_dir = context.run_dir / "real"
    unit_dir = real_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    expected_global: set[str] = set()
    unit_records: list[dict[str, Any]] = []
    current_unit_paths: set[Path] = set()
    current_split_paths: set[Path] = set()
    plans: list[dict[str, Any]] = []
    for suite_name, suite in context.config["suites"].items():
        for record in index:
            if record["dataset"] not in suite["datasets"] or not _condition_allowed(
                record, suite.get("condition_filter")
            ):
                continue
            condition = dict(record)
            dataset = _dataset_spec(context.config, record["dataset"])
            condition["evaluation_size"] = int(dataset["evaluation_size"])
            expected_unit = expected_unit_semantic_keys(
                suite_name=suite_name, suite=suite, methods=methods,
                condition=condition,
            )
            overlap = expected_global.intersection(expected_unit)
            if overlap:
                raise IntegrityError(
                    "two configured units claim the same semantic cells: "
                    f"{sorted(overlap)[:2]}"
                )
            expected_global.update(expected_unit)
            identity = _unit_identity(suite_name, condition)
            unit_key = _unit_key(identity)
            unit_path = unit_dir / f"{sha256_json(identity)}.json"
            current_unit_paths.add(unit_path.resolve())
            input_fingerprint = _unit_input_fingerprint(
                context, suite_name=suite_name, suite=suite, record=condition,
                methods=methods,
            )
            split_paths = _unit_split_paths(
                real_dir, suite_name=suite_name, suite=suite, record=condition,
            )
            current_split_paths.update(path.resolve() for path in split_paths)
            unit_rows = _load_real_unit(
                unit_path, identity=identity, input_fingerprint=input_fingerprint,
                expected_keys=expected_unit, real_dir=real_dir,
            )
            resumed_result = None
            if unit_rows is not None:
                resumed_result = {
                    "rows": unit_rows,
                    "split_artifacts": list(read_json(unit_path)["split_artifacts"]),
                    "resume_status": "verified_resume",
                }
            plans.append({
                "record": dict(record),
                "suite_name": suite_name,
                "suite": dict(suite),
                "methods": methods,
                "condition": condition,
                "real_dir": real_dir,
                "unit_path": unit_path,
                "unit_key": unit_key,
                "identity": identity,
                "input_fingerprint": input_fingerprint,
                "expected_unit": sorted(expected_unit),
                "split_paths": split_paths,
                "global_seed": int(context.config["global_seed"]),
                "metrics_projections": int(context.config.get("metric_projections", 64)),
                "mmd_features": int(context.config.get("mmd_features", 512)),
                "distribution_evaluation_limit": (
                    int(context.config["distribution_evaluation_limit"])
                    if context.config.get("distribution_evaluation_limit") is not None
                    else None
                ),
                "resumed_result": resumed_result,
            })

    pending_plans = [plan for plan in plans if plan["resumed_result"] is None]
    workers = _configured_real_workers() if pending_plans else 1
    print(
        f"real scheduler: units={len(plans)} resume={len(plans) - len(pending_plans)} "
        f"compute={len(pending_plans)} workers={workers} "
        "aggregation=input_order numerical_threads=1",
        flush=True,
    )
    def log_computed(index: int, result: Mapping[str, Any]) -> None:
        plan = pending_plans[index]
        identity = plan["identity"]
        print(
            f"real compute {index + 1}/{len(pending_plans)} complete: "
            f"{identity['dataset']}/{identity['encoder']}/"
            f"{identity['corruption']}/s{identity['severity']} "
            f"rows={len(result['rows'])}",
            flush=True,
        )

    computed_results = iter(_ordered_bounded_map(
        _compute_real_unit, pending_plans, workers=workers,
        on_result=log_computed,
    ))
    for ordinal, plan in enumerate(plans, start=1):
        result = plan["resumed_result"]
        if result is None:
            result = next(computed_results)
        unit_rows = result["rows"]
        split_records = result["split_artifacts"]
        all_rows.extend(unit_rows)
        unit_path = Path(plan["unit_path"])
        unit_records.append({
            "unit_key": plan["unit_key"],
            "identity": plan["identity"],
            "input_fingerprint": plan["input_fingerprint"],
            "record_count": len(unit_rows),
            "envelope": unit_path.resolve().relative_to(
                context.run_dir.resolve()
            ).as_posix(),
            "envelope_sha256": sha256_file(unit_path),
            "split_artifacts": split_records,
            "resume_status": result["resume_status"],
        })
        record = plan["record"]
        print(
            f"real unit {ordinal}/{len(plans)}: "
            f"{record['dataset']}/{record['encoder']}/"
            f"{record['corruption']}/s{record['severity']}: "
            f"rows={len(unit_rows)} status="
            f"{'resume' if result['resume_status'] == 'verified_resume' else 'computed'} "
            f"total={len(all_rows)}",
            flush=True,
        )
    if not all_rows:
        raise RuntimeError("real campaign produced no rows")
    # A changed experiment grid must use a fresh run directory rather than
    # silently publishing stale unit/split artifacts from an older grid.
    observed_units = {path.resolve() for path in unit_dir.glob("*.json")}
    observed_splits = {path.resolve() for path in (real_dir / "splits").glob("*.json")}
    stale_units = sorted(observed_units - current_unit_paths)
    stale_splits = sorted(observed_splits - current_split_paths)
    if stale_units or stale_splits:
        raise IntegrityError(
            "run directory contains artifacts from a different real grid; use a fresh "
            f"run directory (stale_units={len(stale_units)}, "
            f"stale_splits={len(stale_splits)})"
        )
    validate_semantic_grid(all_rows, expected_global)
    primary_outputs = finalize_rows(
        all_rows, real_dir, expected_semantic_keys=expected_global,
        unit_records=unit_records,
    )
    return [
        *primary_outputs,
        *sorted(current_unit_paths),
        *sorted(current_split_paths),
    ]


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _streaming_ids(
    store: Any, *, stream_rows: int, reference_rows: int, seed: int,
) -> tuple[list[Any], list[Any]]:
    """Select a sealed stream and a disjoint clean-reference calibration set."""

    reference = store.endpoint("reference")
    target = store.endpoint("target")
    paired_reference, paired_target = store.pairing().validated()
    model = store.manifest["metadata"]["model_provenance"]
    head_fit_ids = {
        value.item() if isinstance(value, np.generic) else value
        for value in np.asarray(store.array(str(model["head_fit_ids_role"])))
    }
    head_namespace = str(model["head_fit_id_namespace"])
    eligible = [
        (reference_id, target_id)
        for reference_id, target_id in zip(paired_reference, paired_target)
        if not (
            (head_namespace == reference.id_namespace and reference_id in head_fit_ids)
            or (head_namespace == target.id_namespace and target_id in head_fit_ids)
        )
    ]
    required = int(stream_rows) + int(reference_rows)
    if required > len(eligible):
        raise IntegrityError(
            "stream plus disjoint reference calibration exceeds verified eligible pairs: "
            f"required={required}, eligible={len(eligible)}"
        )
    order = np.random.default_rng(int(seed)).permutation(len(eligible))[:required]
    stream_positions = order[:int(stream_rows)]
    reference_positions = order[int(stream_rows):]
    target_order = [eligible[int(index)][1] for index in stream_positions]
    reference_calibration = [eligible[int(index)][0] for index in reference_positions]
    return target_order, reference_calibration


def _streaming_unit_resume(
    unit_dir: Path, *, plan: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    manifest_path = unit_dir / "manifest.json"
    summary_path = unit_dir / "summary.json"
    rows_path = unit_dir / "rows.jsonl"
    if not all(path.is_file() for path in (manifest_path, summary_path, rows_path)):
        return None
    try:
        manifest = read_json(manifest_path)
        summary = read_json(summary_path)
        if manifest.get("status") != "PASS" or summary.get("status") != "PASS":
            raise IntegrityError("streaming unit is not PASS")
        if manifest.get("plan_manifest_sha256") != plan["manifest_sha256"]:
            raise IntegrityError("streaming unit plan changed")
        if manifest.get("feature_store_manifest_sha256") != plan[
            "feature_store_manifest_sha256"
        ]:
            raise IntegrityError("streaming unit feature store changed")
        if manifest.get("rows_sha256") != sha256_file(rows_path):
            raise IntegrityError("streaming unit rows changed")
        if manifest.get("summary_sha256") != sha256_file(summary_path):
            raise IntegrityError("streaming unit summary changed")
        if summary.get("scientific_gates", {}).get("all_passed") is not True:
            raise IntegrityError("streaming unit scientific gates failed")
        aggregates = summary.get("aggregates")
        if not isinstance(aggregates, list) or len(aggregates) != int(
            manifest.get("aggregate_count", -1)
        ):
            raise IntegrityError("streaming unit aggregate grid is malformed")
        return list(aggregates)
    except (IntegrityError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"[streaming unit recompute] {unit_dir.name}: {error}", flush=True)
        return None


def _stage_streaming(context: StageContext) -> Iterable[Path]:
    specification = context.config["streaming_secondary"]
    output = context.run_dir / "streaming"
    output.mkdir(parents=True, exist_ok=True)
    if specification.get("enabled") is not True:
        raise IntegrityError("the locked streaming stage may not be silently disabled")
    records = read_json(context.run_dir / "feature_store_index.json")["records"]
    selected = [
        record for record in records
        if record["dataset"] in specification["datasets"]
        and record["encoder"] in specification["encoders"]
        and _condition_allowed(record, specification.get("condition_filter"))
    ]
    if not selected:
        raise IntegrityError("streaming secondary selected no verified conditions")
    all_aggregates: list[dict[str, Any]] = []
    unit_records: list[dict[str, Any]] = []
    outputs: list[Path] = []
    current_plans: set[Path] = set()
    current_unit_manifests: set[Path] = set()
    for record in selected:
        store = _load_indexed_store(record)
        for order_seed in specification["order_seeds"]:
            selection_seed = _seed(
                context.config["global_seed"], "strict_prequential_order",
                record["dataset"], record["encoder"], record.get("corruption"),
                record.get("severity"), int(order_seed),
            )
            target_order, reference_calibration = _streaming_ids(
                store, stream_rows=int(specification["stream_rows"]),
                reference_rows=int(specification["reference_calibration_rows"]),
                seed=selection_seed,
            )
            identity = {
                "dataset": record["dataset"], "encoder": record["encoder"],
                "corruption": record.get("corruption"),
                "severity": record.get("severity"), "order_seed": int(order_seed),
                "feature_store_manifest_sha256": record["manifest_sha256"],
            }
            token = sha256_json(identity)
            plan_path = output / "plans" / f"{token}.json"
            unit_dir = output / "units" / token
            plan = build_streaming_plan(
                record["manifest"], target_order_ids=target_order,
                reference_calibration_ids=reference_calibration,
                order_provenance={
                    "verified": True,
                    "kind": "prespecified_seeded_permutation_of_verified_pair_table",
                    "source": (
                        f"campaign_config_sha256={context.config_hash}; "
                        f"pairing_sha256={store.pairing().content_hash}; "
                        f"order_seed={int(order_seed)}"
                    ),
                },
                block_size=int(specification["block_size"]),
                metric_seed=_seed(context.config["global_seed"], "stream_metrics", token),
                metric_projections=int(specification["metric_projections"]),
                mmd_features=int(specification["mmd_features"]),
                distribution_limit=int(specification["distribution_limit"]),
                execution_provenance={
                    "config_sha256": context.config_hash,
                    "source_sha256": context.source_hash,
                    "runtime_sha256": context.runtime_hash,
                },
                destination=plan_path,
            )
            current_plans.add(plan_path.resolve())
            current_unit_manifests.add((unit_dir / "manifest.json").resolve())
            aggregates = _streaming_unit_resume(unit_dir, plan=plan)
            resumed = aggregates is not None
            if aggregates is None:
                result = run_prequential_stream(record["manifest"], plan_path)
                if result.get("status") != "PASS":
                    raise RuntimeError(
                        f"streaming scientific gates failed for {identity}: "
                        f"{result.get('scientific_gates')}"
                    )
                unit_outputs = write_streaming_results(unit_dir, result)
                aggregates = list(result["aggregates"])
            else:
                unit_outputs = [
                    unit_dir / "rows.jsonl", unit_dir / "summary.json",
                    unit_dir / "manifest.json",
                ]
            for aggregate in aggregates:
                all_aggregates.append({**identity, **aggregate})
            outputs.extend([plan_path, *unit_outputs])
            unit_records.append({
                **identity,
                "plan": plan_path.relative_to(output).as_posix(),
                "plan_sha256": sha256_file(plan_path),
                "unit_manifest": (unit_dir / "manifest.json").relative_to(output).as_posix(),
                "unit_manifest_sha256": sha256_file(unit_dir / "manifest.json"),
                "aggregate_count": len(aggregates),
                "resume_status": "verified_resume" if resumed else "computed",
            })
            print(
                f"streaming {record['dataset']}/{record['encoder']}/"
                f"{record.get('corruption')}/s{record.get('severity')}/order{order_seed}: "
                f"{'resume' if resumed else 'computed'}",
                flush=True,
            )
    stale_plans = {path.resolve() for path in (output / "plans").glob("*.json")} - current_plans
    stale_units = {
        path.resolve() for path in (output / "units").glob("*/manifest.json")
    } - current_unit_manifests
    if stale_plans or stale_units:
        raise IntegrityError(
            "streaming directory contains artifacts from another locked grid; "
            f"stale_plans={len(stale_plans)}, stale_units={len(stale_units)}"
        )
    aggregate_path = output / "aggregate_rows.jsonl"
    _atomic_jsonl(aggregate_path, all_aggregates)
    manifest_path = output / "manifest.json"
    write_json_atomic(manifest_path, {
        "schema_version": "marginal_alignment.campaign.streaming_collection.v1",
        "status": "PASS",
        "protocol": "strict_predict_before_update",
        "condition_units": len(unit_records),
        "expected_condition_units": len(selected) * len(specification["order_seeds"]),
        "aggregate_rows": len(all_aggregates),
        "aggregate_rows_sha256": sha256_file(aggregate_path),
        "units": unit_records,
    })
    outputs.extend([aggregate_path, manifest_path])
    return sorted({path.resolve() for path in outputs})


def _stage_controlled(context: StageContext) -> Iterable[Path]:
    output = context.run_dir / "controlled"
    config = context.config["synthetic"]
    pair_rows = run_pair_rank_phase_diagram(**dict(config["pair_rank_phase"]))
    orbit_kwargs = dict(config["real_embedding_orbit"])
    orbit_records = []
    index = read_json(context.run_dir / "feature_store_index.json")["records"]
    # One predeclared condition per dataset/encoder suffices because the orbit
    # is constructed from clean/reference geometry, not selected by outcomes.
    seen: set[tuple[str, str]] = set()
    for record in index:
        key = (record["dataset"], record["encoder"])
        if key in seen:
            continue
        seen.add(key)
        store = _load_indexed_store(record)
        reference = store.endpoint("reference")
        head = store.manifest["head"]
        weight = store.array(head["weight_role"])
        bias = store.array(head["bias_role"])
        orbit_records.append(run_real_embedding_orbit(
            np.asarray(reference.features), np.asarray(weight), bias=np.asarray(bias),
            seed=int(context.config["global_seed"]) + len(seen),
            dataset=str(record["dataset"]), encoder=str(record["encoder"]),
            input_manifest_sha256=sha256_file(Path(record["manifest"])),
            **orbit_kwargs,
        ))
    outputs = write_controlled_results(output, pair_rows, orbit_records)
    manifest = read_json(output / "manifest.json")
    if manifest.get("status") != "PASS":
        raise RuntimeError(
            f"controlled scientific gates failed: {manifest.get('scientific_gates')}"
        )
    return outputs


def _stage_inference(context: StageContext) -> Iterable[Path]:
    rows = [json.loads(line) for line in (context.run_dir / "real/rows.jsonl").read_text().splitlines()]
    result = hierarchical_dataset_inference(rows, context.config)
    output = context.run_dir / "inference.json"
    write_json_atomic(output, result)
    return [output]


def _stage_paper_outputs(context: StageContext) -> Iterable[Path]:
    return generate_paper_outputs(context.run_dir)


def _stage_verify(context: StageContext) -> Iterable[Path]:
    # This stage validates only immutable upstream evidence.  The campaign
    # manifest is deliberately published later, after StageRunner has changed
    # verify.json to PASS and closed logs/verify.log.  Otherwise the manifest
    # would contain hashes that are stale at the instant this function returns.
    stage_dir = context.run_dir / "stages"
    discovered = {path.stem for path in stage_dir.glob("*.json")}
    expected_while_running = {*UPSTREAM_STAGE_NAMES, "verify"}
    if discovered != expected_while_running:
        raise RuntimeError(
            "campaign stage set is incomplete or contains unexpected state: "
            f"expected={sorted(expected_while_running)}, found={sorted(discovered)}"
        )
    statuses: dict[str, str | None] = {}
    verified_outputs = 0
    for stage_name in UPSTREAM_STAGE_NAMES:
        state = read_json(stage_dir / f"{stage_name}.json")
        statuses[stage_name] = state.get("status")
        if state.get("status") != "PASS":
            raise RuntimeError(f"upstream stage did not pass: {stage_name}={state.get('status')}")
        outputs = state.get("outputs")
        if not isinstance(outputs, list):
            raise IntegrityError(f"stage {stage_name} has no declared output list")
        for record in outputs:
            if not isinstance(record, Mapping):
                raise IntegrityError(f"stage {stage_name} has a malformed output record")
            label = record.get("path")
            if not isinstance(label, str) or not label:
                raise IntegrityError(f"stage {stage_name} has an invalid output path")
            path = Path(label)
            if path.is_absolute():
                raise IntegrityError(f"campaign stage output is outside the run directory: {label}")
            path = (context.run_dir / path).resolve()
            try:
                path.relative_to(context.run_dir.resolve())
            except ValueError as error:
                raise IntegrityError(f"campaign stage output escapes the run directory: {label}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record.get("bytes", -1)):
                raise IntegrityError(f"stage output size changed after PASS: {label}")
            if sha256_file(path) != record.get("sha256"):
                raise IntegrityError(f"stage output hash changed after PASS: {label}")
            verified_outputs += 1
    required = [
        context.run_dir / "real/rows.jsonl", context.run_dir / "real/summary.json",
        context.run_dir / "real/manifest.json", context.run_dir / "inference.json",
        context.run_dir / "synthetic/manifest.json", context.run_dir / "controlled/manifest.json",
        context.run_dir / "streaming/manifest.json",
        context.run_dir / "paper_outputs/manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = context.run_dir / "verification_report.json"
    write_json_atomic(output, {
        "schema_version": "marginal_alignment.campaign.verification_report.v2",
        "status": "PASS",
        "campaign_id": context.config["campaign_id"],
        "upstream_stages": statuses,
        "verified_output_records": verified_outputs,
    })
    return [output]


def _publish_final_evidence(context: StageContext) -> tuple[Path, Path]:
    """Atomically publish the stable file inventory and its completion marker.

    This function must run only after ``verify`` has returned and StageRunner has
    atomically written its PASS state.  Both verify state and log are therefore
    closed, stable artifacts when their hashes enter the campaign manifest.
    ``CAMPAIGN_MANIFEST.json`` and ``DONE`` are excluded from the inventory to
    avoid self-reference; DONE separately binds the manifest hash and campaign
    identity.
    """

    stage_dir = context.run_dir / "stages"
    discovered = {path.stem for path in stage_dir.glob("*.json")}
    if discovered != set(ALL_STAGE_NAMES):
        raise RuntimeError(
            "cannot finalize an incomplete or ambiguous stage set: "
            f"expected={sorted(ALL_STAGE_NAMES)}, found={sorted(discovered)}"
        )
    statuses: dict[str, str | None] = {}
    for stage_name in ALL_STAGE_NAMES:
        state = read_json(stage_dir / f"{stage_name}.json")
        statuses[stage_name] = state.get("status")
    if any(status != "PASS" for status in statuses.values()):
        raise RuntimeError(f"cannot finalize a non-passing campaign: {statuses}")

    excluded = {"CAMPAIGN_MANIFEST.json", "DONE"}
    files = [
        {"path": path.relative_to(context.run_dir).as_posix(), "sha256": sha256_file(path),
         "bytes": path.stat().st_size}
        for path in sorted(context.run_dir.rglob("*"))
        if path.is_file() and path.relative_to(context.run_dir).as_posix() not in excluded
    ]
    manifest_path = context.run_dir / "CAMPAIGN_MANIFEST.json"
    write_json_atomic(manifest_path, {
        "schema_version": "marginal_alignment.campaign.campaign_manifest.v2",
        "status": "PASS", "campaign_id": context.config["campaign_id"],
        "config_sha256": context.config_hash, "source_sha256": context.source_hash,
        "runtime_sha256": context.runtime_hash,
        "external_input_sha256": context.external_input_hash,
        "stages": statuses, "files": files,
        "finalization_policy": (
            "file inventory captured only after verify PASS publication and log closure; "
            "CAMPAIGN_MANIFEST.json and DONE are not self-inventoried"
        ),
    })
    done_path = context.run_dir / "DONE"
    write_json_atomic(done_path, {
        "schema_version": "marginal_alignment.campaign.done.v2",
        "status": "PASS",
        "campaign_manifest_sha256": sha256_file(manifest_path),
        "campaign_id": context.config["campaign_id"],
    })
    return manifest_path, done_path


def run_campaign(config_path: str | Path, run_dir: str | Path, *,
                 from_stage: str | None = None, force_stage: str | None = None) -> Path:
    config, config_source, config_hash = load_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    lock_path = Path(
        config.get("lock_path", project_root / "protocol" / "experiment_lock.json")
    )
    if not lock_path.is_file():
        raise FileNotFoundError(f"experiment lock is required: {lock_path}")
    lock = read_json(lock_path)
    _validate_lock_status(lock)
    declared_config = (project_root / str(lock.get("config", ""))).resolve()
    if declared_config != config_source:
        raise IntegrityError(
            f"lock/config mismatch: lock declares {declared_config}, runner received {config_source}"
        )
    declared_raw_config_hash = lock.get("config_file_sha256")
    actual_raw_config_hash = sha256_file(config_source)
    if declared_raw_config_hash != actual_raw_config_hash:
        raise IntegrityError(
            "locked config bytes changed after prespecification: "
            f"{actual_raw_config_hash} != {declared_raw_config_hash}"
        )
    feature_build_path = (
        project_root / str(lock.get("feature_build_config", ""))
    ).resolve()
    if not feature_build_path.is_file():
        raise IntegrityError(
            f"lock does not resolve a feature-build config: {feature_build_path}"
        )
    declared_feature_build_hash = lock.get("feature_build_config_file_sha256")
    actual_feature_build_hash = sha256_file(feature_build_path)
    if actual_feature_build_hash != declared_feature_build_hash:
        raise IntegrityError(
            "locked feature-build config bytes changed after prespecification: "
            f"{actual_feature_build_hash} != {declared_feature_build_hash}"
        )
    feature_build_payload = read_json(feature_build_path)
    actual_feature_build_canonical_hash = sha256_json(feature_build_payload)
    declared_feature_build_canonical_hash = lock.get(
        "feature_build_config_canonical_sha256"
    )
    if actual_feature_build_canonical_hash != declared_feature_build_canonical_hash:
        raise IntegrityError(
            "locked feature-build canonical identity is inconsistent with its bytes: "
            f"{actual_feature_build_canonical_hash} != "
            f"{declared_feature_build_canonical_hash}"
        )
    source_hash, source_records = hash_tree(project_root, SOURCE_PATTERNS)
    output = Path(run_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # DONE is the sole authorization marker.  Remove it before touching a reused
    # run directory so a failed forced/restarted run cannot leave stale evidence
    # looking complete.
    (output / "DONE").unlink(missing_ok=True)
    write_json_atomic(output / "resolved_config.json", config)
    shutil.copy2(lock_path, output / "experiment_lock.json")
    write_json_atomic(output / "source_manifest.json", {
        "source_sha256": source_hash, "files": source_records,
    })
    runtime = environment_manifest()
    runtime_hash = sha256_json(runtime["runtime_identity"])
    write_json_atomic(output / "runtime_manifest.json", {
        "runtime_sha256": runtime_hash, "environment": runtime,
    })
    external_input_hash, external_inputs = _current_feature_store_identity(config)
    write_json_atomic(output / "external_input_manifest.json", {
        **external_inputs, "external_input_sha256": external_input_hash,
    })
    context = StageContext(
        project_root, output, config, config_hash, source_hash, runtime_hash,
        external_input_hash,
    )
    runner = StageRunner(context)
    stages = [
        ("preflight", _stage_preflight, []),
        ("tests", _stage_tests, ["preflight"]),
        ("gpu_smoke", _stage_gpu_smoke, ["tests"]),
        ("synthetic", _stage_synthetic, ["gpu_smoke"]),
        ("inventory", _stage_inventory, ["tests"]),
        ("validate_stores", _stage_validate_stores, ["inventory"]),
        ("real", _stage_real, ["validate_stores", "synthetic"]),
        ("streaming", _stage_streaming, ["validate_stores"]),
        ("controlled", _stage_controlled, ["validate_stores", "synthetic"]),
        ("inference", _stage_inference, ["real"]),
        ("paper_outputs", _stage_paper_outputs, ["streaming", "controlled", "inference"]),
        ("verify", _stage_verify, ["paper_outputs"]),
    ]
    # Always traverse earlier stages. StageRunner verifies their current
    # config/source/upstream fingerprint and every output hash before resume;
    # skipping them could certify stale evidence after a code/config change.
    force_flags = _stage_force_flags(
        [name for name, _, _ in stages],
        from_stage=from_stage, force_stage=force_stage,
    )
    for name, function, dependencies in stages:
        runner.run(
            name, function, dependencies=dependencies, force=force_flags[name]
        )
    _publish_final_evidence(context)
    print(f"COMPLETE: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--force-stage", default=None)
    arguments = parser.parse_args()
    run_campaign(arguments.config, arguments.run_dir,
                 from_stage=arguments.from_stage, force_stage=arguments.force_stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
