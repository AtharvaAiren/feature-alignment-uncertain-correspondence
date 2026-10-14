"""Strict upstream feature-cache and frozen-readout preparation utilities.

This module is deliberately independent of Torch/model libraries.  It owns the
parts of feature preparation that must be identical whether feature matrices
are reused from a sealed server cache or rebuilt from raw images:

* canonical condition block indices;
* identity sidecars and clean-only head-fit identities;
* the exact linear-probe realization; and
* checkpoint/preprocessing provenance validation.

No function infers a checkpoint hash from a model name, or correspondence from
labels.  Missing evidence is a hard error.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import IntegrityError, sha256_file, sha256_json
from .cache_import import CIFAR100_C_CORRUPTIONS, IMAGENET_C_CORRUPTIONS


FEATURE_BUILD_SCHEMA = "marginal_alignment.campaign.feature_build.v1"
PREPARED_SOURCE_SCHEMA = "marginal_alignment.campaign.prepared_legacy_source.v1"
MODEL_EVIDENCE_SCHEMA = "marginal_alignment.campaign.model_evidence.v1"
HEAD_REALIZATION_SCHEMA = "marginal_alignment.campaign.readout_realization.v1"


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"expected a JSON object: {source}")
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_npy(path: str | Path, value: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return destination


def load_feature_build_config(path: str | Path) -> dict[str, Any]:
    config = read_json_object(path)
    if config.get("schema_version") != FEATURE_BUILD_SCHEMA:
        raise IntegrityError(f"not a {FEATURE_BUILD_SCHEMA} config")
    encoders = config.get("encoders")
    datasets = config.get("datasets")
    realization = config.get("readout_realization")
    if not isinstance(encoders, Mapping) or set(encoders) != {"dinov3", "clip", "vitb16"}:
        raise IntegrityError("feature config must define exactly dinov3, clip, and vitb16")
    if not isinstance(datasets, Mapping) or set(datasets) != {"cifar100_c", "imagenet_c"}:
        raise IntegrityError("feature config must define CIFAR-100-C and ImageNet-C")
    if not isinstance(realization, Mapping):
        raise IntegrityError("feature config lacks a structured readout_realization")
    for name, expected in (("cifar100_c", 10_000), ("imagenet_c", 50_000)):
        spec = datasets[name]
        if not isinstance(spec, Mapping) or int(spec.get("rows_per_condition", -1)) != expected:
            raise IntegrityError(f"{name} has an invalid rows_per_condition")
        if not isinstance(spec.get("head_protocol"), Mapping):
            raise IntegrityError(f"{name} lacks a structured head_protocol")
    return config


def is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_model_evidence(
    evidence: Mapping[str, Any], *, encoder: str, encoder_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate reusable-cache evidence without synthesizing absent identity."""

    if evidence.get("schema_version") != MODEL_EVIDENCE_SCHEMA:
        raise IntegrityError(f"model evidence is not {MODEL_EVIDENCE_SCHEMA}")
    required = {
        "encoder_id", "model_id", "checkpoint_sha256", "preprocessing_sha256",
        "architecture_config_sha256", "feature_layer", "pooling", "expected_dimension",
    }
    missing = sorted(key for key in required if evidence.get(key) in (None, ""))
    if missing:
        raise IntegrityError(f"model evidence is missing fields: {missing}")
    if evidence["encoder_id"] != encoder:
        raise IntegrityError(f"model evidence encoder {evidence['encoder_id']!r} != {encoder!r}")
    if evidence["model_id"] != encoder_spec["model_id"]:
        raise IntegrityError("model evidence model_id differs from the locked build config")
    for key in ("checkpoint_sha256", "preprocessing_sha256", "architecture_config_sha256"):
        if not is_sha256(evidence[key]):
            raise IntegrityError(f"model evidence {key} is not a SHA-256")
    for key in ("feature_layer", "pooling"):
        if evidence[key] != encoder_spec[key]:
            raise IntegrityError(f"model evidence {key} differs from the locked build config")
    if int(evidence["expected_dimension"]) != int(encoder_spec["expected_dimension"]):
        raise IntegrityError("model evidence dimension differs from the locked build config")
    return dict(evidence)


def checkpoint_tree_fingerprint(root: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash local checkpoint bytes, including all SafeTensors shards.

    The returned digest hashes the sorted ``(relative path, file SHA)`` list.
    It is never a hash of a model identifier or mutable cache directory name.
    """

    source = Path(root).expanduser().resolve()
    if source.is_file():
        records = [{"path": source.name, "sha256": sha256_file(source), "bytes": source.stat().st_size}]
    elif source.is_dir():
        # Transformers behavior depends on config JSON as well as tensor bytes.
        # Bind all snapshot JSON so a mutated architecture/processor config
        # cannot reuse the same weight-only fingerprint.
        suffixes = {".safetensors", ".bin", ".pth", ".pt", ".npz", ".json"}
        candidates = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
        if not candidates:
            raise IntegrityError(f"no checkpoint weight files under {source}")
        records = [
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in candidates
        ]
    else:
        raise FileNotFoundError(source)
    return sha256_json(records), records


def canonical_block_index(
    dataset: str,
    *,
    corruptions: Sequence[str] | None = None,
    severities: Sequence[int] = (1, 3, 5),
    rows_per_condition: int | None = None,
) -> list[dict[str, Any]]:
    """Build a declared block index; this does not itself prove row identity."""

    if dataset == "cifar100_c":
        locked = CIFAR100_C_CORRUPTIONS
        expected_rows = 10_000
    elif dataset == "imagenet_c":
        locked = IMAGENET_C_CORRUPTIONS
        expected_rows = 50_000
    else:
        raise ValueError(f"unsupported corruption dataset: {dataset}")
    selected = tuple(locked if corruptions is None else corruptions)
    unknown = sorted(set(selected).difference(locked))
    if unknown or len(set(selected)) != len(selected):
        raise IntegrityError(f"invalid or duplicate corruptions: {unknown}")
    rows = expected_rows if rows_per_condition is None else int(rows_per_condition)
    if rows != expected_rows:
        raise IntegrityError(f"{dataset} needs {expected_rows} rows per condition")
    levels = tuple(int(value) for value in severities)
    if not levels or len(set(levels)) != len(levels) or any(value not in range(1, 6) for value in levels):
        raise IntegrityError("severities must be distinct values in 1..5")
    result: list[dict[str, Any]] = []
    start = 0
    for corruption in selected:
        for severity in levels:
            result.append({
                "corruption": str(corruption), "severity": severity,
                "start": start, "stop": start + rows,
            })
            start += rows
    return result


def validate_declared_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    total_rows: int,
    require_conditions: Sequence[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Validate explicit source-cache block order, without assuming alphabetical order."""

    rows = 10_000 if dataset == "cifar100_c" else 50_000 if dataset == "imagenet_c" else None
    if rows is None:
        raise ValueError(dataset)
    normalized = []
    previous_stop = 0
    keys: set[tuple[str, int]] = set()
    for record in blocks:
        if not {"corruption", "severity", "start", "stop"}.issubset(record):
            raise IntegrityError("block index has a malformed record")
        item = {
            "corruption": str(record["corruption"]),
            "severity": int(record["severity"]),
            "start": int(record["start"]),
            "stop": int(record["stop"]),
        }
        key = (item["corruption"], item["severity"])
        if key in keys:
            raise IntegrityError(f"duplicate condition block: {key}")
        if item["start"] != previous_stop or item["stop"] - item["start"] != rows:
            raise IntegrityError(f"noncontiguous or wrong-size condition block: {key}")
        if item["severity"] not in range(1, 6):
            raise IntegrityError(f"invalid severity in condition block: {key}")
        keys.add(key)
        normalized.append(item)
        previous_stop = item["stop"]
    if previous_stop != int(total_rows):
        raise IntegrityError(f"block index covers {previous_stop} rows, feature array has {total_rows}")
    if require_conditions is not None:
        missing = sorted(set(require_conditions).difference(keys))
        if missing:
            raise IntegrityError(f"source block index is missing locked conditions: {missing[:5]}")
    return normalized


def deterministic_stratified_positions(
    labels: np.ndarray, *, shots_per_class: int, seed: int
) -> np.ndarray:
    """Select a reproducible, exactly balanced subset without reading target labels."""

    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise IntegrityError("head-fit labels must be a rank-one integer array")
    classes = np.unique(values)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise IntegrityError("head-fit labels must be contiguous 0..K-1")
    seed_sequence = np.random.SeedSequence(int(seed)).spawn(len(classes))
    selected: list[np.ndarray] = []
    for class_index, class_seed in zip(classes, seed_sequence):
        candidates = np.flatnonzero(values == class_index)
        if len(candidates) < int(shots_per_class):
            raise IntegrityError(
                f"class {int(class_index)} has {len(candidates)} rows, needs {shots_per_class}"
            )
        order = np.random.default_rng(class_seed).permutation(candidates)
        selected.append(np.sort(order[: int(shots_per_class)]))
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def head_fit_positions(labels: np.ndarray, protocol: Mapping[str, Any]) -> np.ndarray:
    kind = str(protocol.get("kind", ""))
    if kind == "clean_linear_probe":
        maximum = int(protocol.get("max_train_rows", 0))
        if maximum <= 0 or len(labels) > maximum:
            raise IntegrityError(
                f"clean probe has {len(labels)} rows but locked maximum is {maximum}"
            )
        return np.arange(len(labels), dtype=np.int64)
    if kind == "stratified_clean_linear_probe":
        return deterministic_stratified_positions(
            labels,
            shots_per_class=int(protocol["shots_per_class"]),
            seed=int(protocol["seed"]),
        )
    raise IntegrityError(f"unsupported locked head protocol: {kind!r}")


def fit_frozen_readout(
    features: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    *,
    realization: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the locked multinomial probe and fold standardization into its head."""

    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - exercised on the GPU server
        raise RuntimeError("scikit-learn is required to fit the frozen readout") from error

    x_all = np.asarray(features)
    y_all = np.asarray(labels)
    indices = np.asarray(positions, dtype=np.int64)
    if x_all.ndim != 2 or y_all.shape != (len(x_all),):
        raise IntegrityError("clean features/labels have inconsistent shapes")
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise IntegrityError("head-fit positions must be a unique rank-one array")
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(x_all):
        raise IntegrityError("head-fit positions are empty or outside clean features")
    x = np.asarray(x_all[indices], dtype=np.float64)
    y = np.asarray(y_all[indices], dtype=np.int64)
    if not np.isfinite(x).all():
        raise IntegrityError("clean head-fit features contain nonfinite values")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    constant = scale <= np.finfo(np.float64).eps
    scale[constant] = 1.0
    standardized = (x - mean) / scale
    solver = str(realization.get("solver", "saga"))
    if solver not in {"saga", "lbfgs"}:
        raise IntegrityError(f"unsupported locked logistic-regression solver: {solver!r}")
    requested = {
        "solver": solver,
        "C": float(realization["C"]),
        "fit_intercept": bool(realization["fit_intercept"]),
        "max_iter": int(realization["max_iter"]),
        "tol": float(realization["tol"]),
    }
    # L-BFGS is deterministic and does not consume random_state. SAGA retains
    # the historical seeded behavior for backwards-compatible fixtures.
    if solver == "saga":
        requested["random_state"] = int(seed)
    estimator = LogisticRegression(**requested)
    estimator.fit(standardized, y)
    iterations = np.asarray(estimator.n_iter_, dtype=np.int64)
    if np.any(iterations >= requested["max_iter"]):
        raise IntegrityError(
            f"{solver} reached max_iter; refusing a nonconverged frozen readout "
            f"(n_iter={iterations.tolist()}, max_iter={requested['max_iter']})"
        )
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    intercept = np.asarray(estimator.intercept_, dtype=np.float64)
    if coefficients.shape[0] == 1 and len(estimator.classes_) == 2:
        # sklearn's binary convention stores one signed score. Expand it to
        # two logits so the cache contract always has [classes, dimension].
        coefficients = np.concatenate([-coefficients, coefficients], axis=0) / 2.0
        intercept = np.concatenate([-intercept, intercept], axis=0) / 2.0
    weight = coefficients / scale[None, :]
    bias = intercept - np.sum(coefficients * (mean / scale)[None, :], axis=1)
    prediction = np.empty(len(x_all), dtype=np.int64)
    for start in range(0, len(x_all), 4096):
        stop = min(start + 4096, len(x_all))
        logits = np.asarray(x_all[start:stop], dtype=np.float64) @ weight.T + bias
        prediction[start:stop] = np.argmax(logits, axis=1)
    fit_prediction = prediction[indices]
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise IntegrityError("fitted head contains nonfinite parameters")
    report = {
        "schema_version": HEAD_REALIZATION_SCHEMA,
        "status": "PASS",
        "library": "scikit-learn",
        "library_version": str(sklearn.__version__),
        "estimator": "sklearn.linear_model.LogisticRegression",
        "parameters": requested,
        "classes": [int(value) for value in estimator.classes_],
        "n_iter": [int(value) for value in iterations],
        "fit_rows": int(len(indices)),
        "dimension": int(x_all.shape[1]),
        "constant_feature_count": int(np.sum(constant)),
        "fit_accuracy": float(np.mean(fit_prediction == y_all[indices])),
        "clean_endpoint_accuracy": float(np.mean(prediction == y_all)),
        "standardization_folded_into_weight_bias": True,
    }
    return weight.astype(np.float32), bias.astype(np.float32), report


def build_model_provenance(
    *,
    evidence: Mapping[str, Any],
    head_protocol: Mapping[str, Any],
    head_fit_ids_file: str,
    head_fit_ids_sha256: str,
    head_fit_id_namespace: str,
    reference_id_namespace: str,
    reference_split: str,
    head_fit_split: str,
    readout_report_file: str,
    feature_build_config_sha256: str,
) -> dict[str, Any]:
    if not is_sha256(head_fit_ids_sha256) or not is_sha256(feature_build_config_sha256):
        raise IntegrityError("head-fit/config provenance requires real SHA-256 values")
    return {
        "encoder_id": evidence["encoder_id"],
        "model_id": evidence["model_id"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "architecture_config_sha256": evidence["architecture_config_sha256"],
        "preprocessing_sha256": evidence["preprocessing_sha256"],
        "feature_layer": evidence["feature_layer"],
        "pooling": evidence["pooling"],
        "head_protocol": dict(head_protocol),
        "head_fit_ids_sha256": head_fit_ids_sha256,
        "head_fit_ids_file": str(head_fit_ids_file),
        "head_fit_id_namespace": str(head_fit_id_namespace),
        "reference_id_namespace": str(reference_id_namespace),
        "reference_split": str(reference_split),
        "head_fit_split": str(head_fit_split),
        "readout_realization_file": str(readout_report_file),
        "feature_build_config_sha256": feature_build_config_sha256,
    }


def array_payload_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()
