"""Strict paired-feature cache contract used by real and synthetic audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .linalg import as_samples, stable_hash_arrays


@dataclass(frozen=True)
class PairedFeatureCache:
    reference: np.ndarray
    target: np.ndarray
    labels: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    sample_ids: np.ndarray
    metadata: dict[str, Any]
    path: Path
    content_hash: str


REQUIRED_KEYS = {
    "reference_features",
    "target_features",
    "labels",
    "head_weight",
    "sample_ids",
}


def load_cache(path: str | Path) -> PairedFeatureCache:
    cache_path = Path(path).expanduser().resolve()
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_KEYS.difference(archive.files))
        if missing:
            raise ValueError(f"{cache_path} is missing keys: {missing}")
        reference = as_samples(archive["reference_features"], name="reference_features")
        target = as_samples(archive["target_features"], name="target_features")
        if reference.shape != target.shape:
            raise ValueError("paired reference and target features must have identical shapes")
        labels = np.asarray(archive["labels"], dtype=np.int64)
        sample_ids = np.asarray(archive["sample_ids"])
        if labels.shape != (len(reference),) or sample_ids.shape != (len(reference),):
            raise ValueError("labels and sample_ids must have one entry per paired row")
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique within a cache")
        weight = np.asarray(archive["head_weight"], dtype=np.float64)
        if weight.ndim != 2 or weight.shape[1] != reference.shape[1]:
            raise ValueError("head_weight must have shape [classes,d]")
        if np.any(labels < 0) or np.any(labels >= weight.shape[0]):
            raise ValueError("labels lie outside the frozen head's class range")
        bias = (
            np.asarray(archive["head_bias"], dtype=np.float64)
            if "head_bias" in archive.files
            else np.zeros(weight.shape[0], dtype=np.float64)
        )
        if bias.shape != (weight.shape[0],):
            raise ValueError("head_bias must have one entry per class")
        metadata_raw = str(archive["metadata_json"].item()) if "metadata_json" in archive.files else "{}"
        metadata = json.loads(metadata_raw)
    content_hash = stable_hash_arrays(reference, target, labels, weight, bias, sample_ids)
    return PairedFeatureCache(
        reference, target, labels, weight, bias, sample_ids, metadata, cache_path, content_hash
    )


def write_cache(
    path: str | Path,
    *,
    reference_features: np.ndarray,
    target_features: np.ndarray,
    labels: np.ndarray,
    head_weight: np.ndarray,
    head_bias: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = as_samples(reference_features, name="reference_features")
    target = as_samples(target_features, name="target_features")
    if reference.shape != target.shape:
        raise ValueError("reference and target must be paired and same-shaped")
    n = len(reference)
    ids = np.arange(n, dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    bias = (
        np.zeros(np.asarray(head_weight).shape[0], dtype=np.float64)
        if head_bias is None else np.asarray(head_bias, dtype=np.float64)
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        reference_features=reference.astype(np.float32),
        target_features=target.astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        head_weight=np.asarray(head_weight, dtype=np.float32),
        head_bias=bias.astype(np.float32),
        sample_ids=ids,
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    temporary.replace(destination)
    # Load once so malformed caches fail at creation, not hours into a sweep.
    load_cache(destination)
    return destination
