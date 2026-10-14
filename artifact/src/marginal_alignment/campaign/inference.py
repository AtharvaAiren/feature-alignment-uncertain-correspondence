"""Generic, paired cluster inference for the locked evaluation campaign."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


PAIR_KEYS = (
    "suite", "dataset", "encoder", "corruption", "severity", "split_seed",
    "calibration_size", "pair_budget",
)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def paired_effect_rows(
    rows: Iterable[dict[str, Any]],
    *,
    left: str,
    right: str,
    metric: str,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return exactly paired left-minus-right effects; duplicates fail closed."""

    filters = filters or {}
    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("method") not in {left, right}:
            continue
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        if not _finite(row.get(metric)):
            continue
        key = tuple(row.get(name) for name in PAIR_KEYS)
        method = str(row["method"])
        if method in cells[key]:
            raise ValueError(f"duplicate result cell for {method}: {key}")
        cells[key][method] = row
    effects: list[dict[str, Any]] = []
    for key, methods in cells.items():
        if set(methods) != {left, right}:
            continue
        left_row, right_row = methods[left], methods[right]
        effects.append({
            **{name: value for name, value in zip(PAIR_KEYS, key)},
            "left": left,
            "right": right,
            "metric": metric,
            "left_value": float(left_row[metric]),
            "right_value": float(right_row[metric]),
            "effect": float(left_row[metric]) - float(right_row[metric]),
        })
    if not effects:
        raise ValueError(f"no complete paired cells for {left}-{right} on {metric}")
    return effects


def _cluster_means(
    effects: list[dict[str, Any]], cluster_keys: tuple[str, ...]
) -> tuple[list[tuple[Any, ...]], np.ndarray]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in effects:
        groups[tuple(row.get(key) for key in cluster_keys)].append(float(row["effect"]))
    keys = sorted(groups, key=lambda value: tuple(str(item) for item in value))
    values = np.asarray([np.mean(groups[key]) for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite cluster effect")
    return keys, values


def _sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> float:
    observed = abs(float(values.mean()))
    clusters = len(values)
    if clusters <= 20:
        exceed = 0
        total = 1 << clusters
        for mask in range(total):
            signs = np.asarray([1.0 if mask & (1 << index) else -1.0
                                for index in range(clusters)])
            exceed += abs(float(np.mean(signs * values))) >= observed - 1e-15
        return float(exceed / total)
    count = 0
    for _ in range(int(repetitions)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=clusters)
        count += abs(float(np.mean(signs * values))) >= observed - 1e-15
    return float((count + 1) / (repetitions + 1))


def infer_contrast(
    effects: list[dict[str, Any]],
    *,
    cluster_keys: tuple[str, ...] = ("dataset", "corruption"),
    bootstrap_repetitions: int = 20_000,
    confidence: float = 0.95,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Cluster-macro effect, percentile CI, and paired sign-flip test.

    Corruption clusters are a fixed benchmark panel.  Inference summarizes the
    panel under an exchangeability assumption; it is not universal corruption
    generalization. Split seeds and encoders are repeated measurements inside a
    cluster, not independent datasets.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    keys, values = _cluster_means(effects, cluster_keys)
    if len(values) < 2:
        raise ValueError("at least two clusters are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(bootstrap_repetitions), len(values)))
    bootstrap = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {
        "left": effects[0]["left"],
        "right": effects[0]["right"],
        "metric": effects[0]["metric"],
        "effect_definition": "left_minus_right",
        "effect": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "raw_p": _sign_flip_pvalue(values, rng, bootstrap_repetitions),
        "paired_cells": len(effects),
        "clusters": len(values),
        "cluster_keys": list(cluster_keys),
        "cluster_values": [
            {"cluster": list(key), "effect": float(value)} for key, value in zip(keys, values)
        ],
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "inference_scope": (
            "Fixed corruption panel under cluster exchangeability; split seeds, severities, "
            "and encoders are repeated measurements within clusters."
        ),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return [float(value) for value in adjusted]


def run_locked_inference(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    inference = config["inference"]
    primary = inference["primary_endpoint"]
    results = []
    for index, contrast in enumerate(inference["primary_contrasts"]):
        filters = {"suite": primary["suite"], "calibration_size": primary["calibration_size"]}
        # A correspondence contrast may live in its named suite instead of the
        # primary marginal suite; make that override explicit in the lock.
        filters.update(contrast.get("filters", {}))
        effects = paired_effect_rows(
            rows, left=contrast["left"], right=contrast["right"],
            metric=contrast["metric"], filters=filters,
        )
        results.append(infer_contrast(
            effects,
            cluster_keys=tuple(inference.get("cluster_keys", ["dataset", "corruption"])),
            bootstrap_repetitions=int(inference.get("bootstrap_repetitions", 20_000)),
            confidence=float(inference.get("confidence", 0.95)),
            seed=int(config.get("global_seed", 20260812)) + index,
        ))
    adjusted = holm_adjust([result["raw_p"] for result in results])
    for result, value in zip(results, adjusted):
        result["holm_p"] = value
    return {
        "schema_version": "marginal_alignment.campaign.inference.v2",
        "status": "PASS",
        "contrasts": results,
        "multiplicity": "Holm family-wise correction over locked primary contrasts",
    }


def hierarchical_dataset_inference(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Locked contrasts separately per benchmark, then Holm over the family."""

    inference = config["inference"]
    primary = inference["primary_endpoint"]
    datasets = sorted({row.get("dataset") for row in rows if row.get("dataset")})
    results = []
    for dataset_index, dataset in enumerate(datasets):
        for contrast_index, contrast in enumerate(inference["primary_contrasts"]):
            filters = {
                "suite": primary["suite"], "calibration_size": primary["calibration_size"],
                "dataset": dataset,
            }
            filters.update(contrast.get("filters", {}))
            effects = paired_effect_rows(
                rows, left=contrast["left"], right=contrast["right"],
                metric=contrast["metric"], filters=filters,
            )
            result = infer_contrast(
                effects, cluster_keys=("corruption",),
                bootstrap_repetitions=int(inference.get("bootstrap_repetitions", 20_000)),
                confidence=float(inference.get("confidence", 0.95)),
                seed=int(config.get("global_seed", 20260812))
                     + 1000 * dataset_index + contrast_index,
            )
            result["dataset"] = dataset
            results.append(result)
    adjusted = holm_adjust([result["raw_p"] for result in results])
    for result, value in zip(results, adjusted):
        result["holm_p"] = value
    return {
        "schema_version": "marginal_alignment.campaign.inference.v2",
        "status": "PASS", "contrasts": results,
        "multiplicity": "Holm over every locked dataset-specific primary contrast",
        "pooling_policy": "CIFAR-C and ImageNet-C are analyzed separately; no pseudo-pooling.",
    }


def macro_summary(rows: Iterable[dict[str, Any]], metrics: Iterable[str]) -> list[dict[str, Any]]:
    """Mean/SD across condition-macro split summaries with explicit estimand."""

    first: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("suite"), row.get("dataset"), row.get("encoder"), row.get("method"),
            row.get("calibration_size"), row.get("pair_budget"), row.get("pair_noise_rate"),
            row.get("split_seed"),
        )
        first[key].append(row)
    split_macros = []
    for key, values in first.items():
        record = {name: value for name, value in zip(
            ("suite", "dataset", "encoder", "method", "calibration_size", "pair_budget",
             "pair_noise_rate", "split_seed"), key
        )}
        for metric in metrics:
            vector = [float(row[metric]) for row in values if _finite(row.get(metric))]
            record[metric] = float(np.mean(vector)) if vector else float("nan")
        split_macros.append(record)
    second: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in split_macros:
        key = tuple(row[name] for name in (
            "suite", "dataset", "encoder", "method", "calibration_size", "pair_budget",
            "pair_noise_rate",
        ))
        second[key].append(row)
    summary = []
    for key, values in sorted(second.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {name: value for name, value in zip(
            ("suite", "dataset", "encoder", "method", "calibration_size", "pair_budget",
             "pair_noise_rate"), key
        )}
        record["split_repeats"] = len(values)
        record["variation_unit"] = "SD across split-level condition-macro means"
        for metric in metrics:
            vector = np.asarray([row[metric] for row in values if _finite(row.get(metric))])
            record[f"{metric}_mean"] = float(vector.mean()) if len(vector) else float("nan")
            record[f"{metric}_std"] = (
                float(vector.std(ddof=1)) if len(vector) > 1 else 0.0 if len(vector) else float("nan")
            )
        summary.append(record)
    return summary
