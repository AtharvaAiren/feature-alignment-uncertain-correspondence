"""Reproducible information-ladder runner on paired feature caches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .er import online_er_head
from .linalg import population_mean_cov, relative_frobenius
from .metrics import accuracy, evaluate_alignment, logits, predictions
from .schema import PairedFeatureCache, load_cache
from .transforms import fit_transform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cache_paths(config: dict[str, Any], config_path: Path) -> list[Path]:
    import glob

    values: list[Path] = []
    for expression in config.get("cache_globs", []):
        path = Path(expression)
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        values.extend(Path(item).resolve() for item in glob.glob(str(path), recursive=True))
    values.extend(
        (Path(item) if Path(item).is_absolute() else config_path.parent / item).resolve()
        for item in config.get("cache_paths", [])
    )
    unique = sorted(set(values))
    if not unique:
        raise FileNotFoundError("configuration did not resolve any paired feature caches")
    return unique


def _split_indices(
    cache: PairedFeatureCache,
    *,
    seed: int,
    calibration_size: int,
    evaluation_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(cache.reference)
    if calibration_size < 2 or calibration_size >= n:
        raise ValueError(f"calibration_size must lie in [2,{n - 1}]")
    rng = np.random.default_rng(seed)
    if cache.metadata.get("empirical_multiset_orbit_complete"):
        worlds = int(cache.metadata["worlds"])
        if n % worlds or calibration_size % worlds:
            raise ValueError("orbit-complete splits require sizes divisible by the world count")
        if evaluation_size is not None and int(evaluation_size) % worlds:
            raise ValueError("orbit-complete evaluation size must be divisible by the world count")
        orbit_count = n // worlds
        orbit_order = rng.permutation(orbit_count)
        calibration_orbits = orbit_order[: calibration_size // worlds]
        remaining_orbits = orbit_order[calibration_size // worlds:]
        evaluation_orbits = (
            remaining_orbits
            if evaluation_size is None
            else remaining_orbits[: int(evaluation_size) // worlds]
        )
        calibration = np.concatenate(
            [calibration_orbits + world * orbit_count for world in range(worlds)]
        )
        evaluation = np.concatenate(
            [evaluation_orbits + world * orbit_count for world in range(worlds)]
        )
        rng.shuffle(calibration)
        rng.shuffle(evaluation)
    else:
        order = rng.permutation(n)
        calibration = order[:calibration_size]
        remaining = order[calibration_size:]
        evaluation = remaining if evaluation_size is None else remaining[: int(evaluation_size)]
    if len(evaluation) < 2:
        raise ValueError("evaluation split must contain at least two rows")
    if np.intersect1d(cache.sample_ids[calibration], cache.sample_ids[evaluation]).size:
        raise AssertionError("calibration/evaluation ID leakage")
    return calibration, evaluation


def _evaluate_er(
    cache: PairedFeatureCache,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    *,
    rng: np.random.Generator,
    parameters: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    weight, bias, diagnostics = online_er_head(
        cache.target[calibration], cache.labels[calibration], cache.weight, cache.bias,
        rng=rng,
        learning_rate=float(parameters.get("learning_rate", 1e-2)),
        buffer_size=int(parameters.get("buffer_size", 256)),
        replay_batch_size=int(parameters.get("replay_batch_size", 32)),
        weight_decay=float(parameters.get("weight_decay", 1e-4)),
    )
    reference = cache.reference[evaluation]
    target = cache.target[evaluation]
    labels = cache.labels[evaluation]
    mr, cr = population_mean_cov(reference)
    mt, ct = population_mean_cov(target)
    reference_logits = logits(reference, cache.weight, cache.bias)
    er_logits = logits(target, weight, bias)
    reference_predictions = np.argmax(reference_logits, axis=1)
    er_predictions = np.argmax(er_logits, axis=1)
    reference_acc = accuracy(reference, labels, cache.weight, cache.bias)
    source_acc = accuracy(target, labels, cache.weight, cache.bias)
    er_acc = float(np.mean(er_predictions == labels))
    denominator = reference_acc - source_acc
    recovery = (er_acc - source_acc) / denominator if abs(denominator) > 1e-12 else float("nan")
    return {
        "source_mean_error": float(np.linalg.norm(mt - mr)),
        "aligned_mean_error": float(np.linalg.norm(mt - mr)),
        "source_covariance_relative_error": relative_frobenius(ct, cr),
        "aligned_covariance_relative_error": relative_frobenius(ct, cr),
        "source_gaussian_w2_squared": float("nan"),
        "aligned_gaussian_w2_squared": float("nan"),
        "paired_feature_mse": float(np.mean(np.square(target - reference))),
        "centered_logit_mse": float(np.mean(np.square(er_logits - reference_logits))),
        "reference_prediction_agreement": float(np.mean(er_predictions == reference_predictions)),
        "source_prediction_agreement": float(
            np.mean(predictions(target, cache.weight, cache.bias) == reference_predictions)
        ),
        "reference_accuracy": reference_acc,
        "source_accuracy": source_acc,
        "aligned_accuracy": er_acc,
        "accuracy_recovery_fraction": float(recovery),
    }, diagnostics


def _method_spec(value: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, dict) or "name" not in value:
        raise ValueError("each method must be a name or an object with a name")
    parameters = {key: item for key, item in value.items() if key != "name"}
    return str(value["name"]), parameters


def run_config(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_paths = _resolve_cache_paths(config, config_file)
    methods = [_method_spec(item) for item in config["methods"]]
    split_seeds = [int(item) for item in config.get("split_seeds", [0, 1, 2, 3, 4])]
    calibration_sizes = [int(item) for item in config.get("calibration_sizes", [1024])]
    evaluation_size = config.get("evaluation_size")
    evaluation_size = None if evaluation_size is None else int(evaluation_size)
    shrinkage = float(config.get("shrinkage", 1e-4))
    relative_floor = float(config.get("relative_eigenvalue_floor", 1e-7))
    compute_gaussian_w2 = bool(config.get("compute_gaussian_w2", True))
    records: list[dict[str, Any]] = []
    start = time.time()

    for cache_index, path in enumerate(cache_paths):
        cache = load_cache(path)
        for split_seed in split_seeds:
            for calibration_size in calibration_sizes:
                calibration, evaluation = _split_indices(
                    cache, seed=split_seed, calibration_size=calibration_size,
                    evaluation_size=evaluation_size,
                )
                source_fit = cache.target[calibration]
                reference_fit = cache.reference[calibration]
                for method_index, (method, parameters) in enumerate(methods):
                    method_start = time.time()
                    method_rng = np.random.default_rng(
                        1_000_003 * split_seed + 10_007 * cache_index + 101 * method_index
                    )
                    if method.lower() == "er":
                        metrics, diagnostics = _evaluate_er(
                            cache, calibration, evaluation, rng=method_rng, parameters=parameters
                        )
                        information = "labeled_target_stream_plus_reservoir_replay"
                        changes_consumer = True
                        method_label = "er"
                    else:
                        rank = parameters.get("rank")
                        fitted = fit_transform(
                            method, source_fit, reference_fit, rng=method_rng,
                            rank=None if rank is None else int(rank),
                            shrinkage=float(parameters.get("shrinkage", shrinkage)),
                            relative_floor=float(parameters.get("relative_floor", relative_floor)),
                            readout_weight=cache.weight,
                        )
                        aligned = fitted.apply(cache.target[evaluation])
                        metrics = evaluate_alignment(
                            cache.reference[evaluation], cache.target[evaluation], aligned,
                            labels=cache.labels[evaluation], weight=cache.weight, bias=cache.bias,
                            compute_gaussian_w2=compute_gaussian_w2,
                        )
                        diagnostics = fitted.diagnostics
                        information = fitted.fit_information
                        changes_consumer = False
                        method_label = fitted.method
                    record = {
                        "schema_version": "marginal_alignment.row.v1",
                        "cache_path": str(path),
                        "cache_hash": cache.content_hash,
                        "dataset": cache.metadata.get("dataset", path.stem),
                        "encoder": cache.metadata.get("encoder", "unknown"),
                        "corruption": cache.metadata.get("corruption", cache.metadata.get("hidden_world")),
                        "severity": cache.metadata.get("severity"),
                        "data_seed": cache.metadata.get("seed"),
                        "hidden_world": cache.metadata.get("hidden_world"),
                        "worlds": cache.metadata.get("worlds"),
                        "theoretical_bayes_ceiling": cache.metadata.get("theoretical_bayes_ceiling"),
                        "method": method_label,
                        "requested_method": method,
                        "method_parameters": parameters,
                        "fit_information": information,
                        "changes_frozen_consumer": changes_consumer,
                        "split_seed": split_seed,
                        "calibration_size": len(calibration),
                        "evaluation_size": len(evaluation),
                        "calibration_id_hash": hashlib.sha256(
                            np.ascontiguousarray(cache.sample_ids[calibration]).view(np.uint8)
                        ).hexdigest(),
                        "evaluation_id_hash": hashlib.sha256(
                            np.ascontiguousarray(cache.sample_ids[evaluation]).view(np.uint8)
                        ).hexdigest(),
                        "runtime_seconds": time.time() - method_start,
                        "diagnostics": diagnostics,
                        **metrics,
                    }
                    records.append(record)
                    print(
                        f"[{len(records)}] {path.stem} seed={split_seed} n={calibration_size} "
                        f"{method}: acc={metrics['aligned_accuracy']:.4f} "
                        f"cov={metrics['aligned_covariance_relative_error']:.4f}",
                        flush=True,
                    )

    raw_path = output / "rows.jsonl"
    temporary = raw_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
    temporary.replace(raw_path)
    summary = summarize(records)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    _write_summary_csv(output / "summary.csv", summary["groups"])
    copied_config = output / "config.json"
    copied_config.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": "marginal_alignment.run.v1",
        "config_source": str(config_file),
        "config_sha256": _sha256(copied_config),
        "rows_sha256": _sha256(raw_path),
        "summary_sha256": _sha256(output / "summary.json"),
        "cache_files": [{"path": str(path), "sha256": _sha256(path)} for path in cache_paths],
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "elapsed_seconds": time.time() - start,
        "record_count": len(records),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"summary": summary, "manifest": manifest, "output_dir": str(output)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "aligned_accuracy", "reference_accuracy", "source_accuracy",
        "accuracy_recovery_fraction", "reference_prediction_agreement",
        "aligned_mean_error", "aligned_covariance_relative_error",
        "aligned_gaussian_w2_squared", "paired_feature_mse", "centered_logit_mse",
    ]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["dataset"], record["encoder"], record["method"], record["calibration_size"]
        )
        buckets[key].append(record)
    groups: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
        group: dict[str, Any] = {
            "dataset": key[0], "encoder": key[1], "method": key[2],
            "calibration_size": key[3], "n_rows": len(rows),
        }
        ceilings = [row.get("theoretical_bayes_ceiling") for row in rows]
        ceilings = [float(item) for item in ceilings if item is not None]
        if ceilings:
            group["theoretical_bayes_ceiling"] = float(np.mean(ceilings))
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            group[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            group[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        # In an ambiguity-set experiment, hidden worlds are not independent
        # replicates.  Average/worst-case over all worlds *within each data
        # seed first*, then summarize across seeds.  The raw standard deviation
        # above remains useful for seeing world heterogeneity, but must not be
        # reported as seed uncertainty.
        if all(row.get("data_seed") is not None and row.get("hidden_world") is not None for row in rows):
            ambiguity_units: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                ambiguity_units[(int(row["data_seed"]), int(row["split_seed"]))].append(row)
            complete = True
            uniform_values: list[float] = []
            worst_values: list[float] = []
            uniform_task_values: list[float] = []
            worst_task_values: list[float] = []
            for unit_rows in ambiguity_units.values():
                expected = int(unit_rows[0].get("worlds") or 0)
                observed = {int(row["hidden_world"]) for row in unit_rows}
                if expected < 1 or observed != set(range(expected)):
                    complete = False
                    break
                accuracies = np.asarray(
                    [row["aligned_accuracy"] for row in unit_rows], dtype=np.float64
                )
                task_agreements = np.asarray(
                    [row["reference_prediction_agreement"] for row in unit_rows],
                    dtype=np.float64,
                )
                uniform_values.append(float(accuracies.mean()))
                worst_values.append(float(accuracies.min()))
                uniform_task_values.append(float(task_agreements.mean()))
                worst_task_values.append(float(task_agreements.min()))
            if complete:
                uniform = np.asarray(uniform_values, dtype=np.float64)
                worst = np.asarray(worst_values, dtype=np.float64)
                uniform_task = np.asarray(uniform_task_values, dtype=np.float64)
                worst_task = np.asarray(worst_task_values, dtype=np.float64)
                group["ambiguity_uniform_accuracy_mean"] = float(uniform.mean())
                group["ambiguity_uniform_accuracy_std"] = (
                    float(uniform.std(ddof=1)) if len(uniform) > 1 else 0.0
                )
                group["ambiguity_worst_accuracy_mean"] = float(worst.mean())
                group["ambiguity_worst_accuracy_std"] = (
                    float(worst.std(ddof=1)) if len(worst) > 1 else 0.0
                )
                # These are the quantities matched to the theory: agreement
                # with the frozen reference-frame consumer, not external-label
                # accuracy (the two coincide only when the clean consumer is
                # perfect, as in the orbit-complete synthetic construction).
                group["ambiguity_uniform_task_agreement_mean"] = float(
                    uniform_task.mean()
                )
                group["ambiguity_uniform_task_agreement_std"] = (
                    float(uniform_task.std(ddof=1)) if len(uniform_task) > 1 else 0.0
                )
                group["ambiguity_worst_task_agreement_mean"] = float(worst_task.mean())
                group["ambiguity_worst_task_agreement_std"] = (
                    float(worst_task.std(ddof=1)) if len(worst_task) > 1 else 0.0
                )
        groups.append(group)
    return {"schema_version": "marginal_alignment.summary.v1", "groups": groups}


def _write_summary_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in groups for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(groups)
