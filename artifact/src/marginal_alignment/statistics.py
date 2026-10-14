"""Paired, corruption-clustered inference for the real information ladder."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONTRASTS = (
    ("diagonal_vs_none", "diagonal", "none"),
    ("coral_vs_diagonal", "coral", "diagonal"),
    ("bures_vs_coral", "bures", "coral"),
    ("paired_procrustes_vs_shuffled", "paired_procrustes", "shuffled_pairs"),
    ("paired_covariance_oracle_vs_bures", "paired_oracle", "bures"),
    ("paired_readout_oracle_vs_bures", "paired_readout_oracle", "bures"),
    ("er_vs_none", "er", "none"),
)

METRICS = {
    "aligned_accuracy": "higher_is_better",
    "reference_prediction_agreement": "higher_is_better",
    "centered_logit_mse": "lower_is_better",
    "aligned_covariance_relative_error": "lower_is_better",
}


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _sign_flip_pvalue(cluster_effects: np.ndarray) -> float:
    """Exact two-sided paired randomization p-value over corruption clusters."""

    effects = np.asarray(cluster_effects, dtype=np.float64)
    observed = abs(float(effects.mean()))
    if len(effects) > 20:
        raise ValueError("exact sign-flip enumeration is intentionally capped at 20 clusters")
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        statistic = abs(float(np.mean(effects * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return float(exceed / total)


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(pvalues[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def analyze_real_rows(
    rows_path: str | Path,
    output_path: str | Path,
    *,
    calibration_size: int = 1024,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260811,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(rows_path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if int(row["calibration_size"]) == calibration_size]
    if not rows:
        raise ValueError(f"no rows at calibration size {calibration_size}")
    encoders = sorted({str(row["encoder"]) for row in rows})
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for encoder in encoders:
        encoder_rows = [row for row in rows if row["encoder"] == encoder]
        methods = {str(row["method"]) for row in encoder_rows}
        for contrast_name, candidate, control in CONTRASTS:
            if candidate not in methods or control not in methods:
                continue
            for metric, direction in METRICS.items():
                # Paired cells are corruption x split seed.  Severity is fixed
                # by the locked config but remains part of the key to fail
                # closed if mixed data are supplied.
                cells: dict[tuple[str, int | None, int], dict[str, float]] = defaultdict(dict)
                for row in encoder_rows:
                    if row["method"] in {candidate, control}:
                        key = (
                            str(row["corruption"]),
                            row.get("severity"),
                            int(row["split_seed"]),
                        )
                        cells[key][str(row["method"])] = float(row[metric])
                incomplete = [key for key, value in cells.items() if set(value) != {candidate, control}]
                if incomplete:
                    raise ValueError(f"unpaired cells for {contrast_name}/{metric}: {incomplete[:3]}")
                by_corruption: dict[tuple[str, int | None], list[float]] = defaultdict(list)
                sign = 1.0 if direction == "higher_is_better" else -1.0
                for (corruption, severity, _), value in cells.items():
                    by_corruption[(corruption, severity)].append(
                        sign * (value[candidate] - value[control])
                    )
                clusters = sorted(by_corruption)
                cluster_effects = np.asarray(
                    [np.mean(by_corruption[key]) for key in clusters], dtype=np.float64
                )
                if len(clusters) < 2:
                    raise ValueError("clustered inference needs at least two corruptions")
                boot = np.empty(bootstrap_replicates, dtype=np.float64)
                # Nested bootstrap: resample corruption clusters, then the
                # repeated split cells within each sampled cluster.
                for replicate in range(bootstrap_replicates):
                    picked = rng.integers(0, len(clusters), size=len(clusters))
                    cluster_draws = []
                    for index in picked:
                        values = np.asarray(by_corruption[clusters[int(index)]], dtype=np.float64)
                        within = rng.integers(0, len(values), size=len(values))
                        cluster_draws.append(float(values[within].mean()))
                    boot[replicate] = float(np.mean(cluster_draws))
                records.append({
                    "encoder": encoder,
                    "contrast": contrast_name,
                    "candidate": candidate,
                    "control": control,
                    "metric": metric,
                    "direction_normalization": direction,
                    "effect_positive_means_candidate_better": float(cluster_effects.mean()),
                    "effect_std_across_corruptions": float(cluster_effects.std(ddof=1)),
                    "nested_cluster_bootstrap_95_ci": _percentile_interval(boot),
                    "exact_two_sided_sign_flip_p_raw": _sign_flip_pvalue(cluster_effects),
                    "n_corruptions": len(clusters),
                    "split_repeats_per_corruption": sorted({len(value) for value in by_corruption.values()}),
                    "per_corruption_effects": {
                        f"{corruption}:s{severity}": float(effect)
                        for (corruption, severity), effect in zip(clusters, cluster_effects)
                    },
                })

    # Holm correction is applied within encoder and metric across all declared
    # contrasts; this family definition is fixed here, not selected post hoc.
    for encoder in encoders:
        for metric in METRICS:
            family = [row for row in records if row["encoder"] == encoder and row["metric"] == metric]
            adjusted = _holm([row["exact_two_sided_sign_flip_p_raw"] for row in family])
            for row, value in zip(family, adjusted):
                row["holm_adjusted_p"] = float(value)

    result = {
        "schema_version": "marginal_alignment.real_inference.v1",
        "rows_source": str(Path(rows_path).resolve()),
        "calibration_size": calibration_size,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "independent_cluster": "corruption_x_severity",
        "repeated_measurement": "calibration/evaluation split seed",
        "multiplicity_family": "within encoder x metric across all declared contrasts",
        "records": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return result
