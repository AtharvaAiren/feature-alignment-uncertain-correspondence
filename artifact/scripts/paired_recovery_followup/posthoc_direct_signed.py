#!/usr/bin/env python3
"""Post-hoc direct signed-error calibration for the paired recovery follow-up.

This analysis consumes only the authenticated exported JSON records.  It does
not refit a correction or inspect an evaluation pairing before fixing a bound.
For each cell, contrast centers are differences of the already-frozen fit-role
method-error centers.  Positive contrast scales are the sums of the two
already-frozen fit-role method-error scales (an L1 propagation rule).  The same
29 calibration batches and the same one-sided 90% split-conformal rank used by
the prospective absolute-error comparator then calibrate one shared maximum
score across the three predeclared ordered contrasts.

The analysis is outcome-informed and post-hoc: the rule was specified in
response to reviewer feedback after the prospective study had completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONTRASTS = (
    ("diagonal_minus_identity", "diagonal", "identity"),
    ("bures_minus_identity", "bures", "identity"),
    ("bures_minus_diagonal", "bures", "diagonal"),
)
CALIBRATION_COUNT = 29
RANK_ONE_INDEXED = 27
DELTA = 0.10
TOLERANCE = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_export(root: Path) -> None:
    checksum_path = root / "RESULT_SHA256SUMS"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise RuntimeError(f"missing regular checksum file: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular exported result: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"checksum mismatch: {path}")


def quantile(values: list[float], rank_one_indexed: int) -> float:
    if len(values) != CALIBRATION_COUNT:
        raise RuntimeError(
            f"expected {CALIBRATION_COUNT} calibration scores, got {len(values)}"
        )
    return sorted(values)[rank_one_indexed - 1]


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    contrast_covered = sum(
        int(row["true_contrast"] <= row["upper_bound"] + TOLERANCE)
        for row in rows
    )
    certified = [row for row in rows if row["upper_bound"] < 0.0]
    incorrect = [row for row in certified if row["true_contrast"] >= 0.0]
    return {
        "rows": count,
        "contrast_coverage": contrast_covered,
        "negative_certificates": len(certified),
        "incorrect_strict_orderings": len(incorrect),
        "mean_upper_bound_slack": mean(
            [row["upper_bound"] - row["true_contrast"] for row in rows]
        ),
        "minimum_upper_bound_slack": min(
            row["upper_bound"] - row["true_contrast"] for row in rows
        ),
    }


def run(results_root: Path) -> dict[str, Any]:
    verify_export(results_root)
    prepared_paths = sorted((results_root / "prepared").glob("*.json"))
    if len(prepared_paths) != 8:
        raise RuntimeError(f"expected 8 prepared cells, got {len(prepared_paths)}")

    all_direct_rows: list[dict[str, Any]] = []
    all_physical_rows: list[dict[str, Any]] = []
    cell_results: list[dict[str, Any]] = []
    joint_direct = 0
    joint_physical = 0

    for prepared_path in prepared_paths:
        prepared = read_json(prepared_path)
        cell_id = str(prepared["cell_id"])
        standardizer = prepared["direct_baseline"]["standardizer"]
        method_centers = {
            str(key): float(value)
            for key, value in standardizer["centers"].items()
        }
        method_scales = {
            str(key): float(value)
            for key, value in standardizer["scales"].items()
        }
        if int(standardizer["fitting_batch_count"]) != 16:
            raise RuntimeError(f"unexpected fit-role batch count: {cell_id}")

        centers = {
            name: method_centers[method_a] - method_centers[method_c]
            for name, method_a, method_c in CONTRASTS
        }
        scales = {
            name: method_scales[method_a] + method_scales[method_c]
            for name, method_a, method_c in CONTRASTS
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise RuntimeError(f"invalid contrast scale: {cell_id}")

        calibration_rows = prepared["calibration"]["rows"]
        if len(calibration_rows) != CALIBRATION_COUNT:
            raise RuntimeError(f"unexpected calibration count: {cell_id}")
        scores: list[float] = []
        for row in calibration_rows:
            errors = {
                str(key): float(value)
                for key, value in row["direct_true_method_errors"].items()
            }
            scores.append(
                max(
                    (
                        (errors[method_a] - errors[method_c] - centers[name])
                        / scales[name]
                    )
                    for name, method_a, method_c in CONTRASTS
                )
            )
        threshold = quantile(scores, RANK_ONE_INDEXED)
        upper_bounds = {
            name: max(-1.0, min(1.0, centers[name] + scales[name] * threshold))
            for name, _, _ in CONTRASTS
        }

        cell_direct_rows: list[dict[str, Any]] = []
        cell_physical_rows: list[dict[str, Any]] = []
        cell_joint_direct = 0
        cell_joint_physical = 0
        unit_paths = sorted((results_root / "units" / cell_id).glob("batch_*.json"))
        if len(unit_paths) != 20:
            raise RuntimeError(f"expected 20 evaluation batches: {cell_id}")
        for unit_path in unit_paths:
            unit = read_json(unit_path)
            true_contrasts = {
                str(key): float(value)
                for key, value in unit["post_bound_true_correspondence_scoring"][
                    "true_signed_contrasts"
                ].items()
            }
            feasible = bool(
                unit["post_bound_true_correspondence_scoring"]["physical"]
                ["true_correspondence_feasible"]
            )
            direct_batch_covered = True
            physical_batch_covered = True
            for name, _, _ in CONTRASTS:
                true_value = true_contrasts[name]
                direct_upper = upper_bounds[name]
                direct_covered = true_value <= direct_upper + TOLERANCE
                direct_batch_covered &= direct_covered
                direct_record = {
                    "cell_id": cell_id,
                    "encoder": str(prepared["encoder"]),
                    "corruption": str(prepared["corruption"]),
                    "batch": int(unit["evaluation_batch_index"]),
                    "contrast": name,
                    "true_pair_feasible": feasible,
                    "true_contrast": true_value,
                    "upper_bound": direct_upper,
                    "covered": direct_covered,
                }
                cell_direct_rows.append(direct_record)
                all_direct_rows.append(direct_record)

                physical = unit["physical_geometry"]["contrasts"][name]
                physical_upper = float(physical["certificate"]["upper_bound"])
                physical_covered = true_value <= physical_upper + TOLERANCE
                recorded_covered = bool(
                    physical["upper_bound_covers_true_contrast"]
                )
                if physical_covered != recorded_covered:
                    raise RuntimeError(
                        f"recorded signed coverage disagrees with recomputation: {unit_path}"
                    )
                physical_batch_covered &= physical_covered
                physical_record = {
                    "cell_id": cell_id,
                    "batch": int(unit["evaluation_batch_index"]),
                    "contrast": name,
                    "true_pair_feasible": feasible,
                    "true_contrast": true_value,
                    "upper_bound": physical_upper,
                    "covered": physical_covered,
                    "numerical_tolerance": float(
                        physical["certificate"]["diagnostics"]["numerical_tolerance"]
                    ),
                }
                cell_physical_rows.append(physical_record)
                all_physical_rows.append(physical_record)
            cell_joint_direct += int(direct_batch_covered)
            cell_joint_physical += int(physical_batch_covered)

        joint_direct += cell_joint_direct
        joint_physical += cell_joint_physical
        cell_results.append(
            {
                "cell_id": cell_id,
                "encoder": prepared["encoder"],
                "corruption": prepared["corruption"],
                "repetition": int(prepared["repetition"]),
                "centers": centers,
                "scales": scales,
                "threshold": threshold,
                "upper_bounds": upper_bounds,
                "joint_coverage_batches": cell_joint_direct,
                "physical_joint_signed_coverage_batches": cell_joint_physical,
                "direct": summarize_rows(cell_direct_rows),
            }
        )

    direct_summary = summarize_rows(all_direct_rows)
    physical_summary = summarize_rows(all_physical_rows)
    physical_feasible = summarize_rows(
        [row for row in all_physical_rows if row["true_pair_feasible"]]
    )
    physical_infeasible = summarize_rows(
        [row for row in all_physical_rows if not row["true_pair_feasible"]]
    )
    negative_physical = [row for row in all_physical_rows if row["upper_bound"] < 0.0]
    closest_to_zero = max(negative_physical, key=lambda row: row["upper_bound"])
    ratio = abs(closest_to_zero["upper_bound"]) / closest_to_zero["numerical_tolerance"]

    panel_rows: list[dict[str, Any]] = []
    panels = (
        ("clip_gaussian_noise", "clip", "gaussian_noise"),
        ("clip_motion_blur", "clip", "motion_blur"),
        ("vitb16_pooled", "vitb16", None),
    )
    for panel, encoder, corruption in panels:
        for name, _, _ in CONTRASTS:
            rows = [
                row
                for row in all_direct_rows
                if row["contrast"] == name
                and row["encoder"] == encoder
                and (corruption is None or row["corruption"] == corruption)
            ]
            panel_rows.append(
                {
                    "panel": panel,
                    "contrast": name,
                    "rows": len(rows),
                    "mean_true_contrast": mean(
                        [row["true_contrast"] for row in rows]
                    ),
                    "contrast_coverage": sum(int(row["covered"]) for row in rows),
                    "negative_certificates": sum(
                        int(row["upper_bound"] < 0.0) for row in rows
                    ),
                    "incorrect_strict_orderings": sum(
                        int(
                            row["upper_bound"] < 0.0
                            and row["true_contrast"] >= 0.0
                        )
                        for row in rows
                    ),
                    "mean_upper_bound_slack": mean(
                        [row["upper_bound"] - row["true_contrast"] for row in rows]
                    ),
                }
            )

    return {
        "schema_version": "marginal_alignment.posthoc_direct_signed_calibration.v1",
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "OUTCOME_INFORMED_POST_HOC_COMPARATOR",
        "source_result_checksums_sha256": sha256_file(
            results_root / "RESULT_SHA256SUMS"
        ),
        "protocol": {
            "ordered_contrasts": [name for name, _, _ in CONTRASTS],
            "fit_role_center_rule": "difference_of_frozen_method_error_means",
            "fit_role_scale_rule": "sum_of_frozen_method_error_standard_deviations",
            "scale_interpretation": "positive_L1_normalization_not_a_variance_identity",
            "fit_role_batches": 16,
            "calibration_batches": CALIBRATION_COUNT,
            "delta": DELTA,
            "rank_one_indexed": RANK_ONE_INDEXED,
            "shared_score": "maximum_standardized_signed_error_across_three_contrasts",
            "evaluation_pairing_read_only_after_cell_bounds_frozen": True,
        },
        "direct_signed": {
            **direct_summary,
            "joint_coverage_batches": joint_direct,
            "evaluation_batches": 160,
            "panel_contrasts": panel_rows,
        },
        "physical_signed_revalidation": {
            **physical_summary,
            "joint_coverage_batches": joint_physical,
            "evaluation_batches": 160,
            "true_pair_feasible": physical_feasible,
            "true_pair_infeasible": physical_infeasible,
            "closest_negative_upper_bound": closest_to_zero,
            "closest_negative_magnitude_over_declared_numerical_tolerance": ratio,
        },
        "cells": cell_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.results_root.resolve(strict=True))
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
