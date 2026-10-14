#!/usr/bin/env python3
"""Post-hoc cost-scale audit for the matched Play-3 correspondence panel.

This audit reuses the exact 4,096-calibration/1,024-witness splits and fitted
geometry from the completed ImageNet-Sketch versus ImageNet-C positive-control
panel.  It computes the exact expectation and variance of total cost under a
uniform random bijection, then places the locked 1/5/10 percent budgets and the
ImageNet-C identity pairing on the random-minus-minimum scale.

No encoder forward pass, labels, or outcomes are used to fit a map.  Positive-
control identities are revealed only after reproducing the frozen split and
geometry, exactly as in the original panel.  The audit is explicitly post hoc.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import load_feature_store
from marginal_alignment.iclr_extension.ceiling import (
    CorrespondenceCeilingProtocol,
    _canonical_rows,
    fit_witness_geometry,
)
from marginal_alignment.iclr_extension.cost_scale import (
    assignment_cost,
    summarize_cost_scale,
    uniform_random_bijection_cost_moments,
)
from marginal_alignment.iclr_extension.natural_shift import (
    _content_split,
    _independent_pair_permutation,
    _pair_hash_order,
    load_unpaired_store_inputs,
)


SCHEMA_VERSION = "marginal_alignment.iclr_extension.cost_scale_audit.v1"
EXPECTED_BUDGETS = (0.01, 0.05, 0.10)
EXPECTED_SEEDS = (201, 202, 203)
EXPECTED_ENCODERS = ("clip", "dinov3")
EXPECTED_POSITIVE_CORRUPTIONS = (
    "brightness",
    "defocus_blur",
    "fog",
    "gaussian_noise",
    "jpeg_compression",
    "motion_blur",
)
EXPECTED_NATURAL_UNITS = 18
EXPECTED_POSITIVE_UNITS = 108
EXPECTED_CONDITION_GEOMETRIES = 42
EXPECTED_MARKER_SCHEMA = "marginal_alignment.iclr_extension.rtx_7day_complete.v1"
EXPECTED_NATURAL_SCHEMA = "marginal_alignment.iclr_extension.natural_shift_ceiling.v2"
EXPECTED_POSITIVE_SCHEMA = "marginal_alignment.iclr_extension.positive_control.v2"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise IntegrityError(f"{name} is not finite")
    return result


def _close(observed: float, expected: float, *, name: str) -> None:
    tolerance = (
        4096.0 * np.finfo(np.float64).eps * max(abs(observed), abs(expected), 1.0)
    )
    if abs(observed - expected) > tolerance:
        raise IntegrityError(
            f"{name} differs: observed={observed:.17g}, expected={expected:.17g}, "
            f"tolerance={tolerance:.17g}"
        )


def _protocol(record: Mapping[str, Any]) -> CorrespondenceCeilingProtocol:
    raw = record["unit_plan"]["protocol"]
    return CorrespondenceCeilingProtocol(
        covariance_estimator=str(raw["covariance_estimator"]),
        covariance_relative_floor=float(raw["covariance_relative_floor"]),
        cost_budget_relative=float(raw["cost_budget_relative"]),
        maximum_witness_rows=int(raw["maximum_witness_rows"]),
        swap_neighbors=int(raw["swap_neighbors"]),
        swap_neighbor_sensitivity=tuple(
            int(value) for value in raw["swap_neighbor_sensitivity"]
        ),
        assignment_tie_relative_tolerance=float(
            raw["assignment_tie_relative_tolerance"]
        ),
    )


def _canonical_geometry(
    geometry: Any,
    target_witness: np.ndarray,
    reference_witness: np.ndarray,
    anchor: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target, target_order, target_sha = _canonical_rows(
        target_witness,
        name="cost_scale_target",
        salt=b"play3-target-witness-v1",
    )
    reference, reference_order, reference_sha = _canonical_rows(
        reference_witness,
        name="cost_scale_reference",
        salt=b"play3-reference-witness-v1",
    )
    diagnostics = anchor["ceiling_witness"]["diagnostics"]
    if target_sha != diagnostics["target_content_sha256"]:
        raise IntegrityError("target witness content differs from sealed result")
    if reference_sha != diagnostics["reference_content_sha256"]:
        raise IntegrityError("reference witness content differs from sealed result")
    if (
        geometry.calibration_diagnostics["full_geometry_sha256"]
        != diagnostics["calibration"]["full_geometry_sha256"]
    ):
        raise IntegrityError("reproduced fitted geometry differs from sealed result")
    mapped_target = geometry.bures_map.apply(target)
    target_whitened = (
        mapped_target - geometry.reference_mean
    ) @ geometry.reference_inverse_sqrt
    reference_whitened = (
        reference - geometry.reference_mean
    ) @ geometry.reference_inverse_sqrt
    return target_whitened, reference_whitened, target_order, reference_order


def _natural_geometry(
    store_path: str,
    anchor: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float | None]:
    store = load_feature_store(store_path, verify_hashes=True)
    inputs = load_unpaired_store_inputs(store)
    calibration = int(anchor["calibration_rows"])
    witness_rows = int(anchor["witness_rows"])
    requested = calibration + witness_rows
    seed = int(anchor["split_seed"])
    reference, _, reference_split_sha = _content_split(
        inputs.reference_features,
        rows=requested,
        salt=f"play3-reference-split-v1:{seed}".encode("ascii"),
    )
    target, _, target_split_sha = _content_split(
        inputs.target_features,
        rows=requested,
        salt=f"play3-target-split-v1:{seed}".encode("ascii"),
    )
    if reference_split_sha != anchor["reference_split_content_sha256"]:
        raise IntegrityError("natural reference split differs from sealed result")
    if target_split_sha != anchor["target_split_content_sha256"]:
        raise IntegrityError("natural target split differs from sealed result")
    geometry = fit_witness_geometry(
        target[:calibration],
        reference[:calibration],
        inputs.readout_weight,
        inputs.readout_bias,
        protocol=_protocol(anchor),
    )
    target_whitened, reference_whitened, _, _ = _canonical_geometry(
        geometry, target[calibration:], reference[calibration:], anchor
    )
    return target_whitened, reference_whitened, None


def _positive_geometry(
    store_path: str,
    anchor: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    store = load_feature_store(store_path, verify_hashes=True)
    if store.dataset != "imagenet_c":
        raise IntegrityError("positive-control store is not ImageNet-C")
    pairing = store.pairing()
    paired_reference, paired_target = pairing.validated()
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    reference_lookup = {
        value: index for index, value in enumerate(reference.ids.tolist())
    }
    target_lookup = {value: index for index, value in enumerate(target.ids.tolist())}
    metadata = store.manifest.get("metadata", {})
    model = metadata.get("model_provenance", {})
    if not isinstance(model, Mapping):
        raise IntegrityError("positive-control store lacks model provenance")
    head_fit_ids = set(store.array(str(model["head_fit_ids_role"])).tolist())
    eligible = [
        (reference_id, target_id)
        for reference_id, target_id in zip(paired_reference, paired_target)
        if reference_id not in head_fit_ids
    ]
    calibration = int(anchor["calibration_rows"])
    witness_rows = int(anchor["witness_rows"])
    requested = calibration + witness_rows
    seed = int(anchor["split_seed"])
    order = _pair_hash_order(
        [item[0] for item in eligible],
        [item[1] for item in eligible],
        seed=seed,
    )[:requested]
    selected = [eligible[int(index)] for index in order]
    if len(selected) != requested:
        raise IntegrityError("positive-control split has too few eligible pairs")
    selected_sha = sha256_json(
        [
            {"reference_id": str(reference_id), "target_id": str(target_id)}
            for reference_id, target_id in selected
        ]
    )
    if selected_sha != anchor["selected_pair_split_sha256"]:
        raise IntegrityError("positive-control pair split differs from sealed result")
    reference_rows = np.asarray(
        [reference_lookup[item[0]] for item in selected], dtype=np.int64
    )
    target_rows = np.asarray(
        [target_lookup[item[1]] for item in selected], dtype=np.int64
    )
    reference_features = np.asarray(
        reference.features[reference_rows], dtype=np.float64
    )
    target_features = np.asarray(target.features[target_rows], dtype=np.float64)
    if reference.labels is None or target.labels is None:
        raise IntegrityError("positive-control administrative labels are absent")
    if not np.array_equal(
        np.asarray(reference.labels[reference_rows]),
        np.asarray(target.labels[target_rows]),
    ):
        raise IntegrityError("positive-control paired labels disagree")

    reference_calibration_order = _independent_pair_permutation(
        calibration, seed=seed, endpoint="reference-calibration"
    )
    target_calibration_order = _independent_pair_permutation(
        calibration, seed=seed, endpoint="target-calibration"
    )
    reference_witness_order = _independent_pair_permutation(
        witness_rows, seed=seed, endpoint="reference-witness"
    )
    target_witness_order = _independent_pair_permutation(
        witness_rows, seed=seed, endpoint="target-witness"
    )
    reference_calibration = reference_features[:calibration][
        reference_calibration_order
    ]
    target_calibration = target_features[:calibration][target_calibration_order]
    reference_witness = reference_features[calibration:][reference_witness_order]
    target_witness = target_features[calibration:][target_witness_order]
    head = store.manifest["head"]
    weight = np.asarray(store.array(str(head["weight_role"])), dtype=np.float64)
    bias = np.asarray(store.array(str(head["bias_role"])), dtype=np.float64)
    geometry = fit_witness_geometry(
        target_calibration,
        reference_calibration,
        weight,
        bias,
        protocol=_protocol(anchor),
    )

    (
        target_whitened,
        reference_whitened,
        target_canonical_order,
        (reference_canonical_order),
    ) = _canonical_geometry(geometry, target_witness, reference_witness, anchor)
    target_positions = target_witness_order[target_canonical_order]
    reference_positions = reference_witness_order[reference_canonical_order]
    inverse_reference = np.empty(witness_rows, dtype=np.int64)
    inverse_reference[reference_positions] = np.arange(witness_rows, dtype=np.int64)
    identity_assignment = inverse_reference[target_positions]
    identity_cost = assignment_cost(
        target_whitened, reference_whitened, identity_assignment
    )
    recorded_identity = _finite(
        anchor["posthoc_validation"]["true_assignment_cost"],
        name="recorded identity cost",
    )
    _close(identity_cost, recorded_identity, name="identity assignment cost")
    return target_whitened, reference_whitened, identity_cost


def _condition_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        "positive_control" if "corruption" in record else "natural",
        str(record["dataset"]),
        str(record["encoder"]),
        record.get("corruption"),
        record.get("severity"),
        int(record["split_seed"]),
    )


def _budget_token(budget: float) -> str:
    return f"{int(round(10_000.0 * budget)):04d}bp"


def _expected_unit_specifications() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for encoder in EXPECTED_ENCODERS:
        for seed in EXPECTED_SEEDS:
            for budget in EXPECTED_BUDGETS:
                unit_id = (
                    f"imagenet_sketch__{encoder}__seed_{seed}__"
                    f"budget_{_budget_token(budget)}"
                )
                expected[unit_id] = {
                    "panel": "natural",
                    "schema_version": EXPECTED_NATURAL_SCHEMA,
                    "dataset": "imagenet_sketch",
                    "encoder": encoder,
                    "corruption": None,
                    "severity": None,
                    "split_seed": seed,
                    "cost_budget_relative": budget,
                }
    for corruption in EXPECTED_POSITIVE_CORRUPTIONS:
        for encoder in EXPECTED_ENCODERS:
            for seed in EXPECTED_SEEDS:
                for budget in EXPECTED_BUDGETS:
                    unit_id = (
                        f"positive__{corruption}__{encoder}__seed_{seed}__"
                        f"budget_{_budget_token(budget)}"
                    )
                    expected[unit_id] = {
                        "panel": "positive_control",
                        "schema_version": EXPECTED_POSITIVE_SCHEMA,
                        "dataset": "imagenet_c",
                        "encoder": encoder,
                        "corruption": corruption,
                        "severity": 5,
                        "split_seed": seed,
                        "cost_budget_relative": budget,
                    }
    if len(expected) != EXPECTED_NATURAL_UNITS + EXPECTED_POSITIVE_UNITS:
        raise AssertionError("internal expected-unit inventory has the wrong size")
    return expected


def _validate_completion_marker(marker: Mapping[str, Any]) -> None:
    if (
        marker.get("schema_version") != EXPECTED_MARKER_SCHEMA
        or marker.get("status") != "PASS"
        or marker.get("scientific_result") is not True
        or marker.get("component") != "play3"
        or marker.get("complete_profile_panel") is not True
        or marker.get("complete_original_panel") is not False
        or int(marker.get("complete_unit_count", -1)) != 126
        or int(marker.get("expected_unit_count", -1)) != 126
        or int(marker.get("natural_units", -1)) != EXPECTED_NATURAL_UNITS
        or int(marker.get("positive_control_units", -1)) != EXPECTED_POSITIVE_UNITS
        or marker.get("selection_profile", {}).get("profile_id")
        != "beyond_marginals_rtx_7day_v1"
    ):
        raise IntegrityError(
            "historical Play-3 completion marker is not the exact 126-unit PASS"
        )


def _sealed_unit_paths(
    results_root: Path, marker: Mapping[str, Any]
) -> list[tuple[Path, dict[str, Any]]]:
    """Bind every consumed unit byte to the sealed completion marker and grid."""

    expected = _expected_unit_specifications()
    raw_artifacts = marker.get("unit_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(expected):
        raise IntegrityError("completion marker has an invalid unit-artifact inventory")
    sealed: dict[str, str] = {}
    for artifact in raw_artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {"unit_id", "sha256"}:
            raise IntegrityError(
                "completion marker unit artifact has an invalid schema"
            )
        unit_id = str(artifact["unit_id"])
        digest = str(artifact["sha256"])
        if unit_id in sealed or unit_id not in expected:
            raise IntegrityError(f"unexpected or duplicate sealed unit id: {unit_id}")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise IntegrityError(f"invalid sealed unit digest: {unit_id}")
        sealed[unit_id] = digest
    if set(sealed) != set(expected):
        missing = sorted(set(expected) - set(sealed))
        extra = sorted(set(sealed) - set(expected))
        raise IntegrityError(
            f"completion marker unit grid differs: missing={missing}, extra={extra}"
        )

    unit_paths: list[tuple[Path, dict[str, Any]]] = []
    observed_paths: set[Path] = set()
    for unit_id, specification in sorted(expected.items()):
        directory = results_root / str(specification["panel"])
        if not directory.is_dir() or directory.is_symlink():
            raise IntegrityError(f"unit directory is absent or symbolic: {directory}")
        path = directory / f"{unit_id}.json"
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"sealed unit is absent or symbolic: {path}")
        if sha256_file(path) != sealed[unit_id]:
            raise IntegrityError(f"sealed unit bytes changed: {path}")
        observed_paths.add(path)
        unit_paths.append((path, specification))
    actual_paths = {
        path
        for directory_name in ("natural", "positive_control")
        for path in (results_root / directory_name).glob("*.json")
    }
    if actual_paths != observed_paths:
        unexpected = sorted(str(path) for path in actual_paths - observed_paths)
        missing = sorted(str(path) for path in observed_paths - actual_paths)
        raise IntegrityError(
            "on-disk unit inventory differs from sealed inventory: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return unit_paths


def _validate_unit_record(
    path: Path, value: Mapping[str, Any], specification: Mapping[str, Any]
) -> None:
    unit_plan = value.get("unit_plan")
    if not isinstance(unit_plan, Mapping) or unit_plan.get("unit_id") != path.stem:
        raise IntegrityError(f"unit id does not match its sealed path: {path}")
    exact_fields = {
        "schema_version": specification["schema_version"],
        "dataset": specification["dataset"],
        "encoder": specification["encoder"],
        "split_seed": specification["split_seed"],
        "calibration_rows": 4096,
        "witness_rows": 1024,
    }
    if specification["panel"] == "positive_control":
        exact_fields.update(
            {
                "corruption": specification["corruption"],
                "severity": specification["severity"],
            }
        )
    for name, expected_value in exact_fields.items():
        if value.get(name) != expected_value:
            raise IntegrityError(
                f"sealed unit {name} differs for {path}: "
                f"{value.get(name)!r} != {expected_value!r}"
            )
    if float(unit_plan.get("cost_budget_relative", float("nan"))) != float(
        specification["cost_budget_relative"]
    ):
        raise IntegrityError(f"sealed unit budget differs for {path}")


def _load_groups(
    results_root: Path, marker: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unit_paths = _sealed_unit_paths(results_root, marker)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    source_files: Mapping[str, str] | None = None
    for path, specification in unit_paths:
        value = read_json(path)
        verify_manifest_checksum(value)
        if value.get("status") != "PASS" or value.get("scientific_result") is not True:
            raise IntegrityError(f"unit is not a scientific PASS result: {path}")
        _validate_unit_record(path, value, specification)
        value = dict(value)
        value["_path"] = str(path)
        value["_sha256"] = sha256_file(path)
        groups[_condition_key(value)].append(value)
        embedded = (
            value.get("unit_plan", {}).get("extension_lock", {}).get("source_files")
        )
        if embedded is not None:
            if source_files is None:
                source_files = embedded
            elif dict(embedded) != dict(source_files):
                raise IntegrityError("unit source-file identities disagree")
    if len(groups) != EXPECTED_CONDITION_GEOMETRIES:
        raise IntegrityError(f"expected 42 condition geometries, found {len(groups)}")
    tasks = []
    for key, records in sorted(groups.items(), key=lambda item: str(item[0])):
        records.sort(key=lambda row: float(row["unit_plan"]["cost_budget_relative"]))
        budgets = tuple(
            float(row["unit_plan"]["cost_budget_relative"]) for row in records
        )
        if budgets != EXPECTED_BUDGETS:
            raise IntegrityError(f"condition lacks the three locked budgets: {key}")
        anchor = records[0]
        baseline = _finite(anchor["ceiling_witness"]["baseline_cost"], name="baseline")
        target_sha = anchor["ceiling_witness"]["diagnostics"]["target_content_sha256"]
        reference_sha = anchor["ceiling_witness"]["diagnostics"][
            "reference_content_sha256"
        ]
        geometry_sha = anchor["ceiling_witness"]["diagnostics"]["calibration"][
            "full_geometry_sha256"
        ]
        anchor_plan = anchor["unit_plan"]
        anchor_store_binding = (
            anchor_plan["store_manifest"],
            anchor_plan["store_manifest_sha256"],
        )
        if key[0] == "natural":
            anchor_split_binding = (
                anchor["reference_split_content_sha256"],
                anchor["target_split_content_sha256"],
            )
        else:
            anchor_split_binding = (
                anchor["selected_pair_split_sha256"],
                anchor["posthoc_validation"]["pairing_content_sha256"],
            )
        for row, budget in zip(records, budgets):
            row_plan = row["unit_plan"]
            if (
                row_plan["store_manifest"],
                row_plan["store_manifest_sha256"],
            ) != anchor_store_binding:
                raise IntegrityError("cross-budget feature-store binding changed")
            if key[0] == "natural":
                row_split_binding = (
                    row["reference_split_content_sha256"],
                    row["target_split_content_sha256"],
                )
            else:
                row_split_binding = (
                    row["selected_pair_split_sha256"],
                    row["posthoc_validation"]["pairing_content_sha256"],
                )
            if row_split_binding != anchor_split_binding:
                raise IntegrityError("cross-budget split or pairing binding changed")
            witness = row["ceiling_witness"]
            _close(
                _finite(witness["baseline_cost"], name="baseline"),
                baseline,
                name="cross-budget baseline",
            )
            _close(
                _finite(witness["allowed_cost"], name="allowed cost"),
                baseline * (1.0 + budget),
                name="sealed allowed cost",
            )
            diagnostics = witness["diagnostics"]
            if (
                diagnostics["target_content_sha256"] != target_sha
                or diagnostics["reference_content_sha256"] != reference_sha
                or diagnostics["calibration"]["full_geometry_sha256"] != geometry_sha
            ):
                raise IntegrityError("cross-budget split or fitted geometry changed")
        store_path = str(anchor["unit_plan"]["store_manifest"])
        expected_store_sha = str(anchor["unit_plan"]["store_manifest_sha256"])
        tasks.append(
            {
                "key": list(key),
                "panel": key[0],
                "store_path": store_path,
                "expected_store_sha256": expected_store_sha,
                "records": records,
            }
        )
    if source_files is None:
        raise IntegrityError("positive-control units lack an embedded source manifest")
    return tasks, dict(source_files)


def _authenticated_store_manifest(task: Mapping[str, Any]) -> tuple[Path, Path]:
    raw_store_path = Path(str(task["store_path"])).expanduser()
    if raw_store_path.is_symlink():
        raise IntegrityError(f"feature-store path is symbolic: {raw_store_path}")
    store_path = raw_store_path.resolve(strict=True)
    manifest = (
        store_path / "FEATURE_STORE_MANIFEST.json"
        if store_path.is_dir()
        else store_path
    )
    if not manifest.is_file() or manifest.is_symlink():
        raise IntegrityError(
            f"feature-store manifest is absent or symbolic: {manifest}"
        )
    if sha256_file(manifest) != task["expected_store_sha256"]:
        raise IntegrityError(f"feature-store manifest changed: {manifest}")
    return store_path, manifest


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    records = list(task["records"])
    anchor = records[0]
    store_path, manifest = _authenticated_store_manifest(task)
    if task["panel"] == "natural":
        target, reference, identity_cost = _natural_geometry(str(store_path), anchor)
    else:
        target, reference, identity_cost = _positive_geometry(str(store_path), anchor)
    random_moments = uniform_random_bijection_cost_moments(target, reference)
    baseline = float(anchor["ceiling_witness"]["baseline_cost"])
    cost_scale = summarize_cost_scale(
        minimum_cost=baseline,
        random_moments=random_moments,
        budgets=EXPECTED_BUDGETS,
        identity_cost=identity_cost,
    )
    budget_results = []
    for row, scale_row in zip(records, cost_scale["budgets"]):
        budget_results.append(
            {
                **scale_row,
                "feasible_disagreement": float(row["ceiling_witness"]["disagreement"]),
                "two_world_recovery_ceiling": float(
                    row["ceiling_witness"]["prediction_recovery_ceiling"]
                ),
                "unit_result": row["_path"],
                "unit_result_sha256": row["_sha256"],
            }
        )
    return {
        "panel": task["key"][0],
        "dataset": task["key"][1],
        "encoder": task["key"][2],
        "corruption": task["key"][3],
        "severity": task["key"][4],
        "split_seed": task["key"][5],
        "calibration_rows": int(anchor["calibration_rows"]),
        "witness_rows": int(anchor["witness_rows"]),
        "feature_dimension": int(
            anchor["ceiling_witness"]["diagnostics"]["feature_dimension"]
        ),
        "store_manifest": str(manifest),
        "store_manifest_sha256": str(task["expected_store_sha256"]),
        "fitted_geometry_sha256": anchor["ceiling_witness"]["diagnostics"][
            "calibration"
        ]["full_geometry_sha256"],
        "minimum_cost": baseline,
        "random_bijection": cost_scale["random_bijection"],
        "identity_pairing": cost_scale["identity_pairing"],
        "budgets": budget_results,
    }


def _range(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise IntegrityError("summary received no finite values")
    return {
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _summaries(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for panel in ("natural", "positive_control"):
        selected = [row for row in conditions if row["panel"] == panel]
        record: dict[str, Any] = {
            "panel": panel,
            "condition_geometries": len(selected),
            "random_expected_to_minimum_ratio": _range(
                row["random_bijection"]["expected_to_minimum_ratio"] for row in selected
            ),
            "random_cost_coefficient_of_variation": _range(
                row["random_bijection"]["coefficient_of_variation"] for row in selected
            ),
            "budgets": [],
        }
        identities = [row["identity_pairing"] for row in selected]
        if all(value is None for value in identities):
            record["identity_pairing"] = None
        elif any(value is None for value in identities):
            raise IntegrityError("panel mixes available and unavailable identities")
        else:
            record["identity_pairing"] = {
                "cost_to_minimum_ratio": _range(
                    value["cost_to_minimum_ratio"] for value in identities
                ),
                "excess_fraction_of_random_minus_minimum": _range(
                    value["excess_fraction_of_random_minus_minimum"]
                    for value in identities
                ),
            }
        for index, budget in enumerate(EXPECTED_BUDGETS):
            rows = [row["budgets"][index] for row in selected]
            if any(float(row["cost_budget_relative"]) != budget for row in rows):
                raise IntegrityError("condition budget order changed")
            record["budgets"].append(
                {
                    "cost_budget_relative": budget,
                    "allowed_excess_fraction_of_random_minus_minimum": _range(
                        row["allowed_excess_fraction_of_random_minus_minimum"]
                        for row in rows
                    ),
                    "feasible_disagreement": _range(
                        row["feasible_disagreement"] for row in rows
                    ),
                }
            )
        output.append(record)
    return output


def _verify_marker(marker: Mapping[str, Any], summaries: list[dict[str, Any]]) -> None:
    expected = {
        "natural": marker["natural_budget_macro"],
        "positive_control": marker["positive_control_budget_macro"],
    }
    for summary in summaries:
        rows = expected[summary["panel"]]
        for observed, sealed in zip(summary["budgets"], rows):
            if float(observed["cost_budget_relative"]) != float(
                sealed["cost_budget_relative"]
            ):
                raise IntegrityError("completion marker budget order changed")
            sealed_disagreement = 2.0 * (
                1.0 - float(sealed["equal_prior_randomized_minimax_ceiling_macro_mean"])
            )
            _close(
                float(observed["feasible_disagreement"]["mean"]),
                sealed_disagreement,
                name="completion-marker disagreement macro",
            )


def _verify_historical_source(
    project_root: Path, source_files: Mapping[str, str]
) -> None:
    for relative, expected in sorted(source_files.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise IntegrityError(f"source path is not a safe relative path: {relative}")
        raw_path = project_root / relative_path
        cursor = raw_path
        while cursor != project_root:
            if cursor.is_symlink():
                raise IntegrityError(f"historical source path is symbolic: {relative}")
            cursor = cursor.parent
        if not raw_path.is_file():
            raise IntegrityError(f"historical source file is absent: {relative}")
        path = raw_path.resolve(strict=True)
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise IntegrityError(
                f"source path escapes project root: {relative}"
            ) from error
        observed = sha256_file(path)
        if observed != expected:
            raise IntegrityError(
                f"historical source identity changed for {relative}: {observed} != {expected}"
            )


def _validated_output_path(raw_output: Path, results_root: Path) -> Path:
    output = raw_output.expanduser().resolve(strict=False)
    if output == results_root or results_root in output.parents:
        raise IntegrityError(
            "cost-scale output must be outside the completed historical results root"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite cost-scale audit: {output}")
    return output


def verify_cost_scale_result(value: Mapping[str, Any]) -> None:
    """Authenticate the sealed top-level identity of a cost-scale result.

    The manifest checksum covers every nested condition, budget, summary, and
    provenance field.  Exact scientific recomputation is performed by the
    deployment export gate; this local check makes any intervening byte-level
    or semantic mutation fail closed before a result is accepted.
    """

    if not isinstance(value, Mapping):
        raise IntegrityError("cost-scale result must be a JSON object")
    verify_manifest_checksum(value)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "PASS"
        or value.get("scientific_result") is not True
        or value.get("analysis_timing")
        != "post_hoc_after_inspection_of_correspondence_outcomes"
        or int(value.get("historical_source_files", -1)) != 26
        or value.get("historical_source_identity_verified") is not True
        or int(value.get("condition_geometries", -1)) != EXPECTED_CONDITION_GEOMETRIES
    ):
        raise IntegrityError("cost-scale result identity or grid count drifted")
    conditions = value.get("conditions")
    summaries = value.get("summaries")
    if (
        not isinstance(conditions, list)
        or len(conditions) != EXPECTED_CONDITION_GEOMETRIES
    ):
        raise IntegrityError("cost-scale result condition inventory drifted")
    if (
        not isinstance(summaries, list)
        or len(summaries) != 2
        or not all(isinstance(row, Mapping) for row in summaries)
        or [row.get("panel") for row in summaries] != ["natural", "positive_control"]
    ):
        raise IntegrityError("cost-scale result summary inventory drifted")

    expected_units = _expected_unit_specifications()
    expected_conditions = {
        (
            specification["panel"],
            specification["dataset"],
            specification["encoder"],
            specification["corruption"],
            specification["severity"],
            specification["split_seed"],
        )
        for specification in expected_units.values()
    }
    observed_conditions: set[tuple[Any, ...]] = set()
    observed_units: set[str] = set()
    expected_condition_fields = {
        "panel",
        "dataset",
        "encoder",
        "corruption",
        "severity",
        "split_seed",
        "calibration_rows",
        "witness_rows",
        "feature_dimension",
        "store_manifest",
        "store_manifest_sha256",
        "fitted_geometry_sha256",
        "minimum_cost",
        "random_bijection",
        "identity_pairing",
        "budgets",
    }
    random_fields = {
        "expected_total_cost",
        "variance_total_cost",
        "standard_deviation_total_cost",
        "coefficient_of_variation",
        "expected_to_minimum_ratio",
        "expected_excess_relative_to_minimum",
    }
    scale_budget_fields = {
        "cost_budget_relative",
        "allowed_cost",
        "allowed_to_random_mean_ratio",
        "allowed_excess_fraction_of_random_minus_minimum",
        "random_mean_minus_allowed_in_random_sd",
    }
    budget_fields = scale_budget_fields | {
        "feasible_disagreement",
        "two_world_recovery_ceiling",
        "unit_result",
        "unit_result_sha256",
    }

    for index, condition in enumerate(conditions):
        if (
            not isinstance(condition, Mapping)
            or set(condition) != expected_condition_fields
        ):
            raise IntegrityError(f"cost-scale condition {index} has a malformed schema")
        try:
            identity = (
                condition["panel"],
                condition["dataset"],
                condition["encoder"],
                condition["corruption"],
                condition["severity"],
                int(condition["split_seed"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                f"cost-scale condition {index} has a malformed identity"
            ) from error
        if identity not in expected_conditions or identity in observed_conditions:
            raise IntegrityError(
                f"cost-scale condition grid has an unexpected or duplicate identity: {identity}"
            )
        observed_conditions.add(identity)
        if (
            condition["split_seed"] != identity[-1]
            or condition["calibration_rows"] != 4096
            or condition["witness_rows"] != 1024
            or not isinstance(condition["feature_dimension"], int)
            or isinstance(condition["feature_dimension"], bool)
            or condition["feature_dimension"] < 1
            or not isinstance(condition["store_manifest"], str)
            or not condition["store_manifest"]
        ):
            raise IntegrityError(f"cost-scale condition protocol drifted: {identity}")
        for field in ("store_manifest_sha256", "fitted_geometry_sha256"):
            digest = condition[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise IntegrityError(
                    f"cost-scale condition {field} is invalid: {identity}"
                )

        minimum = _finite(condition["minimum_cost"], name="result minimum cost")
        random = condition["random_bijection"]
        if not isinstance(random, Mapping) or set(random) != random_fields:
            raise IntegrityError(f"cost-scale random moments are malformed: {identity}")
        random_moments = {
            field: _finite(random[field], name=f"result random {field}")
            for field in (
                "expected_total_cost",
                "variance_total_cost",
                "standard_deviation_total_cost",
                "coefficient_of_variation",
            )
        }
        if (
            random_moments["variance_total_cost"] < 0.0
            or random_moments["standard_deviation_total_cost"] < 0.0
            or random_moments["coefficient_of_variation"] < 0.0
        ):
            raise IntegrityError(f"cost-scale random moments are negative: {identity}")

        identity_pairing = condition["identity_pairing"]
        if identity[0] == "natural":
            if identity_pairing is not None:
                raise IntegrityError(
                    "natural-shift cost scale may not reveal an identity pairing"
                )
            identity_cost = None
        else:
            if not isinstance(identity_pairing, Mapping):
                raise IntegrityError(
                    "positive-control cost scale lacks identity pairing"
                )
            identity_cost = _finite(
                identity_pairing.get("cost"), name="result identity cost"
            )

        budget_rows = condition["budgets"]
        if not isinstance(budget_rows, list) or len(budget_rows) != len(
            EXPECTED_BUDGETS
        ):
            raise IntegrityError(f"cost-scale budget inventory drifted: {identity}")
        try:
            reproduced_scale = summarize_cost_scale(
                minimum_cost=minimum,
                random_moments=random_moments,
                budgets=EXPECTED_BUDGETS,
                identity_cost=identity_cost,
            )
        except (TypeError, ValueError) as error:
            raise IntegrityError(
                f"cost-scale numerical payload is invalid: {identity}"
            ) from error
        if dict(random) != reproduced_scale["random_bijection"]:
            raise IntegrityError(f"cost-scale random normalization drifted: {identity}")
        if identity_pairing != reproduced_scale["identity_pairing"]:
            raise IntegrityError(
                f"cost-scale identity normalization drifted: {identity}"
            )

        for budget_row, expected_budget, reproduced_budget in zip(
            budget_rows, EXPECTED_BUDGETS, reproduced_scale["budgets"]
        ):
            if not isinstance(budget_row, Mapping) or set(budget_row) != budget_fields:
                raise IntegrityError(f"cost-scale budget row is malformed: {identity}")
            if any(
                budget_row[field] != reproduced_budget[field]
                for field in scale_budget_fields
            ):
                raise IntegrityError(
                    f"cost-scale budget normalization drifted: {identity}"
                )
            for field in ("feasible_disagreement", "two_world_recovery_ceiling"):
                metric = _finite(budget_row[field], name=f"result {field}")
                if not 0.0 <= metric <= 1.0:
                    raise IntegrityError(f"cost-scale {field} leaves [0,1]: {identity}")
            result_path = budget_row["unit_result"]
            digest = budget_row["unit_result_sha256"]
            if not isinstance(result_path, str) or not result_path:
                raise IntegrityError(
                    f"cost-scale unit-result path is invalid: {identity}"
                )
            unit_id = Path(result_path).stem
            specification = expected_units.get(unit_id)
            if (
                specification is None
                or specification["panel"] != identity[0]
                or specification["dataset"] != identity[1]
                or specification["encoder"] != identity[2]
                or specification["corruption"] != identity[3]
                or specification["severity"] != identity[4]
                or specification["split_seed"] != identity[5]
                or specification["cost_budget_relative"] != expected_budget
                or Path(result_path).parent.name != specification["panel"]
                or unit_id in observed_units
            ):
                raise IntegrityError(
                    f"cost-scale unit-result grid has an unexpected or duplicate path: {result_path}"
                )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise IntegrityError(
                    f"cost-scale unit-result hash is invalid: {unit_id}"
                )
            observed_units.add(unit_id)

    if observed_conditions != expected_conditions or observed_units != set(
        expected_units
    ):
        raise IntegrityError("cost-scale nested condition or unit grid is incomplete")
    try:
        reproduced_summaries = _summaries([dict(row) for row in conditions])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("cost-scale summaries cannot be reproduced") from error
    if summaries != reproduced_summaries:
        raise IntegrityError("cost-scale summaries differ from the sealed conditions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)
    project_root = arguments.project_root.expanduser().resolve(strict=True)
    results_root = arguments.results_root.expanduser().resolve(strict=True)
    output = _validated_output_path(arguments.output, results_root)
    if arguments.workers < 1 or arguments.workers > 8:
        raise ValueError("workers must lie in [1,8]")
    thread_state = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if any(value != "1" for value in thread_state.values()):
        raise RuntimeError(
            "every numerical thread variable must equal '1': "
            f"observed={thread_state}"
        )
    marker_path = results_root / "RTX_PROFILE_COMPLETE.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise IntegrityError(
            "historical Play-3 completion marker is absent or symbolic"
        )
    marker = read_json(marker_path)
    verify_manifest_checksum(marker)
    _validate_completion_marker(marker)
    tasks, source_files = _load_groups(results_root, marker)
    _verify_historical_source(project_root, source_files)
    for task in tasks:
        _authenticated_store_manifest(task)
    if arguments.preflight_only:
        print(
            "COST-SCALE PREFLIGHT PASS: "
            f"{len(tasks)} condition geometries, {len(source_files)} source files"
        )
        return 0
    if arguments.workers == 1:
        conditions = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            conditions = list(executor.map(_run_task, tasks))
    conditions.sort(
        key=lambda row: (
            row["panel"],
            row["dataset"],
            row["encoder"],
            str(row["corruption"]),
            int(row["split_seed"]),
        )
    )
    summaries = _summaries(conditions)
    _verify_marker(marker, summaries)
    payload = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "scientific_result": True,
            "analysis_timing": "post_hoc_after_inspection_of_correspondence_outcomes",
            "interpretation": (
                "descriptive finite-batch cost-scale audit; random-bijection moments "
                "are exact over the uniform permutation law, while identity pairing is "
                "available only in the ImageNet-C positive control"
            ),
            "fit_information": (
                "the audit reproduces each sealed split and full-reference-OAS "
                "Mahalanobis-after-Bures geometry; labels and pairing do not fit maps"
            ),
            "results_root": str(results_root),
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
            "completion_marker": str(marker_path),
            "completion_marker_sha256": sha256_file(marker_path),
            "historical_source_files": len(source_files),
            "historical_source_identity_verified": True,
            "condition_geometries": len(conditions),
            "conditions": conditions,
            "summaries": summaries,
        }
    )
    verify_cost_scale_result(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)
    print(output)
    print(f"sha256={sha256_file(output)}")
    print("CORRESPONDENCE COST-SCALE AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
