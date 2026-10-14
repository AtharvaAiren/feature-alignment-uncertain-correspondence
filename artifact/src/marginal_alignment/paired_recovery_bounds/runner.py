"""Resumable CPU runner, semantic verifier, and deterministic aggregation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    canonical_json_bytes,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import FeatureStore

from .conformal import split_conformal_quantile
from .geometry import (
    FittedRecoveryGeometry,
    allowed_cost,
    array_sha256,
    batch_cost_problem,
    clean_predictions,
    fit_recovery_geometry,
    frozen_predictions,
    score_correspondence,
)
from .optimization import (
    certify_recovery_error,
    solve_exact_recovery_milp,
    verify_certificate,
)
from .protocol import (
    EXPECTED_CORRUPTIONS,
    EXPECTED_METHODS,
    LockedProtocol,
    load_authenticated_store,
    load_locked_protocol,
    runtime_identity,
    verify_external_input_manifest,
)
from .splitting import RecoveryBatch, RecoveryPartition, build_recovery_partition


PREPARED_SCHEMA = "marginal_alignment.paired_recovery_bounds.prepared.v1"
UNIT_SCHEMA = "marginal_alignment.paired_recovery_bounds.unit.v1"
CORRUPTION_DONE_SCHEMA = "marginal_alignment.paired_recovery_bounds.corruption_done.v1"
AGGREGATE_SCHEMA = "marginal_alignment.paired_recovery_bounds.aggregate.v1"
DONE_SCHEMA = "marginal_alignment.paired_recovery_bounds.done.v1"
STATE_SCHEMA = "marginal_alignment.paired_recovery_bounds.operational_state.v1"


@dataclass(frozen=True)
class PreparedContext:
    protocol: LockedProtocol
    store: FeatureStore
    partition: RecoveryPartition
    fitted: FittedRecoveryGeometry
    prepared_payload: Mapping[str, Any]
    qhat: float


def _threshold_payload(value: float) -> dict[str, Any]:
    if math.isinf(float(value)):
        return {"kind": "positive_infinity", "value": None}
    if not math.isfinite(float(value)):
        raise ValueError("threshold must be finite or positive infinity")
    return {"kind": "finite", "value": float(value)}


def _threshold_value(payload: Mapping[str, Any]) -> float:
    kind = payload.get("kind")
    if kind == "positive_infinity" and payload.get("value") is None:
        return math.inf
    if kind == "finite":
        value = float(payload.get("value"))
        if math.isfinite(value) and value >= 0.0:
            return value
    raise IntegrityError("serialized conformal threshold is malformed")


def _state_path(output_root: Path, corruption: str) -> Path:
    return output_root / "state" / f"{corruption}.json"


def _write_state(
    output_root: Path,
    corruption: str,
    *,
    stage: str,
    completed_units: int,
    total_units: int,
    status: str = "RUNNING",
) -> None:
    # Deliberately outcome blind: the monitor may show this entire file.
    write_json_atomic(
        _state_path(output_root, corruption),
        seal_manifest({
            "schema_version": STATE_SCHEMA,
            "status": status,
            "corruption": corruption,
            "stage": stage,
            "completed_units": int(completed_units),
            "total_units": int(total_units),
            "updated_unix_seconds": int(time.time()),
            "scientific_outcomes_in_state": False,
        }),
    )


def _slice(batch: RecoveryBatch, store: FeatureStore) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(
        store.endpoint("reference").features[batch.reference_rows], dtype=np.float64
    )
    target = np.asarray(
        store.endpoint("target").features[batch.target_rows], dtype=np.float64
    )
    if reference.shape != target.shape or reference.shape[0] != batch.size:
        raise IntegrityError("selected paired endpoint slices changed shape")
    return target, reference


def _head(store: FeatureStore) -> tuple[np.ndarray, np.ndarray]:
    head = store.manifest.get("head")
    if not isinstance(head, Mapping):
        raise IntegrityError("feature store lacks a frozen readout")
    return (
        np.asarray(store.array(str(head["weight_role"])), dtype=np.float64),
        np.asarray(store.array(str(head["bias_role"])), dtype=np.float64),
    )


def _corruption_spec(protocol: LockedProtocol, corruption: str) -> Mapping[str, Any]:
    try:
        return protocol.corruption_by_name[corruption]
    except KeyError as error:
        raise IntegrityError(f"corruption is outside prospective lock: {corruption}") from error


def _prepare_context(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    external_input_manifest: str | Path,
    feature_root: str | Path,
    corruption: str,
) -> PreparedContext:
    protocol = load_locked_protocol(
        project_root, config_path=config_path, lock_path=lock_path
    )
    verify_external_input_manifest(protocol, external_input_manifest)
    store = load_authenticated_store(
        protocol, feature_root=feature_root, corruption=corruption
    )
    config = protocol.config
    partition_config = config["partition"]
    specification = _corruption_spec(protocol, corruption)
    partition = build_recovery_partition(
        store,
        seed=int(specification["split_seed"]),
        map_fit_rows=int(partition_config["map_fit_rows"]),
        calibration_batches=int(partition_config["calibration_batches"]),
        evaluation_batches=int(partition_config["evaluation_batches"]),
        batch_size=int(partition_config["batch_rows"]),
        expected_eligible_pairs=int(
            config["shared_store_contract"][
                "eligible_identity_pairs_after_head_fit_exclusion"
            ]
        ),
    )
    target_fit, reference_fit = _slice(partition.map_fit, store)
    weight, bias = _head(store)
    geometry_config = config["common_cost_geometry"]
    fitted = fit_recovery_geometry(
        target_fit,
        reference_fit,
        weight,
        bias,
        covariance_estimator=str(geometry_config["covariance_estimator"]),
        covariance_relative_floor=float(
            geometry_config["covariance_relative_floor"]
        ),
        diagonal_standard_deviation_floor=float(
            config["methods"]["diagonal"]["standard_deviation_floor"]
        ),
    )
    calibration_rows = []
    calibration_scores = []
    for index, batch in enumerate(partition.calibration):
        target, reference = _slice(batch, store)
        problem = batch_cost_problem(
            fitted,
            target,
            reference,
            true_assignment=batch.true_assignment,
        )
        calibration_scores.append(problem.true_normalized_score)
        calibration_rows.append({
            "batch_index": index,
            "batch_identity_sha256": batch.identity_sha256,
            "true_normalized_cost_score": problem.true_normalized_score,
            "minimum_cost": problem.minimum_cost,
            "uniform_random_expected_cost": problem.random_expected_cost,
            "random_minus_minimum_cost": problem.random_minus_minimum,
            "true_correspondence_cost": problem.true_cost,
            "cost_diagnostics": dict(problem.diagnostics),
        })
    qhat, conformal_diagnostics = split_conformal_quantile(
        calibration_scores, delta=float(config["conformal"]["delta"])
    )
    payload = {
        "schema_version": PREPARED_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "study_id": config["study_id"],
        "dataset": "imagenet_c",
        "encoder": "clip",
        "corruption": corruption,
        "severity": int(specification["severity"]),
        "bindings": {
            "config_file_sha256": protocol.config_file_sha256,
            "lock_file_sha256": protocol.lock_file_sha256,
            "lock_manifest_sha256": protocol.lock["manifest_sha256"],
            "external_input_manifest_file_sha256": config[
                "external_input_manifest"
            ]["file_sha256"],
            "feature_store_manifest": str(store.manifest_path),
            "feature_store_manifest_file_sha256": sha256_file(store.manifest_path),
        },
        "partition": partition.summary(),
        "geometry": dict(fitted.diagnostics),
        "calibration": {
            "pairing_role": "true_pairing_used_only_to_score_common_cost_geometry",
            "scores": calibration_rows,
            "score_vector_sha256": array_sha256(
                np.asarray(calibration_scores, dtype=np.float64)
            ),
            "qhat": _threshold_payload(qhat),
            "diagnostics": conformal_diagnostics,
        },
    }
    return PreparedContext(
        protocol=protocol,
        store=store,
        partition=partition,
        fitted=fitted,
        prepared_payload=payload,
        qhat=qhat,
    )


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = seal_manifest(payload)
    if path.exists():
        existing = read_json(path)
        if not isinstance(existing, Mapping):
            raise IntegrityError(f"existing artifact is malformed: {path}")
        verify_manifest_checksum(existing)
        if existing != expected:
            raise IntegrityError(f"existing artifact differs from deterministic reconstruction: {path}")
        return existing
    write_json_atomic(path, expected)
    return expected


def _unit_path(output_root: Path, corruption: str, batch_index: int) -> Path:
    return output_root / "units" / corruption / f"batch_{batch_index:03d}.json"


def _exact_diagnostic(
    context: PreparedContext,
    *,
    target: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    config = context.protocol.config
    exact_config = config["optimization"]["exact_diagnostic"]
    rows = int(exact_config["rows"])
    target_subset = np.asarray(target[:rows], dtype=np.float64)
    reference_subset = np.asarray(reference[:rows], dtype=np.float64)
    problem = batch_cost_problem(context.fitted, target_subset, reference_subset)
    budget = allowed_cost(problem, context.qhat)
    if not math.isfinite(budget):
        raise RuntimeError("locked exact MILP diagnostic unexpectedly has infinite budget")
    clean = clean_predictions(context.fitted, reference_subset)
    methods: dict[str, Any] = {}
    for method in EXPECTED_METHODS:
        prediction = frozen_predictions(
            context.fitted, method=method, target_features=target_subset
        )
        certificate = certify_recovery_error(
            problem.cost,
            prediction,
            clean,
            allowed_cost=budget,
            minimum_assignment=problem.minimum_assignment,
            dual_bisection_iterations=int(
                config["optimization"]["dual_bisection_iterations"]
            ),
        )
        exact_assignment, exact = solve_exact_recovery_milp(
            problem.cost,
            prediction,
            clean,
            allowed_cost=budget,
            time_limit_seconds=float(exact_config["time_limit_seconds"]),
        )
        optimum = float(exact["exact_supremum"])
        if not (
            certificate.lower_bound <= optimum + 1e-10
            and optimum <= certificate.upper_bound + 1e-10
        ):
            raise RuntimeError("exact diagnostic optimum escaped numerical bounds")
        methods[method] = {
            "certificate": certificate.summary(),
            "exact": exact,
            "exact_assignment": [int(value) for value in exact_assignment],
            "exact_optimum_contained": True,
        }
    return {
        "status": "PASS",
        "rows": rows,
        "selection": "literal_prefix_of_locked_evaluation_batch_zero",
        "cost_matrix_sha256": problem.diagnostics["cost_matrix_sha256"],
        "allowed_cost": budget,
        "qhat": _threshold_payload(context.qhat),
        "methods": methods,
    }


def _build_unit(context: PreparedContext, batch_index: int) -> dict[str, Any]:
    config = context.protocol.config
    corruption = str(context.prepared_payload["corruption"])
    batch = context.partition.evaluation[int(batch_index)]
    target, reference = _slice(batch, context.store)
    # Independently permuted endpoint arrays determine the common feasible set.
    # The held-out true assignment is deliberately not accessed until every
    # method-specific optimization bound below has been frozen.
    problem = batch_cost_problem(context.fitted, target, reference)
    budget = allowed_cost(problem, context.qhat)
    clean = clean_predictions(context.fitted, reference)
    prediction_by_method: dict[str, np.ndarray] = {}
    certificate_by_method = {}
    for method in EXPECTED_METHODS:
        prediction = frozen_predictions(
            context.fitted, method=method, target_features=target
        )
        prediction_by_method[method] = prediction
        certificate_by_method[method] = certify_recovery_error(
            problem.cost,
            prediction,
            clean,
            allowed_cost=budget,
            minimum_assignment=problem.minimum_assignment,
            dual_bisection_iterations=int(
                config["optimization"]["dual_bisection_iterations"]
            ),
        )

    true_assignment = np.asarray(batch.true_assignment, dtype=np.int64)
    true_cost, true_score = score_correspondence(problem, true_assignment)
    method_rows: dict[str, Any] = {}
    simultaneous_covered = True
    for method in EXPECTED_METHODS:
        prediction = prediction_by_method[method]
        certificate = certificate_by_method[method]
        true_error = float(np.mean(prediction != clean[true_assignment]))
        covered = bool(true_error <= certificate.upper_bound + 1e-12)
        simultaneous_covered = simultaneous_covered and covered
        method_rows[method] = {
            "prediction_sha256": array_sha256(prediction),
            "clean_prediction_sha256": array_sha256(clean),
            "certificate": certificate.summary(),
            "true_correspondence_error": true_error,
            "bound_covers_true_error": covered,
            "upper_bound_slack_over_true_error": float(
                certificate.upper_bound - true_error
            ),
        }
    true_feasible = bool(
        math.isinf(context.qhat)
        or true_score <= context.qhat
    )
    payload: dict[str, Any] = {
        "schema_version": UNIT_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "study_id": config["study_id"],
        "dataset": "imagenet_c",
        "encoder": "clip",
        "corruption": corruption,
        "severity": int(context.prepared_payload["severity"]),
        "evaluation_batch_index": int(batch_index),
        "batch": batch.summary(),
        "bindings": dict(context.prepared_payload["bindings"]),
        "prepared_payload_sha256": sha256_json(context.prepared_payload),
        "shared_feasible_set": {
            "same_for_all_methods": True,
            "cost_geometry": config["common_cost_geometry"]["name"],
            "cost_matrix_sha256": problem.diagnostics["cost_matrix_sha256"],
            "minimum_assignment_sha256": problem.diagnostics[
                "minimum_assignment_sha256"
            ],
            "minimum_cost": problem.minimum_cost,
            "uniform_random_expected_cost": problem.random_expected_cost,
            "random_minus_minimum_cost": problem.random_minus_minimum,
            "qhat": _threshold_payload(context.qhat),
            "allowed_cost": _threshold_payload(budget),
            "definition": "bijections_with_normalized_cost_score_leq_qhat",
        },
        "post_bound_true_correspondence_scoring": {
            "evaluation_pair_assignment_used_by_map_cost_reward_or_optimization": False,
            "true_assignment": [int(value) for value in true_assignment],
            "true_assignment_sha256": array_sha256(true_assignment),
            "true_cost": true_cost,
            "true_normalized_cost_score": true_score,
            "true_correspondence_feasible": true_feasible,
            "simultaneous_method_bound_coverage": bool(simultaneous_covered),
        },
        "methods": method_rows,
        "ground_truth_class_labels_used": False,
    }
    if int(batch_index) == int(
        config["optimization"]["exact_diagnostic"]["evaluation_batch_index"]
    ):
        payload["exact_diagnostic"] = _exact_diagnostic(
            context, target=target, reference=reference
        )
    else:
        payload["exact_diagnostic"] = None
    return payload


def run_corruption(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    external_input_manifest: str | Path,
    feature_root: str | Path,
    output_root: str | Path,
    corruption: str,
) -> None:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    total = 20
    _write_state(output, corruption, stage="authenticate_and_prepare", completed_units=0, total_units=total)
    context = _prepare_context(
        project_root=project_root,
        config_path=config_path,
        lock_path=lock_path,
        external_input_manifest=external_input_manifest,
        feature_root=feature_root,
        corruption=corruption,
    )
    total = len(context.partition.evaluation)
    prepared_file = output / "prepared" / f"{corruption}.json"
    prepared = _write_or_validate(prepared_file, context.prepared_payload)
    completed = 0
    for index in range(total):
        path = _unit_path(output, corruption, index)
        if path.exists():
            existing = read_json(path)
            if not isinstance(existing, Mapping):
                raise IntegrityError(f"existing unit is malformed: {path}")
            verify_manifest_checksum(existing)
            if (
                existing.get("schema_version") != UNIT_SCHEMA
                or existing.get("status") != "PASS"
                or existing.get("study_id")
                != context.protocol.config["study_id"]
                or existing.get("corruption") != corruption
                or int(existing.get("evaluation_batch_index", -1)) != index
                or existing.get("batch")
                != context.partition.evaluation[index].summary()
                or existing.get("bindings")
                != context.prepared_payload["bindings"]
                or existing.get("prepared_payload_sha256")
                != sha256_json(context.prepared_payload)
                or not isinstance(existing.get("methods"), Mapping)
                or tuple(existing["methods"].keys()) != EXPECTED_METHODS
                or not isinstance(existing.get("shared_feasible_set"), Mapping)
                or existing["shared_feasible_set"].get("same_for_all_methods")
                is not True
                or existing["shared_feasible_set"].get("cost_geometry")
                != context.protocol.config["common_cost_geometry"]["name"]
                or (existing.get("exact_diagnostic") is None)
                != (
                    index
                    != int(
                        context.protocol.config["optimization"][
                            "exact_diagnostic"
                        ]["evaluation_batch_index"]
                    )
                )
            ):
                raise IntegrityError(f"existing unit is foreign or incomplete: {path}")
        else:
            _write_or_validate(path, _build_unit(context, index))
        completed += 1
        _write_state(
            output,
            corruption,
            stage="evaluation_bounds",
            completed_units=completed,
            total_units=total,
        )
        print(f"[{corruption}] completed {completed}/{total} units", flush=True)
    unit_records = [
        {
            "path": str(_unit_path(output, corruption, index).relative_to(output)),
            "sha256": sha256_file(_unit_path(output, corruption, index)),
        }
        for index in range(total)
    ]
    _write_or_validate(
        output / "corruption_done" / f"{corruption}.json",
        {
            "schema_version": CORRUPTION_DONE_SCHEMA,
            "status": "PASS",
            "study_id": context.protocol.config["study_id"],
            "corruption": corruption,
            "prepared_path": str(prepared_file.relative_to(output)),
            "prepared_file_sha256": sha256_file(prepared_file),
            "prepared_manifest_sha256": prepared["manifest_sha256"],
            "complete_unit_count": total,
            "units": unit_records,
        },
    )
    _write_state(
        output,
        corruption,
        stage="complete",
        completed_units=total,
        total_units=total,
        status="PASS",
    )


def _validated_unit_inventory(
    output_root: Path, protocol: LockedProtocol
) -> list[Mapping[str, Any]]:
    units = []
    all_expected_paths = {
        _unit_path(output_root, corruption, index).resolve()
        for corruption in EXPECTED_CORRUPTIONS
        for index in range(20)
    }
    units_root = output_root / "units"
    all_observed_paths = (
        {
            path.resolve()
            for path in units_root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if units_root.is_dir()
        else set()
    )
    if all_observed_paths != all_expected_paths:
        raise IntegrityError("unit directory has missing, extra, or foreign files")
    for corruption in EXPECTED_CORRUPTIONS:
        done_path = output_root / "corruption_done" / f"{corruption}.json"
        done = read_json(done_path)
        if not isinstance(done, Mapping):
            raise IntegrityError(f"corruption completion marker is malformed: {corruption}")
        verify_manifest_checksum(done)
        if (
            done.get("schema_version") != CORRUPTION_DONE_SCHEMA
            or done.get("status") != "PASS"
            or done.get("study_id") != protocol.config["study_id"]
            or done.get("corruption") != corruption
            or int(done.get("complete_unit_count", -1)) != 20
        ):
            raise IntegrityError(f"corruption completion marker changed: {corruption}")
        records = done.get("units")
        if not isinstance(records, list) or len(records) != 20:
            raise IntegrityError(f"corruption unit index is incomplete: {corruption}")
        prepared_path = output_root / "prepared" / f"{corruption}.json"
        prepared = read_json(prepared_path)
        if not isinstance(prepared, Mapping):
            raise IntegrityError(f"prepared artifact is malformed: {corruption}")
        verify_manifest_checksum(prepared)
        if (
            prepared.get("schema_version") != PREPARED_SCHEMA
            or prepared.get("status") != "PASS"
            or prepared.get("study_id") != protocol.config["study_id"]
            or prepared.get("corruption") != corruption
            or done.get("prepared_path")
            != str(prepared_path.relative_to(output_root))
            or done.get("prepared_file_sha256") != sha256_file(prepared_path)
            or done.get("prepared_manifest_sha256")
            != prepared.get("manifest_sha256")
        ):
            raise IntegrityError(f"prepared artifact binding changed: {corruption}")
        expected_paths = {
            str(_unit_path(output_root, corruption, index).relative_to(output_root))
            for index in range(20)
        }
        observed_paths = {str(row.get("path")) for row in records if isinstance(row, Mapping)}
        if observed_paths != expected_paths:
            raise IntegrityError(f"corruption unit path inventory changed: {corruption}")
        for record in records:
            if not isinstance(record, Mapping):
                raise IntegrityError("unit index contains a malformed record")
            path = output_root / str(record["path"])
            if not path.is_file() or path.is_symlink():
                raise IntegrityError(f"unit path is absent or symbolic: {path}")
            if sha256_file(path) != record.get("sha256"):
                raise IntegrityError(f"unit file hash changed: {path}")
            unit = read_json(path)
            if not isinstance(unit, Mapping):
                raise IntegrityError(f"unit is malformed: {path}")
            verify_manifest_checksum(unit)
            if (
                unit.get("schema_version") != UNIT_SCHEMA
                or unit.get("status") != "PASS"
                or unit.get("study_id") != protocol.config["study_id"]
                or unit.get("corruption") != corruption
                or unit.get("bindings") != prepared.get("bindings")
                or unit.get("prepared_payload_sha256")
                != sha256_json(
                    {
                        key: value
                        for key, value in prepared.items()
                        if key != "manifest_sha256"
                    }
                )
                or not isinstance(unit.get("methods"), Mapping)
                or tuple(unit["methods"].keys()) != EXPECTED_METHODS
            ):
                raise IntegrityError(f"unit identity changed: {path}")
            units.append(unit)
    identities = {
        (str(unit["corruption"]), int(unit["evaluation_batch_index"]))
        for unit in units
    }
    expected = {
        (corruption, index)
        for corruption in EXPECTED_CORRUPTIONS
        for index in range(20)
    }
    if identities != expected or len(units) != len(expected):
        raise IntegrityError("complete unit grid is missing, duplicated, or foreign")
    return sorted(
        units, key=lambda row: (str(row["corruption"]), int(row["evaluation_batch_index"]))
    )


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise IntegrityError("aggregate metric vector is empty or nonfinite")
    return {
        "mean": float(array.mean(dtype=np.float64)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _method_summary(units: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    rows = [unit["methods"][method] for unit in units]
    certificates = [row["certificate"] for row in rows]
    return {
        "evaluation_batches": len(rows),
        "bound_coverage_count": int(sum(bool(row["bound_covers_true_error"]) for row in rows)),
        "bound_coverage_rate": float(np.mean([bool(row["bound_covers_true_error"]) for row in rows])),
        "true_correspondence_error": _stats(
            [float(row["true_correspondence_error"]) for row in rows]
        ),
        "bound_lower": _stats([float(row["lower_bound"]) for row in certificates]),
        "bound_upper": _stats([float(row["upper_bound"]) for row in certificates]),
        "bound_width": _stats(
            [float(row["optimization_width"]) for row in certificates]
        ),
        "upper_bound_slack_over_true_error": _stats(
            [float(row["upper_bound_slack_over_true_error"]) for row in rows]
        ),
        "unconstrained_label_count_bound": _stats(
            [float(row["unconstrained_label_count_bound"]) for row in certificates]
        ),
        "certified_tightening_over_unconstrained": _stats(
            [
                float(row["certified_tightening_over_unconstrained"])
                for row in certificates
            ]
        ),
    }


def _aggregate_payload(
    protocol: LockedProtocol, units: Sequence[Mapping[str, Any]], output_root: Path
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for corruption in EXPECTED_CORRUPTIONS:
        subset = [unit for unit in units if unit["corruption"] == corruption]
        feasibility = [
            bool(unit["post_bound_true_correspondence_scoring"]["true_correspondence_feasible"])
            for unit in subset
        ]
        simultaneous = [
            bool(unit["post_bound_true_correspondence_scoring"]["simultaneous_method_bound_coverage"])
            for unit in subset
        ]
        prepared = read_json(output_root / "prepared" / f"{corruption}.json")
        verify_manifest_checksum(prepared)
        summaries[corruption] = {
            "evaluation_batches": len(subset),
            "qhat": prepared["calibration"]["qhat"],
            "true_correspondence_feasibility_count": int(sum(feasibility)),
            "true_correspondence_feasibility_rate": float(np.mean(feasibility)),
            "simultaneous_three_method_bound_coverage_count": int(sum(simultaneous)),
            "simultaneous_three_method_bound_coverage_rate": float(
                np.mean(simultaneous)
            ),
            "methods": {
                method: _method_summary(subset, method)
                for method in EXPECTED_METHODS
            },
        }
    pooled_feasibility = [
        bool(unit["post_bound_true_correspondence_scoring"]["true_correspondence_feasible"])
        for unit in units
    ]
    pooled_simultaneous = [
        bool(unit["post_bound_true_correspondence_scoring"]["simultaneous_method_bound_coverage"])
        for unit in units
    ]
    exact = [unit["exact_diagnostic"] for unit in units if unit["exact_diagnostic"] is not None]
    if len(exact) != 2 or any(row.get("status") != "PASS" for row in exact):
        raise IntegrityError("locked exact diagnostic inventory is incomplete")
    exact_method_checks = sum(
        int(row["methods"][method]["exact_optimum_contained"])
        for row in exact
        for method in EXPECTED_METHODS
    )
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "study_id": protocol.config["study_id"],
        "design": {
            "dataset": "imagenet_c",
            "encoder": "clip",
            "corruptions": list(EXPECTED_CORRUPTIONS),
            "severity": 5,
            "methods": list(EXPECTED_METHODS),
            "map_fit_rows_per_corruption": 4096,
            "calibration_batches_per_corruption": 29,
            "evaluation_batches_per_corruption": 20,
            "batch_rows": 256,
            "delta": 0.1,
            "shared_feasible_set_within_batch": True,
        },
        "bindings": {
            "config_file_sha256": protocol.config_file_sha256,
            "lock_file_sha256": protocol.lock_file_sha256,
            "lock_manifest_sha256": protocol.lock["manifest_sha256"],
            "external_input_manifest_file_sha256": protocol.config[
                "external_input_manifest"
            ]["file_sha256"],
        },
        "complete_evaluation_units": len(units),
        "by_corruption": summaries,
        "pooled_descriptive": {
            "evaluation_batches": len(units),
            "true_correspondence_feasibility_count": int(sum(pooled_feasibility)),
            "true_correspondence_feasibility_rate": float(np.mean(pooled_feasibility)),
            "simultaneous_three_method_bound_coverage_count": int(
                sum(pooled_simultaneous)
            ),
            "simultaneous_three_method_bound_coverage_rate": float(
                np.mean(pooled_simultaneous)
            ),
            "methods": {
                method: _method_summary(units, method)
                for method in EXPECTED_METHODS
            },
        },
        "exact_diagnostics": {
            "corruption_diagnostics": len(exact),
            "method_checks": exact_method_checks,
            "all_exact_optima_inside_numerical_bounds": exact_method_checks == 6,
        },
        "interpretation": {
            "coverage_alone_is_sufficient": False,
            "simultaneity_source": "one_shared_true_correspondence_feasibility_event",
            "not_external_target_label_accuracy": True,
            "not_pure_marginal": True,
            "not_natural_shift": True,
            "not_unpaired": True,
            "fixed_two_corruption_feasibility_panel_not_population_inference": True,
        },
        "unit_file_sha256s": {
            str(_unit_path(output_root, str(unit["corruption"]), int(unit["evaluation_batch_index"])).relative_to(output_root)): sha256_file(
                _unit_path(output_root, str(unit["corruption"]), int(unit["evaluation_batch_index"]))
            )
            for unit in units
        },
    }


def _checksum_payload(output_root: Path, paths: Sequence[Path]) -> bytes:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root)}"
        for path in sorted(paths, key=lambda value: str(value.relative_to(output_root)))
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_checksums(output_root: Path, paths: Sequence[Path]) -> None:
    payload = _checksum_payload(output_root, paths)
    destination = output_root / "RESULT_SHA256SUMS"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def _indexed_result_paths(output_root: Path) -> list[Path]:
    return [
        output_root / "prepared" / f"{corruption}.json"
        for corruption in EXPECTED_CORRUPTIONS
    ] + [
        _unit_path(output_root, corruption, index)
        for corruption in EXPECTED_CORRUPTIONS
        for index in range(20)
    ] + [
        output_root / "corruption_done" / f"{corruption}.json"
        for corruption in EXPECTED_CORRUPTIONS
    ] + [output_root / "AGGREGATE.json"]


def _done_payload(
    protocol: LockedProtocol,
    aggregate: Mapping[str, Any],
    output_root: Path,
    indexed_paths: Sequence[Path],
) -> dict[str, Any]:
    return {
        "schema_version": DONE_SCHEMA,
        "status": "PASS",
        "study_id": protocol.config["study_id"],
        "complete_evaluation_units": 40,
        "aggregate_file_sha256": sha256_file(output_root / "AGGREGATE.json"),
        "aggregate_manifest_sha256": aggregate["manifest_sha256"],
        "indexed_result_files": len(indexed_paths),
        "result_set_sha256": sha256_json({
            str(path.relative_to(output_root)): sha256_file(path)
            for path in indexed_paths
        }),
    }


def aggregate_results(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    output_root: str | Path,
) -> Mapping[str, Any]:
    output = Path(output_root).expanduser().resolve()
    protocol = load_locked_protocol(
        project_root, config_path=config_path, lock_path=lock_path
    )
    units = _validated_unit_inventory(output, protocol)
    aggregate = _write_or_validate(
        output / "AGGREGATE.json",
        _aggregate_payload(protocol, units, output),
    )
    indexed_paths = _indexed_result_paths(output)
    done = _write_or_validate(
        output / "DONE.json",
        _done_payload(protocol, aggregate, output, indexed_paths),
    )
    indexed_paths.append(output / "DONE.json")
    _write_checksums(output, indexed_paths)
    return done


def _verify_exact_diagnostic(
    context: PreparedContext,
    unit: Mapping[str, Any],
    *,
    target: np.ndarray,
    reference: np.ndarray,
    rerun_milp: bool,
) -> None:
    diagnostic = unit.get("exact_diagnostic")
    expected_index = int(
        context.protocol.config["optimization"]["exact_diagnostic"][
            "evaluation_batch_index"
        ]
    )
    index = int(unit["evaluation_batch_index"])
    if index != expected_index:
        if diagnostic is not None:
            raise IntegrityError("exact diagnostic appears on an unlocked batch")
        return
    if not isinstance(diagnostic, Mapping) or diagnostic.get("status") != "PASS":
        raise IntegrityError("locked exact diagnostic is absent")
    rows = int(diagnostic["rows"])
    problem = batch_cost_problem(context.fitted, target[:rows], reference[:rows])
    budget = allowed_cost(problem, context.qhat)
    clean = clean_predictions(context.fitted, reference[:rows])
    if problem.diagnostics["cost_matrix_sha256"] != diagnostic["cost_matrix_sha256"]:
        raise IntegrityError("exact diagnostic cost matrix changed")
    for method in EXPECTED_METHODS:
        prediction = frozen_predictions(
            context.fitted, method=method, target_features=target[:rows]
        )
        record = diagnostic["methods"][method]
        verify_certificate(
            record["certificate"],
            problem.cost,
            prediction,
            clean,
            allowed_cost=budget,
        )
        if rerun_milp:
            _, exact = solve_exact_recovery_milp(
                problem.cost,
                prediction,
                clean,
                allowed_cost=budget,
                time_limit_seconds=float(
                    context.protocol.config["optimization"]["exact_diagnostic"][
                        "time_limit_seconds"
                    ]
                ),
            )
            if abs(float(exact["exact_supremum"]) - float(record["exact"]["exact_supremum"])) > 1e-12:
                raise IntegrityError("exact diagnostic optimum changed on reconstruction")


def verify_results(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    external_input_manifest: str | Path,
    feature_root: str | Path,
    output_root: str | Path,
    rerun_milp: bool = True,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    protocol = load_locked_protocol(
        project_root, config_path=config_path, lock_path=lock_path
    )
    units = _validated_unit_inventory(output, protocol)
    unit_by_key = {
        (str(unit["corruption"]), int(unit["evaluation_batch_index"])): unit
        for unit in units
    }
    for corruption in EXPECTED_CORRUPTIONS:
        context = _prepare_context(
            project_root=project_root,
            config_path=config_path,
            lock_path=lock_path,
            external_input_manifest=external_input_manifest,
            feature_root=feature_root,
            corruption=corruption,
        )
        prepared_path = output / "prepared" / f"{corruption}.json"
        prepared = read_json(prepared_path)
        verify_manifest_checksum(prepared)
        if prepared != seal_manifest(context.prepared_payload):
            raise IntegrityError(f"prepared calibration reconstruction changed: {corruption}")
        for index, batch in enumerate(context.partition.evaluation):
            unit = unit_by_key[(corruption, index)]
            target, reference = _slice(batch, context.store)
            problem = batch_cost_problem(context.fitted, target, reference)
            budget = allowed_cost(problem, context.qhat)
            shared = unit["shared_feasible_set"]
            if (
                shared.get("same_for_all_methods") is not True
                or shared.get("cost_matrix_sha256")
                != problem.diagnostics["cost_matrix_sha256"]
                or abs(float(shared["minimum_cost"]) - problem.minimum_cost) > 1e-10
                or _threshold_value(shared["qhat"]) != context.qhat
                or _threshold_value(shared["allowed_cost"]) != budget
            ):
                raise IntegrityError("shared feasible-set reconstruction changed")
            clean = clean_predictions(context.fitted, reference)
            true_record = unit["post_bound_true_correspondence_scoring"]
            true_assignment = np.asarray(batch.true_assignment, dtype=np.int64)
            true_cost, true_score = score_correspondence(
                problem, true_assignment
            )
            true_feasible = bool(
                math.isinf(context.qhat)
                or true_score <= context.qhat
            )
            simultaneous = True
            for method in EXPECTED_METHODS:
                prediction = frozen_predictions(
                    context.fitted, method=method, target_features=target
                )
                record = unit["methods"][method]
                if record["prediction_sha256"] != array_sha256(prediction):
                    raise IntegrityError("method prediction reconstruction changed")
                verify_certificate(
                    record["certificate"],
                    problem.cost,
                    prediction,
                    clean,
                    allowed_cost=budget,
                )
                true_error = float(np.mean(prediction != clean[true_assignment]))
                if abs(true_error - float(record["true_correspondence_error"])) > 1e-12:
                    raise IntegrityError("true correspondence error reconstruction changed")
                covered = bool(true_error <= float(record["certificate"]["upper_bound"]) + 1e-12)
                if covered != bool(record["bound_covers_true_error"]):
                    raise IntegrityError("method bound coverage reconstruction changed")
                simultaneous = simultaneous and covered
            if (
                true_record.get(
                    "evaluation_pair_assignment_used_by_map_cost_reward_or_optimization"
                )
                is not False
                or true_record.get("true_assignment")
                != [int(value) for value in true_assignment]
                or true_record.get("true_assignment_sha256")
                != array_sha256(true_assignment)
                or abs(float(true_record["true_cost"]) - true_cost) > 1e-10
                or abs(
                    float(true_record["true_normalized_cost_score"])
                    - true_score
                )
                > 1e-12
                or true_feasible
                != bool(true_record["true_correspondence_feasible"])
                or simultaneous
                != bool(true_record["simultaneous_method_bound_coverage"])
            ):
                raise IntegrityError("simultaneous coverage reconstruction changed")
            _verify_exact_diagnostic(
                context,
                unit,
                target=target,
                reference=reference,
                rerun_milp=rerun_milp,
            )
    aggregate = read_json(output / "AGGREGATE.json")
    verify_manifest_checksum(aggregate)
    expected_aggregate = seal_manifest(_aggregate_payload(protocol, units, output))
    if aggregate != expected_aggregate:
        raise IntegrityError("aggregate differs from complete deterministic reaggregation")
    done = read_json(output / "DONE.json")
    verify_manifest_checksum(done)
    indexed_paths = _indexed_result_paths(output)
    expected_done = seal_manifest(
        _done_payload(protocol, aggregate, output, indexed_paths)
    )
    if done != expected_done:
        raise IntegrityError("completion marker does not bind the verified aggregate")
    checksum_path = output / "RESULT_SHA256SUMS"
    if (
        not checksum_path.is_file()
        or checksum_path.is_symlink()
        or checksum_path.read_bytes()
        != _checksum_payload(output, [*indexed_paths, output / "DONE.json"])
    ):
        raise IntegrityError("result checksum index differs from verified result set")
    return {
        "schema_version": "marginal_alignment.paired_recovery_bounds.verification.v1",
        "status": "PASS",
        "source_lock_verified": True,
        "external_input_manifest_verified": True,
        "feature_store_manifests_and_arrays_verified": True,
        "prepared_calibrations_reconstructed": 2,
        "evaluation_units_reconstructed": 40,
        "method_certificates_revalidated": 120,
        "exact_milp_diagnostics_rerun": bool(rerun_milp),
        "deterministic_aggregate_rederived": True,
        "done_file_sha256": sha256_file(output / "DONE.json"),
        "result_checksum_index_sha256": sha256_file(checksum_path),
        "runtime": runtime_identity(),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/paired_recovery_bounds/imagenet_c_clip_v1.json",
    )
    parser.add_argument(
        "--lock",
        default="protocol/paired_recovery_bounds/STUDY_LOCK_20260913.json",
    )
    parser.add_argument("--output-root", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the locked paired conformal recovery-bound study"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-corruption")
    _common_arguments(run)
    run.add_argument("--external-input-manifest", required=True)
    run.add_argument("--feature-root", required=True)
    run.add_argument("--corruption", choices=EXPECTED_CORRUPTIONS, required=True)
    aggregate = subparsers.add_parser("aggregate")
    _common_arguments(aggregate)
    verify = subparsers.add_parser("verify")
    _common_arguments(verify)
    verify.add_argument("--external-input-manifest", required=True)
    verify.add_argument("--feature-root", required=True)
    verify.add_argument("--skip-milp-rerun", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "run-corruption":
        run_corruption(
            project_root=arguments.project_root,
            config_path=arguments.config,
            lock_path=arguments.lock,
            external_input_manifest=arguments.external_input_manifest,
            feature_root=arguments.feature_root,
            output_root=arguments.output_root,
            corruption=arguments.corruption,
        )
    elif arguments.command == "aggregate":
        aggregate_results(
            project_root=arguments.project_root,
            config_path=arguments.config,
            lock_path=arguments.lock,
            output_root=arguments.output_root,
        )
    else:
        report = verify_results(
            project_root=arguments.project_root,
            config_path=arguments.config,
            lock_path=arguments.lock,
            external_input_manifest=arguments.external_input_manifest,
            feature_root=arguments.feature_root,
            output_root=arguments.output_root,
            rerun_milp=not arguments.skip_milp_rerun,
        )
        write_json_atomic(
            Path(arguments.output_root) / "VERIFICATION.json",
            seal_manifest(report),
        )
        print("paired recovery-bound verification: PASS", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
