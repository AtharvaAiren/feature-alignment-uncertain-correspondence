"""Resumable runner and full-reconstruction verifier for the frozen follow-up."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

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
from marginal_alignment.campaign.feature_store import FeatureStore
from marginal_alignment.paired_recovery_bounds.conformal import split_conformal_quantile
from marginal_alignment.paired_recovery_bounds.geometry import (
    FittedRecoveryGeometry,
    allowed_cost,
    array_sha256,
    batch_cost_problem,
    clean_predictions,
    fit_recovery_geometry,
    frozen_predictions,
    score_correspondence,
)
from marginal_alignment.paired_recovery_bounds.optimization import (
    certify_recovery_error,
    solve_exact_recovery_milp,
)
from marginal_alignment.paired_recovery_bounds.splitting import (
    RecoveryBatch,
    RecoveryPartition,
    build_recovery_partition,
)

from .calibration import DirectCalibration, calibrate_direct_recovery_bounds
from .controls import column_permutation, column_permuted_cost_problem
from .protocol import (
    EXPECTED_CELL_KEYS,
    EXPECTED_CONTRASTS,
    EXPECTED_METHODS,
    LockedFollowupProtocol,
    load_authenticated_store,
    load_locked_protocol,
    runtime_identity,
    verify_external_input_manifest,
)
from .signed_optimization import (
    certify_linear_reward,
    signed_contrast_reward_matrix,
    solve_exact_linear_reward_milp,
)


PREPARED_SCHEMA = "marginal_alignment.paired_recovery_followup.prepared.v1"
UNIT_SCHEMA = "marginal_alignment.paired_recovery_followup.unit.v1"
CELL_DONE_SCHEMA = "marginal_alignment.paired_recovery_followup.cell_done.v1"
AGGREGATE_SCHEMA = "marginal_alignment.paired_recovery_followup.aggregate.v1"
DONE_SCHEMA = "marginal_alignment.paired_recovery_followup.done.v1"
STATE_SCHEMA = "marginal_alignment.paired_recovery_followup.operational_state.v1"


@dataclass(frozen=True)
class PreparedContext:
    protocol: LockedFollowupProtocol
    cell: Mapping[str, Any]
    store: FeatureStore
    partition: RecoveryPartition
    fitted: FittedRecoveryGeometry
    physical_qhat: float
    control_qhat: float
    direct: DirectCalibration
    prepared_payload: Mapping[str, Any]

    @property
    def cell_id(self) -> str:
        return cell_id(
            encoder=str(self.cell["encoder"]),
            corruption=str(self.cell["corruption"]),
            repetition=int(self.cell["repetition"]),
        )


def cell_id(*, encoder: str, corruption: str, repetition: int) -> str:
    return f"{encoder}__{corruption}__rep_{int(repetition):02d}"


def _threshold_payload(value: float) -> dict[str, Any]:
    threshold = float(value)
    if math.isinf(threshold) and threshold > 0.0:
        return {"kind": "positive_infinity", "value": None}
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite or positive infinity")
    return {"kind": "finite", "value": threshold}


def _slice(batch: RecoveryBatch, store: FeatureStore) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(
        store.endpoint("reference").features[batch.reference_rows], dtype=np.float64
    )
    target = np.asarray(
        store.endpoint("target").features[batch.target_rows], dtype=np.float64
    )
    if reference.shape != target.shape or len(reference) != batch.size:
        raise IntegrityError("selected endpoint slices changed shape")
    return target, reference


def _head(store: FeatureStore) -> tuple[np.ndarray, np.ndarray]:
    head = store.manifest.get("head")
    if not isinstance(head, Mapping):
        raise IntegrityError("feature store lacks a frozen readout")
    return (
        np.asarray(store.array(str(head["weight_role"])), dtype=np.float64),
        np.asarray(store.array(str(head["bias_role"])), dtype=np.float64),
    )


def _method_predictions(
    fitted: FittedRecoveryGeometry, target: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        method: frozen_predictions(fitted, method=method, target_features=target)
        for method in EXPECTED_METHODS
    }


def _true_method_errors(
    predictions: Mapping[str, np.ndarray],
    clean: np.ndarray,
    true_assignment: np.ndarray,
) -> dict[str, float]:
    return {
        method: float(np.mean(predictions[method] != clean[true_assignment]))
        for method in EXPECTED_METHODS
    }


def _fit_direct_errors(
    fitted: FittedRecoveryGeometry,
    target: np.ndarray,
    reference: np.ndarray,
    true_assignment: np.ndarray,
    *,
    batch_rows: int,
) -> dict[str, list[float]]:
    predictions = _method_predictions(fitted, target)
    clean = clean_predictions(fitted, reference)
    rows = len(target)
    width = int(batch_rows)
    if rows % width != 0:
        raise IntegrityError("map-fit role cannot be divided into locked direct batches")
    result = {method: [] for method in EXPECTED_METHODS}
    for start in range(0, rows, width):
        stop = start + width
        target_positions = np.arange(start, stop, dtype=np.int64)
        matched_reference = true_assignment[target_positions]
        for method in EXPECTED_METHODS:
            result[method].append(
                float(
                    np.mean(
                        predictions[method][target_positions]
                        != clean[matched_reference]
                    )
                )
            )
    return result


def _prepare_context(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    external_input_manifest: str | Path,
    feature_root: str | Path,
    encoder: str,
    corruption: str,
    repetition: int,
) -> PreparedContext:
    protocol = load_locked_protocol(
        project_root, config_path=config_path, lock_path=lock_path
    )
    verify_external_input_manifest(protocol, external_input_manifest)
    cell = protocol.cell(
        encoder=encoder, corruption=corruption, repetition=int(repetition)
    )
    store = load_authenticated_store(
        protocol,
        feature_root=feature_root,
        encoder=encoder,
        corruption=corruption,
    )
    config = protocol.config
    partition_config = config["partition"]
    partition = build_recovery_partition(
        store,
        seed=int(cell["split_seed"]),
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
        covariance_relative_floor=float(geometry_config["covariance_relative_floor"]),
        diagonal_standard_deviation_floor=float(
            geometry_config["diagonal_standard_deviation_floor"]
        ),
    )

    # Pair assignment is first accessed after all correction maps are frozen.
    fit_errors = _fit_direct_errors(
        fitted,
        target_fit,
        reference_fit,
        np.asarray(partition.map_fit.true_assignment, dtype=np.int64),
        batch_rows=int(partition_config["batch_rows"]),
    )
    physical_scores: list[float] = []
    control_scores: list[float] = []
    direct_error_rows: list[dict[str, float]] = []
    calibration_rows: list[dict[str, Any]] = []
    for index, batch in enumerate(partition.calibration):
        target, reference = _slice(batch, store)
        true_assignment = np.asarray(batch.true_assignment, dtype=np.int64)
        physical = batch_cost_problem(
            fitted,
            target,
            reference,
            true_assignment=true_assignment,
        )
        permutation = column_permutation(
            rows=batch.size,
            master_seed=int(cell["control_seed"]),
            key=f"calibration/{index}",
        )
        control = column_permuted_cost_problem(
            physical, permutation, true_assignment=true_assignment
        )
        predictions = _method_predictions(fitted, target)
        clean = clean_predictions(fitted, reference)
        errors = _true_method_errors(predictions, clean, true_assignment)
        physical_scores.append(physical.true_normalized_score)
        control_scores.append(control.true_normalized_score)
        direct_error_rows.append(errors)
        calibration_rows.append({
            "batch_index": int(index),
            "batch_identity_sha256": batch.identity_sha256,
            "physical_true_normalized_cost_score": physical.true_normalized_score,
            "control_true_normalized_cost_score": control.true_normalized_score,
            "physical_cost_matrix_sha256": physical.diagnostics["cost_matrix_sha256"],
            "control_cost_matrix_sha256": control.diagnostics["cost_matrix_sha256"],
            "control_column_permutation_sha256": control.diagnostics[
                "column_permutation_sha256"
            ],
            "direct_true_method_errors": errors,
        })
    delta = float(config["conformal"]["delta"])
    physical_qhat, physical_diagnostics = split_conformal_quantile(
        physical_scores, delta=delta
    )
    control_qhat, control_diagnostics = split_conformal_quantile(
        control_scores, delta=delta
    )
    direct = calibrate_direct_recovery_bounds(
        fit_errors,
        direct_error_rows,
        delta=delta,
        methods=EXPECTED_METHODS,
        scale_floor=float(config["direct_recovery_baseline"]["scale_floor"]),
    )
    encoder_spec = config["encoders"][encoder]
    payload = {
        "schema_version": PREPARED_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "study_id": config["study_id"],
        "cell_id": cell_id(
            encoder=encoder, corruption=corruption, repetition=int(repetition)
        ),
        "dataset": "imagenet_c",
        "encoder": encoder,
        "corruption": corruption,
        "severity": 5,
        "repetition": int(repetition),
        "phase": str(cell["phase"]),
        "bindings": {
            "config_file_sha256": protocol.config_file_sha256,
            "lock_file_sha256": protocol.lock_file_sha256,
            "lock_manifest_sha256": protocol.lock["manifest_sha256"],
            "external_input_manifest_file_sha256": config[
                "external_input_manifest"
            ]["file_sha256"],
            "feature_store_manifest_file_sha256": sha256_file(store.manifest_path),
            "reference_features_sha256": encoder_spec["reference_features_sha256"],
            "target_features_sha256": encoder_spec["stores"][corruption][
                "target_features_sha256"
            ],
        },
        "partition": partition.summary(),
        "geometry": dict(fitted.diagnostics),
        "direct_baseline": direct.summary(),
        "calibration": {
            "pairing_role": (
                "true_pairing_used_only_after_maps_are_frozen_to_score_costs_and_direct_errors"
            ),
            "rows": calibration_rows,
            "physical_score_vector_sha256": array_sha256(
                np.asarray(physical_scores, dtype=np.float64)
            ),
            "control_score_vector_sha256": array_sha256(
                np.asarray(control_scores, dtype=np.float64)
            ),
            "physical_qhat": _threshold_payload(physical_qhat),
            "control_qhat": _threshold_payload(control_qhat),
            "physical_diagnostics": physical_diagnostics,
            "control_diagnostics": control_diagnostics,
            "control_recalibrated_separately": True,
        },
    }
    return PreparedContext(
        protocol=protocol,
        cell=cell,
        store=store,
        partition=partition,
        fitted=fitted,
        physical_qhat=physical_qhat,
        control_qhat=control_qhat,
        direct=direct,
        prepared_payload=payload,
    )


def _certify_objectives(
    context: PreparedContext,
    *,
    problem: Any,
    predictions: Mapping[str, np.ndarray],
    clean: np.ndarray,
    qhat: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    budget = allowed_cost(problem, qhat)
    iterations = int(
        context.protocol.config["optimization"]["dual_bisection_iterations"]
    )
    methods: dict[str, Any] = {}
    for method in EXPECTED_METHODS:
        certificate = certify_recovery_error(
            problem.cost,
            predictions[method],
            clean,
            allowed_cost=budget,
            minimum_assignment=problem.minimum_assignment,
            dual_bisection_iterations=iterations,
        )
        methods[method] = {
            "prediction_sha256": array_sha256(predictions[method]),
            "clean_prediction_sha256": array_sha256(clean),
            "certificate": certificate.summary(),
        }
    contrasts: dict[str, Any] = {}
    for contrast_id, method_a, method_c in EXPECTED_CONTRASTS:
        reward = signed_contrast_reward_matrix(
            predictions[method_a], predictions[method_c], clean
        )
        certificate = certify_linear_reward(
            problem.cost,
            reward,
            allowed_cost=budget,
            minimum_assignment=problem.minimum_assignment,
            dual_bisection_iterations=iterations,
        )
        contrasts[contrast_id] = {
            "method_a": method_a,
            "method_c": method_c,
            "reward_matrix_sha256": array_sha256(reward),
            "signed_rewards_clipped": False,
            "certificate": certificate.summary(),
        }
    return methods, contrasts


def _exact_diagnostic(
    context: PreparedContext,
    *,
    batch_index: int,
    target: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    exact_config = context.protocol.config["optimization"]["exact_diagnostic"]
    rows = int(exact_config["rows"])
    target_subset = np.asarray(target[:rows], dtype=np.float64)
    reference_subset = np.asarray(reference[:rows], dtype=np.float64)
    physical = batch_cost_problem(context.fitted, target_subset, reference_subset)
    permutation = column_permutation(
        rows=rows,
        master_seed=int(context.cell["control_seed"]),
        key=f"exact/{int(batch_index)}",
    )
    control = column_permuted_cost_problem(physical, permutation)
    predictions = _method_predictions(context.fitted, target_subset)
    clean = clean_predictions(context.fitted, reference_subset)
    result: dict[str, Any] = {
        "status": "PASS",
        "rows": rows,
        "selection": "literal_target_and_reference_prefix_of_evaluation_batch_zero",
        "geometries": {},
    }
    for geometry_name, problem, qhat in (
        ("physical", physical, context.physical_qhat),
        ("column_permuted", control, context.control_qhat),
    ):
        budget = allowed_cost(problem, qhat)
        if not math.isfinite(budget):
            raise RuntimeError("exact diagnostic unexpectedly has an infinite budget")
        method_rows: dict[str, Any] = {}
        for method in EXPECTED_METHODS:
            numerical = certify_recovery_error(
                problem.cost,
                predictions[method],
                clean,
                allowed_cost=budget,
                minimum_assignment=problem.minimum_assignment,
                dual_bisection_iterations=int(
                    context.protocol.config["optimization"][
                        "dual_bisection_iterations"
                    ]
                ),
            )
            exact_assignment, exact = solve_exact_recovery_milp(
                problem.cost,
                predictions[method],
                clean,
                allowed_cost=budget,
                time_limit_seconds=float(exact_config["time_limit_seconds"]),
            )
            optimum = float(exact["exact_supremum"])
            if not (
                numerical.lower_bound <= optimum + 1e-10
                and optimum <= numerical.upper_bound + 1e-10
            ):
                raise RuntimeError("absolute exact optimum escaped numerical bracket")
            method_rows[method] = {
                "numerical": numerical.summary(),
                "exact": exact,
                "exact_assignment": [int(value) for value in exact_assignment],
                "exact_optimum_contained": True,
            }
        contrast_rows: dict[str, Any] = {}
        for contrast_id, method_a, method_c in EXPECTED_CONTRASTS:
            reward = signed_contrast_reward_matrix(
                predictions[method_a], predictions[method_c], clean
            )
            numerical = certify_linear_reward(
                problem.cost,
                reward,
                allowed_cost=budget,
                minimum_assignment=problem.minimum_assignment,
                dual_bisection_iterations=int(
                    context.protocol.config["optimization"][
                        "dual_bisection_iterations"
                    ]
                ),
            )
            exact_assignment, exact = solve_exact_linear_reward_milp(
                problem.cost,
                reward,
                allowed_cost=budget,
                time_limit_seconds=float(exact_config["time_limit_seconds"]),
            )
            optimum = float(exact["exact_supremum"])
            if not (
                numerical.lower_bound <= optimum + 1e-10
                and optimum <= numerical.upper_bound + 1e-10
            ):
                raise RuntimeError("signed exact optimum escaped numerical bracket")
            contrast_rows[contrast_id] = {
                "method_a": method_a,
                "method_c": method_c,
                "numerical": numerical.summary(),
                "exact": exact,
                "exact_assignment": [int(value) for value in exact_assignment],
                "exact_optimum_contained": True,
            }
        result["geometries"][geometry_name] = {
            "cost_matrix_sha256": problem.diagnostics["cost_matrix_sha256"],
            "qhat": _threshold_payload(qhat),
            "allowed_cost": _threshold_payload(budget),
            "methods": method_rows,
            "contrasts": contrast_rows,
        }
    return result


def _build_unit(context: PreparedContext, batch_index: int) -> dict[str, Any]:
    config = context.protocol.config
    index = int(batch_index)
    batch = context.partition.evaluation[index]
    target, reference = _slice(batch, context.store)
    # Construct every bound before reading the held-out assignment below.
    physical = batch_cost_problem(context.fitted, target, reference)
    permutation = column_permutation(
        rows=batch.size,
        master_seed=int(context.cell["control_seed"]),
        key=f"evaluation/{index}",
    )
    control = column_permuted_cost_problem(physical, permutation)
    predictions = _method_predictions(context.fitted, target)
    clean = clean_predictions(context.fitted, reference)
    physical_methods, physical_contrasts = _certify_objectives(
        context,
        problem=physical,
        predictions=predictions,
        clean=clean,
        qhat=context.physical_qhat,
    )
    control_methods, control_contrasts = _certify_objectives(
        context,
        problem=control,
        predictions=predictions,
        clean=clean,
        qhat=context.control_qhat,
    )
    exact = (
        _exact_diagnostic(
            context,
            batch_index=index,
            target=target,
            reference=reference,
        )
        if index
        == int(config["optimization"]["exact_diagnostic"]["evaluation_batch_index"])
        else None
    )

    # Post-bound scoring begins here. No bound calculation above consumes this
    # assignment; only its opaque partition hash may have been serialized.
    true_assignment = np.asarray(batch.true_assignment, dtype=np.int64)
    physical_true_cost, physical_true_score = score_correspondence(
        physical, true_assignment
    )
    control_true_cost, control_true_score = score_correspondence(
        control, true_assignment
    )
    true_errors = _true_method_errors(predictions, clean, true_assignment)
    direct_upper = context.direct.upper_by_method()
    direct_covered = {
        method: bool(true_errors[method] <= direct_upper[method] + 1e-12)
        for method in EXPECTED_METHODS
    }

    for geometry_methods in (physical_methods, control_methods):
        for method in EXPECTED_METHODS:
            upper = float(geometry_methods[method]["certificate"]["upper_bound"])
            geometry_methods[method]["true_correspondence_error"] = true_errors[method]
            geometry_methods[method]["bound_covers_true_error"] = bool(
                true_errors[method] <= upper + 1e-12
            )
            geometry_methods[method]["upper_bound_slack_over_true_error"] = float(
                upper - true_errors[method]
            )

    true_contrasts: dict[str, float] = {}
    for contrast_id, method_a, method_c in EXPECTED_CONTRASTS:
        true_value = float(true_errors[method_a] - true_errors[method_c])
        true_contrasts[contrast_id] = true_value
        for geometry_contrasts in (physical_contrasts, control_contrasts):
            upper = float(
                geometry_contrasts[contrast_id]["certificate"]["upper_bound"]
            )
            geometry_contrasts[contrast_id]["true_correspondence_contrast"] = true_value
            geometry_contrasts[contrast_id]["upper_bound_covers_true_contrast"] = bool(
                true_value <= upper + 1e-12
            )
            geometry_contrasts[contrast_id]["certifies_method_a_lower_error"] = bool(
                upper < 0.0
            )

    def feasible(score: float, qhat: float) -> bool:
        return bool(math.isinf(qhat) or score <= qhat)

    physical_feasible = feasible(physical_true_score, context.physical_qhat)
    control_feasible = feasible(control_true_score, context.control_qhat)
    physical_joint_method = all(
        bool(physical_methods[method]["bound_covers_true_error"])
        for method in EXPECTED_METHODS
    )
    control_joint_method = all(
        bool(control_methods[method]["bound_covers_true_error"])
        for method in EXPECTED_METHODS
    )
    physical_joint_all = physical_joint_method and all(
        bool(physical_contrasts[contrast_id]["upper_bound_covers_true_contrast"])
        for contrast_id, _, _ in EXPECTED_CONTRASTS
    )
    control_joint_all = control_joint_method and all(
        bool(control_contrasts[contrast_id]["upper_bound_covers_true_contrast"])
        for contrast_id, _, _ in EXPECTED_CONTRASTS
    )
    return {
        "schema_version": UNIT_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "study_id": config["study_id"],
        "cell_id": context.cell_id,
        "dataset": "imagenet_c",
        "encoder": str(context.cell["encoder"]),
        "corruption": str(context.cell["corruption"]),
        "severity": 5,
        "repetition": int(context.cell["repetition"]),
        "phase": str(context.cell["phase"]),
        "evaluation_batch_index": index,
        "batch": batch.summary(),
        "bindings": dict(context.prepared_payload["bindings"]),
        "prepared_payload_sha256": sha256_json(context.prepared_payload),
        "shared_feasible_sets": {
            "physical": {
                "same_for_all_absolute_and_signed_objectives": True,
                "cost_geometry": config["common_cost_geometry"]["name"],
                "cost_matrix_sha256": physical.diagnostics["cost_matrix_sha256"],
                "minimum_assignment_sha256": physical.diagnostics[
                    "minimum_assignment_sha256"
                ],
                "minimum_cost": physical.minimum_cost,
                "uniform_random_expected_cost": physical.random_expected_cost,
                "random_minus_minimum_cost": physical.random_minus_minimum,
                "qhat": _threshold_payload(context.physical_qhat),
                "allowed_cost": _threshold_payload(
                    allowed_cost(physical, context.physical_qhat)
                ),
            },
            "column_permuted": {
                "same_for_all_absolute_and_signed_objectives": True,
                "cost_geometry": "column_permutation_of_physical_geometry",
                "cost_matrix_sha256": control.diagnostics["cost_matrix_sha256"],
                "minimum_assignment_sha256": control.diagnostics[
                    "minimum_assignment_sha256"
                ],
                "column_permutation_sha256": control.diagnostics[
                    "column_permutation_sha256"
                ],
                "minimum_cost": control.minimum_cost,
                "uniform_random_expected_cost": control.random_expected_cost,
                "random_minus_minimum_cost": control.random_minus_minimum,
                "qhat": _threshold_payload(context.control_qhat),
                "allowed_cost": _threshold_payload(
                    allowed_cost(control, context.control_qhat)
                ),
                "threshold_recalibrated": True,
                "physical_rewards_and_true_assignment_reindexed": False,
            },
        },
        "direct_baseline": {
            "upper_bounds": direct_upper,
            "true_correspondence_errors": true_errors,
            "bound_covers_true_error": direct_covered,
            "simultaneous_method_bound_coverage": all(direct_covered.values()),
            "upper_bound_slack_over_true_error": {
                method: float(direct_upper[method] - true_errors[method])
                for method in EXPECTED_METHODS
            },
        },
        "physical_geometry": {
            "methods": physical_methods,
            "contrasts": physical_contrasts,
        },
        "column_permuted_geometry": {
            "methods": control_methods,
            "contrasts": control_contrasts,
        },
        "post_bound_true_correspondence_scoring": {
            "evaluation_pair_assignment_consumed_by_bound_construction_before_all_bounds_were_frozen": False,
            "true_assignment": [int(value) for value in true_assignment],
            "true_assignment_sha256": array_sha256(true_assignment),
            "true_method_errors": true_errors,
            "true_signed_contrasts": true_contrasts,
            "physical": {
                "true_cost": physical_true_cost,
                "true_normalized_cost_score": physical_true_score,
                "true_correspondence_feasible": physical_feasible,
                "simultaneous_method_bound_coverage": physical_joint_method,
                "simultaneous_all_absolute_and_contrast_upper_coverage": physical_joint_all,
            },
            "column_permuted": {
                "true_cost": control_true_cost,
                "true_normalized_cost_score": control_true_score,
                "true_correspondence_feasible": control_feasible,
                "simultaneous_method_bound_coverage": control_joint_method,
                "simultaneous_all_absolute_and_contrast_upper_coverage": control_joint_all,
            },
        },
        "exact_diagnostic": exact,
        "ground_truth_class_labels_used": False,
    }


def _prepared_path(output_root: Path, identifier: str) -> Path:
    return output_root / "prepared" / f"{identifier}.json"


def _unit_path(output_root: Path, identifier: str, batch_index: int) -> Path:
    return output_root / "units" / identifier / f"batch_{int(batch_index):03d}.json"


def _cell_done_path(output_root: Path, identifier: str) -> Path:
    return output_root / "cell_done" / f"{identifier}.json"


def _state_path(output_root: Path, identifier: str) -> Path:
    return output_root / "state" / f"{identifier}.json"


def _resolve_output_root(value: str | Path, *, create: bool) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise IntegrityError("output root must not be symbolic")
    output = supplied.resolve()
    if create:
        output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir() or output.is_symlink():
        raise IntegrityError("output root is absent or not a regular directory")
    return output


def _require_output_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise IntegrityError(
            f"could not create regular output directory: {path}"
        ) from error
    if not path.is_dir() or path.is_symlink():
        raise IntegrityError(f"output directory is not a regular directory: {path}")


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = seal_manifest(payload)
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"existing artifact is not a regular file: {path}")
        existing = read_json(path)
        if not isinstance(existing, Mapping):
            raise IntegrityError(f"existing artifact is malformed: {path}")
        verify_manifest_checksum(existing)
        if existing != expected:
            raise IntegrityError(
                f"existing artifact differs from deterministic reconstruction: {path}"
            )
        return existing
    write_json_atomic(path, expected)
    return expected


def _write_state(
    output_root: Path,
    identifier: str,
    *,
    stage: str,
    completed_units: int,
    total_units: int,
    status: str = "RUNNING",
) -> None:
    write_json_atomic(
        _state_path(output_root, identifier),
        seal_manifest({
            "schema_version": STATE_SCHEMA,
            "status": status,
            "cell_id": identifier,
            "stage": stage,
            "completed_units": int(completed_units),
            "total_units": int(total_units),
            "updated_unix_seconds": int(time.time()),
            "scientific_outcomes_in_state": False,
        }),
    )


def _validate_resumable_unit(
    unit: Mapping[str, Any], context: PreparedContext, batch_index: int
) -> None:
    verify_manifest_checksum(unit)
    index = int(batch_index)
    expected_exact = index == int(
        context.protocol.config["optimization"]["exact_diagnostic"][
            "evaluation_batch_index"
        ]
    )
    if (
        unit.get("schema_version") != UNIT_SCHEMA
        or unit.get("status") != "PASS"
        or unit.get("study_id") != context.protocol.config["study_id"]
        or unit.get("cell_id") != context.cell_id
        or int(unit.get("evaluation_batch_index", -1)) != index
        or unit.get("batch") != context.partition.evaluation[index].summary()
        or unit.get("bindings") != context.prepared_payload["bindings"]
        or unit.get("prepared_payload_sha256")
        != sha256_json(context.prepared_payload)
        or set(_as_mapping(unit.get("physical_geometry"), "physical geometry").get("methods", {}))
        != set(EXPECTED_METHODS)
        or set(_as_mapping(unit.get("column_permuted_geometry"), "control geometry").get("methods", {}))
        != set(EXPECTED_METHODS)
        or (unit.get("exact_diagnostic") is not None) != expected_exact
    ):
        raise IntegrityError("existing unit is foreign or incomplete")


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a JSON object")
    return value


def _cell_done_payload(
    *,
    study_id: str,
    identifier: str,
    encoder: str,
    corruption: str,
    repetition: int,
    phase: str,
    prepared_path: Path,
    prepared: Mapping[str, Any],
    output_root: Path,
    total: int,
) -> dict[str, Any]:
    return {
        "schema_version": CELL_DONE_SCHEMA,
        "status": "PASS",
        "study_id": study_id,
        "cell_id": identifier,
        "encoder": encoder,
        "corruption": corruption,
        "repetition": int(repetition),
        "phase": phase,
        "prepared_path": str(prepared_path.relative_to(output_root)),
        "prepared_file_sha256": sha256_file(prepared_path),
        "prepared_manifest_sha256": prepared["manifest_sha256"],
        "complete_unit_count": int(total),
        "units": [
            {
                "path": str(
                    _unit_path(output_root, identifier, index).relative_to(output_root)
                ),
                "sha256": sha256_file(_unit_path(output_root, identifier, index)),
            }
            for index in range(int(total))
        ],
    }


def run_cell(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    external_input_manifest: str | Path,
    feature_root: str | Path,
    output_root: str | Path,
    encoder: str,
    corruption: str,
    repetition: int,
) -> None:
    output = _resolve_output_root(output_root, create=True)
    identifier = cell_id(
        encoder=encoder, corruption=corruption, repetition=int(repetition)
    )
    for directory in (
        output / "state",
        output / "prepared",
        output / "units",
        output / "cell_done",
        output / "units" / identifier,
    ):
        _require_output_directory(directory)
    total = 20
    _write_state(
        output,
        identifier,
        stage="authenticate_fit_and_calibrate",
        completed_units=0,
        total_units=total,
    )
    context = _prepare_context(
        project_root=project_root,
        config_path=config_path,
        lock_path=lock_path,
        external_input_manifest=external_input_manifest,
        feature_root=feature_root,
        encoder=encoder,
        corruption=corruption,
        repetition=int(repetition),
    )
    prepared_path = _prepared_path(output, identifier)
    prepared = _write_or_validate(prepared_path, context.prepared_payload)
    total = len(context.partition.evaluation)
    for index in range(total):
        path = _unit_path(output, identifier, index)
        if path.exists() or path.is_symlink():
            if not path.is_file() or path.is_symlink():
                raise IntegrityError(f"existing unit is not a regular file: {path}")
            existing = read_json(path)
            _validate_resumable_unit(
                _as_mapping(existing, "existing unit"), context, index
            )
        else:
            _write_or_validate(path, _build_unit(context, index))
        _write_state(
            output,
            identifier,
            stage="evaluation_bounds",
            completed_units=index + 1,
            total_units=total,
        )
        print(f"[{identifier}] completed {index + 1}/{total} units", flush=True)
    _write_or_validate(
        _cell_done_path(output, identifier),
        _cell_done_payload(
            study_id=str(context.protocol.config["study_id"]),
            identifier=identifier,
            encoder=encoder,
            corruption=corruption,
            repetition=int(repetition),
            phase=str(context.cell["phase"]),
            prepared_path=prepared_path,
            prepared=prepared,
            output_root=output,
            total=total,
        ),
    )
    _write_state(
        output,
        identifier,
        stage="complete",
        completed_units=total,
        total_units=total,
        status="PASS",
    )


def _expected_cell_ids() -> tuple[str, ...]:
    return tuple(
        cell_id(encoder=encoder, corruption=corruption, repetition=repetition)
        for encoder, corruption, repetition in EXPECTED_CELL_KEYS
    )


def _require_exact_regular_files(
    directory: Path, expected: set[Path], *, name: str
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise IntegrityError(f"{name} directory is absent or symbolic")
    observed = {
        path
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != expected or any(
        not path.is_file() or path.is_symlink() for path in observed
    ):
        raise IntegrityError(f"{name} directory has missing, extra, or non-regular files")


def _validated_inventory(
    output_root: Path, protocol: LockedFollowupProtocol
) -> list[Mapping[str, Any]]:
    identifiers = _expected_cell_ids()
    expected_unit_paths = {
        _unit_path(output_root, identifier, index)
        for identifier in _expected_cell_ids()
        for index in range(20)
    }
    expected_prepared_paths = {
        _prepared_path(output_root, identifier) for identifier in identifiers
    }
    expected_cell_done_paths = {
        _cell_done_path(output_root, identifier) for identifier in identifiers
    }
    _require_exact_regular_files(
        output_root / "units", expected_unit_paths, name="unit"
    )
    _require_exact_regular_files(
        output_root / "prepared", expected_prepared_paths, name="prepared"
    )
    _require_exact_regular_files(
        output_root / "cell_done", expected_cell_done_paths, name="cell completion"
    )
    units: list[Mapping[str, Any]] = []
    for encoder, corruption, repetition in EXPECTED_CELL_KEYS:
        identifier = cell_id(
            encoder=encoder, corruption=corruption, repetition=repetition
        )
        prepared_path = _prepared_path(output_root, identifier)
        prepared = _as_mapping(read_json(prepared_path), "prepared artifact")
        verify_manifest_checksum(prepared)
        done_path = _cell_done_path(output_root, identifier)
        done = _as_mapping(read_json(done_path), "cell completion marker")
        verify_manifest_checksum(done)
        cell = protocol.cell(
            encoder=encoder, corruption=corruption, repetition=repetition
        )
        expected_done = seal_manifest(_cell_done_payload(
            study_id=str(protocol.config["study_id"]),
            identifier=identifier,
            encoder=encoder,
            corruption=corruption,
            repetition=repetition,
            phase=str(cell["phase"]),
            prepared_path=prepared_path,
            prepared=prepared,
            output_root=output_root,
            total=20,
        ))
        if done != expected_done:
            raise IntegrityError(f"cell completion marker changed: {identifier}")
        records = done.get("units")
        if not isinstance(records, list) or len(records) != 20:
            raise IntegrityError(f"cell unit index is incomplete: {identifier}")
        record_by_path = {
            str(record.get("path")): record
            for record in records
            if isinstance(record, Mapping)
        }
        for index in range(20):
            path = _unit_path(output_root, identifier, index)
            relative = str(path.relative_to(output_root))
            record = record_by_path.get(relative)
            if record is None or record.get("sha256") != sha256_file(path):
                raise IntegrityError(f"unit checksum index changed: {relative}")
            unit = _as_mapping(read_json(path), "unit")
            verify_manifest_checksum(unit)
            if (
                unit.get("schema_version") != UNIT_SCHEMA
                or unit.get("status") != "PASS"
                or unit.get("study_id") != protocol.config["study_id"]
                or unit.get("cell_id") != identifier
                or int(unit.get("evaluation_batch_index", -1)) != index
                or unit.get("bindings") != prepared.get("bindings")
                or unit.get("prepared_payload_sha256")
                != sha256_json(
                    {key: value for key, value in prepared.items() if key != "manifest_sha256"}
                )
            ):
                raise IntegrityError(f"unit identity changed: {relative}")
            units.append(unit)
    identities = {
        (str(unit["cell_id"]), int(unit["evaluation_batch_index"])) for unit in units
    }
    expected_identities = {
        (identifier, index)
        for identifier in _expected_cell_ids()
        for index in range(20)
    }
    if identities != expected_identities or len(units) != 160:
        raise IntegrityError("complete 160-unit grid is missing, duplicated, or foreign")
    return sorted(
        units,
        key=lambda row: (str(row["cell_id"]), int(row["evaluation_batch_index"])),
    )


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise IntegrityError("aggregate metric vector is empty or nonfinite")
    return {
        "mean": float(array.mean(dtype=np.float64)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "standard_deviation_ddof_1": (
            0.0 if len(array) == 1 else float(array.std(ddof=1, dtype=np.float64))
        ),
    }


def _method_summary(
    units: Sequence[Mapping[str, Any]], *, geometry_key: str, method: str
) -> dict[str, Any]:
    rows = [unit[geometry_key]["methods"][method] for unit in units]
    certificates = [row["certificate"] for row in rows]
    return {
        "evaluation_batches": len(rows),
        "coverage_count": int(sum(bool(row["bound_covers_true_error"]) for row in rows)),
        "true_error": _stats([float(row["true_correspondence_error"]) for row in rows]),
        "upper_bound": _stats([float(row["upper_bound"]) for row in certificates]),
        "optimization_width": _stats(
            [float(row["optimization_width"]) for row in certificates]
        ),
        "upper_slack": _stats(
            [float(row["upper_bound_slack_over_true_error"]) for row in rows]
        ),
        "certified_tightening_over_unconstrained": _stats(
            [
                float(row["certified_tightening_over_unconstrained"])
                for row in certificates
            ]
        ),
    }


def _contrast_summary(
    units: Sequence[Mapping[str, Any]], *, geometry_key: str, contrast_id: str
) -> dict[str, Any]:
    rows = [unit[geometry_key]["contrasts"][contrast_id] for unit in units]
    certificates = [row["certificate"] for row in rows]
    return {
        "evaluation_batches": len(rows),
        "upper_coverage_count": int(
            sum(bool(row["upper_bound_covers_true_contrast"]) for row in rows)
        ),
        "certified_lower_error_count": int(
            sum(bool(row["certifies_method_a_lower_error"]) for row in rows)
        ),
        "true_contrast": _stats(
            [float(row["true_correspondence_contrast"]) for row in rows]
        ),
        "upper_bound": _stats([float(row["upper_bound"]) for row in certificates]),
        "optimization_width": _stats(
            [float(row["optimization_width"]) for row in certificates]
        ),
    }


def _direct_summary(units: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    return {
        "evaluation_batches": len(units),
        "coverage_count": int(
            sum(
                bool(unit["direct_baseline"]["bound_covers_true_error"][method])
                for unit in units
            )
        ),
        "upper_bound": _stats(
            [float(unit["direct_baseline"]["upper_bounds"][method]) for unit in units]
        ),
        "upper_slack": _stats(
            [
                float(
                    unit["direct_baseline"]["upper_bound_slack_over_true_error"][method]
                )
                for unit in units
            ]
        ),
    }


def _subset_summary(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not units:
        raise IntegrityError("cannot summarize an empty subset")
    result: dict[str, Any] = {
        "evaluation_batches": len(units),
        "direct_baseline": {
            "simultaneous_method_coverage_count": int(
                sum(
                    bool(unit["direct_baseline"]["simultaneous_method_bound_coverage"])
                    for unit in units
                )
            ),
            "methods": {
                method: _direct_summary(units, method) for method in EXPECTED_METHODS
            },
        },
    }
    for output_key, geometry_key, post_key in (
        ("physical", "physical_geometry", "physical"),
        ("column_permuted", "column_permuted_geometry", "column_permuted"),
    ):
        result[output_key] = {
            "true_correspondence_feasibility_count": int(
                sum(
                    bool(
                        unit["post_bound_true_correspondence_scoring"][post_key][
                            "true_correspondence_feasible"
                        ]
                    )
                    for unit in units
                )
            ),
            "simultaneous_method_coverage_count": int(
                sum(
                    bool(
                        unit["post_bound_true_correspondence_scoring"][post_key][
                            "simultaneous_method_bound_coverage"
                        ]
                    )
                    for unit in units
                )
            ),
            "methods": {
                method: _method_summary(
                    units, geometry_key=geometry_key, method=method
                )
                for method in EXPECTED_METHODS
            },
            "contrasts": {
                contrast_id: _contrast_summary(
                    units, geometry_key=geometry_key, contrast_id=contrast_id
                )
                for contrast_id, _, _ in EXPECTED_CONTRASTS
            },
        }
    return result


def _aggregate_payload(
    protocol: LockedFollowupProtocol,
    units: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for encoder, corruption, repetition in EXPECTED_CELL_KEYS:
        identifier = cell_id(
            encoder=encoder, corruption=corruption, repetition=repetition
        )
        subset = [unit for unit in units if unit["cell_id"] == identifier]
        prepared = _as_mapping(
            read_json(_prepared_path(output_root, identifier)), "prepared artifact"
        )
        verify_manifest_checksum(prepared)
        by_cell[identifier] = {
            "encoder": encoder,
            "corruption": corruption,
            "repetition": repetition,
            "phase": prepared["phase"],
            "physical_qhat": prepared["calibration"]["physical_qhat"],
            "column_permuted_qhat": prepared["calibration"]["control_qhat"],
            "direct_qhat": prepared["direct_baseline"]["threshold"],
            "summary": _subset_summary(subset),
        }
    primary_by_corruption = {}
    for corruption in ("gaussian_noise", "motion_blur"):
        subset = [
            unit
            for unit in units
            if unit["encoder"] == "clip" and unit["corruption"] == corruption
        ]
        primary_by_corruption[corruption] = _subset_summary(subset)
    primary = [unit for unit in units if unit["encoder"] == "clip"]
    secondary = [unit for unit in units if unit["encoder"] == "vitb16"]
    exact = [unit["exact_diagnostic"] for unit in units if unit["exact_diagnostic"]]
    exact_checks = 0
    for diagnostic in exact:
        for geometry in ("physical", "column_permuted"):
            exact_checks += sum(
                int(row["exact_optimum_contained"])
                for row in diagnostic["geometries"][geometry]["methods"].values()
            )
            exact_checks += sum(
                int(row["exact_optimum_contained"])
                for row in diagnostic["geometries"][geometry]["contrasts"].values()
            )
    if len(exact) != 8 or exact_checks != 96:
        raise IntegrityError("exact diagnostic grid is incomplete")
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "study_id": protocol.config["study_id"],
        "design": {
            "dataset": "imagenet_c",
            "severity": 5,
            "corruptions": ["gaussian_noise", "motion_blur"],
            "methods": list(EXPECTED_METHODS),
            "signed_contrasts": [row[0] for row in EXPECTED_CONTRASTS],
            "primary_encoder": "clip",
            "primary_split_repetitions": 3,
            "secondary_encoder": "vitb16",
            "secondary_split_repetitions": 1,
            "cells": 8,
            "evaluation_batch_units": 160,
            "assignment_certificates": 1920,
            "delta": 0.1,
            "geometry_control_recalibrated": True,
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
        "by_cell": by_cell,
        "primary_clip": {
            "pooled_descriptive": _subset_summary(primary),
            "by_corruption_across_three_partitions": primary_by_corruption,
            "identity_overlap_across_repetitions_means_not_independent_datasets": True,
        },
        "secondary_vitb16": {
            "matched_partition_descriptive_only": True,
            "summary": _subset_summary(secondary),
        },
        "exact_diagnostics": {
            "cell_diagnostics": len(exact),
            "absolute_and_signed_checks_across_both_geometries": exact_checks,
            "all_exact_optima_inside_numerical_brackets": exact_checks == 96,
        },
        "interpretation": {
            "direct_baseline_is_future_batch_bound_not_population_mean_interval": True,
            "physical_and_control_coverage_events_are_separate": True,
            "negative_signed_upper_bound_is_required_for_certified_improvement": True,
            "replications_are_not_new_datasets": True,
            "fixed_corruptions_are_not_a_population": True,
            "ground_truth_class_labels_used": False,
        },
        "unit_file_sha256s": {
            str(
                _unit_path(
                    output_root,
                    str(unit["cell_id"]),
                    int(unit["evaluation_batch_index"]),
                ).relative_to(output_root)
            ): sha256_file(
                _unit_path(
                    output_root,
                    str(unit["cell_id"]),
                    int(unit["evaluation_batch_index"]),
                )
            )
            for unit in units
        },
    }


def _indexed_result_paths(output_root: Path) -> list[Path]:
    identifiers = _expected_cell_ids()
    return (
        [_prepared_path(output_root, identifier) for identifier in identifiers]
        + [
            _unit_path(output_root, identifier, index)
            for identifier in identifiers
            for index in range(20)
        ]
        + [_cell_done_path(output_root, identifier) for identifier in identifiers]
        + [output_root / "AGGREGATE.json"]
    )


def _checksum_payload(output_root: Path, paths: Sequence[Path]) -> bytes:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root)}"
        for path in sorted(paths, key=lambda value: str(value.relative_to(output_root)))
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_checksums(output_root: Path, paths: Sequence[Path]) -> None:
    destination = output_root / "RESULT_SHA256SUMS"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(_checksum_payload(output_root, paths))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _done_payload(
    protocol: LockedFollowupProtocol,
    aggregate: Mapping[str, Any],
    output_root: Path,
    indexed_paths: Sequence[Path],
) -> dict[str, Any]:
    return {
        "schema_version": DONE_SCHEMA,
        "status": "PASS",
        "study_id": protocol.config["study_id"],
        "complete_evaluation_units": 160,
        "assignment_certificates": 1920,
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
    output = _resolve_output_root(output_root, create=False)
    protocol = load_locked_protocol(
        project_root, config_path=config_path, lock_path=lock_path
    )
    units = _validated_inventory(output, protocol)
    aggregate = _write_or_validate(
        output / "AGGREGATE.json",
        _aggregate_payload(protocol, units, output),
    )
    indexed = _indexed_result_paths(output)
    done = _write_or_validate(
        output / "DONE.json",
        _done_payload(protocol, aggregate, output, indexed),
    )
    _write_checksums(output, [*indexed, output / "DONE.json"])
    return done


def verify_results(
    *,
    project_root: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    external_input_manifest: str | Path,
    feature_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = _resolve_output_root(output_root, create=False)
    protocol = load_locked_protocol(
        project_root, config_path=config_path, lock_path=lock_path
    )
    units = _validated_inventory(output, protocol)
    unit_by_key = {
        (str(unit["cell_id"]), int(unit["evaluation_batch_index"])): unit
        for unit in units
    }
    for encoder, corruption, repetition in EXPECTED_CELL_KEYS:
        context = _prepare_context(
            project_root=project_root,
            config_path=config_path,
            lock_path=lock_path,
            external_input_manifest=external_input_manifest,
            feature_root=feature_root,
            encoder=encoder,
            corruption=corruption,
            repetition=repetition,
        )
        prepared = _as_mapping(
            read_json(_prepared_path(output, context.cell_id)), "prepared artifact"
        )
        if prepared != seal_manifest(context.prepared_payload):
            raise IntegrityError(
                f"prepared reconstruction changed: {context.cell_id}"
            )
        for index in range(20):
            expected = seal_manifest(_build_unit(context, index))
            observed = unit_by_key[(context.cell_id, index)]
            if observed != expected:
                raise IntegrityError(
                    f"full unit reconstruction changed: {context.cell_id}/{index}"
                )
    aggregate = _as_mapping(read_json(output / "AGGREGATE.json"), "aggregate")
    verify_manifest_checksum(aggregate)
    expected_aggregate = seal_manifest(_aggregate_payload(protocol, units, output))
    if aggregate != expected_aggregate:
        raise IntegrityError("aggregate differs from deterministic reaggregation")
    done = _as_mapping(read_json(output / "DONE.json"), "completion marker")
    verify_manifest_checksum(done)
    indexed = _indexed_result_paths(output)
    expected_done = seal_manifest(_done_payload(protocol, aggregate, output, indexed))
    if done != expected_done:
        raise IntegrityError("completion marker does not bind the verified aggregate")
    checksum_path = output / "RESULT_SHA256SUMS"
    if (
        not checksum_path.is_file()
        or checksum_path.is_symlink()
        or checksum_path.read_bytes()
        != _checksum_payload(output, [*indexed, output / "DONE.json"])
    ):
        raise IntegrityError("result checksum index differs from verified result set")
    return {
        "schema_version": "marginal_alignment.paired_recovery_followup.verification.v1",
        "status": "PASS",
        "source_lock_verified": True,
        "external_input_manifest_verified": True,
        "feature_store_manifests_and_arrays_verified": True,
        "prepared_calibrations_reconstructed": 8,
        "evaluation_units_fully_reconstructed": 160,
        "assignment_certificates_reconstructed": 1920,
        "exact_milp_diagnostics_rerun": 96,
        "deterministic_aggregate_rederived": True,
        "done_file_sha256": sha256_file(output / "DONE.json"),
        "result_checksum_index_sha256": sha256_file(checksum_path),
        "runtime": runtime_identity(),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--config", default="configs/paired_recovery_followup/followup_v1.json"
    )
    parser.add_argument(
        "--lock",
        default="protocol/paired_recovery_followup/STUDY_LOCK_20260914.json",
    )
    parser.add_argument("--output-root", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the frozen paired-recovery follow-up"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-cell")
    _common_arguments(run)
    run.add_argument("--external-input-manifest", required=True)
    run.add_argument("--feature-root", required=True)
    run.add_argument("--encoder", choices=("clip", "vitb16"), required=True)
    run.add_argument(
        "--corruption", choices=("gaussian_noise", "motion_blur"), required=True
    )
    run.add_argument("--repetition", type=int, choices=(1, 2, 3), required=True)
    aggregate = subparsers.add_parser("aggregate")
    _common_arguments(aggregate)
    verify = subparsers.add_parser("verify")
    _common_arguments(verify)
    verify.add_argument("--external-input-manifest", required=True)
    verify.add_argument("--feature-root", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "run-cell":
        run_cell(
            project_root=arguments.project_root,
            config_path=arguments.config,
            lock_path=arguments.lock,
            external_input_manifest=arguments.external_input_manifest,
            feature_root=arguments.feature_root,
            output_root=arguments.output_root,
            encoder=arguments.encoder,
            corruption=arguments.corruption,
            repetition=arguments.repetition,
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
        )
        write_json_atomic(
            Path(arguments.output_root) / "VERIFICATION.json",
            seal_manifest(report),
        )
        print("paired-recovery follow-up verification: PASS", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
