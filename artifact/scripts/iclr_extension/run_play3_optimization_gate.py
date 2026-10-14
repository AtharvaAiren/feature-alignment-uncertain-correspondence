#!/usr/bin/env python3
"""Freeze, run, aggregate, and verify the additive Play-3 v2 optimization gate.

The completed Play-3 v1 lock is an input lineage record only.  This runner
requires a new lock, frozen before any ImageNet-V2/R outcomes are inspected.
Every unit writes the actual assignments as safe NPY arrays and can be fully
reconstructed from its hash-bound feature store.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    describe_npy,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_file_record,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import load_feature_store
from marginal_alignment.iclr_extension.optimization_certificate import (
    CERTIFICATE_SCHEMA,
    INFORMATION_SCOPE,
    MAX_EXACT_VALIDATION_ROWS,
    SUPPORTED_COST_GEOMETRIES,
    OptimizationCertificateProtocol,
    _array_sha256,
    _unique_canonical_rows,
    construct_optimization_certificate,
    construct_optimization_certificate_from_problem,
    derive_certificate_geometry,
    fit_certificate_geometry,
    prepare_certificate_problem,
    run_exact_subset_validation,
)


CONFIG_SCHEMA = "marginal_alignment.iclr_extension.play3_optimization_gate.v1"
STORE_MAP_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_store_map.v1"
)
LOCK_SCHEMA = "marginal_alignment.iclr_extension.play3_optimization_lock.v1"
UNIT_SCHEMA = "marginal_alignment.iclr_extension.play3_optimization_unit.v1"
AGGREGATE_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_aggregate.v1"
)
PANEL_SHARD_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_panel_shard.v1"
)
VERIFICATION_SHARD_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_verification_shard.v1"
)
COMPUTE_TIMING_SCHEMA = (
    "marginal_alignment.iclr_extension.play3_optimization_compute_timing.v1"
)
SOURCE_FILES = (
    "scripts/iclr_extension/run_play3_optimization_gate.py",
    "src/marginal_alignment/iclr_extension/optimization_certificate.py",
    "scripts/iclr_extension/prepare_play3_optimization_stores.py",
    "src/marginal_alignment/iclr_extension/natural_shift_store.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/linalg.py",
    "docs/play3_optimization_gate.md",
    "docs/extension_compute_accounting.md",
    "docs/play3_store_preparation.md",
    "configs/iclr_extension/play3_store_preparation_inputs.example.json",
    "tests/iclr_extension/test_natural_shift_store.py",
    "tests/iclr_extension/test_optimization_certificate.py",
    "tests/iclr_extension/test_play3_optimization_gate.py",
)
ARRAY_NAMES = (
    "assignment_world_zero",
    "assignment_world_one",
    "prediction_world_zero",
    "prediction_world_one",
    "target_canonical_order",
    "reference_canonical_order",
    "exact_subset_assignment",
)
UNIT_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "certificate_schema_version",
    "status",
    "scientific_result",
    "unit_id",
    "dataset",
    "encoder",
    "cost_geometry",
    "split_seed",
    "cost_budget_relative",
    "lock_file_sha256",
    "lock_manifest_sha256",
    "store_manifest_file_sha256",
    "store_manifest_payload_sha256",
    "information_scope",
    "external_ground_truth_labels_used_for_optimization",
    "frozen_readout_predictions_used_for_optimization",
    "target_wnids_used_for_posthoc_inference",
    "target_wnids_not_used_as_certificate_optimization_inputs",
    "sample_identifiers_used_for_cost_or_reward",
    "sample_identifiers_used_for_indexed_atom_administration",
    "sample_identifiers_parsed_for_semantics",
    "sample_identifier_tie_scope",
    "sample_identifiers_used_for_integrity_validation",
    "prespecified_public_wnid_support_used_for_store_construction",
    "pairing_used",
    "result",
    "exact_small_instance_validation",
    "posthoc_inference",
    "split_diagnostics",
    "arrays",
    "compute",
    "manifest_sha256",
})
ARRAY_RECORD_FIELDS = frozenset({
    "path", "sha256", "size_bytes", "format", "shape", "dtype",
})
CERTIFICATE_RESULT_FIELDS = frozenset({
    "schema_version",
    "status",
    "scientific_interpretation",
    "baseline_cost",
    "primal_cost",
    "allowed_cost",
    "selected_world_disagreement",
    "selected_two_world_minimax_recovery",
    "maximum_disagreement_primal_lower_bound",
    "maximum_disagreement_dual_upper_bound",
    "maximum_disagreement_identification_interval",
    "strongest_two_world_recovery_ceiling_interval",
    "primal_dual_gap",
    "primal_dual_gap_percentage_points",
    "gap_target",
    "gap_target_percentage_points",
    "meets_gap_target",
    "assignment_world_zero_sha256",
    "assignment_world_one_sha256",
    "prediction_world_zero_sha256",
    "prediction_world_one_sha256",
    "target_canonical_order_sha256",
    "reference_canonical_order_sha256",
    "diagnostics",
})
CERTIFICATE_SCIENTIFIC_INTERPRETATION = {
    "selected_two_world_minimax_value": (
        "exact_for_the_two_serialized_assignments"
    ),
    "maximum_disagreement_primal": (
        "integer_feasible_lower_bound_on_maximum_ambiguity"
    ),
    "maximum_disagreement_dual": (
        "sampled_lagrangian_computational_upper_bound_under_the_recorded_"
        "floating_point_solver_model_also_valid_for_the_lp_relaxation"
    ),
    "gap": "optimization_gap_not_a_statistical_confidence_interval",
    "not_external_target_label_accuracy": True,
    "not_a_realized_world_performance_bound": True,
}
CERTIFICATE_DIAGNOSTIC_FIELDS = frozenset({
    "information_scope",
    "external_ground_truth_labels_used_for_optimization",
    "frozen_readout_predictions_used_for_optimization",
    "sample_identifiers_used_for_cost_or_reward",
    "sample_identifiers_used_for_indexed_atom_administration",
    "pairing_used",
    "witness_rows",
    "feature_dimension",
    "indexed_atom_policy",
    "target_witness_indexed_atom_sha256",
    "reference_witness_indexed_atom_sha256",
    "target_witness_indexed_atoms",
    "reference_witness_indexed_atoms",
    "cost_matrix_sha256",
    "reference_prediction_sha256",
    "protocol",
    "geometry",
    "budget_independent_problem_reused",
    "greedy",
    "optimization",
})
POSTHOC_INFERENCE_FIELDS = frozenset({
    "external_ground_truth_labels_used_for_optimization",
    "frozen_readout_predictions_used_for_optimization",
    "target_wnids_preflight_integrity_validated",
    "target_wnids_not_used_as_certificate_optimization_inputs",
    "target_wnids_used_for_posthoc_inference_after_assignment_fixed",
    "target_wnids_used_for_posthoc_inference",
    "sample_identifiers_used_for_cost_or_reward",
    "sample_identifiers_used_for_indexed_atom_administration",
    "sample_identifiers_parsed_for_semantics",
    "exact_duplicate_atom_tie_note",
    "class_intersection_wnids_sha256",
    "target_wnids_role",
    "cluster",
    "declared_cluster_count",
    "observed_witness_cluster_count",
    "per_wnid_sufficient_statistics",
})
POSTHOC_STATISTIC_FIELDS = frozenset({
    "wnid", "rows", "prediction_disagreement_count",
})
SPLIT_DIAGNOSTIC_FIELDS = frozenset({
    "reference_split_parent_content_sha256",
    "target_split_parent_content_sha256",
    "reference_selected_original_rows_sha256",
    "target_selected_original_rows_sha256",
    "grouped_compute_policy",
    "grouped_budget_count",
    "grouped_geometry_count",
})
UNIT_COMPUTE_FIELDS = frozenset({
    "elapsed_seconds",
    "invocation_wall_nanoseconds_before_manifest_write",
    "temporary_directory_and_array_serialization_nanoseconds",
    "shared_group_compute_event",
    "shared_group_compute_event_owner_for_this_unit",
    "shared_event_accounting_policy",
    "manifest_publication_timing_nanoseconds",
    "manifest_publication_timing_status",
    "runtime_contract",
})
COMPUTE_EVENT_FIELDS = frozenset({
    "schema_version",
    "purpose",
    "event_id",
    "group_id",
    "dataset",
    "encoder",
    "split_seed",
    "lock_manifest_sha256",
    "store_manifest_file_sha256",
    "store_manifest_payload_sha256",
    "clock",
    "unit_ids_computed",
    "group_wall_nanoseconds",
    "store_validation_nanoseconds",
    "split_selection_nanoseconds",
    "shared_calibration_and_bures_fit_nanoseconds",
    "geometry_stages",
    "stage_sum_nanoseconds",
    "unattributed_python_overhead_nanoseconds",
    "peak_rss_bytes",
    "peak_rss_status",
    "scope",
    "timing_payload_sha256",
})
COMPUTE_GEOMETRY_FIELDS = frozenset({
    "cost_geometry",
    "geometry_derivation_nanoseconds",
    "dense_problem_preparation_nanoseconds",
    "budgets",
})
COMPUTE_BUDGET_FIELDS = frozenset({
    "cost_budget_basis_points",
    "certificate_nanoseconds",
    "exact_validation_nanoseconds",
    "posthoc_inference_nanoseconds",
})
VERIFICATION_REPORT_FIELDS = frozenset({
    "unit_id",
    "unit_manifest_file_sha256",
    "unit_manifest_sha256",
    "status",
    "assignments_recomputed",
    "costs_recomputed_from_endpoint_features",
    "rewards_recomputed_from_frozen_readout",
    "bijections_revalidated",
    "duplicate_row_admissibility_revalidated",
    "exact_small_instance_milp_revalidated",
    "posthoc_wnid_statistics_revalidated",
    "gap_target_met",
    "primal_dual_gap_percentage_points",
    "verification_compute_event",
    "verification_compute_event_owner_for_this_report",
    "verification_timing",
})
VERIFICATION_TIMING_FIELDS = frozenset({
    "clock",
    "verification_wall_nanoseconds_before_report_seal",
    "pre_reconstruction_artifact_and_contract_validation_nanoseconds",
    "post_reconstruction_comparison_nanoseconds",
    "non_group_validation_and_python_overhead_nanoseconds",
    "verification_manifest_publication_timing_nanoseconds",
    "peak_rss_bytes",
    "peak_rss_status",
})
VERIFICATION_SHARD_FIELDS = frozenset({
    "schema_version",
    "status",
    "lock_file_sha256",
    "lock_manifest_sha256",
    "shard_count",
    "shard_index",
    "expected_units",
    "verified_units",
    "full_endpoint_feature_reconstruction",
    "compute_accounting",
    "reports",
    "manifest_sha256",
})
AGGREGATE_FIELDS = frozenset({
    "schema_version",
    "status",
    "status_interpretation",
    "artifact_status",
    "scientific_result",
    "scientific_gate_status",
    "all_required_gates_pass",
    "required_scientific_gate_keys",
    "failed_required_scientific_gate_keys",
    "lock_file_sha256",
    "lock_manifest_sha256",
    "expected_units",
    "observed_units",
    "zero_silent_drops",
    "full_reconstruction_verification_required",
    "verification_manifests",
    "compute_accounting",
    "inference",
    "conditions",
    "gates",
    "unit_manifests",
    "interpretation",
    "manifest_sha256",
})
AGGREGATE_UNIT_FIELDS = frozenset({
    "unit_id", "manifest_file_sha256", "manifest_sha256",
})
AGGREGATE_VERIFICATION_FIELDS = frozenset({
    "shard_index",
    "manifest_file_sha256",
    "manifest_sha256",
    "verified_units",
    "compute_accounting",
})
AGGREGATE_CONDITION_FIELDS = frozenset({
    "dataset",
    "encoder",
    "cost_geometry",
    "cost_budget_relative",
    "disagreement",
    "unit_ids",
})
AGGREGATE_INTERPRETATION_FIELDS = frozenset({
    "external_ground_truth_labels_used_for_optimization",
    "frozen_readout_predictions_used_for_optimization",
    "target_wnids_preflight_integrity_validated",
    "target_wnids_not_used_as_certificate_optimization_inputs",
    "target_wnids_used_for_posthoc_cluster_inference_after_assignment_fixed",
    "sample_identifiers_used_for_cost_or_reward",
    "sample_identifiers_used_for_indexed_atom_administration",
    "sample_identifiers_parsed_for_semantics",
    "exact_duplicate_atom_posthoc_tie_caveat",
    "gate_passes_are_not_asserted_until_this_aggregate_exists",
})
REQUIRED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REQUIRED_SCIENTIFIC_GATES = (
    "optimization_all_1_5_10_percent_units",
    "exact_64_row_all_units",
    "breadth_1_percent__common_encoders",
    "breadth_5_percent__common_encoders",
    "geometry_gate",
)
SCIENTIFIC_GATE_FAILURE_EXIT_CODE = 2
COMPUTE_PURPOSES = (
    "scientific_panel_compute",
    "full_reconstruction_verification",
)
PEAK_RSS_STATUS = "MISSING_NOT_PORTABLY_AUTHENTICATED_BY_THIS_RUNNER"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str] | set[str], *, name: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise IntegrityError(
            f"{name} schema changed: missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_int(value: Any, expected: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (expected is None or value == expected)
    )


def _finite_number(value: Any, expected: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and (expected is None or float(value) == expected)
    )


def _load_config(path: str | Path, *, require_ready: bool) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = dict(_mapping(read_json(source), name="optimization-gate config"))
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise IntegrityError("unsupported optimization-gate config schema")
    expected_status = "READY_TO_FREEZE_BEFORE_NEW_OUTCOMES"
    if require_ready and value.get("status") != expected_status:
        raise IntegrityError(
            f"config must have status {expected_status!r} before freezing"
        )
    datasets = _mapping(value.get("datasets"), name="datasets")
    encoders = tuple(map(str, value.get("encoders", ())))
    geometries = tuple(map(str, value.get("cost_geometries", ())))
    protocol = _mapping(value.get("certificate_protocol"), name="certificate_protocol")
    if tuple(datasets) != ("imagenet_sketch", "imagenet_v2", "imagenet_r"):
        raise IntegrityError("gate needs exactly Sketch, V2, and R in locked order")
    if encoders != ("dinov3", "clip"):
        raise IntegrityError("gate needs exactly the locked DINOv3 and CLIP encoders")
    encoder_contracts = _mapping(
        value.get("encoder_contracts"), name="encoder_contracts"
    )
    expected_encoder_contracts = {
        "dinov3": {
            "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "expected_dimension": 1024,
            "feature_layer": "last_hidden_state[:,0]",
            "pooling": "class_token",
        },
        "clip": {
            "model_id": "openai/clip-vit-large-patch14",
            "expected_dimension": 768,
            "feature_layer": "vision_model.post_layernorm+visual_projection",
            "pooling": "projected_class_token",
        },
    }
    if encoder_contracts != expected_encoder_contracts:
        raise IntegrityError("gate encoder identity contract changed")
    if geometries != SUPPORTED_COST_GEOMETRIES:
        raise IntegrityError("gate cost geometries changed")
    split_seeds = protocol.get("split_seeds")
    if (
        not isinstance(split_seeds, list)
        or not all(_exact_int(value) for value in split_seeds)
        or tuple(split_seeds) != (201, 202, 203)
    ):
        raise IntegrityError("gate split seeds changed")
    cost_budgets = protocol.get("cost_budgets_relative")
    if (
        not isinstance(cost_budgets, list)
        or not all(_finite_number(value) for value in cost_budgets)
        or tuple(map(float, cost_budgets)) != (0.01, 0.05, 0.10)
    ):
        raise IntegrityError("gate cost budgets changed")
    if (
        not _exact_int(protocol.get("calibration_rows"), 3840)
        or not _exact_int(protocol.get("witness_rows"), 1024)
        or not _exact_int(
            protocol.get("exact_validation_rows"), MAX_EXACT_VALIDATION_ROWS
        )
        or not _finite_number(protocol.get("maximum_primal_dual_gap"), 0.05)
    ):
        raise IntegrityError("gate row counts or optimization target changed")
    if (
        protocol.get("information_scope") != INFORMATION_SCOPE
        or protocol.get("external_ground_truth_labels_used_for_optimization")
        is not False
        or protocol.get("frozen_readout_predictions_used_for_optimization")
        is not True
        or protocol.get("formal_real_arithmetic_certificate_claimed") is not False
        or protocol.get("floating_point_solver_not_formally_verified") is not True
    ):
        raise IntegrityError("gate information scope changed")
    if (
        protocol.get("duplicate_feature_row_policy")
        != (
            "preserve_indexed_atoms_with_authenticated_unique_sample_ids_used_"
            "only_for_exact_feature_ties"
        )
        or not _exact_int(
            protocol.get("maximum_exact_milp_binary_variables"),
            MAX_EXACT_VALIDATION_ROWS ** 2,
        )
    ):
        raise IntegrityError("gate indexed-atom or exact-complexity contract changed")
    if (
        protocol.get("sample_identifier_semantic_parsing") != "forbidden"
        or protocol.get("sample_identifier_integrity_policy")
        != "included_in_indexed_atom_digest_and_serialized_inverse_maps"
        or protocol.get("sample_identifier_posthoc_tie_caveat")
        != (
            "for_exact_duplicate_feature_rows_only_the_opaque_id_tie_order_can_"
            "change_which_indexed_boundary_atom_receives_an_assignment_and_thus_"
            "its_posthoc_wnid_attribution_but_not_feature_cost_or_prediction_"
            "reward"
        )
        or protocol.get("grouped_compute_policy")
        != (
            "split_and_bures_once_per_dataset_encoder_seed;_cost_once_per_"
            "geometry;_three_budgets_reuse_the_prepared_dense_problem"
        )
    ):
        raise IntegrityError("gate indexed-atom administration contract changed")
    if (
        protocol.get("restart_policy")
        != "not_applicable_deterministic_single_schedule_no_random_initialization"
        or protocol.get("rerun_stability_check")
        != "full_reconstruction_requires_bitwise_identical_assignments_and_summaries_under_the_locked_runtime"
    ):
        raise IntegrityError("gate deterministic rerun contract changed")
    accounting = _mapping(
        _mapping(value.get("execution"), name="execution").get(
            "compute_accounting"
        ),
        name="execution.compute_accounting",
    )
    if accounting != {
        "schema_version": COMPUTE_TIMING_SCHEMA,
        "clock": "time.perf_counter_ns_monotonic",
        "shared_event_policy": (
            "one_event_per_actual_dataset_encoder_seed_group_computation_"
            "copied_to_its_unit_records_and_deduplicated_by_authenticated_"
            "event_id"
        ),
        "retry_policy": (
            "distinct_successful_recomputations_reaching_at_least_one_"
            "published_unit_are_counted_once_each;_failed_or_interrupted_"
            "computations_publishing_no_unit_are_not_observable_here"
        ),
        "peak_rss_status": PEAK_RSS_STATUS,
        "manifest_publication_timing_status": (
            "MISSING_SELF_REFERENTIAL_MANIFEST_CANNOT_AUTHENTICATE_POST_"
            "WRITE_DURATION"
        ),
    }:
        raise IntegrityError("gate compute-accounting contract changed")
    requested_rows = int(protocol["calibration_rows"]) + int(
        protocol["witness_rows"]
    )
    for dataset, declaration in datasets.items():
        item = _mapping(declaration, name=f"datasets.{dataset}")
        for field in (
            "expected_reference_rows_after_intersection_and_head_fit_exclusion",
            "expected_target_rows",
            "expected_target_classes",
        ):
            if not _exact_int(item.get(field)) or item[field] < 1:
                raise IntegrityError(
                    f"datasets.{dataset}.{field} must be a positive integer"
                )
        if (
            item.get(
                "expected_reference_rows_after_intersection_and_head_fit_exclusion",
            )
            < requested_rows
        ):
            raise IntegrityError(
                f"{dataset} cannot supply the locked no-head-fit reference split"
            )
    if datasets["imagenet_r"].get(
        "expected_reference_rows_after_intersection_and_head_fit_exclusion", -1
    ) != 5000:
        raise IntegrityError("ImageNet-R reference-row constraint changed")
    lineage = _mapping(value.get("lineage"), name="lineage")
    repository = _repository_root()
    for path_key, hash_key in (
        ("completed_play3_v1_config", "completed_play3_v1_config_file_sha256"),
        ("completed_play3_v1_lock", "completed_play3_v1_lock_file_sha256"),
        (
            "completed_extension_master_lock",
            "completed_extension_master_lock_file_sha256",
        ),
    ):
        lineage_path = Path(str(lineage.get(path_key, "")))
        if lineage_path.is_absolute() or not lineage_path.parts:
            raise IntegrityError(f"gate lineage path is invalid: {path_key}")
        source = (repository / lineage_path).resolve()
        try:
            source.relative_to(repository)
        except ValueError as error:
            raise IntegrityError(f"gate lineage escapes repository: {path_key}") from error
        if not source.is_file() or sha256_file(source) != lineage.get(hash_key):
            raise IntegrityError(f"gate lineage hash changed: {path_key}")
    if (
        protocol.get("class_intersection_order") != "lexicographic_wnid"
        or protocol.get("store_preparation_metadata_key")
        != "optimization_gate_preparation"
        or protocol.get("reference_wnids_role_field")
        != "reference_wnids_role"
        or protocol.get("target_wnids_role_field") != "target_wnids_role"
    ):
        raise IntegrityError("gate class-intersection contract changed")
    inference = _mapping(value.get("inference"), name="inference")
    if (
        inference.get("cluster") != "imagenet_wnid"
        or inference.get("display_label")
        != (
            "descriptive conditional WNID-cluster resampling interval, "
            "assignments fixed/no optimizer refit"
        )
        or inference.get("estimand")
        != "selected_in_sample_witness_disagreement_conditional_on_each_serialized_optimized_assignment"
        or inference.get("optimizer_refit_within_bootstrap") is not False
        or inference.get("interpretation")
        != "descriptive_conditional_cluster_resampling_interval_not_a_sampling_valid_confidence_interval_for_the_population_optimized_functional"
        or inference.get("gate_kind")
        != "prespecified_robustness_screen_not_population_or_confirmatory_inference"
        or inference.get("split_seed_policy")
        != "sum_sufficient_statistics_within_wnid_before_cluster_resampling"
        or inference.get(
            "encoders_geometries_and_budgets_nested_within_dataset_wnid_panel"
        )
        is not True
        or inference.get("rng") != "numpy.random.Generator(PCG64)"
        or not _exact_int(inference.get("seed"), 20260903)
        or not _exact_int(inference.get("replicates"), 10000)
        or not _finite_number(
            inference.get("nominal_resampling_interval_level"), 0.95
        )
        or inference.get("interval") != "descriptive_conditional_percentile"
        or inference.get("zero_silent_drops") is not True
    ):
        raise IntegrityError("gate WNID-clustered inference contract changed")
    execution = _mapping(value.get("execution"), name="execution")
    if execution != {
        "panel_shard_count": 4,
        "verification_shard_count": 4,
        "aggregate_requires_complete_full_reconstruction_verification": True,
        "compute_accounting": accounting,
        "numerical_thread_environment": REQUIRED_THREAD_ENVIRONMENT,
    }:
        raise IntegrityError("gate execution/verification contract changed")
    gate_decisions = _mapping(value.get("gate_decisions"), name="gate_decisions")
    if tuple(map(str, gate_decisions.get("required_scientific_gate_keys", ()))) != (
        REQUIRED_SCIENTIFIC_GATES
    ):
        raise IntegrityError("required scientific gate inventory changed")
    return value


def _source_hashes(repository: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = (repository / relative).resolve()
        if not source.is_file():
            raise IntegrityError(f"optimization-gate source is missing: {relative}")
        hashes[relative] = sha256_file(source)
    return hashes


def _runtime_contract() -> dict[str, Any]:
    try:
        import scipy
        import scipy.optimize._highspy._core as highs_core
    except ImportError as error:  # pragma: no cover - full environment contract
        raise RuntimeError("optimization gate requires SciPy with bundled HiGHS") from error
    try:
        configuration = np.show_config(mode="dicts")
    except TypeError:  # NumPy 1.x does not expose the structured mode.
        configuration = {}
    build_dependencies = configuration.get("Build Dependencies", {})
    blas = build_dependencies.get("blas", {})
    lapack = build_dependencies.get("lapack", {})
    thread_environment = {
        name: os.environ.get(name) for name in REQUIRED_THREAD_ENVIRONMENT
    }
    if thread_environment != REQUIRED_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "optimization gate requires every locked numerical thread "
            "environment variable to equal '1': "
            f"observed={thread_environment}"
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "highs": (
            f"{highs_core.HIGHS_VERSION_MAJOR}."
            f"{highs_core.HIGHS_VERSION_MINOR}."
            f"{highs_core.HIGHS_VERSION_PATCH}"
        ),
        "blas": {
            "name": blas.get("name", "unavailable_in_numpy_version"),
            "version": blas.get("version", "unavailable_in_numpy_version"),
        },
        "lapack": {
            "name": lapack.get("name", "unavailable_in_numpy_version"),
            "version": lapack.get("version", "unavailable_in_numpy_version"),
        },
        "thread_environment": thread_environment,
    }


def _store_key(dataset: str, encoder: str) -> str:
    return f"{dataset}::{encoder}"


def _wnid_string(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("ascii")
        except UnicodeDecodeError as error:
            raise IntegrityError("WNIDs must be ASCII") from error
    return str(value)


def _validated_wnid_values(
    values: np.ndarray,
    *,
    expected_rows: int,
    declared_wnids: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != int(expected_rows):
        raise IntegrityError(f"{name} must have one WNID per endpoint row")
    resolved = tuple(_wnid_string(value) for value in array)
    declared = set(declared_wnids)
    observed = set(resolved)
    if not observed.issubset(declared):
        raise IntegrityError(f"{name} contains a WNID outside the intersection")
    if observed != declared:
        raise IntegrityError(f"{name} does not cover the declared WNID intersection")
    return resolved


def _validated_sample_ids(
    values: np.ndarray, *, expected_rows: int, name: str
) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != int(expected_rows):
        raise IntegrityError(f"{name} must have one scalar ID per indexed atom")
    resolved = tuple(_wnid_string(value) for value in array)
    if len(set(resolved)) != len(resolved):
        raise IntegrityError(f"{name} must be unique for indexed-atom ties")
    return resolved


def _validate_store_map(
    path: str | Path, config: Mapping[str, Any], *, verify_hashes: bool
) -> dict[str, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    value = _mapping(read_json(source), name="store map")
    if set(value) != {"schema_version", "stores"}:
        raise IntegrityError("store map has missing or undeclared fields")
    if value.get("schema_version") != STORE_MAP_SCHEMA:
        raise IntegrityError("unsupported optimization store-map schema")
    records = value.get("stores")
    if not isinstance(records, list):
        raise IntegrityError("stores must be a JSON list")
    expected = {
        _store_key(dataset, encoder)
        for dataset in config["datasets"]
        for encoder in config["encoders"]
    }
    resolved: dict[str, dict[str, Any]] = {}
    encoder_bindings: dict[str, dict[str, Any]] = {}
    for record in records:
        item = _mapping(record, name="store-map record")
        if set(item) != {"dataset", "encoder", "manifest"}:
            raise IntegrityError("store-map record has undeclared fields")
        dataset = str(item["dataset"])
        encoder = str(item["encoder"])
        key = _store_key(dataset, encoder)
        if key not in expected or key in resolved:
            raise IntegrityError(f"unexpected or duplicate store key: {key}")
        manifest = Path(str(item["manifest"])).expanduser().resolve()
        store = load_feature_store(manifest, verify_hashes=verify_hashes)
        dataset_config = _mapping(
            config["datasets"][dataset], name=f"datasets.{dataset}"
        )
        metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
        target_extraction = _mapping(
            metadata.get("target_extraction_manifest"),
            name=f"{key} target extraction manifest",
        )
        source_binding = _mapping(
            metadata.get("source_campaign_reference_store"),
            name=f"{key} source campaign reference store",
        )
        provenance_key = str(dataset_config["store_provenance_metadata_key"])
        provenance = _mapping(
            metadata.get(provenance_key), name=f"{key} dataset provenance"
        )
        preparation_key = str(
            config["certificate_protocol"]["store_preparation_metadata_key"]
        )
        preparation = _mapping(
            metadata.get(preparation_key), name=f"{key} store preparation"
        )
        model = _mapping(
            metadata.get("model_provenance"), name=f"{key} model provenance"
        )
        intersection_key = str(
            config["certificate_protocol"][
                "class_intersection_store_metadata_key"
            ]
        )
        intersection = _mapping(
            metadata.get(intersection_key), name="class intersection"
        )
        wnids = intersection.get("wnids")
        reference_wnids_role = str(intersection.get("reference_wnids_role", ""))
        target_wnids_role = str(intersection.get("target_wnids_role", ""))
        reference_rows = len(store.endpoint("reference").features)
        target_rows = len(store.endpoint("target").features)
        if (
            not isinstance(wnids, list)
            or len(wnids) != int(dataset_config["expected_target_classes"])
            or len(set(map(str, wnids))) != len(wnids)
            or list(map(str, wnids)) != sorted(map(str, wnids))
            or any(
                len(str(wnid)) != 9
                or not str(wnid).startswith("n")
                or not str(wnid)[1:].isdigit()
                for wnid in wnids
            )
            or intersection.get("wnids_sha256") != sha256_json(wnids)
            or intersection.get("policy")
            != dataset_config["class_intersection_policy"]
            or intersection.get("reference_endpoint_restricted") is not True
            or intersection.get("target_endpoint_restricted") is not True
            or reference_wnids_role not in store.records
            or target_wnids_role not in store.records
            or target_rows != int(dataset_config["expected_target_rows"])
            or reference_rows
            != int(dataset_config[
                "expected_reference_rows_after_intersection_and_head_fit_exclusion"
            ])
        ):
            raise IntegrityError(f"class-intersection binding is invalid: {key}")
        declared_wnids = tuple(map(str, wnids))
        _validated_wnid_values(
            store.array(reference_wnids_role),
            expected_rows=reference_rows,
            declared_wnids=declared_wnids,
            name=f"reference WNID role for {key}",
        )
        _validated_wnid_values(
            store.array(target_wnids_role),
            expected_rows=target_rows,
            declared_wnids=declared_wnids,
            name=f"target WNID role for {key}",
        )
        endpoint_reference = store.endpoint("reference")
        endpoint_target = store.endpoint("target")
        reference_endpoint_roles = _mapping(
            store.endpoint_roles.get("reference"),
            name=f"{key} reference endpoint roles",
        )
        target_endpoint_roles = _mapping(
            store.endpoint_roles.get("target"),
            name=f"{key} target endpoint roles",
        )
        if (
            metadata.get("endpoint_labels_not_published") is not True
            or reference_endpoint_roles.get("labels") is not None
            or target_endpoint_roles.get("labels") is not None
            or getattr(endpoint_reference, "labels", None) is not None
            or getattr(endpoint_target, "labels", None) is not None
        ):
            raise IntegrityError(f"v2 store exposes an endpoint label role: {key}")
        encoder_contract = _mapping(
            config["encoder_contracts"][encoder], name=f"encoder {encoder}"
        )
        head = _mapping(store.manifest.get("head"), name=f"{key} frozen head")
        weight_role = str(head.get("weight_role", ""))
        bias_role = str(head.get("bias_role", ""))
        fit_role = str(model.get("head_fit_ids_role", ""))
        reference_ids_role = str(store.endpoint_roles["reference"]["ids"])
        if any(
            role not in store.records
            for role in (weight_role, bias_role, fit_role, reference_ids_role)
        ):
            raise IntegrityError(f"store lacks a required head/ID role: {key}")
        reference_ids = np.asarray(endpoint_reference.ids)
        target_ids = np.asarray(endpoint_target.ids)
        _validated_sample_ids(
            reference_ids,
            expected_rows=reference_rows,
            name=f"reference IDs for {key}",
        )
        _validated_sample_ids(
            target_ids,
            expected_rows=target_rows,
            name=f"target IDs for {key}",
        )
        head_fit_ids = np.asarray(store.array(fit_role))
        try:
            overlap = set(reference_ids.tolist()).intersection(
                set(head_fit_ids.tolist())
            )
        except TypeError as error:
            raise IntegrityError(f"store IDs are not scalar/hashable: {key}") from error
        if overlap:
            raise IntegrityError(
                f"reference endpoint contains frozen head-fit IDs: {key}"
            )
        provenance_required_hashes = (
            "manifest_sha256",
            "file_tree_sha256",
            "class_index_sha256",
        )
        if (
            provenance.get("schema_version")
            != dataset_config["expected_provenance_schema"]
            or any(
                not _valid_sha256(provenance.get(field))
                for field in provenance_required_hashes
            )
            or provenance.get("class_index_sha256")
            != intersection.get("wnids_sha256")
            or int(provenance.get("row_count", -1)) != target_rows
            or (
                dataset != "imagenet_sketch"
                and (
                    provenance.get("status") != "PASS"
                    or provenance.get("dataset") != dataset
                    or provenance.get("variant") != dataset_config["variant"]
                    or not _valid_sha256(provenance.get("target_order_sha256"))
                )
            )
        ):
            raise IntegrityError(f"store dataset provenance is invalid: {key}")
        if (
            model.get("encoder_id") != encoder
            or model.get("model_id") != encoder_contract["model_id"]
            or model.get("feature_layer") != encoder_contract["feature_layer"]
            or model.get("pooling") != encoder_contract["pooling"]
            or endpoint_reference.features.shape[1]
            != int(encoder_contract["expected_dimension"])
            or endpoint_target.features.shape[1]
            != int(encoder_contract["expected_dimension"])
        ):
            raise IntegrityError(f"store encoder identity is invalid: {key}")
        reference_ids_sha256 = str(store.records[reference_ids_role]["sha256"])
        reference_wnids_sha256 = str(
            store.records[reference_wnids_role]["sha256"]
        )
        target_wnids_sha256 = str(store.records[target_wnids_role]["sha256"])
        if (
            preparation.get("schema_version")
            != "marginal_alignment.iclr_extension.optimization_store_preparation.v1"
            or preparation.get("status") != "PASS"
            or preparation.get("dataset") != dataset
            or preparation.get("encoder") != encoder
            or preparation.get("head_fit_disjoint_from_reference") is not True
            or preparation.get("head_fit_ids_sha256")
            != store.records[fit_role]["sha256"]
            or preparation.get("reference_ids_sha256") != reference_ids_sha256
            or preparation.get("reference_wnids_sha256")
            != reference_wnids_sha256
            or preparation.get("target_wnids_sha256") != target_wnids_sha256
            or preparation.get("class_intersection_wnids_sha256")
            != intersection["wnids_sha256"]
            or not _valid_sha256(
                preparation.get("reference_source_manifest_sha256")
            )
            or preparation.get("reference_source_manifest_sha256")
            != metadata.get("store_preparation_source_manifest_sha256")
            or preparation.get("reference_source_manifest_sha256")
            != source_binding.get("actual_manifest_sha256")
            or preparation.get("target_extraction_manifest_sha256")
            != sha256_json(target_extraction)
            or preparation.get("dataset_provenance_manifest_sha256")
            != provenance.get("manifest_sha256")
        ):
            raise IntegrityError(f"store preparation proof is invalid: {key}")
        if (
            store.dataset != dataset
            or store.encoder != encoder
            or store.pairing_spec.get("status") != "unavailable"
        ):
            raise IntegrityError(f"store identity/provenance is invalid: {key}")
        encoder_binding = {
            "encoder_id": encoder,
            "model_id": model["model_id"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "preprocessing_sha256": model["preprocessing_sha256"],
            "feature_layer": model["feature_layer"],
            "pooling": model["pooling"],
            "head_protocol_sha256": sha256_json(model["head_protocol"]),
            "head_fit_ids_sha256": store.records[fit_role]["sha256"],
            "head_weight_sha256": store.records[weight_role]["sha256"],
            "head_bias_sha256": store.records[bias_role]["sha256"],
        }
        if encoder in encoder_bindings and encoder_bindings[encoder] != encoder_binding:
            raise IntegrityError(
                f"dataset stores disagree on frozen encoder/readout: {encoder}"
            )
        encoder_bindings[encoder] = encoder_binding
        resolved[key] = {
            "dataset": dataset,
            "encoder": encoder,
            "manifest": str(manifest),
            "manifest_file_sha256": sha256_file(manifest),
            "manifest_payload_sha256": sha256_json(store.manifest),
            "provenance_metadata_key": provenance_key,
            "provenance_sha256": sha256_json(provenance),
            "store_preparation_metadata_key": preparation_key,
            "store_preparation_sha256": sha256_json(preparation),
            "model_provenance_sha256": sha256_json(model),
            "encoder_binding": encoder_binding,
            "reference_ids_sha256": reference_ids_sha256,
            "head_fit_ids_sha256": str(store.records[fit_role]["sha256"]),
            "head_fit_disjoint_from_reference": True,
            "class_intersection_metadata_key": intersection_key,
            "class_intersection_sha256": sha256_json(intersection),
            "wnids_sha256": str(intersection["wnids_sha256"]),
            "wnids": list(declared_wnids),
            "reference_wnids_role": reference_wnids_role,
            "target_wnids_role": target_wnids_role,
        }
    if set(resolved) != expected:
        raise IntegrityError(
            f"store map coverage mismatch: missing={sorted(expected-set(resolved))}"
        )
    return resolved


def freeze_lock(
    config_path: str | Path,
    store_map_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Freeze code/config/store identities before any new outcome compute."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite optimization lock: {output}")
    config_source = Path(config_path).expanduser().resolve()
    store_map_source = Path(store_map_path).expanduser().resolve()
    config = _load_config(config_source, require_ready=True)
    stores = _validate_store_map(store_map_source, config, verify_hashes=True)
    repository = _repository_root()
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "FROZEN_BEFORE_NEW_OUTCOME_COMPUTE",
        "outcomes_read_during_freeze": False,
        "config": config,
        "config_file_sha256": sha256_file(config_source),
        "store_map_file_sha256": sha256_file(store_map_source),
        "source_files": _source_hashes(repository),
        "stores": {
            key: {field: value for field, value in record.items() if field != "manifest"}
            for key, record in sorted(stores.items())
        },
        "lineage": dict(_mapping(config["lineage"], name="lineage")),
        "freeze_runtime": _runtime_contract(),
    }
    write_json_atomic(output, seal_manifest(payload))
    return output


def _load_lock(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    lock = dict(_mapping(read_json(source), name="optimization lock"))
    verify_manifest_checksum(lock)
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "FROZEN_BEFORE_NEW_OUTCOME_COMPUTE"
        or lock.get("outcomes_read_during_freeze") is not False
    ):
        raise IntegrityError("optimization lock is not execution eligible")
    if lock.get("source_files") != _source_hashes(_repository_root()):
        raise IntegrityError("optimization-gate source differs from the frozen lock")
    if lock.get("freeze_runtime") != _runtime_contract():
        raise IntegrityError("optimization-gate runtime differs from the frozen lock")
    config = _mapping(lock.get("config"), name="locked config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise IntegrityError("locked config schema changed")
    return lock


def _validated_unit_store(
    lock: Mapping[str, Any], store_path: str | Path, dataset: str, encoder: str
):
    key = _store_key(dataset, encoder)
    record = _mapping(
        _mapping(lock.get("stores"), name="locked stores").get(key),
        name=f"locked store {key}",
    )
    store = load_feature_store(store_path, verify_hashes=True)
    metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
    provenance_key = str(record["provenance_metadata_key"])
    provenance = _mapping(metadata.get(provenance_key), name="store provenance")
    preparation = _mapping(
        metadata.get(str(record["store_preparation_metadata_key"])),
        name="store preparation",
    )
    model = _mapping(metadata.get("model_provenance"), name="model provenance")
    intersection = _mapping(
        metadata.get(str(record["class_intersection_metadata_key"])),
        name="class intersection",
    )
    if (
        store.dataset != dataset
        or store.encoder != encoder
        or store.pairing_spec.get("status") != "unavailable"
        or sha256_file(store.manifest_path) != record["manifest_file_sha256"]
        or sha256_json(store.manifest) != record["manifest_payload_sha256"]
        or sha256_json(provenance) != record["provenance_sha256"]
        or sha256_json(preparation) != record["store_preparation_sha256"]
        or sha256_json(model) != record["model_provenance_sha256"]
        or sha256_json(intersection) != record["class_intersection_sha256"]
        or intersection.get("wnids_sha256") != record["wnids_sha256"]
        or list(map(str, intersection.get("wnids", ()))) != record["wnids"]
        or intersection.get("reference_wnids_role")
        != record["reference_wnids_role"]
        or intersection.get("target_wnids_role") != record["target_wnids_role"]
    ):
        raise IntegrityError(f"unit store differs from the frozen lock: {key}")
    fit_role = str(model["head_fit_ids_role"])
    reference_ids = np.asarray(store.endpoint("reference").ids)
    head_fit_ids = np.asarray(store.array(fit_role))
    if set(reference_ids.tolist()).intersection(set(head_fit_ids.tolist())):
        raise IntegrityError(f"unit reference endpoint contains head-fit IDs: {key}")
    return store


def _locked_store_map(
    lock: Mapping[str, Any], store_map_path: str | Path
) -> dict[str, dict[str, Any]]:
    resolved = _validate_store_map(
        store_map_path, lock["config"], verify_hashes=True
    )
    locked = _mapping(lock.get("stores"), name="locked stores")
    if set(resolved) != set(locked):
        raise IntegrityError("execution store map coverage differs from the lock")
    for key, record in resolved.items():
        without_path = {
            field: value for field, value in record.items() if field != "manifest"
        }
        if without_path != locked[key]:
            raise IntegrityError(f"execution store map differs from lock: {key}")
    return resolved


def _content_select(
    features: np.ndarray,
    *,
    sample_ids: np.ndarray,
    requested: int,
    name: str,
    salt: bytes,
) -> tuple[np.ndarray, np.ndarray, str]:
    canonical, order, digest, _ = _unique_canonical_rows(
        features, name=name, salt=salt, sample_ids=sample_ids
    )
    if int(requested) > len(canonical):
        raise IntegrityError(f"{name} has too few rows for the locked split")
    # Copy only the locked subset so the full canonicalized endpoint can be
    # released before canonicalizing the other (potentially 50k-row) endpoint.
    return (
        np.array(canonical[:requested], copy=True),
        np.array(order[:requested], copy=True),
        digest,
    )


def _split_inputs(store, config: Mapping[str, Any], *, seed: int):
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    protocol = config["certificate_protocol"]
    requested = int(protocol["calibration_rows"]) + int(protocol["witness_rows"])
    reference_rows, reference_selection, reference_digest = _content_select(
        reference.features,
        sample_ids=reference.ids,
        requested=requested,
        name="reference_store_features",
        salt=(
            f"play3-v2:{store.dataset}:{seed}:reference-split"
        ).encode("ascii"),
    )
    target_rows, target_selection, target_digest = _content_select(
        target.features,
        sample_ids=target.ids,
        requested=requested,
        name="target_store_features",
        salt=(f"play3-v2:{store.dataset}:{seed}:target-split").encode("ascii"),
    )
    head = _mapping(store.manifest.get("head"), name="frozen head")
    weight = np.asarray(store.array(str(head["weight_role"])), dtype=np.float64)
    bias = np.asarray(store.array(str(head["bias_role"])), dtype=np.float64)
    reference_ids = np.asarray(reference.ids)[reference_selection]
    target_ids = np.asarray(target.ids)[target_selection]
    return (
        reference_rows,
        target_rows,
        weight,
        bias,
        reference_digest,
        target_digest,
        reference_selection,
        target_selection,
        reference_ids,
        target_ids,
    )


def _unit_protocol(
    config: Mapping[str, Any], *, geometry: str, budget: float
) -> OptimizationCertificateProtocol:
    declared = config["certificate_protocol"]
    if geometry not in config["cost_geometries"]:
        raise IntegrityError("unit geometry is outside the lock")
    if budget not in [float(value) for value in declared["cost_budgets_relative"]]:
        raise IntegrityError("unit cost budget is outside the lock")
    return OptimizationCertificateProtocol(
        cost_geometry=geometry,
        cost_budget_relative=budget,
        covariance_estimator=declared["covariance_estimator"],
        covariance_relative_floor=float(declared["covariance_relative_floor"]),
        maximum_witness_rows=int(declared["witness_rows"]),
        greedy_neighbors=int(declared["greedy_neighbors"]),
        dual_bisection_iterations=int(declared["dual_bisection_iterations"]),
        primal_dual_gap_target=float(declared["maximum_primal_dual_gap"]),
        assignment_tie_relative_tolerance=float(
            declared["assignment_tie_relative_tolerance"]
        ),
    )


def _posthoc_wnid_statistics(
    store,
    lock: Mapping[str, Any],
    result,
    *,
    dataset: str,
    encoder: str,
    calibration_rows: int,
    target_selection: np.ndarray,
) -> dict[str, Any]:
    """Load class membership only after optimization and form cluster counts."""

    record = _mapping(
        _mapping(lock["stores"], name="locked stores").get(
            _store_key(dataset, encoder)
        ),
        name="locked unit store",
    )
    role = str(record["target_wnids_role"])
    declared_wnids = tuple(map(str, record["wnids"]))
    _validated_wnid_values(
        store.array(str(record["reference_wnids_role"])),
        expected_rows=len(store.endpoint("reference").features),
        declared_wnids=declared_wnids,
        name=f"posthoc reference WNID role for {dataset}::{encoder}",
    )
    target_wnids = np.asarray(store.array(role))
    _validated_wnid_values(
        target_wnids,
        expected_rows=len(store.endpoint("target").features),
        declared_wnids=declared_wnids,
        name=f"posthoc target WNID role for {dataset}::{encoder}",
    )
    witness_indices = np.asarray(target_selection[calibration_rows:], dtype=np.int64)
    if witness_indices.shape != result.target_canonical_order.shape:
        raise IntegrityError("posthoc target selection has the wrong witness size")
    selected = target_wnids[witness_indices]
    canonical = selected[np.asarray(result.target_canonical_order, dtype=np.int64)]
    canonical_wnids = tuple(_wnid_string(value) for value in canonical)
    if not set(canonical_wnids).issubset(set(declared_wnids)):
        raise IntegrityError("posthoc witness WNID escaped the locked intersection")
    disagreement = np.asarray(
        result.prediction_world_zero != result.prediction_world_one, dtype=bool
    )
    if disagreement.shape != (len(canonical_wnids),):
        raise IntegrityError("posthoc disagreement vector has the wrong shape")
    sufficient_statistics: list[dict[str, Any]] = []
    for wnid in declared_wnids:
        mask = np.fromiter(
            (value == wnid for value in canonical_wnids),
            dtype=bool,
            count=len(canonical_wnids),
        )
        rows = int(np.count_nonzero(mask))
        if rows:
            sufficient_statistics.append({
                "wnid": wnid,
                "rows": rows,
                "prediction_disagreement_count": int(
                    np.count_nonzero(disagreement[mask])
                ),
            })
    if (
        sum(row["rows"] for row in sufficient_statistics) != len(canonical_wnids)
        or sum(
            row["prediction_disagreement_count"]
            for row in sufficient_statistics
        )
        != int(np.count_nonzero(disagreement))
    ):
        raise IntegrityError("posthoc WNID sufficient statistics do not close")
    return {
        "external_ground_truth_labels_used_for_optimization": False,
        "frozen_readout_predictions_used_for_optimization": True,
        "target_wnids_preflight_integrity_validated": True,
        "target_wnids_not_used_as_certificate_optimization_inputs": True,
        "target_wnids_used_for_posthoc_inference_after_assignment_fixed": True,
        "target_wnids_used_for_posthoc_inference": True,
        "sample_identifiers_used_for_cost_or_reward": False,
        "sample_identifiers_used_for_indexed_atom_administration": True,
        "sample_identifiers_parsed_for_semantics": False,
        "exact_duplicate_atom_tie_note": (
            "opaque_authenticated_sample_ids_break_only_exact_feature_ties;_"
            "the_tie_choice_can_change_posthoc_wnid_attribution_for_boundary_"
            "atoms_but_not_feature_cost_or_prediction_reward"
        ),
        "class_intersection_wnids_sha256": record["wnids_sha256"],
        "target_wnids_role": role,
        "cluster": "imagenet_wnid",
        "declared_cluster_count": len(declared_wnids),
        "observed_witness_cluster_count": len(sufficient_statistics),
        "per_wnid_sufficient_statistics": sufficient_statistics,
    }


def _compute_unit(
    lock: Mapping[str, Any],
    store_path: str | Path,
    *,
    dataset: str,
    encoder: str,
    geometry_name: str,
    seed: int,
    budget: float,
    purpose: str = "scientific_panel_compute",
):
    if purpose not in COMPUTE_PURPOSES:
        raise ValueError(f"unsupported grouped-compute purpose: {purpose}")
    cache_key = (
        str(lock["manifest_sha256"]),
        str(Path(store_path).expanduser().resolve()),
        str(dataset),
        str(encoder),
        int(seed),
        str(purpose),
    )
    cache_miss = cache_key not in _COMPUTE_GROUP_CACHE
    if cache_miss:
        _COMPUTE_GROUP_CACHE[cache_key] = _compute_group(
            lock,
            store_path,
            dataset=dataset,
            encoder=encoder,
            seed=seed,
            purpose=purpose,
        )
    store, computed, compute_event = _COMPUTE_GROUP_CACHE[cache_key]
    identity = (str(geometry_name), _budget_token(budget))
    if identity not in computed:
        raise IntegrityError("requested unit is outside its grouped computation")
    result, exact_assignment, exact, posthoc, split_diagnostics = computed[identity]
    return (
        store,
        result,
        exact_assignment,
        exact,
        posthoc,
        split_diagnostics,
        compute_event,
        cache_miss,
    )


_COMPUTE_GROUP_CACHE: dict[
    tuple[str, str, str, str, int, str],
    tuple[Any, dict[tuple[str, int], tuple[Any, ...]], dict[str, Any]],
] = {}


def _compute_group(
    lock: Mapping[str, Any],
    store_path: str | Path,
    *,
    dataset: str,
    encoder: str,
    seed: int,
    purpose: str,
):
    """Compute all three geometries and budgets with shared immutable work."""

    group_started_ns = time.perf_counter_ns()
    event_nonce = sha256_json({
        "lock_manifest_sha256": lock["manifest_sha256"],
        "dataset": dataset,
        "encoder": encoder,
        "split_seed": int(seed),
        "purpose": purpose,
        "wall_clock_nonce_ns": time.time_ns(),
        "monotonic_nonce_ns": group_started_ns,
        "process_id_nonce": os.getpid(),
    })
    config = lock["config"]
    if dataset not in config["datasets"] or encoder not in config["encoders"]:
        raise IntegrityError("unit dataset or encoder is outside the lock")
    if seed not in [int(value) for value in config["certificate_protocol"]["split_seeds"]]:
        raise IntegrityError("unit seed is outside the lock")
    stage_started_ns = time.perf_counter_ns()
    store = _validated_unit_store(lock, store_path, dataset, encoder)
    store_validation_ns = time.perf_counter_ns() - stage_started_ns
    stage_started_ns = time.perf_counter_ns()
    (
        reference,
        target,
        weight,
        bias,
        reference_digest,
        target_digest,
        reference_selection,
        target_selection,
        reference_ids,
        target_ids,
    ) = _split_inputs(store, config, seed=seed)
    split_selection_ns = time.perf_counter_ns() - stage_started_ns
    calibration = int(config["certificate_protocol"]["calibration_rows"])
    budgets = [
        float(value)
        for value in config["certificate_protocol"]["cost_budgets_relative"]
    ]
    primary_geometry = str(config["cost_geometries"][0])
    base_protocol = _unit_protocol(
        config, geometry=primary_geometry, budget=budgets[0]
    )
    stage_started_ns = time.perf_counter_ns()
    base_geometry = fit_certificate_geometry(
        target[:calibration],
        reference[:calibration],
        weight,
        bias,
        protocol=base_protocol,
        target_calibration_ids=target_ids[:calibration],
        reference_calibration_ids=reference_ids[:calibration],
    )
    shared_calibration_and_bures_fit_ns = (
        time.perf_counter_ns() - stage_started_ns
    )
    witness_target = target[calibration:]
    witness_reference = reference[calibration:]
    split_diagnostics = {
        "reference_split_parent_content_sha256": reference_digest,
        "target_split_parent_content_sha256": target_digest,
        "reference_selected_original_rows_sha256": sha256_json(
            reference_selection.tolist()
        ),
        "target_selected_original_rows_sha256": sha256_json(
            target_selection.tolist()
        ),
        "grouped_compute_policy": (
            "split_and_bures_once_per_dataset_encoder_seed;_cost_once_per_"
            "geometry;_three_budgets_reuse_the_prepared_dense_problem"
        ),
        "grouped_budget_count": len(budgets),
        "grouped_geometry_count": len(config["cost_geometries"]),
    }
    computed: dict[tuple[str, int], tuple[Any, ...]] = {}
    geometry_timings: list[dict[str, Any]] = []
    for geometry_name in config["cost_geometries"]:
        geometry_protocol = _unit_protocol(
            config, geometry=str(geometry_name), budget=budgets[0]
        )
        stage_started_ns = time.perf_counter_ns()
        geometry = (
            base_geometry
            if geometry_name == primary_geometry
            else derive_certificate_geometry(
                base_geometry, protocol=geometry_protocol
            )
        )
        geometry_derivation_ns = time.perf_counter_ns() - stage_started_ns
        stage_started_ns = time.perf_counter_ns()
        problem = prepare_certificate_problem(
            geometry,
            witness_target,
            witness_reference,
            target_sample_ids=target_ids[calibration:],
            reference_sample_ids=reference_ids[calibration:],
        )
        dense_problem_preparation_ns = time.perf_counter_ns() - stage_started_ns
        budget_timings: list[dict[str, Any]] = []
        for budget in budgets:
            unit_protocol = _unit_protocol(
                config, geometry=str(geometry_name), budget=budget
            )
            stage_started_ns = time.perf_counter_ns()
            result = construct_optimization_certificate_from_problem(
                problem, protocol=unit_protocol
            )
            certificate_ns = time.perf_counter_ns() - stage_started_ns
            unit_geometry = replace(
                geometry,
                protocol=unit_protocol,
                diagnostics={
                    **dict(geometry.diagnostics),
                    "protocol": unit_protocol.as_dict(),
                },
            )
            stage_started_ns = time.perf_counter_ns()
            exact_assignment, exact = run_exact_subset_validation(
                unit_geometry,
                witness_target,
                witness_reference,
                target_sample_ids=target_ids[calibration:],
                reference_sample_ids=reference_ids[calibration:],
                validation_rows=int(
                    config["certificate_protocol"]["exact_validation_rows"]
                ),
                time_limit_seconds=float(
                    config["certificate_protocol"][
                        "exact_milp_time_limit_seconds"
                    ]
                ),
            )
            exact_validation_ns = time.perf_counter_ns() - stage_started_ns
            # WNID arrays may already have been read by store-map preflight for
            # integrity. They enter this scientific computation only after the
            # assignment is fixed and never enter the feature-only objective.
            stage_started_ns = time.perf_counter_ns()
            posthoc = _posthoc_wnid_statistics(
                store,
                lock,
                result,
                dataset=dataset,
                encoder=encoder,
                calibration_rows=calibration,
                target_selection=target_selection,
            )
            posthoc_inference_ns = time.perf_counter_ns() - stage_started_ns
            budget_timings.append({
                "cost_budget_basis_points": _budget_token(budget),
                "certificate_nanoseconds": int(certificate_ns),
                "exact_validation_nanoseconds": int(exact_validation_ns),
                "posthoc_inference_nanoseconds": int(posthoc_inference_ns),
            })
            computed[(str(geometry_name), _budget_token(budget))] = (
                result,
                exact_assignment,
                exact,
                posthoc,
                dict(split_diagnostics),
            )
        geometry_timings.append({
            "cost_geometry": str(geometry_name),
            "geometry_derivation_nanoseconds": int(geometry_derivation_ns),
            "dense_problem_preparation_nanoseconds": int(
                dense_problem_preparation_ns
            ),
            "budgets": budget_timings,
        })
    group_wall_ns = time.perf_counter_ns() - group_started_ns
    stage_sum_ns = (
        store_validation_ns
        + split_selection_ns
        + shared_calibration_and_bures_fit_ns
        + sum(
            int(row["geometry_derivation_nanoseconds"])
            + int(row["dense_problem_preparation_nanoseconds"])
            + sum(
                int(budget_row["certificate_nanoseconds"])
                + int(budget_row["exact_validation_nanoseconds"])
                + int(budget_row["posthoc_inference_nanoseconds"])
                for budget_row in row["budgets"]
            )
            for row in geometry_timings
        )
    )
    if stage_sum_ns > group_wall_ns:
        raise RuntimeError(
            "non-overlapping compute timing stages exceed group wall time"
        )
    group_id = _group_id(dataset, encoder, seed)
    event_body = {
        "schema_version": COMPUTE_TIMING_SCHEMA,
        "purpose": purpose,
        "event_id": event_nonce,
        "group_id": group_id,
        "dataset": dataset,
        "encoder": encoder,
        "split_seed": int(seed),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "store_manifest_file_sha256": sha256_file(store.manifest_path),
        "store_manifest_payload_sha256": sha256_json(store.manifest),
        "clock": "time.perf_counter_ns_monotonic",
        "unit_ids_computed": [
            _unit_id(_unit_identity(dataset, encoder, str(geometry_name), seed, budget))
            for geometry_name in config["cost_geometries"]
            for budget in budgets
        ],
        "group_wall_nanoseconds": int(group_wall_ns),
        "store_validation_nanoseconds": int(store_validation_ns),
        "split_selection_nanoseconds": int(split_selection_ns),
        "shared_calibration_and_bures_fit_nanoseconds": int(
            shared_calibration_and_bures_fit_ns
        ),
        "geometry_stages": geometry_timings,
        "stage_sum_nanoseconds": int(stage_sum_ns),
        "unattributed_python_overhead_nanoseconds": int(
            group_wall_ns - stage_sum_ns
        ),
        "peak_rss_bytes": None,
        "peak_rss_status": PEAK_RSS_STATUS,
        "scope": (
            "successful_group_computation_reaching_at_least_one_sealed_unit_"
            "or_verification_report;_events_interrupted_before_publication_"
            "are_not_observable"
        ),
    }
    compute_event = {
        **event_body,
        "timing_payload_sha256": sha256_json(event_body),
    }
    return store, computed, compute_event


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    np.save(path, np.asarray(value), allow_pickle=False)
    return describe_npy(path, relative_to=root)


def run_unit(
    lock_path: str | Path,
    store_path: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    encoder: str,
    geometry_name: str,
    seed: int,
    budget: float,
) -> Path:
    lock_source = Path(lock_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite unit output: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    started_ns = time.perf_counter_ns()
    lock = _load_lock(lock_source)
    (
        store,
        result,
        exact_assignment,
        exact,
        posthoc_inference,
        split_diagnostics,
        compute_event,
        compute_event_owner,
    ) = _compute_unit(
        lock,
        store_path,
        dataset=dataset,
        encoder=encoder,
        geometry_name=geometry_name,
        seed=seed,
        budget=budget,
        purpose="scientific_panel_compute",
    )
    output_stage_started_ns = time.perf_counter_ns()
    temporary.mkdir(parents=True)
    arrays = {
        "assignment_world_zero": _save_array(
            temporary, "assignment_world_zero", result.assignment_world_zero
        ),
        "assignment_world_one": _save_array(
            temporary, "assignment_world_one", result.assignment_world_one
        ),
        "prediction_world_zero": _save_array(
            temporary, "prediction_world_zero", result.prediction_world_zero
        ),
        "prediction_world_one": _save_array(
            temporary, "prediction_world_one", result.prediction_world_one
        ),
        "target_canonical_order": _save_array(
            temporary, "target_canonical_order", result.target_canonical_order
        ),
        "reference_canonical_order": _save_array(
            temporary, "reference_canonical_order", result.reference_canonical_order
        ),
        "exact_subset_assignment": _save_array(
            temporary, "exact_subset_assignment", exact_assignment
        ),
    }
    output_stage_ns = time.perf_counter_ns() - output_stage_started_ns
    unit_id = _unit_id(
        _unit_identity(dataset, encoder, geometry_name, seed, budget)
    )
    invocation_ns = time.perf_counter_ns() - started_ns
    payload = {
        "schema_version": UNIT_SCHEMA,
        "certificate_schema_version": CERTIFICATE_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "unit_id": unit_id,
        "dataset": dataset,
        "encoder": encoder,
        "cost_geometry": geometry_name,
        "split_seed": int(seed),
        "cost_budget_relative": float(budget),
        "lock_file_sha256": sha256_file(lock_source),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "store_manifest_file_sha256": sha256_file(store.manifest_path),
        "store_manifest_payload_sha256": sha256_json(store.manifest),
        "information_scope": INFORMATION_SCOPE,
        "external_ground_truth_labels_used_for_optimization": False,
        "frozen_readout_predictions_used_for_optimization": True,
        "target_wnids_used_for_posthoc_inference": True,
        "target_wnids_not_used_as_certificate_optimization_inputs": True,
        "sample_identifiers_used_for_cost_or_reward": False,
        "sample_identifiers_used_for_indexed_atom_administration": True,
        "sample_identifiers_parsed_for_semantics": False,
        "sample_identifier_tie_scope": "exact_duplicate_feature_rows_only",
        "sample_identifiers_used_for_integrity_validation": True,
        "prespecified_public_wnid_support_used_for_store_construction": True,
        "pairing_used": False,
        "result": result.summary(),
        "exact_small_instance_validation": exact,
        "posthoc_inference": posthoc_inference,
        "split_diagnostics": split_diagnostics,
        "arrays": arrays,
        "compute": {
            # Both representations are derived from one clock read so the
            # serialized seconds cannot drift independently of the exact
            # integer duration authenticated below.
            "elapsed_seconds": invocation_ns / 1e9,
            "invocation_wall_nanoseconds_before_manifest_write": int(
                invocation_ns
            ),
            "temporary_directory_and_array_serialization_nanoseconds": int(
                output_stage_ns
            ),
            "shared_group_compute_event": compute_event,
            "shared_group_compute_event_owner_for_this_unit": bool(
                compute_event_owner
            ),
            "shared_event_accounting_policy": (
                "event_is_repeated_in_its_unit_manifests_and_must_be_"
                "deduplicated_by_event_id_before_summing"
            ),
            "manifest_publication_timing_nanoseconds": None,
            "manifest_publication_timing_status": lock["config"]["execution"][
                "compute_accounting"
            ]["manifest_publication_timing_status"],
            "runtime_contract": _runtime_contract(),
        },
    }
    write_json_atomic(temporary / "UNIT_MANIFEST.json", seal_manifest(payload))
    os.replace(temporary, output)
    return output / "UNIT_MANIFEST.json"


def verify_unit(
    lock_path: str | Path,
    store_path: str | Path,
    unit_path: str | Path,
) -> dict[str, Any]:
    verification_started_ns = time.perf_counter_ns()
    lock_source = Path(lock_path).expanduser().resolve()
    unit_source = Path(unit_path).expanduser().resolve()
    manifest_path = (
        unit_source / "UNIT_MANIFEST.json" if unit_source.is_dir() else unit_source
    )
    manifest = _mapping(read_json(manifest_path), name="unit manifest")
    verify_manifest_checksum(manifest)
    _require_exact_keys(
        manifest, UNIT_MANIFEST_FIELDS, name="optimization unit manifest"
    )
    if manifest.get("schema_version") != UNIT_SCHEMA:
        raise IntegrityError("unsupported optimization-unit schema")
    root = manifest_path.parent
    arrays = _mapping(manifest.get("arrays"), name="unit arrays")
    if set(arrays) != set(ARRAY_NAMES):
        raise IntegrityError("unit assignment-array inventory is incomplete")
    for name in ARRAY_NAMES:
        _require_exact_keys(
            _mapping(arrays[name], name=f"unit array {name}"),
            ARRAY_RECORD_FIELDS,
            name=f"unit array record {name}",
        )
    stored_arrays = {
        name: np.asarray(np.load(
            verify_file_record(root, _mapping(arrays[name], name=name)),
            allow_pickle=False,
        ))
        for name in ARRAY_NAMES
    }
    lock = _load_lock(lock_source)
    dataset = str(manifest.get("dataset"))
    encoder = str(manifest.get("encoder"))
    seed_value = manifest.get("split_seed")
    if not _exact_int(seed_value):
        raise IntegrityError("optimization unit split seed is not an integer")
    seed = seed_value
    _validated_unit_compute_record(
        manifest.get("compute"),
        lock=lock,
        dataset=dataset,
        encoder=encoder,
        seed=seed,
    )
    if (
        manifest.get("lock_file_sha256") != sha256_file(lock_source)
        or manifest.get("lock_manifest_sha256") != lock["manifest_sha256"]
        or manifest.get("external_ground_truth_labels_used_for_optimization")
        is not False
        or manifest.get("frozen_readout_predictions_used_for_optimization")
        is not True
        or manifest.get("target_wnids_used_for_posthoc_inference") is not True
        or manifest.get("target_wnids_not_used_as_certificate_optimization_inputs")
        is not True
        or manifest.get("sample_identifiers_used_for_cost_or_reward") is not False
        or manifest.get("sample_identifiers_used_for_indexed_atom_administration")
        is not True
        or manifest.get("sample_identifiers_parsed_for_semantics") is not False
        or manifest.get("sample_identifier_tie_scope")
        != "exact_duplicate_feature_rows_only"
        or manifest.get("sample_identifiers_used_for_integrity_validation") is not True
        or manifest.get(
            "prespecified_public_wnid_support_used_for_store_construction"
        )
        is not True
        or manifest.get("pairing_used") is not False
    ):
        raise IntegrityError("unit lock or information-scope declaration changed")
    pre_reconstruction_validation_ns = (
        time.perf_counter_ns() - verification_started_ns
    )
    (
        store,
        result,
        exact_assignment,
        exact,
        posthoc_inference,
        split_diagnostics,
        verification_event,
        verification_event_owner,
    ) = _compute_unit(
        lock,
        store_path,
        dataset=dataset,
        encoder=encoder,
        geometry_name=str(manifest["cost_geometry"]),
        seed=seed,
        budget=float(manifest["cost_budget_relative"]),
        purpose="full_reconstruction_verification",
    )
    post_reconstruction_started_ns = time.perf_counter_ns()
    expected_arrays = {
        "assignment_world_zero": result.assignment_world_zero,
        "assignment_world_one": result.assignment_world_one,
        "prediction_world_zero": result.prediction_world_zero,
        "prediction_world_one": result.prediction_world_one,
        "target_canonical_order": result.target_canonical_order,
        "reference_canonical_order": result.reference_canonical_order,
        "exact_subset_assignment": exact_assignment,
    }
    for name, expected in expected_arrays.items():
        if not np.array_equal(stored_arrays[name], expected):
            raise IntegrityError(f"unit deterministic reconstruction failed: {name}")
    if (
        manifest.get("result") != result.summary()
        or manifest.get("exact_small_instance_validation") != exact
        or manifest.get("posthoc_inference") != posthoc_inference
        or manifest.get("split_diagnostics") != split_diagnostics
        or manifest.get("store_manifest_file_sha256")
        != sha256_file(store.manifest_path)
    ):
        raise IntegrityError("unit metrics/cost/reward reconstruction failed")
    verification_event = _validated_compute_event(
        verification_event,
        lock=lock,
        dataset=dataset,
        encoder=encoder,
        seed=seed,
        purpose="full_reconstruction_verification",
    )
    post_reconstruction_validation_ns = (
        time.perf_counter_ns() - post_reconstruction_started_ns
    )
    verification_wall_ns = time.perf_counter_ns() - verification_started_ns
    non_group_ns = verification_wall_ns - (
        int(verification_event["group_wall_nanoseconds"])
        if verification_event_owner
        else 0
    )
    if non_group_ns < 0:
        raise RuntimeError("verification group timing exceeds its enclosing call")
    return {
        "status": "PASS",
        "assignments_recomputed": True,
        "costs_recomputed_from_endpoint_features": True,
        "rewards_recomputed_from_frozen_readout": True,
        "bijections_revalidated": True,
        "duplicate_row_admissibility_revalidated": True,
        "exact_small_instance_milp_revalidated": True,
        "posthoc_wnid_statistics_revalidated": True,
        "gap_target_met": bool(result.summary()["meets_gap_target"]),
        "primal_dual_gap_percentage_points": result.summary()[
            "primal_dual_gap_percentage_points"
        ],
        "verification_compute_event": verification_event,
        "verification_compute_event_owner_for_this_report": bool(
            verification_event_owner
        ),
        "verification_timing": {
            "clock": "time.perf_counter_ns_monotonic",
            "verification_wall_nanoseconds_before_report_seal": int(
                verification_wall_ns
            ),
            "pre_reconstruction_artifact_and_contract_validation_nanoseconds": int(
                pre_reconstruction_validation_ns
            ),
            "post_reconstruction_comparison_nanoseconds": int(
                post_reconstruction_validation_ns
            ),
            "non_group_validation_and_python_overhead_nanoseconds": int(
                non_group_ns
            ),
            "verification_manifest_publication_timing_nanoseconds": None,
            "peak_rss_bytes": None,
            "peak_rss_status": PEAK_RSS_STATUS,
        },
    }


def _budget_token(value: float) -> int:
    if not _finite_number(value):
        raise IntegrityError(f"cost budget must be a finite JSON number: {value}")
    token = int(round(10_000.0 * float(value)))
    if abs(float(value) - token / 10_000.0) > 1e-12:
        raise IntegrityError(f"cost budget is not representable in basis points: {value}")
    return token


def _unit_identity(
    dataset: str,
    encoder: str,
    geometry: str,
    seed: int,
    budget: float,
) -> tuple[str, str, str, int, int]:
    if not _exact_int(seed):
        raise IntegrityError(f"split seed must be an integer: {seed}")
    return dataset, encoder, geometry, seed, _budget_token(budget)


def _unit_id(identity: tuple[str, str, str, int, int]) -> str:
    dataset, encoder, geometry, seed, budget = identity
    return (
        f"{dataset}__{encoder}__{geometry}__seed_{seed}__"
        f"budget_{budget:04d}bp"
    )


def _group_id(dataset: str, encoder: str, seed: int) -> str:
    if not _exact_int(seed):
        raise IntegrityError(f"split seed must be an integer: {seed}")
    return f"{dataset}__{encoder}__seed_{seed}"


def _nonnegative_nanoseconds(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrityError(f"{name} must be a nonnegative integer")
    return value


def _validated_compute_event(
    value: Any,
    *,
    lock: Mapping[str, Any],
    dataset: str,
    encoder: str,
    seed: int,
    purpose: str,
) -> dict[str, Any]:
    """Validate one sealed-in-parent, deduplicable grouped timing event."""

    value = _mapping(value, name="grouped compute timing event")
    _require_exact_keys(value, COMPUTE_EVENT_FIELDS, name="grouped compute event")
    event = dict(_mapping(value, name="grouped compute timing event"))
    digest = event.pop("timing_payload_sha256", None)
    config = lock["config"]
    expected_units = [
        _unit_id(_unit_identity(dataset, encoder, geometry, seed, budget))
        for geometry in config["cost_geometries"]
        for budget in config["certificate_protocol"]["cost_budgets_relative"]
    ]
    store_record = _mapping(
        lock["stores"].get(_store_key(dataset, encoder)), name="locked store"
    )
    if (
        value.get("schema_version") != COMPUTE_TIMING_SCHEMA
        or value.get("purpose") != purpose
        or not _valid_sha256(value.get("event_id"))
        or value.get("group_id") != _group_id(dataset, encoder, seed)
        or value.get("dataset") != dataset
        or value.get("encoder") != encoder
        or not isinstance(value.get("split_seed"), int)
        or isinstance(value.get("split_seed"), bool)
        or value.get("split_seed") != seed
        or value.get("lock_manifest_sha256") != lock["manifest_sha256"]
        or value.get("store_manifest_file_sha256")
        != store_record["manifest_file_sha256"]
        or value.get("store_manifest_payload_sha256")
        != store_record["manifest_payload_sha256"]
        or value.get("clock") != "time.perf_counter_ns_monotonic"
        or value.get("unit_ids_computed") != expected_units
        or value.get("peak_rss_bytes") is not None
        or value.get("peak_rss_status") != PEAK_RSS_STATUS
        or value.get("scope")
        != (
            "successful_group_computation_reaching_at_least_one_sealed_unit_"
            "or_verification_report;_events_interrupted_before_publication_"
            "are_not_observable"
        )
        or digest != sha256_json(event)
    ):
        raise IntegrityError("grouped compute timing identity or checksum changed")
    group_wall_ns = _nonnegative_nanoseconds(
        value.get("group_wall_nanoseconds"), name="group wall timing"
    )
    fixed_stages = sum(
        _nonnegative_nanoseconds(value.get(field), name=field)
        for field in (
            "store_validation_nanoseconds",
            "split_selection_nanoseconds",
            "shared_calibration_and_bures_fit_nanoseconds",
        )
    )
    geometry_rows = value.get("geometry_stages")
    if not isinstance(geometry_rows, list) or [
        row.get("cost_geometry") if isinstance(row, Mapping) else None
        for row in geometry_rows
    ] != list(config["cost_geometries"]):
        raise IntegrityError("grouped compute geometry timing inventory changed")
    stage_sum_ns = fixed_stages
    expected_budgets = [
        _budget_token(value)
        for value in config["certificate_protocol"]["cost_budgets_relative"]
    ]
    for geometry_row_value in geometry_rows:
        geometry_row = _mapping(
            geometry_row_value, name="compute geometry timing"
        )
        _require_exact_keys(
            geometry_row,
            COMPUTE_GEOMETRY_FIELDS,
            name="compute geometry timing",
        )
        stage_sum_ns += _nonnegative_nanoseconds(
            geometry_row.get("geometry_derivation_nanoseconds"),
            name="geometry derivation timing",
        )
        stage_sum_ns += _nonnegative_nanoseconds(
            geometry_row.get("dense_problem_preparation_nanoseconds"),
            name="dense problem preparation timing",
        )
        budget_rows = geometry_row.get("budgets")
        if not isinstance(budget_rows, list) or [
            row.get("cost_budget_basis_points")
            if isinstance(row, Mapping)
            else None
            for row in budget_rows
        ] != expected_budgets:
            raise IntegrityError("grouped compute budget timing inventory changed")
        for budget_row_value in budget_rows:
            budget_row = _mapping(budget_row_value, name="budget timing")
            _require_exact_keys(
                budget_row, COMPUTE_BUDGET_FIELDS, name="compute budget timing"
            )
            stage_sum_ns += sum(
                _nonnegative_nanoseconds(budget_row.get(field), name=field)
                for field in (
                    "certificate_nanoseconds",
                    "exact_validation_nanoseconds",
                    "posthoc_inference_nanoseconds",
                )
            )
    declared_stage_sum_ns = _nonnegative_nanoseconds(
        value.get("stage_sum_nanoseconds"), name="compute stage sum"
    )
    overhead_ns = _nonnegative_nanoseconds(
        value.get("unattributed_python_overhead_nanoseconds"),
        name="unattributed compute overhead",
    )
    if (
        declared_stage_sum_ns != stage_sum_ns
        or stage_sum_ns + overhead_ns != group_wall_ns
    ):
        raise IntegrityError("grouped compute timing stages do not close")
    return dict(value)


def _compute_accounting_summary(
    rows: list[tuple[Mapping[str, Any], bool, str]],
    *,
    expected_group_ids: set[str],
    purpose: str,
) -> dict[str, Any]:
    """Deduplicate repeated group events and sum non-overlapping timings."""

    events: dict[str, dict[str, Any]] = {}
    owners: dict[str, int] = {}
    represented_units: dict[str, set[str]] = {}
    for event_value, owner, unit_id in rows:
        event = dict(event_value)
        event_id = str(event["event_id"])
        if event.get("purpose") != purpose or unit_id not in event[
            "unit_ids_computed"
        ]:
            raise IntegrityError("compute event is attached to the wrong unit")
        if event_id in events and events[event_id] != event:
            raise IntegrityError("compute event ID has conflicting timing payloads")
        events[event_id] = event
        owners[event_id] = owners.get(event_id, 0) + int(bool(owner))
        represented_units.setdefault(event_id, set()).add(unit_id)
    if not events or any(count != 1 for count in owners.values()):
        raise IntegrityError("each represented compute event needs exactly one owner")
    observed_group_ids = {str(event["group_id"]) for event in events.values()}
    if observed_group_ids != expected_group_ids:
        raise IntegrityError("compute timing group coverage is incomplete")
    totals = {
        "group_wall_nanoseconds": 0,
        "store_validation_nanoseconds": 0,
        "split_selection_nanoseconds": 0,
        "shared_calibration_and_bures_fit_nanoseconds": 0,
        "geometry_derivation_nanoseconds": 0,
        "dense_problem_preparation_nanoseconds": 0,
        "certificate_nanoseconds": 0,
        "exact_validation_nanoseconds": 0,
        "posthoc_inference_nanoseconds": 0,
        "unattributed_python_overhead_nanoseconds": 0,
    }
    totals_by_geometry: dict[str, dict[str, Any]] = {}
    for event in events.values():
        for field in (
            "group_wall_nanoseconds",
            "store_validation_nanoseconds",
            "split_selection_nanoseconds",
            "shared_calibration_and_bures_fit_nanoseconds",
            "unattributed_python_overhead_nanoseconds",
        ):
            totals[field] += int(event[field])
        for geometry_row in event["geometry_stages"]:
            geometry_name = str(geometry_row["cost_geometry"])
            geometry_totals = totals_by_geometry.setdefault(geometry_name, {
                "geometry_derivation_nanoseconds": 0,
                "dense_problem_preparation_nanoseconds": 0,
                "budgets": {},
            })
            totals["geometry_derivation_nanoseconds"] += int(
                geometry_row["geometry_derivation_nanoseconds"]
            )
            geometry_totals["geometry_derivation_nanoseconds"] += int(
                geometry_row["geometry_derivation_nanoseconds"]
            )
            totals["dense_problem_preparation_nanoseconds"] += int(
                geometry_row["dense_problem_preparation_nanoseconds"]
            )
            geometry_totals["dense_problem_preparation_nanoseconds"] += int(
                geometry_row["dense_problem_preparation_nanoseconds"]
            )
            for budget_row in geometry_row["budgets"]:
                budget_key = (
                    f"{int(budget_row['cost_budget_basis_points']):04d}bp"
                )
                budget_totals = geometry_totals["budgets"].setdefault(
                    budget_key,
                    {
                        "certificate_nanoseconds": 0,
                        "exact_validation_nanoseconds": 0,
                        "posthoc_inference_nanoseconds": 0,
                    },
                )
                for field in (
                    "certificate_nanoseconds",
                    "exact_validation_nanoseconds",
                    "posthoc_inference_nanoseconds",
                ):
                    totals[field] += int(budget_row[field])
                    budget_totals[field] += int(budget_row[field])
    return {
        "schema_version": COMPUTE_TIMING_SCHEMA,
        "purpose": purpose,
        "accounting_scope": (
            "successful_authenticated_group_events_represented_by_published_"
            "records_only"
        ),
        "expected_distinct_groups": len(expected_group_ids),
        "observed_distinct_groups": len(observed_group_ids),
        "successful_group_compute_events": len(events),
        "successful_retry_group_compute_events": (
            len(events) - len(observed_group_ids)
        ),
        "events_deduplicated_across_unit_records": len(rows) - len(events),
        "event_inventory": [
            {
                "event_id": event_id,
                "timing_payload_sha256": events[event_id][
                    "timing_payload_sha256"
                ],
                "group_id": events[event_id]["group_id"],
                "group_wall_nanoseconds": events[event_id][
                    "group_wall_nanoseconds"
                ],
                "represented_unit_records": len(represented_units[event_id]),
            }
            for event_id in sorted(events)
        ],
        "totals_nanoseconds": totals,
        "totals_seconds": {
            field.removesuffix("_nanoseconds") + "_seconds": value / 1e9
            for field, value in totals.items()
        },
        "totals_by_geometry_nanoseconds": totals_by_geometry,
        "totals_by_geometry_seconds": {
            geometry: {
                "geometry_derivation_seconds": row[
                    "geometry_derivation_nanoseconds"
                ] / 1e9,
                "dense_problem_preparation_seconds": row[
                    "dense_problem_preparation_nanoseconds"
                ] / 1e9,
                "budgets": {
                    budget: {
                        field.removesuffix("_nanoseconds") + "_seconds": (
                            value / 1e9
                        )
                        for field, value in budget_row.items()
                    }
                    for budget, budget_row in row["budgets"].items()
                },
            }
            for geometry, row in totals_by_geometry.items()
        },
        "peak_rss_bytes": None,
        "peak_rss_status": PEAK_RSS_STATUS,
        "not_included": [
            "events_interrupted_or_failed_before_any_sealed_record",
            "manifest_seal_write_fsync_and_atomic_rename_duration",
        ],
    }


def _validated_unit_compute_record(
    value: Any,
    *,
    lock: Mapping[str, Any],
    dataset: str,
    encoder: str,
    seed: int,
) -> tuple[dict[str, Any], bool]:
    compute = _mapping(value, name="unit compute record")
    _require_exact_keys(compute, UNIT_COMPUTE_FIELDS, name="unit compute record")
    elapsed_value = compute.get("elapsed_seconds")
    if (
        not isinstance(elapsed_value, (int, float))
        or isinstance(elapsed_value, bool)
    ):
        raise IntegrityError("optimization unit elapsed timing is not numeric")
    elapsed = float(elapsed_value)
    invocation_ns = _nonnegative_nanoseconds(
        compute.get("invocation_wall_nanoseconds_before_manifest_write"),
        name="unit invocation timing",
    )
    output_ns = _nonnegative_nanoseconds(
        compute.get("temporary_directory_and_array_serialization_nanoseconds"),
        name="unit array serialization timing",
    )
    event = _validated_compute_event(
        compute.get("shared_group_compute_event"),
        lock=lock,
        dataset=dataset,
        encoder=encoder,
        seed=seed,
        purpose="scientific_panel_compute",
    )
    owner = compute.get("shared_group_compute_event_owner_for_this_unit")
    if (
        compute.get("runtime_contract") != lock["freeze_runtime"]
        or not np.isfinite(elapsed)
        or elapsed < 0.0
        or elapsed != invocation_ns / 1e9
        or invocation_ns < output_ns
        or not isinstance(owner, bool)
        or (owner and invocation_ns < int(event["group_wall_nanoseconds"]))
        or compute.get("shared_event_accounting_policy")
        != (
            "event_is_repeated_in_its_unit_manifests_and_must_be_"
            "deduplicated_by_event_id_before_summing"
        )
        or compute.get("manifest_publication_timing_nanoseconds") is not None
        or compute.get("manifest_publication_timing_status")
        != lock["config"]["execution"]["compute_accounting"][
            "manifest_publication_timing_status"
        ]
    ):
        raise IntegrityError("optimization unit runtime record changed")
    return event, owner


def _validated_verification_timing(
    report: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    identity: tuple[str, str, str, int, int],
) -> tuple[dict[str, Any], bool]:
    dataset, encoder, _, seed, _ = identity
    event = _validated_compute_event(
        report.get("verification_compute_event"),
        lock=lock,
        dataset=dataset,
        encoder=encoder,
        seed=seed,
        purpose="full_reconstruction_verification",
    )
    owner = report.get("verification_compute_event_owner_for_this_report")
    timing = _mapping(
        report.get("verification_timing"), name="verification timing"
    )
    _require_exact_keys(
        timing, VERIFICATION_TIMING_FIELDS, name="verification timing"
    )
    wall_ns = _nonnegative_nanoseconds(
        timing.get("verification_wall_nanoseconds_before_report_seal"),
        name="verification wall timing",
    )
    pre_ns = _nonnegative_nanoseconds(
        timing.get(
            "pre_reconstruction_artifact_and_contract_validation_nanoseconds"
        ),
        name="verification pre-reconstruction timing",
    )
    post_ns = _nonnegative_nanoseconds(
        timing.get("post_reconstruction_comparison_nanoseconds"),
        name="verification post-reconstruction timing",
    )
    non_group_ns = _nonnegative_nanoseconds(
        timing.get("non_group_validation_and_python_overhead_nanoseconds"),
        name="verification non-group timing",
    )
    if (
        not isinstance(owner, bool)
        or timing.get("clock") != "time.perf_counter_ns_monotonic"
        or wall_ns
        != non_group_ns
        + (int(event["group_wall_nanoseconds"]) if owner else 0)
        or pre_ns + post_ns > non_group_ns
        or timing.get("verification_manifest_publication_timing_nanoseconds")
        is not None
        or timing.get("peak_rss_bytes") is not None
        or timing.get("peak_rss_status") != PEAK_RSS_STATUS
    ):
        raise IntegrityError("verification timing stages do not close")
    return event, owner


def _verification_compute_accounting(
    reports: list[Mapping[str, Any]], *, expected_group_ids: set[str]
) -> dict[str, Any]:
    event_summary = _compute_accounting_summary(
        [
            (
                _mapping(
                    report.get("verification_compute_event"),
                    name="verification compute event",
                ),
                bool(
                    report.get(
                        "verification_compute_event_owner_for_this_report"
                    )
                ),
                str(report["unit_id"]),
            )
            for report in reports
        ],
        expected_group_ids=expected_group_ids,
        purpose="full_reconstruction_verification",
    )
    timing_rows = [
        _mapping(report.get("verification_timing"), name="verification timing")
        for report in reports
    ]
    non_group_ns = sum(
        int(row["non_group_validation_and_python_overhead_nanoseconds"])
        for row in timing_rows
    )
    return {
        **event_summary,
        "unit_report_count": len(reports),
        "unit_non_group_validation_and_python_overhead_nanoseconds": int(
            non_group_ns
        ),
        "unit_non_group_validation_and_python_overhead_seconds": (
            non_group_ns / 1e9
        ),
        "pre_reconstruction_artifact_and_contract_validation_nanoseconds": int(
            sum(
                int(row[
                    "pre_reconstruction_artifact_and_contract_validation_nanoseconds"
                ])
                for row in timing_rows
            )
        ),
        "post_reconstruction_comparison_nanoseconds": int(sum(
            int(row["post_reconstruction_comparison_nanoseconds"])
            for row in timing_rows
        )),
        "verification_manifest_publication_timing_nanoseconds": None,
    }


def _expected_identities(
    config: Mapping[str, Any],
) -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(sorted(
        _unit_identity(dataset, encoder, geometry, seed, budget)
        for dataset in config["datasets"]
        for encoder in config["encoders"]
        for geometry in config["cost_geometries"]
        for seed in config["certificate_protocol"]["split_seeds"]
        for budget in config["certificate_protocol"]["cost_budgets_relative"]
    ))


def _sharded_identities(
    config: Mapping[str, Any], *, shard_count: int, shard_index: int
) -> tuple[tuple[str, str, str, int, int], ...]:
    if not _exact_int(shard_count) or not _exact_int(shard_index):
        raise ValueError("shard count and index must be integers")
    count = shard_count
    index = shard_index
    if count < 1 or index < 0 or index >= count:
        raise ValueError("shard index must lie in [0, shard count)")
    # Keep all nine geometry/budget units for a dataset/encoder/seed group in
    # one process. This is a scientific no-op, but lets the runner fit Bures
    # once and construct each dense geometry cost once rather than nine times.
    grouped: dict[tuple[str, str, int], list[tuple[str, str, str, int, int]]] = {}
    for identity in _expected_identities(config):
        dataset, encoder, _, seed, _ = identity
        grouped.setdefault((dataset, encoder, seed), []).append(identity)
    selected: list[tuple[str, str, str, int, int]] = []
    for position, key in enumerate(sorted(grouped)):
        if position % count == index:
            selected.extend(sorted(grouped[key]))
    return tuple(selected)


def _validate_aggregate_unit(
    manifest_path: Path,
    *,
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    allow_release_provenance: bool = False,
) -> tuple[tuple[str, str, str, int, int], dict[str, Any]]:
    manifest = _mapping(read_json(manifest_path), name="optimization unit")
    verify_manifest_checksum(manifest)
    _require_exact_keys(
        manifest,
        UNIT_MANIFEST_FIELDS
        | ({"release_provenance"} if allow_release_provenance else set()),
        name="optimization unit manifest",
    )
    split_seed = manifest.get("split_seed")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise IntegrityError("optimization unit split seed is not an integer")
    budget_value = manifest.get("cost_budget_relative")
    if not _finite_number(budget_value):
        raise IntegrityError("optimization unit cost budget is not numeric")
    identity = _unit_identity(
        str(manifest.get("dataset")),
        str(manifest.get("encoder")),
        str(manifest.get("cost_geometry")),
        split_seed,
        budget_value,
    )
    dataset, encoder, geometry, seed, budget = identity
    config = lock["config"]
    if (
        manifest.get("schema_version") != UNIT_SCHEMA
        or manifest.get("certificate_schema_version") != CERTIFICATE_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("scientific_result") is not True
        or manifest.get("unit_id") != _unit_id(identity)
        or dataset not in config["datasets"]
        or encoder not in config["encoders"]
        or geometry not in config["cost_geometries"]
        or seed not in map(int, config["certificate_protocol"]["split_seeds"])
        or budget
        not in {
            _budget_token(value)
            for value in config["certificate_protocol"][
                "cost_budgets_relative"
            ]
        }
        or manifest.get("lock_file_sha256") != lock_file_sha256
        or manifest.get("lock_manifest_sha256") != lock["manifest_sha256"]
        or manifest.get("external_ground_truth_labels_used_for_optimization")
        is not False
        or manifest.get("frozen_readout_predictions_used_for_optimization")
        is not True
        or manifest.get("target_wnids_used_for_posthoc_inference") is not True
        or manifest.get("target_wnids_not_used_as_certificate_optimization_inputs")
        is not True
        or manifest.get("sample_identifiers_used_for_cost_or_reward") is not False
        or manifest.get("sample_identifiers_used_for_indexed_atom_administration")
        is not True
        or manifest.get("sample_identifiers_parsed_for_semantics") is not False
        or manifest.get("sample_identifier_tie_scope")
        != "exact_duplicate_feature_rows_only"
        or manifest.get("sample_identifiers_used_for_integrity_validation") is not True
        or manifest.get(
            "prespecified_public_wnid_support_used_for_store_construction"
        )
        is not True
        or manifest.get("pairing_used") is not False
    ):
        raise IntegrityError(f"optimization unit contract changed: {manifest_path}")
    compute_event, compute_event_owner = _validated_unit_compute_record(
        manifest.get("compute"),
        lock=lock,
        dataset=dataset,
        encoder=encoder,
        seed=seed,
    )
    compute = _mapping(manifest.get("compute"), name="unit compute record")
    store_record = _mapping(
        lock["stores"].get(_store_key(dataset, encoder)), name="locked store"
    )
    if (
        manifest.get("store_manifest_file_sha256")
        != store_record["manifest_file_sha256"]
        or manifest.get("store_manifest_payload_sha256")
        != store_record["manifest_payload_sha256"]
    ):
        raise IntegrityError("optimization unit uses the wrong feature store")
    arrays = _mapping(manifest.get("arrays"), name="unit arrays")
    if set(arrays) != set(ARRAY_NAMES):
        raise IntegrityError("optimization unit array inventory changed")
    for name in ARRAY_NAMES:
        _require_exact_keys(
            _mapping(arrays[name], name=f"unit array {name}"),
            ARRAY_RECORD_FIELDS,
            name=f"unit array record {name}",
        )
    loaded = {
        name: np.asarray(np.load(
            verify_file_record(
                manifest_path.parent, _mapping(arrays[name], name=name)
            ),
            allow_pickle=False,
        ))
        for name in ARRAY_NAMES
    }
    witness_rows = int(config["certificate_protocol"]["witness_rows"])
    exact_rows = int(config["certificate_protocol"]["exact_validation_rows"])
    for name in (
        "assignment_world_zero",
        "assignment_world_one",
        "target_canonical_order",
        "reference_canonical_order",
    ):
        if (
            loaded[name].shape != (witness_rows,)
            or not np.array_equal(
                np.sort(loaded[name].astype(np.int64, copy=False)),
                np.arange(witness_rows, dtype=np.int64),
            )
        ):
            raise IntegrityError(f"unit has a non-bijection array: {name}")
    for name in ("prediction_world_zero", "prediction_world_one"):
        if loaded[name].shape != (witness_rows,):
            raise IntegrityError(f"unit prediction array has wrong shape: {name}")
    if (
        loaded["exact_subset_assignment"].shape != (exact_rows,)
        or not np.array_equal(
            np.sort(loaded["exact_subset_assignment"].astype(np.int64, copy=False)),
            np.arange(exact_rows, dtype=np.int64),
        )
    ):
        raise IntegrityError("unit exact-subset assignment is not a bijection")
    result = _mapping(manifest.get("result"), name="unit certificate result")
    _require_exact_keys(
        result, CERTIFICATE_RESULT_FIELDS, name="unit certificate result"
    )
    interpretation = _mapping(
        result.get("scientific_interpretation"),
        name="unit certificate scientific interpretation",
    )
    if dict(interpretation) != CERTIFICATE_SCIENTIFIC_INTERPRETATION:
        raise IntegrityError("unit certificate scientific interpretation changed")
    diagnostics = _mapping(
        result.get("diagnostics"), name="unit certificate diagnostics"
    )
    _require_exact_keys(
        diagnostics,
        CERTIFICATE_DIAGNOSTIC_FIELDS,
        name="unit certificate diagnostics",
    )
    if (
        result.get("schema_version") != CERTIFICATE_SCHEMA
        or result.get("status") != "PASS"
        or result.get("assignment_world_zero_sha256")
        != _array_sha256(loaded["assignment_world_zero"])
        or result.get("assignment_world_one_sha256")
        != _array_sha256(loaded["assignment_world_one"])
        or result.get("prediction_world_zero_sha256")
        != _array_sha256(loaded["prediction_world_zero"])
        or result.get("prediction_world_one_sha256")
        != _array_sha256(loaded["prediction_world_one"])
        or result.get("target_canonical_order_sha256")
        != _array_sha256(loaded["target_canonical_order"])
        or result.get("reference_canonical_order_sha256")
        != _array_sha256(loaded["reference_canonical_order"])
    ):
        raise IntegrityError("unit result hashes do not authenticate its arrays")
    disagreement_count = int(np.count_nonzero(
        loaded["prediction_world_zero"] != loaded["prediction_world_one"]
    ))
    disagreement = disagreement_count / witness_rows
    result_numeric_fields = (
        "baseline_cost",
        "primal_cost",
        "allowed_cost",
        "maximum_disagreement_primal_lower_bound",
        "maximum_disagreement_dual_upper_bound",
        "selected_world_disagreement",
        "selected_two_world_minimax_recovery",
        "primal_dual_gap",
        "primal_dual_gap_percentage_points",
        "gap_target",
        "gap_target_percentage_points",
    )
    if any(
        not isinstance(result.get(field), (int, float))
        or isinstance(result.get(field), bool)
        or not np.isfinite(float(result[field]))
        for field in result_numeric_fields
    ):
        raise IntegrityError("unit optimization bounds are not finite JSON numbers")
    primal = float(result["maximum_disagreement_primal_lower_bound"])
    dual = float(result["maximum_disagreement_dual_upper_bound"])
    gap = float(result["primal_dual_gap"])
    gap_target = float(config["certificate_protocol"]["maximum_primal_dual_gap"])
    if (
        not 0.0 <= primal <= dual <= 1.0
        or float(result["baseline_cost"]) < 0.0
        or float(result["primal_cost"]) < 0.0
        or float(result["allowed_cost"]) < 0.0
        or float(result["primal_cost"]) > float(result["allowed_cost"]) + 1e-10
        or abs(primal - disagreement) > 1e-12
        or abs(float(result["selected_world_disagreement"]) - disagreement)
        > 1e-12
        or abs(
            float(result["selected_two_world_minimax_recovery"])
            - (1.0 - disagreement / 2.0)
        )
        > 1e-12
        or abs(gap - (dual - primal)) > 1e-12
        or float(result["primal_dual_gap_percentage_points"]) != 100.0 * gap
        or float(result["gap_target"]) != gap_target
        or float(result["gap_target_percentage_points"]) != 100.0 * gap_target
        or result.get("maximum_disagreement_identification_interval")
        != [primal, dual]
        or result.get("strongest_two_world_recovery_ceiling_interval")
        != [1.0 - dual / 2.0, 1.0 - primal / 2.0]
        or result.get("meets_gap_target") is not (gap <= gap_target)
        or diagnostics.get("information_scope") != INFORMATION_SCOPE
        or diagnostics.get("external_ground_truth_labels_used_for_optimization")
        is not False
        or diagnostics.get("frozen_readout_predictions_used_for_optimization")
        is not True
        or diagnostics.get("sample_identifiers_used_for_cost_or_reward") is not False
        or diagnostics.get("sample_identifiers_used_for_indexed_atom_administration")
        is not True
        or diagnostics.get("pairing_used") is not False
        or not _exact_int(diagnostics.get("witness_rows"), witness_rows)
        or not _exact_int(diagnostics.get("feature_dimension"))
        or diagnostics.get("feature_dimension") <= 0
        or diagnostics.get("indexed_atom_policy")
        != (
            "preserve_all_rows;_authenticated_unique_sample_ids_break_only_"
            "exact_feature_ties"
        )
        or not _valid_sha256(diagnostics.get("target_witness_indexed_atom_sha256"))
        or not _valid_sha256(
            diagnostics.get("reference_witness_indexed_atom_sha256")
        )
        or not _valid_sha256(diagnostics.get("cost_matrix_sha256"))
        or not _valid_sha256(diagnostics.get("reference_prediction_sha256"))
        or not isinstance(diagnostics.get("budget_independent_problem_reused"), bool)
    ):
        raise IntegrityError("unit optimization bounds do not close")
    exact = _mapping(
        manifest.get("exact_small_instance_validation"), name="exact validation"
    )
    exact_expected_fields = {
        "status",
        "solver",
        "rows",
        "binary_variables",
        "relative_mip_gap_requested",
        "reported_mip_gap",
        "maximum_disagreement_exact",
        "assignment_cost",
        "allowed_cost",
        "assignment_sha256",
        "solver_message",
        "selection",
        "target_parent_indexed_atom_sha256",
        "reference_parent_indexed_atom_sha256",
        "target_parent_indexed_atoms",
        "reference_parent_indexed_atoms",
        "certificate_primal_lower_bound",
        "certificate_dual_upper_bound",
        "certificate_contains_exact_optimum",
    }
    atom_expected_fields = {
        "indexed_atom_rows",
        "exact_duplicate_feature_groups",
        "exact_duplicate_feature_rows_beyond_first",
        "sample_identifiers_supplied",
        "sample_identifiers_unique",
        "sample_identifiers_used_only_for_exact_duplicate_feature_ties",
        "sample_identifier_bytes_included_in_indexed_atom_integrity_digest",
        "sample_identifier_semantic_parsing",
        "original_row_position_used_as_tie_key",
        "cost_and_reward_depend_only_on_features_and_frozen_readout",
        "joint_row_and_id_permutation_invariant",
    }
    exact_rows_value = exact.get("rows")
    binary_variables = exact.get("binary_variables")
    exact_value = exact.get("maximum_disagreement_exact")
    assignment_cost = exact.get("assignment_cost")
    allowed_cost = exact.get("allowed_cost")
    exact_primal = exact.get("certificate_primal_lower_bound")
    exact_dual = exact.get("certificate_dual_upper_bound")
    numeric_values = (
        exact.get("relative_mip_gap_requested"),
        exact.get("reported_mip_gap"),
        exact_value,
        assignment_cost,
        allowed_cost,
        exact_primal,
        exact_dual,
    )
    atom_diagnostics = []
    for endpoint in (
        "target_parent_indexed_atoms",
        "reference_parent_indexed_atoms",
    ):
        diagnostic = _mapping(exact.get(endpoint), name=endpoint)
        if set(diagnostic) != atom_expected_fields:
            raise IntegrityError("unit exact-validation atom schema changed")
        atom_diagnostics.append(diagnostic)
    for endpoint in (
        "target_witness_indexed_atoms",
        "reference_witness_indexed_atoms",
    ):
        diagnostic = _mapping(
            diagnostics.get(endpoint), name=f"certificate diagnostic {endpoint}"
        )
        if set(diagnostic) != atom_expected_fields:
            raise IntegrityError("unit certificate indexed-atom schema changed")
    if (
        diagnostics.get("target_witness_indexed_atom_sha256")
        != exact.get("target_parent_indexed_atom_sha256")
        or diagnostics.get("reference_witness_indexed_atom_sha256")
        != exact.get("reference_parent_indexed_atom_sha256")
        or diagnostics.get("target_witness_indexed_atoms")
        != exact.get("target_parent_indexed_atoms")
        or diagnostics.get("reference_witness_indexed_atoms")
        != exact.get("reference_parent_indexed_atoms")
    ):
        raise IntegrityError("unit certificate/exact indexed-atom records disagree")
    atom_counts_valid = all(
        isinstance(diagnostic[field], int)
        and not isinstance(diagnostic[field], bool)
        and 0 <= diagnostic[field] <= witness_rows
        for diagnostic in atom_diagnostics
        for field in (
            "indexed_atom_rows",
            "exact_duplicate_feature_groups",
            "exact_duplicate_feature_rows_beyond_first",
        )
    )
    atom_flags_valid = all(
        diagnostic["indexed_atom_rows"] == witness_rows
        and diagnostic["sample_identifiers_supplied"] is True
        and diagnostic["sample_identifiers_unique"] is True
        and diagnostic[
            "sample_identifiers_used_only_for_exact_duplicate_feature_ties"
        ]
        is True
        and diagnostic[
            "sample_identifier_bytes_included_in_indexed_atom_integrity_digest"
        ]
        is True
        and diagnostic["sample_identifier_semantic_parsing"] is False
        and diagnostic["original_row_position_used_as_tie_key"] is False
        and diagnostic[
            "cost_and_reward_depend_only_on_features_and_frozen_readout"
        ]
        is True
        and diagnostic["joint_row_and_id_permutation_invariant"] is True
        and diagnostic["exact_duplicate_feature_rows_beyond_first"]
        >= diagnostic["exact_duplicate_feature_groups"]
        for diagnostic in atom_diagnostics
    )
    exact_numbers_valid = all(
        isinstance(number, (int, float))
        and not isinstance(number, bool)
        and np.isfinite(float(number))
        for number in numeric_values
    )
    if (
        set(exact) != exact_expected_fields
        or exact.get("status") != "PROVEN_OPTIMAL"
        or exact.get("solver") != "scipy_optimize_milp_highs"
        or not isinstance(exact_rows_value, int)
        or isinstance(exact_rows_value, bool)
        or exact_rows_value != exact_rows
        or not isinstance(binary_variables, int)
        or isinstance(binary_variables, bool)
        or binary_variables != exact_rows * exact_rows
        or not exact_numbers_valid
        or float(exact["relative_mip_gap_requested"]) != 0.0
        or float(exact["reported_mip_gap"]) != 0.0
        or not 0.0 <= float(exact_value) <= 1.0
        or not 0.0 <= float(exact_primal) <= float(exact_value)
        or not float(exact_value) <= float(exact_dual) <= 1.0
        or float(assignment_cost) < 0.0
        or float(allowed_cost) < 0.0
        or float(assignment_cost) > float(allowed_cost) + 1e-10
        or exact.get("certificate_contains_exact_optimum") is not True
        or exact.get("assignment_sha256")
        != _array_sha256(loaded["exact_subset_assignment"])
        or not isinstance(exact.get("solver_message"), str)
        or not exact.get("solver_message")
        or exact.get("selection")
        != (
            "first_indexed_atoms_after_independent_feature_primary_and_"
            "sample_id_duplicate_tie_canonicalization_of_each_endpoint"
        )
        or not isinstance(exact.get("target_parent_indexed_atom_sha256"), str)
        or len(exact["target_parent_indexed_atom_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in exact["target_parent_indexed_atom_sha256"]
        )
        or not isinstance(exact.get("reference_parent_indexed_atom_sha256"), str)
        or len(exact["reference_parent_indexed_atom_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in exact["reference_parent_indexed_atom_sha256"]
        )
        or not atom_counts_valid
        or not atom_flags_valid
    ):
        raise IntegrityError(
            "unit exact validation is not a complete zero-gap diagnostic proof"
        )
    posthoc = _mapping(manifest.get("posthoc_inference"), name="posthoc inference")
    _require_exact_keys(
        posthoc, POSTHOC_INFERENCE_FIELDS, name="unit posthoc inference"
    )
    statistics = posthoc.get("per_wnid_sufficient_statistics")
    declared_wnids = tuple(map(str, store_record["wnids"]))
    declared_cluster_count = posthoc.get("declared_cluster_count")
    observed_cluster_count = posthoc.get("observed_witness_cluster_count")
    if (
        posthoc.get("external_ground_truth_labels_used_for_optimization")
        is not False
        or posthoc.get("frozen_readout_predictions_used_for_optimization")
        is not True
        or posthoc.get("target_wnids_preflight_integrity_validated")
        is not True
        or posthoc.get("target_wnids_not_used_as_certificate_optimization_inputs")
        is not True
        or posthoc.get(
            "target_wnids_used_for_posthoc_inference_after_assignment_fixed"
        )
        is not True
        or posthoc.get("target_wnids_used_for_posthoc_inference") is not True
        or posthoc.get("sample_identifiers_used_for_cost_or_reward") is not False
        or posthoc.get("sample_identifiers_used_for_indexed_atom_administration")
        is not True
        or posthoc.get("sample_identifiers_parsed_for_semantics") is not False
        or posthoc.get("exact_duplicate_atom_tie_note")
        != (
            "opaque_authenticated_sample_ids_break_only_exact_feature_ties;_"
            "the_tie_choice_can_change_posthoc_wnid_attribution_for_boundary_"
            "atoms_but_not_feature_cost_or_prediction_reward"
        )
        or posthoc.get("cluster") != "imagenet_wnid"
        or posthoc.get("class_intersection_wnids_sha256")
        != store_record["wnids_sha256"]
        or posthoc.get("target_wnids_role") != store_record["target_wnids_role"]
        or not isinstance(declared_cluster_count, int)
        or isinstance(declared_cluster_count, bool)
        or declared_cluster_count != len(declared_wnids)
        or not isinstance(observed_cluster_count, int)
        or isinstance(observed_cluster_count, bool)
        or observed_cluster_count < 0
        or not isinstance(statistics, list)
    ):
        raise IntegrityError("unit posthoc inference contract changed")
    rows = np.zeros(len(declared_wnids), dtype=np.int64)
    disagreements = np.zeros(len(declared_wnids), dtype=np.int64)
    index = {wnid: position for position, wnid in enumerate(declared_wnids)}
    seen: set[str] = set()
    for statistic in statistics:
        row = _mapping(statistic, name="per-WNID statistic")
        _require_exact_keys(
            row, POSTHOC_STATISTIC_FIELDS, name="per-WNID statistic"
        )
        wnid = str(row.get("wnid"))
        count = row.get("rows")
        different = row.get("prediction_disagreement_count")
        if (
            wnid not in index
            or wnid in seen
            or not isinstance(count, int)
            or isinstance(count, bool)
            or not isinstance(different, int)
            or isinstance(different, bool)
            or count < 1
            or different < 0
            or different > count
        ):
            raise IntegrityError("unit has an invalid per-WNID statistic")
        seen.add(wnid)
        rows[index[wnid]] = count
        disagreements[index[wnid]] = different
    if (
        len(seen) != observed_cluster_count
        or int(rows.sum()) != witness_rows
        or int(disagreements.sum()) != disagreement_count
        or [str(row["wnid"]) for row in statistics]
        != [
            wnid
            for wnid in declared_wnids
            if rows[index[wnid]] > 0
        ]
    ):
        raise IntegrityError("unit per-WNID statistics do not reproduce its result")
    split_diagnostics = _mapping(
        manifest.get("split_diagnostics"), name="unit split diagnostics"
    )
    _require_exact_keys(
        split_diagnostics,
        SPLIT_DIAGNOSTIC_FIELDS,
        name="unit split diagnostics",
    )
    if (
        not all(
            _valid_sha256(split_diagnostics.get(field))
            for field in (
                "reference_split_parent_content_sha256",
                "target_split_parent_content_sha256",
                "reference_selected_original_rows_sha256",
                "target_selected_original_rows_sha256",
            )
        )
        or split_diagnostics.get("grouped_compute_policy")
        != (
            "split_and_bures_once_per_dataset_encoder_seed;_cost_once_per_"
            "geometry;_three_budgets_reuse_the_prepared_dense_problem"
        )
        or not _exact_int(
            split_diagnostics.get("grouped_budget_count"),
            len(config["certificate_protocol"]["cost_budgets_relative"]),
        )
        or not _exact_int(
            split_diagnostics.get("grouped_geometry_count"),
            len(config["cost_geometries"]),
        )
    ):
        raise IntegrityError("unit split diagnostics contract changed")
    return identity, {
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "rows": rows,
        "disagreements": disagreements,
        "primal_dual_gap": gap,
        "meets_gap_target": bool(result["meets_gap_target"]),
        "exact_proven": True,
        "compute_event": compute_event,
        "compute_event_owner": compute_event_owner,
        "unit_array_serialization_nanoseconds": int(
            compute["temporary_directory_and_array_serialization_nanoseconds"]
        ),
        "unit_invocation_wall_nanoseconds_before_manifest_write": int(
            compute["invocation_wall_nanoseconds_before_manifest_write"]
        ),
    }


def _validate_panel_shard_completion(
    completion: Path,
    *,
    root: Path,
    identities: tuple[tuple[str, str, str, int, int], ...],
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    shard_count: int,
    shard_index: int,
) -> None:
    """Reauthenticate a completed shard and every persisted unit payload."""

    payload = _mapping(read_json(completion), name="panel shard completion")
    verify_manifest_checksum(payload)
    expected_ids = [_unit_id(value) for value in identities]
    units = payload.get("units")
    required_keys = {
        "schema_version",
        "status",
        "lock_file_sha256",
        "lock_manifest_sha256",
        "shard_count",
        "shard_index",
        "expected_units",
        "completed_units",
        "unit_ids",
        "units",
        "resume_policy",
        "manifest_sha256",
    }
    if (
        set(payload) != required_keys
        or payload.get("schema_version") != PANEL_SHARD_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("lock_file_sha256") != lock_file_sha256
        or payload.get("lock_manifest_sha256") != lock["manifest_sha256"]
        or not _exact_int(payload.get("shard_count"), shard_count)
        or not _exact_int(payload.get("shard_index"), shard_index)
        or not _exact_int(payload.get("expected_units"), len(identities))
        or not _exact_int(payload.get("completed_units"), len(identities))
        or payload.get("unit_ids") != expected_ids
        or not isinstance(units, list)
        or len(units) != len(identities)
        or payload.get("resume_policy")
        != (
            "hash_and_contract_validate_complete_units_without_reconstruction;_"
            "use_verify-panel-for_full_reconstruction"
        )
    ):
        raise IntegrityError("existing panel-shard completion marker is invalid")

    for position, identity in enumerate(identities):
        identifier = expected_ids[position]
        unit_record = _mapping(units[position], name="completed panel unit")
        manifest_path = root / identifier / "UNIT_MANIFEST.json"
        if (
            set(unit_record)
            != {"unit_id", "manifest_file_sha256", "resumed"}
            or unit_record.get("unit_id") != identifier
            or not isinstance(unit_record.get("resumed"), bool)
            or not manifest_path.is_file()
        ):
            raise IntegrityError(
                f"completed panel unit is missing or invalid: {identifier}"
            )
        observed, validated = _validate_aggregate_unit(
            manifest_path,
            lock=lock,
            lock_file_sha256=lock_file_sha256,
        )
        if (
            observed != identity
            or unit_record.get("manifest_file_sha256")
            != validated["manifest_file_sha256"]
        ):
            raise IntegrityError(
                f"completed panel unit no longer matches its marker: {identifier}"
            )


def run_panel_shard(
    lock_path: str | Path,
    store_map_path: str | Path,
    output_root: str | Path,
    *,
    shard_count: int,
    shard_index: int,
    resume: bool,
) -> Path:
    """Run one deterministic shard of the complete 162-unit locked panel."""

    lock_source = Path(lock_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    lock = _load_lock(lock_source)
    stores = _locked_store_map(lock, store_map_path)
    identities = _sharded_identities(
        lock["config"], shard_count=shard_count, shard_index=shard_index
    )
    root.mkdir(parents=True, exist_ok=True)
    completion = root / f"SHARD_{int(shard_index):03d}_DONE.json"
    if completion.exists():
        _validate_panel_shard_completion(
            completion,
            root=root,
            identities=identities,
            lock=lock,
            lock_file_sha256=sha256_file(lock_source),
            shard_count=shard_count,
            shard_index=shard_index,
        )
        return completion
    completed: list[dict[str, Any]] = []
    for identity in identities:
        dataset, encoder, geometry, seed, budget = identity
        identifier = _unit_id(identity)
        destination = root / identifier
        manifest_path = destination / "UNIT_MANIFEST.json"
        if destination.exists():
            if not resume or not manifest_path.is_file():
                raise FileExistsError(
                    f"refusing unverified existing unit directory: {destination}"
                )
            observed, record = _validate_aggregate_unit(
                manifest_path,
                lock=lock,
                lock_file_sha256=sha256_file(lock_source),
            )
            if observed != identity:
                raise IntegrityError("resumed unit identity changed")
            completed.append({
                "unit_id": identifier,
                "manifest_file_sha256": record["manifest_file_sha256"],
                "resumed": True,
            })
            continue
        output = run_unit(
            lock_source,
            stores[_store_key(dataset, encoder)]["manifest"],
            destination,
            dataset=dataset,
            encoder=encoder,
            geometry_name=geometry,
            seed=seed,
            budget=budget / 10_000.0,
        )
        completed.append({
            "unit_id": identifier,
            "manifest_file_sha256": sha256_file(output),
            "resumed": False,
        })
    write_json_atomic(completion, seal_manifest({
        "schema_version": PANEL_SHARD_SCHEMA,
        "status": "PASS",
        "lock_file_sha256": sha256_file(lock_source),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "expected_units": len(identities),
        "completed_units": len(completed),
        "unit_ids": [_unit_id(value) for value in identities],
        "units": completed,
        "resume_policy": (
            "hash_and_contract_validate_complete_units_without_reconstruction;_"
            "use_verify-panel-for_full_reconstruction"
        ),
    }))
    return completion


def verify_panel_shard(
    lock_path: str | Path,
    store_map_path: str | Path,
    units_root: str | Path,
    output_path: str | Path,
    *,
    shard_count: int,
    shard_index: int,
) -> Path:
    """Fully reconstruct one deterministic panel shard from locked stores."""

    lock_source = Path(lock_path).expanduser().resolve()
    root = Path(units_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification output: {output}")
    lock = _load_lock(lock_source)
    stores = _locked_store_map(lock, store_map_path)
    identities = _sharded_identities(
        lock["config"], shard_count=shard_count, shard_index=shard_index
    )
    reports: list[dict[str, Any]] = []
    for identity in identities:
        dataset, encoder, _, _, _ = identity
        identifier = _unit_id(identity)
        manifest = root / identifier / "UNIT_MANIFEST.json"
        if not manifest.is_file():
            raise IntegrityError(f"verification unit is missing: {identifier}")
        report = verify_unit(
            lock_source,
            stores[_store_key(dataset, encoder)]["manifest"],
            manifest,
        )
        unit_payload = _mapping(read_json(manifest), name="verified unit manifest")
        verify_manifest_checksum(unit_payload)
        reports.append({
            "unit_id": identifier,
            "unit_manifest_file_sha256": sha256_file(manifest),
            "unit_manifest_sha256": unit_payload["manifest_sha256"],
            **report,
        })
    expected_group_ids = {
        _group_id(dataset, encoder, seed)
        for dataset, encoder, _, seed, _ in identities
    }
    compute_accounting = _verification_compute_accounting(
        reports, expected_group_ids=expected_group_ids
    )
    write_json_atomic(output, seal_manifest({
        "schema_version": VERIFICATION_SHARD_SCHEMA,
        "status": "PASS",
        "lock_file_sha256": sha256_file(lock_source),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "expected_units": len(identities),
        "verified_units": len(reports),
        "full_endpoint_feature_reconstruction": True,
        "compute_accounting": compute_accounting,
        "reports": reports,
    }))
    return output


def _clustered_interval(
    rows_by_seed: np.ndarray,
    disagreements_by_seed: np.ndarray,
    *,
    bootstrap_weights: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray(rows_by_seed, dtype=np.float64)
    disagreements = np.asarray(disagreements_by_seed, dtype=np.float64)
    weights = np.asarray(bootstrap_weights, dtype=np.float64)
    if (
        rows.ndim != 2
        or disagreements.shape != rows.shape
        or weights.ndim != 2
        or weights.shape[1] != rows.shape[1]
        or np.any(rows < 0.0)
        or np.any(disagreements < 0.0)
        or np.any(disagreements > rows)
    ):
        raise IntegrityError("WNID-clustered inference inputs are malformed")
    # The overlapping split repeats are collapsed inside each WNID before
    # WNIDs are resampled as the independent benchmark-panel units.
    pooled_rows = rows.sum(axis=0)
    pooled_disagreements = disagreements.sum(axis=0)
    denominators = weights @ pooled_rows
    if np.any(denominators <= 0.0):
        raise IntegrityError("WNID bootstrap produced an empty resample")
    samples = (weights @ pooled_disagreements) / denominators
    lower, upper = np.quantile(samples, [0.025, 0.975])
    total_rows = float(pooled_rows.sum())
    if total_rows <= 0.0:
        raise IntegrityError("WNID-clustered inference has no witness rows")
    return {
        "point_estimate": float(pooled_disagreements.sum() / total_rows),
        "lower": float(lower),
        "upper": float(upper),
        "display_label": (
            "descriptive conditional WNID-cluster resampling interval, "
            "assignments fixed/no optimizer refit"
        ),
        "nominal_resampling_interval_level": 0.95,
        "interval": "descriptive_conditional_percentile",
        "replicates": int(weights.shape[0]),
        "dropped_replicates": 0,
        "cluster": "imagenet_wnid",
        "declared_clusters": int(rows.shape[1]),
        "observed_clusters_after_pooling": int(np.count_nonzero(pooled_rows)),
        "split_seed_policy": (
            "sum_sufficient_statistics_within_wnid_before_cluster_resampling"
        ),
        "estimand": (
            "selected_in_sample_witness_disagreement_conditional_on_each_"
            "serialized_optimized_assignment"
        ),
        "optimizer_refit_within_bootstrap": False,
        "interpretation": (
            "descriptive_conditional_cluster_resampling_interval_not_a_"
            "sampling_valid_confidence_interval_for_the_population_optimized_"
            "functional"
        ),
        "gate_kind": (
            "prespecified_robustness_screen_not_population_or_confirmatory_"
            "inference"
        ),
    }


def _gate_summary(
    conditions: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    primary_geometry = str(config["gate_decisions"]["breadth_primary_geometry"])
    primary_budget = _budget_token(
        config["certificate_protocol"]["primary_cost_budget_relative"]
    )
    one_budget = _budget_token(0.01)
    decisions: dict[str, Any] = {}
    for token, threshold, name in (
        (one_budget, 0.25, "breadth_1_percent"),
        (primary_budget, 0.50, "breadth_5_percent"),
    ):
        passing_by_encoder: dict[str, set[str]] = {}
        for encoder in config["encoders"]:
            selected = [
                row
                for row in conditions
                if row["encoder"] == encoder
                and row["cost_geometry"] == primary_geometry
                and _budget_token(row["cost_budget_relative"]) == token
            ]
            passing = sorted(
                row["dataset"] for row in selected if row["disagreement"]["lower"] > threshold
            )
            passing_by_encoder[str(encoder)] = set(passing)
            decisions[f"{name}__{encoder}"] = {
                "status": "PASS" if len(passing) >= 2 else "FAIL",
                "threshold_strictly_greater_than": threshold,
                "passing_datasets": passing,
                "required_datasets": 2,
            }
        common = sorted(set.intersection(*passing_by_encoder.values()))
        decisions[f"{name}__common_encoders"] = {
            "status": "PASS" if len(common) >= 2 else "FAIL",
            "threshold_strictly_greater_than": threshold,
            "passing_datasets_common_to_all_encoders": common,
            "required_common_datasets": 2,
            "encoder_passing_datasets": {
                encoder: sorted(datasets)
                for encoder, datasets in passing_by_encoder.items()
            },
        }
    geometry_rows: list[dict[str, Any]] = []
    for geometry in config["cost_geometries"]:
        encoder_rows = []
        passing_by_encoder = {}
        for encoder in config["encoders"]:
            selected = [
                row
                for row in conditions
                if row["encoder"] == encoder
                and row["cost_geometry"] == geometry
                and _budget_token(row["cost_budget_relative"]) == primary_budget
            ]
            passing = sorted(
                row["dataset"]
                for row in selected
                if row["disagreement"]["lower"] > 0.50
            )
            passing_by_encoder[str(encoder)] = set(passing)
            encoder_rows.append({
                "encoder": encoder,
                "passing_datasets": passing,
                "status": "PASS" if len(passing) >= 2 else "FAIL",
            })
        common = sorted(set.intersection(*passing_by_encoder.values()))
        geometry_rows.append({
            "cost_geometry": geometry,
            "encoder_rows": encoder_rows,
            "passing_datasets_common_to_all_encoders": common,
            "required_common_datasets": 2,
            "status": "PASS" if len(common) >= 2 else "FAIL",
        })
    passing_geometries = sum(row["status"] == "PASS" for row in geometry_rows)
    decisions["geometry_gate"] = {
        "status": "PASS" if passing_geometries >= 2 else "FAIL",
        "required_geometries": 2,
        "passing_geometries": passing_geometries,
        "rows": geometry_rows,
    }
    return decisions


def _scientific_gate_outcome(
    gates: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Separate aggregate-artifact validity from its scientific gate outcome."""

    required = tuple(map(
        str,
        _mapping(config.get("gate_decisions"), name="gate_decisions").get(
            "required_scientific_gate_keys", ()
        ),
    ))
    if required != REQUIRED_SCIENTIFIC_GATES:
        raise IntegrityError("required scientific gate inventory changed")
    failed: list[str] = []
    for key in required:
        record = _mapping(gates.get(key), name=f"scientific gate {key}")
        status = record.get("status")
        if status not in {"PASS", "FAIL"}:
            raise IntegrityError(f"scientific gate has invalid status: {key}")
        if status == "FAIL":
            failed.append(key)
    return {
        "scientific_gate_status": "PASS" if not failed else "FAIL",
        "all_required_gates_pass": not failed,
        "required_scientific_gate_keys": list(required),
        "failed_required_scientific_gate_keys": failed,
    }


def _validated_verification_inventory(
    verification_root: str | Path,
    *,
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    expected: Mapping[
        tuple[str, str, str, int, int], Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    root = Path(verification_root).expanduser().resolve()
    shard_count = int(lock["config"]["execution"]["verification_shard_count"])
    paths = sorted(root.glob("*.json"))
    if len(paths) != shard_count:
        raise IntegrityError(
            f"expected {shard_count} verification manifests, found {len(paths)}"
        )
    shard_indices: set[int] = set()
    verified_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    expected_ids = {
        _unit_id(identity): (identity, record)
        for identity, record in expected.items()
    }
    for path in paths:
        payload = _mapping(read_json(path), name="verification shard")
        verify_manifest_checksum(payload)
        _require_exact_keys(
            payload,
            VERIFICATION_SHARD_FIELDS,
            name="verification shard",
        )
        index_value = payload.get("shard_index")
        index = index_value if _exact_int(index_value) else -1
        reports = payload.get("reports")
        if (
            payload.get("schema_version") != VERIFICATION_SHARD_SCHEMA
            or payload.get("status") != "PASS"
            or payload.get("lock_file_sha256") != lock_file_sha256
            or payload.get("lock_manifest_sha256") != lock["manifest_sha256"]
            or not _exact_int(payload.get("shard_count"), shard_count)
            or index < 0
            or index >= shard_count
            or index in shard_indices
            or payload.get("full_endpoint_feature_reconstruction") is not True
            or not isinstance(reports, list)
            or not _exact_int(payload.get("expected_units"), len(reports))
            or not _exact_int(payload.get("verified_units"), len(reports))
        ):
            raise IntegrityError(f"verification shard is invalid: {path}")
        shard_indices.add(index)
        validated_reports: list[Mapping[str, Any]] = []
        shard_identities: list[tuple[str, str, str, int, int]] = []
        for report_value in reports:
            report = _mapping(report_value, name="verification report")
            _require_exact_keys(
                report,
                VERIFICATION_REPORT_FIELDS,
                name="verification report",
            )
            identifier = str(report.get("unit_id", ""))
            expected_pair = expected_ids.get(identifier)
            identity = expected_pair[0] if expected_pair is not None else None
            expected_record = expected_pair[1] if expected_pair is not None else None
            if (
                expected_record is None
                or identifier in verified_ids
                or report.get("status") != "PASS"
                or report.get("unit_manifest_file_sha256")
                != expected_record["manifest_file_sha256"]
                or report.get("unit_manifest_sha256")
                != expected_record["manifest_sha256"]
                or report.get("assignments_recomputed") is not True
                or report.get("costs_recomputed_from_endpoint_features") is not True
                or report.get("rewards_recomputed_from_frozen_readout") is not True
                or report.get("bijections_revalidated") is not True
                or report.get("duplicate_row_admissibility_revalidated") is not True
                or report.get("exact_small_instance_milp_revalidated") is not True
                or report.get("posthoc_wnid_statistics_revalidated") is not True
                or report.get("gap_target_met")
                is not expected_record["meets_gap_target"]
                or not _finite_number(
                    report.get("primal_dual_gap_percentage_points"),
                    100.0 * float(expected_record["primal_dual_gap"]),
                )
            ):
                raise IntegrityError(
                    f"verification report is invalid or duplicate: {identifier}"
                )
            _validated_verification_timing(
                report,
                lock=lock,
                identity=identity,
            )
            verified_ids.add(identifier)
            validated_reports.append(report)
            shard_identities.append(identity)
        expected_group_ids = {
            _group_id(dataset, encoder, seed)
            for dataset, encoder, _, seed, _ in shard_identities
        }
        observed_accounting = _verification_compute_accounting(
            validated_reports, expected_group_ids=expected_group_ids
        )
        if payload.get("compute_accounting") != observed_accounting:
            raise IntegrityError("verification shard compute accounting changed")
        records.append({
            "shard_index": index,
            "manifest_file_sha256": sha256_file(path),
            "manifest_sha256": payload["manifest_sha256"],
            "verified_units": len(reports),
            "compute_accounting": observed_accounting,
        })
    if shard_indices != set(range(shard_count)) or verified_ids != set(expected_ids):
        raise IntegrityError("full-reconstruction verification coverage is incomplete")
    return sorted(records, key=lambda row: int(row["shard_index"]))


def _combined_verification_compute_accounting(
    shard_records: list[Mapping[str, Any]], *, expected_group_ids: set[str]
) -> dict[str, Any]:
    """Combine already rederived, disjoint verification-shard timing totals."""

    accounting_rows = [
        _mapping(row.get("compute_accounting"), name="shard compute accounting")
        for row in shard_records
    ]
    event_inventory = [
        dict(_mapping(event, name="verification timing event inventory row"))
        for accounting in accounting_rows
        for event in accounting["event_inventory"]
    ]
    event_ids = [str(row["event_id"]) for row in event_inventory]
    group_ids = [str(row["group_id"]) for row in event_inventory]
    if (
        len(set(event_ids)) != len(event_ids)
        or len(set(group_ids)) != len(group_ids)
        or set(group_ids) != expected_group_ids
    ):
        raise IntegrityError("verification compute events overlap or omit a group")
    totals_fields = tuple(
        accounting_rows[0]["totals_nanoseconds"]
    ) if accounting_rows else ()
    totals_ns = {
        field: sum(int(row["totals_nanoseconds"][field]) for row in accounting_rows)
        for field in totals_fields
    }
    totals_by_geometry: dict[str, dict[str, Any]] = {}
    for accounting in accounting_rows:
        for geometry, source_row in accounting[
            "totals_by_geometry_nanoseconds"
        ].items():
            destination = totals_by_geometry.setdefault(geometry, {
                "geometry_derivation_nanoseconds": 0,
                "dense_problem_preparation_nanoseconds": 0,
                "budgets": {},
            })
            for field in (
                "geometry_derivation_nanoseconds",
                "dense_problem_preparation_nanoseconds",
            ):
                destination[field] += int(source_row[field])
            for budget, budget_row in source_row["budgets"].items():
                destination_budget = destination["budgets"].setdefault(
                    budget,
                    {
                        "certificate_nanoseconds": 0,
                        "exact_validation_nanoseconds": 0,
                        "posthoc_inference_nanoseconds": 0,
                    },
                )
                for field, value in budget_row.items():
                    destination_budget[field] += int(value)
    non_group_ns = sum(
        int(row["unit_non_group_validation_and_python_overhead_nanoseconds"])
        for row in accounting_rows
    )
    group_wall_ns = int(totals_ns.get("group_wall_nanoseconds", 0))
    return {
        "schema_version": COMPUTE_TIMING_SCHEMA,
        "purpose": "full_reconstruction_verification",
        "accounting_scope": (
            "sum_of_disjoint_authenticated_verification_shard_accounting"
        ),
        "expected_distinct_groups": len(expected_group_ids),
        "observed_distinct_groups": len(set(group_ids)),
        "successful_group_compute_events": len(event_ids),
        "successful_retry_group_compute_events": sum(
            int(row["successful_retry_group_compute_events"])
            for row in accounting_rows
        ),
        "events_deduplicated_across_unit_records": sum(
            int(row["events_deduplicated_across_unit_records"])
            for row in accounting_rows
        ),
        "event_inventory": sorted(
            event_inventory, key=lambda row: str(row["event_id"])
        ),
        "totals_nanoseconds": totals_ns,
        "totals_seconds": {
            field.removesuffix("_nanoseconds") + "_seconds": value / 1e9
            for field, value in totals_ns.items()
        },
        "totals_by_geometry_nanoseconds": totals_by_geometry,
        "totals_by_geometry_seconds": {
            geometry: {
                "geometry_derivation_seconds": row[
                    "geometry_derivation_nanoseconds"
                ] / 1e9,
                "dense_problem_preparation_seconds": row[
                    "dense_problem_preparation_nanoseconds"
                ] / 1e9,
                "budgets": {
                    budget: {
                        field.removesuffix("_nanoseconds") + "_seconds": (
                            value / 1e9
                        )
                        for field, value in budget_row.items()
                    }
                    for budget, budget_row in row["budgets"].items()
                },
            }
            for geometry, row in totals_by_geometry.items()
        },
        "unit_non_group_validation_and_python_overhead_nanoseconds": int(
            non_group_ns
        ),
        "unit_non_group_validation_and_python_overhead_seconds": (
            non_group_ns / 1e9
        ),
        "observed_verification_wall_before_manifest_publication_nanoseconds": int(
            group_wall_ns + non_group_ns
        ),
        "observed_verification_wall_before_manifest_publication_seconds": (
            group_wall_ns + non_group_ns
        ) / 1e9,
        "peak_rss_bytes": None,
        "peak_rss_status": PEAK_RSS_STATUS,
        "not_included": [
            "events_interrupted_or_failed_before_any_sealed_verification_shard",
            "verification_manifest_seal_write_fsync_and_atomic_rename_duration",
        ],
    }


def aggregate_units(
    lock_path: str | Path,
    units_root: str | Path,
    verification_root: str | Path,
    output_path: str | Path,
) -> Path:
    lock_source = Path(lock_path).expanduser().resolve()
    root = Path(units_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {output}")
    lock = _load_lock(lock_source)
    config = lock["config"]
    expected = set(_expected_identities(config))
    manifest_paths = sorted(root.rglob("UNIT_MANIFEST.json"))
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    lock_file_sha256 = sha256_file(lock_source)
    for manifest_path in manifest_paths:
        identity, record = _validate_aggregate_unit(
            manifest_path, lock=lock, lock_file_sha256=lock_file_sha256
        )
        if identity in records:
            raise IntegrityError(f"duplicate optimization unit: {_unit_id(identity)}")
        records[identity] = record
    if set(records) != expected:
        raise IntegrityError(
            "optimization unit inventory differs from the sealed panel: "
            f"missing={len(expected-set(records))}, extra={len(set(records)-expected)}"
        )
    expected_group_ids = {
        _group_id(dataset, encoder, seed)
        for dataset in config["datasets"]
        for encoder in config["encoders"]
        for seed in config["certificate_protocol"]["split_seeds"]
    }
    panel_compute_accounting = _compute_accounting_summary(
        [
            (
                record["compute_event"],
                bool(record["compute_event_owner"]),
                _unit_id(identity),
            )
            for identity, record in records.items()
        ],
        expected_group_ids=expected_group_ids,
        purpose="scientific_panel_compute",
    )
    unit_non_group_ns = sum(
        int(record["unit_invocation_wall_nanoseconds_before_manifest_write"])
        - (
            int(record["compute_event"]["group_wall_nanoseconds"])
            if record["compute_event_owner"]
            else 0
        )
        for record in records.values()
    )
    unit_array_serialization_ns = sum(
        int(record["unit_array_serialization_nanoseconds"])
        for record in records.values()
    )
    if unit_non_group_ns < unit_array_serialization_ns:
        raise IntegrityError("unit output timing exceeds non-group invocation timing")
    panel_compute_accounting.update({
        "unit_record_count": len(records),
        "unit_non_group_invocation_nanoseconds_before_manifest_write": int(
            unit_non_group_ns
        ),
        "unit_non_group_invocation_seconds_before_manifest_write": (
            unit_non_group_ns / 1e9
        ),
        "temporary_directory_and_array_serialization_nanoseconds": int(
            unit_array_serialization_ns
        ),
        "temporary_directory_and_array_serialization_seconds": (
            unit_array_serialization_ns / 1e9
        ),
        "observed_process_time_before_manifest_publication_nanoseconds": int(
            panel_compute_accounting["totals_nanoseconds"][
                "group_wall_nanoseconds"
            ]
            + unit_non_group_ns
        ),
        "observed_process_time_before_manifest_publication_seconds": (
            panel_compute_accounting["totals_nanoseconds"][
                "group_wall_nanoseconds"
            ]
            + unit_non_group_ns
        ) / 1e9,
        "elapsed_interpretation": (
            "sum_of_nonoverlapping_recorded_process_intervals_not_parallel_"
            "campaign_wall_clock"
        ),
    })
    verification_manifests = _validated_verification_inventory(
        verification_root,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        expected=records,
    )
    verification_compute_accounting = _combined_verification_compute_accounting(
        verification_manifests, expected_group_ids=expected_group_ids
    )
    inference = config["inference"]
    replicates = int(inference["replicates"])
    seed = int(inference["seed"])
    weights_by_dataset: dict[str, np.ndarray] = {}
    for dataset in config["datasets"]:
        wnids = lock["stores"][_store_key(dataset, config["encoders"][0])]["wnids"]
        for encoder in config["encoders"][1:]:
            if lock["stores"][_store_key(dataset, encoder)]["wnids"] != wnids:
                raise IntegrityError("encoder stores disagree on class intersection")
        dataset_seed = int(sha256_json({
            "seed": seed,
            "dataset": dataset,
            "wnids_sha256": sha256_json(wnids),
        })[:16], 16)
        rng = np.random.Generator(np.random.PCG64(dataset_seed))
        weights_by_dataset[dataset] = rng.multinomial(
            len(wnids),
            np.full(len(wnids), 1.0 / len(wnids), dtype=np.float64),
            size=replicates,
        )
    conditions: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for encoder in config["encoders"]:
            for geometry in config["cost_geometries"]:
                for budget in config["certificate_protocol"][
                    "cost_budgets_relative"
                ]:
                    token = _budget_token(budget)
                    selected = [
                        records[_unit_identity(
                            dataset, encoder, geometry, seed_value, budget
                        )]
                        for seed_value in config["certificate_protocol"][
                            "split_seeds"
                        ]
                    ]
                    interval = _clustered_interval(
                        np.stack([row["rows"] for row in selected]),
                        np.stack([row["disagreements"] for row in selected]),
                        bootstrap_weights=weights_by_dataset[dataset],
                    )
                    conditions.append({
                        "dataset": dataset,
                        "encoder": encoder,
                        "cost_geometry": geometry,
                        "cost_budget_relative": token / 10_000.0,
                        "disagreement": interval,
                        "unit_ids": [
                            _unit_id(_unit_identity(
                                dataset, encoder, geometry, seed_value, budget
                            ))
                            for seed_value in config["certificate_protocol"][
                                "split_seeds"
                            ]
                        ],
                    })
    optimization_failures = sorted(
        _unit_id(identity)
        for identity, record in records.items()
        if not record["meets_gap_target"]
    )
    exact_failures = sorted(
        _unit_id(identity)
        for identity, record in records.items()
        if not record["exact_proven"]
    )
    decision_rows = _gate_summary(conditions, config)
    gates = {
        "optimization_all_1_5_10_percent_units": {
            "status": "PASS" if not optimization_failures else "FAIL",
            "maximum_gap": float(
                config["certificate_protocol"]["maximum_primal_dual_gap"]
            ),
            "failed_unit_ids": optimization_failures,
        },
        "exact_64_row_all_units": {
            "status": "PASS" if not exact_failures else "FAIL",
            "failed_unit_ids": exact_failures,
        },
        **decision_rows,
    }
    scientific_outcome = _scientific_gate_outcome(gates, config)
    payload = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "status_interpretation": "artifact_integrity_and_completeness_only",
        "artifact_status": "PASS",
        "scientific_result": True,
        **scientific_outcome,
        "lock_file_sha256": lock_file_sha256,
        "lock_manifest_sha256": lock["manifest_sha256"],
        "expected_units": len(expected),
        "observed_units": len(records),
        "zero_silent_drops": True,
        "full_reconstruction_verification_required": True,
        "verification_manifests": verification_manifests,
        "compute_accounting": {
            "panel": panel_compute_accounting,
            "full_reconstruction_verification": (
                verification_compute_accounting
            ),
            "peak_rss_bytes": None,
            "peak_rss_status": PEAK_RSS_STATUS,
            "coverage_limit": (
                "failed_or_interrupted_work_that_published_no_authenticated_"
                "unit_or_verification_shard_is_not_observable_and_requires_"
                "external_campaign_log_accounting"
            ),
        },
        "inference": dict(inference),
        "conditions": conditions,
        "gates": gates,
        "unit_manifests": [
            {
                "unit_id": _unit_id(identity),
                "manifest_file_sha256": records[identity]["manifest_file_sha256"],
                "manifest_sha256": records[identity]["manifest_sha256"],
            }
            for identity in sorted(records)
        ],
        "interpretation": {
            "external_ground_truth_labels_used_for_optimization": False,
            "frozen_readout_predictions_used_for_optimization": True,
            "target_wnids_preflight_integrity_validated": True,
            "target_wnids_not_used_as_certificate_optimization_inputs": True,
            "target_wnids_used_for_posthoc_cluster_inference_after_assignment_fixed": (
                True
            ),
            "sample_identifiers_used_for_cost_or_reward": False,
            "sample_identifiers_used_for_indexed_atom_administration": True,
            "sample_identifiers_parsed_for_semantics": False,
            "exact_duplicate_atom_posthoc_tie_caveat": config[
                "certificate_protocol"
            ]["sample_identifier_posthoc_tie_caveat"],
            "gate_passes_are_not_asserted_until_this_aggregate_exists": True,
        },
    }
    write_json_atomic(output, seal_manifest(payload))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-lock")
    freeze.add_argument("--config", required=True, type=Path)
    freeze.add_argument("--store-map", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run-unit")
    run.add_argument("--lock", required=True, type=Path)
    run.add_argument("--store", required=True, type=Path)
    run.add_argument("--dataset", required=True)
    run.add_argument("--encoder", required=True)
    run.add_argument("--geometry", required=True)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--budget", required=True, type=float)
    run.add_argument("--output", required=True, type=Path)
    panel = subparsers.add_parser("run-panel")
    panel.add_argument("--lock", required=True, type=Path)
    panel.add_argument("--store-map", required=True, type=Path)
    panel.add_argument("--output-root", required=True, type=Path)
    panel.add_argument("--shard-count", required=True, type=int)
    panel.add_argument("--shard-index", required=True, type=int)
    panel.add_argument("--resume", action="store_true")
    verify = subparsers.add_parser("verify-unit")
    verify.add_argument("--lock", required=True, type=Path)
    verify.add_argument("--store", required=True, type=Path)
    verify.add_argument("--unit", required=True, type=Path)
    verify_panel = subparsers.add_parser("verify-panel")
    verify_panel.add_argument("--lock", required=True, type=Path)
    verify_panel.add_argument("--store-map", required=True, type=Path)
    verify_panel.add_argument("--units-root", required=True, type=Path)
    verify_panel.add_argument("--output", required=True, type=Path)
    verify_panel.add_argument("--shard-count", required=True, type=int)
    verify_panel.add_argument("--shard-index", required=True, type=int)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--lock", required=True, type=Path)
    aggregate.add_argument("--units-root", required=True, type=Path)
    aggregate.add_argument("--verification-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "freeze-lock":
        output = freeze_lock(arguments.config, arguments.store_map, arguments.output)
        print(json.dumps({
            "status": "PASS",
            "output": str(output),
            "sha256": sha256_file(output),
        }, sort_keys=True))
        return 0
    if arguments.command == "run-unit":
        output = run_unit(
            arguments.lock,
            arguments.store,
            arguments.output,
            dataset=arguments.dataset,
            encoder=arguments.encoder,
            geometry_name=arguments.geometry,
            seed=arguments.seed,
            budget=arguments.budget,
        )
        print(json.dumps({
            "status": "PASS",
            "output": str(output),
            "sha256": sha256_file(output),
        }, sort_keys=True))
        return 0
    if arguments.command == "run-panel":
        output = run_panel_shard(
            arguments.lock,
            arguments.store_map,
            arguments.output_root,
            shard_count=arguments.shard_count,
            shard_index=arguments.shard_index,
            resume=arguments.resume,
        )
        print(json.dumps({
            "status": "PASS",
            "output": str(output),
            "sha256": sha256_file(output),
        }, sort_keys=True))
        return 0
    if arguments.command == "verify-panel":
        output = verify_panel_shard(
            arguments.lock,
            arguments.store_map,
            arguments.units_root,
            arguments.output,
            shard_count=arguments.shard_count,
            shard_index=arguments.shard_index,
        )
        print(json.dumps({
            "status": "PASS",
            "output": str(output),
            "sha256": sha256_file(output),
        }, sort_keys=True))
        return 0
    if arguments.command == "aggregate":
        output = aggregate_units(
            arguments.lock,
            arguments.units_root,
            arguments.verification_root,
            arguments.output,
        )
        aggregate_payload = _mapping(
            read_json(output), name="optimization aggregate"
        )
        verify_manifest_checksum(aggregate_payload)
        gate_status = str(aggregate_payload.get("scientific_gate_status", ""))
        all_required = aggregate_payload.get("all_required_gates_pass")
        if gate_status not in {"PASS", "FAIL"} or all_required is not (
            gate_status == "PASS"
        ):
            raise IntegrityError("aggregate scientific gate status is invalid")
        print(json.dumps({
            "status": gate_status,
            "artifact_status": "PASS",
            "scientific_gate_status": gate_status,
            "all_required_gates_pass": all_required,
            "failed_required_scientific_gate_keys": aggregate_payload[
                "failed_required_scientific_gate_keys"
            ],
            "output": str(output),
            "sha256": sha256_file(output),
        }, sort_keys=True))
        return 0 if all_required else SCIENTIFIC_GATE_FAILURE_EXIT_CODE
    report = verify_unit(arguments.lock, arguments.store, arguments.unit)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
