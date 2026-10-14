"""Test-suite classifications that must not mutate historically locked tests."""

from __future__ import annotations

import pytest


# These tests authenticate records that are deliberately absent from the
# anonymous source-only package: 21 depend on the original campaign/evidence
# tree and two bind pre-redaction historical launcher bytes.  Keep the marker
# assignment here because several target test modules are themselves hashed by
# pre-outcome protocol locks and therefore must remain byte-for-byte stable.
EVIDENCE_INTEGRATED_NODE_IDS = (
    "tests/iclr_extension/test_aggregation.py::test_complete_panel_aggregation_is_seed_nested_and_immutable",
    "tests/iclr_extension/test_aggregation.py::test_aggregation_rejects_incomplete_duplicate_nonfinite_or_drifted_panels",
    "tests/iclr_extension/test_natural_shift.py::test_play3_config_and_locks_bind_the_exact_base_external_manifest",
    "tests/iclr_extension/test_natural_shift.py::test_locked_plan_only_rejects_a_structurally_valid_substitute_store",
    "tests/iclr_extension/test_rotation_runner.py::RotationRunnerTests::test_config_matches_master_lock_and_declares_complete_panels",
    "tests/iclr_extension/test_rotation_runner.py::RotationRunnerTests::test_config_drift_from_the_prespecification_fails_closed",
    "tests/iclr_extension/test_rotation_runner.py::RotationRunnerTests::test_play1_lock_rejects_source_execution_and_source_code_drift",
    "tests/iclr_extension/test_rotation_runner.py::RotationRunnerTests::test_shards_are_disjoint_complete_and_deterministic",
    "tests/iclr_extension/test_rotation_runner.py::RotationRunnerTests::test_aggregate_averages_splits_before_condition_macro",
    "tests/iclr_extension/test_rotation_runner.py::RotationRunnerTests::test_sealed_store_runs_both_methods_resumes_and_completes",
    "tests/iclr_extension/test_rtx_7day_profile.py::test_profile_amendment_and_lock_are_fully_sealed_and_tamper_evident",
    "tests/iclr_extension/test_rtx_7day_profile.py::test_sealed_campaign_plan_advertises_exact_selected_methods",
    "tests/tta_baselines/test_protocol.py::test_locked_primary_grid_and_controlled_scope",
    "tests/tta_baselines/test_protocol.py::test_deterministic_shards_are_disjoint_and_complete",
    "tests/tta_baselines/test_protocol.py::test_invalid_shard_fails_before_compute",
    "tests/tta_baselines/test_protocol.py::test_prespecification_lock_binds_feature_build_and_source_execution_evidence",
    "tests/tta_baselines/test_runner_contract.py::test_source_parity_gate_accepts_frozen_record_and_rejects_drift",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    evidence_integrated = pytest.mark.evidence_integrated
    for item in items:
        if any(
            item.nodeid == node_id or item.nodeid.startswith(f"{node_id}[")
            for node_id in EVIDENCE_INTEGRATED_NODE_IDS
        ):
            item.add_marker(evidence_integrated)
