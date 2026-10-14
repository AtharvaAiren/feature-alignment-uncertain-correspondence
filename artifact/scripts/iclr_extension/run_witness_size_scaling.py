#!/usr/bin/env python3
"""Run the leakage-safe nested witness-size scaling follow-up.

This is an additive, post-hoc reviewer-motivated study.  It leaves R5 and the
completed reviewer follow-up untouched.  One maximum endpoint selection and
one calibration geometry are frozen per condition/encoder/seed; fresh
certificates are then computed at n=1024, 4096, and 8192 for q=0.05.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

# The completed follow-up runner provides already-tested store binding and
# runtime helpers.  Keep this additive runner importable both as a script and
# from tests loaded with importlib.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_SRC = _REPOSITORY_ROOT / "src"
if str(_REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_SRC))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import run_reviewer_followup as prior  # noqa: E402

from marginal_alignment.campaign.artifacts import (  # noqa: E402
    IntegrityError,
    describe_file,
    describe_npy,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_file_record,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.iclr_extension.natural_shift_store import (  # noqa: E402
    load_official_class_index,
)
from marginal_alignment.iclr_extension.optimization_certificate import (  # noqa: E402
    FULL_MAHALANOBIS,
    OptimizationCertificateProtocol,
    _array_sha256,
    _unique_canonical_rows,
    fit_certificate_geometry,
    prepare_certificate_problem,
    solve_exact_budgeted_disagreement_milp,
)
from marginal_alignment.iclr_extension.pair_structure import (  # noqa: E402
    relative_budget_for_random_gap_share,
)
from marginal_alignment.iclr_extension.witness_scaling import (  # noqa: E402
    WitnessSizeProtocol,
    build_nested_natural_selection,
    build_nested_pair_selection,
    class_coverage,
    construct_scaling_certificate_from_problem,
    cross_endpoint_class_coverage,
    materialize_prefix,
    run_scaling_exact_subset_validation,
    uniform_random_cost_moments_blockwise,
    validate_nested_selection,
    validate_nested_selection_manifest,
)


CONFIG_SCHEMA = "marginal_alignment.iclr_extension.witness_size_scaling.v1"
CONFIG_PAYLOAD_SHA256 = (
    "bddd415022066d298d84e6682ca0f51a9374626888a526c288e3bedc8ed12ada"
)
LOCK_SCHEMA = "marginal_alignment.witness_size_scaling_lock.v1"
GROUP_SCHEMA = "marginal_alignment.witness_size_scaling_group.v1"
VERIFY_SCHEMA = "marginal_alignment.witness_size_scaling_verification.v1"
AGGREGATE_SCHEMA = "marginal_alignment.witness_size_scaling_aggregate.v1"
STATE_SCHEMA = "marginal_alignment.witness_size_scaling_state.v1"
NATURAL_PANEL = "imagenet_sketch_natural_unpaired"
CONTROL_PANEL = "imagenet_c_disjoint_unpaired"
SELECTION_SALT = "bm-witness-size-scaling-20260910-v1"
WITNESS_SIZES = (1024, 4096, 8192)
ENCODERS = ("clip", "dinov3")
SEEDS = (201, 202, 203)
CORRUPTIONS = tuple(prior.IMAGENET_C_CORRUPTIONS)
_SCALING_SOURCE_FILES = (
    "configs/iclr_extension/witness_size_scaling_20260910.json",
    "docs/witness_size_scaling_20260910.md",
    "scripts/iclr_extension/run_witness_size_scaling.py",
    "src/marginal_alignment/iclr_extension/witness_scaling.py",
    "tests/iclr_extension/test_witness_scaling.py",
    "tests/iclr_extension/test_witness_size_scaling_runner.py",
)
SOURCE_FILES = tuple(dict.fromkeys(_SCALING_SOURCE_FILES + tuple(prior.SOURCE_FILES)))


@dataclass(frozen=True)
class GroupSpec:
    panel: str
    dataset: str
    corruption: str | None
    encoder: str
    seed: int

    @property
    def condition(self) -> str:
        return self.dataset if self.corruption is None else str(self.corruption)

    @property
    def group_id(self) -> str:
        return f"{self.panel}__{self.condition}__{self.encoder}__seed_{self.seed}"


def _repository_root() -> Path:
    return _REPOSITORY_ROOT


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a mapping")
    return value


def _load_config(path: str | Path) -> dict[str, Any]:
    value = dict(_mapping(read_json(path), name="scaling config"))
    common = _mapping(value.get("common_protocol"), name="common protocol")
    panels = _mapping(value.get("panels"), name="panels")
    c_panel = _mapping(panels.get(CONTROL_PANEL), name="ImageNet-C panel")
    resources = _mapping(value.get("resource_contract"), name="resource contract")
    counts = _mapping(value.get("expected_counts"), name="expected counts")
    if (
        sha256_json(value) != CONFIG_PAYLOAD_SHA256
        or value.get("schema_version") != CONFIG_SCHEMA
        or value.get("status")
        != "PRESPECIFIED_POST_HOC_REVIEWER_MOTIVATED_BEFORE_SCALING_OUTCOMES"
        or value.get("scientific_result") is not False
        or tuple(common.get("encoders", ())) != ENCODERS
        or tuple(map(int, common.get("split_seeds", ()))) != SEEDS
        or int(common.get("calibration_rows", -1)) != 3840
        or tuple(map(int, common.get("witness_sizes", ()))) != WITNESS_SIZES
        or int(common.get("maximum_witness_rows", -1)) != 8192
        or float(common.get("shared_random_gap_q", -1.0)) != 0.05
        or common.get("cost_geometry") != FULL_MAHALANOBIS
        or common.get("covariance_estimator") != "oas"
        or float(common.get("covariance_relative_floor", -1.0)) != 1e-6
        or int(common.get("exact_validation_rows", -1)) != 64
        or float(common.get("exact_milp_time_limit_seconds", -1.0)) != 900.0
        or float(common.get("exact_reported_mip_gap_tolerance", -1.0)) != 1e-12
        or int(common.get("maximum_dual_bisection_iterations", -1)) != 48
        or float(common.get("maximum_primal_dual_gap", -1.0)) != 0.01
        or common.get("early_stop_policy")
        != "stop_at_the_first_evaluation_with_a_valid_primal_dual_gap_at_most_0.01"
        or common.get("primal_initializer")
        != "minimum_cost_assignment_no_python_greedy_pair_set"
        or float(common.get("assignment_tie_relative_tolerance", -1.0)) != 1e-12
        or int(common.get("random_cost_moment_block_rows", -1)) != 256
        or common.get("selected_disagreement_role")
        != "integer_feasible_lower_bound_on_maximum_disagreement"
        or common.get("upper_bound_role")
        != (
            "sampled_lagrangian_computational_upper_bound_under_the_locked_"
            "floating_point_solver_model"
        )
        or common.get("exact_validation_role")
        != (
            "tractable_64_row_implementation_diagnostic_not_an_exact_solution_"
            "of_any_full_witness_problem"
        )
        or tuple(c_panel.get("corruptions", ())) != CORRUPTIONS
        or int(c_panel.get("eligible_identity_pairs", -1)) != 25000
        or int(c_panel.get("required_identity_pairs_at_maximum", -1)) != 24064
        or int(c_panel.get("mathematical_maximum_witness_rows", -1)) != 8660
        or int(resources.get("parallel_workers", -1)) != 2
        or resources.get("cpu_only") is not True
        or int(resources.get("maximum_concurrent_campaigns", -1)) != 1
        or int(resources.get("conservative_peak_gib_per_worker", -1)) != 12
        or int(resources.get("required_memavailable_gib_before_launch", -1)) != 28
        or int(counts.get("groups", -1)) != 96
        or int(counts.get("total_fresh_certificates", -1)) != 288
        or int(counts.get("external_completed_1024_certificates_used_as_baseline", -1))
        != 0
    ):
        raise IntegrityError("critical witness-size scaling protocol changed")
    protocol = WitnessSizeProtocol(
        calibration_rows=int(common["calibration_rows"]),
        witness_sizes=tuple(map(int, common["witness_sizes"])),
    )
    if protocol.required_disjoint_identity_rows != 24064:
        raise IntegrityError("witness-size identity capacity arithmetic changed")
    return value


def enumerate_groups(config: Mapping[str, Any]) -> list[GroupSpec]:
    _ = config
    groups = [
        GroupSpec(NATURAL_PANEL, "imagenet_sketch", None, encoder, seed)
        for encoder in ENCODERS
        for seed in SEEDS
    ]
    groups.extend(
        GroupSpec(CONTROL_PANEL, "imagenet_c", corruption, encoder, seed)
        for corruption in CORRUPTIONS
        for encoder in ENCODERS
        for seed in SEEDS
    )
    groups.sort(key=lambda item: item.group_id)
    if len(groups) != 96 or len({item.group_id for item in groups}) != 96:
        raise IntegrityError("scaling inventory must contain 96 unique groups")
    return groups


def shard_groups(
    groups: Sequence[GroupSpec], shard_count: int
) -> list[list[GroupSpec]]:
    if isinstance(shard_count, bool) or int(shard_count) != 2:
        raise ValueError("the memory-locked scaling scheduler requires two shards")
    # Keep all seeds for one store on one worker, and balance the 32 stores.
    store_keys = sorted(
        {(item.dataset, item.corruption, item.encoder) for item in groups}
    )
    allocation = {key: index % 2 for index, key in enumerate(store_keys)}
    shards: list[list[GroupSpec]] = [[], []]
    for item in groups:
        shards[allocation[(item.dataset, item.corruption, item.encoder)]].append(item)
    for values in shards:
        # Each group has the same {8192,4096,1024} workload; deterministic
        # descending condition order avoids accidental post-outcome scheduling.
        values.sort(key=lambda item: item.group_id, reverse=True)
    if sorted(item.group_id for values in shards for item in values) != sorted(
        item.group_id for item in groups
    ):
        raise IntegrityError("scaling sharding lost or duplicated a group")
    return shards


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    records = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"scaling source file is absent or symbolic: {path}")
        records[relative] = sha256_file(path)
    return records


def _runtime_contract() -> Mapping[str, Any]:
    return prior._runtime_contract()


def _store_key(spec: GroupSpec) -> str:
    return (
        f"imagenet_sketch::{spec.encoder}"
        if spec.corruption is None
        else f"imagenet_c/{spec.corruption}/severity_5::{spec.encoder}"
    )


def _selection_key(spec: GroupSpec) -> str:
    return spec.group_id


def _build_selection(store: Any, spec: GroupSpec, protocol: WitnessSizeProtocol):
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    if spec.panel == NATURAL_PANEL:
        return build_nested_natural_selection(
            reference_features=reference.features,
            target_features=target.features,
            reference_ids=reference.ids,
            target_ids=target.ids,
            dataset=spec.dataset,
            seed=spec.seed,
            protocol=protocol,
        )
    pairing = store.pairing()
    paired_reference, paired_target = pairing.validated()
    metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
    provenance = _mapping(metadata.get("model_provenance"), name="model provenance")
    role = str(provenance.get("head_fit_ids_role", ""))
    return build_nested_pair_selection(
        reference_ids=reference.ids,
        target_ids=target.ids,
        paired_reference_ids=paired_reference,
        paired_target_ids=paired_target,
        head_fit_reference_ids=store.array(role),
        seed=spec.seed,
        selection_salt=SELECTION_SALT,
        protocol=protocol,
    )


def _validate_completed_inputs(r5: Path, historical: Path) -> dict[str, Any]:
    r5_aggregate_path = r5 / "PLAY3_OPTIMIZATION_AGGREGATE.json"
    historical_aggregate_path = historical / "REVIEWER_FOLLOWUP_AGGREGATE.json"
    historical_done_path = historical / "DONE.json"
    for path in (r5_aggregate_path, historical_aggregate_path, historical_done_path):
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"completed input is absent or symbolic: {path}")
    r5_aggregate = _mapping(read_json(r5_aggregate_path), name="R5 aggregate")
    historical_aggregate = _mapping(
        read_json(historical_aggregate_path), name="historical aggregate"
    )
    historical_done = _mapping(read_json(historical_done_path), name="historical done")
    for value in (r5_aggregate, historical_aggregate, historical_done):
        verify_manifest_checksum(value)
    if (
        r5_aggregate.get("status") != "PASS"
        or r5_aggregate.get("all_required_gates_pass") is not True
        or historical_aggregate.get("status") != "PASS"
        or historical_done.get("status") != "PASS"
        or historical_done.get("schema_version")
        != "marginal_alignment.reviewer_followup_done.v1"
        or historical_done.get("aggregate_file_sha256")
        != sha256_file(historical_aggregate_path)
        or historical_done.get("aggregate_manifest_sha256")
        != historical_aggregate.get("manifest_sha256")
    ):
        raise IntegrityError("R5 or historical reviewer follow-up is not complete")
    return {
        "r5_aggregate": {
            "path": str(r5_aggregate_path),
            "file_sha256": sha256_file(r5_aggregate_path),
            "manifest_sha256": r5_aggregate["manifest_sha256"],
        },
        "historical_reviewer_followup": {
            "root": str(historical),
            "aggregate_file_sha256": sha256_file(historical_aggregate_path),
            "aggregate_manifest_sha256": historical_aggregate["manifest_sha256"],
            "done_file_sha256": sha256_file(historical_done_path),
            "used_as_scaling_baseline": False,
            "reason": "new_maximum_selection_changes_endpoint_prefixes",
        },
    }


def freeze_lock(
    *,
    config_path: str | Path,
    r5_root: str | Path,
    historical_followup_root: str | Path,
    feature_root: str | Path,
    external_input_manifest: str | Path,
    class_index: str | Path,
    output_root: str | Path,
) -> Path:
    config_path = Path(config_path).resolve(strict=True)
    r5 = Path(r5_root).resolve(strict=True)
    historical = Path(historical_followup_root).resolve(strict=True)
    features = Path(feature_root).resolve(strict=True)
    external = Path(external_input_manifest).resolve(strict=True)
    class_index_path = Path(class_index).resolve(strict=True)
    destination = Path(output_root).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to reuse scaling root: {destination}")
    config = _load_config(config_path)
    groups = enumerate_groups(config)
    completed = _validate_completed_inputs(r5, historical)
    external_hashes = prior._external_imagenet_c_hashes(external)
    official = load_official_class_index(class_index_path)
    protocol = WitnessSizeProtocol(3840, WITNESS_SIZES)

    stores: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for spec in groups:
        key = _store_key(spec)
        if key not in stores:
            if spec.corruption is None:
                manifest = (
                    r5
                    / "stores"
                    / "imagenet_sketch"
                    / spec.encoder
                    / "FEATURE_STORE_MANIFEST.json"
                )
            else:
                manifest = (
                    features
                    / "imagenet_c"
                    / spec.encoder
                    / spec.corruption
                    / "severity_5"
                    / "FEATURE_STORE_MANIFEST.json"
                )
            record = prior._bind_store(
                manifest,
                dataset=spec.dataset,
                encoder=spec.encoder,
                corruption=spec.corruption,
            )
            if spec.corruption is not None and record["manifest_file_sha256"] != (
                external_hashes.get((spec.corruption, spec.encoder))
            ):
                raise IntegrityError(f"ImageNet-C store is outside input lock: {key}")
            stores[key] = record
        store = prior._load_verified_store(str(stores[key]["manifest"]))
        selection = _build_selection(store, spec, protocol)
        if (
            spec.corruption is not None
            and int(selection.manifest["parent_selection"]["eligible_pair_count"])
            != 25000
        ):
            raise IntegrityError("ImageNet-C eligible identity count changed")
        selections[_selection_key(spec)] = dict(selection.manifest)

    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "FROZEN_BEFORE_WITNESS_SIZE_OUTCOMES",
        "scientific_result": False,
        "historical_outcomes_inspected_before_design": True,
        "new_scaling_outcomes_read_during_freeze": False,
        "config": config,
        "config_file_sha256": sha256_file(config_path),
        "source_files": _source_hashes(),
        "freeze_runtime": _runtime_contract(),
        **completed,
        "external_input_manifest": {
            "path": str(external),
            "file_sha256": sha256_file(external),
        },
        "class_index": {
            "path": str(class_index_path),
            "file_sha256": official.file_sha256,
            "mapping_sha256": official.canonical_mapping_sha256,
        },
        "stores": stores,
        "nested_selections": selections,
        "expected_groups": 96,
        "expected_certificates": 288,
    }
    destination.mkdir(parents=True)
    lock_path = destination / "WITNESS_SIZE_SCALING_LOCK.json"
    write_json_atomic(lock_path, seal_manifest(payload))
    write_json_atomic(
        destination / "STATE.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": "LOCKED",
            "completed_groups": 0,
            "expected_groups": 96,
        },
    )
    return lock_path


def _reauthenticate_frozen_inputs(
    lock: Mapping[str, Any],
) -> Mapping[tuple[str, str], str]:
    """Re-read every external completion/input bound into the scaling lock."""

    r5_record = _mapping(lock.get("r5_aggregate"), name="locked R5 aggregate")
    historical_record = _mapping(
        lock.get("historical_reviewer_followup"),
        name="locked historical reviewer follow-up",
    )
    r5_aggregate_path = Path(str(r5_record.get("path", ""))).resolve(strict=True)
    historical_root = Path(str(historical_record.get("root", ""))).resolve(strict=True)
    reauthenticated = _validate_completed_inputs(
        r5_aggregate_path.parent, historical_root
    )
    if (
        reauthenticated["r5_aggregate"] != r5_record
        or reauthenticated["historical_reviewer_followup"] != historical_record
    ):
        raise IntegrityError("completed input binding changed after freeze")
    external_record = _mapping(
        lock.get("external_input_manifest"), name="external input binding"
    )
    external_path = Path(str(external_record.get("path", ""))).resolve(strict=True)
    if sha256_file(external_path) != external_record.get("file_sha256"):
        raise IntegrityError("external input manifest changed after freeze")
    external_hashes = prior._external_imagenet_c_hashes(external_path)
    class_record = _mapping(lock.get("class_index"), name="class-index binding")
    class_path = Path(str(class_record.get("path", ""))).resolve(strict=True)
    official = load_official_class_index(class_path)
    if official.file_sha256 != class_record.get(
        "file_sha256"
    ) or official.canonical_mapping_sha256 != class_record.get("mapping_sha256"):
        raise IntegrityError("official class-index binding changed after freeze")
    return external_hashes


def _load_lock(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    lock = dict(_mapping(read_json(source), name="scaling lock"))
    verify_manifest_checksum(lock)
    config_path = (
        _repository_root() / "configs/iclr_extension/witness_size_scaling_20260910.json"
    )
    config = _load_config(config_path)
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "FROZEN_BEFORE_WITNESS_SIZE_OUTCOMES"
        or lock.get("new_scaling_outcomes_read_during_freeze") is not False
        or lock.get("config") != config
        or lock.get("config_file_sha256") != sha256_file(config_path)
        or lock.get("source_files") != _source_hashes()
        or lock.get("freeze_runtime") != _runtime_contract()
        or int(lock.get("expected_groups", -1)) != 96
        or int(lock.get("expected_certificates", -1)) != 288
    ):
        raise IntegrityError("witness-size scaling lock identity changed")
    external_hashes = _reauthenticate_frozen_inputs(lock)
    stores = _mapping(lock.get("stores"), name="locked stores")
    for key, raw in stores.items():
        record = _mapping(raw, name=f"store {key}")
        manifest = Path(str(record["manifest"])).resolve(strict=True)
        if sha256_file(manifest) != record["manifest_file_sha256"]:
            raise IntegrityError(f"locked store changed: {key}")
        if key.startswith("imagenet_c/"):
            parts = key.split("::", 1)
            corruption = parts[0].split("/")[1]
            encoder = parts[1]
            if record["manifest_file_sha256"] != external_hashes.get(
                (corruption, encoder)
            ):
                raise IntegrityError(f"locked ImageNet-C input changed: {key}")
    selections = _mapping(lock.get("nested_selections"), name="nested selections")
    if set(selections) != {item.group_id for item in enumerate_groups(config)}:
        raise IntegrityError("nested selection inventory changed")
    for manifest in selections.values():
        validate_nested_selection_manifest(_mapping(manifest, name="selection"))
    return lock


def _load_group_store(lock: Mapping[str, Any], spec: GroupSpec):
    record = _mapping(lock["stores"][_store_key(spec)], name="store binding")
    store = prior._load_verified_store(str(record["manifest"]))
    if (
        sha256_json(store.manifest) != record["manifest_payload_sha256"]
        or store.dataset != spec.dataset
        or store.encoder != spec.encoder
    ):
        raise IntegrityError("execution store differs from lock")
    return store, record


def _base_protocol(config: Mapping[str, Any]) -> OptimizationCertificateProtocol:
    common = _mapping(config["common_protocol"], name="common protocol")
    return OptimizationCertificateProtocol(
        cost_geometry=FULL_MAHALANOBIS,
        cost_budget_relative=0.0,
        covariance_estimator=str(common["covariance_estimator"]),
        covariance_relative_floor=float(common["covariance_relative_floor"]),
        maximum_witness_rows=8192,
        # Required by the historical protocol dataclass but deliberately
        # unused by the scaling solver, which constructs no Python greedy set.
        greedy_neighbors=1,
        dual_bisection_iterations=int(common["maximum_dual_bisection_iterations"]),
        primal_dual_gap_target=float(common["maximum_primal_dual_gap"]),
        assignment_tie_relative_tolerance=float(
            common["assignment_tie_relative_tolerance"]
        ),
    )


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    np.save(path, np.asarray(value), allow_pickle=False)
    return describe_npy(path, relative_to=root)


def _endpoint_class_arrays(
    store: Any, spec: GroupSpec
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """First interpret class values after all three certificates are frozen."""

    reference = store.endpoint("reference")
    target = store.endpoint("target")
    if spec.panel == NATURAL_PANEL:
        metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
        intersection = _mapping(
            metadata.get("class_intersection"), name="class intersection"
        )
        reference_role = str(intersection["reference_wnids_role"])
        target_role = str(intersection["target_wnids_role"])
        return (
            np.asarray(store.array(reference_role)),
            np.asarray(store.array(target_role)),
            {
                "phase": "after_all_three_certificates_frozen",
                "kind": "authenticated_endpoint_wnids",
                "reference_role": reference_role,
                "target_role": target_role,
            },
        )
    if reference.labels is None or target.labels is None:
        raise IntegrityError("ImageNet-C endpoint labels are absent")
    return (
        np.asarray(reference.labels),
        np.asarray(target.labels),
        {
            "phase": "after_all_three_certificates_frozen",
            "kind": "sealed_endpoint_class_indices",
        },
    )


def _compute_group(lock: Mapping[str, Any], spec: GroupSpec, *, staging_root: Path):
    config = _mapping(lock["config"], name="config")
    common = _mapping(config["common_protocol"], name="common protocol")
    protocol = WitnessSizeProtocol(3840, WITNESS_SIZES)
    store, _ = _load_group_store(lock, spec)
    selection = _build_selection(store, spec, protocol)
    locked_selection = _mapping(
        lock["nested_selections"][_selection_key(spec)], name="locked selection"
    )
    if dict(selection.manifest) != dict(locked_selection):
        raise IntegrityError("execution endpoint selection differs from lock")
    reference_endpoint = store.endpoint("reference")
    target_endpoint = store.endpoint("target")
    maximum = materialize_prefix(
        selection,
        reference_features=reference_endpoint.features,
        target_features=target_endpoint.features,
        witness_rows=8192,
    )
    calibration = protocol.calibration_rows
    head = _mapping(store.manifest.get("head"), name="frozen head")
    weight = np.asarray(store.array(str(head["weight_role"])), dtype=np.float64)
    bias = np.asarray(store.array(str(head["bias_role"])), dtype=np.float64)
    base_protocol = _base_protocol(config)
    geometry = fit_certificate_geometry(
        maximum["target_features"][:calibration],
        maximum["reference_features"][:calibration],
        weight,
        bias,
        protocol=base_protocol,
        target_calibration_ids=maximum["target_ids"][:calibration],
        reference_calibration_ids=maximum["reference_ids"][:calibration],
    )

    certificates: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    frozen_prefixes: dict[int, Any] = {}
    # Largest first exposes memory/runtime failures early and every group uses
    # the same predetermined order.
    for size in reversed(WITNESS_SIZES):
        prefix = selection.prefix(size)
        witness_target = np.asarray(
            target_endpoint.features[prefix.target_witness_rows], dtype=np.float64
        )
        witness_reference = np.asarray(
            reference_endpoint.features[prefix.reference_witness_rows], dtype=np.float64
        )
        target_ids = np.asarray(prefix.target_ids[calibration:])
        reference_ids = np.asarray(prefix.reference_ids[calibration:])
        problem = prepare_certificate_problem(
            geometry,
            witness_target,
            witness_reference,
            target_sample_ids=target_ids,
            reference_sample_ids=reference_ids,
        )
        moments = uniform_random_cost_moments_blockwise(
            problem.cost,
            block_rows=int(common["random_cost_moment_block_rows"]),
        )
        relative = relative_budget_for_random_gap_share(
            float(problem.baseline_cost),
            float(moments["expected_total_cost"]),
            float(common["shared_random_gap_q"]),
        )
        unit_protocol = replace(base_protocol, cost_budget_relative=relative)
        result = construct_scaling_certificate_from_problem(
            problem, protocol=unit_protocol
        )
        exact_geometry = replace(
            geometry,
            protocol=unit_protocol,
            diagnostics={
                **dict(geometry.diagnostics),
                "protocol": unit_protocol.as_dict(),
            },
        )
        (
            exact_assignment,
            exact,
            exact_problem_arrays,
        ) = run_scaling_exact_subset_validation(
            exact_geometry,
            witness_target,
            witness_reference,
            target_sample_ids=target_ids,
            reference_sample_ids=reference_ids,
            validation_rows=int(common["exact_validation_rows"]),
            time_limit_seconds=float(common["exact_milp_time_limit_seconds"]),
        )
        if (
            exact.get("status") != "PROVEN_OPTIMAL"
            or float(exact.get("reported_mip_gap", float("inf")))
            > float(common["exact_reported_mip_gap_tolerance"])
            or exact.get("certificate_contains_exact_optimum") is not True
            or result.primal_dual_gap > float(common["maximum_primal_dual_gap"])
        ):
            raise IntegrityError("scaling numerical certificate gate failed")
        key = f"n_{size:05d}"
        arrays.update(
            {
                f"{key}__assignment_world_zero": result.assignment_world_zero,
                f"{key}__assignment_world_one": result.assignment_world_one,
                f"{key}__prediction_world_zero": result.prediction_world_zero,
                f"{key}__prediction_world_one": result.prediction_world_one,
                f"{key}__target_canonical_order": result.target_canonical_order,
                f"{key}__reference_canonical_order": result.reference_canonical_order,
                f"{key}__exact_subset_assignment": exact_assignment,
                **{
                    f"{key}__exact_subset_{name}": value
                    for name, value in exact_problem_arrays.items()
                },
            }
        )
        certificates.append(
            {
                "witness_rows": size,
                "shared_random_gap_q": 0.05,
                "derived_cost_budget_relative_to_minimum": float(relative),
                "random_bijection_cost": moments,
                "result": result.summary(),
                "exact_small_instance_validation": exact,
            }
        )
        frozen_prefixes[size] = prefix
        del problem, witness_target, witness_reference, result
        gc.collect()

    certificates.sort(key=lambda row: int(row["witness_rows"]))
    prelabel_scientific = {
        "schema_version": (
            "marginal_alignment.witness_size_scaling_prelabel_scientific.v1"
        ),
        "group_id": spec.group_id,
        "panel": spec.panel,
        "dataset": spec.dataset,
        "corruption": spec.corruption,
        "encoder": spec.encoder,
        "split_seed": spec.seed,
        "calibration_rows": calibration,
        "witness_sizes": list(WITNESS_SIZES),
        "shared_random_gap_q": 0.05,
        "selection_manifest_sha256": selection.manifest["manifest_sha256"],
        "geometry": dict(geometry.diagnostics),
        "geometry_fit_count": 1,
        "geometry_reused_across_sizes": True,
        "pairing_available_to_geometry_cost_reward_or_optimizer": False,
        "historical_1024_used_as_scaling_baseline": False,
        "certificates": certificates,
    }
    array_records = {
        name: _save_array(staging_root, name, value)
        for name, value in sorted(arrays.items())
    }
    prelabel_path = staging_root / "PRELABEL_CERTIFICATE_MANIFEST.json"
    write_json_atomic(
        prelabel_path,
        seal_manifest(
            {
                "schema_version": (
                    "marginal_alignment.witness_size_scaling_prelabel_layer.v1"
                ),
                "status": "PASS",
                "scientific_result": True,
                "label_values_sliced_or_interpreted": False,
                "lock_manifest_sha256": lock["manifest_sha256"],
                "scientific": prelabel_scientific,
                "arrays": array_records,
            }
        ),
    )
    frozen_prelabel = dict(read_json(prelabel_path))
    verify_manifest_checksum(frozen_prelabel)
    if frozen_prelabel["scientific"] != prelabel_scientific:
        raise IntegrityError("pre-label certificate layer changed while sealing")
    for record in array_records.values():
        verify_file_record(staging_root, record)

    # Store authentication may open label memmaps, but no label/WNID values
    # are sliced or interpreted before the immutable layer above is sealed.
    reference_classes, target_classes, class_source = _endpoint_class_arrays(
        store, spec
    )
    certificates = [dict(row) for row in frozen_prelabel["scientific"]["certificates"]]
    certificate_by_size = {int(row["witness_rows"]): row for row in certificates}
    for size in WITNESS_SIZES:
        prefix = frozen_prefixes[size]
        reference_witness_classes = np.asarray(
            reference_classes[prefix.reference_witness_rows]
        )
        target_witness_classes = np.asarray(target_classes[prefix.target_witness_rows])
        certificate_by_size[size]["class_coverage"] = {
            "phase": "after_all_three_certificates_frozen",
            "source": class_source,
            "reference": class_coverage(reference_witness_classes),
            "target": class_coverage(target_witness_classes),
            "cross_endpoint": cross_endpoint_class_coverage(
                target_witness_classes, reference_witness_classes
            ),
        }
    return (
        {
            "schema_version": "marginal_alignment.witness_size_scaling_scientific.v1",
            "group_id": spec.group_id,
            "panel": spec.panel,
            "dataset": spec.dataset,
            "corruption": spec.corruption,
            "encoder": spec.encoder,
            "split_seed": spec.seed,
            "calibration_rows": calibration,
            "witness_sizes": list(WITNESS_SIZES),
            "shared_random_gap_q": 0.05,
            "selection_manifest_sha256": selection.manifest["manifest_sha256"],
            "geometry": dict(geometry.diagnostics),
            "geometry_fit_count": 1,
            "geometry_reused_across_sizes": True,
            "pairing_available_to_geometry_cost_reward_or_optimizer": False,
            "label_values_sliced_or_interpreted_only_after_all_three_"
            "certificates_frozen": True,
            "prelabel_certificate_manifest_sha256": frozen_prelabel["manifest_sha256"],
            "prelabel_certificate_manifest_file_sha256": sha256_file(prelabel_path),
            "historical_1024_used_as_scaling_baseline": False,
            "certificates": certificates,
        },
        array_records,
        describe_file(prelabel_path, relative_to=staging_root),
    )


def _group_path(root: Path, spec: GroupSpec) -> Path:
    return root / "groups" / spec.group_id


def _expected_arrays() -> set[str]:
    return {
        f"n_{size:05d}__{name}"
        for size in WITNESS_SIZES
        for name in (
            "assignment_world_zero",
            "assignment_world_one",
            "prediction_world_zero",
            "prediction_world_one",
            "target_canonical_order",
            "reference_canonical_order",
            "exact_subset_assignment",
            "exact_subset_cost",
            "exact_subset_reference_prediction",
            "exact_subset_prediction_world_zero",
            "exact_subset_scaling_assignment_world_zero",
            "exact_subset_scaling_assignment_world_one",
            "exact_subset_scaling_prediction_world_zero",
            "exact_subset_scaling_prediction_world_one",
            "exact_subset_scaling_target_canonical_order",
            "exact_subset_scaling_reference_canonical_order",
        )
    }


def _numerically_equal(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return bool(
        math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=atol)
    )


def _validate_optimization_trace(
    result: Mapping[str, Any],
    *,
    size: int,
    gap_target: float,
) -> None:
    """Reconstruct the reported primal/dual bracket from the serialized trace."""

    diagnostics = _mapping(result.get("diagnostics"), name="result diagnostics")
    optimization = _mapping(
        diagnostics.get("optimization"), name="optimization diagnostics"
    )
    trace = optimization.get("trace")
    scale = float(optimization.get("cost_normalization_scale", float("nan")))
    normalized_budget = float(optimization.get("normalized_budget", float("nan")))
    feasibility_tolerance = float(
        optimization.get("feasibility_tolerance", float("nan"))
    )
    allowed = float(result.get("allowed_cost", float("nan")))
    baseline = float(result.get("baseline_cost", float("nan")))
    primal_cost = float(result.get("primal_cost", float("nan")))
    assignment_zero_hash = result.get("assignment_world_zero_sha256")
    assignment_one_hash = result.get("assignment_world_one_sha256")
    if (
        not isinstance(trace, list)
        or not trace
        or not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(normalized_budget)
        or not _numerically_equal(normalized_budget, allowed / scale)
        or not math.isfinite(feasibility_tolerance)
        or feasibility_tolerance < 0.0
        or optimization.get("primal_solver") != "best_feasible_lagrangian_assignment"
        or optimization.get("dual_solver")
        != "scipy_linear_sum_assignment_lagrangian_search"
        or not _numerically_equal(
            float(optimization.get("early_stop_target", float("nan"))), gap_target
        )
        or int(optimization.get("maximum_bisection_iterations", -1)) != 48
        or int(optimization.get("evaluations", -1)) != len(trace)
        or optimization.get("objective_buffer_reused") is not True
        or optimization.get("dense_float_reward_matrix_created") is not False
        or optimization.get("upper_bounds_lp_relaxation_optimum") is not True
        or optimization.get("optimized_lagrangian_dual_value_claimed") is not False
        or optimization.get("formal_real_arithmetic_certificate_claimed") is not False
        or optimization.get("floating_point_solver_not_formally_verified") is not True
        or optimization.get("roundoff_upper_cushion_applied") is not True
        or not isinstance(assignment_zero_hash, str)
        or not isinstance(assignment_one_hash, str)
    ):
        raise IntegrityError("optimization trace header is inconsistent")

    epsilon = np.finfo(np.float64).eps
    best_primal = 0.0
    best_dual = 1.0
    best_dual_multiplier = 0.0
    selected_candidates = [(baseline, 0.0, assignment_zero_hash)]
    first_gap_target_evaluation: int | None = None
    for index, raw_record in enumerate(trace):
        record = _mapping(raw_record, name="optimization trace record")
        scaled_multiplier = float(record.get("scaled_multiplier", float("nan")))
        mean_multiplier = float(record.get("mean_reward_multiplier", float("nan")))
        assignment_cost = float(record.get("assignment_cost", float("nan")))
        disagreement = float(record.get("disagreement", float("nan")))
        reported_dual = float(record.get("dual_upper_bound", float("nan")))
        assignment_hash = record.get("assignment_sha256")
        if (
            not math.isfinite(scaled_multiplier)
            or scaled_multiplier < 0.0
            or not _numerically_equal(mean_multiplier, scaled_multiplier / size)
            or not math.isfinite(assignment_cost)
            or assignment_cost < 0.0
            or not math.isfinite(disagreement)
            or not 0.0 <= disagreement <= 1.0
            or not isinstance(assignment_hash, str)
            or len(assignment_hash) != 64
        ):
            raise IntegrityError("optimization trace record is malformed")
        normalized_assignment_cost = assignment_cost / scale
        raw_dual = float(
            disagreement
            + mean_multiplier * (normalized_budget - normalized_assignment_cost)
        )
        cushion = float(
            2048.0
            * size
            * epsilon
            * max(
                1.0,
                abs(raw_dual),
                abs(mean_multiplier)
                * (abs(normalized_budget) + abs(normalized_assignment_cost)),
            )
        )
        expected_dual = min(1.0, raw_dual + cushion)
        feasible = assignment_cost <= allowed
        if (
            not math.isfinite(reported_dual)
            or not 0.0 <= reported_dual <= 1.0
            or not _numerically_equal(reported_dual, expected_dual)
            or record.get("budget_feasible") is not feasible
        ):
            raise IntegrityError("optimization trace dual or feasibility changed")
        if expected_dual < best_dual:
            best_dual = expected_dual
            best_dual_multiplier = mean_multiplier
        if feasible:
            selected_candidates.append((assignment_cost, disagreement, assignment_hash))
            best_primal = max(best_primal, disagreement)
        trace_upper = float(min(1.0, max(best_primal, best_dual)))
        trace_gap = float(max(trace_upper - best_primal, 0.0))
        if not _numerically_equal(
            float(record.get("best_gap_after_evaluation", float("nan"))), trace_gap
        ):
            raise IntegrityError("optimization trace running gap changed")
        if trace_gap <= gap_target and first_gap_target_evaluation is None:
            first_gap_target_evaluation = index

    lower = float(result.get("maximum_disagreement_primal_lower_bound", float("nan")))
    upper = float(result.get("maximum_disagreement_dual_upper_bound", float("nan")))
    gap = float(result.get("primal_dual_gap", float("nan")))
    stopped = optimization.get("early_stop_reached")
    if (
        not _numerically_equal(lower, best_primal)
        or not _numerically_equal(upper, min(1.0, max(best_primal, best_dual)))
        or not _numerically_equal(gap, upper - lower)
        or not _numerically_equal(
            float(optimization.get("best_dual_multiplier", float("nan"))),
            best_dual_multiplier,
        )
        or stopped is not (first_gap_target_evaluation is not None)
        or (
            first_gap_target_evaluation is not None
            and first_gap_target_evaluation != len(trace) - 1
        )
        or not any(
            candidate_hash == assignment_one_hash
            and _numerically_equal(candidate_disagreement, lower)
            and _numerically_equal(candidate_cost, primal_cost)
            for candidate_cost, candidate_disagreement, candidate_hash in selected_candidates
        )
    ):
        raise IntegrityError("optimization trace does not derive the result bracket")


def _validate_result_numbers(
    result: Mapping[str, Any],
    *,
    size: int,
    relative: float,
    gap_target: float,
    require_gap_target: bool,
) -> None:
    diagnostics = _mapping(result.get("diagnostics"), name="result diagnostics")
    protocol = _mapping(diagnostics.get("protocol"), name="result protocol")
    baseline = float(result.get("baseline_cost", float("nan")))
    allowed = float(result.get("allowed_cost", float("nan")))
    primal_cost = float(result.get("primal_cost", float("nan")))
    selected = float(result.get("selected_world_disagreement", float("nan")))
    minimax = float(result.get("selected_two_world_minimax_recovery", float("nan")))
    lower = float(result.get("maximum_disagreement_primal_lower_bound", float("nan")))
    upper = float(result.get("maximum_disagreement_dual_upper_bound", float("nan")))
    gap = float(result.get("primal_dual_gap", float("nan")))
    identification = result.get("maximum_disagreement_identification_interval")
    recovery = result.get("strongest_two_world_recovery_ceiling_interval")
    cost_tolerance = 1e-12 * max(abs(allowed), abs(primal_cost), 1.0)
    expected_meets = bool(gap <= gap_target)
    if (
        result.get("status") != "PASS"
        or result.get("meets_gap_target") is not expected_meets
        or (require_gap_target and not expected_meets)
        or int(diagnostics.get("witness_rows", -1)) != size
        or not 0.0 <= lower <= upper <= 1.0
        or not 0.0 <= selected <= 1.0
        or not _numerically_equal(selected, lower)
        or not _numerically_equal(gap, upper - lower)
        or not _numerically_equal(
            float(result.get("primal_dual_gap_percentage_points", float("nan"))),
            100.0 * gap,
        )
        or not _numerically_equal(
            float(result.get("gap_target", float("nan"))), gap_target
        )
        or not _numerically_equal(
            float(result.get("gap_target_percentage_points", float("nan"))),
            100.0 * gap_target,
        )
        or not _numerically_equal(minimax, 1.0 - selected / 2.0)
        or not isinstance(identification, list)
        or len(identification) != 2
        or not _numerically_equal(float(identification[0]), lower)
        or not _numerically_equal(float(identification[1]), upper)
        or not isinstance(recovery, list)
        or len(recovery) != 2
        or not _numerically_equal(float(recovery[0]), 1.0 - upper / 2.0)
        or not _numerically_equal(float(recovery[1]), 1.0 - lower / 2.0)
        or not math.isfinite(relative)
        or relative < 0.0
        or not math.isfinite(baseline)
        or not math.isfinite(allowed)
        or not math.isfinite(primal_cost)
        or baseline < 0.0
        or allowed < 0.0
        or primal_cost < 0.0
        or not _numerically_equal(allowed, baseline * (1.0 + relative))
        or primal_cost > allowed + cost_tolerance
        or not _numerically_equal(
            float(protocol.get("cost_budget_relative", float("nan"))), relative
        )
        or not _numerically_equal(
            float(protocol.get("primal_dual_gap_target", float("nan"))), gap_target
        )
        or int(protocol.get("dual_bisection_iterations", -1)) != 48
        or float(protocol.get("assignment_tie_relative_tolerance", -1.0)) != 1e-12
    ):
        raise IntegrityError("certificate numerical identities are inconsistent")
    _validate_optimization_trace(result, size=size, gap_target=gap_target)


def _validate_certificate_numbers(row: Mapping[str, Any], *, size: int) -> None:
    """Re-derive every numerical identity used to interpret one certificate."""

    result = _mapping(row.get("result"), name="certificate result")
    exact = _mapping(
        row.get("exact_small_instance_validation"), name="exact diagnostic"
    )
    moments = _mapping(row.get("random_bijection_cost"), name="random-cost moments")
    relative = float(row.get("derived_cost_budget_relative_to_minimum", float("nan")))
    baseline = float(result.get("baseline_cost", float("nan")))
    random_mean = float(moments.get("expected_total_cost", float("nan")))
    if (
        not math.isfinite(baseline)
        or baseline <= 0.0
        or not math.isfinite(random_mean)
        or random_mean <= baseline
        or not math.isfinite(relative)
        or relative < 0.0
        or not _numerically_equal(
            relative,
            relative_budget_for_random_gap_share(baseline, random_mean, 0.05),
        )
    ):
        raise IntegrityError("certificate numerical identities are inconsistent")
    _validate_result_numbers(
        result,
        size=size,
        relative=relative,
        gap_target=0.01,
        require_gap_target=True,
    )

    scaling = _mapping(exact.get("scaling_certificate"), name="exact scaling result")
    _validate_result_numbers(
        scaling,
        size=64,
        relative=relative,
        gap_target=0.0,
        require_gap_target=False,
    )
    exact_lower = float(exact.get("certificate_primal_lower_bound", float("nan")))
    exact_upper = float(exact.get("certificate_dual_upper_bound", float("nan")))
    exact_gap = float(exact.get("certificate_primal_dual_gap", float("nan")))
    exact_optimum = float(exact.get("maximum_disagreement_exact", float("nan")))
    exact_assignment_cost = float(exact.get("assignment_cost", float("nan")))
    exact_allowed = float(exact.get("allowed_cost", float("nan")))
    exact_cost_tolerance = 1e-12 * max(
        abs(exact_assignment_cost), abs(exact_allowed), 1.0
    )
    reported_mip_gap = float(exact.get("reported_mip_gap", float("inf")))
    recomputed_containment = bool(
        exact_lower - 1e-10 <= exact_optimum <= exact_upper + 1e-10
    )
    if (
        exact.get("status") != "PROVEN_OPTIMAL"
        or exact.get("certificate_solver") != "witness_scaling_no_greedy_early_stop_v1"
        or int(exact.get("rows", -1)) != 64
        or not math.isfinite(reported_mip_gap)
        or not 0.0 <= reported_mip_gap <= 1e-12
        or not 0.0 <= exact_lower <= exact_upper <= 1.0
        or not _numerically_equal(
            exact_lower,
            float(scaling.get("maximum_disagreement_primal_lower_bound", float("nan"))),
        )
        or not _numerically_equal(
            exact_upper,
            float(scaling.get("maximum_disagreement_dual_upper_bound", float("nan"))),
        )
        or not _numerically_equal(exact_gap, exact_upper - exact_lower)
        or not _numerically_equal(
            exact_gap, float(scaling.get("primal_dual_gap", float("nan")))
        )
        or not 0.0 <= exact_optimum <= 1.0
        or not recomputed_containment
        or exact.get("certificate_contains_exact_optimum") is not True
        or not math.isfinite(exact_assignment_cost)
        or not math.isfinite(exact_allowed)
        or exact_assignment_cost < 0.0
        or exact_allowed < 0.0
        or exact_assignment_cost > exact_allowed + exact_cost_tolerance
        or not _numerically_equal(
            exact_allowed, float(scaling.get("allowed_cost", float("nan")))
        )
    ):
        raise IntegrityError("exact diagnostic numerical identities are inconsistent")


def _is_permutation(value: np.ndarray, size: int) -> bool:
    return bool(
        value.shape == (size,)
        and np.issubdtype(value.dtype, np.integer)
        and np.array_equal(np.sort(value), np.arange(size))
    )


def _validate_exact_diagnostic_arrays(
    exact: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    key: str,
    rerun_exact_solver: bool,
) -> None:
    """Recompute the 64-row MILP and scaling objectives from sealed arrays."""

    names = {
        name: np.asarray(arrays[f"{key}__exact_subset_{name}"])
        for name in (
            "assignment",
            "cost",
            "reference_prediction",
            "prediction_world_zero",
            "scaling_assignment_world_zero",
            "scaling_assignment_world_one",
            "scaling_prediction_world_zero",
            "scaling_prediction_world_one",
            "scaling_target_canonical_order",
            "scaling_reference_canonical_order",
        )
    }
    count = 64
    cost = names["cost"]
    reference_prediction = names["reference_prediction"]
    prediction_zero = names["prediction_world_zero"]
    exact_assignment = names["assignment"]
    scaling_zero = names["scaling_assignment_world_zero"]
    scaling_one = names["scaling_assignment_world_one"]
    scaling_prediction_zero = names["scaling_prediction_world_zero"]
    scaling_prediction_one = names["scaling_prediction_world_one"]
    target_order = names["scaling_target_canonical_order"]
    reference_order = names["scaling_reference_canonical_order"]
    scaling = _mapping(exact.get("scaling_certificate"), name="exact scaling result")
    scaling_diagnostics = _mapping(
        scaling.get("diagnostics"), name="exact scaling diagnostics"
    )
    permutations = (
        exact_assignment,
        scaling_zero,
        scaling_one,
        target_order,
        reference_order,
    )
    if (
        cost.shape != (count, count)
        or not np.issubdtype(cost.dtype, np.floating)
        or not np.all(np.isfinite(cost))
        or np.any(cost < 0.0)
        or reference_prediction.shape != (count,)
        or prediction_zero.shape != (count,)
        or scaling_prediction_zero.shape != (count,)
        or scaling_prediction_one.shape != (count,)
        or any(not _is_permutation(value, count) for value in permutations)
        or exact.get("cost_matrix_sha256") != _array_sha256(cost)
        or exact.get("reference_prediction_sha256")
        != _array_sha256(reference_prediction)
        or exact.get("prediction_world_zero_sha256") != _array_sha256(prediction_zero)
        or exact.get("assignment_sha256") != _array_sha256(exact_assignment)
        or scaling.get("assignment_world_zero_sha256") != _array_sha256(scaling_zero)
        or scaling.get("assignment_world_one_sha256") != _array_sha256(scaling_one)
        or scaling.get("prediction_world_zero_sha256")
        != _array_sha256(scaling_prediction_zero)
        or scaling.get("prediction_world_one_sha256")
        != _array_sha256(scaling_prediction_one)
        or scaling.get("target_canonical_order_sha256") != _array_sha256(target_order)
        or scaling.get("reference_canonical_order_sha256")
        != _array_sha256(reference_order)
        or scaling_diagnostics.get("cost_matrix_sha256") != _array_sha256(cost)
        or scaling_diagnostics.get("reference_prediction_sha256")
        != _array_sha256(reference_prediction)
    ):
        raise IntegrityError("exact diagnostic arrays or hashes are inconsistent")

    row_index = np.arange(count, dtype=np.int64)
    reconstructed_scaling_zero = reference_prediction[scaling_zero]
    reconstructed_scaling_one = reference_prediction[scaling_one]
    exact_assignment_cost = float(
        cost[row_index, exact_assignment].sum(dtype=np.float64)
    )
    exact_disagreement = float(
        np.mean(prediction_zero != reference_prediction[exact_assignment])
    )
    scaling_baseline = float(cost[row_index, scaling_zero].sum(dtype=np.float64))
    scaling_primal_cost = float(cost[row_index, scaling_one].sum(dtype=np.float64))
    scaling_disagreement = float(
        np.mean(scaling_prediction_zero != scaling_prediction_one)
    )
    exact_allowed = float(exact.get("allowed_cost", float("nan")))
    if (
        not np.array_equal(prediction_zero, reconstructed_scaling_zero)
        or not np.array_equal(scaling_prediction_zero, reconstructed_scaling_zero)
        or not np.array_equal(scaling_prediction_one, reconstructed_scaling_one)
        or not _numerically_equal(
            exact_assignment_cost,
            float(exact.get("assignment_cost", float("nan"))),
        )
        or not _numerically_equal(
            exact_disagreement,
            float(exact.get("maximum_disagreement_exact", float("nan"))),
        )
        or not _numerically_equal(
            scaling_baseline, float(scaling.get("baseline_cost", float("nan")))
        )
        or not _numerically_equal(
            scaling_primal_cost, float(scaling.get("primal_cost", float("nan")))
        )
        or not _numerically_equal(
            scaling_disagreement,
            float(scaling.get("selected_world_disagreement", float("nan"))),
        )
        or exact_assignment_cost > exact_allowed + 1e-12 * max(abs(exact_allowed), 1.0)
    ):
        raise IntegrityError("exact diagnostic objectives fail reconstruction")
    if rerun_exact_solver:
        _, independently_solved = solve_exact_budgeted_disagreement_milp(
            cost,
            reference_prediction,
            prediction_zero,
            allowed_cost=exact_allowed,
            time_limit_seconds=900.0,
        )
        if (
            independently_solved.get("status") != "PROVEN_OPTIMAL"
            or float(independently_solved.get("reported_mip_gap", float("inf"))) > 1e-12
            or not _numerically_equal(
                float(
                    independently_solved.get("maximum_disagreement_exact", float("nan"))
                ),
                exact_disagreement,
            )
        ):
            raise IntegrityError(
                "stored exact assignment is not independently proven optimal"
            )


def _validate_recomputed_certificate(
    stored: Mapping[str, Any], recomputed: Any, *, context: str
) -> None:
    if recomputed.summary() != dict(stored):
        raise IntegrityError(f"{context} certificate and LAP trace do not reconstruct")


def _validate_group_against_frozen_inputs(
    store: Any,
    selection: Any,
    config: Mapping[str, Any],
    scientific: Mapping[str, Any],
    loaded_arrays: Mapping[str, np.ndarray],
) -> None:
    """Independently reconstruct all costs, LAP traces, and exact diagnostics."""

    common = _mapping(config.get("common_protocol"), name="common protocol")
    calibration = int(common["calibration_rows"])
    reference_endpoint = store.endpoint("reference")
    target_endpoint = store.endpoint("target")
    maximum = materialize_prefix(
        selection,
        reference_features=reference_endpoint.features,
        target_features=target_endpoint.features,
        witness_rows=8192,
    )
    head = _mapping(store.manifest.get("head"), name="frozen head")
    weight = np.asarray(store.array(str(head["weight_role"])), dtype=np.float64)
    bias = np.asarray(store.array(str(head["bias_role"])), dtype=np.float64)
    base_protocol = _base_protocol(config)
    geometry = fit_certificate_geometry(
        maximum["target_features"][:calibration],
        maximum["reference_features"][:calibration],
        weight,
        bias,
        protocol=base_protocol,
        target_calibration_ids=maximum["target_ids"][:calibration],
        reference_calibration_ids=maximum["reference_ids"][:calibration],
    )
    if dict(geometry.diagnostics) != dict(scientific.get("geometry", {})):
        raise IntegrityError("frozen geometry does not reconstruct")

    by_size = {int(row["witness_rows"]): row for row in scientific["certificates"]}
    for size in WITNESS_SIZES:
        certificate = by_size[size]
        prefix = selection.prefix(size)
        witness_target = np.asarray(
            target_endpoint.features[prefix.target_witness_rows], dtype=np.float64
        )
        witness_reference = np.asarray(
            reference_endpoint.features[prefix.reference_witness_rows], dtype=np.float64
        )
        target_ids = np.asarray(prefix.target_ids[calibration:])
        reference_ids = np.asarray(prefix.reference_ids[calibration:])
        problem = prepare_certificate_problem(
            geometry,
            witness_target,
            witness_reference,
            target_sample_ids=target_ids,
            reference_sample_ids=reference_ids,
        )
        moments = uniform_random_cost_moments_blockwise(
            problem.cost,
            block_rows=int(common["random_cost_moment_block_rows"]),
        )
        if moments != certificate.get("random_bijection_cost"):
            raise IntegrityError("random-cost moments do not reconstruct")
        relative = float(certificate["derived_cost_budget_relative_to_minimum"])
        unit_protocol = replace(base_protocol, cost_budget_relative=relative)
        recomputed = construct_scaling_certificate_from_problem(
            problem, protocol=unit_protocol
        )
        _validate_recomputed_certificate(
            _mapping(certificate.get("result"), name="stored full certificate"),
            recomputed,
            context="full scaling",
        )

        exact_geometry = replace(
            geometry,
            protocol=unit_protocol,
            diagnostics={
                **dict(geometry.diagnostics),
                "protocol": unit_protocol.as_dict(),
            },
        )
        canonical_target, target_order, _, _ = _unique_canonical_rows(
            witness_target,
            name="scaling_exact_validation_target_features",
            salt=b"witness-scaling-exact-target-v1",
            sample_ids=target_ids,
        )
        canonical_reference, reference_order, _, _ = _unique_canonical_rows(
            witness_reference,
            name="scaling_exact_validation_reference_features",
            salt=b"witness-scaling-exact-reference-v1",
            sample_ids=reference_ids,
        )
        exact_problem = prepare_certificate_problem(
            exact_geometry,
            canonical_target[:64],
            canonical_reference[:64],
            target_sample_ids=target_ids[target_order][:64],
            reference_sample_ids=reference_ids[reference_order][:64],
        )
        key = f"n_{size:05d}"
        expected_exact_arrays = {
            "cost": exact_problem.cost,
            "reference_prediction": exact_problem.reference_prediction,
            "prediction_world_zero": exact_problem.prediction_world_zero,
        }
        if any(
            not np.array_equal(loaded_arrays[f"{key}__exact_subset_{name}"], expected)
            for name, expected in expected_exact_arrays.items()
        ):
            raise IntegrityError("exact subset problem does not reconstruct")
        diagnostic_protocol = replace(unit_protocol, primal_dual_gap_target=0.0)
        reconstructed_scaling = construct_scaling_certificate_from_problem(
            exact_problem,
            protocol=diagnostic_protocol,
            require_gap_target=False,
        )
        exact = _mapping(
            certificate.get("exact_small_instance_validation"),
            name="exact diagnostic",
        )
        _validate_recomputed_certificate(
            _mapping(
                exact.get("scaling_certificate"),
                name="stored exact scaling certificate",
            ),
            reconstructed_scaling,
            context="exact subset scaling",
        )
        _validate_exact_diagnostic_arrays(
            exact,
            loaded_arrays,
            key=key,
            rerun_exact_solver=True,
        )
        del problem, exact_problem, recomputed, reconstructed_scaling
        del witness_target, witness_reference
        gc.collect()


def _validate_group(
    lock_path: Path,
    spec: GroupSpec,
    path: Path,
    *,
    recompute_scientific_inputs: bool = False,
    loaded_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = path / "GROUP_MANIFEST.json"
    if not path.is_dir() or path.is_symlink() or not manifest_path.is_file():
        raise IntegrityError(f"scaling group is incomplete: {path}")
    manifest = dict(_mapping(read_json(manifest_path), name="group manifest"))
    verify_manifest_checksum(manifest)
    scientific = _mapping(manifest.get("scientific"), name="scientific payload")
    certificates = scientific.get("certificates")
    if (
        manifest.get("schema_version") != GROUP_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("scientific_result") is not True
        or manifest.get("lock_file_sha256") != sha256_file(lock_path)
        or scientific.get("group_id") != spec.group_id
        or scientific.get("geometry_fit_count") != 1
        or scientific.get("geometry_reused_across_sizes") is not True
        or scientific.get(
            "label_values_sliced_or_interpreted_only_after_all_three_"
            "certificates_frozen"
        )
        is not True
        or scientific.get("historical_1024_used_as_scaling_baseline") is not False
        or not isinstance(certificates, list)
        or [int(row.get("witness_rows", -1)) for row in certificates]
        != list(WITNESS_SIZES)
    ):
        raise IntegrityError(f"invalid scaling group: {spec.group_id}")
    prelabel_record = _mapping(
        manifest.get("prelabel_certificate_manifest"),
        name="pre-label certificate manifest record",
    )
    prelabel_path = verify_file_record(path, prelabel_record)
    prelabel = dict(_mapping(read_json(prelabel_path), name="pre-label layer"))
    verify_manifest_checksum(prelabel)
    prelabel_scientific = _mapping(
        prelabel.get("scientific"), name="pre-label scientific payload"
    )
    locked_header = _mapping(read_json(lock_path), name="locked header")
    verify_manifest_checksum(locked_header)
    if (
        prelabel.get("schema_version")
        != "marginal_alignment.witness_size_scaling_prelabel_layer.v1"
        or prelabel.get("status") != "PASS"
        or prelabel.get("scientific_result") is not True
        or prelabel.get("label_values_sliced_or_interpreted") is not False
        or prelabel.get("lock_manifest_sha256") != locked_header.get("manifest_sha256")
        or scientific.get("prelabel_certificate_manifest_sha256")
        != prelabel.get("manifest_sha256")
        or scientific.get("prelabel_certificate_manifest_file_sha256")
        != sha256_file(prelabel_path)
        or prelabel_scientific.get("group_id") != spec.group_id
        or prelabel_scientific.get("selection_manifest_sha256")
        != scientific.get("selection_manifest_sha256")
    ):
        raise IntegrityError("pre-label certificate layer binding failed")
    prelabel_certificates = prelabel_scientific.get("certificates")
    if (
        not isinstance(prelabel_certificates, list)
        or [
            {key: value for key, value in row.items() if key != "class_coverage"}
            for row in certificates
        ]
        != prelabel_certificates
    ):
        raise IntegrityError("post-label layer changed a frozen certificate")

    arrays = _mapping(manifest.get("arrays"), name="group arrays")
    if arrays != prelabel.get("arrays") or set(arrays) != _expected_arrays():
        raise IntegrityError("scaling group array inventory or layer binding changed")
    loaded: dict[str, np.ndarray] = {}
    for name, record in arrays.items():
        array_path = verify_file_record(path, _mapping(record, name="array record"))
        loaded[name] = np.load(array_path, allow_pickle=False)

    for row in certificates:
        size = int(row.get("witness_rows", -1))
        _validate_certificate_numbers(row, size=size)
        result = _mapping(row.get("result"), name="certificate result")
        exact = _mapping(
            row.get("exact_small_instance_validation"), name="exact diagnostic"
        )
        coverage = _mapping(row.get("class_coverage"), name="class coverage")
        cross = _mapping(coverage.get("cross_endpoint"), name="cross coverage")
        result_diagnostics = _mapping(
            result.get("diagnostics"), name="certificate diagnostics"
        )
        lower_bound = float(
            result.get("maximum_disagreement_primal_lower_bound", float("nan"))
        )
        upper_bound = float(
            result.get("maximum_disagreement_dual_upper_bound", float("nan"))
        )
        if (
            size not in WITNESS_SIZES
            or float(row.get("shared_random_gap_q", -1.0)) != 0.05
            or result.get("status") != "PASS"
            or result.get("meets_gap_target") is not True
            or float(result.get("primal_dual_gap", float("inf"))) > 0.01
            or not np.isfinite(lower_bound)
            or not np.isfinite(upper_bound)
            or not 0.0 <= lower_bound <= upper_bound <= 1.0
            or int(result_diagnostics.get("witness_rows", -1)) != size
            or exact.get("status") != "PROVEN_OPTIMAL"
            or int(exact.get("rows", -1)) != 64
            or float(exact.get("reported_mip_gap", float("inf"))) > 1e-12
            or exact.get("certificate_contains_exact_optimum") is not True
            or coverage.get("phase") != "after_all_three_certificates_frozen"
            or int(
                _mapping(coverage.get("reference"), name="reference coverage").get(
                    "rows", -1
                )
            )
            != size
            or int(
                _mapping(coverage.get("target"), name="target coverage").get("rows", -1)
            )
            != size
            or int(cross.get("rows_per_endpoint", -1)) != size
            or "target_fraction_with_at_least_one_reference_same_class_partner"
            not in cross
            or "maximum_class_preserving_matching_rows" not in cross
            or "maximum_class_preserving_matching_fraction" not in cross
        ):
            raise IntegrityError("certificate or class-coverage gate failed")
        key = f"n_{size:05d}"
        zero = loaded[f"{key}__assignment_world_zero"]
        one = loaded[f"{key}__assignment_world_one"]
        prediction_zero = loaded[f"{key}__prediction_world_zero"]
        prediction_one = loaded[f"{key}__prediction_world_one"]
        target_order = loaded[f"{key}__target_canonical_order"]
        reference_order = loaded[f"{key}__reference_canonical_order"]
        permutations = (zero, one, target_order, reference_order)
        if any(
            value.shape != (size,)
            or not np.issubdtype(value.dtype, np.integer)
            or not np.array_equal(np.sort(value), np.arange(size))
            for value in permutations
        ):
            raise IntegrityError("serialized full-witness permutation is invalid")
        reconstructed_reference_prediction = np.empty_like(prediction_zero)
        reconstructed_reference_prediction[zero] = prediction_zero
        if (
            prediction_zero.shape != (size,)
            or prediction_one.shape != (size,)
            or result.get("assignment_world_zero_sha256") != _array_sha256(zero)
            or result.get("assignment_world_one_sha256") != _array_sha256(one)
            or result.get("prediction_world_zero_sha256")
            != _array_sha256(prediction_zero)
            or result.get("prediction_world_one_sha256")
            != _array_sha256(prediction_one)
            or result.get("target_canonical_order_sha256")
            != _array_sha256(target_order)
            or result.get("reference_canonical_order_sha256")
            != _array_sha256(reference_order)
            or result_diagnostics.get("reference_prediction_sha256")
            != _array_sha256(reconstructed_reference_prediction)
            or not np.array_equal(
                prediction_one, reconstructed_reference_prediction[one]
            )
            or abs(
                float(result.get("selected_world_disagreement", -1.0))
                - float(np.mean(prediction_zero != prediction_one))
            )
            > 1e-12
        ):
            raise IntegrityError("serialized certificate arrays fail semantic binding")
        _validate_exact_diagnostic_arrays(
            exact, loaded, key=key, rerun_exact_solver=False
        )

    if recompute_scientific_inputs:
        lock = _load_lock(lock_path) if loaded_lock is None else loaded_lock
        protocol = WitnessSizeProtocol(3840, WITNESS_SIZES)
        store, _ = _load_group_store(lock, spec)
        selection = _build_selection(store, spec, protocol)
        validate_nested_selection(selection)
        locked_selection = _mapping(
            lock["nested_selections"][_selection_key(spec)],
            name="locked selection",
        )
        if dict(selection.manifest) != dict(locked_selection):
            raise IntegrityError("live nested selection differs from frozen lock")
        _validate_group_against_frozen_inputs(
            store, selection, lock["config"], scientific, loaded
        )
        reference_classes, target_classes, class_source = _endpoint_class_arrays(
            store, spec
        )
        by_size = {int(row["witness_rows"]): row for row in certificates}
        for size in WITNESS_SIZES:
            prefix = selection.prefix(size)
            observed = {
                "phase": "after_all_three_certificates_frozen",
                "source": class_source,
                "reference": class_coverage(
                    reference_classes[prefix.reference_witness_rows]
                ),
                "target": class_coverage(target_classes[prefix.target_witness_rows]),
                "cross_endpoint": cross_endpoint_class_coverage(
                    target_classes[prefix.target_witness_rows],
                    reference_classes[prefix.reference_witness_rows],
                ),
            }
            if observed != by_size[size]["class_coverage"]:
                raise IntegrityError("post-freeze class coverage reconstruction failed")
    return manifest


def _write_group(lock_path: Path, spec: GroupSpec) -> Path:
    destination = _group_path(lock_path.parent, spec)
    if destination.exists() or destination.is_symlink():
        _validate_group(lock_path, spec, destination)
        return destination / "GROUP_MANIFEST.json"
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale temporary group exists: {temporary}")
    temporary.mkdir(parents=True)
    started = time.perf_counter_ns()
    scientific, records, prelabel_record = _compute_group(
        _load_lock(lock_path), spec, staging_root=temporary
    )
    payload = {
        "schema_version": GROUP_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "lock_file_sha256": sha256_file(lock_path),
        "scientific": scientific,
        "arrays": records,
        "prelabel_certificate_manifest": prelabel_record,
        "compute": {
            "elapsed_seconds_before_manifest_write": (time.perf_counter_ns() - started)
            / 1e9,
            "maximum_resident_set_size_raw": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "runtime": _runtime_contract(),
        },
    }
    write_json_atomic(temporary / "GROUP_MANIFEST.json", seal_manifest(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return destination / "GROUP_MANIFEST.json"


def run_shard(lock_path: str | Path, *, shard_index: int, shard_count: int) -> Path:
    lock_path = Path(lock_path).resolve(strict=True)
    lock = _load_lock(lock_path)
    shards = shard_groups(enumerate_groups(lock["config"]), shard_count)
    if not 0 <= int(shard_index) < len(shards):
        raise ValueError("invalid shard index")
    rows = []
    for spec in shards[int(shard_index)]:
        path = _write_group(lock_path, spec)
        rows.append(
            {
                "group_id": spec.group_id,
                "manifest_file_sha256": sha256_file(path),
            }
        )
        print(
            f"Completed groups: {sum(1 for _ in lock_path.parent.glob('groups/*/GROUP_MANIFEST.json'))}/96",
            flush=True,
        )
    output = lock_path.parent / "run_shards" / f"shard_{int(shard_index):03d}.json"
    write_json_atomic(
        output,
        seal_manifest(
            {
                "schema_version": "marginal_alignment.witness_size_scaling_run_shard.v1",
                "status": "PASS",
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "groups": rows,
            }
        ),
    )
    return output


def verify_shard(lock_path: str | Path, *, shard_index: int, shard_count: int) -> Path:
    lock_path = Path(lock_path).resolve(strict=True)
    lock = _load_lock(lock_path)
    shards = shard_groups(enumerate_groups(lock["config"]), shard_count)
    if not 0 <= int(shard_index) < len(shards):
        raise ValueError("invalid shard index")
    rows = []
    protocol = WitnessSizeProtocol(3840, WITNESS_SIZES)
    for spec in shards[int(shard_index)]:
        path = _group_path(lock_path.parent, spec)
        manifest = _validate_group(
            lock_path,
            spec,
            path,
            recompute_scientific_inputs=True,
            loaded_lock=lock,
        )
        store, _ = _load_group_store(lock, spec)
        selection = _build_selection(store, spec, protocol)
        if dict(selection.manifest) != dict(
            lock["nested_selections"][_selection_key(spec)]
        ):
            raise IntegrityError("verification selection reconstruction failed")
        rows.append(
            {
                "group_id": spec.group_id,
                "manifest_file_sha256": sha256_file(path / "GROUP_MANIFEST.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "selection_reconstructed": True,
            }
        )
        print(
            f"Verified groups in shard {int(shard_index)}: "
            f"{len(rows)}/{len(shards[int(shard_index)])}",
            flush=True,
        )
    output = lock_path.parent / "verification" / f"shard_{int(shard_index):03d}.json"
    write_json_atomic(
        output,
        seal_manifest(
            {
                "schema_version": VERIFY_SCHEMA,
                "status": "PASS",
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "groups": rows,
                "optimizer_rerun_against_frozen_inputs": True,
                "exact_milp_rerun_from_sealed_problem": True,
            }
        ),
    )
    return output


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def interval_certified_direction(
    *, left_lower: float, left_upper: float, right_lower: float, right_upper: float
) -> tuple[float, float, str]:
    """Classify only changes identified by two numerical brackets."""

    values = np.asarray(
        [left_lower, left_upper, right_lower, right_upper], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(values))
        or left_lower > left_upper
        or right_lower > right_upper
    ):
        raise ValueError("invalid numerical identification brackets")
    lower = float(right_lower - left_upper)
    upper = float(right_upper - left_lower)
    direction = (
        "certified_decrease"
        if upper < 0.0
        else "certified_increase" if lower > 0.0 else "inconclusive"
    )
    return lower, upper, direction


def _derive_aggregate_tables(
    specs: Sequence[GroupSpec],
    manifests_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive every exported table solely from authenticated group manifests."""

    certificate_rows: list[dict[str, Any]] = []
    for spec in specs:
        manifest = _mapping(
            manifests_by_id.get(spec.group_id), name="authenticated group manifest"
        )
        scientific = _mapping(manifest.get("scientific"), name="group scientific")
        for certificate in scientific["certificates"]:
            result = certificate["result"]
            coverage = certificate["class_coverage"]
            certificate_rows.append(
                {
                    "group_id": spec.group_id,
                    "panel": spec.panel,
                    "dataset": spec.dataset,
                    "corruption": spec.corruption,
                    "encoder": spec.encoder,
                    "split_seed": spec.seed,
                    "witness_rows": certificate["witness_rows"],
                    "shared_random_gap_q": 0.05,
                    "lower_disagreement": result[
                        "maximum_disagreement_primal_lower_bound"
                    ],
                    "upper_disagreement": result[
                        "maximum_disagreement_dual_upper_bound"
                    ],
                    "gap": result["primal_dual_gap"],
                    "target_unique_classes": coverage["target"]["unique_classes"],
                    "reference_unique_classes": coverage["reference"]["unique_classes"],
                    "common_classes": coverage["cross_endpoint"]["common_classes"],
                    "target_same_class_partner_fraction": coverage["cross_endpoint"][
                        "target_fraction_with_at_least_one_reference_same_class_partner"
                    ],
                    "same_class_candidate_pair_fraction": coverage["cross_endpoint"][
                        "same_class_candidate_pair_fraction"
                    ],
                    "maximum_class_preserving_matching_fraction": coverage[
                        "cross_endpoint"
                    ]["maximum_class_preserving_matching_fraction"],
                }
            )
    certificate_rows.sort(key=lambda row: (row["group_id"], row["witness_rows"]))
    if len(certificate_rows) != 288:
        raise IntegrityError("aggregate requires all 288 fresh certificates")

    summaries: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in certificate_rows:
        buckets[(row["panel"], row["encoder"], row["witness_rows"])].append(row)
    for (panel, encoder, size), values in sorted(buckets.items()):
        summaries.append(
            {
                "panel": panel,
                "encoder": encoder,
                "witness_rows": size,
                "cells": len(values),
                "lower_disagreement_mean": float(
                    np.mean([row["lower_disagreement"] for row in values])
                ),
                "lower_disagreement_minimum": float(
                    np.min([row["lower_disagreement"] for row in values])
                ),
                "lower_disagreement_maximum": float(
                    np.max([row["lower_disagreement"] for row in values])
                ),
                "upper_disagreement_mean": float(
                    np.mean([row["upper_disagreement"] for row in values])
                ),
                "target_unique_classes_mean": float(
                    np.mean([row["target_unique_classes"] for row in values])
                ),
                "same_class_partner_fraction_mean": float(
                    np.mean(
                        [row["target_same_class_partner_fraction"] for row in values]
                    )
                ),
                "maximum_class_preserving_matching_fraction_mean": float(
                    np.mean(
                        [
                            row["maximum_class_preserving_matching_fraction"]
                            for row in values
                        ]
                    )
                ),
            }
        )
    index = {(row["group_id"], row["witness_rows"]): row for row in certificate_rows}
    deltas: list[dict[str, Any]] = []
    for spec in specs:
        for left, right in ((1024, 4096), (1024, 8192), (4096, 8192)):
            a = index[(spec.group_id, left)]
            b = index[(spec.group_id, right)]
            difference_lower, difference_upper, direction = (
                interval_certified_direction(
                    left_lower=a["lower_disagreement"],
                    left_upper=a["upper_disagreement"],
                    right_lower=b["lower_disagreement"],
                    right_upper=b["upper_disagreement"],
                )
            )
            deltas.append(
                {
                    "group_id": spec.group_id,
                    "panel": spec.panel,
                    "corruption": spec.corruption,
                    "encoder": spec.encoder,
                    "split_seed": spec.seed,
                    "left_witness_rows": left,
                    "right_witness_rows": right,
                    "selected_lower_difference_right_minus_left": (
                        b["lower_disagreement"] - a["lower_disagreement"]
                    ),
                    "numerical_bracket_difference_lower_bound": difference_lower,
                    "numerical_bracket_difference_upper_bound": difference_upper,
                    "interval_certified_direction": direction,
                    "direction_rule": (
                        "decrease_iff_upper_below_zero;increase_iff_lower_above_"
                        "zero;otherwise_inconclusive"
                    ),
                    "target_unique_class_difference": (
                        b["target_unique_classes"] - a["target_unique_classes"]
                    ),
                    "same_class_partner_fraction_difference": (
                        b["target_same_class_partner_fraction"]
                        - a["target_same_class_partner_fraction"]
                    ),
                }
            )
    return certificate_rows, deltas, summaries


def _validate_aggregate_table_contents(
    observed: Mapping[str, Sequence[Mapping[str, Any]]],
    expected: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if set(observed) != set(expected) or any(
        list(observed[name]) != list(expected[name]) for name in expected
    ):
        raise IntegrityError(
            "aggregate table content differs from authenticated group manifests"
        )


def aggregate(lock_path: str | Path, *, verification_shards: int) -> Path:
    lock_path = Path(lock_path).resolve(strict=True)
    lock = _load_lock(lock_path)
    specs = enumerate_groups(lock["config"])
    shards = shard_groups(specs, verification_shards)
    verified: dict[str, tuple[str, str]] = {}
    for index, expected in enumerate(shards):
        path = lock_path.parent / "verification" / f"shard_{index:03d}.json"
        report = _mapping(read_json(path), name="verification report")
        verify_manifest_checksum(report)
        if (
            report.get("schema_version") != VERIFY_SCHEMA
            or report.get("status") != "PASS"
            or report.get("shard_index") != index
            or report.get("shard_count") != verification_shards
            or report.get("optimizer_rerun_against_frozen_inputs") is not True
            or report.get("exact_milp_rerun_from_sealed_problem") is not True
            or [row["group_id"] for row in report["groups"]]
            != [item.group_id for item in expected]
            or any(
                row.get("selection_reconstructed") is not True
                for row in report["groups"]
            )
        ):
            raise IntegrityError("verification coverage changed")
        for row in report["groups"]:
            verified[row["group_id"]] = (
                row["manifest_file_sha256"],
                row["manifest_sha256"],
            )
    group_records = []
    manifests_by_id: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        path = _group_path(lock_path.parent, spec)
        manifest = _validate_group(
            lock_path,
            spec,
            path,
            recompute_scientific_inputs=False,
            loaded_lock=lock,
        )
        binding = (
            sha256_file(path / "GROUP_MANIFEST.json"),
            manifest["manifest_sha256"],
        )
        if verified.get(spec.group_id) != binding:
            raise IntegrityError("group changed after verification")
        manifests_by_id[spec.group_id] = manifest
        group_records.append(
            {
                "group_id": spec.group_id,
                "manifest_file_sha256": binding[0],
                "manifest_sha256": binding[1],
            }
        )
    certificate_rows, deltas, summaries = _derive_aggregate_tables(
        specs, manifests_by_id
    )
    _write_jsonl(lock_path.parent / "WITNESS_SIZE_ROWS.jsonl", certificate_rows)
    _write_jsonl(lock_path.parent / "WITNESS_SIZE_DELTAS.jsonl", deltas)
    _write_jsonl(lock_path.parent / "WITNESS_SIZE_SUMMARIES.jsonl", summaries)
    payload = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "post_hoc_reviewer_motivated_supportive",
        "lock_file_sha256": sha256_file(lock_path),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "groups": len(group_records),
        "fresh_certificates": len(certificate_rows),
        "all_gaps_at_most_0_01": all(row["gap"] <= 0.01 for row in certificate_rows),
        "nested_selection_reconstructed": True,
        "label_values_sliced_or_interpreted_postfreeze_only": True,
        "historical_1024_comparison": {
            "status": "OMITTED",
            "reason": "historical_endpoint_sets_are_not_the_new_maximum_selection_prefixes",
            "historical_results_used_as_scaling_baseline": False,
        },
        "fixed_panel_population_inference_claimed": False,
        "artifacts": {
            "rows_sha256": sha256_file(lock_path.parent / "WITNESS_SIZE_ROWS.jsonl"),
            "deltas_sha256": sha256_file(
                lock_path.parent / "WITNESS_SIZE_DELTAS.jsonl"
            ),
            "summaries_sha256": sha256_file(
                lock_path.parent / "WITNESS_SIZE_SUMMARIES.jsonl"
            ),
        },
        "group_manifests": group_records,
    }
    output = lock_path.parent / "WITNESS_SIZE_SCALING_AGGREGATE.json"
    write_json_atomic(output, seal_manifest(payload))
    done = lock_path.parent / "DONE.json"
    write_json_atomic(
        done,
        seal_manifest(
            {
                "schema_version": "marginal_alignment.witness_size_scaling_done.v1",
                "status": "PASS",
                "aggregate_file_sha256": sha256_file(output),
                "aggregate_manifest_sha256": read_json(output)["manifest_sha256"],
            }
        ),
    )
    write_json_atomic(
        lock_path.parent / "STATE.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": "COMPLETE",
            "completed_groups": 96,
            "expected_groups": 96,
            "done_file_sha256": sha256_file(done),
        },
    )
    return output


def verify_complete(lock_path: str | Path) -> Path:
    """Authenticate a stable completed campaign without changing it."""

    lock_path = Path(lock_path).resolve(strict=True)
    lock = _load_lock(lock_path)
    root = lock_path.parent
    aggregate_path = root / "WITNESS_SIZE_SCALING_AGGREGATE.json"
    done_path = root / "DONE.json"
    aggregate_manifest = dict(
        _mapping(read_json(aggregate_path), name="scaling aggregate")
    )
    done = dict(_mapping(read_json(done_path), name="scaling done marker"))
    verify_manifest_checksum(aggregate_manifest)
    verify_manifest_checksum(done)
    if (
        aggregate_manifest.get("schema_version") != AGGREGATE_SCHEMA
        or aggregate_manifest.get("status") != "PASS"
        or aggregate_manifest.get("groups") != 96
        or aggregate_manifest.get("fresh_certificates") != 288
        or aggregate_manifest.get("all_gaps_at_most_0_01") is not True
        or aggregate_manifest.get("lock_file_sha256") != sha256_file(lock_path)
        or done.get("status") != "PASS"
        or done.get("schema_version")
        != "marginal_alignment.witness_size_scaling_done.v1"
        or done.get("aggregate_file_sha256") != sha256_file(aggregate_path)
        or done.get("aggregate_manifest_sha256")
        != aggregate_manifest.get("manifest_sha256")
    ):
        raise IntegrityError("completed scaling marker or aggregate is invalid")
    artifacts = _mapping(
        aggregate_manifest.get("artifacts"), name="aggregate artifacts"
    )
    expected_artifacts = {
        "rows_sha256": root / "WITNESS_SIZE_ROWS.jsonl",
        "deltas_sha256": root / "WITNESS_SIZE_DELTAS.jsonl",
        "summaries_sha256": root / "WITNESS_SIZE_SUMMARIES.jsonl",
    }
    if set(artifacts) != set(expected_artifacts):
        raise IntegrityError("aggregate artifact inventory changed")
    for key, path in expected_artifacts.items():
        if artifacts[key] != sha256_file(path):
            raise IntegrityError(f"aggregate artifact changed: {path}")
    expected_jsonl_counts = {
        "WITNESS_SIZE_ROWS.jsonl": 288,
        "WITNESS_SIZE_DELTAS.jsonl": 288,
        "WITNESS_SIZE_SUMMARIES.jsonl": 12,
    }
    parsed_jsonl: dict[str, list[Mapping[str, Any]]] = {}
    for filename, count in expected_jsonl_counts.items():
        rows = []
        for line in (root / filename).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(_mapping(json.loads(line), name=f"{filename} row"))
        if len(rows) != count:
            raise IntegrityError(f"{filename} row count changed")
        parsed_jsonl[filename] = rows
    row_keys = {
        (row.get("group_id"), int(row.get("witness_rows", -1)))
        for row in parsed_jsonl["WITNESS_SIZE_ROWS.jsonl"]
    }
    if len(row_keys) != 288 or {size for _, size in row_keys} != set(WITNESS_SIZES):
        raise IntegrityError("certificate row coverage changed")
    if any(
        row.get("interval_certified_direction")
        not in {"certified_decrease", "certified_increase", "inconclusive"}
        for row in parsed_jsonl["WITNESS_SIZE_DELTAS.jsonl"]
    ):
        raise IntegrityError("delta direction classification changed")
    expected_groups = []
    manifests_by_id: dict[str, Mapping[str, Any]] = {}
    complete_specs = enumerate_groups(lock["config"])
    for completed_index, spec in enumerate(complete_specs, start=1):
        path = _group_path(root, spec)
        manifest = _validate_group(
            lock_path,
            spec,
            path,
            recompute_scientific_inputs=True,
            loaded_lock=lock,
        )
        expected_groups.append(
            {
                "group_id": spec.group_id,
                "manifest_file_sha256": sha256_file(path / "GROUP_MANIFEST.json"),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
        manifests_by_id[spec.group_id] = manifest
        print(
            f"Final-authenticated groups: {completed_index}/{len(complete_specs)}",
            flush=True,
        )
    if aggregate_manifest.get("group_manifests") != expected_groups:
        raise IntegrityError("aggregate group bindings changed")
    expected_rows, expected_deltas, expected_summaries = _derive_aggregate_tables(
        enumerate_groups(lock["config"]), manifests_by_id
    )
    _validate_aggregate_table_contents(
        parsed_jsonl,
        {
            "WITNESS_SIZE_ROWS.jsonl": expected_rows,
            "WITNESS_SIZE_DELTAS.jsonl": expected_deltas,
            "WITNESS_SIZE_SUMMARIES.jsonl": expected_summaries,
        },
    )
    expected_group_by_id = {row["group_id"]: row for row in expected_groups}
    for index, expected_specs in enumerate(
        shard_groups(enumerate_groups(lock["config"]), 2)
    ):
        report_path = root / "verification" / f"shard_{index:03d}.json"
        report = _mapping(read_json(report_path), name="verification report")
        verify_manifest_checksum(report)
        if (
            report.get("schema_version") != VERIFY_SCHEMA
            or report.get("status") != "PASS"
            or report.get("shard_index") != index
            or report.get("shard_count") != 2
            or report.get("optimizer_rerun_against_frozen_inputs") is not True
            or report.get("exact_milp_rerun_from_sealed_problem") is not True
            or [row.get("group_id") for row in report.get("groups", ())]
            != [spec.group_id for spec in expected_specs]
            or any(
                row.get("manifest_file_sha256")
                != expected_group_by_id[row.get("group_id")]["manifest_file_sha256"]
                or row.get("manifest_sha256")
                != expected_group_by_id[row.get("group_id")]["manifest_sha256"]
                or row.get("selection_reconstructed") is not True
                for row in report.get("groups", ())
            )
        ):
            raise IntegrityError("verification report changed")
    state = _mapping(read_json(root / "STATE.json"), name="scientific state")
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("status") != "COMPLETE"
        or state.get("completed_groups") != 96
        or state.get("expected_groups") != 96
        or state.get("done_file_sha256") != sha256_file(done_path)
    ):
        raise IntegrityError("completed scientific state changed")
    return done_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--r5-root", required=True)
    freeze.add_argument("--historical-followup-root", required=True)
    freeze.add_argument("--feature-root", required=True)
    freeze.add_argument("--external-input-manifest", required=True)
    freeze.add_argument("--class-index", required=True)
    freeze.add_argument("--output-root", required=True)
    for name in ("run-shard", "verify-shard"):
        command = commands.add_parser(name)
        command.add_argument("--lock", required=True)
        command.add_argument("--shard-index", type=int, required=True)
        command.add_argument("--shard-count", type=int, required=True)
    final = commands.add_parser("aggregate")
    final.add_argument("--lock", required=True)
    final.add_argument("--verification-shards", type=int, required=True)
    verify = commands.add_parser("verify-complete")
    verify.add_argument("--lock", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        output = freeze_lock(
            config_path=arguments.config,
            r5_root=arguments.r5_root,
            historical_followup_root=arguments.historical_followup_root,
            feature_root=arguments.feature_root,
            external_input_manifest=arguments.external_input_manifest,
            class_index=arguments.class_index,
            output_root=arguments.output_root,
        )
    elif arguments.command == "run-shard":
        output = run_shard(
            arguments.lock,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    elif arguments.command == "verify-shard":
        output = verify_shard(
            arguments.lock,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    elif arguments.command == "aggregate":
        output = aggregate(
            arguments.lock, verification_shards=arguments.verification_shards
        )
    else:
        output = verify_complete(arguments.lock)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
