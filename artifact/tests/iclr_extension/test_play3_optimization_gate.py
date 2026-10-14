from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)
from scripts.iclr_extension import run_play3_optimization_gate as gate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    PROJECT_ROOT / "configs" / "iclr_extension" / "play3_optimization_gate.json"
)


def test_sha256_validator_rejects_coercible_nonstring_values() -> None:
    digest = "1" * 64
    assert gate._valid_sha256(digest)
    assert not gate._valid_sha256(int(digest))
    assert not gate._valid_sha256(True)


class _FakeFeatures:
    def __init__(self, rows: int, dimension: int) -> None:
        self.shape = (int(rows), int(dimension))

    def __len__(self) -> int:
        return self.shape[0]


class _FakeStore:
    def __init__(
        self,
        *,
        dataset: str,
        encoder: str,
        reference_rows: int,
        target_rows: int,
        wnids: list[str],
        provenance_key: str,
        provenance_schema: str,
        variant: str,
        policy: str,
        encoder_contract: dict,
    ) -> None:
        reference_wnids = np.asarray([
            wnids[index % len(wnids)] for index in range(reference_rows)
        ])
        target_wnids = np.asarray([
            wnids[index % len(wnids)] for index in range(target_rows)
        ])
        intersection = {
            "policy": policy,
            "wnids": wnids,
            "wnids_sha256": sha256_json(wnids),
            "reference_endpoint_restricted": True,
            "target_endpoint_restricted": True,
            "reference_wnids_role": "reference_wnids",
            "target_wnids_role": "target_wnids",
        }
        role_hashes = {}
        for role in (
            "reference_ids",
            "head_fit_ids",
            "head_weight",
            "head_bias",
            "reference_wnids",
            "target_wnids",
        ):
            role_hashes[role] = sha256_json({
                "dataset": (
                    None
                    if role in {"head_fit_ids", "head_weight", "head_bias"}
                    else dataset
                ),
                "encoder": encoder,
                "role": role,
            })
        provenance = {
            "schema_version": provenance_schema,
            "manifest_sha256": "1" * 64,
            "file_tree_sha256": "2" * 64,
            "class_index_sha256": intersection["wnids_sha256"],
            "row_count": target_rows,
        }
        if dataset != "imagenet_sketch":
            provenance.update({
                "status": "PASS",
                "dataset": dataset,
                "variant": variant,
                "target_order_sha256": "3" * 64,
            })
        target_extraction = {
            "schema_version": "fixture.target_extraction.v1",
            "dataset": dataset,
            "encoder": encoder,
        }
        source_manifest_sha256 = "6" * 64
        model = {
            "encoder_id": encoder,
            "model_id": encoder_contract["model_id"],
            "checkpoint_sha256": "4" * 64,
            "preprocessing_sha256": "5" * 64,
            "feature_layer": encoder_contract["feature_layer"],
            "pooling": encoder_contract["pooling"],
            "head_protocol": {"kind": "fixture_frozen_head"},
            "head_fit_ids_role": "head_fit_ids",
        }
        preparation = {
            "schema_version": (
                "marginal_alignment.iclr_extension."
                "optimization_store_preparation.v1"
            ),
            "status": "PASS",
            "dataset": dataset,
            "encoder": encoder,
            "head_fit_disjoint_from_reference": True,
            "head_fit_ids_sha256": role_hashes["head_fit_ids"],
            "reference_ids_sha256": role_hashes["reference_ids"],
            "reference_wnids_sha256": role_hashes["reference_wnids"],
            "target_wnids_sha256": role_hashes["target_wnids"],
            "class_intersection_wnids_sha256": intersection["wnids_sha256"],
            "reference_source_manifest_sha256": source_manifest_sha256,
            "target_extraction_manifest_sha256": sha256_json(target_extraction),
            "dataset_provenance_manifest_sha256": provenance["manifest_sha256"],
        }
        self.dataset = dataset
        self.encoder = encoder
        self.pairing_spec = {"status": "unavailable"}
        self.records = {
            role: {"sha256": digest} for role, digest in role_hashes.items()
        }
        self.manifest = {
            "head": {
                "weight_role": "head_weight",
                "bias_role": "head_bias",
            },
            "metadata": {
                provenance_key: provenance,
                "class_intersection": intersection,
                "model_provenance": model,
                "optimization_gate_preparation": preparation,
                "endpoint_labels_not_published": True,
                "source_campaign_reference_store": {
                    "actual_manifest_sha256": source_manifest_sha256,
                },
                "target_extraction_manifest": target_extraction,
                "store_preparation_source_manifest_sha256": (
                    source_manifest_sha256
                ),
            }
        }
        dimension = int(encoder_contract["expected_dimension"])
        reference_ids = np.arange(reference_rows, dtype=np.int64)
        self._endpoints = {
            "reference": SimpleNamespace(
                features=_FakeFeatures(reference_rows, dimension),
                ids=reference_ids,
            ),
            "target": SimpleNamespace(
                features=_FakeFeatures(target_rows, dimension),
                ids=np.arange(2_000_000, 2_000_000 + target_rows),
            ),
        }
        self.endpoint_roles = {
            "reference": {"ids": "reference_ids", "labels": None},
            "target": {"ids": "target_ids", "labels": None},
        }
        self._arrays = {
            "reference_wnids": reference_wnids,
            "target_wnids": target_wnids,
            "head_fit_ids": np.arange(1_000_000, 1_001_000),
        }

    def endpoint(self, name: str):
        return self._endpoints[name]

    def array(self, role: str) -> np.ndarray:
        return self._arrays[role]


def _fake_store_map(tmp_path: Path, config: dict):
    stores = {}
    records = []
    for dataset, dataset_config in config["datasets"].items():
        count = int(dataset_config["expected_target_classes"])
        wnids = [f"n{index:08d}" for index in range(count)]
        for encoder in config["encoders"]:
            manifest = tmp_path / f"{dataset}__{encoder}.json"
            manifest.write_text("{}\n", encoding="utf-8")
            stores[manifest.resolve()] = _FakeStore(
                dataset=dataset,
                encoder=encoder,
                reference_rows=int(dataset_config[
                    "expected_reference_rows_after_intersection_and_head_fit_exclusion"
                ]),
                target_rows=int(dataset_config["expected_target_rows"]),
                wnids=wnids,
                provenance_key=str(
                    dataset_config["store_provenance_metadata_key"]
                ),
                provenance_schema=str(
                    dataset_config["expected_provenance_schema"]
                ),
                variant=str(dataset_config["variant"]),
                policy=str(dataset_config["class_intersection_policy"]),
                encoder_contract=dict(config["encoder_contracts"][encoder]),
            )
            records.append({
                "dataset": dataset,
                "encoder": encoder,
                "manifest": str(manifest),
            })
    store_map = tmp_path / "store_map.json"
    store_map.write_text(json.dumps({
        "schema_version": gate.STORE_MAP_SCHEMA,
        "stores": records,
    }), encoding="utf-8")
    return store_map, stores


def _set_locked_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in gate.REQUIRED_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _timing_store_records(config: dict) -> dict:
    return {
        gate._store_key(dataset, encoder): {
            "manifest_file_sha256": sha256_json({
                "dataset": dataset, "encoder": encoder, "kind": "file"
            }),
            "manifest_payload_sha256": sha256_json({
                "dataset": dataset, "encoder": encoder, "kind": "payload"
            }),
            "wnids": ["n00000001"],
        }
        for dataset in config["datasets"]
        for encoder in config["encoders"]
    }


def _completed_panel_fixture(
    tmp_path: Path,
    identities: tuple[tuple[str, str, str, int, int], ...],
) -> tuple[Path, Path, Path, dict, dict[str, Path]]:
    lock_path = tmp_path / "LOCK.json"
    lock_path.write_text("fixture lock\n", encoding="utf-8")
    store_map_path = tmp_path / "store_map.json"
    store_map_path.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "units"
    root.mkdir()
    lock = {"manifest_sha256": "a" * 64, "config": {}}
    array_paths: dict[str, Path] = {}
    completed = []
    for position, identity in enumerate(identities):
        identifier = gate._unit_id(identity)
        unit_root = root / identifier
        unit_root.mkdir()
        arrays = {}
        for array_position, name in enumerate(gate.ARRAY_NAMES):
            path = unit_root / f"{name}.npy"
            np.save(path, np.asarray([position * 10 + array_position]))
            arrays[name] = gate.describe_npy(path, relative_to=unit_root)
            array_paths[f"{identifier}::{name}"] = path
        manifest_path = unit_root / "UNIT_MANIFEST.json"
        write_json_atomic(manifest_path, seal_manifest({
            "unit_id": identifier,
            "arrays": arrays,
        }))
        completed.append({
            "unit_id": identifier,
            "manifest_file_sha256": gate.sha256_file(manifest_path),
            "resumed": False,
        })
    completion = root / "SHARD_000_DONE.json"
    write_json_atomic(completion, seal_manifest({
        "schema_version": gate.PANEL_SHARD_SCHEMA,
        "status": "PASS",
        "lock_file_sha256": gate.sha256_file(lock_path),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "shard_count": 1,
        "shard_index": 0,
        "expected_units": len(identities),
        "completed_units": len(identities),
        "unit_ids": [gate._unit_id(identity) for identity in identities],
        "units": completed,
        "resume_policy": (
            "hash_and_contract_validate_complete_units_without_reconstruction;_"
            "use_verify-panel-for_full_reconstruction"
        ),
    }))
    return lock_path, store_map_path, root, lock, array_paths


def _install_completed_panel_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lock: dict,
    identities: tuple[tuple[str, str, str, int, int], ...],
    calls: list[str],
) -> None:
    by_id = {gate._unit_id(identity): identity for identity in identities}

    def validate(manifest_path: Path, *, lock, lock_file_sha256):
        assert lock is not None
        assert len(lock_file_sha256) == 64
        manifest = read_json(manifest_path)
        verify_manifest_checksum(manifest)
        arrays = manifest["arrays"]
        assert set(arrays) == set(gate.ARRAY_NAMES)
        for name in gate.ARRAY_NAMES:
            gate.verify_file_record(manifest_path.parent, arrays[name])
        identifier = manifest["unit_id"]
        calls.append(identifier)
        return by_id[identifier], {
            "manifest_file_sha256": gate.sha256_file(manifest_path),
        }

    monkeypatch.setattr(gate, "_load_lock", lambda path: lock)
    monkeypatch.setattr(gate, "_locked_store_map", lambda lock, path: {})
    monkeypatch.setattr(
        gate,
        "_sharded_identities",
        lambda config, shard_count, shard_index: identities,
    )
    monkeypatch.setattr(gate, "_validate_aggregate_unit", validate)


def _fixture_compute_event(
    lock: dict,
    identity: tuple[str, str, str, int, int],
    *,
    purpose: str,
    retry: int = 0,
) -> dict:
    dataset, encoder, _, seed, _ = identity
    config = lock["config"]
    geometry_stages = []
    for geometry in config["cost_geometries"]:
        geometry_stages.append({
            "cost_geometry": geometry,
            "geometry_derivation_nanoseconds": 11,
            "dense_problem_preparation_nanoseconds": 13,
            "budgets": [{
                "cost_budget_basis_points": gate._budget_token(budget),
                "certificate_nanoseconds": 17,
                "exact_validation_nanoseconds": 19,
                "posthoc_inference_nanoseconds": 23,
            } for budget in config["certificate_protocol"][
                "cost_budgets_relative"
            ]],
        })
    stage_sum = 2 + 3 + 5 + sum(
        row["geometry_derivation_nanoseconds"]
        + row["dense_problem_preparation_nanoseconds"]
        + sum(
            budget["certificate_nanoseconds"]
            + budget["exact_validation_nanoseconds"]
            + budget["posthoc_inference_nanoseconds"]
            for budget in row["budgets"]
        )
        for row in geometry_stages
    )
    group_id = gate._group_id(dataset, encoder, seed)
    store = lock["stores"][gate._store_key(dataset, encoder)]
    body = {
        "schema_version": gate.COMPUTE_TIMING_SCHEMA,
        "purpose": purpose,
        "event_id": sha256_json({
            "group_id": group_id, "purpose": purpose, "retry": retry
        }),
        "group_id": group_id,
        "dataset": dataset,
        "encoder": encoder,
        "split_seed": seed,
        "lock_manifest_sha256": lock["manifest_sha256"],
        "store_manifest_file_sha256": store["manifest_file_sha256"],
        "store_manifest_payload_sha256": store[
            "manifest_payload_sha256"
        ],
        "clock": "time.perf_counter_ns_monotonic",
        "unit_ids_computed": [
            gate._unit_id(gate._unit_identity(
                dataset, encoder, geometry, seed, budget
            ))
            for geometry in config["cost_geometries"]
            for budget in config["certificate_protocol"][
                "cost_budgets_relative"
            ]
        ],
        "group_wall_nanoseconds": stage_sum + 29,
        "store_validation_nanoseconds": 2,
        "split_selection_nanoseconds": 3,
        "shared_calibration_and_bures_fit_nanoseconds": 5,
        "geometry_stages": geometry_stages,
        "stage_sum_nanoseconds": stage_sum,
        "unattributed_python_overhead_nanoseconds": 29,
        "peak_rss_bytes": None,
        "peak_rss_status": gate.PEAK_RSS_STATUS,
        "scope": (
            "successful_group_computation_reaching_at_least_one_sealed_unit_"
            "or_verification_report;_events_interrupted_before_publication_"
            "are_not_observable"
        ),
    }
    return {**body, "timing_payload_sha256": sha256_json(body)}


def _fixture_verification_report(
    lock: dict,
    identity: tuple[str, str, str, int, int],
    expected_record: dict,
    *,
    owner: bool,
) -> dict:
    event = _fixture_compute_event(
        lock, identity, purpose="full_reconstruction_verification"
    )
    non_group = 37
    wall = non_group + (event["group_wall_nanoseconds"] if owner else 0)
    return {
        "unit_id": gate._unit_id(identity),
        "unit_manifest_file_sha256": expected_record["manifest_file_sha256"],
        "unit_manifest_sha256": expected_record["manifest_sha256"],
        "status": "PASS",
        "assignments_recomputed": True,
        "costs_recomputed_from_endpoint_features": True,
        "rewards_recomputed_from_frozen_readout": True,
        "bijections_revalidated": True,
        "duplicate_row_admissibility_revalidated": True,
        "exact_small_instance_milp_revalidated": True,
        "posthoc_wnid_statistics_revalidated": True,
        "gap_target_met": expected_record["meets_gap_target"],
        "primal_dual_gap_percentage_points": (
            100.0 * expected_record["primal_dual_gap"]
        ),
        "verification_compute_event": event,
        "verification_compute_event_owner_for_this_report": owner,
        "verification_timing": {
            "clock": "time.perf_counter_ns_monotonic",
            "verification_wall_nanoseconds_before_report_seal": wall,
            "pre_reconstruction_artifact_and_contract_validation_nanoseconds": 7,
            "post_reconstruction_comparison_nanoseconds": 11,
            "non_group_validation_and_python_overhead_nanoseconds": non_group,
            "verification_manifest_publication_timing_nanoseconds": None,
            "peak_rss_bytes": None,
            "peak_rss_status": gate.PEAK_RSS_STATUS,
        },
    }


def test_freeze_lock_seals_code_stores_and_class_intersections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_locked_threads(monkeypatch)
    config = gate._load_config(CONFIG, require_ready=True)
    store_map, stores = _fake_store_map(tmp_path, config)
    monkeypatch.setattr(
        gate,
        "load_feature_store",
        lambda path, verify_hashes=True: stores[Path(path).resolve()],
    )
    output = tmp_path / "LOCK.json"
    gate.freeze_lock(CONFIG, store_map, output)
    lock = read_json(output)
    verify_manifest_checksum(lock)
    assert lock["status"] == "FROZEN_BEFORE_NEW_OUTCOME_COMPUTE"
    assert lock["outcomes_read_during_freeze"] is False
    assert len(lock["stores"]) == 6
    assert all("manifest" not in record for record in lock["stores"].values())
    assert all(record["wnids"] for record in lock["stores"].values())
    assert set(lock["source_files"]) == set(gate.SOURCE_FILES)
    assert lock["freeze_runtime"]["thread_environment"] == (
        gate.REQUIRED_THREAD_ENVIRONMENT
    )


def test_runtime_contract_rejects_unspecified_numerical_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in gate.REQUIRED_THREAD_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="numerical thread"):
        gate._runtime_contract()
    _set_locked_threads(monkeypatch)
    assert gate._runtime_contract()["thread_environment"] == (
        gate.REQUIRED_THREAD_ENVIRONMENT
    )


def test_store_map_rejects_incomplete_target_class_intersection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = gate._load_config(CONFIG, require_ready=True)
    store_map, stores = _fake_store_map(tmp_path, config)
    first = next(iter(stores.values()))
    first._arrays["target_wnids"][:] = first._arrays["target_wnids"][0]
    monkeypatch.setattr(
        gate,
        "load_feature_store",
        lambda path, verify_hashes=True: stores[Path(path).resolve()],
    )
    with pytest.raises(IntegrityError, match="does not cover"):
        gate._validate_store_map(store_map, config, verify_hashes=True)


def test_config_prevents_overrunning_five_thousand_row_r_reference(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["datasets"]["imagenet_r"][
        "expected_reference_rows_after_intersection_and_head_fit_exclusion"
    ] = 4800
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(IntegrityError, match="no-head-fit reference split"):
        gate._load_config(malformed, require_ready=True)


@pytest.mark.parametrize(
    ("case", "value", "message"),
    [
        ("split_seed", "201", "split seeds"),
        ("cost_budget", "0.01", "cost budgets"),
        ("witness_rows", 1024.0, "row counts"),
        ("exact_rows", True, "row counts"),
        ("binary_variables", 4096.0, "indexed-atom"),
        ("inference_seed", "20260903", "inference contract"),
        ("inference_replicates", 10000.0, "inference contract"),
        ("dataset_rows", "25000", "positive integer"),
    ],
)
def test_config_rejects_coercible_noncanonical_numeric_types(
    tmp_path: Path, case: str, value: object, message: str
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if case == "split_seed":
        config["certificate_protocol"]["split_seeds"][0] = value
    elif case == "cost_budget":
        config["certificate_protocol"]["cost_budgets_relative"][0] = value
    elif case == "witness_rows":
        config["certificate_protocol"]["witness_rows"] = value
    elif case == "exact_rows":
        config["certificate_protocol"]["exact_validation_rows"] = value
    elif case == "binary_variables":
        config["certificate_protocol"][
            "maximum_exact_milp_binary_variables"
        ] = value
    elif case == "inference_seed":
        config["inference"]["seed"] = value
    elif case == "inference_replicates":
        config["inference"]["replicates"] = value
    elif case == "dataset_rows":
        config["datasets"]["imagenet_sketch"][
            "expected_reference_rows_after_intersection_and_head_fit_exclusion"
        ] = value
    else:  # pragma: no cover - parametrization programming error
        raise AssertionError(case)
    malformed = tmp_path / f"malformed_{case}.json"
    malformed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(IntegrityError, match=message):
        gate._load_config(malformed, require_ready=True)


def test_wnid_bootstrap_pools_split_repeats_inside_class() -> None:
    rows = np.asarray([[2, 2, 0], [2, 0, 2]], dtype=np.int64)
    disagreements = np.asarray([[2, 0, 0], [0, 0, 2]], dtype=np.int64)
    weights = np.asarray([
        [1, 1, 1],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
    ], dtype=np.int64)
    summary = gate._clustered_interval(
        rows, disagreements, bootstrap_weights=weights
    )
    assert summary["point_estimate"] == pytest.approx(0.5)
    assert summary["split_seed_policy"] == (
        "sum_sufficient_statistics_within_wnid_before_cluster_resampling"
    )
    assert summary["display_label"] == (
        "descriptive conditional WNID-cluster resampling interval, "
        "assignments fixed/no optimizer refit"
    )
    assert summary["optimizer_refit_within_bootstrap"] is False
    assert summary["gate_kind"] == (
        "prespecified_robustness_screen_not_population_or_confirmatory_"
        "inference"
    )
    assert summary["declared_clusters"] == 3
    assert summary["dropped_replicates"] == 0


def test_posthoc_wnid_metadata_distinguishes_preflight_read_from_optimizer_use() -> None:
    wnids = ["n00000001", "n00000002"]
    store = SimpleNamespace(
        endpoint=lambda name: SimpleNamespace(features=_FakeFeatures(2, 1)),
        array=lambda role: np.asarray(wnids),
    )
    lock = {
        "stores": {
            gate._store_key("imagenet_v2", "dinov3"): {
                "target_wnids_role": "target_wnids",
                "reference_wnids_role": "reference_wnids",
                "wnids": wnids,
                "wnids_sha256": sha256_json(wnids),
            }
        }
    }
    result = SimpleNamespace(
        target_canonical_order=np.asarray([0, 1], dtype=np.int64),
        prediction_world_zero=np.asarray([0, 1], dtype=np.int64),
        prediction_world_one=np.asarray([1, 1], dtype=np.int64),
    )
    report = gate._posthoc_wnid_statistics(
        store,
        lock,
        result,
        dataset="imagenet_v2",
        encoder="dinov3",
        calibration_rows=0,
        target_selection=np.asarray([0, 1], dtype=np.int64),
    )
    assert report["target_wnids_preflight_integrity_validated"] is True
    assert report["target_wnids_not_used_as_certificate_optimization_inputs"] is True
    assert (
        report["target_wnids_used_for_posthoc_inference_after_assignment_fixed"]
        is True
    )
    assert "target_wnids_loaded_after_certificate_construction" not in report


def test_gate_evaluation_covers_all_cost_anchors_and_uses_wnid_intervals() -> None:
    config = gate._load_config(CONFIG, require_ready=True)
    assert tuple(config["certificate_protocol"]["cost_budgets_relative"]) == (
        0.01,
        0.05,
        0.1,
    )
    assert config["certificate_protocol"]["calibration_rows"] == 3840
    assert config["certificate_protocol"]["witness_rows"] == 1024
    assert config["certificate_protocol"]["exact_validation_rows"] == 64
    assert config["certificate_protocol"][
        "external_ground_truth_labels_used_for_optimization"
    ] is False
    assert config["certificate_protocol"][
        "frozen_readout_predictions_used_for_optimization"
    ] is True
    assert config["certificate_protocol"][
        "formal_real_arithmetic_certificate_claimed"
    ] is False
    assert config["certificate_protocol"][
        "floating_point_solver_not_formally_verified"
    ] is True
    assert (
        config["certificate_protocol"]["maximum_exact_milp_binary_variables"]
        == 4096
    )
    assert config["certificate_protocol"]["restart_policy"].startswith(
        "not_applicable_deterministic_single_schedule"
    )
    assert (
        config["certificate_protocol"]["calibration_rows"]
        + config["certificate_protocol"]["witness_rows"]
        <= config["datasets"]["imagenet_r"][
            "expected_reference_rows_after_intersection_and_head_fit_exclusion"
        ]
        == 5000
    )
    assert "Every 1024-row unit" in config["gate_decisions"]["optimization_gate"]
    assert "descriptive conditional WNID-cluster" in config[
        "gate_decisions"
    ]["breadth_gate_1_percent"]
    assert config["inference"]["optimizer_refit_within_bootstrap"] is False
    assert config["inference"]["display_label"].endswith(
        "assignments fixed/no optimizer refit"
    )
    assert config["inference"]["gate_kind"].startswith(
        "prespecified_robustness_screen"
    )
    assert "not_a_sampling_valid_confidence_interval" in config[
        "inference"
    ]["interpretation"]
    identities = gate._expected_identities(config)
    assert len(identities) == 162
    shards = [
        gate._sharded_identities(config, shard_count=4, shard_index=index)
        for index in range(4)
    ]
    assert sorted(identity for shard in shards for identity in shard) == list(
        identities
    )
    assert sorted(map(len, shards)) == [36, 36, 45, 45]
    group_owner = {}
    for shard_index, shard in enumerate(shards):
        for dataset, encoder, _, seed, _ in shard:
            group = (dataset, encoder, seed)
            assert group_owner.setdefault(group, shard_index) == shard_index
    assert len(group_owner) == 18
    assert {identity[-1] for identity in identities} == {100, 500, 1000}
    conditions = []
    for dataset in config["datasets"]:
        for encoder in config["encoders"]:
            for geometry in config["cost_geometries"]:
                for budget in config["certificate_protocol"][
                    "cost_budgets_relative"
                ]:
                    conditions.append({
                        "dataset": dataset,
                        "encoder": encoder,
                        "cost_geometry": geometry,
                        "cost_budget_relative": budget,
                        "disagreement": {
                            "lower": 0.60 if budget >= 0.05 else 0.30
                        },
                    })
    decisions = gate._gate_summary(conditions, config)
    assert decisions["breadth_1_percent__dinov3"]["status"] == "PASS"
    assert decisions["breadth_5_percent__clip"]["status"] == "PASS"
    assert decisions["breadth_1_percent__common_encoders"]["status"] == "PASS"
    assert decisions["breadth_5_percent__common_encoders"]["status"] == "PASS"
    assert decisions["geometry_gate"]["status"] == "PASS"


def test_completed_panel_marker_revalidates_every_unit_and_all_seven_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = "full_reference_oas_mahalanobis_after_bures"
    identities = (
        ("imagenet_v2", "dinov3", geometry, 201, 100),
        ("imagenet_r", "clip", geometry, 203, 500),
    )
    lock_path, store_map, root, lock, _ = _completed_panel_fixture(
        tmp_path, identities
    )
    calls: list[str] = []
    _install_completed_panel_stubs(
        monkeypatch, lock=lock, identities=identities, calls=calls
    )

    output = gate.run_panel_shard(
        lock_path,
        store_map,
        root,
        shard_count=1,
        shard_index=0,
        resume=True,
    )

    assert output == root / "SHARD_000_DONE.json"
    assert calls == [gate._unit_id(identity) for identity in identities]


@pytest.mark.parametrize("mutation", ["delete_manifest", "delete_array", "modify_array"])
def test_completed_panel_marker_rejects_payload_damage_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    geometry = "full_reference_oas_mahalanobis_after_bures"
    identities = (("imagenet_v2", "dinov3", geometry, 201, 100),)
    lock_path, store_map, root, lock, array_paths = _completed_panel_fixture(
        tmp_path, identities
    )
    identifier = gate._unit_id(identities[0])
    if mutation == "delete_manifest":
        (root / identifier / "UNIT_MANIFEST.json").unlink()
    else:
        array_path = array_paths[f"{identifier}::assignment_world_zero"]
        if mutation == "delete_array":
            array_path.unlink()
        else:
            np.save(array_path, np.asarray([999], dtype=np.int64))
    _install_completed_panel_stubs(
        monkeypatch, lock=lock, identities=identities, calls=[]
    )

    with pytest.raises(IntegrityError):
        gate.run_panel_shard(
            lock_path,
            store_map,
            root,
            shard_count=1,
            shard_index=0,
            resume=True,
        )


def test_breadth_gate_requires_the_same_datasets_for_both_encoders() -> None:
    config = gate._load_config(CONFIG, require_ready=True)
    conditions = []
    for dataset in config["datasets"]:
        for encoder in config["encoders"]:
            for geometry in config["cost_geometries"]:
                for budget in config["certificate_protocol"][
                    "cost_budgets_relative"
                ]:
                    if encoder == "dinov3":
                        passing = dataset in {"imagenet_sketch", "imagenet_v2"}
                    else:
                        passing = dataset in {"imagenet_v2", "imagenet_r"}
                    conditions.append({
                        "dataset": dataset,
                        "encoder": encoder,
                        "cost_geometry": geometry,
                        "cost_budget_relative": budget,
                        "disagreement": {"lower": 0.60 if passing else 0.0},
                    })
    decisions = gate._gate_summary(conditions, config)
    assert decisions["breadth_1_percent__dinov3"]["status"] == "PASS"
    assert decisions["breadth_1_percent__clip"]["status"] == "PASS"
    assert decisions["breadth_1_percent__common_encoders"]["status"] == "FAIL"
    assert decisions["breadth_5_percent__common_encoders"]["status"] == "FAIL"
    assert decisions["geometry_gate"]["status"] == "FAIL"


def test_content_selection_is_multiset_invariant_and_duplicate_safe() -> None:
    features = np.asarray([[3.0, 1.0], [0.0, 2.0], [-1.0, 4.0]])
    sample_ids = np.asarray(["sample-c", "sample-a", "sample-b"])
    first, _, first_hash = gate._content_select(
        features,
        sample_ids=sample_ids,
        requested=2,
        name="fixture",
        salt=b"locked-split",
    )
    second, _, second_hash = gate._content_select(
        features[[2, 0, 1]],
        sample_ids=sample_ids[[2, 0, 1]],
        requested=2,
        name="fixture",
        salt=b"locked-split",
    )
    assert np.array_equal(first, second)
    assert first_hash == second_hash
    duplicate, _, duplicate_hash = gate._content_select(
        np.asarray([[1.0], [1.0]]),
        sample_ids=np.asarray(["indexed-b", "indexed-a"]),
        requested=2,
        name="duplicate fixture",
        salt=b"locked-split",
    )
    permuted, _, permuted_hash = gate._content_select(
        np.asarray([[1.0], [1.0]])[[1, 0]],
        sample_ids=np.asarray(["indexed-b", "indexed-a"])[[1, 0]],
        requested=2,
        name="duplicate fixture",
        salt=b"locked-split",
    )
    assert np.array_equal(duplicate, permuted)
    assert duplicate_hash == permuted_hash
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        gate._content_select(
            np.asarray([[1.0], [1.0]]),
            sample_ids=np.asarray(["reused", "reused"]),
            requested=2,
            name="duplicate fixture",
            salt=b"locked-split",
        )


def test_group_timing_is_checksum_bound_and_retries_are_deduplicated() -> None:
    config = gate._load_config(CONFIG, require_ready=True)
    lock = {
        "manifest_sha256": "a" * 64,
        "config": config,
        "stores": _timing_store_records(config),
    }
    identities = [
        identity
        for identity in gate._expected_identities(config)
        if identity[0] == "imagenet_v2"
        and identity[1] == "dinov3"
        and identity[3] == 201
    ]
    first = _fixture_compute_event(
        lock, identities[0], purpose="scientific_panel_compute"
    )
    retry = _fixture_compute_event(
        lock, identities[0], purpose="scientific_panel_compute", retry=1
    )
    validated = gate._validated_compute_event(
        first,
        lock=lock,
        dataset="imagenet_v2",
        encoder="dinov3",
        seed=201,
        purpose="scientific_panel_compute",
    )
    assert validated["stage_sum_nanoseconds"] < validated[
        "group_wall_nanoseconds"
    ]
    rows = []
    for position, identity in enumerate(identities):
        event = first if position < 3 else retry
        rows.append((
            event,
            position in {0, 3},
            gate._unit_id(identity),
        ))
    summary = gate._compute_accounting_summary(
        rows,
        expected_group_ids={gate._group_id("imagenet_v2", "dinov3", 201)},
        purpose="scientific_panel_compute",
    )
    assert summary["successful_group_compute_events"] == 2
    assert summary["successful_retry_group_compute_events"] == 1
    assert summary["events_deduplicated_across_unit_records"] == 7
    assert summary["totals_nanoseconds"]["group_wall_nanoseconds"] == (
        2 * first["group_wall_nanoseconds"]
    )
    primary_geometry = config["cost_geometries"][0]
    assert summary["totals_by_geometry_nanoseconds"][primary_geometry][
        "dense_problem_preparation_nanoseconds"
    ] == 26
    assert summary["totals_by_geometry_nanoseconds"][primary_geometry][
        "budgets"
    ]["0100bp"]["exact_validation_nanoseconds"] == 38
    tampered = json.loads(json.dumps(first))
    tampered["geometry_stages"][0]["budgets"][0][
        "certificate_nanoseconds"
    ] += 1
    with pytest.raises(IntegrityError, match="checksum"):
        gate._validated_compute_event(
            tampered,
            lock=lock,
            dataset="imagenet_v2",
            encoder="dinov3",
            seed=201,
            purpose="scientific_panel_compute",
        )


def test_aggregate_requires_complete_authenticated_reconstruction_inventory(
    tmp_path: Path,
) -> None:
    config = gate._load_config(CONFIG, require_ready=True)
    identities = set(gate._expected_identities(config))
    expected = {
        identity: {
            "manifest_file_sha256": sha256_json({"file": gate._unit_id(identity)}),
            "manifest_sha256": sha256_json({"payload": gate._unit_id(identity)}),
            "meets_gap_target": True,
            "primal_dual_gap": 0.01,
        }
        for identity in identities
    }
    lock = {
        "manifest_sha256": "a" * 64,
        "config": config,
        "stores": _timing_store_records(config),
    }
    lock_file_sha256 = "b" * 64
    partitions = [
        gate._sharded_identities(config, shard_count=4, shard_index=index)
        for index in range(4)
    ]
    for index, partition in enumerate(partitions):
        owned_groups = set()
        reports = []
        for identity in partition:
            group_id = gate._group_id(identity[0], identity[1], identity[3])
            owner = group_id not in owned_groups
            owned_groups.add(group_id)
            reports.append(_fixture_verification_report(
                lock, identity, expected[identity], owner=owner
            ))
        compute_accounting = gate._verification_compute_accounting(
            reports,
            expected_group_ids={
                gate._group_id(dataset, encoder, seed)
                for dataset, encoder, _, seed, _ in partition
            },
        )
        write_json_atomic(tmp_path / f"shard_{index}.json", seal_manifest({
            "schema_version": gate.VERIFICATION_SHARD_SCHEMA,
            "status": "PASS",
            "lock_file_sha256": lock_file_sha256,
            "lock_manifest_sha256": lock["manifest_sha256"],
            "shard_count": 4,
            "shard_index": index,
            "expected_units": len(reports),
            "verified_units": len(reports),
            "full_endpoint_feature_reconstruction": True,
            "compute_accounting": compute_accounting,
            "reports": reports,
        }))
    inventory = gate._validated_verification_inventory(
        tmp_path,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        expected=expected,
    )
    assert [row["shard_index"] for row in inventory] == [0, 1, 2, 3]
    assert sum(
        row["compute_accounting"]["successful_group_compute_events"]
        for row in inventory
    ) == 18
    (tmp_path / "shard_3.json").unlink()
    with pytest.raises(IntegrityError, match="verification manifests"):
        gate._validated_verification_inventory(
            tmp_path,
            lock=lock,
            lock_file_sha256=lock_file_sha256,
            expected=expected,
        )


def test_failed_scientific_aggregate_is_preserved_and_cli_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = gate._load_config(CONFIG, require_ready=True)
    config = json.loads(json.dumps(config))
    config["inference"]["replicates"] = 2
    identities = gate._expected_identities(config)
    units_root = tmp_path / "units"
    records_by_path = {}
    for position, identity in enumerate(identities):
        manifest = units_root / f"unit_{position:03d}" / "UNIT_MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        records_by_path[manifest.resolve()] = (
            identity,
            {
                "manifest_file_sha256": sha256_json({"file": position}),
                "manifest_sha256": sha256_json({"manifest": position}),
                "rows": np.asarray([1024], dtype=np.int64),
                "disagreements": np.asarray([0], dtype=np.int64),
                "primal_dual_gap": 0.0,
                "meets_gap_target": True,
                "exact_proven": True,
            },
        )
    only_wnid = ["n00000001"]
    lock = {
        "manifest_sha256": "a" * 64,
        "config": config,
        "stores": {
            key: {**record, "wnids": only_wnid}
            for key, record in _timing_store_records(config).items()
        },
    }
    owned_groups = set()
    for identity, record in records_by_path.values():
        group_id = gate._group_id(identity[0], identity[1], identity[3])
        owner = group_id not in owned_groups
        owned_groups.add(group_id)
        event = _fixture_compute_event(
            lock, identity, purpose="scientific_panel_compute"
        )
        record.update({
            "compute_event": event,
            "compute_event_owner": owner,
            "unit_array_serialization_nanoseconds": 10,
            "unit_invocation_wall_nanoseconds_before_manifest_write": (
                1000 if owner else 100
            ),
        })
    lock_path = tmp_path / "LOCK.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    verification_root = tmp_path / "verification"
    verification_root.mkdir()
    output = tmp_path / "AGGREGATE.json"
    monkeypatch.setattr(gate, "_load_lock", lambda path: lock)
    monkeypatch.setattr(
        gate,
        "_validate_aggregate_unit",
        lambda path, **kwargs: records_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(
        gate,
        "_validated_verification_inventory",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        gate,
        "_combined_verification_compute_accounting",
        lambda *args, **kwargs: {"status": "fixture"},
    )
    produced = gate.aggregate_units(
        lock_path, units_root, verification_root, output
    )
    assert produced == output.resolve()
    aggregate = read_json(output)
    verify_manifest_checksum(aggregate)
    assert aggregate["status"] == "PASS"
    assert aggregate["artifact_status"] == "PASS"
    assert aggregate["scientific_gate_status"] == "FAIL"
    assert aggregate["all_required_gates_pass"] is False
    assert aggregate["failed_required_scientific_gate_keys"] == [
        "breadth_1_percent__common_encoders",
        "breadth_5_percent__common_encoders",
        "geometry_gate",
    ]
    assert output.is_file()

    monkeypatch.setattr(gate, "aggregate_units", lambda *args, **kwargs: produced)
    exit_code = gate.main([
        "aggregate",
        "--lock", str(lock_path),
        "--units-root", str(units_root),
        "--verification-root", str(verification_root),
        "--output", str(output),
    ])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == gate.SCIENTIFIC_GATE_FAILURE_EXIT_CODE == 2
    assert report["status"] == "FAIL"
    assert report["artifact_status"] == "PASS"
    assert report["scientific_gate_status"] == "FAIL"
    assert report["all_required_gates_pass"] is False
    assert output.is_file()
