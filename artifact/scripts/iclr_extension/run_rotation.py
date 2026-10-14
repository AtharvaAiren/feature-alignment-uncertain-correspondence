#!/usr/bin/env python3
"""Run the locked Bures versus soft-pseudo-label orientation extension.

Each dataset/encoder/corruption/split cell is written atomically as a sealed
JSON artifact.  Existing units are resumed only when their full identity and
checksums match.  The aggregate is rebuilt from those immutable units and is
marked PASS only when the selected panel is complete.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    sha256_file,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import (
    DiscoveredFeatureStore,
    FeatureStore,
    discover_feature_stores,
    load_feature_store,
)
from marginal_alignment.campaign.metrics import classification_metrics
from marginal_alignment.campaign.split_plan import build_split_plan
from marginal_alignment.iclr_extension.pseudo_orientation import (
    fit_bures_pseudo_orientation,
)
from marginal_alignment.metrics import logits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "iclr_extension" / "rotation.json"
UNIT_SCHEMA = "beyond_marginals.rotation_extension.unit.v1"
AGGREGATE_SCHEMA = "beyond_marginals.rotation_extension.aggregate.v1"
SUITE_NAME = "play1_rotation_breaker"
PLAY1_LOCK_SCHEMA = "marginal_alignment.iclr_extension.play1_lock.v1"
PLAY1_LOCK_STATUS = "PRESPECIFIED_BEFORE_ROTATION_EXTENSION_OUTCOME_INSPECTION"
PLAY1_REQUIRED_IMPORTED_DEPENDENCIES = frozenset({
    # Python executes these package initializers (and their eager imports)
    # before resolving the campaign/extension submodules below.
    "src/marginal_alignment/__init__.py",
    "src/marginal_alignment/ceiling.py",
    "src/marginal_alignment/transforms.py",
    "src/marginal_alignment/campaign/__init__.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/campaign/method_registry.py",
    "src/marginal_alignment/campaign/metrics.py",
    "src/marginal_alignment/campaign/split_plan.py",
    "src/marginal_alignment/linalg.py",
    "src/marginal_alignment/metrics.py",
})
PLAY1_SOURCE_FILES = frozenset({
    "scripts/iclr_extension/__init__.py",
    "scripts/iclr_extension/run_rotation.py",
    "src/marginal_alignment/iclr_extension/__init__.py",
    "src/marginal_alignment/iclr_extension/pseudo_orientation.py",
    "tests/iclr_extension/__init__.py",
    "tests/iclr_extension/test_pseudo_orientation.py",
    "tests/iclr_extension/test_rotation_runner.py",
}) | PLAY1_REQUIRED_IMPORTED_DEPENDENCIES
PLAY1_EXECUTION_ARTIFACTS = frozenset({
    "campaign_manifest",
    "external_input_manifest",
    "experiment_lock",
    "resolved_config",
    "real_manifest",
    "real_rows",
    "runtime_manifest",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _indices(ids: np.ndarray, selected: Iterable[Any], *, name: str) -> np.ndarray:
    lookup = {
        value.item() if isinstance(value, np.generic) else value: index
        for index, value in enumerate(np.asarray(ids))
    }
    try:
        return np.asarray([lookup[value] for value in selected], dtype=np.int64)
    except KeyError as error:
        raise IntegrityError(
            f"{name} references absent endpoint ID {error.args[0]!r}"
        ) from error


def _head(store: FeatureStore) -> tuple[np.ndarray, np.ndarray]:
    roles = store.manifest.get("head", {})
    _require(isinstance(roles, Mapping), "feature store lacks frozen-readout declaration")
    weight_role, bias_role = roles.get("weight_role"), roles.get("bias_role")
    _require(bool(weight_role) and bool(bias_role), (
        "frozen readout must declare weight_role and bias_role"
    ))
    weight = np.asarray(store.array(str(weight_role)), dtype=np.float64)
    bias = np.asarray(store.array(str(bias_role)), dtype=np.float64)
    _require(weight.ndim == 2 and bias.shape == (weight.shape[0],), (
        "frozen readout arrays have invalid shapes"
    ))
    return weight, bias


def _labels(
    reference_labels: np.ndarray | None,
    target_labels: np.ndarray | None,
    reference_index: np.ndarray,
    target_index: np.ndarray,
) -> np.ndarray:
    _require(
        reference_labels is not None and target_labels is not None,
        "evaluation requires labels on both endpoints",
    )
    reference = np.asarray(reference_labels[reference_index], dtype=np.int64)
    target = np.asarray(target_labels[target_index], dtype=np.int64)
    _require(
        np.array_equal(reference, target),
        "verified evaluation pairs have unequal endpoint labels",
    )
    return target


def _seed(*values: Any) -> int:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def _feature_store_key(
    dataset: Any, encoder: Any, corruption: Any, severity: Any
) -> tuple[str, str, str, int]:
    return str(dataset), str(encoder), str(corruption), int(severity)


def _external_feature_store_index(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str, str, int], str]:
    """Validate and index the base campaign's exact feature-store identities."""

    _require(
        payload.get("schema_version")
        == "marginal_alignment.iclr.external_inputs.v1"
        and payload.get("status") == "PASS",
        "source external-input manifest is not a passing V1 manifest",
    )
    rows = payload.get("native_feature_stores")
    _require(isinstance(rows, list), "source external-input store rows are malformed")
    _require(
        int(payload.get("native_feature_store_count", -1)) == len(rows),
        "source external-input feature-store count is inconsistent",
    )
    index: dict[tuple[str, str, str, int], str] = {}
    required_fields = {
        "dataset", "encoder", "corruption", "severity", "manifest",
        "manifest_sha256",
    }
    for row in rows:
        _require(
            isinstance(row, Mapping) and set(row) == required_fields,
            "source external-input feature-store row is malformed",
        )
        key = _feature_store_key(
            row["dataset"], row["encoder"], row["corruption"], row["severity"]
        )
        _require(key not in index, f"duplicate source external-input store: {key}")
        expected = row["manifest_sha256"]
        _require(_valid_sha256(expected), f"invalid source feature-store hash: {key}")
        index[key] = str(expected)
    return index


def _runtime_identity() -> dict[str, Any]:
    """Capture behavior-relevant numerical versions without host identity."""

    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {
            "numpy": str(np.__version__),
            "scipy": importlib_metadata.version("scipy"),
            "sklearn": importlib_metadata.version("scikit-learn"),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def _project_file(value: Any, *, name: str, required: bool = True) -> Path:
    _require(isinstance(value, str) and value, f"{name} path must be nonempty")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{name} path must be repository-relative")
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise IntegrityError(f"{name} path escapes repository: {value}") from error
    if required:
        _require(candidate.is_file(), f"{name} is absent: {candidate}")
    return candidate


def _validate_play1_lock(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    lock_path: Path,
    lock: Mapping[str, Any],
    master_path: Path,
    master: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the complete outcome-blind Play-1 evidence chain.

    The imported ``resolved_config`` and compressed rows are cleaned transport
    copies rather than byte-identical members of the original campaign
    directory.  Their exact local bytes are therefore bound separately while
    the original experiment-lock, real-manifest, and uncompressed-row hashes
    are cross-checked through ``CAMPAIGN_MANIFEST.json``.
    """

    _require(lock.get("schema_version") == PLAY1_LOCK_SCHEMA, "unexpected Play-1 lock schema")
    _require(lock.get("status") == PLAY1_LOCK_STATUS, "Play-1 lock is not prespecified")

    locked_config = str(lock.get("config", ""))
    expected_config = _project_file(locked_config, name="locked rotation config")
    _require(expected_config == config_path.resolve(), "rotation config path differs from Play-1 lock")
    _require(_valid_sha256(lock.get("config_file_sha256")), "invalid locked rotation-config hash")
    _require(
        sha256_file(expected_config) == lock["config_file_sha256"],
        "rotation config differs from Play-1 lock",
    )
    expected_lock = _project_file(config.get("play1_lock"), name="configured Play-1 lock")
    _require(expected_lock == lock_path.resolve(), "configured Play-1 lock path changed")

    locked_master = str(lock.get("master_lock", ""))
    expected_master = _project_file(locked_master, name="locked extension master lock")
    _require(expected_master == master_path.resolve(), "master-lock path differs from Play-1 lock")
    _require(str(config.get("extension_lock", "")) == locked_master, "configured master-lock path changed")
    _require(_valid_sha256(lock.get("master_lock_file_sha256")), "invalid locked master-lock hash")
    _require(
        sha256_file(expected_master) == lock["master_lock_file_sha256"],
        "extension master lock differs from Play-1 lock",
    )
    _require(
        master.get("schema_version") == "beyond_marginals.iclr_extension_lock.v1"
        and master.get("status") == "PRESPECIFIED_BEFORE_EXTENSION_RESULTS",
        "extension master lock is not prespecified",
    )

    source_record = lock.get("source_campaign_config")
    _require(isinstance(source_record, Mapping), "Play-1 lock lacks source_campaign_config")
    _require(
        set(source_record) == {"path", "sha256"},
        "source_campaign_config record is malformed",
    )
    _require(
        str(source_record["path"]) == str(config.get("source_campaign_config", "")),
        "source campaign config path changed",
    )
    source_path = _project_file(source_record["path"], name="source campaign config")
    _require(_valid_sha256(source_record["sha256"]), "invalid source campaign config hash")
    _require(
        sha256_file(source_path) == source_record["sha256"],
        "source campaign config differs from Play-1 lock",
    )

    source_files = lock.get("source_files")
    _require(isinstance(source_files, Mapping), "Play-1 lock lacks source-file hashes")
    _require(set(source_files) == PLAY1_SOURCE_FILES, "Play-1 source-file inventory drift")
    required_dependencies = lock.get("required_imported_dependencies")
    _require(
        isinstance(required_dependencies, list)
        and set(map(str, required_dependencies))
        == PLAY1_REQUIRED_IMPORTED_DEPENDENCIES,
        "Play-1 imported-dependency inventory drift",
    )
    _require(
        set(map(str, required_dependencies)).issubset(set(source_files)),
        "Play-1 lock does not hash every required imported dependency",
    )
    for relative, expected in source_files.items():
        _require(_valid_sha256(expected), f"invalid Play-1 source hash: {relative}")
        candidate = _project_file(relative, name="locked Play-1 source")
        _require(sha256_file(candidate) == expected, f"locked Play-1 source changed: {relative}")

    execution = lock.get("source_execution_evidence")
    _require(isinstance(execution, Mapping), "Play-1 lock lacks source execution evidence")
    _require(
        set(execution) == {"artifacts", "semantic_reconstruction_policy"},
        "source execution evidence has unexpected fields",
    )
    _require(
        execution.get("semantic_reconstruction_policy")
        == "cleaned_transport_copies_bound_separately_from_original_campaign_members",
        "source execution reconstruction policy changed",
    )
    artifacts = execution.get("artifacts")
    _require(isinstance(artifacts, Mapping), "source execution artifacts are malformed")
    _require(set(artifacts) == PLAY1_EXECUTION_ARTIFACTS, "source execution artifact inventory drift")
    verified_artifacts: dict[str, Any] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    for name in sorted(PLAY1_EXECUTION_ARTIFACTS):
        record = artifacts[name]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256", "required_at_play1_runtime"},
            f"source execution record is malformed: {name}",
        )
        _require(_valid_sha256(record["sha256"]), f"invalid source execution hash: {name}")
        required = record["required_at_play1_runtime"]
        _require(isinstance(required, bool), f"invalid source execution requirement: {name}")
        candidate = _project_file(
            record["path"], name=f"source execution artifact {name}", required=required
        )
        present = candidate.is_file()
        if present:
            _require(
                sha256_file(candidate) == record["sha256"],
                f"source execution artifact changed: {name}",
            )
            if name != "real_rows":
                value = read_json(candidate)
                _require(isinstance(value, Mapping), f"source execution artifact is not an object: {name}")
                payloads[name] = value
        verified_artifacts[name] = {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "required_at_play1_runtime": required,
            "present_and_verified": present,
        }

    campaign = payloads["campaign_manifest"]
    external_inputs = payloads["external_input_manifest"]
    experiment_lock = payloads["experiment_lock"]
    resolved = payloads["resolved_config"]
    real_manifest = payloads["real_manifest"]
    runtime_manifest = payloads["runtime_manifest"]
    _require(
        campaign.get("schema_version") == "marginal_alignment.iclr.campaign_manifest.v2"
        and campaign.get("status") == "PASS",
        "source campaign manifest is not a passing V2 campaign",
    )
    _require(
        campaign.get("campaign_id") == resolved.get("campaign_id"),
        "source campaign and resolved config IDs differ",
    )
    _require(
        resolved.get("schema_version") == "marginal_alignment.iclr.campaign.v2",
        "source resolved config has unexpected schema",
    )
    _require(
        experiment_lock.get("schema_version") == "marginal_alignment.iclr.lock.v2"
        and "BEFORE_OUTCOME_INSPECTION" in str(experiment_lock.get("status", "")),
        "source experiment lock lacks a pre-outcome amendment status",
    )
    _require(
        real_manifest.get("schema_version") == "marginal_alignment.iclr.real_manifest.v2"
        and real_manifest.get("status") == "PASS"
        and real_manifest.get("semantic_grid_validation", {}).get("status") == "PASS",
        "source real panel is not complete and passing",
    )
    _require(
        campaign.get("runtime_sha256") == runtime_manifest.get("runtime_sha256"),
        "source runtime identity differs between campaign and runtime manifests",
    )
    external_store_index = _external_feature_store_index(external_inputs)
    _require(
        external_inputs.get("external_input_sha256")
        == campaign.get("external_input_sha256"),
        "source external-input semantic identity differs from campaign manifest",
    )
    _require(
        len(external_store_index)
        == int(master.get("base_release", {}).get("feature_store_count", -1)),
        "source external-input store count differs from the extension master lock",
    )
    required_store_keys = {
        _feature_store_key(dataset, encoder, corruption, config["severity"])
        for dataset, specification in config["datasets"].items()
        for encoder in specification["encoders"]
        for corruption in (
            list(config["panels"]["primary"])
            + list(config["panels"]["diagnostic"])
        )
    }
    missing_locked_stores = sorted(required_store_keys - set(external_store_index))
    _require(
        not missing_locked_stores,
        "rotation panel is absent from the source external-input manifest: "
        f"{missing_locked_stores[:5]}",
    )
    _require(
        artifacts["campaign_manifest"]["sha256"]
        == master.get("base_release", {}).get("base_campaign_manifest_sha256"),
        "source campaign manifest differs from the extension master lock",
    )
    campaign_files = {
        str(record.get("path")): str(record.get("sha256"))
        for record in campaign.get("files", [])
        if isinstance(record, Mapping)
    }
    _require(
        campaign_files.get("ICLR_EXPERIMENT_LOCK_V2.json")
        == artifacts["experiment_lock"]["sha256"],
        "source experiment-lock copy differs from campaign manifest",
    )
    _require(
        campaign_files.get("real/manifest.json") == artifacts["real_manifest"]["sha256"],
        "source real-manifest copy differs from campaign manifest",
    )
    _require(
        _valid_sha256(campaign_files.get("runtime_manifest.json")),
        "source campaign does not bind its original runtime manifest",
    )
    _require(
        _valid_sha256(campaign_files.get("external_input_manifest.json")),
        "source campaign does not bind its original external-input manifest",
    )
    _require(
        campaign_files.get("real/rows.jsonl") == real_manifest.get("rows_sha256"),
        "source real rows identity differs between campaign and real manifests",
    )
    _require(
        int(resolved.get("global_seed", -1)) == int(config.get("global_seed", -2)),
        "source execution global seed differs from rotation config",
    )

    return {
        "schema_version": "marginal_alignment.iclr_extension.play1_provenance.v1",
        "status": "VALIDATED",
        "play1_lock": {
            "path": str(lock_path.resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(lock_path),
            "status": str(lock["status"]),
        },
        "master_lock": {
            "path": locked_master,
            "sha256": str(lock["master_lock_file_sha256"]),
        },
        "source_campaign_config": {
            "path": str(source_record["path"]),
            "sha256": str(source_record["sha256"]),
        },
        "source_execution_evidence": verified_artifacts,
        "source_external_input_sha256": str(
            external_inputs["external_input_sha256"]
        ),
        "source_feature_store_manifests": [
            {
                "dataset": key[0],
                "encoder": key[1],
                "corruption": key[2],
                "severity": key[3],
                "manifest_sha256": external_store_index[key],
            }
            for key in sorted(required_store_keys)
        ],
        "source_runtime_identity": runtime_manifest.get("environment", {}).get(
            "runtime_identity"
        ),
        "extension_runtime_identity": _runtime_identity(),
        "source_files": {key: str(source_files[key]) for key in sorted(source_files)},
    }


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    config_path = path.expanduser().resolve()
    config = read_json(config_path)
    _require(isinstance(config, dict), "rotation config must be a JSON object")
    master_path = _project_file(config.get("extension_lock"), name="extension master lock")
    master = read_json(master_path)
    _require(isinstance(master, dict), "extension master lock must be a JSON object")
    _validate_config_against_lock(config, master)
    lock_path = _project_file(config.get("play1_lock"), name="Play-1 lock")
    lock = read_json(lock_path)
    _require(isinstance(lock, dict), "Play-1 lock must be a JSON object")
    provenance = _validate_play1_lock(
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock=lock,
        master_path=master_path,
        master=master,
    )
    return config, lock_path, master, provenance


def _validate_config_against_lock(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    """Reject any drift from the Play 1 master lock before reading results."""

    _require(
        config.get("schema_version")
        == "beyond_marginals.rotation_extension.config.v1",
        "unexpected rotation config schema",
    )
    _require(
        lock.get("schema_version") == "beyond_marginals.iclr_extension_lock.v1",
        "unexpected extension lock schema",
    )
    _require(
        lock.get("status") == "PRESPECIFIED_BEFORE_EXTENSION_RESULTS",
        "extension lock is not prespecified",
    )
    rules = lock.get("global_rules", {})
    for key in (
        "target_or_evaluation_labels_available_to_fit",
        "sample_correspondence_available_to_unpaired_estimators",
    ):
        _require(rules.get(key) is False, f"lock permits forbidden information: {key}")
    for key in (
        "evaluation_disjoint_from_calibration",
        "all_selection_uses_calibration_data_only",
    ):
        _require(rules.get(key) is True, f"lock lacks required separation: {key}")

    play = lock.get("play1_rotation_breaker", {})
    _require(int(config["severity"]) == int(play["severity"]), "severity drift")
    _require(
        int(config["calibration_size"]) == int(play["calibration_size"]),
        "calibration-size drift",
    )
    _require(
        list(config["split_seeds"]) == list(play["split_seeds"]),
        "split-seed drift",
    )
    _require(
        list(config["panels"]["primary"]) == list(play["primary_corruptions"]),
        "primary-corruption drift",
    )
    _require(
        list(config["panels"]["diagnostic"])
        == list(play["diagnostic_corruptions"]),
        "diagnostic-corruption drift",
    )
    _require(
        list(config["datasets"]) == list(play["datasets"]), "dataset drift"
    )
    for dataset in play["datasets"]:
        _require(
            list(config["datasets"][dataset]["encoders"])
            == list(play["encoders"]),
            f"encoder drift for {dataset}",
        )

    declared = config["method"]
    locked = play["methods"]["bures_pseudo_orientation"]
    exact_keys = (
        "covariance_estimator",
        "candidate_ranks",
        "procrustes_shrinkages",
        "confidence_threshold",
        "temperature",
        "validation_fraction",
        "minimum_active_classes",
        "minimum_effective_class_mass",
        "rank_support_rule",
        "minimum_explained_discrepancy_energy",
        "dataset_mass_thresholds",
        "failure_policy",
    )
    for key in exact_keys:
        _require(declared.get(key) == locked.get(key), f"method drift: {key}")
    _require(
        float(declared["minimum_relative_improvement"])
        == float(locked["minimum_relative_validation_improvement"]),
        "method drift: minimum relative validation improvement",
    )
    _require(
        float(declared["minimum_fit_validation_subspace_stability"])
        == float(locked["minimum_fit_validation_subspace_stability"]),
        "method drift: fit/validation subspace stability",
    )
    locked_null = locked.get("null_calibration", {})
    _require(
        int(declared["null_permutations"]) == int(locked_null.get("permutations", -1))
        and float(declared["null_quantile"]) == float(locked_null.get("quantile", -1.0)),
        "method drift: responsibility-permutation null calibration",
    )
    _require(
        declared.get("null_interpretation") == locked_null.get("interpretation")
        == "conservative_heuristic_selection_gate_not_a_significance_test_or_identification_claim",
        "method drift: responsibility-permutation null interpretation",
    )
    _require(
        declared.get("refit_selected_specification_on_all_calibration") is True
        and locked.get("final_refit_policy")
        == "selected_specification_refit_on_all_calibration_rows_before_evaluation",
        "method drift: selected specification refit policy",
    )
    _require(
        config["metrics"]["primary"] == play["primary_metric"],
        "primary-metric drift",
    )
    _require(
        list(config["metrics"]["secondary"])
        == list(play["secondary_metrics"]),
        "secondary-metric drift",
    )
    _require(
        config["metrics"].get("directions") == play.get("metric_directions"),
        "metric-direction drift",
    )
    _require(
        config["metrics"].get("sign_normalized_improvement")
        == "positive_is_better_for_every_metric",
        "sign-normalized metric convention changed",
    )
    uncertainty = config.get("uncertainty", {})
    locked_uncertainty = play.get("uncertainty", {})
    uncertainty_pairs = {
        "unit": "cluster_unit",
        "panel_cluster_counts": "panel_cluster_counts",
        "split_reduction": "split_seed_policy",
        "bootstrap": "bootstrap",
        "replicates": "replicates",
        "seed": "seed",
        "confidence_level": "confidence_level",
        "encoders": "encoders",
        "interval_interpretation": "interval_interpretation",
    }
    for configured_key, locked_key in uncertainty_pairs.items():
        _require(
            uncertainty.get(configured_key) == locked_uncertainty.get(locked_key),
            f"uncertainty drift: {configured_key}",
        )

    source_path = Path(str(config["source_campaign_config"]))
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source = read_json(source_path)
    _require(
        int(config["global_seed"]) == int(source["global_seed"]),
        "global seed does not match the source campaign",
    )
    source_datasets = {row["name"]: row for row in source["datasets"]}
    for dataset, specification in config["datasets"].items():
        _require(dataset in source_datasets, f"unknown source dataset: {dataset}")
        source_specification = source_datasets[dataset]
        _require(
            int(specification["evaluation_size"])
            == int(source_specification["evaluation_size"]),
            f"evaluation-size drift for {dataset}",
        )
        declared_corruptions = set(config["panels"]["primary"]) | set(
            config["panels"]["diagnostic"]
        )
        _require(
            declared_corruptions.issubset(source_specification["corruptions"]),
            f"extension contains unknown corruptions for {dataset}",
        )


def _panels(config: Mapping[str, Any], requested: str) -> tuple[str, ...]:
    return ("primary", "diagnostic") if requested == "all" else (requested,)


def _expected_units(
    config: Mapping[str, Any], requested_panel: str
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for panel in _panels(config, requested_panel):
        for dataset, dataset_specification in config["datasets"].items():
            for encoder in dataset_specification["encoders"]:
                for corruption in config["panels"][panel]:
                    for split_seed in config["split_seeds"]:
                        units.append({
                            "panel": panel,
                            "dataset": dataset,
                            "encoder": encoder,
                            "corruption": corruption,
                            "severity": int(config["severity"]),
                            "split_seed": int(split_seed),
                            "calibration_size": int(config["calibration_size"]),
                            "evaluation_size": int(
                                dataset_specification["evaluation_size"]
                            ),
                        })
    return units


def _sharded_units(
    expected: list[dict[str, Any]], *, shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    """Partition the prespecified unit order without changing unit identity.

    Shards may safely share an output root: unit and split paths are disjoint,
    and a final ``--aggregate-only`` pass deterministically seals the complete
    panel after every shard exits.
    """

    shard_index, num_shards = int(shard_index), int(num_shards)
    _require(num_shards >= 1, "num_shards must be at least one")
    _require(0 <= shard_index < num_shards, "shard_index must lie in [0,num_shards)")
    return [
        identity for position, identity in enumerate(expected)
        if position % num_shards == shard_index
    ]


def _store_index(
    feature_root: Path,
    *,
    expected_manifest_sha256: Mapping[tuple[str, str, str, int], str],
) -> dict[tuple[str, str, str, int], DiscoveredFeatureStore]:
    index: dict[tuple[str, str, str, int], DiscoveredFeatureStore] = {}
    for discovered in discover_feature_stores([feature_root]):
        if discovered.kind != "native_v2":
            continue
        manifest = read_json(discovered.manifest_path)
        metadata = manifest.get("metadata", {})
        key = _feature_store_key(
            discovered.dataset,
            discovered.encoder,
            metadata.get("corruption"),
            metadata.get("severity"),
        )
        if key in index:
            raise IntegrityError(
                f"duplicate feature store for {key}: "
                f"{index[key].manifest_path} and {discovered.manifest_path}"
            )
        if key in expected_manifest_sha256:
            observed_sha256 = sha256_file(discovered.manifest_path)
            _require(
                observed_sha256 == expected_manifest_sha256[key],
                "feature-store manifest differs from locked source campaign: "
                f"{key}",
            )
        index[key] = discovered
    return index


def _head_exclusions(
    store: FeatureStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Reproduce the source campaign's head-fit exclusion contract."""

    reference = store.endpoint("reference")
    target = store.endpoint("target")
    pairing = store.pairing()
    model = store.manifest["metadata"]["model_provenance"]
    head_fit_ids = np.asarray(store.array(str(model["head_fit_ids_role"])))
    head_namespace = str(model["head_fit_id_namespace"])
    if (
        str(model["head_fit_split"]) == str(model["reference_split"])
        and head_namespace != reference.id_namespace
    ):
        raise IntegrityError(
            "head fit and reference endpoint share a split but not an ID namespace"
        )
    excluded_reference = np.asarray([], dtype=reference.ids.dtype)
    excluded_target = np.asarray([], dtype=target.ids.dtype)
    paired_reference, paired_target = pairing.validated()
    if head_namespace == reference.id_namespace:
        target_by_reference = {
            left.item() if isinstance(left, np.generic) else left:
            right.item() if isinstance(right, np.generic) else right
            for left, right in zip(paired_reference, paired_target)
        }
        values = [
            value.item() if isinstance(value, np.generic) else value
            for value in head_fit_ids
        ]
        missing = [value for value in values if value not in target_by_reference]
        if missing:
            raise IntegrityError(
                f"head-fit reference IDs lack target counterparts: {missing[:3]}"
            )
        excluded_reference = np.asarray(values, dtype=reference.ids.dtype)
        excluded_target = np.asarray(
            [target_by_reference[value] for value in values], dtype=target.ids.dtype
        )
    elif head_namespace == target.id_namespace:
        reference_by_target = {
            right.item() if isinstance(right, np.generic) else right:
            left.item() if isinstance(left, np.generic) else left
            for left, right in zip(paired_reference, paired_target)
        }
        values = [
            value.item() if isinstance(value, np.generic) else value
            for value in head_fit_ids
        ]
        missing = [value for value in values if value not in reference_by_target]
        if missing:
            raise IntegrityError(
                f"head-fit target IDs lack reference counterparts: {missing[:3]}"
            )
        excluded_target = np.asarray(values, dtype=target.ids.dtype)
        excluded_reference = np.asarray(
            [reference_by_target[value] for value in values],
            dtype=reference.ids.dtype,
        )
    return excluded_reference, excluded_target, head_fit_ids, head_namespace


def _artifact_path(root: Path, folder: str, identity: Mapping[str, Any]) -> Path:
    return (
        root
        / folder
        / str(identity["panel"])
        / str(identity["dataset"])
        / str(identity["encoder"])
        / str(identity["corruption"])
        / f"severity_{int(identity['severity'])}"
        / f"seed_{int(identity['split_seed'])}.json"
    )


def _persist_immutable(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_manifest(payload)
    if path.exists():
        existing = read_json(path)
        verify_manifest_checksum(existing)
        if existing != sealed:
            raise IntegrityError(f"immutable artifact conflicts with current unit: {path}")
        return existing
    write_json_atomic(path, sealed)
    return sealed


def _finite_metrics(
    aligned: np.ndarray,
    reference: np.ndarray,
    *,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    reference_logits: np.ndarray,
) -> dict[str, float]:
    aligned_logits = logits(aligned, weight, bias)
    task = classification_metrics(
        aligned_logits, labels, reference_logits=reference_logits
    )
    logit_delta = aligned_logits - reference_logits
    centered_logit_delta = logit_delta - logit_delta.mean(axis=1, keepdims=True)
    metrics = {
        "reference_prediction_agreement": task["reference_prediction_agreement"],
        "aligned_accuracy": task["accuracy"],
        "centered_logit_mse": float(np.mean(np.square(centered_logit_delta))),
        "paired_feature_mse": float(np.mean(np.square(aligned - reference))),
        "negative_log_likelihood": task["negative_log_likelihood"],
        "expected_calibration_error": task["expected_calibration_error"],
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise IntegrityError("a required rotation-extension metric is non-finite")
    return {key: float(value) for key, value in metrics.items()}


def _split_plan(
    *,
    store: FeatureStore,
    identity: Mapping[str, Any],
    output_root: Path,
    global_seed: int,
) -> dict[str, Any]:
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    excluded_reference, excluded_target, head_fit_ids, head_namespace = (
        _head_exclusions(store)
    )
    plan = build_split_plan(
        reference_ids=reference.ids,
        target_ids=target.ids,
        pairing=store.pairing(),
        seed=_seed(
            global_seed,
            SUITE_NAME,
            identity["dataset"],
            identity["encoder"],
            identity["corruption"],
            identity["severity"],
            identity["split_seed"],
        ),
        evaluation_size=int(identity["evaluation_size"]),
        calibration_sizes=[int(identity["calibration_size"])],
        pair_budgets=[],
        reference_namespace=reference.id_namespace,
        target_namespace=target.id_namespace,
        excluded_reference_ids=excluded_reference,
        excluded_target_ids=excluded_target,
        head_fit_exclusion_required=head_namespace in {
            reference.id_namespace,
            target.id_namespace,
        },
        head_fit_ids_count=int(len(head_fit_ids)),
    )
    return _persist_immutable(
        _artifact_path(output_root, "splits", identity),
        {key: value for key, value in plan.items() if key != "manifest_sha256"},
    )


def _run_unit(
    *,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    discovered: DiscoveredFeatureStore,
    output_root: Path,
    config_sha256: str,
    lock_sha256: str,
    runner_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        provenance.get("schema_version")
        == "marginal_alignment.iclr_extension.play1_provenance.v1"
        and provenance.get("status") == "VALIDATED",
        "unit lacks validated Play-1 provenance",
    )
    _require(
        provenance.get("play1_lock", {}).get("sha256") == lock_sha256,
        "unit Play-1 lock hash differs from validated provenance",
    )
    unit_path = _artifact_path(output_root, "units", identity)
    store_manifest_sha256 = sha256_file(discovered.manifest_path)
    expected_store_records = {
        _feature_store_key(
            row["dataset"], row["encoder"], row["corruption"], row["severity"]
        ): str(row["manifest_sha256"])
        for row in provenance.get("source_feature_store_manifests", [])
    }
    identity_store_key = _feature_store_key(
        identity["dataset"],
        identity["encoder"],
        identity["corruption"],
        identity["severity"],
    )
    _require(
        expected_store_records.get(identity_store_key) == store_manifest_sha256,
        "unit feature-store manifest differs from locked source campaign",
    )
    if unit_path.exists():
        existing = read_json(unit_path)
        verify_manifest_checksum(existing)
        _require(existing.get("schema_version") == UNIT_SCHEMA, "unit schema drift")
        _require(existing.get("status") == "PASS", "resumed unit is not PASS")
        _require(existing.get("identity") == dict(identity), "resumed unit identity drift")
        _require(existing.get("config_sha256") == config_sha256, "resumed config drift")
        _require(existing.get("play1_lock_sha256") == lock_sha256, "resumed Play-1 lock drift")
        _require(existing.get("runner_sha256") == runner_sha256, "resumed runner drift")
        _require(existing.get("provenance") == dict(provenance), "resumed provenance drift")
        _require(
            existing.get("feature_store_manifest_sha256") == store_manifest_sha256,
            "resumed feature-store drift",
        )
        split_path = _artifact_path(output_root, "splits", identity)
        _require(split_path.is_file(), f"resumed split artifact is absent: {split_path}")
        resumed_split = read_json(split_path)
        verify_manifest_checksum(resumed_split)
        _require(
            resumed_split.get("schema_version") == "marginal_alignment.split_plan.v2",
            "resumed split schema drift",
        )
        _require(
            resumed_split.get("manifest_sha256")
            == existing.get("split_plan_sha256"),
            "resumed split manifest differs from sealed unit",
        )
        _require(
            resumed_split.get("evaluation_sha256")
            == existing.get("evaluation_sha256"),
            "resumed split evaluation identity differs from sealed unit",
        )
        expected_split_seed = _seed(
            int(config["global_seed"]),
            SUITE_NAME,
            identity["dataset"],
            identity["encoder"],
            identity["corruption"],
            identity["severity"],
            identity["split_seed"],
        )
        _require(
            int(resumed_split.get("seed", -1)) == expected_split_seed,
            "resumed split seed differs from unit identity",
        )
        calibration_key = str(identity["calibration_size"])
        calibration_record = resumed_split.get("calibrations", {}).get(
            calibration_key, {}
        )
        _require(
            resumed_split.get("calibration_sizes")
            == [int(identity["calibration_size"])]
            and int(calibration_record.get("size", -1))
            == int(identity["calibration_size"])
            and int(resumed_split.get("evaluation", {}).get("size", -1))
            == int(identity["evaluation_size"]),
            "resumed split sizes differ from unit identity",
        )
        return existing

    started = time.time()
    store = load_feature_store(discovered.manifest_path, verify_hashes=True)
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    weight, bias = _head(store)
    plan = _split_plan(
        store=store,
        identity=identity,
        output_root=output_root,
        global_seed=int(config["global_seed"]),
    )
    evaluation = plan["evaluation"]
    calibration = plan["calibrations"][str(identity["calibration_size"])]
    for endpoint in ("reference_ids", "target_ids"):
        evaluation_ids = {
            value.item() if isinstance(value, np.generic) else value
            for value in evaluation[endpoint]
        }
        calibration_ids = {
            value.item() if isinstance(value, np.generic) else value
            for value in calibration[endpoint]
        }
        _require(
            evaluation_ids.isdisjoint(calibration_ids),
            f"calibration and evaluation {endpoint} overlap",
        )
    reference_eval_index = _indices(
        reference.ids, evaluation["reference_ids"], name="evaluation reference"
    )
    target_eval_index = _indices(
        target.ids, evaluation["target_ids"], name="evaluation target"
    )
    reference_fit_index = _indices(
        reference.ids, calibration["reference_ids"], name="calibration reference"
    )
    target_fit_index = _indices(
        target.ids, calibration["target_ids"], name="calibration target"
    )
    reference_fit = np.asarray(
        reference.features[reference_fit_index], dtype=np.float64
    )
    target_fit = np.asarray(target.features[target_fit_index], dtype=np.float64)

    method = config["method"]
    thresholds = method["dataset_mass_thresholds"][identity["dataset"]]
    fitted = fit_bures_pseudo_orientation(
        target_fit,
        reference_fit,
        readout_weight=weight,
        readout_bias=bias,
        covariance_estimator=method["covariance_estimator"],
        relative_floor=float(method["relative_floor"]),
        candidate_ranks=method["candidate_ranks"],
        procrustes_shrinkages=method["procrustes_shrinkages"],
        confidence_threshold=float(method["confidence_threshold"]),
        temperature=float(method["temperature"]),
        validation_fraction=float(method["validation_fraction"]),
        minimum_fit_rows=int(method["minimum_fit_rows"]),
        minimum_validation_rows=int(method["minimum_validation_rows"]),
        minimum_class_mass=float(thresholds["fit"]),
        minimum_validation_class_mass=float(thresholds["validation"]),
        minimum_active_classes=int(method["minimum_active_classes"]),
        minimum_effective_class_mass=float(
            method["minimum_effective_class_mass"]
        ),
        minimum_explained_discrepancy_energy=float(
            method["minimum_explained_discrepancy_energy"]
        ),
        minimum_subspace_stability=float(
            method["minimum_fit_validation_subspace_stability"]
        ),
        null_permutations=int(method["null_permutations"]),
        null_quantile=float(method["null_quantile"]),
        minimum_relative_improvement=float(
            method["minimum_relative_improvement"]
        ),
    )
    _require(fitted.base_map.method == "bures", "constructive method lost Bures baseline")
    _require(
        fitted.base_map.diagnostics["source_covariance"]["covariance_estimator"]
        == "oas",
        "direct baseline is not Bures-OAS",
    )
    if fitted.diagnostics["fallback_used"]:
        _require(
            np.array_equal(fitted.matrix, fitted.base_map.matrix)
            and np.array_equal(fitted.offset, fitted.base_map.offset),
            "identity-orientation fallback is not the exact Bures map",
        )
    # Evaluation features and labels are materialized only after fitting.  The
    # fitter's API receives calibration features and the frozen readout only.
    reference_eval = np.asarray(
        reference.features[reference_eval_index], dtype=np.float64
    )
    target_eval = np.asarray(target.features[target_eval_index], dtype=np.float64)
    labels = _labels(
        reference.labels,
        target.labels,
        reference_eval_index,
        target_eval_index,
    )
    reference_logits = logits(reference_eval, weight, bias)
    base_metrics = _finite_metrics(
        fitted.base_map.apply(target_eval),
        reference_eval,
        labels=labels,
        weight=weight,
        bias=bias,
        reference_logits=reference_logits,
    )
    orientation_metrics = _finite_metrics(
        fitted.apply(target_eval),
        reference_eval,
        labels=labels,
        weight=weight,
        bias=bias,
        reference_logits=reference_logits,
    )
    metric_names = [config["metrics"]["primary"], *config["metrics"]["secondary"]]
    differences = {
        name: float(orientation_metrics[name] - base_metrics[name])
        for name in metric_names
    }
    directions = dict(config["metrics"]["directions"])
    sign_normalized_improvements = {
        name: (
            differences[name]
            if directions[name] == "higher_is_better"
            else -differences[name]
        )
        for name in metric_names
    }
    payload = {
        "schema_version": UNIT_SCHEMA,
        "status": "PASS",
        "identity": dict(identity),
        "config_sha256": config_sha256,
        "play1_lock_sha256": lock_sha256,
        "extension_lock_sha256": provenance["master_lock"]["sha256"],
        "runner_sha256": runner_sha256,
        "provenance": dict(provenance),
        "feature_store_manifest": str(discovered.manifest_path.resolve()),
        "feature_store_manifest_sha256": store_manifest_sha256,
        "split_plan_sha256": plan["manifest_sha256"],
        "evaluation_sha256": plan["evaluation_sha256"],
        "information_audit": {
            "pairing_used_for_split_administration_only": True,
            "pairing_available_to_orientation_fitter": False,
            "pairing_used_by_orientation_fitter": False,
            "calibration_reference_target_row_assignment_available_to_fitter": False,
            "calibration_labels_used": False,
            "target_or_evaluation_labels_available_to_fit": False,
            "frozen_classifier_predictions_used": True,
            "evaluation_labels_used_for_metrics_only": True,
            "evaluation_disjoint_from_calibration": True,
            "calibration_evaluation_id_disjointness_checked_for_both_endpoints": True,
            "selection_bures_whitening_and_orientation_fit_rows_only": True,
            "validation_used_for_rank_and_regularization_selection_only": True,
            "selected_specification_refit_on_all_calibration_before_evaluation": True,
        },
        "metric_metadata": {
            "directions": directions,
            "sign_normalized_improvement": "positive_is_better_for_every_metric",
        },
        "methods": {
            "bures_oas": {
                "metrics": base_metrics,
                "diagnostics": fitted.base_map.diagnostics,
            },
            "bures_pseudo_orientation": {
                "metrics": orientation_metrics,
                "diagnostics": fitted.diagnostics,
            },
        },
        "paired_differences_pseudo_orientation_minus_bures": differences,
        "sign_normalized_improvements_pseudo_orientation_over_bures": (
            sign_normalized_improvements
        ),
        "orientation_harm_by_metric": {
            name: bool(value < 0.0)
            for name, value in sign_normalized_improvements.items()
        },
        "wall_seconds": float(time.time() - started),
    }
    return _persist_immutable(unit_path, payload)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise IntegrityError("bootstrap requires a finite nonempty condition vector")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _nested_cluster_values(
    condition_values: Mapping[tuple[str, str, str], float],
) -> tuple[np.ndarray, int]:
    """Average encoders inside each dataset/corruption bootstrap cluster.

    Encoders evaluated on the same dataset/corruption are deliberately kept
    together in every resample.  This interval describes heterogeneity across
    the finite prespecified panel; it is not population-sampling inference.
    """

    nested: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (dataset, _encoder, corruption), value in condition_values.items():
        nested[(dataset, corruption)].append(float(value))
    if not nested:
        raise IntegrityError("cluster bootstrap has no dataset/corruption cells")
    encoder_counts = {len(values) for values in nested.values()}
    if len(encoder_counts) != 1:
        raise IntegrityError(
            "dataset/corruption clusters contain unequal encoder panels"
        )
    values = np.asarray(
        [np.mean(nested[key]) for key in sorted(nested)], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise IntegrityError("cluster bootstrap contains non-finite values")
    return values, int(next(iter(encoder_counts)))


def _complete_panel_summary(
    units: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    panel: str,
) -> dict[str, Any]:
    selected = [unit for unit in units if unit["identity"]["panel"] == panel]
    metric_names = [config["metrics"]["primary"], *config["metrics"]["secondary"]]
    condition_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    normalized_condition_values: dict[
        tuple[str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for unit in selected:
        identity = unit["identity"]
        key = (identity["dataset"], identity["encoder"], identity["corruption"])
        for metric in metric_names:
            condition_values[key][metric].append(
                unit["paired_differences_pseudo_orientation_minus_bures"][metric]
            )
            normalized_condition_values[key][metric].append(
                unit[
                    "sign_normalized_improvements_pseudo_orientation_over_bures"
                ][metric]
            )
    expected_splits = len(config["split_seeds"])
    for key, metric_rows in condition_values.items():
        for metric in metric_names:
            _require(
                len(metric_rows[metric]) == expected_splits,
                f"condition {key} has an incomplete split set for {metric}",
            )
            _require(
                len(normalized_condition_values[key][metric]) == expected_splits,
                f"condition {key} has incomplete normalized splits for {metric}",
            )
    condition_mean_maps = {
        metric: {
            key: float(np.mean(metric_rows[metric]))
            for key, metric_rows in sorted(condition_values.items())
        }
        for metric in metric_names
    }
    condition_means = {
        metric: np.asarray(list(values.values()), dtype=np.float64)
        for metric, values in condition_mean_maps.items()
    }
    normalized_condition_means = {
        metric: np.asarray([
            np.mean(normalized_condition_values[key][metric])
            for key in sorted(normalized_condition_values)
        ])
        for metric in metric_names
    }
    uncertainty = config["uncertainty"]
    primary = config["metrics"]["primary"]
    primary_clusters, encoders_per_cluster = _nested_cluster_values(
        condition_mean_maps[primary]
    )
    lower, upper = _bootstrap_interval(
        primary_clusters,
        replicates=int(uncertainty["replicates"]),
        seed=int(uncertainty["seed"]) + (0 if panel == "primary" else 1),
        confidence_level=float(uncertainty["confidence_level"]),
    )
    pseudo_diagnostics = [
        unit["methods"]["bures_pseudo_orientation"]["diagnostics"]
        for unit in selected
    ]
    selected_ranks = Counter(str(row["selected_rank"]) for row in pseudo_diagnostics)
    selected_shrinkages = Counter(
        "identity" if row["selected_shrinkage"] is None else str(row["selected_shrinkage"])
        for row in pseudo_diagnostics
    )
    active_counts = [
        len(row["active_fit_classes"])
        for row in pseudo_diagnostics
        if "active_fit_classes" in row
    ]
    return {
        "condition_cells": len(condition_values),
        "null_gate_interpretation": config["method"]["null_interpretation"],
        "dataset_corruption_clusters": int(len(primary_clusters)),
        "encoders_nested_per_cluster": encoders_per_cluster,
        "split_seeds_per_condition": expected_splits,
        "condition_macro_mean_differences": {
            metric: float(np.mean(values))
            for metric, values in condition_means.items()
        },
        "condition_macro_mean_sign_normalized_improvements": {
            metric: float(np.mean(values))
            for metric, values in normalized_condition_means.items()
        },
        "condition_harm_rates": {
            metric: float(np.mean(values < 0.0))
            for metric, values in normalized_condition_means.items()
        },
        "primary_estimand": {
            "metric": primary,
            "pseudo_orientation_minus_bures": float(
                np.mean(condition_means[primary])
            ),
            "sign_normalized_improvement": float(
                np.mean(normalized_condition_means[primary])
            ),
            "direction": config["metrics"]["directions"][primary],
            "cluster_bootstrap_confidence_interval": [lower, upper],
            "confidence_level": float(uncertainty["confidence_level"]),
            "bootstrap_replicates": int(uncertainty["replicates"]),
            "cluster_unit": uncertainty["unit"],
            "encoders": uncertainty["encoders"],
            "interval_interpretation": uncertainty["interval_interpretation"],
        },
        "fallback_rate": float(np.mean([
            bool(row["fallback_used"]) for row in pseudo_diagnostics
        ])),
        "selected_rank_distribution": dict(sorted(selected_ranks.items())),
        "selected_shrinkage_distribution": dict(
            sorted(selected_shrinkages.items())
        ),
        "active_fit_class_count_mean": (
            float(np.mean(active_counts)) if active_counts else None
        ),
        "maximum_orientation_orthogonality_error": float(max(
            row["orientation_orthogonality_error"] for row in pseudo_diagnostics
        )),
        "maximum_fitted_covariance_relative_error": float(max(
            row["fitted_covariance_relative_error"] for row in pseudo_diagnostics
        )),
        "maximum_empirical_covariance_relative_error": float(max(
            row["empirical_covariance_relative_error"] for row in pseudo_diagnostics
        )),
        "orientation_reflection_rate": float(np.mean([
            bool(row["orientation_includes_reflection"])
            for row in pseudo_diagnostics
        ])),
    }


def _write_aggregate(
    *,
    output_root: Path,
    requested_panel: str,
    expected: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    config_sha256: str,
    lock_sha256: str,
    runner_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        provenance.get("schema_version")
        == "marginal_alignment.iclr_extension.play1_provenance.v1"
        and provenance.get("status") == "VALIDATED"
        and provenance.get("play1_lock", {}).get("sha256") == lock_sha256,
        "aggregate lacks validated Play-1 provenance",
    )
    completed: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for identity in expected:
        path = _artifact_path(output_root, "units", identity)
        if not path.exists():
            continue
        unit = read_json(path)
        verify_manifest_checksum(unit)
        _require(unit.get("identity") == dict(identity), f"unit identity mismatch: {path}")
        _require(unit.get("schema_version") == UNIT_SCHEMA, f"unit schema mismatch: {path}")
        _require(unit.get("status") == "PASS", f"unit is not PASS: {path}")
        _require(unit.get("config_sha256") == config_sha256, f"unit config drift: {path}")
        _require(
            unit.get("play1_lock_sha256") == lock_sha256,
            f"unit Play-1 lock drift: {path}",
        )
        _require(unit.get("runner_sha256") == runner_sha256, f"unit runner drift: {path}")
        _require(unit.get("provenance") == dict(provenance), f"unit provenance drift: {path}")
        completed.append(unit)
        artifacts.append({
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
            "manifest_sha256": unit["manifest_sha256"],
        })
    complete = len(completed) == len(expected)
    panel_summaries = {}
    if complete:
        for panel in _panels(config, requested_panel):
            panel_summaries[panel] = _complete_panel_summary(
                completed, config=config, panel=panel
            )
    payload = seal_manifest({
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS" if complete else "RUNNING",
        "requested_panel": requested_panel,
        "config_sha256": config_sha256,
        "play1_lock_sha256": lock_sha256,
        "extension_lock_sha256": provenance["master_lock"]["sha256"],
        "runner_sha256": runner_sha256,
        "provenance": dict(provenance),
        "expected_units": len(expected),
        "completed_units": len(completed),
        "missing_units": len(expected) - len(completed),
        "unit_artifacts": artifacts,
        "panel_summaries": panel_summaries,
    })
    write_json_atomic(output_root / f"AGGREGATE.{requested_panel}.json", payload)
    return payload


def _resolve_environment_path(config: Mapping[str, Any], key: str) -> Path:
    variable = str(config[key])
    value = os.environ.get(variable)
    if not value:
        raise IntegrityError(f"required environment variable is unset: {variable}")
    return Path(value).expanduser().resolve()


def run(
    *,
    config_path: Path,
    requested_panel: str,
    feature_root: Path | None = None,
    output_root: Path | None = None,
    verify_only: bool = False,
    aggregate_only: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    config, lock_path, _, provenance = _load_config(config_path)
    output_root = output_root or _resolve_environment_path(
        config, "output_root_environment"
    )
    expected = _expected_units(config, requested_panel)
    assigned = _sharded_units(
        expected, shard_index=shard_index, num_shards=num_shards
    )
    config_sha256 = sha256_file(config_path)
    lock_sha256 = sha256_file(lock_path)
    runner_sha256 = sha256_file(Path(__file__))
    if aggregate_only:
        _require(not verify_only, "--aggregate-only and --verify-only are mutually exclusive")
        _require(num_shards == 1 and shard_index == 0, (
            "--aggregate-only consumes the full panel and does not accept shard selection"
        ))
        _require(output_root.is_dir(), f"result root is absent: {output_root}")
        aggregate = _write_aggregate(
            output_root=output_root,
            requested_panel=requested_panel,
            expected=expected,
            config=config,
            config_sha256=config_sha256,
            lock_sha256=lock_sha256,
            runner_sha256=runner_sha256,
            provenance=provenance,
        )
        _require(aggregate["status"] == "PASS", (
            f"panel is incomplete: {aggregate['completed_units']}/"
            f"{aggregate['expected_units']} units"
        ))
        return aggregate

    feature_root = feature_root or _resolve_environment_path(
        config, "feature_root_environment"
    )
    _require(feature_root.is_dir(), f"feature root is absent: {feature_root}")
    expected_store_manifests = {
        _feature_store_key(
            row["dataset"], row["encoder"], row["corruption"], row["severity"]
        ): str(row["manifest_sha256"])
        for row in provenance["source_feature_store_manifests"]
    }
    stores = _store_index(
        feature_root,
        expected_manifest_sha256=expected_store_manifests,
    )
    missing = sorted({
        (
            unit["dataset"],
            unit["encoder"],
            unit["corruption"],
            unit["severity"],
        )
        for unit in expected
        if (
            unit["dataset"],
            unit["encoder"],
            unit["corruption"],
            unit["severity"],
        )
        not in stores
    })
    _require(not missing, f"{len(missing)} required stores are missing; first={missing[:5]}")
    if verify_only:
        for key in sorted({
            (unit["dataset"], unit["encoder"], unit["corruption"], unit["severity"])
            for unit in expected
        }):
            load_feature_store(stores[key].manifest_path, verify_hashes=True)
        return {
            "status": "PASS",
            "verified_stores": len({
                (unit["dataset"], unit["encoder"], unit["corruption"], unit["severity"])
                for unit in expected
            }),
            "expected_units": len(expected),
            "num_shards": int(num_shards),
            "assigned_units": len(assigned),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    for number, identity in enumerate(assigned, start=1):
        key = (
            identity["dataset"],
            identity["encoder"],
            identity["corruption"],
            identity["severity"],
        )
        print(
            f"[rotation] shard={shard_index}/{num_shards} "
            f"{number}/{len(assigned)} "
            f"{identity['panel']} {identity['dataset']}/{identity['encoder']} "
            f"{identity['corruption']} s{identity['severity']} "
            f"seed={identity['split_seed']}",
            flush=True,
        )
        _run_unit(
            config=config,
            identity=identity,
            discovered=stores[key],
            output_root=output_root,
            config_sha256=config_sha256,
            lock_sha256=lock_sha256,
            runner_sha256=runner_sha256,
            provenance=provenance,
        )
        # Multiple shards intentionally share one immutable unit directory.
        # Only a single-shard run may rewrite the live aggregate; concurrent
        # aggregate writers would otherwise race even though unit paths are
        # disjoint.  Multi-shard campaigns finalize explicitly after every
        # shard has exited via --aggregate-only.
        if num_shards == 1:
            _write_aggregate(
                output_root=output_root,
                requested_panel=requested_panel,
                expected=expected,
                config=config,
                config_sha256=config_sha256,
                lock_sha256=lock_sha256,
                runner_sha256=runner_sha256,
                provenance=provenance,
            )
    if num_shards == 1:
        aggregate = _write_aggregate(
            output_root=output_root,
            requested_panel=requested_panel,
            expected=expected,
            config=config,
            config_sha256=config_sha256,
            lock_sha256=lock_sha256,
            runner_sha256=runner_sha256,
            provenance=provenance,
        )
        _require(aggregate["status"] == "PASS", "single-shard rotation panel is incomplete")
        return aggregate
    return {
        "status": "PASS",
        "schema_version": "beyond_marginals.rotation_extension.shard.v1",
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "assigned_units": len(assigned),
        "completed_assigned_units": len(assigned),
        "expected_full_panel_units": len(expected),
        "full_panel_status": "REQUIRES_EXPLICIT_AGGREGATION",
        "finalization_command": "rerun with --aggregate-only after all shards pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--panel", choices=("primary", "diagnostic", "all"), default="all"
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    arguments = parser.parse_args()
    result = run(
        config_path=arguments.config.expanduser().resolve(),
        requested_panel=arguments.panel,
        feature_root=(
            None
            if arguments.feature_root is None
            else arguments.feature_root.expanduser().resolve()
        ),
        output_root=(
            None
            if arguments.output_root is None
            else arguments.output_root.expanduser().resolve()
        ),
        verify_only=arguments.verify_only,
        aggregate_only=arguments.aggregate_only,
        shard_index=arguments.shard_index,
        num_shards=arguments.num_shards,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
