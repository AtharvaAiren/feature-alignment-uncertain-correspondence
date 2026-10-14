from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.artifacts import sha256_file
from marginal_alignment.campaign.feature_store import (
    DiscoveredFeatureStore,
    build_native_manifest,
)
from scripts.iclr_extension.run_rotation import (
    _complete_panel_summary,
    _expected_units,
    _load_config,
    _persist_immutable,
    _run_unit,
    _sharded_units,
    _store_index,
    _validate_config_against_lock,
    _validate_play1_lock,
    _write_aggregate,
    PLAY1_EXECUTION_ARTIFACTS,
    PLAY1_LOCK_STATUS,
    PLAY1_REQUIRED_IMPORTED_DEPENDENCIES,
    PLAY1_SOURCE_FILES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "iclr_extension" / "rotation.json"


def _toy_store(root: Path) -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(81)
    rows, dimension, classes = 400, 6, 10
    prototypes = np.zeros((classes, dimension))
    angles = 2.0 * np.pi * np.arange(classes) / classes
    prototypes[:, :2] = 2.0 * np.column_stack([
        np.cos(angles), np.sin(angles)
    ])
    labels = np.repeat(np.arange(classes), rows // classes)
    rng.shuffle(labels)
    reference = prototypes[labels] + rng.normal(scale=0.4, size=(rows, dimension))
    angle = np.deg2rad(15.0)
    rotation = np.eye(dimension)
    rotation[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    target = reference @ rotation
    weight = 3.0 * prototypes
    arrays = {
        "reference_features": reference.astype(np.float32),
        "target_features": target.astype(np.float32),
        "reference_ids": np.arange(rows, dtype=np.int64),
        "target_ids": np.arange(1000, 1000 + rows, dtype=np.int64),
        "reference_labels": labels.astype(np.int64),
        "target_labels": labels.astype(np.int64),
        "pair_reference_ids": np.arange(rows, dtype=np.int64),
        "pair_target_ids": np.arange(1000, 1000 + rows, dtype=np.int64),
        "head_weight": weight.astype(np.float32),
        "head_bias": np.zeros(classes, dtype=np.float32),
        "head_fit_ids": np.arange(10_000, 10_032, dtype=np.int64),
    }
    paths = {}
    for role, value in arrays.items():
        path = root / f"{role}.npy"
        np.save(path, value)
        paths[role] = path.name
    head_fit_hash = sha256_file(root / "head_fit_ids.npy")
    return build_native_manifest(
        root,
        dataset="toy",
        encoder="tiny",
        arrays=paths,
        id_provenance={
            "reference": {
                "verified": True,
                "kind": "fixture",
                "source": "test",
                "namespace": "toy:clean",
            },
            "target": {
                "verified": True,
                "kind": "fixture",
                "source": "test",
                "namespace": "toy:corrupt",
            },
        },
        pairing={
            "status": "available",
            "reference_ids_role": "pair_reference_ids",
            "target_ids_role": "pair_target_ids",
            "provenance": {
                "verified": True,
                "kind": "fixture_pair",
                "source": "test",
            },
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata={
            "corruption": "contrast",
            "severity": 5,
            "model_provenance": {
                "encoder_id": "tiny",
                "checkpoint_sha256": "0" * 64,
                "preprocessing_sha256": "1" * 64,
                "feature_layer": "fixture",
                "pooling": "fixture",
                "head_protocol": "fixture",
                "head_fit_ids_sha256": head_fit_hash,
                "head_fit_ids_role": "head_fit_ids",
                "head_fit_id_namespace": "toy:head_fit",
                "reference_id_namespace": "toy:clean",
                "reference_split": "test",
                "head_fit_split": "train",
            },
        },
    )


class RotationRunnerTests(unittest.TestCase):
    def test_config_matches_master_lock_and_declares_complete_panels(self):
        config, play1_lock_path, _, provenance = _load_config(CONFIG_PATH)
        primary = _expected_units(config, "primary")
        diagnostic = _expected_units(config, "diagnostic")
        complete = _expected_units(config, "all")

        self.assertEqual(len(primary), 150)
        self.assertEqual(len(diagnostic), 90)
        self.assertEqual(len(complete), 240)
        identities = {
            tuple(sorted(row.items())) for row in complete
        }
        self.assertEqual(len(identities), len(complete))
        self.assertEqual(play1_lock_path.name, "iclr_extension_play1_lock.json")
        self.assertEqual(provenance["status"], "VALIDATED")
        self.assertEqual(provenance["play1_lock"]["status"], PLAY1_LOCK_STATUS)
        self.assertEqual(set(provenance["source_files"]), PLAY1_SOURCE_FILES)
        self.assertEqual(
            set(read_json(play1_lock_path)["required_imported_dependencies"]),
            PLAY1_REQUIRED_IMPORTED_DEPENDENCIES,
        )
        self.assertEqual(
            set(provenance["source_execution_evidence"]),
            PLAY1_EXECUTION_ARTIFACTS,
        )
        self.assertEqual(len(provenance["source_feature_store_manifests"]), 48)
        self.assertEqual(
            provenance["source_external_input_sha256"],
            "3a8348746ec7a924968c68a29c34e9fa4ac0ea32c6d728d40dc82e0e6844ba97",
        )
        self.assertEqual(
            set(provenance["extension_runtime_identity"]["packages"]),
            {"numpy", "scipy", "sklearn"},
        )
        self.assertIsInstance(provenance["source_runtime_identity"], dict)
        self.assertFalse(
            provenance["source_execution_evidence"]["real_rows"][
                "required_at_play1_runtime"
            ]
        )

    def test_config_drift_from_the_prespecification_fails_closed(self):
        config, _, lock, _ = _load_config(CONFIG_PATH)
        drifted = deepcopy(config)
        drifted["method"]["confidence_threshold"] = 0.7
        with self.assertRaisesRegex(IntegrityError, "method drift"):
            _validate_config_against_lock(drifted, lock)

    def test_play1_lock_rejects_source_execution_and_source_code_drift(self):
        config, lock_path, master, _ = _load_config(CONFIG_PATH)
        master_path = PROJECT_ROOT / config["extension_lock"]
        locked = read_json(lock_path)

        execution_drift = deepcopy(locked)
        execution_drift["source_execution_evidence"]["artifacts"][
            "real_manifest"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(IntegrityError, "source execution artifact changed"):
            _validate_play1_lock(
                config=config,
                config_path=CONFIG_PATH,
                lock_path=lock_path,
                lock=execution_drift,
                master_path=master_path,
                master=master,
            )

        external_input_drift = deepcopy(locked)
        external_input_drift["source_execution_evidence"]["artifacts"][
            "external_input_manifest"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(IntegrityError, "source execution artifact changed"):
            _validate_play1_lock(
                config=config,
                config_path=CONFIG_PATH,
                lock_path=lock_path,
                lock=external_input_drift,
                master_path=master_path,
                master=master,
            )

        source_drift = deepcopy(locked)
        source_drift["source_files"][
            "src/marginal_alignment/iclr_extension/pseudo_orientation.py"
        ] = "0" * 64
        with self.assertRaisesRegex(IntegrityError, "locked Play-1 source changed"):
            _validate_play1_lock(
                config=config,
                config_path=CONFIG_PATH,
                lock_path=lock_path,
                lock=source_drift,
                master_path=master_path,
                master=master,
            )

        dependency_inventory_drift = deepcopy(locked)
        dependency_inventory_drift["required_imported_dependencies"] = [
            "src/marginal_alignment/linalg.py"
        ]
        with self.assertRaisesRegex(
            IntegrityError, "imported-dependency inventory drift"
        ):
            _validate_play1_lock(
                config=config,
                config_path=CONFIG_PATH,
                lock_path=lock_path,
                lock=dependency_inventory_drift,
                master_path=master_path,
                master=master,
            )

        master_drift = deepcopy(locked)
        master_drift["master_lock_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(IntegrityError, "extension master lock differs"):
            _validate_play1_lock(
                config=config,
                config_path=CONFIG_PATH,
                lock_path=lock_path,
                lock=master_drift,
                master_path=master_path,
                master=master,
            )

    def test_shards_are_disjoint_complete_and_deterministic(self):
        config, _, _, _ = _load_config(CONFIG_PATH)
        expected = _expected_units(config, "all")
        shards = [
            _sharded_units(expected, shard_index=index, num_shards=7)
            for index in range(7)
        ]
        identity = lambda row: tuple(sorted(row.items()))
        shard_sets = [{identity(row) for row in rows} for rows in shards]
        self.assertEqual(sum(map(len, shard_sets)), len(expected))
        self.assertEqual(set.union(*shard_sets), {identity(row) for row in expected})
        for left, values in enumerate(shard_sets):
            for other in shard_sets[left + 1:]:
                self.assertTrue(values.isdisjoint(other))
        self.assertEqual(
            shards[3], _sharded_units(expected, shard_index=3, num_shards=7)
        )
        with self.assertRaisesRegex(IntegrityError, "shard_index"):
            _sharded_units(expected, shard_index=7, num_shards=7)

    def test_unit_artifact_is_content_sealed_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit.json"
            first = _persist_immutable(path, {"schema_version": "test", "value": 1})
            second = _persist_immutable(path, {"schema_version": "test", "value": 1})
            verify_manifest_checksum(first)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(IntegrityError, "immutable artifact conflicts"):
                _persist_immutable(path, {"schema_version": "test", "value": 2})

    def test_store_index_rejects_a_substituted_base_campaign_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _toy_store(Path(directory) / "store")
            key = ("toy", "tiny", "contrast", 5)
            accepted = _store_index(
                manifest.parent,
                expected_manifest_sha256={key: sha256_file(manifest)},
            )
            self.assertEqual(set(accepted), {key})
            with self.assertRaisesRegex(
                IntegrityError, "differs from locked source campaign"
            ):
                _store_index(
                    manifest.parent,
                    expected_manifest_sha256={key: "0" * 64},
                )

    def test_aggregate_averages_splits_before_condition_macro(self):
        config, _, _, _ = _load_config(CONFIG_PATH)
        metrics = [config["metrics"]["primary"], *config["metrics"]["secondary"]]
        units = []
        for corruption, difference in (("contrast", 1.0), ("zoom_blur", 3.0)):
            for encoder in ("clip", "dinov3"):
                for seed in config["split_seeds"]:
                    units.append({
                        "identity": {
                            "panel": "primary",
                            "dataset": "cifar100_c",
                            "encoder": encoder,
                            "corruption": corruption,
                            "split_seed": seed,
                        },
                        "paired_differences_pseudo_orientation_minus_bures": {
                            metric: difference for metric in metrics
                        },
                        "sign_normalized_improvements_pseudo_orientation_over_bures": {
                            metric: (
                                difference
                                if config["metrics"]["directions"][metric]
                                == "higher_is_better"
                                else -difference
                            )
                            for metric in metrics
                        },
                        "methods": {
                            "bures_pseudo_orientation": {
                                "diagnostics": {
                                    "fallback_used": False,
                                    "selected_rank": 2,
                                    "selected_shrinkage": 0.1,
                                    "active_fit_classes": list(range(8)),
                                    "orientation_orthogonality_error": 1e-14,
                                    "fitted_covariance_relative_error": 1e-12,
                                    "empirical_covariance_relative_error": 1e-4,
                                    "orientation_includes_reflection": False,
                                }
                            }
                        },
                    })
        summary = _complete_panel_summary(units, config=config, panel="primary")
        self.assertEqual(summary["condition_cells"], 4)
        self.assertEqual(summary["dataset_corruption_clusters"], 2)
        self.assertEqual(summary["encoders_nested_per_cluster"], 2)
        self.assertEqual(
            summary["primary_estimand"]["cluster_unit"],
            "dataset_corruption_condition_cluster",
        )
        for metric in metrics:
            self.assertEqual(
                summary["condition_macro_mean_differences"][metric], 2.0
            )
        self.assertEqual(
            summary["primary_estimand"]["pseudo_orientation_minus_bures"], 2.0
        )
        self.assertEqual(
            summary["primary_estimand"]["sign_normalized_improvement"], 2.0
        )
        self.assertEqual(
            summary["null_gate_interpretation"],
            "conservative_heuristic_selection_gate_not_a_significance_test_or_identification_claim",
        )
        self.assertEqual(
            summary["condition_macro_mean_sign_normalized_improvements"]
            ["centered_logit_mse"],
            -2.0,
        )

    def test_sealed_store_runs_both_methods_resumes_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = _toy_store(temporary / "store")
            output = temporary / "results"
            base_config, _, _, provenance = _load_config(CONFIG_PATH)
            provenance = deepcopy(provenance)
            provenance["source_feature_store_manifests"] = [{
                "dataset": "toy",
                "encoder": "tiny",
                "corruption": "contrast",
                "severity": 5,
                "manifest_sha256": sha256_file(manifest),
            }]
            config = deepcopy(base_config)
            config["split_seeds"] = [101]
            config["method"].update({
                "candidate_ranks": [2],
                "confidence_threshold": 0.55,
                "minimum_fit_rows": 32,
                "minimum_validation_rows": 16,
                "minimum_active_classes": 2,
                "dataset_mass_thresholds": {
                    "toy": {"fit": 2.0, "validation": 0.5}
                },
                "minimum_explained_discrepancy_energy": 0.5,
                "minimum_fit_validation_subspace_stability": 0.0,
            })
            identity = {
                "panel": "primary",
                "dataset": "toy",
                "encoder": "tiny",
                "corruption": "contrast",
                "severity": 5,
                "split_seed": 101,
                "calibration_size": 200,
                "evaluation_size": 100,
            }
            discovered = DiscoveredFeatureStore(
                kind="native_v2",
                root=manifest.parent,
                manifest_path=manifest,
                dataset="toy",
                encoder="tiny",
            )
            hashes = {
                "config_sha256": "a" * 64,
                "lock_sha256": provenance["play1_lock"]["sha256"],
                "runner_sha256": "c" * 64,
                "provenance": provenance,
            }
            first = _run_unit(
                config=config,
                identity=identity,
                discovered=discovered,
                output_root=output,
                **hashes,
            )
            resumed = _run_unit(
                config=config,
                identity=identity,
                discovered=discovered,
                output_root=output,
                **hashes,
            )
            self.assertEqual(first, resumed)
            split_path = next((output / "splits").rglob("*.json"))
            original_split_bytes = split_path.read_bytes()
            split_path.unlink()
            with self.assertRaisesRegex(
                IntegrityError, "resumed split artifact is absent"
            ):
                _run_unit(
                    config=config,
                    identity=identity,
                    discovered=discovered,
                    output_root=output,
                    **hashes,
                )
            split_path.write_bytes(original_split_bytes)

            tampered_split = read_json(split_path)
            unsigned_tampered_split = {
                key: value for key, value in tampered_split.items()
                if key != "manifest_sha256"
            }
            unsigned_tampered_split["seed"] = int(
                unsigned_tampered_split["seed"]
            ) + 1
            write_json_atomic(split_path, seal_manifest(unsigned_tampered_split))
            with self.assertRaisesRegex(
                IntegrityError, "resumed split manifest differs from sealed unit"
            ):
                _run_unit(
                    config=config,
                    identity=identity,
                    discovered=discovered,
                    output_root=output,
                    **hashes,
                )
            split_path.write_bytes(original_split_bytes)
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(
                set(first["methods"]),
                {"bures_oas", "bures_pseudo_orientation"},
            )
            self.assertFalse(
                first["information_audit"][
                    "pairing_available_to_orientation_fitter"
                ]
            )
            self.assertTrue(
                first["information_audit"][
                    "pairing_used_for_split_administration_only"
                ]
            )
            self.assertTrue(
                first["information_audit"][
                    "calibration_evaluation_id_disjointness_checked_for_both_endpoints"
                ]
            )
            self.assertEqual(
                first["metric_metadata"]["sign_normalized_improvement"],
                "positive_is_better_for_every_metric",
            )
            self.assertEqual(first["provenance"], provenance)
            self.assertEqual(
                first["play1_lock_sha256"], provenance["play1_lock"]["sha256"]
            )
            self.assertEqual(
                first["extension_lock_sha256"], provenance["master_lock"]["sha256"]
            )
            aggregate = _write_aggregate(
                output_root=output,
                requested_panel="primary",
                expected=[identity],
                config=config,
                **hashes,
            )
            self.assertEqual(aggregate["status"], "PASS")
            self.assertEqual(aggregate["completed_units"], 1)
            self.assertEqual(aggregate["provenance"], provenance)
            verify_manifest_checksum(aggregate)


if __name__ == "__main__":
    unittest.main()
