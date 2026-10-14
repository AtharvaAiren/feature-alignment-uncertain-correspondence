from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_file, write_json_atomic
from marginal_alignment.campaign.cache_import import (
    EXPECTED_STORE_COUNT,
    MATERIALIZATION_SCHEMA,
    convert_indexed_imagenet_cache,
    estimate_v2_materialization_space,
    expected_conditions,
    import_condition,
    seal_legacy_arrays,
    validate_store_completeness,
)
from marginal_alignment.campaign.feature_store import load_feature_store


class IdempotentCacheImportTests(unittest.TestCase):
    def _sources(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        reference = root / "reference"
        target = root / "target"
        reference.mkdir()
        target.mkdir()
        rng = np.random.default_rng(17)
        np.save(reference / "features.npy", rng.normal(size=(8, 4)).astype(np.float32))
        np.save(reference / "labels.npy", np.arange(8, dtype=np.int64) % 2)
        np.save(reference / "head_weight.npy", rng.normal(size=(2, 4)).astype(np.float32))
        np.save(reference / "head_bias.npy", np.zeros(2, dtype=np.float32))
        np.save(reference / "head_fit_ids.npy", np.arange(100, 106, dtype=np.int64))
        np.save(target / "features.npy", rng.normal(size=(16, 4)).astype(np.float32))
        np.save(target / "labels.npy", np.tile(np.arange(8, dtype=np.int64) % 2, 2))
        seal_legacy_arrays(reference)
        seal_legacy_arrays(target)
        model = {
            "encoder_id": "unit_encoder",
            "checkpoint_sha256": "a" * 64,
            "preprocessing_sha256": "b" * 64,
            "feature_layer": "fixture",
            "pooling": "fixture",
            "head_protocol": "fixture",
            "head_fit_ids_sha256": sha256_file(reference / "head_fit_ids.npy"),
            "head_fit_ids_file": "head_fit_ids.npy",
            "head_fit_id_namespace": "fixture:train",
            "reference_id_namespace": "fixture:reference",
            "reference_split": "test",
            "head_fit_split": "train",
        }
        return reference, target, model

    def _import(
        self, root: Path, reference: Path, target: Path, model: dict[str, object],
        *, severity: int = 1,
    ) -> Path:
        ids = np.arange(8, dtype=np.int64)
        return import_condition(
            source_reference_root=reference,
            source_target_root=target,
            destination=root / "stores" / "cifar100_c" / "unit_encoder" /
            "brightness" / "severity_1",
            dataset="cifar100_c",
            encoder="unit_encoder",
            reference_feature_file="features.npy",
            reference_label_file="labels.npy",
            target_feature_file="features.npy",
            target_label_file="labels.npy",
            reference_rows=slice(0, 8),
            target_rows=slice(0, 8),
            reference_ids=ids,
            target_ids=ids + 10_000,
            pairing_reference_ids=ids,
            pairing_target_ids=ids + 10_000,
            pairing_provenance={
                "verified": True,
                "kind": "natural_clean_corruption_identity",
                "source": "unit canonical identity",
            },
            head_weight_file="head_weight.npy",
            head_bias_file="head_bias.npy",
            metadata={"corruption": "brightness", "severity": severity},
            model_provenance=model,
            shared_root=root / "stores" / "_shared" / "cifar100_c" / "unit_encoder",
        )

    def test_reuses_exact_condition_and_hardlinks_invariant_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, target, model = self._sources(root)
            manifest = self._import(root, reference, target, model)
            condition = manifest.parent
            shared = root / "stores" / "_shared" / "cifar100_c" / "unit_encoder"
            for role in ("reference_features", "reference_labels", "head_weight", "head_fit_ids"):
                condition_stat = (condition / f"{role}.npy").stat()
                shared_stat = (shared / f"{role}.npy").stat()
                self.assertEqual((condition_stat.st_dev, condition_stat.st_ino),
                                 (shared_stat.st_dev, shared_stat.st_ino))
                self.assertGreaterEqual(condition_stat.st_nlink, 2)
            target_inode = (condition / "target_features.npy").stat().st_ino
            manifest_mtime = manifest.stat().st_mtime_ns
            second = self._import(root, reference, target, model)
            self.assertEqual(second, manifest)
            self.assertEqual(second.stat().st_mtime_ns, manifest_mtime)
            self.assertEqual((condition / "target_features.npy").stat().st_ino, target_inode)

    def test_tampering_or_expected_metadata_change_rebuilds_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, target, model = self._sources(root)
            manifest = self._import(root, reference, target, model)
            condition = manifest.parent
            values = np.load(condition / "target_features.npy", mmap_mode="r+")
            values[0, 0] += 100.0
            values.flush()
            del values
            rebuilt = self._import(root, reference, target, model)
            store = load_feature_store(rebuilt)
            expected = np.load(target / "features.npy")[:8]
            np.testing.assert_array_equal(store.endpoint("target").features, expected)
            changed = self._import(root, reference, target, model, severity=3)
            changed_store = load_feature_store(changed)
            self.assertEqual(changed_store.manifest["metadata"]["severity"], 3)
            self.assertEqual(
                changed_store.manifest["metadata"]["materialization"]["schema_version"],
                MATERIALIZATION_SCHEMA,
            )

    def test_locked_topology_and_incomplete_report_are_explicit(self) -> None:
        self.assertEqual(len(expected_conditions()), EXPECTED_STORE_COUNT)
        self.assertEqual(EXPECTED_STORE_COUNT, 306)
        with tempfile.TemporaryDirectory() as directory:
            report = validate_store_completeness(directory)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["expected_count"], 306)
        self.assertEqual(report["missing_count"], 306)
        self.assertEqual(report["valid_count"], 0)

    def test_cached_source_seal_still_detects_mid_process_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, target, model = self._sources(root)
            self._import(root, reference, target, model)
            values = np.load(reference / "features.npy", mmap_mode="r+")
            values[0, 0] += 1.0
            values.flush()
            del values
            with self.assertRaisesRegex(IntegrityError, "source array hash mismatch"):
                self._import(root, reference, target, model)

    def test_disk_preflight_is_data_dependent_and_reports_peak_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            for reference_name, target_name in (
                ("cifar100", "cifar100c"), ("imagenet", "imagenet_c"),
            ):
                reference = legacy / reference_name / "unit_encoder"
                target = legacy / target_name / "unit_encoder"
                reference.mkdir(parents=True)
                target.mkdir(parents=True)
                np.save(reference / "features.npy", np.ones((2, 3), dtype=np.float32))
                np.save(reference / "labels.npy", np.zeros(2, dtype=np.int64))
                np.save(reference / "head_weight.npy", np.ones((2, 3), dtype=np.float32))
                np.save(reference / "head_bias.npy", np.zeros(2, dtype=np.float32))
                np.save(reference / "head_fit_ids.npy", np.arange(2, dtype=np.int64))
                np.save(target / "features.npy", np.ones((2, 3), dtype=np.float32))
                model = {
                    "encoder_id": "unit_encoder", "checkpoint_sha256": "a" * 64,
                    "preprocessing_sha256": "b" * 64, "feature_layer": "fixture",
                    "pooling": "fixture", "head_protocol": "fixture",
                    "head_fit_ids_sha256": sha256_file(reference / "head_fit_ids.npy"),
                    "head_fit_ids_file": "head_fit_ids.npy",
                    "head_fit_id_namespace": "fixture:train",
                    "reference_id_namespace": "fixture:reference",
                    "reference_split": "test",
                    "head_fit_split": "train",
                }
                write_json_atomic(reference / "metadata.json", {
                    "reference_feature_file": "features.npy",
                    "reference_label_file": "labels.npy",
                    "model_provenance": model,
                })
                write_json_atomic(target / "metadata.json", {
                    "target_feature_file": "features.npy",
                })
            report = estimate_v2_materialization_space(
                legacy_root=legacy,
                destination_root=root / "stores",
                encoders=("unit_encoder",),
                severities=(1,),
            )
            self.assertEqual(len(report["details"]), 2)
            self.assertGreater(report["required_free_bytes"], report["estimated_net_new_bytes"])
            self.assertGreaterEqual(report["safety_margin_bytes"], 2 * 1024**3)

    def test_imagenet_conversion_preserves_head_fit_identity_namespace(self) -> None:
        """ImageNet V2 IDs must remain the sealed WNID/basename keys.

        A previous converter revision replaced the endpoint and pairing IDs with
        integer row numbers while retaining string-valued head-fit IDs.  The
        campaign correctly rejected that store because the head-fit identities
        could not be a subset of the declared reference namespace.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "imagenet" / "unit_encoder"
            target = root / "imagenet_c" / "unit_encoder"
            reference.mkdir(parents=True)
            target.mkdir(parents=True)
            rng = np.random.default_rng(29)
            identity_keys = np.asarray([
                "n00000001/image_0001.JPEG",
                "n00000001/image_0002.JPEG",
                "n00000002/image_0001.JPEG",
                "n00000002/image_0002.JPEG",
            ])
            labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
            head_fit_ids = identity_keys[[0, 2]]
            np.save(reference / "test_features.npy", rng.normal(size=(4, 3)).astype(np.float32))
            np.save(reference / "test_labels.npy", labels)
            np.save(reference / "sample_ids.npy", identity_keys)
            np.save(reference / "head_weight.npy", rng.normal(size=(2, 3)).astype(np.float32))
            np.save(reference / "head_bias.npy", np.zeros(2, dtype=np.float32))
            np.save(reference / "head_fit_ids.npy", head_fit_ids)
            np.save(target / "test_features.npy", rng.normal(size=(4, 3)).astype(np.float32))
            np.save(target / "test_labels.npy", labels)
            np.save(target / "sample_ids.npy", identity_keys)
            model = {
                "encoder_id": "unit_encoder",
                "checkpoint_sha256": "a" * 64,
                "preprocessing_sha256": "b" * 64,
                "feature_layer": "fixture",
                "pooling": "fixture",
                "head_protocol": "fixture",
                "head_fit_ids_sha256": sha256_file(reference / "head_fit_ids.npy"),
                "head_fit_ids_file": "head_fit_ids.npy",
                "head_fit_id_namespace": "imagenet:val:relative_image_path",
                "reference_id_namespace": "imagenet:val:relative_image_path",
                "reference_split": "val",
                "head_fit_split": "val",
            }
            write_json_atomic(reference / "metadata.json", {
                "reference_feature_file": "test_features.npy",
                "reference_label_file": "test_labels.npy",
                "sample_ids_file": "sample_ids.npy",
                "head_weight_file": "head_weight.npy",
                "head_bias_file": "head_bias.npy",
                "model_provenance": model,
            })
            write_json_atomic(target / "metadata.json", {
                "target_feature_file": "test_features.npy",
                "target_label_file": "test_labels.npy",
                "sample_ids_file": "sample_ids.npy",
                "sample_id_kind": "relative_image_path",
                "blocks": [{
                    "corruption": "fog", "severity": 5, "start": 0, "stop": 4,
                }],
            })
            seal_legacy_arrays(reference)
            seal_legacy_arrays(target)

            manifests = convert_indexed_imagenet_cache(
                reference_root=reference,
                target_root=target,
                destination_root=root / "stores",
                encoder="unit_encoder",
                corruptions=["fog"],
                severities=[5],
                rows_per_condition=4,
            )
            self.assertEqual(len(manifests), 1)
            store = load_feature_store(manifests[0])
            np.testing.assert_array_equal(store.endpoint("reference").ids, identity_keys)
            np.testing.assert_array_equal(store.endpoint("target").ids, identity_keys)
            paired_reference, paired_target = store.pairing().validated()
            self.assertEqual(paired_reference, identity_keys.tolist())
            self.assertEqual(paired_target, identity_keys.tolist())
            stored_fit_ids = np.asarray(store.array("head_fit_ids"))
            self.assertTrue(set(stored_fit_ids.tolist()).issubset(
                set(store.endpoint("reference").ids.tolist())
            ))
            self.assertEqual(
                store.manifest["metadata"]["endpoint_identity_encoding"],
                "sealed_sample_identity_keys_v1",
            )


if __name__ == "__main__":
    unittest.main()
