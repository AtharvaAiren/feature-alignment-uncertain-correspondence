from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_file, write_json_atomic
from marginal_alignment.campaign.feature_store import (
    build_native_manifest,
    discover_feature_stores,
    load_feature_store,
)


class NativeFeatureStoreTests(unittest.TestCase):
    def _store(self, root: Path) -> Path:
        rng = np.random.default_rng(4)
        arrays = {
            "reference_features": rng.normal(size=(24, 5)).astype(np.float32),
            "target_features": rng.normal(size=(24, 5)).astype(np.float32),
            "reference_ids": np.asarray([f"r{index}" for index in range(24)]),
            "target_ids": np.asarray([f"t{index}" for index in range(24)]),
            "reference_labels": np.arange(24, dtype=np.int64) % 3,
            "target_labels": np.arange(24, dtype=np.int64) % 3,
            "pair_reference_ids": np.asarray([f"r{index}" for index in range(24)]),
            "pair_target_ids": np.asarray([f"t{index}" for index in range(24)]),
            "head_weight": rng.normal(size=(3, 5)).astype(np.float32),
            "head_bias": np.zeros(3, dtype=np.float32),
            "head_fit_ids": np.arange(100, 110, dtype=np.int64),
        }
        paths = {}
        for role, value in arrays.items():
            path = root / f"{role}.npy"
            np.save(path, value)
            paths[role] = path.name
        head_fit_hash = sha256_file(root / "head_fit_ids.npy")
        return build_native_manifest(
            root,
            dataset="unit_dataset",
            encoder="unit_encoder",
            arrays=paths,
            id_provenance={
                "reference": {
                    "verified": True,
                    "kind": "dataset_sample_key",
                    "source": "unit fixture reference index",
                    "namespace": "unit:reference",
                },
                "target": {
                    "verified": True,
                    "kind": "dataset_sample_key",
                    "source": "unit fixture target index",
                    "namespace": "unit:target",
                },
            },
            pairing={
                "status": "available",
                "reference_ids_role": "pair_reference_ids",
                "target_ids_role": "pair_target_ids",
                "provenance": {
                    "verified": True,
                    "kind": "natural_clean_corruption_identity",
                    "source": "unit fixture construction",
                },
            },
            head={"weight_role": "head_weight", "bias_role": "head_bias"},
            metadata={
                "model_provenance": {
                    "encoder_id": "unit_encoder", "checkpoint_sha256": "0" * 64,
                    "preprocessing_sha256": "1" * 64, "feature_layer": "fixture",
                    "pooling": "fixture", "head_protocol": "fixture",
                    "head_fit_ids_sha256": head_fit_hash,
                    "head_fit_ids_role": "head_fit_ids",
                    "head_fit_id_namespace": "unit:head_fit",
                    "reference_id_namespace": "unit:reference",
                    "reference_split": "test",
                    "head_fit_split": "train",
                }
            },
        )

    def test_native_store_is_mmap_and_pairing_is_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._store(Path(directory))
            store = load_feature_store(manifest)
            self.assertEqual(store.kind, "native_v2")
            self.assertTrue(store.hashes_verified)
            self.assertIsInstance(store.endpoint("reference").features, np.memmap)
            self.assertEqual(store.endpoint("target").features.shape, (24, 5))
            reference, target = store.pairing().validated()
            self.assertEqual(reference[3], "r3")
            self.assertEqual(target[3], "t3")

    def test_skipping_file_hashes_disables_pairing_information(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._store(Path(directory))
            store = load_feature_store(manifest, verify_hashes=False)
            self.assertFalse(store.hashes_verified)
            with self.assertRaisesRegex(IntegrityError, "hashes were not verified"):
                store.pairing()

    def test_native_store_detects_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._store(root)
            values = np.load(root / "target_features.npy", mmap_mode="r+")
            values[0, 0] += 7.0
            values.flush()
            with self.assertRaisesRegex(IntegrityError, "SHA-256 mismatch"):
                load_feature_store(manifest)

    def test_pairing_provenance_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(8)
            for name, value in {
                "reference_features": rng.normal(size=(5, 2)).astype(np.float32),
                "target_features": rng.normal(size=(5, 2)).astype(np.float32),
                "reference_ids": np.arange(5, dtype=np.int64),
                "target_ids": np.arange(10, 15, dtype=np.int64),
                "head_weight": rng.normal(size=(2, 2)).astype(np.float32),
                "head_bias": np.zeros(2, dtype=np.float32),
                "head_fit_ids": np.arange(100, 105, dtype=np.int64),
            }.items():
                np.save(root / f"{name}.npy", value)
            manifest = build_native_manifest(
                root,
                dataset="unpaired",
                encoder="identity",
                arrays={name: f"{name}.npy" for name in (
                    "reference_features", "target_features", "reference_ids", "target_ids",
                    "head_weight", "head_bias", "head_fit_ids",
                )},
                id_provenance={
                    endpoint: {
                        "verified": True,
                        "kind": "endpoint_key",
                        "source": "fixture",
                        "namespace": endpoint,
                    }
                    for endpoint in ("reference", "target")
                },
                metadata={
                    "model_provenance": {
                        "encoder_id": "identity", "checkpoint_sha256": "0" * 64,
                        "preprocessing_sha256": "1" * 64, "feature_layer": "fixture",
                        "pooling": "fixture", "head_protocol": "fixture",
                        "head_fit_ids_sha256": sha256_file(root / "head_fit_ids.npy"),
                        "head_fit_ids_role": "head_fit_ids",
                        "head_fit_id_namespace": "head_fit",
                        "reference_id_namespace": "reference",
                        "reference_split": "test",
                        "head_fit_split": "train",
                    }
                },
                head={"weight_role": "head_weight", "bias_role": "head_bias"},
            )
            store = load_feature_store(manifest)
            with self.assertRaisesRegex(IntegrityError, "no usable correspondence"):
                store.pairing()


class LegacyServerAdapterTests(unittest.TestCase):
    def _legacy(self, root: Path) -> Path:
        store = root / "cifar100c" / "dinov3"
        store.mkdir(parents=True)
        rng = np.random.default_rng(12)
        for split, count in (("train", 14), ("test", 9)):
            np.save(store / f"{split}_features.npy", rng.normal(size=(count, 4)).astype(np.float32))
            np.save(store / f"{split}_labels.npy", np.arange(count, dtype=np.int64) % 2)
        files = {
            path.name: {"sha256": sha256_file(path)}
            for path in sorted(store.glob("*.npy"))
        }
        write_json_atomic(store / "manifest.json", {
            "verified": True,
            "dataset": "cifar100c",
            "encoder": "dinov3",
            "files": files,
        })
        return store

    def test_discovers_and_adapts_server_layout_without_claiming_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            root = self._legacy(cache)
            discovered = discover_feature_stores([cache])
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].kind, "legacy_server_cache")
            store = load_feature_store(root)
            train, test = store.endpoint("train"), store.endpoint("test")
            self.assertIsInstance(train.features, np.memmap)
            self.assertEqual(len(train.ids), 14)
            self.assertNotEqual(train.id_namespace, test.id_namespace)
            self.assertFalse(train.id_provenance["cross_endpoint_correspondence"])
            with self.assertRaisesRegex(IntegrityError, "no usable correspondence"):
                store.pairing()

    def test_legacy_manifest_without_per_file_hashes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "test_features.npy", np.ones((4, 2), dtype=np.float32))
            np.save(root / "test_labels.npy", np.zeros(4, dtype=np.int64))
            write_json_atomic(root / "manifest.json", {"verified": True, "files": {}})
            with self.assertRaisesRegex(IntegrityError, r"no \*_features"):
                load_feature_store(root)


if __name__ == "__main__":
    unittest.main()
