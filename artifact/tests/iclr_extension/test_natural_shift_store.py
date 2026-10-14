from __future__ import annotations

import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_json
from marginal_alignment.campaign.feature_store import load_feature_store as real_load_store
from marginal_alignment.iclr_extension import natural_shift_store as stores


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_class_index(path: Path, count: int) -> tuple[str, ...]:
    wnids = tuple(f"n{index:08d}" for index in range(count))
    path.write_text(json.dumps({
        str(index): [wnid, f"class {index}"]
        for index, wnid in enumerate(wnids)
    }), encoding="utf-8")
    return wnids


def _write_tar_from_tree(archive: Path, root: Path, prefix: str) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=f"{prefix}/{path.relative_to(root)}")


def test_v2_provenance_binds_archive_integer_layout_and_class_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stores, "EXPECTED_IMAGENET_CLASSES", 3)
    monkeypatch.setitem(stores.DATASET_CONTRACTS, "imagenet_v2", {
        **stores.DATASET_CONTRACTS["imagenet_v2"], "rows": 6, "classes": 3,
        "rows_per_class": 2,
    })
    class_index = tmp_path / "imagenet_class_index.json"
    _write_class_index(class_index, 3)
    root = tmp_path / "imagenetv2-matched-frequency-format-val"
    for class_index_value in range(3):
        directory = root / str(class_index_value)
        directory.mkdir(parents=True)
        for image_index in range(2):
            (directory / f"{image_index}.jpeg").write_bytes(
                f"{class_index_value}:{image_index}".encode("ascii")
            )
    archive = tmp_path / "v2.tar.gz"
    _write_tar_from_tree(archive, root, root.name)
    output = tmp_path / "v2_provenance.json"
    stores.freeze_natural_shift_provenance(
        dataset="imagenet_v2",
        dataset_root=root,
        archive=archive,
        class_index_path=class_index,
        output=output,
    )
    validated = stores.validate_natural_shift_provenance(output)
    assert len(validated.discovery.paths) == 6
    assert validated.discovery.labels.tolist() == [0, 0, 1, 1, 2, 2]
    assert validated.discovery.wnids.tolist() == [
        "n00000000", "n00000000", "n00000001", "n00000001",
        "n00000002", "n00000002",
    ]
    (root / "1" / "0.jpeg").write_bytes(b"changed")
    with pytest.raises(IntegrityError, match="tree/order changed"):
        stores.validate_natural_shift_provenance(output)


def test_checked_in_v2_config_matches_store_preparation_contract() -> None:
    config, source = stores._load_v2_config(
        PROJECT_ROOT / "configs" / "iclr_extension" /
        "play3_optimization_gate.json"
    )
    assert source.is_file()
    assert config["datasets"]["imagenet_v2"]["expected_target_rows"] == 10_000
    assert config["datasets"]["imagenet_r"]["expected_target_rows"] == 30_000


def test_r_discovery_requires_exact_wnid_intersection_and_total_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stores, "EXPECTED_IMAGENET_CLASSES", 3)
    monkeypatch.setitem(stores.DATASET_CONTRACTS, "imagenet_r", {
        **stores.DATASET_CONTRACTS["imagenet_r"], "rows": 3, "classes": 2,
    })
    index_path = tmp_path / "classes.json"
    wnids = _write_class_index(index_path, 3)
    index = stores.load_official_class_index(index_path)
    root = tmp_path / "imagenet-r"
    for wnid, count in ((wnids[0], 1), (wnids[2], 2)):
        directory = root / wnid
        directory.mkdir(parents=True)
        for row in range(count):
            (directory / f"{row}.jpg").write_bytes(f"{wnid}:{row}".encode())
    discovery = stores.discover_natural_shift("imagenet_r", root, index)
    assert discovery.intersection_wnids == (wnids[0], wnids[2])
    assert discovery.labels.tolist() == [0, 2, 2]
    (root / wnids[0] / "extra.jpg").write_bytes(b"extra")
    with pytest.raises(IntegrityError, match="needs 3 unique target rows"):
        stores.discover_natural_shift("imagenet_r", root, index)


class _ReferenceStore:
    def __init__(self, wnids: tuple[str, ...]) -> None:
        ids = []
        labels = []
        fit_ids = []
        for label, wnid in enumerate(wnids):
            current = [f"{wnid}/ILSVRC2012_val_{row:08d}.JPEG" for row in range(50)]
            ids.extend(current)
            labels.extend([label] * 50)
            fit_ids.extend(current[:25])
        self._ids = np.asarray(ids)
        self._labels = np.asarray(labels, dtype=np.int64)
        self._fit = np.asarray(fit_ids)
        self.records = {"head_fit_ids": {}, "head_weight": {}, "head_bias": {}}
        self.manifest = {
            "head": {"weight_role": "head_weight", "bias_role": "head_bias"},
            "metadata": {"model_provenance": {
                "head_fit_ids_role": "head_fit_ids",
                "head_protocol": {"shots_per_class": 25},
            }},
        }

    def endpoint(self, name: str):
        assert name == "reference"
        return SimpleNamespace(
            features=np.arange(len(self._ids), dtype=np.float32)[:, None],
            labels=self._labels,
            ids=self._ids,
            id_namespace="imagenet:val",
        )

    def array(self, role: str) -> np.ndarray:
        return {
            "head_fit_ids": self._fit,
            "head_weight": np.zeros((1000, 1), dtype=np.float32),
            "head_bias": np.zeros(1000, dtype=np.float32),
        }[role]


def test_reference_intersection_excludes_all_head_fit_ids() -> None:
    wnids = tuple(f"n{index:08d}" for index in range(1000))
    class_index = stores.OfficialClassIndex(
        path=Path("/unused"),
        file_sha256="0" * 64,
        wnids_by_index=wnids,
        canonical_mapping_sha256="1" * 64,
    )
    source = _ReferenceStore(wnids)
    selected, evidence = stores._reference_arrays(
        source,
        class_index=class_index,
        intersection_wnids=wnids[:200],
        expected_rows=5_000,
    )
    assert selected["reference_features"].shape == (5_000, 1)
    assert set(selected["reference_wnids"].tolist()) == set(wnids[:200])
    assert not set(selected["reference_ids"].tolist()).intersection(
        set(selected["head_fit_ids"].tolist())
    )
    assert set(evidence["reference_counts_by_wnid"].values()) == {25}


def test_store_metadata_hashes_real_wnid_arrays_and_disallows_outcomes(
    tmp_path: Path
) -> None:
    wnids = ["n00000000", "n00000001"]
    arrays = {
        "head_fit_ids": np.asarray(["fit"]),
        "reference_ids": np.asarray(["ref0", "ref1"]),
        "reference_wnids": np.asarray(wnids),
        "target_wnids": np.asarray(wnids),
    }
    for role, value in arrays.items():
        with (tmp_path / f"{role}.npy").open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
    extraction = {"schema_version": "fixture", "outcomes_read": False}
    metadata = stores._store_metadata(
        dataset="imagenet_v2",
        encoder_id="clip",
        provenance_binding={"natural_shift_provenance": {}},
        provenance_manifest_sha256="0" * 64,
        intersection_wnids=wnids,
        reference_source_manifest_sha256="1" * 64,
        source_binding={},
        model_provenance={},
        class_identity={},
        arrays_root=tmp_path,
        target_extraction=extraction,
        v2_config_sha256="2" * 64,
    )
    preparation = metadata["optimization_gate_preparation"]
    assert preparation["outcomes_read"] is False
    assert preparation["class_intersection_wnids_sha256"] == sha256_json(wnids)
    assert metadata["class_intersection"]["target_wnids_role"] == "target_wnids"


def test_structural_label_checkpoint_is_removed_before_publication(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "target_labels.npy"
    with label_path.open("wb") as handle:
        np.save(handle, np.asarray([0, 1], dtype=np.int64), allow_pickle=False)
    stores._remove_structural_label_checkpoint(label_path)
    assert not label_path.exists()
    with pytest.raises(IntegrityError, match="absent before removal"):
        stores._remove_structural_label_checkpoint(label_path)


def test_completed_block_labels_can_be_recreated_without_feature_compute(
    tmp_path: Path,
) -> None:
    expected = np.asarray([4, 5, 6], dtype=np.int64)
    feature_path = tmp_path / "target_features.npy"
    with feature_path.open("wb") as handle:
        np.save(handle, np.arange(6, dtype=np.float32).reshape(3, 2))
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    state = {
        "completed_blocks": [{
            "start": 0,
            "stop": 3,
            "features_sha256": stores._block_sha256(features),
            "labels_sha256": stores._block_sha256(expected),
        }]
    }
    recreated = tmp_path / "target_labels.npy"
    stores._save_npy_fsynced(recreated, expected)
    labels = np.load(recreated, mmap_mode="r", allow_pickle=False)
    assert stores._validate_completed_blocks(
        state,
        features=features,
        labels=labels,
        expected_labels=expected,
        rows=3,
        block_rows=3,
    ) == 3


def test_build_state_binding_uses_canonical_json_identity(tmp_path: Path) -> None:
    """A freshly persisted HF config must survive JSON key normalization."""

    binding = {
        "encoder_evidence": {
            "architecture_config": {
                "config": {"id2label": {0: "zero", 1: "one"}}
            }
        }
    }
    fingerprint = sha256_json(binding)
    state_path = tmp_path / "BUILD_STATE.json"
    stores._write_build_state(state_path, {
        "schema_version": stores.BUILD_STATE_SCHEMA,
        "build_fingerprint": fingerprint,
        "build_binding": binding,
        "phase": "extracting",
    })

    restored = stores._read_state(state_path)
    assert restored["build_binding"] != binding
    assert stores._build_state_matches_binding(
        restored, binding=binding, fingerprint=fingerprint
    )

    restored["build_binding"]["encoder_evidence"]["architecture_config"][
        "config"
    ]["id2label"]["0"] = "changed"
    assert not stores._build_state_matches_binding(
        restored, binding=binding, fingerprint=fingerprint
    )


class _TinySketchStore:
    def __init__(self, root: Path, wnids: tuple[str, ...]) -> None:
        root.mkdir()
        self.root = root
        self.manifest_path = root / "FEATURE_STORE_MANIFEST.json"
        self.manifest_path.write_text("{}\n", encoding="utf-8")
        self.dataset = "imagenet_sketch"
        self.encoder = "clip"
        self._reference = SimpleNamespace(
            features=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            labels=np.asarray([0, 1], dtype=np.int64),
            ids=np.asarray([f"{wnids[0]}/r.jpg", f"{wnids[1]}/r.jpg"]),
        )
        # The first two indexed atoms intentionally have identical features.
        self._target = SimpleNamespace(
            features=np.asarray([
                [1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]
            ], dtype=np.float32),
            labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            ids=np.asarray([
                f"{wnids[0]}/a.jpg", f"{wnids[0]}/b.jpg",
                f"{wnids[1]}/a.jpg", f"{wnids[1]}/b.jpg",
            ]),
        )
        self._arrays = {
            "head_fit_ids": np.asarray([f"{wnids[0]}/fit.jpg"]),
            "head_weight": np.zeros((2, 2), dtype=np.float32),
            "head_bias": np.zeros(2, dtype=np.float32),
        }
        self.manifest = {
            "head": {"weight_role": "head_weight", "bias_role": "head_bias"},
            "id_provenance": {
                "reference": {
                    "verified": True, "kind": "fixture", "source": "fixture",
                    "namespace": "imagenet:val",
                },
                "target": {
                    "verified": True, "kind": "fixture", "source": "fixture",
                    "namespace": "imagenet_sketch:relative_path",
                },
            },
            "metadata": {
                "imagenet_sketch_provenance": {
                    "schema_version": (
                        "marginal_alignment.iclr_extension."
                        "imagenet_sketch_provenance.v2"
                    ),
                    "class_index_sha256": sha256_json(list(wnids)),
                },
                "reference_class_identity": {"status": "PASS"},
                "model_provenance": {
                    "encoder_id": "clip",
                    "model_id": "openai/clip-vit-large-patch14",
                    "checkpoint_sha256": "1" * 64,
                    "preprocessing_sha256": "2" * 64,
                    "feature_layer": "vision_model.post_layernorm+visual_projection",
                    "pooling": "projected_class_token",
                    "head_protocol": {"kind": "fixture"},
                    "head_fit_ids_sha256": "3" * 64,
                    "head_fit_ids_role": "head_fit_ids",
                    "head_fit_id_namespace": "imagenet:val",
                    "reference_id_namespace": "imagenet:val",
                    "reference_split": "val_without_fit",
                    "head_fit_split": "val_fit",
                },
            },
        }

    def endpoint(self, name: str):
        value = self._reference if name == "reference" else self._target
        return SimpleNamespace(**vars(value), id_namespace=(
            "imagenet:val" if name == "reference"
            else "imagenet_sketch:relative_path"
        ))

    def array(self, role: str) -> np.ndarray:
        return self._arrays[role]


def test_sketch_upgrade_publishes_indexed_duplicates_without_endpoint_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wnids = ("n00000000", "n00000001")
    source = _TinySketchStore(tmp_path / "source", wnids)
    provenance_path = tmp_path / "sketch_provenance.json"
    provenance_path.write_text("{}\n", encoding="utf-8")
    dataset_root = tmp_path / "raw_sketch"
    dataset_root.mkdir()
    discovery = SimpleNamespace(
        sample_ids=np.asarray([
            f"{wnids[0]}/a.jpg", f"{wnids[0]}/b.jpg",
            f"{wnids[1]}/a.jpg", f"{wnids[1]}/b.jpg",
        ]),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        wnids=wnids,
        file_tree_sha256="4" * 64,
        paths=(Path("a"), Path("b"), Path("c"), Path("d")),
        duplicate_digest_groups=1,
        duplicate_rows_beyond_first=1,
    )
    provenance = SimpleNamespace(
        manifest_path=provenance_path,
        manifest={
            "schema_version": (
                "marginal_alignment.iclr_extension."
                "imagenet_sketch_provenance.v2"
            ),
            "class_index_sha256": sha256_json(list(wnids)),
        },
        discovery=discovery,
        dataset_root=dataset_root,
    )
    v2_config = {
        "encoders": ["clip"],
        "encoder_contracts": {"clip": {
            "model_id": "openai/clip-vit-large-patch14",
            "expected_dimension": 2,
            "feature_layer": "vision_model.post_layernorm+visual_projection",
            "pooling": "projected_class_token",
        }},
    }
    v2_path = tmp_path / "v2.json"
    v2_path.write_text("{}\n", encoding="utf-8")
    v1_path = tmp_path / "v1.json"
    v1_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(stores, "EXPECTED_SKETCH_REFERENCE_ROWS", 2)
    monkeypatch.setattr(stores, "EXPECTED_SKETCH_TARGET_ROWS", 4)
    monkeypatch.setattr(stores, "EXPECTED_SKETCH_REFERENCE_ROWS_PER_CLASS", 1)
    monkeypatch.setattr(stores, "_load_v2_config", lambda path: (v2_config, v2_path))
    monkeypatch.setattr(
        stores, "validate_imagenet_sketch_provenance", lambda *args, **kwargs: provenance
    )
    monkeypatch.setattr(
        stores, "load_locked_play3_config",
        lambda path: ({
            "base_campaign_feature_stores": {"natural_reference_condition": {}}
        }, v1_path, {}, PROJECT_ROOT),
    )
    monkeypatch.setattr(
        stores, "_load_locked_base_campaign_feature_store_evidence",
        lambda *args: object(),
    )
    monkeypatch.setattr(
        stores, "_validate_store_provenance_binding",
        lambda *args, **kwargs: {
            "schema_version": "fixture", "status": "PASS",
            "actual_manifest_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        stores,
        "load_feature_store",
        lambda path, verify_hashes=True: (
            source
            if Path(path).expanduser().resolve() == source.manifest_path.resolve()
            else real_load_store(path, verify_hashes=verify_hashes)
        ),
    )
    output = tmp_path / "published"
    manifest = stores.upgrade_imagenet_sketch_feature_store(
        v2_config_path=v2_path,
        v1_config_path=v1_path,
        provenance_path=provenance_path,
        source_store_path=source.manifest_path,
        destination=output,
        encoder_id="clip",
        dataset_root=dataset_root,
        archive=tmp_path / "unused.zip",
    )
    published = real_load_store(manifest, verify_hashes=True)
    assert published.endpoint("reference").labels is None
    assert published.endpoint("target").labels is None
    assert not (output / "reference_labels.npy").exists()
    assert not (output / "target_labels.npy").exists()
    assert len(published.endpoint("target").features) == 4
    assert np.array_equal(
        published.endpoint("target").features[0],
        published.endpoint("target").features[1],
    )
    binding = published.manifest["metadata"]["imagenet_sketch_provenance"]
    assert binding["indexed_atom_policy"] == "preserve_all_rows_and_unique_sample_ids"
    assert binding["duplicate_rows_beyond_first"] == 1
