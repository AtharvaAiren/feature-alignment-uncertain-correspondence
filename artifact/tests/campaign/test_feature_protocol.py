from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_json
from marginal_alignment.campaign.cache_import import CIFAR100_C_CORRUPTIONS
from marginal_alignment.campaign.feature_protocol import (
    MODEL_EVIDENCE_SCHEMA,
    canonical_block_index,
    checkpoint_tree_fingerprint,
    deterministic_stratified_positions,
    fit_frozen_readout,
    validate_declared_blocks,
)
from marginal_alignment.campaign.legacy_prepare import _source_blocks, audit_reuse_pair
from marginal_alignment.campaign.raw_extract import (
    ArrayImageDataset, _verify_raw_array_manifest, _write_raw_array_manifest,
    cifar100c_condition_labels, extract_dataset,
)


def test_canonical_cifar_blocks_are_condition_complete_and_contiguous():
    blocks = canonical_block_index(
        "cifar100_c", corruptions=CIFAR100_C_CORRUPTIONS, severities=(1, 3, 5)
    )
    assert len(blocks) == 57
    assert blocks[0] == {
        "corruption": "brightness", "severity": 1, "start": 0, "stop": 10_000,
    }
    assert blocks[-1]["stop"] == 570_000
    validated = validate_declared_blocks(
        blocks, dataset="cifar100_c", total_rows=570_000,
        require_conditions=[(name, severity) for name in CIFAR100_C_CORRUPTIONS for severity in (1, 3, 5)],
    )
    assert validated == blocks


def test_declared_blocks_never_fill_an_undeclared_gap():
    blocks = canonical_block_index("cifar100_c", corruptions=("brightness",), severities=(1, 3))
    blocks[1]["start"] += 1
    with pytest.raises(IntegrityError, match="noncontiguous"):
        validate_declared_blocks(blocks, dataset="cifar100_c", total_rows=20_000)


def test_legacy_list_block_index_is_normalized_only_from_sequential_records(
    tmp_path: Path,
):
    target = tmp_path / "imagenet_c" / "dinov3"
    target.mkdir(parents=True)
    records = [
        {
            "block": index,
            "corruption": corruption,
            "severity": severity,
            "n_classes": 1000,
        }
        for index, (corruption, severity) in enumerate(
            (name, level)
            for name in (
                "brightness", "contrast", "defocus_blur", "elastic_transform",
                "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
                "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
                "zoom_blur",
            )
            for level in range(1, 6)
        )
    ]
    (target / "block_index.json").write_text(json.dumps(records))
    blocks = _source_blocks(
        target, {}, dataset="imagenet_c", total_rows=4_750_000,
    )
    assert len(blocks) == 95
    assert blocks[0] == {
        "corruption": "brightness", "severity": 1, "start": 0, "stop": 50_000,
    }
    assert blocks[-1]["stop"] == 4_750_000

    records[4]["block"] = 99
    (target / "block_index.json").write_text(json.dumps(records))
    with pytest.raises(IntegrityError, match="sequential"):
        _source_blocks(target, {}, dataset="imagenet_c", total_rows=4_750_000)


def test_stratified_head_subset_is_exact_balanced_and_deterministic():
    labels = np.repeat(np.arange(4), 20)
    first = deterministic_stratified_positions(labels, shots_per_class=7, seed=1729)
    second = deterministic_stratified_positions(labels, shots_per_class=7, seed=1729)
    assert np.array_equal(first, second)
    assert np.array_equal(np.bincount(labels[first], minlength=4), np.full(4, 7))


def test_checkpoint_fingerprint_hashes_bytes_not_directory_name(tmp_path: Path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    config = tmp_path / "config.json"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    config.write_text('{"hidden_size":1024}')
    digest, records = checkpoint_tree_fingerprint(tmp_path)
    assert len(records) == 3
    assert digest == sha256_json(records)
    previous = digest
    second.write_bytes(b"changed")
    changed, _ = checkpoint_tree_fingerprint(tmp_path)
    assert changed != previous
    second.write_bytes(b"two")
    config.write_text('{"hidden_size":768}')
    changed_config, _ = checkpoint_tree_fingerprint(tmp_path)
    assert changed_config != previous


@pytest.mark.parametrize("solver", ["saga", "lbfgs"])
def test_binary_logistic_head_is_expanded_to_two_raw_compatible_logits(solver):
    rng = np.random.default_rng(4)
    features = rng.normal(size=(240, 4)).astype(np.float32)
    labels = (features[:, 0] - 0.4 * features[:, 1] > 0).astype(np.int64)
    weight, bias, report = fit_frozen_readout(
        features, labels, np.arange(len(labels)),
        realization={
            "solver": solver, "C": 1.0, "fit_intercept": True,
            "max_iter": 2000, "tol": 1e-3,
        },
        seed=1729,
    )
    assert weight.shape == (2, 4)
    assert bias.shape == (2,)
    assert report["status"] == "PASS"
    assert np.mean(np.argmax(features @ weight.T + bias, axis=1) == labels) > 0.9


def test_raw_extractor_writes_condition_labels_at_global_offset():
    class Encoder:
        encoder_id = "fixture"
        device = "cpu"

        @staticmethod
        def encode(images):
            return np.asarray([[index, index + 0.5] for index in range(len(images))], dtype=np.float32)

    images = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    dataset = ArrayImageDataset(images, np.asarray([4, 5, 6]))
    features = np.full((8, 2), -1, dtype=np.float32)
    labels = np.full(8, -1, dtype=np.int64)
    extract_dataset(
        Encoder(), dataset, features, destination_offset=5, batch_size=2, workers=0,
        label_destination=labels,
    )
    assert np.array_equal(labels[5:], [4, 5, 6])
    assert np.all(labels[:5] == -1)
    assert np.all(features[:5] == -1)


def test_cifar_c_labels_accept_one_vector_or_five_proven_repeats():
    shared = np.arange(10_000, dtype=np.int64) % 100
    assert np.array_equal(cifar100c_condition_labels(shared, 5), shared)
    repeated = np.tile(shared, 5)
    assert np.array_equal(cifar100c_condition_labels(repeated, 3), shared)
    corrupted = repeated.copy()
    corrupted[20_000] = 99
    with pytest.raises(IntegrityError, match="not five exact repeats"):
        cifar100c_condition_labels(corrupted, 3)
    with pytest.raises(IntegrityError, match="expected one 10k"):
        cifar100c_condition_labels(shared[:9999], 1)


def test_raw_array_manifest_rejects_changed_feature_bytes(tmp_path: Path):
    features = tmp_path / "features.npy"
    labels = tmp_path / "labels.npy"
    np.save(features, np.zeros((4, 2), dtype=np.float32), allow_pickle=False)
    np.save(labels, np.arange(4, dtype=np.int64), allow_pickle=False)
    manifest = tmp_path / "RAW_TARGET_MANIFEST.json"
    files = {"features": features, "labels": labels}
    _write_raw_array_manifest(
        manifest, fingerprint="f" * 64, files=files, metadata={"dataset": "fixture"},
    )
    assert _verify_raw_array_manifest(
        manifest, fingerprint="f" * 64, files=files, required_if_any_final=True,
    )
    np.save(features, np.ones((4, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(IntegrityError, match="SHA-256 changed"):
        _verify_raw_array_manifest(
            manifest, fingerprint="f" * 64, files=files, required_if_any_final=True,
        )


def test_raw_array_manifest_rejects_missing_identity_sidecar(tmp_path: Path):
    features = tmp_path / "features.npy"
    ids = tmp_path / "ids.npy"
    np.save(features, np.zeros((4, 2), dtype=np.float32), allow_pickle=False)
    np.save(ids, np.arange(4, dtype=np.int64), allow_pickle=False)
    manifest = tmp_path / "RAW_TARGET_MANIFEST.json"
    files = {"features": features, "ids": ids}
    _write_raw_array_manifest(
        manifest, fingerprint="a" * 64, files=files, metadata={"dataset": "fixture"},
    )
    ids.unlink()
    with pytest.raises(IntegrityError, match="partial"):
        _verify_raw_array_manifest(
            manifest, fingerprint="a" * 64, files=files, required_if_any_final=True,
        )


def _save(path: Path, shape: tuple[int, ...], dtype) -> None:
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[:] = 0
    value.flush()


def test_reuse_audit_requires_exact_model_evidence_then_passes(tmp_path: Path):
    source = tmp_path / "source"
    reference = source / "cifar100" / "dinov3"
    target = source / "cifar100c" / "dinov3"
    reference.mkdir(parents=True)
    target.mkdir(parents=True)
    _save(reference / "train_features.npy", (50_000, 2), np.float32)
    _save(reference / "train_labels.npy", (50_000,), np.int64)
    _save(reference / "test_features.npy", (10_000, 2), np.float32)
    _save(reference / "test_labels.npy", (10_000,), np.int64)
    _save(target / "test_features.npy", (570_000, 2), np.float32)
    _save(target / "test_labels.npy", (570_000,), np.int64)
    blocks = canonical_block_index(
        "cifar100_c", corruptions=CIFAR100_C_CORRUPTIONS, severities=(1, 3, 5)
    )
    (target / "metadata.json").write_text(json.dumps({
        "blocks": blocks,
        "pairing_provenance": "official_cifar100c_canonical_test_index",
    }))
    config = {
        "encoders": {"dinov3": {
            "model_id": "locked/model", "expected_dimension": 2,
            "feature_layer": "cls", "pooling": "cls",
        }},
        "datasets": {"cifar100_c": {
            "rows_per_condition": 10_000,
            "head_protocol": {"kind": "clean_linear_probe", "seed": 1729, "max_train_rows": 50_000},
            "reference_split": "test", "head_fit_split": "train",
        }},
        "readout_realization": {},
    }
    blocked = audit_reuse_pair(
        source_root=source, dataset="cifar100_c", encoder="dinov3", config=config
    )
    assert blocked["status"] == "BLOCKED"
    assert "model evidence" in blocked["blockers"][0]
    evidence = {
        "schema_version": MODEL_EVIDENCE_SCHEMA,
        "encoder_id": "dinov3", "model_id": "locked/model",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "architecture_config_sha256": hashlib.sha256(b"architecture").hexdigest(),
        "preprocessing_sha256": hashlib.sha256(b"preprocess").hexdigest(),
        "feature_layer": "cls", "pooling": "cls", "expected_dimension": 2,
    }
    (reference / "MODEL_EVIDENCE.json").write_text(json.dumps(evidence))
    ready = audit_reuse_pair(
        source_root=source, dataset="cifar100_c", encoder="dinov3", config=config
    )
    assert ready["status"] == "READY"
    assert ready["block_count"] == 57
