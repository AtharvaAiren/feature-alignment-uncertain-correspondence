from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_file
from marginal_alignment.campaign.feature_store import build_native_manifest
from marginal_alignment.iclr_extension.ceiling import CorrespondenceCeilingProtocol
import marginal_alignment.iclr_extension.natural_shift as natural_shift


class _TrackingStore:
    def __init__(self, arrays: dict[str, np.ndarray], *, pairing_status: str = "unavailable"):
        self.arrays = arrays
        self.calls: list[str] = []
        self.hashes_verified = True
        self.dataset = "imagenet_sketch"
        self.encoder = "fixture_encoder"
        self.pairing_spec = {
            "status": pairing_status,
            "reason": "independent endpoint collections",
        }
        self.endpoint_roles = {
            "reference": {
                "features": "reference_features",
                "ids": "reference_ids",
                "labels": "reference_labels",
            },
            "target": {
                "features": "target_features",
                "ids": "target_ids",
                "labels": "target_labels",
            },
        }
        self.manifest = {
            "head": {
                "weight_role": "head_weight",
                "bias_role": "head_bias",
            },
            "metadata": {},
        }

    def array(self, role: str) -> np.ndarray:
        self.calls.append(role)
        if "label" in role or "id" in role or "pair" in role:
            raise AssertionError(f"forbidden role was accessed: {role}")
        return self.arrays[role]

    def pairing(self):  # pragma: no cover - a call is itself a test failure
        raise AssertionError("pairing() must not be called")


def _source_reference_binding(
    *, encoder: str, manifest_sha256: str = "4" * 64
) -> dict[str, object]:
    return {
        "schema_version": (
            "marginal_alignment.iclr_extension.base_campaign_store_binding.v1"
        ),
        "status": "PASS",
        "external_input_manifest_file_sha256": "5" * 64,
        "external_input_sha256": "6" * 64,
        "condition": {
            "dataset": "imagenet_c",
            "encoder": encoder,
            "corruption": "brightness",
            "severity": 5,
        },
        "expected_manifest_sha256": manifest_sha256,
        "actual_manifest_sha256": manifest_sha256,
    }


@pytest.fixture
def tiny_imagenet_sketch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(natural_shift, "EXPECTED_CLASSES", 2)
    monkeypatch.setattr(natural_shift, "EXPECTED_IMAGES_PER_CLASS", 4)
    monkeypatch.setattr(natural_shift, "EXPECTED_ROWS", 8)
    wnids = ("n00000001", "n00000002")
    reference = tmp_path / "reference"
    dataset = tmp_path / "sketch"
    for wnid in wnids:
        (reference / wnid).mkdir(parents=True)
        (dataset / wnid).mkdir(parents=True)
        for index in range(4):
            (dataset / wnid / f"image_{index}.JPEG").write_bytes(
                f"{wnid}:{index}".encode("ascii")
            )
    archive = tmp_path / "imagenet-sketch.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for wnid in wnids:
            for index in range(4):
                image = dataset / wnid / f"image_{index}.JPEG"
                handle.write(
                    image,
                    arcname=f"ImageNet-Sketch/{wnid}/{image.name}",
                )
    manifest = tmp_path / "provenance.json"
    natural_shift.freeze_imagenet_sketch_provenance(
        dataset_root=dataset,
        reference_root=reference,
        archive=archive,
        output=manifest,
    )
    return dataset, reference, archive, manifest, wnids


def test_freeze_validates_exact_tree_and_builds_unpaired_plan(
    tiny_imagenet_sketch,
) -> None:
    dataset, _, archive, manifest, wnids = tiny_imagenet_sketch
    validated = natural_shift.validate_imagenet_sketch_provenance(manifest)
    assert validated.discovery.wnids == wnids
    assert len(validated.discovery.paths) == 8
    plan = natural_shift.create_imagenet_sketch_build_plan(manifest)
    assert plan.feature_store_pairing["status"] == "unavailable"
    assert np.array_equal(plan.target_labels, np.repeat([0, 1], 4))
    assert plan.provenance_manifest_sha256 == sha256_file(manifest)
    assert plan.archive_path == archive.resolve()
    assert plan.dataset_root == dataset.resolve()


def test_freeze_hashes_uneven_source_and_selects_first_basenames(
    tiny_imagenet_sketch,
    tmp_path: Path,
) -> None:
    dataset, reference, archive, _, wnids = tiny_imagenet_sketch
    manifest = tmp_path / "uneven_provenance.json"
    extra = dataset / wnids[0] / "zz_extra.JPEG"
    extra.write_bytes(b"outcome-blind-extra-source-row")
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as handle:
        handle.write(
            extra,
            arcname=f"ImageNet-Sketch/{wnids[0]}/{extra.name}",
        )

    natural_shift.freeze_imagenet_sketch_provenance(
        dataset_root=dataset,
        reference_root=reference,
        archive=archive,
        output=manifest,
    )
    validated = natural_shift.validate_imagenet_sketch_provenance(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert len(validated.discovery.paths) == 8
    assert extra.resolve() not in validated.discovery.paths
    assert validated.discovery.source_rows == 9
    assert validated.discovery.source_class_counts == {
        wnids[0]: 5,
        wnids[1]: 4,
    }
    assert payload["selection_policy"] == (
        "lexicographically_first_50_basenames_per_wnid"
    )
    assert payload["archive"]["normalized_source_image_rows"] == 9
    assert payload["archive"]["normalized_selected_image_rows"] == 8


def test_freeze_is_immutable_and_tampering_is_detected(
    tiny_imagenet_sketch,
) -> None:
    dataset, reference, archive, manifest, _ = tiny_imagenet_sketch
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        natural_shift.freeze_imagenet_sketch_provenance(
            dataset_root=dataset,
            reference_root=reference,
            archive=archive,
            output=manifest,
        )
    image = dataset / "n00000001" / "image_0.JPEG"
    original = image.read_bytes()
    image.write_bytes(b"X" * len(original))
    with pytest.raises(IntegrityError, match="file-tree bytes"):
        natural_shift.validate_imagenet_sketch_provenance(manifest)


def test_freeze_rejects_an_archive_unrelated_to_the_extracted_tree(
    tiny_imagenet_sketch,
    tmp_path: Path,
) -> None:
    dataset, reference, _, _, wnids = tiny_imagenet_sketch
    unrelated = tmp_path / "unrelated.zip"
    with zipfile.ZipFile(unrelated, "w", compression=zipfile.ZIP_STORED) as handle:
        for wnid in wnids:
            for index in range(4):
                handle.writestr(
                    f"ImageNet-Sketch/{wnid}/image_{index}.JPEG",
                    f"unrelated:{wnid}:{index}".encode("ascii"),
                )
    with pytest.raises(IntegrityError, match="does not match the supplied archive"):
        natural_shift.freeze_imagenet_sketch_provenance(
            dataset_root=dataset,
            reference_root=reference,
            archive=unrelated,
            output=tmp_path / "must_not_exist.json",
        )


def test_loader_reads_only_feature_and_frozen_readout_roles() -> None:
    rng = np.random.default_rng(13)
    store = _TrackingStore({
        "reference_features": rng.normal(size=(8, 2)),
        "target_features": rng.normal(size=(8, 2)),
        "head_weight": rng.normal(size=(2, 2)),
        "head_bias": np.zeros(2),
    })
    loaded = natural_shift.load_unpaired_store_inputs(store)  # type: ignore[arg-type]
    assert loaded.reference_features.shape == (8, 2)
    assert store.calls == [
        "reference_features",
        "target_features",
        "head_weight",
        "head_bias",
    ]


def test_loader_rejects_available_pairing_before_any_array_access() -> None:
    store = _TrackingStore({}, pairing_status="available")
    with pytest.raises(IntegrityError, match="pairing to be unavailable"):
        natural_shift.load_unpaired_store_inputs(store)  # type: ignore[arg-type]
    assert store.calls == []


def test_reference_id_label_parity_fails_closed(
    tiny_imagenet_sketch,
) -> None:
    _, _, _, _, wnids = tiny_imagenet_sketch
    identifiers = np.asarray([
        f"{wnid}/image_{index}.JPEG"
        for class_index, wnid in enumerate(wnids)
        for index in range(4)
    ])
    labels = np.repeat(np.arange(2, dtype=np.int64), 4)
    fit_ids = identifiers[[0, 4]]
    passed = natural_shift._validate_reference_class_identity(
        reference_ids=identifiers,
        reference_labels=labels,
        head_fit_ids=fit_ids,
        class_index=wnids,
        shots_per_class=1,
    )
    assert passed["status"] == "PASS"
    corrupted = labels.copy()
    corrupted[0] = 1
    with pytest.raises(IntegrityError, match="ID-to-label mapping"):
        natural_shift._validate_reference_class_identity(
            reference_ids=identifiers,
            reference_labels=corrupted,
            head_fit_ids=fit_ids,
            class_index=wnids,
            shots_per_class=1,
        )


def test_full_store_run_validates_provenance_and_never_reads_labels_or_pairs(
    tiny_imagenet_sketch,
) -> None:
    _, _, _, manifest, _ = tiny_imagenet_sketch
    reference = np.asarray([
        [-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]
    ])
    target = np.asarray([
        [-2.8], [-1.9], [-0.8], [-0.4], [0.4], [0.9], [2.1], [2.9]
    ])
    store = _TrackingStore({
        "reference_features": reference,
        "target_features": target,
        "head_weight": np.asarray([[-1.0], [1.0]]),
        "head_bias": np.zeros(2),
    })
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    store.manifest["metadata"] = {
        "imagenet_sketch_provenance": {
            "manifest_sha256": sha256_file(manifest),
            "file_tree_sha256": payload["file_tree_sha256"],
            "class_index_sha256": payload["class_index_sha256"],
            "row_count": 8,
        },
        "reference_class_identity": {
            "status": "PASS",
            "reference_id_kind": "wnid_basename",
            "class_index_sha256": payload["class_index_sha256"],
            "reference_id_label_mapping_sha256": "3" * 64,
        },
        "source_campaign_reference_store": _source_reference_binding(
            encoder="fixture_encoder"
        ),
    }
    result = natural_shift.run_unpaired_store_ceiling(
        store,  # type: ignore[arg-type]
        manifest,
        source_campaign_reference_store=store.manifest["metadata"][
            "source_campaign_reference_store"
        ],
        split_seed=3,
        calibration_rows=4,
        witness_rows=4,
        protocol=CorrespondenceCeilingProtocol(
            maximum_witness_rows=4,
            swap_neighbors=4,
            swap_neighbor_sensitivity=(1, 2),
        ),
    )
    assert result["status"] == "PASS"
    assert result["labels_used"] is False
    assert result["pairing_used"] is False
    assert store.calls == [
        "reference_features",
        "target_features",
        "head_weight",
        "head_bias",
    ]


def _native_store(tmp_path: Path, manifest: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    arrays = {
        "reference_features": np.linspace(-3.0, 3.0, 8)[:, None],
        "target_features": np.linspace(-2.8, 2.9, 8)[:, None],
        "reference_ids": np.asarray([f"reference:{index}" for index in range(8)]),
        "target_ids": np.asarray([f"sketch:{index}" for index in range(8)]),
        "target_labels": (np.linspace(-2.8, 2.9, 8) > 0).astype(np.int64),
        "head_weight": np.asarray([[-1.0], [1.0]]),
        "head_bias": np.zeros(2),
        "head_fit_ids": np.asarray([f"fit:{index}" for index in range(4)]),
    }
    for role, value in arrays.items():
        np.save(root / f"{role}.npy", value)
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    return build_native_manifest(
        root,
        dataset="imagenet_sketch",
        encoder="dinov3",
        arrays={role: f"{role}.npy" for role in arrays},
        id_provenance={
            "reference": {
                "verified": True,
                "kind": "independent_clean_image_key",
                "source": "fixture",
                "namespace": "fixture:reference",
            },
            "target": {
                "verified": True,
                "kind": "independent_sketch_image_key",
                "source": "fixture",
                "namespace": "fixture:target",
            },
        },
        pairing={
            "status": "unavailable",
            "reason": "independent endpoint collections",
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata={
            "imagenet_sketch_provenance": {
                "manifest_sha256": sha256_file(manifest),
                "file_tree_sha256": provenance["file_tree_sha256"],
                "class_index_sha256": provenance["class_index_sha256"],
                "row_count": 8,
            },
            "reference_class_identity": {
                "status": "PASS",
                "reference_id_kind": "wnid_basename",
                "class_index_sha256": provenance["class_index_sha256"],
                "reference_id_label_mapping_sha256": "3" * 64,
            },
            "source_campaign_reference_store": _source_reference_binding(
                encoder="dinov3"
            ),
            "model_provenance": {
                "encoder_id": "dinov3",
                "model_id": "fixture/model",
                "checkpoint_sha256": "0" * 64,
                "architecture_config_sha256": "2" * 64,
                "preprocessing_sha256": "1" * 64,
                "feature_layer": "fixture",
                "pooling": "fixture",
                "head_protocol": "fixture",
                "head_fit_ids_sha256": sha256_file(root / "head_fit_ids.npy"),
                "head_fit_ids_role": "head_fit_ids",
                "head_fit_id_namespace": "fixture:head_fit",
                "reference_id_namespace": "fixture:reference",
                "reference_split": "validation_without_head_fit",
                "head_fit_split": "validation_head_fit",
            },
        },
    )


def _paired_imagenetc_store(
    tmp_path: Path,
    *,
    rows: int = 12,
    corruption: str = "brightness",
    store_name: str = "paired_store",
) -> Path:
    root = tmp_path / store_name
    root.mkdir(parents=True)
    if rows % 2:
        raise ValueError("fixture rows must be even")
    labels = np.repeat(np.arange(2, dtype=np.int64), rows // 2)
    reference_ids = np.asarray([
        f"n{int(label) + 1:08d}/reference_{index}.JPEG"
        for index, label in enumerate(labels)
    ])
    target_ids = np.asarray([f"t:{index}" for index in range(rows)])
    reference_features = np.linspace(-3.0, 3.0, rows)[:, None]
    target_features = reference_features + 0.15 * np.sin(np.arange(rows))[:, None]
    head_fit_indices = np.asarray([0, rows // 2], dtype=np.int64)
    arrays = {
        "reference_features": reference_features.astype(np.float32),
        "target_features": target_features.astype(np.float32),
        "reference_ids": reference_ids,
        "target_ids": target_ids,
        "reference_labels": labels,
        "target_labels": labels,
        "pair_reference_ids": reference_ids,
        "pair_target_ids": target_ids,
        "head_weight": np.asarray([[-1.0], [1.0]], dtype=np.float32),
        "head_bias": np.zeros(2, dtype=np.float32),
        "head_fit_ids": reference_ids[head_fit_indices],
    }
    for role, value in arrays.items():
        np.save(root / f"{role}.npy", value)
    return build_native_manifest(
        root,
        dataset="imagenet_c",
        encoder="dinov3",
        arrays={role: f"{role}.npy" for role in arrays},
        id_provenance={
            "reference": {
                "verified": True,
                "kind": "dataset_sample_key",
                "source": "fixture",
                "namespace": "imagenet:reference",
            },
            "target": {
                "verified": True,
                "kind": "dataset_sample_key",
                "source": "fixture",
                "namespace": "imagenet_c:target",
            },
        },
        pairing={
            "status": "available",
            "reference_ids_role": "pair_reference_ids",
            "target_ids_role": "pair_target_ids",
            "provenance": {
                "verified": True,
                "kind": "natural_clean_corruption_identity",
                "source": "fixture",
            },
        },
        head={"weight_role": "head_weight", "bias_role": "head_bias"},
        metadata={
            "corruption": corruption,
            "severity": 5,
            "model_provenance": {
                "encoder_id": "dinov3",
                "model_id": "fixture/model",
                "checkpoint_sha256": "0" * 64,
                "architecture_config_sha256": "2" * 64,
                "preprocessing_sha256": "1" * 64,
                "feature_layer": "fixture",
                "pooling": "fixture",
                "head_protocol": {"kind": "fixture", "shots_per_class": 1},
                "head_fit_ids_sha256": sha256_file(root / "head_fit_ids.npy"),
                "head_fit_ids_role": "head_fit_ids",
                "head_fit_id_namespace": "imagenet:reference",
                "reference_id_namespace": "imagenet:reference",
                "reference_split": "val",
                "head_fit_split": "val",
            },
        },
    )


def _base_campaign_evidence(
    tmp_path: Path,
    records: list[tuple[Path, str, str, str, int]],
) -> natural_shift.BaseCampaignFeatureStoreEvidence:
    external_input_sha256 = "a" * 64
    payload = {
        "schema_version": "marginal_alignment.iclr.external_inputs.v1",
        "status": "PASS",
        "external_input_sha256": external_input_sha256,
        "native_feature_store_count": len(records),
        "native_feature_stores": [
            {
                "dataset": dataset,
                "encoder": encoder,
                "corruption": corruption,
                "severity": severity,
                "manifest": (
                    f"<FEATURE_STORE_ROOT>/{dataset}/{encoder}/{corruption}/"
                    f"severity_{severity}/FEATURE_STORE_MANIFEST.json"
                ),
                "manifest_sha256": sha256_file(manifest),
            }
            for manifest, dataset, encoder, corruption, severity in records
        ],
    }
    path = tmp_path / "external_input_manifest.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return natural_shift.load_base_campaign_feature_store_evidence(
        path,
        expected_file_sha256=sha256_file(path),
        expected_external_input_sha256=external_input_sha256,
        expected_feature_store_count=len(records),
    )


def test_base_campaign_store_allowlist_rejects_valid_store_substitution(
    tmp_path: Path,
) -> None:
    locked_manifest = _paired_imagenetc_store(
        tmp_path, rows=12, store_name="locked_store"
    )
    substitute_manifest = _paired_imagenetc_store(
        tmp_path, rows=14, store_name="substitute_store"
    )
    evidence = _base_campaign_evidence(tmp_path, [
        (locked_manifest, "imagenet_c", "dinov3", "brightness", 5)
    ])
    locked_store = natural_shift.load_feature_store(locked_manifest)
    binding = natural_shift.verify_base_campaign_feature_store(
        locked_store,
        evidence,
        dataset="imagenet_c",
        encoder="dinov3",
        corruption="brightness",
        severity=5,
    )
    assert binding["actual_manifest_sha256"] == sha256_file(locked_manifest)
    substitute_store = natural_shift.load_feature_store(substitute_manifest)
    with pytest.raises(IntegrityError, match="exact base-campaign manifest"):
        natural_shift.verify_base_campaign_feature_store(
            substitute_store,
            evidence,
            dataset="imagenet_c",
            encoder="dinov3",
            corruption="brightness",
            severity=5,
        )


def test_play3_config_and_locks_bind_the_exact_base_external_manifest() -> None:
    repository = Path(natural_shift.__file__).resolve().parents[3]
    config = json.loads(
        (repository / "configs/iclr_extension/play3_natural.json").read_text()
    )
    child = json.loads(
        (repository / "protocol/iclr_extension_play3_lock.json").read_text()
    )
    master = json.loads(
        (repository / "protocol/iclr_extension_master_lock.json").read_text()
    )
    declaration = config["base_campaign_feature_stores"]
    assert declaration == child["base_campaign_feature_stores"]
    assert declaration == master[
        "play3_empirical_ceiling_and_natural_shift"
    ]["base_campaign_feature_stores"]
    external = repository / declaration["external_input_manifest"]
    assert sha256_file(external) == declaration[
        "external_input_manifest_file_sha256"
    ]
    payload = json.loads(external.read_text(encoding="utf-8"))
    assert payload["external_input_sha256"] == declaration[
        "external_input_sha256"
    ]
    assert payload["native_feature_store_count"] == declaration[
        "feature_store_count"
    ]
    assert set(child["source_files"]) == natural_shift.PLAY3_SOURCE_FILES
    assert set(child["required_imported_dependencies"]) == (
        natural_shift.PLAY3_REQUIRED_IMPORTED_DEPENDENCIES
    )
    for relative, expected in child["source_files"].items():
        assert sha256_file(repository / relative) == expected


def test_base_campaign_external_manifest_byte_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    store_manifest = _paired_imagenetc_store(tmp_path)
    evidence = _base_campaign_evidence(tmp_path, [
        (store_manifest, "imagenet_c", "dinov3", "brightness", 5)
    ])
    manifest = evidence.manifest_path
    locked_file_sha256 = evidence.manifest_file_sha256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["native_feature_stores"][0]["manifest_sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(IntegrityError, match="manifest bytes changed"):
        natural_shift.load_base_campaign_feature_store_evidence(
            manifest,
            expected_file_sha256=locked_file_sha256,
            expected_external_input_sha256=evidence.external_input_sha256,
            expected_feature_store_count=1,
        )


def test_generated_natural_store_reference_binding_tampering_is_rejected(
    tiny_imagenet_sketch,
    tmp_path: Path,
) -> None:
    dataset, _, archive, provenance_manifest, _ = tiny_imagenet_sketch
    store = natural_shift.load_feature_store(
        _native_store(tmp_path, provenance_manifest)
    )
    provenance = natural_shift.validate_imagenet_sketch_provenance(
        provenance_manifest, dataset_root=dataset, archive=archive
    )
    original = dict(
        store.manifest["metadata"]["source_campaign_reference_store"]
    )
    store.manifest["metadata"]["source_campaign_reference_store"] = {
        **original,
        "actual_manifest_sha256": "f" * 64,
    }
    with pytest.raises(IntegrityError, match="malformed base-campaign"):
        natural_shift._validate_store_provenance_binding(
            store,
            provenance,
            supplied_source_binding=original,
        )


def test_locked_plan_only_rejects_a_structurally_valid_substitute_store(
    tiny_imagenet_sketch, tmp_path: Path
) -> None:
    dataset, _, archive, manifest, _ = tiny_imagenet_sketch
    store_manifest = _native_store(tmp_path, manifest)
    repository = Path(natural_shift.__file__).resolve().parents[3]
    with pytest.raises(IntegrityError, match="locked base campaign"):
        natural_shift.main([
            "--config", str(repository / "configs/iclr_extension/play3_natural.json"),
            "--store", str(store_manifest),
            "--provenance", str(manifest),
            "--dataset-root", str(dataset),
            "--archive", str(archive),
            "--split-seed", "201",
            "--cost-budget-relative", "0.05",
            "--plan-only",
        ])


def test_immutable_unit_writer_rejects_resume_or_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "unit.json"
    natural_shift.write_immutable_unit_result(output, {"status": "PASS"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        natural_shift.write_immutable_unit_result(output, {"status": "PASS"})


def test_positive_control_hides_pairing_from_fit_then_validates_posthoc(
    tmp_path: Path,
) -> None:
    store_manifest = _paired_imagenetc_store(tmp_path)
    store = natural_shift.load_feature_store(store_manifest)
    evidence = _base_campaign_evidence(tmp_path, [
        (store_manifest, "imagenet_c", "dinov3", "brightness", 5)
    ])
    binding = natural_shift.verify_base_campaign_feature_store(
        store,
        evidence,
        dataset="imagenet_c",
        encoder="dinov3",
        corruption="brightness",
        severity=5,
    )
    result = natural_shift.run_paired_positive_control(
        store,
        source_campaign_store=binding,
        split_seed=201,
        calibration_rows=4,
        witness_rows=4,
        protocol=CorrespondenceCeilingProtocol(
            maximum_witness_rows=4,
            swap_neighbors=4,
            swap_neighbor_sensitivity=(1, 2),
        ),
    )
    assert result["status"] == "PASS"
    assert result["pairing_unavailable_to_estimator"] is True
    assert result["pairing_used_for_split_administration_before_fit"] is True
    assert result[
        "labels_used_for_administrative_pair_validation_before_fit"
    ] is True
    assert result["labels_unavailable_to_estimator"] is True
    assert result["labels_used_by_estimator"] is False
    assert [
        row["swap_neighbors"]
        for row in result["swap_neighborhood_sensitivity"]
    ] == [1, 2]
    primary = result["ceiling_witness"]["diagnostics"]
    assert all(
        row["ceiling_witness"]["diagnostics"]["calibration"][
            "full_geometry_sha256"
        ] == primary["calibration"]["full_geometry_sha256"]
        for row in result["swap_neighborhood_sensitivity"]
    )
    assert result["posthoc_validation"]["pairing_content_sha256"]
    assert set(result["posthoc_validation"][
        "per_method_reference_prediction_agreement"
    ]) == set(natural_shift.MARGINAL_FRONTIER_METHODS)
    assert 0.0 <= result["posthoc_validation"][
        "bures_reference_prediction_agreement"
    ] <= 1.0


def test_positive_control_rejects_store_binding_tamper_before_estimation(
    tmp_path: Path,
) -> None:
    store_manifest = _paired_imagenetc_store(tmp_path)
    store = natural_shift.load_feature_store(store_manifest)
    evidence = _base_campaign_evidence(tmp_path, [
        (store_manifest, "imagenet_c", "dinov3", "brightness", 5)
    ])
    binding = natural_shift.verify_base_campaign_feature_store(
        store,
        evidence,
        dataset="imagenet_c",
        encoder="dinov3",
        corruption="brightness",
        severity=5,
    )
    with pytest.raises(IntegrityError, match="exact locked base-campaign"):
        natural_shift.run_paired_positive_control(
            store,
            source_campaign_store={
                **binding,
                "actual_manifest_sha256": "f" * 64,
            },
            split_seed=201,
            calibration_rows=4,
            witness_rows=4,
            protocol=CorrespondenceCeilingProtocol(
                maximum_witness_rows=4,
                swap_neighbors=4,
                swap_neighbor_sensitivity=(1, 2),
            ),
        )


def test_natural_accuracy_is_a_separate_post_witness_artifact(
    tiny_imagenet_sketch, tmp_path: Path
) -> None:
    _, _, _, manifest, _ = tiny_imagenet_sketch
    store = natural_shift.load_feature_store(_native_store(tmp_path, manifest))
    result = natural_shift.run_unpaired_store_ceiling(
        store,
        manifest,
        source_campaign_reference_store=store.manifest["metadata"][
            "source_campaign_reference_store"
        ],
        split_seed=201,
        calibration_rows=4,
        witness_rows=4,
        protocol=CorrespondenceCeilingProtocol(
            maximum_witness_rows=4,
            swap_neighbors=4,
            swap_neighbor_sensitivity=(1, 2),
        ),
        include_posthoc_accuracy=True,
    )
    posthoc = result["posthoc_evaluation"]
    assert result["labels_used"] is False
    assert posthoc["evaluation_phase"] == (
        "after_geometry_and_two_world_witness_freeze"
    )
    assert posthoc["used_for_geometry_or_map_fitting"] is False
    assert posthoc["posthoc_oracle_envelope_deployable"] is False
    assert [
        row["swap_neighbors"]
        for row in result["swap_neighborhood_sensitivity"]
    ] == [1, 2]
    assert set(posthoc["per_method_target_accuracy"]) == set(
        natural_shift.MARGINAL_FRONTIER_METHODS
    )
    assert 0.0 <= posthoc["source_target_accuracy"] <= 1.0
    assert 0.0 <= posthoc["bures_target_accuracy"] <= 1.0


def test_actual_sketch_store_builder_extracts_and_publishes_unpaired_store(
    tiny_imagenet_sketch,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _, archive, manifest, _ = tiny_imagenet_sketch
    reference_manifest = _paired_imagenetc_store(tmp_path, rows=8)
    reference_store = natural_shift.load_feature_store(reference_manifest)
    source_evidence = _base_campaign_evidence(tmp_path, [
        (reference_manifest, "imagenet_c", "dinov3", "brightness", 5)
    ])
    monkeypatch.setattr(
        natural_shift,
        "_load_locked_base_campaign_feature_store_evidence",
        lambda *args, **kwargs: source_evidence,
    )
    model = reference_store.manifest["metadata"]["model_provenance"]

    class _Encoder:
        expected_dimension = 1
        evidence = {
            "encoder_id": "dinov3",
            "model_id": model["model_id"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "architecture_config_sha256": model[
                "architecture_config_sha256"
            ],
            "preprocessing_sha256": model["preprocessing_sha256"],
            "feature_layer": model["feature_layer"],
            "pooling": model["pooling"],
            "expected_dimension": 1,
        }

    monkeypatch.setattr(
        natural_shift,
        "load_feature_build_config",
        lambda _: {
            "batch_size": 2,
            "workers": 0,
            "encoders": {"dinov3": {"fixture": True}},
        },
    )
    monkeypatch.setattr(
        natural_shift, "load_frozen_encoder", lambda *args, **kwargs: _Encoder()
    )

    calls: list[int] = []
    fail_once = {"value": True}

    def fake_extract(encoder, image_dataset, destination, **kwargs):
        start = int(kwargs["destination_offset"])
        calls.append(start)
        if start == 3 and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("simulated power interruption")
        stop = start + len(image_dataset)
        destination[start:stop] = np.arange(
            start, stop, dtype=np.float32
        )[:, None]
        label_start = int(kwargs["label_destination_offset"])
        kwargs["label_destination"][label_start:label_start + len(image_dataset)] = (
            image_dataset.labels
        )
        if hasattr(destination, "flush"):
            destination.flush()

    monkeypatch.setattr(natural_shift, "extract_dataset", fake_extract)
    repository = Path(natural_shift.__file__).resolve().parents[3]
    config_path = repository / "configs/iclr_extension/play3_natural.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    lock_payload = json.loads(
        (repository / config_payload["lock_path"]).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        natural_shift,
        "load_locked_play3_config",
        lambda _: (config_payload, config_path, lock_payload, repository),
    )
    output = tmp_path / "published_sketch_store"
    work = tmp_path / "resumable_work"
    with pytest.raises(RuntimeError, match="simulated power interruption"):
        natural_shift.build_imagenet_sketch_feature_store(
            play3_config_path=config_path,
            provenance_manifest=manifest,
            reference_store_path=reference_manifest,
            destination=output,
            encoder_id="dinov3",
            dataset_root=dataset,
            archive=archive,
            device="cpu",
            work_dir=work,
            checkpoint_rows=3,
        )
    interrupted_state = json.loads((work / "BUILD_STATE.json").read_text())
    assert output.exists() is False
    assert [(item["start"], item["stop"]) for item in interrupted_state[
        "completed_blocks"
    ]] == [(0, 3)]
    built = natural_shift.build_imagenet_sketch_feature_store(
        play3_config_path=config_path,
        provenance_manifest=manifest,
        reference_store_path=reference_manifest,
        destination=output,
        encoder_id="dinov3",
        dataset_root=dataset,
        archive=archive,
        device="cpu",
        work_dir=work,
        checkpoint_rows=3,
    )
    store = natural_shift.load_feature_store(built)
    assert store.dataset == "imagenet_sketch"
    assert store.pairing_spec["status"] == "unavailable"
    assert len(store.endpoint("reference").features) == 6
    assert len(store.endpoint("target").features) == 8
    assert calls == [0, 3, 3, 6]
    final_state = json.loads((output / "BUILD_STATE.json").read_text())
    assert final_state["status"] == "PASS"
    assert final_state["phase"] == "published_ready"
    assert len(final_state["completed_blocks"]) == 3
    assert not (output / "extracted_target_labels.npy").exists()
    assert not (output / "target_labels.partial.npy").exists()
    assert store.records["target_labels"]["path"] == "target_labels.npy"
    assert store.manifest["metadata"]["resumable_build"][
        "provenance_sidecar"
    ]["path"] == "BUILD_STATE.json"
    source_binding = store.manifest["metadata"][
        "source_campaign_reference_store"
    ]
    assert source_binding["condition"] == {
        "dataset": "imagenet_c",
        "encoder": "dinov3",
        "corruption": "brightness",
        "severity": 5,
    }
    assert source_binding["actual_manifest_sha256"] == sha256_file(
        reference_manifest
    )
    assert source_binding["external_input_manifest_file_sha256"] == (
        source_evidence.manifest_file_sha256
    )
    with pytest.raises(IntegrityError, match="no usable correspondence"):
        store.pairing()
