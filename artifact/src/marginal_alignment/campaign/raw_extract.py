"""GPU extraction fallback for strict CIFAR-100/C and ImageNet/C caches.

The normal campaign path reuses existing feature matrices.  This module is the
fail-closed fallback when those matrices lack model or identity provenance.
It requires local model weights and local raw datasets; it never downloads or
substitutes a model.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import IntegrityError, sha256_file, sha256_json
from .cache_import import CIFAR100_C_CORRUPTIONS, IMAGENET_C_CORRUPTIONS
from .feature_protocol import (
    MODEL_EVIDENCE_SCHEMA,
    atomic_json,
    atomic_npy,
    canonical_block_index,
    checkpoint_tree_fingerprint,
    load_feature_build_config,
    read_json_object,
)


RAW_BUILD_SCHEMA = "marginal_alignment.campaign.raw_feature_build.v1"
CIFAR_PROVENANCE_SCHEMA = "marginal_alignment.campaign.cifar100c_raw_provenance.v1"
IMAGENET_PROVENANCE_SCHEMA = "marginal_alignment.campaign.imagenetc_raw_provenance.v1"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
RAW_REFERENCE_MANIFEST = "RAW_REFERENCE_MANIFEST.json"
RAW_TARGET_MANIFEST = "RAW_TARGET_MANIFEST.json"


def _resolve_local_transformers_snapshot(model_id: str, environment_name: str) -> Path:
    explicit = os.environ.get(environment_name)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - server only
        raise RuntimeError("huggingface-hub is required for local checkpoint discovery") from error
    try:
        return Path(snapshot_download(repo_id=model_id, local_files_only=True)).resolve()
    except Exception as error:
        raise FileNotFoundError(
            f"no local snapshot for {model_id}; set {environment_name} to an exact local path"
        ) from error


def _resolve_required_file(environment_name: str) -> Path:
    raw = os.environ.get(environment_name)
    if not raw:
        raise FileNotFoundError(f"set {environment_name} to the exact local checkpoint file")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _torch_dependencies():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:  # pragma: no cover - server only
        raise RuntimeError("Torch is required for raw feature extraction") from error
    return torch, DataLoader, Dataset


def _load_strict_safetensors(model: Any, checkpoint: Path) -> None:
    """Load a timm state dict without letting timm guess the file format.

    ``timm`` routes a pretrained ``file`` for Vision Transformers through its
    legacy JAX/NumPy loader.  A Hugging Face ``model.safetensors`` file must
    instead be decoded explicitly and then checked against the complete model
    state dict.  Strict loading prevents a plausible-looking but incompatible
    checkpoint from silently producing features.
    """

    if checkpoint.suffix.lower() != ".safetensors":
        raise IntegrityError(
            f"locked timm checkpoint must be SafeTensors, got: {checkpoint}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - server dependency gate
        raise RuntimeError("safetensors is required for supervised ViT extraction") from error
    state_dict = load_file(str(checkpoint), device="cpu")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise IntegrityError(f"SafeTensors checkpoint has no state dictionary: {checkpoint}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise IntegrityError(
            f"SafeTensors checkpoint is incompatible with the locked timm architecture: "
            f"{checkpoint}: {error}"
        ) from error


@dataclass
class FrozenEncoder:
    encoder_id: str
    model: Any
    processor: Any
    device: str
    expected_dimension: int
    backend: str
    evidence: dict[str, Any]

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        torch, _, _ = _torch_dependencies()
        with torch.inference_mode():
            if self.backend == "transformers_auto_cls":
                inputs = self.processor(images=list(images), return_tensors="pt")
                inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
                output = self.model(**inputs)
                value = output.last_hidden_state[:, 0]
            elif self.backend == "transformers_clip_projection":
                inputs = self.processor(images=list(images), return_tensors="pt")
                pixels = inputs["pixel_values"].to(self.device, non_blocking=True)
                output = self.model(pixel_values=pixels)
                value = output.image_embeds
            elif self.backend == "timm_pre_logits":
                tensor = torch.stack([self.processor(image) for image in images]).to(
                    self.device, non_blocking=True
                )
                value = self.model.forward_head(self.model.forward_features(tensor), pre_logits=True)
            else:  # pragma: no cover - validated by config
                raise RuntimeError(self.backend)
        array = value.detach().float().cpu().numpy()
        if array.ndim != 2 or array.shape[1] != self.expected_dimension:
            raise IntegrityError(
                f"{self.encoder_id} emitted {array.shape}, expected [batch,{self.expected_dimension}]"
            )
        if not np.isfinite(array).all():
            raise IntegrityError(f"{self.encoder_id} emitted nonfinite features")
        return array.astype(np.float32, copy=False)


def load_frozen_encoder(
    encoder_id: str, encoder_spec: Mapping[str, Any], *, device: str,
) -> FrozenEncoder:
    """Load only exact local weights and derive a content-backed evidence record."""

    torch, _, _ = _torch_dependencies()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested but torch.cuda.is_available() is false")
    backend = str(encoder_spec["backend"])
    model_id = str(encoder_spec["model_id"])
    environment_name = str(encoder_spec["model_path_environment"])
    if backend in {"transformers_auto_cls", "transformers_clip_projection"}:
        checkpoint = _resolve_local_transformers_snapshot(model_id, environment_name)
        try:
            if backend == "transformers_auto_cls":
                from transformers import AutoImageProcessor, AutoModel
                processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
                model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
            else:
                from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
                processor = CLIPImageProcessor.from_pretrained(checkpoint, local_files_only=True)
                model = CLIPVisionModelWithProjection.from_pretrained(
                    checkpoint, local_files_only=True
                )
        except ImportError as error:  # pragma: no cover - server only
            raise RuntimeError("transformers is required for DINOv3/CLIP extraction") from error
        preprocessing = {
            "backend": backend,
            "processor_class": processor.__class__.__name__,
            "processor": processor.to_dict(),
        }
    elif backend == "timm_pre_logits":
        checkpoint = _resolve_required_file(environment_name)
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except ImportError as error:  # pragma: no cover - server only
            raise RuntimeError("timm is required for supervised ViT extraction") from error
        # Construct the complete locked architecture, including its original
        # classifier, so the downloaded state dictionary can be validated
        # strictly. Feature extraction still uses forward_head(pre_logits=True)
        # and therefore returns the 768-dimensional penultimate representation.
        model = timm.create_model(model_id, pretrained=False)
        _load_strict_safetensors(model, checkpoint)
        data_config = resolve_data_config({}, model=model)
        processor = create_transform(**data_config, is_training=False)
        preprocessing = {
            "backend": backend, "timm_version": str(timm.__version__),
            "data_config": data_config, "transform_repr": repr(processor),
        }
    else:
        raise IntegrityError(f"unknown encoder backend: {backend}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_sha, checkpoint_files = checkpoint_tree_fingerprint(checkpoint)
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        architecture_config = {
            "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "config": model.config.to_dict(),
        }
    else:
        architecture_config = {
            "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "model_repr": repr(model),
            "pretrained_cfg": dict(getattr(model, "pretrained_cfg", {}) or {}),
        }
    architecture_config_sha = sha256_json(architecture_config)
    preprocessing_sha = sha256_json(preprocessing)
    evidence = {
        "schema_version": MODEL_EVIDENCE_SCHEMA,
        "encoder_id": encoder_id,
        "model_id": model_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_files": checkpoint_files,
        "architecture_config": architecture_config,
        "architecture_config_sha256": architecture_config_sha,
        "preprocessing": preprocessing,
        "preprocessing_sha256": preprocessing_sha,
        "feature_layer": encoder_spec["feature_layer"],
        "pooling": encoder_spec["pooling"],
        "expected_dimension": int(encoder_spec["expected_dimension"]),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "output_dtype": "float32",
    }
    return FrozenEncoder(
        encoder_id=encoder_id, model=model, processor=processor, device=device,
        expected_dimension=int(encoder_spec["expected_dimension"]),
        backend=backend, evidence=evidence,
    )


def _collate_pil(batch):
    images, labels, positions = zip(*batch)
    return list(images), np.asarray(labels, dtype=np.int64), np.asarray(positions, dtype=np.int64)


class ImagePathDataset:
    def __init__(self, paths: Sequence[Path], labels: np.ndarray):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.paths) != len(self.labels):
            raise ValueError("image paths and labels differ")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            value = image.convert("RGB").copy()
        return value, int(self.labels[index]), index


class ArrayImageDataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.images) != len(self.labels):
            raise ValueError("image arrays and labels differ")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        from PIL import Image
        image = Image.fromarray(np.asarray(self.images[index], dtype=np.uint8))
        return image, int(self.labels[index]), index


class TorchvisionWrapper:
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, int(label), index


def extract_dataset(
    encoder: FrozenEncoder, dataset: Any, destination: np.ndarray, *,
    destination_offset: int, batch_size: int, workers: int, label_destination: np.ndarray,
    label_destination_offset: int | None = None,
) -> None:
    torch, DataLoader, _ = _torch_dependencies()
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(workers),
        pin_memory=str(encoder.device).startswith("cuda"), collate_fn=_collate_pil,
        persistent_workers=int(workers) > 0,
    )
    seen = 0
    label_offset = destination_offset if label_destination_offset is None else int(
        label_destination_offset
    )
    for images, labels, positions in loader:
        features = encoder.encode(images)
        rows = destination_offset + positions
        label_rows = label_offset + positions
        destination[rows] = features
        label_destination[label_rows] = labels
        seen += len(rows)
        if seen == len(dataset) or seen % max(int(batch_size) * 100, 1) == 0:
            print(
                f"[extract] {encoder.encoder_id}: {seen}/{len(dataset)} rows "
                f"(offset={destination_offset})", flush=True,
            )
    if seen != len(dataset):
        raise IntegrityError(f"extracted {seen} rows, expected {len(dataset)}")
    if hasattr(destination, "flush"):
        destination.flush()
    if hasattr(label_destination, "flush"):
        label_destination.flush()
    del torch


def discover_image_folder(root: str | Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Discover ImageNet rows in stable WNID/basename identity order."""

    base = Path(root).expanduser().resolve()
    classes = sorted(path for path in base.iterdir() if path.is_dir()) if base.is_dir() else []
    if len(classes) != 1000:
        raise IntegrityError(f"expected 1000 ImageNet WNID directories under {base}, found {len(classes)}")
    paths: list[Path] = []
    labels: list[int] = []
    identities: list[str] = []
    for class_index, directory in enumerate(classes):
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in files:
            paths.append(path)
            labels.append(class_index)
            identities.append(f"{directory.name}/{path.name}")
    ids = np.asarray(identities, dtype=f"<U{max(map(len, identities), default=1)}")
    if len(paths) != 50_000 or len(np.unique(ids)) != len(ids):
        raise IntegrityError(f"ImageNet endpoint needs 50,000 unique identities; found {len(paths)}")
    return paths, np.asarray(labels, dtype=np.int64), ids


def align_image_folder_to_ids(
    root: str | Path, reference_ids: np.ndarray,
) -> tuple[list[Path], np.ndarray]:
    paths, labels, identities = discover_image_folder(root)
    lookup = {str(identity): (path, int(label)) for identity, path, label in zip(identities, paths, labels)}
    if len(lookup) != len(reference_ids) or set(lookup) != set(map(str, reference_ids)):
        missing = sorted(set(map(str, reference_ids)).difference(lookup))[:3]
        extra = sorted(set(lookup).difference(map(str, reference_ids)))[:3]
        raise IntegrityError(f"ImageNet clean/corrupt identity mismatch: missing={missing}, extra={extra}")
    ordered_paths = [lookup[str(identity)][0] for identity in reference_ids]
    ordered_labels = np.asarray([lookup[str(identity)][1] for identity in reference_ids], dtype=np.int64)
    return ordered_paths, ordered_labels


def validate_cifar100c_provenance(path: str | Path) -> dict[str, Any]:
    provenance = read_json_object(path)
    if provenance.get("schema_version") != CIFAR_PROVENANCE_SCHEMA:
        raise IntegrityError(f"CIFAR provenance is not {CIFAR_PROVENANCE_SCHEMA}")
    source_url = str(provenance.get("source_url", ""))
    if "zenodo" not in source_url.lower() or "3555552" not in source_url:
        raise IntegrityError("CIFAR-100-C provenance must cite Zenodo record 3555552")
    archive = Path(str(provenance.get("archive_file", ""))).expanduser().resolve()
    expected = str(provenance.get("archive_sha256", ""))
    if not archive.is_file() or len(expected) != 64 or sha256_file(archive) != expected:
        raise IntegrityError("CIFAR-100-C archive path/hash evidence does not verify")
    return provenance


def validate_imagenetc_provenance(path: str | Path) -> dict[str, Any]:
    provenance = read_json_object(path)
    if provenance.get("schema_version") != IMAGENET_PROVENANCE_SCHEMA:
        raise IntegrityError(f"ImageNet-C provenance is not {IMAGENET_PROVENANCE_SCHEMA}")
    if str(provenance.get("dataset", "")).lower() not in {"imagenet-c", "imagenet_c"}:
        raise IntegrityError("ImageNet-C provenance has the wrong dataset identity")
    source_url = str(provenance.get("source_url", ""))
    if not source_url.startswith(("https://", "http://")):
        raise IntegrityError("ImageNet-C provenance needs an explicit source URL")
    archives = provenance.get("archives")
    if not isinstance(archives, list) or not archives:
        raise IntegrityError("ImageNet-C provenance needs one or more archive path/hash records")
    for record in archives:
        if not isinstance(record, Mapping):
            raise IntegrityError("ImageNet-C archive record is malformed")
        archive = Path(str(record.get("file", ""))).expanduser().resolve()
        expected = str(record.get("sha256", ""))
        if not archive.is_file() or len(expected) != 64 or sha256_file(archive) != expected:
            raise IntegrityError(f"ImageNet-C archive path/hash evidence does not verify: {archive}")
    return provenance


def cifar100c_condition_labels(raw_labels: np.ndarray, severity: int) -> np.ndarray:
    """Accept only the two documented label encodings, proving repeat equality."""

    values = np.asarray(raw_labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise IntegrityError("CIFAR-100-C labels must be a rank-one integer array")
    if len(values) == 10_000:
        return np.asarray(values, dtype=np.int64)
    if len(values) == 50_000:
        blocks = values.reshape(5, 10_000)
        if not all(np.array_equal(blocks[0], blocks[index]) for index in range(1, 5)):
            raise IntegrityError("CIFAR-100-C 50k label blocks are not five exact repeats")
        return np.asarray(blocks[int(severity) - 1], dtype=np.int64)
    raise IntegrityError(
        f"CIFAR-100-C labels have {len(values)} rows; expected one 10k vector or five 10k repeats"
    )


def _verify_raw_array_manifest(
    path: Path, *, fingerprint: str, files: Mapping[str, Path], required_if_any_final: bool,
) -> bool:
    existing = [destination.is_file() for destination in files.values()]
    if not any(existing):
        if path.is_file():
            raise IntegrityError(f"raw manifest exists but all declared arrays are absent: {path}")
        return False
    if not all(existing) or not path.is_file():
        if required_if_any_final:
            raise IntegrityError(
                f"raw final arrays are partial or lack a manifest; use a fresh raw root: {path.parent}"
            )
        return False
    manifest = read_json_object(path)
    if manifest.get("schema_version") != RAW_BUILD_SCHEMA:
        raise IntegrityError(f"raw array manifest schema mismatch: {path}")
    if manifest.get("fingerprint") != fingerprint:
        raise IntegrityError(f"raw final-array fingerprint changed: {path}")
    records = manifest.get("arrays")
    if not isinstance(records, Mapping) or set(records) != set(files):
        raise IntegrityError(f"raw array manifest roles are incomplete: {path}")
    for role, destination in files.items():
        record = records[role]
        if not isinstance(record, Mapping):
            raise IntegrityError(f"raw array record is malformed: {role}")
        if int(record.get("bytes", -1)) != destination.stat().st_size:
            raise IntegrityError(f"raw array byte count changed: {destination}")
        if str(record.get("sha256", "")) != sha256_file(destination):
            raise IntegrityError(f"raw array SHA-256 changed: {destination}")
    return True


def _write_raw_array_manifest(
    path: Path, *, fingerprint: str, files: Mapping[str, Path], metadata: Mapping[str, Any],
) -> None:
    missing = [str(destination) for destination in files.values() if not destination.is_file()]
    if missing:
        raise IntegrityError(f"cannot seal absent raw arrays: {missing}")
    atomic_json(path, {
        "schema_version": RAW_BUILD_SCHEMA, "status": "PASS",
        "fingerprint": fingerprint, "metadata": dict(metadata),
        "arrays": {
            role: {
                "path": destination.name, "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
            for role, destination in files.items()
        },
    })
    _verify_raw_array_manifest(
        path, fingerprint=fingerprint, files=files, required_if_any_final=True,
    )


def _new_memmap(path: Path, *, rows: int, dimension: int) -> np.ndarray:
    if path.exists():
        value = np.load(path, mmap_mode="r+", allow_pickle=False)
        if value.shape != (rows, dimension) or value.dtype != np.float32:
            raise IntegrityError(f"partial feature shape/dtype mismatch: {path}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(int(rows), int(dimension)),
    )


def _publish_partial(partial: Path, destination: Path, *, replace_generated: bool = False) -> None:
    if destination.exists():
        value = np.load(destination, mmap_mode="r", allow_pickle=False)
        partial_value = np.load(partial, mmap_mode="r", allow_pickle=False)
        if value.shape != partial_value.shape or value.dtype != partial_value.dtype:
            raise IntegrityError(f"existing final feature array conflicts: {destination}")
        if not replace_generated:
            return
    os.replace(partial, destination)


def _dataset_environment(config: Mapping[str, Any], dataset: str, key: str) -> Path:
    environment_name = str(config["datasets"][dataset][key])
    raw = os.environ.get(environment_name)
    if not raw:
        raise FileNotFoundError(f"set {environment_name} for raw {dataset} extraction")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _complete_blocks(state: Mapping[str, Any]) -> set[str]:
    values = state.get("completed_blocks", [])
    return {str(value) for value in values} if isinstance(values, list) else set()


def _build_cifar_raw(
    *, encoder: FrozenEncoder, output_root: Path, config: Mapping[str, Any],
    batch_size: int, workers: int,
) -> dict[str, Any]:
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as error:  # pragma: no cover - server only
        raise RuntimeError("torchvision is required for CIFAR extraction") from error
    dataset_spec = config["datasets"]["cifar100_c"]
    clean_root = _dataset_environment(config, "cifar100_c", "clean_root_environment")
    corruption_root = _dataset_environment(config, "cifar100_c", "corruption_root_environment")
    provenance_env = str(dataset_spec["corruption_provenance_environment"])
    provenance_path = os.environ.get(provenance_env)
    if not provenance_path:
        raise FileNotFoundError(f"set {provenance_env} to the official archive provenance JSON")
    raw_provenance = validate_cifar100c_provenance(provenance_path)
    reference = output_root / "cifar100" / encoder.encoder_id
    target = output_root / "cifar100c" / encoder.encoder_id
    reference.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    dimension = encoder.expected_dimension
    clean_train = CIFAR100(clean_root, train=True, download=False)
    clean_test = CIFAR100(clean_root, train=False, download=False)
    reference_files = {
        "train_features": reference / "train_features.npy",
        "train_labels": reference / "train_labels.npy",
        "test_features": reference / "test_features.npy",
        "test_labels": reference / "test_labels.npy",
    }
    reference_fingerprint = sha256_json({
        "schema": RAW_BUILD_SCHEMA, "dataset": "cifar100", "encoder": encoder.evidence,
        "train_rows": len(clean_train), "test_rows": len(clean_test),
        "torchvision_integrity_checked": True,
    })
    reference_reusable = _verify_raw_array_manifest(
        reference / RAW_REFERENCE_MANIFEST,
        fingerprint=reference_fingerprint, files=reference_files,
        required_if_any_final=False,
    )
    for name, raw_dataset, rows in (
        ("train", clean_train, 50_000), ("test", clean_test, 10_000),
    ):
        destination = reference / f"{name}_features.npy"
        labels_path = reference / f"{name}_labels.npy"
        if not reference_reusable:
            partial = destination.with_suffix(".partial.npy")
            matrix = _new_memmap(partial, rows=rows, dimension=dimension)
            labels = np.empty(rows, dtype=np.int64)
            extract_dataset(
                encoder, TorchvisionWrapper(raw_dataset), matrix,
                destination_offset=0, batch_size=batch_size, workers=workers,
                label_destination=labels,
            )
            if not np.array_equal(labels, np.asarray(raw_dataset.targets, dtype=np.int64)):
                raise IntegrityError(f"CIFAR {name} labels changed during extraction")
            atomic_npy(labels_path, labels)
            _publish_partial(partial, destination, replace_generated=True)
    if not reference_reusable:
        _write_raw_array_manifest(
            reference / RAW_REFERENCE_MANIFEST,
            fingerprint=reference_fingerprint, files=reference_files,
            metadata={"dataset": "cifar100", "encoder": encoder.encoder_id},
        )
    test_labels = np.load(reference / "test_labels.npy", allow_pickle=False)
    blocks = canonical_block_index(
        "cifar100_c", corruptions=CIFAR100_C_CORRUPTIONS,
        severities=dataset_spec["severities"],
    )
    total_rows = int(blocks[-1]["stop"])
    destination = target / "test_features.npy"
    labels_path = target / "test_labels.npy"
    state_path = target / "RAW_EXTRACTION_STATE.json"
    fingerprint = sha256_json({
        "schema": RAW_BUILD_SCHEMA, "dataset": "cifar100_c",
        "encoder_evidence": encoder.evidence, "blocks": blocks,
        "raw_archive_sha256": raw_provenance["archive_sha256"],
    })
    target_files = {
        "target_features": destination,
        "target_labels": labels_path,
    }
    target_reusable = _verify_raw_array_manifest(
        target / RAW_TARGET_MANIFEST, fingerprint=fingerprint, files=target_files,
        required_if_any_final=False,
    )
    state = read_json_object(state_path) if state_path.is_file() else {
        "schema_version": RAW_BUILD_SCHEMA, "fingerprint": fingerprint,
        "completed_blocks": [],
    }
    if state.get("fingerprint") != fingerprint:
        raise IntegrityError("CIFAR partial extraction fingerprint changed; use a fresh raw output root")
    if not target_reusable:
        partial = destination.with_suffix(".partial.npy")
        partial_existed = partial.is_file()
        labels_existed = labels_path.is_file()
        matrix = _new_memmap(partial, rows=total_rows, dimension=dimension)
        labels_output = (
            np.load(labels_path, mmap_mode="r+", allow_pickle=False)
            if labels_path.is_file() else np.lib.format.open_memmap(
                labels_path, mode="w+", dtype=np.int64, shape=(total_rows,)
            )
        )
        complete = _complete_blocks(state)
        if complete and (not partial_existed or not labels_existed):
            complete = set()
            state["completed_blocks"] = []
            atomic_json(state_path, state)
        raw_labels = np.load(corruption_root / "labels.npy", mmap_mode="r", allow_pickle=False)
        for block in blocks:
            key = f"{block['corruption']}:{block['severity']}"
            if key in complete:
                continue
            raw = np.load(
                corruption_root / f"{block['corruption']}.npy", mmap_mode="r", allow_pickle=False
            )
            raw_start = (int(block["severity"]) - 1) * 10_000
            images = raw[raw_start:raw_start + 10_000]
            labels = cifar100c_condition_labels(raw_labels, int(block["severity"]))
            if not np.array_equal(labels, test_labels):
                raise IntegrityError(f"official CIFAR identity label check failed: {key}")
            extract_dataset(
                encoder, ArrayImageDataset(images, labels), matrix,
                destination_offset=int(block["start"]), batch_size=batch_size,
                workers=workers, label_destination=labels_output,
                label_destination_offset=int(block["start"]),
            )
            complete.add(key)
            state["completed_blocks"] = sorted(complete)
            atomic_json(state_path, state)
        _publish_partial(partial, destination, replace_generated=True)
        _write_raw_array_manifest(
            target / RAW_TARGET_MANIFEST, fingerprint=fingerprint, files=target_files,
            metadata={
                "dataset": "cifar100_c", "encoder": encoder.encoder_id,
                "blocks": blocks, "raw_provenance": raw_provenance,
            },
        )
    atomic_json(reference / "MODEL_EVIDENCE.json", encoder.evidence)
    atomic_json(reference / "metadata.json", {
        "reference_feature_file": "test_features.npy", "reference_label_file": "test_labels.npy",
        "head_fit_feature_file": "train_features.npy", "head_fit_label_file": "train_labels.npy",
    })
    atomic_json(target / "metadata.json", {
        "target_feature_file": "test_features.npy", "target_label_file": "test_labels.npy",
        "blocks": blocks, "pairing_provenance": "official_cifar100c_canonical_test_index",
        "raw_provenance": raw_provenance,
    })
    atomic_json(target / "block_index.json", {"blocks": blocks})
    return {"dataset": "cifar100_c", "encoder": encoder.encoder_id, "blocks": len(blocks)}


def _build_imagenet_raw(
    *, encoder: FrozenEncoder, output_root: Path, config: Mapping[str, Any],
    batch_size: int, workers: int,
) -> dict[str, Any]:
    dataset_spec = config["datasets"]["imagenet_c"]
    clean_root = _dataset_environment(config, "imagenet_c", "clean_root_environment")
    corruption_root = _dataset_environment(config, "imagenet_c", "corruption_root_environment")
    provenance_env = str(dataset_spec["corruption_provenance_environment"])
    provenance_path = os.environ.get(provenance_env)
    if not provenance_path:
        raise FileNotFoundError(f"set {provenance_env} to the ImageNet-C archive provenance JSON")
    raw_provenance = validate_imagenetc_provenance(provenance_path)
    reference = output_root / "imagenet" / encoder.encoder_id
    target = output_root / "imagenet_c" / encoder.encoder_id
    reference.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    dimension = encoder.expected_dimension
    clean_paths, clean_labels, clean_ids = discover_image_folder(clean_root)
    destination = reference / "test_features.npy"
    reference_files = {
        "reference_features": destination,
        "reference_labels": reference / "test_labels.npy",
        "reference_ids": reference / "sample_ids.npy",
    }
    reference_fingerprint = sha256_json({
        "schema": RAW_BUILD_SCHEMA, "dataset": "imagenet", "encoder": encoder.evidence,
        "identity_sha256": sha256_json([str(value) for value in clean_ids]),
    })
    reference_reusable = _verify_raw_array_manifest(
        reference / RAW_REFERENCE_MANIFEST,
        fingerprint=reference_fingerprint, files=reference_files,
        required_if_any_final=False,
    )
    if not reference_reusable:
        partial = destination.with_suffix(".partial.npy")
        matrix = _new_memmap(partial, rows=50_000, dimension=dimension)
        extracted_labels = np.empty(50_000, dtype=np.int64)
        extract_dataset(
            encoder, ImagePathDataset(clean_paths, clean_labels), matrix,
            destination_offset=0, batch_size=batch_size, workers=workers,
            label_destination=extracted_labels,
        )
        if not np.array_equal(extracted_labels, clean_labels):
            raise IntegrityError("ImageNet clean labels changed during extraction")
        atomic_npy(reference / "test_labels.npy", clean_labels)
        atomic_npy(reference / "sample_ids.npy", clean_ids)
        _publish_partial(partial, destination, replace_generated=True)
        _write_raw_array_manifest(
            reference / RAW_REFERENCE_MANIFEST,
            fingerprint=reference_fingerprint, files=reference_files,
            metadata={"dataset": "imagenet", "encoder": encoder.encoder_id},
        )
    blocks = canonical_block_index(
        "imagenet_c", corruptions=IMAGENET_C_CORRUPTIONS,
        severities=dataset_spec["severities"],
    )
    total_rows = int(blocks[-1]["stop"])
    target_features_path = target / "test_features.npy"
    target_labels_path = target / "test_labels.npy"
    target_ids_path = target / "sample_ids.npy"
    state_path = target / "RAW_EXTRACTION_STATE.json"
    identity_sha = sha256_json([str(value) for value in clean_ids])
    fingerprint = sha256_json({
        "schema": RAW_BUILD_SCHEMA, "dataset": "imagenet_c",
        "encoder_evidence": encoder.evidence, "blocks": blocks,
        "clean_identity_sha256": identity_sha,
        "raw_provenance_sha256": sha256_json(raw_provenance),
    })
    target_files = {
        "target_features": target_features_path,
        "target_labels": target_labels_path,
        "target_ids": target_ids_path,
    }
    target_reusable = _verify_raw_array_manifest(
        target / RAW_TARGET_MANIFEST, fingerprint=fingerprint, files=target_files,
        required_if_any_final=False,
    )
    state = read_json_object(state_path) if state_path.is_file() else {
        "schema_version": RAW_BUILD_SCHEMA, "fingerprint": fingerprint,
        "completed_blocks": [],
    }
    if state.get("fingerprint") != fingerprint:
        raise IntegrityError("ImageNet partial extraction fingerprint changed; use a fresh raw output root")
    if not target_reusable:
        partial = target_features_path.with_suffix(".partial.npy")
        partial_existed = partial.is_file()
        labels_existed = target_labels_path.is_file()
        ids_existed = target_ids_path.is_file()
        matrix = _new_memmap(partial, rows=total_rows, dimension=dimension)
        labels_output = (
            np.load(target_labels_path, mmap_mode="r+", allow_pickle=False)
            if target_labels_path.is_file() else np.lib.format.open_memmap(
                target_labels_path, mode="w+", dtype=np.int64, shape=(total_rows,)
            )
        )
        ids_output = (
            np.load(target_ids_path, mmap_mode="r+", allow_pickle=False)
            if target_ids_path.is_file() else np.lib.format.open_memmap(
                target_ids_path, mode="w+", dtype=clean_ids.dtype, shape=(total_rows,)
            )
        )
        complete = _complete_blocks(state)
        if complete and (not partial_existed or not labels_existed or not ids_existed):
            complete = set()
            state["completed_blocks"] = []
            atomic_json(state_path, state)
        for block in blocks:
            key = f"{block['corruption']}:{block['severity']}"
            if key in complete:
                continue
            condition_root = (
                corruption_root / str(block["corruption"]) / str(block["severity"])
            )
            paths, labels = align_image_folder_to_ids(condition_root, clean_ids)
            if not np.array_equal(labels, clean_labels):
                raise IntegrityError(f"ImageNet ID-aligned labels disagree: {key}")
            start, stop = int(block["start"]), int(block["stop"])
            ids_output[start:stop] = clean_ids
            extract_dataset(
                encoder, ImagePathDataset(paths, labels), matrix,
                destination_offset=start, batch_size=batch_size, workers=workers,
                label_destination=labels_output, label_destination_offset=start,
            )
            if hasattr(ids_output, "flush"):
                ids_output.flush()
            complete.add(key)
            state["completed_blocks"] = sorted(complete)
            atomic_json(state_path, state)
        _publish_partial(partial, target_features_path, replace_generated=True)
        _write_raw_array_manifest(
            target / RAW_TARGET_MANIFEST, fingerprint=fingerprint, files=target_files,
            metadata={
                "dataset": "imagenet_c", "encoder": encoder.encoder_id,
                "blocks": blocks, "raw_provenance": raw_provenance,
            },
        )
    atomic_json(reference / "MODEL_EVIDENCE.json", encoder.evidence)
    atomic_json(reference / "metadata.json", {
        "reference_feature_file": "test_features.npy", "reference_label_file": "test_labels.npy",
        "sample_ids_file": "sample_ids.npy", "sample_id_kind": "wnid_basename",
    })
    atomic_json(target / "metadata.json", {
        "target_feature_file": "test_features.npy", "target_label_file": "test_labels.npy",
        "sample_ids_file": "sample_ids.npy", "sample_id_kind": "wnid_basename",
        "blocks": blocks, "raw_provenance": raw_provenance,
    })
    atomic_json(target / "block_index.json", {"blocks": blocks})
    return {"dataset": "imagenet_c", "encoder": encoder.encoder_id, "blocks": len(blocks)}


def build_raw_pair(
    *, config_path: str | Path, output_root: str | Path, dataset: str,
    encoder_id: str, device: str = "cuda", batch_size: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    config = load_feature_build_config(config_path)
    if dataset not in config["datasets"] or encoder_id not in config["encoders"]:
        raise IntegrityError(f"unlocked raw build pair: {dataset}/{encoder_id}")
    encoder = load_frozen_encoder(encoder_id, config["encoders"][encoder_id], device=device)
    actual_batch = int(config["batch_size"] if batch_size is None else batch_size)
    actual_workers = int(config["workers"] if workers is None else workers)
    output = Path(output_root).expanduser().resolve()
    if dataset == "cifar100_c":
        result = _build_cifar_raw(
            encoder=encoder, output_root=output, config=config,
            batch_size=actual_batch, workers=actual_workers,
        )
    else:
        result = _build_imagenet_raw(
            encoder=encoder, output_root=output, config=config,
            batch_size=actual_batch, workers=actual_workers,
        )
    result.update({
        "schema_version": RAW_BUILD_SCHEMA, "status": "PASS",
        "feature_build_config_sha256": sha256_file(config_path),
        "model_evidence_sha256": sha256_json(encoder.evidence),
    })
    return result
