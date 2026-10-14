#!/usr/bin/env python3
"""Build paired clean/CIFAR-C feature caches with a frozen clean readout.

This script is intentionally separate from the NumPy audit.  It runs on the
GPU server where Torch, torchvision, timm, raw CIFAR-100, and model weights are
available.  Existing DINOv2 pooled corruption features can be reused, avoiding
all corrupted-image inference; the script then extracts only clean train/test
features and trains one clean-only linear readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "glass_blur", "motion_blur", "zoom_blur",
    "fog", "frost", "snow", "contrast", "elastic_transform",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cifar-root", required=True, help="torchvision CIFAR-100 root")
    result.add_argument("--cifar-c-root", help="directory containing CIFAR-100-C .npy files")
    result.add_argument("--output-root", required=True)
    result.add_argument("--model-id", default="vit_base_patch14_dinov2.lvd142m")
    result.add_argument("--encoder-name", default="dinov2_vitb14")
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--checkpoint", help="exact local checkpoint; disables downloads")
    result.add_argument(
        "--discover-local-checkpoint", action="store_true",
        help="find the timm checkpoint in the local Torch/Hugging Face caches only",
    )
    result.add_argument("--severities", type=int, nargs="+", default=[5])
    result.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--device", default="cuda")
    result.add_argument("--head-epochs", type=int, default=30)
    result.add_argument("--head-learning-rate", type=float, default=0.003)
    result.add_argument("--head-weight-decay", type=float, default=1e-4)
    result.add_argument("--head-seed", type=int, default=1729)
    result.add_argument("--download", action="store_true")
    result.add_argument(
        "--reuse-target-pooled",
        help="optional [contexts*10000,d] pooled.npy in the order supplied by --reuse-contexts",
    )
    result.add_argument("--reuse-target-labels", help="optional matching labels.npy")
    result.add_argument("--reuse-contexts", nargs="+", default=DEFAULT_CORRUPTIONS)
    return result


def _dependencies():
    import torch
    import timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torchvision.datasets import CIFAR100

    return torch, timm, Image, DataLoader, Dataset, TensorDataset, CIFAR100


def _find_local_checkpoint(timm, model_id: str) -> Path:
    config = timm.models.get_pretrained_cfg(model_id)
    candidates: list[Path] = []
    # Prefer Hugging Face SafeTensors over legacy URL artifacts.  Some ViT
    # configs point at an official JAX ``.npz`` file in the Torch hub cache;
    # that file is valid, but must be handled by timm's native ViT loader and
    # must never be passed to ``torch.load``.
    repo_id = getattr(config, "hf_hub_id", None)
    if repo_id:
        root = Path.home() / ".cache" / "huggingface" / "hub" / (
            "models--" + str(repo_id).replace("/", "--")
        )
        for pattern in (
            "snapshots/*/*.safetensors", "snapshots/*/*.bin",
            "snapshots/*/*.pth", "snapshots/*/*.pt",
        ):
            candidates.extend(sorted(root.glob(pattern)))
    url = getattr(config, "url", None)
    if url:
        import torch

        candidates.append(Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no local checkpoint found for {model_id}")


def _load_state_dict(path: Path):
    import torch

    # Hugging Face cache aliases do not always retain the underlying format's
    # extension (e.g. SafeTensors content may be exposed as ``*.bin``).  Detect
    # by parsing content, not by trusting the filename.  SafeTensors parsing is
    # non-executable; the fallback keeps PyTorch's restricted weights-only mode
    # explicit and never opts into arbitrary pickle execution.
    try:
        from safetensors.torch import load_file
        from safetensors import SafetensorError

        try:
            state = load_file(str(path))
        except SafetensorError:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
    except ImportError:
        if path.suffix.lower() == ".safetensors":
            raise RuntimeError(
                f"SafeTensors checkpoint found but safetensors is not installed: {path}"
            )
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    return state


def _load_checkpoint_into_model(model, checkpoint_path: Path) -> None:
    """Load a locally pinned checkpoint without confusing NPZ and pickle formats."""
    if checkpoint_path.suffix.lower() == ".npz":
        loader = getattr(model, "load_pretrained", None)
        if loader is None:
            raise RuntimeError(
                f"model has no native loader for JAX/NumPy checkpoint {checkpoint_path}"
            )
        loader(str(checkpoint_path))
        return
    missing, unexpected = model.load_state_dict(
        _load_state_dict(checkpoint_path), strict=False
    )
    del unexpected
    if any(key.startswith("blocks.") for key in missing):
        raise RuntimeError(f"checkpoint did not populate transformer blocks: {missing[:8]}")


def _load_model(
    model_id: str,
    device: str,
    image_size: int,
    *,
    checkpoint: str | None,
    discover_local_checkpoint: bool,
):
    torch, timm, *_ = _dependencies()
    # Pin the input grid.  The server's existing DINOv2 pooled cache was built
    # with img_size=224; omitting this silently instantiates the native 518px
    # positional grid for some timm checkpoints and makes clean/corrupt
    # features incomparable.
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None
    if discover_local_checkpoint:
        if checkpoint_path is not None:
            raise ValueError("choose --checkpoint or --discover-local-checkpoint, not both")
        checkpoint_path = _find_local_checkpoint(timm, model_id)
    if checkpoint_path is None:
        model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=image_size)
        # Record the local file selected by timm when it can be discovered.
        # Failure to discover is recorded explicitly rather than replaced by a
        # guessed path; the server source/result hashes still pin the run.
        try:
            checkpoint_path = _find_local_checkpoint(timm, model_id)
        except FileNotFoundError:
            checkpoint_path = None
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model = timm.create_model(model_id, pretrained=False, num_classes=0, img_size=image_size)
        _load_checkpoint_into_model(model, checkpoint_path)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["std"], dtype=torch.float32).view(3, 1, 1)

    def transform(image):
        # Match the already-cached corruption path exactly: uint8 -> float/255,
        # bicubic interpolate with align_corners=False and no antialias flag,
        # then normalize.  Using timm's PIL pipeline here would introduce a
        # clean/corrupt preprocessing mismatch into a correspondence audit.
        array = np.array(image, dtype=np.uint8, copy=True)
        value = torch.from_numpy(array).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        value = torch.nn.functional.interpolate(
            value, size=image_size, mode="bicubic", align_corners=False
        ).squeeze(0)
        return (value - mean) / std

    config = {
        **config,
        "audit_resize": image_size,
        "audit_preprocessing": "legacy_cache_exact",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "timm_pretrained_loader",
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path is not None else None,
    }
    return model, transform, config


def _extract_dataset(model, dataset, transform, output: Path, *, batch_size: int,
                     workers: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _, DataLoader, Dataset, _, _ = _dependencies()

    class Wrapped(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            image, label = dataset[index]
            return transform(image), int(label), index

    output.parent.mkdir(parents=True, exist_ok=True)
    labels_path = output.with_name(output.stem + "_labels.npy")
    if output.exists() and labels_path.exists():
        return np.load(output, mmap_mode="r"), np.load(labels_path, mmap_mode="r")
    loader = DataLoader(
        Wrapped(), batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    features = None
    labels = np.empty(len(dataset), dtype=np.int64)
    with torch.no_grad():
        for images, batch_labels, indices in loader:
            values = model(images.to(device, non_blocking=True))
            if isinstance(values, (tuple, list)):
                values = values[0]
            values = values.detach().float().cpu().numpy()
            if features is None:
                features = np.lib.format.open_memmap(
                    output, mode="w+", dtype=np.float32, shape=(len(dataset), values.shape[1])
                )
            idx = indices.numpy()
            features[idx] = values
            labels[idx] = batch_labels.numpy()
    assert features is not None
    features.flush()
    np.save(labels_path, labels)
    return np.load(output, mmap_mode="r"), labels


def _train_head(train_features: np.ndarray, train_labels: np.ndarray, classes: int, *,
                epochs: int, learning_rate: float, weight_decay: float, seed: int,
                batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _, DataLoader, _, TensorDataset, _ = _dependencies()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x = torch.as_tensor(np.asarray(train_features), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(train_labels), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )
    head = torch.nn.Linear(x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for epoch in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                head(features.to(device, non_blocking=True)), labels.to(device, non_blocking=True)
            )
            loss.backward()
            optimizer.step()
        print(f"head epoch {epoch + 1}/{epochs} loss={float(loss):.5f}", flush=True)
    return head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy()


def _corruption_dataset(array: np.ndarray, labels: np.ndarray, Image, Dataset):
    class CorruptionDataset(Dataset):
        def __len__(self):
            return len(array)

        def __getitem__(self, index):
            return Image.fromarray(np.asarray(array[index], dtype=np.uint8)), int(labels[index])

    return CorruptionDataset()


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    torch, _, Image, _, Dataset, _, CIFAR100 = _dependencies()
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = Path(arguments.output_root).resolve() / arguments.encoder_name
    work = output_root / "_features"
    work.mkdir(parents=True, exist_ok=True)
    model, transform, preprocessing = _load_model(
        arguments.model_id,
        arguments.device,
        arguments.image_size,
        checkpoint=arguments.checkpoint,
        discover_local_checkpoint=arguments.discover_local_checkpoint,
    )
    clean_train = CIFAR100(arguments.cifar_root, train=True, download=arguments.download)
    clean_test = CIFAR100(arguments.cifar_root, train=False, download=arguments.download)
    train_features, train_labels = _extract_dataset(
        model, clean_train, transform, work / "clean_train.npy",
        batch_size=arguments.batch_size, workers=arguments.workers, device=arguments.device,
    )
    test_features, test_labels = _extract_dataset(
        model, clean_test, transform, work / "clean_test.npy",
        batch_size=arguments.batch_size, workers=arguments.workers, device=arguments.device,
    )
    weight_path, bias_path = work / "head_weight.npy", work / "head_bias.npy"
    if weight_path.exists() and bias_path.exists():
        weight, bias = np.load(weight_path), np.load(bias_path)
    else:
        weight, bias = _train_head(
            train_features, train_labels, 100, epochs=arguments.head_epochs,
            learning_rate=arguments.head_learning_rate,
            weight_decay=arguments.head_weight_decay, seed=arguments.head_seed,
            batch_size=max(arguments.batch_size, 512), device=arguments.device,
        )
        np.save(weight_path, weight)
        np.save(bias_path, bias)
    clean_accuracy = float(np.mean(np.argmax(np.asarray(test_features) @ weight.T + bias, axis=1) == test_labels))
    print(f"frozen clean-readout accuracy={clean_accuracy:.4f}", flush=True)

    # Import only now so running --help does not require PYTHONPATH configuration.
    from marginal_alignment.schema import write_cache

    reused = None
    reused_labels = None
    if arguments.reuse_target_pooled:
        reused = np.load(arguments.reuse_target_pooled, mmap_mode="r")
        reused_labels = (
            np.load(arguments.reuse_target_labels, mmap_mode="r")
            if arguments.reuse_target_labels else None
        )
        expected = len(arguments.reuse_contexts) * 10000
        if reused.shape[0] != expected or reused.shape[1] != test_features.shape[1]:
            raise ValueError(
                f"reused target cache has shape {reused.shape}; expected ({expected},{test_features.shape[1]})"
            )

    if reused is None and not arguments.cifar_c_root:
        raise ValueError("provide either --cifar-c-root or --reuse-target-pooled")
    raw_labels = None
    if arguments.cifar_c_root:
        raw_labels = np.load(Path(arguments.cifar_c_root) / "labels.npy")
    written = []
    for corruption in arguments.corruptions:
        for severity in arguments.severities:
            if reused is not None:
                if severity != 5:
                    raise ValueError("the known reused pooled cache contains severity 5 only")
                context = arguments.reuse_contexts.index(corruption)
                target_features = np.asarray(reused[context * 10000:(context + 1) * 10000])
                target_labels = (
                    np.asarray(reused_labels[context * 10000:(context + 1) * 10000])
                    if reused_labels is not None else test_labels
                )
            else:
                raw = np.load(Path(arguments.cifar_c_root) / f"{corruption}.npy", mmap_mode="r")
                start = (severity - 1) * 10000
                images = raw[start:start + 10000]
                target_labels = np.asarray(raw_labels[start:start + 10000], dtype=np.int64)
                dataset = _corruption_dataset(images, target_labels, Image, Dataset)
                target_path = work / f"{corruption}_s{severity}.npy"
                target_features, extracted_labels = _extract_dataset(
                    model, dataset, transform, target_path,
                    batch_size=arguments.batch_size, workers=arguments.workers,
                    device=arguments.device,
                )
                if not np.array_equal(extracted_labels, target_labels):
                    raise AssertionError("corruption labels changed during extraction")
            if not np.array_equal(target_labels, test_labels):
                raise AssertionError(f"clean/corrupted label order mismatch for {corruption} severity {severity}")
            destination = output_root / corruption / f"severity_{severity}.npz"
            write_cache(
                destination,
                reference_features=np.asarray(test_features),
                target_features=np.asarray(target_features),
                labels=test_labels,
                head_weight=weight,
                head_bias=bias,
                sample_ids=np.arange(len(test_labels), dtype=np.int64),
                metadata={
                    "dataset": "cifar100_c",
                    "encoder": arguments.encoder_name,
                    "model_id": arguments.model_id,
                    "corruption": corruption,
                    "severity": severity,
                    "clean_readout_accuracy": clean_accuracy,
                    "head_fit_data": "clean CIFAR-100 train only",
                    "target_labels_used_for_fit": False,
                    "preprocessing": preprocessing,
                    "reused_target_cache": bool(reused is not None),
                },
            )
            written.append(str(destination))
            print(f"wrote {destination}", flush=True)
    manifest = {
        "status": "PASS", "encoder": arguments.encoder_name,
        "model_id": arguments.model_id, "clean_accuracy": clean_accuracy,
        "caches": written, "torch": torch.__version__, "preprocessing": preprocessing,
    }
    (output_root / "PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
