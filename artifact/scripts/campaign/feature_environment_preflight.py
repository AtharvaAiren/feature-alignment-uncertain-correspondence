#!/usr/bin/env python3
"""Fast dependency/CUDA gate before any large cache hashing or extraction."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
from pathlib import Path

from marginal_alignment.campaign.cache_import import (
    CIFAR100_C_CORRUPTIONS,
    IMAGENET_C_CORRUPTIONS,
)
from marginal_alignment.campaign.feature_protocol import load_feature_build_config
from marginal_alignment.campaign.raw_extract import (
    validate_cifar100c_provenance,
    validate_imagenetc_provenance,
)


REQUIRED_MODULES = (
    "numpy", "scipy", "sklearn", "torch", "torchvision", "timm",
    "transformers", "huggingface_hub", "safetensors", "PIL",
)


def _required_environment_path(name: str, *, kind: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise FileNotFoundError(f"set {name}")
    path = Path(raw).expanduser().resolve()
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")
    return path


def validate_raw_inputs(config: dict) -> dict:
    """Fail before extraction unless every locked raw/model input is explicit."""

    paths: dict[str, str] = {}
    for encoder_id, spec in config["encoders"].items():
        name = str(spec["model_path_environment"])
        kind = "file" if str(spec["backend"]) == "timm_pre_logits" else "directory"
        paths[name] = str(_required_environment_path(name, kind=kind))

    for dataset, spec in config["datasets"].items():
        clean_name = str(spec["clean_root_environment"])
        target_name = str(spec["corruption_root_environment"])
        provenance_name = str(spec["corruption_provenance_environment"])
        paths[clean_name] = str(_required_environment_path(clean_name, kind="directory"))
        target = _required_environment_path(target_name, kind="directory")
        paths[target_name] = str(target)
        provenance = _required_environment_path(provenance_name, kind="file")
        paths[provenance_name] = str(provenance)
        if dataset == "cifar100_c":
            validate_cifar100c_provenance(provenance)
            required = [target / "labels.npy"] + [
                target / f"{name}.npy"
                for name in CIFAR100_C_CORRUPTIONS
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"CIFAR-100-C raw files are missing: {missing[:3]}")
        else:
            validate_imagenetc_provenance(provenance)
            clean_root = Path(paths[clean_name])
            clean_classes = [path for path in clean_root.iterdir() if path.is_dir()]
            if len(clean_classes) != 1000:
                raise FileNotFoundError(
                    f"ImageNet val root needs 1000 WNID directories, found {len(clean_classes)}"
                )
            missing_conditions = [
                str(target / corruption / str(severity))
                for corruption in IMAGENET_C_CORRUPTIONS
                for severity in spec["severities"]
                if not (target / corruption / str(severity)).is_dir()
            ]
            if missing_conditions:
                raise FileNotFoundError(
                    f"ImageNet-C condition directories are missing: {missing_conditions[:3]}"
                )
    return {"status": "PASS", "paths": paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-raw-inputs", action="store_true")
    arguments = parser.parse_args(argv)
    config = load_feature_build_config(arguments.config)
    versions = {}
    failures = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "present"))
        except Exception as error:
            failures.append({"module": name, "error": f"{type(error).__name__}: {error}"})
    cuda = {}
    if "torch" in versions:
        import torch
        available = bool(torch.cuda.is_available())
        cuda = {
            "available": available,
            "device_count": int(torch.cuda.device_count()),
            "device_0": torch.cuda.get_device_name(0) if available else None,
        }
        if arguments.require_cuda and not available:
            failures.append({"module": "torch.cuda", "error": "CUDA is required but unavailable"})
        if available:
            # Force context creation now, rather than fail after cache conversion.
            value = torch.ones(1, device="cuda")
            cuda["smoke_value"] = float((value + 1).cpu().item())
    raw_inputs = {"status": "NOT_REQUESTED"}
    if arguments.require_raw_inputs:
        try:
            raw_inputs = validate_raw_inputs(config)
        except Exception as error:
            raw_inputs = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
            failures.append({"module": "raw_inputs", "error": raw_inputs["error"]})
    report = {
        "schema_version": "marginal_alignment.campaign.feature_environment.v1",
        "status": "PASS" if not failures else "FAIL",
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "cuda": cuda, "raw_inputs": raw_inputs,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
