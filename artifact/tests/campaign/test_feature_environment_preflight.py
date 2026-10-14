from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.campaign.feature_environment_preflight import validate_raw_inputs


def _config(tmp_path: Path) -> dict:
    return {
        "encoders": {
            "dinov3": {
                "backend": "transformers_auto_cls",
                "model_path_environment": "MARGINAL_DINOV3_MODEL",
            },
            "clip": {
                "backend": "transformers_clip_projection",
                "model_path_environment": "MARGINAL_CLIP_MODEL",
            },
            "vitb16": {
                "backend": "timm_pre_logits",
                "model_path_environment": "MARGINAL_VITB16_CHECKPOINT",
            },
        },
        "datasets": {
            "cifar100_c": {
                "clean_root_environment": "MARGINAL_CIFAR100_ROOT",
                "corruption_root_environment": "MARGINAL_CIFAR100C_ROOT",
                "corruption_provenance_environment": "MARGINAL_CIFAR100C_PROVENANCE",
                "severities": [1, 3, 5],
            },
            "imagenet_c": {
                "clean_root_environment": "MARGINAL_IMAGENET_VAL_ROOT",
                "corruption_root_environment": "MARGINAL_IMAGENETC_ROOT",
                "corruption_provenance_environment": "MARGINAL_IMAGENETC_PROVENANCE",
                "severities": [1, 3, 5],
            },
        },
    }


def test_raw_input_gate_fails_before_extraction_when_model_path_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    for name in (
        "MARGINAL_DINOV3_MODEL", "MARGINAL_CLIP_MODEL", "MARGINAL_VITB16_CHECKPOINT",
        "MARGINAL_CIFAR100_ROOT", "MARGINAL_CIFAR100C_ROOT",
        "MARGINAL_CIFAR100C_PROVENANCE", "MARGINAL_IMAGENET_VAL_ROOT",
        "MARGINAL_IMAGENETC_ROOT", "MARGINAL_IMAGENETC_PROVENANCE",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(FileNotFoundError, match="MARGINAL_DINOV3_MODEL"):
        validate_raw_inputs(_config(tmp_path))


def test_raw_input_gate_rejects_directory_where_vit_checkpoint_file_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    dinov3 = tmp_path / "dinov3"
    clip = tmp_path / "clip"
    vit = tmp_path / "vit"
    for path in (dinov3, clip, vit):
        path.mkdir()
    monkeypatch.setenv("MARGINAL_DINOV3_MODEL", str(dinov3))
    monkeypatch.setenv("MARGINAL_CLIP_MODEL", str(clip))
    monkeypatch.setenv("MARGINAL_VITB16_CHECKPOINT", str(vit))
    with pytest.raises(FileNotFoundError, match="not a file"):
        validate_raw_inputs(_config(tmp_path))
