from pathlib import Path

import pytest

from marginal_alignment.campaign.artifacts import IntegrityError
from marginal_alignment.campaign.raw_extract import _load_strict_safetensors


def test_strict_safetensors_loader_accepts_exact_state_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.safetensors"
    safetensors.save_file(
        {name: value.detach().clone() for name, value in source.state_dict().items()},
        str(checkpoint),
    )

    restored = torch.nn.Linear(3, 2)
    _load_strict_safetensors(restored, checkpoint)
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)

    incompatible = torch.nn.Linear(4, 2)
    with pytest.raises(IntegrityError, match="incompatible with the locked timm architecture"):
        _load_strict_safetensors(incompatible, checkpoint)


def test_strict_safetensors_loader_rejects_ambiguous_file_format(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(IntegrityError, match="must be SafeTensors"):
        _load_strict_safetensors(torch.nn.Linear(1, 1), checkpoint)
