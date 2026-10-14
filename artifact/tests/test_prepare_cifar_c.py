from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import prepare_cifar_c as preparation


class _FakeModel:
    def __init__(self):
        self.native_paths: list[str] = []
        self.state_dict_calls: list[tuple[object, bool]] = []

    def load_pretrained(self, path: str) -> None:
        self.native_paths.append(path)

    def load_state_dict(self, state, *, strict: bool):
        self.state_dict_calls.append((state, strict))
        return [], []


class CheckpointDispatchTests(unittest.TestCase):
    def test_npz_uses_native_vit_loader_not_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official_augreg_checkpoint.npz"
            path.write_bytes(b"validity is delegated to timm")
            model = _FakeModel()
            with mock.patch.object(
                preparation, "_load_state_dict",
                side_effect=AssertionError("NPZ must not reach torch.load"),
            ):
                preparation._load_checkpoint_into_model(model, path)
            self.assertEqual(model.native_paths, [str(path)])
            self.assertEqual(model.state_dict_calls, [])

    def test_pytorch_or_safetensors_uses_state_dict_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(b"fixture")
            model = _FakeModel()
            state = {"blocks.0.weight": object()}
            with mock.patch.object(preparation, "_load_state_dict", return_value=state):
                preparation._load_checkpoint_into_model(model, path)
            self.assertEqual(model.native_paths, [])
            self.assertEqual(model.state_dict_calls, [(state, False)])


if __name__ == "__main__":
    unittest.main()
