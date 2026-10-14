#!/usr/bin/env python3
"""Download the locked encoder bytes once and emit exact local-path exports.

DINOv3 is gated.  The caller must already have accepted its license and expose
an authorized Hugging Face token through the standard Hugging Face mechanism.
No alternate repository or model is substituted on access failure.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from marginal_alignment.campaign.artifacts import sha256_file
from marginal_alignment.campaign.feature_protocol import (
    atomic_json,
    checkpoint_tree_fingerprint,
    load_feature_build_config,
)


REPOSITORIES = {
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "clip": "openai/clip-vit-large-patch14",
    "vitb16": "timm/vit_base_patch16_224.augreg_in21k_ft_in1k",
}


def download_models(*, config_path: Path, output_root: Path) -> dict:
    # huggingface_hub reads this at import time. hf-xet 1.5.1 has a known
    # gated-blob failure ("Unable to parse string as hex hash value"). Plain
    # Hub HTTP preserves the same immutable revision/bytes and resumes files.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    except ImportError as error:  # pragma: no cover - server only
        raise RuntimeError("huggingface-hub is required") from error

    config = load_feature_build_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    records = {}
    exports = {}
    for encoder in ("dinov3", "clip", "vitb16"):
        repository = REPOSITORIES[encoder]
        if config["encoders"][encoder]["model_id"] != (
            repository if encoder != "vitb16" else repository.split("/", 1)[1]
        ):
            raise RuntimeError(f"locked model identifier changed for {encoder}")
        info = api.model_info(repository)
        revision = str(info.sha)
        local_directory = output_root / encoder / revision
        if encoder in {"dinov3", "clip"}:
            selected = sorted(
                name for name in list_repo_files(repository, revision=revision)
                if name.endswith((".json", ".safetensors"))
            )
            if not selected or not any(name.endswith(".safetensors") for name in selected):
                raise RuntimeError(f"{repository}@{revision} has no SafeTensors checkpoint")
            print(
                f"[models] {encoder}: downloading {len(selected)} files at {revision} "
                "with Xet disabled",
                flush=True,
            )
            for name in selected:
                hf_hub_download(
                    repo_id=repository,
                    filename=name,
                    revision=revision,
                    local_dir=local_directory,
                )
            resolved = local_directory.resolve()
            export_path = resolved
        else:
            resolved_file = Path(hf_hub_download(
                repo_id=repository,
                filename="model.safetensors",
                revision=revision,
                local_dir=local_directory,
            )).resolve()
            export_path = resolved_file
        checkpoint_sha, files = checkpoint_tree_fingerprint(export_path)
        records[encoder] = {
            "repository": repository,
            "revision": revision,
            "local_path": str(export_path),
            "checkpoint_sha256": checkpoint_sha,
            "files": files,
        }
        exports[str(config["encoders"][encoder]["model_path_environment"])] = str(
            export_path
        )

    manifest = {
        "schema_version": "marginal_alignment.campaign.model_acquisition.v1",
        "status": "PASS",
        "feature_build_config": str(config_path.resolve()),
        "feature_build_config_sha256": sha256_file(config_path),
        "models": records,
        "exports": exports,
    }
    atomic_json(output_root / "MODEL_ACQUISITION_MANIFEST.json", manifest)
    shell_path = output_root / "model_paths.sh"
    shell_path.write_text(
        "".join(f"export {name}={json.dumps(path)}\n" for name, path in exports.items()),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "manifest": str(output_root / "MODEL_ACQUISITION_MANIFEST.json"),
        "shell_exports": str(shell_path),
        "revisions": {key: value["revision"] for key, value in records.items()},
    }, indent=2, sort_keys=True))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    download_models(config_path=arguments.config, output_root=arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
