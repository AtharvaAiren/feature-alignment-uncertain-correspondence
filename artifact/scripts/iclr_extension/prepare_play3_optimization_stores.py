#!/usr/bin/env python3
"""Prepare authenticated feature stores for the additive Play-3 v2 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.campaign.artifacts import sha256_file
from marginal_alignment.iclr_extension.natural_shift_store import (
    build_natural_shift_feature_store,
    freeze_natural_shift_provenance,
    prepare_six_store_panel,
    upgrade_imagenet_sketch_feature_store,
    validate_natural_shift_provenance,
    write_optimization_store_map,
)


def _common_build(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v2-config", required=True, type=Path)
    parser.add_argument("--v1-config", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--encoder", required=True, choices=("dinov3", "clip"))
    parser.add_argument("--output-store", required=True, type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-provenance")
    freeze.add_argument("--dataset", required=True, choices=("imagenet_v2", "imagenet_r"))
    freeze.add_argument("--dataset-root", required=True, type=Path)
    freeze.add_argument("--archive", required=True, type=Path)
    freeze.add_argument("--class-index", required=True, type=Path)
    freeze.add_argument("--source-url")
    freeze.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate-provenance")
    validate.add_argument("--provenance", required=True, type=Path)
    validate.add_argument("--dataset-root", type=Path)
    validate.add_argument("--archive", type=Path)
    validate.add_argument("--class-index", type=Path)

    build = subparsers.add_parser("build-store")
    _common_build(build)
    build.add_argument("--dataset", required=True, choices=("imagenet_v2", "imagenet_r"))
    build.add_argument("--class-index", required=True, type=Path)
    build.add_argument("--reference-store", required=True, type=Path)
    build.add_argument("--device", default="cuda")
    build.add_argument("--batch-size", type=int)
    build.add_argument("--workers", type=int)
    build.add_argument("--checkpoint-rows", type=int, default=5_000)
    build.add_argument("--work-dir", type=Path)

    sketch = subparsers.add_parser("upgrade-sketch-store")
    _common_build(sketch)
    sketch.add_argument("--source-store", required=True, type=Path)

    panel = subparsers.add_parser("prepare-panel")
    panel.add_argument("--input-map", required=True, type=Path)

    store_map = subparsers.add_parser("write-store-map")
    store_map.add_argument("--v2-config", required=True, type=Path)
    store_map.add_argument("--store-root", required=True, type=Path)
    store_map.add_argument("--output", required=True, type=Path)

    arguments = parser.parse_args(argv)
    if arguments.command == "freeze-provenance":
        output = freeze_natural_shift_provenance(
            dataset=arguments.dataset,
            dataset_root=arguments.dataset_root,
            archive=arguments.archive,
            class_index_path=arguments.class_index,
            output=arguments.output,
            source_url=arguments.source_url,
        )
    elif arguments.command == "validate-provenance":
        value = validate_natural_shift_provenance(
            arguments.provenance,
            dataset_root=arguments.dataset_root,
            archive=arguments.archive,
            class_index_path=arguments.class_index,
        )
        print(json.dumps({
            "status": "PASS",
            "dataset": value.discovery.dataset,
            "rows": len(value.discovery.paths),
            "classes": len(value.discovery.intersection_wnids),
            "provenance": str(value.path),
            "provenance_sha256": sha256_file(value.path),
        }, sort_keys=True))
        return 0
    elif arguments.command == "build-store":
        output = build_natural_shift_feature_store(
            dataset=arguments.dataset,
            v2_config_path=arguments.v2_config,
            v1_config_path=arguments.v1_config,
            provenance_path=arguments.provenance,
            reference_store_path=arguments.reference_store,
            destination=arguments.output_store,
            encoder_id=arguments.encoder,
            dataset_root=arguments.dataset_root,
            archive=arguments.archive,
            class_index_path=arguments.class_index,
            device=arguments.device,
            batch_size=arguments.batch_size,
            workers=arguments.workers,
            checkpoint_rows=arguments.checkpoint_rows,
            work_dir=arguments.work_dir,
        )
    elif arguments.command == "upgrade-sketch-store":
        output = upgrade_imagenet_sketch_feature_store(
            v2_config_path=arguments.v2_config,
            v1_config_path=arguments.v1_config,
            provenance_path=arguments.provenance,
            source_store_path=arguments.source_store,
            destination=arguments.output_store,
            encoder_id=arguments.encoder,
            dataset_root=arguments.dataset_root,
            archive=arguments.archive,
        )
    elif arguments.command == "prepare-panel":
        output = prepare_six_store_panel(arguments.input_map)
    else:
        output = write_optimization_store_map(
            v2_config_path=arguments.v2_config,
            output_root=arguments.store_root,
            destination=arguments.output,
        )
    print(json.dumps({
        "status": "PASS",
        "scientific_result": False,
        "outcomes_read": False,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
