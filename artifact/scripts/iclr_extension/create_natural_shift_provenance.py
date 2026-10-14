#!/usr/bin/env python3
"""Freeze or validate local ImageNet-Sketch acquisition bytes.

This command never downloads data and never produces a scientific result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.iclr_extension.natural_shift import (
    freeze_imagenet_sketch_provenance,
    validate_imagenet_sketch_provenance,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="hash an already-downloaded archive and extracted tree"
    )
    freeze.add_argument("--dataset-root", required=True, type=Path)
    freeze.add_argument("--reference-root", required=True, type=Path)
    freeze.add_argument("--archive", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser(
        "validate", help="rehash all bytes bound by an existing manifest"
    )
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset-root", type=Path)
    validate.add_argument("--archive", type=Path)

    arguments = parser.parse_args(argv)
    if arguments.mode == "freeze":
        output = freeze_imagenet_sketch_provenance(
            dataset_root=arguments.dataset_root,
            reference_root=arguments.reference_root,
            archive=arguments.archive,
            output=arguments.output,
        )
        validated = validate_imagenet_sketch_provenance(output)
    else:
        output = arguments.manifest.expanduser().resolve()
        validated = validate_imagenet_sketch_provenance(
            output,
            dataset_root=arguments.dataset_root,
            archive=arguments.archive,
        )
    print(json.dumps({
        "status": "PASS",
        "mode": arguments.mode,
        "scientific_result": False,
        "manifest": str(output),
        "dataset_root": str(validated.dataset_root),
        "archive": str(validated.archive_path),
        "rows": len(validated.discovery.paths),
        "classes": len(validated.discovery.wnids),
        "file_tree_sha256": validated.discovery.file_tree_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
