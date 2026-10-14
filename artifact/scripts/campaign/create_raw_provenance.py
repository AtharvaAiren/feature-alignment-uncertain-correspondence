#!/usr/bin/env python3
"""Create content-backed CIFAR-100-C or ImageNet-C archive provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.campaign.artifacts import sha256_file
from marginal_alignment.campaign.feature_protocol import atomic_json
from marginal_alignment.campaign.raw_extract import (
    CIFAR_PROVENANCE_SCHEMA,
    IMAGENET_PROVENANCE_SCHEMA,
    validate_cifar100c_provenance,
    validate_imagenetc_provenance,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    cifar = subparsers.add_parser("cifar100c")
    cifar.add_argument("--archive", required=True, type=Path)
    cifar.add_argument("--output", required=True, type=Path)
    imagenet = subparsers.add_parser("imagenetc")
    imagenet.add_argument("--archive", required=True, action="append", type=Path)
    imagenet.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    if arguments.dataset == "cifar100c":
        archive = arguments.archive.expanduser().resolve()
        value = {
            "schema_version": CIFAR_PROVENANCE_SCHEMA,
            "dataset": "CIFAR-100-C",
            "source_url": "https://zenodo.org/records/3555552",
            "archive_file": str(archive),
            "archive_sha256": sha256_file(archive),
        }
        atomic_json(arguments.output, value)
        validate_cifar100c_provenance(arguments.output)
    else:
        archives = [path.expanduser().resolve() for path in arguments.archive]
        value = {
            "schema_version": IMAGENET_PROVENANCE_SCHEMA,
            "dataset": "ImageNet-C",
            "source_url": "https://zenodo.org/records/2235448",
            "archives": [
                {"file": str(path), "sha256": sha256_file(path)} for path in archives
            ],
        }
        atomic_json(arguments.output, value)
        validate_imagenetc_provenance(arguments.output)
    print(json.dumps({
        "status": "PASS", "dataset": arguments.dataset,
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
