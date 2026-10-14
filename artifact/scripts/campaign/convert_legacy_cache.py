#!/usr/bin/env python3
"""Convert sealed canonical CIFAR/ImageNet cache arrays into V2 stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.campaign.cache_import import (
    convert_canonical_cifar_cache, convert_indexed_imagenet_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar100_c", "imagenet_c"), required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--corruptions", required=True,
                        help="comma-separated locked corruption names")
    parser.add_argument("--severities", default="1,3,5")
    arguments = parser.parse_args()
    kwargs = {
        "reference_root": arguments.reference_root,
        "target_root": arguments.target_root,
        "destination_root": arguments.destination_root,
        "encoder": arguments.encoder,
        "corruptions": [value for value in arguments.corruptions.split(",") if value],
        "severities": [int(value) for value in arguments.severities.split(",") if value],
    }
    if arguments.dataset == "cifar100_c":
        outputs = convert_canonical_cifar_cache(**kwargs)
    else:
        outputs = convert_indexed_imagenet_cache(**kwargs)
    print(json.dumps({"status": "PASS", "stores": len(outputs),
                      "manifests": [str(path) for path in outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
