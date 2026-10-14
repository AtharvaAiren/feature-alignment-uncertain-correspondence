#!/usr/bin/env python3
"""Read-only inventory of documented server feature caches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from marginal_alignment.campaign.cache_import import write_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("MARGINAL_FEATURE_ROOT", "data/feature_stores_v2"),
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    root = Path(arguments.cache_root)
    stores = [path for dataset in ("cifar100", "cifar100c", "imagenet", "imagenet_c")
              for path in (root / dataset).glob("*") if path.is_dir()]
    output = write_inventory([str(path) for path in stores], arguments.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
