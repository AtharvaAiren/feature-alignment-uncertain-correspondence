#!/usr/bin/env python3
"""Estimate peak additional disk needed by the idempotent V2 conversion."""

from __future__ import annotations

import argparse
import json

from marginal_alignment.campaign.cache_import import estimate_v2_materialization_space


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--destination-root", required=True)
    arguments = parser.parse_args()
    report = estimate_v2_materialization_space(
        legacy_root=arguments.legacy_root,
        destination_root=arguments.destination_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
