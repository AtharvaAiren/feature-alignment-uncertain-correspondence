#!/usr/bin/env python3
"""Fail closed unless the complete locked 306-store V2 cache is valid."""

from __future__ import annotations

import argparse
import json

from marginal_alignment.campaign.cache_import import validate_store_completeness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    arguments = parser.parse_args()
    report = validate_store_completeness(arguments.root, progress=True)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
