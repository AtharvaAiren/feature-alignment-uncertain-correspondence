#!/usr/bin/env python3
"""Content-seal one reusable legacy feature directory."""

from __future__ import annotations

import argparse

from marginal_alignment.campaign.cache_import import seal_legacy_arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    arguments = parser.parse_args()
    print(seal_legacy_arrays(arguments.store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
