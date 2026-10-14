#!/usr/bin/env python3
"""Dedicated verifier entry point; accepts the runner's ``verify`` arguments."""

import sys

from marginal_alignment.paired_recovery_bounds.runner import main


if __name__ == "__main__":
    raise SystemExit(main(["verify", *sys.argv[1:]]))
