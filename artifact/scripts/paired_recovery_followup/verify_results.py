#!/usr/bin/env python3
import sys

from marginal_alignment.paired_recovery_followup.runner import main


if __name__ == "__main__":
    raise SystemExit(main(["verify", *sys.argv[1:]]))
