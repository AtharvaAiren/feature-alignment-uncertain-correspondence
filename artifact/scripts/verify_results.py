#!/usr/bin/env python3
"""Read-only structural verification of all public scientific evidence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

required_pass = [
    "primary/inference.json",
    "primary/paper_outputs/manifest.json",
    "orientation/RTX_PROFILE_COMPLETE.json",
    "ambiguity/full_rank/DONE.json",
    "ambiguity/common_cost/DONE.json",
    "ambiguity/common_cost/REVIEWER_FOLLOWUP_AGGREGATE.json",
    "ambiguity/witness_scaling/DONE.json",
    "ambiguity/witness_scaling/WITNESS_SIZE_SCALING_AGGREGATE.json",
    "ambiguity/play3/aggregate.json",
    "paired_recovery/precursor/AGGREGATE.json",
    "paired_recovery/precursor/DONE.json",
    "paired_recovery/precursor/VERIFICATION.json",
    "paired_recovery/followup/results/AGGREGATE.json",
    "paired_recovery/followup/results/DONE.json",
    "paired_recovery/followup/results/VERIFICATION.json",
]

parsed = 0
for path in sorted(RESULTS.rglob("*")):
    if not path.is_file():
        continue
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        parsed += 1
    elif path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
        parsed += 1
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
        parsed += 1

for relative in required_pass:
    value = json.loads((RESULTS / relative).read_text(encoding="utf-8"))
    assert value.get("status") == "PASS", (relative, value.get("status"))

def nonempty_lines(relative: str) -> int:
    return sum(bool(x.strip()) for x in (RESULTS / relative).read_text().splitlines())

assert nonempty_lines("ambiguity/common_cost/CERTIFICATE_ROWS.jsonl") == 720
assert nonempty_lines("ambiguity/common_cost/PAIR_STRUCTURE_CONTRASTS.jsonl") == 288
assert nonempty_lines("ambiguity/common_cost/REALIZED_PROFILE_ROWS.jsonl") == 108
assert nonempty_lines("ambiguity/common_cost/SKETCH_PAIRED_BOUND_SEPARATIONS.jsonl") == 180
assert nonempty_lines("ambiguity/witness_scaling/WITNESS_SIZE_ROWS.jsonl") == 288
unit_files = list((RESULTS / "paired_recovery/followup/results/units").rglob("*.json"))
assert len(unit_files) == 160, len(unit_files)

print(f"RESULT VERIFICATION: PASS ({parsed} result files; 720 common-cost certificates; 288 witness rows; 160 recovery units)")

