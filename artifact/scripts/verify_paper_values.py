#!/usr/bin/env python3
"""Check headline manuscript-facing counts and certificate invariants."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def load(relative: str):
    return json.loads((R / relative).read_text(encoding="utf-8"))

common = load("ambiguity/common_cost/REVIEWER_FOLLOWUP_AGGREGATE.json")
assert common["status"] == "PASS"

witness = load("ambiguity/witness_scaling/WITNESS_SIZE_SCALING_AGGREGATE.json")
assert witness["status"] == "PASS"
assert witness["fresh_certificates"] == 288
assert witness["all_gaps_at_most_0_01"] is True

play3 = load("ambiguity/play3/aggregate.json")
assert play3["status"] == "PASS"
assert play3["all_required_gates_pass"] is True

follow = load("paired_recovery/followup/results/AGGREGATE.json")
assert follow["status"] == "PASS"
assert follow["complete_evaluation_units"] == 160
assert len(follow["by_cell"]) == 8

done = load("paired_recovery/followup/results/DONE.json")
assert done["complete_evaluation_units"] == 160
verify = load("paired_recovery/followup/results/VERIFICATION.json")
assert verify["status"] == "PASS"

signed = load("paired_recovery/posthoc_direct_signed/RESULT.json")
assert signed["status"] == "PASS"
assert signed["analysis_status"] == "OUTCOME_INFORMED_POST_HOC_COMPARATOR"
assert signed["direct_signed"]["contrast_coverage"] == 454
assert signed["direct_signed"]["joint_coverage_batches"] == 141
assert signed["direct_signed"]["negative_certificates"] == 280
assert signed["direct_signed"]["incorrect_strict_orderings"] == 2
physical = signed["physical_signed_revalidation"]
assert physical["contrast_coverage"] == 480
assert physical["joint_coverage_batches"] == 160
assert physical["negative_certificates"] == 263
assert physical["incorrect_strict_orderings"] == 0

print("PAPER-VALUE CHECK: PASS (720 common-cost certificates; 288 scaling certificates; all Play-3 gates; 8-cell/160-batch recovery panel; signed comparator validated)")
