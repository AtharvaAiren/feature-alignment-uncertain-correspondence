# Claim-to-artifact map

| Manuscript claim or analysis | Authoritative artifact |
|---|---|
| Primary synthetic and real alignment comparisons | `results/primary/paper_outputs/*.json` |
| Confirmatory intervals and multiplicity | `results/primary/inference.json` |
| Complete primary row inventory | `results/primary/rows.jsonl.gz` and `summary.json` |
| Orientation selector and optimization gate | `results/orientation/AGGREGATE.primary.json`, `results/ambiguity/play3/aggregate.json` |
| Full-rank paired sensitivity | `results/ambiguity/full_rank/SUMMARY.json`, `INFERENCE.json` |
| Common-cost unpaired/paired profiles and realized accuracy | `results/ambiguity/common_cost/REVIEWER_FOLLOWUP_AGGREGATE.json` and JSONL companions |
| Witness-size scaling and class opportunity | `results/ambiguity/witness_scaling/WITNESS_SIZE_SCALING_AGGREGATE.json` and JSONL companions |
| Normalized-cost interpretation | `results/ambiguity/cost_scale/aggregate.json` |
| Paired recovery precursor | `results/paired_recovery/precursor/{AGGREGATE,DONE,VERIFICATION}.json` |
| Final paired recovery/calibration study | `results/paired_recovery/followup/results/{AGGREGATE,DONE,VERIFICATION}.json` and 160 unit files |
| Post-hoc direct signed comparator | `results/paired_recovery/posthoc_direct_signed/RESULT.json`, reproduced by `scripts/paired_recovery_followup/posthoc_direct_signed.py` |

Feasible assignments provide lower bounds; Lagrangian relaxations provide
upper bounds. The profile quantifies ambiguity inside the declared model, not
whether an alignment method will improve accuracy. The paired-recovery study is
a distinct, calibration-based estimand.
The direct signed comparator is explicitly outcome-informed and post-hoc; it
does not alter or relabel the prospectively frozen follow-up.
