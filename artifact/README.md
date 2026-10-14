# Experimental artifact B

Anonymous source, locked configurations, and compact evidence for the
accompanying ICLR submission. The artifact audits task-relevant correspondence
ambiguity under explicit cost models, then tests what paired calibration can
and cannot certify about recovery error. It includes only studies reported in
the manuscript or appendix; exploratory branches not used by the paper,
engineering failures, deployment scripts, logs, feature stores, and raw
datasets are excluded.

## Fast verification

The frozen evidence is independently checkable on a CPU-only machine:

```bash
python3 scripts/verify_results.py
python3 scripts/verify_paper_values.py
python3 scripts/audit_anonymity.py
```

Check the release manifest with `shasum -a 256 -c SHA256SUMS` on macOS or
`sha256sum -c SHA256SUMS` on Linux.

Or run `make verify audit`. The first verifier parses the complete public
result inventory, checks terminal records, and authenticates the expected row
and unit counts. The second checks manuscript-facing invariants.

## Included studies

| Study family | What it establishes | Frozen evidence |
|---|---|---|
| Primary alignment campaign | Synthetic identification controls, natural/corruption benchmarks, inference, and method comparisons | `results/primary` |
| Orientation and optimization panels | Held-out orientation checks and exact optimization gates | `results/orientation`, `results/ambiguity/play3` |
| Full-rank paired confirmation | Untruncated paired-ridge sensitivity | `results/ambiguity/full_rank` |
| Common-cost audit | Unpaired versus paired endpoint profiles, realized accuracy, and numerical brackets | `results/ambiguity/common_cost` |
| Witness-size scaling | Nested 1,024--8,192-row certificates and class-opportunity diagnostics | `results/ambiguity/witness_scaling` |
| Cost-scale audit | Cross-condition interpretation of the normalized budget | `results/ambiguity/cost_scale` |
| Paired recovery bounds | Precursor audit, final eight-cell 160-batch follow-up, and labeled post-hoc direct signed comparator | `results/paired_recovery` |

See [the evidence map](docs/EVIDENCE.md) and [reproduction guide](docs/REPRODUCE.md).

## Repository layout

```text
configs/              paper experiment configurations
protocol/             prospective locks and protocol records
src/                   estimators, certificates, and campaign code
scripts/               scientific entry points and release checks
tests/                 focused numerical/protocol tests
results/               compact, read-only paper evidence
provenance/            source and public-sanitization lineage
docs/                  data, evidence, provenance, and run guides
```

Public result copies preserve scientific values while replacing machine-local
path strings. The original and public SHA-256 values are linked in
`provenance/SANITIZATION_MANIFEST.json`. New runs belong under
`results/reproduction/`; never overwrite the frozen evidence.

Raw datasets, pretrained encoders, and large feature stores are not
redistributed. Their acquisition and identity requirements are documented in
[the data guide](docs/DATA.md). Code is provided under [LICENSE](LICENSE).
