# Reproduction guide

## 1. Verify the published evidence

```bash
python3 scripts/verify_results.py
python3 scripts/verify_paper_values.py
```

This path is CPU-only and needs no external data.

The explicitly post-hoc direct signed comparator is also reproducible from the
published paired records alone:

```bash
mkdir -p results/reproduction
python3 scripts/paired_recovery_followup/posthoc_direct_signed.py \
  results/paired_recovery/followup/results \
  results/reproduction/POSTHOC_DIRECT_SIGNED_CALIBRATION.json
```

## 2. Install the code

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
export PYTHONPATH="$PWD:$PWD/src"
```

Install the `vision` group only to rebuild feature stores or rerun image
campaigns.

## 3. Scientific entry points

Configurations and protocol locks define the reported axes. Entry points expose
their exact CLI through `--help`:

```bash
python -m marginal_alignment.campaign.campaign --help
python scripts/iclr_extension/run_rotation.py --help
python scripts/iclr_extension/run_play3.py --help
python scripts/iclr_extension/run_play3_optimization_gate.py --help
python scripts/iclr_extension/run_full_rank_paired_confirmatory.py --help
python scripts/iclr_extension/run_reviewer_followup.py --help
python scripts/iclr_extension/run_witness_size_scaling.py --help
python scripts/paired_recovery_bounds/run_study.py --help
python scripts/paired_recovery_followup/run_study.py --help
```

Supply external datasets, weights, and feature-store paths explicitly and send
all outputs to a fresh directory outside `results/`.
