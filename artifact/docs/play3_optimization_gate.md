# Play-3 v2 optimization and breadth gate

Status: protocol and execution code are ready; no v2 scientific result exists.
Do not describe the five-point optimization gate, ImageNet-V2/ImageNet-R
replication, or geometry gate as passed until the sealed aggregate described
below has been produced and verified on the server.

The 1,024-row result is a floating-point numerical primal--dual certificate,
not a formally verified real-arithmetic proof of the global optimum. The
separate 64-row MILP diagnostic is required to terminate with a proven solver
optimum at zero requested relative gap, but it validates only that smaller
subproblem.

## Additive lineage and the 5,000-row ImageNet-R constraint

This gate does not modify the completed Play-3 v1 config, lock, units, or
hashes. The new config records those files as prior lineage and discloses that
the v1 ImageNet-Sketch/full-Mahalanobis outcomes had already been inspected.

The frozen ImageNet head used 25 of the 50 validation images in every class.
After those head-fit IDs are excluded, restricting the reference endpoint to
ImageNet-R's 200-WNID intersection leaves exactly `200 * 25 = 5000` rows. A
4,096-row calibration plus 1,024-row witness would require 5,120 rows and is
therefore impossible without leakage or leaving the declared intersection.
The not-yet-frozen v2 protocol instead uses the same 3,840-row calibration and
1,024-row witness for all three shifts (4,864 reference rows total). Every
1,024-row certificate is accompanied by a separate 64-row exact diagnostic.
The smaller exact subset is deliberate: it has 4,096 binary variables, while
the abandoned 256-row design had 65,536 and exceeded its nominal time limit in
a local pilot. The exact diagnostic validates implementation of the bounds on
a tractable subproblem; it does not make the 1,024-row search exact.

The search has no stochastic initialization or random restart: canonical row
selection, Hungarian assignments, greedy ordering, and the multiplier schedule
are deterministic. Consequently there is no separate restart distribution to
summarize. Under the frozen Python/SciPy/HiGHS/BLAS/thread runtime,
`verify-panel` instead requires bitwise-identical assignments and identical
summaries on a complete rerun. Any mismatch fails the verification gate.

The six stores preserve indexed atoms, including repeated images and exact
duplicate feature rows. Immutable authenticated sample IDs are opaque
administrative tie keys: feature bytes determine the primary order, and IDs
are consulted only among exact feature ties. IDs never enter a distance,
readout, prediction reward, or budget. They are included in indexed-atom
digests and inverse maps, and they are never parsed for WNID semantics. This
defines permutation invariance under a joint permutation of rows and IDs and
avoids silently dropping genuine duplicate observations. One narrow caveat is
recorded explicitly: for exact duplicate features, the ID tie order can decide
which indexed boundary atom receives an otherwise exchangeable assignment and
therefore can change its posthoc WNID attribution; it cannot change the
feature-only optimization objective. The WNID statistics remain descriptive
and conditional on the serialized assignment.

## Required six-store contract

The raw ImageNet-V2/R extraction adapter and authenticated Sketch repackager
are implemented in `prepare_play3_optimization_stores.py`; the remaining
blocking task is to run them against the official server-side inputs. Do not
fabricate or merely relabel the completed Sketch v1 store as a v2 store. The
preparer revalidates and republishes six native v2 feature stores (three
datasets by two encoders), after which `freeze-lock` performs a full hash and
semantic validation before any new witness outcome is computed. See
`docs/play3_store_preparation.md` for the exact preparation commands.

Each store must contain these additional NPY roles:

- `reference_wnids`: one WNID for each reference row;
- `target_wnids`: one WNID for each target row.

Its metadata must contain the dataset-specific provenance object named in the
config and this exact logical object (the concrete WNID list and hashes vary):

```json
{
  "class_intersection": {
    "policy": "all_official_imagenet_1k_wnids",
    "wnids": ["n01440764", "..."],
    "wnids_sha256": "SHA256_OF_CANONICAL_JSON_WNID_LIST",
    "reference_endpoint_restricted": true,
    "target_endpoint_restricted": true,
    "reference_wnids_role": "reference_wnids",
    "target_wnids_role": "target_wnids"
  }
}
```

The WNID list must be unique and lexicographically ordered. Both endpoint
WNID arrays must cover exactly that list and may contain no value outside it.
The runner reads these arrays during store-map preflight to validate hashes,
shapes, and declared class support. That integrity read is recorded explicitly;
the arrays are not supplied to the certificate optimizer. Their scientific use
begins only after an assignment is fixed, when per-WNID descriptive sufficient
statistics are formed.

The assignment reward is prediction disagreement, so the optimizer does use
the two frozen-readout prediction arrays. It does not use external ground-truth
labels, row-level WNIDs, semantic sample identities, or clean--target pairings.
Artifacts encode this distinction explicitly as
`frozen_readout_predictions_used_for_optimization: true` and
`external_ground_truth_labels_used_for_optimization: false`; the older,
ambiguous phrase “labels not used” is not the v2 contract.

The expected target row counts are 50,000 (the already specified balanced
Sketch subset), 10,000 (V2 matched-frequency), and 30,000 (R). The expected
reference counts after both WNID restriction and head-fit exclusion are
25,000, 25,000, and 5,000 respectively. The store must preserve the frozen
readout and model provenance from its encoder's authenticated ImageNet source
store. Pairing must remain `unavailable`.

The preparation step additionally authenticates the raw dataset variant,
file tree, official class-to-WNID map, encoder checkpoint, preprocessing,
feature layer/pooling, target extraction order, and exclusion of every frozen
head-fit ID. Until that executable path is run on the server, store
construction—and therefore the empirical breadth gate—remains blocked.

The required `optimization_gate_preparation` record must be `PASS`, use schema
`marginal_alignment.iclr_extension.optimization_store_preparation.v1`, bind
the dataset and encoder, bind the hashes of the head-fit IDs, reference IDs,
both WNID arrays, and class-intersection list, bind authenticated reference and
target-extraction manifests, and assert that head-fit IDs are disjoint from
the restricted reference endpoint. `freeze-lock` checks those fields and also
checks the actual ID sets for disjointness.

## Exact server runbook after store preparation

Work from a clean copy of this source tree. Set absolute server paths without
editing the frozen config:

```bash
set -Eeuo pipefail

PROJECT=/absolute/server/path/beyond-marginals
PY=/absolute/server/path/python
STORE_ROOT=/absolute/server/path/play3_v2_feature_stores
STORE_MAP="$STORE_ROOT/play3_optimization_store_map.json"
LOCK=/absolute/server/path/PLAY3_OPTIMIZATION_LOCK.json
UNITS=/absolute/server/path/play3_v2_units
VERIFY=/absolute/server/path/play3_v2_verification
AGGREGATE=/absolute/server/path/PLAY3_OPTIMIZATION_AGGREGATE.json
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Use the store map generated and semantically validated by `prepare-panel`; do
not manually reconstruct its six paths from the example file. Before reading
any V2/R witness outcome, freeze the new lock:

```bash
cd "$PROJECT"
test -f "$STORE_MAP"
test ! -e "$LOCK"
PYTHONPATH=src "$PY" scripts/iclr_extension/run_play3_optimization_gate.py \
  freeze-lock \
  --config configs/iclr_extension/play3_optimization_gate.json \
  --store-map "$STORE_MAP" \
  --output "$LOCK"
sha256sum "$LOCK"
```

Run the complete 162-unit panel in four deterministic shards. Each unit is
one dataset, encoder, geometry, split seed, and 1/5/10-percent cost anchor.
`--resume` accepts only complete units whose manifests, arrays, lock, and
sufficient statistics validate; it never accepts a partial directory.
Sharding keeps all nine geometry-by-budget units for a
dataset/encoder/split-seed group in one process. Within that group the runner
selects rows and fits Bures once, constructs each geometry's 1,024-by-1,024
cost matrix once, and reuses that immutable problem for all three budgets.
This changes no unit definition. A separately launched `verify-panel` process
reconstructs the same shared work independently, once per group.

```bash
mkdir -p "$UNITS/logs"
panel_pids=()
for SHARD in 0 1 2 3; do
  PYTHONPATH=src "$PY" scripts/iclr_extension/run_play3_optimization_gate.py \
    run-panel --lock "$LOCK" --store-map "$STORE_MAP" \
    --output-root "$UNITS" --shard-count 4 --shard-index "$SHARD" --resume \
    >"$UNITS/logs/shard_${SHARD}.log" 2>&1 &
  panel_pids+=("$!")
done
panel_failed=0
for pid in "${panel_pids[@]}"; do
  if ! wait "$pid"; then
    panel_failed=1
  fi
done
(( panel_failed == 0 )) || {
  echo "At least one Play-3 optimization shard failed" >&2
  exit 1
}

UNIT_COUNT=$(find "$UNITS" -type f -name UNIT_MANIFEST.json | wc -l)
[[ "$UNIT_COUNT" -eq 162 ]]
for SHARD in 0 1 2 3; do
  test -f "$UNITS/SHARD_$(printf '%03d' "$SHARD")_DONE.json"
done
```

Progress appears in jumps of nine units because a process computes an entire
dataset/encoder/split-seed group before publishing its unit directories. The
four shards contain 45, 45, 36, and 36 units, respectively. A quiet log while a
process remains active is therefore not evidence of a stall.

### Authenticated compute accounting

Each actual dataset/encoder/split-seed computation produces one timing event
using Python's monotonic `time.perf_counter_ns` clock. The event separately
records store validation, deterministic split selection, the shared
calibration/Bures geometry fit, each geometry derivation, each dense problem
preparation, and certificate, exact-MILP, and posthoc stages for every budget.
The event binds the lock, store, group identity, all nine unit IDs, and its own
canonical-JSON checksum. It is copied into the nine unit manifests so every
unit remains independently auditable.

Those nine copies are **not nine measurements**. Aggregation validates their
checksums, requires exactly one owner record for each event, and deduplicates
by event ID. A clean run therefore has 18 group events. If a partial resume
recomputes a group and both successful attempts are represented by sealed
units, both distinct events are counted once, truthfully recording the retry.
The aggregate also sums non-group unit invocation time and NPY serialization
time without recharging an owner unit for its embedded group computation.
Because four shards may execute concurrently, this total is a sum of
non-overlapping process intervals, not campaign wall-clock time.

`verify-panel` creates a separate set of events with purpose
`full_reconstruction_verification`. Each verification-shard manifest contains
its rederived, deduplicated timing summary, and the final aggregate combines
only disjoint shard inventories. This makes scientific execution and
independent reconstruction costs separately reportable.

Three limits stay explicit rather than being estimated: portable authenticated
peak RSS is unavailable; a self-contained manifest cannot record the duration
of its own seal/write/fsync/atomic rename; and a failed or interrupted attempt
that publishes no sealed unit or verification shard is not observable from
these artifacts. Server campaign logs are required for those quantities.

Fully reconstruct all assignments, costs, prediction rewards, bijections, and
64-row diagnostic MILPs from the locked endpoints. These verification shards are
compute-intensive by design:

```bash
mkdir -p "$VERIFY"
verify_pids=()
for SHARD in 0 1 2 3; do
  # A verification manifest is written atomically only after that whole shard
  # passes. Preserve completed shards when resuming an interrupted verification.
  if [[ -f "$VERIFY/shard_${SHARD}.json" ]]; then
    continue
  fi
  PYTHONPATH=src "$PY" scripts/iclr_extension/run_play3_optimization_gate.py \
    verify-panel --lock "$LOCK" --store-map "$STORE_MAP" \
    --units-root "$UNITS" --shard-count 4 --shard-index "$SHARD" \
    --output "$VERIFY/shard_${SHARD}.json" \
    >"$VERIFY/shard_${SHARD}.log" 2>&1 &
  verify_pids+=("$!")
done
verify_failed=0
for pid in "${verify_pids[@]}"; do
  if ! wait "$pid"; then
    verify_failed=1
  fi
done
(( verify_failed == 0 )) || {
  echo "At least one Play-3 verification shard failed" >&2
  exit 1
}

VERIFY_COUNT=$(find "$VERIFY" -maxdepth 1 -type f -name 'shard_*.json' | wc -l)
[[ "$VERIFY_COUNT" -eq 4 ]]
```

Only after all four verification manifests report `PASS`, aggregate the exact
locked inventory. The following descriptive calculation keeps the serialized
assignments fixed and does not refit the optimizer:

```bash
set +e
PYTHONPATH=src "$PY" scripts/iclr_extension/run_play3_optimization_gate.py \
  aggregate --lock "$LOCK" --units-root "$UNITS" \
  --verification-root "$VERIFY" --output "$AGGREGATE"
AGGREGATE_EXIT=$?
set -e
[[ "$AGGREGATE_EXIT" -eq 0 || "$AGGREGATE_EXIT" -eq 2 ]]
sha256sum "$AGGREGATE" "$VERIFY"/shard_*.json

PYTHONPATH=src "$PY" - "$AGGREGATE" <<'PY'
import json
import sys
from pathlib import Path

from marginal_alignment.campaign.artifacts import verify_manifest_checksum

aggregate = json.loads(Path(sys.argv[1]).read_text())
verify_manifest_checksum(aggregate)
assert aggregate["status"] == "PASS"
assert aggregate["artifact_status"] == "PASS"
assert aggregate["status_interpretation"] == "artifact_integrity_and_completeness_only"
assert aggregate["scientific_gate_status"] in {"PASS", "FAIL"}
assert aggregate["all_required_gates_pass"] == (
    aggregate["scientific_gate_status"] == "PASS"
)
assert aggregate["expected_units"] == aggregate["observed_units"] == 162
assert aggregate["zero_silent_drops"] is True
print(json.dumps({
    "artifact_status": aggregate["artifact_status"],
    "scientific_gate_status": aggregate["scientific_gate_status"],
    "failed_required_scientific_gate_keys": aggregate[
        "failed_required_scientific_gate_keys"
    ],
    "gates": {
        name: gate["status"] for name, gate in aggregate["gates"].items()
    },
}, indent=2, sort_keys=True))
PY

if [[ "$AGGREGATE_EXIT" -eq 2 ]]; then
  echo "Aggregate artifact preserved; at least one required scientific gate failed" >&2
  exit 2
fi
```

The aggregate's top-level `status: PASS` and `artifact_status: PASS`
authenticate complete aggregation; they do **not** mean every scientific gate
passed. `scientific_gate_status` and `all_required_gates_pass` are the
fail-closed scientific decision. The CLI prints both statuses and returns zero
only when all required gates pass. If a required scientific gate fails, it
first writes and seals the aggregate, prints `scientific_gate_status: FAIL`,
and returns exit code 2 so campaign automation stops while preserving the
failed evidence for reporting.

The aggregate fails closed on missing or extra units. Split repeats are
pooled as sufficient statistics inside each WNID before the fixed WNID panel
is resampled. Its label is **descriptive conditional WNID-cluster resampling
interval, assignments fixed/no optimizer refit**. It is a prespecified
robustness screen—not a population interval, sampling-valid confidence
interval, or confirmatory inference procedure. Optimization succeeds only if
**every** 1%, 5%, and 10% unit
has a gap at most five percentage points; an average cannot hide a failure.
The aggregate records separate PASS/FAIL decisions for exact MILPs, each
encoder's diagnostic breadth thresholds, the primary breadth gates, and the
geometry gate. Each scientific breadth gate requires the threshold on the
same two or more datasets for both encoders; disjoint encoder-specific
success sets cannot pass. A scientific conclusion
must follow those recorded decisions, including a narrowed claim after any
failure.

The lock also captures the exact Python, NumPy, SciPy, bundled HiGHS,
BLAS/LAPACK, platform, and numerical-thread environment. Execution and
verification reject a runtime that differs from the pre-outcome frozen
contract.

## Local validation

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src python -m pytest \
  tests/iclr_extension/test_optimization_certificate.py \
  tests/iclr_extension/test_play3_optimization_gate.py -q
python -m py_compile \
  src/marginal_alignment/iclr_extension/optimization_certificate.py \
  scripts/iclr_extension/run_play3_optimization_gate.py
```
