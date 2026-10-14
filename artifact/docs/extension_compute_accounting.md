# RTX extension compute accounting

This prose is included for protocol transparency in the anonymous source
export, but it is not an evidence payload. The machine-readable terminal
records and generated `COMPUTE_ACCOUNTING.json` live only in the separately
authenticated RTX extension evidence bundle. “Authenticated” below means
authenticated by that bundle's verifier and independently supplied top-level
manifest digest; the prose and source checks alone do not establish the values.

The frozen TTA terminal records provide exact algorithm counters, algorithm
wall time, and peak VRAM for all 1,080 planned units. A deterministic audit
deduplicates shared raw-source work by 180 `identity_sha256` values and shared
EATA-Fisher work by 4 identities. The audit requires exactly 180 terminals per
method, exactly six references to every raw-source identity (one per method),
and exactly 45 EATA-only references to every Fisher identity (15 corruptions
by three split seeds). It also rejects missing, additional, negative, or
non-finite fields in each authenticated compute record.

## What the separate extension evidence authenticates

| Quantity | Authenticated value |
|---|---:|
| TTA terminal units | 1,080 |
| Finite PASS units | 1,026 |
| Authenticated divergence units | 54 |
| Sum of per-unit algorithm wall time | 236,658.291 s (65.738 h) |
| Deduplicated shared raw-source wall time | 22,750.326 s (6.320 h) |
| Deduplicated shared EATA-Fisher wall time | 174.998 s (0.049 h) |
| Recorded TTA work total | 259,583.615 s (72.107 h) |
| Maximum per-unit algorithm peak VRAM | 20,998,524,928 bytes |
| Forward-equivalent examples in terminal-unit algorithm records | 19,497,762 |
| Backward examples in terminal-unit algorithm records | 3,538,709 |
| Optimizer steps in terminal-unit algorithm records | 52,424 |
| Augmentation views in terminal-unit algorithm records | 2,833,729 |

The 72.107-hour value is a sum of recorded serial work, not elapsed campaign
wall-clock time. Shared work is counted once rather than once per referencing
method. The 54 divergence records stop at their authenticated failure boundary
and therefore contain lower-bound compute. Source-method algorithm time is
zero because its inference is represented by the separately deduplicated
shared raw-source artifact.

All terminal records bind the same pre-compute runtime manifest: Linux 6.8 on
x86-64, Python 3.12.3, an NVIDIA RTX PRO 4500 Blackwell (compute capability
12.0), CUDA 13.0, cuDNN 9.2, PyTorch 2.13.0, torchvision 0.28.0,
Transformers 5.14.1, NumPy 2.5.1, timm 1.0.28, safetensors 0.8.0, and Pillow
12.3.0. Numerical libraries and PyTorch intra/inter-op execution use one
thread, with one data-loader worker. These facts authenticate the accelerator
and software environment, but not the host CPU/RAM quantities listed below.

The exact counter totals and the 24 dataset/encoder/method breakdowns are
generated as `COMPUTE_ACCOUNTING.json` by
`scripts/iclr_extension/build_extension_release.py`; they should be used in
tables rather than manually transcribed values.

## Play-3 v2 accounting after the not-yet-run extension

The additive Play-3 v2 runner now has checksum-bound, grouped stage timing.
When the locked 162-unit panel is executed, its aggregate will deduplicate the
nine unit references to each actual dataset/encoder/split-seed event and
separately total split selection, shared calibration/Bures fitting, geometry
and dense-cost preparation, per-budget certificate search, exact validation,
posthoc attribution, and unit-array serialization. Full-reconstruction
verification has an independent timing inventory. No numerical v2 compute
value exists yet; code readiness is not execution evidence. The precise scope
and exclusions are documented in `docs/play3_optimization_gate.md`.

## What must remain disclosed as missing

The transferred execution tree does not authenticate:

- the CPU model/count or host-RAM capacity at run time;
- peak host-memory or peak storage use;
- per-unit Play-1 and completed Play-3 v1 time and peak resources;
- complete feature-extraction and end-to-end campaign elapsed time; or
- total compute across preliminary and failed research runs.

The server-side capture helper can record current host hardware, but its output
is explicitly posthoc and cannot retroactively establish run-time or peak
usage. Until the launcher logs and failed/preliminary-run inventory are
collected, this evidence does not support changing the compute checklist answer
to “Yes.”
