# Play-3 v2 natural-shift store preparation

Status: executable input preparation only. No ImageNet-V2/ImageNet-R witness
outcome has been computed, and this procedure does not freeze or pass the
scientific gate.

The preparation command produces the six native stores required by
`play3_optimization_gate.json`: ImageNet-Sketch, ImageNet-V2 matched-frequency,
and ImageNet-R, each under the locked DINOv3 and CLIP encoders. It repackages
the two completed, authenticated Sketch stores without recomputing their
features and performs fresh resumable extraction for V2/R.

## Fail-closed input contract

The preparer rejects an input unless all of the following revalidate:

- the raw archive hash and its normalized image members equal the extracted
  image tree byte for byte;
- the ImageNet-V2 root has exactly integer directories `0` through `999`, ten
  images per class, and 10,000 rows total;
- the ImageNet-R root has exactly 200 official ImageNet WNID directories and
  30,000 rows total;
- an exact official `imagenet_class_index.json` maps keys `0..999` to 1,000
  unique, lexicographically ordered WNIDs;
- the reference store is one exact manifest in the completed base-campaign
  allowlist and its checkpoint, preprocessing, feature layer, pooling, frozen
  head, and 25-shot-per-class fit IDs agree with the loaded encoder;
- every head-fit ID is excluded after WNID restriction, leaving exactly
  25,000 reference rows for Sketch/V2 and 5,000 for R;
- each output has actual `reference_wnids` and `target_wnids` arrays, covers
  exactly the declared intersection, and declares pairing unavailable.

The gate's `class_index_sha256` is the canonical hash of the declared
intersection WNID list (1,000 for Sketch/V2 and 200 for R). The separate
`official_class_to_wnid_mapping_file_sha256` and
`official_class_to_wnid_mapping_sha256` fields bind, respectively, the exact
class-index JSON bytes and its canonical full 1,000-class mapping. These fields
must not be conflated.

Endpoint labels are deliberately omitted from the published v2 stores. They
are checked as structural extraction inputs and reduced to a digest, while
row-level WNIDs remain in separate post-hoc roles. The optimizer receives only
features and the frozen readout.

## Exact server procedure

First transfer a clean source tree and set paths to already-present official
archives, extracted roots, class index, authenticated base stores, and the two
completed Sketch v1 stores. This code never downloads or substitutes inputs.

```bash
set -Eeuo pipefail

PROJECT=/absolute/server/path/beyond-marginals
PY=/absolute/server/path/python
MODEL_PATHS=/absolute/server/path/model_paths.sh
CLASS_INDEX=/absolute/server/path/imagenet_class_index.json
V2_ROOT=/absolute/server/path/imagenetv2-matched-frequency-format-val
V2_ARCHIVE=/absolute/server/path/imagenetv2-matched-frequency-format-val.tar.gz
R_ROOT=/absolute/server/path/imagenet-r
R_ARCHIVE=/absolute/server/path/imagenet-r.tar
INPUTS=/absolute/server/path/play3_store_preparation_inputs.json
V2_PROVENANCE=/absolute/server/path/imagenet_v2_provenance.json
R_PROVENANCE=/absolute/server/path/imagenet_r_provenance.json
STORE_ROOT=/absolute/server/path/play3_v2_feature_stores
STORE_MAP="$STORE_ROOT/play3_optimization_store_map.json"
PREPARATION_LOG=/absolute/server/path/play3_store_preparation.log

cd "$PROJECT"
test -f "$MODEL_PATHS"
# This trusted server-side file must set the exact local checkpoint paths
# used by the completed campaign. It must not download or change a checkpoint.
source "$MODEL_PATHS"
: "${MARGINAL_DINOV3_MODEL:?MODEL_PATHS did not set MARGINAL_DINOV3_MODEL}"
: "${MARGINAL_CLIP_MODEL:?MODEL_PATHS did not set MARGINAL_CLIP_MODEL}"
# Make the two validated shell values visible to the Python extraction process
# even if MODEL_PATHS assigned rather than exported them.
export MARGINAL_DINOV3_MODEL MARGINAL_CLIP_MODEL
test -d "$MARGINAL_DINOV3_MODEL"
test -d "$MARGINAL_CLIP_MODEL"
test -f "$CLASS_INDEX"
test -d "$V2_ROOT"
test -f "$V2_ARCHIVE"
test -d "$R_ROOT"
test -f "$R_ARCHIVE"
sha256sum "$MODEL_PATHS"

PYTHONPATH=src "$PY" -m pytest \
  tests/iclr_extension/test_natural_shift_store.py \
  tests/iclr_extension/test_optimization_certificate.py \
  tests/iclr_extension/test_play3_optimization_gate.py -q
```

The raw V2/R roots, archives, and official class-index JSON above are
prerequisites, not bundled assets. Their presence and bytes have not been
verified in this local workspace; the two `freeze-provenance` calls below are
the first server-side authentication gate and fail before model loading.

Freeze V2/R acquisition evidence before feature or witness computation. Each
command re-reads the complete archive and extracted image tree:

```bash
if [[ -f "$V2_PROVENANCE" ]]; then
  PYTHONPATH=src "$PY" scripts/iclr_extension/prepare_play3_optimization_stores.py \
    validate-provenance --provenance "$V2_PROVENANCE" \
    --dataset-root "$V2_ROOT" --archive "$V2_ARCHIVE" \
    --class-index "$CLASS_INDEX"
else
  PYTHONPATH=src "$PY" scripts/iclr_extension/prepare_play3_optimization_stores.py \
    freeze-provenance --dataset imagenet_v2 --dataset-root "$V2_ROOT" \
    --archive "$V2_ARCHIVE" --class-index "$CLASS_INDEX" \
    --output "$V2_PROVENANCE"
fi

if [[ -f "$R_PROVENANCE" ]]; then
  PYTHONPATH=src "$PY" scripts/iclr_extension/prepare_play3_optimization_stores.py \
    validate-provenance --provenance "$R_PROVENANCE" \
    --dataset-root "$R_ROOT" --archive "$R_ARCHIVE" \
    --class-index "$CLASS_INDEX"
else
  PYTHONPATH=src "$PY" scripts/iclr_extension/prepare_play3_optimization_stores.py \
    freeze-provenance --dataset imagenet_r --dataset-root "$R_ROOT" \
    --archive "$R_ARCHIVE" --class-index "$CLASS_INDEX" \
    --output "$R_PROVENANCE"
fi
```

Copy `configs/iclr_extension/play3_store_preparation_inputs.example.json` to
`$INPUTS` and replace every placeholder. In particular, use the exact base
ImageNet-C reference manifests named in the completed campaign's external
input manifest, and the completed Play-3 v1 Sketch manifests. Then run the
single resumable panel command:

```bash
test -f "$INPUTS"
PYTHONPATH=src "$PY" scripts/iclr_extension/prepare_play3_optimization_stores.py \
  prepare-panel --input-map "$INPUTS" \
  > "$PREPARATION_LOG" 2>&1
```

V2/R extraction checkpoints only resume when the sealed fingerprint and every
completed block hash still match. A completed destination is never
overwritten. After all six stores verify, the command writes
`play3_optimization_store_map.json` and runs the scientific gate's full
six-store semantic validator. It still does not freeze the scientific lock.
On a retry after an interruption, run the same `prepare-panel` command. If the
store map already exists, preparation has reached its terminal publication
step and deliberately refuses to overwrite it; validate and use that generated
map rather than recreating it by hand.

Inspect the non-scientific outputs before proceeding:

```bash
find "$STORE_ROOT" -name FEATURE_STORE_MANIFEST.json -print | sort
sha256sum "$STORE_MAP" \
  "$STORE_ROOT"/*/*/FEATURE_STORE_MANIFEST.json
grep -nEi 'traceback|error|exception|failed|killed|out of memory|no space|integrity' \
  "$PREPARATION_LOG" || true
```

Only after this succeeds should the separate commands in
`docs/play3_optimization_gate.md` freeze the new pre-outcome lock, execute the
162 units, independently reconstruct them, and aggregate the prespecified
decisions.
