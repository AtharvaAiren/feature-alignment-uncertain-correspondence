#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${MARGINAL_PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
python_bin="${MARGINAL_PYTHON:-$(command -v python3)}"
source_root="${MARGINAL_LEGACY_FEATURE_ROOT:-${project_root}/data/legacy_features}"
prepared_root="${MARGINAL_PREPARED_FEATURE_ROOT:-${project_root}/data/prepared_features}"
raw_root="${MARGINAL_RAW_FEATURE_ROOT:-${project_root}/data/raw_features}"
v2_root="${MARGINAL_FEATURE_ROOT:-${project_root}/data/feature_stores_v2}"
config="${MARGINAL_FEATURE_BUILD_CONFIG:-${project_root}/configs/campaign/feature_build.json}"
audit_root="${project_root}/results/feature_preflight"
force_raw="${MARGINAL_FORCE_RAW_EXTRACTION:-0}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${audit_root}" "${prepared_root}" "${raw_root}"

case "${force_raw}" in
  0|1) ;;
  *)
    echo "FATAL: MARGINAL_FORCE_RAW_EXTRACTION must be 0 or 1, got ${force_raw}" >&2
    exit 2
    ;;
esac

if [[ "${force_raw}" == "1" ]]; then
  echo "[feature-bootstrap] FORCE-RAW mode: legacy feature matrices will not be read or qualified"
else
  echo "[feature-bootstrap] read-only reuse audit, then raw fallback only where required"
fi
for encoder in dinov3 clip vitb16; do
  for dataset in cifar100_c imagenet_c; do
    pair="${dataset}__${encoder}"
    audit="${audit_root}/${pair}.json"
    pair_source="${source_root}"
    if [[ "${force_raw}" == "1" ]]; then
      echo "[feature-bootstrap] forced raw rebuild: ${pair}"
      "${python_bin}" "${project_root}/scripts/campaign/extract_legacy_from_raw.py" \
        --config "${config}" \
        --output-root "${raw_root}" \
        --datasets "${dataset}" \
        --encoders "${encoder}" \
        --device cuda
      pair_source="${raw_root}"
    elif "${python_bin}" "${project_root}/scripts/campaign/prepare_legacy_sources.py" audit \
      --config "${config}" \
      --source-root "${source_root}" \
      --datasets "${dataset}" \
      --encoders "${encoder}" \
      --output "${audit}"; then
      echo "[feature-bootstrap] reusable: ${pair}"
    else
      echo "[feature-bootstrap] raw rebuild required: ${pair}; blockers in ${audit}"
      "${python_bin}" "${project_root}/scripts/campaign/extract_legacy_from_raw.py" \
        --config "${config}" \
        --output-root "${raw_root}" \
        --datasets "${dataset}" \
        --encoders "${encoder}" \
        --device cuda
      pair_source="${raw_root}"
    fi

    reference_name="imagenet"
    target_name="imagenet_c"
    if [[ "${dataset}" == "cifar100_c" ]]; then
      reference_name="cifar100"
      target_name="cifar100c"
    fi
    reference_seal="${prepared_root}/${reference_name}/${encoder}/SOURCE_ARRAY_SHA256.json"
    target_seal="${prepared_root}/${target_name}/${encoder}/SOURCE_ARRAY_SHA256.json"
    if [[ -f "${reference_seal}" && -f "${target_seal}" ]]; then
      prepared_audit="${audit_root}/${pair}__prepared.json"
      if ! "${python_bin}" "${project_root}/scripts/campaign/prepare_legacy_sources.py" audit \
        --config "${config}" \
        --source-root "${prepared_root}" \
        --datasets "${dataset}" \
        --encoders "${encoder}" \
        --output "${prepared_audit}"; then
        echo "FATAL: sealed prepared pair is stale or invalid: ${pair}; see ${prepared_audit}" >&2
        exit 4
      fi
      echo "[feature-bootstrap] prepared pair already sealed and exact: ${pair}"
    elif [[ -f "${reference_seal}" || -f "${target_seal}" ]]; then
      echo "FATAL: only one endpoint is sealed for prepared pair ${pair}" >&2
      exit 4
    else
      "${python_bin}" "${project_root}/scripts/campaign/prepare_legacy_sources.py" prepare \
        --config "${config}" \
        --source-root "${pair_source}" \
        --destination-root "${prepared_root}" \
        --datasets "${dataset}" \
        --encoders "${encoder}" \
        --seal
    fi
  done
done

echo "[feature-bootstrap] materializing strict V2 stores"
env \
  MARGINAL_PROJECT_ROOT="${project_root}" \
  MARGINAL_PYTHON="${python_bin}" \
  MARGINAL_LEGACY_FEATURE_ROOT="${prepared_root}" \
  MARGINAL_FEATURE_ROOT="${v2_root}" \
  bash "${project_root}/scripts/campaign/build_v2_stores.sh"
