#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${MARGINAL_PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
feature_root="${MARGINAL_FEATURE_ROOT:-${project_root}/data/feature_stores_v2}"
python_bin="${MARGINAL_PYTHON:-$(command -v python3)}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

lock_path="${MARGINAL_LAUNCH_LOCK:-${project_root}/results/campaign_launcher.lock}"
mkdir -p "$(dirname "${lock_path}")"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${lock_path}"
  if ! flock -n 9; then
    echo "FATAL: another campaign launcher owns ${lock_path}; refusing concurrent run" >&2
    exit 75
  fi
  printf 'pid=%s started_utc=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&9
else
  echo "FATAL: flock is required to guard cache conversion and campaign state" >&2
  exit 2
fi

echo "[preflight] checking the complete reuse/extraction environment and CUDA"
preflight_arguments=(
  --config "${project_root}/configs/campaign/feature_build.json"
  --require-cuda
)
if [[ "${MARGINAL_FORCE_RAW_EXTRACTION:-0}" == "1" ]]; then
  preflight_arguments+=(--require-raw-inputs)
fi
"${python_bin}" "${project_root}/scripts/campaign/feature_environment_preflight.py" \
  "${preflight_arguments[@]}"

echo "[cache] validating locked 306-store cache before campaign launch"
if [[ "${MARGINAL_FORCE_RAW_EXTRACTION:-0}" == "1" ]]; then
  echo "[cache] FORCE-RAW mode bypasses legacy/V2 reuse and resumes only the declared raw build"
  bash "${project_root}/scripts/campaign/bootstrap_v2_stores.sh"
elif "${python_bin}" "${project_root}/scripts/campaign/check_v2_stores.py" --root "${feature_root}"; then
  echo "[cache] complete V2 cache verified; conversion skipped"
else
  echo "[cache] cache is absent, partial, stale, or invalid; bootstrapping strict sources"
  bash "${project_root}/scripts/campaign/bootstrap_v2_stores.sh"
fi

echo "[cache] final pre-campaign cache validation"
"${python_bin}" "${project_root}/scripts/campaign/check_v2_stores.py" --root "${feature_root}"
exec bash "${project_root}/scripts/run_campaign.sh" "$@"
