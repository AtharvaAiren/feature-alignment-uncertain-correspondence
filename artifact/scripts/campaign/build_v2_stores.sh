#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${MARGINAL_PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
python_bin="${MARGINAL_PYTHON:-$(command -v python3)}"
legacy_root="${MARGINAL_LEGACY_FEATURE_ROOT:-${project_root}/data/prepared_features}"
destination_root="${MARGINAL_FEATURE_ROOT:-${project_root}/data/feature_stores_v2}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

cifar_corruptions="brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_blur,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,saturate,shot_noise,snow,spatter,speckle_noise,zoom_blur"
imagenet_corruptions="brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur"

echo "[cache] V2 conversion preflight: expecting 306 stores under ${destination_root}"
for encoder in dinov3 clip vitb16; do
  for store in \
    "${legacy_root}/cifar100/${encoder}" \
    "${legacy_root}/cifar100c/${encoder}" \
    "${legacy_root}/imagenet/${encoder}" \
    "${legacy_root}/imagenet_c/${encoder}"; do
    if [[ ! -d "${store}" ]]; then
      echo "FATAL: required legacy cache directory missing: ${store}" >&2
      exit 2
    fi
    if [[ -f "${store}/SOURCE_ARRAY_SHA256.json" ]]; then
      echo "[cache] source seal present (converter will verify): ${store}"
    else
      echo "[cache] creating source seal: ${store}"
      "${python_bin}" "${project_root}/scripts/campaign/seal_legacy_arrays.py" "${store}"
    fi
  done
done

echo "[cache] checking data-dependent peak free-space requirement"
"${python_bin}" "${project_root}/scripts/campaign/preflight_v2_disk.py" \
  --legacy-root "${legacy_root}" \
  --destination-root "${destination_root}"

conversion_index=0
for encoder in dinov3 clip vitb16; do
  conversion_index=$((conversion_index + 1))
  echo "[cache] conversion ${conversion_index}/6: cifar100_c ${encoder} (57 conditions)"
  "${python_bin}" "${project_root}/scripts/campaign/convert_legacy_cache.py" \
    --dataset cifar100_c \
    --encoder "${encoder}" \
    --reference-root "${legacy_root}/cifar100/${encoder}" \
    --target-root "${legacy_root}/cifar100c/${encoder}" \
    --destination-root "${destination_root}" \
    --corruptions "${cifar_corruptions}" \
    --severities 1,3,5

  conversion_index=$((conversion_index + 1))
  echo "[cache] conversion ${conversion_index}/6: imagenet_c ${encoder} (45 conditions)"
  "${python_bin}" "${project_root}/scripts/campaign/convert_legacy_cache.py" \
    --dataset imagenet_c \
    --encoder "${encoder}" \
    --reference-root "${legacy_root}/imagenet/${encoder}" \
    --target-root "${legacy_root}/imagenet_c/${encoder}" \
    --destination-root "${destination_root}" \
    --corruptions "${imagenet_corruptions}" \
    --severities 1,3,5
done

echo "[cache] final content validation of all 306 expected stores"
"${python_bin}" "${project_root}/scripts/campaign/check_v2_stores.py" --root "${destination_root}"
echo "V2 feature-store conversion PASS: ${destination_root}"
