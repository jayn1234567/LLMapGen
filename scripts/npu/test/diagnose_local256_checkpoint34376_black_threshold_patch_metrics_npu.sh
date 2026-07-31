#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

RUN_ROOT=${RUN_ROOT:-/cache/jn/outputs/rc_e2e_original_crop256_local256_550k_checkpoint34376_20260728_095546}
PREDICTION_DIR=${PREDICTION_DIR:-${RUN_ROOT}/inference/json}
DATASET_ROOT=${DATASET_ROOT:-/cache/jn/e2e_eval/e2e_data_original_crop256_black1_local256_550k_v2}
INFER_JSONL=${INFER_JSONL:-${DATASET_ROOT}/infer.jsonl}
BLACK_RATIO_MANIFEST=${BLACK_RATIO_MANIFEST:-${DATASET_ROOT}/patch_black_ratio_manifest.json}
RAW_E2E_ROOT=${RAW_E2E_ROOT:-/cache/e2e_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/blackratio_patch_ab}
DIAG_RUN_ID=${DIAG_RUN_ID:-local256_checkpoint34376_blackratio_patch_ab}
BLACK_RATIO_WORKERS=${BLACK_RATIO_WORKERS:-16}
REBUILD_BLACK_RATIO_MANIFEST=${REBUILD_BLACK_RATIO_MANIFEST:-False}
VISUALIZE_PATCH_AB=${VISUALIZE_PATCH_AB:-True}
VIS_LIMIT=${VIS_LIMIT:-200}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cd "${REPO_ROOT}"

for required_path in "${INFER_JSONL}" "${PREDICTION_DIR}" "${RAW_E2E_ROOT}"; do
  if [ ! -e "${required_path}" ]; then
    echo "ERROR: required local256 path not found: ${required_path}" >&2
    exit 2
  fi
done

if is_true "${REBUILD_BLACK_RATIO_MANIFEST}" || [ ! -s "${BLACK_RATIO_MANIFEST}" ]; then
  echo "[local256-threshold-ab] computing black ratios from existing crop images"
  python scripts/tools/build_e2e_black_ratio_manifest.py \
    --infer-jsonl "${INFER_JSONL}" \
    --image-root "${DATASET_ROOT}" \
    --output-json "${BLACK_RATIO_MANIFEST}" \
    --workers "${BLACK_RATIO_WORKERS}"
else
  echo "[local256-threshold-ab] reuse black-ratio manifest: ${BLACK_RATIO_MANIFEST}"
fi

BASE_RUN_ID=$(basename "${RUN_ROOT}") \
DATASET_ROOT="${DATASET_ROOT}" \
MANIFEST_JSON="${BLACK_RATIO_MANIFEST}" \
INFER_JSONL="${INFER_JSONL}" \
PREDICTION_DIR="${PREDICTION_DIR}" \
RAW_E2E_ROOT="${RAW_E2E_ROOT}" \
DIAG_RUN_ID="${DIAG_RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
IMAGE_FOLDER="${DATASET_ROOT}" \
VISUALIZE_PATCH_AB="${VISUALIZE_PATCH_AB}" \
VIS_LIMIT="${VIS_LIMIT}" \
RUN_WHOLEMAP_AB=False \
bash "${SCRIPT_DIR}/diagnose_context512_black_threshold_checkpoint12504_npu.sh"
