#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

SOURCE_RUN_ID=${SOURCE_RUN_ID:-context512_checkpoint12504_fresh_original_20260730_125433}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-/cache/jn/outputs/${SOURCE_RUN_ID}}
PREDICTION_DIR=${PREDICTION_DIR:-${SOURCE_OUTPUT_ROOT}/inference/json}
E2E_RAW_ROOT=${E2E_RAW_ROOT:-/cache/jn/e2e_eval/raw_e2e_data}
EVAL_ROOT=${EVAL_ROOT:-${SOURCE_OUTPUT_ROOT}/roi_clipped_patch_metrics}
CLIPPED_PREDICTION_DIR=${CLIPPED_PREDICTION_DIR:-${EVAL_ROOT}/predictions}
SANITIZE_REPORT=${SANITIZE_REPORT:-${EVAL_ROOT}/prediction_roi_clip_report.json}

cd "${REPO_ROOT}"
mkdir -p "${EVAL_ROOT}"

python scripts/tools/sanitize_rc_e2e_predictions_for_original_formatter.py \
  --input-dir "${PREDICTION_DIR}" \
  --output-dir "${CLIPPED_PREDICTION_DIR}" \
  --report-json "${SANITIZE_REPORT}" \
  --roi-min 0 \
  --roi-max 1000 \
  --reset

E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${E2E_RAW_ROOT}" \
PREDICTION_DIR="${CLIPPED_PREDICTION_DIR}" \
EVAL_RUN_ID="${SOURCE_RUN_ID}_roi_clipped_patch_metrics" \
EVAL_ROOT="${EVAL_ROOT}" \
METRICS_JSON="${EVAL_ROOT}/metrics.json" \
EVAL_JSONL="${EVAL_ROOT}/eval_records.jsonl" \
METRICS_OBS_PATH="" \
REQUIRE_ALL=True \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"

echo "============================================================"
echo "ROI-CLIPPED PATCH METRICS COMPLETE"
echo "Clip report: ${SANITIZE_REPORT}"
echo "Metrics:     ${EVAL_ROOT}/metrics.json"
echo "Eval JSONL:  ${EVAL_ROOT}/eval_records.jsonl"
echo "============================================================"
