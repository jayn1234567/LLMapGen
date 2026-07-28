#!/usr/bin/env bash
set -euo pipefail

# One command: run full local256 E2E inference, then compute the repository's
# patch-level lane metrics from the same local prediction JSON files.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

RUN_ID=${RUN_ID:-rc_e2e_local256_550k_checkpoint34376_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}
INFER_RESULT_OBS_PATH=${INFER_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/${RUN_ID}}
PATCH_METRICS_OBS_PATH=${PATCH_METRICS_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RUN_ID}_patch_metrics}

echo "[local256-e2e] stage 1/2: full checkpoint-34376 inference"
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
INFER_RESULT_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
bash "${SCRIPT_DIR}/run_rc_e2e_local256_550k_checkpoint34376_npu.sh"

echo "[local256-e2e] stage 2/2: repository patch metrics"
EVAL_RUN_ID="${RUN_ID}_patch_metrics" \
EVAL_ROOT="${OUTPUT_ROOT}/patch_metrics" \
PREDICTION_DIR="${RAW_RESULT_DIR}" \
PREDICTION_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
METRICS_OBS_PATH="${PATCH_METRICS_OBS_PATH}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"

echo "============================================================"
echo "LOCAL256 CHECKPOINT-34376 E2E INFERENCE + PATCH METRICS COMPLETE"
echo "Inference JSON: ${RAW_RESULT_DIR}"
echo "Patch metrics:  ${OUTPUT_ROOT}/patch_metrics/metrics.json"
echo "Inference OBS:  ${INFER_RESULT_OBS_PATH}"
echo "Metrics OBS:    ${PATCH_METRICS_OBS_PATH}"
echo "============================================================"
