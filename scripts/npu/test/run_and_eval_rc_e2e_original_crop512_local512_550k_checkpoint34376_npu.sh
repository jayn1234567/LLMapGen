#!/usr/bin/env bash
set -euo pipefail

# One command: original RC E2E plain-512 crop, local512-550k checkpoint-34376
# inference, repository patch metrics, and original all/low/high evaluation.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

RUN_ID=${RUN_ID:-rc_e2e_original_crop512_local512_550k_checkpoint34376_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}
INFER_RESULT_OBS_PATH=${INFER_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/${RUN_ID}}
PATCH_METRICS_OBS_PATH=${PATCH_METRICS_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RUN_ID}_patch_metrics}
ORIGINAL_EVAL_RUN_ID=${ORIGINAL_EVAL_RUN_ID:-${RUN_ID}_original_pipeline}
ORIGINAL_RESULT_ROOT=${ORIGINAL_RESULT_ROOT:-${OUTPUT_ROOT}/original_pipeline_metrics}
ORIGINAL_RESULT_OBS_PATH=${ORIGINAL_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RUN_ID}_original_pipeline}

echo "[original-crop512-e2e] stage 1/3: original splitter + local512 checkpoint-34376 inference"
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
INFER_RESULT_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
bash "${SCRIPT_DIR}/run_rc_e2e_original_crop512_local512_550k_checkpoint34376_npu.sh"

echo "[original-crop512-e2e] stage 2/3: repository patch metrics"
PATCH_SIZE=512 \
EVAL_RUN_ID="${RUN_ID}_patch_metrics" \
EVAL_ROOT="${OUTPUT_ROOT}/patch_metrics" \
PREDICTION_DIR="${RAW_RESULT_DIR}" \
PREDICTION_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
METRICS_OBS_PATH="${PATCH_METRICS_OBS_PATH}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"

echo "[original-crop512-e2e] stage 3/3: original RC E2E all/low/high metrics"
PREDICTION_COORD_SCALE=0.512 \
RUN_ID="${ORIGINAL_EVAL_RUN_ID}" \
RUN_WORK_ROOT="/cache/jn/e2e_eval/original_pipeline_runs/${ORIGINAL_EVAL_RUN_ID}" \
RESULT_ROOT="${ORIGINAL_RESULT_ROOT}" \
RESULT_OBS_PATH="${ORIGINAL_RESULT_OBS_PATH}" \
PREDICTION_CACHE="${RAW_RESULT_DIR}" \
PREDICTION_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "============================================================"
echo "LOCAL512 CHECKPOINT-34376 FULL E2E EVALUATION COMPLETE"
echo "Inference JSON: ${RAW_RESULT_DIR}"
echo "Patch metrics:  ${OUTPUT_ROOT}/patch_metrics/metrics.json"
echo "Original all:   ${ORIGINAL_RESULT_ROOT}/eval_result_all"
echo "Original low:   ${ORIGINAL_RESULT_ROOT}/eval_result_low"
echo "Original high:  ${ORIGINAL_RESULT_ROOT}/eval_result_high"
echo "Inference OBS:  ${INFER_RESULT_OBS_PATH}"
echo "Patch OBS:      ${PATCH_METRICS_OBS_PATH}"
echo "Original OBS:   ${ORIGINAL_RESULT_OBS_PATH}"
echo "============================================================"
