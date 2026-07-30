#!/usr/bin/env bash
set -euo pipefail

# One command: RawLane-local256 inference, repository patch metrics, and the
# original RC E2E all/low/high post-processing and evaluation pipeline.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

RUN_ID=${RUN_ID:-rc_e2e_rawlane_local256_checkpoint12504_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}
INFER_RESULT_OBS_PATH=${INFER_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/${RUN_ID}}
PATCH_METRICS_OBS_PATH=${PATCH_METRICS_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RUN_ID}_patch_metrics}
ORIGINAL_EVAL_RUN_ID=${ORIGINAL_EVAL_RUN_ID:-${RUN_ID}_original_pipeline}
ORIGINAL_RUN_WORK_ROOT=${ORIGINAL_RUN_WORK_ROOT:-/cache/jn/e2e_eval/original_pipeline_runs/${ORIGINAL_EVAL_RUN_ID}}
ORIGINAL_E2E_DATA_ROOT=${ORIGINAL_E2E_DATA_ROOT:-${ORIGINAL_RUN_WORK_ROOT}/e2e_data}
ORIGINAL_RESULT_ROOT=${ORIGINAL_RESULT_ROOT:-${OUTPUT_ROOT}/original_pipeline_metrics}
ORIGINAL_RESULT_OBS_PATH=${ORIGINAL_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RUN_ID}_original_pipeline}
E2E_DATA_SOURCE=${E2E_DATA_SOURCE:-local_archive}
E2E_DATA_ARCHIVE=${E2E_DATA_ARCHIVE:-/cache/jn/e2e_eval/e2e_data.zip}

echo "[rawlane-local256-e2e] stage 1/3: full checkpoint-12504 inference"
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
INFER_RESULT_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
bash "${SCRIPT_DIR}/run_rc_e2e_rawlane_local256_checkpoint12504_npu.sh"

echo "[rawlane-local256-e2e] stage 2/3: repository patch metrics"
EVAL_RUN_ID="${RUN_ID}_patch_metrics" \
EVAL_ROOT="${OUTPUT_ROOT}/patch_metrics" \
PREDICTION_DIR="${RAW_RESULT_DIR}" \
PREDICTION_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
METRICS_OBS_PATH="${PATCH_METRICS_OBS_PATH}" \
REQUIRE_ALL=True \
E2E_DATA_SOURCE="${E2E_DATA_SOURCE}" \
E2E_ARCHIVE_PATH="${E2E_DATA_ARCHIVE}" \
E2E_RAW_ROOT="${ORIGINAL_E2E_DATA_ROOT}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"

echo "[rawlane-local256-e2e] stage 3/3: original RC E2E all/low/high metrics"
RUN_ID="${ORIGINAL_EVAL_RUN_ID}" \
RUN_WORK_ROOT="${ORIGINAL_RUN_WORK_ROOT}" \
E2E_DATA_ROOT="${ORIGINAL_E2E_DATA_ROOT}" \
RESULT_ROOT="${ORIGINAL_RESULT_ROOT}" \
RESULT_OBS_PATH="${ORIGINAL_RESULT_OBS_PATH}" \
PREDICTION_CACHE="${RAW_RESULT_DIR}" \
PREDICTION_OBS_PATH="${INFER_RESULT_OBS_PATH}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
E2E_DATA_SOURCE="${E2E_DATA_SOURCE}" \
E2E_DATA_ARCHIVE="${E2E_DATA_ARCHIVE}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "============================================================"
echo "RAWLANE LOCAL256 CHECKPOINT-12504 FULL E2E EVALUATION COMPLETE"
echo "Inference JSON: ${RAW_RESULT_DIR}"
echo "Patch metrics:  ${OUTPUT_ROOT}/patch_metrics/metrics.json"
echo "Original all:   ${ORIGINAL_RESULT_ROOT}/eval_result_all"
echo "Original low:   ${ORIGINAL_RESULT_ROOT}/eval_result_low"
echo "Original high:  ${ORIGINAL_RESULT_ROOT}/eval_result_high"
echo "Inference OBS:  ${INFER_RESULT_OBS_PATH}"
echo "Patch OBS:      ${PATCH_METRICS_OBS_PATH}"
echo "Original OBS:   ${ORIGINAL_RESULT_OBS_PATH}"
echo "============================================================"
