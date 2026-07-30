#!/usr/bin/env bash
set -euo pipefail

# Resume the July 30 context512/ROI256 run at original metric evaluation only.
# infer_result_format and center_lane_rule outputs are reused in-place.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

BASE_RUN_ID=${BASE_RUN_ID:-context512_roi256_checkpoint12504_fresh_20260730_122847}
E2E_DATA_ROOT=${E2E_DATA_ROOT:-/cache/jn/e2e_eval/raw_e2e_data}
PREDICTION_CACHE=${PREDICTION_CACHE:-/cache/jn/outputs/${BASE_RUN_ID}/inference/json}
RESULT_ROOT=${RESULT_ROOT:-/cache/jn/outputs/${BASE_RUN_ID}/original_pipeline_metrics}
RESUME_RUN_ID=${RESUME_RUN_ID:-${BASE_RUN_ID}_metrics_resume_$(date -u +%Y%m%d_%H%M%S)}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RESUME_RUN_ID}}

E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${E2E_DATA_ROOT}" \
E2E_DATA_ROOT="${E2E_DATA_ROOT}" \
RUN_ID="${RESUME_RUN_ID}" \
RUN_WORK_ROOT="/cache/jn/e2e_eval/original_pipeline_runs/${RESUME_RUN_ID}" \
RESULT_ROOT="${RESULT_ROOT}" \
RESULT_OBS_PATH="${RESULT_OBS_PATH}" \
PREDICTION_CACHE="${PREDICTION_CACHE}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
INSTALL_ENGINE_DEPS=False \
RUN_FORMAT_STEP=False \
RUN_RULE_STEP=False \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=True \
RUN_HIGH_EVAL=True \
EVAL_SIMPLIFY_PATH=False \
EVAL_VIS_FLAG=False \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"
