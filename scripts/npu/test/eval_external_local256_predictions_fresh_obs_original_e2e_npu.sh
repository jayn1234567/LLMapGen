#!/usr/bin/env bash
set -euo pipefail

# Evaluate existing local256 per-patch JSON with the original formatter,
# center-lane post-processing, and all/low/high E2E metrics. This entry does
# not run model inference and does not modify predictions using ground truth.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

SOURCE_PREDICTION_DIR=${SOURCE_PREDICTION_DIR:-/cache/xyk/results/xyk_test_phase_a_lane_intersection_output_256_e2e_20260731_063027/json}
E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
RUN_ID=${RUN_ID:-xyk_external_local256_original_e2e_$(date +%Y%m%d_%H%M%S)}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/e2e_eval/original_pipeline_runs/${RUN_ID}}
E2E_DATA_ARCHIVE=${E2E_DATA_ARCHIVE:-${RUN_WORK_ROOT}/e2e_data.zip}
E2E_DATA_ROOT=${E2E_DATA_ROOT:-${RUN_WORK_ROOT}/e2e_data}
RESULT_ROOT=${RESULT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
RULE_WORKERS=${RULE_WORKERS:-16}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
INSTALL_ENGINE_DEPS=${INSTALL_ENGINE_DEPS:-True}
REUSE_ENGINE_ARCHIVE=${REUSE_ENGINE_ARCHIVE:-True}
RESET_PREPARED_E2E_DATA=${RESET_PREPARED_E2E_DATA:-False}
FAIL_ON_INVALID_PREDICTIONS=${FAIL_ON_INVALID_PREDICTIONS:-False}

cd "${REPO_ROOT}"

if [ ! -d "${SOURCE_PREDICTION_DIR}" ]; then
  echo "ERROR: external prediction directory not found: ${SOURCE_PREDICTION_DIR}" >&2
  exit 2
fi

PREDICTION_COUNT=$(find "${SOURCE_PREDICTION_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [ "${PREDICTION_COUNT}" -le 0 ]; then
  echo "ERROR: no per-patch JSON files found below ${SOURCE_PREDICTION_DIR}" >&2
  exit 2
fi

echo "============================================================"
echo "EXTERNAL LOCAL256 ORIGINAL E2E EVALUATION"
echo "Predictions:      ${SOURCE_PREDICTION_DIR}"
echo "Prediction count: ${PREDICTION_COUNT}"
echo "Fresh E2E OBS:    ${E2E_DATA_OBS_PATH}"
echo "Run E2E data:     ${E2E_DATA_ROOT}"
echo "Results:          ${RESULT_ROOT}"
echo "Protocol:         unmodified predictions; original formatter/rule/all-low-high"
echo "Invalid policy:   report and count as no prediction (fail=${FAIL_ON_INVALID_PREDICTIONS})"
echo "============================================================"

PREDICTION_CACHE="${SOURCE_PREDICTION_DIR}" \
REUSE_PREDICTIONS=True \
E2E_DATA_SOURCE=auto \
E2E_DATA_OBS_PATH="${E2E_DATA_OBS_PATH}" \
E2E_DATA_ARCHIVE="${E2E_DATA_ARCHIVE}" \
E2E_RAW_ROOT="${RUN_WORK_ROOT}/unused_raw_e2e_data" \
E2E_DATA_ROOT="${E2E_DATA_ROOT}" \
RESET_PREPARED_E2E_DATA="${RESET_PREPARED_E2E_DATA}" \
RUN_ID="${RUN_ID}" \
RUN_WORK_ROOT="${RUN_WORK_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}" \
RESULT_OBS_PATH="${RESULT_OBS_PATH}" \
PREDICTION_COORD_SCALE=0.256 \
RUN_FORMAT_STEP=True \
RUN_RULE_STEP=True \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=True \
RUN_HIGH_EVAL=True \
EVAL_SIMPLIFY_PATH=False \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
FILL_MISSING_SCENE_PREDICTIONS=True \
FAIL_ON_INVALID_PREDICTIONS="${FAIL_ON_INVALID_PREDICTIONS}" \
INSTALL_ENGINE_DEPS="${INSTALL_ENGINE_DEPS}" \
REUSE_ENGINE_ARCHIVE="${REUSE_ENGINE_ARCHIVE}" \
UPLOAD_RESULTS=False \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "============================================================"
echo "EXTERNAL LOCAL256 E2E COMPLETE"
echo "All roads:  ${RESULT_ROOT}/eval_result_all"
echo "Low roads:  ${RESULT_ROOT}/eval_result_low"
echo "High roads: ${RESULT_ROOT}/eval_result_high"
echo "Logs:       ${RESULT_ROOT}/logs"
echo "NOTE: predictions were not suppressed, clipped, or otherwise changed by GT."
echo "============================================================"
