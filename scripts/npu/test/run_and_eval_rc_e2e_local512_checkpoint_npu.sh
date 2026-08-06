#!/usr/bin/env bash
set -euo pipefail

# One command from a local512 checkpoint to original RC E2E all/low/high metrics.
# The workflow suppresses predictions in patches without lane GT, but deliberately
# skips repository patch-level metric calculation.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

if [ "$#" -gt 1 ]; then
  echo "Usage: CHECKPOINT_OBS_PATH=obs://... bash $0" >&2
  echo "   or: bash $0 obs://.../checkpoint-N/" >&2
  exit 2
fi

CHECKPOINT_OBS_PATH=${1:-${CHECKPOINT_OBS_PATH:-}}
if [ -z "${CHECKPOINT_OBS_PATH}" ]; then
  echo "ERROR: provide a checkpoint through CHECKPOINT_OBS_PATH or as the first argument." >&2
  exit 2
fi

CHECKPOINT_URI=${CHECKPOINT_OBS_PATH%/}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-$(basename "${CHECKPOINT_URI}")}
if [ -z "${CHECKPOINT_NAME}" ]; then
  echo "ERROR: unable to derive CHECKPOINT_NAME from ${CHECKPOINT_OBS_PATH}" >&2
  exit 2
fi

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}

RUN_ID=${RUN_ID:-local512_${CHECKPOINT_NAME}_e2e_$(date +%Y%m%d_%H%M%S)}
FRESH_RUN_ROOT=${FRESH_RUN_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/${RUN_ID}}
E2E_ARCHIVE=${E2E_ARCHIVE:-${FRESH_RUN_ROOT}/e2e_data.zip}
E2E_ROOT=${E2E_ROOT:-${FRESH_RUN_ROOT}/e2e_data}
INFERENCE_DATASET_ROOT=${INFERENCE_DATASET_ROOT:-${FRESH_RUN_ROOT}/local512_dataset}

OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}
POSTPROCESS_ROOT=${POSTPROCESS_ROOT:-${OUTPUT_ROOT}/postprocess}
PATCH_REFERENCE_JSONL=${PATCH_REFERENCE_JSONL:-${POSTPROCESS_ROOT}/patch_reference.jsonl}
PATCH_REFERENCE_REPORT=${PATCH_REFERENCE_REPORT:-${POSTPROCESS_ROOT}/patch_reference_report.json}
E2E_PREDICTION_DIR=${E2E_PREDICTION_DIR:-${POSTPROCESS_ROOT}/predictions}
FILTER_REPORT=${FILTER_REPORT:-${POSTPROCESS_ROOT}/filter_report.json}

CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/e2e_local512_${CHECKPOINT_NAME}}
ORIGINAL_PIPELINE_WORK_ROOT=${ORIGINAL_PIPELINE_WORK_ROOT:-${FRESH_RUN_ROOT}/original_pipeline}
ORIGINAL_ENGINE_ROOT=${ORIGINAL_ENGINE_ROOT:-${ORIGINAL_PIPELINE_WORK_ROOT}/engine_lane_grid512}

REBUILD_E2E_DATASET=${REBUILD_E2E_DATASET:-True}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
RULE_WORKERS=${RULE_WORKERS:-16}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-True}
INSTALL_ENGINE_DEPS=${INSTALL_ENGINE_DEPS:-True}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-False}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}

safe_source() {
  local path=$1
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
safe_source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${FRESH_RUN_ROOT}" "${OUTPUT_ROOT}" "${POSTPROCESS_ROOT}"

echo "============================================================"
echo "LOCAL512 CHECKPOINT -> ORIGINAL RC E2E"
echo "Run id:          ${RUN_ID}"
echo "Checkpoint:      ${CHECKPOINT_OBS_PATH}"
echo "Fresh E2E OBS:   ${E2E_DATA_OBS_PATH}"
echo "Fresh E2E root:  ${E2E_ROOT}"
echo "Predictions:     ${RAW_RESULT_DIR}"
echo "E2E input JSON:  ${E2E_PREDICTION_DIR}"
echo "Output:          ${OUTPUT_ROOT}"
echo "Patch metrics:   disabled"
echo "Original eval:   all + low + high"
echo "============================================================"

echo "[local512-e2e] stage 1/5: download and extract E2E data"
if [ ! -s "${E2E_ARCHIVE}" ]; then
  python - "${E2E_DATA_OBS_PATH}" "${E2E_ARCHIVE}" <<'PY'
import sys
import moxing as mox

print(f"[local512-e2e] download {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy(sys.argv[1], sys.argv[2])
PY
else
  echo "[local512-e2e] reuse completed download: ${E2E_ARCHIVE}"
fi

if [ ! -f "${E2E_ROOT}/.extract_complete" ]; then
  python scripts/tools/prepare_rc_e2e_original_run_data.py \
    --archive "${E2E_ARCHIVE}" \
    --destination "${E2E_ROOT}" \
    --allowed-root /cache/jn/e2e_eval/fresh_obs_runs \
    --reset
else
  echo "[local512-e2e] reuse extracted E2E data: ${E2E_ROOT}"
fi

echo "[local512-e2e] stage 2/5: build local512 inputs and run inference"
E2E_DATA_OBS_PATH="${E2E_DATA_OBS_PATH}" \
E2E_WORK_ROOT="${FRESH_RUN_ROOT}" \
E2E_ARCHIVE_PATH="${E2E_ARCHIVE}" \
E2E_RAW_ROOT="${E2E_ROOT}" \
E2E_VIEW_MODE=local512 \
E2E_TARGET_SIZE=512 \
E2E_CONTEXT_SIZE=512 \
E2E_PROMPT_PROFILE=current \
E2E_INPUT_RASTER=inter \
E2E_DATASET_ROOT="${INFERENCE_DATASET_ROOT}" \
REBUILD_E2E_DATASET="${REBUILD_E2E_DATASET}" \
BLACK_RATIO_THRESHOLD=1.0 \
RASTER_ALIGNMENT_REPORT="${FRESH_RUN_ROOT}/raster_alignment_report.json" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" \
CHECKPOINT_CACHE_ROOT="${CHECKPOINT_CACHE_ROOT}" \
CHECKPOINT_DIR="${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
INFER_RESULT_OBS_PATH="" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"

echo "[local512-e2e] stage 3/5: build GT-presence reference without patch metrics"
python scripts/tools/build_rc_e2e_patch_gt_presence.py \
  --raw-e2e-root "${E2E_ROOT}" \
  --prediction-dir "${RAW_RESULT_DIR}" \
  --output-jsonl "${PATCH_REFERENCE_JSONL}" \
  --report-json "${PATCH_REFERENCE_REPORT}" \
  --patch-size 512 \
  --require-all

echo "[local512-e2e] stage 4/5: suppress predictions in patches without lane GT"
python scripts/tools/suppress_e2e_predictions_without_patch_gt.py \
  --eval-jsonl "${PATCH_REFERENCE_JSONL}" \
  --prediction-dir "${RAW_RESULT_DIR}" \
  --output-dir "${E2E_PREDICTION_DIR}" \
  --report-json "${FILTER_REPORT}" \
  --reset \
  --strict

echo "[local512-e2e] stage 5/5: original formatter, rule engine, and all/low/high metrics"
E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${E2E_ROOT}" \
RUN_ID="${RUN_ID}" \
RUN_WORK_ROOT="${ORIGINAL_PIPELINE_WORK_ROOT}" \
RESULT_ROOT="${OUTPUT_ROOT}" \
RESULT_OBS_PATH="${RESULT_OBS_PATH}" \
PREDICTION_CACHE="${E2E_PREDICTION_DIR}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
INSTALL_ENGINE_DEPS="${INSTALL_ENGINE_DEPS}" \
ENGINE_EXTRACT_ROOT="${ORIGINAL_ENGINE_ROOT}" \
PREDICTION_COORD_SCALE=0.512 \
ORIGINAL_E2E_LANE_GRID_SIZE=512 \
RULE_WORKERS="${RULE_WORKERS}" \
RUN_FORMAT_STEP=True \
RUN_RULE_STEP=True \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=True \
RUN_HIGH_EVAL=True \
EVAL_SIMPLIFY_PATH=False \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
FILL_MISSING_SCENE_PREDICTIONS=True \
RESET_EXISTING_MODEL_OUTPUTS=False \
UPLOAD_RESULTS="${UPLOAD_RESULTS}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "============================================================"
echo "LOCAL512 ORIGINAL RC E2E COMPLETE"
echo "Inference JSON:  ${RAW_RESULT_DIR}"
echo "Filter report:   ${FILTER_REPORT}"
echo "All roads:       ${OUTPUT_ROOT}/eval_result_all"
echo "Low roads:       ${OUTPUT_ROOT}/eval_result_low"
echo "High roads:      ${OUTPUT_ROOT}/eval_result_high"
echo "Run data:        ${E2E_ROOT}"
echo "Patch metrics:   not calculated"
echo "============================================================"
