#!/usr/bin/env bash
set -euo pipefail

# Clean-BEV local256 E2E for original DINOv2-Large + CapRL-derived checkpoint.
# The model receives only *_inter.tif crops; no RawLane raster is read or drawn.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

if [ "$#" -gt 1 ]; then
  echo "Usage: bash $0 [obs://.../checkpoint-N/]" >&2
  exit 2
fi

DEFAULT_CHECKPOINT_OBS_PATH=obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/
CHECKPOINT_OBS_PATH=${1:-${CHECKPOINT_OBS_PATH:-${DEFAULT_CHECKPOINT_OBS_PATH}}}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-$(basename "${CHECKPOINT_OBS_PATH%/}")}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/clean_local256_550k_${CHECKPOINT_NAME}}
CHECKPOINT_EXPECTED_KIND=${CHECKPOINT_EXPECTED_KIND:-auto}
ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}

E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
RUN_ID=${RUN_ID:-clean_local256_550k_${CHECKPOINT_NAME}_e2e_$(date +%Y%m%d_%H%M%S)}
FRESH_RUN_ROOT=${FRESH_RUN_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/${RUN_ID}}
E2E_ARCHIVE=${E2E_ARCHIVE:-${FRESH_RUN_ROOT}/e2e_data.zip}
E2E_ROOT=${E2E_ROOT:-${FRESH_RUN_ROOT}/e2e_data}
INFERENCE_DATASET_ROOT=${INFERENCE_DATASET_ROOT:-${FRESH_RUN_ROOT}/clean_local256_dataset}

OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}
POSTPROCESS_ROOT=${POSTPROCESS_ROOT:-${OUTPUT_ROOT}/postprocess}
PATCH_REFERENCE_JSONL=${PATCH_REFERENCE_JSONL:-${POSTPROCESS_ROOT}/patch_reference.jsonl}
PATCH_REFERENCE_REPORT=${PATCH_REFERENCE_REPORT:-${POSTPROCESS_ROOT}/patch_reference_report.json}
E2E_PREDICTION_DIR=${E2E_PREDICTION_DIR:-${POSTPROCESS_ROOT}/gt_empty_suppressed_predictions}
FILTER_REPORT=${FILTER_REPORT:-${POSTPROCESS_ROOT}/filter_report.json}

LANE_RESULT_ROOT=${LANE_RESULT_ROOT:-${OUTPUT_ROOT}/lane_original_pipeline_metrics}
LANE_RUN_WORK_ROOT=${LANE_RUN_WORK_ROOT:-${FRESH_RUN_ROOT}/lane_original_pipeline}
RUN_INTERSECTION_E2E=${RUN_INTERSECTION_E2E:-True}
INTERSECTION_RESULT_ROOT=${INTERSECTION_RESULT_ROOT:-${OUTPUT_ROOT}/intersection_original_pipeline_metrics}
INTERSECTION_RUN_WORK_ROOT=${INTERSECTION_RUN_WORK_ROOT:-${FRESH_RUN_ROOT}/intersection_original_pipeline}
INTERSECTION_ENGINE_ROOT=${INTERSECTION_ENGINE_ROOT:-${INTERSECTION_RUN_WORK_ROOT}/original_engine_grid256}
INTERSECTION_COLLAPSE_TYPE_TO_ONE=${INTERSECTION_COLLAPSE_TYPE_TO_ONE:-False}
INTERSECTION_EVAL_ONLY_TYPE1=${INTERSECTION_EVAL_ONLY_TYPE1:-False}
INTERSECTION_GT_EMPTY_SUPPRESSION=${INTERSECTION_GT_EMPTY_SUPPRESSION:-False}

REBUILD_E2E_DATASET=${REBUILD_E2E_DATASET:-True}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
RULE_WORKERS=${RULE_WORKERS:-16}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
INSTALL_ENGINE_DEPS=${INSTALL_ENGINE_DEPS:-True}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-False}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
set +u
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
set -u

cd "${REPO_ROOT}"
mkdir -p "${FRESH_RUN_ROOT}" "${OUTPUT_ROOT}" "${POSTPROCESS_ROOT}"

echo "============================================================"
echo "CLEAN LOCAL256-550K ORIGINAL DINOV2 E2E"
echo "Run id:             ${RUN_ID}"
echo "Checkpoint OBS:     ${CHECKPOINT_OBS_PATH}"
echo "E2E data OBS:       ${E2E_DATA_OBS_PATH}"
echo "Inference raster:   inter_patch_tif/*_inter.tif"
echo "RawLane input:      disabled"
echo "Prompt profile:     local256_550k_v1"
echo "Per-device batch:   ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Lane evaluation:    original all + low + high"
echo "Intersection eval:  ${RUN_INTERSECTION_E2E}"
echo "Output:             ${OUTPUT_ROOT}"
echo "============================================================"

echo "[clean-local256-e2e] stage 1/6: download and extract E2E data"
if [ ! -s "${E2E_ARCHIVE}" ]; then
  python - "${E2E_DATA_OBS_PATH}" "${E2E_ARCHIVE}" <<'PY'
import sys
import moxing as mox

print(f"[clean-local256-e2e] download {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy(sys.argv[1], sys.argv[2])
PY
else
  echo "[clean-local256-e2e] reuse archive: ${E2E_ARCHIVE}"
fi

if [ ! -f "${E2E_ROOT}/.e2e_source_prepare_complete.json" ]; then
  python scripts/tools/prepare_rc_e2e_original_run_data.py \
    --archive "${E2E_ARCHIVE}" \
    --destination "${E2E_ROOT}" \
    --allowed-root /cache/jn/e2e_eval/fresh_obs_runs \
    --reset
else
  echo "[clean-local256-e2e] reuse extracted E2E data: ${E2E_ROOT}"
fi

echo "[clean-local256-e2e] stage 2/6: clean local256 DINOv2 inference"
E2E_DATA_OBS_PATH="${E2E_DATA_OBS_PATH}" \
E2E_WORK_ROOT="${FRESH_RUN_ROOT}" \
E2E_ARCHIVE_PATH="${E2E_ARCHIVE}" \
E2E_RAW_ROOT="${E2E_ROOT}" \
E2E_VIEW_MODE=local256 \
E2E_TARGET_SIZE=256 \
E2E_CONTEXT_SIZE=256 \
E2E_PROMPT_PROFILE=local256_550k_v1 \
E2E_INPUT_RASTER=inter \
E2E_DATASET_ROOT="${INFERENCE_DATASET_ROOT}" \
REBUILD_E2E_DATASET="${REBUILD_E2E_DATASET}" \
BLACK_RATIO_THRESHOLD=1.0 \
VALIDATE_RASTER_ALIGNMENT=False \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" \
CHECKPOINT_CACHE_ROOT="${CHECKPOINT_CACHE_ROOT}" \
CHECKPOINT_DIR="${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}" \
CHECKPOINT_EXPECTED_KIND="${CHECKPOINT_EXPECTED_KIND}" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
INFER_RESULT_OBS_PATH="" \
bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"

python - "${INFERENCE_DATASET_ROOT}/dataset_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "view_mode": "local256",
    "target_size": 256,
    "context_size": 256,
    "prompt_profile": "local256_550k_v1",
    "input_raster": "inter",
    "raw_lane_overlay": False,
}
errors = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
if errors:
    raise RuntimeError(f"Clean no-RawLane dataset contract failed: {errors}; summary={path}")
print(f"[clean-local256-e2e] no-RawLane contract passed: patches={summary['patch_count']}")
PY

echo "[clean-local256-e2e] stage 3/6: build lane-GT presence reference"
python scripts/tools/build_rc_e2e_patch_gt_presence.py \
  --raw-e2e-root "${E2E_ROOT}" \
  --prediction-dir "${RAW_RESULT_DIR}" \
  --output-jsonl "${PATCH_REFERENCE_JSONL}" \
  --report-json "${PATCH_REFERENCE_REPORT}" \
  --patch-size 256 \
  --require-all

echo "[clean-local256-e2e] stage 4/6: suppress predictions in lane-GT-empty patches"
python scripts/tools/suppress_e2e_predictions_without_patch_gt.py \
  --eval-jsonl "${PATCH_REFERENCE_JSONL}" \
  --prediction-dir "${RAW_RESULT_DIR}" \
  --output-dir "${E2E_PREDICTION_DIR}" \
  --report-json "${FILTER_REPORT}" \
  --reset \
  --strict

echo "[clean-local256-e2e] stage 5/6: original lane all/low/high evaluation"
E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${E2E_ROOT}" \
E2E_USE_RAW_ROOT_DIRECTLY=True \
RUN_ID="${RUN_ID}_lane_original" \
RUN_WORK_ROOT="${LANE_RUN_WORK_ROOT}" \
RESULT_ROOT="${LANE_RESULT_ROOT}" \
RESULT_OBS_PATH="${RESULT_OBS_PATH}" \
PREDICTION_CACHE="${E2E_PREDICTION_DIR}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
INSTALL_ENGINE_DEPS="${INSTALL_ENGINE_DEPS}" \
PREDICTION_COORD_SCALE=0.256 \
ORIGINAL_E2E_LANE_GRID_SIZE=256 \
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
RESET_EXISTING_MODEL_OUTPUTS=True \
UPLOAD_RESULTS="${UPLOAD_RESULTS}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

if is_true "${RUN_INTERSECTION_E2E}"; then
  echo "[clean-local256-e2e] stage 6/6: original whole-map intersection evaluation"
  PREDICTION_DIR="${RAW_RESULT_DIR}" \
  E2E_DATA_ROOT="${E2E_ROOT}" \
  RUN_ID="${RUN_ID}_intersection_original" \
  RESULT_ROOT="${INTERSECTION_RESULT_ROOT}" \
  RUN_WORK_ROOT="${INTERSECTION_RUN_WORK_ROOT}" \
  ENGINE_EXTRACT_ROOT="${INTERSECTION_ENGINE_ROOT}" \
  WINDOW_SIZE=256 \
  INTERSECTION_STRIDE=256 \
  ORIGINAL_E2E_LANE_GRID_SIZE=256 \
  PREDICTION_COORD_SCALE=0.256 \
  COLLAPSE_INTERSECTION_TYPE_TO_ONE="${INTERSECTION_COLLAPSE_TYPE_TO_ONE}" \
  EVAL_INTERSECTION_ONLY_TYPE1="${INTERSECTION_EVAL_ONLY_TYPE1}" \
  SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION="${INTERSECTION_GT_EMPTY_SUPPRESSION}" \
  E2E_USE_RAW_ROOT_DIRECTLY=True \
  EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
  EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
  INSTALL_ENGINE_DEPS=False \
  UPLOAD_RESULTS="${UPLOAD_RESULTS}" \
  bash "${SCRIPT_DIR}/eval_local512_predictions_original_intersection_e2e_npu.sh"
else
  echo "[clean-local256-e2e] skip intersection E2E: RUN_INTERSECTION_E2E=${RUN_INTERSECTION_E2E}"
fi

echo "============================================================"
echo "CLEAN LOCAL256 ORIGINAL DINOV2 E2E COMPLETE"
echo "Inference JSON:       ${RAW_RESULT_DIR}"
echo "No-RawLane summary:   ${INFERENCE_DATASET_ROOT}/dataset_summary.json"
echo "GT-empty filter:      ${FILTER_REPORT}"
echo "Lane all roads:       ${LANE_RESULT_ROOT}/eval_result_all"
echo "Lane low roads:       ${LANE_RESULT_ROOT}/eval_result_low"
echo "Lane high roads:      ${LANE_RESULT_ROOT}/eval_result_high"
if is_true "${RUN_INTERSECTION_E2E}"; then
  echo "Intersection metrics: ${INTERSECTION_RESULT_ROOT}/eval_result_all"
fi
echo "============================================================"
