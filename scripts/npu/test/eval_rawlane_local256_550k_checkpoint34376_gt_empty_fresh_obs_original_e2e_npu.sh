#!/usr/bin/env bash
set -euo pipefail

# Fresh-OBS, full end-to-end evaluation for the Raw-Lane local256 550k
# LLM-LoRA checkpoint. The checkpoint contains the Qwen adapter plus ordinary
# DINOv2/projector parameters in non_lora_trainables.bin. The input is the
# already composited lane_patch_tif raster; do not draw Raw-Lane a second time.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}

MODEL_OBS_ROOT=${MODEL_OBS_ROOT:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
QWEN_VL_MODEL=${QWEN_VL_MODEL:-/cache/jn/model/CapRL-Qwen3VL-4B}
QWEN_EXTRACTED_LLM=${QWEN_EXTRACTED_LLM:-/cache/jn/model/CapRL-Qwen3VL-4B_llm_extracted}
VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-${MODEL_OBS_ROOT}/facebook_dinov2-large}

CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/08/05/5e26ea0190634699828ff4b28df4c608/output/ma-job-e95d0d2c-ee85-420e-a36f-2785ad6ce6c8/checkpoint-34376/}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_local256_550k_lora_checkpoint34376}
CHECKPOINT_EXPECTED_KIND=${CHECKPOINT_EXPECTED_KIND:-lora}

RUN_ID=${RUN_ID:-rawlane_local256_550k_lora_checkpoint34376_gt_empty_fresh_obs_e2e_$(date +%Y%m%d_%H%M%S)}
FRESH_RUN_ROOT=${FRESH_RUN_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/${RUN_ID}}
FRESH_ARCHIVE=${FRESH_ARCHIVE:-${FRESH_RUN_ROOT}/e2e_data.zip}
FRESH_E2E_ROOT=${FRESH_E2E_ROOT:-${FRESH_RUN_ROOT}/e2e_data}
INFERENCE_DATASET_ROOT=${INFERENCE_DATASET_ROOT:-${FRESH_RUN_ROOT}/rawlane_local256_dataset}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
RULE_WORKERS=${RULE_WORKERS:-16}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}

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
export QWEN3VL_EXTRACTED_LLM_PATH="${QWEN_EXTRACTED_LLM}"
export QWEN_BASE_MODEL_PATH="${QWEN_EXTRACTED_LLM}"

mkdir -p "${FRESH_RUN_ROOT}" "${OUTPUT_ROOT}" "$(dirname "${QWEN_VL_MODEL}")" \
  "$(dirname "${VISION_TOWER}")"

download_obs_directory() {
  local source=$1
  local destination=$2
  local sentinel=$3
  if [ -s "${destination}/${sentinel}" ]; then
    echo "[model-asset] reuse ${destination}"
    return
  fi
  echo "[model-asset] ${source} -> ${destination}"
  SOURCE="${source}" DESTINATION="${destination}" python - <<'PY'
import os
import moxing as mox

mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
}

echo "============================================================"
echo "RAWLANE LOCAL256-550K LLM-LORA CHECKPOINT-34376 FRESH OBS ORIGINAL E2E"
echo "Run id:          ${RUN_ID}"
echo "Fresh OBS data:  ${E2E_DATA_OBS_PATH}"
echo "Fresh archive:   ${FRESH_ARCHIVE}"
echo "Fresh E2E root:  ${FRESH_E2E_ROOT}"
echo "Inference set:   ${INFERENCE_DATASET_ROOT}"
echo "Checkpoint OBS:  ${CHECKPOINT_OBS_PATH}"
echo "Checkpoint kind: ${CHECKPOINT_EXPECTED_KIND}"
echo "Qwen-VL base:    ${QWEN_VL_MODEL}"
echo "Text LLM base:   ${QWEN_EXTRACTED_LLM}"
echo "DINOv2:          ${VISION_TOWER}"
echo "New predictions: ${RAW_RESULT_DIR}"
echo "Generation cap:  ${MAX_NEW_TOKENS}"
echo "Per-device batch:${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Output:          ${OUTPUT_ROOT}"
echo "Evaluations:     all + low + high"
echo "============================================================"

echo "[rawlane550k-fresh-e2e] stage 1/4: prepare CapRL text base and DINOv2"
download_obs_directory "${MODEL_OBS_ROOT}/CapRL-Qwen3VL-4B" "${QWEN_VL_MODEL}" config.json
download_obs_directory "${VISION_TOWER_OBS_PATH}" "${VISION_TOWER}" config.json

QWEN_VL_MODEL="${QWEN_VL_MODEL}" QWEN_EXTRACTED_LLM="${QWEN_EXTRACTED_LLM}" python - <<'PY'
import os
from mllm.model.qwen3vl_extractor import ensure_extracted_llm_from_qwen3vl

resolved = ensure_extracted_llm_from_qwen3vl(os.environ["QWEN_VL_MODEL"])
expected = os.path.abspath(os.environ["QWEN_EXTRACTED_LLM"])
if os.path.abspath(resolved) != expected:
    raise SystemExit(f"Unexpected extracted LLM path: {resolved}; expected {expected}")
print(f"[qwen-base] extracted text LLM: {resolved}")
PY

echo "[rawlane550k-fresh-e2e] stage 2/4: force-download and extract clean E2E data"
rm -f "${FRESH_ARCHIVE}"
python - "${E2E_DATA_OBS_PATH}" "${FRESH_ARCHIVE}" <<'PY'
import sys
import moxing as mox

print(f"[rawlane550k-fresh-e2e] download {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy(sys.argv[1], sys.argv[2])
PY

python scripts/tools/prepare_rc_e2e_original_run_data.py \
  --archive "${FRESH_ARCHIVE}" \
  --destination "${FRESH_E2E_ROOT}" \
  --allowed-root /cache/jn/e2e_eval/fresh_obs_runs \
  --reset

echo "[rawlane550k-fresh-e2e] stage 3/4: rebuild Raw-Lane local256 inputs and run inference"
E2E_DATA_OBS_PATH="${E2E_DATA_OBS_PATH}" \
E2E_WORK_ROOT="${FRESH_RUN_ROOT}" \
E2E_ARCHIVE_PATH="${FRESH_ARCHIVE}" \
E2E_RAW_ROOT="${FRESH_E2E_ROOT}" \
E2E_VIEW_MODE=local256 \
E2E_TARGET_SIZE=256 \
E2E_CONTEXT_SIZE=256 \
E2E_PROMPT_PROFILE=rawlane_local256_550k_v1 \
E2E_INPUT_RASTER=rawlane \
E2E_DATASET_ROOT="${INFERENCE_DATASET_ROOT}" \
REBUILD_E2E_DATASET=True \
BLACK_RATIO_THRESHOLD=1.0 \
RASTER_ALIGNMENT_REPORT="${FRESH_RUN_ROOT}/raster_alignment_report.json" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" \
CHECKPOINT_CACHE_ROOT="${CHECKPOINT_CACHE_ROOT}" \
CHECKPOINT_DIR="${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}" \
CHECKPOINT_EXPECTED_KIND="${CHECKPOINT_EXPECTED_KIND}" \
VISION_TOWER="${VISION_TOWER}" \
VISION_TOWER_OBS_PATH="${VISION_TOWER_OBS_PATH}" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
INFER_RESULT_OBS_PATH="" \
bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"

echo "[rawlane550k-fresh-e2e] stage 4/4: suppress GT-empty patches and run original all/low/high evaluation"
SOURCE_PREDICTION_DIR="${RAW_RESULT_DIR}" \
RAW_E2E_ROOT="${FRESH_E2E_ROOT}" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_ORIGINAL_E2E=True \
ORIGINAL_E2E_DATA_SOURCE=raw_direct \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=True \
RUN_HIGH_EVAL=True \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
FILL_MISSING_SCENE_PREDICTIONS=True \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
RULE_WORKERS="${RULE_WORKERS}" \
UPLOAD_RESULTS=False \
bash "${SCRIPT_DIR}/eval_local256_checkpoint34376_gt_nonempty_oracle_original_e2e_npu.sh"

echo "============================================================"
echo "RAWLANE LOCAL256-550K LLM-LORA CHECKPOINT-34376 FRESH OBS E2E COMPLETE"
echo "Fresh E2E data: ${FRESH_E2E_ROOT}"
echo "Fresh inference:${RAW_RESULT_DIR}"
echo "Suppression:    ${OUTPUT_ROOT}/gt_oracle_suppression_report.json"
echo "Scene audit:    ${OUTPUT_ROOT}/original_pipeline_metrics/scene_output_completeness.json"
echo "All roads:      ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_all"
echo "Low roads:      ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_low"
echo "High roads:     ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_high"
echo "============================================================"
