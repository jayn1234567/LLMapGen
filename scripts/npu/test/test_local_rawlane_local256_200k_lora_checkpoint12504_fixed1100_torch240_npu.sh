#!/usr/bin/env bash
set -euo pipefail

# Evaluate the Raw-Lane local256 200k Qwen-LoRA checkpoint. The visual tower
# and projector are restored from non_lora_trainables.bin; the Qwen adapter is
# merged onto the text LLM extracted from the original CapRL-Qwen3VL-4B base.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

MODEL_OBS_ROOT=${MODEL_OBS_ROOT:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
QWEN_VL_MODEL=${QWEN_VL_MODEL:-/cache/jn/model/CapRL-Qwen3VL-4B}
QWEN_EXTRACTED_LLM=${QWEN_EXTRACTED_LLM:-/cache/jn/model/CapRL-Qwen3VL-4B_llm_extracted}
VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_200k_rawlane/local256_200k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/rawlane_local256_200k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/rawlane_local256_200k_extract}
DATASET_ROOT=${DATASET_ROOT:-${DATASET_EXTRACT_ROOT}/local256}
EVAL_SOURCE_JSONL=${EVAL_SOURCE_JSONL:-${DATASET_ROOT}/phase_a/eval.jsonl}
FIXED_EVAL_ROOT=${FIXED_EVAL_ROOT:-/cache/jn/eval_sets/rawlane_local256_200k_fixed1100_e300_m300_h300_vh200_seed42_v1}

CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-12504}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/08/02/8cfad7c8fd884a8ea34ad63cd92fbda4/output/ma-job-157ed3f3-a797-47bf-9027-ce7e5d4cf29b/${CHECKPOINT_NAME}/}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_local256_200k_lora_20260802}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-2}
VIS_LIMIT=${VIS_LIMIT:-50}
RUN_ID=${RUN_ID:-rawlane_local256_200k_lora_checkpoint12504_fixed1100_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}

export QWEN3VL_EXTRACTED_LLM_PATH="${QWEN_EXTRACTED_LLM}"
export QWEN_BASE_MODEL_PATH="${QWEN_EXTRACTED_LLM}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${QWEN_VL_MODEL}")" "$(dirname "${VISION_TOWER}")" \
  "${CHECKPOINT_DIR}" "${OUTPUT_ROOT}"

download_obs_directory() {
  local source=$1
  local destination=$2
  local sentinel=$3
  if [ -e "${destination}/${sentinel}" ]; then
    echo "[asset] reuse ${destination}"
    return
  fi
  echo "[asset] ${source} -> ${destination}"
  SOURCE="${source}" DESTINATION="${destination}" python - <<'PY'
import os
import moxing as mox

mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
}

download_obs_directory "${MODEL_OBS_ROOT}/CapRL-Qwen3VL-4B" "${QWEN_VL_MODEL}" config.json
download_obs_directory "${MODEL_OBS_ROOT}/facebook_dinov2-large" "${VISION_TOWER}" config.json

if [ ! -f "${CHECKPOINT_DIR}/adapter_config.json" ] || \
   { [ ! -f "${CHECKPOINT_DIR}/adapter_model.safetensors" ] && [ ! -f "${CHECKPOINT_DIR}/adapter_model.bin" ]; } || \
   [ ! -f "${CHECKPOINT_DIR}/non_lora_trainables.bin" ]; then
  echo "[checkpoint] ${CHECKPOINT_OBS_PATH} -> ${CHECKPOINT_DIR}"
  CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" python - <<'PY'
import os
import moxing as mox

mox.file.copy_parallel(os.environ["CHECKPOINT_OBS_PATH"], os.environ["CHECKPOINT_DIR"])
PY
fi

if [ ! -f "${CHECKPOINT_DIR}/adapter_config.json" ]; then
  echo "ERROR: adapter_config.json is missing from ${CHECKPOINT_DIR}" >&2
  exit 2
fi
if [ ! -f "${CHECKPOINT_DIR}/adapter_model.safetensors" ] && [ ! -f "${CHECKPOINT_DIR}/adapter_model.bin" ]; then
  echo "ERROR: LoRA adapter weights are missing from ${CHECKPOINT_DIR}" >&2
  exit 2
fi
if [ ! -f "${CHECKPOINT_DIR}/non_lora_trainables.bin" ]; then
  echo "ERROR: non_lora_trainables.bin is missing from ${CHECKPOINT_DIR}" >&2
  exit 2
fi

QWEN_VL_MODEL="${QWEN_VL_MODEL}" QWEN_EXTRACTED_LLM="${QWEN_EXTRACTED_LLM}" python - <<'PY'
import os
from mllm.model.qwen3vl_extractor import ensure_extracted_llm_from_qwen3vl

resolved = ensure_extracted_llm_from_qwen3vl(os.environ["QWEN_VL_MODEL"])
expected = os.path.abspath(os.environ["QWEN_EXTRACTED_LLM"])
if os.path.abspath(resolved) != expected:
    raise SystemExit(f"Unexpected extracted LLM path: {resolved}; expected {expected}")
print(f"[qwen-base] extracted text LLM: {resolved}")
PY

echo "============================================================"
echo "Raw-Lane local256 200k LoRA fixed-1100 evaluation"
echo "Checkpoint:     ${CHECKPOINT_DIR}"
echo "Qwen-VL base:   ${QWEN_VL_MODEL}"
echo "Text LLM base:  ${QWEN_EXTRACTED_LLM}"
echo "DINOv2:         ${VISION_TOWER}"
echo "Dataset:        ${DATASET_OBS_PATH}"
echo "Fixed eval:     ${FIXED_EVAL_ROOT}"
echo "Visible NPUs:   ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Output:         ${OUTPUT_ROOT}"
echo "============================================================"

SKIP_ENV_ACTIVATION=True \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}" \
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
VIS_LIMIT="${VIS_LIMIT}" \
DATASET_OBS_PATH="${DATASET_OBS_PATH}" \
DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
DATASET_ROOT="${DATASET_ROOT}" \
EVAL_SOURCE_JSONL="${EVAL_SOURCE_JSONL}" \
FIXED_EVAL_ROOT="${FIXED_EVAL_ROOT}" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" \
CHECKPOINT_CACHE_ROOT="${CHECKPOINT_CACHE_ROOT}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
VISION_TOWER="${VISION_TOWER}" \
PATCH_SIZE=256 \
RUN_LABEL="rawlane_local256_200k_lora_checkpoint12504_fixed1100" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

