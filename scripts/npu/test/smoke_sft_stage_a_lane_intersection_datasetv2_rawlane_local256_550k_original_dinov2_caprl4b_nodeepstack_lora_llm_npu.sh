#!/usr/bin/env bash
set -euo pipefail

# Single-node Ascend smoke for the formal Raw-Lane 550k Qwen-LLM-LoRA recipe.
# Step 10 and step 20 each run loss-only eval, release unused NPU cache, and
# write an ordinary PEFT adapter plus non-LoRA vision/projector parameters.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh"
cd "${REPO_ROOT}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}

python - <<'PY'
import json
import sys

try:
    import torch
    import torch_npu
except Exception as exc:
    raise SystemExit(f"NPU Python preflight import failed: {exc!r}") from exc

report = {
    "python": sys.executable,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
}
print("[lora-smoke] preflight=" + json.dumps(report), flush=True)
if not report["npu_available"] or report["npu_count"] < 8:
    raise SystemExit(f"Expected eight available NPUs, got {report}")
PY

OBS_CACHE=${OBS_CACHE:-/cache/jn}
RUN_ID=${RUN_ID:-rawlane550k_original_dinov2_caprl4b_lora_llm_smoke_$(date -u +%Y%m%d_%H%M%S)}
SMOKE_ROOT=${SMOKE_ROOT:-${OBS_CACHE}/outputs/rawlane550k_original_dinov2_caprl4b_lora_llm_smoke}
OUTPUT_URL=${OUTPUT_URL:-${SMOKE_ROOT}/completed}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-${SMOKE_ROOT}/work}
LOG_DIR=${LOG_DIR:-${SMOKE_ROOT}/logs/${RUN_ID}}
TRAIN_LOG=${TRAIN_LOG:-${LOG_DIR}/train.log}

DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/data/rawlane_local256_550k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/data/rawlane_local256_550k_extract}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/rawlane_local256_550k}

mkdir -p "${OUTPUT_URL}" "${LOCAL_MODEL_SAVE_ROOT}" "${LOG_DIR}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"

echo "============================================================"
echo "Raw-Lane 550k Qwen LLM-LoRA checkpoint/eval smoke"
echo "NPUs:             ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Topology:         1 node x 8 NPUs"
echo "Batch:            per_device=4, accumulation=4, effective=128"
echo "Schedule:         max_steps=20, eval_steps=10, save_steps=10"
echo "Learning rates:   Qwen LoRA/projector=2e-4, DINOv2=2e-5"
echo "Dataset archive:  ${DATASET_ARCHIVE_PATH}"
echo "Dataset root:     ${DATASET_PATH}"
echo "Output:           ${OUTPUT_URL}/${RUN_ID}"
echo "============================================================"

set +e
set +u
OUTPUT_URL="${OUTPUT_URL}" \
RUN_ID="${RUN_ID}" \
OBS_CACHE="${OBS_CACHE}" \
LOCAL_MODEL_SAVE_ROOT="${LOCAL_MODEL_SAVE_ROOT}" \
DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
DATASET_PATH="${DATASET_PATH}" \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
EXPECTED_NNODES=1 \
EXPECTED_NPROC_PER_NODE=8 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${MASTER_PORT:-29683}" \
MAX_STEPS=20 \
NUM_EPOCHS=100 \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=2 \
LOGGING_STEPS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
TARGET_GLOBAL_BATCH_SIZE=128 \
ENABLE_EVAL=True \
EVAL_STEPS=10 \
EVAL_SAMPLE_LIMIT=256 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
SAVE_BEST_TRAIN_LOSS=False \
SAVE_BEST_EVAL_LOSS=False \
SAVE_BEST_INFER_INDEX=False \
SWANLAB_ENABLE=False \
INSTALL_DEPS=False \
ENABLE_MOXING_UPGRADE=False \
REUSE_LOCAL_ASSETS=True \
DATASET_INSPECT_MAX_SAMPLES="${DATASET_INSPECT_MAX_SAMPLES:-2000}" \
DATASET_IMAGE_CHECKS_PER_SPLIT="${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}" \
MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True \
MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=False \
DDP_FIND_UNUSED_PARAMETERS=True \
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}" \
bash "${FORMAL_SCRIPT}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -u
set -e

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "ERROR: LoRA checkpoint/eval smoke failed (exit=${TRAIN_EXIT}); log=${TRAIN_LOG}" >&2
  exit "${TRAIN_EXIT}"
fi

FINAL_OUTPUT="${OUTPUT_URL%/}/${RUN_ID}"
CHECKPOINT_DIR="${FINAL_OUTPUT}/checkpoint-20"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
  echo "ERROR: expected checkpoint was not produced: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT_DIR}/adapter_config.json" ]; then
  echo "ERROR: LoRA adapter config is missing: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! find "${CHECKPOINT_DIR}" -maxdepth 1 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \) -print -quit | grep -q .; then
  echo "ERROR: LoRA adapter weights are missing: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [ ! -s "${CHECKPOINT_DIR}/non_lora_trainables.bin" ]; then
  echo "ERROR: vision/projector trainables are missing: ${CHECKPOINT_DIR}/non_lora_trainables.bin" >&2
  exit 1
fi
if ! grep -Fq "Released unused NPU cache before checkpoint save." "${TRAIN_LOG}"; then
  echo "ERROR: NPU pre-save synchronization/cache cleanup did not run." >&2
  exit 1
fi
if ! grep -Eq "eval_loss|eval/loss" "${TRAIN_LOG}"; then
  echo "ERROR: eval loss was not reported during the smoke run." >&2
  exit 1
fi

echo "============================================================"
echo "RAW-LANE LLM-LORA CHECKPOINT/EVAL SMOKE PASSED"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Log:        ${TRAIN_LOG}"
echo "============================================================"
