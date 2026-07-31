#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
cd "${REPO_ROOT}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAX_STEPS=${MAX_STEPS:-2}
SAVE_STEPS=${SAVE_STEPS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
OBS_CACHE=${OBS_CACHE:-/cache/jn}
RUN_ID=${RUN_ID:-rawlane550k_zero3_sharded_checkpoint_smoke_$(date -u +%Y%m%d_%H%M%S)}
SMOKE_ROOT=${SMOKE_ROOT:-${OBS_CACHE}/outputs/rawlane550k_zero3_sharded_checkpoint_smoke}
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
echo "Raw-Lane 550k ZeRO-3 sharded checkpoint smoke"
echo "NPUs:             ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Batch:            per_device=${PER_DEVICE_TRAIN_BATCH_SIZE}, global=${TARGET_GLOBAL_BATCH_SIZE}"
echo "Steps/save:       max_steps=${MAX_STEPS}, save_steps=${SAVE_STEPS}"
echo "DeepSpeed:        scripts/deepspeed_zero3_no_merge.json"
echo "Dataset archive:  ${DATASET_ARCHIVE_PATH}"
echo "Dataset root:     ${DATASET_PATH}"
echo "Output:           ${OUTPUT_URL}/${RUN_ID}"
echo "============================================================"

set +e
OUTPUT_URL="${OUTPUT_URL}" \
RUN_ID="${RUN_ID}" \
OBS_CACHE="${OBS_CACHE}" \
LOCAL_MODEL_SAVE_ROOT="${LOCAL_MODEL_SAVE_ROOT}" \
DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
DATASET_PATH="${DATASET_PATH}" \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${MASTER_PORT:-29671}" \
MAX_STEPS="${MAX_STEPS}" \
NUM_EPOCHS=100 \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT=1 \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE}" \
CHECKPOINT_SAVE_MODE=sharded \
DEEPSPEED_CONFIG=scripts/deepspeed_zero3_no_merge.json \
MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True \
MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=True \
ENABLE_EVAL=False \
SAVE_BEST_TRAIN_LOSS=False \
SAVE_BEST_EVAL_LOSS=False \
SAVE_BEST_INFER_INDEX=False \
SWANLAB_ENABLE=False \
INSTALL_DEPS=False \
ENABLE_MOXING_UPGRADE=False \
REUSE_LOCAL_ASSETS=True \
DATASET_INSPECT_MAX_SAMPLES="${DATASET_INSPECT_MAX_SAMPLES:-2000}" \
DATASET_IMAGE_CHECKS_PER_SPLIT="${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}" \
LOGGING_STEPS=1 \
bash "${FORMAL_SCRIPT}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "ERROR: sharded checkpoint smoke failed (exit=${TRAIN_EXIT}); log=${TRAIN_LOG}" >&2
  exit "${TRAIN_EXIT}"
fi

FINAL_OUTPUT="${OUTPUT_URL%/}/${RUN_ID}"
NODE_OUTPUT="${FINAL_OUTPUT}/zero_shards/node_0"
CHECKPOINT_DIR=$(find "${NODE_OUTPUT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [ -z "${CHECKPOINT_DIR}" ]; then
  echo "ERROR: no checkpoint-* directory found under ${FINAL_OUTPUT}" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT_DIR}/zero_to_fp32.py" ]; then
  echo "ERROR: checkpoint is not a recoverable ZeRO shard set: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! find "${CHECKPOINT_DIR}" -type f \( -name '*model_states.pt' -o -name '*optim_states.pt' \) -print -quit | grep -q .; then
  echo "ERROR: DeepSpeed state shards were not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if find "${CHECKPOINT_DIR}" -maxdepth 1 -type f \( -name 'model.safetensors' -o -name 'pytorch_model.bin' \) -print -quit | grep -q .; then
  echo "ERROR: checkpoint unexpectedly contains a gathered full model: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! grep -Fq "Released unused NPU cache before checkpoint save." "${TRAIN_LOG}"; then
  echo "ERROR: checkpoint cache-release guard did not run." >&2
  exit 1
fi

echo "============================================================"
echo "ZERO-3 SHARDED CHECKPOINT SMOKE PASSED"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Log:        ${TRAIN_LOG}"
echo "Merge later: bash scripts/tools/merge_zero3_multinode_checkpoint.sh '${FINAL_OUTPUT}' '$(basename "${CHECKPOINT_DIR}")' '${FINAL_OUTPUT}/merged_$(basename "${CHECKPOINT_DIR}")'"
echo "============================================================"
