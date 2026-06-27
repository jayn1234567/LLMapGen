#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-0}}"

TRAINROOT="${TRAINROOT:-${DATA_ROOT:-}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_PATH:-}}"
DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-${DINOV2_PATH:-}}"

: "${TRAINROOT:?Set TRAINROOT to the trainroot containing train.jsonl/meta_train.jsonl.}"
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the Qwen/Qwen3 checkpoint.}"
: "${DINOV2_MODEL_NAME_OR_PATH:?Set DINOV2_MODEL_NAME_OR_PATH to the DINOv2 checkpoint.}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PATH:-outputs/dinov2_centerline_qwen_lora_npu}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
PREPARE_TRAINROOT="${PREPARE_TRAINROOT:-false}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"

ARGS=(
  scripts/train_dinov2_centerline.py
  --trainroot "${TRAINROOT}"
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --dinov2-model-name-or-path "${DINOV2_MODEL_NAME_OR_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --logging-steps "${LOGGING_STEPS}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --max-samples "${MAX_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --device-backend npu
  --ddp-backend hccl
)

if [[ "${BF16}" == "true" ]]; then
  ARGS+=(--bf16)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  ARGS+=(--gradient-checkpointing)
fi
if [[ "${PREPARE_TRAINROOT}" == "true" ]]; then
  ARGS+=(--prepare-trainroot)
fi
if [[ -n "${PREPARED_TRAINROOT:-}" ]]; then
  ARGS+=(--prepared-trainroot "${PREPARED_TRAINROOT}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "true" ]]; then
  ARGS+=(--local-files-only)
fi
if [[ -n "${TOKENIZER_NAME_OR_PATH:-}" ]]; then
  ARGS+=(--tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}")
fi
if [[ -n "${VISUAL_ENCODER_CHECKPOINT_PATH:-}" ]]; then
  ARGS+=(--visual-encoder-checkpoint-path "${VISUAL_ENCODER_CHECKPOINT_PATH}")
fi
if [[ -n "${BRIDGE_MODULES_STATE_PATH:-}" ]]; then
  ARGS+=(--bridge-modules-state-path "${BRIDGE_MODULES_STATE_PATH}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${ARGS[@]}"
else
  python "${ARGS[@]}"
fi
