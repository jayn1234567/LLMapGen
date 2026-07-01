#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINROOT="${TRAINROOT:-${DATA_ROOT:-}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_PATH:-}}"
DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-${DINOV2_PATH:-}}"

: "${TRAINROOT:?Set TRAINROOT to the trainroot containing train.jsonl/meta_train.jsonl.}"
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the Qwen/Qwen3 checkpoint.}"
: "${DINOV2_MODEL_NAME_OR_PATH:?Set DINOV2_MODEL_NAME_OR_PATH to the DINOv2 checkpoint.}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PATH:-outputs/dinov2_centerline_qwen_lora_gpu}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
USE_TORCHRUN="${USE_TORCHRUN:-true}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
PREPARE_TRAINROOT="${PREPARE_TRAINROOT:-false}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
VISION_TRAIN_LAST_N_LAYERS="${VISION_TRAIN_LAST_N_LAYERS:-0}"
MAP_TASK="${MAP_TASK:-lane}"
USE_GLOBAL_LOCAL_VIEWS="${USE_GLOBAL_LOCAL_VIEWS:-false}"
GLOBAL_LOCAL_VIEW_COUNT="${GLOBAL_LOCAL_VIEW_COUNT:-2}"
CONTEXT_IMAGE_KEY="${CONTEXT_IMAGE_KEY:-context_image}"
REQUIRE_CONTEXT_IMAGE="${REQUIRE_CONTEXT_IMAGE:-false}"
GLOBAL_LOCAL_PROMPT="${GLOBAL_LOCAL_PROMPT:-true}"
USE_VIEW_TYPE_EMBEDDING="${USE_VIEW_TYPE_EMBEDDING:-auto}"
VIEW_TYPE_EMBEDDING_COUNT="${VIEW_TYPE_EMBEDDING_COUNT:-2}"
VIEW_TYPE_EMBEDDING_INIT_STD="${VIEW_TYPE_EMBEDDING_INIT_STD:-0.02}"

set -- \
  scripts/train_dinov2_centerline.py \
  --trainroot "${TRAINROOT}" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --dinov2-model-name-or-path "${DINOV2_MODEL_NAME_OR_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --logging-steps "${LOGGING_STEPS}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --map-task "${MAP_TASK}" \
  --device-backend cuda \
  --ddp-backend nccl

if [ "${BF16}" = "true" ]; then
  set -- "$@" --bf16
fi
if [ "${GRADIENT_CHECKPOINTING}" = "true" ]; then
  set -- "$@" --gradient-checkpointing
fi
if [ "${PREPARE_TRAINROOT}" = "true" ]; then
  set -- "$@" --prepare-trainroot
fi
if [ "${USE_GLOBAL_LOCAL_VIEWS}" = "true" ]; then
  set -- "$@" \
    --use-global-local-views \
    --global-local-view-count "${GLOBAL_LOCAL_VIEW_COUNT}" \
    --context-image-key "${CONTEXT_IMAGE_KEY}"
fi
if [ "${REQUIRE_CONTEXT_IMAGE}" = "true" ]; then
  set -- "$@" --require-context-image
fi
if [ "${GLOBAL_LOCAL_PROMPT}" = "false" ]; then
  set -- "$@" --no-global-local-prompt
fi
if [ "${USE_VIEW_TYPE_EMBEDDING}" = "true" ]; then
  set -- "$@" --use-view-type-embedding
elif [ "${USE_VIEW_TYPE_EMBEDDING}" = "false" ]; then
  set -- "$@" --no-use-view-type-embedding
fi
set -- "$@" \
  --view-type-embedding-count "${VIEW_TYPE_EMBEDDING_COUNT}" \
  --view-type-embedding-init-std "${VIEW_TYPE_EMBEDDING_INIT_STD}"
if [ -n "${PREPARED_TRAINROOT:-}" ]; then
  set -- "$@" --prepared-trainroot "${PREPARED_TRAINROOT}"
fi
if [ "${LOCAL_FILES_ONLY}" = "true" ]; then
  set -- "$@" --local-files-only
fi
if [ "${FREEZE_LANGUAGE_MODEL}" = "true" ]; then
  set -- "$@" --freeze-language-model
else
  set -- "$@" --no-freeze-language-model
fi
if [ "${FREEZE_VISION_ENCODER}" = "true" ]; then
  set -- "$@" --freeze-vision-encoder
else
  set -- "$@" --no-freeze-vision-encoder
fi
if [ "${VISION_TRAIN_LAST_N_LAYERS}" != "0" ]; then
  set -- "$@" --vision-train-last-n-layers "${VISION_TRAIN_LAST_N_LAYERS}"
fi
if [ -n "${TOKENIZER_NAME_OR_PATH:-}" ]; then
  set -- "$@" --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}"
fi
if [ -n "${VISUAL_ENCODER_CHECKPOINT_PATH:-}" ]; then
  set -- "$@" --visual-encoder-checkpoint-path "${VISUAL_ENCODER_CHECKPOINT_PATH}"
fi
if [ -n "${BRIDGE_MODULES_STATE_PATH:-}" ]; then
  set -- "$@" --bridge-modules-state-path "${BRIDGE_MODULES_STATE_PATH}"
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  set -- "$@" --resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}"
fi
if [ -n "${SYSTEM_PROMPT:-}" ]; then
  set -- "$@" --system-prompt "${SYSTEM_PROMPT}"
fi
if [ -n "${USER_PROMPT:-}" ]; then
  set -- "$@" --user-prompt "${USER_PROMPT}"
fi

if [ "${USE_TORCHRUN}" = "true" ]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "$@"
else
  "${PYTHON_BIN}" "$@"
fi
