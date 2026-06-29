#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${TRAINROOT:?Set TRAINROOT to the prepared trainroot.}"
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the local Qwen3-8B path.}"
: "${DINOV2_MODEL_NAME_OR_PATH:?Set DINOV2_MODEL_NAME_OR_PATH to the local DINOv2-large path.}"

# This smoke intentionally starts without the public-data visual/bridge assets.
# The DINOv2 backbone and Qwen3-8B are loaded from local model directories, while
# the alignment modules are randomly initialized MLP layers.
export VISUAL_ENCODER_CHECKPOINT_PATH=""
export BRIDGE_MODULES_STATE_PATH=""

export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-0}}"
export MAX_STEPS="${MAX_STEPS:-10}"
export MAX_SAMPLES="${MAX_SAMPLES:-16}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export BF16="${BF16:-true}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-false}"
export FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
export VISION_TRAIN_LAST_N_LAYERS="${VISION_TRAIN_LAST_N_LAYERS:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/dinov2_qwen3_8b_random_align_npu_smoke}"

echo "=== LLMapGen DINOv2 + Qwen random-align NPU smoke ==="
echo "repo=${REPO_ROOT}"
echo "trainroot=${TRAINROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "qwen=${MODEL_NAME_OR_PATH}"
echo "dinov2=${DINOV2_MODEL_NAME_OR_PATH}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "ascend_devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "vision_train_last_n_layers=${VISION_TRAIN_LAST_N_LAYERS}"

exec sh scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
