#!/usr/bin/env bash
set -euo pipefail

# DI NPU SFT entrypoint:
# - private-data segmentation-trained DINOv3 checkpoint from DINOV3_MODEL_OBS_PATH
# - randomly initialized visual alignment modules
# - fully unfrozen DINOv3 vision encoder
# - Qwen3-8B trained with LoRA

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_dinov2_qwen3_lora_jjh_style_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

if [ -z "${DINOV3_MODEL_OBS_PATH:-}" ]; then
  echo "ERROR: DINOV3_MODEL_OBS_PATH is required."
  echo "Example:"
  echo "  DINOV3_MODEL_OBS_PATH=obs://bucket/path/to/private_dinov3 bash ${SCRIPT_PATH}"
  exit 1
fi

export USE_PRETRAINED_VISUAL_BRIDGE=false
export FREEZE_VISION_ENCODER=false
export VISION_TRAIN_LAST_N_LAYERS=0
export DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.zip"
export QWEN_MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/checkpoint/Qwen3-8B"

export VISION_MODEL_FAMILY=dinov3
export VISION_MODEL_OBS_PATH="${DINOV3_MODEL_OBS_PATH}"
export VISION_PATH="${VISION_PATH:-/cache/llmapgen/model/dinov3-private-seg}"
export DINOV2_MODEL_OBS_PATH="${VISION_MODEL_OBS_PATH}"
export DINOV2_PATH="${VISION_PATH}"
export VISION_PATCH_SIZE=${VISION_PATCH_SIZE:-16}
export VISION_NUM_PREFIX_TOKENS=${VISION_NUM_PREFIX_TOKENS:--1}
export ENCODER_INPUT_PAD_SIZE=${ENCODER_INPUT_PAD_SIZE:-512}

export VISUAL_TOKEN_COMPRESSOR=${VISUAL_TOKEN_COMPRESSOR:-none}
export VISUAL_TOKEN_COMPRESSOR_GRID_SIZE=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE:-0}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-5}
export MAX_STEPS=${MAX_STEPS:--1}
export RUN_ID=${RUN_ID:-dinov3_private_seg_random_align_full_unfreeze_lora_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: private DINOv3 segmentation backbone + random alignment + full DINOv3 unfreeze + Qwen LoRA"
echo "USE_PRETRAINED_VISUAL_BRIDGE=${USE_PRETRAINED_VISUAL_BRIDGE}"
echo "FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER}"
echo "VISION_TRAIN_LAST_N_LAYERS=${VISION_TRAIN_LAST_N_LAYERS}"
echo "DATASET_OBS_PATH=${DATASET_OBS_PATH}"
echo "QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH}"
echo "DINOV3_MODEL_OBS_PATH=${DINOV3_MODEL_OBS_PATH}"
echo "VISION_PATH=${VISION_PATH}"
echo "VISION_PATCH_SIZE=${VISION_PATCH_SIZE}"
echo "VISION_NUM_PREFIX_TOKENS=${VISION_NUM_PREFIX_TOKENS}"
echo "ENCODER_INPUT_PAD_SIZE=${ENCODER_INPUT_PAD_SIZE}"
echo "VISUAL_TOKEN_COMPRESSOR=${VISUAL_TOKEN_COMPRESSOR}"
echo "VISUAL_TOKEN_COMPRESSOR_GRID_SIZE=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
