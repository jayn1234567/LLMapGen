#!/usr/bin/env bash
set -euo pipefail

# DI NPU SFT entrypoint:
# - original DINOv2 checkpoint from DINOV2_MODEL_OBS_PATH
# - randomly initialized visual alignment modules
# - fully unfrozen DINOv2 vision encoder
# - Qwen3-8B trained with LoRA

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_dinov2_qwen3_lora_jjh_style_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export USE_PRETRAINED_VISUAL_BRIDGE=false
export FREEZE_VISION_ENCODER=false
export VISION_TRAIN_LAST_N_LAYERS=0
export DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.zip"
export QWEN_MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/checkpoint/Qwen3-8B"
export DINOV2_MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large"
export VISUAL_TOKEN_COMPRESSOR=${VISUAL_TOKEN_COMPRESSOR:-none}
export VISUAL_TOKEN_COMPRESSOR_GRID_SIZE=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE:-0}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-5}
export MAX_STEPS=${MAX_STEPS:--1}
export RUN_ID=${RUN_ID:-dinov2_original_random_align_full_unfreeze_lora_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: original DINOv2 + random alignment + full DINOv2 unfreeze + Qwen LoRA"
echo "USE_PRETRAINED_VISUAL_BRIDGE=${USE_PRETRAINED_VISUAL_BRIDGE}"
echo "FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER}"
echo "VISION_TRAIN_LAST_N_LAYERS=${VISION_TRAIN_LAST_N_LAYERS}"
echo "DATASET_OBS_PATH=${DATASET_OBS_PATH}"
echo "QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH}"
echo "DINOV2_MODEL_OBS_PATH=${DINOV2_MODEL_OBS_PATH}"
echo "VISUAL_TOKEN_COMPRESSOR=${VISUAL_TOKEN_COMPRESSOR}"
echo "VISUAL_TOKEN_COMPRESSOR_GRID_SIZE=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
