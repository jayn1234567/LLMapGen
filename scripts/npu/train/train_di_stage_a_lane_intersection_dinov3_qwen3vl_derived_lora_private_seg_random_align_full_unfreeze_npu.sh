#!/usr/bin/env bash
set -euo pipefail

# DI NPU SFT entrypoint:
# - DINOv3 base model from DINOV3_BASE_MODEL_OBS_PATH
# - private-data segmentation-trained DINOv3 weights from DINOV3_VISUAL_CHECKPOINT_OBS_PATH
# - randomly initialized visual alignment modules
# - Qwen3-VL-8B-Instruct text LLM extracted to Qwen3ForCausalLM
# - fully unfrozen DINOv3 vision encoder
# - extracted Qwen3 text LLM trained with LoRA

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_dinov2_qwen3_lora_jjh_style_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export USE_PRETRAINED_VISUAL_BRIDGE=false
export USE_VISUAL_ENCODER_CHECKPOINT=true
export FREEZE_VISION_ENCODER=false
export VISION_TRAIN_LAST_N_LAYERS=0
export EXTRACT_QWEN3VL_TEXT_LLM=true

export DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.zip"
export QWEN3VL_MODEL_OBS_PATH=${QWEN3VL_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/Qwen3-VL-8B-Instruct}
export QWEN_MODEL_OBS_PATH="${QWEN3VL_MODEL_OBS_PATH}"
export QWEN_PATH="${QWEN_PATH:-/cache/llmapgen/model/Qwen3-VL-8B-Instruct}"
export QWEN_EXTRACTED_TEXT_PATH="${QWEN_EXTRACTED_TEXT_PATH:-/cache/llmapgen/model/Qwen3-VL-8B-Instruct_llm_extracted}"

export DINOV3_BASE_MODEL_OBS_PATH=${DINOV3_BASE_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
export DINOV3_VISUAL_CHECKPOINT_OBS_PATH=${DINOV3_VISUAL_CHECKPOINT_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov3_lora.pt}
export VISION_MODEL_FAMILY=dinov3
export VISION_MODEL_OBS_PATH="${DINOV3_BASE_MODEL_OBS_PATH}"
export VISION_PATH="${VISION_PATH:-/cache/llmapgen/model/facebook_dinov3-vitl16-pretrain-lvd1689m}"
export DINOV2_MODEL_OBS_PATH="${VISION_MODEL_OBS_PATH}"
export DINOV2_PATH="${VISION_PATH}"
export VISUAL_ENCODER_CHECKPOINT_OBS_PATH="${DINOV3_VISUAL_CHECKPOINT_OBS_PATH}"
export VISUAL_ENCODER_CHECKPOINT_PATH="${VISUAL_ENCODER_CHECKPOINT_PATH:-/cache/llmapgen/model/dinov3_lora.pt}"
export VISION_PATCH_SIZE=${VISION_PATCH_SIZE:-16}
export VISION_NUM_PREFIX_TOKENS=${VISION_NUM_PREFIX_TOKENS:--1}
export ENCODER_INPUT_PAD_SIZE=${ENCODER_INPUT_PAD_SIZE:-512}

export VISUAL_TOKEN_COMPRESSOR=${VISUAL_TOKEN_COMPRESSOR:-none}
export VISUAL_TOKEN_COMPRESSOR_GRID_SIZE=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE:-0}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-5}
export MAX_STEPS=${MAX_STEPS:--1}
export RUN_ID=${RUN_ID:-dinov3_private_seg_random_align_qwen3vl_derived_lora_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: private DINOv3 segmentation backbone + random alignment + Qwen3-VL-derived text LLM LoRA"
echo "USE_VISUAL_ENCODER_CHECKPOINT=${USE_VISUAL_ENCODER_CHECKPOINT}"
echo "USE_PRETRAINED_VISUAL_BRIDGE=${USE_PRETRAINED_VISUAL_BRIDGE}"
echo "FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER}"
echo "VISION_TRAIN_LAST_N_LAYERS=${VISION_TRAIN_LAST_N_LAYERS}"
echo "EXTRACT_QWEN3VL_TEXT_LLM=${EXTRACT_QWEN3VL_TEXT_LLM}"
echo "DATASET_OBS_PATH=${DATASET_OBS_PATH}"
echo "QWEN3VL_MODEL_OBS_PATH=${QWEN3VL_MODEL_OBS_PATH}"
echo "QWEN_PATH=${QWEN_PATH}"
echo "QWEN_EXTRACTED_TEXT_PATH=${QWEN_EXTRACTED_TEXT_PATH}"
echo "DINOV3_BASE_MODEL_OBS_PATH=${DINOV3_BASE_MODEL_OBS_PATH}"
echo "DINOV3_VISUAL_CHECKPOINT_OBS_PATH=${DINOV3_VISUAL_CHECKPOINT_OBS_PATH}"
echo "VISION_PATH=${VISION_PATH}"
echo "VISUAL_ENCODER_CHECKPOINT_PATH=${VISUAL_ENCODER_CHECKPOINT_PATH}"
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
