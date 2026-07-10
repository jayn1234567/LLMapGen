#!/usr/bin/env bash
set -euo pipefail

# Jiangjihua-style recipe implemented in LLMapGen:
# - norm1000 lane+intersection prepared trainroot
# - CapRL-Qwen3VL-4B -> extracted Qwen3 text LLM
# - original DINOv2-L/14 vision tower
# - DINOv2 layers 6/12/18/23 mean-fused into the main visual stream
# - randomly initialized LLMapGen visual alignment modules
# - full-parameter SFT with DeepSpeed ZeRO-3

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_dinov2_qwen3_lora_jjh_style_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot_norm1000.zip}
export TRAINROOT_DIR_NAME=${TRAINROOT_DIR_NAME:-prepared_lane_intersection_trainroot_norm1000}
export QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/CapRL-Qwen3VL-4B}
export QWEN_PATH=${QWEN_PATH:-${WORK_ROOT:-/cache/llmapgen}/model/CapRL-Qwen3VL-4B}
export EXTRACT_QWEN3VL_TEXT_LLM=true
export QWEN_EXTRACTED_TEXT_PATH=${QWEN_EXTRACTED_TEXT_PATH:-${WORK_ROOT:-/cache/llmapgen}/model/CapRL-Qwen3VL-4B_llm_extracted}

export DINOV2_MODEL_OBS_PATH=${DINOV2_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
export VISION_MODEL_OBS_PATH=${VISION_MODEL_OBS_PATH:-${DINOV2_MODEL_OBS_PATH}}
export VISION_MODEL_FAMILY=dinov2
export VISION_PATH=${VISION_PATH:-${WORK_ROOT:-/cache/llmapgen}/model/dinov2-large}
export VISION_PATCH_SIZE=14
export VISION_NUM_PREFIX_TOKENS=-1
export ENCODER_INPUT_PAD_SIZE=518
export VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-6,12,18,23}
export VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}

export USE_PRETRAINED_VISUAL_BRIDGE=false
export USE_VISUAL_ENCODER_CHECKPOINT=false
export BRIDGE_MODULES_STATE_PATH=
export VISUAL_ENCODER_CHECKPOINT_PATH=

export MAP_TASK=lane_intersection
export IMAGE_SIZE=512
export CUTOFF_LEN=${CUTOFF_LEN:-4096}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-5}
export MAX_STEPS=${MAX_STEPS:--1}
export TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-64}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export LANGUAGE_MODEL_LR=${LANGUAGE_MODEL_LR:-2e-5}
export ALIGNMENT_LR=${ALIGNMENT_LR:-2e-5}
export VISION_ENCODER_LR=${VISION_ENCODER_LR:-2e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
export NO_LORA=true
export FREEZE_LANGUAGE_MODEL=false
export FREEZE_VISION_ENCODER=false
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-configs/deepspeed_zero3.json}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
export LOGGING_STEPS=${LOGGING_STEPS:-1}
export BF16=true
export GRADIENT_CHECKPOINTING=true
export RUN_ID=${RUN_ID:-dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_norm1000_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: norm1000 DINOv2 layer-fusion + CapRL-Qwen3VL-derived text LLM full-parameter SFT"
echo "DATASET_OBS_PATH=${DATASET_OBS_PATH}"
echo "QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH}"
echo "VISION_MODEL_OBS_PATH=${VISION_MODEL_OBS_PATH}"
echo "VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES}"
echo "VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE}"
echo "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
