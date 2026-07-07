#!/usr/bin/env bash
set -euo pipefail

# DI/Ascend entrypoint:
# - norm1000 lane+intersection trainroot
# - private-data segmentation-trained DINOv3 weights
# - randomly initialized visual alignment modules
# - Qwen3-VL-derived text LLM full-parameter training (no LoRA)
# - fully unfrozen DINOv3 vision encoder
#
# Warning: Qwen3 8B full-parameter training is memory-heavy without ZeRO/FSDP.
# For local Ascend smoke tests, start with MAX_STEPS=5 and OPTIM=adafactor.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_norm1000_dinov3_qwen3vl_derived_lora_private_seg_random_align_full_unfreeze_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: norm1000 DINOv3 Qwen3VL-derived DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export NO_LORA=true
export FREEZE_LANGUAGE_MODEL=false
export FREEZE_VISION_ENCODER=false
export VISION_TRAIN_LAST_N_LAYERS=0
export OPTIM=${OPTIM:-adafactor}
export LEARNING_RATE=${LEARNING_RATE:-5e-6}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-5}
export RUN_ID=${RUN_ID:-dinov3_private_seg_random_align_qwen3vl_derived_fullparam_norm1000_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: norm1000 DINOv3 + random alignment + Qwen3-VL-derived full-parameter training"
echo "NO_LORA=${NO_LORA}"
echo "OPTIM=${OPTIM}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "MAX_STEPS=${MAX_STEPS:--1}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
