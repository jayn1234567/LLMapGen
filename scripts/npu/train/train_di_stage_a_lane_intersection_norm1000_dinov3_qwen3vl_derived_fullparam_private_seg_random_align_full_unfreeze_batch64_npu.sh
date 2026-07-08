#!/usr/bin/env bash
set -euo pipefail

# Formal DI launcher for the validated batch-64 full-parameter recipe:
# - norm1000 lane+intersection trainroot from OBS
# - private-data segmentation-trained DINOv3 weights
# - randomly initialized visual alignment modules
# - Qwen3-VL-derived text LLM full-parameter training
# - fully unfrozen DINOv3 vision encoder

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_norm1000_dinov3_qwen3vl_derived_fullparam_private_seg_random_align_full_unfreeze_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: full-parameter norm1000 DINOv3 Qwen3VL-derived DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8
export TARGET_GLOBAL_BATCH_SIZE=64
export CUTOFF_LEN=4096
export SAVE_STEPS=1000
export LOGGING_STEPS=1
export BF16=true
export GRADIENT_CHECKPOINTING=true
export RUN_ID=${RUN_ID:-dinov3_private_seg_random_align_qwen3vl_derived_fullparam_norm1000_batch64_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: formal DI batch64 norm1000 DINOv3 full-parameter training"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
echo "CUTOFF_LEN=${CUTOFF_LEN}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "BF16=${BF16}"
echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
