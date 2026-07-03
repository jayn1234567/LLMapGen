#!/usr/bin/env bash
set -euo pipefail

# DI NPU SFT entrypoint for full DINOv2 fine-tuning.
# This wrapper keeps the production DI launcher as the single source of truth
# and only changes the recipe defaults needed for full vision-tower unfreezing.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_dinov2_qwen3_lora_jjh_style_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export FREEZE_VISION_ENCODER=false
export VISION_TRAIN_LAST_N_LAYERS=0
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-6}
export MAX_STEPS=${MAX_STEPS:--1}
export RUN_ID=${RUN_ID:-dinov2_full_unfreeze_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: full DINOv2 unfreeze"
echo "FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER}"
echo "VISION_TRAIN_LAST_N_LAYERS=${VISION_TRAIN_LAST_N_LAYERS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
