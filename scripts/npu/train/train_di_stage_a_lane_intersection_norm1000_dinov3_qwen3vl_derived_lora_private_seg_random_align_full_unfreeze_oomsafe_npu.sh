#!/usr/bin/env bash
set -euo pipefail

# Conservative DI NPU launcher for the norm1000 lane+intersection recipe.
# Use this when full DINOv3 unfreeze OOMs with larger per-device batches.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_norm1000_dinov3_qwen3vl_derived_lora_private_seg_random_align_full_unfreeze_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: norm1000 DINOv3 Qwen3VL-derived DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export CUTOFF_LEN=${CUTOFF_LEN:-4096}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
export TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-32}
export BF16=${BF16:-true}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export LOGGING_STEPS=${LOGGING_STEPS:-1}
export RUN_ID=${RUN_ID:-dinov3_private_seg_random_align_qwen3vl_derived_lora_norm1000_oomsafe_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: OOM-safe norm1000 DINOv3 full-unfreeze LoRA"
echo "CUTOFF_LEN=${CUTOFF_LEN}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
echo "BF16=${BF16}"
echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
