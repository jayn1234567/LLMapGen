#!/usr/bin/env bash
set -euo pipefail

# DI NPU SFT entrypoint:
# - norm1000 lane+intersection prepared trainroot from OBS
# - DINOv3 base model from DINOV3_BASE_MODEL_OBS_PATH
# - private-data segmentation-trained DINOv3 weights from DINOV3_VISUAL_CHECKPOINT_OBS_PATH
# - randomly initialized visual alignment modules
# - Qwen3-VL-8B-Instruct text LLM extracted to Qwen3ForCausalLM
# - fully unfrozen DINOv3 vision encoder
# - extracted Qwen3 text LLM trained with LoRA

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_dinov3_qwen3vl_derived_lora_private_seg_random_align_full_unfreeze_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: DINOv3 Qwen3VL-derived DI launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot_norm1000.zip"
export TRAINROOT_DIR_NAME="prepared_lane_intersection_trainroot_norm1000"
export MAP_TASK=lane_intersection

export RUN_ID=${RUN_ID:-dinov3_private_seg_random_align_qwen3vl_derived_lora_norm1000_latest_prompt_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: norm1000 dataset + latest lane/intersection prompt"
echo "DATASET_OBS_PATH=${DATASET_OBS_PATH}"
echo "TRAINROOT_DIR_NAME=${TRAINROOT_DIR_NAME}"
echo "MAP_TASK=${MAP_TASK}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
