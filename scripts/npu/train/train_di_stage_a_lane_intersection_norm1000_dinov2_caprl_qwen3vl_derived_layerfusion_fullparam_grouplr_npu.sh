#!/usr/bin/env bash
set -euo pipefail

# Grouped-LR variant of the CapRL-Qwen3VL-derived DINOv2 layer-fusion recipe.
# Keeps the same data/model route, but uses a smaller LR for DINOv2 and a
# larger LR for randomly initialized alignment modules.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_norm1000_dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base CapRL/DINOv2 layer-fusion launcher not found: ${BASE_SCRIPT}"
  exit 1
fi

export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export LANGUAGE_MODEL_LR=${LANGUAGE_MODEL_LR:-2e-5}
export VISION_ENCODER_LR=${VISION_ENCODER_LR:-5e-6}
export ALIGNMENT_LR=${ALIGNMENT_LR:-1e-4}
export RUN_ID=${RUN_ID:-dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_norm1000_$(date -u +%Y%m%d_%H%M%S)}

echo "Recipe override: grouped LR"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "LANGUAGE_MODEL_LR=${LANGUAGE_MODEL_LR}"
echo "VISION_ENCODER_LR=${VISION_ENCODER_LR}"
echo "ALIGNMENT_LR=${ALIGNMENT_LR}"
echo "RUN_ID=${RUN_ID}"

exec bash "${BASE_SCRIPT}"
