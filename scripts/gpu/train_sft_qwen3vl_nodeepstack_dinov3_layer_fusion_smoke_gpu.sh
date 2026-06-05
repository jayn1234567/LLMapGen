#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export VISION_BACKBONE=${VISION_BACKBONE:-dinov3}
export VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-"6 12 18 23"}
export VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}
export RUN_ID=${RUN_ID:-${FLOW_PHASE:-phase_a}_${MAP_TASK:-lane}_dinov3_layer_fusion_${TRAIN_MODE:-lora}_gpu02}

exec bash "${SCRIPT_DIR}/train_sft_qwen3vl_nodeepstack_smoke_gpu.sh"
