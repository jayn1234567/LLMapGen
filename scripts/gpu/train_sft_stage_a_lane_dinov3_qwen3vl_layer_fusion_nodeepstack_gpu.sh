#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FLOW_PHASE=${FLOW_PHASE:-phase_a_lane}
export MAP_TASK=${MAP_TASK:-lane}
export VISION_BACKBONE=${VISION_BACKBONE:-dinov3}
export MODEL_FAMILY=${MODEL_FAMILY:-qwen3vl}
export DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}
export VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-"6 12 18 23"}
export VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}
exec bash "${SCRIPT_DIR}/train_sft_qwen_family_formal_gpu.sh"
