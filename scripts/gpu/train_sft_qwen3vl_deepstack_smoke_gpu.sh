#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export VISION_BACKBONE=${VISION_BACKBONE:-dinov2}
export DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-False}
export DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-"6 12 18 23"}
export RUN_ID=${RUN_ID:-${FLOW_PHASE:-phase_a}_${MAP_TASK:-lane}_${VISION_BACKBONE}_deepstack_${TRAIN_MODE:-lora}_gpu02}

exec bash "${SCRIPT_DIR}/train_sft_qwen3vl_nodeepstack_smoke_gpu.sh"
