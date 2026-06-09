#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FLOW_PHASE=${FLOW_PHASE:-phase_a_lane}
export MAP_TASK=${MAP_TASK:-lane}
export VISION_BACKBONE=${VISION_BACKBONE:-dinov2_siglip_concat}
export MODEL_FAMILY=${MODEL_FAMILY:-qwen3vl}
export DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}
exec bash "${SCRIPT_DIR}/train_sft_qwen_family_formal_gpu.sh"
