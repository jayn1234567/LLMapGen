#!/usr/bin/env bash
set -euo pipefail

# Common NPU SFT launcher for the explicit flow wrappers in this directory.
# Required envs are set by wrapper scripts:
#   VISION_BACKBONE=dinov2|dinov3
#   DATASET_PHASE=phase_a|phase_b
#   MAP_TASK=lane|lane_intersection

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:?set VISION_BACKBONE to dinov2 or dinov3}
DATASET_PHASE=${DATASET_PHASE:?set DATASET_PHASE to phase_a or phase_b}
MAP_TASK=${MAP_TASK:?set MAP_TASK to lane or lane_intersection}

case "${VISION_BACKBONE}" in
  dinov2|dinov3) ;;
  *) echo "ERROR: unsupported VISION_BACKBONE=${VISION_BACKBONE}"; exit 1 ;;
esac
case "${DATASET_PHASE}" in
  phase_a|phase_b) ;;
  *) echo "ERROR: unsupported DATASET_PHASE=${DATASET_PHASE}"; exit 1 ;;
esac
case "${MAP_TASK}" in
  lane|lane_intersection) ;;
  *) echo "ERROR: unsupported MAP_TASK=${MAP_TASK}"; exit 1 ;;
esac

export DATASET_PHASE
export MAP_TASK
export OUTPUT_PATH=${OUTPUT_PATH:-/cache/unimapgen_v2/train_output/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
export SWANLAB_PROJECT=${SWANLAB_PROJECT:-mllm-sft-33w-${DATASET_PHASE}-${MAP_TASK}-${VISION_BACKBONE}-nodeepstack}
export SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_33w_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
export SWANLAB_TAGS=${SWANLAB_TAGS:-sft,33w,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},qwen3vl8b,nodeepstack}

case "${VISION_BACKBONE}" in
  dinov2)
    exec bash scripts/npu/train_sft_dinov2_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh
    ;;
  dinov3)
    exec bash scripts/npu/train_sft_dinov3_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh
    ;;
esac
