#!/usr/bin/env bash
set -euo pipefail

# Common NPU SFT launcher for explicit stage/task/vision wrappers.
#
# Wrapper-selected parameters:
#   VISION_BACKBONE=dinov2|dinov3
#     dinov2 uses DINOv2-L 518 input; dinov3 uses DINOv3-L 512 input.
#   DATASET_PHASE=phase_a|phase_b
#     phase_a: no incoming state hints; phase_b: left/top state-update hints.
#   MAP_TASK=lane|lane_intersection
#     lane predicts centerlines only; lane_intersection predicts centerlines and intersections.
#
# Main paths to edit when running on cloud:
#   DATASET_PATH: dataset root containing phase_a/phase_b train/eval/test jsonl and images.
#   OUTPUT_PATH: local/cloud output directory for checkpoints.
#   OUTPUT_URL: cloud platform output OBS path, usually injected by the platform.

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
export SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
export SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
export SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
export SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-sft}
export SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_33w_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
export SWANLAB_TAGS=${SWANLAB_TAGS:-sft,33w,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},qwen3vl8b,nodeepstack}

case "${VISION_BACKBONE}" in
  dinov2)
    exec bash scripts/npu/train/base_train_sft_dinov2_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh
    ;;
  dinov3)
    exec bash scripts/npu/train/base_train_sft_dinov3_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh
    ;;
esac
