#!/usr/bin/env bash
set -euo pipefail

# Production NPU inference entry: DINOv2 + Qwen3VL-8B + no DeepStack.
# Set DATASET_PHASE, MAP_TASK and TRAINED_CHECKPOINT_OBS as needed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VISION_BACKBONE=dinov2
export DATASET_PHASE=${DATASET_PHASE:-phase_a}
export MAP_TASK=${MAP_TASK:-lane}
exec bash "${SCRIPT_DIR}/run_infer_nodeepstack_npu.sh" "$@"
