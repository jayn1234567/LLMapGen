#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# This wrapper enables train-loss best checkpointing inside the canonical
# DINOv3 DeepStack script. Edit the values here for this variant.
export SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}
export BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000}
export BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best}

exec bash "${SCRIPT_DIR}/train_full_dinov3_qwen3vl-8b_deepstack_npu.sh" "$@"
