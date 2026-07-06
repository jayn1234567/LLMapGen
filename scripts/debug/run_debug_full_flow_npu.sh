#!/usr/bin/env bash
set -euo pipefail

# One-command Ascend NPU debug flow for one phase/task/backbone.
# Default: SFT -> inference. GRPO is opt-in and uses vLLM-Ascend.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:-dinov2}
DATASET_PHASE=${DATASET_PHASE:-phase_a}
MAP_TASK=${MAP_TASK:-lane}
RUN_SFT=${RUN_SFT:-True}
RUN_INFER=${RUN_INFER:-True}
RUN_GRPO=${RUN_GRPO:-False}
RUN_GRPO_INFER=${RUN_GRPO_INFER:-False}
DEBUG_RUN_NAME=${DEBUG_RUN_NAME:-local_debug}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/debug}

export VISION_BACKBONE DATASET_PHASE MAP_TASK DEBUG_RUN_NAME OUTPUT_ROOT

if [[ "${RUN_SFT}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  bash scripts/debug/train_sft_debug_npu.sh
fi

if [[ "${RUN_INFER}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  bash scripts/debug/infer_debug_npu.sh
fi

if [[ "${RUN_GRPO}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  bash scripts/debug/train_grpo_debug_npu.sh
  if [[ "${RUN_GRPO_INFER}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    GRPO_OUTPUT_DIR="${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack"
    if CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
        --output-dir "${GRPO_OUTPUT_DIR}" \
        --best-name best_reward \
        --allow-direct 2>/dev/null); then
      export CHECKPOINT_DIR
    else
      export CHECKPOINT_DIR="${GRPO_OUTPUT_DIR}/merged"
    fi
    export OUTPUT_DIR="${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/infer_grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack"
    bash scripts/debug/infer_debug_npu.sh
  fi
fi

echo "Debug flow finished:"
echo "  run=${DEBUG_RUN_NAME}"
echo "  output_root=${OUTPUT_ROOT}/${DEBUG_RUN_NAME}"
