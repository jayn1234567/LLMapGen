#!/usr/bin/env bash
# set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ====================== Edit this block for this test job ======================
export DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip}
export DATASET_DIR_NAME=${DATASET_DIR_NAME:-MLLM20260427_rc_jjh}
export MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}

# Single or multiple weights:
# CHECKPOINT_OBS_LIST=obs://.../checkpoint-500,obs://.../eval_best_candidates
# TRAINED_CHECKPOINT_OBS=obs://.../train_output CHECKPOINT_NAMES=checkpoint-500,eval_best_candidates,best_candidates,merged
# CHECKPOINT_DIRS=/cache/.../checkpoint-500,/cache/.../eval_best_candidates
export CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}
export TRAINED_CHECKPOINT_OBS=${TRAINED_CHECKPOINT_OBS:-}
export CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-}
export CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}

export OUTPUT_DIR=${OUTPUT_DIR:-}
export TEST_RESULT_OBS=${TEST_RESULT_OBS:-}
export NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
export INSTALL_DEPS=${INSTALL_DEPS:-}
export ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-}
# =============================================================================

export VISION_BACKBONE=dinov2
export DATASET_PHASE=phase_b
export MAP_TASK=lane
exec bash "${SCRIPT_DIR}/run_infer_nodeepstack_npu.sh" "$@"
