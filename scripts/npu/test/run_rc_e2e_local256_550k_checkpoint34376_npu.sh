#!/usr/bin/env bash
set -euo pipefail

# Full RC E2E inference for the original-DINOv2 model trained on Dataset V2
# local256-550k. The immutable v1 prompt matches checkpoint-34376 training.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

export E2E_VIEW_MODE=local256
export E2E_TARGET_SIZE=256
export E2E_CONTEXT_SIZE=256
export E2E_PROMPT_PROFILE=local256_550k_v1
export E2E_DATASET_ROOT=${E2E_DATASET_ROOT:-/cache/jn/e2e_eval/e2e_data_local256_550k_v1}

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}
export CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/}
export CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/local256_550k_checkpoint34376}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}}

export RUN_ID=${RUN_ID:-rc_e2e_local256_550k_checkpoint34376_$(date -u +%Y%m%d_%H%M%S)}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
export INFER_RESULT_OBS_PATH=${INFER_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/${RUN_ID}}

exec bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"
