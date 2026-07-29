#!/usr/bin/env bash
set -euo pipefail

# Full-image local512 Dataset V2 checkpoint. Coordinates cover the complete
# 512x512 patch, so this is not the context512/center-ROI256 experiment.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

export E2E_VIEW_MODE=local512
export E2E_TARGET_SIZE=512
export E2E_CONTEXT_SIZE=512
export E2E_PROMPT_PROFILE=current

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}
export CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/2aafa8a3aab146edb06805144537050e/output/ma-job-64e0ebda-df88-4285-8cc1-5f7d7699dc84/checkpoint-34376/}
export CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/local512_550k_checkpoint34376}

export RUN_ID=${RUN_ID:-rc_e2e_local512_550k_checkpoint34376_$(date -u +%Y%m%d_%H%M%S)}
export PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}

exec bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"
