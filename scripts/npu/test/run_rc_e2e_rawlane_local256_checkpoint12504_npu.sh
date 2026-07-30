#!/usr/bin/env bash
set -euo pipefail

# Full RC E2E inference for the model trained on RawLane local256. The aligned
# lane_patch_tif/*_lane.tif is already a complete BEV image with RawLane drawn
# on it, so the dataset builder crops that raster directly without re-overlay.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

export E2E_VIEW_MODE=local256
export E2E_TARGET_SIZE=256
export E2E_CONTEXT_SIZE=256
export E2E_PROMPT_PROFILE=rawlane_local256_550k_v1
export E2E_INPUT_RASTER=rawlane
export BLACK_RATIO_THRESHOLD=${BLACK_RATIO_THRESHOLD:-1.0}
export VALIDATE_RASTER_ALIGNMENT=${VALIDATE_RASTER_ALIGNMENT:-True}
export E2E_DATASET_ROOT=${E2E_DATASET_ROOT:-/cache/jn/e2e_eval/e2e_data_rawlane_local256_black1_v1}

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-12504}
export CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/29/3bf5a8001ec6433ca4ee973564c29976/output/ma-job-a782316a-32ec-4958-ae1f-44c69fdedd3f/checkpoint-12504/}
export CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_local256_checkpoint12504}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}}

export RUN_ID=${RUN_ID:-rc_e2e_rawlane_local256_checkpoint12504_$(date -u +%Y%m%d_%H%M%S)}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
export INFER_RESULT_OBS_PATH=${INFER_RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/${RUN_ID}}

exec bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"
