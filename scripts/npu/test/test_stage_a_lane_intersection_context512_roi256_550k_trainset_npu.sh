#!/usr/bin/env bash
set -euo pipefail

# One-command DI entry for full Context512/ROI256 550k train-set inference.
# The implementation lives in the resumable Stage-A launcher next to this
# profile; all experiment-specific values are fixed here for reproducibility.
# DI must provide OUTPUT_URL as an obs:// URI (or the caller may set
# OBS_OUTPUT_URL/RESULT_OBS_ROOT); the delegated launcher verifies the remote
# result tree before declaring success.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

if [ -n "${MA_VJ_NAME:-}" ]; then
  DEFAULT_RUN_ID=$(printf '%s' "${MA_VJ_NAME}" | tr -c 'A-Za-z0-9_.-' '_')
else
  DEFAULT_RUN_ID=context512_roi256_550k_trainset_infer_b32_$(date -u +%Y%m%d_%H%M%S)
fi
export RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}}"
export NNODES="${NNODES:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE:-32}"

export USE_STRATIFIED_SUBSET=False
export EXPECTED_TRAIN_SAMPLES=550000
export DATASET_OBS_PATH="${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar}"
export CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/4f735c63da7a4f86829b26246143e219/output/ma-job-81341482-55b8-4c28-887b-0e4166776561/checkpoint-12504/}"
export INSTALL_INFER_DEPS="${INSTALL_INFER_DEPS:-False}"
export REQUIRE_SHAPELY="${REQUIRE_SHAPELY:-False}"
export REQUIRE_OBS_UPLOAD=True
export UPLOAD_RESULTS="${UPLOAD_RESULTS:-True}"
export RESUME_INFERENCE="${RESUME_INFERENCE:-True}"

echo "[di-entry] full 550k train-set profile"
echo "[di-entry] run=${RUN_ID} nnodes=${NNODES} nproc=${NPROC_PER_NODE} per_device_batch=${PER_DEVICE_INFER_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/test_stage_a_lane_intersection_context512_roi256_200k_trainset_npu.sh"
