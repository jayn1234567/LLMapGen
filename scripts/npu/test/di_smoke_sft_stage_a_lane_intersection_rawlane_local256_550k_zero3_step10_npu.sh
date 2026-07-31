#!/usr/bin/env bash
set -euo pipefail

# DI checkpoint-path verification for the Raw-Lane local256 550k recipe.
# The formal launcher detects DI topology and platform output paths. This wrapper
# only shortens the run and forces two early ZeRO-3 sharded checkpoints.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
cd "${REPO_ROOT}"

echo "[di-step10-smoke] reached Raw-Lane 550k ZeRO-3 checkpoint verification entry"
echo "[di-step10-smoke] max_steps=20 save_steps=10 per_device_batch=4 global_batch=128"
echo "DI_throughput: 0.00 samples/s/npu"

export MAX_STEPS=20
export NUM_EPOCHS=100
export SAVE_STEPS=10
export SAVE_TOTAL_LIMIT=2
export LOGGING_STEPS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export TARGET_GLOBAL_BATCH_SIZE=128
export DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-2000}
export DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}
export SAVE_BEST_TRAIN_LOSS=False
export SAVE_BEST_EVAL_LOSS=False
export SAVE_BEST_INFER_INDEX=False
export SWANLAB_ENABLE=False
export MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True
export MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=True

exec bash "${FORMAL_SCRIPT}"
