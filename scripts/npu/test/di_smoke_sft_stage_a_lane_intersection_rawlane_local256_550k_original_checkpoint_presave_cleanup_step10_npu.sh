#!/usr/bin/env bash
set -euo pipefail

# Test the original non-sharded checkpoint path with one isolated change:
# synchronize the NPU and release unused cached allocations immediately before
# the ordinary ZeRO-3 gather/save operation.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
cd "${REPO_ROOT}"

echo "[di-original-save-step10-smoke] reached original checkpoint save experiment"
echo "[di-original-save-step10-smoke] save sequence: npu.synchronize -> gc.collect -> npu.empty_cache -> ordinary ZeRO-3 gather/save"
echo "[di-original-save-step10-smoke] zero_shards=False cpu_merge=False"
echo "[di-original-save-step10-smoke] max_steps=20 save_steps=10 per_device_batch=4 global_batch=128"
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
export CHECKPOINT_SAVE_MODE=original
export DEEPSPEED_CONFIG=scripts/deepspeed_zero3.json
export ENABLE_EVAL=False
export MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True
export MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=False

exec bash "${FORMAL_SCRIPT}"
