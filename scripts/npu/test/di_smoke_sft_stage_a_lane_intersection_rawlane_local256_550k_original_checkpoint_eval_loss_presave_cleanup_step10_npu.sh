#!/usr/bin/env bash
set -euo pipefail

# Extend the proven original-checkpoint smoke with eval_loss. At steps 10 and
# 20 the trainer evaluates a deterministic 256-sample subset in loss-only mode,
# then synchronizes/cleans NPU cache before the ordinary consolidated save.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
cd "${REPO_ROOT}"

echo "[di-original-save-eval-step10-smoke] reached eval-loss plus original checkpoint experiment"
echo "[di-original-save-eval-step10-smoke] step10: eval_loss(256 samples, batch1, loss-only) -> NPU cleanup -> ordinary checkpoint"
echo "[di-original-save-eval-step10-smoke] zero_shards=False cpu_merge=False best_eval_copy=False"
echo "DI_throughput: 0.00 samples/s/npu"

export MAX_STEPS=20
export NUM_EPOCHS=100
export SAVE_STEPS=10
export SAVE_TOTAL_LIMIT=2
export LOGGING_STEPS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export TARGET_GLOBAL_BATCH_SIZE=128
export ENABLE_EVAL=True
export EVAL_STEPS=10
export EVAL_SAMPLE_LIMIT=${EVAL_SAMPLE_LIMIT:-256}
export PER_DEVICE_EVAL_BATCH_SIZE=1
export DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-2000}
export DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}
export SAVE_BEST_TRAIN_LOSS=False
export SAVE_BEST_EVAL_LOSS=False
export SAVE_BEST_INFER_INDEX=False
export SWANLAB_ENABLE=False
export CHECKPOINT_SAVE_MODE=original
export DEEPSPEED_CONFIG=scripts/deepspeed_zero3.json
export MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True
export MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=False

exec bash "${FORMAL_SCRIPT}"
