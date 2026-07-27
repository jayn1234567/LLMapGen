#!/usr/bin/env bash
set -euo pipefail

# Evaluate the context512_roi256 checkpoint on one persistent 1100-sample set.
# Geometry metrics stay in the centered 256x256 ROI coordinate frame; the
# visualizer projects those local coordinates back onto the 512x512 image.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

export DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar}
export DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/context512_roi256_550k.tar}
export DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/context512_roi256_extract}
export DATASET_ROOT=${DATASET_ROOT:-${DATASET_EXTRACT_ROOT}/context512_roi256}
export EVAL_SOURCE_JSONL=${EVAL_SOURCE_JSONL:-${DATASET_ROOT}/phase_a/eval.jsonl}

export FIXED_EVAL_ROOT=${FIXED_EVAL_ROOT:-/cache/jn/eval_sets/datasetv2_context512_roi256_550k_fixed1100_e300_m300_h300_vh200_seed42_v1}
export FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}
export FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}
export REBUILD_FIXED_EVAL=${REBUILD_FIXED_EVAL:-False}

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-12504}
export CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/4f735c63da7a4f86829b26246143e219/output/ma-job-81341482-55b8-4c28-887b-0e4166776561/checkpoint-12504/}
export CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/context512_roi256_checkpoint12504}
export RUN_LABEL=${RUN_LABEL:-context512_roi256_checkpoint12504_fixed1100}

# The image canvas is 512x512, while generated norm1000 coordinates describe
# only the centered 256x256 target ROI.
export PATCH_SIZE=256
export VIS_LIMIT=${VIS_LIMIT:-50}
export PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-2}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

exec bash "${SCRIPT_DIR}/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh"
