#!/usr/bin/env bash
set -euo pipefail

# Evaluate the full local512-550k checkpoint on one persistent 1100-sample set.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

export DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local512/local512_550k.tar}
export DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/local512_550k.tar}
export DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/local512_extract}
export DATASET_ROOT=${DATASET_ROOT:-${DATASET_EXTRACT_ROOT}/local512}
export EVAL_SOURCE_JSONL=${EVAL_SOURCE_JSONL:-${DATASET_ROOT}/phase_a/eval.jsonl}

export FIXED_EVAL_ROOT=${FIXED_EVAL_ROOT:-/cache/jn/eval_sets/datasetv2_local512_550k_fixed1100_e300_m300_h300_vh200_seed42_v1}
export FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}
export FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}
export REBUILD_FIXED_EVAL=${REBUILD_FIXED_EVAL:-False}

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}
export CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/2aafa8a3aab146edb06805144537050e/output/ma-job-64e0ebda-df88-4285-8cc1-5f7d7699dc84/checkpoint-34376/}
export CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/local512_550k_checkpoint34376}
export RUN_LABEL=${RUN_LABEL:-local512_550k_checkpoint34376_fixed1100}

# Unlike context512_roi256, norm1000 covers the complete 512x512 image.
export PATCH_SIZE=512
export VIS_LIMIT=${VIS_LIMIT:-50}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

exec bash "${SCRIPT_DIR}/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh"
