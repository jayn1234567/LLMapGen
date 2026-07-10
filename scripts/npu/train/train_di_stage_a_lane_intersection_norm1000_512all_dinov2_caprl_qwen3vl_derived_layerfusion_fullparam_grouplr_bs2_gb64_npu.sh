#!/usr/bin/env bash
set -euo pipefail

# DI entrypoint:
# - OBS raw/full 512 lane+intersection dataset
# - convert to LLMapGen trainroot with norm1000 labels
# - DINOv2/CapRL + Qwen3-VL-derived text LLM
# - layer fusion + full-parameter grouped-LR SFT
# - per-device batch 2, target global batch 64

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_norm1000_dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_npu.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base CapRL grouped-LR launcher not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

export DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/xxh/data_lane_intersection_norm_sample_512_all.tar}
export DATASET_KIND=raw
export DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_lane_intersection_norm_sample_512_all}
export TRAINROOT_DIR_NAME=${TRAINROOT_DIR_NAME:-prepared_lane_intersection_trainroot_norm1000_512_all}
export DATASET_PHASE=${DATASET_PHASE:-phase_a}
export DATASET_IMAGE_ROOT=${DATASET_IMAGE_ROOT:-images}
export DATASET_PATCH_SIZE=${DATASET_PATCH_SIZE:-512}
export DATASET_COORD_MAX=${DATASET_COORD_MAX:-1000}
export DATASET_EVAL_OUTPUT_NAME=${DATASET_EVAL_OUTPUT_NAME:-val.jsonl}
export DATASET_MEDIA_MODE=${DATASET_MEDIA_MODE:-symlink}
export FORCE_PREPARE_TRAINROOT=${FORCE_PREPARE_TRAINROOT:-true}

export RUN_ID=${RUN_ID:-dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_norm1000_512all_bs2_gb64_$(date -u +%Y%m%d_%H%M%S)}
export PREPARED_TRAINROOT=${PREPARED_TRAINROOT:-${WORK_ROOT:-/cache/llmapgen}/prepared_lane_intersection_trainroot_norm1000_512_all_${RUN_ID}}

export PER_DEVICE_TRAIN_BATCH_SIZE=2
export TARGET_GLOBAL_BATCH_SIZE=64
export GRADIENT_ACCUMULATION_STEPS=1
export CUTOFF_LEN=4096
export MODEL_MAX_LENGTH=4096
export BF16=true
export GRADIENT_CHECKPOINTING=true
export SAVE_STEPS=${SAVE_STEPS:-1000}
export LOGGING_STEPS=${LOGGING_STEPS:-1}

printf '[di-entry] reached %s
' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[di-entry] recipe=norm1000_512all_dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_bs2_gb64"
echo "[di-entry] DATASET_OBS_PATH=${DATASET_OBS_PATH}"
echo "[di-entry] DATASET_KIND=${DATASET_KIND}"
echo "[di-entry] DATASET_DIR_NAME=${DATASET_DIR_NAME}"
echo "[di-entry] DATASET_PHASE=${DATASET_PHASE}"
echo "[di-entry] DATASET_IMAGE_ROOT=${DATASET_IMAGE_ROOT}"
echo "[di-entry] DATASET_PATCH_SIZE=${DATASET_PATCH_SIZE}"
echo "[di-entry] DATASET_COORD_MAX=${DATASET_COORD_MAX}"
echo "[di-entry] PREPARED_TRAINROOT=${PREPARED_TRAINROOT}"
echo "[di-entry] PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "[di-entry] TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
echo "[di-entry] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "DI_throughput: 0.00 samples/s/npu"

exec bash "${BASE_SCRIPT}"
