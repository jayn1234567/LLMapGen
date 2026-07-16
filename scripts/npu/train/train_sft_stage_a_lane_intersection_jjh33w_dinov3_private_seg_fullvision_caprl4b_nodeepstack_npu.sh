#!/usr/bin/env bash

# ============================================================
# NPU SFT training
# Jiangjihua v9 original 33w/256 dataset + CapRL-Qwen3VL-4B + no DeepStack,
# replacing the reference DINOv2 tower with HF DINOv3-L initialized from the
# private-data scripted DINOv3 segmentation checkpoint.
# ============================================================

set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_SCRIPT="${SCRIPT_DIR}/train_sft_stage_a_lane_intersection_jjh33w_latest_private_dinov2_last2_caprl4b_nodeepstack_npu.sh"

echo "[di-entry] reached Jiangjihua original-33w DINOv3-private-seg CapRL SFT launcher"
echo "[di-entry] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"
echo "DI_throughput: 0.00 samples/s/npu"

export VISION_BACKBONE=${VISION_BACKBONE:-dinov3_private_seg}
export VISION_TOWER_NAME=${VISION_TOWER_NAME:-facebook_dinov3-vitl16-pretrain-lvd1689m_private_seg_lora}
export MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov3_private_seg}
export INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
export VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
export VISION_TOWER_CHECKPOINT_OBS_PATH=${VISION_TOWER_CHECKPOINT_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov3_lora.pt}
export VISION_TOWER_CHECKPOINT=${VISION_TOWER_CHECKPOINT:-${OBS_CACHE:-/cache}/checkpoints/dinov3_lora.pt}
export DINOV3_CHECKPOINT_ALLOW_ZERO_SHAPE=${DINOV3_CHECKPOINT_ALLOW_ZERO_SHAPE:-false}

# Keep Jiangjihua's best original 33w dataset and CapRL/Qwen settings from the base script.
# DINOv3 is fully trainable in this experiment, but with a lower LR than the LLM/projector.
export MM_VISION_UNFREEZE_LAST_N_BLOCKS=${MM_VISION_UNFREEZE_LAST_N_BLOCKS:--1}
export MM_VISION_SELECT_LAYER=${MM_VISION_SELECT_LAYER:-24}
export MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-5e-6}
export MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
export LR=${LR:-2e-5}
export NUM_EPOCHS=${NUM_EPOCHS:-8}
export TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}

if [ -n "${MA_VJ_NAME:-}" ]; then
  DEFAULT_DINOV3_RUN_ID=$(printf '%s' "${MA_VJ_NAME}" | tr -c 'A-Za-z0-9_.-' '_')
else
  DEFAULT_DINOV3_RUN_ID=jjh33w_dinov3_private_seg_fullvision_caprl4b_$(date -u +%Y%m%d_%H%M%S)
fi
export RUN_ID=${RUN_ID:-${DEFAULT_DINOV3_RUN_ID}}
export RDZV_ID=${RDZV_ID:-sft_phase_a_lane_intersection_dinov3_${RUN_ID}}
export SWANLAB_GROUP=${SWANLAB_GROUP:-sft_phase_a_lane_intersection_jjh33w_dinov3_private_seg_caprl4b}
export SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_jjh33w_dinov3_private_seg_fullvision_caprl4b_ep8}
export SWANLAB_TAGS=${SWANLAB_TAGS:-sft,phase_a,lane_intersection,jjh33w,dinov3,private_seg,fullvision,caprl4b,nodeepstack}

echo "VISION_BACKBONE=${VISION_BACKBONE}"
echo "MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE}"
echo "DINOV3_CHECKPOINT_ALLOW_ZERO_SHAPE=${DINOV3_CHECKPOINT_ALLOW_ZERO_SHAPE}"

exec bash "${BASE_SCRIPT}"
