#!/usr/bin/env bash
set -euo pipefail

# DI entrypoint wrapper for:
#   phase_a | lane+intersection | norm1000 | DINOv2/CapRL | Qwen3-VL-derived LLM
#   layer fusion | full-parameter training | grouped learning rates
#
# Fixed batch recipe:
#   4 nodes x 8 NPUs x per-device batch 2 x grad-accum 1 = global batch 64.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

BASE_SCRIPT="${SCRIPT_DIR}/train_di_stage_a_lane_intersection_norm1000_dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_npu.sh"

export PER_DEVICE_TRAIN_BATCH_SIZE=2
export TARGET_GLOBAL_BATCH_SIZE=64
export GRADIENT_ACCUMULATION_STEPS=1
export CUTOFF_LEN=4096
export MODEL_MAX_LENGTH=4096
export BF16=true
export GRADIENT_CHECKPOINTING=true
export SAVE_STEPS=1000
export LOGGING_STEPS=1

export RUN_ID=${RUN_ID:-dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_norm1000_bs2_gb64_$(date -u +%Y%m%d_%H%M%S)}

printf '[di-entry] reached %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[di-entry] recipe=dinov2_caprl_qwen3vl_derived_layerfusion_fullparam_grouplr_bs2_gb64"
echo "[di-entry] RUN_ID=${RUN_ID}"
echo "[di-entry] PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "[di-entry] TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
echo "[di-entry] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[di-entry] CUTOFF_LEN=${CUTOFF_LEN}"
echo "[di-entry] MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH}"
echo "[di-entry] BF16=${BF16}"
echo "[di-entry] GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "[di-entry] SAVE_STEPS=${SAVE_STEPS}"
echo "[di-entry] LOGGING_STEPS=${LOGGING_STEPS}"
echo "DI_throughput: 0.00 samples/s/npu"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "ERROR: base training script not found: ${BASE_SCRIPT}" >&2
  echo "This entrypoint only fixes the launch parameters. Please add the base CAPRL/group-LR script to the same directory." >&2
  exit 1
fi

exec bash "${BASE_SCRIPT}"
