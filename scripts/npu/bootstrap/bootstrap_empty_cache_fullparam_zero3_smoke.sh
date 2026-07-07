#!/usr/bin/env bash
set -euo pipefail

# Bootstrap an empty Ascend /cache for LLMapGen full-parameter ZeRO-3 smoke.
# This script assumes the repository has already been cloned/pulled locally.
# It creates the Python env if needed, downloads OBS dataset/model assets through
# the training launcher, and runs a short full-parameter DINOv3 + Qwen3VL-derived
# text LLM smoke test.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

echo "[bootstrap-empty-cache] repo=${REPO_ROOT}"

# Keep code/env outside /cache by default so a cache cleanup is less painful.
ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/llmapgen-npu-py311}
OBS_CACHE=${OBS_CACHE:-/cache}
WORK_ROOT=${WORK_ROOT:-/cache/llmapgen}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs}
RUN_ID=${RUN_ID:-fullparam_zero3_empty_cache_smoke_$(date -u +%Y%m%d_%H%M%S)}

mkdir -p "${OBS_CACHE}" "${WORK_ROOT}" "${OUTPUT_ROOT}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[bootstrap-empty-cache] creating env: ${ENV_DIR}"
  ENV_DIR="${ENV_DIR}" bash scripts/npu/setup/create_llmapgen_npu_py311_env.sh
else
  echo "[bootstrap-empty-cache] reusing env: ${ENV_DIR}"
fi

# shellcheck disable=SC1090
source "${ENV_DIR}/activate_llmapgen_npu.sh"

python - <<'PYCHECK'
import sys
print("[bootstrap-empty-cache] python=", sys.executable)
try:
    import torch, torch_npu
    print("[bootstrap-empty-cache] torch=", torch.__version__)
    print("[bootstrap-empty-cache] torch_npu=", getattr(torch_npu, "__version__", "unknown"))
    print("[bootstrap-empty-cache] npu_available=", torch.npu.is_available())
    print("[bootstrap-empty-cache] npu_count=", torch.npu.device_count())
except Exception as exc:
    print("[bootstrap-empty-cache] torch_npu preflight error:", repr(exc))
    raise
try:
    import deepspeed, accelerate, transformers, moxing
    print("[bootstrap-empty-cache] deepspeed=", deepspeed.__version__)
    print("[bootstrap-empty-cache] accelerate=", accelerate.__version__)
    print("[bootstrap-empty-cache] transformers=", transformers.__version__)
    print("[bootstrap-empty-cache] moxing=", getattr(moxing, "__file__", "ok"))
except Exception as exc:
    print("[bootstrap-empty-cache] dependency preflight error:", repr(exc))
    raise
PYCHECK

export OBS_CACHE
export WORK_ROOT
export RUN_ID
export INSTALL_DEPS=${INSTALL_DEPS:-False}
export ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False}
export MAX_STEPS=${MAX_STEPS:-5}
export MAX_SAMPLES=${MAX_SAMPLES:-32}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export LOGGING_STEPS=${LOGGING_STEPS:-1}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-8}
export LEARNING_RATE=${LEARNING_RATE:-5e-6}
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
export LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${OUTPUT_ROOT}/${RUN_ID}}
export OUTPUT_PATH=${OUTPUT_PATH:-${LOCAL_MODEL_SAVE_PATH}}

# Local single-node defaults. Override from the shell for a different device set.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-6060}

echo "[bootstrap-empty-cache] running full-param ZeRO-3 smoke"
echo "[bootstrap-empty-cache] RUN_ID=${RUN_ID}"
echo "[bootstrap-empty-cache] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[bootstrap-empty-cache] MAX_STEPS=${MAX_STEPS}, MAX_SAMPLES=${MAX_SAMPLES}"
echo "[bootstrap-empty-cache] DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"

bash scripts/npu/train/train_di_stage_a_lane_intersection_norm1000_dinov3_qwen3vl_derived_fullparam_private_seg_random_align_full_unfreeze_npu.sh
