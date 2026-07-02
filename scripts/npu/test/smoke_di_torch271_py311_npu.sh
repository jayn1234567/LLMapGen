#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PY311_ENV_DIR="${PY311_ENV_DIR:-/home/ma-user/.conda/envs/llmapgen-npu-py311}"
PY311_ACTIVATE_SCRIPT="${PY311_ACTIVATE_SCRIPT:-${PY311_ENV_DIR}/activate_llmapgen_npu.sh}"
if [ ! -f "${PY311_ACTIVATE_SCRIPT}" ]; then
  echo "[npu-py311-smoke] missing py311 env activation script: ${PY311_ACTIVATE_SCRIPT}" >&2
  echo "[npu-py311-smoke] create it first: bash scripts/npu/setup/create_llmapgen_npu_py311_env.sh" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${PY311_ACTIVATE_SCRIPT}"

python - <<'PY'
import sys

print(f"[npu-py311-smoke] python={sys.executable} version={sys.version.split()[0]}", flush=True)
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This smoke test needs Python 3.11 for the torch_npu cp311 wheel.")
PY

export RUN_ID="${RUN_ID:-torch271_py311_smoke_$(date +%Y%m%d_%H%M%S)}"
export MA_VJ_NAME="${MA_VJ_NAME:-local_di_torch271_py311_smoke}"
export MA_NUM_HOSTS="${MA_NUM_HOSTS:-1}"
export VC_TASK_INDEX="${VC_TASK_INDEX:-0}"
export MA_NUM_GPUS="${MA_NUM_GPUS:-8}"
export VC_WORKER_HOSTS="${VC_WORKER_HOSTS:-127.0.0.1}"
export OUTPUT_URL="${OUTPUT_URL:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export TRAINROOT="${TRAINROOT:-/cache/jn/prepared_lane_intersection_trainroot}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/cache/jn/model/Qwen3-8B}"
export DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-/cache/jn/model/dinov2-large}"
export ASSET_DIR="${ASSET_DIR:-/cache/jn/model/dinov2seg_bridge/dinov2_centerline_assets_qwen3_8b}"
export VISUAL_ENCODER_CHECKPOINT_PATH="${VISUAL_ENCODER_CHECKPOINT_PATH:-${ASSET_DIR}/visual_encoder_checkpoint.pt}"
export BRIDGE_MODULES_STATE_PATH="${BRIDGE_MODULES_STATE_PATH:-${ASSET_DIR}/bridge_modules_state.pt}"

export DATASET_KIND="${DATASET_KIND:-prepared}"
export INSTALL_TORCH_NPU="${INSTALL_TORCH_NPU:-true}"
export NPU_PREFLIGHT="${NPU_PREFLIGHT:-true}"
export MAP_TASK="${MAP_TASK:-lane_intersection}"
export MAX_STEPS="${MAX_STEPS:-2}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-8}"
export BF16="${BF16:-true}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

bash scripts/npu/train/train_di_dinov2_centerline_qwen_lora_npu.sh
