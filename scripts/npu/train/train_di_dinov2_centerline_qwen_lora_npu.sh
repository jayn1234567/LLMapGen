#!/usr/bin/env bash
set -euo pipefail

# DI/ModelArts style launcher for LLMapGen DINOv2 + Qwen3-8B LoRA SFT.
# Defaults follow the current Ascend debug machine paths. In DI jobs, keep the
# same command and override the OBS variables below from the platform UI.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
WORK_ROOT="${WORK_ROOT:-/cache/jn}"
OBS_CACHE="${OBS_CACHE:-/cache}"

# Current user paths on the Ascend server.
TRAINROOT="${TRAINROOT:-${WORK_ROOT}/prepared_lane_intersection_trainroot}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${WORK_ROOT}/model/Qwen3-8B}"
DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-${WORK_ROOT}/model/dinov2-large}"
ASSET_DIR="${ASSET_DIR:-${WORK_ROOT}/dinov2seg_bridge/dinov2_centerline_assets_qwen3_8b}"
VISUAL_ENCODER_CHECKPOINT_PATH="${VISUAL_ENCODER_CHECKPOINT_PATH:-${ASSET_DIR}/visual_encoder_checkpoint.pt}"
BRIDGE_MODULES_STATE_PATH="${BRIDGE_MODULES_STATE_PATH:-${ASSET_DIR}/bridge_modules_state.pt}"

# OBS inputs for DI. Leave empty for local/debug runs when the paths above exist.
DATASET_OBS_PATH="${DATASET_OBS_PATH:-}"
QWEN_MODEL_OBS_PATH="${QWEN_MODEL_OBS_PATH:-}"
DINOV2_MODEL_OBS_PATH="${DINOV2_MODEL_OBS_PATH:-}"
ASSET_OBS_PATH="${ASSET_OBS_PATH:-}"

# Raw private dataset layout. Used only when TRAINROOT does not already exist.
DATASET_PHASE="${DATASET_PHASE:-phase_a}"
DATASET_DIR_NAME="${DATASET_DIR_NAME:-data_lane_intersection_norm_sample_512_33w}"
DATASET_IMAGE_ROOT="${DATASET_IMAGE_ROOT:-images}"
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}"
DATASET_INPUT_ROOT="${DATASET_INPUT_ROOT:-}"
PREPARED_TRAINROOT="${PREPARED_TRAINROOT:-${WORK_ROOT}/prepared_lane_intersection_trainroot_${RUN_ID}}"
VALIDATE_TRAINROOT="${VALIDATE_TRAINROOT:-true}"
VALIDATE_MAX_SAMPLES="${VALIDATE_MAX_SAMPLES:-200}"

# Training defaults. The task defaults to centerline + intersection from now on.
MAP_TASK="${MAP_TASK:-lane_intersection}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs/di_dinov2_bridge_qwen_lora_${MAP_TASK}_${RUN_ID}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-6}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
VISION_TRAIN_LAST_N_LAYERS="${VISION_TRAIN_LAST_N_LAYERS:-4}"

# Environment. If the cloned/created env exists, source it; otherwise use current.
ACTIVATE_ENV_SCRIPT="${ACTIVATE_ENV_SCRIPT:-/home/ma-user/.conda/envs/llmapgen-npu/activate_llmapgen_npu.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -f "${ACTIVATE_ENV_SCRIPT}" ]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_ENV_SCRIPT}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

source_if_exists() {
  if [ -f "$1" ]; then
    # shellcheck disable=SC1090
    source "$1"
    echo "[di-train] sourced $1"
  fi
}

source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source_if_exists "/usr/local/Ascend/nnal/atb/set_env.sh"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-eth0}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-64000}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_OP_BASE_FFTS_MODE_ENABLE="${HCCL_OP_BASE_FFTS_MODE_ENABLE:-FALSE}"
export WITHOUT_JIT_COMPILE="${WITHOUT_JIT_COMPILE:-1}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ -n "${MA_VJ_NAME:-}" ]; then
  WORKER_HOSTS="${VC_WORKER_HOSTS:-127.0.0.1}"
  NNODES="${NNODES:-${MA_NUM_HOSTS:-1}}"
  NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-0}}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-${MA_NUM_GPUS:-8}}"
  MASTER_ADDR="${MASTER_ADDR:-${WORKER_HOSTS%%,*}}"
else
  NNODES="${NNODES:-1}"
  NODE_RANK="${NODE_RANK:-0}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
MASTER_PORT="${MASTER_PORT:-6060}"

copy_obs_file() {
  src="$1"
  dst="$2"
  mkdir -p "$(dirname "${dst}")"
  "${PYTHON_BIN}" - "$src" "$dst" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
}

copy_obs_dir() {
  src="$1"
  dst="$2"
  mkdir -p "${dst}"
  "${PYTHON_BIN}" - "$src" "$dst" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
}

require_path_or_obs() {
  path="$1"
  obs="$2"
  label="$3"
  if [ -e "${path}" ]; then
    return 0
  fi
  if [ -z "${obs}" ]; then
    echo "[di-train] ERROR: ${label} not found: ${path}" >&2
    echo "[di-train] ERROR: set the local path or the matching OBS variable." >&2
    exit 2
  fi
}

ensure_model_dir() {
  local_path="$1"
  obs_path="$2"
  label="$3"
  if [ -f "${local_path}/config.json" ]; then
    echo "[di-train] ${label} exists: ${local_path}"
    return 0
  fi
  require_path_or_obs "${local_path}/config.json" "${obs_path}" "${label}"
  echo "[di-train] downloading ${label}: ${obs_path} -> ${local_path}"
  copy_obs_dir "${obs_path}" "${local_path}"
}

ensure_asset_dir() {
  if [ -f "${VISUAL_ENCODER_CHECKPOINT_PATH}" ] && [ -f "${BRIDGE_MODULES_STATE_PATH}" ]; then
    echo "[di-train] visual/bridge assets exist: ${ASSET_DIR}"
    return 0
  fi
  if [ -z "${ASSET_OBS_PATH}" ]; then
    echo "[di-train] ERROR: visual/bridge asset files are missing under ${ASSET_DIR}" >&2
    echo "[di-train] ERROR: set ASSET_DIR/VISUAL_ENCODER_CHECKPOINT_PATH/BRIDGE_MODULES_STATE_PATH or ASSET_OBS_PATH." >&2
    exit 2
  fi
  mkdir -p "$(dirname "${ASSET_DIR}")"
  case "${ASSET_OBS_PATH}" in
    *.tar|*.tar.gz|*.tgz)
      asset_archive="${OBS_CACHE}/dinov2_centerline_assets_${RUN_ID}.tar"
      echo "[di-train] downloading asset archive: ${ASSET_OBS_PATH} -> ${asset_archive}"
      copy_obs_file "${ASSET_OBS_PATH}" "${asset_archive}"
      tar -xf "${asset_archive}" -C "$(dirname "${ASSET_DIR}")"
      ;;
    *)
      echo "[di-train] downloading asset dir: ${ASSET_OBS_PATH} -> ${ASSET_DIR}"
      copy_obs_dir "${ASSET_OBS_PATH}" "${ASSET_DIR}"
      ;;
  esac
  if [ ! -f "${VISUAL_ENCODER_CHECKPOINT_PATH}" ] || [ ! -f "${BRIDGE_MODULES_STATE_PATH}" ]; then
    echo "[di-train] ERROR: asset files are still missing under ${ASSET_DIR}" >&2
    exit 2
  fi
}

extract_dataset_if_needed() {
  if [ -n "${DATASET_INPUT_ROOT}" ] && [ -d "${DATASET_INPUT_ROOT}" ]; then
    echo "[di-train] raw dataset exists: ${DATASET_INPUT_ROOT}"
    return 0
  fi
  if [ -f "${TRAINROOT}/train.jsonl" ]; then
    echo "[di-train] prepared trainroot exists: ${TRAINROOT}"
    return 0
  fi
  if [ -z "${DATASET_OBS_PATH}" ]; then
    echo "[di-train] ERROR: TRAINROOT not found and DATASET_INPUT_ROOT is empty." >&2
    echo "[di-train] ERROR: set TRAINROOT, DATASET_INPUT_ROOT, or DATASET_OBS_PATH." >&2
    exit 2
  fi
  mkdir -p "${DATASET_EXTRACT_ROOT}"
  case "${DATASET_OBS_PATH}" in
    *.zip)
      dataset_archive="${OBS_CACHE}/dataset_${RUN_ID}.zip"
      echo "[di-train] downloading dataset zip: ${DATASET_OBS_PATH} -> ${dataset_archive}"
      copy_obs_file "${DATASET_OBS_PATH}" "${dataset_archive}"
      unzip -q "${dataset_archive}" -d "${DATASET_EXTRACT_ROOT}"
      ;;
    *.tar|*.tar.gz|*.tgz)
      dataset_archive="${OBS_CACHE}/dataset_${RUN_ID}.tar"
      echo "[di-train] downloading dataset tar: ${DATASET_OBS_PATH} -> ${dataset_archive}"
      copy_obs_file "${DATASET_OBS_PATH}" "${dataset_archive}"
      tar -xf "${dataset_archive}" -C "${DATASET_EXTRACT_ROOT}"
      ;;
    *)
      target_dir="${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}"
      echo "[di-train] downloading dataset dir: ${DATASET_OBS_PATH} -> ${target_dir}"
      copy_obs_dir "${DATASET_OBS_PATH}" "${target_dir}"
      ;;
  esac
}

resolve_dataset_input_root() {
  if [ -n "${DATASET_INPUT_ROOT}" ] && [ -d "${DATASET_INPUT_ROOT}" ]; then
    printf '%s\n' "${DATASET_INPUT_ROOT}"
    return 0
  fi
  if [ -d "${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}" ]; then
    printf '%s\n' "${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}"
    return 0
  fi
  if [ -f "${DATASET_EXTRACT_ROOT}/dataset_info.json" ] || [ -d "${DATASET_EXTRACT_ROOT}/${DATASET_PHASE}" ]; then
    printf '%s\n' "${DATASET_EXTRACT_ROOT}"
    return 0
  fi
  first_dir="$(find "${DATASET_EXTRACT_ROOT}" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  if [ -n "${first_dir}" ]; then
    printf '%s\n' "${first_dir}"
    return 0
  fi
  echo "[di-train] ERROR: cannot resolve dataset root under ${DATASET_EXTRACT_ROOT}" >&2
  exit 2
}

prepare_trainroot_if_needed() {
  if [ -f "${TRAINROOT}/train.jsonl" ]; then
    return 0
  fi
  extract_dataset_if_needed
  DATASET_INPUT_ROOT="$(resolve_dataset_input_root)"
  echo "[di-train] preparing trainroot from ${DATASET_INPUT_ROOT} -> ${PREPARED_TRAINROOT}"
  "${PYTHON_BIN}" scripts/tools/prepare_di_qa_trainroot.py \
    --input-root "${DATASET_INPUT_ROOT}" \
    --phase "${DATASET_PHASE}" \
    --image-root "${DATASET_IMAGE_ROOT}" \
    --task "${MAP_TASK}" \
    --output-root "${PREPARED_TRAINROOT}"
  TRAINROOT="${PREPARED_TRAINROOT}"
}

upload_output_if_possible() {
  if [ "${NODE_RANK}" != "0" ]; then
    echo "[di-train] node_rank=${NODE_RANK}; skip output upload."
    return 0
  fi
  if [ -z "${OUTPUT_URL:-}" ]; then
    echo "[di-train] OUTPUT_URL is empty; local output kept at ${OUTPUT_DIR}"
    return 0
  fi
  cloud_output="${OUTPUT_URL%/}/${RUN_ID}"
  echo "[di-train] uploading output: ${OUTPUT_DIR} -> ${cloud_output}"
  copy_obs_dir "${OUTPUT_DIR}" "${cloud_output}"
  echo "[di-train] cloud output: ${cloud_output}"
}

mkdir -p "${WORK_ROOT}" "${OBS_CACHE}" "$(dirname "${OUTPUT_DIR}")"

ensure_model_dir "${MODEL_NAME_OR_PATH}" "${QWEN_MODEL_OBS_PATH}" "Qwen model"
ensure_model_dir "${DINOV2_MODEL_NAME_OR_PATH}" "${DINOV2_MODEL_OBS_PATH}" "DINOv2 model"
ensure_asset_dir
prepare_trainroot_if_needed

if [ "${VALIDATE_TRAINROOT}" = "true" ]; then
  "${PYTHON_BIN}" scripts/tools/validate_di_trainroot.py \
    --trainroot "${TRAINROOT}" \
    --task "${MAP_TASK}" \
    --max-samples "${VALIDATE_MAX_SAMPLES}"
fi

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
if [ -z "${GRADIENT_ACCUMULATION_STEPS}" ]; then
  MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
  GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
  if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi

echo "============================================================"
echo "LLMapGen DI DINOv2 + Qwen3 LoRA SFT"
echo "run_id:       ${RUN_ID}"
echo "task:         ${MAP_TASK}"
echo "trainroot:    ${TRAINROOT}"
echo "qwen:         ${MODEL_NAME_OR_PATH}"
echo "dinov2:       ${DINOV2_MODEL_NAME_OR_PATH}"
echo "visual ckpt:  ${VISUAL_ENCODER_CHECKPOINT_PATH}"
echo "bridge state: ${BRIDGE_MODULES_STATE_PATH}"
echo "output:       ${OUTPUT_DIR}"
echo "nnodes:       ${NNODES}"
echo "node_rank:    ${NODE_RANK}"
echo "nproc/node:   ${NPROC_PER_NODE}"
echo "master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "global batch: target=${TARGET_GLOBAL_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "vision train: freeze=${FREEZE_VISION_ENCODER}, last_n=${VISION_TRAIN_LAST_N_LAYERS}"
echo "============================================================"

export TRAINROOT
export MODEL_NAME_OR_PATH
export DINOV2_MODEL_NAME_OR_PATH
export VISUAL_ENCODER_CHECKPOINT_PATH
export BRIDGE_MODULES_STATE_PATH
export OUTPUT_DIR
export MAP_TASK
export PYTHON_BIN
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export USE_TORCHRUN=true
export KEEP_DISTRIBUTED_ENV=true
export NUM_TRAIN_EPOCHS MAX_STEPS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE SAVE_STEPS SAVE_TOTAL_LIMIT LOGGING_STEPS DATALOADER_NUM_WORKERS
export BF16 GRADIENT_CHECKPOINTING LOCAL_FILES_ONLY
export FREEZE_LANGUAGE_MODEL FREEZE_VISION_ENCODER VISION_TRAIN_LAST_N_LAYERS

bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh

upload_output_if_possible
