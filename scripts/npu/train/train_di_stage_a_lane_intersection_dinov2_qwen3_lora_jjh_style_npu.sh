#!/usr/bin/env bash

# ============================================================
# DI NPU SFT training
# Fixed recipe: prepared trainroot | lane + intersection | DINOv2 + Qwen3-8B | LoRA
# This file follows the jiangjihua DI launcher style: one self-contained script
# declares cloud paths, installs runtime deps, downloads OBS assets, launches
# torchrun/HCCL, then stages rank-0 outputs to OUTPUT_URL when provided.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

set -euo pipefail

echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "Recipe: prepared_trainroot | lane_intersection | dinov2 + qwen3-8b lora"

# ====================== cloud paths ======================
# OUTPUT_URL is injected by the DI/ModelArts training platform.
CLUSTER_SAVE=${OUTPUT_URL:-}
OSB_SHARE_PATH="${CLUSTER_SAVE}"
echo "System defined obs share path: ${OSB_SHARE_PATH:-<empty>}"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}
WORK_ROOT=${WORK_ROOT:-/cache/llmapgen}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.zip}
QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/checkpoint/Qwen3-8B}
DINOV2_MODEL_OBS_PATH=${DINOV2_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
ASSET_OBS_PATH=${ASSET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_centerline_assets_qwen3_8b}

DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
TRAINROOT_DIR_NAME=${TRAINROOT_DIR_NAME:-prepared_lane_intersection_trainroot}
TRAINROOT=${TRAINROOT:-${DATASET_EXTRACT_ROOT}/${TRAINROOT_DIR_NAME}}

QWEN_PATH=${QWEN_PATH:-${WORK_ROOT}/model/Qwen3-8B}
DINOV2_PATH=${DINOV2_PATH:-${WORK_ROOT}/model/dinov2-large}
ASSET_DIR=${ASSET_DIR:-${WORK_ROOT}/model/dinov2_centerline_assets_qwen3_8b}
VISUAL_ENCODER_CHECKPOINT_PATH=${VISUAL_ENCODER_CHECKPOINT_PATH:-${ASSET_DIR}/visual_encoder_checkpoint.pt}
BRIDGE_MODULES_STATE_PATH=${BRIDGE_MODULES_STATE_PATH:-${ASSET_DIR}/bridge_modules_state.pt}

CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH:+${OSB_SHARE_PATH%/}/${RUN_ID}}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}
OUTPUT_PATH=${OUTPUT_PATH:-${LOCAL_MODEL_SAVE_PATH}}

# ====================== training params ======================
MAP_TASK=${MAP_TASK:-lane_intersection}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-32}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-6}
MAX_STEPS=${MAX_STEPS:--1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LOGGING_STEPS=${LOGGING_STEPS:-10}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
CUTOFF_LEN=${CUTOFF_LEN:-7168}
IMAGE_SIZE=${IMAGE_SIZE:-512}
ENCODER_INPUT_PAD_SIZE=${ENCODER_INPUT_PAD_SIZE:-518}
FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL:-false}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}
VISION_TRAIN_LAST_N_LAYERS=${VISION_TRAIN_LAST_N_LAYERS:-4}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# ====================== Ascend environment ======================
source_if_exists() {
  if [ -f "$1" ]; then
    local nounset_was_on=0
    case "$-" in
      *u*)
        nounset_was_on=1
        set +u
        ;;
    esac
    export ZSH_VERSION="${ZSH_VERSION:-}"
    # shellcheck disable=SC1090
    source "$1"
    if [ "${nounset_was_on}" = "1" ]; then
      set -u
    fi
    echo "[di-train] sourced $1"
  fi
}

export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}
source_if_exists /usr/local/Ascend/ascend-toolkit/set_env.sh
source_if_exists /usr/local/Ascend/nnal/atb/set_env.sh
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export INF_NAN_MODE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export WITHOUT_JIT_COMPILE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE
export COMBINED_ENABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Dependency installation for managed NPU images. Set INSTALL_DEPS=False on prebuilt images.
INSTALL_DEPS=${INSTALL_DEPS:-True}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}
MOXING_WHL_OBS_PATH=${MOXING_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}
TORCH_NPU_WHL_OBS_PATH=${TORCH_NPU_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
TORCH_NPU_WHL_LOCAL_PATH=${TORCH_NPU_WHL_LOCAL_PATH:-/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}

if [[ "${ENABLE_MOXING_UPGRADE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  USE_MEMARTS=0 python -c "import moxing as mox; mox.file.copy('${MOXING_WHL_OBS_PATH}', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
  pip uninstall moxing-framework -y || true
  pip cache purge || true
  pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
  export MOX_PROFILE=1
  export MOX_RECORD_OBS=1
fi

if [[ "${INSTALL_DEPS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  pip install torch==2.7.1 torch_npu==2.7.1rc1
  python -c "import moxing as mox; mox.file.copy('${TORCH_NPU_WHL_OBS_PATH}', '${TORCH_NPU_WHL_LOCAL_PATH}')"
  pip install --force-reinstall "${TORCH_NPU_WHL_LOCAL_PATH}"
  pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "transformers==4.56.2" "tokenizers>=0.22.0,<0.23.0"
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26,<2.0' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless==4.11.0.86'
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15
  pip install 'numpy>=1.26,<2.0' 'opencv-python-headless==4.11.0.86'
fi

python - <<'PY'
import os
import sys

print(f"[di-preflight] python={sys.executable} version={sys.version.split()[0]}", flush=True)
for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
    print(f"[di-preflight] env {name}={os.environ.get(name, '<empty>')}", flush=True)
import torch
import torch_npu

print(f"[di-preflight] torch={torch.__version__}", flush=True)
print(f"[di-preflight] torch_npu={getattr(torch_npu, '__version__', 'unknown')}", flush=True)
print(f"[di-preflight] npu_available={torch.npu.is_available()}", flush=True)
print(f"[di-preflight] npu_count={torch.npu.device_count()}", flush=True)
if not torch.npu.is_available():
    raise SystemExit("NPU is not available after torch/torch_npu installation.")
PY

# ====================== distributed setup ======================
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi
MASTER_PORT=${MASTER_PORT:-6060}
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-llmapgen_dinov2_qwen3_lora_${RUN_ID}}

mkdir -p "${OUTPUT_PATH}" "${WORK_ROOT}/model" "${DATASET_EXTRACT_ROOT}"

# ====================== downloads ======================
echo "[di-download] dataset: ${DATASET_OBS_PATH} -> ${DATASET_ZIP_PATH}"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

echo "[di-download] qwen: ${QWEN_MODEL_OBS_PATH} -> ${QWEN_PATH}"
python -c "import moxing as mox; mox.file.copy_parallel('${QWEN_MODEL_OBS_PATH}', '${QWEN_PATH}')"
echo "[di-download] dinov2: ${DINOV2_MODEL_OBS_PATH} -> ${DINOV2_PATH}"
python -c "import moxing as mox; mox.file.copy_parallel('${DINOV2_MODEL_OBS_PATH}', '${DINOV2_PATH}')"
echo "[di-download] bridge assets: ${ASSET_OBS_PATH} -> ${ASSET_DIR}"
python -c "import moxing as mox; mox.file.copy_parallel('${ASSET_OBS_PATH}', '${ASSET_DIR}')"

if [ ! -f "${TRAINROOT}/train.jsonl" ]; then
  FOUND_TRAINROOT=$(find "${DATASET_EXTRACT_ROOT}" -maxdepth 6 -type f -name train.jsonl -printf '%h\n' 2>/dev/null | sort | head -n 1 || true)
  if [ -n "${FOUND_TRAINROOT}" ]; then
    TRAINROOT="${FOUND_TRAINROOT}"
  fi
fi

for path in \
  "${TRAINROOT}/train.jsonl" \
  "${QWEN_PATH}/config.json" \
  "${DINOV2_PATH}/config.json" \
  "${VISUAL_ENCODER_CHECKPOINT_PATH}" \
  "${BRIDGE_MODULES_STATE_PATH}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    echo "Dataset extract summary:"
    find "${DATASET_EXTRACT_ROOT}" -maxdepth 4 -mindepth 1 -printf '  %P\n' 2>/dev/null | head -n 80 || true
    exit 1
  fi
done

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))}
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
  GRADIENT_ACCUMULATION_STEPS=1
fi

FREEZE_LANGUAGE_ARGS=(--no-freeze-language-model)
if [[ "${FREEZE_LANGUAGE_MODEL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  FREEZE_LANGUAGE_ARGS=(--freeze-language-model)
fi
FREEZE_VISION_ARGS=(--no-freeze-vision-encoder)
if [[ "${FREEZE_VISION_ENCODER}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  FREEZE_VISION_ARGS=(--freeze-vision-encoder --vision-train-last-n-layers "${VISION_TRAIN_LAST_N_LAYERS}")
fi

echo "============================================================"
echo "Run id:       ${RUN_ID}"
echo "Trainroot:    ${TRAINROOT}"
echo "Qwen:         ${QWEN_PATH}"
echo "DINOv2:       ${DINOV2_PATH}"
echo "Visual ckpt:  ${VISUAL_ENCODER_CHECKPOINT_PATH}"
echo "Bridge state: ${BRIDGE_MODULES_STATE_PATH}"
echo "Output:       ${OUTPUT_PATH}"
echo "Cloud output: ${CLOUD_OUTPUT_PATH:-<empty>}"
echo "NNODES/NPROC: ${NNODES}/${NPROC_PER_NODE}, rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Global batch: target=${TARGET_GLOBAL_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "Vision train: freeze=${FREEZE_VISION_ENCODER}, last_n=${VISION_TRAIN_LAST_N_LAYERS}"
echo "============================================================"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_dinov2_centerline.py \
  --trainroot "${TRAINROOT}" \
  --model-name-or-path "${QWEN_PATH}" \
  --dinov2-model-name-or-path "${DINOV2_PATH}" \
  --visual-encoder-checkpoint-path "${VISUAL_ENCODER_CHECKPOINT_PATH}" \
  --bridge-modules-state-path "${BRIDGE_MODULES_STATE_PATH}" \
  --output-dir "${OUTPUT_PATH}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --logging-steps "${LOGGING_STEPS}" \
  --save-strategy steps \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --cutoff-len "${CUTOFF_LEN}" \
  --image-size "${IMAGE_SIZE}" \
  --encoder-input-pad-size "${ENCODER_INPUT_PAD_SIZE}" \
  --map-task "${MAP_TASK}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --device-backend npu \
  --ddp-backend hccl \
  --local-files-only \
  --bf16 \
  --gradient-checkpointing \
  "${FREEZE_LANGUAGE_ARGS[@]}" \
  "${FREEZE_VISION_ARGS[@]}"

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "Training failed with exit code ${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  if [ -n "${CLOUD_OUTPUT_PATH:-}" ]; then
    if [ -e "${CLOUD_OUTPUT_PATH}" ]; then
      echo "ERROR: cloud output path already exists, refusing to overwrite: ${CLOUD_OUTPUT_PATH}"
      exit 1
    fi
    echo "Moving rank0 local output to cloud output: ${OUTPUT_PATH} -> ${CLOUD_OUTPUT_PATH}"
    mv "${OUTPUT_PATH}" "${CLOUD_OUTPUT_PATH}"
    echo "Final cloud output path: ${CLOUD_OUTPUT_PATH}"
  else
    echo "OUTPUT_URL is empty; local output kept at ${OUTPUT_PATH}"
  fi
else
  echo "Non-master node ${NODE_RANK}: skip cloud output move."
fi
