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
DI_THROUGHPUT_VALUE=${DI_THROUGHPUT_VALUE:-1.00}
echo "DI_throughput: ${DI_THROUGHPUT_VALUE} samples/s/npu"

# ====================== cloud paths ======================
# OUTPUT_URL is injected by the DI/ModelArts training platform.
CLUSTER_SAVE=${OUTPUT_URL:-}
OSB_SHARE_PATH="${CLUSTER_SAVE}"
echo "System defined obs share path: ${OSB_SHARE_PATH:-<empty>}"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}
WORK_ROOT=${WORK_ROOT:-/cache/llmapgen}
SKIP_OBS_DOWNLOADS=${SKIP_OBS_DOWNLOADS:-false}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.zip}
QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/checkpoint/Qwen3-8B}
DINOV2_MODEL_OBS_PATH=${DINOV2_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
VISION_MODEL_OBS_PATH=${VISION_MODEL_OBS_PATH:-${DINOV2_MODEL_OBS_PATH}}
VISION_MODEL_FAMILY=${VISION_MODEL_FAMILY:-dinov2}
ASSET_OBS_PATH=${ASSET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_centerline_assets_qwen3_8b}

DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
TRAINROOT_DIR_NAME=${TRAINROOT_DIR_NAME:-prepared_lane_intersection_trainroot}
TRAINROOT=${TRAINROOT:-${DATASET_EXTRACT_ROOT}/${TRAINROOT_DIR_NAME}}

QWEN_PATH=${QWEN_PATH:-${WORK_ROOT}/model/Qwen3-8B}
EXTRACT_QWEN3VL_TEXT_LLM=${EXTRACT_QWEN3VL_TEXT_LLM:-false}
QWEN_EXTRACTED_TEXT_PATH=${QWEN_EXTRACTED_TEXT_PATH:-${WORK_ROOT}/model/Qwen3-VL-8B-Instruct_llm_extracted}
DINOV2_PATH=${DINOV2_PATH:-${WORK_ROOT}/model/dinov2-large}
VISION_PATH=${VISION_PATH:-${DINOV2_PATH}}
ASSET_DIR=${ASSET_DIR:-${WORK_ROOT}/model/dinov2_centerline_assets_qwen3_8b}
VISUAL_ENCODER_CHECKPOINT_PATH=${VISUAL_ENCODER_CHECKPOINT_PATH:-${ASSET_DIR}/visual_encoder_checkpoint.pt}
BRIDGE_MODULES_STATE_PATH=${BRIDGE_MODULES_STATE_PATH:-${ASSET_DIR}/bridge_modules_state.pt}
USE_PRETRAINED_VISUAL_BRIDGE=${USE_PRETRAINED_VISUAL_BRIDGE:-true}
USE_VISUAL_ENCODER_CHECKPOINT=${USE_VISUAL_ENCODER_CHECKPOINT:-${USE_PRETRAINED_VISUAL_BRIDGE}}
VISUAL_ENCODER_CHECKPOINT_OBS_PATH=${VISUAL_ENCODER_CHECKPOINT_OBS_PATH:-}

CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH:+${OSB_SHARE_PATH%/}/${RUN_ID}}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR:-${LOCAL_MODEL_SAVE_PATH}}}

# ====================== training params ======================
MAP_TASK=${MAP_TASK:-lane_intersection}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}
USER_PROMPT=${USER_PROMPT:-}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-32}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-6}
MAX_STEPS=${MAX_STEPS:--1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LANGUAGE_MODEL_LR=${LANGUAGE_MODEL_LR:-}
ALIGNMENT_LR=${ALIGNMENT_LR:-}
VISION_ENCODER_LR=${VISION_ENCODER_LR:-}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
OPTIM=${OPTIM:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LOGGING_STEPS=${LOGGING_STEPS:-10}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
CUTOFF_LEN=${CUTOFF_LEN:-7168}
IMAGE_SIZE=${IMAGE_SIZE:-512}
ENCODER_INPUT_PAD_SIZE=${ENCODER_INPUT_PAD_SIZE:-518}
VISION_PATCH_SIZE=${VISION_PATCH_SIZE:-14}
VISION_NUM_PREFIX_TOKENS=${VISION_NUM_PREFIX_TOKENS:--1}
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}
FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL:-false}
NO_LORA=${NO_LORA:-false}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}
VISION_TRAIN_LAST_N_LAYERS=${VISION_TRAIN_LAST_N_LAYERS:-4}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
VISUAL_TOKEN_COMPRESSOR=${VISUAL_TOKEN_COMPRESSOR:-none}
VISUAL_TOKEN_COMPRESSOR_GRID_SIZE=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE:-0}
VISUAL_TOKEN_COMPRESSOR_HIDDEN_DIM=${VISUAL_TOKEN_COMPRESSOR_HIDDEN_DIM:-512}
VISUAL_TOKEN_COMPRESSOR_DEPTH=${VISUAL_TOKEN_COMPRESSOR_DEPTH:-2}
VISUAL_TOKEN_COMPRESSOR_DROPOUT=${VISUAL_TOKEN_COMPRESSOR_DROPOUT:-0.0}

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
source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists /usr/local/Ascend/ascend-toolkit/set_env.sh
source_if_exists /usr/local/Ascend/nnal/atb/set_env.sh
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
if [ -n "${MA_VJ_NAME:-}" ]; then
  # ModelArts/DI injects ASCEND_VISIBLE_DEVICES and NPU_VISIBLE_DEVICES. Do not
  # synthesize ASCEND_RT_VISIBLE_DEVICES from the DI physical device order, since
  # torch_npu can then report zero visible devices on managed workers.
  if [ -z "${ASCEND_VISIBLE_DEVICES:-}" ] && [ -n "${NPU_VISIBLE_DEVICES:-}" ]; then
    export ASCEND_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}"
  fi
  if [ -z "${NPU_VISIBLE_DEVICES:-}" ] && [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    export NPU_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}"
  fi
  if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ] && [[ ! "${LLMAPGEN_KEEP_ASCEND_RT_VISIBLE_DEVICES:-false}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    echo "[di-train] unsetting ASCEND_RT_VISIBLE_DEVICES on DI worker; set LLMAPGEN_KEEP_ASCEND_RT_VISIBLE_DEVICES=true to keep it."
    unset ASCEND_RT_VISIBLE_DEVICES
  fi
else
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}}
  export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
  export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
fi
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
export LLMAPGEN_DISABLE_TRAINER_FLOS=${LLMAPGEN_DISABLE_TRAINER_FLOS:-true}

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
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15 'protobuf==4.25.7'
  pip install 'numpy>=1.26,<2.0' 'opencv-python-headless==4.11.0.86' 'protobuf==4.25.7'
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
if [[ "${SKIP_OBS_DOWNLOADS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  echo "[di-download] SKIP_OBS_DOWNLOADS=true; using local TRAINROOT/QWEN_PATH/VISION_PATH/checkpoints."
else
  echo "[di-download] dataset: ${DATASET_OBS_PATH} -> ${DATASET_ZIP_PATH}"
  python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
  unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

  echo "[di-download] qwen: ${QWEN_MODEL_OBS_PATH} -> ${QWEN_PATH}"
  python -c "import moxing as mox; mox.file.copy_parallel('${QWEN_MODEL_OBS_PATH}', '${QWEN_PATH}')"
fi
if [[ "${EXTRACT_QWEN3VL_TEXT_LLM}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  if [ -f "${QWEN_EXTRACTED_TEXT_PATH}/config.json" ]; then
    echo "[di-extract] use existing Qwen3-VL text LLM: ${QWEN_EXTRACTED_TEXT_PATH}"
  else
    echo "[di-extract] Qwen3-VL text LLM: ${QWEN_PATH} -> ${QWEN_EXTRACTED_TEXT_PATH}"
    python scripts/tools/extract_qwen3vl_text_llm.py \
      --input-dir "${QWEN_PATH}" \
      --output-dir "${QWEN_EXTRACTED_TEXT_PATH}"
  fi
  QWEN_PATH="${QWEN_EXTRACTED_TEXT_PATH}"
  echo "[di-extract] training Qwen text path: ${QWEN_PATH}"
fi
if [[ "${SKIP_OBS_DOWNLOADS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  echo "[di-download] skip vision download; using local ${VISION_PATH}"
else
  echo "[di-download] vision(${VISION_MODEL_FAMILY}): ${VISION_MODEL_OBS_PATH} -> ${VISION_PATH}"
  python -c "import moxing as mox; mox.file.copy_parallel('${VISION_MODEL_OBS_PATH}', '${VISION_PATH}')"
fi
if [[ "${USE_PRETRAINED_VISUAL_BRIDGE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  if [[ "${SKIP_OBS_DOWNLOADS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    echo "[di-download] skip bridge asset download; using local ${ASSET_DIR}"
  else
    echo "[di-download] bridge assets: ${ASSET_OBS_PATH} -> ${ASSET_DIR}"
    python -c "import moxing as mox; mox.file.copy_parallel('${ASSET_OBS_PATH}', '${ASSET_DIR}')"
  fi
else
  echo "[di-download] skip bridge assets; using ${VISION_MODEL_FAMILY} vision checkpoint and randomly initialized alignment modules."
  BRIDGE_MODULES_STATE_PATH=""
fi
if [[ "${USE_VISUAL_ENCODER_CHECKPOINT}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  if [ -n "${VISUAL_ENCODER_CHECKPOINT_OBS_PATH}" ]; then
    mkdir -p "$(dirname "${VISUAL_ENCODER_CHECKPOINT_PATH}")"
    if [[ "${SKIP_OBS_DOWNLOADS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
      echo "[di-download] skip visual encoder checkpoint download; using local ${VISUAL_ENCODER_CHECKPOINT_PATH}"
    else
      echo "[di-download] visual encoder checkpoint: ${VISUAL_ENCODER_CHECKPOINT_OBS_PATH} -> ${VISUAL_ENCODER_CHECKPOINT_PATH}"
      python -c "import moxing as mox; mox.file.copy('${VISUAL_ENCODER_CHECKPOINT_OBS_PATH}', '${VISUAL_ENCODER_CHECKPOINT_PATH}')"
    fi
  fi
else
  VISUAL_ENCODER_CHECKPOINT_PATH=""
fi

if [ ! -f "${TRAINROOT}/train.jsonl" ]; then
  FOUND_TRAINROOT=$(find "${DATASET_EXTRACT_ROOT}" -maxdepth 6 -type f -name train.jsonl -printf '%h\n' 2>/dev/null | sort | head -n 1 || true)
  if [ -n "${FOUND_TRAINROOT}" ]; then
    TRAINROOT="${FOUND_TRAINROOT}"
  fi
fi

for path in \
  "${TRAINROOT}/train.jsonl" \
  "${QWEN_PATH}/config.json" \
  "${VISION_PATH}/config.json"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    echo "Dataset extract summary:"
    find "${DATASET_EXTRACT_ROOT}" -maxdepth 4 -mindepth 1 -printf '  %P\n' 2>/dev/null | head -n 80 || true
    exit 1
  fi
done
if [[ "${USE_VISUAL_ENCODER_CHECKPOINT}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  if [ ! -e "${VISUAL_ENCODER_CHECKPOINT_PATH}" ]; then
    echo "ERROR: required visual encoder checkpoint path not found: ${VISUAL_ENCODER_CHECKPOINT_PATH}"
    exit 1
  fi
fi
if [[ "${USE_PRETRAINED_VISUAL_BRIDGE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  for path in \
    "${BRIDGE_MODULES_STATE_PATH}"; do
    if [ ! -e "${path}" ]; then
      echo "ERROR: required pretrained bridge path not found: ${path}"
      echo "Set USE_PRETRAINED_VISUAL_BRIDGE=false to train from original DINOv2 and randomly initialized alignment modules."
      exit 1
    fi
  done
fi

VISUAL_BRIDGE_ARGS=()
if [[ "${USE_VISUAL_ENCODER_CHECKPOINT}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  VISUAL_BRIDGE_ARGS+=(
    --visual-encoder-checkpoint-path "${VISUAL_ENCODER_CHECKPOINT_PATH}"
  )
fi
if [[ "${USE_PRETRAINED_VISUAL_BRIDGE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  VISUAL_BRIDGE_ARGS+=(
    --bridge-modules-state-path "${BRIDGE_MODULES_STATE_PATH}"
  )
fi

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

LORA_ARGS=(
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
)
if [[ "${NO_LORA}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  LORA_ARGS=(--no-lora)
fi

OPTIM_ARGS=()
if [ -n "${OPTIM}" ]; then
  OPTIM_ARGS=(--optim "${OPTIM}")
fi

DEEPSPEED_ARGS=()
if [ -n "${DEEPSPEED_CONFIG}" ]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

LR_GROUP_ARGS=()
if [ -n "${LANGUAGE_MODEL_LR}" ]; then
  LR_GROUP_ARGS+=(--language-model-lr "${LANGUAGE_MODEL_LR}")
fi
if [ -n "${ALIGNMENT_LR}" ]; then
  LR_GROUP_ARGS+=(--alignment-lr "${ALIGNMENT_LR}")
fi
if [ -n "${VISION_ENCODER_LR}" ]; then
  LR_GROUP_ARGS+=(--vision-encoder-lr "${VISION_ENCODER_LR}")
fi

VISION_LAYER_FUSION_ARGS=()
if [ -n "${VISION_LAYER_FUSION_INDEXES}" ]; then
  VISION_LAYER_FUSION_ARGS=(
    --vision-layer-fusion-indexes "${VISION_LAYER_FUSION_INDEXES}"
    --vision-layer-fusion-type "${VISION_LAYER_FUSION_TYPE}"
  )
fi

PROMPT_ARGS=()
if [ -n "${SYSTEM_PROMPT}" ]; then
  PROMPT_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")
fi
if [ -n "${USER_PROMPT}" ]; then
  PROMPT_ARGS+=(--user-prompt "${USER_PROMPT}")
fi

echo "============================================================"
echo "Run id:       ${RUN_ID}"
echo "Trainroot:    ${TRAINROOT}"
echo "Qwen:         ${QWEN_PATH}"
echo "Vision:       ${VISION_PATH} (${VISION_MODEL_FAMILY}, patch=${VISION_PATCH_SIZE}, prefix=${VISION_NUM_PREFIX_TOKENS})"
echo "Layer fusion: ${VISION_LAYER_FUSION_INDEXES:-off} (${VISION_LAYER_FUSION_TYPE})"
echo "Visual ckpt:  ${VISUAL_ENCODER_CHECKPOINT_PATH}"
echo "Bridge state: ${BRIDGE_MODULES_STATE_PATH}"
echo "Pretrained visual bridge: ${USE_PRETRAINED_VISUAL_BRIDGE}"
echo "Visual encoder checkpoint enabled: ${USE_VISUAL_ENCODER_CHECKPOINT}"
echo "Output:       ${OUTPUT_PATH}"
echo "Cloud output: ${CLOUD_OUTPUT_PATH:-<empty>}"
echo "NNODES/NPROC: ${NNODES}/${NPROC_PER_NODE}, rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Global batch: target=${TARGET_GLOBAL_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "Vision train: freeze=${FREEZE_VISION_ENCODER}, last_n=${VISION_TRAIN_LAST_N_LAYERS}"
echo "Qwen train:    no_lora=${NO_LORA}, optim=${OPTIM:-<default>}"
echo "LR groups:     language=${LANGUAGE_MODEL_LR:-<base>}, alignment=${ALIGNMENT_LR:-<base>}, vision=${VISION_ENCODER_LR:-<base>}"
echo "DeepSpeed:     ${DEEPSPEED_CONFIG:-<disabled>}"
echo "Visual token compressor: mode=${VISUAL_TOKEN_COMPRESSOR}, grid=${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE}, hidden=${VISUAL_TOKEN_COMPRESSOR_HIDDEN_DIM}, depth=${VISUAL_TOKEN_COMPRESSOR_DEPTH}"
if [ -n "${SYSTEM_PROMPT}" ] || [ -n "${USER_PROMPT}" ]; then
  echo "Prompt override: system=$( [ -n "${SYSTEM_PROMPT}" ] && echo yes || echo no ), user=$( [ -n "${USER_PROMPT}" ] && echo yes || echo no )"
else
  echo "Prompt override: none; using Python defaults for MAP_TASK=${MAP_TASK}"
fi
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
  --dinov2-model-name-or-path "${VISION_PATH}" \
  --vision-model-name-or-path "${VISION_PATH}" \
  --vision-patch-size "${VISION_PATCH_SIZE}" \
  --vision-num-prefix-tokens "${VISION_NUM_PREFIX_TOKENS}" \
  "${VISION_LAYER_FUSION_ARGS[@]}" \
  "${VISUAL_BRIDGE_ARGS[@]}" \
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
  "${PROMPT_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  "${OPTIM_ARGS[@]}" \
  "${DEEPSPEED_ARGS[@]}" \
  "${LR_GROUP_ARGS[@]}" \
  --visual-token-compressor "${VISUAL_TOKEN_COMPRESSOR}" \
  --visual-token-compressor-grid-size "${VISUAL_TOKEN_COMPRESSOR_GRID_SIZE}" \
  --visual-token-compressor-hidden-dim "${VISUAL_TOKEN_COMPRESSOR_HIDDEN_DIM}" \
  --visual-token-compressor-depth "${VISUAL_TOKEN_COMPRESSOR_DEPTH}" \
  --visual-token-compressor-dropout "${VISUAL_TOKEN_COMPRESSOR_DROPOUT}" \
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
echo "DI_throughput: ${DI_THROUGHPUT_VALUE} samples/s/npu"

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
