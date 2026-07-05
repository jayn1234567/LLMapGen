#!/usr/bin/env bash

# DI/ModelArts evaluation launcher for LLMapGen DINOv2 + Qwen3-8B LoRA.
# It downloads the prepared dataset, base models, DINOv2 bridge assets, and a
# DI training output checkpoint root, then runs sharded NPU inference/eval/vis
# and uploads the evaluation directory to OUTPUT_URL when the platform provides
# it.

set -euo pipefail

printf '[di-eval-entry] reached LLMapGen DI eval launcher at %s\n' "$(date -Iseconds 2>/dev/null || date)"
printf '[di-eval-entry] argv0=%s argc=%s pwd=%s\n' "$0" "$#" "$(pwd)"
for name in OUTPUT_URL MA_VJ_NAME MA_NUM_HOSTS VC_TASK_INDEX MA_NUM_GPUS VC_WORKER_HOSTS ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES NPU_VISIBLE_DEVICES; do
  eval "value=\${${name}:-}"
  if [ -n "${value}" ]; then
    printf '[di-eval-entry] env %s=%s\n' "${name}" "${value}"
  else
    printf '[di-eval-entry] env %s=<empty>\n' "${name}"
  fi
done

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_ID=${RUN_ID:-eval_dinov2_full_unfreeze_checkpoint_29610_$(date -u +%Y%m%d_%H%M%S)}
WORK_ROOT=${WORK_ROOT:-/cache/llmapgen_eval}
OBS_CACHE=${OBS_CACHE:-/cache}
PYTHON_BIN=${PYTHON_BIN:-python}

CLOUD_OUTPUT_ROOT=${OUTPUT_URL:-}
CLOUD_OUTPUT_PATH=${CLOUD_OUTPUT_ROOT:+${CLOUD_OUTPUT_ROOT%/}/${RUN_ID}}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.zip}
QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/checkpoint/Qwen3-8B}
DINOV2_MODEL_OBS_PATH=${DINOV2_MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
ASSET_OBS_PATH=${ASSET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_centerline_assets_qwen3_8b}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/03/47e3beecf8f34c92a61be56f1372dab2/output/dinov2_full_unfreeze_20260703_232023}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-29610}

DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/llmapgen_eval_dataset_${RUN_ID}.zip}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/llmapgen_eval_dataset_extract_${RUN_ID}}
TRAINROOT_DIR_NAME=${TRAINROOT_DIR_NAME:-prepared_lane_intersection_trainroot}
TRAINROOT=${TRAINROOT:-${DATASET_EXTRACT_ROOT}/${TRAINROOT_DIR_NAME}}

MODEL_ROOT=${MODEL_ROOT:-${WORK_ROOT}/model}
QWEN_PATH=${QWEN_PATH:-${MODEL_ROOT}/Qwen3-8B}
DINOV2_PATH=${DINOV2_PATH:-${MODEL_ROOT}/dinov2-large}
ASSET_DIR=${ASSET_DIR:-${MODEL_ROOT}/dinov2_centerline_assets_qwen3_8b}
VISUAL_ENCODER_CHECKPOINT_PATH=${VISUAL_ENCODER_CHECKPOINT_PATH:-${ASSET_DIR}/visual_encoder_checkpoint.pt}
BRIDGE_MODULES_STATE_PATH=${BRIDGE_MODULES_STATE_PATH:-${ASSET_DIR}/bridge_modules_state.pt}

CHECKPOINT_RUN_NAME=${CHECKPOINT_RUN_NAME:-$(basename "${CHECKPOINT_OBS_PATH%/}")}
LOCAL_RUN_ROOT=${LOCAL_RUN_ROOT:-${WORK_ROOT}/di_outputs/${CHECKPOINT_RUN_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${LOCAL_RUN_ROOT}/${CHECKPOINT_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-${WORK_ROOT}/eval_outputs/${RUN_ID}}

SPLIT=${SPLIT:-val}
MAP_TASK=${MAP_TASK:-lane_intersection}
CATEGORIES=${CATEGORIES:-centerline,intersection}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
VIS_LIMIT=${VIS_LIMIT:-512}
IMAGE_SIZE=${IMAGE_SIZE:-512}
ENCODER_INPUT_PAD_SIZE=${ENCODER_INPUT_PAD_SIZE:-518}
INFER_HEARTBEAT_SECONDS=${INFER_HEARTBEAT_SECONDS:-60}
AUTO_INSTALL_EVAL_DEPS=${AUTO_INSTALL_EVAL_DEPS:-true}

INSTALL_DEPS=${INSTALL_DEPS:-True}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}
MOXING_WHL_OBS_PATH=${MOXING_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}
TORCH_NPU_WHL_OBS_PATH=${TORCH_NPU_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
TORCH_NPU_WHL_LOCAL_PATH=${TORCH_NPU_WHL_LOCAL_PATH:-/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}

source_if_exists() {
  if [ -f "$1" ]; then
    local nounset_was_on=0
    case "$-" in
      *u*) nounset_was_on=1; set +u ;;
    esac
    export ZSH_VERSION="${ZSH_VERSION:-}"
    # shellcheck disable=SC1090
    source "$1"
    if [ "${nounset_was_on}" = "1" ]; then
      set -u
    fi
    echo "[di-eval] sourced $1"
  fi
}

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES)$ ]]
}

copy_obs_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  "${PYTHON_BIN}" - "$src" "$dst" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
}

copy_obs_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  "${PYTHON_BIN}" - "$src" "$dst" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
}

NODE_RANK=0
if [ -n "${MA_VJ_NAME:-}" ]; then
  NODE_RANK=${VC_TASK_INDEX:-0}
fi
if [ "${NODE_RANK}" != "0" ]; then
  echo "[di-eval] node_rank=${NODE_RANK}; evaluation is executed only on node 0 to avoid duplicate metrics."
  exit 0
fi

source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists /usr/local/Ascend/ascend-toolkit/set_env.sh
source_if_exists /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-FALSE}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [ -n "${MA_VJ_NAME:-}" ]; then
  NPROC_PER_NODE=${NPROC_PER_NODE:-${MA_NUM_GPUS:-8}}
else
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
fi

if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  if [ -n "${NPU_VISIBLE_DEVICES:-}" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}"
  elif [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}"
  else
    ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
    export ASCEND_RT_VISIBLE_DEVICES
  fi
fi
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}

if bool_enabled "${ENABLE_MOXING_UPGRADE}"; then
  echo "[di-eval] installing moxing wheel"
  USE_MEMARTS=0 "${PYTHON_BIN}" -c "import moxing as mox; mox.file.copy('${MOXING_WHL_OBS_PATH}', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
  "${PYTHON_BIN}" -m pip uninstall moxing-framework -y || true
  "${PYTHON_BIN}" -m pip cache purge || true
  "${PYTHON_BIN}" -m pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
  export MOX_PROFILE=1
  export MOX_RECORD_OBS=1
fi

if bool_enabled "${INSTALL_DEPS}"; then
  echo "[di-eval] installing runtime dependencies"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  "${PYTHON_BIN}" -m pip install torch==2.7.1 torch_npu==2.7.1rc1
  copy_obs_file "${TORCH_NPU_WHL_OBS_PATH}" "${TORCH_NPU_WHL_LOCAL_PATH}"
  "${PYTHON_BIN}" -m pip install --force-reinstall "${TORCH_NPU_WHL_LOCAL_PATH}"
  "${PYTHON_BIN}" -m pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "transformers==4.56.2" "tokenizers>=0.22.0,<0.23.0"
  "${PYTHON_BIN}" -m pip install accelerate==1.6.0 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  "${PYTHON_BIN}" -m pip install "peft>=0.10.0" pydantic "numpy>=1.26,<2.0" "scipy>=1.10" "scikit-learn>=1.2"
  "${PYTHON_BIN}" -m pip install requests "einops>=0.6" "timm>=0.9.0" "opencv-python-headless==4.11.0.86"
  "${PYTHON_BIN}" -m pip install "loguru>=0.7.0" "shapely>=2.0.0" "huggingface-hub==0.36.2" urllib3==1.26.15 "protobuf==4.25.7"
  "${PYTHON_BIN}" -m pip install "numpy>=1.26,<2.0" "opencv-python-headless==4.11.0.86" "protobuf==4.25.7"
fi

"${PYTHON_BIN}" - <<'PY'
import os
import sys

print(f"[di-eval-preflight] python={sys.executable} version={sys.version.split()[0]}", flush=True)
for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
    print(f"[di-eval-preflight] env {name}={os.environ.get(name, '<empty>')}", flush=True)
import torch
import torch_npu

print(f"[di-eval-preflight] torch={torch.__version__}", flush=True)
print(f"[di-eval-preflight] torch_npu={getattr(torch_npu, '__version__', 'unknown')}", flush=True)
print(f"[di-eval-preflight] npu_available={torch.npu.is_available()}", flush=True)
print(f"[di-eval-preflight] npu_count={torch.npu.device_count()}", flush=True)
if not torch.npu.is_available():
    raise SystemExit("NPU is not available after torch/torch_npu installation.")
PY

mkdir -p "${WORK_ROOT}" "${MODEL_ROOT}" "${DATASET_EXTRACT_ROOT}" "${OUTPUT_DIR}"

echo "[di-eval-download] dataset: ${DATASET_OBS_PATH} -> ${DATASET_ZIP_PATH}"
copy_obs_file "${DATASET_OBS_PATH}" "${DATASET_ZIP_PATH}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

echo "[di-eval-download] qwen: ${QWEN_MODEL_OBS_PATH} -> ${QWEN_PATH}"
copy_obs_dir "${QWEN_MODEL_OBS_PATH}" "${QWEN_PATH}"
echo "[di-eval-download] dinov2: ${DINOV2_MODEL_OBS_PATH} -> ${DINOV2_PATH}"
copy_obs_dir "${DINOV2_MODEL_OBS_PATH}" "${DINOV2_PATH}"
echo "[di-eval-download] bridge assets: ${ASSET_OBS_PATH} -> ${ASSET_DIR}"
copy_obs_dir "${ASSET_OBS_PATH}" "${ASSET_DIR}"

echo "[di-eval-download] checkpoint root: ${CHECKPOINT_OBS_PATH} -> ${LOCAL_RUN_ROOT}"
CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" \
CHECKPOINT_NAMES="${CHECKPOINT_NAME}" \
OBS_CACHE="${WORK_ROOT}/di_outputs" \
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT}" \
bash scripts/npu/test/download_di_output_for_infer.sh

if [ ! -f "${TRAINROOT}/${SPLIT}.jsonl" ]; then
  FOUND_TRAINROOT=$(find "${DATASET_EXTRACT_ROOT}" -maxdepth 6 -type f -name "${SPLIT}.jsonl" -printf '%h\n' 2>/dev/null | sort | head -n 1 || true)
  if [ -n "${FOUND_TRAINROOT}" ]; then
    TRAINROOT="${FOUND_TRAINROOT}"
  fi
fi

for path in \
  "${TRAINROOT}/${SPLIT}.jsonl" \
  "${QWEN_PATH}/config.json" \
  "${DINOV2_PATH}/config.json" \
  "${VISUAL_ENCODER_CHECKPOINT_PATH}" \
  "${CHECKPOINT_DIR}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    echo "Dataset extract summary:"
    find "${DATASET_EXTRACT_ROOT}" -maxdepth 4 -mindepth 1 -printf '  %P\n' 2>/dev/null | head -n 120 || true
    echo "Checkpoint run root summary:"
    find "${LOCAL_RUN_ROOT}" -maxdepth 2 -mindepth 1 -printf '  %P\n' 2>/dev/null | head -n 120 || true
    exit 1
  fi
done

echo "============================================================"
echo "DI eval run id: ${RUN_ID}"
echo "Checkpoint OBS: ${CHECKPOINT_OBS_PATH}"
echo "Checkpoint:     ${CHECKPOINT_DIR}"
echo "Run root:       ${LOCAL_RUN_ROOT}"
echo "Trainroot:      ${TRAINROOT}"
echo "Qwen:           ${QWEN_PATH}"
echo "DINOv2:         ${DINOV2_PATH}"
echo "Visual ckpt:    ${VISUAL_ENCODER_CHECKPOINT_PATH}"
echo "Output:         ${OUTPUT_DIR}"
echo "Cloud output:   ${CLOUD_OUTPUT_PATH:-<empty>}"
echo "NPROC:          ${NPROC_PER_NODE}"
echo "Visible:        ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Split/samples:  ${SPLIT}/${MAX_SAMPLES}"
echo "============================================================"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
INFER_HEARTBEAT_SECONDS="${INFER_HEARTBEAT_SECONDS}" \
RUN_ROOT="${LOCAL_RUN_ROOT}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
TRAINROOT="${TRAINROOT}" \
MODEL_NAME_OR_PATH="${QWEN_PATH}" \
TOKENIZER_NAME_OR_PATH="${QWEN_PATH}" \
DINOV2_MODEL_NAME_OR_PATH="${DINOV2_PATH}" \
VISUAL_ENCODER_CHECKPOINT_PATH="${VISUAL_ENCODER_CHECKPOINT_PATH}" \
BRIDGE_MODULES_STATE_PATH="${BRIDGE_MODULES_STATE_PATH}" \
SPLIT="${SPLIT}" \
MAP_TASK="${MAP_TASK}" \
CATEGORIES="${CATEGORIES}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
VIS_LIMIT="${VIS_LIMIT}" \
IMAGE_SIZE="${IMAGE_SIZE}" \
ENCODER_INPUT_PAD_SIZE="${ENCODER_INPUT_PAD_SIZE}" \
AUTO_INSTALL_EVAL_DEPS="${AUTO_INSTALL_EVAL_DEPS}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash scripts/npu/test/infer_eval_visualize_dinov2_centerline_qwen_lora_npu.sh

if [ -n "${CLOUD_OUTPUT_PATH:-}" ]; then
  echo "[di-eval-upload] ${OUTPUT_DIR} -> ${CLOUD_OUTPUT_PATH}"
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${CLOUD_OUTPUT_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
  echo "[di-eval-upload] done: ${CLOUD_OUTPUT_PATH}"
else
  echo "[di-eval-upload] OUTPUT_URL is empty; local output kept at ${OUTPUT_DIR}"
fi
