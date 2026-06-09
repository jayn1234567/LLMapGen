#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NPU SFT training
# Fixed recipe: phase_a | lane+intersection | DINOv2 direct ViT layer fusion, no DeepStack | Qwen3 text LLM
# This file is self-contained and does not call another project .sh file.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"

DATASET_PHASE=phase_a
MAP_TASK=lane_intersection
VISION_RECIPE=dinov2_layer_fusion
MODEL_FAMILY=qwen3
MODEL_LABEL=qwen3
TRAIN_VARIANT=full

case "${DATASET_PHASE}" in phase_a|phase_b) ;; *) echo "ERROR: DATASET_PHASE must be phase_a or phase_b"; exit 1 ;; esac
case "${MAP_TASK}" in lane|lane_intersection) ;; *) echo "ERROR: MAP_TASK must be lane or lane_intersection"; exit 1 ;; esac
case "${MODEL_FAMILY}" in qwen3|qwen3_5) ;; *) echo "ERROR: MODEL_FAMILY must be qwen3 or qwen3_5"; exit 1 ;; esac
case "${TRAIN_VARIANT}" in full|lora_llm) ;; *) echo "ERROR: TRAIN_VARIANT must be full or lora_llm"; exit 1 ;; esac

CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="${CLUSTER_SAVE}"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}

DINO_V2_TOWER_NAME=${DINO_V2_TOWER_NAME:-facebook_dinov2-large}
DINO_V3_TOWER_NAME=${DINO_V3_TOWER_NAME:-facebook_dinov3-vitl16-pretrain-lvd1689m}
SIGLIP_TOWER_NAME=${SIGLIP_TOWER_NAME:-google_siglip-large-patch16-384}
DINO_V2_TOWER=${DINO_V2_TOWER:-${OBS_CACHE}/checkpoints/${DINO_V2_TOWER_NAME}}
DINO_V3_TOWER=${DINO_V3_TOWER:-${OBS_CACHE}/checkpoints/${DINO_V3_TOWER_NAME}}
SIGLIP_TOWER=${SIGLIP_TOWER:-${OBS_CACHE}/checkpoints/${SIGLIP_TOWER_NAME}}

if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
  DEFAULT_QWEN_MODEL_NAME=Qwen3.5-4B-Instruct
else
  DEFAULT_QWEN_MODEL_NAME=Qwen3-8B
fi
QWEN_MODEL_NAME=${QWEN_MODEL_NAME:-${DEFAULT_QWEN_MODEL_NAME}}
QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH:-${MODEL_OBS_PATH}/${QWEN_MODEL_NAME}}
QWEN_PATH=${QWEN_PATH:-${OBS_CACHE}/checkpoints/${QWEN_MODEL_NAME}}

DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}
STAGE_A_CHECKPOINT_OBS_PATH=${STAGE_A_CHECKPOINT_OBS_PATH:-}
STAGE_A_CHECKPOINT_PATH=${STAGE_A_CHECKPOINT_PATH:-}
STAGE_A_DOWNLOAD_DIR=${STAGE_A_DOWNLOAD_DIR:-${OBS_CACHE}/stage_a_checkpoint_${RUN_ID}}

case "${VISION_RECIPE}" in
  dinov2|dinov2_layer_fusion|dinov2_deepstack)
    VISION_BACKBONE=dinov2
    VISION_TOWER=${VISION_TOWER:-${DINO_V2_TOWER}}
    MM_VISION_TOWER_TYPE=dinov2
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    DOWNLOAD_TOWER_NAMES=("${DINO_V2_TOWER_NAME}")
    REQUIRED_VISION_TOWERS=("${DINO_V2_TOWER}")
    ;;
  dinov3|dinov3_layer_fusion|dinov3_deepstack)
    VISION_BACKBONE=dinov3
    VISION_TOWER=${VISION_TOWER:-${DINO_V3_TOWER}}
    MM_VISION_TOWER_TYPE=dinov3
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    DOWNLOAD_TOWER_NAMES=("${DINO_V3_TOWER_NAME}")
    REQUIRED_VISION_TOWERS=("${DINO_V3_TOWER}")
    ;;
  dinov2_siglip_concat)
    VISION_BACKBONE=dinov2_siglip_concat
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINO_V2_TOWER},${SIGLIP_TOWER}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov2,siglip}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,384}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-0}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-concat_projector}
    VISION_TOWER=${VISION_TOWER:-${MULTI_VISION_TOWERS}}
    MM_VISION_TOWER_TYPE=multi_concat
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    DOWNLOAD_TOWER_NAMES=("${DINO_V2_TOWER_NAME}" "${SIGLIP_TOWER_NAME}")
    REQUIRED_VISION_TOWERS=("${DINO_V2_TOWER}" "${SIGLIP_TOWER}")
    ;;
  dinov3_siglip_concat)
    VISION_BACKBONE=dinov3_siglip_concat
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINO_V3_TOWER},${SIGLIP_TOWER}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov3,siglip}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,384}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-0}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-concat_projector}
    VISION_TOWER=${VISION_TOWER:-${MULTI_VISION_TOWERS}}
    MM_VISION_TOWER_TYPE=multi_concat
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    DOWNLOAD_TOWER_NAMES=("${DINO_V3_TOWER_NAME}" "${SIGLIP_TOWER_NAME}")
    REQUIRED_VISION_TOWERS=("${DINO_V3_TOWER}" "${SIGLIP_TOWER}")
    ;;
  multi_moe)
    VISION_BACKBONE=multi_moe
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINO_V2_TOWER},${DINO_V3_TOWER}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov2,dinov3}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,512}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-1}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-softmax_router}
    VISION_TOWER=${VISION_TOWER:-${MULTI_VISION_TOWERS}}
    MM_VISION_TOWER_TYPE=multi_moe
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    DOWNLOAD_TOWER_NAMES=("${DINO_V2_TOWER_NAME}" "${DINO_V3_TOWER_NAME}")
    REQUIRED_VISION_TOWERS=("${DINO_V2_TOWER}" "${DINO_V3_TOWER}")
    ;;
  *) echo "ERROR: unsupported VISION_RECIPE=${VISION_RECIPE}"; exit 1 ;;
esac

DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}
case "${VISION_RECIPE}" in
  dinov2_deepstack|dinov3_deepstack)
    if [[ "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then DISABLE_DEEPSTACK=False; fi
    DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-"6 12 18 23"}
    ;;
  dinov2_layer_fusion|dinov3_layer_fusion)
    VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-"6 12 18 23"}
    ;;
esac

TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}
MM_VISION_FUSION_LR=${MM_VISION_FUSION_LR:-${MM_PROJECTOR_LR}}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-500}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
ENABLE_EVAL=${ENABLE_EVAL:-False}
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-False}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-5000}
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}
BEST_INFER_INDEX_METRIC=${BEST_INFER_INDEX_METRIC:-length_f1}
BEST_INFER_INDEX_NUM_SAMPLES=${BEST_INFER_INDEX_NUM_SAMPLES:-0}
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-5}

if [ "${TRAIN_VARIANT}" = "lora_llm" ]; then
  LORA_ENABLE=${LORA_ENABLE:-True}
  LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
  LORA_R=${LORA_R:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
  LORA_DROPOUT=${LORA_DROPOUT:-0.05}
  LORA_BIAS=${LORA_BIAS:-none}
  UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}
else
  LORA_ENABLE=${LORA_ENABLE:-False}
  LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
  LORA_R=${LORA_R:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
  LORA_DROPOUT=${LORA_DROPOUT:-0.05}
  LORA_BIAS=${LORA_BIAS:-none}
  UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}
fi

SWANLAB_ENABLE=${SWANLAB_ENABLE:-True}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}}
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,${DATASET_PHASE},${MAP_TASK},${VISION_RECIPE},${MODEL_LABEL},${TRAIN_VARIANT},unimapgen_v9}
SWANLAB_MODE=${SWANLAB_MODE:-offline}
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}

export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
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
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

INSTALL_DEPS=${INSTALL_DEPS:-True}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers>=5.7.0"}
TOKENIZERS_SPEC=${TOKENIZERS_SPEC:-"tokenizers>=0.22.0"}
if [[ "${ENABLE_MOXING_UPGRADE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
  pip uninstall moxing-framework -y
  pip cache purge
  pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
  export MOX_PROFILE=1
  export MOX_RECORD_OBS=1
fi
if [[ "${INSTALL_DEPS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  pip install torch==2.7.1 torch_npu==2.7.1rc1
  python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
  pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl
  pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "${TRANSFORMERS_SPEC}" "${TOKENIZERS_SPEC}"
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless>=4.8.0'
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15
fi

resolve_training_checkpoint() {
  python - "$1" <<'PYRESOLVE'
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin")):
    print(root)
    raise SystemExit(0)
cmd = [sys.executable, "scripts/tools/resolve_best_checkpoint.py", "--output-dir", str(root), "--best-name", "infer_best", "--best-name", "eval_best", "--best-name", "best", "--best-name", "best_reward", "--allow-direct"]
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
if result.returncode == 0 and result.stdout.strip():
    print(result.stdout.strip())
    raise SystemExit(0)
checkpoints = []
for path in root.glob("checkpoint-*"):
    if path.is_dir():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        checkpoints.append((step, path))
if checkpoints:
    print(sorted(checkpoints)[-1][1])
    raise SystemExit(0)
raise SystemExit(f"cannot resolve a usable checkpoint under: {root}")
PYRESOLVE
}

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
export RDZV_ID=${RDZV_ID:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}_${RUN_ID}}
mkdir -p "${LOCAL_MODEL_SAVE_PATH}"
OUTPUT_PATH="${LOCAL_MODEL_SAVE_PATH}"
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab}

for i in "${!DOWNLOAD_TOWER_NAMES[@]}"; do
  python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${DOWNLOAD_TOWER_NAMES[$i]}', '${REQUIRED_VISION_TOWERS[$i]}')"
done
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

if [ "${DATASET_PHASE}" = "phase_b" ]; then
  if [ -n "${STAGE_A_CHECKPOINT_OBS_PATH}" ]; then
    python -c "import moxing as mox; mox.file.copy_parallel('${STAGE_A_CHECKPOINT_OBS_PATH}', '${STAGE_A_DOWNLOAD_DIR}')"
    CHECKPOINT_INPUT_PATH="${STAGE_A_DOWNLOAD_DIR}"
  elif [ -n "${STAGE_A_CHECKPOINT_PATH}" ]; then
    CHECKPOINT_INPUT_PATH="${STAGE_A_CHECKPOINT_PATH}"
  else
    echo "ERROR: set STAGE_A_CHECKPOINT_OBS_PATH or STAGE_A_CHECKPOINT_PATH for Stage-B SFT."
    exit 1
  fi
  INIT_MODEL_PATH=$(resolve_training_checkpoint "${CHECKPOINT_INPUT_PATH}")
  if [ "${TRAIN_VARIANT}" = "lora_llm" ] && compgen -G "${INIT_MODEL_PATH}/adapter_model*" > /dev/null; then
    echo "ERROR: Stage-B lora_llm expects a full Stage-A checkpoint. Adapter-only Stage-A LoRA continuation is not supported by this train entrypoint."
    exit 1
  fi
else
  if [ ! -e "${QWEN_PATH}/config.json" ]; then
    python -c "import moxing as mox; mox.file.copy_parallel('${QWEN_MODEL_OBS_PATH}', '${QWEN_PATH}')"
  fi
  INIT_MODEL_PATH="${QWEN_PATH}"
fi

TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
for path in "${INIT_MODEL_PATH}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}" "${REQUIRED_VISION_TOWERS[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then GRADIENT_ACCUMULATION_STEPS=1; fi

EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")
EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(--eval_data_path "${EVAL_PATH}" --eval_image_folder "${IMAGE_FOLDER}" "${EVAL_STRATEGY_ARG}" steps --eval_steps "${EVAL_STEPS}" --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}" --best_eval_loss_dir eval_best)
fi
VISION_ARGS=(--vision_tower "${VISION_TOWER}" --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" --input_image_size "${INPUT_IMAGE_SIZE}")
if [[ "${MM_VISION_TOWER_TYPE}" == "multi_moe" || "${MM_VISION_TOWER_TYPE}" == "multi_concat" ]]; then
  VISION_ARGS+=(--multi_vision_towers "${MULTI_VISION_TOWERS}" --multi_vision_tower_types "${MULTI_VISION_TOWER_TYPES}" --multi_vision_input_image_sizes "${MULTI_VISION_INPUT_IMAGE_SIZES}" --multi_vision_primary_index "${MULTI_VISION_PRIMARY_INDEX}" --multi_vision_hidden_size "${MULTI_VISION_HIDDEN_SIZE}" --multi_vision_target_grid "${MULTI_VISION_TARGET_GRID}" --multi_vision_fusion "${MULTI_VISION_FUSION}")
fi
if [ -n "${VISION_LAYER_FUSION_INDEXES}" ]; then
  VISION_ARGS+=(--vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES} --vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}")
fi
if [[ ! "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ && -n "${DEEPSTACK_VISUAL_INDEXES}" ]]; then
  VISION_ARGS+=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi

BEST_INFER_VISION_TOWER="${VISION_TOWER}"
if [[ "${MM_VISION_TOWER_TYPE}" == "multi_moe" || "${MM_VISION_TOWER_TYPE}" == "multi_concat" ]]; then
  BEST_INFER_VISION_TOWER="${MULTI_VISION_TOWERS}"
fi

echo "============================================================"
echo "Recipe:       ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_RECIPE} | ${MODEL_FAMILY} | ${TRAIN_VARIANT}"
echo "Init model:   ${INIT_MODEL_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Vision type:  ${MM_VISION_TOWER_TYPE} fusion=${MULTI_VISION_FUSION:-single}"
echo "DeepStack disabled: ${DISABLE_DEEPSTACK}, indexes=${DEEPSTACK_VISUAL_INDEXES:-auto}"
echo "Layer fusion: ${VISION_LAYER_FUSION_INDEXES:-off} (${VISION_LAYER_FUSION_TYPE})"
echo "LoRA:         enable=${LORA_ENABLE}, scope=${LORA_TARGET_SCOPE}, r=${LORA_R}"
echo "Train:        ${TRAIN_PATH}"
echo "Eval:         ${EVAL_PATH}"
echo "Output:       ${OUTPUT_PATH}"
echo "Cloud output: ${CLOUD_OUTPUT_PATH}"
echo "============================================================"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  --model_name_or_path "${INIT_MODEL_PATH}" \
  --version conv_qwen_3_Dinov2_huawei \
  "${VISION_ARGS[@]}" \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
  --disable_deepstack "${DISABLE_DEEPSTACK}" \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --sample_seed 42 \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_target_scope "${LORA_TARGET_SCOPE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_bias "${LORA_BIAS}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --mm_vision_fusion_lr "${MM_VISION_FUSION_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
  --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
  --best_train_loss_dir best \
  --save_best_infer_index "${SAVE_BEST_INFER_INDEX}" \
  --best_infer_index_dir infer_best \
  --best_infer_index_metric "${BEST_INFER_INDEX_METRIC}" \
  --best_infer_index_phase "${DATASET_PHASE}" \
  --best_infer_index_eval_data_path "${EVAL_PATH}" \
  --best_infer_index_image_folder "${IMAGE_FOLDER}" \
  --best_infer_index_vision_tower "${BEST_INFER_VISION_TOWER}" \
  --best_infer_index_input_image_size "${INPUT_IMAGE_SIZE}" \
  --best_infer_index_conv_template conv_qwen_3_Dinov2_huawei \
  --best_infer_index_map_task "${MAP_TASK}" \
  --best_infer_index_num_samples "${BEST_INFER_INDEX_NUM_SAMPLES}" \
  --best_infer_index_eval_steps "${SAVE_STEPS}" \
  --best_infer_index_max_new_tokens 2048 \
  --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
  --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
  --use_hf_progress_bar True \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to none \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type sft \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --swanlab_log_dir "${SWANLAB_LOG_DIR}" \
  --swanlab_api_host "${SWANLAB_API_HOST}" \
  --swanlab_web_host "${SWANLAB_WEB_HOST}" \
  --ddp_find_unused_parameters False \
  --ddp_backend hccl \
  --deepspeed "${DEEPSPEED_CONFIG}"

if [[ "${NODE_RANK}" == "0" ]]; then
  if [ -e "${CLOUD_OUTPUT_PATH}" ]; then
    echo "ERROR: cloud output path already exists, refusing to overwrite: ${CLOUD_OUTPUT_PATH}"
    exit 1
  fi
  echo "Moving rank0 local output to cloud output: ${OUTPUT_PATH} -> ${CLOUD_OUTPUT_PATH}"
  mv "${OUTPUT_PATH}" "${CLOUD_OUTPUT_PATH}"
  echo "Final cloud output path: ${CLOUD_OUTPUT_PATH}"
else
  echo "Non-master node ${NODE_RANK}: skip cloud output move."
fi
