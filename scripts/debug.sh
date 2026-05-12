#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Local NPU debug: Qwen3-VL-8B + DINOv3-L + DeepStack full-param
# - No dependency installation
# - No OBS download/upload
# - Runs one training step by default to verify the DINOv3 training path
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/..")
cd "${REPO_ROOT}"

echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"

# ====================== NPU environment ======================
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ====================== distributed parameters ======================
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
export RDZV_ID=${RDZV_ID:-debug_dinov3_qwen3vl}

echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# ====================== HCCL & NPU settings ======================
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export INF_NAN_MODE_ENABLE=${INF_NAN_MODE_ENABLE:-1}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-FALSE}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export LLAVA_LOG_RANK0_ONLY=${LLAVA_LOG_RANK0_ONLY:-1}

# ====================== local paths ======================
DATASET_PATH=${DATASET_PATH:-/cache/MLLM20260427_rc_jjh}
TRAIN_PATH=${TRAIN_PATH:-${DATASET_PATH}/train.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
DINOV3_PATH=${DINOV3_PATH:-/cache/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
QWEN3VL_PATH=${QWEN3VL_PATH:-${Qwen3VL_PATH:-/cache/checkpoints/Qwen3-VL-8B-Instruct}}
OUTPUT_PATH=${OUTPUT_PATH:-/cache/debug_dinov3_qwen3vl_npu}

for path in "${TRAIN_PATH}" "${IMAGE_FOLDER}" "${DINOV3_PATH}" "${QWEN3VL_PATH}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required local path not found: ${path}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "========== key deps =========="
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
echo "==============================="

# ====================== debug training params ======================
MM_VISION_SELECT_LAYER=${MM_VISION_SELECT_LAYER:--2}
MM_PROJECTOR_TYPE=${MM_PROJECTOR_TYPE:-mlp2x_gelu}
UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-"6 12 18 23"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-1}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-5e-5}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SAMPLE_SEED=${SAMPLE_SEED:-42}

DEEPSTACK_ARGS=()
if [ -n "${DEEPSTACK_VISUAL_INDEXES}" ]; then
    DEEPSTACK_ARGS=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi

echo "============================================================"
echo "Debug:      DINOv3 full-param train smoke"
echo "Model:      ${QWEN3VL_PATH}"
echo "ViT:        ${DINOV3_PATH}"
echo "Train:      ${TRAIN_PATH}"
echo "Images:     ${IMAGE_FOLDER}"
echo "Output:     ${OUTPUT_PATH}"
echo "DeepStack:  ${DEEPSTACK_VISUAL_INDEXES:-disabled}"
echo "Max steps:  ${MAX_STEPS}"
echo "Batch:      ${PER_DEVICE_TRAIN_BATCH_SIZE}/npu x ${GRADIENT_ACCUMULATION_STEPS}"
echo "DeepSpeed:  ${DEEPSPEED_CONFIG}"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m llava.train.train_qwen \
    --model_name_or_path "${QWEN3VL_PATH}" \
    --version conv_qwen_3_Dinov2_huawei \
    --vision_tower "${DINOV3_PATH}" \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
    "${DEEPSTACK_ARGS[@]}" \
    --data_path "${TRAIN_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --sample_seed "${SAMPLE_SEED}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_PATH}" \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay 0.0 \
    --warmup_steps 0 \
    --lr_scheduler_type constant \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    --save_strategy no \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl \
    --deepspeed "${DEEPSPEED_CONFIG}"

echo "Debug training finished."
