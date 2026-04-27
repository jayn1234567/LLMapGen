#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# NPU (Ascend) training script
# Reference: adapts llava.train.train_qwen for Ascend NPU
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR/.."  # project root

# -------------------- NPU environment --------------------
export ASCEND_CUSTOM_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# HCCL / distributed settings
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export HCCL_ASYNC_ERROR_HANDLING=0
export INF_NAN_MODE_ENABLE=1
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

# -------------------- user-adjustable paths --------------------
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-}          # e.g. obs://bucket/path/to/models/
DATASET_OBS_PATH=${DATASET_OBS_PATH:-}     # e.g. obs://bucket/path/to/dataset.zip
OUTPUT_URL=${OUTPUT_URL:-outputs/npu_train}

# local paths
DINOV2_PATH=${DINOV2_PATH:-${OBS_CACHE}/dinov2-large}
FASTVLM_PATH=${FASTVLM_PATH:-${OBS_CACHE}/llava-fastvithd_1.5b_stage2}
DATASET_PATH=${DATASET_PATH:-${OBS_CACHE}/data}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}/images}

# training params
NUM_EPOCHS=${NUM_EPOCHS:-3}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-5e-5}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}

# -------------------- download from OBS (optional) --------------------
download_from_obs() {
    echo ">>> Downloading models/data from OBS..."
    if [ -n "${MODEL_OBS_PATH}" ]; then
        python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/dinov2-large', '${DINOV2_PATH}')" 2>/dev/null || echo "[WARN] moxing not available, expecting local files"
        python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/llava-fastvithd_1.5b_stage2', '${FASTVLM_PATH}')" 2>/dev/null || echo "[WARN] moxing not available"
    fi
    if [ -n "${DATASET_OBS_PATH}" ]; then
        python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset.zip')" 2>/dev/null || true
        if [ -f "${OBS_CACHE}/dataset.zip" ]; then
            unzip -o "${OBS_CACHE}/dataset.zip" -d "${DATASET_PATH}"
        fi
    fi
    echo ">>> Download done."
}

# -------------------- distributed setup --------------------
if [[ -z "${MA_VJ_NAME:-}" ]]; then
    NNODES=1
    NODE_RANK=0
    NPROC_PER_NODE=${NPROC_PER_NODE:-8}
    MASTER_ADDR=localhost
else
    NNODES="$MA_NUM_HOSTS"
    NODE_RANK="$VC_TASK_INDEX"
    NPROC_PER_NODE="$MA_NUM_GPUS"   # NPU count per node
    MASTER_HOST="$VC_WORKER_HOSTS"
    MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
fi
MASTER_PORT="6060"

export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
echo "NNODES=$NNODES  NODE_RANK=$NODE_RANK  NPROC_PER_NODE=$NPROC_PER_NODE"

# -------------------- main --------------------
download_from_obs

# Ensure data paths exist
if [ ! -f "${DATASET_PATH}/train.jsonl" ]; then
    echo "ERROR: train.jsonl not found at ${DATASET_PATH}/train.jsonl"
    exit 1
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

echo "=== Starting NPU training ==="
echo "DINOv2: ${DINOV2_PATH}"
echo "FastVLM: ${FASTVLM_PATH}"
echo "Data: ${DATASET_PATH}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m llava.train.train_qwen \
    --model_name_or_path "${FASTVLM_PATH}" \
    --version qwen_2_centerline_coord \
    --unfreeze_mm_vision_tower False \
    --vision_tower "${DINOV2_PATH}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATASET_PATH}/train.jsonl" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_URL}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --evaluation_strategy no \
    --load_best_model_at_end False \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 10 \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl

echo "=== Training finished ==="
