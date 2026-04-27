#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NPU cloud training script (ModelArts / Ascend)
# Usage:
#   bash train_npu.sh --model_obs_path <obs_path> --dataset_obs_path <obs_path>
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR/.."  # project root
echo "Script path: $SCRIPT_PATH"
echo "Script folder path: $SCRIPT_DIR"
echo "Current working path: $PWD"

# ==================== ASCEND env ====================
export ASCEND_CUSTOM_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

workerID=$(echo $HOSTNAME | awk -F'-' '{print $(NF-1)"-"$NF}')
echo ${workerID}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
sudo chmod -R 777 /usr/local/Ascend/ascend-toolkit/
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# ==================== moxing upgrade ====================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> changing moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
pip uninstall moxing-framework -y
pip cache purge
pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
export MOX_PROFILE=1
export MOX_RECORD_OBS=1
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> moxing change finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> origin pip package version >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
pip list
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> origin pip package version >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ==================== proxy ====================
echo $https_proxy
echo $HTTPS_PROXY
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# ==================== pip install deps ====================
pip install loguru shapely
pip install -e . 2>/dev/null || pip install -e . --no-deps
pip install torch==2.7.1
pip install torch_npu==2.7.1rc1
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
pip install /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

pip install transformers==4.57.3
pip install deepspeed==0.14.4
pip install tokenizers==0.22.1
pip install accelerate==1.6.0
pip install safetensors
pip install Pillow
pip install urllib3==1.26.15

pip list

# ==================== distributed config ====================
if [[ -z "${MA_VJ_NAME}" ]]; then
    NNODES=1
    NODE_RANK=0
    NPROC_PER_NODE=8
    MASTER_ADDR=localhost
else
    NNODES="$MA_NUM_HOSTS"
    NODE_RANK="$VC_TASK_INDEX"
    NPROC_PER_NODE="$MA_NUM_GPUS"
    MASTER_HOST="$VC_WORKER_HOSTS"
    MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
fi

MASTER_PORT="6060"

export NNODES=$NNODES
export NODE_RANK=$NODE_RANK
export NPROC_PER_NODE=$NPROC_PER_NODE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export RDZV_ID='1234'

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

if [[ $NODE_RANK == 0 ]]; then
    EXT_ARGS="--rdzv_conf=is_host=1"
else
    EXT_ARGS=""
fi

# HCCL / network
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
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
export OMP_NUM_THREADS=1

# ==================== OBS output path ====================
CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="$CLUSTER_SAVE"
echo "System defined obs share path: $OSB_SHARE_PATH"

LOCAL_MODEL_SAVE_PATH='/cache/local_model_save_path'
mkdir -p $LOCAL_MODEL_SAVE_PATH

if [[ $NODE_RANK == 0 ]]; then
    OUTPUT_PATH=$OSB_SHARE_PATH
else
    OUTPUT_PATH=$LOCAL_MODEL_SAVE_PATH
fi

# ==================== parse CLI args ====================
MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/LLM/Qwen3-VL-8B-Instruct_202604071747/"
DEBUG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_obs_path)
            MODEL_OBS_PATH="$2"
            shift 2
            ;;
        --dataset_obs_path)
            DATASET_OBS_PATH="$2"
            shift 2
            ;;
        --debug)
            DEBUG="$2"
            shift 2
            ;;
        *)
            echo "Unknown param: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${DATASET_OBS_PATH:-}" ]]; then
    echo "ERROR: must provide --dataset_obs_path"
    exit 1
fi

echo "model_obs_path: $MODEL_OBS_PATH"
echo "dataset_obs_path: $DATASET_OBS_PATH"
echo "debug: $DEBUG"

# ==================== download model ====================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
DINOV2_PATH='/cache/dinov2-large'
FASTVLM_PATH='/cache/llava-fastvithd_1.5b_stage2'
mkdir -p $DINOV2_PATH $FASTVLM_PATH

python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/dinov2-large/', '$DINOV2_PATH')"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/llava-fastvithd_1.5b_stage2/', '$FASTVLM_PATH')"

# ==================== download dataset ====================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
DATASET_FOLDER='/cache/Data'
mkdir -p $DATASET_FOLDER
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_FOLDER}/dataset.zip')"
cd $DATASET_FOLDER
unzip -o dataset.zip
cd "$SCRIPT_DIR/.."

if [[ -f "${DATASET_FOLDER}/train.jsonl" ]]; then
    DATASET_PATH="${DATASET_FOLDER}"
else
    DATASET_PATH=$(find "${DATASET_FOLDER}" -maxdepth 1 -type d -exec test -f "{}/train.jsonl" \; -print -quit)
    if [[ -z "$DATASET_PATH" ]]; then
        echo "ERROR: train.jsonl not found in DATASET_FOLDER"
        exit 1
    fi
fi

echo "DATASET_PATH: $DATASET_PATH"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> finish moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ==================== training ====================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> start training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Current working path: $PWD"
export ROOT_IMAGE_DIR="$DATASET_PATH"
echo "ROOT_IMAGE_DIR: $ROOT_IMAGE_DIR"

TAR_EQUAL_BATCH_SIZE=${TAR_EQUAL_BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}

total_gpus=$(( NNODES * NPROC_PER_NODE ))
gas=$((TAR_EQUAL_BATCH_SIZE / (total_gpus * PER_DEVICE_BATCH_SIZE) ))
if [ $gas -lt 1 ]; then
    GRADIENT_ACCUMULATION=1
else
    GRADIENT_ACCUMULATION=$gas
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${EXT_ARGS} \
    -m llava.train.train_qwen \
    --model_name_or_path "${FASTVLM_PATH}" \
    --version qwen_2_centerline_coord \
    --unfreeze_mm_vision_tower False \
    --vision_tower "${DINOV2_PATH}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATASET_PATH}/train.jsonl" \
    --image_folder "${DATASET_PATH}/images" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_PATH}" \
    --num_train_epochs "${NUM_EPOCHS:-3}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --learning_rate "${LR:-2e-5}" \
    --mm_projector_lr "${MM_PROJECTOR_LR:-5e-5}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --model_max_length "${MODEL_MAX_LENGTH:-4096}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-1000}" \
    --evaluation_strategy no \
    --load_best_model_at_end False \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-10}" \
    --logging_steps 10 \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> training finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

if [ "$DEBUG" = true ]; then
    python ${SCRIPT_DIR}/tool.py 2>/dev/null || echo "no tool.py, skip"
else
    echo "[Training Stage] task finished"
fi
