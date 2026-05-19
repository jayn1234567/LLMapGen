#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU (Ascend) SFT recipe for larger 33w-scale data.
# Qwen3-VL-8B LLM (auto-extract) + DINOv2 + no DeepStack.
# Default recipe:
# - Global batch 128, per-device batch 4.
# - 3 epochs as the first large-data run.
# - LLM/projector LR 2e-5, DINO vision tower LR 2e-6.
# - Weight decay 0.0, cosine scheduler, warmup ratio 0.03.
# - Eval every EVAL_STEPS and keep the best eval-loss checkpoint in eval_best/.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd $SCRIPT_DIR
echo "Script path: $SCRIPT_PATH"
echo "Script folder path: $SCRIPT_DIR"
echo "Current working path: $PWD"

# ====================== NPU environment ======================
export ASCEND_CUSTOM_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

workerID=$(echo $HOSTNAME | awk -F'-' '{print $(NF-1)"-"$NF}')
echo ${workerID}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
sudo chmod -R 777 /usr/local/Ascend/ascend-toolkit/
source /usr/local/Ascend/nnal/atb/set_env.sh

# ====================== moxing upgrade ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> changing moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
pip uninstall moxing-framework -y
pip cache purge
pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
export MOX_PROFILE=1
export MOX_RECORD_OBS=1
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> moxing change finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== dependencies (strictly from step.sh) ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing dependencies (step.sh) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pip install torch==2.7.1
pip install torch_npu==2.7.1rc1

python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

# -------------------- tokenizer prerequisites (before transformers) --------------------
pip install sentencepiece
pip install tiktoken

# -------------------- core ML (step.sh) --------------------
pip install "transformers>=4.51.0"
pip install "tokenizers>=0.21"
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install safetensors
pip install packaging
pip install Pillow
pip install torchvision==0.22.1

# -------------------- llava project dependencies (from pyproject.toml) --------------------
pip install shortuuid
pip install peft
pip install pydantic
pip install 'markdown2[all]'
pip install 'numpy>=1.26'
pip install 'scikit-learn>=1.2'
pip install 'gradio>=5.0'
pip install requests
pip install uvicorn
pip install fastapi
pip install 'einops>=0.6'
pip install 'einops-exts>=0.0.4'
pip install 'timm>=0.9.0'

pip install "huggingface-hub>=0.25.1" --force-reinstall
pip install urllib3==1.26.15

# -------------------- verification --------------------
echo "========== key deps =========="
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
echo "==============================="

pip list

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dependencies installed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== distributed parameters ======================
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

# ====================== HCCL & NPU settings ======================
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
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}

# ====================== output management ======================
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

# ====================== OBS paths ======================
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints"
DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip"

DINOV2_PATH=${DINOV2_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov2-large}
Qwen3VL_PATH=${Qwen3VL_PATH:-${OBS_CACHE}/checkpoints/Qwen3-VL-8B-Instruct}

DATASET_PATH="/cache/MLLM20260427_rc_jjh"
IMAGE_FOLDER="${DATASET_PATH}"

# ====================== download ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading models >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov2-large', '${DINOV2_PATH}')"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/Qwen3-VL-8B-Instruct', '${Qwen3VL_PATH}')"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset.zip')"

cd /cache
unzip -o dataset.zip
cd $SCRIPT_DIR

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Expected dataset directory $DATASET_PATH not found after unzip."
    ls -l /cache/
    exit 1
fi

# ====================== dataset split ======================
# DATASET_PHASE selects the A/B state-update axis. MAP_TASK selects the output schema.
# If the dataset has phase_a/phase_b directories, use those; otherwise fall back
# to the legacy flat train.jsonl/test.jsonl layout.
DATASET_PHASE=${DATASET_PHASE:-phase_a}   # phase_a or phase_b.
MAP_TASK=${MAP_TASK:-lane}                # lane or lane_intersection; used by inference/eval scripts.
EVAL_RATIO=${EVAL_RATIO:-0.2}             # Split this ratio from test as eval.
EVAL_COUNT=${EVAL_COUNT:--1}              # >=0 overrides EVAL_RATIO.
EVAL_SPLIT_SEED=${EVAL_SPLIT_SEED:-42}

if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
    TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
    TEST_SOURCE_PATH="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
else
    TRAIN_PATH="${DATASET_PATH}/train.jsonl"
    TEST_SOURCE_PATH="${DATASET_PATH}/test.jsonl"
fi
SPLIT_DIR="${OBS_CACHE}/eval_split/${DATASET_PHASE}_${MAP_TASK}"
mkdir -p "${SPLIT_DIR}"
python "${SCRIPT_DIR}/../../scripts/data/split_eval_from_test.py" \
    --test-json "${TEST_SOURCE_PATH}" \
    --output-test "${SPLIT_DIR}/test.jsonl" \
    --output-eval "${SPLIT_DIR}/eval.jsonl" \
    --eval-ratio "${EVAL_RATIO}" \
    --eval-count "${EVAL_COUNT}" \
    --seed "${EVAL_SPLIT_SEED}"
TEST_PATH="${SPLIT_DIR}/test.jsonl"
EVAL_PATH="${SPLIT_DIR}/eval.jsonl"
EVAL_IMAGE_FOLDER="${IMAGE_FOLDER}"
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: $TRAIN_PATH not found"
    exit 1
fi
if [ ! -f "$TEST_PATH" ]; then
    echo "ERROR: split test path $TEST_PATH not found"
    exit 1
fi

echo "DATASET_PATH: $DATASET_PATH"
echo "DATASET_PHASE:$DATASET_PHASE"
echo "MAP_TASK:     $MAP_TASK"
echo "TRAIN_PATH:   $TRAIN_PATH"
echo "EVAL_PATH:    $EVAL_PATH"
echo "TEST_PATH:    $TEST_PATH"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== batch size ======================
# Edit these values in this script. Do not pass them as one-off shell prefixes.
# Target total train batch = TARGET_GLOBAL_BATCH_SIZE.
# Actual total batch = NNODES * NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * gradient_accumulation_steps.
TARGET_GLOBAL_BATCH_SIZE=128
PER_DEVICE_TRAIN_BATCH_SIZE=4

total_gpus=$(( NNODES * NPROC_PER_NODE ))
micro_batch=$(( total_gpus * PER_DEVICE_TRAIN_BATCH_SIZE ))
gradient_accumulation_steps=$(( (TARGET_GLOBAL_BATCH_SIZE + micro_batch - 1) / micro_batch ))
if [ "${gradient_accumulation_steps}" -lt 1 ]; then
    gradient_accumulation_steps=1
fi
ACTUAL_GLOBAL_BATCH_SIZE=$(( micro_batch * gradient_accumulation_steps ))
if [ "${ACTUAL_GLOBAL_BATCH_SIZE}" -ne "${TARGET_GLOBAL_BATCH_SIZE}" ]; then
    echo "WARNING: actual global batch is ${ACTUAL_GLOBAL_BATCH_SIZE}, not target ${TARGET_GLOBAL_BATCH_SIZE}, because total_gpus*per_device=${micro_batch} does not divide the target."
fi

echo ">>> Target global batch: ${TARGET_GLOBAL_BATCH_SIZE}"
echo ">>> Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}, Total GPUs: ${total_gpus}"
echo ">>> Gradient accumulation steps: ${gradient_accumulation_steps}"
echo ">>> Actual global batch: ${ACTUAL_GLOBAL_BATCH_SIZE}"

# ====================== training ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd "$SCRIPT_DIR/../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ====================== training parameters ======================
# Edit this block for experiments. Each variable is passed to train_qwen.py below.
MM_VISION_SELECT_LAYER=-2              # --mm_vision_select_layer; -2 means use the penultimate ViT layer as main feature.
MM_PROJECTOR_TYPE=mlp2x_gelu           # --mm_projector_type.
UNFREEZE_MM_VISION_TOWER=True          # --unfreeze_mm_vision_tower; True means full-param ViT training.
DISABLE_DEEPSTACK=True                  # Fixed no-DeepStack mode.
GRADIENT_CHECKPOINTING=True            # --gradient_checkpointing; keep enabled for large full-param jobs.
DEEPSPEED_CONFIG="scripts/deepspeed_zero3.json" # --deepspeed; ZeRO3 with gathered checkpoint saves.
NUM_EPOCHS=3                           # --num_train_epochs; first 33w run targets roughly the old 3.4w/6ep update count.
LR=2e-5                                # --learning_rate for LLM and any non-special trainable modules.
MM_PROJECTOR_LR=2e-5                   # --mm_projector_lr; projector follows the LLM LR.
MM_VISION_TOWER_LR=2e-6                # --mm_vision_tower_lr; keep DINO updates conservative during full-param finetune.
WEIGHT_DECAY=0.0                       # --weight_decay; keep disabled for this full-param Qwen3VL+DINO recipe.
WARMUP_RATIO=0.03                      # --warmup_ratio; about 232 warmup steps for 33w data, batch 128, 3 epochs.
LR_SCHEDULER_TYPE=cosine               # --lr_scheduler_type.
MODEL_MAX_LENGTH=4096                  # --model_max_length.
SAVE_STEPS=500                         # --save_steps; should be aligned with EVAL_STEPS when keeping eval_best.
SAVE_TOTAL_LIMIT=10                    # --save_total_limit.
LOGGING_STEPS=10                       # --logging_steps.
EVAL_STEPS=500                         # --eval_steps; eval loss is computed during training when ENABLE_EVAL=True.
SAMPLE_SEED=42                         # --sample_seed.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False} # --save_best_train_loss.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000} # --best_train_loss_start_step.
BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best} # --best_train_loss_dir.
ENABLE_EVAL=${ENABLE_EVAL:-True}        # If True, run eval_loss on EVAL_PATH by steps.
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-True} # Maintain OUTPUT_PATH/eval_best.
BEST_EVAL_LOSS_DIR=${BEST_EVAL_LOSS_DIR:-eval_best}
USE_HF_PROGRESS_BAR=True               # --use_hf_progress_bar; True prints HF tqdm progress on console.
EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")

EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && [ -s "${EVAL_PATH}" ]; then
    EVAL_ARGS=(
        --eval_data_path "${EVAL_PATH}"
        --eval_image_folder "${EVAL_IMAGE_FOLDER}"
        "${EVAL_STRATEGY_ARG}" steps
        --eval_steps "${EVAL_STEPS}"
        --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}"
        --best_eval_loss_dir "${BEST_EVAL_LOSS_DIR}"
    )
fi

# ---------- no DeepStack ----------
DEEPSTACK_ARGS=(--disable_deepstack True)
DEEPSTACK_LABEL="disabled"

echo "============================================================"
echo "Model:      ${Qwen3VL_PATH} (Qwen3-VL-8B auto-extract LLM)"
echo "ViT:        ${DINOV2_PATH}"
echo "DeepStack:  ${DEEPSTACK_LABEL}"
echo "Grad ckpt:  ${GRADIENT_CHECKPOINTING}"
echo "DeepSpeed:  ${DEEPSPEED_CONFIG}"
echo "LR:         llm=${LR}, projector=${MM_PROJECTOR_LR}, vision=${MM_VISION_TOWER_LR}"
echo "Best train: ${SAVE_BEST_TRAIN_LOSS}, start_step=${BEST_TRAIN_LOSS_START_STEP}, dir=${BEST_TRAIN_LOSS_DIR}"
echo "Eval:       ${ENABLE_EVAL}, eval_steps=${EVAL_STEPS}, best_eval=${SAVE_BEST_EVAL_LOSS}, dir=${BEST_EVAL_LOSS_DIR}"
echo "HF tqdm:    ${USE_HF_PROGRESS_BAR}"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_qwen \
    --model_name_or_path "${Qwen3VL_PATH}" \
    --version conv_qwen_3_Dinov2_huawei \
    --vision_tower "${DINOV2_PATH}" \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
    "${DEEPSTACK_ARGS[@]}" \
    --data_path "${TRAIN_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    "${EVAL_ARGS[@]}" \
    --sample_seed "${SAMPLE_SEED}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_PATH}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
    --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
    --best_train_loss_dir "${BEST_TRAIN_LOSS_DIR}" \
    --use_hf_progress_bar "${USE_HF_PROGRESS_BAR}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl \
    --deepspeed "${DEEPSPEED_CONFIG}"

echo "=== Training finished ==="
