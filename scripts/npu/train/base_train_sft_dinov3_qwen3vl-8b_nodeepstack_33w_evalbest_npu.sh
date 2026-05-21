#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU (Ascend) SFT recipe for larger 33w-scale data.
# Qwen3-VL-8B LLM (auto-extract) + DINOv3 + no DeepStack.
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
pip install "sentencepiece>=0.1.99"
pip install "tiktoken>=0.7.0"

# -------------------- core ML (step.sh) --------------------
pip install "transformers==4.56.2"
pip install "tokenizers>=0.22.0,<0.23.0"
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install "safetensors>=0.4.3"
pip install packaging
pip install "Pillow>=10.0.0"
pip install torchvision==0.22.1

# -------------------- llava project dependencies (from pyproject.toml) --------------------
pip install shortuuid
pip install "peft>=0.10.0"
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
pip install 'opencv-python-headless>=4.8.0'
pip install 'loguru>=0.7.0'
pip install 'shapely>=2.0.0'
pip install 'geopandas>=0.14.0'
pip install 'rasterio>=1.3.0'
pip install 'pyproj>=3.6.0'
pip install 'fiona>=1.9.0'
pip install wandb
pip install swanlab

pip install "huggingface-hub==0.36.2" --force-reinstall
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

# The cloud training filesystem is create-only in some jobs: avoid reusing an
# existing output root because checkpoint rotation/best replacement may need
# delete/overwrite permissions. Override RUN_ID if all nodes must share a fixed id.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}
LOCAL_MODEL_SAVE_PATH="${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}"
mkdir -p "${LOCAL_MODEL_SAVE_PATH}"

if [[ $NODE_RANK == 0 ]]; then
    OUTPUT_PATH="${OSB_SHARE_PATH%/}/${RUN_ID}"
else
    OUTPUT_PATH=$LOCAL_MODEL_SAVE_PATH
fi
echo "Run id: $RUN_ID"
echo "Output path: $OUTPUT_PATH"

# ====================== OBS paths ======================
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints"
DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip"

DINOV3_PATH=${DINOV3_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
Qwen3VL_PATH=${Qwen3VL_PATH:-${OBS_CACHE}/checkpoints/Qwen3-VL-8B-Instruct}

DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
DATASET_PATH="${DATASET_EXTRACT_ROOT}/MLLM20260427_rc_jjh"
IMAGE_FOLDER="${DATASET_PATH}"

# ====================== download ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading models >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov3-vitl16-pretrain-lvd1689m', '${DINOV3_PATH}')"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/Qwen3-VL-8B-Instruct', '${Qwen3VL_PATH}')"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"

mkdir -p "${DATASET_EXTRACT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Expected dataset directory $DATASET_PATH not found after unzip."
    ls -l "${DATASET_EXTRACT_ROOT}"
    exit 1
fi

# ====================== dataset paths ======================
# DATASET_PHASE selects the A/B state-update axis. MAP_TASK selects the output schema.
# New datasets are split at raw-sample level and already contain train/eval/test.
# If the dataset has phase_a/phase_b directories, use those; otherwise fall back
# to the legacy flat train.jsonl/eval.jsonl/test.jsonl layout.
DATASET_PHASE=${DATASET_PHASE:-phase_a}   # phase_a or phase_b.
MAP_TASK=${MAP_TASK:-lane}                # lane or lane_intersection; used by inference/eval scripts.

if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
    TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
    EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
    TEST_PATH="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
else
    TRAIN_PATH="${DATASET_PATH}/train.jsonl"
    EVAL_PATH="${DATASET_PATH}/eval.jsonl"
    TEST_PATH="${DATASET_PATH}/test.jsonl"
fi
EVAL_IMAGE_FOLDER="${IMAGE_FOLDER}"
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: $TRAIN_PATH not found"
    exit 1
fi
if [ ! -f "$TEST_PATH" ]; then
    echo "ERROR: $TEST_PATH not found"
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
cd "$SCRIPT_DIR/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ====================== training parameters ======================
# Edit this block for experiments. Each variable is passed to train_qwen.py below.
MM_VISION_SELECT_LAYER=-2              # --mm_vision_select_layer; -2 means use the penultimate ViT layer as main feature.
INPUT_IMAGE_SIZE=512                   # --input_image_size; 256 BEV patch -> 512, DINOv3-L patch16 -> 32x32=1024 visual tokens.
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
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10} # --save_total_limit; deletes old regular checkpoint-* dirs.
LOGGING_STEPS=10                       # --logging_steps.
EVAL_STEPS=500                         # --eval_steps; eval loss is computed during training when ENABLE_EVAL=True.
SAMPLE_SEED=42                         # --sample_seed.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False} # --save_best_train_loss.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000} # --best_train_loss_start_step.
BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best} # --best_train_loss_dir.
ENABLE_EVAL=${ENABLE_EVAL:-True}        # If True, run eval_loss on EVAL_PATH by steps.
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-True} # Maintain OUTPUT_PATH/eval_best.
BEST_EVAL_LOSS_DIR=${BEST_EVAL_LOSS_DIR:-eval_best}
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only} # New best creates a unique dir, then old successful best dirs are deleted.
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-1} # Keep only the latest successful best candidate by default.
USE_HF_PROGRESS_BAR=True               # --use_hf_progress_bar; True prints HF tqdm progress on console.
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False} # --swanlab_enable; enable SwanLab monitoring when True.
SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}  # Runtime API key; can be overridden by exporting SWANLAB_API_KEY.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3} # One SwanLab project for all SFT/GRPO runs.
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-} # Optional SwanLab workspace/org.
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_dinov3_nodeepstack} # Groups related runs in the same project.
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-sft} # SwanLab job type for filtering.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_33w_${DATASET_PHASE}_${MAP_TASK}_dinov3_qwen3vl8b_nodeepstack}
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,33w,${DATASET_PHASE},${MAP_TASK},dinov3,qwen3vl8b,nodeepstack}
SWANLAB_MODE=${SWANLAB_MODE:-}
export SWANLAB_API_KEY
EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")

EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    if [ ! -s "${EVAL_PATH}" ]; then
        echo "ERROR: ENABLE_EVAL=True but eval json is missing or empty: ${EVAL_PATH}"
        exit 1
    fi
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
echo "ViT:        ${DINOV3_PATH}"
echo "Input size: ${INPUT_IMAGE_SIZE}"
echo "DeepStack:  ${DEEPSTACK_LABEL}"
echo "Grad ckpt:  ${GRADIENT_CHECKPOINTING}"
echo "DeepSpeed:  ${DEEPSPEED_CONFIG}"
echo "LR:         llm=${LR}, projector=${MM_PROJECTOR_LR}, vision=${MM_VISION_TOWER_LR}"
echo "Best train: ${SAVE_BEST_TRAIN_LOSS}, start_step=${BEST_TRAIN_LOSS_START_STEP}, dir=${BEST_TRAIN_LOSS_DIR}"
echo "Eval:       ${ENABLE_EVAL}, eval_steps=${EVAL_STEPS}, best_eval=${SAVE_BEST_EVAL_LOSS}, dir=${BEST_EVAL_LOSS_DIR}"
echo "Best mode:  ${BEST_CHECKPOINT_SAVE_MODE}"
echo "Best keep:  ${BEST_CHECKPOINT_KEEP_LIMIT}"
echo "HF tqdm:    ${USE_HF_PROGRESS_BAR}"
echo "SwanLab:    ${SWANLAB_ENABLE}, project=${SWANLAB_PROJECT}, group=${SWANLAB_GROUP}, job=${SWANLAB_JOB_TYPE}, exp=${SWANLAB_EXPERIMENT_NAME}"
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
    --vision_tower "${DINOV3_PATH}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
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
    --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
    --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
    --use_hf_progress_bar "${USE_HF_PROGRESS_BAR}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --swanlab_enable "${SWANLAB_ENABLE}" \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_workspace "${SWANLAB_WORKSPACE}" \
    --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
    --swanlab_group "${SWANLAB_GROUP}" \
    --swanlab_job_type "${SWANLAB_JOB_TYPE}" \
    --swanlab_tags "${SWANLAB_TAGS}" \
    --swanlab_mode "${SWANLAB_MODE}" \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl \
    --deepspeed "${DEEPSPEED_CONFIG}"

echo "=== Training finished ==="
