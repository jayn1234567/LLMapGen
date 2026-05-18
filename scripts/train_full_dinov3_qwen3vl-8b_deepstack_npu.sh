#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU (Ascend) llava training + eval script
# Qwen3-VL-8B LLM (auto-extract) + DINOv3 + DeepStack
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
export LLAVA_LOG_RANK0_ONLY=${LLAVA_LOG_RANK0_ONLY:-1}

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

DINOV3_PATH=${DINOV3_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
Qwen3VL_PATH=${Qwen3VL_PATH:-${OBS_CACHE}/checkpoints/Qwen3-VL-8B-Instruct}

DATASET_PATH="/cache/MLLM20260427_rc_jjh"
IMAGE_FOLDER="${DATASET_PATH}"

# ====================== download ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading models >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov3-vitl16-pretrain-lvd1689m', '${DINOV3_PATH}')"
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

TRAIN_PATH="${DATASET_PATH}/train.jsonl"
TEST_PATH="${DATASET_PATH}/test.jsonl"
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: $TRAIN_PATH not found"
    exit 1
fi

echo "DATASET_PATH: $DATASET_PATH"
echo "TRAIN_PATH:   $TRAIN_PATH"
echo "TEST_PATH:    $TEST_PATH"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== batch size ======================
# Edit these values in this script. Do not pass them as one-off shell prefixes.
# Target total train batch = TARGET_GLOBAL_BATCH_SIZE.
# Actual total batch = NNODES * NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * gradient_accumulation_steps.
TARGET_GLOBAL_BATCH_SIZE=32
PER_DEVICE_TRAIN_BATCH_SIZE=2

total_gpus=$(( NNODES * NPROC_PER_NODE ))
gas=$((TARGET_GLOBAL_BATCH_SIZE / (total_gpus * PER_DEVICE_TRAIN_BATCH_SIZE) ))
if [ $gas -lt 1 ]; then
    gradient_accumulation_steps=1
else
    gradient_accumulation_steps=$gas
fi

echo ">>> Target global batch: ${TARGET_GLOBAL_BATCH_SIZE}"
echo ">>> Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}, Total GPUs: ${total_gpus}"
echo ">>> Gradient accumulation steps: ${gradient_accumulation_steps}"

# ====================== training ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ====================== training parameters ======================
# Edit this block for experiments. Each variable is passed to train_qwen.py below.
MM_VISION_SELECT_LAYER=-2              # --mm_vision_select_layer; -2 means use the penultimate ViT layer as main feature.
MM_PROJECTOR_TYPE=mlp2x_gelu           # --mm_projector_type.
UNFREEZE_MM_VISION_TOWER=True          # --unfreeze_mm_vision_tower; True means full-param ViT training.
DEEPSTACK_VISUAL_INDEXES="6 12 18 23"  # --deepstack_visual_indexes; fixed DeepStack layers for DINOv3-L.
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-False}  # Internal wrapper override used by the no-deepstack script.
GRADIENT_CHECKPOINTING=True            # --gradient_checkpointing; keep enabled for large full-param jobs.
DEEPSPEED_CONFIG="scripts/deepspeed_zero3.json" # --deepspeed; ZeRO3 with gathered checkpoint saves.
NUM_EPOCHS=6                           # --num_train_epochs.
LR=2e-5                                # --learning_rate.
MM_PROJECTOR_LR=2e-5                   # --mm_projector_lr; set empty in command block if not needed.
WEIGHT_DECAY=0.1                       # --weight_decay.
WARMUP_STEPS=100                       # --warmup_steps.
LR_SCHEDULER_TYPE=cosine               # --lr_scheduler_type.
MODEL_MAX_LENGTH=4096                  # --model_max_length.
SAVE_STEPS=300                         # --save_steps.
SAVE_TOTAL_LIMIT=10                    # --save_total_limit.
LOGGING_STEPS=10                       # --logging_steps.
SAMPLE_SEED=42                         # --sample_seed.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False} # --save_best_train_loss; train_best wrapper sets this True.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000} # --best_train_loss_start_step.
BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best} # --best_train_loss_dir.
USE_HF_PROGRESS_BAR=True               # --use_hf_progress_bar; True prints HF tqdm progress on console.

# ---------- DeepStack ----------
DEEPSTACK_ARGS=()
if [[ "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    DEEPSTACK_ARGS=(--disable_deepstack True)
    DEEPSTACK_LABEL="disabled"
elif [ -n "${DEEPSTACK_VISUAL_INDEXES}" ]; then
    DEEPSTACK_ARGS=(--disable_deepstack False --deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
    DEEPSTACK_LABEL="${DEEPSTACK_VISUAL_INDEXES}"
else
    DEEPSTACK_LABEL="disabled"
fi

echo "============================================================"
echo "Model:      ${Qwen3VL_PATH} (Qwen3-VL-8B → auto-extract LLM)"
echo "ViT:        ${DINOV3_PATH}"
echo "DeepStack:  ${DEEPSTACK_LABEL}"
echo "Grad ckpt:  ${GRADIENT_CHECKPOINTING}"
echo "DeepSpeed:  ${DEEPSPEED_CONFIG}"
echo "Best train: ${SAVE_BEST_TRAIN_LOSS}, start_step=${BEST_TRAIN_LOSS_START_STEP}, dir=${BEST_TRAIN_LOSS_DIR}"
echo "HF tqdm:    ${USE_HF_PROGRESS_BAR}"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m llava.train.train_qwen \
    --model_name_or_path "${Qwen3VL_PATH}" \
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
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_steps "${WARMUP_STEPS}" \
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

# # ====================== inference ======================
# echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
# cd "$SCRIPT_DIR/.."

# # 👇 只在主节点跑推理
# if [ ${NODE_RANK} -ne 0 ]; then
#     echo "✅ Skip inference on non-master node"
#     exit 0
# fi

# # 👇 主节点使用单机8卡推理（正确）
# if [ -f "$TEST_PATH" ] && [ -d "$IMAGE_FOLDER" ]; then
#     echo ">>> Running inference on ${TEST_PATH}"

#     torchrun --nproc_per_node=8 \
#         --master_addr=127.0.0.1 \
#         --master_port=29501 \
#         scripts/infer_centerline_checkpoint.py \
#         --checkpoint-dir "${OUTPUT_PATH}" \
#         --test-json "${TEST_PATH}" \
#         --image-folder "${IMAGE_FOLDER}" \
#         --num-samples -1 \
#         --conv-template conv_qwen_3_Dinov2_huawei \
#         --device npu \
#         --max-new-tokens 2048 \
#         --output-json "${OUTPUT_PATH}/summary.json" \
#         --output-dir "${OUTPUT_PATH}/predictions" \
#         --print-full-output
# else
#     echo ">>> No test.jsonl found, skipping inference"
# fi

# TEST_OUTPUT_LOCAL="${OUTPUT_PATH}/predictions"

# # ===================== 【自动合并 rank 文件】 =====================
# echo "🔗 正在合并所有 summary_rank*.json → summary.json"
# python3 - << EOF
# import json, glob, os
# output_dir = "$TEST_OUTPUT_LOCAL"
# files = sorted(glob.glob(os.path.join(output_dir, "summary_rank*.json")))
# merged = []
# for f in files:
#     with open(f, "r", encoding="utf-8") as fp:
#         for line in fp:
#             line = line.strip()
#             if line:
#                 merged.append(json.loads(line))
# merged.sort(key=lambda x: x.get("idx", 0))
# with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fp:
#     for item in merged:
#         fp.write(json.dumps(item, ensure_ascii=False) + "\n")
# print(f"✅ 合并完成，共 {len(merged)} 条记录")
# EOF
# # ==================================================================

# if [ -f "scripts/visualize_centerline.py" ]; then
#     python scripts/visualize_centerline.py \
#       --input-dir "${OUTPUT_PATH}/predictions" \
#       --image-folder "${IMAGE_FOLDER}" \
#       --output-dir "${TEST_OUTPUT_LOCAL}/viz"
# fi

# echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> inference finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
