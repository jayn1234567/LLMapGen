#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU (Ascend) llava training + eval script
# Qwen2 + DINOv3 + DeepStack, freeze ViT
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
LLM_PATH=${LLM_PATH:-${OBS_CACHE}/checkpoints/llava-fastvithd_1.5b_stage2}

DATASET_PATH="/cache/MLLM20260427_rc_jjh"
IMAGE_FOLDER="${DATASET_PATH}"

# ====================== download ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading models >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov3-vitl16-pretrain-lvd1689m', '${DINOV3_PATH}')"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/llava-fastvithd_1.5b_stage2', '${LLM_PATH}')"

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

# ====================== auto gradient accumulation ======================
tar_equal_batch_size=64
per_device_train_batch_size=2

total_gpus=$(( NNODES * NPROC_PER_NODE ))
gas=$((tar_equal_batch_size / (total_gpus * per_device_train_batch_size) ))
if [ $gas -lt 1 ]; then
    gradient_accumulation_steps=1
else
    gradient_accumulation_steps=$gas
fi

echo ">>> Target global batch: ${tar_equal_batch_size}"
echo ">>> Per-device batch: ${per_device_train_batch_size}, Total GPUs: ${total_gpus}"
echo ">>> Gradient accumulation steps: ${gradient_accumulation_steps}"

# ====================== training ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd "$SCRIPT_DIR/../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---------- Training params ----------
MM_VISION_SELECT_LAYER=-2
INPUT_IMAGE_SIZE=512
MM_PROJECTOR_TYPE=mlp2x_gelu
UNFREEZE_MM_VISION_TOWER=False
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}
DEEPSPEED_CONFIG="scripts/deepspeed_zero3.json"
NUM_EPOCHS=6
LR=2e-5
MM_PROJECTOR_LR=2e-5
WEIGHT_DECAY=0.0
WARMUP_STEPS=50
LR_SCHEDULER_TYPE=cosine
MODEL_MAX_LENGTH=4096
WARMUP_STEPS=50
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=10
LOGGING_STEPS=10
SAMPLE_SEED=42

# ---------- DeepStack ----------
DEEPSTACK_ARGS=()
if [[ "${DISABLE_DEEPSTACK:-False}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    DEEPSTACK_ARGS=(--disable_deepstack True)
    DEEPSTACK_LABEL="disabled"
    GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
elif [ -n "${DEEPSTACK_VISUAL_INDEXES}" ]; then
    DEEPSTACK_ARGS=(--disable_deepstack False --deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
    DEEPSTACK_LABEL="${DEEPSTACK_VISUAL_INDEXES}"
    GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
else
    DEEPSTACK_LABEL="disabled"
    GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
fi

echo "============================================================"
echo "Model:      ${LLM_PATH}"
echo "ViT:        ${DINOV3_PATH}"
echo "Input size: ${INPUT_IMAGE_SIZE}"
echo "DeepStack:  ${DEEPSTACK_LABEL}"
echo "DeepSpeed:  ${DEEPSPEED_CONFIG}"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_qwen \
    --model_name_or_path "${LLM_PATH}" \
    --version conv_qwen_2_Dinov2_huawei \
    --vision_tower "${DINOV3_PATH}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
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
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-True}" \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl \
    --deepspeed "${DEEPSPEED_CONFIG}"

echo "=== Training finished ==="

# # ====================== inference ======================
# echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
# cd "$SCRIPT_DIR/../.."

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
#         --conv-template conv_qwen_2_Dinov2_huawei \
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
# echo "🔗 正在使用鲁棒合并处理所有 summary_rank*.json → summary.json"
# export TEST_OUTPUT_LOCAL="${TEST_OUTPUT_LOCAL}"
# python3 - << 'EOF'
# import json, glob, os, sys

# def extract_json_objects(content):
#     brace_count = 0
#     start = None
#     in_string = False
#     escape = False
#     i = 0
#     length = len(content)
#     while i < length:
#         ch = content[i]
#         if not escape and ch == '\\':
#             escape = True
#             i += 1
#             continue
#         if not escape and ch == '"':
#             in_string = not in_string
#         escape = False
#         if not in_string:
#             if ch == '{':
#                 if brace_count == 0:
#                     start = i
#                 brace_count += 1
#             elif ch == '}':
#                 brace_count -= 1
#                 if brace_count == 0 and start is not None:
#                     obj_str = content[start:i+1]
#                     yield obj_str
#                     start = None
#         i += 1

# output_dir = os.environ.get("TEST_OUTPUT_LOCAL", "/cache/test_output")
# files = sorted(glob.glob(os.path.join(output_dir, "summary_rank*.json")))
# if not files:
#     print("❌ 未找到任何 summary_rank*.json 文件")
#     sys.exit(1)

# merged = []
# bad = 0
# for f in files:
#     with open(f, "r", encoding="utf-8-sig") as fp:
#         data = fp.read()
#     for idx, obj_str in enumerate(extract_json_objects(data)):
#         try:
#             obj = json.loads(obj_str)
#             merged.append(obj)
#         except json.JSONDecodeError as e:
#             bad += 1
#             print(f"⚠️ 解析失败 {f} 对象{idx}: {e}", file=sys.stderr)

# if not merged:
#     print("❌ 没有解析到任何有效 JSON 对象", file=sys.stderr)
#     sys.exit(1)

# merged.sort(key=lambda x: x.get("idx", 0))
# out_path = os.path.join(output_dir, "summary.json")
# with open(out_path, "w", encoding="utf-8") as fp:
#     for item in merged:
#         fp.write(json.dumps(item, ensure_ascii=False) + "\n")

# print(f"✅ 合并完成，有效记录 {len(merged)} 条，跳过 {bad} 条无效对象")
# EOF
# # ==================================================================

# if [ -f "scripts/visualize_centerline.py" ]; then
#     python scripts/visualize_centerline.py \
#       --input-dir "${OUTPUT_PATH}/predictions" \
#       --image-folder "${IMAGE_FOLDER}" \
#       --output-dir "${TEST_OUTPUT_LOCAL}/viz"
# fi

# echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> inference finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
