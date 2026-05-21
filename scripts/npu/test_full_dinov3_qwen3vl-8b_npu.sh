#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU (Ascend) full-parameter finetune llava testing script
# - Downloads: dataset + trained checkpoint from specified OBS path
# - Downloads the external ViT base model and passes it to inference explicitly
# - Results uploaded to ${OUTPUT_URL}/test_results (platform injected)
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

# ====================== dependencies ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing dependencies >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pip install torch==2.7.1
pip install torch_npu==2.7.1rc1

python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

pip install "sentencepiece>=0.1.99"
pip install "tiktoken>=0.7.0"

pip install "transformers==4.56.2"
pip install "tokenizers>=0.22.0,<0.23.0"
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install "safetensors>=0.4.3"
pip install packaging
pip install "Pillow>=10.0.0"
pip install torchvision==0.22.1

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

echo "========== key deps =========="
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
echo "==============================="

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
if [[ -z "${OUTPUT_URL}" ]]; then
    echo "ERROR: OUTPUT_URL is not set."
    exit 1
fi

CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="$CLUSTER_SAVE"
echo "System defined obs share path: $OSB_SHARE_PATH"

LOCAL_MODEL_SAVE_PATH='/cache/local_model_save_path'
mkdir -p $LOCAL_MODEL_SAVE_PATH

# ====================== OBS paths ======================
OBS_CACHE=${OBS_CACHE:-/cache}

MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints"
DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip"

TRAINED_CHECKPOINT_OBS="obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/05/11/91d5397b1e5e4d21a16177081527425e/output/checkpoint-2400"

DINOV3_PATH=${DINOV3_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}

DEEPSTACK_ARGS=()
if [[ "${DISABLE_DEEPSTACK:-False}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    DEEPSTACK_ARGS=(--disable_deepstack)
    DEEPSTACK_LABEL="disabled"
elif [ -n "${DEEPSTACK_VISUAL_INDEXES}" ]; then
    DEEPSTACK_ARGS=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
    DEEPSTACK_LABEL="override ${DEEPSTACK_VISUAL_INDEXES}"
else
    DEEPSTACK_LABEL="from checkpoint config"
fi

DATASET_PATH="/cache/MLLM20260427_rc_jjh"
IMAGE_FOLDER="${DATASET_PATH}"

# ====================== download dataset ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov3-vitl16-pretrain-lvd1689m', '${DINOV3_PATH}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset.zip')"

cd /cache
unzip -o dataset.zip
cd $SCRIPT_DIR

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Expected dataset directory $DATASET_PATH not found after unzip."
    ls -l /cache/
    exit 1
fi

# Use the dataset's prebuilt raw-sample-level split. Eval is no longer carved
# out of test at runtime.
DATASET_PHASE=${DATASET_PHASE:-phase_a}
MAP_TASK=${MAP_TASK:-lane}
COORD_MODE=${COORD_MODE:-auto}     # auto reads meta.coord_mode; new datasets use normalized 0-1000 coordinates.
COORD_RANGE=${COORD_RANGE:-1000}
if [ -f "${DATASET_PATH}/${DATASET_PHASE}/test.jsonl" ]; then
    TEST_JSON="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
    EVAL_JSON="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
else
    TEST_JSON="${DATASET_PATH}/test.jsonl"
    EVAL_JSON="${DATASET_PATH}/eval.jsonl"
fi
if [ ! -f "$TEST_JSON" ]; then
    echo "ERROR: test json missing: $TEST_JSON"
    exit 1
fi

echo "DATASET_PATH: $DATASET_PATH"
echo "TEST_JSON: $TEST_JSON"
echo "EVAL_JSON: $EVAL_JSON"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== download trained checkpoint ======================
CHECKPOINT_LOCAL="/cache/train_output"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading checkpoint from ${TRAINED_CHECKPOINT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${TRAINED_CHECKPOINT_OBS}', '${CHECKPOINT_LOCAL}')"

CHECKPOINT_DIR="${CHECKPOINT_LOCAL}"

echo "Using checkpoint root directory: $CHECKPOINT_DIR"
echo "Checkpoint directory contents:"
ls -la ${CHECKPOINT_DIR}

if [ ! -f "${CHECKPOINT_DIR}/model.safetensors" ] && [ ! -f "${CHECKPOINT_DIR}/pytorch_model.bin" ]; then
    echo "ERROR: No model.safetensors or pytorch_model.bin found."
    exit 1
fi
echo "ViT base is loaded from DINOV3_PATH and passed with --vision_tower."

NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0} # 0 means all final-test rows after eval split.

echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "DINOV3_PATH:    $DINOV3_PATH"
echo "INPUT_IMAGE_SIZE: ${INPUT_IMAGE_SIZE}"
echo "MAP_TASK:       $MAP_TASK"
echo "COORD_MODE:     ${COORD_MODE} (range=${COORD_RANGE})"
echo "NUM_TEST_SAMPLES: ${NUM_TEST_SAMPLES} (0 means all final-test rows)"
echo "DeepStack:      ${DEEPSTACK_LABEL}"
echo "DeepStack is auto-detected from checkpoint config unless DISABLE_DEEPSTACK or DEEPSTACK_VISUAL_INDEXES is set."

# ====================== inference ======================
TEST_OUTPUT_LOCAL="/cache/test_output"
SAMPLE_JSON_DIR="${TEST_OUTPUT_LOCAL}/json"
VIZ_DIR="${TEST_OUTPUT_LOCAL}/viz"
METRICS_JSON="${TEST_OUTPUT_LOCAL}/eval.json"
mkdir -p "$TEST_OUTPUT_LOCAL" "$SAMPLE_JSON_DIR" "$VIZ_DIR"

cd "$SCRIPT_DIR/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/infer_centerline_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --vision_tower "${DINOV3_PATH}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    "${DEEPSTACK_ARGS[@]}" \
    --test-json "${TEST_JSON}" \
    --num-samples "${NUM_TEST_SAMPLES}" \
    --image-folder "${IMAGE_FOLDER}" \
    --prompt-mode dataset \
    --map-task "${MAP_TASK}" \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template "conv_qwen_3_Dinov2_huawei" \
    --output-dir "${TEST_OUTPUT_LOCAL}" \
    --sample-json-dir "${SAMPLE_JSON_DIR}" \
    --output-json "${TEST_OUTPUT_LOCAL}/summary.json" \
    --temperature 0.0 \
    --max-new-tokens 2048

echo "=== Testing finished ==="

if [ "${NODE_RANK}" -ne 0 ]; then
    echo "Skip merge/upload on non-master node ${NODE_RANK}"
    exit 0
fi

# ===================== 【合并 rank 文件】 =====================
echo "Merging summary_rank*.json -> summary.json"
export TEST_OUTPUT_LOCAL="${TEST_OUTPUT_LOCAL}"
python3 - << 'EOF'
import json, glob, os, sys

def extract_json_objects(content):
    brace_count = 0
    start = None
    in_string = False
    escape = False
    i = 0
    length = len(content)
    while i < length:
        ch = content[i]
        if not escape and ch == '\\':
            escape = True
            i += 1
            continue
        if not escape and ch == '"':
            in_string = not in_string
        escape = False
        if not in_string:
            if ch == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    obj_str = content[start:i+1]
                    yield obj_str
                    start = None
        i += 1

output_dir = os.environ.get("TEST_OUTPUT_LOCAL", "/cache/test_output")
files = sorted(glob.glob(os.path.join(output_dir, "summary_rank*.json")))
if not files:
    print("No summary_rank*.json files found")
    sys.exit(1)

merged = []
bad = 0
for f in files:
    with open(f, "r", encoding="utf-8-sig") as fp:
        data = fp.read().strip()
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list):
            merged.extend(parsed)
            continue
        if isinstance(parsed, dict):
            merged.append(parsed)
            continue
    except json.JSONDecodeError:
        pass
    for idx, obj_str in enumerate(extract_json_objects(data)):
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "record_id" in obj:
                merged.append(obj)
        except json.JSONDecodeError as e:
            bad += 1
            print(f"Parse failed {f} obj{idx}: {e}", file=sys.stderr)

if not merged:
    print("No valid JSON objects parsed", file=sys.stderr)
    sys.exit(1)

merged.sort(key=lambda x: x.get("idx", 0))
out_path = os.path.join(output_dir, "summary.json")
with open(out_path, "w", encoding="utf-8") as fp:
    for item in merged:
        fp.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Merged: {len(merged)} records, skipped {bad} invalid objects")
EOF

if [ $? -ne 0 ]; then
    echo "Merge failed, skipping visualization"
    exit 1
fi
# ==========================================

if [ -f "scripts/visualize_centerline.py" ]; then
    python scripts/visualize_centerline.py \
      --input-dir "${TEST_OUTPUT_LOCAL}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-dir "${VIZ_DIR}" \
      --eval-output-json "${METRICS_JSON}"
fi

# ====================== upload results ======================
TEST_RESULT_OBS="${OUTPUT_URL}/test_results"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Uploading results to ${TEST_RESULT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${TEST_OUTPUT_LOCAL}', '${TEST_RESULT_OBS}')"
echo "Results saved to ${TEST_RESULT_OBS}"
