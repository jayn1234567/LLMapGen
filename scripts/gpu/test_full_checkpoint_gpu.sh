#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# GPU full-checkpoint llava testing script
# - Downloads: dataset + trained checkpoint from specified OBS path
# - Uses an external ViT base model path during inference
# - Results uploaded to ${OUTPUT_URL}/test_results (platform injected)
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd $SCRIPT_DIR
echo "Script path: $SCRIPT_PATH"
echo "Script folder path: $SCRIPT_DIR"
echo "Current working path: $PWD"

# ====================== moxing upgrade ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> changing moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
pip uninstall moxing-framework -y
pip cache purge
pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
export MOX_PROFILE=1
export MOX_RECORD_OBS=1
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> moxing change finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== dependencies ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing dependencies >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pip install torch==2.7.1

pip install sentencepiece
pip install tiktoken

pip install "transformers>=4.51.0"
pip install "tokenizers>=0.21"
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install safetensors
pip install packaging
pip install Pillow
pip install torchvision==0.22.1

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

echo "========== key deps =========="
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
echo "==============================="

# ====================== output management (platform injected) ======================
if [[ -z "${OUTPUT_URL}" ]]; then
    echo "ERROR: OUTPUT_URL is not set."
    exit 1
fi

CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="$CLUSTER_SAVE"
OUTPUT_PATH=$OSB_SHARE_PATH

echo "Platform OUTPUT_URL: $OUTPUT_URL"
echo "OSB_SHARE_PATH: $OSB_SHARE_PATH"

# ====================== OBS paths ======================
OBS_CACHE=${OBS_CACHE:-/cache}

DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip"

# 【修改】全参训练输出目录（DeepSpeed merge 后的根目录，包含 model.safetensors）
TRAINED_CHECKPOINT_OBS="obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/05/06/c9017063151248669d7d57b48790b6a0/output/"

DATASET_PATH="/cache/MLLM20260427_rc_jjh"
IMAGE_FOLDER="${DATASET_PATH}"

# ====================== download dataset ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset.zip')"
cd /cache
unzip -o dataset.zip
cd $SCRIPT_DIR

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset dir not found."
    exit 1
fi

TEST_JSON="${DATASET_PATH}/test.jsonl"
if [ ! -f "$TEST_JSON" ]; then
    echo "ERROR: test.jsonl missing."
    exit 1
fi

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
echo "ViT base must be supplied separately during inference."

COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "TEST_JSON: $TEST_JSON"
echo "COORD_MODE: ${COORD_MODE} (range=${COORD_RANGE})"

# ====================== inference ======================
TEST_OUTPUT_LOCAL="/cache/test_output"
mkdir -p "$TEST_OUTPUT_LOCAL"

cd "$SCRIPT_DIR/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 8 GPU distributed inference
torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --test-json "${TEST_JSON}" \
  --num-samples -1 \
  --image-folder "${IMAGE_FOLDER}" \
  --prompt-mode dataset \
  --map-task lane \
  --patch-size 256 \
  --coord-mode "${COORD_MODE}" \
  --coord-range "${COORD_RANGE}" \
  --conv-template "conv_qwen_2_Dinov2_huawei" \
  --output-dir "${TEST_OUTPUT_LOCAL}" \
  --output-json "${TEST_OUTPUT_LOCAL}/summary.json" \
  --temperature 0.0 \
  --max-new-tokens 2048

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
        data = fp.read()
    for idx, obj_str in enumerate(extract_json_objects(data)):
        try:
            obj = json.loads(obj_str)
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

if [ -f "scripts/tools/visualize_centerline.py" ]; then
    python scripts/tools/visualize_centerline.py \
      --input-dir "${TEST_OUTPUT_LOCAL}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-dir "${TEST_OUTPUT_LOCAL}/viz"
fi

echo "=== Testing finished ==="

# ====================== upload results ======================
TEST_RESULT_OBS="${OUTPUT_URL}/test_results"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Uploading results to ${TEST_RESULT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${TEST_OUTPUT_LOCAL}', '${TEST_RESULT_OBS}')"
echo "Results saved to ${TEST_RESULT_OBS}"
