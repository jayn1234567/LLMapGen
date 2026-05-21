#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# infer_dinov2_centerline_gpu.sh
# GPU 推理测试: 加载 DINOv2 + Qwen checkpoint，输出中心线 JSON + 可视化
# ============================================================

# ---------- Paths ----------
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

CHECKPOINT_DIR=outputs/dinov2_qwen2_1.5b
IMAGE=data/test.png
IMAGE_FOLDER=""
TEST_JSON=""
# 0 means all records in TEST_JSON. Set a positive value only for a quick debug subset.
NUM_SAMPLES=0
SAMPLE_OFFSET=0
PROMPT_MODE=default
COORD_MODE=auto
COORD_RANGE=1000
CONV_TEMPLATE=conv_qwen_2_Dinov2_huawei
PROMPT="<image>\nPredict the complete road map from the current patch in the BEV image."
OUTPUT_JSON=outputs/prediction.json
OUTPUT_DIR=outputs/predictions

DEVICE=cuda
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
PRINT_FULL_OUTPUT=true

# ====================== env ======================
CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Image:      ${IMAGE:-<from test_json>}"
echo "Template:   ${CONV_TEMPLATE}"
echo "Coords:     ${COORD_MODE} (range=${COORD_RANGE})"
echo "Device:     ${DEVICE}"
echo "============================================================"

python scripts/tools/infer_centerline_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --image "${IMAGE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --test-json "${TEST_JSON}" \
    --num-samples "${NUM_SAMPLES}" \
    --sample-offset "${SAMPLE_OFFSET}" \
    --prompt-mode "${PROMPT_MODE}" \
    --map-task lane \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template "${CONV_TEMPLATE}" \
    --prompt "${PROMPT}" \
    --device "${DEVICE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --output-json "${OUTPUT_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --print-full-output
