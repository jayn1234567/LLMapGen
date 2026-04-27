#!/usr/bin/env bash
set -euo pipefail

TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-outputs/centerline_coord_ft_1.5b}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${TRAIN_OUTPUT_DIR}/best}
TEST_JSON=${TEST_JSON:-data/test.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
NUM_SAMPLES=${NUM_SAMPLES:-50}
SAMPLE_OFFSET=${SAMPLE_OFFSET:-0}
CUDA_DEVICE=${CUDA_DEVICE:-2}
PROMPT_VERSION=${PROMPT_VERSION:-qwen_2_centerline_coord}
OUTPUT_DIR=${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}_eval_${NUM_SAMPLES}_offset${SAMPLE_OFFSET}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --test-json "${TEST_JSON}" \
  --image-folder "${IMAGE_FOLDER}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-offset "${SAMPLE_OFFSET}" \
  --prompt-mode dataset \
  --conv-template "${PROMPT_VERSION}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/summary.json" \
  --temperature 0.0 \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python scripts/summarize_centerline_eval.py \
  --summary-json "${OUTPUT_DIR}/summary.json"
