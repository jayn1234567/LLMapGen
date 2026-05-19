#!/usr/bin/env bash
set -euo pipefail

# 单卡 GRPO smoke test: DINOv2 + Qwen3VL + no DeepStack + lane/intersection.
# 路口样本来自 debug 数据中的 synthetic intersection，用于验证训练和解析链路。

# ---------- Environment ----------
CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
GPU_ID=2

# ---------- Model paths ----------
SFT_CHECKPOINT="/media/q/data2/jjh/project/MLLM_project/outputs/test_qwen3vl"
VISION_TOWER="/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large"

# ---------- Data paths ----------
TRAIN_JSONL="data/grpo_debug_lane_intersection20/train.jsonl"
TEST_JSONL="data/grpo_debug_lane_intersection20/test.jsonl"
IMAGE_FOLDER="data/av2_patch_256_fullimage_cutflag_test_v2"
OUTPUT_DIR="outputs/grpo_debug_lane_intersection_dinov2_qwen3vl_nodeepstack_gpu"
INFER_DIR="${OUTPUT_DIR}/infer_test"

# ---------- Model config ----------
VERSION="conv_qwen_3_Dinov2_huawei"
INPUT_IMAGE_SIZE=518
DISABLE_DEEPSTACK=True
MODEL_MAX_LENGTH=2048
TRAINING_BRANCH="auto_lane_intersection"  # Debug data is task-only; use phase_a/b_lane_intersection for A/B JSONL.
MAP_TASK="lane_intersection"
COORD_MODE=auto      # auto reads meta.coord_mode; new datasets use normalized 0-1000 coordinates.
COORD_RANGE=1000

# ---------- GRPO sampling/reward ----------
NUM_GENERATIONS=2
MAX_NEW_TOKENS=384
TEMPERATURE=0.7
TOP_P=0.9
KL_BETA=0.02
CLIP_RANGE=0.2

# lane+intersection 任务下，infer_index 线匹配仍是主信号，路口奖励只作辅助。
REWARD_FORMAT_WEIGHT=0.07
REWARD_CENTERLINE_INSTANCE_WEIGHT=0.33
REWARD_CENTERLINE_LENGTH_WEIGHT=0.42
REWARD_CUT_TYPE_WEIGHT=0.04
REWARD_CUT_CONTINUITY_WEIGHT=0.04
REWARD_INTERSECTION_WEIGHT=0.10

# ---------- LoRA config ----------
LORA_TARGET_SCOPE="llm"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# ---------- Training config ----------
MAX_STEPS=1
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-6
BF16=True
LOGGING_STEPS=1
SAVE_STEPS=1
SAVE_TOTAL_LIMIT=1

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

[ -d "${SFT_CHECKPOINT}" ] || { echo "SFT checkpoint not found: ${SFT_CHECKPOINT}"; exit 1; }
[ -d "${VISION_TOWER}" ] || { echo "Vision tower not found: ${VISION_TOWER}"; exit 1; }
[ -f "${TRAIN_JSONL}" ] || { echo "Train JSONL not found: ${TRAIN_JSONL}"; exit 1; }
[ -f "${TEST_JSONL}" ] || { echo "Test JSONL not found: ${TEST_JSONL}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "GRPO lane+intersection debug: DINOv2 + Qwen3VL + no DeepStack"
echo "GPU:       ${CUDA_VISIBLE_DEVICES}"
echo "Model:     ${SFT_CHECKPOINT}"
echo "ViT:       ${VISION_TOWER}"
echo "Train:     ${TRAIN_JSONL}"
echo "Output:    ${OUTPUT_DIR}"
echo "Coords:    ${COORD_MODE} (range=${COORD_RANGE})"
echo "============================================================"

python -m mllm.train.train_grpo \
    --model_name_or_path "${SFT_CHECKPOINT}" \
    --version "${VERSION}" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack "${DISABLE_DEEPSTACK}" \
    --data_path "${TRAIN_JSONL}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio pad \
    --training_branch "${TRAINING_BRANCH}" \
    --map_task "${MAP_TASK}" \
    --coord_mode "${COORD_MODE}" \
    --coord_range "${COORD_RANGE}" \
    --output_dir "${OUTPUT_DIR}" \
    --grpo_backend custom \
    --num_generations "${NUM_GENERATIONS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --kl_beta "${KL_BETA}" \
    --clip_range "${CLIP_RANGE}" \
    --reward_format_weight "${REWARD_FORMAT_WEIGHT}" \
    --reward_centerline_instance_weight "${REWARD_CENTERLINE_INSTANCE_WEIGHT}" \
    --reward_centerline_length_weight "${REWARD_CENTERLINE_LENGTH_WEIGHT}" \
    --reward_cut_type_weight "${REWARD_CUT_TYPE_WEIGHT}" \
    --reward_cut_continuity_weight "${REWARD_CUT_CONTINUITY_WEIGHT}" \
    --reward_intersection_weight "${REWARD_INTERSECTION_WEIGHT}" \
    --lora_enable True \
    --lora_target_scope "${LORA_TARGET_SCOPE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_steps "${MAX_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --remove_unused_columns False \
    --report_to none

python scripts/infer_centerline_checkpoint.py \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoint-1" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack \
    --test-json "${TEST_JSONL}" \
    --image-folder "${IMAGE_FOLDER}" \
    --num-samples 2 \
    --prompt-mode dataset \
    --map-task lane_intersection \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template "${VERSION}" \
    --device cuda \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --output-dir "${INFER_DIR}" \
    --output-json "${INFER_DIR}/summary.json" \
    --eval-centerline
