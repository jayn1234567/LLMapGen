#!/usr/bin/env bash
set -euo pipefail

# GRPO ZeRO-3 smoke: DINOv2 + Qwen3VL + no DeepStack + lane_intersection.
# Uses Phase-style lane+intersection JSONL and validates checkpoint inference.

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
GPU_IDS=0,2
NUM_GPUS=2
MASTER_PORT=29664

SFT_CHECKPOINT="/media/q/data2/jjh/project/MLLM_project/outputs/test_qwen3vl"
VISION_TOWER="/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large"
TRAIN_JSONL="data/grpo_debug_lane_intersection20/train.jsonl"
TEST_JSONL="data/grpo_debug_lane_intersection20/test.jsonl"
IMAGE_FOLDER="data/av2_patch_256_fullimage_cutflag_test_v2"
OUTPUT_DIR="outputs/grpo_debug_lane_intersection_dinov2_qwen3vl_nodeepstack_zero3_gpu"
INFER_DIR="${OUTPUT_DIR}/infer_test"

VERSION="conv_qwen_3_Dinov2_huawei"
INPUT_IMAGE_SIZE=518
DISABLE_DEEPSTACK=True
MODEL_MAX_LENGTH=2048
DEEPSPEED_CONFIG="scripts/deepspeed_zero3.json"
TRAINING_BRANCH="auto_lane_intersection"  # Debug data is task-only; use phase_a/b_lane_intersection for A/B JSONL.
MAP_TASK="lane_intersection"
COORD_MODE=auto      # auto reads meta.coord_mode; new datasets use normalized 0-1000 coordinates.
COORD_RANGE=1000

NUM_GENERATIONS=2
MAX_NEW_TOKENS=384
TEMPERATURE=0.7
TOP_P=0.9
# ZeRO-3 LoRA uses the same DeepSpeed-wrapped policy with LoRA adapters disabled
# as the reference model. Full-parameter ZeRO-3 still requires KL_BETA=0.0.
KL_BETA=0.02
CLIP_RANGE=0.2

# Keep infer_index centerline matching as the main signal. Intersection reward
# is enabled only for lane_intersection and stays secondary.
REWARD_FORMAT_WEIGHT=0.07
REWARD_CENTERLINE_INSTANCE_WEIGHT=0.33
REWARD_CENTERLINE_LENGTH_WEIGHT=0.42
REWARD_CUT_TYPE_WEIGHT=0.04
REWARD_CUT_CONTINUITY_WEIGHT=0.04
REWARD_INTERSECTION_WEIGHT=0.10

LORA_TARGET_SCOPE="llm"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

MAX_STEPS=1
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-6
BF16=True

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

[ -d "${SFT_CHECKPOINT}" ] || { echo "SFT checkpoint not found: ${SFT_CHECKPOINT}"; exit 1; }
[ -d "${VISION_TOWER}" ] || { echo "Vision tower not found: ${VISION_TOWER}"; exit 1; }
[ -f "${TRAIN_JSONL}" ] || { echo "Train JSONL not found: ${TRAIN_JSONL}"; exit 1; }
[ -f "${TEST_JSONL}" ] || { echo "Test JSONL not found: ${TEST_JSONL}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "GRPO lane+intersection ZeRO3 debug: DINOv2 + Qwen3VL + no DeepStack"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "Train:     ${TRAIN_JSONL}"
echo "Output:    ${OUTPUT_DIR}"
echo "Coords:    ${COORD_MODE} (range=${COORD_RANGE})"
echo "============================================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_grpo \
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
    --logging_steps 1 \
    --save_steps 1 \
    --save_total_limit 1 \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --remove_unused_columns False \
    --report_to none \
    --ddp_find_unused_parameters False \
    --deepspeed "${DEEPSPEED_CONFIG}"

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
