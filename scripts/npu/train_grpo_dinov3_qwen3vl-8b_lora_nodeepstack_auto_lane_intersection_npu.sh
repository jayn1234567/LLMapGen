#!/usr/bin/env bash
set -euo pipefail

# GRPO lane+intersection RL finetuning for the generic MLLM framework.
# Edit the values below in this script before launching.

# ---------- Runtime ----------
NPUS="0,1,2"
NPROC_PER_NODE=3
MASTER_PORT=29632

# ---------- Paths ----------
SFT_CHECKPOINT="outputs/sft_lane_intersection/checkpoint-3000"
VISION_TOWER="checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m"
TRAIN_JSONL="data/train_lane_intersection_cut.jsonl"
IMAGE_FOLDER="data/images"
OUTPUT_DIR="outputs/grpo_lane_intersection_dinov3_qwen3vl8b_lora"

# ---------- Model ----------
VERSION="conv_qwen_3_Dinov2_huawei"
INPUT_IMAGE_SIZE=512                      # 256 BEV patch -> 512, DINOv3-L patch16 -> 32x32=1024 visual tokens.
DEEPSTACK_VISUAL_INDEXES=""
DISABLE_DEEPSTACK=True

# ---------- GRPO ----------
TRAINING_BRANCH="auto_lane_intersection"  # Use phase_a/b_lane_intersection when TRAIN_JSONL is A/B data.
MAP_TASK="lane_intersection"              # Predict centerlines and intersection polygons.
COORD_MODE=auto                           # auto reads meta.coord_mode; new datasets use normalized 0-1000 coordinates.
COORD_RANGE=1000
NUM_GENERATIONS=4
MAX_NEW_TOKENS=768                        # Usually longer than lane-only JSON.
TEMPERATURE=0.7
TOP_P=0.9
KL_BETA=0.02
CLIP_RANGE=0.2

# ---------- Reward Weights ----------
# Keep infer_index centerline matching as the main signal. Intersection reward
# is enabled only for lane_intersection and stays secondary.
REWARD_FORMAT_WEIGHT=0.07
REWARD_CENTERLINE_INSTANCE_WEIGHT=0.33
REWARD_CENTERLINE_LENGTH_WEIGHT=0.42
REWARD_CUT_TYPE_WEIGHT=0.04
REWARD_CUT_CONTINUITY_WEIGHT=0.04
REWARD_INTERSECTION_WEIGHT=0.10

# ---------- LoRA ----------
LORA_TARGET_SCOPE="llm"
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.05

# ---------- Optimization ----------
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-6
WEIGHT_DECAY=0.0
MAX_STEPS=1000
LOGGING_STEPS=1
SAVE_STEPS=100
BF16=True
MODEL_MAX_LENGTH=4096

DEEPSTACK_ARGS=()
if [[ "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    DEEPSTACK_ARGS=(--disable_deepstack True)
elif [ -n "${DEEPSTACK_VISUAL_INDEXES}" ]; then
    DEEPSTACK_ARGS=(--disable_deepstack False --deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    -m mllm.train.train_grpo \
    --model_name_or_path "${SFT_CHECKPOINT}" \
    --version "${VERSION}" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    "${DEEPSTACK_ARGS[@]}" \
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
    --weight_decay "${WEIGHT_DECAY}" \
    --max_steps "${MAX_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --remove_unused_columns False
