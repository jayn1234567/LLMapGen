#!/usr/bin/env bash
set -euo pipefail

# Current production GRPO entry: DINOv2 + Qwen3VL-8B + no DeepStack + LoRA.
# Phase and task are independent:
#   phase_a/phase_b controls incoming hints.
#   lane/lane_intersection controls reward schema.
# Tokenizer handling is internal to mllm/model/builder.py: GRPO does not pass
# --tokenizer_use_fast and does not install tiktoken/sentencepiece at runtime.

SCRIPT_PATH=$(readlink -f "$0")
NPU_SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(cd "${NPU_SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# ---------- Runtime ----------
NPUS="0,1,2"
NPROC_PER_NODE=3
MASTER_PORT=29631
OBS_CACHE=${OBS_CACHE:-/cache}

# ---------- OBS / local paths ----------
MODEL_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints"
DATASET_OBS_PATH="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip"
SFT_CHECKPOINT_OBS="obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/outputs/sft_phase_a_lane/checkpoint-3000"

DINOV2_PATH=${DINOV2_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov2-large}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-${OBS_CACHE}/checkpoints/sft_qwen3vl_dinov2_nodeepstack}
DATASET_PATH=${DATASET_PATH:-${OBS_CACHE}/MLLM20260427_rc_jjh}
IMAGE_FOLDER="${DATASET_PATH}"

# ---------- Branch / data ----------
DATASET_PHASE="phase_a"                  # phase_a or phase_b.
MAP_TASK="lane"                          # lane or lane_intersection.
TRAINING_BRANCH="phase_a_lane"           # phase_a_lane, phase_b_lane, phase_a_lane_intersection, phase_b_lane_intersection.
COORD_MODE=auto                          # auto reads meta.coord_mode; new datasets use normalized 0-1000 coordinates.
COORD_RANGE=1000

# ---------- Model ----------
VERSION="conv_qwen_3_Dinov2_huawei"
INPUT_IMAGE_SIZE=518
DISABLE_DEEPSTACK=True
MODEL_MAX_LENGTH=4096
DEEPSPEED_CONFIG="scripts/deepspeed_zero3.json"

# ---------- GRPO sampling/reward ----------
NUM_GENERATIONS=4
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
KL_BETA=0.02
CLIP_RANGE=0.2

# For lane-only, infer_index centerline matching dominates the reward.
REWARD_FORMAT_WEIGHT=0.08
REWARD_CENTERLINE_INSTANCE_WEIGHT=0.37
REWARD_CENTERLINE_LENGTH_WEIGHT=0.45
REWARD_CUT_TYPE_WEIGHT=0.05
REWARD_CUT_CONTINUITY_WEIGHT=0.05
REWARD_INTERSECTION_WEIGHT=0.0
if [ "${MAP_TASK}" = "lane_intersection" ]; then
    MAX_NEW_TOKENS=768
    REWARD_FORMAT_WEIGHT=0.07
    REWARD_CENTERLINE_INSTANCE_WEIGHT=0.33
    REWARD_CENTERLINE_LENGTH_WEIGHT=0.42
    REWARD_CUT_TYPE_WEIGHT=0.04
    REWARD_CUT_CONTINUITY_WEIGHT=0.04
    REWARD_INTERSECTION_WEIGHT=0.10
fi

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

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading DINOv2 / SFT / dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov2-large', '${DINOV2_PATH}')"
python -c "import moxing as mox; mox.file.copy_parallel('${SFT_CHECKPOINT_OBS}', '${SFT_CHECKPOINT}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset.zip')"

cd "${OBS_CACHE}"
unzip -o dataset.zip
cd "${REPO_ROOT}"

if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
    TRAIN_JSONL="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
    EVAL_JSONL="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
    TEST_JSONL="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
else
    TRAIN_JSONL="${DATASET_PATH}/train.jsonl"
    EVAL_JSONL="${DATASET_PATH}/eval.jsonl"
    TEST_JSONL="${DATASET_PATH}/test.jsonl"
fi

OUTPUT_DIR="${CLUSTER_SAVE:-outputs/grpo_dinov2_qwen3vl_nodeepstack_${TRAINING_BRANCH}}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "GRPO NPU:      DINOv2 + Qwen3VL + no DeepStack"
echo "Branch:        ${TRAINING_BRANCH}"
echo "Train JSONL:   ${TRAIN_JSONL}"
echo "Eval JSONL:    ${EVAL_JSONL}"
echo "Final test:    ${TEST_JSONL}"
echo "Coords:        ${COORD_MODE} (range=${COORD_RANGE})"
echo "DeepSpeed:     ${DEEPSPEED_CONFIG}"
echo "Tokenizer:     slow/fallback in mllm/model/builder.py"
echo "Output:        ${OUTPUT_DIR}"
echo "============================================================"

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    -m mllm.train.train_grpo \
    --model_name_or_path "${SFT_CHECKPOINT}" \
    --version "${VERSION}" \
    --vision_tower "${DINOV2_PATH}" \
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
    --weight_decay "${WEIGHT_DECAY}" \
    --max_steps "${MAX_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --remove_unused_columns False \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl \
    --deepspeed "${DEEPSPEED_CONFIG}"
