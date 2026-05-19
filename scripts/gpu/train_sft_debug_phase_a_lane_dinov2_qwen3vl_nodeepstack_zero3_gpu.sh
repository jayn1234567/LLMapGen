#!/usr/bin/env bash
set -euo pipefail

# SFT Phase A smoke: DINOv2 + Qwen3VL + no DeepStack + ZeRO-3.
# Phase A uses empty incoming hints and tests ordinary patch recognition training/inference.

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
GPU_IDS=0,2
NUM_GPUS=2
MASTER_PORT=29651

MODEL_NAME_OR_PATH="/media/q/data2/jjh/project/MLLM_project/outputs/test_qwen3vl"
VISION_TOWER="/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large"
TRAIN_JSONL="data/debug_phase_a_lane20/train.jsonl"
TEST_JSONL="data/debug_phase_a_lane20/test.jsonl"
IMAGE_FOLDER="data/av2_patch_256_fullimage_cutflag_test_v2"
OUTPUT_DIR="outputs/sft_debug_phase_a_lane_dinov2_qwen3vl_nodeepstack_zero3_gpu"
INFER_DIR="${OUTPUT_DIR}/infer_test"

VERSION="conv_qwen_3_Dinov2_huawei"
INPUT_IMAGE_SIZE=518
DISABLE_DEEPSTACK=True
DEEPSPEED_CONFIG="scripts/deepspeed_zero3.json"

MAX_STEPS=1
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-6
MM_PROJECTOR_LR=1e-6
WEIGHT_DECAY=0.0
MODEL_MAX_LENGTH=2048
BF16=True
COORD_MODE=auto      # auto reads meta.coord_mode; new datasets use normalized 0-1000 coordinates.
COORD_RANGE=1000

LORA_TARGET_SCOPE="llm"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

[ -d "${MODEL_NAME_OR_PATH}" ] || { echo "Model not found: ${MODEL_NAME_OR_PATH}"; exit 1; }
[ -d "${VISION_TOWER}" ] || { echo "Vision tower not found: ${VISION_TOWER}"; exit 1; }
[ -f "${TRAIN_JSONL}" ] || { echo "Train JSONL not found: ${TRAIN_JSONL}"; exit 1; }
[ -f "${TEST_JSONL}" ] || { echo "Test JSONL not found: ${TEST_JSONL}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "SFT Phase A ZeRO3 debug: DINOv2 + Qwen3VL + no DeepStack"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "Train:     ${TRAIN_JSONL}"
echo "Output:    ${OUTPUT_DIR}"
echo "Coords:    ${COORD_MODE} (range=${COORD_RANGE})"
echo "============================================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_sft \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version "${VERSION}" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack "${DISABLE_DEEPSTACK}" \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower False \
    --data_path "${TRAIN_JSONL}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio pad \
    --output_dir "${OUTPUT_DIR}" \
    --lora_enable True \
    --lora_target_scope "${LORA_TARGET_SCOPE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_steps "${MAX_STEPS}" \
    --warmup_steps 0 \
    --lr_scheduler_type constant \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps 1 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf_progress_bar True \
    --report_to none \
    --tf32 False \
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
    --map-task lane \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template "${VERSION}" \
    --device cuda \
    --max-new-tokens 256 \
    --temperature 0 \
    --output-dir "${INFER_DIR}" \
    --output-json "${INFER_DIR}/summary.json" \
    --eval-centerline
