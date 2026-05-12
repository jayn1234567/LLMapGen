#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# train_llm_align_dinov3_qwen2-1.5b_freeze-vit_gpu.sh
# 单卡训练: DINOv3-L + Qwen2-1.5B + DeepStack, freeze ViT
# ============================================================

# ---------- Paths ----------
MODEL_NAME_OR_PATH=checkpoints/llava-fastvithd_1.5b_stage2
VISION_TOWER=checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m
DATA_PATH=data/train.jsonl
IMAGE_FOLDER=data/images
OUTPUT_DIR=outputs/dinov3_qwen2_1.5b

# ---------- Model ----------
VERSION=conv_qwen_2_Dinov2_huawei
MM_VISION_SELECT_LAYER=-2
MM_PROJECTOR_TYPE=mlp2x_gelu
MM_VISION_SELECT_FEATURE=patch
UNFREEZE_MM_VISION_TOWER=False
DEEPSTACK_VISUAL_INDEXES="6 12 18 23"

# ---------- Training ----------
CUDA_VISIBLE_DEVICES=0
NUM_EPOCHS=3
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
LR=2e-5
MM_PROJECTOR_LR=5e-5
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE=cosine
MODEL_MAX_LENGTH=4096

# ---------- Checkpoint ----------
SAVE_STRATEGY=steps
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=10
LOGGING_STEPS=10
SAMPLE_SEED=42

# ---------- DeepSpeed (set to "scripts/deepspeed_zero2.json" to enable) ----------
DEEPSPEED_CONFIG=""
# DEEPSPEED_CONFIG="scripts/deepspeed_zero2.json"

# ====================== env ======================
CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

[ -d "${MODEL_NAME_OR_PATH}" ] || { echo "Model not found: ${MODEL_NAME_OR_PATH}"; exit 1; }
[ -f "${DATA_PATH}" ] || { echo "Data not found: ${DATA_PATH}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ====================== args ======================
DEEPSTACK_ARGS=()
[ -n "${DEEPSTACK_VISUAL_INDEXES}" ] && DEEPSTACK_ARGS=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})

DEEPSPEED_CMD=()
[ -n "${DEEPSPEED_CONFIG}" ] && DEEPSPEED_CMD=(--deepspeed "${DEEPSPEED_CONFIG}")

echo "============================================================"
echo "Model:    ${MODEL_NAME_OR_PATH}"
echo "ViT:      ${VISION_TOWER}"
echo "Version:  ${VERSION}"
echo "DeepStack:${DEEPSTACK_VISUAL_INDEXES:-disabled}"
echo "DeepSpeed:${DEEPSPEED_CONFIG:-disabled}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

python -m llava.train.train_qwen \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version "${VERSION}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
    --mm_vision_select_feature "${MM_VISION_SELECT_FEATURE}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
    "${DEEPSTACK_ARGS[@]}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --sample_seed "${SAMPLE_SEED}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --tf32 False \
    "${DEEPSPEED_CMD[@]}"
