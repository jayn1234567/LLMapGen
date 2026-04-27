#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Multi-GPU training (DDP via torchrun)
# Usage:
#   bash scripts/train_multi_gpu.sh
#
# Override defaults via env vars:
#   NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_multi_gpu.sh
# ============================================================

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/llava-fastvithd_1.5b_stage2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${CHECKPOINT_DIR}}
DATA_PATH=${DATA_PATH:-data/train.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/centerline_coord_ft_1.5b}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_EPOCHS=${NUM_EPOCHS:-3}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-5e-5}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
VERSION=${VERSION:-qwen_2_centerline_coord}
VISION_TOWER=${VISION_TOWER:-checkpoints/facebook_dinov2-large}
PYTHON_BIN=${PYTHON_BIN:-python}
TF32=${TF32:-False}

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "Batch: ${PER_DEVICE_BATCH_SIZE}/gpu x ${GRADIENT_ACCUMULATION} accum x ${NUM_GPUS} gpus = $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS)) total"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m llava.train.train_qwen \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version "${VERSION}" \
    --unfreeze_mm_vision_tower False \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy no \
    --load_best_model_at_end False \
    --save_total_limit 10 \
    --logging_steps 10 \
    --report_to none \
    --tf32 "${TF32}" \
    --ddp_find_unused_parameters False \
    --ddp_backend nccl
