#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/llava-fastvithd_1.5b_stage2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${CHECKPOINT_DIR}}
DATA_PATH=${DATA_PATH:-data/train.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-""}
EVAL_IMAGE_FOLDER=${EVAL_IMAGE_FOLDER:-""}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/centerline_coord_ft_1.5b}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
NUM_EPOCHS=${NUM_EPOCHS:-3}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-5e-5}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
VERSION=${VERSION:-conv_qwen_2_Dinov2_huawei}
VISION_TOWER=${VISION_TOWER:-checkpoints/facebook_dinov2-large}
PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_SAMPLE_LIMIT=${EVAL_SAMPLE_LIMIT:-256}
SAMPLE_SEED=${SAMPLE_SEED:-42}
TF32=${TF32:-False}

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
    echo "Model checkpoint not found: ${MODEL_NAME_OR_PATH}"
    exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "Training json not found: ${DATA_PATH}"
    exit 1
fi

if [ ! -d "${IMAGE_FOLDER}" ]; then
    echo "Image folder not found: ${IMAGE_FOLDER}"
    exit 1
fi

if [ ! -f "${EVAL_DATA_PATH}" ]; then
    echo "Eval json not found: ${EVAL_DATA_PATH}"
    exit 1
fi

if [ ! -d "${EVAL_IMAGE_FOLDER}" ]; then
    echo "Eval image folder not found: ${EVAL_IMAGE_FOLDER}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Using PYTHONPATH=${PYTHONPATH}"
"${PYTHON_BIN}" - <<'PY'
import llava
print("Using llava from:", llava.__file__)
PY

"${PYTHON_BIN}" -m llava.train.train_qwen \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version "${VERSION}" \
    --unfreeze_mm_vision_tower False \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --eval_image_folder "${EVAL_IMAGE_FOLDER}" \
    --eval_sample_limit "${EVAL_SAMPLE_LIMIT}" \
    --sample_seed "${SAMPLE_SEED}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
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
    --tf32 "${TF32}"
