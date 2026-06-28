#!/usr/bin/env bash
#SBATCH --job-name=llmapgen-dinov2-smoke
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mingli01/project/jn/LLMapGen
#SBATCH --output=slurm-logs/%j-dinov2-centerline-gpu-smoke.out
#SBATCH --error=slurm-logs/%j-dinov2-centerline-gpu-smoke.err

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/file_storage01/home/mingli/project/jn/UniMapGen/.condaenvs/llamafactory-cu128-dtok-resume-20260326}"
export PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python3}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONUSERBASE || true
unset PYTHONHOME || true

REPO_ROOT="${REPO_ROOT:-/mingli01/project/jn/LLMapGen}"
cd "${REPO_ROOT}"
mkdir -p slurm-logs

export TRAINROOT="${TRAINROOT:-${REPO_ROOT}/data_samples/dinov2_centerline_tiny_trainroot}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mingli01/models/Qwen3-8B}"
export DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-/mingli01/data/ckpts/dinov2-large}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/dinov2_qwen3_8b_gpu_smoke}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export USE_TORCHRUN="${USE_TORCHRUN:-true}"
if [ -z "${MASTER_PORT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    else
        export MASTER_PORT=29501
    fi
else
    export MASTER_PORT
fi
export MAX_SAMPLES="${MAX_SAMPLES:-2}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1}"
export MAX_STEPS="${MAX_STEPS:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export BF16="${BF16:-true}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-false}"
export FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
export VISION_TRAIN_LAST_N_LAYERS="${VISION_TRAIN_LAST_N_LAYERS:-0}"

echo "=== LLMapGen DINOv2 centerline GPU smoke ==="
echo "date=$(date)"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "env_prefix=${ENV_PREFIX}"
echo "python_bin=${PYTHON_BIN}"
echo "master_port=${MASTER_PORT}"
echo "trainroot=${TRAINROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "model=${MODEL_NAME_OR_PATH}"
echo "dinov2=${DINOV2_MODEL_NAME_OR_PATH}"
echo "visual_encoder_checkpoint=${VISUAL_ENCODER_CHECKPOINT_PATH:-}"
echo "bridge_modules_state=${BRIDGE_MODULES_STATE_PATH:-}"
echo "freeze_language_model=${FREEZE_LANGUAGE_MODEL}"
echo "freeze_vision_encoder=${FREEZE_VISION_ENCODER}"
echo "vision_train_last_n_layers=${VISION_TRAIN_LAST_N_LAYERS}"
nvidia-smi || true

sh scripts/gpu/train/train_dinov2_centerline_qwen_lora_gpu.sh

echo "=== smoke finished at $(date) ==="
