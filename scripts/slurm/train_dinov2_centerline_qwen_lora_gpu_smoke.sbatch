#!/usr/bin/env bash
#SBATCH --job-name=llmapgen-dinov2-smoke
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/file_storage01/home/mingli/project/jn/UniMapGen_douglas_dino_sft_worktree
#SBATCH --output=slurm-logs/%j-dinov2-centerline-gpu-smoke.out
#SBATCH --error=slurm-logs/%j-dinov2-centerline-gpu-smoke.err

set -euo pipefail

if [ -f /persist_data/apps/miniconda3/etc/profile.d/conda.sh ]; then
  source /persist_data/apps/miniconda3/etc/profile.d/conda.sh
else
  eval "$(/persist_data/apps/miniconda3/bin/conda shell.bash hook)"
fi
conda activate "${CONDA_ENV_NAME:-unimapgen}"

REPO_ROOT="${REPO_ROOT:-/file_storage01/home/mingli/project/jn/UniMapGen_douglas_dino_sft_worktree}"
cd "${REPO_ROOT}"
mkdir -p slurm-logs

export TRAINROOT="${TRAINROOT:-${REPO_ROOT}/data_samples/dinov2_centerline_tiny_trainroot}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mingli01/data/ckpts/Qwen3-8B}"
export DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-/file_storage01/home/mingli/data/ckpts/dinov2-large}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/dinov2_qwen3_8b_gpu_smoke}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
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

echo "=== LLMapGen DINOv2 centerline GPU smoke ==="
echo "date=$(date)"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "trainroot=${TRAINROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "model=${MODEL_NAME_OR_PATH}"
echo "dinov2=${DINOV2_MODEL_NAME_OR_PATH}"
nvidia-smi || true

sh scripts/gpu/train/train_dinov2_centerline_qwen_lora_gpu.sh

echo "=== smoke finished at $(date) ==="
