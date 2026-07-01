#!/usr/bin/env bash
#SBATCH --job-name=gl-dino2-qwen
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --chdir=/mingli01/project/jn/LLMapGen
#SBATCH --output=slurm-logs/%j-dinov2-qwen-global-local.out
#SBATCH --error=slurm-logs/%j-dinov2-qwen-global-local.err

set -euo pipefail

if [ -f /usr/share/lmod/lmod/init/bash ]; then
  source /usr/share/lmod/lmod/init/bash
fi
if command -v module >/dev/null 2>&1; then
  export MODULEPATH=/persist_data/modulefiles:/share/modulefiles
  module load conda/3 || true
  module load cuda/13.0 || true
fi

REPO_ROOT="${REPO_ROOT:-/mingli01/project/jn/LLMapGen}"
ENV_PREFIX="${ENV_PREFIX:-/file_storage01/home/mingli/project/jn/UniMapGen/.condaenvs/llamafactory-cu128-dtok-resume-20260326}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/file_storage01/home/mingli/data/ckpts/Qwen3-8B}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-/file_storage01/home/mingli/data/ckpts/dinov2-large}"
VISUAL_ENCODER_CKPT="${VISUAL_ENCODER_CKPT:-/mingli01/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/latest.pt}"
BRIDGE_MODULES="${BRIDGE_MODULES:-/mingli01/data/outputs/stage3_rc_nonqwen_modules_from21176_ckpt21000_20260427/rc_dinov2_centerline_json_modules.pt}"

TRAINROOT="${TRAINROOT:-/mingli01/data/outputs/rc_clean_dataset_xiuzheng_joint512_trainroot_douglas_merge6h22_nolinefix_20260506}"
CONTEXT_IMAGE_KEY="${CONTEXT_IMAGE_KEY:-context_image}"
GLOBAL_LOCAL_TRAINROOT="${GLOBAL_LOCAL_TRAINROOT:-/mingli01/data/outputs/rc_clean_dataset_xiuzheng_joint512_global_local1024_resize512_trainroot}"
SMOKE_GLOBAL_LOCAL_TRAINROOT="${SMOKE_GLOBAL_LOCAL_TRAINROOT:-/mingli01/data/outputs/llmapgen_global_local_smoke_${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-/mingli01/data/outputs/llmapgen_dinov2_qwen3_8b_global_local1024_resize512_seginit_align_6epoch_${SLURM_JOB_ID:-manual}}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/mingli01/data/outputs/llmapgen_dinov2_qwen3_8b_global_local_smoke_${SLURM_JOB_ID:-manual}}"

RUN_MODE="${RUN_MODE:-smoke_then_full}"
PREPARE_GLOBAL_LOCAL="${PREPARE_GLOBAL_LOCAL:-true}"
SMOKE_MAX_TRAIN_SAMPLES="${SMOKE_MAX_TRAIN_SAMPLES:-16}"
SMOKE_MAX_EVAL_SAMPLES="${SMOKE_MAX_EVAL_SAMPLES:-4}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-3}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-6}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MASTER_PORT="${MASTER_PORT:-$((21000 + ${SLURM_JOB_ID:-1234} % 20000))}"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/file_storage01/home/mingli/data/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/file_storage01/home/mingli/data/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/file_storage01/home/mingli/data/hf_cache/transformers}"
export TORCH_HOME="${TORCH_HOME:-/persist_data/home/mingli/.cache/torch}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_cache_${SLURM_JOB_ID:-manual}}"
unset PYTHONPATH || true
unset PYTHONUSERBASE || true
unset PYTHONHOME || true

cd "${REPO_ROOT}"
mkdir -p slurm-logs "${TRITON_CACHE_DIR}" "$(dirname "${OUTPUT_DIR}")"

echo "[global-local] date=$(date)"
echo "[global-local] host=$(hostname)"
echo "[global-local] repo=${REPO_ROOT}"
echo "[global-local] branch=$(git branch --show-current 2>/dev/null || true)"
echo "[global-local] python=${PYTHON_BIN}"
echo "[global-local] model=${MODEL_NAME_OR_PATH}"
echo "[global-local] tokenizer=${TOKENIZER_NAME_OR_PATH}"
echo "[global-local] dinov2=${DINOV2_MODEL_NAME_OR_PATH}"
echo "[global-local] seg_dinov2_ckpt=${VISUAL_ENCODER_CKPT}"
echo "[global-local] bridge_modules=${BRIDGE_MODULES}"
echo "[global-local] trainroot=${TRAINROOT}"
echo "[global-local] global_local_trainroot=${GLOBAL_LOCAL_TRAINROOT}"
echo "[global-local] output_dir=${OUTPUT_DIR}"
echo "[global-local] run_mode=${RUN_MODE}"

test -d "${MODEL_NAME_OR_PATH}"
test -d "${DINOV2_MODEL_NAME_OR_PATH}"
test -f "${VISUAL_ENCODER_CKPT}"
test -f "${BRIDGE_MODULES}"
test -f "${TRAINROOT}/train.jsonl"
test -f "${TRAINROOT}/meta_train.jsonl"
test -f "${TRAINROOT}/val.jsonl"
test -f "${TRAINROOT}/meta_val.jsonl"

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m py_compile \
  unimapgen/data/rc_centerline_json_sft_dataset.py \
  unimapgen/models/qwen3_rc_dinov2_centerline_json_sft.py \
  unimapgen/dinov2_centerline/train.py \
  unimapgen/dinov2_centerline/predict.py \
  scripts/tools/add_global_local_context.py \
  scripts/train_dinov2_centerline.py

nvidia-smi || true

prepare_context_trainroot() {
  local output_trainroot="$1"
  shift
  PYTHONNOUSERSITE=1 "${PYTHON_BIN}" scripts/tools/add_global_local_context.py \
    --trainroot "${TRAINROOT}" \
    --output-trainroot "${output_trainroot}" \
    --context-image-key "${CONTEXT_IMAGE_KEY}" \
    --context-size 1024 \
    --output-size 512 \
    --local-patch-size 512 \
    "$@"
}

run_train() {
  local nproc="$1"
  local trainroot="$2"
  local output_dir="$3"
  local max_steps="$4"
  local epochs="$5"
  local save_strategy="$6"
  local eval_strategy="$7"
  local master_port="$8"
  local best_args=()
  if [ "${save_strategy}" != "no" ] && [ "${eval_strategy}" != "no" ]; then
    best_args+=(--load-best-model-at-end --metric-for-best-model eval_loss --no-greater-is-better)
  fi
  PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${nproc}" \
    --master_port="${master_port}" \
    scripts/train_dinov2_centerline.py \
    --trainroot "${trainroot}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}" \
    --dinov2-model-name-or-path "${DINOV2_MODEL_NAME_OR_PATH}" \
    --visual-encoder-checkpoint-path "${VISUAL_ENCODER_CKPT}" \
    --bridge-modules-state-path "${BRIDGE_MODULES}" \
    --output-dir "${output_dir}" \
    --num-train-epochs "${epochs}" \
    --max-steps "${max_steps}" \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 2 \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay 0.0 \
    --warmup-ratio 0.03 \
    --logging-steps 1 \
    --save-strategy "${save_strategy}" \
    --save-steps "${SAVE_STEPS}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    --evaluation-strategy "${eval_strategy}" \
    --eval-steps "${EVAL_STEPS}" \
    --dataloader-num-workers 0 \
    --cutoff-len 7168 \
    --image-size 512 \
    --encoder-input-pad-size 518 \
    --visual-projector-hidden-dim 4096 \
    --geometric-mlp-hidden-dim 512 \
    --token-alignment-hidden-dim 4096 \
    --token-alignment-num-layers 2 \
    --token-alignment-dropout 0.0 \
    --model-dtype bfloat16 \
    --freeze-vision-encoder \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --gradient-checkpointing \
    --no-ddp-find-unused-parameters \
    --map-task lane \
    --use-global-local-views \
    --global-local-view-count 2 \
    --context-image-key "${CONTEXT_IMAGE_KEY}" \
    --require-context-image \
    --use-view-type-embedding \
    --view-type-embedding-count 2 \
    --view-type-embedding-init-std 0.02 \
    --local-files-only \
    --bf16 \
    --device-backend cuda \
    --ddp-backend nccl \
    "${best_args[@]}"
}

if [ "${RUN_MODE}" = "smoke" ] || [ "${RUN_MODE}" = "smoke_then_full" ]; then
  rm -rf "${SMOKE_GLOBAL_LOCAL_TRAINROOT}" "${SMOKE_OUTPUT_DIR}"
  prepare_context_trainroot "${SMOKE_GLOBAL_LOCAL_TRAINROOT}" \
    --max-train-samples "${SMOKE_MAX_TRAIN_SAMPLES}" \
    --max-eval-samples "${SMOKE_MAX_EVAL_SAMPLES}"
  run_train 1 "${SMOKE_GLOBAL_LOCAL_TRAINROOT}" "${SMOKE_OUTPUT_DIR}" "${SMOKE_MAX_STEPS}" 1 "no" "no" "$((MASTER_PORT + 1))"
fi

if [ "${RUN_MODE}" = "full" ] || [ "${RUN_MODE}" = "smoke_then_full" ]; then
  if [ "${PREPARE_GLOBAL_LOCAL}" = "true" ]; then
    prepare_context_trainroot "${GLOBAL_LOCAL_TRAINROOT}"
  fi
  run_train 8 "${GLOBAL_LOCAL_TRAINROOT}" "${OUTPUT_DIR}" -1 "${NUM_TRAIN_EPOCHS}" "steps" "steps" "${MASTER_PORT}"
fi

echo "[global-local] finished at $(date)"
