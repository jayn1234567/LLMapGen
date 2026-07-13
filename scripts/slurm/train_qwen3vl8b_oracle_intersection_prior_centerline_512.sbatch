#!/usr/bin/env bash
#SBATCH --job-name=oracle-int-cl
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --exclude=gpu49
#SBATCH --chdir=/mingli01/project/jn/LLMapGen
#SBATCH --output=slurm-logs/%j-qwen3vl8b-oracle-intersection-centerline.out
#SBATCH --error=slurm-logs/%j-qwen3vl8b-oracle-intersection-centerline.err

set -euo pipefail

source ~/.bashrc || true

REPO_ROOT="${REPO_ROOT:-/mingli01/project/jn/LLMapGen}"
UNIMAPGEN_ROOT="${UNIMAPGEN_ROOT:-/file_storage01/home/mingli/project/jn/UniMapGen}"
MODEL_DIR="${MODEL_DIR:-/mingli01/data/outputs/qwen3vl8b_instruct_fixedproc_no_empty_processor_config}"
SRC_TRAINROOT="${SRC_TRAINROOT:-/mingli01/data/outputs/rc_clean_dataset_xiuzheng_joint512_trainroot_douglas_merge6h22_nolinefix_20260506}"
ORACLE_TRAINROOT="${ORACLE_TRAINROOT:-/mingli01/data/outputs/rc_clean_dataset_xiuzheng_joint512_oracle_intersection_prior_centerline_trainroot_20260706}"
PATCH_ROOT="${PATCH_ROOT:-/mingli01/data/outputs/rc_clean_dataset_xiuzheng_joint512_patch_root_nolinefix_20260506}"
MEDIA_DIR="${MEDIA_DIR:-${PATCH_ROOT}}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/mingli01/data/outputs/stage2_qwen3vl8b_joint512_oracle_intersection_prior_centerline_smoke_20260706}"
OUTPUT_DIR="${OUTPUT_DIR:-/mingli01/data/outputs/stage2_qwen3vl8b_joint512_oracle_intersection_prior_centerline_lora_full1epoch_20260706}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${UNIMAPGEN_ROOT}/scripts/train_qwen3_vl_direct_rc_centerline_json.py}"

RUN_MODE="${RUN_MODE:-smoke_then_full}"
PREPARE_DATASET="${PREPARE_DATASET:-true}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-64}"
SMOKE_MAX_EVAL_SAMPLES="${SMOKE_MAX_EVAL_SAMPLES:-16}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-20}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

export PYTHONPATH="${UNIMAPGEN_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
unset PYTHONNOUSERSITE || true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/file_storage01/home/mingli/data/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/file_storage01/home/mingli/data/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/file_storage01/home/mingli/data/hf_cache/transformers}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_cache_${SLURM_JOB_ID:-manual}}"

cd "${REPO_ROOT}"
mkdir -p slurm-logs "${TRITON_CACHE_DIR}" "${SMOKE_OUTPUT_DIR}" "${OUTPUT_DIR}"

echo "[oracle-int-cl] date=$(date)"
echo "[oracle-int-cl] host=$(hostname)"
echo "[oracle-int-cl] branch=$(git branch --show-current 2>/dev/null || true)"
echo "[oracle-int-cl] model=${MODEL_DIR}"
echo "[oracle-int-cl] src_trainroot=${SRC_TRAINROOT}"
echo "[oracle-int-cl] oracle_trainroot=${ORACLE_TRAINROOT}"
echo "[oracle-int-cl] media_dir=${MEDIA_DIR}"
echo "[oracle-int-cl] output_dir=${OUTPUT_DIR}"
echo "[oracle-int-cl] run_mode=${RUN_MODE} lr=${LEARNING_RATE} epochs=${NUM_TRAIN_EPOCHS}"

test -d "${MODEL_DIR}"
test -f "${SRC_TRAINROOT}/train.jsonl"
test -f "${SRC_TRAINROOT}/val.jsonl"
test -f "${SRC_TRAINROOT}/meta_train.jsonl"
test -f "${SRC_TRAINROOT}/meta_val.jsonl"
test -f "${TRAIN_SCRIPT}"

python3 -m py_compile scripts/tools/build_oracle_intersection_prior_centerline_dataset.py

if [ "${PREPARE_DATASET}" = "true" ]; then
  python3 scripts/tools/build_oracle_intersection_prior_centerline_dataset.py \
    --src-trainroot "${SRC_TRAINROOT}" \
    --output-trainroot "${ORACLE_TRAINROOT}" \
    --overwrite
fi

if [ -d "${PATCH_ROOT}/patches512_joint" ]; then
  rm -f "${ORACLE_TRAINROOT}/patches512_joint"
  ln -s "${PATCH_ROOT}/patches512_joint" "${ORACLE_TRAINROOT}/patches512_joint"
fi

wc -l \
  "${ORACLE_TRAINROOT}/train.jsonl" \
  "${ORACLE_TRAINROOT}/val.jsonl" \
  "${ORACLE_TRAINROOT}/meta_train.jsonl" \
  "${ORACLE_TRAINROOT}/meta_val.jsonl"

run_train() {
  local output_dir="$1"
  local max_samples="$2"
  local max_eval_samples="$3"
  local max_steps="$4"
  local epochs="$5"
  local eval_strategy="$6"
  local save_steps="$7"
  local best_args=()
  if [ "${eval_strategy}" != "no" ]; then
    best_args+=(--load-best-model-at-end --no-greater-is-better --metric-for-best-model eval_loss)
  fi
  conda run --no-capture-output -n getok python -u "${TRAIN_SCRIPT}" \
    --model-name-or-path "${MODEL_DIR}" \
    --dataset-jsonl "${ORACLE_TRAINROOT}/train.jsonl" \
    --dataset-meta-jsonl "${ORACLE_TRAINROOT}/meta_train.jsonl" \
    --eval-dataset-jsonl "${ORACLE_TRAINROOT}/val.jsonl" \
    --eval-dataset-meta-jsonl "${ORACLE_TRAINROOT}/meta_val.jsonl" \
    --media-dir "${MEDIA_DIR}" \
    --output-dir "${output_dir}" \
    --max-samples "${max_samples}" \
    --max-eval-samples "${max_eval_samples}" \
    --num-train-epochs "${epochs}" \
    --max-steps "${max_steps}" \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay 0.0 \
    --warmup-ratio 0.03 \
    --logging-steps 10 \
    --evaluation-strategy "${eval_strategy}" \
    --eval-steps "${EVAL_STEPS}" \
    --save-steps "${save_steps}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    --dataloader-num-workers 2 \
    --cutoff-len 4096 \
    --image-size 0 \
    --source-patch-size 512 \
    --target-coord-max 511 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-target-modules all-linear \
    --freeze-vision-tower \
    --freeze-merger \
    --use-dataset-system-prompt \
    --use-dataset-user-prompt \
    "${best_args[@]}"
}

if [ "${RUN_MODE}" = "smoke" ] || [ "${RUN_MODE}" = "smoke_then_full" ]; then
  rm -rf "${SMOKE_OUTPUT_DIR}"
  mkdir -p "${SMOKE_OUTPUT_DIR}"
  run_train "${SMOKE_OUTPUT_DIR}" "${SMOKE_MAX_SAMPLES}" "${SMOKE_MAX_EVAL_SAMPLES}" "${SMOKE_MAX_STEPS}" 1 "no" "${SMOKE_MAX_STEPS}"
fi

if [ "${RUN_MODE}" = "full" ] || [ "${RUN_MODE}" = "smoke_then_full" ]; then
  run_train "${OUTPUT_DIR}" 0 "${MAX_EVAL_SAMPLES}" -1 "${NUM_TRAIN_EPOCHS}" "steps" "${SAVE_STEPS}"
fi

echo "[oracle-int-cl] finished at $(date)"
