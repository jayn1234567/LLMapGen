#!/bin/bash

#SBATCH -J jnrcstg2
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o /file_storage01/home/mingli/project/jn/UniMapGen/logs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_%j.out
#SBATCH -e /file_storage01/home/mingli/project/jn/UniMapGen/logs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_%j.err

set -euo pipefail
if [ -f /usr/share/lmod/lmod/init/bash ]; then
  source /usr/share/lmod/lmod/init/bash
fi
if command -v module >/dev/null 2>&1; then
  export MODULEPATH=/persist_data/modulefiles:/share/modulefiles
  module load conda/3 || true
  module load cuda/13.0 || true
fi

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/mingli/project/jn/UniMapGen}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/.condaenvs/llamafactory-cu128-dtok-resume-20260326}
PYTHON_BIN=${PYTHON_BIN:-$ENV_PREFIX/bin/python3}
SCRIPT_PATH=${SCRIPT_PATH:-$PROJECT_ROOT/train_required/scripts/train_qwen3_rc_dinov2_caption_llava.py}

MODEL_DIR=${MODEL_DIR:-/file_storage01/home/mingli/data/ckpts/Qwen3-4B}
TOKENIZER_DIR=${TOKENIZER_DIR:-$MODEL_DIR}
DINO_DIR=${DINO_DIR:-/file_storage01/home/mingli/data/ckpts/dinov2-large}
VISUAL_ENCODER_CKPT=${VISUAL_ENCODER_CKPT:-/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/latest.pt}
BRIDGE_MODULES=${BRIDGE_MODULES:-/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/rc_dinov2_clip_align_modules.pt}

DATA_DIR=${DATA_DIR:-/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411}
TRAIN_JSON=${TRAIN_JSON:-$DATA_DIR/train.jsonl}
TRAIN_META_JSON=${TRAIN_META_JSON:-$DATA_DIR/meta_train.jsonl}
EVAL_JSON=${EVAL_JSON:-$DATA_DIR/val.jsonl}
EVAL_META_JSON=${EVAL_META_JSON:-$DATA_DIR/meta_val.jsonl}
OUT_DIR=${OUT_DIR:-/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412}

MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
EVALUATION_STRATEGY=${EVALUATION_STRATEGY:-steps}
EVAL_STEPS=${EVAL_STEPS:-1000}

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/file_storage01/home/mingli/data/hf_cache
export HF_DATASETS_CACHE=/file_storage01/home/mingli/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/file_storage01/home/mingli/data/hf_cache/transformers
export TORCH_HOME=/persist_data/home/mingli/.cache/torch
export NNODES=1
export NPROC_PER_NODE=4
export MASTER_PORT=${MASTER_PORT:-29793}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/${USER}/triton_cache_${SLURM_JOB_ID}}
unset PYTHONPATH || true
unset PYTHONUSERBASE || true
unset PYTHONHOME || true

mkdir -p /file_storage01/home/mingli/project/jn/UniMapGen/logs
mkdir -p /file_storage01/home/mingli/data/outputs
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

echo "[stage2-caption] host            : $(hostname)"
echo "[stage2-caption] model           : $MODEL_DIR"
echo "[stage2-caption] tokenizer       : $TOKENIZER_DIR"
echo "[stage2-caption] dino            : $DINO_DIR"
echo "[stage2-caption] dino_ckpt       : $VISUAL_ENCODER_CKPT"
echo "[stage2-caption] stage1_bridge   : $BRIDGE_MODULES"
echo "[stage2-caption] train_json      : $TRAIN_JSON"
echo "[stage2-caption] train_meta      : $TRAIN_META_JSON"
echo "[stage2-caption] eval_json       : $EVAL_JSON"
echo "[stage2-caption] eval_meta       : $EVAL_META_JSON"
echo "[stage2-caption] output          : $OUT_DIR"
echo "[stage2-caption] schema          : Scene + GridStates(8x8)"
echo "[stage2-caption] trainable       : visual_norm + projector + geometric_position_mlp + token_alignment"
date

test -f "$TRAIN_JSON"
test -f "$TRAIN_META_JSON"
test -f "$EVAL_JSON"
test -f "$EVAL_META_JSON"
test -f "$VISUAL_ENCODER_CKPT"
test -f "$BRIDGE_MODULES"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m py_compile \
  "$PROJECT_ROOT/unimapgen/data/rc_caption_short_dataset.py" \
  "$PROJECT_ROOT/unimapgen/models/qwen3_rc_dinov2_caption_llava.py" \
  "$SCRIPT_PATH"

nvidia-smi

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
  "$SCRIPT_PATH" \
  --model-name-or-path "$MODEL_DIR" \
  --tokenizer-name-or-path "$TOKENIZER_DIR" \
  --dinov2-model-name-or-path "$DINO_DIR" \
  --visual-encoder-checkpoint-path "$VISUAL_ENCODER_CKPT" \
  --bridge-modules-state-path "$BRIDGE_MODULES" \
  --dataset-jsonl "$TRAIN_JSON" \
  --dataset-meta-jsonl "$TRAIN_META_JSON" \
  --eval-dataset-jsonl "$EVAL_JSON" \
  --eval-dataset-meta-jsonl "$EVAL_META_JSON" \
  --media-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --weight-decay 0.0 \
  --warmup-ratio 0.03 \
  --logging-steps "$LOGGING_STEPS" \
  --save-strategy "$SAVE_STRATEGY" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --evaluation-strategy "$EVALUATION_STRATEGY" \
  --eval-steps "$EVAL_STEPS" \
  --dataloader-num-workers 0 \
  --cutoff-len 4096 \
  --image-size 512 \
  --encoder-input-pad-size 518 \
  --visual-projector-hidden-dim 4096 \
  --geometric-mlp-hidden-dim 512 \
  --token-alignment-hidden-dim 4096 \
  --token-alignment-num-layers 2 \
  --token-alignment-dropout 0.0 \
  --model-dtype bfloat16 \
  --freeze-language-model \
  --freeze-vision-encoder \
  --local-files-only \
  --bf16

date
echo "[stage2-caption] done"
