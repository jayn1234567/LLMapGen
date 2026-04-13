#!/bin/bash

#SBATCH -J rcclipev
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /file_storage01/home/mingli/project/jn/UniMapGen/logs/stage1_rc_dinov2_clip_retrieval_eval_val_%j.out
#SBATCH -e /file_storage01/home/mingli/project/jn/UniMapGen/logs/stage1_rc_dinov2_clip_retrieval_eval_val_%j.err

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
SCRIPT_PATH=${SCRIPT_PATH:-$PROJECT_ROOT/eval_optional/scripts/eval_qwen3_rc_dinov2_clip_retrieval.py}

TRAIN_OUT_DIR=${TRAIN_OUT_DIR:-/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412}
DATA_ROOT=${DATA_ROOT:-/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412}
FULL_MEDIA_ROOT=${FULL_MEDIA_ROOT:-/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411}
OUT_DIR=${OUT_DIR:-$TRAIN_OUT_DIR/retrieval_eval_val}
MAX_SAMPLES=${MAX_SAMPLES:-0}
BATCH_SIZE=${BATCH_SIZE:-16}

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME=/file_storage01/home/mingli/data/hf_cache
export HF_DATASETS_CACHE=/file_storage01/home/mingli/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/file_storage01/home/mingli/data/hf_cache/transformers
unset PYTHONPATH || true
unset PYTHONUSERBASE || true
unset PYTHONHOME || true

mkdir -p /file_storage01/home/mingli/project/jn/UniMapGen/logs
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

echo "[stage1-retrieval-eval] host      : $(hostname)"
echo "[stage1-retrieval-eval] train_out : $TRAIN_OUT_DIR"
echo "[stage1-retrieval-eval] data_root : $DATA_ROOT"
echo "[stage1-retrieval-eval] media     : $FULL_MEDIA_ROOT"
echo "[stage1-retrieval-eval] out_dir   : $OUT_DIR"
date

test -f "$SCRIPT_PATH"
test -f "$TRAIN_OUT_DIR/args.json"
test -f "$TRAIN_OUT_DIR/rc_dinov2_clip_align_modules.pt"
test -f "$DATA_ROOT/val.jsonl"
test -f "$DATA_ROOT/meta_val.jsonl"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m py_compile \
  "$PROJECT_ROOT/unimapgen/data/rc_semantic_align_dataset.py" \
  "$PROJECT_ROOT/unimapgen/models/qwen3_rc_dinov2_clip_align.py" \
  "$SCRIPT_PATH"

nvidia-smi

PYTHONNOUSERSITE=1 "$PYTHON_BIN" "$SCRIPT_PATH" \
  --train-output-dir "$TRAIN_OUT_DIR" \
  --dataset-jsonl "$DATA_ROOT/val.jsonl" \
  --dataset-meta-jsonl "$DATA_ROOT/meta_val.jsonl" \
  --media-dir "$FULL_MEDIA_ROOT" \
  --output-dir "$OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --device cuda \
  --local-files-only

date
echo "[stage1-retrieval-eval] done"
