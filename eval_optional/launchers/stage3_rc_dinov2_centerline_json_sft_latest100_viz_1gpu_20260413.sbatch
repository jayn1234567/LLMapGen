#!/bin/bash

#SBATCH -J rcjsviz
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o /file_storage01/home/mingli/project/jn/UniMapGen/logs/stage3_rc_dinov2_centerline_json_sft_latest100_viz_%j.out
#SBATCH -e /file_storage01/home/mingli/project/jn/UniMapGen/logs/stage3_rc_dinov2_centerline_json_sft_latest100_viz_%j.err

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
PREDICT_SCRIPT=${PREDICT_SCRIPT:-$PROJECT_ROOT/eval_optional/scripts/predict_qwen3_rc_dinov2_centerline_json_sft.py}
RENDER_SCRIPT=${RENDER_SCRIPT:-$PROJECT_ROOT/eval_optional/scripts/render_eval_predictions_jsonl.py}

RUN_ROOT=${RUN_ROOT:-/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413}
DATA_ROOT=${DATA_ROOT:-/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408}
SPLIT_JSON=${SPLIT_JSON:-$DATA_ROOT/val.jsonl}
SPLIT_META_JSON=${SPLIT_META_JSON:-$DATA_ROOT/meta_val.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
SEED=${SEED:-42}

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
cd "$PROJECT_ROOT"

LATEST_CKPT=$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [ -z "$LATEST_CKPT" ]; then
  echo "[stage3-json-viz] no checkpoint-* found under $RUN_ROOT" >&2
  exit 1
fi
CKPT_NAME=$(basename "$LATEST_CKPT")
OUT_DIR=${OUT_DIR:-$RUN_ROOT/inference_viz_${CKPT_NAME}_val${MAX_SAMPLES}_20260413}
PRED_JSONL="$OUT_DIR/predictions.jsonl"
SUMMARY_JSON="$OUT_DIR/summary.json"
PANELS_DIR="$OUT_DIR/panels"

mkdir -p "$OUT_DIR" "$PANELS_DIR"

echo "[stage3-json-viz] host        : $(hostname)"
echo "[stage3-json-viz] run_root    : $RUN_ROOT"
echo "[stage3-json-viz] latest_ckpt : $LATEST_CKPT"
echo "[stage3-json-viz] split_json  : $SPLIT_JSON"
echo "[stage3-json-viz] split_meta  : $SPLIT_META_JSON"
echo "[stage3-json-viz] output_dir  : $OUT_DIR"
echo "[stage3-json-viz] max_samples : $MAX_SAMPLES"
echo "[stage3-json-viz] max_new     : $MAX_NEW_TOKENS"
date

test -f "$PREDICT_SCRIPT"
test -f "$RENDER_SCRIPT"
test -d "$LATEST_CKPT"
test -f "$LATEST_CKPT/pytorch_model.bin"
test -f "$RUN_ROOT/args.json"
test -f "$SPLIT_JSON"
test -f "$SPLIT_META_JSON"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m py_compile \
  "$PROJECT_ROOT/unimapgen/data/rc_centerline_json_sft_dataset.py" \
  "$PROJECT_ROOT/unimapgen/models/qwen3_rc_dinov2_centerline_json_sft.py" \
  "$PREDICT_SCRIPT" \
  "$RENDER_SCRIPT"

nvidia-smi

PYTHONNOUSERSITE=1 "$PYTHON_BIN" "$PREDICT_SCRIPT" \
  --checkpoint-dir "$LATEST_CKPT" \
  --dataset-jsonl "$SPLIT_JSON" \
  --dataset-meta-jsonl "$SPLIT_META_JSON" \
  --media-dir "$DATA_ROOT" \
  --output-jsonl "$PRED_JSONL" \
  --summary-json "$SUMMARY_JSON" \
  --max-samples "$MAX_SAMPLES" \
  --shuffle \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device cuda

PYTHONNOUSERSITE=1 "$PYTHON_BIN" "$RENDER_SCRIPT" \
  --predictions-jsonl "$PRED_JSONL" \
  --dataset-root "$DATA_ROOT" \
  --output-dir "$PANELS_DIR" \
  --limit "$MAX_SAMPLES"

date
echo "[stage3-json-viz] done"
