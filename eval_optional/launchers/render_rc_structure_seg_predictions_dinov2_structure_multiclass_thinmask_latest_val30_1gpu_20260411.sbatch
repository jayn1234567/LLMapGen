#!/bin/bash

#SBATCH -J jnrcdmclv
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /file_storage01/home/mingli/project/jn/UniMapGen/logs/render_rc_structure_seg_predictions_dinov2_structure_multiclass_thinmask_latest_%j.out
#SBATCH -e /file_storage01/home/mingli/project/jn/UniMapGen/logs/render_rc_structure_seg_predictions_dinov2_structure_multiclass_thinmask_latest_%j.err

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
SCRIPT_PATH=${SCRIPT_PATH:-$PROJECT_ROOT/eval_optional/scripts/render_rc_structure_seg_predictions.py}

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME=/file_storage01/home/mingli/data/hf_cache
export HF_DATASETS_CACHE=/file_storage01/home/mingli/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/file_storage01/home/mingli/data/hf_cache/transformers
export TORCH_HOME=/persist_data/home/mingli/.cache/torch
unset PYTHONUSERBASE || true
unset PYTHONHOME || true
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}

DATA_DIR=${DATA_DIR:-/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408}
VAL_JSON=${VAL_JSON:-$DATA_DIR/val.jsonl}
VAL_META_JSON=${VAL_META_JSON:-$DATA_DIR/meta_val.jsonl}
CHECKPOINT=${CHECKPOINT:-/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/latest.pt}
OUT_DIR=${OUT_DIR:-/file_storage01/home/mingli/data/outputs/render_rc_structure_seg_dinov2_structure_multiclass3_thinmask_latest_val30_20260411}
MAX_SAMPLES=${MAX_SAMPLES:-30}
THRESHOLD=${THRESHOLD:-0.5}

mkdir -p /file_storage01/home/mingli/project/jn/UniMapGen/logs
mkdir -p /file_storage01/home/mingli/data/outputs
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

echo "[render-rc-structure-seg-dinov2-thinmask] host       : $(hostname)"
echo "[render-rc-structure-seg-dinov2-thinmask] checkpoint : $CHECKPOINT"
echo "[render-rc-structure-seg-dinov2-thinmask] val_json   : $VAL_JSON"
echo "[render-rc-structure-seg-dinov2-thinmask] val_meta   : $VAL_META_JSON"
echo "[render-rc-structure-seg-dinov2-thinmask] output_dir : $OUT_DIR"
echo "[render-rc-structure-seg-dinov2-thinmask] max_samples: $MAX_SAMPLES"
echo "[render-rc-structure-seg-dinov2-thinmask] threshold  : $THRESHOLD"
date

test -f "$CHECKPOINT"
test -f "$VAL_JSON"
test -f "$VAL_META_JSON"

"$PYTHON_BIN" -m py_compile \
  "$PROJECT_ROOT/unimapgen/data/rc_structure_seg_dataset.py" \
  "$PROJECT_ROOT/unimapgen/models/encoders/satellite_encoder.py" \
  "$PROJECT_ROOT/unimapgen/models/rc_structure_seg.py" \
  "$PROJECT_ROOT/eval_optional/scripts/render_rc_structure_seg_predictions.py"

nvidia-smi || true

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -s "$SCRIPT_PATH" \
  --checkpoint "$CHECKPOINT" \
  --dataset-jsonl "$VAL_JSON" \
  --dataset-meta-jsonl "$VAL_META_JSON" \
  --media-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --image-size 512 \
  --mask-size 512 \
  --max-samples "$MAX_SAMPLES" \
  --batch-size 4 \
  --threshold "$THRESHOLD" \
  --device cuda

date
echo "[render-rc-structure-seg-dinov2-thinmask] done"
