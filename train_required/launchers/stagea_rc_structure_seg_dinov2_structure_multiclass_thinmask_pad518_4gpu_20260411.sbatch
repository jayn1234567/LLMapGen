#!/bin/bash

#SBATCH -J jnrcdmcl
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH -t 72:00:00
#SBATCH -o /file_storage01/home/mingli/project/jn/UniMapGen/logs/stagea_rc_structure_seg_dinov2_structure_multiclass_thinmask_pad518_4gpu_%j.out
#SBATCH -e /file_storage01/home/mingli/project/jn/UniMapGen/logs/stagea_rc_structure_seg_dinov2_structure_multiclass_thinmask_pad518_4gpu_%j.err

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
SCRIPT_PATH=${SCRIPT_PATH:-$PROJECT_ROOT/train_required/scripts/train_rc_structure_seg.py}

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME=/file_storage01/home/mingli/data/hf_cache
export HF_DATASETS_CACHE=/file_storage01/home/mingli/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/file_storage01/home/mingli/data/hf_cache/transformers
export TORCH_HOME=/persist_data/home/mingli/.cache/torch
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
unset PYTHONUSERBASE || true
unset PYTHONHOME || true

DATA_DIR=${DATA_DIR:-/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408}
TRAIN_JSON=${TRAIN_JSON:-$DATA_DIR/train.jsonl}
TRAIN_META_JSON=${TRAIN_META_JSON:-$DATA_DIR/meta_train.jsonl}
VAL_JSON=${VAL_JSON:-$DATA_DIR/val.jsonl}
VAL_META_JSON=${VAL_META_JSON:-$DATA_DIR/meta_val.jsonl}
MEDIA_DIR=${MEDIA_DIR:-$DATA_DIR}
DINO_REPO=${DINO_REPO:-facebook/dinov2-large}
DINO_DIR=${DINO_DIR:-/file_storage01/home/mingli/data/ckpts/dinov2-large}
OUT_DIR=${OUT_DIR:-/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411}
export DINO_REPO
export DINO_DIR

mkdir -p /file_storage01/home/mingli/project/jn/UniMapGen/logs
mkdir -p /file_storage01/home/mingli/data/outputs
mkdir -p /file_storage01/home/mingli/data/ckpts
mkdir -p "$HF_HOME"
cd "$PROJECT_ROOT"

echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] host          : $(hostname)"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] train_json    : $TRAIN_JSON"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] train_meta    : $TRAIN_META_JSON"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] val_json      : $VAL_JSON"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] val_meta      : $VAL_META_JSON"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] media_dir     : $MEDIA_DIR"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] model_repo    : $DINO_REPO"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] model_dir     : $DINO_DIR"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] output_dir    : $OUT_DIR"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] hf_endpoint   : $HF_ENDPOINT"
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] pad_input_to  : 518"
date

test -f "$TRAIN_JSON"
test -f "$TRAIN_META_JSON"
test -f "$VAL_JSON"
test -f "$VAL_META_JSON"

if [ ! -f "$DINO_DIR/config.json" ]; then
  echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] local weights missing, downloading snapshot..."
  "$PYTHON_BIN" -s - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["DINO_REPO"]
local_dir = Path(os.environ["DINO_DIR"])
endpoint = os.environ.get("HF_ENDPOINT", "").strip() or None
local_dir.mkdir(parents=True, exist_ok=True)

kwargs = {
    "repo_id": repo_id,
    "local_dir": str(local_dir),
    "max_workers": 8,
}
if endpoint:
    kwargs["endpoint"] = endpoint

print("[rc-structure-seg-dinov2-structure-multiclass-thinmask] snapshot_download start", flush=True)
path = snapshot_download(**kwargs)
print(f"[rc-structure-seg-dinov2-structure-multiclass-thinmask] snapshot path: {path}", flush=True)
required = ["config.json"]
missing = [name for name in required if not (local_dir / name).exists()]
if missing:
    raise SystemExit(f"missing required files after download: {missing}")
print("[rc-structure-seg-dinov2-structure-multiclass-thinmask] required files present", flush=True)
PY
fi

"$PYTHON_BIN" -m py_compile \
  "$PROJECT_ROOT/unimapgen/data/rc_structure_seg_dataset.py" \
  "$PROJECT_ROOT/unimapgen/models/encoders/satellite_encoder.py" \
  "$PROJECT_ROOT/unimapgen/models/rc_structure_seg.py" \
  "$PROJECT_ROOT/train_required/scripts/train_rc_structure_seg.py"

nvidia-smi || true

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON_BIN" -s "$SCRIPT_PATH" \
  --encoder-type dinov2_vitl14 \
  --seg-supervision structure_multiclass \
  --multiclass-class-weights 0.2,1.0,1.0 \
  --dinov2-model-name-or-path "$DINO_DIR" \
  --dinov2-unfreeze-last-n-blocks 12 \
  --train-jsonl "$TRAIN_JSON" \
  --train-meta-jsonl "$TRAIN_META_JSON" \
  --val-jsonl "$VAL_JSON" \
  --val-meta-jsonl "$VAL_META_JSON" \
  --media-dir "$MEDIA_DIR" \
  --output-dir "$OUT_DIR" \
  --image-size 512 \
  --mask-size 512 \
  --encoder-input-pad-size 518 \
  --decoder-dim 256 \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 8 \
  --lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-2 \
  --warmup-steps 500 \
  --bce-weight 1.0 \
  --dice-weight 1.0 \
  --fixed-pos-weight 4.0 \
  --grad-clip-norm 1.0 \
  --amp \
  --seed 42 \
  --aug-rot90-prob 0.5 \
  --aug-hflip-prob 0.5 \
  --aug-vflip-prob 0.0 \
  --log-every 20

date
echo "[rc-structure-seg-dinov2-structure-multiclass-thinmask] done"
