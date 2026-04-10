#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EVAL_ROOT=${EVAL_ROOT:-$ROOT/dataset/paper16_patch_only_full_trainval_fixed16_unseenlite305515_empty08_from_refs_allboxes}
DATASET_JSONL=${DATASET_JSONL:-$EVAL_ROOT/val.jsonl}
META_JSONL=${META_JSONL:-$EVAL_ROOT/meta_val.jsonl}
BASE_MODEL=${BASE_MODEL:-$ROOT/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct}
PROCESSOR_PATH=${PROCESSOR_PATH:-$BASE_MODEL}
ADAPTER=${ADAPTER:-$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/checkpoint-157182}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/stagea_patch_only_full_trainval_geomdedup_fixed16_unseenlite_eval_ckpt157182}
DEVICE=${DEVICE:-cuda:0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_SOURCE_PATCHES=${MAX_SOURCE_PATCHES:-0}

if [[ ! -f "$DATASET_JSONL" ]]; then
  echo "[stagea-fixed16-eval] missing dataset jsonl: $DATASET_JSONL" >&2
  exit 1
fi
if [[ ! -f "$META_JSONL" ]]; then
  echo "[stagea-fixed16-eval] missing meta jsonl: $META_JSONL" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL" ]]; then
  echo "[stagea-fixed16-eval] missing base model dir: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -e "$ADAPTER" ]]; then
  echo "[stagea-fixed16-eval] missing adapter path: $ADAPTER" >&2
  exit 1
fi

echo "[stagea-fixed16-eval] DATASET_JSONL=$DATASET_JSONL"
echo "[stagea-fixed16-eval] META_JSONL=$META_JSONL"
echo "[stagea-fixed16-eval] BASE_MODEL=$BASE_MODEL"
echo "[stagea-fixed16-eval] ADAPTER=$ADAPTER"
echo "[stagea-fixed16-eval] OUTPUT_DIR=$OUTPUT_DIR"

cmd=(
  python "$ROOT/scripts/run_fixed16_grouped_eval.py"
  --dataset-jsonl "$DATASET_JSONL"
  --meta-jsonl "$META_JSONL"
  --dataset-root "$EVAL_ROOT"
  --base-model "$BASE_MODEL"
  --adapter "$ADAPTER"
  --processor-path "$PROCESSOR_PATH"
  --engine custom
  --output-dir "$OUTPUT_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --paper-categories road
)

if [[ "$MAX_SOURCE_PATCHES" != "0" ]]; then
  cmd+=(--max-source-patches "$MAX_SOURCE_PATCHES")
fi

"${cmd[@]}"
