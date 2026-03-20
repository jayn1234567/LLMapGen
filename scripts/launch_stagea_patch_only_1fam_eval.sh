#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
FAMILY_ID=${FAMILY_ID:-ATX_-1_0_sat__paper16_r0_c0}
SAMPLE_IDS=${SAMPLE_IDS:-ATX_-1_0_sat__paper16_r0_c0_p00,ATX_-1_0_sat__paper16_r0_c0_p01,ATX_-1_0_sat__paper16_r0_c0_p02,ATX_-1_0_sat__paper16_r0_c0_p03,ATX_-1_0_sat__paper16_r0_c0_p04,ATX_-1_0_sat__paper16_r0_c0_p05,ATX_-1_0_sat__paper16_r0_c0_p06,ATX_-1_0_sat__paper16_r0_c0_p07,ATX_-1_0_sat__paper16_r0_c0_p08,ATX_-1_0_sat__paper16_r0_c0_p09,ATX_-1_0_sat__paper16_r0_c0_p10,ATX_-1_0_sat__paper16_r0_c0_p11,ATX_-1_0_sat__paper16_r0_c0_p12,ATX_-1_0_sat__paper16_r0_c0_p13,ATX_-1_0_sat__paper16_r0_c0_p14,ATX_-1_0_sat__paper16_r0_c0_p15}
DATASET_ROOT=${DATASET_ROOT:-$ROOT/dataset/paper16_patch_only_100img_system}
DATASET_JSONL=${DATASET_JSONL:-$DATASET_ROOT/train.jsonl}
BASE_MODEL=${BASE_MODEL:-$ROOT/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct}
PROCESSOR_PATH=${PROCESSOR_PATH:-$BASE_MODEL}
if [[ -z "${ADAPTER:-}" ]]; then
  ADAPTER_BASE="$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora"
  if [[ ! -d "$ADAPTER_BASE" ]]; then
    echo "[stagea-eval] missing adapter root: $ADAPTER_BASE" >&2
    exit 1
  fi
  if [[ -f "$ADAPTER_BASE/adapter_config.json" || -f "$ADAPTER_BASE/adapter_model.safetensors" ]]; then
    ADAPTER="$ADAPTER_BASE"
  elif find "$ADAPTER_BASE" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
    ADAPTER="$(find "$ADAPTER_BASE" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  else
    echo "[stagea-eval] no adapter files and no checkpoint-* found under: $ADAPTER_BASE" >&2
    exit 1
  fi
fi
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/stagea_eval_1fam_${FAMILY_ID}}
DEVICE=${DEVICE:-cuda:0}

if [[ ! -f "$DATASET_JSONL" ]]; then
  echo "[stagea-eval] missing dataset jsonl: $DATASET_JSONL" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL" ]]; then
  echo "[stagea-eval] missing base model dir: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -e "$ADAPTER" ]]; then
  echo "[stagea-eval] resolved adapter path does not exist: $ADAPTER" >&2
  exit 1
fi
echo "[stagea-eval] DATASET_JSONL=$DATASET_JSONL"
echo "[stagea-eval] BASE_MODEL=$BASE_MODEL"
echo "[stagea-eval] ADAPTER=$ADAPTER"
echo "[stagea-eval] OUTPUT_DIR=$OUTPUT_DIR"

python "$ROOT/scripts/run_qwen2_5vl_lora_small_eval.py" \
  --dataset-jsonl "$DATASET_JSONL" \
  --dataset-root "$DATASET_ROOT" \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --processor-path "$PROCESSOR_PATH" \
  --engine custom \
  --output-dir "$OUTPUT_DIR" \
  --max-samples 16 \
  --sample-ids "$SAMPLE_IDS" \
  --max-new-tokens 2048 \
  --device "$DEVICE"
