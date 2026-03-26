#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_DIR=${MODEL_DIR:-$ROOT/ckpts/modelscope/Qwen2___5-VL-3B-Instruct}
DATASET_ROOT=${DATASET_ROOT:-$ROOT/dataset}
DATASET_JSONL=${DATASET_JSONL:-}
RUN_DIR=${RUN_DIR:-$ROOT/outputs/stagea_discrete_tokens_full}
CHECKPOINT_OR_MODEL=${CHECKPOINT_OR_MODEL:-}
PROCESSOR_PATH=${PROCESSOR_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/stagea_discrete_eval}
DEVICE=${DEVICE:-cuda:0}

MAX_SAMPLES=${MAX_SAMPLES:-16}
SAMPLE_IDS=${SAMPLE_IDS:-}
ID_PREFIXES=${ID_PREFIXES:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
METER_PER_PIXEL=${METER_PER_PIXEL:-0.15}
LINE_WIDTH_PX=${LINE_WIDTH_PX:-6}
PAPER_CATEGORIES=${PAPER_CATEGORIES:-}
IMAGE_SIZE=${IMAGE_SIZE:-896}
DISCRETE_CATEGORIES=${DISCRETE_CATEGORIES:-road}
DISCRETE_COORD_NUM_BINS=${DISCRETE_COORD_NUM_BINS:-896}
DISCRETE_TOKEN_SCHEMA=${DISCRETE_TOKEN_SCHEMA:-shared_numbers}
DISABLE_LEGACY_TEXT_PROMPT_TOKENS=${DISABLE_LEGACY_TEXT_PROMPT_TOKENS:-1}
SKIP_VIZ=${SKIP_VIZ:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-10}

if [[ -z "$DATASET_JSONL" ]]; then
  if [[ -f "$DATASET_ROOT/val.jsonl" ]]; then
    DATASET_JSONL="$DATASET_ROOT/val.jsonl"
  else
    DATASET_JSONL="$DATASET_ROOT/train.jsonl"
  fi
fi

if [[ -z "$CHECKPOINT_OR_MODEL" ]]; then
  if find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
    CHECKPOINT_OR_MODEL="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  elif [[ -f "$RUN_DIR/config.json" ]]; then
    CHECKPOINT_OR_MODEL="$RUN_DIR"
  else
    echo "[portable-dtok-eval] cannot resolve checkpoint/model under: $RUN_DIR" >&2
    exit 1
  fi
fi

if [[ -z "$PROCESSOR_PATH" ]]; then
  if [[ -d "$RUN_DIR/processor" ]]; then
    PROCESSOR_PATH="$RUN_DIR/processor"
  elif [[ -f "$CHECKPOINT_OR_MODEL/preprocessor_config.json" || -f "$CHECKPOINT_OR_MODEL/processor_config.json" ]]; then
    PROCESSOR_PATH="$CHECKPOINT_OR_MODEL"
  else
    PROCESSOR_PATH="$MODEL_DIR"
  fi
fi

if [[ ! -f "$DATASET_JSONL" ]]; then
  echo "[portable-dtok-eval] missing dataset jsonl: $DATASET_JSONL" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT/images" ]]; then
  echo "[portable-dtok-eval] missing images dir: $DATASET_ROOT/images" >&2
  exit 1
fi
if [[ ! -e "$CHECKPOINT_OR_MODEL" ]]; then
  echo "[portable-dtok-eval] missing checkpoint/model path: $CHECKPOINT_OR_MODEL" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[portable-dtok-eval] missing base model dir: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$ROOT/logs"

cmd=(
  "$PYTHON_BIN"
  "$ROOT/scripts/eval_stagea_discrete.py"
  --dataset-jsonl "$DATASET_JSONL"
  --dataset-root "$DATASET_ROOT"
  --model-or-checkpoint "$CHECKPOINT_OR_MODEL"
  --processor-path "$PROCESSOR_PATH"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --meter-per-pixel "$METER_PER_PIXEL"
  --line-width-px "$LINE_WIDTH_PX"
  --paper-categories "$PAPER_CATEGORIES"
  --image-size "$IMAGE_SIZE"
  --discrete-categories "$DISCRETE_CATEGORIES"
  --discrete-coord-num-bins "$DISCRETE_COORD_NUM_BINS"
  --discrete-token-schema "$DISCRETE_TOKEN_SCHEMA"
  --progress-every "$PROGRESS_EVERY"
)

if [[ -n "$SAMPLE_IDS" ]]; then
  cmd+=(--sample-ids "$SAMPLE_IDS")
fi
if [[ -n "$ID_PREFIXES" ]]; then
  cmd+=(--id-prefixes "$ID_PREFIXES")
fi
if [[ "$DISABLE_LEGACY_TEXT_PROMPT_TOKENS" == "1" ]]; then
  cmd+=(--disable-legacy-text-prompt-tokens)
fi
if [[ "$SKIP_VIZ" == "1" ]]; then
  cmd+=(--skip-viz)
fi

exec "${cmd[@]}"
