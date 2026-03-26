#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_DIR=${MODEL_DIR:-$ROOT/ckpts/modelscope/Qwen2___5-VL-3B-Instruct}
DATASET_ROOT=${DATASET_ROOT:-$ROOT/dataset}
TRAIN_JSON=${TRAIN_JSON:-$DATASET_ROOT/train.jsonl}
EVAL_JSON=${EVAL_JSON:-$DATASET_ROOT/val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/stagea_discrete_tokens_full}
PROCESSOR_OUTPUT_DIR=${PROCESSOR_OUTPUT_DIR:-$OUTPUT_DIR/processor}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29588}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-6}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
SEED=${SEED:-42}
CUTOFF_LEN=${CUTOFF_LEN:-8192}
EVALUATION_STRATEGY=${EVALUATION_STRATEGY:-epoch}
EVAL_STEPS=${EVAL_STEPS:-0}
COORD_NUM_BINS=${COORD_NUM_BINS:-896}
TOKEN_SCHEMA=${TOKEN_SCHEMA:-shared_numbers}
CATEGORIES=${CATEGORIES:-road}
IMAGE_SIZE=${IMAGE_SIZE:-896}
BF16=${BF16:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
DISABLE_LEGACY_TEXT_PROMPT_TOKENS=${DISABLE_LEGACY_TEXT_PROMPT_TOKENS:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[portable-dtok-train] missing model dir: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_JSON" ]]; then
  echo "[portable-dtok-train] missing train jsonl: $TRAIN_JSON" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT/images" ]]; then
  echo "[portable-dtok-train] missing images dir: $DATASET_ROOT/images" >&2
  exit 1
fi

if [[ ! -f "$EVAL_JSON" ]]; then
  EVAL_JSON=""
  EVALUATION_STRATEGY=no
fi

mkdir -p "$OUTPUT_DIR" "$ROOT/logs"

cmd=(
  "$PYTHON_BIN"
  "$ROOT/scripts/train_stagea_discrete.py"
  --model-name-or-path "$MODEL_DIR"
  --dataset-jsonl "$TRAIN_JSON"
  --media-dir "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --processor-output-dir "$PROCESSOR_OUTPUT_DIR"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-ratio "$WARMUP_RATIO"
  --logging-steps "$LOGGING_STEPS"
  --evaluation-strategy "$EVALUATION_STRATEGY"
  --eval-steps "$EVAL_STEPS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
  --seed "$SEED"
  --cutoff-len "$CUTOFF_LEN"
  --image-size "$IMAGE_SIZE"
  --coord-num-bins "$COORD_NUM_BINS"
  --token-schema "$TOKEN_SCHEMA"
  --categories "$CATEGORIES"
  --no-lora
)

if [[ -n "$EVAL_JSON" ]]; then
  cmd+=(--eval-dataset-jsonl "$EVAL_JSON")
fi
if [[ "$BF16" == "1" ]]; then
  cmd+=(--bf16)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  cmd+=(--gradient-checkpointing)
fi
if [[ "$DISABLE_LEGACY_TEXT_PROMPT_TOKENS" == "1" ]]; then
  cmd+=(--disable-legacy-text-prompt-tokens)
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" "${cmd[@]:1}"
else
  exec "${cmd[@]}"
fi
