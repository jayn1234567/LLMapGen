#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR=${1:?"usage: merge_zero3_checkpoint.sh CHECKPOINT_DIR [OUTPUT_DIR]"}
CHECKPOINT_DIR=$(readlink -f "${CHECKPOINT_DIR}")
OUTPUT_DIR=${2:-${CHECKPOINT_DIR}_merged}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(readlink -f "${OUTPUT_DIR}")

CONVERTER=${CHECKPOINT_DIR}/zero_to_fp32.py
if [ ! -f "${CONVERTER}" ]; then
  echo "ERROR: DeepSpeed converter not found: ${CONVERTER}" >&2
  exit 1
fi

echo "[zero3-merge] checkpoint=${CHECKPOINT_DIR}"
echo "[zero3-merge] output=${OUTPUT_DIR}"
python "${CONVERTER}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}/pytorch_model.bin"

for name in \
  config.json generation_config.json trainer_state.json training_args.bin \
  tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json \
  chat_template.jinja vocab.json merges.txt tokenizer.model preprocessor_config.json \
  args.json qwen_multimodal_checkpoint.json \
  rc_dinov2_centerline_json_modules.pt rc_dinov2_centerline_json_modules.pth; do
  if [ -f "${CHECKPOINT_DIR}/${name}" ]; then
    cp -f "${CHECKPOINT_DIR}/${name}" "${OUTPUT_DIR}/${name}"
  fi
done

python - "${OUTPUT_DIR}/pytorch_model.bin" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"Merged model was not produced: {path}")
print(f"[zero3-merge] merged model bytes={path.stat().st_size}")
PY

echo "[zero3-merge] done: ${OUTPUT_DIR}"
