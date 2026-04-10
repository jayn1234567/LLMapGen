#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_DIR=${MODEL_DIR:-$ROOT/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct}
DATASET_ROOT=${DATASET_ROOT:-$ROOT/dataset/paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10}

CONFIG_TEMPLATE="$ROOT/configs/llamafactory_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system/qwen2_5vl_3b_lora_sft_stdce_8gpu_e10.yaml"
CONFIG_RUNTIME_DIR="$ROOT/.runtime"
CONFIG_RUNTIME="$CONFIG_RUNTIME_DIR/qwen2_5vl_3b_lora_sft_stdce_8gpu_e10.runtime.yaml"
DATASET_RUNTIME_DIR="$CONFIG_RUNTIME_DIR/llamafactory_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system"
DATASET_INFO_TEMPLATE="$ROOT/configs/llamafactory_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system/dataset_info.json"
DATASET_INFO_RUNTIME="$DATASET_RUNTIME_DIR/dataset_info.json"

mkdir -p "$CONFIG_RUNTIME_DIR" "$DATASET_RUNTIME_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[stagea-fixed16-train] missing model dir: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$DATASET_ROOT/train.jsonl" ]]; then
  echo "[stagea-fixed16-train] missing dataset jsonl: $DATASET_ROOT/train.jsonl" >&2
  exit 1
fi
if [[ ! -f "$DATASET_ROOT/val.jsonl" ]]; then
  echo "[stagea-fixed16-train] missing dataset jsonl: $DATASET_ROOT/val.jsonl" >&2
  exit 1
fi

echo "[stagea-fixed16-train] MODEL_DIR=$MODEL_DIR"
echo "[stagea-fixed16-train] DATASET_ROOT=$DATASET_ROOT"
echo "[stagea-fixed16-train] OUTPUT_DIR=$OUTPUT_DIR"

python - "$DATASET_INFO_TEMPLATE" "$DATASET_INFO_RUNTIME" "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

src, dst, dataset_root = sys.argv[1:4]
dataset_root = Path(dataset_root)
obj = json.loads(Path(src).read_text(encoding="utf-8"))
for name, value in obj.items():
    if name.endswith("_train"):
        value["file_name"] = str(dataset_root / "train.jsonl")
    elif name.endswith("_val"):
        value["file_name"] = str(dataset_root / "val.jsonl")
Path(dst).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

python - "$CONFIG_TEMPLATE" "$CONFIG_RUNTIME" "$DATASET_RUNTIME_DIR" "$DATASET_ROOT" "$MODEL_DIR" "$OUTPUT_DIR" <<'PY'
import re
import sys

src, dst, dataset_dir, media_dir, model_dir, output_dir = sys.argv[1:7]
text = open(src, "r", encoding="utf-8").read()
repls = {
    r"(?m)^model_name_or_path:.*$": f"model_name_or_path: {model_dir}",
    r"(?m)^dataset_dir:.*$": f"dataset_dir: {dataset_dir}",
    r"(?m)^media_dir:.*$": f"media_dir: {media_dir}",
    r"(?m)^output_dir:.*$": f"output_dir: {output_dir}",
}
for pat, rep in repls.items():
    text = re.sub(pat, rep, text)
open(dst, "w", encoding="utf-8").write(text)
PY

llamafactory-cli train "$CONFIG_RUNTIME"
