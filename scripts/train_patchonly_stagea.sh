#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/data/project/jn/UniMapGen/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct}"
LF_ENV="${LF_ENV:-$ROOT/.envs/llamafactory-cu128}"

if [[ ! -x "$LF_ENV/bin/llamafactory-cli" ]]; then
  echo "[ERROR] Missing llamafactory-cli: $LF_ENV/bin/llamafactory-cli" >&2
  exit 1
fi

exec "$LF_ENV/bin/llamafactory-cli" train "$ROOT/configs/llamafactory_paper16_patch_only_100img_system/qwen2_5vl_3b_lora_sft.yaml"
