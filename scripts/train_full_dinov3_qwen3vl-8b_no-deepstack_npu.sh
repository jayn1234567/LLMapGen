#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export DISABLE_DEEPSTACK=True
exec bash "${SCRIPT_DIR}/train_full_dinov3_qwen3vl-8b_deepstack_npu.sh" "$@"
