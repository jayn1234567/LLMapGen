#!/usr/bin/env bash
set -euo pipefail

# Current production test entry: DINOv2 + Qwen3VL-8B + no DeepStack.
# It delegates to the cloud test script and reads the prebuilt test.jsonl directly.

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export DISABLE_DEEPSTACK=True
exec bash "${SCRIPT_DIR}/test_full_dinov2_qwen3vl-8b_npu.sh" "$@"
