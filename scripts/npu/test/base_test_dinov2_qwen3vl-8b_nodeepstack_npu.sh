#!/usr/bin/env bash
set -euo pipefail

# Current production test entry: DINOv2 + Qwen3VL-8B + no DeepStack.
# It delegates to the cloud test script and reads the prebuilt test.jsonl directly.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
export DISABLE_DEEPSTACK=True
exec bash "${REPO_ROOT}/scripts/npu/test/base_test_full_dinov2_qwen3vl-8b_npu.sh" "$@"
