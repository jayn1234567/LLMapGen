#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec bash "${SCRIPT_DIR}/test_full_dinov2_qwen3vl-8b_npu.sh" "$@"
