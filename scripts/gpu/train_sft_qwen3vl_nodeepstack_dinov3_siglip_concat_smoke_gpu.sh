#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VISION_BACKBONE=${VISION_BACKBONE:-dinov3_siglip_concat}
exec bash "${SCRIPT_DIR}/train_sft_qwen3vl_nodeepstack_smoke_gpu.sh"
