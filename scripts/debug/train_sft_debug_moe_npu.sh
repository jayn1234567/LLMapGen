#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VISION_BACKBONE=${VISION_BACKBONE:-multi_moe}
exec bash "${SCRIPT_DIR}/train_sft_debug_npu.sh"
