#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VISION_BACKBONE=dinov2
export DATASET_PHASE=phase_a
export MAP_TASK=lane
exec bash "${SCRIPT_DIR}/run_sft_nodeepstack_npu.sh"

