#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
exec bash "${REPO_ROOT}/scripts/train_sft_dinov3_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh"
