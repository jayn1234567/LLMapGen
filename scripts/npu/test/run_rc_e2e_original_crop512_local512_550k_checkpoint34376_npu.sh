#!/usr/bin/env bash
set -euo pipefail

# Original RC E2E splitter at 512x512/stride 512, followed by full-image
# local512-550k checkpoint-34376 inference.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

export ORIGINAL_PATCH_SIZE=512
export E2E_VIEW_MODE=local512
export E2E_PROMPT_PROFILE=current
export E2E_DATASET_TAG=${E2E_DATASET_TAG:-original_crop512_black1_local512_550k_v1}
export E2E_DATASET_ROOT=${E2E_DATASET_ROOT:-/cache/jn/e2e_eval/e2e_data_${E2E_DATASET_TAG}}
export BASE_INFERENCE_SCRIPT=${BASE_INFERENCE_SCRIPT:-${SCRIPT_DIR}/run_rc_e2e_local512_550k_checkpoint34376_npu.sh}

exec bash "${SCRIPT_DIR}/run_rc_e2e_original_crop256_local256_550k_checkpoint34376_npu.sh"
