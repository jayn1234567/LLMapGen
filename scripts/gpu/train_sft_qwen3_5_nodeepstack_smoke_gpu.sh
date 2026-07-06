#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-checkpoints/qwen3.5-8b-instruct}
export CONDA_ENV=${CONDA_ENV:-fastvlm}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/debug_runs/qwen3_5_nodeepstack_smoke}
export DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}

exec bash "${SCRIPT_DIR}/train_sft_qwen3vl_nodeepstack_smoke_gpu.sh"
