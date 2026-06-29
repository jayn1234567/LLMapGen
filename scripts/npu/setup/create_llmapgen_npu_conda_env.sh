#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_CONDA=true
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-llmapgen-npu}"

exec bash "${SCRIPT_DIR}/create_llmapgen_npu_env.sh"
