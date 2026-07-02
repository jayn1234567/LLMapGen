#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clone a platform-provided Python 3.11 Ascend environment, then install the
# lightweight LLMapGen runtime dependencies. The torch/torch_npu override is
# intentionally left to the DI smoke/train launcher so failures are visible at
# job startup.
SOURCE_ENV_DIR="${SOURCE_ENV_DIR:-/home/ma-user/anaconda3/envs/PyTorch-2.5.1}"
SOURCE_CONDA_ENV_NAME="${SOURCE_CONDA_ENV_NAME:-}"
TARGET_ENV_DIR="${TARGET_ENV_DIR:-/home/ma-user/.conda/envs/llmapgen-npu-py311}"
CLONE_FORCE="${CLONE_FORCE:-false}"

export SOURCE_ENV_DIR
export SOURCE_CONDA_ENV_NAME
export TARGET_ENV_DIR
export CLONE_FORCE
export INSTALL_TORCH_STACK="${INSTALL_TORCH_STACK:-false}"
export INSTALL_PROJECT_DEPS="${INSTALL_PROJECT_DEPS:-true}"

# Match the dependency family used by the DI NPU reference launcher more closely
# while keeping torch/torch_npu installation in the training smoke script.
export TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.56.2}"
export ACCELERATE_SPEC="${ACCELERATE_SPEC:-accelerate==1.6.0}"
export TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers>=0.22.0,<0.23.0}"
export HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:-huggingface-hub==0.36.2}"
export PEFT_SPEC="${PEFT_SPEC:-peft>=0.10.0,<0.20.0}"
export EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-packaging psutil}"

bash "${SCRIPT_DIR}/clone_llmapgen_npu_conda_env.sh"

"${TARGET_ENV_DIR}/bin/python" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Expected Python 3.11 for the torch_npu cp311 wheel, got {sys.version.split()[0]} at {sys.executable}"
    )
print(f"[npu-py311-env] python ok: {sys.executable} {sys.version.split()[0]}", flush=True)
PY

echo "[npu-py311-env] activate with: source ${TARGET_ENV_DIR}/activate_llmapgen_npu.sh"
