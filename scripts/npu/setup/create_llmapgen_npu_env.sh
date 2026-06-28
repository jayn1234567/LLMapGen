#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_DIR="${ENV_DIR:-${REPO_ROOT}/.venv-llmapgen-npu}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
USE_CONDA="${USE_CONDA:-false}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# torch/torch-npu must match the CANN stack installed on the Ascend image.
# Override these in DI when the platform provides a different compatibility matrix.
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.21.0}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-torchaudio==2.6.0}"
TORCH_NPU_SPEC="${TORCH_NPU_SPEC:-torch-npu==2.6.0}"

PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"

source_if_exists() {
  if [ -f "$1" ]; then
    # shellcheck disable=SC1090
    source "$1"
    echo "[npu-env] sourced $1"
  fi
}

source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source_if_exists "/usr/local/Ascend/nnal/atb/set_env.sh"

if [ "${USE_CONDA}" = "true" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[npu-env] USE_CONDA=true but conda was not found in PATH." >&2
    exit 2
  fi
  if [ ! -x "${ENV_DIR}/bin/python" ]; then
    conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_DIR}"
else
  if [ ! -x "${ENV_DIR}/bin/python" ]; then
    "${PYTHON_BOOTSTRAP}" -m venv "${ENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

PIP_ARGS=()
if [ -n "${PIP_INDEX_URL}" ]; then
  PIP_ARGS+=(--index-url "${PIP_INDEX_URL}")
fi
if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then
  PIP_ARGS+=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
fi
if [ -n "${PIP_TRUSTED_HOST}" ]; then
  PIP_ARGS+=(--trusted-host "${PIP_TRUSTED_HOST}")
fi

python -m pip install "${PIP_ARGS[@]}" \
  "${TORCH_SPEC}" \
  "${TORCHVISION_SPEC}" \
  "${TORCHAUDIO_SPEC}" \
  "${TORCH_NPU_SPEC}"

python -m pip install "${PIP_ARGS[@]}" \
  "numpy<2" \
  "pillow>=10.0.0" \
  "opencv-python-headless>=4.8.0" \
  "tqdm>=4.66.0" \
  "transformers>=4.51.0" \
  "accelerate>=0.33.0" \
  "peft>=0.12.0" \
  "safetensors>=0.4.3" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "einops>=0.7.0" \
  "pyyaml>=6.0.1"

if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then
  python -m pip install "${PIP_ARGS[@]}" flash-attn --no-build-isolation
fi

if [ -n "${EXTRA_PIP_PACKAGES}" ]; then
  # shellcheck disable=SC2086
  python -m pip install "${PIP_ARGS[@]}" ${EXTRA_PIP_PACKAGES}
fi

ACTIVATE_SCRIPT="${ENV_DIR}/activate_llmapgen_npu.sh"
cat > "${ACTIVATE_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source_if_exists() {
  if [ -f "\$1" ]; then
    # shellcheck disable=SC1090
    source "\$1"
  fi
}
source_if_exists "\${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source_if_exists "/usr/local/Ascend/nnal/atb/set_env.sh"
source "${ENV_DIR}/bin/activate"
export PYTHON_BIN="${ENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="\${TOKENIZERS_PARALLELISM:-false}"
export HCCL_CONNECT_TIMEOUT="\${HCCL_CONNECT_TIMEOUT:-1800}"
EOF
chmod +x "${ACTIVATE_SCRIPT}"

python - <<'PY'
import json
import sys

result = {"python": sys.executable}
try:
    import torch
    result["torch"] = torch.__version__
except Exception as exc:
    result["torch_error"] = repr(exc)
try:
    import torch_npu
    result["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
except Exception as exc:
    result["torch_npu_error"] = repr(exc)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

echo "[npu-env] environment ready: ${ENV_DIR}"
echo "[npu-env] activate with: source ${ACTIVATE_SCRIPT}"
