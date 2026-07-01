#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SOURCE_CONDA_ENV_NAME="${SOURCE_CONDA_ENV_NAME:-${SOURCE_ENV_NAME:-}}"
SOURCE_ENV_DIR="${SOURCE_ENV_DIR:-${SOURCE_CONDA_ENV_DIR:-}}"
TARGET_CONDA_ENV_NAME="${CONDA_ENV_NAME:-${TARGET_CONDA_ENV_NAME:-llmapgen-npu}}"
TARGET_ENV_DIR="${ENV_DIR:-${TARGET_ENV_DIR:-}}"
CLONE_FORCE="${CLONE_FORCE:-false}"

# A cloned Ascend image environment often already has a working torch/torch-npu
# pair. Keep it by default; set INSTALL_TORCH_STACK=true only when you need to
# replace the stack explicitly.
INSTALL_TORCH_STACK="${INSTALL_TORCH_STACK:-false}"
INSTALL_PROJECT_DEPS="${INSTALL_PROJECT_DEPS:-true}"

TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.21.0}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-torchaudio==2.6.0}"
TORCH_NPU_SPEC="${TORCH_NPU_SPEC:-torch-npu==2.6.0}"

PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.51.0,<5.0.0}"
ACCELERATE_SPEC="${ACCELERATE_SPEC:-accelerate>=0.33.0,<1.0.0}"
HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:-huggingface-hub<1.0.0}"
TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers<0.22.0}"
PEFT_SPEC="${PEFT_SPEC:-peft>=0.12.0,<0.20.0}"

source_if_exists() {
  if [ -f "$1" ]; then
    local nounset_was_on=0
    case "$-" in
      *u*)
        nounset_was_on=1
        set +u
        ;;
    esac
    export ZSH_VERSION="${ZSH_VERSION:-}"
    # shellcheck disable=SC1090
    source "$1"
    if [ "${nounset_was_on}" = "1" ]; then
      set -u
    fi
    echo "[npu-clone-env] sourced $1"
  fi
}

source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source_if_exists "/usr/local/Ascend/nnal/atb/set_env.sh"

CONDA_SH_FROM_EXE="/nonexistent/conda.sh"
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_SH_FROM_EXE="$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
fi
for conda_sh in \
  "${CONDA_SH_FROM_EXE}" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [ -f "${conda_sh}" ]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
    break
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[npu-clone-env] conda was not found in PATH." >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ -z "${SOURCE_CONDA_ENV_NAME}" ] && [ -z "${SOURCE_ENV_DIR}" ]; then
  echo "[npu-clone-env] Set SOURCE_CONDA_ENV_NAME or SOURCE_ENV_DIR." >&2
  echo "[npu-clone-env] Available conda envs:" >&2
  conda env list >&2 || true
  exit 2
fi

if [ -n "${SOURCE_ENV_DIR}" ]; then
  SOURCE_ENV_DIR="$(cd "${SOURCE_ENV_DIR}" && pwd)"
  CLONE_SOURCE="${SOURCE_ENV_DIR}"
else
  CLONE_SOURCE="${SOURCE_CONDA_ENV_NAME}"
fi

target_exists=false
if [ -n "${TARGET_ENV_DIR}" ]; then
  if [ -x "${TARGET_ENV_DIR}/bin/python" ]; then
    target_exists=true
  fi
else
  if conda env list | awk '{print $1}' | grep -Fxq "${TARGET_CONDA_ENV_NAME}"; then
    target_exists=true
  fi
fi

if [ "${target_exists}" = "true" ] && [ "${CLONE_FORCE}" = "true" ]; then
  if [ -n "${TARGET_ENV_DIR}" ]; then
    conda env remove -y -p "${TARGET_ENV_DIR}" || true
  else
    conda env remove -y -n "${TARGET_CONDA_ENV_NAME}" || true
  fi
  target_exists=false
fi

if [ "${target_exists}" = "false" ]; then
  if [ -n "${TARGET_ENV_DIR}" ]; then
    conda create -y -p "${TARGET_ENV_DIR}" --clone "${CLONE_SOURCE}"
  else
    conda create -y -n "${TARGET_CONDA_ENV_NAME}" --clone "${CLONE_SOURCE}"
  fi
else
  echo "[npu-clone-env] target env already exists; reuse it. Set CLONE_FORCE=true to recreate."
fi

if [ -n "${TARGET_ENV_DIR}" ]; then
  conda activate "${TARGET_ENV_DIR}"
else
  conda activate "${TARGET_CONDA_ENV_NAME}"
fi

ENV_DIR="$(python - <<'PY'
from pathlib import Path
import sys
print(Path(sys.prefix).resolve())
PY
)"

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

if [ "${INSTALL_TORCH_STACK}" = "true" ]; then
  python -m pip install "${PIP_ARGS[@]}" \
    "${TORCH_SPEC}" \
    "${TORCHVISION_SPEC}" \
    "${TORCHAUDIO_SPEC}" \
    "${TORCH_NPU_SPEC}"
fi

if [ "${INSTALL_PROJECT_DEPS}" = "true" ]; then
  python -m pip install "${PIP_ARGS[@]}" \
    "numpy<2" \
    "pillow>=10.0.0" \
    "opencv-python-headless>=4.8.0" \
    "tqdm>=4.66.0" \
    "${TRANSFORMERS_SPEC}" \
    "${ACCELERATE_SPEC}" \
    "${HUGGINGFACE_HUB_SPEC}" \
    "${TOKENIZERS_SPEC}" \
    "${PEFT_SPEC}" \
    "safetensors>=0.4.3" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25.0" \
    "einops>=0.7.0" \
    "pyyaml>=6.0.1"
fi

if [ -n "${EXTRA_PIP_PACKAGES}" ]; then
  # shellcheck disable=SC2086
  python -m pip install "${PIP_ARGS[@]}" ${EXTRA_PIP_PACKAGES}
fi

ACTIVATE_SCRIPT="${ENV_DIR}/activate_llmapgen_npu.sh"
CONDA_ACTIVATE_TARGET="${ENV_DIR}"
if [ -z "${TARGET_ENV_DIR}" ]; then
  CONDA_ACTIVATE_TARGET="${TARGET_CONDA_ENV_NAME}"
fi
cat > "${ACTIVATE_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source_if_exists() {
  if [ -f "\$1" ]; then
    local nounset_was_on=0
    case "\$-" in
      *u*)
        nounset_was_on=1
        set +u
        ;;
    esac
    export ZSH_VERSION="\${ZSH_VERSION:-}"
    # shellcheck disable=SC1090
    source "\$1"
    if [ "\${nounset_was_on}" = "1" ]; then
      set -u
    fi
  fi
}
source_if_exists "\${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source_if_exists "/usr/local/Ascend/nnal/atb/set_env.sh"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ACTIVATE_TARGET}"
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

echo "[npu-clone-env] cloned source: ${CLONE_SOURCE}"
echo "[npu-clone-env] environment ready: ${ENV_DIR}"
echo "[npu-clone-env] activate with: source ${ACTIVATE_SCRIPT}"
