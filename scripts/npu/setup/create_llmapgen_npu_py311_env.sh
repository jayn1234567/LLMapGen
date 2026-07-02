#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/llmapgen-npu-py311}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
# This script is specifically for the torch_npu cp311 wheel. Ignore ambient
# PYTHON_VERSION values that may be exported by the platform base environment.
PYTHON_VERSION="${LLMAPGEN_PY311_PYTHON_VERSION:-3.11}"
PYTHON_VERSION="${PYTHON_VERSION#python=}"
PYTHON_VERSION="${PYTHON_VERSION#python}"
if [ -z "${PYTHON_VERSION}" ]; then
  PYTHON_VERSION="3.11"
fi
CLONE_FORCE="${CLONE_FORCE:-false}"
HOST_PYTHON="${HOST_PYTHON:-}"

TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1}"
TORCH_NPU_SPEC="${TORCH_NPU_SPEC:-torch_npu==2.7.1rc1}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.22.1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.56.2}"
TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers>=0.22.0,<0.23.0}"
ACCELERATE_SPEC="${ACCELERATE_SPEC:-accelerate==1.6.0}"
HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:-huggingface-hub==0.36.2}"
PEFT_SPEC="${PEFT_SPEC:-peft>=0.10.0,<0.20.0}"

ENABLE_MOXING_INSTALL="${ENABLE_MOXING_INSTALL:-true}"
ENABLE_TORCH_NPU_WHL="${ENABLE_TORCH_NPU_WHL:-true}"
MOXING_WHL_OBS_PATH="${MOXING_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}"
MOXING_WHL_LOCAL_PATH="${MOXING_WHL_LOCAL_PATH:-/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl}"
TORCH_NPU_WHL_OBS_PATH="${TORCH_NPU_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}"
TORCH_NPU_WHL_LOCAL_PATH="${TORCH_NPU_WHL_LOCAL_PATH:-/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}"

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES)$ ]]
}

python_has_moxing() {
  "$1" - <<'PY' >/dev/null 2>&1
import moxing  # noqa: F401
PY
}

resolve_host_python_with_moxing() {
  local candidates=()
  local candidate=""
  local resolved=""
  local seen=":"

  if [ -n "${HOST_PYTHON}" ]; then
    candidates+=("${HOST_PYTHON}")
  fi
  candidates+=(
    "/home/ma-user/anaconda3/bin/python"
    "/home/ma-user/miniconda3/bin/python"
    "/modelarts/authoring/notebook-conda/bin/python"
    "python"
    "python3"
    "/usr/bin/python3"
  )

  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" == */* ]]; then
      resolved="${candidate}"
    else
      resolved="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [ -z "${resolved}" ] || [ ! -x "${resolved}" ]; then
      continue
    fi
    case "${seen}" in
      *:"${resolved}":*) continue ;;
    esac
    seen="${seen}${resolved}:"
    if python_has_moxing "${resolved}"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done

  echo "[npu-py311-env] could not find a host python that can import moxing." >&2
  echo "[npu-py311-env] Set HOST_PYTHON=/path/to/python_with_moxing, or pre-place:" >&2
  echo "[npu-py311-env]   ${MOXING_WHL_LOCAL_PATH}" >&2
  echo "[npu-py311-env]   ${TORCH_NPU_WHL_LOCAL_PATH}" >&2
  return 2
}

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
    echo "[npu-py311-env] sourced $1"
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
  echo "[npu-py311-env] conda was not found in PATH." >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[npu-py311-env] requested python version: ${PYTHON_VERSION}"

if bool_enabled "${CLONE_FORCE}"; then
  if [ -n "${CONDA_ENV_NAME}" ]; then
    conda env remove -y -n "${CONDA_ENV_NAME}" || true
  elif [ -x "${ENV_DIR}/bin/python" ]; then
    conda env remove -y -p "${ENV_DIR}" || true
  fi
fi

if [ -n "${CONDA_ENV_NAME}" ]; then
  if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${CONDA_ENV_NAME}"
else
  if [ ! -x "${ENV_DIR}/bin/python" ]; then
    conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_DIR}"
fi

ENV_DIR="$(python - <<'PY'
from pathlib import Path
import sys
print(Path(sys.prefix).resolve())
PY
)"

DOWNLOAD_MOXING_WHL=false
DOWNLOAD_TORCH_NPU_WHL=false
if bool_enabled "${ENABLE_MOXING_INSTALL}" && [ ! -f "${MOXING_WHL_LOCAL_PATH}" ]; then
  DOWNLOAD_MOXING_WHL=true
fi
if bool_enabled "${ENABLE_TORCH_NPU_WHL}" && [ ! -f "${TORCH_NPU_WHL_LOCAL_PATH}" ]; then
  DOWNLOAD_TORCH_NPU_WHL=true
fi

if bool_enabled "${DOWNLOAD_MOXING_WHL}" || bool_enabled "${DOWNLOAD_TORCH_NPU_WHL}"; then
  HOST_PYTHON="$(resolve_host_python_with_moxing)"
  echo "[npu-py311-env] host python for OBS downloads: ${HOST_PYTHON}"
  USE_MEMARTS=0 "${HOST_PYTHON}" - "$MOXING_WHL_OBS_PATH" "$MOXING_WHL_LOCAL_PATH" "$TORCH_NPU_WHL_OBS_PATH" "$TORCH_NPU_WHL_LOCAL_PATH" "${DOWNLOAD_MOXING_WHL}" "${DOWNLOAD_TORCH_NPU_WHL}" <<'PY'
import sys
import moxing as mox

moxing_obs, moxing_local, torch_obs, torch_local, install_moxing, install_torch_wheel = sys.argv[1:]
truthy = {"1", "true", "True", "TRUE", "yes", "YES"}
if install_moxing in truthy:
    mox.file.copy(moxing_obs, moxing_local)
    print(f"[npu-py311-env] downloaded moxing wheel: {moxing_local}", flush=True)
if install_torch_wheel in truthy:
    mox.file.copy(torch_obs, torch_local)
    print(f"[npu-py311-env] downloaded torch_npu wheel: {torch_local}", flush=True)
PY
else
  echo "[npu-py311-env] OBS wheels already exist locally; skip host moxing downloads."
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
python -m pip install --upgrade pip setuptools wheel

if bool_enabled "${ENABLE_MOXING_INSTALL}" && [ -f "${MOXING_WHL_LOCAL_PATH}" ]; then
  python -m pip uninstall -y moxing-framework || true
  python -m pip install "${MOXING_WHL_LOCAL_PATH}"
fi

python -m pip install \
  "${TORCH_SPEC}" \
  "${TORCH_NPU_SPEC}"

if bool_enabled "${ENABLE_TORCH_NPU_WHL}" && [ -f "${TORCH_NPU_WHL_LOCAL_PATH}" ]; then
  python -m pip install --force-reinstall "${TORCH_NPU_WHL_LOCAL_PATH}"
fi

python -m pip install \
  "sentencepiece>=0.1.99" \
  "tiktoken>=0.7.0" \
  "${TRANSFORMERS_SPEC}" \
  "${TOKENIZERS_SPEC}"

python -m pip install \
  "${ACCELERATE_SPEC}" \
  "deepspeed==0.14.4" \
  "safetensors>=0.4.3" \
  "packaging" \
  "Pillow>=10.0.0" \
  "${TORCHVISION_SPEC}"

python -m pip install \
  "shortuuid" \
  "${PEFT_SPEC}" \
  "attrs>=23.0.0" \
  "cloudpickle>=3.0.0" \
  "decorator>=5.1.1" \
  "ml-dtypes>=0.4.0" \
  "pydantic" \
  "markdown2[all]" \
  "numpy>=1.26" \
  "scipy>=1.10" \
  "scikit-learn>=1.2"

python -m pip install \
  "requests" \
  "absl-py>=2.0.0" \
  "uvicorn" \
  "fastapi" \
  "einops>=0.6" \
  "einops-exts>=0.0.4" \
  "timm>=0.9.0" \
  "opencv-python-headless>=4.8.0" \
  "tornado>=6.3.0"

python -m pip install \
  "loguru>=0.7.0" \
  "shapely>=2.0.0" \
  "wandb" \
  "swanlab" \
  "${HUGGINGFACE_HUB_SPEC}" \
  "urllib3==1.26.15" \
  "psutil"

ACTIVATE_SCRIPT="${ENV_DIR}/activate_llmapgen_npu.sh"
CONDA_ACTIVATE_TARGET="${ENV_DIR}"
if [ -n "${CONDA_ENV_NAME}" ]; then
  CONDA_ACTIVATE_TARGET="${CONDA_ENV_NAME}"
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

result = {"python": sys.executable, "python_version": sys.version.split()[0]}
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version.split()[0]} at {sys.executable}")
try:
    import moxing  # noqa: F401
    result["moxing"] = "ok"
except Exception as exc:
    result["moxing_error"] = repr(exc)
try:
    import torch
    result["torch"] = torch.__version__
    result["torch_npu_available_attr"] = hasattr(torch, "npu")
    if hasattr(torch, "npu"):
        try:
            result["torch_npu_is_available"] = bool(torch.npu.is_available())
            result["torch_npu_device_count"] = int(torch.npu.device_count())
        except Exception as exc:
            result["torch_npu_runtime_error"] = repr(exc)
except Exception as exc:
    result["torch_error"] = repr(exc)
try:
    import torch_npu
    result["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
except Exception as exc:
    result["torch_npu_error"] = repr(exc)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

echo "[npu-py311-env] environment ready: ${ENV_DIR}"
echo "[npu-py311-env] activate with: source ${ACTIVATE_SCRIPT}"
