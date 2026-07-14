#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/mllm-dinov2-seg-npu-py311}"
PYTHON_VERSION="${MLLM_DINO_SEG_PYTHON_VERSION:-3.11}"
CLONE_FROM="${CLONE_FROM:-}"
RECREATE="${RECREATE:-false}"
HOST_PYTHON="${HOST_PYTHON:-}"
REQUIRE_NPU="${REQUIRE_NPU:-true}"

TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.22.1}"
TORCH_NPU_SPEC="${TORCH_NPU_SPEC:-torch_npu==2.7.1rc1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.56.2}"
TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers>=0.22.0,<0.23.0}"
HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:-huggingface-hub==0.36.2}"
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools==75.8.0}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==1.26.4}"
PROTOBUF_SPEC="${PROTOBUF_SPEC:-protobuf==4.25.7}"

MOXING_WHL_OBS_PATH="${MOXING_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}"
MOXING_WHL_LOCAL_PATH="${MOXING_WHL_LOCAL_PATH:-/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl}"
TORCH_NPU_WHL_OBS_PATH="${TORCH_NPU_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}"
TORCH_NPU_WHL_LOCAL_PATH="${TORCH_NPU_WHL_LOCAL_PATH:-/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}"

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES)$ ]]
}

source_if_exists() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    return
  fi
  local nounset_was_on=0
  case "$-" in
    *u*)
      nounset_was_on=1
      set +u
      ;;
  esac
  export ZSH_VERSION="${ZSH_VERSION:-}"
  # shellcheck disable=SC1090
  source "${path}"
  if [ "${nounset_was_on}" = "1" ]; then
    set -u
  fi
  echo "[dinov2-seg-env] sourced ${path}"
}

python_has_moxing() {
  "$1" - <<'PY' >/dev/null 2>&1
import moxing
assert hasattr(moxing, "file")
PY
}

resolve_host_python_with_moxing() {
  local candidate=""
  local resolved=""
  local candidates=()
  if [ -n "${HOST_PYTHON}" ]; then
    candidates+=("${HOST_PYTHON}")
  fi
  candidates+=(
    "/home/ma-user/anaconda3/envs/PyTorch-2.5.1/bin/python"
    "/modelarts/authoring/notebook-conda/bin/python"
    "/home/ma-user/anaconda3/bin/python"
    "/home/ma-user/miniconda3/bin/python"
    "python"
    "python3"
  )
  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" == */* ]]; then
      resolved="${candidate}"
    else
      resolved="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [ -n "${resolved}" ] && [ -x "${resolved}" ] && python_has_moxing "${resolved}"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  echo "[dinov2-seg-env] no host Python can import Huawei moxing-framework." >&2
  echo "[dinov2-seg-env] Set HOST_PYTHON, or place these wheels manually:" >&2
  echo "  ${MOXING_WHL_LOCAL_PATH}" >&2
  echo "  ${TORCH_NPU_WHL_LOCAL_PATH}" >&2
  return 2
}

CONDA_SH=""
for candidate in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [ -f "${candidate}" ]; then
    CONDA_SH="${candidate}"
    break
  fi
done
if [ -z "${CONDA_SH}" ] && command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
if [ ! -f "${CONDA_SH}" ]; then
  echo "[dinov2-seg-env] conda.sh was not found." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"

if bool_enabled "${RECREATE}" && [ -e "${ENV_DIR}" ]; then
  case "${ENV_DIR}" in
    "${HOME}"/.conda/envs/*|"${HOME}"/anaconda3/envs/*|"${HOME}"/miniconda3/envs/*)
      conda env remove -y -p "${ENV_DIR}"
      ;;
    *)
      echo "[dinov2-seg-env] refusing to recreate unexpected ENV_DIR=${ENV_DIR}" >&2
      exit 2
      ;;
  esac
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  if [ -n "${CLONE_FROM}" ]; then
    if [ ! -x "${CLONE_FROM}/bin/python" ]; then
      echo "[dinov2-seg-env] CLONE_FROM has no Python: ${CLONE_FROM}" >&2
      exit 2
    fi
    echo "[dinov2-seg-env] cloning ${CLONE_FROM} -> ${ENV_DIR}"
    conda create -y -p "${ENV_DIR}" --clone "${CLONE_FROM}"
  else
    echo "[dinov2-seg-env] creating ${ENV_DIR} with Python ${PYTHON_VERSION}"
    if ! conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip; then
      echo "[dinov2-seg-env] conda could not obtain Python ${PYTHON_VERSION}." >&2
      echo "[dinov2-seg-env] Retry with CLONE_FROM=/path/to/a/python3.11/conda/env." >&2
      exit 2
    fi
  fi
else
  echo "[dinov2-seg-env] reusing existing environment: ${ENV_DIR}"
fi
conda activate "${ENV_DIR}"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version} at {sys.executable}")
print(f"[dinov2-seg-env] python={sys.executable} version={sys.version.split()[0]}")
PY

DOWNLOAD_MOXING=false
DOWNLOAD_TORCH_NPU=false
if [ ! -f "${MOXING_WHL_LOCAL_PATH}" ]; then
  DOWNLOAD_MOXING=true
fi
if [ ! -f "${TORCH_NPU_WHL_LOCAL_PATH}" ]; then
  DOWNLOAD_TORCH_NPU=true
fi
if bool_enabled "${DOWNLOAD_MOXING}" || bool_enabled "${DOWNLOAD_TORCH_NPU}"; then
  HOST_PYTHON="$(resolve_host_python_with_moxing)"
  echo "[dinov2-seg-env] host Python for OBS wheels: ${HOST_PYTHON}"
  USE_MEMARTS=0 "${HOST_PYTHON}" - \
    "${MOXING_WHL_OBS_PATH}" "${MOXING_WHL_LOCAL_PATH}" \
    "${TORCH_NPU_WHL_OBS_PATH}" "${TORCH_NPU_WHL_LOCAL_PATH}" \
    "${DOWNLOAD_MOXING}" "${DOWNLOAD_TORCH_NPU}" <<'PY'
import sys
from pathlib import Path
import moxing as mox

mox_obs, mox_local, npu_obs, npu_local, get_mox, get_npu = sys.argv[1:]
truthy = {"1", "true", "True", "TRUE", "yes", "YES"}
for source, target, enabled in (
    (mox_obs, mox_local, get_mox),
    (npu_obs, npu_local, get_npu),
):
    if enabled not in truthy:
        continue
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    print(f"[dinov2-seg-env] download {source} -> {target}", flush=True)
    mox.file.copy(source, target)
PY
fi

export PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
python -m pip install --upgrade pip "${SETUPTOOLS_SPEC}" wheel
python -m pip install "${TORCH_SPEC}" "${TORCHVISION_SPEC}"

if [ -f "${TORCH_NPU_WHL_LOCAL_PATH}" ]; then
  python -m pip install --force-reinstall --no-deps "${TORCH_NPU_WHL_LOCAL_PATH}"
else
  python -m pip install "${TORCH_NPU_SPEC}"
fi

python -m pip uninstall -y moxing moxing-framework >/dev/null 2>&1 || true
if [ ! -f "${MOXING_WHL_LOCAL_PATH}" ]; then
  echo "[dinov2-seg-env] Huawei moxing-framework wheel is missing." >&2
  exit 2
fi
python -m pip install "${MOXING_WHL_LOCAL_PATH}"

python -m pip install \
  "${TRANSFORMERS_SPEC}" \
  "${TOKENIZERS_SPEC}" \
  "${HUGGINGFACE_HUB_SPEC}" \
  "safetensors>=0.4.3" \
  "Pillow>=10.0.0" \
  "tqdm>=4.66" \
  "requests>=2.31,<3" \
  "packaging>=23" \
  "absl-py>=2.0" \
  "attrs>=23.0" \
  "cloudpickle>=3.0" \
  "decorator>=5.1" \
  "ml-dtypes>=0.4,<1" \
  "scipy>=1.10,<2" \
  "tornado>=6.3" \
  "psutil>=5.9" \
  "${NUMPY_SPEC}" \
  "${PROTOBUF_SPEC}"

# Re-pin CANN-sensitive packages after dependency resolution.
python -m pip install "${SETUPTOOLS_SPEC}" "${NUMPY_SPEC}" "${PROTOBUF_SPEC}"
python -m pip install -e "${REPO_ROOT}" --no-deps

ACTIVATE_SCRIPT="${ENV_DIR}/activate_mllm_dinov2_seg_npu.sh"
cat > "${ACTIVATE_SCRIPT}" <<EOF
#!/usr/bin/env bash
source_if_exists() {
  if [ -f "\$1" ]; then
    local nounset_was_on=0
    case "\$-" in
      *u*) nounset_was_on=1; set +u ;;
    esac
    export ZSH_VERSION="\${ZSH_VERSION:-}"
    source "\$1"
    if [ "\${nounset_was_on}" = "1" ]; then
      set -u
    fi
  fi
}
source_if_exists "\${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source "${CONDA_SH}"
conda activate "${ENV_DIR}"
export PYTHON="${ENV_DIR}/bin/python"
export PYTHON_BIN="${ENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="\${TOKENIZERS_PARALLELISM:-false}"
export HCCL_CONNECT_TIMEOUT="\${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="\${HCCL_EXEC_TIMEOUT:-7200}"
EOF
chmod +x "${ACTIVATE_SCRIPT}"

source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"

REQUIRE_NPU="${REQUIRE_NPU}" python - <<'PY'
import json
import os
import sys

import numpy
import torch
import torch_npu
import transformers
import moxing
from transformers import Dinov2Model

del Dinov2Model
result = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "moxing_file_api": hasattr(moxing, "file"),
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
}
print(json.dumps(result, indent=2, ensure_ascii=True))
if int(numpy.__version__.split(".")[0]) >= 2:
    raise SystemExit("NumPy 2.x is incompatible with the current CANN stack.")
if not result["moxing_file_api"]:
    raise SystemExit("Huawei moxing-framework was not imported correctly.")
required = os.environ.get("REQUIRE_NPU", "true").lower() in {"1", "true", "yes"}
if required and not result["npu_available"]:
    raise SystemExit("NPU is not available in the newly created environment.")
if result["npu_available"]:
    value = (torch.ones(1, device="npu") + 1).cpu().item()
    if value != 2:
        raise SystemExit(f"Unexpected NPU tensor smoke result: {value}")
PY

echo "[dinov2-seg-env] environment ready: ${ENV_DIR}"
echo "[dinov2-seg-env] activate with: source ${ACTIVATE_SCRIPT}"
