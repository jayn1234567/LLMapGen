#!/usr/bin/env bash
set -euo pipefail

# Build an isolated Python 3.11 inference environment for the CANN stack that
# requires PyTorch/torch-npu 2.4.0. The source mapgen environment only provides
# the local Python/conda payload; runtime packages are pinned below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SOURCE_ENV_NAME="${SOURCE_ENV_NAME:-mapgen}"
SOURCE_ENV_PREFIX="${SOURCE_ENV_PREFIX:-}"
ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}"
RECREATE_ENV="${RECREATE_ENV:-false}"
REQUIRE_NPU="${REQUIRE_NPU:-true}"
TORCH_NPU_INSTALL_SPEC="${TORCH_NPU_INSTALL_SPEC:-torch-npu==2.4.0}"

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES)$ ]]
}

find_conda_sh() {
  local candidate=""
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    candidate="$(conda info --base)/etc/profile.d/conda.sh"
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  return 1
}

source_if_exists() {
  if [ -f "$1" ]; then
    local nounset_was_on=0
    case "$-" in
      *u*) nounset_was_on=1; set +u ;;
    esac
    export ZSH_VERSION="${ZSH_VERSION:-}"
    # shellcheck disable=SC1090
    source "$1"
    if [ "${nounset_was_on}" = "1" ]; then
      set -u
    fi
  fi
}

CONDA_SH="$(find_conda_sh || true)"
if [ -z "${CONDA_SH}" ]; then
  echo "[torch240-env] conda.sh was not found." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"

if [ -z "${SOURCE_ENV_PREFIX}" ]; then
  nounset_was_on=0
  case "$-" in
    *u*) nounset_was_on=1; set +u ;;
  esac
  if ! conda activate "${SOURCE_ENV_NAME}"; then
    echo "[torch240-env] unable to activate source environment: ${SOURCE_ENV_NAME}" >&2
    echo "[torch240-env] set SOURCE_ENV_PREFIX to the absolute mapgen prefix." >&2
    exit 2
  fi
  SOURCE_ENV_PREFIX="${CONDA_PREFIX}"
  conda deactivate || true
  if [ "${nounset_was_on}" = "1" ]; then
    set -u
  fi
fi

if [ ! -x "${SOURCE_ENV_PREFIX}/bin/python" ]; then
  echo "[torch240-env] source environment has no Python: ${SOURCE_ENV_PREFIX}" >&2
  exit 2
fi
SOURCE_ENV_PREFIX="$(cd "${SOURCE_ENV_PREFIX}" && pwd)"
SOURCE_PYTHON_VERSION="$(${SOURCE_ENV_PREFIX}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${SOURCE_PYTHON_VERSION}" != "3.11" ]; then
  echo "[torch240-env] source environment must provide Python 3.11, got ${SOURCE_PYTHON_VERSION}." >&2
  exit 2
fi
if [ "${SOURCE_ENV_PREFIX}" = "${ENV_DIR}" ]; then
  echo "[torch240-env] source and target environments must differ." >&2
  exit 2
fi
case "${ENV_DIR}" in
  /home/ma-user/.conda/envs/*) ;;
  *)
    echo "[torch240-env] refusing target outside /home/ma-user/.conda/envs: ${ENV_DIR}" >&2
    exit 2
    ;;
esac

if [ -d "${ENV_DIR}" ] && bool_enabled "${RECREATE_ENV}"; then
  echo "[torch240-env] removing existing target environment: ${ENV_DIR}"
  conda env remove -y -p "${ENV_DIR}" || rm -rf "${ENV_DIR}"
fi
if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[torch240-env] cloning ${SOURCE_ENV_PREFIX} -> ${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" --clone "${SOURCE_ENV_PREFIX}"
else
  echo "[torch240-env] updating existing environment: ${ENV_DIR}"
fi

source_if_exists "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
conda activate "${ENV_DIR}"
export PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "[torch240-env] installing exact CANN-compatible runtime versions"
python -m pip install --upgrade pip setuptools==75.8.0 wheel
python -m pip uninstall -y torch torch-npu torch_npu torchvision opencv-python opencv-python-headless filelock || true
python -m pip install --no-cache-dir --force-reinstall \
  "torch==2.4.0" \
  "${TORCH_NPU_INSTALL_SPEC}" \
  "torchvision==0.19.0" \
  "opencv-python-headless==4.10.0.84" \
  "filelock==3.22.0"

python -m pip install \
  "transformers==4.56.2" \
  "tokenizers>=0.22.0,<0.23.0" \
  "huggingface-hub==0.36.2" \
  "accelerate==1.6.0" \
  "deepspeed==0.14.4" \
  "safetensors>=0.4.3" \
  "sentencepiece==0.2.1" \
  "tiktoken==0.13.0" \
  "shortuuid==1.0.13" \
  "peft==0.19.1" \
  "Pillow>=10.0.0" \
  "pydantic>=2.0" \
  "markdown2[all]" \
  "packaging>=23" \
  "einops==0.8.2" \
  "einops-exts==0.0.4" \
  "timm==1.0.27" \
  "numpy==1.26.4" \
  "protobuf==4.25.7" \
  "scipy>=1.10,<2" \
  "scikit-learn>=1.2,<2" \
  "shapely==2.1.2" \
  "loguru==0.7.3" \
  "urllib3==1.26.15"

# Reassert ABI-sensitive versions after all transitive dependency resolution.
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "torch==2.4.0" \
  "${TORCH_NPU_INSTALL_SPEC}" \
  "torchvision==0.19.0" \
  "opencv-python-headless==4.10.0.84" \
  "filelock==3.22.0" \
  "numpy==1.26.4" \
  "protobuf==4.25.7"
python -m pip install -e "${REPO_ROOT}" --no-deps

ACTIVATE_SCRIPT="${ENV_DIR}/activate_mllm_infer_torch240.sh"
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

# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
REQUIRE_NPU="${REQUIRE_NPU}" python - <<'PY'
import json
import os
import sys

import cv2
import deepspeed
import filelock
import numpy
import torch
import torch_npu
import torchvision

result = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "deepspeed": deepspeed.__version__,
    "filelock": filelock.__version__,
    "opencv": cv2.__version__,
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "torchvision": torchvision.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
}
print(json.dumps(result, indent=2, ensure_ascii=True))

expected = {
    "deepspeed": "0.14.4",
    "filelock": "3.22.0",
    "opencv": "4.10.0",
    "torch": "2.4.0",
    "torch_npu": "2.4.0",
    "torchvision": "0.19.0",
}
for key, version in expected.items():
    actual = str(result[key]).split("+")[0]
    if actual != version:
        raise SystemExit(f"Expected {key}={version}, got {result[key]}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version}")
required = os.environ.get("REQUIRE_NPU", "true").lower() in {"1", "true", "yes"}
if required and not result["npu_available"]:
    raise SystemExit("NPU is not available in the torch-2.4.0 inference environment.")
if result["npu_available"]:
    value = (torch.ones(4, device="npu") * 2).sum().cpu().item()
    if value != 8:
        raise SystemExit(f"Unexpected NPU tensor result: {value}")
PY

echo "[torch240-env] environment ready: ${ENV_DIR}"
echo "[torch240-env] activate with: source ${ACTIVATE_SCRIPT}"
