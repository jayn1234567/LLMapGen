#!/usr/bin/env bash
# Build the discrete-coordinate SFT environment by cloning the local mapgen env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SOURCE_ENV_NAME="${SOURCE_ENV_NAME:-mapgen}"
SOURCE_ENV_PREFIX="${SOURCE_ENV_PREFIX:-}"
ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/mllm-coordtokens-npu-py311}"
RECREATE_ENV="${RECREATE_ENV:-false}"
REQUIRE_NPU="${REQUIRE_NPU:-true}"
HOST_PYTHON="${HOST_PYTHON:-}"

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

CONDA_SH="$(find_conda_sh || true)"
if [ -z "${CONDA_SH}" ]; then
  echo "[coordtoken-env] conda.sh was not found." >&2
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
    echo "[coordtoken-env] unable to activate source conda env: ${SOURCE_ENV_NAME}" >&2
    echo "[coordtoken-env] set SOURCE_ENV_PREFIX=/absolute/path/to/mapgen if it is prefix-only." >&2
    exit 2
  fi
  SOURCE_ENV_PREFIX="${CONDA_PREFIX}"
  conda deactivate || true
  if [ "${nounset_was_on}" = "1" ]; then
    set -u
  fi
fi

if [ ! -x "${SOURCE_ENV_PREFIX}/bin/python" ]; then
  echo "[coordtoken-env] source env has no Python: ${SOURCE_ENV_PREFIX}" >&2
  exit 2
fi

SOURCE_ENV_PREFIX="$(cd "${SOURCE_ENV_PREFIX}" && pwd)"
SOURCE_PYTHON_VERSION="$(${SOURCE_ENV_PREFIX}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${SOURCE_PYTHON_VERSION}" != "3.11" ]; then
  echo "[coordtoken-env] mapgen must provide Python 3.11, got ${SOURCE_PYTHON_VERSION}: ${SOURCE_ENV_PREFIX}" >&2
  exit 2
fi
if [ "${SOURCE_ENV_PREFIX}" = "${ENV_DIR}" ]; then
  echo "[coordtoken-env] source and target environments must be different: ${ENV_DIR}" >&2
  exit 2
fi

echo "[coordtoken-env] source env: ${SOURCE_ENV_PREFIX} (Python ${SOURCE_PYTHON_VERSION})"
echo "[coordtoken-env] target env: ${ENV_DIR}"

BASE_SETUP="${SCRIPT_DIR}/create_mllm_dinov2_seg_npu_env.sh"
if [ ! -f "${BASE_SETUP}" ]; then
  echo "[coordtoken-env] base setup script not found: ${BASE_SETUP}" >&2
  exit 2
fi

if [ -z "${HOST_PYTHON}" ]; then
  HOST_PYTHON="${SOURCE_ENV_PREFIX}/bin/python"
fi

CLONE_FROM="${SOURCE_ENV_PREFIX}" \
ENV_DIR="${ENV_DIR}" \
RECREATE="${RECREATE_ENV}" \
REQUIRE_NPU="${REQUIRE_NPU}" \
HOST_PYTHON="${HOST_PYTHON}" \
bash "${BASE_SETUP}"

# shellcheck disable=SC1090
source "${ENV_DIR}/activate_mllm_dinov2_seg_npu.sh"
export PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "[coordtoken-env] installing full SFT/DeepSpeed dependency set"
python -m pip install \
  "accelerate==1.6.0" \
  "deepspeed==0.14.4" \
  "sentencepiece>=0.1.99" \
  "tiktoken>=0.7.0" \
  "shortuuid>=1.0.13" \
  "peft>=0.10.0" \
  "pydantic>=2.0" \
  "markdown2[all]" \
  "scikit-learn>=1.2,<2" \
  "einops>=0.6" \
  "einops-exts>=0.0.4" \
  "timm>=0.9.0" \
  "opencv-python-headless==4.11.0.86" \
  "loguru>=0.7.0" \
  "shapely>=2.0.0" \
  "wandb>=0.17" \
  "swanlab==0.7.19" \
  "urllib3==1.26.15" \
  "ninja>=1.11" \
  "hjson>=3.1"

# Keep CANN-sensitive versions last so transitive dependencies cannot upgrade them.
python -m pip install \
  "setuptools==75.8.0" \
  "numpy==1.26.4" \
  "protobuf==4.25.7" \
  "opencv-python-headless==4.11.0.86" \
  "transformers==4.56.2" \
  "tokenizers>=0.22.0,<0.23.0" \
  "huggingface-hub==0.36.2"
python -m pip install -e "${REPO_ROOT}" --no-deps

ACTIVATE_SCRIPT="${ENV_DIR}/activate_mllm_coordtokens_npu.sh"
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

import accelerate
import deepspeed
import moxing
import numpy
import torch
import torch_npu
import transformers

result = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "deepspeed": deepspeed.__version__,
    "moxing_file_api": hasattr(moxing, "file"),
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
}
print(json.dumps(result, indent=2, ensure_ascii=True))
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version}")
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"Expected NumPy 1.26.4 for CANN, got {numpy.__version__}")
if transformers.__version__ != "4.56.2":
    raise SystemExit(f"Expected Transformers 4.56.2, got {transformers.__version__}")
if not result["moxing_file_api"]:
    raise SystemExit("Huawei moxing-framework is not available.")
required = os.environ.get("REQUIRE_NPU", "true").lower() in {"1", "true", "yes"}
if required and not result["npu_available"]:
    raise SystemExit("NPU is not available in the discrete-coordinate environment.")
if result["npu_available"]:
    value = (torch.ones(4, device="npu") * 2).sum().cpu().item()
    if value != 8:
        raise SystemExit(f"Unexpected NPU tensor result: {value}")
PY

echo "[coordtoken-env] environment ready: ${ENV_DIR}"
echo "[coordtoken-env] activate with: source ${ACTIVATE_SCRIPT}"
