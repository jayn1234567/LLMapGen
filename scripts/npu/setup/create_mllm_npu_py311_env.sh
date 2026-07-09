#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Create a clean Ascend/NPU conda environment for MLLM training.
#
# Usage:
#   bash scripts/npu/setup/create_mllm_npu_py311_env.sh
#
# Optional overrides:
#   ENV_NAME=mllm-npu-py311
#   ENV_PREFIX=/home/ma-user/.conda/envs/mllm-npu-py311
#   PYTHON_VERSION=3.11
#   CONDA_CHANNEL=http://192.168.214.30:8088/repository/conda-proxy/main
#   FORCE_RECREATE=true
# ============================================================

ENV_NAME=${ENV_NAME:-mllm-npu-py311}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
ENV_PREFIX=${ENV_PREFIX:-${HOME}/.conda/envs/${ENV_NAME}}
FORCE_RECREATE=${FORCE_RECREATE:-false}
CONDA_CHANNEL=${CONDA_CHANNEL:-}
PIP_INDEX_URL=${PIP_INDEX_URL:-http://repo.huaweicloud.com/repository/pypi/simple/}
MOXING_WHEEL_OBS=${MOXING_WHEEL_OBS:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}
MOXING_WHEEL_LOCAL=${MOXING_WHEEL_LOCAL:-/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl}
TORCH_NPU_WHEEL_OBS=${TORCH_NPU_WHEEL_OBS:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
TORCH_NPU_WHEEL_LOCAL=${TORCH_NPU_WHEEL_LOCAL:-/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)

log() {
  echo "[mllm-npu-env] $*"
}

die() {
  echo "[mllm-npu-env][ERROR] $*" >&2
  exit 1
}

as_bool() {
  [[ "${1:-}" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]]
}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  log "sourced /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
  log "sourced /usr/local/Ascend/nnal/atb/set_env.sh"
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

command -v conda >/dev/null 2>&1 || die "conda not found. Source your conda profile first."

if [ -d "${ENV_PREFIX}" ] && as_bool "${FORCE_RECREATE}"; then
  log "removing existing environment: ${ENV_PREFIX}"
  conda env remove -p "${ENV_PREFIX}" -y || rm -rf "${ENV_PREFIX}"
fi

if [ ! -d "${ENV_PREFIX}" ]; then
  log "creating conda environment: ${ENV_PREFIX} python=${PYTHON_VERSION}"
  CONDA_CREATE_ARGS=(create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip setuptools wheel)
  if [ -n "${CONDA_CHANNEL}" ]; then
    CONDA_CREATE_ARGS+=(-c "${CONDA_CHANNEL}" --override-channels)
  fi
  conda "${CONDA_CREATE_ARGS[@]}"
else
  log "environment already exists: ${ENV_PREFIX}"
fi

PYTHON="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"
ACTIVATE_SCRIPT="${ENV_PREFIX}/activate_mllm_npu.sh"

[ -x "${PYTHON}" ] || die "python not found in environment: ${PYTHON}"

log "python=$("${PYTHON}" -V 2>&1)"
log "pip=$("${PIP}" --version)"

pip_install() {
  "${PIP}" install --trusted-host repo.huaweicloud.com --index-url "${PIP_INDEX_URL}" "$@"
}

download_with_moxing() {
  local obs_path="$1"
  local local_path="$2"
  shift 2 || true
  local candidates=()
  if [ -n "${MOXING_BOOTSTRAP_PYTHON:-}" ]; then
    candidates+=("${MOXING_BOOTSTRAP_PYTHON}")
  fi
  candidates+=(
    "${PYTHON}"
    "/modelarts/authoring/notebook-conda/bin/python"
    "/home/ma-user/anaconda3/bin/python"
  )
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi

  for candidate in "${candidates[@]}"; do
    [ -x "${candidate}" ] || continue
    if USE_MEMARTS=0 "${candidate}" - "$obs_path" "$local_path" <<'PY'
import sys
try:
    import moxing as mox
except Exception:
    raise SystemExit(2)
src, dst = sys.argv[1], sys.argv[2]
mox.file.copy_parallel(src, dst) if src.rstrip("/").endswith("/") else mox.file.copy(src, dst)
print(dst)
PY
    then
      return 0
    fi
  done
  return 1
}

log "upgrading pip/setuptools/wheel"
pip_install --upgrade pip setuptools wheel

log "installing moxing-framework"
if [ ! -f "${MOXING_WHEEL_LOCAL}" ]; then
  if ! download_with_moxing "${MOXING_WHEEL_OBS}" "${MOXING_WHEEL_LOCAL}"; then
    die "failed to download moxing wheel. Set MOXING_WHEEL_LOCAL to an existing wheel or run from a platform python that can import moxing."
  fi
fi
pip_install --force-reinstall "${MOXING_WHEEL_LOCAL}"

log "installing torch and torch_npu"
pip_install torch==2.7.1 torch_npu==2.7.1rc1
if [ ! -f "${TORCH_NPU_WHEEL_LOCAL}" ]; then
  USE_MEMARTS=0 "${PYTHON}" - "$TORCH_NPU_WHEEL_OBS" "$TORCH_NPU_WHEEL_LOCAL" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
print(sys.argv[2])
PY
fi
pip_install --force-reinstall "${TORCH_NPU_WHEEL_LOCAL}"

log "installing MLLM training dependencies"
pip_install \
  "sentencepiece>=0.1.99" \
  "tiktoken>=0.7.0" \
  "transformers==4.56.2" \
  "tokenizers>=0.22.0,<0.23.0" \
  "huggingface-hub==0.36.2" \
  "accelerate==1.6.0" \
  "deepspeed==0.14.4" \
  "safetensors>=0.4.3" \
  packaging \
  "Pillow>=10.0.0" \
  torchvision==0.22.1 \
  shortuuid \
  "peft>=0.10.0" \
  pydantic \
  "markdown2[all]" \
  "numpy>=1.26,<2.0" \
  "scipy>=1.10" \
  "scikit-learn>=1.2" \
  requests \
  uvicorn \
  fastapi \
  "einops>=0.6" \
  "einops-exts>=0.0.4" \
  "timm>=0.9.0" \
  opencv-python-headless==4.11.0.86 \
  "loguru>=0.7.0" \
  "shapely>=2.0.0" \
  wandb \
  swanlab \
  urllib3==1.26.15 \
  tqdm \
  ninja

log "installing project in editable mode without dependency resolution"
cd "${ROOT_DIR}"
"${PIP}" install -e . --no-deps

cat > "${ACTIVATE_SCRIPT}" <<EOF
#!/usr/bin/env bash
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
export PYTHONPATH="${ROOT_DIR}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
EOF
chmod +x "${ACTIVATE_SCRIPT}"

log "running preflight"
"${PYTHON}" - <<'PY'
import json
import os
import sys

result = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
}
try:
    import torch
    result["torch"] = torch.__version__
except Exception as exc:
    result["torch_error"] = repr(exc)
try:
    import torch_npu
    result["torch_npu"] = getattr(torch_npu, "__version__", "<unknown>")
    result["npu_available"] = bool(torch.npu.is_available())
    result["npu_count"] = int(torch.npu.device_count())
except Exception as exc:
    result["torch_npu_error"] = repr(exc)
try:
    import transformers
    result["transformers"] = transformers.__version__
except Exception as exc:
    result["transformers_error"] = repr(exc)
try:
    import moxing
    result["moxing"] = getattr(moxing, "__version__", "<installed>")
except Exception as exc:
    result["moxing_error"] = repr(exc)
print(json.dumps(result, ensure_ascii=False, indent=2))
if result.get("python_version", "0") < "3.10":
    raise SystemExit("python version is too old")
if not result.get("npu_available", False):
    raise SystemExit("torch_npu installed, but NPU is not available")
PY

"${PYTHON}" -m py_compile \
  mllm/model/multimodal_encoder/dinov3_encoder.py \
  scripts/tools/build_oracle_intersection_centerline_dataset.py

log "environment ready: ${ENV_PREFIX}"
log "activate with: source ${ACTIVATE_SCRIPT}"
