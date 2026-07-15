#!/usr/bin/env bash
set -euo pipefail

# CPU-only environment for RC Dataset V2 generation on an Ascend/DI host.
# This environment deliberately does not install torch or torch_npu.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/rc-dataset-v2-py311}"
PYTHON_VERSION="${RC_DATASET_PYTHON_VERSION:-3.11}"
CLONE_FROM="${CLONE_FROM:-/home/ma-user/.conda/envs/mllm-npu-py311}"
HOST_PYTHON="${HOST_PYTHON:-}"
RUN_BUILD="${RUN_BUILD:-false}"

MOXING_WHL_OBS_PATH="${MOXING_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}"
MOXING_WHL_LOCAL_PATH="${MOXING_WHL_LOCAL_PATH:-/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl}"

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]]
}

python_is_requested_version() {
  "$1" - "${PYTHON_VERSION}" <<'PY' >/dev/null 2>&1
import sys

requested = tuple(int(part) for part in sys.argv[1].split(".")[:2])
raise SystemExit(0 if sys.version_info[:2] == requested else 1)
PY
}

python_has_moxing() {
  USE_MEMARTS=0 "$1" - <<'PY' >/dev/null 2>&1
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
    "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python3.9"
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
  return 1
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
if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi

ENV_KIND="existing"
if [ ! -x "${ENV_DIR}/bin/python" ]; then
  if [ -z "${CONDA_SH}" ] || [ ! -f "${CONDA_SH}" ]; then
    echo "[dataset-v2-env] conda is required to clone the existing Python environment." >&2
    exit 2
  fi
  if [ ! -x "${CLONE_FROM}/bin/python" ]; then
    echo "[dataset-v2-env] CLONE_FROM has no Python: ${CLONE_FROM}" >&2
    exit 2
  fi
  echo "[dataset-v2-env] cloning existing environment"
  echo "[dataset-v2-env] source: ${CLONE_FROM}"
  echo "[dataset-v2-env] target: ${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" --clone "${CLONE_FROM}"
  ENV_KIND="conda-clone"
else
  echo "[dataset-v2-env] reusing existing environment: ${ENV_DIR}"
fi

ENV_PYTHON="${ENV_DIR}/bin/python"
if ! python_is_requested_version "${ENV_PYTHON}"; then
  echo "[dataset-v2-env] expected Python ${PYTHON_VERSION}, found:" >&2
  "${ENV_PYTHON}" --version >&2
  exit 2
fi

echo "[dataset-v2-env] installing CPU/geospatial dependencies"
export PYTHONNOUSERSITE=1
"${ENV_PYTHON}" -m pip install "setuptools<81" wheel
"${ENV_PYTHON}" -m pip install -r "${REPO_ROOT}/data_process/requirements.txt"

if [ ! -f "${MOXING_WHL_LOCAL_PATH}" ]; then
  RESOLVED_HOST_PYTHON="$(resolve_host_python_with_moxing || true)"
  if [ -z "${RESOLVED_HOST_PYTHON}" ]; then
    echo "[dataset-v2-env] no host Python can download the Huawei MoXing wheel." >&2
    echo "[dataset-v2-env] Set HOST_PYTHON to a Python with moxing.file, or place the wheel at:" >&2
    echo "  ${MOXING_WHL_LOCAL_PATH}" >&2
    exit 2
  fi
  echo "[dataset-v2-env] downloading Huawei MoXing wheel with ${RESOLVED_HOST_PYTHON}"
  USE_MEMARTS=0 "${RESOLVED_HOST_PYTHON}" - \
    "${MOXING_WHL_OBS_PATH}" "${MOXING_WHL_LOCAL_PATH}" <<'PY'
import sys
from pathlib import Path

import moxing as mox

source, target = sys.argv[1:]
Path(target).parent.mkdir(parents=True, exist_ok=True)
print(f"[dataset-v2-env] download {source} -> {target}", flush=True)
mox.file.copy(source, target)
PY
fi

"${ENV_PYTHON}" -m pip uninstall -y moxing moxing-framework >/dev/null 2>&1 || true
"${ENV_PYTHON}" -m pip install "${MOXING_WHL_LOCAL_PATH}"
# Keep NumPy and setuptools on versions compatible with the data pipeline and MoXing.
"${ENV_PYTHON}" -m pip install "numpy>=1.26,<2.0" "setuptools<81"

ACTIVATE_SCRIPT="${ENV_DIR}/activate_rc_dataset_v2.sh"
cat > "${ACTIVATE_SCRIPT}" <<EOF
#!/usr/bin/env bash
if [ -f "${ENV_DIR}/conda-meta/history" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${ENV_DIR}"
else
  source "${ENV_DIR}/bin/activate"
fi
export PYTHON="${ENV_DIR}/bin/python"
export PYTHON_BIN="${ENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"
EOF
chmod +x "${ACTIVATE_SCRIPT}"

"${ENV_PYTHON}" - <<'PY'
import json
import sys

import geopandas
import moxing
import numpy
import PIL
import pyproj
import rasterio
import shapely

try:
    import pyogrio
    vector_io = f"pyogrio {pyogrio.__version__}"
except ImportError:
    import fiona
    vector_io = f"fiona {fiona.__version__}"

result = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
    "geopandas": geopandas.__version__,
    "rasterio": rasterio.__version__,
    "shapely": shapely.__version__,
    "pyproj": pyproj.__version__,
    "vector_io": vector_io,
    "moxing_file_api": hasattr(moxing, "file"),
}
print(json.dumps(result, indent=2, ensure_ascii=True))
if int(numpy.__version__.split(".")[0]) >= 2:
    raise SystemExit("Dataset V2 requires NumPy < 2.0.")
if not result["moxing_file_api"]:
    raise SystemExit("Huawei moxing-framework was not installed correctly.")
PY

echo "============================================================"
echo "[dataset-v2-env] environment kind: ${ENV_KIND}"
echo "[dataset-v2-env] cloned from: ${CLONE_FROM}"
echo "[dataset-v2-env] environment ready: ${ENV_DIR}"
echo "[dataset-v2-env] activate with: source ${ACTIVATE_SCRIPT}"
echo "[dataset-v2-env] build with: bash ${REPO_ROOT}/scripts/npu/data/build_rc_dataset_v2_balanced_noempty_i30_npu.sh"
echo "============================================================"

if bool_enabled "${RUN_BUILD}"; then
  echo "[dataset-v2-env] RUN_BUILD=${RUN_BUILD}; starting Dataset V2 build"
  PYTHON_BIN="${ENV_PYTHON}" \
    bash "${REPO_ROOT}/scripts/npu/data/build_rc_dataset_v2_balanced_noempty_i30_npu.sh"
fi
