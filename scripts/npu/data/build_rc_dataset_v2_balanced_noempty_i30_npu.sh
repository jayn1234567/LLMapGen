#!/usr/bin/env bash
set -euo pipefail

# Build the balanced RC Dataset V2 from all seven raw OBS sources on an Ascend host.
# This is a CPU/data-preparation job; it does not allocate or initialize an NPU.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/cache/jn/rc_dataset_v2_noempty_i30}
OUTPUT_OBS_ROOT=${OUTPUT_OBS_ROOT:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/rc_dataset_v2_noempty_i30/}
TRAIN_TARGET_SAMPLES=${TRAIN_TARGET_SAMPLES:-450000}
DIFFICULTY_RATIOS=${DIFFICULTY_RATIOS:-empty=0,easy=0.35,medium=0.30,hard=0.25,very_hard=0.10}
INTERSECTION_TARGET_RATIO=${INTERSECTION_TARGET_RATIO:-0.30}
VIEWS=${VIEWS:-both}
OBS_BACKEND=${OBS_BACKEND:-moxing}
OBSUTIL_PATH=${OBSUTIL_PATH:-}
INSTALL_DATA_DEPS=${INSTALL_DATA_DEPS:-false}
RESUME=${RESUME:-true}
KEEP_ARCHIVES=${KEEP_ARCHIVES:-true}
REMOVE_PACKAGE_AFTER_UPLOAD=${REMOVE_PACKAGE_AFTER_UPLOAD:-false}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-false}
SKIP_BUILD=${SKIP_BUILD:-false}
SKIP_UPLOAD=${SKIP_UPLOAD:-false}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "${WORK_ROOT}"

if is_true "${INSTALL_DATA_DEPS}"; then
  "${PYTHON_BIN}" -m pip install "setuptools<81"
  "${PYTHON_BIN}" -m pip install -r data_process/requirements.txt
fi

"${PYTHON_BIN}" - "${OBS_BACKEND}" <<'PY'
import sys

backend = sys.argv[1]
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required, found {sys.version}")

import geopandas  # noqa: F401
import numpy  # noqa: F401
import PIL  # noqa: F401
import pyproj  # noqa: F401
import rasterio  # noqa: F401
import shapely  # noqa: F401

if backend == "moxing":
    import moxing as mox
    if not hasattr(mox, "file"):
        raise SystemExit("Huawei moxing-framework with mox.file is required")

print(f"[dataset-v2-npu] python preflight passed: {sys.version.split()[0]}")
PY

echo "============================================================"
echo "[dataset-v2-npu] repository:          ${REPO_ROOT}"
echo "[dataset-v2-npu] work root:           ${WORK_ROOT}"
echo "[dataset-v2-npu] output OBS:          ${OUTPUT_OBS_ROOT}"
echo "[dataset-v2-npu] train records:       ${TRAIN_TARGET_SAMPLES}"
echo "[dataset-v2-npu] difficulty ratios:   ${DIFFICULTY_RATIOS}"
echo "[dataset-v2-npu] global intersection: ${INTERSECTION_TARGET_RATIO}"
echo "[dataset-v2-npu] views:               ${VIEWS}"
echo "[dataset-v2-npu] OBS backend:         ${OBS_BACKEND}"
echo "============================================================"
df -h "${WORK_ROOT}"

command=(
  "${PYTHON_BIN}" scripts/tools/build_rc_dataset_v2_from_obs.py
  --work-root "${WORK_ROOT}"
  --output-obs-root "${OUTPUT_OBS_ROOT}"
  --views "${VIEWS}"
  --train-target-samples "${TRAIN_TARGET_SAMPLES}"
  --difficulty-ratios "${DIFFICULTY_RATIOS}"
  --intersection-target-ratio "${INTERSECTION_TARGET_RATIO}"
  --obs-backend "${OBS_BACKEND}"
  --upload-mode tar
)

if [ -n "${OBSUTIL_PATH}" ]; then
  command+=(--obsutil-path "${OBSUTIL_PATH}")
fi
if is_true "${RESUME}"; then
  command+=(--resume)
fi
if is_true "${KEEP_ARCHIVES}"; then
  command+=(--keep-archives)
fi
if is_true "${REMOVE_PACKAGE_AFTER_UPLOAD}"; then
  command+=(--remove-package-after-upload)
fi
if is_true "${SKIP_DOWNLOAD}"; then
  command+=(--skip-download)
fi
if is_true "${SKIP_BUILD}"; then
  command+=(--skip-build)
fi
if is_true "${SKIP_UPLOAD}"; then
  command+=(--skip-upload)
fi

printf '[dataset-v2-npu] command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

echo "============================================================"
echo "[dataset-v2-npu] complete"
echo "[dataset-v2-npu] local output: ${WORK_ROOT}/output"
echo "[dataset-v2-npu] OBS output:   ${OUTPUT_OBS_ROOT}"
echo "============================================================"
