#!/usr/bin/env bash
set -euo pipefail

# Download a fresh original E2E dataset, reuse existing local512 per-patch
# predictions, and run only intersection formatting/merging/original metrics.
# Model inference is intentionally not part of this entry.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

E2E_ENV_DIR=${E2E_ENV_DIR:-/home/ma-user/.conda/envs/rc-e2e-original-py311}
CONDA_SH=${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}

PREDICTION_DIR=${PREDICTION_DIR:-/cache/jn/outputs/local512_550k_checkpoint34376_gt_empty_fresh_obs_e2e_20260805_103759/inference/json}
E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}

RUN_ID=${RUN_ID:-local512_550k_checkpoint34376_fresh_obs_intersection_e2e_$(date +%Y%m%d_%H%M%S)}
FRESH_RUN_ROOT=${FRESH_RUN_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/${RUN_ID}}
E2E_DATA_ARCHIVE=${E2E_DATA_ARCHIVE:-${FRESH_RUN_ROOT}/e2e_data.zip}
E2E_DATA_ROOT=${E2E_DATA_ROOT:-${FRESH_RUN_ROOT}/e2e_data}
RESULT_ROOT=${RESULT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/e2e_eval/original_pipeline_runs/${RUN_ID}}

EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
COLLAPSE_INTERSECTION_TYPE_TO_ONE=${COLLAPSE_INTERSECTION_TYPE_TO_ONE:-False}
EVAL_INTERSECTION_ONLY_TYPE1=${EVAL_INTERSECTION_ONLY_TYPE1:-False}
SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION=${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION:-True}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
INSTALL_ENGINE_DEPS=${INSTALL_ENGINE_DEPS:-False}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-False}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: conda activation script not found: ${CONDA_SH}" >&2
  exit 2
fi
if [ ! -d "${PREDICTION_DIR}" ]; then
  echo "ERROR: prediction directory not found: ${PREDICTION_DIR}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${E2E_ENV_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PREDICTION_COUNT=$(find "${PREDICTION_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [ "${PREDICTION_COUNT}" -eq 0 ]; then
  echo "ERROR: no per-patch prediction JSON found below ${PREDICTION_DIR}" >&2
  exit 2
fi

mkdir -p "${FRESH_RUN_ROOT}" "${RESULT_ROOT}" "${RUN_WORK_ROOT}"

echo "============================================================"
echo "FRESH OBS LOCAL512 INTERSECTION E2E"
echo "Predictions:       ${PREDICTION_DIR} (${PREDICTION_COUNT} JSON files)"
echo "E2E OBS:           ${E2E_DATA_OBS_PATH}"
echo "Fresh archive:     ${E2E_DATA_ARCHIVE}"
echo "Fresh E2E root:    ${E2E_DATA_ROOT}"
echo "Output:            ${RESULT_ROOT}"
echo "Run inference:     False"
echo "Numeric type map:  enabled"
echo "Collapse all type: ${COLLAPSE_INTERSECTION_TYPE_TO_ONE}"
echo "Evaluator Type=1:  ${EVAL_INTERSECTION_ONLY_TYPE1}"
echo "GT-empty oracle:   ${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION}"
echo "Visualization:     ${EVAL_VIS_FLAG}"
echo "============================================================"

echo "[fresh-intersection-e2e] stage 1/2: download and extract fresh E2E data"
python - "${E2E_DATA_OBS_PATH}" "${E2E_DATA_ARCHIVE}" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

import moxing as mox

source = sys.argv[1]
destination = Path(sys.argv[2])
temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary.unlink(missing_ok=True)
try:
    print(f"[fresh-intersection-e2e] download {source} -> {destination}", flush=True)
    mox.file.copy(source, str(temporary))
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded E2E archive is missing or empty: {temporary}")
    if not zipfile.is_zipfile(temporary):
        raise RuntimeError(f"Downloaded E2E archive is not a valid ZIP file: {temporary}")
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
print(f"[fresh-intersection-e2e] archive bytes={destination.stat().st_size}", flush=True)
PY

python scripts/tools/prepare_rc_e2e_original_run_data.py \
  --archive "${E2E_DATA_ARCHIVE}" \
  --destination "${E2E_DATA_ROOT}" \
  --allowed-root /cache/jn/e2e_eval/fresh_obs_runs \
  --reset

SCENE_COUNT=$(find "${E2E_DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | wc -l)
if [ "${SCENE_COUNT}" -ne "${EXPECTED_E2E_SCENES}" ]; then
  echo "ERROR: expected ${EXPECTED_E2E_SCENES} E2E scenes, found ${SCENE_COUNT} below ${E2E_DATA_ROOT}" >&2
  exit 2
fi

echo "[fresh-intersection-e2e] stage 2/2: format, merge, and run original intersection metrics"
PREDICTION_DIR="${PREDICTION_DIR}" \
E2E_DATA_ROOT="${E2E_DATA_ROOT}" \
RUN_ID="${RUN_ID}" \
RESULT_ROOT="${RESULT_ROOT}" \
RUN_WORK_ROOT="${RUN_WORK_ROOT}" \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
COLLAPSE_INTERSECTION_TYPE_TO_ONE="${COLLAPSE_INTERSECTION_TYPE_TO_ONE}" \
EVAL_INTERSECTION_ONLY_TYPE1="${EVAL_INTERSECTION_ONLY_TYPE1}" \
SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION="${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION}" \
E2E_USE_RAW_ROOT_DIRECTLY=True \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
INSTALL_ENGINE_DEPS="${INSTALL_ENGINE_DEPS}" \
UPLOAD_RESULTS="${UPLOAD_RESULTS}" \
RESULT_OBS_PATH="${RESULT_OBS_PATH}" \
bash "${SCRIPT_DIR}/eval_local512_predictions_original_intersection_e2e_npu.sh"

echo "============================================================"
echo "FRESH OBS INTERSECTION E2E COMPLETE"
echo "Metrics/output: ${RESULT_ROOT}/eval_result_all"
echo "Evaluator log:  ${RESULT_ROOT}/logs/03_eval_all.log"
echo "Fresh E2E data: ${E2E_DATA_ROOT}"
echo "============================================================"
