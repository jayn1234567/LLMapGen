#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}

E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
E2E_WORK_ROOT=${E2E_WORK_ROOT:-/cache/jn/e2e_eval}
E2E_ARCHIVE_PATH=${E2E_ARCHIVE_PATH:-${E2E_WORK_ROOT}/e2e_data.zip}
E2E_RAW_ROOT=${E2E_RAW_ROOT:-${E2E_WORK_ROOT}/raw_e2e_data}

PREDICTION_OBS_PATH=${PREDICTION_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/context512_roi256_checkpoint12504_e2e_data_full_v1}
EVAL_RUN_ID=${EVAL_RUN_ID:-context512_roi256_checkpoint12504_e2e_data_full_v1_wholemap_lane_metrics}
EVAL_ROOT=${EVAL_ROOT:-${E2E_WORK_ROOT}/metrics/${EVAL_RUN_ID}}
PATCH_METRICS_PREDICTION_DIR=${PATCH_METRICS_PREDICTION_DIR:-${E2E_WORK_ROOT}/metrics/context512_roi256_checkpoint12504_e2e_data_full_v1_patch_metrics/predictions}
PREDICTION_DIR=${PREDICTION_DIR:-${PATCH_METRICS_PREDICTION_DIR}}
METRICS_JSON=${METRICS_JSON:-${EVAL_ROOT}/wholemap_lane_metrics.json}
METRICS_OBS_PATH=${METRICS_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${EVAL_RUN_ID}}

MAX_SAMPLES=${MAX_SAMPLES:-0}
IGNORE_LANE_TYPES=${IGNORE_LANE_TYPES:-3,4,22}
LANE_BUFFER_SIZE=${LANE_BUFFER_SIZE:-2.5}
LANE_OVERLAP_THRESHOLD=${LANE_OVERLAP_THRESHOLD:-0.8}
DIRECTION_THRESHOLD_DEG=${DIRECTION_THRESHOLD_DEG:-10}
STITCH_DISTANCE=${STITCH_DISTANCE:-1.0}
STITCH_DIRECTION_THRESHOLD_DEG=${STITCH_DIRECTION_THRESHOLD_DEG:-20}
CUT_PREDICTED_INTERSECTIONS=${CUT_PREDICTED_INTERSECTIONS:-False}
REQUIRE_MASKS=${REQUIRE_MASKS:-False}
REQUIRE_ALL=${REQUIRE_ALL:-True}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-True}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

has_extracted_e2e_data() {
  [ -f "${E2E_RAW_ROOT}/.extract_complete" ] || \
    find "${E2E_RAW_ROOT}" -type d -name rc_one_patch_release -print -quit 2>/dev/null | grep -q .
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${E2E_WORK_ROOT}" "${EVAL_ROOT}" "${PREDICTION_DIR}"

if has_extracted_e2e_data; then
  echo "[e2e-wholemap-eval] reuse extracted E2E data: ${E2E_RAW_ROOT}"
else
  if [ ! -s "${E2E_ARCHIVE_PATH}" ]; then
    echo "[e2e-wholemap-eval] downloading data ${E2E_DATA_OBS_PATH} -> ${E2E_ARCHIVE_PATH}"
    python - "${E2E_DATA_OBS_PATH}" "${E2E_ARCHIVE_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
  else
    echo "[e2e-wholemap-eval] reuse data archive: ${E2E_ARCHIVE_PATH}"
  fi

  echo "[e2e-wholemap-eval] extracting ${E2E_ARCHIVE_PATH} -> ${E2E_RAW_ROOT}"
  mkdir -p "${E2E_RAW_ROOT}"
  python - "${E2E_ARCHIVE_PATH}" "${E2E_RAW_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path
archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with zipfile.ZipFile(archive) as handle:
    handle.extractall(destination)
(destination / ".extract_complete").write_text("ok\n", encoding="utf-8")
PY
fi

if ! find "${PREDICTION_DIR}" -type f -name '*.json' -print -quit | grep -q .; then
  echo "[e2e-wholemap-eval] downloading predictions ${PREDICTION_OBS_PATH} -> ${PREDICTION_DIR}"
  python - "${PREDICTION_OBS_PATH}" "${PREDICTION_DIR}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
else
  echo "[e2e-wholemap-eval] reuse predictions: ${PREDICTION_DIR}"
fi

python scripts/tools/validate_rc_e2e_raster_alignment.py \
  --input-root "${E2E_RAW_ROOT}" \
  --patch-size 256 \
  --output-json "${EVAL_ROOT}/raster_alignment_report.json"

REQUIRE_ALL_FLAG=--require-all
if ! is_true "${REQUIRE_ALL}"; then
  REQUIRE_ALL_FLAG=--no-require-all
fi
REQUIRE_MASKS_FLAG=--require-masks
if ! is_true "${REQUIRE_MASKS}"; then
  REQUIRE_MASKS_FLAG=--no-require-masks
fi
CUT_INTERSECTIONS_FLAG=--cut-predicted-intersections
if ! is_true "${CUT_PREDICTED_INTERSECTIONS}"; then
  CUT_INTERSECTIONS_FLAG=--no-cut-predicted-intersections
fi

echo "============================================================"
echo "RC E2E OFFICIAL-STYLE WHOLE-MAP LANE EVALUATION"
echo "Raw data:         ${E2E_RAW_ROOT}"
echo "Predictions:      ${PREDICTION_DIR}"
echo "Buffer:           ${LANE_BUFFER_SIZE} m"
echo "Overlap threshold:${LANE_OVERLAP_THRESHOLD}"
echo "Direction limit:  ${DIRECTION_THRESHOLD_DEG} deg"
echo "Stitch distance:  ${STITCH_DISTANCE} m"
echo "Output:           ${METRICS_JSON}"
echo "============================================================"

python scripts/tools/evaluate_rc_e2e_wholemap_lane_metrics.py \
  --raw-e2e-root "${E2E_RAW_ROOT}" \
  --prediction-dir "${PREDICTION_DIR}" \
  --output-json "${METRICS_JSON}" \
  --baseline-name gt \
  --gt-crs EPSG:4326 \
  --patch-size 256 \
  --coord-range 1000 \
  --ignore-lane-types "${IGNORE_LANE_TYPES}" \
  --lane-buffer-size "${LANE_BUFFER_SIZE}" \
  --lane-overlap-threshold "${LANE_OVERLAP_THRESHOLD}" \
  --direction-threshold-deg "${DIRECTION_THRESHOLD_DEG}" \
  --stitch-distance "${STITCH_DISTANCE}" \
  --stitch-direction-threshold-deg "${STITCH_DIRECTION_THRESHOLD_DEG}" \
  --max-samples "${MAX_SAMPLES}" \
  "${CUT_INTERSECTIONS_FLAG}" \
  "${REQUIRE_MASKS_FLAG}" \
  "${REQUIRE_ALL_FLAG}"

if is_true "${UPLOAD_RESULTS}" && [ -n "${METRICS_OBS_PATH}" ]; then
  echo "[e2e-wholemap-eval] uploading ${EVAL_ROOT} -> ${METRICS_OBS_PATH}"
  python - "${EVAL_ROOT}" "${METRICS_OBS_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
fi

echo "============================================================"
echo "RC E2E WHOLE-MAP LANE METRICS COMPLETE"
echo "Metrics:     ${METRICS_JSON}"
echo "Metrics OBS: ${METRICS_OBS_PATH}"
echo "============================================================"
