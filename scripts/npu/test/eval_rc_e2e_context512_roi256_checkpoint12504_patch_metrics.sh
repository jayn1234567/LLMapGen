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
EVAL_RUN_ID=${EVAL_RUN_ID:-context512_roi256_checkpoint12504_e2e_data_full_v1_patch_metrics}
EVAL_ROOT=${EVAL_ROOT:-${E2E_WORK_ROOT}/metrics/${EVAL_RUN_ID}}
PREDICTION_DIR=${PREDICTION_DIR:-${EVAL_ROOT}/predictions}
METRICS_JSON=${METRICS_JSON:-${EVAL_ROOT}/metrics.json}
EVAL_JSONL=${EVAL_JSONL:-${EVAL_ROOT}/eval_records.jsonl}
METRICS_OBS_PATH=${METRICS_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${EVAL_RUN_ID}}

MAX_SAMPLES=${MAX_SAMPLES:-0}
METER_PER_PIXEL=${METER_PER_PIXEL:-0.2}
BUFFER_SIZE=${BUFFER_SIZE:-1.0}
MATCH_THRESHOLD=${MATCH_THRESHOLD:-0.33}
IGNORE_LANE_TYPES=${IGNORE_LANE_TYPES:-3,4,22}
REQUIRE_ALL=${REQUIRE_ALL:-True}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
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

if [ ! -s "${E2E_ARCHIVE_PATH}" ]; then
  echo "[e2e-patch-eval] downloading data ${E2E_DATA_OBS_PATH} -> ${E2E_ARCHIVE_PATH}"
  python - "${E2E_DATA_OBS_PATH}" "${E2E_ARCHIVE_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
else
  echo "[e2e-patch-eval] reuse data archive: ${E2E_ARCHIVE_PATH}"
fi

if [ ! -f "${E2E_RAW_ROOT}/.extract_complete" ]; then
  echo "[e2e-patch-eval] extracting ${E2E_ARCHIVE_PATH} -> ${E2E_RAW_ROOT}"
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
else
  echo "[e2e-patch-eval] reuse extracted E2E data: ${E2E_RAW_ROOT}"
fi

if ! find "${PREDICTION_DIR}" -type f -name '*.json' -print -quit | grep -q .; then
  echo "[e2e-patch-eval] downloading predictions ${PREDICTION_OBS_PATH} -> ${PREDICTION_DIR}"
  python - "${PREDICTION_OBS_PATH}" "${PREDICTION_DIR}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
else
  echo "[e2e-patch-eval] reuse predictions: ${PREDICTION_DIR}"
fi

python scripts/tools/validate_rc_e2e_raster_alignment.py \
  --input-root "${E2E_RAW_ROOT}" \
  --patch-size 256 \
  --output-json "${EVAL_ROOT}/raster_alignment_report.json"

REQUIRE_FLAG=--require-all
if ! is_true "${REQUIRE_ALL}"; then
  REQUIRE_FLAG=--no-require-all
fi

python scripts/tools/evaluate_rc_e2e_patch_metrics.py \
  --raw-e2e-root "${E2E_RAW_ROOT}" \
  --prediction-dir "${PREDICTION_DIR}" \
  --output-json "${METRICS_JSON}" \
  --output-eval-jsonl "${EVAL_JSONL}" \
  --baseline-name gt \
  --gt-crs EPSG:4326 \
  --patch-size 256 \
  --coord-range 1000 \
  --meter-per-pixel "${METER_PER_PIXEL}" \
  --buffer-size "${BUFFER_SIZE}" \
  --match-threshold "${MATCH_THRESHOLD}" \
  --ignore-lane-types "${IGNORE_LANE_TYPES}" \
  --max-samples "${MAX_SAMPLES}" \
  "${REQUIRE_FLAG}"

if [ -n "${METRICS_OBS_PATH}" ]; then
  echo "[e2e-patch-eval] uploading ${EVAL_ROOT} -> ${METRICS_OBS_PATH}"
  python - "${EVAL_ROOT}" "${METRICS_OBS_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
fi

echo "============================================================"
echo "RC E2E PATCH METRICS COMPLETE"
echo "Metrics:      ${METRICS_JSON}"
echo "Eval JSONL:   ${EVAL_JSONL}"
echo "Metrics OBS:  ${METRICS_OBS_PATH}"
echo "============================================================"
