#!/usr/bin/env bash
set -euo pipefail

# Reuse local512 patch predictions, build the dedicated RC inter512 artifacts,
# and run the untouched original evaluator once with both high/low roads enabled.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

E2E_ENV_DIR=${E2E_ENV_DIR:-/home/ma-user/.conda/envs/rc-e2e-original-py311}
CONDA_SH=${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}

PREDICTION_DIR=${PREDICTION_DIR:-/cache/jn/outputs/local512_550k_checkpoint34376_gt_empty_fresh_obs_e2e_20260805_103759/inference/json}
E2E_DATA_ROOT=${E2E_DATA_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/local512_550k_checkpoint34376_e2e_native512_20260805_150253/e2e_data}
QUERY_NAME=${QUERY_NAME:-output_llm_intersection_jn}
INTER_RESULT_SUBDIR=${INTER_RESULT_SUBDIR:-inter512/tif_512_256}

RUN_ID=${RUN_ID:-local512_original_intersection_e2e_$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/e2e_eval/original_pipeline_runs/${RUN_ID}}
FORMAT_REPORT=${FORMAT_REPORT:-${RESULT_ROOT}/intersection_format_report.json}
GEOJSON_REPORT=${GEOJSON_REPORT:-${RESULT_ROOT}/intersection_geojson_report.json}
GT_ORACLE_REPORT=${GT_ORACLE_REPORT:-${RESULT_ROOT}/intersection_gt_empty_suppression_report.json}

WINDOW_SIZE=${WINDOW_SIZE:-512}
INTERSECTION_STRIDE=${INTERSECTION_STRIDE:-512}
ORIGINAL_E2E_LANE_GRID_SIZE=${ORIGINAL_E2E_LANE_GRID_SIZE:-${WINDOW_SIZE}}
PREDICTION_COORD_SCALE=${PREDICTION_COORD_SCALE:-0.${WINDOW_SIZE}}
ENGINE_EXTRACT_ROOT=${ENGINE_EXTRACT_ROOT:-${RUN_WORK_ROOT}/original_engine_grid${ORIGINAL_E2E_LANE_GRID_SIZE}}
COORD_RANGE=${COORD_RANGE:-1000}
MERGE_BUFFER_METERS=${MERGE_BUFFER_METERS:-0.5}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-True}
INSTALL_ENGINE_DEPS=${INSTALL_ENGINE_DEPS:-False}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-False}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}
COLLAPSE_INTERSECTION_TYPE_TO_ONE=${COLLAPSE_INTERSECTION_TYPE_TO_ONE:-False}
SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION=${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION:-False}
E2E_USE_RAW_ROOT_DIRECTLY=${E2E_USE_RAW_ROOT_DIRECTLY:-False}
EVAL_INTERSECTION_ONLY_TYPE1=${EVAL_INTERSECTION_ONLY_TYPE1:-True}

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: conda activation script not found: ${CONDA_SH}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${E2E_ENV_DIR}"
python -c "import rasterio, shapely, pyproj"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ ! -d "${PREDICTION_DIR}" ]; then
  echo "ERROR: prediction directory not found: ${PREDICTION_DIR}" >&2
  exit 2
fi
if [ ! -d "${E2E_DATA_ROOT}" ]; then
  echo "ERROR: extracted E2E root not found: ${E2E_DATA_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}" "${RUN_WORK_ROOT}"

echo "============================================================"
echo "LOCAL512 PREDICTIONS -> ORIGINAL INTERSECTION E2E"
echo "Predictions:      ${PREDICTION_DIR}"
echo "E2E data:         ${E2E_DATA_ROOT}"
echo "Window/stride:    ${WINDOW_SIZE}/${INTERSECTION_STRIDE}"
echo "Evaluator grid:   ${ORIGINAL_E2E_LANE_GRID_SIZE} (scale=${PREDICTION_COORD_SCALE})"
echo "Patch result dir: ${INTER_RESULT_SUBDIR}"
echo "Query name:       ${QUERY_NAME}"
echo "Evaluation:       one all-roads pass (high=True, low=True)"
echo "Output:           ${RESULT_ROOT}"
echo "Collapse type:    ${COLLAPSE_INTERSECTION_TYPE_TO_ONE}"
echo "GT-empty oracle:  ${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION}"
echo "============================================================"

echo "[intersection-e2e] stage 1/3: format patch intersection predictions"
FORMAT_TYPE_ARGS=()
case "${COLLAPSE_INTERSECTION_TYPE_TO_ONE}" in
  1|true|TRUE|True|yes|YES|on|ON) FORMAT_TYPE_ARGS+=(--collapse-type-to-one) ;;
esac
python scripts/tools/format_rc_e2e_intersection_predictions.py \
  --prediction-dir "${PREDICTION_DIR}" \
  --e2e-root "${E2E_DATA_ROOT}" \
  --report-json "${FORMAT_REPORT}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${INTERSECTION_STRIDE}" \
  --coord-range "${COORD_RANGE}" \
  --result-subdir "${INTER_RESULT_SUBDIR}" \
  --reset \
  --strict \
  "${FORMAT_TYPE_ARGS[@]}"

python - "${FORMAT_REPORT}" "${COLLAPSE_INTERSECTION_TYPE_TO_ONE}" "${EVAL_INTERSECTION_ONLY_TYPE1}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
collapse = sys.argv[2].strip().lower() in {"1", "true", "yes", "on"}
only_type1 = sys.argv[3].strip().lower() in {"1", "true", "yes", "on"}
report = json.loads(report_path.read_text(encoding="utf-8"))
counts = report.get("counts") or {}
formatted = int(counts.get("formatted_intersections", 0))
visible = int(counts.get("label:1_1", 0)) + int(counts.get("label:1_2", 0))
if formatted > 0 and visible == 0 and not collapse and only_type1:
    raise RuntimeError(
        "All formatted intersections map to non-type-1 labels, so the original evaluator "
        "would report pred_num=0. This checkpoint has no evaluator-visible common/T labels. "
        "For a geometry-only metric, rerun with COLLAPSE_INTERSECTION_TYPE_TO_ONE=True. "
        f"Inspect {report_path} for missing/explicit intersection types."
    )
print(
    f"[intersection-e2e] evaluator-visible type-1 predictions={visible}/{formatted} "
    f"collapse_type_to_one={collapse} evaluator_only_type1={only_type1}",
    flush=True,
)
PY

case "${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION}" in
  1|true|TRUE|True|yes|YES|on|ON)
    echo "[intersection-e2e] diagnostic: suppress predictions in GT-empty intersection patches"
    python scripts/tools/suppress_rc_e2e_intersections_without_patch_gt.py \
      --e2e-root "${E2E_DATA_ROOT}" \
      --report-json "${GT_ORACLE_REPORT}" \
      --result-subdir "${INTER_RESULT_SUBDIR}" \
      --window-size "${WINDOW_SIZE}" \
      --stride "${INTERSECTION_STRIDE}" \
      --gt-intersection-type 1 \
      --expected-scenes "${EXPECTED_E2E_SCENES}"
    ;;
esac

echo "[intersection-e2e] stage 2/3: merge patch polygons into per-scene GeoJSON"
python scripts/tools/build_rc_e2e_intersection_geojson.py \
  --e2e-root "${E2E_DATA_ROOT}" \
  --report-json "${GEOJSON_REPORT}" \
  --stride "${INTERSECTION_STRIDE}" \
  --merge-buffer-meters "${MERGE_BUFFER_METERS}" \
  --result-subdir "${INTER_RESULT_SUBDIR}" \
  --query-name "${QUERY_NAME}" \
  --expected-scenes "${EXPECTED_E2E_SCENES}" \
  --reset-query

echo "[intersection-e2e] stage 3/3: original RC E2E intersection metrics"
E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${E2E_DATA_ROOT}" \
E2E_USE_RAW_ROOT_DIRECTLY="${E2E_USE_RAW_ROOT_DIRECTLY}" \
RUN_ID="${RUN_ID}" \
RUN_WORK_ROOT="${RUN_WORK_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}" \
RESULT_OBS_PATH="${RESULT_OBS_PATH}" \
PREDICTION_CACHE="${PREDICTION_DIR}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
ENGINE_EXTRACT_ROOT="${ENGINE_EXTRACT_ROOT}" \
INSTALL_ENGINE_DEPS="${INSTALL_ENGINE_DEPS}" \
PREDICTION_COORD_SCALE="${PREDICTION_COORD_SCALE}" \
ORIGINAL_E2E_LANE_GRID_SIZE="${ORIGINAL_E2E_LANE_GRID_SIZE}" \
RUN_FORMAT_STEP=False \
RUN_RULE_STEP=False \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=False \
RUN_HIGH_EVAL=False \
EVAL_QUERY_NAME="${QUERY_NAME}" \
EVAL_INTERSECTION_ONLY_TYPE1="${EVAL_INTERSECTION_ONLY_TYPE1}" \
EVAL_SIMPLIFY_PATH=False \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
FILL_MISSING_SCENE_PREDICTIONS=False \
RESET_EXISTING_MODEL_OUTPUTS=False \
FAIL_ON_INVALID_PREDICTIONS=False \
UPLOAD_RESULTS="${UPLOAD_RESULTS}" \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "============================================================"
echo "ORIGINAL INTERSECTION E2E COMPLETE"
echo "Metrics/output:   ${RESULT_ROOT}/eval_result_all"
echo "Evaluator log:    ${RESULT_ROOT}/logs/03_eval_all.log"
echo "Format report:    ${FORMAT_REPORT}"
echo "GeoJSON report:   ${GEOJSON_REPORT}"
if [ -f "${GT_ORACLE_REPORT}" ]; then
  echo "GT oracle report: ${GT_ORACLE_REPORT}"
fi
echo "============================================================"
