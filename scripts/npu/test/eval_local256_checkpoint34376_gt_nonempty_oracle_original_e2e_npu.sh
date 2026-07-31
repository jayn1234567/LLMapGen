#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT:-/cache/jn/outputs/rc_e2e_original_crop256_local256_550k_checkpoint34376_20260728_095546}
SOURCE_PREDICTION_DIR=${SOURCE_PREDICTION_DIR:-${SOURCE_RUN_ROOT}/inference/json}
RAW_E2E_ROOT=${RAW_E2E_ROOT:-/cache/e2e_data}
RUN_ID=${RUN_ID:-local256_checkpoint34376_gt_nonempty_oracle_$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
PATCH_REFERENCE_ROOT=${PATCH_REFERENCE_ROOT:-${OUTPUT_ROOT}/patch_reference}
PATCH_EVAL_JSONL=${PATCH_EVAL_JSONL:-${PATCH_REFERENCE_ROOT}/eval_records.jsonl}
ORACLE_PREDICTION_DIR=${ORACLE_PREDICTION_DIR:-${OUTPUT_ROOT}/oracle_predictions}
ORACLE_REPORT=${ORACLE_REPORT:-${OUTPUT_ROOT}/gt_oracle_suppression_report.json}
ORACLE_PATCH_ROOT=${ORACLE_PATCH_ROOT:-${OUTPUT_ROOT}/oracle_patch_metrics}
PATCH_COMPARISON_JSON=${PATCH_COMPARISON_JSON:-${OUTPUT_ROOT}/patch_metric_comparison.json}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/e2e_eval/original_pipeline_runs/${RUN_ID}}
ORIGINAL_RESULT_ROOT=${ORIGINAL_RESULT_ROOT:-${OUTPUT_ROOT}/original_pipeline_metrics}
RULE_WORKERS=${RULE_WORKERS:-16}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
RUN_ORIGINAL_E2E=${RUN_ORIGINAL_E2E:-False}
RUN_ALL_EVAL=${RUN_ALL_EVAL:-True}
RUN_LOW_EVAL=${RUN_LOW_EVAL:-True}
RUN_HIGH_EVAL=${RUN_HIGH_EVAL:-True}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cd "${REPO_ROOT}"

if [ ! -d "${SOURCE_PREDICTION_DIR}" ]; then
  echo "ERROR: source prediction directory not found: ${SOURCE_PREDICTION_DIR}" >&2
  exit 2
fi
if [ ! -d "${RAW_E2E_ROOT}" ]; then
  echo "ERROR: clean E2E data root not found: ${RAW_E2E_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${PATCH_REFERENCE_ROOT}"

if [ ! -s "${PATCH_EVAL_JSONL}" ] || [ ! -s "${PATCH_REFERENCE_ROOT}/metrics.json" ]; then
  echo "[gt-oracle-e2e] stage 1/4: build original patch metrics and GT reference"
  E2E_DATA_SOURCE=raw_direct \
  E2E_RAW_ROOT="${RAW_E2E_ROOT}" \
  EVAL_RUN_ID="${RUN_ID}_patch_reference" \
  EVAL_ROOT="${PATCH_REFERENCE_ROOT}" \
  PREDICTION_DIR="${SOURCE_PREDICTION_DIR}" \
  METRICS_JSON="${PATCH_REFERENCE_ROOT}/metrics.json" \
  EVAL_JSONL="${PATCH_EVAL_JSONL}" \
  METRICS_OBS_PATH="" \
  REQUIRE_ALL=True \
  bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"
else
  echo "[gt-oracle-e2e] reuse patch reference: ${PATCH_EVAL_JSONL}"
fi

echo "[gt-oracle-e2e] stage 2/4: suppress predictions in GT-empty patches"
python scripts/tools/suppress_e2e_predictions_without_patch_gt.py \
  --eval-jsonl "${PATCH_EVAL_JSONL}" \
  --prediction-dir "${SOURCE_PREDICTION_DIR}" \
  --output-dir "${ORACLE_PREDICTION_DIR}" \
  --report-json "${ORACLE_REPORT}" \
  --reset \
  --strict

echo "[gt-oracle-e2e] stage 3/4: evaluate GT-oracle patch metrics"
E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${RAW_E2E_ROOT}" \
EVAL_RUN_ID="${RUN_ID}_oracle_patch" \
EVAL_ROOT="${ORACLE_PATCH_ROOT}" \
PREDICTION_DIR="${ORACLE_PREDICTION_DIR}" \
METRICS_JSON="${ORACLE_PATCH_ROOT}/metrics.json" \
EVAL_JSONL="${ORACLE_PATCH_ROOT}/eval_records.jsonl" \
METRICS_OBS_PATH="" \
REQUIRE_ALL=True \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"

python - \
  "${PATCH_REFERENCE_ROOT}/metrics.json" \
  "${ORACLE_PATCH_ROOT}/metrics.json" \
  "${PATCH_COMPARISON_JSON}" <<'PY'
import json
import sys
from pathlib import Path

original_path = Path(sys.argv[1])
oracle_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

def select(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = payload["centerline_eval"]
    return {
        "metrics_json": str(path),
        "evaluated_records": payload["coverage"]["evaluated_records"],
        "instance_precision": metric["instance_pre"],
        "instance_recall": metric["instance_recall"],
        "instance_f1": metric["instance_f1"],
        "length_precision": metric["length_pre"],
        "length_recall": metric["length_recall"],
        "length_f1": metric["length_f1"],
        "raw_totals": metric.get("raw_totals", {}),
    }

original = select(original_path)
oracle = select(oracle_path)
metric_keys = (
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "length_precision",
    "length_recall",
    "length_f1",
)
summary = {
    "warning": "GT-oracle diagnostic; ground truth is used to suppress predictions in empty patches.",
    "original": original,
    "gt_empty_suppressed": oracle,
    "gt_empty_suppressed_minus_original": {
        key: oracle[key] - original[key] for key in metric_keys
    },
}
output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

if ! is_true "${RUN_ORIGINAL_E2E}"; then
  echo "============================================================"
  echo "GT-NONEMPTY ORACLE PATCH EVALUATION COMPLETE"
  echo "Suppression report: ${ORACLE_REPORT}"
  echo "Patch comparison:   ${PATCH_COMPARISON_JSON}"
  echo "Oracle metrics:     ${ORACLE_PATCH_ROOT}/metrics.json"
  echo "Original E2E skipped: RUN_ORIGINAL_E2E=${RUN_ORIGINAL_E2E}"
  echo "============================================================"
  exit 0
fi

echo "[gt-oracle-e2e] stage 4/4: original all/low/high road evaluation pipeline"
E2E_DATA_SOURCE=raw_copy \
E2E_RAW_ROOT="${RAW_E2E_ROOT}" \
E2E_DATA_ROOT="${RUN_WORK_ROOT}/e2e_data" \
E2E_PREPARE_MODE=hardlink \
RESET_PREPARED_E2E_DATA=False \
RUN_ID="${RUN_ID}_original_pipeline" \
RUN_WORK_ROOT="${RUN_WORK_ROOT}" \
RESULT_ROOT="${ORIGINAL_RESULT_ROOT}" \
RESULT_OBS_PATH="" \
PREDICTION_CACHE="${ORACLE_PREDICTION_DIR}" \
REUSE_PREDICTIONS=True \
REUSE_ENGINE_ARCHIVE=True \
INSTALL_ENGINE_DEPS=False \
RULE_WORKERS="${RULE_WORKERS}" \
RUN_FORMAT_STEP=True \
RUN_RULE_STEP=True \
RUN_ALL_EVAL="${RUN_ALL_EVAL}" \
RUN_LOW_EVAL="${RUN_LOW_EVAL}" \
RUN_HIGH_EVAL="${RUN_HIGH_EVAL}" \
EVAL_SIMPLIFY_PATH=False \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
UPLOAD_RESULTS=False \
bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "============================================================"
echo "GT-NONEMPTY ORACLE E2E COMPLETE"
echo "Suppression report: ${ORACLE_REPORT}"
echo "Patch comparison:   ${PATCH_COMPARISON_JSON}"
echo "Oracle predictions: ${ORACLE_PREDICTION_DIR}"
echo "Original all:       ${ORIGINAL_RESULT_ROOT}/eval_result_all"
echo "Original low:       ${ORIGINAL_RESULT_ROOT}/eval_result_low"
echo "Original high:      ${ORIGINAL_RESULT_ROOT}/eval_result_high"
echo "Run-local E2E data: ${RUN_WORK_ROOT}/e2e_data"
echo "WARNING: diagnostic oracle metric; do not report as production performance."
echo "============================================================"
