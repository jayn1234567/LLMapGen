#!/usr/bin/env bash
set -euo pipefail

# Compare <0.98 and <1.0 patch sets with the same predictions and patch GT.
# Whole-map A/B remains available but is disabled by default because it reruns
# the original formatter and rule engine twice.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

BASE_RUN_ID=${BASE_RUN_ID:-context512_roi256_checkpoint12504_fresh_20260730_122847}
DATASET_ROOT=${DATASET_ROOT:-/cache/jn/e2e_eval/e2e_data_context512_roi256_black1_20260729_114312}
MANIFEST_JSON=${MANIFEST_JSON:-${DATASET_ROOT}/patch_manifest.json}
INFER_JSONL=${INFER_JSONL:-${DATASET_ROOT}/infer.jsonl}
PREDICTION_DIR=${PREDICTION_DIR:-/cache/jn/outputs/${BASE_RUN_ID}/inference/json}
PREDICTION_PARENT=$(dirname "${PREDICTION_DIR}")
SANITIZE_REPORT=${SANITIZE_REPORT:-${PREDICTION_PARENT}/prediction_roi_clip_report.json}
RAW_E2E_ROOT=${RAW_E2E_ROOT:-/cache/jn/e2e_eval/raw_e2e_data}
MAX_OLD_BLACK_RATIO=${MAX_OLD_BLACK_RATIO:-0.98}
DIAG_RUN_ID=${DIAG_RUN_ID:-context512_black_threshold_ab_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${DIAG_RUN_ID}}
FILTERED_PREDICTION_DIR=${FILTERED_PREDICTION_DIR:-${OUTPUT_ROOT}/predictions_black_lt_098}
ADDED_PREDICTION_DIR=${ADDED_PREDICTION_DIR:-${OUTPUT_ROOT}/predictions_black_ge_098_lt_100}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_ROOT}}
VISUALIZE_PATCH_AB=${VISUALIZE_PATCH_AB:-True}
VIS_LIMIT=${VIS_LIMIT:-200}
RUN_WHOLEMAP_AB=${RUN_WHOLEMAP_AB:-False}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cd "${REPO_ROOT}"

for required_path in "${MANIFEST_JSON}" "${INFER_JSONL}"; do
  if [ ! -s "${required_path}" ]; then
    echo "ERROR: required file not found or empty: ${required_path}" >&2
    exit 2
  fi
done
if [ ! -d "${PREDICTION_DIR}" ]; then
  echo "ERROR: prediction directory not found: ${PREDICTION_DIR}" >&2
  exit 2
fi
if [ ! -d "${RAW_E2E_ROOT}" ]; then
  echo "ERROR: raw E2E root not found: ${RAW_E2E_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[threshold-ab] verifying the complete <1.0 prediction set"
python scripts/tools/verify_e2e_inference_completeness.py \
  --infer-jsonl "${INFER_JSONL}" \
  --prediction-dir "${PREDICTION_DIR}" \
  --output-json "${OUTPUT_ROOT}/full_prediction_completeness.json"

echo "[threshold-ab] selecting the legacy <${MAX_OLD_BLACK_RATIO} subset"
python scripts/tools/filter_e2e_predictions_by_black_ratio.py \
  --manifest-json "${MANIFEST_JSON}" \
  --prediction-dir "${PREDICTION_DIR}" \
  --output-dir "${FILTERED_PREDICTION_DIR}" \
  --max-black-ratio "${MAX_OLD_BLACK_RATIO}" \
  --max-exclusive \
  --reset \
  --strict

echo "[threshold-ab] selecting newly added sparse patches"
python scripts/tools/filter_e2e_predictions_by_black_ratio.py \
  --manifest-json "${MANIFEST_JSON}" \
  --prediction-dir "${PREDICTION_DIR}" \
  --output-dir "${ADDED_PREDICTION_DIR}" \
  --min-black-ratio "${MAX_OLD_BLACK_RATIO}" \
  --max-black-ratio 1.0 \
  --max-exclusive \
  --reset \
  --strict

run_patch_eval() {
  local label=$1
  local predictions=$2
  local eval_root=${OUTPUT_ROOT}/${label}_patch_metrics

  echo "[threshold-ab] patch evaluation ${label}: ${predictions}"
  E2E_DATA_SOURCE=raw_direct \
  E2E_RAW_ROOT="${RAW_E2E_ROOT}" \
  EVAL_RUN_ID="${DIAG_RUN_ID}_${label}_patch" \
  EVAL_ROOT="${eval_root}" \
  PREDICTION_DIR="${predictions}" \
  METRICS_OBS_PATH="" \
  REQUIRE_ALL=False \
  bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh"

  if is_true "${VISUALIZE_PATCH_AB}"; then
    local viz_input=${eval_root}/visualization_input
    local viz_output=${eval_root}/visualizations
    local eval_jsonl=${eval_root}/eval_records.jsonl
    if [ ! -s "${eval_jsonl}" ]; then
      echo "ERROR: visualization source JSONL not found or empty: ${eval_jsonl}" >&2
      exit 2
    fi
    mkdir -p "${viz_input}" "${viz_output}"
    rm -f "${viz_input}/summary.json" "${viz_input}/summary.jsonl"
    cp "${eval_jsonl}" "${viz_input}/summary.jsonl"
    python scripts/tools/visualize_centerline.py \
      --input-dir "${viz_input}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-dir "${viz_output}" \
      --map-task lane \
      --max-samples "${VIS_LIMIT}" \
      --no-eval-centerline \
      --skip-whole-map-viz
  fi
}

run_patch_eval black_lt_098 "${FILTERED_PREDICTION_DIR}"
run_patch_eval black_ge_098_lt_100 "${ADDED_PREDICTION_DIR}"
run_patch_eval black_lt_100 "${PREDICTION_DIR}"

PATCH_SUMMARY_PATH=${OUTPUT_ROOT}/patch_metric_comparison.json
ROI_RELATION_STATS_PATH=${OUTPUT_ROOT}/roi_relation_stats.json
python - \
  "${OUTPUT_ROOT}/black_lt_098_patch_metrics/metrics.json" \
  "${OUTPUT_ROOT}/black_ge_098_lt_100_patch_metrics/metrics.json" \
  "${OUTPUT_ROOT}/black_lt_100_patch_metrics/metrics.json" \
  "${PATCH_SUMMARY_PATH}" \
  "${SANITIZE_REPORT}" \
  "${ROI_RELATION_STATS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

labels = ("black_ratio_lt_0.98", "black_ratio_ge_0.98_lt_1.0", "black_ratio_lt_1.0")
summary = {}
for label, value in zip(labels, sys.argv[1:4]):
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = payload["centerline_eval"]
    coverage = payload["coverage"]
    summary[label] = {
        "metrics_json": str(path),
        "evaluated_records": coverage["evaluated_records"],
        "pairing_errors": coverage["pairing_errors"],
        "prediction_conversion_errors": coverage["prediction_conversion_errors"],
        "instance_precision": metric["instance_pre"],
        "instance_recall": metric["instance_recall"],
        "instance_f1": metric["instance_f1"],
        "length_precision": metric["length_pre"],
        "length_recall": metric["length_recall"],
        "length_f1": metric["length_f1"],
        "raw_totals": metric.get("raw_totals", {}),
    }

old = summary[labels[0]]
new = summary[labels[2]]
summary["full_minus_legacy"] = {
    "evaluated_records": new["evaluated_records"] - old["evaluated_records"],
    "instance_precision": new["instance_precision"] - old["instance_precision"],
    "instance_recall": new["instance_recall"] - old["instance_recall"],
    "instance_f1": new["instance_f1"] - old["instance_f1"],
    "length_precision": new["length_precision"] - old["length_precision"],
    "length_recall": new["length_recall"] - old["length_recall"],
    "length_f1": new["length_f1"] - old["length_f1"],
}

output_path = Path(sys.argv[4])
sanitize_report = Path(sys.argv[5])
relation_output_path = Path(sys.argv[6])
if sanitize_report.is_file():
    sanitize_payload = json.loads(sanitize_report.read_text(encoding="utf-8"))
    relation_stats = sanitize_payload.get("roi_relation_stats", {})
    summary["roi_relation_stats"] = relation_stats
    relation_output_path.write_text(
        json.dumps(relation_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
else:
    summary["roi_relation_stats_error"] = f"sanitize report not found: {sanitize_report}"
output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "============================================================"
echo "CONTEXT512 PATCH BLACK-THRESHOLD A/B COMPLETE"
echo "Comparison: ${PATCH_SUMMARY_PATH}"
echo "Legacy:     ${OUTPUT_ROOT}/black_lt_098_patch_metrics/metrics.json"
echo "Added only: ${OUTPUT_ROOT}/black_ge_098_lt_100_patch_metrics/metrics.json"
echo "Full set:   ${OUTPUT_ROOT}/black_lt_100_patch_metrics/metrics.json"
echo "Relations:  ${ROI_RELATION_STATS_PATH}"
if is_true "${VISUALIZE_PATCH_AB}"; then
  echo "Legacy viz: ${OUTPUT_ROOT}/black_lt_098_patch_metrics/visualizations"
  echo "Added viz:  ${OUTPUT_ROOT}/black_ge_098_lt_100_patch_metrics/visualizations"
  echo "Full viz:   ${OUTPUT_ROOT}/black_lt_100_patch_metrics/visualizations"
fi
echo "============================================================"

if ! is_true "${RUN_WHOLEMAP_AB}"; then
  exit 0
fi

run_wholemap_eval() {
  local label=$1
  local predictions=$2
  local result_root=${OUTPUT_ROOT}/${label}
  local run_id=${DIAG_RUN_ID}_${label}

  echo "[threshold-ab] evaluating ${label}: ${predictions}"
  E2E_DATA_SOURCE=raw_direct \
  E2E_RAW_ROOT="${RAW_E2E_ROOT}" \
  E2E_DATA_ROOT="${RAW_E2E_ROOT}" \
  RUN_ID="${run_id}" \
  RUN_WORK_ROOT="/cache/jn/e2e_eval/original_pipeline_runs/${run_id}" \
  RESULT_ROOT="${result_root}" \
  RESULT_OBS_PATH="" \
  PREDICTION_CACHE="${predictions}" \
  REUSE_PREDICTIONS=True \
  REUSE_ENGINE_ARCHIVE=True \
  INSTALL_ENGINE_DEPS=False \
  RESET_EXISTING_MODEL_OUTPUTS=True \
  RUN_FORMAT_STEP=True \
  RUN_RULE_STEP=True \
  RUN_ALL_EVAL=True \
  RUN_LOW_EVAL=False \
  RUN_HIGH_EVAL=False \
  EVAL_SIMPLIFY_PATH=False \
  EVAL_VIS_FLAG=False \
  UPLOAD_RESULTS=False \
  bash "${SCRIPT_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"
}

# Run the subset first so the raw tree is left with the current full-set output.
run_wholemap_eval black_lt_098 "${FILTERED_PREDICTION_DIR}"
run_wholemap_eval black_lt_100 "${PREDICTION_DIR}"

SUMMARY_PATH=${OUTPUT_ROOT}/metric_comparison.txt
{
  echo "legacy subset: black_ratio < ${MAX_OLD_BLACK_RATIO}"
  grep -E "Lane instance total matched num|Lane length gt matched|patch.*found|patch evaluated" \
    "${OUTPUT_ROOT}/black_lt_098/logs/03_eval_all.log" || true
  echo
  echo "current full set: black_ratio < 1.0"
  grep -E "Lane instance total matched num|Lane length gt matched|patch.*found|patch evaluated" \
    "${OUTPUT_ROOT}/black_lt_100/logs/03_eval_all.log" || true
} | tee "${SUMMARY_PATH}"

echo "============================================================"
echo "CONTEXT512 BLACK-THRESHOLD A/B COMPLETE"
echo "Comparison: ${SUMMARY_PATH}"
echo "Subset:     ${OUTPUT_ROOT}/black_lt_098"
echo "Full set:   ${OUTPUT_ROOT}/black_lt_100"
echo "The raw E2E tree now contains the clean full-set model outputs."
echo "============================================================"
