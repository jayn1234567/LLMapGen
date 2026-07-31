#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}

RUN_ID=${RUN_ID:-context512_checkpoint12504_gt_empty_fresh_obs_e2e_$(date +%Y%m%d_%H%M%S)}
FRESH_RUN_ROOT=${FRESH_RUN_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/${RUN_ID}}
FRESH_ARCHIVE=${FRESH_ARCHIVE:-${FRESH_RUN_ROOT}/e2e_data.zip}
FRESH_E2E_ROOT=${FRESH_E2E_ROOT:-${FRESH_RUN_ROOT}/e2e_data}
INFERENCE_DATASET_ROOT=${INFERENCE_DATASET_ROOT:-${FRESH_RUN_ROOT}/context512_roi256_dataset}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${OUTPUT_ROOT}/inference/json}

RULE_WORKERS=${RULE_WORKERS:-16}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}

safe_source() {
  local path=$1
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
safe_source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${FRESH_RUN_ROOT}" "${OUTPUT_ROOT}"
echo "============================================================"
echo "CONTEXT512 GT-EMPTY ORACLE + FRESH OBS ORIGINAL E2E"
echo "Run id:          ${RUN_ID}"
echo "Fresh OBS data:  ${E2E_DATA_OBS_PATH}"
echo "Fresh archive:   ${FRESH_ARCHIVE}"
echo "Fresh E2E root:  ${FRESH_E2E_ROOT}"
echo "Inference set:   ${INFERENCE_DATASET_ROOT}"
echo "New predictions: ${RAW_RESULT_DIR}"
echo "Output:          ${OUTPUT_ROOT}"
echo "Evaluations:     all + low + high"
echo "============================================================"

echo "[fresh-obs-e2e] stage 1/3: force-download and extract a clean E2E dataset"
rm -f "${FRESH_ARCHIVE}"
python - "${E2E_DATA_OBS_PATH}" "${FRESH_ARCHIVE}" <<'PY'
import sys
import moxing as mox

print(f"[fresh-obs-e2e] download {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy(sys.argv[1], sys.argv[2])
PY

python scripts/tools/prepare_rc_e2e_original_run_data.py \
  --archive "${FRESH_ARCHIVE}" \
  --destination "${FRESH_E2E_ROOT}" \
  --allowed-root /cache/jn/e2e_eval/fresh_obs_runs \
  --reset

echo "[fresh-obs-e2e] stage 2/3: rebuild context512/ROI256 inputs and run fresh inference"
E2E_DATA_OBS_PATH="${E2E_DATA_OBS_PATH}" \
E2E_WORK_ROOT="${FRESH_RUN_ROOT}" \
E2E_ARCHIVE_PATH="${FRESH_ARCHIVE}" \
E2E_RAW_ROOT="${FRESH_E2E_ROOT}" \
E2E_DATASET_ROOT="${INFERENCE_DATASET_ROOT}" \
REBUILD_E2E_DATASET=True \
BLACK_RATIO_THRESHOLD=1.0 \
RASTER_ALIGNMENT_REPORT="${FRESH_RUN_ROOT}/raster_alignment_report.json" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
INFER_RESULT_OBS_PATH="" \
bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"

echo "[fresh-obs-e2e] stage 3/3: count/suppress GT-empty patches and run original all/low/high E2E"
SOURCE_PREDICTION_DIR="${RAW_RESULT_DIR}" \
RAW_E2E_ROOT="${FRESH_E2E_ROOT}" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_ORIGINAL_E2E=True \
ORIGINAL_E2E_DATA_SOURCE=raw_direct \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=True \
RUN_HIGH_EVAL=True \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
FILL_MISSING_SCENE_PREDICTIONS=True \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
RULE_WORKERS="${RULE_WORKERS}" \
UPLOAD_RESULTS=False \
bash "${SCRIPT_DIR}/eval_local256_checkpoint34376_gt_nonempty_oracle_original_e2e_npu.sh"

echo "============================================================"
echo "CONTEXT512 FRESH-OBS GT-EMPTY E2E COMPLETE"
echo "Fresh E2E data: ${FRESH_E2E_ROOT}"
echo "Fresh inference:${RAW_RESULT_DIR}"
echo "Suppression:    ${OUTPUT_ROOT}/gt_oracle_suppression_report.json"
echo "Scene audit:    ${OUTPUT_ROOT}/original_pipeline_metrics/scene_output_completeness.json"
echo "All roads:      ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_all"
echo "Low roads:      ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_low"
echo "High roads:     ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_high"
echo "============================================================"
