#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-0}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cache/jn/checkpoint-29610}"
RUN_ROOT="${RUN_ROOT:-}"
RUN_ARGS_JSON="${RUN_ARGS_JSON:-}"
TRAINROOT="${TRAINROOT:-/cache/jn/prepared_lane_intersection_trainroot}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-npu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
VIS_LIMIT="${VIS_LIMIT:-64}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
CATEGORIES="${CATEGORIES:-centerline,intersection}"
METER_PER_PIXEL="${METER_PER_PIXEL:-0.2}"
LINE_WIDTH_PX="${LINE_WIDTH_PX:-6}"

RUN_NAME="${RUN_NAME:-$(basename "${CHECKPOINT_DIR}")_${SPLIT}_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/cache/jn/outputs/infer_eval_visualize_${RUN_NAME}}"
PRED_JSONL="${PRED_JSONL:-${OUTPUT_DIR}/predictions.jsonl}"
PRED_SUMMARY_JSON="${PRED_SUMMARY_JSON:-${OUTPUT_DIR}/predict_summary.json}"

mkdir -p "${OUTPUT_DIR}"

PREDICT_ARGS=(
  scripts/predict_dinov2_centerline.py
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --trainroot "${TRAINROOT}"
  --split "${SPLIT}"
  --output-jsonl "${PRED_JSONL}"
  --summary-json "${PRED_SUMMARY_JSON}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --device "${DEVICE}"
)

if [ -n "${RUN_ROOT}" ]; then
  PREDICT_ARGS+=(--run-root "${RUN_ROOT}")
fi
if [ -n "${RUN_ARGS_JSON}" ]; then
  PREDICT_ARGS+=(--run-args-json "${RUN_ARGS_JSON}")
fi
if [ "${LOCAL_FILES_ONLY}" = "true" ]; then
  PREDICT_ARGS+=(--local-files-only)
fi

echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Run root:   ${RUN_ROOT:-<auto>}"
echo "Args json:  ${RUN_ARGS_JSON:-<auto>}"
echo "Trainroot:  ${TRAINROOT}"
echo "Split:      ${SPLIT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Device:     ${DEVICE}, visible=${ASCEND_RT_VISIBLE_DEVICES}"
echo "============================================================"

python "${PREDICT_ARGS[@]}"

python scripts/tools/eval_visualize_dinov2_centerline_predictions.py \
  --pred-jsonl "${PRED_JSONL}" \
  --trainroot "${TRAINROOT}" \
  --out-dir "${OUTPUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --categories "${CATEGORIES}" \
  --meter-per-pixel "${METER_PER_PIXEL}" \
  --line-width-px "${LINE_WIDTH_PX}" \
  --vis-limit "${VIS_LIMIT}"

echo "============================================================"
echo "Done."
echo "Prediction JSONL: ${PRED_JSONL}"
echo "Prediction summary: ${PRED_SUMMARY_JSON}"
echo "Eval summary: ${OUTPUT_DIR}/eval_visualization_summary.json"
echo "Official eval: ${OUTPUT_DIR}/eval_official.json"
echo "Engineering eval: ${OUTPUT_DIR}/eval_engineering.json"
echo "Visualization sheet: ${OUTPUT_DIR}/visualization/prediction_overlay_sheet.png"
echo "============================================================"
