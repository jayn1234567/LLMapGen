#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-0}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CKPT_DIR:-}}"
TRAINROOT="${TRAINROOT:-${DATA_ROOT:-}}"

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the training output or checkpoint directory.}"
: "${TRAINROOT:?Set TRAINROOT to the trainroot containing split jsonl files.}"

SPLIT="${SPLIT:-val}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${PRED_OUTPUT_JSONL:-outputs/dinov2_centerline_qwen_lora_npu_${SPLIT}_pred.jsonl}}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"

set -- \
  scripts/predict_dinov2_centerline.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --trainroot "${TRAINROOT}" \
  --split "${SPLIT}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device npu

if [ -n "${SUMMARY_JSON:-}" ]; then
  set -- "$@" --summary-json "${SUMMARY_JSON}"
fi
if [ "${LOCAL_FILES_ONLY}" = "true" ]; then
  set -- "$@" --local-files-only
fi

python "$@"
