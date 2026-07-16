#!/usr/bin/env bash
set -euo pipefail

# Classify Jiangjihua-style UniMapGen records with the shared geometry-v2
# difficulty rule and render deterministic ground-truth samples per bucket.
# This is a CPU-only inspection job and does not initialize an NPU.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/cache/jn/data_line_samples_33w}
PHASE=${PHASE:-phase_a}
SPLIT=${SPLIT:-train}
JSONL_PATH=${JSONL_PATH:-}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_ROOT}}
OUTPUT_DIR=${OUTPUT_DIR:-/cache/jn/outputs/jiangjihua_difficulty_${PHASE}_${SPLIT}}
SAMPLES_PER_DIFFICULTY=${SAMPLES_PER_DIFFICULTY:-100}
MAX_SAMPLES=${MAX_SAMPLES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-10000}
SEED=${SEED:-42}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
INCLUDE_EMPTY=${INCLUDE_EMPTY:-false}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [ -z "${JSONL_PATH}" ]; then
  candidates=(
    "${DATASET_ROOT}/${PHASE}/${SPLIT}.jsonl"
    "${DATASET_ROOT}/${PHASE//_/}/${SPLIT}.jsonl"
    "${DATASET_ROOT}/${SPLIT}.jsonl"
  )
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      JSONL_PATH="${candidate}"
      break
    fi
  done
fi

if [ -z "${JSONL_PATH}" ] || [ ! -f "${JSONL_PATH}" ]; then
  echo "ERROR: unable to find ${SPLIT}.jsonl under ${DATASET_ROOT}."
  echo "Set JSONL_PATH=/absolute/path/to/${SPLIT}.jsonl explicitly."
  exit 1
fi
if [ ! -d "${IMAGE_FOLDER}" ]; then
  echo "ERROR: IMAGE_FOLDER does not exist: ${IMAGE_FOLDER}"
  exit 1
fi
if ! [[ "${SAMPLES_PER_DIFFICULTY}" =~ ^[0-9]+$ ]] || [ "${SAMPLES_PER_DIFFICULTY}" -lt 1 ]; then
  echo "ERROR: SAMPLES_PER_DIFFICULTY must be a positive integer."
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 9):
    raise SystemExit(f"Python >= 3.9 is required, found {sys.version}")
from PIL import Image  # noqa: F401

print(f"[difficulty-viz] python preflight passed: {sys.version.split()[0]}")
PY

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "[difficulty-viz] repository:       ${REPO_ROOT}"
echo "[difficulty-viz] dataset root:     ${DATASET_ROOT}"
echo "[difficulty-viz] input JSONL:      ${JSONL_PATH}"
echo "[difficulty-viz] image folder:     ${IMAGE_FOLDER}"
echo "[difficulty-viz] phase/split:      ${PHASE}/${SPLIT}"
echo "[difficulty-viz] samples/bucket:   ${SAMPLES_PER_DIFFICULTY}"
echo "[difficulty-viz] include empty:    ${INCLUDE_EMPTY}"
echo "[difficulty-viz] coordinate mode:  ${COORD_MODE}"
echo "[difficulty-viz] output:           ${OUTPUT_DIR}"
echo "============================================================"

command=(
  "${PYTHON_BIN}" scripts/tools/tag_hard_map_samples.py
  --dataset-root "${DATASET_ROOT}"
  --phase "${PHASE}"
  --split "${SPLIT}"
  --jsonl "${JSONL_PATH}"
  --image-folder "${IMAGE_FOLDER}"
  --output-dir "${OUTPUT_DIR}"
  --max-samples "${MAX_SAMPLES}"
  --progress-every "${PROGRESS_EVERY}"
  --visualize-top-k 0
  --visualize-per-difficulty "${SAMPLES_PER_DIFFICULTY}"
  --visualize-difficulties easy medium hard very_hard
  --coord-mode "${COORD_MODE}"
  --coord-range "${COORD_RANGE}"
  --seed "${SEED}"
)
if is_true "${INCLUDE_EMPTY}"; then
  command+=(--include-empty)
fi

printf '[difficulty-viz] command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

echo "============================================================"
echo "[difficulty-viz] completed"
echo "[difficulty-viz] all tags:       ${OUTPUT_DIR}/sample_tags.jsonl"
echo "[difficulty-viz] summary:        ${OUTPUT_DIR}/summary.json"
echo "[difficulty-viz] easy images:    ${OUTPUT_DIR}/viz_by_difficulty/easy"
echo "[difficulty-viz] medium images:  ${OUTPUT_DIR}/viz_by_difficulty/medium"
echo "[difficulty-viz] hard images:    ${OUTPUT_DIR}/viz_by_difficulty/hard"
echo "[difficulty-viz] very hard:      ${OUTPUT_DIR}/viz_by_difficulty/very_hard"
echo "[difficulty-viz] contact sheets: ${OUTPUT_DIR}/contact_sheet_*.png"
echo "============================================================"
