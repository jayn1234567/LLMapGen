#!/usr/bin/env bash
set -euo pipefail

# One-time Ascend-side preparation for the type-clean 512 lane/intersection SFT data.
# The generated archive is self-contained and can be consumed directly by DI training jobs.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

RAW_DATASET_OBS_PATH=${RAW_DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/xxh/data_lane_intersection_norm_sample_512_typeclean.zip}
PREPARED_DATASET_OBS_PATH=${PREPARED_DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/data_lane_intersection_norm_sample_512_typeclean_sft_taxonomy.zip}
WORK_ROOT=${WORK_ROOT:-/cache/jn/typeclean512_sft_prepare_$(date -u +%Y%m%d_%H%M%S)}
RAW_ARCHIVE=${RAW_ARCHIVE:-${WORK_ROOT}/raw_typeclean512.zip}
EXTRACT_ROOT=${EXTRACT_ROOT:-${WORK_ROOT}/extract}
NORMALIZED_ROOT=${NORMALIZED_ROOT:-${WORK_ROOT}/normalized}
PREPARED_ARCHIVE=${PREPARED_ARCHIVE:-${WORK_ROOT}/data_lane_intersection_norm_sample_512_typeclean_sft_taxonomy.zip}
PHASE=${PHASE:-phase_a}
PROGRESS_EVERY=${PROGRESS_EVERY:-50000}

mkdir -p "${WORK_ROOT}" "${EXTRACT_ROOT}" "${NORMALIZED_ROOT}"

echo "============================================================"
echo "[typeclean-package] repository: ${REPO_ROOT}"
echo "[typeclean-package] raw OBS:    ${RAW_DATASET_OBS_PATH}"
echo "[typeclean-package] output OBS: ${PREPARED_DATASET_OBS_PATH}"
echo "[typeclean-package] work root:  ${WORK_ROOT}"
echo "============================================================"

python - "${RAW_DATASET_OBS_PATH}" "${RAW_ARCHIVE}" <<'PY'
import sys
import moxing as mox

source, target = sys.argv[1:]
print(f"[typeclean-package] download {source} -> {target}", flush=True)
mox.file.copy(source, target)
PY

unzip -q "${RAW_ARCHIVE}" -d "${EXTRACT_ROOT}"

RAW_DATASET_ROOT=$(python - "${EXTRACT_ROOT}" "${PHASE}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
phase_candidates = [
    path for path in root.rglob(f"{phase}/train.jsonl") if "__MACOSX" not in path.parts
]
flat_candidates = [
    path
    for path in root.rglob("train.jsonl")
    if path.parent.name not in {"phase_a", "phase_b"} and "__MACOSX" not in path.parts
]
if len(phase_candidates) == 1:
    print(phase_candidates[0].parent.parent)
elif len(flat_candidates) == 1:
    print(flat_candidates[0].parent)
else:
    candidates = phase_candidates + flat_candidates
    preview = "\n".join(str(path) for path in candidates[:20]) or "<none>"
    raise SystemExit(f"Unable to resolve exactly one dataset root below {root}:\n{preview}")
PY
)

echo "[typeclean-package] resolved dataset root: ${RAW_DATASET_ROOT}"
CONVERSION_REPORT=${WORK_ROOT}/dataset_conversion_summary.json
python scripts/tools/prepare_typeclean_lane_intersection_sft.py \
  --input-root "${RAW_DATASET_ROOT}" \
  --output-root "${NORMALIZED_ROOT}" \
  --phase "${PHASE}" \
  --splits train eval test \
  --progress-every "${PROGRESS_EVERY}" \
  --summary-report "${CONVERSION_REPORT}" \
  --overwrite

if [ -f "${RAW_DATASET_ROOT}/${PHASE}/train.jsonl" ]; then
  TARGET_JSONL_ROOT="${RAW_DATASET_ROOT}/${PHASE}"
  SOURCE_JSONL_ROOT="${NORMALIZED_ROOT}/${PHASE}"
else
  TARGET_JSONL_ROOT="${RAW_DATASET_ROOT}"
  SOURCE_JSONL_ROOT="${NORMALIZED_ROOT}"
fi

for split in train eval test; do
  if [ -f "${SOURCE_JSONL_ROOT}/${split}.jsonl" ]; then
    mv -f "${SOURCE_JSONL_ROOT}/${split}.jsonl" "${TARGET_JSONL_ROOT}/${split}.jsonl"
  fi
done
cp "${CONVERSION_REPORT}" "${RAW_DATASET_ROOT}/dataset_conversion_summary.json"

INSPECTION_REPORT=${RAW_DATASET_ROOT}/dataset_inspection.json
python scripts/tools/inspect_lane_intersection_training_dataset.py \
  --dataset-root "${RAW_DATASET_ROOT}" \
  --phase "${PHASE}" \
  --expected-image-size 512 \
  --coord-min 0 \
  --coord-max 1000 \
  --image-checks-per-split 64 \
  --forbid-lane-type 3 \
  --allowed-centerline-type common \
  --allowed-centerline-type right_turn \
  --allowed-centerline-type other \
  --allowed-intersection-pair '1|1' \
  --allowed-intersection-pair '1|2' \
  --allowed-intersection-pair '1|3' \
  --allowed-intersection-pair '4|1' \
  --require-centerline-type-field \
  --require-intersection-type-fields \
  --require-taxonomy-prompt \
  --strict \
  --report "${INSPECTION_REPORT}"

if ! command -v zip >/dev/null 2>&1; then
  echo "ERROR: zip is required to package the prepared dataset." >&2
  exit 1
fi

DATASET_PARENT=$(dirname "${RAW_DATASET_ROOT}")
DATASET_NAME=$(basename "${RAW_DATASET_ROOT}")
echo "[typeclean-package] creating ${PREPARED_ARCHIVE}"
(
  cd "${DATASET_PARENT}"
  zip -q -1 -r "${PREPARED_ARCHIVE}" "${DATASET_NAME}"
)
sha256sum "${PREPARED_ARCHIVE}" > "${PREPARED_ARCHIVE}.sha256"

python - "${PREPARED_ARCHIVE}" "${PREPARED_DATASET_OBS_PATH}" <<'PY'
import sys
import moxing as mox

source, target = sys.argv[1:]
print(f"[typeclean-package] upload {source} -> {target}", flush=True)
mox.file.copy(source, target)
mox.file.copy(f"{source}.sha256", f"{target}.sha256")
PY

echo "============================================================"
echo "[typeclean-package] complete"
echo "[typeclean-package] local archive: ${PREPARED_ARCHIVE}"
echo "[typeclean-package] OBS archive:   ${PREPARED_DATASET_OBS_PATH}"
cat "${PREPARED_ARCHIVE}.sha256"
echo "============================================================"
