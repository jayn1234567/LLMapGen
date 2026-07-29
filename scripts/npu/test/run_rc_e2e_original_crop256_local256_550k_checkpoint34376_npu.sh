#!/usr/bin/env bash
set -euo pipefail

# Use the archived RC E2E project's own splitter to create plain local patches,
# then run the matching checkpoint and prompt profile. Defaults preserve the
# original local256 checkpoint-34376 experiment.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}

E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
E2E_WORK_ROOT=${E2E_WORK_ROOT:-/cache/jn/e2e_eval}
E2E_ARCHIVE_PATH=${E2E_ARCHIVE_PATH:-${E2E_WORK_ROOT}/e2e_data.zip}
E2E_RAW_ROOT=${E2E_RAW_ROOT:-${E2E_WORK_ROOT}/raw_e2e_data}

ORIGINAL_ENGINE_OBS_PATH=${ORIGINAL_ENGINE_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/code/rc_nn-sjn_e2e_eval.zip}
ORIGINAL_ENGINE_CACHE=${ORIGINAL_ENGINE_CACHE:-${E2E_WORK_ROOT}/original_pipeline_cache}
ORIGINAL_ENGINE_ARCHIVE=${ORIGINAL_ENGINE_ARCHIVE:-${ORIGINAL_ENGINE_CACHE}/rc_nn-sjn_e2e_eval.zip}
ORIGINAL_ENGINE_EXTRACT_ROOT=${ORIGINAL_ENGINE_EXTRACT_ROOT:-${ORIGINAL_ENGINE_CACHE}/engine}

ORIGINAL_BLACK_RATIO_THRESHOLD=${ORIGINAL_BLACK_RATIO_THRESHOLD:-1.0}
ORIGINAL_PATCH_SIZE=${ORIGINAL_PATCH_SIZE:-256}
E2E_VIEW_MODE=${E2E_VIEW_MODE:-local256}
E2E_PROMPT_PROFILE=${E2E_PROMPT_PROFILE:-local256_550k_v1}
E2E_DATASET_TAG=${E2E_DATASET_TAG:-original_crop${ORIGINAL_PATCH_SIZE}_black1_${E2E_VIEW_MODE}_550k_v2}
E2E_DATASET_ROOT=${E2E_DATASET_ROOT:-${E2E_WORK_ROOT}/e2e_data_${E2E_DATASET_TAG}}
ORIGINAL_MANIFEST=${ORIGINAL_MANIFEST:-${E2E_DATASET_ROOT}/patch_manifest.json}
REBUILD_E2E_DATASET=${REBUILD_E2E_DATASET:-False}
BASE_INFERENCE_SCRIPT=${BASE_INFERENCE_SCRIPT:-${SCRIPT_DIR}/run_rc_e2e_local256_550k_checkpoint34376_npu.sh}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${E2E_WORK_ROOT}" "${ORIGINAL_ENGINE_CACHE}" "$(dirname "${E2E_ARCHIVE_PATH}")"

if [ ! -s "${E2E_ARCHIVE_PATH}" ]; then
  echo "[original-crop256] downloading ${E2E_DATA_OBS_PATH} -> ${E2E_ARCHIVE_PATH}"
  python - "${E2E_DATA_OBS_PATH}" "${E2E_ARCHIVE_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
fi

if [ ! -f "${E2E_RAW_ROOT}/.extract_complete" ]; then
  echo "[original-crop256] extracting E2E data -> ${E2E_RAW_ROOT}"
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

if [ ! -s "${ORIGINAL_ENGINE_ARCHIVE}" ]; then
  echo "[original-crop256] downloading original E2E engine -> ${ORIGINAL_ENGINE_ARCHIVE}"
  python - "${ORIGINAL_ENGINE_OBS_PATH}" "${ORIGINAL_ENGINE_ARCHIVE}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
fi

if [ ! -f "${ORIGINAL_ENGINE_EXTRACT_ROOT}/.extract_complete" ]; then
  echo "[original-crop256] extracting original E2E engine -> ${ORIGINAL_ENGINE_EXTRACT_ROOT}"
  python - "${ORIGINAL_ENGINE_ARCHIVE}" "${ORIGINAL_ENGINE_EXTRACT_ROOT}" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path
archive = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
allowed = Path("/cache/jn/e2e_eval/original_pipeline_cache").resolve()
if allowed not in destination.parents and destination != allowed:
    raise ValueError(f"Refusing to replace unexpected extraction path: {destination}")
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
with zipfile.ZipFile(archive) as handle:
    handle.extractall(destination)
(destination / ".extract_complete").write_text("ok\n", encoding="utf-8")
PY
fi

ORIGINAL_SPLITTER=$(python - "${ORIGINAL_ENGINE_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
matches = sorted(root.rglob("split_inter_tif_for_inference.py"))
if not matches:
    raise FileNotFoundError(f"Original split_inter_tif_for_inference.py not found below {root}")
print(matches[0])
PY
)

ORIGINAL_INPUT_ROOT=$(python - "${E2E_RAW_ROOT}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
candidates = [root, *sorted(path for path in root.iterdir() if path.is_dir())]
for candidate in candidates:
    if any((child / "rc_one_patch_release").is_dir() for child in candidate.iterdir() if child.is_dir()):
        print(candidate)
        break
else:
    raise FileNotFoundError(f"Unable to locate E2E scene directories below {root}")
PY
)

if is_true "${REBUILD_E2E_DATASET}" || [ ! -s "${E2E_DATASET_ROOT}/infer.jsonl" ]; then
  if is_true "${REBUILD_E2E_DATASET}" && [ -e "${E2E_DATASET_ROOT}" ]; then
    python - "${E2E_DATASET_ROOT}" "${E2E_DATASET_TAG}" <<'PY'
import shutil
import sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
expected_name = f"e2e_data_{sys.argv[2]}"
allowed = Path("/cache/jn/e2e_eval").resolve()
if allowed not in target.parents or target.name != expected_name:
    raise ValueError(f"Refusing to remove unexpected dataset path: {target}")
shutil.rmtree(target)
PY
  fi
  mkdir -p "${E2E_DATASET_ROOT}/images" "${E2E_DATASET_ROOT}/tools"
  RUNTIME_SPLITTER=${E2E_DATASET_ROOT}/tools/split_inter_tif_for_inference_black1.py
  python - "${ORIGINAL_SPLITTER}" "${RUNTIME_SPLITTER}" "${ORIGINAL_BLACK_RATIO_THRESHOLD}" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
threshold = float(sys.argv[3])
if not 0.0 <= threshold <= 1.0:
    raise ValueError(f"black ratio threshold must be within [0,1], got {threshold}")
text = source.read_text(encoding="utf-8")
pattern = r"if\s+black_ratio\(pil_img\)\s*>\s*0\.98\s*:"
replacement = f"if black_ratio(pil_img) >= {threshold:g}:"
patched, count = re.subn(pattern, replacement, text)
if count != 1:
    raise RuntimeError(f"Expected exactly one original black-ratio filter, found {count}: {source}")
destination.write_text(patched, encoding="utf-8")
print(f"[original-crop256] runtime filter override: {replacement}")
PY
  echo "[original-crop] running original splitter with black-only filter: ${RUNTIME_SPLITTER}"
  python "${RUNTIME_SPLITTER}" \
    --input-root "${ORIGINAL_INPUT_ROOT}" \
    --output-images "${E2E_DATASET_ROOT}/images" \
    --output-json "${ORIGINAL_MANIFEST}" \
    --patch-size "${ORIGINAL_PATCH_SIZE}" \
    --stride "${ORIGINAL_PATCH_SIZE}"

  python scripts/tools/build_rc_e2e_jsonl_from_original_manifest.py \
    --manifest-json "${ORIGINAL_MANIFEST}" \
    --output-root "${E2E_DATASET_ROOT}" \
    --prompt-profile "${E2E_PROMPT_PROFILE}" \
    --patch-size "${ORIGINAL_PATCH_SIZE}" \
    --coord-range 1000 \
    --black-ratio-threshold "${ORIGINAL_BLACK_RATIO_THRESHOLD}"
else
  echo "[original-crop256] reuse original-crop inference dataset: ${E2E_DATASET_ROOT}"
fi

export E2E_VIEW_MODE
export E2E_TARGET_SIZE=${ORIGINAL_PATCH_SIZE}
export E2E_CONTEXT_SIZE=${ORIGINAL_PATCH_SIZE}
export E2E_PROMPT_PROFILE
export E2E_DATASET_ROOT
export REBUILD_E2E_DATASET=False

exec bash "${BASE_INFERENCE_SCRIPT}"
