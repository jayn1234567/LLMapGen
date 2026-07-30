#!/usr/bin/env bash
set -euo pipefail

# Fresh OBS E2E data -> current context512/ROI256 crop and inference -> the
# original RC formatter, rule engine, and evaluator without repository fixes.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

INFER_ENV_DIR=${INFER_ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
INFER_ACTIVATE_SCRIPT=${INFER_ACTIVATE_SCRIPT:-${INFER_ENV_DIR}/activate_mllm_infer_torch240.sh}
E2E_ENV_DIR=${E2E_ENV_DIR:-/home/ma-user/.conda/envs/rc-e2e-original-py311}
CONDA_SH=${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}

E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
ORIGINAL_ENGINE_OBS_PATH=${ORIGINAL_ENGINE_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/code/rc_nn-sjn_e2e_eval.zip}
ENGINE_ARCHIVE=${ENGINE_ARCHIVE:-/cache/jn/e2e_eval/original_pipeline_cache/rc_nn-sjn_e2e_eval.zip}

RUN_ID=${RUN_ID:-context512_checkpoint12504_fresh_original_$(date -u +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/cache/jn/e2e_eval/fresh_original_runs/${RUN_ID}}
FRESH_E2E_ARCHIVE=${FRESH_E2E_ARCHIVE:-${RUN_ROOT}/e2e_data.zip}
FRESH_E2E_EXTRACT_ROOT=${FRESH_E2E_EXTRACT_ROOT:-${RUN_ROOT}/fresh_e2e_extract}
RESOLVED_E2E_ROOT_FILE=${RESOLVED_E2E_ROOT_FILE:-${RUN_ROOT}/resolved_e2e_root.txt}
INFERENCE_DATASET_ROOT=${INFERENCE_DATASET_ROOT:-${RUN_ROOT}/context512_roi256_dataset}
ORIGINAL_ENGINE_EXTRACT_ROOT=${ORIGINAL_ENGINE_EXTRACT_ROOT:-${RUN_ROOT}/original_engine}
INFERENCE_OUTPUT_ROOT=${INFERENCE_OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${INFERENCE_OUTPUT_ROOT}/inference/json}
ORIGINAL_RESULT_ROOT=${ORIGINAL_RESULT_ROOT:-${INFERENCE_OUTPUT_ROOT}/original_engine_all_roads}

EXPECTED_SCENES=${EXPECTED_SCENES:-110}
BLACK_RATIO_THRESHOLD=${BLACK_RATIO_THRESHOLD:-1.0}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
RULE_WORKERS=${RULE_WORKERS:-16}
PREDICTION_COORD_SCALE=${PREDICTION_COORD_SCALE:-0.256}

safe_source() {
  local path=$1
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: required file not found: $1" >&2
    exit 2
  fi
}

if [ ! -f "${INFER_ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference activation script not found: ${INFER_ACTIVATE_SCRIPT}" >&2
  exit 2
fi
if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: conda activation script not found: ${CONDA_SH}" >&2
  exit 2
fi
if [ -e "${RUN_ROOT}" ]; then
  echo "ERROR: fresh run directory already exists: ${RUN_ROOT}" >&2
  echo "ERROR: choose a new RUN_ID; this entry never reuses a previous E2E data tree." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${INFERENCE_OUTPUT_ROOT}" "$(dirname "${ENGINE_ARCHIVE}")"
safe_source "${INFER_ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

echo "[fresh-original] downloading a fresh E2E archive"
echo "[fresh-original] ${E2E_DATA_OBS_PATH} -> ${FRESH_E2E_ARCHIVE}"
python - "${E2E_DATA_OBS_PATH}" "${FRESH_E2E_ARCHIVE}" <<'PY'
import sys
import moxing as mox

mox.file.copy(sys.argv[1], sys.argv[2])
PY

echo "[fresh-original] extracting fresh E2E data -> ${FRESH_E2E_EXTRACT_ROOT}"
python - \
  "${FRESH_E2E_ARCHIVE}" \
  "${FRESH_E2E_EXTRACT_ROOT}" \
  "${RESOLVED_E2E_ROOT_FILE}" \
  "${EXPECTED_SCENES}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
extract_root = Path(sys.argv[2]).resolve()
result_file = Path(sys.argv[3]).resolve()
expected_scenes = int(sys.argv[4])
extract_root.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(archive) as handle:
    handle.extractall(extract_root)

candidates = []
for candidate in [extract_root, *(path for path in extract_root.rglob("*") if path.is_dir())]:
    scene_dirs = [
        child
        for child in candidate.iterdir()
        if child.is_dir() and (child / "rc_one_patch_release").is_dir()
    ]
    if scene_dirs:
        candidates.append((len(scene_dirs), candidate, scene_dirs))
if not candidates:
    raise FileNotFoundError(f"Unable to find the E2E scene root below {extract_root}")

count, data_root, scene_dirs = max(candidates, key=lambda item: item[0])
entries = list(data_root.iterdir())
non_scene_entries = sorted(str(path) for path in entries if path not in scene_dirs)
if count != expected_scenes:
    raise RuntimeError(f"Fresh E2E archive has {count} direct scenes, expected {expected_scenes}: {data_root}")
if non_scene_entries:
    raise RuntimeError(
        "Original simplify_path=True requires a scene-only root, but extra entries were found: "
        f"{non_scene_entries[:20]}"
    )

result_file.write_text(str(data_root) + "\n", encoding="utf-8")
print(f"[fresh-original] resolved scene-only root: {data_root}")
print(f"[fresh-original] direct scene count: {count}")
PY

FRESH_E2E_ROOT=$(head -n 1 "${RESOLVED_E2E_ROOT_FILE}")
if [ -z "${FRESH_E2E_ROOT}" ] || [ ! -d "${FRESH_E2E_ROOT}" ]; then
  echo "ERROR: resolved fresh E2E root is invalid: ${FRESH_E2E_ROOT}" >&2
  exit 2
fi

echo "[fresh-original] current context512/ROI256 crop and inference"
E2E_DATA_OBS_PATH="${E2E_DATA_OBS_PATH}" \
E2E_ARCHIVE_PATH="${FRESH_E2E_ARCHIVE}" \
E2E_RAW_ROOT="${FRESH_E2E_ROOT}" \
E2E_WORK_ROOT="${RUN_ROOT}" \
E2E_DATASET_ROOT="${INFERENCE_DATASET_ROOT}" \
REBUILD_E2E_DATASET=True \
BLACK_RATIO_THRESHOLD="${BLACK_RATIO_THRESHOLD}" \
VALIDATE_RASTER_ALIGNMENT=True \
RASTER_ALIGNMENT_REPORT="${RUN_ROOT}/raster_alignment_report.json" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT}" \
RAW_RESULT_DIR="${RAW_RESULT_DIR}" \
INFER_RESULT_OBS_PATH="" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
bash "${SCRIPT_DIR}/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"

echo "[fresh-original] preparing an untouched original-engine checkout"
if [ ! -s "${ENGINE_ARCHIVE}" ]; then
  python - "${ORIGINAL_ENGINE_OBS_PATH}" "${ENGINE_ARCHIVE}" <<'PY'
import sys
import moxing as mox

mox.file.copy(sys.argv[1], sys.argv[2])
PY
else
  echo "[fresh-original] reuse original engine archive: ${ENGINE_ARCHIVE}"
fi

python - "${ENGINE_ARCHIVE}" "${ORIGINAL_ENGINE_EXTRACT_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(archive) as handle:
    handle.extractall(destination)
PY

ORIGINAL_ROOT=$(python - "${ORIGINAL_ENGINE_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = (
    Path("data_process/infer_result_format.py"),
    Path("center-lane-preprocess-p2p_run/center_lane_rule/test_rule.py"),
    Path("E2E_EVAL/Evaluation/multi_main.py"),
    Path("E2E_EVAL/Evaluation/config.yaml"),
)
candidates = [root, *(path for path in root.rglob("*") if path.is_dir())]
matches = [candidate for candidate in candidates if all((candidate / item).is_file() for item in required)]
if len(matches) != 1:
    raise RuntimeError(f"Expected one untouched original project root, found {matches}")
print(matches[0])
PY
)

FORMAT_SCRIPT=${ORIGINAL_ROOT}/data_process/infer_result_format.py
RULE_PROJECT=${ORIGINAL_ROOT}/center-lane-preprocess-p2p_run
RULE_ENTRY=${RULE_PROJECT}/center_lane_rule/test_rule.py
EVAL_PROJECT=${ORIGINAL_ROOT}/E2E_EVAL
EVAL_DIR=${EVAL_PROJECT}/Evaluation
EVAL_ENTRY=${EVAL_DIR}/multi_main.py
EVAL_CONFIG=${EVAL_DIR}/config.yaml
require_file "${FORMAT_SCRIPT}"
require_file "${RULE_ENTRY}"
require_file "${EVAL_ENTRY}"
require_file "${EVAL_CONFIG}"

safe_source "${CONDA_SH}"
conda activate "${E2E_ENV_DIR}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb

mkdir -p "${ORIGINAL_RESULT_ROOT}/logs"
echo "[fresh-original] original step 1/3: infer_result_format.py"
python "${FORMAT_SCRIPT}" \
  -i "${RAW_RESULT_DIR}" \
  -o "${FRESH_E2E_ROOT}" \
  --scale "${PREDICTION_COORD_SCALE}" \
  2>&1 | tee "${ORIGINAL_RESULT_ROOT}/logs/01_infer_result_format.log"

echo "[fresh-original] original step 2/3: center_lane_rule/test_rule.py"
export PYTHONPATH="${RULE_PROJECT}:${PYTHONPATH:-}"
(
  cd "${RULE_PROJECT}/center_lane_rule"
  python "${RULE_ENTRY}" -i "${FRESH_E2E_ROOT}" -n "${RULE_WORKERS}"
) 2>&1 | tee "${ORIGINAL_RESULT_ROOT}/logs/02_center_lane_rule.log"

echo "[fresh-original] requiring all ${EXPECTED_SCENES} scenes before original evaluation"
python - "${FRESH_E2E_ROOT}" "${EXPECTED_SCENES}" "${ORIGINAL_RESULT_ROOT}/scene_completeness.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
output_path = Path(sys.argv[3]).resolve()
scenes = sorted(path for path in root.iterdir() if path.is_dir())

def lane_file(scene: Path, suffix: str) -> str:
    matches = sorted(
        str(path / "Lane.geojson")
        for path in scene.iterdir()
        if path.is_dir() and path.name.endswith(suffix) and (path / "Lane.geojson").is_file()
    )
    return matches[0] if matches else ""

records = []
for scene in scenes:
    gt = lane_file(scene, "gt")
    prediction = lane_file(scene, "output_base")
    records.append({"scene_id": scene.name, "gt": gt, "prediction": prediction})

report = {
    "root": str(root),
    "expected_scenes": expected,
    "scene_count": len(scenes),
    "gt_count": sum(bool(item["gt"]) for item in records),
    "prediction_count": sum(bool(item["prediction"]) for item in records),
    "missing_gt": [item["scene_id"] for item in records if not item["gt"]],
    "missing_prediction": [item["scene_id"] for item in records if not item["prediction"]],
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if report["scene_count"] != expected or report["gt_count"] != expected or report["prediction_count"] != expected:
    raise RuntimeError("Refusing to run the original evaluator without complete 110-scene GT and predictions")
PY

echo "[fresh-original] original step 3/3: E2E_EVAL all roads"
python - "${EVAL_CONFIG}" "${FRESH_E2E_ROOT}" "${ORIGINAL_RESULT_ROOT}/eval_result_all" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).resolve()
payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
payload["rootpath"] = sys.argv[2]
payload["baseline_name"] = "gt"
payload["query_name"] = "output_base"
payload["outpath"] = sys.argv[3]
payload["check_high_road"] = True
payload["check_low_road"] = True
payload["simplify_path"] = True
payload["visFlag"] = True
config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"[fresh-original] untouched evaluator config: {payload}")
PY

export PYTHONPATH="${EVAL_PROJECT}:${PYTHONPATH:-}"
(
  cd "${EVAL_DIR}"
  python "${EVAL_ENTRY}"
) 2>&1 | tee "${ORIGINAL_RESULT_ROOT}/logs/03_original_eval_all.log"

if ! grep -Eq "${EXPECTED_SCENES} patch evaluated" "${ORIGINAL_RESULT_ROOT}/logs/03_original_eval_all.log"; then
  echo "ERROR: original evaluator did not report ${EXPECTED_SCENES} evaluated scenes." >&2
  exit 2
fi

echo "============================================================"
echo "FRESH-DATA UNTOUCHED ORIGINAL E2E COMPLETE"
echo "Fresh archive: ${FRESH_E2E_ARCHIVE}"
echo "Fresh E2E root: ${FRESH_E2E_ROOT}"
echo "Inference JSON: ${RAW_RESULT_DIR}"
echo "Completeness:   ${ORIGINAL_RESULT_ROOT}/scene_completeness.json"
echo "Original eval:  ${ORIGINAL_RESULT_ROOT}/eval_result_all"
echo "Original log:   ${ORIGINAL_RESULT_ROOT}/logs/03_original_eval_all.log"
echo "============================================================"
