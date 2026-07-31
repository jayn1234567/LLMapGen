#!/usr/bin/env bash
set -euo pipefail

# Run the original RC E2E post-processing and evaluation projects unchanged:
# raw MLLM patch JSON -> infer_result_format.py -> center_lane_rule/test_rule.py
# -> E2E_EVAL/Evaluation/multi_main.py (all / low / high).

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

SOURCE_ENV_DIR=${SOURCE_ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
SOURCE_ACTIVATE_SCRIPT=${SOURCE_ACTIVATE_SCRIPT:-${SOURCE_ENV_DIR}/activate_mllm_infer_torch240.sh}
E2E_ENV_DIR=${E2E_ENV_DIR:-/home/ma-user/.conda/envs/rc-e2e-original-py311}
CONDA_SH=${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}

ORIGINAL_ENGINE_OBS_PATH=${ORIGINAL_ENGINE_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/code/rc_nn-sjn_e2e_eval.zip}
E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
PREDICTION_OBS_PATH=${PREDICTION_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_infer/context512_roi256_checkpoint12504_e2e_data_full_v1}

E2E_CACHE_ROOT=${E2E_CACHE_ROOT:-/cache/jn/e2e_eval/original_pipeline_cache}
ENGINE_ARCHIVE=${ENGINE_ARCHIVE:-${E2E_CACHE_ROOT}/rc_nn-sjn_e2e_eval.zip}
ENGINE_EXTRACT_ROOT=${ENGINE_EXTRACT_ROOT:-${E2E_CACHE_ROOT}/engine}
E2E_DATA_ARCHIVE=${E2E_DATA_ARCHIVE:-/cache/jn/e2e_eval/e2e_data.zip}
E2E_RAW_ROOT=${E2E_RAW_ROOT:-/cache/jn/e2e_eval/raw_e2e_data}
PREDICTION_CACHE=${PREDICTION_CACHE:-${E2E_CACHE_ROOT}/predictions/context512_roi256_checkpoint12504_e2e_data_full_v1}

RUN_ID=${RUN_ID:-context512_roi256_checkpoint12504_original_e2e_$(date +%Y%m%d_%H%M%S)}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/e2e_eval/original_pipeline_runs/${RUN_ID}}
E2E_DATA_ROOT=${E2E_DATA_ROOT:-${RUN_WORK_ROOT}/e2e_data}
PREDICTION_INPUT_DIR=${PREDICTION_INPUT_DIR:-${RUN_WORK_ROOT}/prediction_input}
RESULT_ROOT=${RESULT_ROOT:-/cache/jn/outputs/${RUN_ID}}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/output/e2e_metrics/${RUN_ID}}

RULE_WORKERS=${RULE_WORKERS:-16}
INSTALL_ENGINE_DEPS=${INSTALL_ENGINE_DEPS:-True}
RECREATE_E2E_ENV=${RECREATE_E2E_ENV:-False}
REUSE_ENGINE_ARCHIVE=${REUSE_ENGINE_ARCHIVE:-True}
REUSE_PREDICTIONS=${REUSE_PREDICTIONS:-True}
RESET_PREPARED_E2E_DATA=${RESET_PREPARED_E2E_DATA:-False}
E2E_PREPARE_MODE=${E2E_PREPARE_MODE:-hardlink}
E2E_USE_RAW_ROOT_DIRECTLY=${E2E_USE_RAW_ROOT_DIRECTLY:-False}
E2E_DATA_SOURCE=${E2E_DATA_SOURCE:-local_archive}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-True}
PREDICTION_COORD_SCALE=${PREDICTION_COORD_SCALE:-0.256}
RUN_FORMAT_STEP=${RUN_FORMAT_STEP:-True}
RUN_RULE_STEP=${RUN_RULE_STEP:-True}
RUN_ALL_EVAL=${RUN_ALL_EVAL:-True}
RUN_LOW_EVAL=${RUN_LOW_EVAL:-True}
RUN_HIGH_EVAL=${RUN_HIGH_EVAL:-True}
EVAL_SIMPLIFY_PATH=${EVAL_SIMPLIFY_PATH:-False}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-True}
RESET_EXISTING_MODEL_OUTPUTS=${RESET_EXISTING_MODEL_OUTPUTS:-False}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
FILL_MISSING_SCENE_PREDICTIONS=${FILL_MISSING_SCENE_PREDICTIONS:-True}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

has_extracted_e2e_data() {
  [ -f "${E2E_RAW_ROOT}/.extract_complete" ] || \
    find "${E2E_RAW_ROOT}" -type d -name rc_one_patch_release -print -quit 2>/dev/null | grep -q .
}

safe_source() {
  local path=$1
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: required original-engine file not found: $1" >&2
    exit 2
  fi
}

if [ ! -f "${SOURCE_ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: source environment activation script not found: ${SOURCE_ACTIVATE_SCRIPT}" >&2
  exit 2
fi
safe_source "${SOURCE_ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

python - <<'PY'
import moxing as mox
print(f"[original-e2e] moxing={mox.__file__}")
PY

mkdir -p "${E2E_CACHE_ROOT}" "$(dirname "${E2E_DATA_ARCHIVE}")" "${RESULT_ROOT}"

if ! is_true "${REUSE_ENGINE_ARCHIVE}" || [ ! -s "${ENGINE_ARCHIVE}" ]; then
  echo "[original-e2e] downloading original engine ${ORIGINAL_ENGINE_OBS_PATH} -> ${ENGINE_ARCHIVE}"
  python - "${ORIGINAL_ENGINE_OBS_PATH}" "${ENGINE_ARCHIVE}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
else
  echo "[original-e2e] reuse original engine archive: ${ENGINE_ARCHIVE}"
fi

if [ ! -f "${ENGINE_EXTRACT_ROOT}/.extract_complete" ]; then
  echo "[original-e2e] extracting original engine -> ${ENGINE_EXTRACT_ROOT}"
  python - "${ENGINE_ARCHIVE}" "${ENGINE_EXTRACT_ROOT}" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
with zipfile.ZipFile(archive) as handle:
    handle.extractall(destination)
(destination / ".extract_complete").write_text("ok\n", encoding="utf-8")
PY
else
  echo "[original-e2e] reuse extracted original engine: ${ENGINE_EXTRACT_ROOT}"
fi

ORIGINAL_ROOT=$(python - "${ENGINE_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
candidates = [extract_root, *sorted(path for path in extract_root.rglob("*") if path.is_dir())]
for candidate in candidates:
    required = (
        candidate / "data_process" / "infer_result_format.py",
        candidate / "center-lane-preprocess-p2p_run" / "center_lane_rule" / "test_rule.py",
        candidate / "E2E_EVAL" / "Evaluation" / "multi_main.py",
        candidate / "E2E_EVAL" / "Evaluation" / "config.yaml",
    )
    if all(path.is_file() for path in required):
        print(candidate)
        break
else:
    raise FileNotFoundError(
        "Unable to locate an original E2E project root containing data_process, "
        "center-lane-preprocess-p2p_run, and E2E_EVAL below " + str(extract_root)
    )
PY
)

FORMAT_SCRIPT=${ORIGINAL_ROOT}/data_process/infer_result_format.py
RULE_PROJECT=${ORIGINAL_ROOT}/center-lane-preprocess-p2p_run
RULE_ENTRY=${RULE_PROJECT}/center_lane_rule/test_rule.py
RULE_REQUIREMENTS=${RULE_PROJECT}/requirements.txt
EVAL_PROJECT=${ORIGINAL_ROOT}/E2E_EVAL
EVAL_DIR=${EVAL_PROJECT}/Evaluation
EVAL_ENTRY=${EVAL_DIR}/multi_main.py
EVAL_CONFIG=${EVAL_DIR}/config.yaml

require_file "${FORMAT_SCRIPT}"
require_file "${RULE_ENTRY}"
require_file "${RULE_REQUIREMENTS}"
require_file "${EVAL_ENTRY}"
require_file "${EVAL_CONFIG}"

echo "[original-e2e] resolved project root: ${ORIGINAL_ROOT}"
echo "[original-e2e] original requirements: ${RULE_REQUIREMENTS}"

if is_true "${RECREATE_E2E_ENV}" && [ -d "${E2E_ENV_DIR}" ]; then
  python - "${E2E_ENV_DIR}" <<'PY'
import shutil
import sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
allowed_parent = Path("/home/ma-user/.conda/envs").resolve()
if target.parent != allowed_parent or target.name != "rc-e2e-original-py311":
    raise ValueError(f"Refusing to remove unexpected environment path: {target}")
shutil.rmtree(target)
PY
fi

if [ ! -x "${E2E_ENV_DIR}/bin/python" ]; then
  require_file "${CONDA_SH}"
  safe_source "${CONDA_SH}"
  echo "[original-e2e] cloning isolated environment ${SOURCE_ENV_DIR} -> ${E2E_ENV_DIR}"
  conda create -p "${E2E_ENV_DIR}" --clone "${SOURCE_ENV_DIR}" -y
fi

require_file "${CONDA_SH}"
safe_source "${CONDA_SH}"
conda activate "${E2E_ENV_DIR}"

DEPS_SENTINEL=${E2E_ENV_DIR}/.rc_e2e_original_deps_ready
if is_true "${INSTALL_ENGINE_DEPS}" && [ ! -f "${DEPS_SENTINEL}" ]; then
  echo "[original-e2e] installing the original rule-engine requirements once"
  (
    cd "${RULE_PROJECT}"
    python -m pip install -r requirements.txt
  )
  python -m pip install \
    loguru \
    geojson \
    utm \
    pyyaml \
    openpyxl \
    "pandas==2.2.3" \
    ujson \
    scikit-image
  python - "${DEPS_SENTINEL}" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[1]).write_text("ok\n", encoding="utf-8")
PY
else
  echo "[original-e2e] reuse dependency-ready environment: ${E2E_ENV_DIR}"
fi

# The archived evaluator imports pandas from problemExcelWriter.py, but its
# original requirements omit it. Repair already-cached environments too.
if ! python -c "import pandas" >/dev/null 2>&1; then
  echo "[original-e2e] installing evaluator dependency missing from original requirements: pandas"
  python -m pip install "pandas==2.2.3"
fi

python - <<'PY'
import cv2
import geojson
import loguru
import moxing
import numpy
import pandas
import rasterio
import scipy
import shapely
import yaml
print("[original-e2e] dependency preflight passed")
print(f"[original-e2e] numpy={numpy.__version__} shapely={shapely.__version__} rasterio={rasterio.__version__}")
PY

if [ "${E2E_DATA_SOURCE}" = "local_archive" ]; then
  if [ ! -s "${E2E_DATA_ARCHIVE}" ]; then
    echo "ERROR: local E2E archive not found or empty: ${E2E_DATA_ARCHIVE}" >&2
    echo "ERROR: local_archive mode never downloads from OBS or reuses E2E_RAW_ROOT." >&2
    exit 2
  fi
  echo "[original-e2e] preparing a run-local E2E tree from local archive: ${E2E_DATA_ARCHIVE}"
  PREPARE_RESET_FLAG=()
  if is_true "${RESET_PREPARED_E2E_DATA}"; then
    PREPARE_RESET_FLAG=(--reset)
  fi
  python scripts/tools/prepare_rc_e2e_original_run_data.py \
    --archive "${E2E_DATA_ARCHIVE}" \
    --destination "${E2E_DATA_ROOT}" \
    --allowed-root /cache/jn/e2e_eval/original_pipeline_runs \
    "${PREPARE_RESET_FLAG[@]}"
elif [ "${E2E_DATA_SOURCE}" = "raw_direct" ]; then
  if ! has_extracted_e2e_data; then
    echo "ERROR: raw_direct mode requires extracted data below ${E2E_RAW_ROOT}" >&2
    exit 2
  fi
  E2E_DATA_ROOT="${E2E_RAW_ROOT}"
  echo "[original-e2e] DIRECT mode: original pipeline will read and write ${E2E_DATA_ROOT}"
  echo "[original-e2e] WARNING: output_base and post-processing artifacts will modify the raw E2E tree"
elif [ "${E2E_DATA_SOURCE}" = "raw_copy" ]; then
  if ! has_extracted_e2e_data; then
    echo "ERROR: raw_copy mode requires extracted data below ${E2E_RAW_ROOT}" >&2
    exit 2
  fi
  PREPARE_RESET_FLAG=()
  if is_true "${RESET_PREPARED_E2E_DATA}"; then
    PREPARE_RESET_FLAG=(--reset)
  fi
  python scripts/tools/prepare_rc_e2e_original_run_data.py \
    --source-root "${E2E_RAW_ROOT}" \
    --destination "${E2E_DATA_ROOT}" \
    --allowed-root /cache/jn/e2e_eval/original_pipeline_runs \
    --copy-mode "${E2E_PREPARE_MODE}" \
    "${PREPARE_RESET_FLAG[@]}"
elif [ "${E2E_DATA_SOURCE}" = "auto" ] && has_extracted_e2e_data; then
  echo "[original-e2e] reuse extracted E2E data: ${E2E_RAW_ROOT}"
  if is_true "${E2E_USE_RAW_ROOT_DIRECTLY}"; then
    E2E_DATA_ROOT="${E2E_RAW_ROOT}"
    echo "[original-e2e] DIRECT mode: original pipeline will read and write ${E2E_DATA_ROOT}"
    echo "[original-e2e] WARNING: output_base and post-processing artifacts will modify the raw E2E tree"
  else
    PREPARE_RESET_FLAG=()
    if is_true "${RESET_PREPARED_E2E_DATA}"; then
      PREPARE_RESET_FLAG=(--reset)
    fi
    python scripts/tools/prepare_rc_e2e_original_run_data.py \
      --source-root "${E2E_RAW_ROOT}" \
      --destination "${E2E_DATA_ROOT}" \
      --allowed-root /cache/jn/e2e_eval/original_pipeline_runs \
      --copy-mode "${E2E_PREPARE_MODE}" \
      "${PREPARE_RESET_FLAG[@]}"
  fi
elif [ "${E2E_DATA_SOURCE}" = "auto" ]; then
  if [ ! -s "${E2E_DATA_ARCHIVE}" ]; then
    echo "[original-e2e] downloading full E2E data ${E2E_DATA_OBS_PATH} -> ${E2E_DATA_ARCHIVE}"
    python - "${E2E_DATA_OBS_PATH}" "${E2E_DATA_ARCHIVE}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
  else
    echo "[original-e2e] reuse E2E archive: ${E2E_DATA_ARCHIVE}"
  fi

  PREPARE_RESET_FLAG=()
  if is_true "${RESET_PREPARED_E2E_DATA}"; then
    PREPARE_RESET_FLAG=(--reset)
  fi
  python scripts/tools/prepare_rc_e2e_original_run_data.py \
    --archive "${E2E_DATA_ARCHIVE}" \
    --destination "${E2E_DATA_ROOT}" \
    --allowed-root /cache/jn/e2e_eval/original_pipeline_runs \
    "${PREPARE_RESET_FLAG[@]}"
else
  echo "ERROR: unsupported E2E_DATA_SOURCE=${E2E_DATA_SOURCE}; expected local_archive, raw_direct, raw_copy, or auto" >&2
  exit 2
fi

if ! is_true "${REUSE_PREDICTIONS}" || ! find "${PREDICTION_CACHE}" -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "[original-e2e] downloading raw model predictions ${PREDICTION_OBS_PATH} -> ${PREDICTION_CACHE}"
  python - "${PREDICTION_OBS_PATH}" "${PREDICTION_CACHE}" <<'PY'
import shutil
import sys
from pathlib import Path
import moxing as mox

destination = Path(sys.argv[2]).resolve()
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
mox.file.copy_parallel(sys.argv[1], str(destination))
PY
else
  echo "[original-e2e] reuse raw predictions: ${PREDICTION_CACHE}"
fi

PREDICTION_COUNT=$(find "${PREDICTION_CACHE}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [ "${PREDICTION_COUNT}" -le 0 ]; then
  echo "ERROR: no prediction JSON files found below ${PREDICTION_CACHE}" >&2
  exit 2
fi
echo "[original-e2e] prediction JSON count: ${PREDICTION_COUNT}"

echo "[original-e2e] validating raw prediction files for the original formatter"
python - "${PREDICTION_CACHE}" "${PREDICTION_INPUT_DIR}" "${RESULT_ROOT}/invalid_predictions.json" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
report_path = Path(sys.argv[3]).resolve()
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

invalid = []
valid = 0
for path in sorted(source.glob("*.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        image = payload["image"]
        prediction = payload["prediction_json"]
        if not isinstance(image, str) or len(image.replace("\\", "/").split("/")) < 2:
            raise ValueError("invalid image path")
        if not any(part.isdigit() and len(part) > 10 for part in image.replace("\\", "/").split("/")):
            raise ValueError("image path does not contain a numeric patch id")
        parsed = json.loads(prediction)
        lines = parsed["lines"]
        if not isinstance(lines, list):
            raise TypeError("prediction lines is not a list")
        for line_index, line in enumerate(lines):
            if not isinstance(line, dict) or "category" not in line or "points" not in line:
                raise ValueError(f"line {line_index} misses category or points")
            if not isinstance(line["points"], list):
                raise TypeError(f"line {line_index} points is not a list")
            for point_index, point in enumerate(line["points"]):
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    raise ValueError(f"line {line_index} point {point_index} is not a coordinate pair")
                float(point[0])
                float(point[1])
    except Exception as exc:
        invalid.append({"file": str(path), "error": repr(exc)})
        continue

    target = destination / path.name
    try:
        os.link(path, target)
    except OSError:
        shutil.copy2(path, target)
    valid += 1

report = {
    "source": str(source),
    "formatter_input": str(destination),
    "valid": valid,
    "invalid": len(invalid),
    "invalid_records": invalid,
    "policy": "Malformed predictions are omitted and therefore contribute no predicted lanes.",
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: report[key] for key in ("valid", "invalid", "formatter_input")}, ensure_ascii=False))
if valid <= 0:
    raise RuntimeError("No structurally valid prediction JSON files remain")
PY

VALID_PREDICTION_COUNT=$(find "${PREDICTION_INPUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
INVALID_PREDICTION_COUNT=$((PREDICTION_COUNT - VALID_PREDICTION_COUNT))
echo "[original-e2e] formatter input: valid=${VALID_PREDICTION_COUNT} invalid=${INVALID_PREDICTION_COUNT}"

mkdir -p "${RESULT_ROOT}/logs"
cat > "${RESULT_ROOT}/run_manifest.txt" <<EOF
run_id=${RUN_ID}
original_engine_obs=${ORIGINAL_ENGINE_OBS_PATH}
original_root=${ORIGINAL_ROOT}
e2e_data_obs=${E2E_DATA_OBS_PATH}
prediction_obs=${PREDICTION_OBS_PATH}
prediction_count=${PREDICTION_COUNT}
valid_prediction_count=${VALID_PREDICTION_COUNT}
invalid_prediction_count=${INVALID_PREDICTION_COUNT}
e2e_environment=${E2E_ENV_DIR}
EOF

echo "============================================================"
echo "ORIGINAL RC E2E PIPELINE"
echo "Original project: ${ORIGINAL_ROOT}"
echo "E2E data:         ${E2E_DATA_ROOT}"
echo "Predictions:      ${PREDICTION_INPUT_DIR} (${VALID_PREDICTION_COUNT} valid, ${INVALID_PREDICTION_COUNT} invalid)"
echo "Results:          ${RESULT_ROOT}"
echo "Result OBS:       ${RESULT_OBS_PATH}"
echo "============================================================"

if is_true "${RESET_EXISTING_MODEL_OUTPUTS}"; then
  if ! is_true "${RUN_FORMAT_STEP}" || ! is_true "${RUN_RULE_STEP}"; then
    echo "ERROR: RESET_EXISTING_MODEL_OUTPUTS=True requires both format and rule steps enabled." >&2
    exit 2
  fi
  echo "[original-e2e] removing stale model-generated outputs below ${E2E_DATA_ROOT}"
  python - "${E2E_DATA_ROOT}" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
if root != Path("/cache/jn/e2e_eval/raw_e2e_data").resolve():
    raise ValueError(f"Refusing to clean model outputs outside raw_e2e_data: {root}")

removed = []
for scene in sorted(path for path in root.iterdir() if path.is_dir()):
    targets = [
        scene / "output_base",
        scene / "debug_base",
        scene / "rc_one_patch_release" / "center_line_v2" / "lane_ins_res",
    ]
    for target in targets:
        resolved = target.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Unsafe cleanup target: {resolved}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
print(f"[original-e2e] removed {len(removed)} stale model-output directories")
PY
fi

if is_true "${RUN_FORMAT_STEP}"; then
  echo "[original-e2e] step 1/5: original infer_result_format.py"
  python "${FORMAT_SCRIPT}" \
    -i "${PREDICTION_INPUT_DIR}" \
    -o "${E2E_DATA_ROOT}" \
    --scale "${PREDICTION_COORD_SCALE}" \
    2>&1 | tee "${RESULT_ROOT}/logs/01_infer_result_format.log"
else
  echo "[original-e2e] step 1/5: SKIP original infer_result_format.py"
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb
export PYTHONPATH="${RULE_PROJECT}:${PYTHONPATH:-}"
if is_true "${RUN_RULE_STEP}"; then
  echo "[original-e2e] step 2/5: original center_lane_rule/test_rule.py"
  (
    cd "${RULE_PROJECT}/center_lane_rule"
    python "${RULE_ENTRY}" -i "${E2E_DATA_ROOT}" -n "${RULE_WORKERS}"
  ) 2>&1 | tee "${RESULT_ROOT}/logs/02_center_lane_rule.log"
else
  OUTPUT_BASE_COUNT=$(find "${E2E_DATA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name '*output_base' | wc -l)
  if [ "${OUTPUT_BASE_COUNT}" -le 0 ]; then
    echo "ERROR: RUN_RULE_STEP=False but no output_base directories exist below ${E2E_DATA_ROOT}" >&2
    exit 2
  fi
  echo "[original-e2e] step 2/5: SKIP center_lane_rule; reuse ${OUTPUT_BASE_COUNT} output_base directories"
fi

SCENE_COMPLETENESS_ARGS=()
if is_true "${FILL_MISSING_SCENE_PREDICTIONS}"; then
  SCENE_COMPLETENESS_ARGS+=(--fill-missing-predictions)
fi
python scripts/tools/ensure_rc_e2e_scene_outputs.py \
  --e2e-root "${E2E_DATA_ROOT}" \
  --report-json "${RESULT_ROOT}/scene_output_completeness.json" \
  --expected-scenes "${EXPECTED_E2E_SCENES}" \
  --baseline-suffix gt \
  --query-suffix output_base \
  "${SCENE_COMPLETENESS_ARGS[@]}"

CONFIG_BACKUP=${RUN_WORK_ROOT}/original_eval_config.yaml
cp "${EVAL_CONFIG}" "${CONFIG_BACKUP}"
restore_config() {
  cp "${CONFIG_BACKUP}" "${EVAL_CONFIG}" 2>/dev/null || true
}
trap restore_config EXIT

run_original_eval() {
  local mode=$1
  local high=$2
  local low=$3
  local step=$4
  local output=${RESULT_ROOT}/eval_result_${mode}
  python - "${EVAL_CONFIG}" "${E2E_DATA_ROOT}" "${output}" "${high}" "${low}" "${EVAL_SIMPLIFY_PATH}" "${EVAL_VIS_FLAG}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
payload["rootpath"] = sys.argv[2]
payload["baseline_name"] = "gt"
payload["query_name"] = "output_base"
payload["outpath"] = sys.argv[3]
payload["check_high_road"] = sys.argv[4].lower() == "true"
payload["check_low_road"] = sys.argv[5].lower() == "true"
payload["simplify_path"] = sys.argv[6].lower() == "true"
payload["visFlag"] = sys.argv[7].lower() == "true"
config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"[original-e2e] configured {config_path}: {payload}")
PY
  export PYTHONPATH="${EVAL_PROJECT}:${PYTHONPATH:-}"
  (
    cd "${EVAL_DIR}"
    python "${EVAL_ENTRY}"
  ) 2>&1 | tee "${RESULT_ROOT}/logs/${step}_eval_${mode}.log"
}

if is_true "${RUN_ALL_EVAL}"; then
  echo "[original-e2e] step 3/5: original E2E_EVAL all roads"
  run_original_eval all True True 03
else
  echo "[original-e2e] step 3/5: SKIP all-roads evaluation"
fi
if is_true "${RUN_LOW_EVAL}"; then
  echo "[original-e2e] step 4/5: original E2E_EVAL low roads"
  run_original_eval low False True 04
else
  echo "[original-e2e] step 4/5: SKIP low-roads evaluation"
fi
if is_true "${RUN_HIGH_EVAL}"; then
  echo "[original-e2e] step 5/5: original E2E_EVAL high roads"
  run_original_eval high True False 05
else
  echo "[original-e2e] step 5/5: SKIP high-roads evaluation"
fi

restore_config
trap - EXIT

if is_true "${UPLOAD_RESULTS}" && [ -n "${RESULT_OBS_PATH}" ]; then
  echo "[original-e2e] uploading ${RESULT_ROOT} -> ${RESULT_OBS_PATH}"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    python - "${RESULT_ROOT}" "${RESULT_OBS_PATH}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
fi

echo "============================================================"
echo "ORIGINAL RC E2E EVALUATION COMPLETE"
echo "All roads:  ${RESULT_ROOT}/eval_result_all"
echo "Low roads:  ${RESULT_ROOT}/eval_result_low"
echo "High roads: ${RESULT_ROOT}/eval_result_high"
echo "Logs:       ${RESULT_ROOT}/logs"
echo "Result OBS: ${RESULT_OBS_PATH}"
echo "============================================================"
