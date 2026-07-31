#!/usr/bin/env bash
set -euo pipefail

# Sanitize model output outside the original project, then run the original
# formatter, rule engine, and all-roads evaluator without editing their code.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

INFER_ENV_DIR=${INFER_ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
INFER_ACTIVATE_SCRIPT=${INFER_ACTIVATE_SCRIPT:-${INFER_ENV_DIR}/activate_mllm_infer_torch240.sh}
E2E_ENV_DIR=${E2E_ENV_DIR:-/home/ma-user/.conda/envs/rc-e2e-original-py311}
CONDA_SH=${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}
ORIGINAL_ENGINE_OBS_PATH=${ORIGINAL_ENGINE_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/code/rc_nn-sjn_e2e_eval.zip}
ENGINE_ARCHIVE=${ENGINE_ARCHIVE:-/cache/jn/e2e_eval/original_pipeline_cache/rc_nn-sjn_e2e_eval.zip}

E2E_DATA_ROOT=${E2E_DATA_ROOT:?Set E2E_DATA_ROOT to the fresh 110-scene root}
PREDICTION_DIR=${PREDICTION_DIR:?Set PREDICTION_DIR to raw per-patch prediction JSON}
RUN_WORK_ROOT=${RUN_WORK_ROOT:?Set RUN_WORK_ROOT to an isolated work directory}
RESULT_ROOT=${RESULT_ROOT:?Set RESULT_ROOT for original evaluator outputs}
ORIGINAL_ENGINE_EXTRACT_ROOT=${ORIGINAL_ENGINE_EXTRACT_ROOT:-${RUN_WORK_ROOT}/original_engine}
FORMAT_INPUT_DIR=${FORMAT_INPUT_DIR:-${RUN_WORK_ROOT}/original_formatter_input}
SANITIZE_REPORT=${SANITIZE_REPORT:-${RESULT_ROOT}/prediction_sanitize_report.json}

EXPECTED_SCENES=${EXPECTED_SCENES:-110}
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

require_file "${INFER_ACTIVATE_SCRIPT}"
require_file "${CONDA_SH}"
if [ ! -d "${E2E_DATA_ROOT}" ] || [ ! -d "${PREDICTION_DIR}" ]; then
  echo "ERROR: E2E_DATA_ROOT or PREDICTION_DIR does not exist." >&2
  exit 2
fi

mkdir -p "${RUN_WORK_ROOT}" "${RESULT_ROOT}/logs" "$(dirname "${ENGINE_ARCHIVE}")"
safe_source "${INFER_ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

python scripts/tools/sanitize_rc_e2e_predictions_for_original_formatter.py \
  --input-dir "${PREDICTION_DIR}" \
  --output-dir "${FORMAT_INPUT_DIR}" \
  --report-json "${SANITIZE_REPORT}" \
  --reset

if [ ! -s "${ENGINE_ARCHIVE}" ]; then
  python - "${ORIGINAL_ENGINE_OBS_PATH}" "${ENGINE_ARCHIVE}" <<'PY'
import sys
import moxing as mox
mox.file.copy(sys.argv[1], sys.argv[2])
PY
fi

if [ ! -d "${ORIGINAL_ENGINE_EXTRACT_ROOT}" ]; then
  python - "${ENGINE_ARCHIVE}" "${ORIGINAL_ENGINE_EXTRACT_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(sys.argv[1]) as handle:
    handle.extractall(destination)
PY
else
  echo "[untouched-original] reuse extracted original engine: ${ORIGINAL_ENGINE_EXTRACT_ROOT}"
fi

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
matches = [path for path in [root, *(item for item in root.rglob("*") if item.is_dir())] if all((path / name).is_file() for name in required)]
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

echo "[untouched-original] clearing only generated outputs from the fresh E2E tree"
python - "${E2E_DATA_ROOT}" "${EXPECTED_SCENES}" <<'PY'
import shutil
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
scenes = sorted(path for path in root.iterdir() if path.is_dir() and (path / "rc_one_patch_release").is_dir())
if len(scenes) != expected:
    raise RuntimeError(f"Expected {expected} direct E2E scenes before cleanup, found {len(scenes)}")
removed = []
for scene in scenes:
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
print(f"[untouched-original] removed generated directories: {len(removed)}")
PY

safe_source "${CONDA_SH}"
conda activate "${E2E_ENV_DIR}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb

echo "[untouched-original] step 1/3: original infer_result_format.py"
python "${FORMAT_SCRIPT}" -i "${FORMAT_INPUT_DIR}" -o "${E2E_DATA_ROOT}" --scale "${PREDICTION_COORD_SCALE}" \
  2>&1 | tee "${RESULT_ROOT}/logs/01_infer_result_format.log"

echo "[untouched-original] step 2/3: original center_lane_rule/test_rule.py"
export PYTHONPATH="${RULE_PROJECT}:${PYTHONPATH:-}"
(
  cd "${RULE_PROJECT}/center_lane_rule"
  python "${RULE_ENTRY}" -i "${E2E_DATA_ROOT}" -n "${RULE_WORKERS}"
) 2>&1 | tee "${RESULT_ROOT}/logs/02_center_lane_rule.log"

python - "${E2E_DATA_ROOT}" "${EXPECTED_SCENES}" "${RESULT_ROOT}/scene_completeness.json" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
output_path = Path(sys.argv[3]).resolve()
scenes = sorted(path for path in root.iterdir() if path.is_dir())

def lane_file(scene, suffix):
    paths = [path / "Lane.geojson" for path in scene.iterdir() if path.is_dir() and path.name.endswith(suffix)]
    return next((str(path) for path in paths if path.is_file()), "")

records = [{"scene_id": scene.name, "gt": lane_file(scene, "gt"), "prediction": lane_file(scene, "output_base")} for scene in scenes]
report = {
    "scene_count": len(scenes),
    "gt_count": sum(bool(item["gt"]) for item in records),
    "prediction_count": sum(bool(item["prediction"]) for item in records),
    "missing_gt": [item["scene_id"] for item in records if not item["gt"]],
    "missing_prediction": [item["scene_id"] for item in records if not item["prediction"]],
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if any(report[key] != expected for key in ("scene_count", "gt_count", "prediction_count")):
    raise RuntimeError("Refusing original evaluation without complete 110-scene GT and predictions")
PY

echo "[untouched-original] step 3/3: original E2E_EVAL all roads"
python - "${EVAL_CONFIG}" "${E2E_DATA_ROOT}" "${RESULT_ROOT}/eval_result_all" <<'PY'
import sys
from pathlib import Path
import yaml
config_path = Path(sys.argv[1]).resolve()
payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
payload.update({
    "rootpath": sys.argv[2],
    "baseline_name": "gt",
    "query_name": "output_base",
    "outpath": sys.argv[3],
    "check_high_road": True,
    "check_low_road": True,
    "simplify_path": True,
    "visFlag": True,
})
config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"[untouched-original] original evaluator config: {payload}")
PY
export PYTHONPATH="${EVAL_PROJECT}:${PYTHONPATH:-}"
(
  cd "${EVAL_DIR}"
  python "${EVAL_ENTRY}"
) 2>&1 | tee "${RESULT_ROOT}/logs/03_original_eval_all.log"

if ! grep -Eq "${EXPECTED_SCENES} patch evaluated" "${RESULT_ROOT}/logs/03_original_eval_all.log"; then
  echo "ERROR: original evaluator did not report ${EXPECTED_SCENES} evaluated scenes." >&2
  exit 2
fi

echo "============================================================"
echo "UNTOUCHED ORIGINAL RC E2E COMPLETE"
echo "Sanitize report: ${SANITIZE_REPORT}"
echo "Completeness:    ${RESULT_ROOT}/scene_completeness.json"
echo "Original eval:   ${RESULT_ROOT}/eval_result_all"
echo "Original log:    ${RESULT_ROOT}/logs/03_original_eval_all.log"
echo "============================================================"
