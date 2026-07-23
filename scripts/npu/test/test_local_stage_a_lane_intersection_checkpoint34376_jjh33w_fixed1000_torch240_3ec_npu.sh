#!/usr/bin/env bash
set -euo pipefail

# Evaluate the local256-550k checkpoint on the exact fixed 1000-sample JJH33W
# reference set. Model and dataset assets are downloaded fresh from OBS; only
# the immutable four-bucket evaluation lists are intentionally reused.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ENV_DIR}/activate_mllm_infer_torch240.sh
REFERENCE_SPLIT_ROOT=${REFERENCE_SPLIT_ROOT:-/cache/jn/eval_sets/jjh33w_1000_e300_m300_h300_vh100_seed42_v1}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

python - "${REFERENCE_SPLIT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 100}
seen_ids = set()
for difficulty, expected_count in expected.items():
    path = root / f"{difficulty}.jsonl"
    if not path.is_file():
        raise SystemExit(f"Required fixed evaluation split is missing: {path}")
    count = 0
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
        if not sample_id:
            raise SystemExit(f"Missing sample id in {path}:{line_number}")
        if sample_id in seen_ids:
            raise SystemExit(f"Duplicate sample id in fixed evaluation set: {sample_id}")
        seen_ids.add(sample_id)
        count += 1
    if count != expected_count:
        raise SystemExit(f"Expected {expected_count} records in {path}, found {count}")
if len(seen_ids) != 1000:
    raise SystemExit(f"Expected 1000 unique fixed evaluation records, found {len(seen_ids)}")
print(f"[fixed-eval] validated exact JJH33W reference set: {root} ({len(seen_ids)} unique samples)")
PY

RUN_ID=${RUN_ID:-checkpoint34376_jjh33w_fixed1000_torch240_$(date -u +%Y%m%d_%H%M%S)}
RUNTIME_ROOT=${RUNTIME_ROOT:-/cache/jn/fresh_assets/${RUN_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}

for target in "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"; do
  case "${target}" in
    /cache/jn/*) ;;
    *)
      echo "ERROR: refusing recursive cleanup outside /cache/jn: ${target}" >&2
      exit 2
      ;;
  esac
done
rm -rf "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"
mkdir -p "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export MASTER_ADDR=127.0.0.1
unset MASTER_PORT

export RUN_ID
export OUTPUT_URL=${OUTPUT_URL:-${OUTPUT_ROOT}}
export LOCAL_OUTPUT_ROOT=${OUTPUT_ROOT}
export OBS_CACHE=${RUNTIME_ROOT}
export MODEL_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints
export VISION_TOWER=${RUNTIME_ROOT}/models/facebook_dinov2-large
export CHECKPOINT_OBS_LIST=obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/
export CHECKPOINT_DOWNLOAD_ROOT=${RUNTIME_ROOT}/checkpoints
export DATASET_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip
export DATASET_ARCHIVE_PATH=${RUNTIME_ROOT}/dataset/data_line_samples_33w.zip
export DATASET_EXTRACT_ROOT=${RUNTIME_ROOT}/dataset/data_line_samples_33w_extract
export DATASET_DIR_NAME=data_line_samples_33w
export DATASET_PATH=${DATASET_EXTRACT_ROOT}/data_line_samples_33w
export IMAGE_FOLDER=${DATASET_PATH}
export TEST_JSON=${DATASET_PATH}/phase_a/test.jsonl
export UPLOAD_RESULTS=False
export REUSE_LOCAL_ASSETS=False

export DIFFICULTY_EVAL=True
export DIFFICULTIES=easy,medium,hard,very_hard
export DIFFICULTY_SAMPLES_PER_BUCKET_SPEC=easy=300,medium=300,hard=300,very_hard=100
export DIFFICULTY_SPLIT_ROOT=${REFERENCE_SPLIT_ROOT}
export DIFFICULTY_REUSE_SPLITS=True
export DIFFICULTY_REBUILD_SPLITS=False
export DIFFICULTY_INCLUDE_EMPTY=False
export DIFFICULTY_VIS_LIMIT=50
export DIFFICULTY_TOTAL_EVAL=True
export NUM_TEST_SAMPLES=0
export MAX_NEW_TOKENS=2048
export CHECKPOINT_DEEPSTACK_MODE=disabled

export INSTALL_DEPS=False
export ENABLE_MOXING_UPGRADE=False

echo "[fixed-eval] checkpoint: checkpoint-34376"
echo "[fixed-eval] dataset: fresh JJH33W download from ${DATASET_OBS_PATH}"
echo "[fixed-eval] split root: ${REFERENCE_SPLIT_ROOT}"
echo "[fixed-eval] output root: ${OUTPUT_ROOT}"

exec bash "${SCRIPT_DIR}/test_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh"
