#!/usr/bin/env bash
set -euo pipefail

# Compare the local256-550k and context512-roi256 checkpoints on the same
# sample identities, difficulty buckets, prompts appropriate to each input
# variant, and byte-identical local-ROI assistant ground truth.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-4}
export USE_MEMARTS=${USE_MEMARTS:-0}

RUN_ID=${RUN_ID:-local256_vs_context512_roi256_shared1100_$(date -u +%Y%m%d_%H%M%S)}
COMPARE_ROOT=${COMPARE_ROOT:-/cache/jn/outputs/${RUN_ID}}
VIS_LIMIT=${VIS_LIMIT:-50}

VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}

LOCAL256_FIXED_EVAL_ROOT=${LOCAL256_FIXED_EVAL_ROOT:-/cache/jn/eval_sets/datasetv2_local256_550k_fixed1100_e300_m300_h300_vh200_seed42_v1}
CONTEXT_SHARED_EVAL_ROOT=${CONTEXT_SHARED_EVAL_ROOT:-/cache/jn/eval_sets/context512_roi256_from_local256_shared_fixed1100_v1}

LOCAL256_CHECKPOINT_OBS_PATH=${LOCAL256_CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/}
LOCAL256_CHECKPOINT_CACHE_ROOT=${LOCAL256_CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/local256_550k_checkpoint34376}

CONTEXT_DATASET_OBS_PATH=${CONTEXT_DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar}
CONTEXT_DATASET_ARCHIVE_PATH=${CONTEXT_DATASET_ARCHIVE_PATH:-/cache/jn/data/context512_roi256_550k.tar}
CONTEXT_DATASET_EXTRACT_ROOT=${CONTEXT_DATASET_EXTRACT_ROOT:-/cache/jn/data/context512_roi256_extract}
CONTEXT_DATASET_ROOT=${CONTEXT_DATASET_ROOT:-${CONTEXT_DATASET_EXTRACT_ROOT}/context512_roi256}
CONTEXT_CHECKPOINT_OBS_PATH=${CONTEXT_CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/4f735c63da7a4f86829b26246143e219/output/ma-job-81341482-55b8-4c28-887b-0e4166776561/checkpoint-12504/}
CONTEXT_CHECKPOINT_CACHE_ROOT=${CONTEXT_CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/context512_roi256_checkpoint12504}

copy_obs_parallel() {
  local source_path=$1
  local target_path=$2
  python - "${source_path}" "${target_path}" <<'PY'
import sys
import moxing as mox

print(f"[obs-download] {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
}

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  mkdir -p "$(dirname "${VISION_TOWER}")"
  copy_obs_parallel "${VISION_TOWER_OBS_PATH}" "${VISION_TOWER}"
fi
if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: DINOv2 download is incomplete: ${VISION_TOWER}" >&2
  exit 2
fi

echo "============================================================"
echo "Shared evaluation comparison"
echo "Output root:       ${COMPARE_ROOT}"
echo "Visible NPUs:      ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Processes:         ${NPROC_PER_NODE}"
echo "Per-device batch:  ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "============================================================"

# This first run also creates the canonical local256 fixed-1100 reference set
# when /cache was cleared.
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}" \
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
VISION_TOWER="${VISION_TOWER}" \
FIXED_EVAL_ROOT="${LOCAL256_FIXED_EVAL_ROOT}" \
CHECKPOINT_OBS_PATH="${LOCAL256_CHECKPOINT_OBS_PATH}" \
CHECKPOINT_CACHE_ROOT="${LOCAL256_CHECKPOINT_CACHE_ROOT}" \
OUTPUT_ROOT="${COMPARE_ROOT}/local256_550k_checkpoint34376" \
VIS_LIMIT="${VIS_LIMIT}" \
REBUILD_FIXED_EVAL=False \
bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

if [ ! -f "${CONTEXT_DATASET_ROOT}/phase_a/eval.jsonl" ]; then
  mkdir -p "$(dirname "${CONTEXT_DATASET_ARCHIVE_PATH}")" "${CONTEXT_DATASET_EXTRACT_ROOT}"
  if [ ! -s "${CONTEXT_DATASET_ARCHIVE_PATH}" ]; then
    copy_obs_parallel "${CONTEXT_DATASET_OBS_PATH}" "${CONTEXT_DATASET_ARCHIVE_PATH}"
  fi
  echo "[context-dataset] extracting ${CONTEXT_DATASET_ARCHIVE_PATH}"
  tar -xf "${CONTEXT_DATASET_ARCHIVE_PATH}" -C "${CONTEXT_DATASET_EXTRACT_ROOT}"
fi
if [ ! -f "${CONTEXT_DATASET_ROOT}/phase_a/eval.jsonl" ]; then
  echo "ERROR: context512_roi256 eval JSONL not found below ${CONTEXT_DATASET_ROOT}" >&2
  exit 2
fi

validate_shared_context_eval() {
  python - "${CONTEXT_SHARED_EVAL_ROOT}" "${CONTEXT_DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 200}
ids = set()
for difficulty, expected_count in expected.items():
    path = root / f"{difficulty}.jsonl"
    if not path.is_file():
        raise SystemExit(1)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != expected_count:
        raise SystemExit(1)
    for row in rows:
        sample_id = str(row.get("id", row.get("sample_id", "")))
        if not sample_id or sample_id in ids:
            raise SystemExit(1)
        ids.add(sample_id)
        image = row.get("image", row.get("images", ""))
        if isinstance(image, list):
            image = image[0] if image else ""
        path = Path(str(image))
        if not path.is_absolute():
            path = dataset_root / path
        if not path.is_file():
            raise SystemExit(1)
for name in ("all_selected.jsonl", "manifest.jsonl", "summary.json"):
    if not (root / name).is_file():
        raise SystemExit(1)
if len(ids) != 1100:
    raise SystemExit(1)
print(f"[shared-eval] validated {len(ids)} mapped context samples", flush=True)
PY
}

if ! validate_shared_context_eval; then
  BUILD_ROOT="${CONTEXT_SHARED_EVAL_ROOT}.building.$$"
  rm -rf "${BUILD_ROOT}"
  python scripts/tools/remap_fixed_eval_to_dataset.py \
    --reference-dir "${LOCAL256_FIXED_EVAL_ROOT}" \
    --target-dataset-root "${CONTEXT_DATASET_ROOT}" \
    --output-dir "${BUILD_ROOT}" \
    --target-phase phase_a \
    --scan-target-splits eval test train \
    --allowed-target-splits eval test \
    --patch-size 256 \
    --ground-truth-source reference \
    --require-all

  cp "${LOCAL256_FIXED_EVAL_ROOT}/manifest.jsonl" "${BUILD_ROOT}/manifest.jsonl"
  cp "${LOCAL256_FIXED_EVAL_ROOT}/summary.json" "${BUILD_ROOT}/summary.json"
  python - "${LOCAL256_FIXED_EVAL_ROOT}" "${CONTEXT_DATASET_ROOT}" "${BUILD_ROOT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "reference_eval_root": str(Path(sys.argv[1]).resolve()),
    "target_dataset_root": str(Path(sys.argv[2]).resolve()),
    "mapped_eval_root": str(Path(sys.argv[3]).resolve()),
    "ground_truth_source": "reference",
    "sample_identity": "local256 fixed-1100 IDs and difficulty buckets",
    "context_input_policy": "target context512_roi256 image and prompt",
}
(Path(sys.argv[3]) / "fixed_eval_identity.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY
  rm -rf "${CONTEXT_SHARED_EVAL_ROOT}"
  mv "${BUILD_ROOT}" "${CONTEXT_SHARED_EVAL_ROOT}"
  validate_shared_context_eval
fi

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}" \
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
VISION_TOWER="${VISION_TOWER}" \
FIXED_EVAL_ROOT="${CONTEXT_SHARED_EVAL_ROOT}" \
CHECKPOINT_OBS_PATH="${CONTEXT_CHECKPOINT_OBS_PATH}" \
CHECKPOINT_CACHE_ROOT="${CONTEXT_CHECKPOINT_CACHE_ROOT}" \
OUTPUT_ROOT="${COMPARE_ROOT}/context512_roi256_checkpoint12504" \
VIS_LIMIT="${VIS_LIMIT}" \
REBUILD_FIXED_EVAL=False \
bash scripts/npu/test/test_local_stage_a_lane_intersection_context512_roi256_checkpoint12504_fixed1100_torch240_npu.sh

python - "${COMPARE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    "local256_550k_checkpoint34376": root / "local256_550k_checkpoint34376/checkpoint-34376/by_difficulty/all_selected/eval.json",
    "context512_roi256_checkpoint12504": root / "context512_roi256_checkpoint12504/checkpoint-12504/by_difficulty/all_selected/eval.json",
}
payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
payload["comparison_protocol"] = {
    "same_sample_ids": True,
    "same_difficulty_buckets": True,
    "same_ground_truth": True,
    "local256_input": "256x256 local patch",
    "context_input": "512x512 context with centered 256x256 target ROI",
}
output = root / "shared_fixed1100_comparison.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[comparison] {output}")
PY

echo "============================================================"
echo "COMPARISON COMPLETE"
echo "Local256 results: ${COMPARE_ROOT}/local256_550k_checkpoint34376"
echo "Context results:  ${COMPARE_ROOT}/context512_roi256_checkpoint12504"
echo "Comparison JSON:  ${COMPARE_ROOT}/shared_fixed1100_comparison.json"
echo "Shared eval set:  ${CONTEXT_SHARED_EVAL_ROOT}"
echo "============================================================"
