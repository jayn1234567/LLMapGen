#!/usr/bin/env bash
set -euo pipefail

# Evaluate the Raw-Lane local256 and context512/ROI256 200k checkpoints on
# independently sampled, deterministic fixed-1100 holdouts. Both use seed 42
# and the same difficulty counts, but sample identities are not forced equal.

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

safe_source() {
  local path=$1
  if [ -f "${path}" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${path}"
    set -u
  fi
}

safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
safe_source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-4}
export USE_MEMARTS=${USE_MEMARTS:-0}

RUN_ID=${RUN_ID:-rawlane200k_local256_vs_context512_roi256_seed42_$(date -u +%Y%m%d_%H%M%S)}
COMPARE_ROOT=${COMPARE_ROOT:-/cache/jn/outputs/${RUN_ID}}
VIS_LIMIT=${VIS_LIMIT:-50}
FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}
FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}
REBUILD_FIXED_EVAL=${REBUILD_FIXED_EVAL:-False}

VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}

RAWLANE_LOCAL_DATASET_OBS_PATH=${RAWLANE_LOCAL_DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_200k_rawlane/local256_200k.tar}
RAWLANE_LOCAL_ARCHIVE_PATH=${RAWLANE_LOCAL_ARCHIVE_PATH:-/cache/jn/data/rawlane_local256_200k.tar}
RAWLANE_LOCAL_EXTRACT_ROOT=${RAWLANE_LOCAL_EXTRACT_ROOT:-/cache/jn/data/rawlane_local256_200k_extract}
RAWLANE_LOCAL_DATASET_ROOT=${RAWLANE_LOCAL_DATASET_ROOT:-${RAWLANE_LOCAL_EXTRACT_ROOT}/local256}
RAWLANE_LOCAL_FIXED_EVAL_ROOT=${RAWLANE_LOCAL_FIXED_EVAL_ROOT:-/cache/jn/eval_sets/rawlane_local256_200k_fixed1100_e300_m300_h300_vh200_seed42_v1}
RAWLANE_LOCAL_CHECKPOINT_OBS_PATH=${RAWLANE_LOCAL_CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/29/3bf5a8001ec6433ca4ee973564c29976/output/ma-job-a782316a-32ec-4958-ae1f-44c69fdedd3f/checkpoint-12504/}
RAWLANE_LOCAL_CHECKPOINT_CACHE_ROOT=${RAWLANE_LOCAL_CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_local256_200k_checkpoint12504}

RAWLANE_CONTEXT_DATASET_OBS_PATH=${RAWLANE_CONTEXT_DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256_200k_rawlane/context512_roi256_200k.tar}
RAWLANE_CONTEXT_ARCHIVE_PATH=${RAWLANE_CONTEXT_ARCHIVE_PATH:-/cache/jn/data/rawlane_context512_roi256_200k.tar}
RAWLANE_CONTEXT_EXTRACT_ROOT=${RAWLANE_CONTEXT_EXTRACT_ROOT:-/cache/jn/data/rawlane_context512_roi256_200k_extract}
RAWLANE_CONTEXT_DATASET_ROOT=${RAWLANE_CONTEXT_DATASET_ROOT:-${RAWLANE_CONTEXT_EXTRACT_ROOT}/context512_roi256_200k}
RAWLANE_CONTEXT_FIXED_EVAL_ROOT=${RAWLANE_CONTEXT_FIXED_EVAL_ROOT:-/cache/jn/eval_sets/rawlane_context512_roi256_200k_fixed1100_e300_m300_h300_vh200_seed42_v1}
RAWLANE_CONTEXT_CHECKPOINT_OBS_PATH=${RAWLANE_CONTEXT_CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/30/ea77c85a9d54442b825b86cd7f547a26/output/ma-job-2e7c82dd-3a05-440b-b686-db5c3bcc2512/checkpoint-12504/}
RAWLANE_CONTEXT_CHECKPOINT_CACHE_ROOT=${RAWLANE_CONTEXT_CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_context512_roi256_200k_checkpoint12504}

copy_obs_parallel() {
  local source_path=$1
  local target_path=$2
  mkdir -p "$(dirname "${target_path}")"
  python - "${source_path}" "${target_path}" <<'PY'
import sys
import moxing as mox

print(f"[obs-download] {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
}

copy_obs_file() {
  local source_path=$1
  local target_path=$2
  mkdir -p "$(dirname "${target_path}")"
  python - "${source_path}" "${target_path}" <<'PY'
import sys
import moxing as mox

print(f"[obs-download] {sys.argv[1]} -> {sys.argv[2]}", flush=True)
mox.file.copy(sys.argv[1], sys.argv[2])
PY
}

prepare_and_resolve_dataset() {
  local obs_path=$1
  local archive_path=$2
  local extract_root=$3
  local preferred_name=$4

  if ! find "${extract_root}" -type f \( -path '*/phase_a/eval.jsonl' -o -path '*/phase_a/val.jsonl' \) -print -quit 2>/dev/null | grep -q .; then
    if [ ! -s "${archive_path}" ]; then
      copy_obs_file "${obs_path}" "${archive_path}" >&2
    else
      echo "[dataset] reuse archive: ${archive_path}" >&2
    fi
    mkdir -p "${extract_root}"
    echo "[dataset] extracting ${archive_path} -> ${extract_root}" >&2
    tar -xf "${archive_path}" -C "${extract_root}"
  else
    echo "[dataset] reuse extracted files: ${extract_root}" >&2
  fi

  python - "${extract_root}" "${preferred_name}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
preferred_name = sys.argv[2]
preferred = extract_root / preferred_name
phase_roots = []
flat_roots = []

def add(root):
    root = root.resolve()
    phase = root / "phase_a"
    if (phase / "train.jsonl").is_file() and any(
        path.is_file() for path in (phase / "eval.jsonl", phase / "val.jsonl")
    ):
        if root not in phase_roots:
            phase_roots.append(root)
        return
    if (root / "train.jsonl").is_file() and any(
        path.is_file() for path in (root / "eval.jsonl", root / "val.jsonl")
    ):
        if root not in flat_roots:
            flat_roots.append(root)

add(preferred)
add(extract_root)
for train_path in sorted(extract_root.rglob("train.jsonl")):
    if "__MACOSX" in train_path.parts:
        continue
    root = train_path.parent.parent if train_path.parent.name == "phase_a" else train_path.parent
    add(root)

resolved = phase_roots or flat_roots
if len(resolved) != 1:
    preview = "\n".join(str(root) for root in resolved[:20]) or "<none>"
    raise SystemExit(f"Unable to resolve exactly one Dataset V2 root below {extract_root}:\n{preview}")
print(resolved[0])
PY
}

resolve_eval_jsonl() {
  local dataset_root=$1
  local candidate
  for candidate in \
    "${dataset_root}/phase_a/eval.jsonl" \
    "${dataset_root}/phase_a/val.jsonl" \
    "${dataset_root}/eval.jsonl" \
    "${dataset_root}/val.jsonl"; do
    if [ -f "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "ERROR: no eval.jsonl or val.jsonl found below ${dataset_root}" >&2
  return 2
}

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  copy_obs_parallel "${VISION_TOWER_OBS_PATH}" "${VISION_TOWER}"
fi
if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: DINOv2 download is incomplete: ${VISION_TOWER}" >&2
  exit 2
fi

RAWLANE_LOCAL_DATASET_ROOT=$(prepare_and_resolve_dataset \
  "${RAWLANE_LOCAL_DATASET_OBS_PATH}" \
  "${RAWLANE_LOCAL_ARCHIVE_PATH}" \
  "${RAWLANE_LOCAL_EXTRACT_ROOT}" \
  local256)
RAWLANE_CONTEXT_DATASET_ROOT=$(prepare_and_resolve_dataset \
  "${RAWLANE_CONTEXT_DATASET_OBS_PATH}" \
  "${RAWLANE_CONTEXT_ARCHIVE_PATH}" \
  "${RAWLANE_CONTEXT_EXTRACT_ROOT}" \
  context512_roi256_200k)
RAWLANE_LOCAL_EVAL_JSONL=$(resolve_eval_jsonl "${RAWLANE_LOCAL_DATASET_ROOT}")
RAWLANE_CONTEXT_EVAL_JSONL=$(resolve_eval_jsonl "${RAWLANE_CONTEXT_DATASET_ROOT}")

mkdir -p "${COMPARE_ROOT}"
echo "============================================================"
echo "Raw-Lane 200k independent seed-42 evaluation"
echo "Output root:        ${COMPARE_ROOT}"
echo "Visible NPUs:       ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Processes:          ${NPROC_PER_NODE}"
echo "Per-device batch:   ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Difficulty counts:  ${FIXED_EVAL_COUNTS}"
echo "Seed for each set:  ${FIXED_EVAL_SEED}"
echo "Same sample IDs:    false"
echo "Local dataset root: ${RAWLANE_LOCAL_DATASET_ROOT}"
echo "Context data root:  ${RAWLANE_CONTEXT_DATASET_ROOT}"
echo "============================================================"

COMMON_EVAL_ENV=(
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
  ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}"
  NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${NPROC_PER_NODE}"
  PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}"
  VISION_TOWER="${VISION_TOWER}"
  FIXED_EVAL_COUNTS="${FIXED_EVAL_COUNTS}"
  FIXED_EVAL_SEED="${FIXED_EVAL_SEED}"
  REBUILD_FIXED_EVAL="${REBUILD_FIXED_EVAL}"
  PATCH_SIZE=256
  VIS_LIMIT="${VIS_LIMIT}"
)

env "${COMMON_EVAL_ENV[@]}" \
  DATASET_OBS_PATH="${RAWLANE_LOCAL_DATASET_OBS_PATH}" \
  DATASET_ARCHIVE_PATH="${RAWLANE_LOCAL_ARCHIVE_PATH}" \
  DATASET_EXTRACT_ROOT="${RAWLANE_LOCAL_EXTRACT_ROOT}" \
  DATASET_ROOT="${RAWLANE_LOCAL_DATASET_ROOT}" \
  EVAL_SOURCE_JSONL="${RAWLANE_LOCAL_EVAL_JSONL}" \
  FIXED_EVAL_ROOT="${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" \
  CHECKPOINT_NAME=checkpoint-12504 \
  CHECKPOINT_OBS_PATH="${RAWLANE_LOCAL_CHECKPOINT_OBS_PATH}" \
  CHECKPOINT_CACHE_ROOT="${RAWLANE_LOCAL_CHECKPOINT_CACHE_ROOT}" \
  OUTPUT_ROOT="${COMPARE_ROOT}/rawlane_local256_200k" \
  RUN_LABEL=rawlane_local256_200k_checkpoint12504_seed42_fixed1100 \
  bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

env "${COMMON_EVAL_ENV[@]}" \
  DATASET_OBS_PATH="${RAWLANE_CONTEXT_DATASET_OBS_PATH}" \
  DATASET_ARCHIVE_PATH="${RAWLANE_CONTEXT_ARCHIVE_PATH}" \
  DATASET_EXTRACT_ROOT="${RAWLANE_CONTEXT_EXTRACT_ROOT}" \
  DATASET_ROOT="${RAWLANE_CONTEXT_DATASET_ROOT}" \
  EVAL_SOURCE_JSONL="${RAWLANE_CONTEXT_EVAL_JSONL}" \
  FIXED_EVAL_ROOT="${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}" \
  CHECKPOINT_NAME=checkpoint-12504 \
  CHECKPOINT_OBS_PATH="${RAWLANE_CONTEXT_CHECKPOINT_OBS_PATH}" \
  CHECKPOINT_CACHE_ROOT="${RAWLANE_CONTEXT_CHECKPOINT_CACHE_ROOT}" \
  OUTPUT_ROOT="${COMPARE_ROOT}/rawlane_context512_roi256_200k" \
  RUN_LABEL=rawlane_context512_roi256_200k_checkpoint12504_seed42_fixed1100 \
  bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

python - "${COMPARE_ROOT}" "${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" "${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}" "${FIXED_EVAL_SEED}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    "rawlane_local256_200k": root / "rawlane_local256_200k/checkpoint-12504/by_difficulty/all_selected/eval.json",
    "rawlane_context512_roi256_200k": root / "rawlane_context512_roi256_200k/checkpoint-12504/by_difficulty/all_selected/eval.json",
}
payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
payload["comparison_protocol"] = {
    "rawlane_local_eval_root": sys.argv[2],
    "rawlane_context_eval_root": sys.argv[3],
    "same_seed": True,
    "seed": int(sys.argv[4]),
    "same_difficulty_counts": True,
    "same_sample_ids": False,
    "same_ground_truth": False,
    "strict_pairwise_comparison": False,
    "sample_counts_per_set": {"easy": 300, "medium": 300, "hard": 300, "very_hard": 200},
    "local_input": "Raw-Lane 256x256 local patch",
    "context_input": "Raw-Lane 512x512 context with centered 256x256 target ROI",
}
output = root / "rawlane200k_seed42_independent_eval_summary.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[comparison] {output}")
PY

echo "============================================================"
echo "RAW-LANE 200K SEED-42 EVALUATION COMPLETE"
echo "Local256 results:   ${COMPARE_ROOT}/rawlane_local256_200k"
echo "Context results:    ${COMPARE_ROOT}/rawlane_context512_roi256_200k"
echo "Combined summary:   ${COMPARE_ROOT}/rawlane200k_seed42_independent_eval_summary.json"
echo "Local eval set:     ${RAWLANE_LOCAL_FIXED_EVAL_ROOT}"
echo "Context eval set:   ${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}"
echo "============================================================"
