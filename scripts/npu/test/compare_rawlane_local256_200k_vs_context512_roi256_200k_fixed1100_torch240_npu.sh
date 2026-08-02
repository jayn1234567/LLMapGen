#!/usr/bin/env bash
set -euo pipefail

# Rebuild the canonical Dataset V2 local256 fixed-1100 evaluation identities,
# map those identities to both Raw-Lane 200k views, then evaluate both models
# with identical difficulty buckets and byte-identical assistant ground truth.

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

RUN_ID=${RUN_ID:-rawlane200k_local256_vs_context512_roi256_fixed1100_$(date -u +%Y%m%d_%H%M%S)}
COMPARE_ROOT=${COMPARE_ROOT:-/cache/jn/outputs/${RUN_ID}}
VIS_LIMIT=${VIS_LIMIT:-50}
FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}
FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}

VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}

# This is the exact source used to create the earlier 256 fixed-1100 set.
REFERENCE_DATASET_OBS_PATH=${REFERENCE_DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256/local256.tar}
REFERENCE_ARCHIVE_PATH=${REFERENCE_ARCHIVE_PATH:-/cache/jn/data/fixed1100_reference_local256.tar}
REFERENCE_EXTRACT_ROOT=${REFERENCE_EXTRACT_ROOT:-/cache/jn/data/fixed1100_reference_local256_extract}
REFERENCE_DATASET_DIR_NAME=${REFERENCE_DATASET_DIR_NAME:-local256}
REFERENCE_FIXED_EVAL_ROOT=${REFERENCE_FIXED_EVAL_ROOT:-/cache/jn/eval_sets/datasetv2_local256_550k_fixed1100_e300_m300_h300_vh200_seed42_v1}

RAWLANE_LOCAL_DATASET_OBS_PATH=${RAWLANE_LOCAL_DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_200k_rawlane/local256_200k.tar}
RAWLANE_LOCAL_ARCHIVE_PATH=${RAWLANE_LOCAL_ARCHIVE_PATH:-/cache/jn/data/rawlane_local256_200k.tar}
RAWLANE_LOCAL_EXTRACT_ROOT=${RAWLANE_LOCAL_EXTRACT_ROOT:-/cache/jn/data/rawlane_local256_200k_extract}
RAWLANE_LOCAL_DATASET_DIR_NAME=${RAWLANE_LOCAL_DATASET_DIR_NAME:-local256_200k}
RAWLANE_LOCAL_FIXED_EVAL_ROOT=${RAWLANE_LOCAL_FIXED_EVAL_ROOT:-/cache/jn/eval_sets/rawlane_local256_200k_from_local256_fixed1100_v1}
RAWLANE_LOCAL_CHECKPOINT_OBS_PATH=${RAWLANE_LOCAL_CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/29/3bf5a8001ec6433ca4ee973564c29976/output/ma-job-a782316a-32ec-4958-ae1f-44c69fdedd3f/checkpoint-12504/}
RAWLANE_LOCAL_CHECKPOINT_CACHE_ROOT=${RAWLANE_LOCAL_CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_local256_200k_checkpoint12504}

RAWLANE_CONTEXT_DATASET_OBS_PATH=${RAWLANE_CONTEXT_DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256_200k_rawlane/context512_roi256_200k.tar}
RAWLANE_CONTEXT_ARCHIVE_PATH=${RAWLANE_CONTEXT_ARCHIVE_PATH:-/cache/jn/data/rawlane_context512_roi256_200k.tar}
RAWLANE_CONTEXT_EXTRACT_ROOT=${RAWLANE_CONTEXT_EXTRACT_ROOT:-/cache/jn/data/rawlane_context512_roi256_200k_extract}
RAWLANE_CONTEXT_DATASET_DIR_NAME=${RAWLANE_CONTEXT_DATASET_DIR_NAME:-context512_roi256_200k}
RAWLANE_CONTEXT_FIXED_EVAL_ROOT=${RAWLANE_CONTEXT_FIXED_EVAL_ROOT:-/cache/jn/eval_sets/rawlane_context512_roi256_200k_from_local256_fixed1100_v1}
RAWLANE_CONTEXT_CHECKPOINT_OBS_PATH=${RAWLANE_CONTEXT_CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/30/ea77c85a9d54442b825b86cd7f547a26/output/ma-job-2e7c82dd-3a05-440b-b686-db5c3bcc2512/checkpoint-12504/}
RAWLANE_CONTEXT_CHECKPOINT_CACHE_ROOT=${RAWLANE_CONTEXT_CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/rawlane_context512_roi256_200k_checkpoint12504}

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

resolve_dataset_root() {
  local extract_root=$1
  local preferred_name=$2
  python - "${extract_root}" "${preferred_name}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
preferred_name = sys.argv[2]
candidates = []

def add(path):
    path = path.resolve()
    if path not in candidates and (path / "phase_a" / "eval.jsonl").is_file() and (path / "images").is_dir():
        candidates.append(path)

add(extract_root / preferred_name)
add(extract_root)
for eval_path in sorted(extract_root.rglob("phase_a/eval.jsonl")):
    add(eval_path.parent.parent)

if not candidates:
    raise SystemExit(f"Unable to locate dataset root below {extract_root}")
if len(candidates) > 1 and candidates[0].name != preferred_name:
    raise SystemExit(f"Ambiguous dataset roots below {extract_root}: {candidates}")
print(candidates[0])
PY
}

prepare_dataset() {
  local obs_path=$1
  local archive_path=$2
  local extract_root=$3
  local preferred_name=$4

  if [ ! -s "${archive_path}" ]; then
    copy_obs_file "${obs_path}" "${archive_path}" >&2
  else
    echo "[dataset] reuse archive: ${archive_path}" >&2
  fi
  mkdir -p "${extract_root}"
  if ! find "${extract_root}" -type f -path '*/phase_a/eval.jsonl' -print -quit | grep -q .; then
    echo "[dataset] extracting ${archive_path} -> ${extract_root}" >&2
    tar -xf "${archive_path}" -C "${extract_root}"
  else
    echo "[dataset] reuse extraction: ${extract_root}" >&2
  fi
  resolve_dataset_root "${extract_root}" "${preferred_name}"
}

validate_eval_set() {
  local split_root=$1
  local dataset_root=$2
  python - "${split_root}" "${dataset_root}" <<'PY'
import json
import sys
from pathlib import Path

split_root = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 200}
seen = set()

for difficulty, expected_count in expected.items():
    path = split_root / f"{difficulty}.jsonl"
    if not path.is_file():
        raise SystemExit(1)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != expected_count:
        raise SystemExit(1)
    for row in rows:
        sample_id = str(row.get("id", row.get("sample_id", ""))).strip()
        if not sample_id or sample_id in seen:
            raise SystemExit(1)
        seen.add(sample_id)
        image = row.get("image", row.get("images", ""))
        if isinstance(image, list):
            image = image[0] if image else ""
        image_path = Path(str(image))
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        if not image_path.is_file():
            raise SystemExit(1)

for name in ("all_selected.jsonl", "manifest.jsonl", "summary.json"):
    if not (split_root / name).is_file():
        raise SystemExit(1)
if len(seen) != 1100:
    raise SystemExit(1)
print(f"[fixed-eval] validated {len(seen)} samples: {split_root}", flush=True)
PY
}

build_reference_eval() {
  local dataset_root=$1
  local eval_jsonl=${dataset_root}/phase_a/eval.jsonl
  if validate_eval_set "${REFERENCE_FIXED_EVAL_ROOT}" "${dataset_root}"; then
    echo "[fixed-eval] reuse canonical local256 fixed-1100 set"
    return
  fi

  local build_root=${REFERENCE_FIXED_EVAL_ROOT}.building.$$
  rm -rf "${build_root}"
  mkdir -p "${build_root}"
  python scripts/tools/build_difficulty_eval_splits.py \
    --input-jsonl "${eval_jsonl}" \
    --output-dir "${build_root}" \
    --samples-per-difficulty 0 \
    --samples-per-difficulty-spec "${FIXED_EVAL_COUNTS}" \
    --difficulties easy medium hard very_hard \
    --seed "${FIXED_EVAL_SEED}" \
    --coord-mode auto \
    --coord-range 1000

  python - "${build_root}" "${eval_jsonl}" "${FIXED_EVAL_SEED}" "${FIXED_EVAL_COUNTS}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = [source, root / "all_selected.jsonl", root / "manifest.jsonl"]
files.extend(root / f"{name}.jsonl" for name in ("easy", "medium", "hard", "very_hard"))
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_jsonl": str(source),
    "seed": int(sys.argv[3]),
    "requested_counts": sys.argv[4],
    "total_samples": 1100,
    "sha256": {str(path): sha256(path) for path in files},
    "identity": "canonical Dataset V2 local256 fixed-1100 evaluation set",
}
(root / "fixed_eval_identity.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY
  rm -rf "${REFERENCE_FIXED_EVAL_ROOT}"
  mv "${build_root}" "${REFERENCE_FIXED_EVAL_ROOT}"
  validate_eval_set "${REFERENCE_FIXED_EVAL_ROOT}" "${dataset_root}"
}

map_eval_set() {
  local reference_root=$1
  local target_dataset_root=$2
  local output_root=$3
  local ground_truth_source=$4

  if validate_eval_set "${output_root}" "${target_dataset_root}"; then
    echo "[fixed-eval-remap] reuse ${output_root}"
    return
  fi

  local build_root=${output_root}.building.$$
  rm -rf "${build_root}"
  python scripts/tools/remap_fixed_eval_to_dataset.py \
    --reference-dir "${reference_root}" \
    --target-dataset-root "${target_dataset_root}" \
    --output-dir "${build_root}" \
    --target-phase phase_a \
    --scan-target-splits eval test \
    --allowed-target-splits eval test \
    --patch-size 256 \
    --ground-truth-source "${ground_truth_source}" \
    --require-all
  cp "${reference_root}/manifest.jsonl" "${build_root}/manifest.jsonl"
  cp "${reference_root}/summary.json" "${build_root}/summary.json"
  if [ -f "${reference_root}/fixed_eval_identity.json" ]; then
    cp "${reference_root}/fixed_eval_identity.json" "${build_root}/reference_fixed_eval_identity.json"
  fi
  rm -rf "${output_root}"
  mv "${build_root}" "${output_root}"
  validate_eval_set "${output_root}" "${target_dataset_root}"
}

mkdir -p "${COMPARE_ROOT}"
echo "============================================================"
echo "Raw-Lane 200k shared fixed-1100 comparison"
echo "Output root:        ${COMPARE_ROOT}"
echo "Visible NPUs:       ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Processes:          ${NPROC_PER_NODE}"
echo "Per-device batch:   ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Difficulty counts:  ${FIXED_EVAL_COUNTS}"
echo "Reference seed:     ${FIXED_EVAL_SEED}"
echo "============================================================"

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  copy_obs_parallel "${VISION_TOWER_OBS_PATH}" "${VISION_TOWER}"
fi
if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: DINOv2 download is incomplete: ${VISION_TOWER}" >&2
  exit 2
fi

REFERENCE_DATASET_ROOT=$(prepare_dataset \
  "${REFERENCE_DATASET_OBS_PATH}" \
  "${REFERENCE_ARCHIVE_PATH}" \
  "${REFERENCE_EXTRACT_ROOT}" \
  "${REFERENCE_DATASET_DIR_NAME}")
RAWLANE_LOCAL_DATASET_ROOT=$(prepare_dataset \
  "${RAWLANE_LOCAL_DATASET_OBS_PATH}" \
  "${RAWLANE_LOCAL_ARCHIVE_PATH}" \
  "${RAWLANE_LOCAL_EXTRACT_ROOT}" \
  "${RAWLANE_LOCAL_DATASET_DIR_NAME}")
RAWLANE_CONTEXT_DATASET_ROOT=$(prepare_dataset \
  "${RAWLANE_CONTEXT_DATASET_OBS_PATH}" \
  "${RAWLANE_CONTEXT_ARCHIVE_PATH}" \
  "${RAWLANE_CONTEXT_EXTRACT_ROOT}" \
  "${RAWLANE_CONTEXT_DATASET_DIR_NAME}")

build_reference_eval "${REFERENCE_DATASET_ROOT}"

# The local Raw-Lane target contributes its current image, prompt and GT.
map_eval_set \
  "${REFERENCE_FIXED_EVAL_ROOT}" \
  "${RAWLANE_LOCAL_DATASET_ROOT}" \
  "${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" \
  target

# Context uses its 512 image and ROI-aware prompt, but keeps byte-identical GT
# from the mapped Raw-Lane local256 set.
map_eval_set \
  "${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" \
  "${RAWLANE_CONTEXT_DATASET_ROOT}" \
  "${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}" \
  reference

python - "${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" "${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}" "${COMPARE_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

local_root = Path(sys.argv[1])
context_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
difficulties = ("easy", "medium", "hard", "very_hard")
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 200}

def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def record_id(record):
    return str(record.get("id", record.get("sample_id", ""))).strip()

def assistant_value(record):
    conversations = record.get("conversations") or []
    for message in reversed(conversations):
        if str(message.get("from", "")).lower() in {"gpt", "assistant"}:
            return message.get("value")
    raise ValueError(f"No assistant value for sample {record_id(record)}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

counts = {}
for difficulty in difficulties:
    local_path = local_root / f"{difficulty}.jsonl"
    context_path = context_root / f"{difficulty}.jsonl"
    local_rows = read_jsonl(local_path)
    context_rows = read_jsonl(context_path)
    if len(local_rows) != expected[difficulty] or len(context_rows) != expected[difficulty]:
        raise SystemExit(f"Unexpected {difficulty} counts: local={len(local_rows)} context={len(context_rows)}")
    for index, (local, context) in enumerate(zip(local_rows, context_rows)):
        local_id = record_id(local)
        context_id = record_id(context)
        if local_id != context_id:
            raise SystemExit(
                f"Sample identity mismatch at {difficulty}[{index}]: local={local_id!r} context={context_id!r}"
            )
        if assistant_value(local) != assistant_value(context):
            raise SystemExit(f"Ground-truth mismatch at {difficulty}[{index}] sample={local_id}")
    counts[difficulty] = len(local_rows)

payload = {
    "same_sample_ids": True,
    "same_bucket_order": True,
    "same_assistant_ground_truth": True,
    "counts": counts,
    "local_all_selected_sha256": sha256(local_root / "all_selected.jsonl"),
    "context_all_selected_sha256": sha256(context_root / "all_selected.jsonl"),
    "note": "File hashes differ because image paths and user prompts are view-specific; IDs, bucket order and assistant GT are equal.",
}
identity_path = output_root / "paired_eval_identity.json"
identity_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
PY

COMMON_EVAL_ENV=(
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
  ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}"
  NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${NPROC_PER_NODE}"
  PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}"
  VISION_TOWER="${VISION_TOWER}"
  FIXED_EVAL_COUNTS="${FIXED_EVAL_COUNTS}"
  FIXED_EVAL_SEED="${FIXED_EVAL_SEED}"
  REBUILD_FIXED_EVAL=False
  PATCH_SIZE=256
  VIS_LIMIT="${VIS_LIMIT}"
)

env "${COMMON_EVAL_ENV[@]}" \
  DATASET_OBS_PATH="${RAWLANE_LOCAL_DATASET_OBS_PATH}" \
  DATASET_ARCHIVE_PATH="${RAWLANE_LOCAL_ARCHIVE_PATH}" \
  DATASET_EXTRACT_ROOT="${RAWLANE_LOCAL_EXTRACT_ROOT}" \
  DATASET_ROOT="${RAWLANE_LOCAL_DATASET_ROOT}" \
  EVAL_SOURCE_JSONL="${RAWLANE_LOCAL_DATASET_ROOT}/phase_a/eval.jsonl" \
  FIXED_EVAL_ROOT="${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" \
  CHECKPOINT_NAME=checkpoint-12504 \
  CHECKPOINT_OBS_PATH="${RAWLANE_LOCAL_CHECKPOINT_OBS_PATH}" \
  CHECKPOINT_CACHE_ROOT="${RAWLANE_LOCAL_CHECKPOINT_CACHE_ROOT}" \
  OUTPUT_ROOT="${COMPARE_ROOT}/rawlane_local256_200k" \
  RUN_LABEL=rawlane_local256_200k_checkpoint12504_fixed1100 \
  bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

env "${COMMON_EVAL_ENV[@]}" \
  DATASET_OBS_PATH="${RAWLANE_CONTEXT_DATASET_OBS_PATH}" \
  DATASET_ARCHIVE_PATH="${RAWLANE_CONTEXT_ARCHIVE_PATH}" \
  DATASET_EXTRACT_ROOT="${RAWLANE_CONTEXT_EXTRACT_ROOT}" \
  DATASET_ROOT="${RAWLANE_CONTEXT_DATASET_ROOT}" \
  EVAL_SOURCE_JSONL="${RAWLANE_CONTEXT_DATASET_ROOT}/phase_a/eval.jsonl" \
  FIXED_EVAL_ROOT="${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}" \
  CHECKPOINT_NAME=checkpoint-12504 \
  CHECKPOINT_OBS_PATH="${RAWLANE_CONTEXT_CHECKPOINT_OBS_PATH}" \
  CHECKPOINT_CACHE_ROOT="${RAWLANE_CONTEXT_CHECKPOINT_CACHE_ROOT}" \
  OUTPUT_ROOT="${COMPARE_ROOT}/rawlane_context512_roi256_200k" \
  RUN_LABEL=rawlane_context512_roi256_200k_checkpoint12504_fixed1100 \
  bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

python - "${COMPARE_ROOT}" "${REFERENCE_FIXED_EVAL_ROOT}" "${RAWLANE_LOCAL_FIXED_EVAL_ROOT}" "${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}" <<'PY'
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
    "canonical_reference_eval_root": sys.argv[2],
    "rawlane_local_eval_root": sys.argv[3],
    "rawlane_context_eval_root": sys.argv[4],
    "same_sample_ids": True,
    "same_difficulty_buckets": True,
    "same_ground_truth": True,
    "sample_counts": {"easy": 300, "medium": 300, "hard": 300, "very_hard": 200},
    "local_input": "Raw-Lane 256x256 local patch",
    "context_input": "Raw-Lane 512x512 context with centered 256x256 target ROI",
}
output = root / "rawlane200k_shared_fixed1100_comparison.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[comparison] {output}")
PY

echo "============================================================"
echo "RAW-LANE 200K COMPARISON COMPLETE"
echo "Local256 results:   ${COMPARE_ROOT}/rawlane_local256_200k"
echo "Context results:    ${COMPARE_ROOT}/rawlane_context512_roi256_200k"
echo "Comparison JSON:    ${COMPARE_ROOT}/rawlane200k_shared_fixed1100_comparison.json"
echo "Pair identity:      ${COMPARE_ROOT}/paired_eval_identity.json"
echo "Canonical eval set: ${REFERENCE_FIXED_EVAL_ROOT}"
echo "Local mapped set:   ${RAWLANE_LOCAL_FIXED_EVAL_ROOT}"
echo "Context mapped set: ${RAWLANE_CONTEXT_FIXED_EVAL_ROOT}"
echo "============================================================"
