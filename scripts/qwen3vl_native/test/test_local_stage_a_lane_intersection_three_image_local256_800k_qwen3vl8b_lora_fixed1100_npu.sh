#!/usr/bin/env bash
set -euo pipefail

# Evaluate the released 800k three-image native-Qwen3-VL-8B LoRA checkpoint
# on one persistent 1100-sample local256 set. The model runs once on all 1100
# records; metrics are then split into easy/medium/hard/very_hard buckets.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

DEFAULT_CHECKPOINT_OBS_ROOT=obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/08/08/22c8ec9bf4754b5489d614a7a7d726f6/output/ma-job-94c3eef0-5be2-456d-8a8b-ed8e1943baa3
CHECKPOINT_OBS_LIST=${1:-${CHECKPOINT_OBS_LIST:-${DEFAULT_CHECKPOINT_OBS_ROOT}/checkpoint-36000/,${DEFAULT_CHECKPOINT_OBS_ROOT}/checkpoint-50000/}}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/native_qwen3vl_three_image_local256_800k}
REFRESH_CHECKPOINT=${REFRESH_CHECKPOINT:-False}

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-native-qwen3vl-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_native_qwen3vl_torch240.sh}
CREATE_ENV_IF_MISSING=${CREATE_ENV_IF_MISSING:-True}

MODEL_OBS_ROOT=${MODEL_OBS_ROOT:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
QWEN3VL_MODEL_NAME=${QWEN3VL_MODEL_NAME:-Qwen3-VL-8B-Instruct}
QWEN3VL_OBS_PATH=${QWEN3VL_OBS_PATH:-${MODEL_OBS_ROOT}/${QWEN3VL_MODEL_NAME}/}
QWEN3VL_PATH=${QWEN3VL_PATH:-/cache/jn/model/${QWEN3VL_MODEL_NAME}}
REFRESH_BASE_MODEL=${REFRESH_BASE_MODEL:-False}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_rawpos/local256_rawlane_pose_800k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/local256_rawlane_pose_800k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/local256_rawlane_pose_800k_fixed_eval_extract}
DATASET_TAR_MANIFEST=${DATASET_TAR_MANIFEST:-${DATASET_ARCHIVE_PATH}.members.txt}
DATASET_ROOT=${DATASET_ROOT:-}
EVAL_SOURCE_JSONL=${EVAL_SOURCE_JSONL:-}
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}
REFRESH_DATASET_ARCHIVE=${REFRESH_DATASET_ARCHIVE:-False}
REFRESH_DATASET_TAR_MANIFEST=${REFRESH_DATASET_TAR_MANIFEST:-False}
EXTRACT_FULL_DATASET=${EXTRACT_FULL_DATASET:-False}

FIXED_EVAL_ROOT=${FIXED_EVAL_ROOT:-/cache/jn/eval_sets/three_image_local256_800k_fixed1100_e300_m300_h300_vh200_seed42_v1}
FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}
FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}
REBUILD_FIXED_EVAL=${REBUILD_FIXED_EVAL:-False}

RUN_LABEL=${RUN_LABEL:-three_image_local256_800k_native_qwen3vl8b_lora_checkpoint36000_vs_50000_fixed1100}
RUN_ID=${RUN_ID:-${RUN_LABEL}_$(date -u +%Y%m%d_%H%M%S)}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/eval_runs/${RUN_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
COMPARISON_ROOT=${COMPARISON_ROOT:-${OUTPUT_ROOT}/comparison}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}
FIXED_EVAL_OBS_PATH=${FIXED_EVAL_OBS_PATH:-}

# Match the user's established Ascend inference default. Both values remain
# overridable without changing deterministic shard ownership.
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
LOAD_STAGGER_SECONDS=${LOAD_STAGGER_SECONDS:-10}
VIS_LIMIT=${VIS_LIMIT:-0}

EVAL_METER_PER_PIXEL=${EVAL_METER_PER_PIXEL:-0.2}
EVAL_BUFFER_SIZE=${EVAL_BUFFER_SIZE:-1.0}
EVAL_MATCH_THRESHOLD=${EVAL_MATCH_THRESHOLD:-0.33}
EVAL_INTERSECTION_IOU_THRESHOLD=${EVAL_INTERSECTION_IOU_THRESHOLD:-0.5}

DEFAULT_SYSTEM_PROMPT=$'You are a road-map reconstruction assistant designed to process BEV (Bird\'s Eye View) images generated from LiDAR data.\nPredict the complete road map from the current patch in the BEV image.\nReturn only valid JSON in the required schema.\nDo not output markdown fences or extra explanation.\nKeep all coordinates in the patch-local coordinate system.'
SYSTEM_PROMPT=${SYSTEM_PROMPT:-${DEFAULT_SYSTEM_PROMPT}}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

safe_source() {
  local source_path=$1
  set +u
  # shellcheck disable=SC1090
  source "${source_path}"
  set -u
}

download_obs_file() {
  local source=$1
  local destination=$2
  local refresh=$3
  mkdir -p "$(dirname "${destination}")"
  if ! is_true "${refresh}" && [ -s "${destination}" ]; then
    echo "[fixed1100] reuse file: ${destination}"
    return
  fi
  rm -f "${destination}"
  SOURCE="${source}" DESTINATION="${destination}" python - <<'PY'
import os
import moxing as mox

print(f"[fixed1100] download {os.environ['SOURCE']} -> {os.environ['DESTINATION']}", flush=True)
mox.file.copy(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
}

download_obs_directory() {
  local source=$1
  local destination=$2
  local sentinel=$3
  local refresh=$4
  if ! is_true "${refresh}" && [ -s "${destination}/${sentinel}" ]; then
    echo "[fixed1100] reuse directory: ${destination}"
    return
  fi
  if [ -e "${destination}" ]; then
    destination="${destination}.validated_$(date -u +%Y%m%d_%H%M%S)"
  fi
  SOURCE="${source}" DESTINATION="${destination}" python - <<'PY'
import os
import moxing as mox

print(f"[fixed1100] download {os.environ['SOURCE']} -> {os.environ['DESTINATION']}", flush=True)
mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
  printf '%s\n' "${destination}"
}

has_native_base_model() {
  local root=$1
  [ -s "${root}/config.json" ] && \
    { [ -s "${root}/preprocessor_config.json" ] || [ -s "${root}/processor_config.json" ]; } && \
    { [ -s "${root}/model.safetensors" ] || [ -s "${root}/model.safetensors.index.json" ] || \
      [ -s "${root}/pytorch_model.bin" ] || [ -s "${root}/pytorch_model.bin.index.json" ]; }
}

has_lora_checkpoint() {
  local root=$1
  [ -s "${root}/adapter_config.json" ] && \
    { [ -s "${root}/adapter_model.safetensors" ] || [ -s "${root}/adapter_model.bin" ]; }
}

safe_label() {
  python - "$1" <<'PY'
import re
import sys
value = sys.argv[1].strip().rstrip("/") or "checkpoint"
print(re.sub(r"[^A-Za-z0-9._-]+", "_", value.split("/")[-1]).strip("._-") or "checkpoint")
PY
}

read_checkpoint_list() {
  python - "$1" <<'PY'
import re
import sys
for item in re.split(r"[,;\n]+", sys.argv[1] or ""):
    item = item.strip()
    if item:
        print(item)
PY
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  if ! is_true "${CREATE_ENV_IF_MISSING}"; then
    echo "ERROR: native Qwen3-VL environment is missing: ${ACTIVATE_SCRIPT}" >&2
    exit 2
  fi
  echo "[fixed1100] creating isolated native-Qwen3-VL Torch-2.4 environment"
  ENV_DIR="${ENV_DIR}" RECREATE_ENV=False REQUIRE_NPU=True \
    bash scripts/npu/setup/create_mllm_native_qwen3vl_torch240_npu_env_from_infer.sh
fi
safe_source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  safe_source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}

python - <<'PY'
import json
import peft
import scipy
import shapely
import torch
import torch_npu
import transformers

versions = {
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "scipy": scipy.__version__,
    "shapely": shapely.__version__,
    "npu_available": bool(torch.npu.is_available()),
}
print(json.dumps(versions, indent=2))
if transformers.__version__ != "4.57.3":
    raise SystemExit(f"Expected transformers=4.57.3, got {transformers.__version__}")
if peft.__version__ != "0.18.0":
    raise SystemExit(f"Expected peft=0.18.0, got {peft.__version__}")
if not torch.npu.is_available():
    raise SystemExit("NPU is unavailable in the native-Qwen3-VL inference environment")
PY

mkdir -p "${DATASET_EXTRACT_ROOT}" "${CHECKPOINT_CACHE_ROOT}" "$(dirname "${QWEN3VL_PATH}")" \
  "$(dirname "${FIXED_EVAL_ROOT}")" "${RUN_WORK_ROOT}" "${OUTPUT_ROOT}" "${COMPARISON_ROOT}"

resolve_extracted_eval() {
  python - "${DATASET_EXTRACT_ROOT}" "${DATASET_ROOT}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
preferred = Path(sys.argv[2]).resolve() if sys.argv[2] else None
candidates = []
if preferred:
    preferred_eval = preferred / "phase_a" / "eval.jsonl"
    preferred_val = preferred / "phase_a" / "val.jsonl"
    if preferred_eval.is_file():
        print(preferred_eval)
        raise SystemExit(0)
    if preferred_val.is_file():
        print(preferred_val)
        raise SystemExit(0)
for name in ("eval.jsonl", "val.jsonl"):
    candidates = [
        path for path in extract_root.rglob(name)
        if path.parent.name == "phase_a" and "__MACOSX" not in path.parts
    ]
    if len(candidates) == 1:
        print(candidates[0])
        raise SystemExit(0)
    if len(candidates) > 1:
        preview = "\n".join(str(path) for path in candidates[:20])
        raise SystemExit(f"Unable to resolve one phase_a/{name}; candidates:\n{preview}")
raise SystemExit("Unable to resolve phase_a/eval.jsonl or phase_a/val.jsonl")
PY
}

ensure_dataset_archive() {
  if is_true "${REUSE_LOCAL_ASSETS}" && ! is_true "${REFRESH_DATASET_ARCHIVE}" && [ -s "${DATASET_ARCHIVE_PATH}" ]; then
    echo "[fixed1100] reuse dataset archive: ${DATASET_ARCHIVE_PATH}"
  else
    download_obs_file "${DATASET_OBS_PATH}" "${DATASET_ARCHIVE_PATH}" "${REFRESH_DATASET_ARCHIVE}"
  fi
}

if [ -z "${EVAL_SOURCE_JSONL}" ] || [ ! -s "${EVAL_SOURCE_JSONL}" ]; then
  EVAL_SOURCE_JSONL=$(resolve_extracted_eval 2>/dev/null || true)
fi

if [ -z "${EVAL_SOURCE_JSONL}" ] || [ ! -s "${EVAL_SOURCE_JSONL}" ]; then
  ensure_dataset_archive
  if is_true "${REFRESH_DATASET_TAR_MANIFEST}" || [ ! -s "${DATASET_TAR_MANIFEST}" ]; then
    echo "[fixed1100] indexing dataset TAR once: ${DATASET_ARCHIVE_PATH}"
    tar -tf "${DATASET_ARCHIVE_PATH}" > "${DATASET_TAR_MANIFEST}"
  fi
  EVAL_ARCHIVE_MEMBER=$(python - "${DATASET_TAR_MANIFEST}" <<'PY'
import sys
from pathlib import PurePosixPath

members = [line.strip() for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
for filename in ("eval.jsonl", "val.jsonl"):
    candidates = []
    for member in members:
        path = PurePosixPath(member.removeprefix("./"))
        if len(path.parts) >= 2 and path.parts[-2:] == ("phase_a", filename):
            if "__MACOSX" not in path.parts:
                candidates.append(member)
    if len(candidates) == 1:
        print(candidates[0])
        raise SystemExit(0)
    if len(candidates) > 1:
        preview = "\n".join(candidates[:20])
        raise SystemExit(f"Expected one phase_a/{filename} in TAR, found {len(candidates)}:\n{preview}")
raise SystemExit("No phase_a/eval.jsonl or phase_a/val.jsonl found in TAR")
PY
  )
  echo "[fixed1100] extracting evaluation manifest: ${EVAL_ARCHIVE_MEMBER}"
  tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}" "${EVAL_ARCHIVE_MEMBER}"
  EVAL_SOURCE_JSONL=$(resolve_extracted_eval)
fi

DATASET_ROOT=$(python - "${EVAL_SOURCE_JSONL}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
if path.parent.name != "phase_a":
    raise SystemExit(f"Evaluation JSONL is not below phase_a: {path}")
print(path.parent.parent)
PY
)

write_fixed_identity() {
  python - "${FIXED_EVAL_ROOT}" "${EVAL_SOURCE_JSONL}" "${FIXED_EVAL_SEED}" "${FIXED_EVAL_COUNTS}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = [root / f"{name}.jsonl" for name in ("easy", "medium", "hard", "very_hard")]
files.extend([root / "all_selected.jsonl", root / "manifest.jsonl"])
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_jsonl": str(source.resolve()),
    "source_jsonl_sha256": sha256(source),
    "seed": int(sys.argv[3]),
    "requested_counts": sys.argv[4],
    "total_samples": 1100,
    "split_sha256": {path.name: sha256(path) for path in files},
    "reuse_policy": "Reuse these exact records and hashes for every comparison checkpoint.",
}
(root / "fixed_eval_identity.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

validate_fixed_structure() {
  python - "${FIXED_EVAL_ROOT}" "${EVAL_SOURCE_JSONL}" "${FIXED_EVAL_COUNTS}" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
expected = {}
for item in sys.argv[3].split(","):
    name, count = item.split("=", 1)
    expected[name] = int(count)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

identity_path = root / "fixed_eval_identity.json"
if not identity_path.is_file():
    raise SystemExit(1)
identity = json.loads(identity_path.read_text(encoding="utf-8"))
if identity.get("source_jsonl_sha256") != sha256(source):
    raise SystemExit("Fixed set source hash does not match the current evaluation JSONL")
if int(identity.get("total_samples", -1)) != sum(expected.values()):
    raise SystemExit("Fixed-set identity has the wrong total")

seen = set()
bucket_ids = []
for difficulty, expected_count in expected.items():
    path = root / f"{difficulty}.jsonl"
    if not path.is_file():
        raise SystemExit(1)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != expected_count:
        raise SystemExit(f"Expected {expected_count} records in {path}, found {len(records)}")
    for record in records:
        sample_id = str(record.get("id") or record.get("sample_id") or "").strip()
        if not sample_id or sample_id in seen:
            raise SystemExit(f"Invalid or duplicate sample id: {sample_id!r}")
        seen.add(sample_id)
        bucket_ids.append(sample_id)
        images = record.get("images")
        conversations = record.get("conversations")
        if not isinstance(images, list) or len(images) != 3:
            raise SystemExit(f"Expected three ordered images for {sample_id}: {images!r}")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise SystemExit(f"Missing prompt/ground truth for {sample_id}")
        prompt = str(conversations[0].get("value", ""))
        if prompt.count("<image>") != 3:
            raise SystemExit(f"Expected three image tokens for {sample_id}")
        target = json.loads(str(conversations[1].get("value", "")))
        if not isinstance(target, dict) or not isinstance(target.get("lines"), list):
            raise SystemExit(f"Invalid structured target for {sample_id}")

all_path = root / "all_selected.jsonl"
all_records = [json.loads(line) for line in all_path.read_text(encoding="utf-8").splitlines() if line.strip()]
all_ids = [str(record.get("id") or record.get("sample_id") or "").strip() for record in all_records]
if len(all_ids) != sum(expected.values()) or Counter(all_ids) != Counter(bucket_ids):
    raise SystemExit("all_selected.jsonl does not exactly match the four difficulty buckets")
for name, expected_hash in identity.get("split_sha256", {}).items():
    path = root / name
    if not path.is_file() or sha256(path) != expected_hash:
        raise SystemExit(f"Fixed split hash mismatch: {path}")
print(f"[fixed1100] validated immutable split: {root} ({len(all_ids)} records)")
PY
}

if ! is_true "${REBUILD_FIXED_EVAL}" && validate_fixed_structure; then
  echo "[fixed1100] reuse exact saved evaluation set"
else
  if [ -e "${FIXED_EVAL_ROOT}" ] && [ -n "$(find "${FIXED_EVAL_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] && ! is_true "${REBUILD_FIXED_EVAL}"; then
    echo "ERROR: fixed evaluation root exists but failed validation: ${FIXED_EVAL_ROOT}" >&2
    echo "Set REBUILD_FIXED_EVAL=True only when replacing this evaluation-set identity intentionally." >&2
    exit 2
  fi
  BUILD_ROOT="${FIXED_EVAL_ROOT}.building.$$"
  rm -rf "${BUILD_ROOT}"
  mkdir -p "${BUILD_ROOT}"
  echo "[fixed1100] selecting deterministic 300/300/300/200 samples from ${EVAL_SOURCE_JSONL}"
  python scripts/tools/build_difficulty_eval_splits.py \
    --input-jsonl "${EVAL_SOURCE_JSONL}" \
    --output-dir "${BUILD_ROOT}" \
    --samples-per-difficulty 0 \
    --samples-per-difficulty-spec "${FIXED_EVAL_COUNTS}" \
    --difficulties easy medium hard very_hard \
    --seed "${FIXED_EVAL_SEED}" \
    --coord-mode auto \
    --coord-range 1000
  rm -rf "${FIXED_EVAL_ROOT}"
  mv "${BUILD_ROOT}" "${FIXED_EVAL_ROOT}"
  write_fixed_identity
  validate_fixed_structure
fi

if is_true "${EXTRACT_FULL_DATASET}"; then
  ensure_dataset_archive
  echo "[fixed1100] extracting the complete dataset by explicit request"
  tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}"
else
  MISSING_MEMBERS_FILE="${RUN_WORK_ROOT}/missing_fixed1100_image_members.txt"
  ASSET_REPORT="${RUN_WORK_ROOT}/fixed1100_asset_report.json"
  set +e
  python - "${FIXED_EVAL_ROOT}/all_selected.jsonl" "${DATASET_ROOT}" \
    "${DATASET_EXTRACT_ROOT}" "${DATASET_TAR_MANIFEST}" "${MISSING_MEMBERS_FILE}" "${ASSET_REPORT}" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath

selected_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2]).resolve()
extract_root = Path(sys.argv[3]).resolve()
manifest_path = Path(sys.argv[4])
missing_output = Path(sys.argv[5])
report_path = Path(sys.argv[6])
records = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
image_values = []
for record in records:
    image_values.extend(str(value) for value in record["images"])
image_values = list(dict.fromkeys(image_values))
missing_paths = []
for value in image_values:
    path = Path(value)
    target = path if path.is_absolute() else dataset_root / path
    if not target.is_file():
        missing_paths.append((value, target))

report = {
    "records": len(records),
    "unique_image_assets": len(image_values),
    "existing_image_assets": len(image_values) - len(missing_paths),
    "missing_image_assets": len(missing_paths),
}
if not missing_paths:
    missing_output.write_text("", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0)
if not manifest_path.is_file():
    raise SystemExit(3)
try:
    dataset_prefix = dataset_root.relative_to(extract_root).as_posix().strip("/")
except ValueError as exc:
    raise SystemExit(f"Missing fixed-eval images cannot be extracted because dataset root is outside extract root: {exc}")
members = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
by_normalized = {member.removeprefix("./").rstrip("/"): member for member in members}
requested_members = []
unresolved = []
for value, target in missing_paths:
    path = Path(value)
    if path.is_absolute():
        unresolved.append(str(target))
        continue
    normalized = str(PurePosixPath(dataset_prefix) / PurePosixPath(value.replace("\\", "/"))).lstrip("/")
    member = by_normalized.get(normalized)
    if member is None:
        unresolved.append(normalized)
    else:
        requested_members.append(member)
if unresolved:
    raise SystemExit(f"Unable to resolve {len(unresolved)} fixed-eval images in TAR; examples={unresolved[:10]}")
missing_output.write_text("\n".join(dict.fromkeys(requested_members)) + "\n", encoding="utf-8")
report["tar_members_to_extract"] = len(set(requested_members))
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
  asset_status=$?
  set -e
  if [ "${asset_status}" -ne 0 ]; then
    status=${asset_status}
    if [ "${status}" -ne 3 ]; then
      exit "${status}"
    fi
    ensure_dataset_archive
    if is_true "${REFRESH_DATASET_TAR_MANIFEST}" || [ ! -s "${DATASET_TAR_MANIFEST}" ]; then
      echo "[fixed1100] indexing dataset TAR for selected image extraction"
      tar -tf "${DATASET_ARCHIVE_PATH}" > "${DATASET_TAR_MANIFEST}"
    fi
    python - "${FIXED_EVAL_ROOT}/all_selected.jsonl" "${DATASET_ROOT}" \
      "${DATASET_EXTRACT_ROOT}" "${DATASET_TAR_MANIFEST}" "${MISSING_MEMBERS_FILE}" "${ASSET_REPORT}" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath

selected_path, dataset_root, extract_root, manifest_path, output_path, report_path = map(Path, sys.argv[1:])
dataset_root = dataset_root.resolve()
extract_root = extract_root.resolve()
records = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
values = list(dict.fromkeys(str(value) for record in records for value in record["images"]))
prefix = dataset_root.relative_to(extract_root).as_posix().strip("/")
members = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
lookup = {member.removeprefix("./").rstrip("/"): member for member in members}
requested = []
unresolved = []
for value in values:
    path = Path(value)
    target = path if path.is_absolute() else dataset_root / path
    if target.is_file():
        continue
    if path.is_absolute():
        unresolved.append(str(path))
        continue
    normalized = str(PurePosixPath(prefix) / PurePosixPath(value.replace("\\", "/"))).lstrip("/")
    member = lookup.get(normalized)
    (requested if member else unresolved).append(member or normalized)
if unresolved:
    raise SystemExit(f"Unable to resolve {len(unresolved)} fixed-eval images in TAR; examples={unresolved[:10]}")
requested = list(dict.fromkeys(requested))
output_path.write_text("\n".join(requested) + ("\n" if requested else ""), encoding="utf-8")
report = {"records": len(records), "unique_image_assets": len(values), "tar_members_to_extract": len(requested)}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
  fi
  if [ -s "${MISSING_MEMBERS_FILE}" ]; then
    ensure_dataset_archive
    echo "[fixed1100] extracting only selected three-image assets"
    tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}" -T "${MISSING_MEMBERS_FILE}"
  fi
fi

python - "${FIXED_EVAL_ROOT}" "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

split_root = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
missing = []
checked = 0
for name in ("easy", "medium", "hard", "very_hard"):
    path = split_root / f"{name}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for value in record["images"]:
            image = Path(str(value))
            if not image.is_absolute():
                image = dataset_root / image
            checked += 1
            if not image.is_file():
                missing.append(str(image))
if missing:
    raise FileNotFoundError(f"Missing {len(missing)} fixed-eval image assets; examples={missing[:10]}")
print(f"[fixed1100] image preflight passed: {checked} references")
PY

if is_true "${REFRESH_BASE_MODEL}" || ! has_native_base_model "${QWEN3VL_PATH}"; then
  QWEN3VL_PATH=$(download_obs_directory "${QWEN3VL_OBS_PATH}" "${QWEN3VL_PATH}" config.json True | tail -n 1)
else
  echo "[fixed1100] reuse native base model: ${QWEN3VL_PATH}"
fi

CHECKPOINT_OBS_ITEMS=()
CHECKPOINT_LABELS=()
CHECKPOINT_DIRS=()
while IFS= read -r checkpoint_obs; do
  label=$(safe_label "${checkpoint_obs}")
  checkpoint_dir="${CHECKPOINT_CACHE_ROOT}/${label}"
  if is_true "${REFRESH_CHECKPOINT}" || ! has_lora_checkpoint "${checkpoint_dir}"; then
    checkpoint_dir=$(download_obs_directory "${checkpoint_obs}" "${checkpoint_dir}" adapter_config.json True | tail -n 1)
  else
    echo "[fixed1100] reuse LoRA checkpoint: ${checkpoint_dir}"
  fi
  CHECKPOINT_OBS_ITEMS+=("${checkpoint_obs}")
  CHECKPOINT_LABELS+=("${label}")
  CHECKPOINT_DIRS+=("${checkpoint_dir}")
done < <(read_checkpoint_list "${CHECKPOINT_OBS_LIST}")
if [ "${#CHECKPOINT_DIRS[@]}" -lt 1 ]; then
  echo "ERROR: CHECKPOINT_OBS_LIST did not contain a checkpoint" >&2
  exit 2
fi
if [ "${#CHECKPOINT_DIRS[@]}" -ne "$(printf '%s\n' "${CHECKPOINT_LABELS[@]}" | sort -u | wc -l)" ]; then
  echo "ERROR: checkpoint labels must be unique: ${CHECKPOINT_LABELS[*]}" >&2
  exit 2
fi

for index in "${!CHECKPOINT_DIRS[@]}"; do
  python - "${CHECKPOINT_DIRS[$index]}" "${QWEN3VL_PATH}" "${CHECKPOINT_LABELS[$index]}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
base = Path(sys.argv[2])
expected_label = sys.argv[3]
if checkpoint.name != expected_label:
    raise ValueError(f"Checkpoint cache label mismatch: expected {expected_label}, got {checkpoint.name}")
adapter = checkpoint / "adapter_config.json"
weights = [checkpoint / "adapter_model.safetensors", checkpoint / "adapter_model.bin"]
base_weights = [
    base / "model.safetensors", base / "model.safetensors.index.json",
    base / "pytorch_model.bin", base / "pytorch_model.bin.index.json",
]
processor = [base / "preprocessor_config.json", base / "processor_config.json"]
if not adapter.is_file() or not any(path.is_file() and path.stat().st_size for path in weights):
    raise FileNotFoundError(f"Incomplete LoRA checkpoint: {checkpoint}")
if not (base / "config.json").is_file() or not any(path.is_file() and path.stat().st_size for path in base_weights):
    raise FileNotFoundError(f"Incomplete native Qwen3-VL base: {base}")
if not any(path.is_file() for path in processor):
    raise FileNotFoundError(f"Native processor config is missing: {base}")
payload = json.loads(adapter.read_text(encoding="utf-8"))
print(json.dumps({
    "checkpoint": str(checkpoint), "base": str(base),
    "peft_type": payload.get("peft_type"),
    "base_model_name_or_path": payload.get("base_model_name_or_path"),
}, indent=2))
PY
done

IFS=',' read -r -a DEVICE_IDS <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [ "${#DEVICE_IDS[@]}" -ne "${NPROC_PER_NODE}" ]; then
  echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE}, visible devices=${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [ "${PER_DEVICE_INFER_BATCH_SIZE}" -lt 1 ]; then
  echo "ERROR: PER_DEVICE_INFER_BATCH_SIZE must be >= 1" >&2
  exit 2
fi

echo "============================================================"
echo "NATIVE QWEN3-VL-8B THREE-IMAGE FIXED-1100 PATCH EVAL"
echo "Checkpoint OBS:    ${CHECKPOINT_OBS_ITEMS[*]}"
echo "Checkpoints:       ${CHECKPOINT_DIRS[*]}"
echo "Native base:       ${QWEN3VL_PATH}"
echo "Dataset:           ${DATASET_ROOT}"
echo "Source eval JSONL: ${EVAL_SOURCE_JSONL}"
echo "Fixed eval root:   ${FIXED_EVAL_ROOT}"
echo "Counts/seed:       ${FIXED_EVAL_COUNTS} / ${FIXED_EVAL_SEED}"
echo "Inputs:            3 (clean BEV, Raw-Lane, Pose)"
echo "Visible NPUs:      ${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPU processes:     ${NPROC_PER_NODE}"
echo "Per-device batch:  ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Generation cap:    ${MAX_NEW_TOKENS}"
echo "Output:            ${OUTPUT_ROOT}"
echo "============================================================"

SHARED_SHARD_JSONL_ROOT="${RUN_WORK_ROOT}/fixed1100_shards"
ACTIVE_INFER_JSONL="${SHARED_SHARD_JSONL_ROOT}/selected.jsonl"
rm -rf "${SHARED_SHARD_JSONL_ROOT}"
mkdir -p "${SHARED_SHARD_JSONL_ROOT}"
python scripts/tools/split_jsonl_for_inference.py \
  --input-jsonl "${FIXED_EVAL_ROOT}/all_selected.jsonl" \
  --output-root "${SHARED_SHARD_JSONL_ROOT}" \
  --num-shards "${NPROC_PER_NODE}" \
  --num-samples 0

for checkpoint_index in "${!CHECKPOINT_DIRS[@]}"; do
  CHECKPOINT_DIR="${CHECKPOINT_DIRS[$checkpoint_index]}"
  CHECKPOINT_LABEL="${CHECKPOINT_LABELS[$checkpoint_index]}"
  CHECKPOINT_OUTPUT_ROOT="${OUTPUT_ROOT}/${CHECKPOINT_LABEL}"
  SHARD_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}/inference/shards"
  INFERENCE_ROOT="${CHECKPOINT_OUTPUT_ROOT}/inference"
  RAW_RESULT_DIR="${INFERENCE_ROOT}/json"
  METRICS_ROOT="${CHECKPOINT_OUTPUT_ROOT}/by_difficulty"

  echo "[fixed1100] evaluating ${CHECKPOINT_LABEL}: ${CHECKPOINT_DIR}"
  rm -rf "${SHARD_OUTPUT_ROOT}" "${RAW_RESULT_DIR}" "${METRICS_ROOT}"
  mkdir -p "${SHARD_OUTPUT_ROOT}" "${INFERENCE_ROOT}/logs" "${METRICS_ROOT}"

  pids=()
  for rank in "${!DEVICE_IDS[@]}"; do
    device=$(echo "${DEVICE_IDS[$rank]}" | xargs)
    shard_name=$(printf 'shard_%05d' "${rank}")
    shard_jsonl="${SHARED_SHARD_JSONL_ROOT}/${shard_name}.jsonl"
    shard_output="${SHARD_OUTPUT_ROOT}/${shard_name}"
    shard_log="${INFERENCE_ROOT}/logs/${shard_name}.log"
    mkdir -p "${shard_output}"
    (
      export ASCEND_RT_VISIBLE_DEVICES="${device}"
      export ASCEND_VISIBLE_DEVICES="${device}"
      export NPU_VISIBLE_DEVICES="${device}"
      python -m mllm.native_qwen3vl.infer \
        --model-name-or-path "${CHECKPOINT_DIR}" \
        --model-base "${QWEN3VL_PATH}" \
        --test-json "${shard_jsonl}" \
        --image-folder "${DATASET_ROOT}" \
        --output-dir "${shard_output}" \
        --phase phase_a \
        --map-task lane_intersection \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature 0.0 \
        --coord-mode auto \
        --coord-range 1000 \
        --default-patch-size 256 \
        --per-device-infer-batch-size "${PER_DEVICE_INFER_BATCH_SIZE}" \
        --device npu:0 \
        --bf16 \
        --include-intersections \
        --system-prompt "${SYSTEM_PROMPT}" \
        --skip-eval \
        --skip-visualize
    ) >"${shard_log}" 2>&1 &
    pids+=("$!")
    echo "[fixed1100] ${CHECKPOINT_LABEL} rank=${rank} physical_npu=${device} pid=$! log=${shard_log}"
    if [ "${LOAD_STAGGER_SECONDS}" -gt 0 ] && [ "${rank}" -lt "$((NPROC_PER_NODE - 1))" ]; then
      sleep "${LOAD_STAGGER_SECONDS}"
    fi
  done

  failed=0
  for rank in "${!pids[@]}"; do
    if ! wait "${pids[$rank]}"; then
      echo "ERROR: ${CHECKPOINT_LABEL} native inference shard ${rank} failed; tail follows" >&2
      tail -n 120 "${INFERENCE_ROOT}/logs/$(printf 'shard_%05d' "${rank}").log" >&2 || true
      failed=1
    fi
  done
  if [ "${failed}" -ne 0 ]; then
    exit 1
  fi

  for rank in "${!DEVICE_IDS[@]}"; do
    shard_log="${INFERENCE_ROOT}/logs/$(printf 'shard_%05d' "${rank}").log"
    throughput_line=$(grep 'DI_throughput:' "${shard_log}" | tail -n 1 || true)
    if [ -z "${throughput_line}" ]; then
      echo "ERROR: ${CHECKPOINT_LABEL} shard ${rank} did not report DI_throughput: ${shard_log}" >&2
      exit 1
    fi
    echo "[fixed1100] ${CHECKPOINT_LABEL} rank=${rank} ${throughput_line}"
  done

  python scripts/tools/merge_native_qwen3vl_inference_shards.py \
    --infer-jsonl "${ACTIVE_INFER_JSONL}" \
    --shard-root "${SHARD_OUTPUT_ROOT}" \
    --output-dir "${INFERENCE_ROOT}" \
    --prediction-dir "${RAW_RESULT_DIR}" \
    --reset

  SPLIT_VIS_ARGS=()
  if [ "${VIS_LIMIT}" -gt 0 ]; then
    SPLIT_VIS_ARGS=(
      --image-folder "${DATASET_ROOT}"
      --visualize-max-samples "${VIS_LIMIT}"
    )
  fi

  python scripts/tools/split_single_pass_eval_by_difficulty.py \
    --summary-json "${INFERENCE_ROOT}/summary.json" \
    --split-root "${FIXED_EVAL_ROOT}" \
    --output-root "${METRICS_ROOT}" \
    --expected-counts "${FIXED_EVAL_COUNTS}" \
    --meter-per-pixel "${EVAL_METER_PER_PIXEL}" \
    --buffer-size "${EVAL_BUFFER_SIZE}" \
    --match-threshold "${EVAL_MATCH_THRESHOLD}" \
    --intersection-iou-threshold "${EVAL_INTERSECTION_IOU_THRESHOLD}" \
    --map-task lane_intersection \
    "${SPLIT_VIS_ARGS[@]}"
done

COMPARE_ARGS=()
for checkpoint_index in "${!CHECKPOINT_LABELS[@]}"; do
  label="${CHECKPOINT_LABELS[$checkpoint_index]}"
  COMPARE_ARGS+=(--checkpoint "${label}=${OUTPUT_ROOT}/${label}/by_difficulty")
done
python scripts/tools/compare_fixed1100_patch_metrics.py \
  "${COMPARE_ARGS[@]}" \
  --output-json "${COMPARISON_ROOT}/comparison.json" \
  --output-markdown "${COMPARISON_ROOT}/comparison.md"

if [ -n "${FIXED_EVAL_OBS_PATH}" ]; then
  SOURCE="${FIXED_EVAL_ROOT}" DESTINATION="${FIXED_EVAL_OBS_PATH}" python - <<'PY'
import os
import moxing as mox
mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
fi
if [ -n "${RESULT_OBS_PATH}" ]; then
  SOURCE="${OUTPUT_ROOT}" DESTINATION="${RESULT_OBS_PATH}" python - <<'PY'
import os
import moxing as mox
mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
fi

echo "============================================================"
echo "THREE-IMAGE FIXED-1100 CHECKPOINT COMPARISON COMPLETE"
echo "Fixed eval identity: ${FIXED_EVAL_ROOT}/fixed_eval_identity.json"
echo "Checkpoint outputs:  ${OUTPUT_ROOT}/{${CHECKPOINT_LABELS[*]}}"
echo "Comparison JSON:     ${COMPARISON_ROOT}/comparison.json"
echo "Comparison report:   ${COMPARISON_ROOT}/comparison.md"
if [ "${VIS_LIMIT}" -gt 0 ]; then
  echo "Visualizations:      each checkpoint/by_difficulty/{easy,medium,hard,very_hard,all_selected}/viz"
fi
if [ -n "${RESULT_OBS_PATH}" ]; then
  echo "Result OBS:          ${RESULT_OBS_PATH}"
fi
echo "============================================================"
