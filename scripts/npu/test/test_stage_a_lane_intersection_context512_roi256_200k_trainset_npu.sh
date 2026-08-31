#!/usr/bin/env bash
set -euo pipefail

# DI / Ascend inference for the context512_roi256 200k training subset.
# Results are written as resumable JSONL streams, validated, and uploaded.

echo "[di-entry] reached context512_roi256 200k train-set inference launcher"
echo "[di-entry] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"
echo "DI_throughput: 0.00 samples/s/npu"

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}
safe_source() {
  local path="$1"
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

# DI supplies the pinned torch/torch_npu environment.  No package installation
# is performed here; ACTIVATE_SCRIPT is only for local reproduction.
if [ -n "${ACTIVATE_SCRIPT:-}" ]; then
  [ -f "${ACTIVATE_SCRIPT}" ] || { echo "ERROR: missing ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT}" >&2; exit 2; }
  safe_source "${ACTIVATE_SCRIPT}"
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}

RUN_ID=${RUN_ID:-context512_roi256_200k_trainset_infer_$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/datasets/context512_roi256_550k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/datasets/context512_roi256_550k_extract}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-context512_roi256}
DATASET_ROOT=${DATASET_ROOT:-}
IMAGE_FOLDER=${IMAGE_FOLDER:-}
TRAIN_JSON=${TRAIN_JSON:-}

# These match the historical 200k context512/ROI256 training recipe.
USE_STRATIFIED_SUBSET=${USE_STRATIFIED_SUBSET:-True}
SUBSET_TARGET_SAMPLES=${SUBSET_TARGET_SAMPLES:-200000}
SUBSET_DIFFICULTY_RATIOS=${SUBSET_DIFFICULTY_RATIOS:-easy=0.30,medium=0.3560290909,hard=0.2439709091,very_hard=0.10}
SUBSET_SHORTAGE_POLICY=${SUBSET_SHORTAGE_POLICY:-redistribute}
SUBSET_SHORTAGE_FILL_BUCKETS=${SUBSET_SHORTAGE_FILL_BUCKETS:-medium,hard}
SUBSET_SEED=${SUBSET_SEED:-42}
SUBSET_PROGRESS_EVERY=${SUBSET_PROGRESS_EVERY:-50000}
SUBSET_REUSE=${SUBSET_REUSE:-True}
EXPECTED_TRAIN_SAMPLES=${EXPECTED_TRAIN_SAMPLES:-200000}
SUBSET_SUMMARY=${SUBSET_SUMMARY:-}

MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
QWEN_MODEL_NAME=${QWEN_MODEL_NAME:-CapRL-Qwen3VL-4B}
VISION_TOWER_NAME=${VISION_TOWER_NAME:-facebook_dinov2-large}
QWEN_MODEL_OBS_PATH=${QWEN_MODEL_OBS_PATH:-${MODEL_OBS_PATH}/${QWEN_MODEL_NAME}}
DINOV2_MODEL_OBS_PATH=${DINOV2_MODEL_OBS_PATH:-${MODEL_OBS_PATH}/${VISION_TOWER_NAME}}
QWEN_PATH=${QWEN_PATH:-${OBS_CACHE}/checkpoints/${QWEN_MODEL_NAME}}
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}

# Default is the previously recorded 200k checkpoint-12504. Override it for
# another result; the resolved layout is saved in checkpoint_inventory.json.
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/4f735c63da7a4f86829b26246143e219/output/ma-job-81341482-55b8-4c28-887b-0e4166776561/checkpoint-12504/}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/checkpoints/context512_roi256_200k_infer}

LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-${OBS_CACHE}/outputs/${RUN_ID}}
PREDICTION_JSONL=${PREDICTION_JSONL:-${LOCAL_OUTPUT_ROOT}/train_predictions.jsonl}
INFERENCE_CONFIG_PATH=${INFERENCE_CONFIG_PATH:-${LOCAL_OUTPUT_ROOT}/inference_config.json}
RESULT_OBS_ROOT=${RESULT_OBS_ROOT:-${OUTPUT_URL:-}}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}
[ -z "${RESULT_OBS_PATH}" ] || RESULT_OBS_ROOT=${RESULT_OBS_PATH}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-True}
KEEP_RANK_JSONL=${KEEP_RANK_JSONL:-True}
RESUME_INFERENCE=${RESUME_INFERENCE:-True}
STREAM_FSYNC_EVERY=${STREAM_FSYNC_EVERY:-100}
INFER_PROGRESS_EVERY=${INFER_PROGRESS_EVERY:-1000}
INFER_QUIET_SAMPLES=${INFER_QUIET_SAMPLES:-True}
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
MAP_TASK=${MAP_TASK:-lane_intersection}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}

if [ -n "${MA_VJ_NAME:-}" ] || [ -n "${MA_NUM_HOSTS:-}" ]; then
  NNODES=${NNODES:-${MA_NUM_HOSTS:-1}}
  NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
  NPROC_PER_NODE=${NPROC_PER_NODE:-${MA_NUM_GPUS:-8}}
else
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  if [ -n "${VC_WORKER_HOSTS:-}" ]; then MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"; else MASTER_ADDR=127.0.0.1; fi
fi
if [ -z "${MASTER_PORT:-}" ]; then
  if [ "${NNODES}" -eq 1 ]; then
    MASTER_PORT=$(python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)
  else
    MASTER_PORT=6060
  fi
fi
TOTAL_DEVICES=$((NNODES * NPROC_PER_NODE))
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-context512_roi256_200k_infer_${RUN_ID}}
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ] && [ "${NPROC_PER_NODE}" -le 8 ]; then
  ASCEND_RT_VISIBLE_DEVICES=$(python - "${NPROC_PER_NODE}" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}

mkdir -p "${LOCAL_OUTPUT_ROOT}" "${CHECKPOINT_DOWNLOAD_ROOT}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"

copy_parallel() {
  local source="$1" target="$2" threads="${3:-64}"
  SOURCE="${source}" TARGET="${target}" THREADS="${threads}" python - <<'PY'
import os
import moxing as mox
print(f"[obs] copy {os.environ['SOURCE']} -> {os.environ['TARGET']} threads={os.environ['THREADS']}", flush=True)
mox.file.copy_parallel(os.environ["SOURCE"], os.environ["TARGET"], threads=int(os.environ["THREADS"]))
PY
}
archive_is_valid() {
  python - "$1" <<'PY'
import sys
import tarfile
import zipfile
from pathlib import Path
p = Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p)) else 1)
PY
}
model_asset_is_complete() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    index = root / name
    if index.is_file():
        names = set((json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}).values())
        raise SystemExit(0 if names and all((root / n).is_file() for n in names) else 1)
raise SystemExit(0 if any((root / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")) else 1)
PY
}
count_jsonl() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
with Path(sys.argv[1]).open("r", encoding="utf-8-sig") as handle:
    print(sum(1 for line in handle if line.strip()))
PY
}
sha256_file() {
  python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path
digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

python - <<'PY'
import moxing
print(f"[di-infer] moxing={moxing.__file__}", flush=True)
PY

# Download and extract the 550k source only when no explicit local root was given.
if [ -z "${DATASET_ROOT}" ]; then
  if is_true "${REUSE_LOCAL_ASSETS}" && archive_is_valid "${DATASET_ARCHIVE_PATH}"; then
    echo "[dataset] reuse archive: ${DATASET_ARCHIVE_PATH}"
  else
    rm -f "${DATASET_ARCHIVE_PATH}"
    echo "[dataset] downloading ${DATASET_OBS_PATH} -> ${DATASET_ARCHIVE_PATH}"
    copy_parallel "${DATASET_OBS_PATH}" "${DATASET_ARCHIVE_PATH}" 128
  fi
  archive_is_valid "${DATASET_ARCHIVE_PATH}" || { echo "ERROR: invalid dataset archive" >&2; exit 2; }
  DATASET_MARKER="${DATASET_EXTRACT_ROOT}/.extract_complete"
  if is_true "${REUSE_LOCAL_ASSETS}" && [ -f "${DATASET_MARKER}" ]; then
    echo "[dataset] reuse extracted root: ${DATASET_EXTRACT_ROOT}"
  else
    ARCHIVE="${DATASET_ARCHIVE_PATH}" DESTINATION="${DATASET_EXTRACT_ROOT}" MARKER="${DATASET_MARKER}" python - <<'PY'
import os
import tarfile
import zipfile
from pathlib import Path
archive = Path(os.environ["ARCHIVE"])
destination = Path(os.environ["DESTINATION"])
destination.mkdir(parents=True, exist_ok=True)
if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as handle: handle.extractall(destination)
elif tarfile.is_tarfile(archive):
    with tarfile.open(archive) as handle: handle.extractall(destination)
else:
    raise RuntimeError(f"Unsupported archive: {archive}")
Path(os.environ["MARKER"]).write_text("complete\n", encoding="utf-8")
PY
  fi
  DATASET_ROOT=$(python - "${DATASET_EXTRACT_ROOT}" "${DATASET_DIR_NAME}" <<'PY'
import sys
from pathlib import Path
extract = Path(sys.argv[1]).resolve()
preferred = sys.argv[2].strip()
candidates = ([extract / preferred] if preferred else []) + [extract]
for path in sorted(extract.rglob("train.jsonl")):
    if "__MACOSX" not in path.parts:
        candidates.append(path.parent.parent if path.parent.name in {"phase_a", "phasea"} else path.parent)
seen = set()
for candidate in candidates:
    candidate = candidate.resolve()
    if candidate in seen: continue
    seen.add(candidate)
    if any((candidate / rel).is_file() for rel in ("phase_a/train.jsonl", "phasea/train.jsonl", "train.jsonl")):
        print(candidate)
        raise SystemExit(0)
raise SystemExit(f"Unable to resolve Dataset V2 root below {extract}")
PY
)
fi
[ -d "${DATASET_ROOT}" ] || { echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2; exit 2; }
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_ROOT}}

resolve_train_json() {
  python - "${DATASET_ROOT}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for path in (root / "phase_a/train.jsonl", root / "phasea/train.jsonl", root / "train.jsonl"):
    if path.is_file():
        print(path)
        raise SystemExit(0)
raise SystemExit(f"train.jsonl not found below {root}")
PY
}
if [ -z "${TRAIN_JSON}" ]; then
  FULL_TRAIN_JSON=$(resolve_train_json)
  FULL_TRAIN_COUNT=$(count_jsonl "${FULL_TRAIN_JSON}")
  echo "[dataset] source train records=${FULL_TRAIN_COUNT}"
  if is_true "${USE_STRATIFIED_SUBSET}" && [ "${FULL_TRAIN_COUNT}" -ne "${SUBSET_TARGET_SAMPLES}" ]; then
    SUBSET_ROOT=${SUBSET_ROOT:-${DATASET_EXTRACT_ROOT}/.llmapgen_stratified_subsets}
    TRAIN_JSON="${SUBSET_ROOT}/train_${SUBSET_TARGET_SAMPLES}_seed${SUBSET_SEED}.jsonl"
    SUBSET_SUMMARY="${SUBSET_ROOT}/train_${SUBSET_TARGET_SAMPLES}_seed${SUBSET_SEED}.summary.json"
    subset_args=(
      --input-jsonl "${FULL_TRAIN_JSON}" --output-jsonl "${TRAIN_JSON}"
      --summary-json "${SUBSET_SUMMARY}" --target-samples "${SUBSET_TARGET_SAMPLES}"
      --difficulty-ratios "${SUBSET_DIFFICULTY_RATIOS}"
      --shortage-policy "${SUBSET_SHORTAGE_POLICY}"
      --shortage-fill-buckets "${SUBSET_SHORTAGE_FILL_BUCKETS}"
      --seed "${SUBSET_SEED}" --progress-every "${SUBSET_PROGRESS_EVERY}"
    )
    if is_true "${SUBSET_REUSE}"; then subset_args+=(--reuse-if-valid); fi
    echo "[dataset] rebuilding/reusing deterministic 200k subset: ${TRAIN_JSON}"
    python scripts/tools/build_stratified_train_subset.py "${subset_args[@]}"
    TRAIN_SELECTION_MODE=recreated_stratified_subset
  else
    TRAIN_JSON="${FULL_TRAIN_JSON}"
    TRAIN_SELECTION_MODE=packaged_train_split
  fi
else
  TRAIN_JSON=$(readlink -f "${TRAIN_JSON}")
  TRAIN_SELECTION_MODE=explicit_preserved_train_jsonl
fi
[ -s "${TRAIN_JSON}" ] || { echo "ERROR: missing train JSONL: ${TRAIN_JSON}" >&2; exit 2; }
ACTUAL_TRAIN_SAMPLES=$(count_jsonl "${TRAIN_JSON}")
[ "${ACTUAL_TRAIN_SAMPLES}" -eq "${EXPECTED_TRAIN_SAMPLES}" ] || {
  echo "ERROR: expected ${EXPECTED_TRAIN_SAMPLES} records, found ${ACTUAL_TRAIN_SAMPLES}" >&2; exit 2;
}

resolve_checkpoint_dir() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
candidates = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
def lora(p):
    return (p / "adapter_config.json").is_file() and any((p / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin"))
def full(p):
    return (p / "config.json").is_file() and any((p / n).is_file() for n in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
usable = [p for p in candidates if lora(p) or full(p)]
if not usable: raise SystemExit(f"No usable checkpoint below {root}")
if root in usable:
    print(root)
    raise SystemExit(0)
numbered = []
for path in usable:
    if not path.name.startswith("checkpoint-"):
        continue
    try:
        step = int(path.name.rsplit("-", 1)[1])
    except ValueError:
        continue
    numbered.append((step, path))
if numbered:
    print(max(numbered, key=lambda item: (item[0], str(item[1])))[1])
else:
    print(sorted(usable, key=lambda p: (len(p.parts), str(p)))[0])
PY
}
if [ -z "${CHECKPOINT_DIR}" ]; then
  CHECKPOINT_SOURCE="${CHECKPOINT_DOWNLOAD_ROOT}/source"
  if is_true "${REUSE_LOCAL_ASSETS}" && [ -d "${CHECKPOINT_SOURCE}" ] \
    && resolve_checkpoint_dir "${CHECKPOINT_SOURCE}" >/dev/null 2>&1; then
    echo "[checkpoint] reuse ${CHECKPOINT_SOURCE}"
  else
    rm -rf "${CHECKPOINT_SOURCE}"
    copy_parallel "${CHECKPOINT_OBS_PATH}" "${CHECKPOINT_SOURCE}" 128
  fi
  CHECKPOINT_DIR=$(resolve_checkpoint_dir "${CHECKPOINT_SOURCE}")
fi
CHECKPOINT_DIR=$(readlink -f "${CHECKPOINT_DIR}")
CHECKPOINT_MODE=$(python - "${CHECKPOINT_DIR}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
if (root / "adapter_config.json").is_file() and any((root / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
    if not (root / "non_lora_trainables.bin").is_file(): raise SystemExit("LoRA checkpoint missing non_lora_trainables.bin")
    print("lora")
elif (root / "config.json").is_file() and any((root / n).is_file() for n in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")):
    print("full")
else:
    raise SystemExit(f"Unsupported checkpoint layout: {root}")
PY
)

if ! is_true "${REUSE_LOCAL_ASSETS}" || ! model_asset_is_complete "${VISION_TOWER}"; then
  mkdir -p "${VISION_TOWER}"
  copy_parallel "${DINOV2_MODEL_OBS_PATH}" "${VISION_TOWER}" 128
else
  echo "[model] reuse DINOv2: ${VISION_TOWER}"
fi
model_asset_is_complete "${VISION_TOWER}" || { echo "ERROR: incomplete DINOv2 tower" >&2; exit 2; }

if [ "${CHECKPOINT_MODE}" = lora ]; then
  if ! is_true "${REUSE_LOCAL_ASSETS}" || ! model_asset_is_complete "${QWEN_PATH}"; then
    mkdir -p "${QWEN_PATH}"
    copy_parallel "${QWEN_MODEL_OBS_PATH}" "${QWEN_PATH}" 128
  else
    echo "[model] reuse CapRL base: ${QWEN_PATH}"
  fi
  model_asset_is_complete "${QWEN_PATH}" || { echo "ERROR: incomplete CapRL base model" >&2; exit 2; }
  export QWEN_BASE_MODEL_PATH="${QWEN_PATH}"
  export MODEL_BASE="${QWEN_PATH}"
fi

if [ "${NODE_RANK}" -eq 0 ]; then
  python - "${CHECKPOINT_DIR}" "${LOCAL_OUTPUT_ROOT}/checkpoint_inventory.json" "${CHECKPOINT_OBS_PATH}" "${CHECKPOINT_MODE}" <<'PY'
import json
import sys
from pathlib import Path
checkpoint = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
payload = {
    "checkpoint_dir": str(checkpoint),
    "checkpoint_obs_path": sys.argv[3],
    "mode": sys.argv[4],
    "files": [
        {"path": str(p.relative_to(checkpoint)), "bytes": p.stat().st_size}
        for p in sorted(checkpoint.rglob("*")) if p.is_file()
    ],
}
for name in ("adapter_config.json", "qwen_multimodal_checkpoint.json", "llava_checkpoint.json", "config.json"):
    path = checkpoint / name
    if path.is_file():
        try: payload[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc: payload[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

SUCCESS_MARKER="${LOCAL_OUTPUT_ROOT}/_SUCCESS"
if is_true "${RESUME_INFERENCE}" && [ -f "${SUCCESS_MARKER}" ]; then
  echo "[infer] successful output already exists: ${LOCAL_OUTPUT_ROOT}"
  exit 0
fi
export RUN_ID DATASET_OBS_PATH DATASET_ROOT TRAIN_JSON TRAIN_SELECTION_MODE ACTUAL_TRAIN_SAMPLES
export CHECKPOINT_OBS_PATH CHECKPOINT_DIR CHECKPOINT_MODE VISION_TOWER INPUT_IMAGE_SIZE MAP_TASK COORD_RANGE MAX_NEW_TOKENS RESUME_INFERENCE PER_DEVICE_INFER_BATCH_SIZE

if [ "${NODE_RANK}" -eq 0 ]; then
  TRAIN_JSON_SHA256=$(sha256_file "${TRAIN_JSON}")
  DATASET_ARCHIVE_SHA256=""
  if [ -f "${DATASET_ARCHIVE_PATH}" ]; then
    DATASET_ARCHIVE_SHA256=$(sha256_file "${DATASET_ARCHIVE_PATH}")
  fi
  INFERENCE_CONFIG_PATH="${INFERENCE_CONFIG_PATH}" \
  RUN_ID="${RUN_ID}" DATASET_OBS_PATH="${DATASET_OBS_PATH}" DATASET_ROOT="${DATASET_ROOT}" \
  TRAIN_JSON="${TRAIN_JSON}" TRAIN_SELECTION_MODE="${TRAIN_SELECTION_MODE}" \
  ACTUAL_TRAIN_SAMPLES="${ACTUAL_TRAIN_SAMPLES}" EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES}" \
  TRAIN_JSON_SHA256="${TRAIN_JSON_SHA256}" DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
  DATASET_ARCHIVE_SHA256="${DATASET_ARCHIVE_SHA256}" SUBSET_SUMMARY="${SUBSET_SUMMARY}" \
  SUBSET_TARGET_SAMPLES="${SUBSET_TARGET_SAMPLES}" SUBSET_DIFFICULTY_RATIOS="${SUBSET_DIFFICULTY_RATIOS}" \
  SUBSET_SHORTAGE_POLICY="${SUBSET_SHORTAGE_POLICY}" SUBSET_SHORTAGE_FILL_BUCKETS="${SUBSET_SHORTAGE_FILL_BUCKETS}" \
  SUBSET_SEED="${SUBSET_SEED}" CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" CHECKPOINT_MODE="${CHECKPOINT_MODE}" \
  VISION_TOWER="${VISION_TOWER}" INPUT_IMAGE_SIZE="${INPUT_IMAGE_SIZE}" MAP_TASK="${MAP_TASK}" \
  COORD_MODE="${COORD_MODE}" COORD_RANGE="${COORD_RANGE}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
  NNODES="${NNODES}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
  python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "format": "llmapgen_stage_a_train_inference_config",
    "status": "prepared",
    "run_id": os.environ["RUN_ID"],
    "dataset_obs_path": os.environ["DATASET_OBS_PATH"],
    "dataset_root": os.environ["DATASET_ROOT"],
    "dataset_archive_path": os.environ["DATASET_ARCHIVE_PATH"],
    "dataset_archive_sha256": os.environ["DATASET_ARCHIVE_SHA256"],
    "train_json": os.environ["TRAIN_JSON"],
    "train_json_sha256": os.environ["TRAIN_JSON_SHA256"],
    "train_selection_mode": os.environ["TRAIN_SELECTION_MODE"],
    "actual_train_samples": int(os.environ["ACTUAL_TRAIN_SAMPLES"]),
    "expected_train_samples": int(os.environ["EXPECTED_TRAIN_SAMPLES"]),
    "subset_summary": os.environ["SUBSET_SUMMARY"],
    "subset": {
        "target_samples": int(os.environ["SUBSET_TARGET_SAMPLES"]),
        "difficulty_ratios": os.environ["SUBSET_DIFFICULTY_RATIOS"],
        "shortage_policy": os.environ["SUBSET_SHORTAGE_POLICY"],
        "shortage_fill_buckets": os.environ["SUBSET_SHORTAGE_FILL_BUCKETS"],
        "seed": int(os.environ["SUBSET_SEED"]),
    },
    "checkpoint_obs_path": os.environ["CHECKPOINT_OBS_PATH"],
    "checkpoint_dir": os.environ["CHECKPOINT_DIR"],
    "checkpoint_mode": os.environ["CHECKPOINT_MODE"],
    "vision_tower": os.environ["VISION_TOWER"],
    "input_image_size": int(os.environ["INPUT_IMAGE_SIZE"]),
    "map_task": os.environ["MAP_TASK"],
    "coord_mode": os.environ["COORD_MODE"],
    "coord_range": int(os.environ["COORD_RANGE"]),
    "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "per_device_infer_batch_size": int(os.environ["PER_DEVICE_INFER_BATCH_SIZE"]),
    "topology": {
        "nnodes": int(os.environ["NNODES"]),
        "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    },
    "output_contract": {
        "prediction_jsonl": "train_predictions.jsonl",
        "rank_jsonl": "train_predictions_rank*.jsonl",
        "per_sample_json": False,
        "resumable": True,
    },
}
path = Path(os.environ["INFERENCE_CONFIG_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

echo "============================================================"
echo "Run id:            ${RUN_ID}"
echo "Dataset root:      ${DATASET_ROOT}"
echo "Train JSONL:       ${TRAIN_JSON}"
echo "Selection:         ${TRAIN_SELECTION_MODE} (${ACTUAL_TRAIN_SAMPLES} samples)"
echo "Canvas / target:   512x512 / centered ROI 256x256"
echo "Checkpoint:        ${CHECKPOINT_DIR} (${CHECKPOINT_MODE})"
echo "DINOv2:            ${VISION_TOWER} input=${INPUT_IMAGE_SIZE} layer=-2"
echo "Output JSONL:      ${PREDICTION_JSONL}"
echo "Per-device batch:  ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Topology:          nnodes=${NNODES} node_rank=${NODE_RANK} nproc=${NPROC_PER_NODE}"
echo "Visible NPUs:      ${ASCEND_RT_VISIBLE_DEVICES}"
echo "============================================================"

infer_args=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --vision_tower "${VISION_TOWER}"
  --mm_vision_tower_type dinov2
  --input_image_size "${INPUT_IMAGE_SIZE}"
  --disable_deepstack
  --test-json "${TRAIN_JSON}"
  --num-samples 0
  --image-folder "${IMAGE_FOLDER}"
  --prompt-mode dataset
  --map-task "${MAP_TASK}"
  --patch-size 256
  --coord-mode "${COORD_MODE}"
  --coord-range "${COORD_RANGE}"
  --conv-template conv_qwen_3_Dinov2_huawei
  --output-jsonl "${PREDICTION_JSONL}"
  --stream-fsync-every "${STREAM_FSYNC_EVERY}"
  --temperature "${TEMPERATURE}"
  --per-device-infer-batch-size "${PER_DEVICE_INFER_BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --no-sample-json
  --progress-every "${INFER_PROGRESS_EVERY}"
)
if is_true "${INFER_QUIET_SAMPLES}"; then infer_args+=(--quiet-samples); fi
if is_true "${RESUME_INFERENCE}"; then infer_args+=(--resume-output-jsonl); fi

INFER_START_NS=$(date +%s%N)
set +e
torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  scripts/tools/infer_centerline_checkpoint.py "${infer_args[@]}"
INFER_EXIT=$?
set -e
INFER_END_NS=$(date +%s%N)

if [ "${INFER_EXIT}" -ne 0 ]; then
  echo "ERROR: inference exited with code ${INFER_EXIT}" >&2
  if [ "${NODE_RANK}" -eq 0 ]; then
    python - "${LOCAL_OUTPUT_ROOT}/inference_failure.json" "${INFER_EXIT}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status": "failed", "exit_code": int(sys.argv[2])}, indent=2) + "\n", encoding="utf-8")
PY
  fi
  exit "${INFER_EXIT}"
fi
if [ "${NODE_RANK}" -ne 0 ]; then
  echo "[infer] non-master node ${NODE_RANK} completed."
  exit 0
fi
[ -s "${PREDICTION_JSONL}" ] || { echo "ERROR: aggregate JSONL missing: ${PREDICTION_JSONL}" >&2; exit 1; }

python scripts/tools/validate_stream_inference_output.py \
  --expected-jsonl "${TRAIN_JSON}" --prediction-jsonl "${PREDICTION_JSONL}" \
  --output-manifest "${LOCAL_OUTPUT_ROOT}/inference_manifest.json" \
  --expected-count "${EXPECTED_TRAIN_SAMPLES}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" --dataset-root "${DATASET_ROOT}"

INFER_END_NS=$(date +%s%N)
python - "${LOCAL_OUTPUT_ROOT}/inference_manifest.json" "${LOCAL_OUTPUT_ROOT}/inference_complete.json" \
  "${INFER_START_NS}" "${INFER_END_NS}" "${TOTAL_DEVICES}" "${RUN_ID}" \
  "${CHECKPOINT_MODE}" "${TRAIN_JSON}" "${PREDICTION_JSONL}" "${INFERENCE_CONFIG_PATH}" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
complete_path = Path(sys.argv[2])
elapsed = max((int(sys.argv[4]) - int(sys.argv[3])) / 1e9, 1e-9)
total_devices = max(1, int(sys.argv[5]))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
count = int(manifest.get("prediction_count", 0))
throughput = count / elapsed / total_devices
payload = {
    "status": "complete",
    "run_id": sys.argv[6],
    "checkpoint_mode": sys.argv[7],
    "train_json": sys.argv[8],
    "prediction_jsonl": sys.argv[9],
    "expected_samples": int(manifest.get("expected_count", 0)),
    "predicted_samples": count,
    "parse_ok_count": int(manifest.get("parse_ok_count", 0)),
    "parse_error_count": int(manifest.get("parse_error_count", 0)),
    "prediction_sha256": manifest.get("prediction_sha256", ""),
    "elapsed_seconds": elapsed,
    "total_npus": total_devices,
    "throughput_samples_per_second_per_npu": throughput,
    "validator_manifest": str(manifest_path),
}
complete_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

config_path = Path(sys.argv[10])
if config_path.is_file():
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prediction_jsonl": sys.argv[9],
            "prediction_count": count,
            "prediction_sha256": manifest.get("prediction_sha256", ""),
            "inference_manifest": str(manifest_path),
            "inference_complete": str(complete_path),
        }
    )
    temporary = config_path.with_name(f".{config_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(config_path)
print(f"DI_throughput: {throughput:.2f} samples/s/npu", flush=True)
PY

echo "[infer] COMPLETE: ${LOCAL_OUTPUT_ROOT}"
echo "[infer] prediction JSONL: ${PREDICTION_JSONL}"
echo "[infer] validator: ${LOCAL_OUTPUT_ROOT}/inference_manifest.json"

if is_true "${UPLOAD_RESULTS}" && [ -n "${RESULT_OBS_ROOT}" ]; then
  CLOUD_OUTPUT_DIR="${RESULT_OBS_ROOT%/}/${RUN_ID}"
  echo "[infer] uploading result tree -> ${CLOUD_OUTPUT_DIR}"
  if [[ "${CLOUD_OUTPUT_DIR}" == obs://* ]]; then
    SOURCE="${LOCAL_OUTPUT_ROOT}" TARGET="${CLOUD_OUTPUT_DIR}" python - <<'PY'
import os
import moxing as mox
mox.file.copy_parallel(os.environ["SOURCE"], os.environ["TARGET"], threads=128)
PY
  else
    mkdir -p "${CLOUD_OUTPUT_DIR}"
    cp -a "${LOCAL_OUTPUT_ROOT}/." "${CLOUD_OUTPUT_DIR}/"
  fi
  echo "[infer] saved result tree: ${CLOUD_OUTPUT_DIR}"
else
  echo "[infer] upload skipped; local result tree is authoritative."
fi

# Mark success only after the optional cloud copy has completed.  A failed
# upload must remain resumable instead of looking complete on the next run.
SUCCESS_PENDING="${SUCCESS_MARKER}.pending"
printf 'status=complete\nrun_id=%s\n' "${RUN_ID}" > "${SUCCESS_PENDING}"
if is_true "${UPLOAD_RESULTS}" && [ -n "${RESULT_OBS_ROOT}" ]; then
  CLOUD_OUTPUT_DIR="${RESULT_OBS_ROOT%/}/${RUN_ID}"
  if [[ "${CLOUD_OUTPUT_DIR}" == obs://* ]]; then
    SOURCE="${SUCCESS_PENDING}" TARGET="${CLOUD_OUTPUT_DIR}/_SUCCESS" python - <<'PY'
import os
import moxing as mox
mox.file.copy(os.environ["SOURCE"], os.environ["TARGET"])
PY
  else
    cp "${SUCCESS_PENDING}" "${CLOUD_OUTPUT_DIR}/_SUCCESS"
  fi
fi
mv "${SUCCESS_PENDING}" "${SUCCESS_MARKER}"
echo "[infer] success marker: ${SUCCESS_MARKER}"
if ! is_true "${KEEP_RANK_JSONL}"; then
  find "${LOCAL_OUTPUT_ROOT}" -maxdepth 1 -type f -name 'train_predictions_rank*.jsonl' -delete
fi
