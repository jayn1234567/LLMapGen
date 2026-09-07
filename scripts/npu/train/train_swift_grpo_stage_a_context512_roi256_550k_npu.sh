#!/usr/bin/env bash
set -euo pipefail

# Experimental ms-swift GRPO entry for the Dataset V2 Context512/ROI256 route.
#
# The preferred input is the original 550k Dataset V2 train split.  An
# inference JSONL may still be supplied for later hard-pool experiments.
#   TRAIN_JSONL / DATASET_OBS_PATH      : original Dataset V2 train records
#   IMAGE_ROOT                          : Dataset V2 root containing images/
#   MODEL_PATH                          : CapRL-Qwen3-derived text model or
#                                         a project-compatible starting model
#   UNIMAPGEN_VISION_TOWER              : local DINOv2-Large directory
#
# The inference records are converted to Swift's messages/images/solution
# contract before Swift starts.  The prediction in an inference record is
# audit-only and is never used as the GRPO reference answer.

echo "[di-entry] reached Swift UniMapGen GRPO launcher"
echo "[di-entry] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"
echo "DI_throughput: 0.00 samples/s/npu"

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
REPO_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')
echo "[di-entry] repo=${REPO_ROOT} commit=${REPO_COMMIT}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

safe_source() {
  local path="$1"
  [ -f "${path}" ] || return 0
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

if [ -n "${ACTIVATE_SCRIPT:-}" ]; then
  safe_source "${ACTIVATE_SCRIPT}"
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # Ascend's set_env.sh reads positional parameters under some CANN builds.
  # Temporarily disabling nounset prevents a harmless environment-script
  # warning from aborting this entry point.
  safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export USE_MEMARTS=${USE_MEMARTS:-0}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}

echo "[swift-grpo] python=$(command -v python)"
python - <<'PY'
import importlib.util
import sys

required = ("torch", "torch_npu", "transformers", "PIL", "mllm")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing project/NPU modules: {missing}")
if importlib.util.find_spec("swift") is None:
    raise SystemExit(
        "ms-swift is not installed in this environment. Install the pinned DI "
        "Swift package in the image, then rerun; this launcher does not mutate "
        "the accelerator environment."
    )
import torch
import torch_npu  # noqa: F401
import transformers
import swift

npu = getattr(torch, "npu", None)
available = bool(npu is not None and npu.is_available())
count = int(npu.device_count()) if npu is not None else 0
print(f"[swift-grpo] python={sys.executable}", flush=True)
print(f"[swift-grpo] torch={torch.__version__}", flush=True)
print(f"[swift-grpo] torch_npu={getattr(torch_npu, '__version__', 'unknown')}", flush=True)
print(f"[swift-grpo] transformers={transformers.__version__}", flush=True)
print(f"[swift-grpo] swift={getattr(swift, '__version__', 'unknown')}", flush=True)
print(f"[swift-grpo] npu_available={available} npu_count={count}", flush=True)
if not available:
    raise SystemExit("NPU is unavailable; run this entry point in a DI/Ascend NPU environment.")
PY

RUN_ID=${RUN_ID:-swift_grpo_context512_roi256_550k_$(date -u +%Y%m%d_%H%M%S)}
if [ -z "${OUTPUT_ROOT:-}" ]; then
  # Keep the training source separate from the DI output mount.  This avoids
  # copying a directory onto itself when OUTPUT_URL is a local mount.
  LOCAL_OUTPUT_BASE=${LOCAL_OUTPUT_ROOT:-/cache/local_model_save_path}
  OUTPUT_ROOT="${LOCAL_OUTPUT_BASE%/}/${RUN_ID}"
fi
WORK_ROOT=${WORK_ROOT:-${OUTPUT_ROOT}/work}
mkdir -p "${WORK_ROOT}"

# Match the existing DI training convention: train locally, then publish the
# complete run directory below OUTPUT_URL/<RUN_ID> after Swift exits cleanly.
CLOUD_OUTPUT_DIR=${GRPO_RESULT_OBS:-}
if [ -z "${CLOUD_OUTPUT_DIR}" ] && [ -n "${OUTPUT_URL:-}" ]; then
  CLOUD_OUTPUT_DIR="${OUTPUT_URL%/}/${RUN_ID}"
fi

TRAIN_JSONL=${TRAIN_JSONL:-}
DATASET_ROOT=${DATASET_ROOT:-}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${WORK_ROOT}/context512_roi256_550k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${WORK_ROOT}/dataset_extract}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-context512_roi256_550k}
INFERENCE_JSONL=${INFERENCE_JSONL:-}
INFERENCE_OBS_PATH=${INFERENCE_OBS_PATH:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
IMAGE_ROOT_OBS_PATH=${IMAGE_ROOT_OBS_PATH:-}
SWIFT_DATASET=${SWIFT_DATASET:-${WORK_ROOT}/swift_grpo.jsonl}
CONVERT_SUMMARY=${CONVERT_SUMMARY:-${WORK_ROOT}/swift_grpo_conversion_summary.json}
VALIDATE_SUMMARY=${VALIDATE_SUMMARY:-${WORK_ROOT}/swift_grpo_dataset_validation.json}
EXPECTED_IMAGES=${EXPECTED_IMAGES:-1}
TRAIN_LIMIT=${TRAIN_LIMIT:-0}
COPY_THREADS=${COPY_THREADS:-128}

copy_obs() {
  local source="$1" target="$2" threads="${3:-128}"
  SOURCE="${source}" TARGET="${target}" THREADS="${threads}" python - <<'PY'
import os
from pathlib import Path
import moxing as mox

source = os.environ["SOURCE"]
target = os.environ["TARGET"]
Path(target).parent.mkdir(parents=True, exist_ok=True)
print(f"[obs] copy {source} -> {target} threads={os.environ['THREADS']}", flush=True)
if source.endswith("/") or not Path(target).suffix:
    mox.file.copy_parallel(source, target, threads=int(os.environ["THREADS"]))
else:
    mox.file.copy(source, target)
PY
}

if [ -z "${INFERENCE_JSONL}" ] && [ -n "${INFERENCE_OBS_PATH}" ]; then
  INFERENCE_JSONL="${WORK_ROOT}/source_inference.jsonl"
  copy_obs "${INFERENCE_OBS_PATH}" "${INFERENCE_JSONL}" "${COPY_THREADS}"
fi

if [ -z "${TRAIN_JSONL}" ] && [ -z "${INFERENCE_JSONL}" ]; then
  if [ -z "${DATASET_ROOT}" ]; then
    if [ ! -f "${DATASET_ARCHIVE_PATH}" ]; then
      [ -n "${DATASET_OBS_PATH}" ] || { echo "ERROR: set DATASET_OBS_PATH or TRAIN_JSONL" >&2; exit 2; }
      copy_obs "${DATASET_OBS_PATH}" "${DATASET_ARCHIVE_PATH}" "${COPY_THREADS}"
    else
      echo "[dataset] reuse archive: ${DATASET_ARCHIVE_PATH}"
    fi
    if [ ! -d "${DATASET_EXTRACT_ROOT}" ] || [ -z "$(find "${DATASET_EXTRACT_ROOT}" -type f -name train.jsonl -print -quit)" ]; then
      mkdir -p "${DATASET_EXTRACT_ROOT}"
      DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" python - <<'PY'
import os
import tarfile
from pathlib import Path

archive = Path(os.environ["DATASET_ARCHIVE_PATH"])
root = Path(os.environ["DATASET_EXTRACT_ROOT"])
print(f"[dataset] extracting {archive} -> {root}", flush=True)
with tarfile.open(archive) as handle:
    handle.extractall(root)
PY
    else
      echo "[dataset] reuse extraction: ${DATASET_EXTRACT_ROOT}"
    fi
    DATASET_ROOT=$(DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" DATASET_DIR_NAME="${DATASET_DIR_NAME}" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DATASET_EXTRACT_ROOT"])
name = os.environ.get("DATASET_DIR_NAME", "").strip()
candidates = []
if name:
    candidates.extend([root / name, root / name / "phase_a"])
candidates.append(root)
for candidate in candidates:
    if (candidate / "phase_a" / "train.jsonl").is_file():
        print(candidate)
        break
    if (candidate / "train.jsonl").is_file():
        print(candidate.parent)
        break
else:
    matches = sorted(root.rglob("train.jsonl"))
    if not matches:
        raise SystemExit(f"No train.jsonl found below {root}")
    print(matches[0].parent.parent if matches[0].parent.name in {"phase_a", "phasea"} else matches[0].parent)
PY
)
  fi
  for candidate in \
    "${DATASET_ROOT}/phase_a/train.jsonl" \
    "${DATASET_ROOT}/phasea/train.jsonl" \
    "${DATASET_ROOT}/train.jsonl"; do
    if [ -f "${candidate}" ]; then
      TRAIN_JSONL="${candidate}"
      break
    fi
  done
fi

if [ -z "${DATASET_ROOT}" ] && [ -n "${TRAIN_JSONL}" ]; then
  case "${TRAIN_JSONL}" in
    */phase_a/train.jsonl|*/phasea/train.jsonl) DATASET_ROOT=$(dirname "$(dirname "${TRAIN_JSONL}")") ;;
  esac
fi

# Accept either a JSONL file or a completed inference output directory.
if [ -z "${TRAIN_JSONL}" ] && [ -d "${INFERENCE_JSONL}" ]; then
  for candidate in \
    "${INFERENCE_JSONL}/train_predictions.jsonl" \
    "${INFERENCE_JSONL}/predictions.jsonl" \
    "${INFERENCE_JSONL}/summary.jsonl"; do
    if [ -f "${candidate}" ]; then
      INFERENCE_JSONL="${candidate}"
      break
    fi
  done
fi

SOURCE_JSONL=${TRAIN_JSONL:-${INFERENCE_JSONL}}
[ -f "${SOURCE_JSONL}" ] || { echo "ERROR: train/inference JSONL not found: ${SOURCE_JSONL}" >&2; exit 2; }
if [ -n "${TRAIN_JSONL}" ]; then
  SOURCE_KIND=dataset
else
  SOURCE_KIND=inference
fi

if [ -z "${IMAGE_ROOT}" ] && [ -n "${IMAGE_ROOT_OBS_PATH}" ]; then
  IMAGE_ROOT="${WORK_ROOT}/images_dataset"
  mkdir -p "${IMAGE_ROOT}"
  copy_obs "${IMAGE_ROOT_OBS_PATH}" "${IMAGE_ROOT}/" "${COPY_THREADS}"
fi
[ -z "${IMAGE_ROOT}" ] && [ -n "${DATASET_ROOT}" ] && IMAGE_ROOT="${DATASET_ROOT}"
[ -n "${IMAGE_ROOT}" ] || { echo "ERROR: set IMAGE_ROOT (Dataset V2 root containing images/)." >&2; exit 2; }
[ -d "${IMAGE_ROOT}" ] || { echo "ERROR: IMAGE_ROOT not found: ${IMAGE_ROOT}" >&2; exit 2; }

echo "[swift-grpo] source jsonl=${SOURCE_JSONL}"
echo "[swift-grpo] image root=${IMAGE_ROOT}"
echo "[swift-grpo] converted dataset=${SWIFT_DATASET}"
echo "[swift-grpo] local output=${OUTPUT_ROOT}"
echo "[swift-grpo] cloud output=${CLOUD_OUTPUT_DIR:-<disabled>}"

convert_args=(
  --input-jsonl "${SOURCE_JSONL}"
  --output-jsonl "${SWIFT_DATASET}"
  --image-root "${IMAGE_ROOT}"
  --source-kind "${SOURCE_KIND}"
  --summary-json "${CONVERT_SUMMARY}"
  --strict
)
if [ "${TRAIN_LIMIT}" -gt 0 ]; then
  convert_args+=(--limit "${TRAIN_LIMIT}")
fi
python scripts/tools/prepare_swift_grpo_dataset.py "${convert_args[@]}"

python scripts/tools/validate_swift_grpo_dataset.py \
  --dataset-jsonl "${SWIFT_DATASET}" \
  --image-root "${IMAGE_ROOT}" \
  --expected-images "${EXPECTED_IMAGES}" \
  --summary-json "${VALIDATE_SUMMARY}"

DATASET_ROWS=$(python - "${SWIFT_DATASET}" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
)
[ "${DATASET_ROWS}" -gt 0 ] || { echo "ERROR: converted Swift dataset is empty" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-${QWEN_MODEL_PATH:-}}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/CapRL-Qwen3VL-4B}
MODEL_CACHE=${MODEL_CACHE:-${WORK_ROOT}/model}
if [ -z "${MODEL_PATH}" ] && [ -n "${MODEL_OBS_PATH}" ]; then
  MODEL_PATH="${MODEL_CACHE}/qwen"
  mkdir -p "${MODEL_PATH}"
  copy_obs "${MODEL_OBS_PATH}" "${MODEL_PATH}/" "${COPY_THREADS}"
fi
[ -n "${MODEL_PATH}" ] || {
  echo "ERROR: set MODEL_PATH to the CapRL-Qwen3-derived text model (or MODEL_OBS_PATH)." >&2
  exit 2
}
[ -d "${MODEL_PATH}" ] || { echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2; exit 2; }

UNIMAPGEN_MODEL_BASE=${UNIMAPGEN_MODEL_BASE:-${MODEL_PATH}}
UNIMAPGEN_VISION_TOWER=${UNIMAPGEN_VISION_TOWER:-}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
VISION_TOWER_CACHE=${VISION_TOWER_CACHE:-${WORK_ROOT}/vision_tower}
if [ -z "${UNIMAPGEN_VISION_TOWER}" ] && [ -n "${VISION_TOWER_OBS_PATH}" ]; then
  UNIMAPGEN_VISION_TOWER="${VISION_TOWER_CACHE}/dinov2"
  mkdir -p "${UNIMAPGEN_VISION_TOWER}"
  copy_obs "${VISION_TOWER_OBS_PATH}" "${UNIMAPGEN_VISION_TOWER}/" "${COPY_THREADS}"
fi
[ -n "${UNIMAPGEN_VISION_TOWER}" ] || {
  echo "ERROR: set UNIMAPGEN_VISION_TOWER to the local DINOv2-Large directory." >&2
  exit 2
}
[ -d "${UNIMAPGEN_VISION_TOWER}" ] || {
  echo "ERROR: DINOv2 directory not found: ${UNIMAPGEN_VISION_TOWER}" >&2
  exit 2
}

export ROOT_IMAGE_DIR="${IMAGE_ROOT}"
export UNIMAPGEN_MODEL_BASE
export UNIMAPGEN_VISION_TOWER
export UNIMAPGEN_INPUT_IMAGE_SIZE=${UNIMAPGEN_INPUT_IMAGE_SIZE:-518}
export UNIMAPGEN_VISION_SELECT_LAYER=${UNIMAPGEN_VISION_SELECT_LAYER:--2}
export UNIMAPGEN_VISION_SELECT_FEATURE=${UNIMAPGEN_VISION_SELECT_FEATURE:-patch}
export UNIMAPGEN_PROJECTOR_TYPE=${UNIMAPGEN_PROJECTOR_TYPE:-mlp2x_gelu}
export UNIMAPGEN_MODEL_MAX_LENGTH=${UNIMAPGEN_MODEL_MAX_LENGTH:-4096}
export UNIMAPGEN_MAP_TASK=${UNIMAPGEN_MAP_TASK:-lane_intersection}
export UNIMAPGEN_COORD_MODE=${UNIMAPGEN_COORD_MODE:-norm1000}
export UNIMAPGEN_COORD_RANGE=${UNIMAPGEN_COORD_RANGE:-1000}
export UNIMAPGEN_PATCH_SIZE=${UNIMAPGEN_PATCH_SIZE:-256}

# An optional SFT adapter/full checkpoint is loaded by the external plugin
# before Swift attaches the new GRPO LoRA.  It is intentionally explicit.
export UNIMAPGEN_START_CHECKPOINT=${UNIMAPGEN_START_CHECKPOINT:-}

PLUGIN_PATH=${PLUGIN_PATH:-${REPO_ROOT}/scripts/rl/swift_unimapgen_plugin.py}
[ -f "${PLUGIN_PATH}" ] || { echo "ERROR: Swift plugin not found: ${PLUGIN_PATH}" >&2; exit 2; }

NUM_GENERATIONS=${NUM_GENERATIONS:-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}
MAX_STEPS=${MAX_STEPS:-20}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/swift_grpo_output}
SAVE_STEPS=${SAVE_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}

if command -v swift >/dev/null 2>&1; then
  SWIFT_CMD=(swift)
else
  SWIFT_CMD=(python -m swift.cli)
fi

COMMAND_FILE="${WORK_ROOT}/swift_command.txt"
printf '%q ' "${SWIFT_CMD[@]}" rlhf \
  --rlhf_type grpo --model "${MODEL_PATH}" --model_type unimapgen_qwen3_dinov2 \
  --template unimapgen_qwen3_dinov2 --dataset "${SWIFT_DATASET}" \
  --reward_funcs unimapgen_map --external_plugins "${PLUGIN_PATH}" \
  --use_vllm false --tuner_type lora --freeze_vit true --freeze_aligner true \
  --freeze_llm false --target_modules all-linear --lora_rank 8 --lora_alpha 16 \
  --num_generations "${NUM_GENERATIONS}" --steps_per_generation 1 \
  --max_length "${MAX_LENGTH}" --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --temperature 0.7 --beta 0.04 --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps "${SAVE_STEPS}" --logging_steps "${LOGGING_STEPS}" \
  --output_dir "${OUTPUT_DIR}" --bf16 true --gradient_checkpointing true \
  --dataloader_num_workers 0 --remove_unused_columns false --report_to none \
  > "${COMMAND_FILE}"
echo "[swift-grpo] command saved: ${COMMAND_FILE}"
echo "[swift-grpo] rows=${DATASET_ROWS} model=${MODEL_PATH} vision=${UNIMAPGEN_VISION_TOWER}"
echo "[swift-grpo] num_generations=${NUM_GENERATIONS} max_length=${MAX_LENGTH} max_completion_length=${MAX_COMPLETION_LENGTH}"

START_SECONDS=${SECONDS}
"${SWIFT_CMD[@]}" rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --model_type unimapgen_qwen3_dinov2 \
  --template unimapgen_qwen3_dinov2 \
  --dataset "${SWIFT_DATASET}" \
  --reward_funcs unimapgen_map \
  --external_plugins "${PLUGIN_PATH}" \
  --use_vllm false \
  --tuner_type lora \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_generations "${NUM_GENERATIONS}" \
  --steps_per_generation 1 \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --temperature 0.7 \
  --beta 0.04 \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --output_dir "${OUTPUT_DIR}" \
  --bf16 true \
  --gradient_checkpointing true \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  --report_to none

ELAPSED=$((SECONDS - START_SECONDS))
if [ -n "${CLOUD_OUTPUT_DIR}" ]; then
  if [[ "${CLOUD_OUTPUT_DIR}" == obs://* ]]; then
    SOURCE="${OUTPUT_ROOT}" TARGET="${CLOUD_OUTPUT_DIR}" python -c '
import os
from pathlib import Path
import moxing as mox
source = Path(os.environ["SOURCE"])
target = os.environ["TARGET"]
if not source.is_dir():
    raise SystemExit(f"local GRPO output is missing: {source}")
print(f"[swift-grpo] upload {source} -> {target}", flush=True)
mox.file.copy_parallel(str(source), target, threads=128)
'
  else
    mkdir -p "${CLOUD_OUTPUT_DIR}"
    cp -a "${OUTPUT_ROOT}/." "${CLOUD_OUTPUT_DIR}/"
  fi
  echo "[swift-grpo] published output=${CLOUD_OUTPUT_DIR}"
fi
if [ "${ELAPSED}" -gt 0 ]; then
  THROUGHPUT=$(python - "${DATASET_ROWS}" "${ELAPSED}" <<'PY'
import sys
rows = float(sys.argv[1])
seconds = max(float(sys.argv[2]), 1.0)
print(f"{rows / seconds:.2f}")
PY
)
else
  THROUGHPUT=0.00
fi
echo "DI_throughput: ${THROUGHPUT} samples/s/npu"
echo "[swift-grpo] completed output=${OUTPUT_DIR}"
