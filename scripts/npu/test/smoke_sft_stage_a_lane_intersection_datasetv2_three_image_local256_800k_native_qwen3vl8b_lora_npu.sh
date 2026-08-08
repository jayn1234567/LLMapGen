#!/usr/bin/env bash
set -euo pipefail

# Single-node DI-like smoke for native Qwen3-VL-8B LoRA on the released
# local256 Raw-Lane + Pose three-image 800k dataset.
# It reuses the formal DI launcher and only overrides runtime knobs.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/qwen3vl_native/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_qwen3vl8b_lora_npu.sh"
cd "${REPO_ROOT}"

# Keep native Qwen3-VL dependencies isolated from the stable Torch-2.4
# inference environment. DI continues to install its own Torch-2.7 runtime.
ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-native-qwen3vl-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_native_qwen3vl_torch240.sh}
SKIP_ENV_ACTIVATION=${SKIP_ENV_ACTIVATION:-False}
if [[ ! "${SKIP_ENV_ACTIVATION}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]; then
  if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
    echo "ERROR: native Qwen3-VL Torch-2.4 activation script not found: ${ACTIVATE_SCRIPT}" >&2
    echo "Run: bash scripts/npu/setup/create_mllm_native_qwen3vl_torch240_npu_env_from_infer.sh" >&2
    exit 2
  fi
  set +u
  source "${ACTIVATE_SCRIPT}"
  set -u
fi

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29641}

MAX_STEPS=${MAX_STEPS:-5}
NUM_EPOCHS=${NUM_EPOCHS:-100}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
SAVE_STEPS=${SAVE_STEPS:-${MAX_STEPS}}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
LOGGING_STEPS=${LOGGING_STEPS:-1}

OBS_CACHE=${OBS_CACHE:-/cache/jn}
RUN_ID=${RUN_ID:-datasetv2_three_image_local256_800k_native_qwen3vl8b_di_like_smoke_$(date -u +%Y%m%d_%H%M%S)}
SMOKE_ROOT=${SMOKE_ROOT:-${OBS_CACHE}/outputs/datasetv2_three_image_local256_800k_native_qwen3vl8b_di_like_smoke}
OUTPUT_URL=${OUTPUT_URL:-${SMOKE_ROOT}/completed}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-${SMOKE_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${SMOKE_ROOT}/logs/${RUN_ID}}
TRAIN_LOG=${TRAIN_LOG:-${LOG_ROOT}/train.log}
NPU_MEMORY_LOG=${NPU_MEMORY_LOG:-${LOG_ROOT}/npu_smi.log}
NPU_MONITOR_SECONDS=${NPU_MONITOR_SECONDS:-10}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_rawpos/local256_rawlane_pose_800k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/datasets/local256_rawlane_pose_800k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/datasets/local256_rawlane_pose_800k_extract}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/local256_rawlane_pose_800k}
DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-5000}
DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}

INSTALL_DEPS=${INSTALL_DEPS:-False}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False}
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
VERIFY_LORA_GRADIENTS=${VERIFY_LORA_GRADIENTS:-True}

mkdir -p "${OUTPUT_URL}" "${LOCAL_MODEL_SAVE_ROOT}" "${LOG_ROOT}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"

echo "============================================================"
echo "Dataset V2 three-image local256-800k native Qwen3-VL-8B LoRA smoke"
echo "Repo:             ${REPO_ROOT}"
echo "Python:           ${PYTHON}"
echo "Visible NPUs:     ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Topology:         1 node x ${NPROC_PER_NODE} NPUs"
echo "Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Global target:    ${TARGET_GLOBAL_BATCH_SIZE}"
echo "Max steps:        ${MAX_STEPS}"
echo "Checkpoint step:  ${SAVE_STEPS}"
echo "Native vision:    Qwen3-VL dynamic local256 input, three images"
echo "Distributed:      HCCL DDP (LLM+vision-attention+merger LoRA; no DeepSpeed)"
echo "Output root:      ${OUTPUT_URL}/${RUN_ID}"
echo "Training log:     ${TRAIN_LOG}"
echo "NPU memory log:   ${NPU_MEMORY_LOG}"
echo "============================================================"

NPROC_PER_NODE="${NPROC_PER_NODE}" "${PYTHON}" - <<'PY'
import json
import os
import platform
import sys

import moxing
import torch
import torch_npu
import torchvision
import transformers

from mllm.native_qwen3vl.modeling import resolve_native_model_class

payload = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
    "moxing_file_api": hasattr(moxing, "file"),
}
print(f"[di-like-smoke-preflight] {json.dumps(payload, ensure_ascii=True)}", flush=True)
failures = []
if sys.version_info[:2] != (3, 11):
    failures.append(f"python={platform.python_version()} expected 3.11.x")
if not torch.__version__.startswith("2.4.0"):
    failures.append(f"torch={torch.__version__} expected 2.4.0")
if not torch_npu.__version__.startswith("2.4.0"):
    failures.append(f"torch_npu={torch_npu.__version__} expected 2.4.0")
if not torchvision.__version__.startswith("0.19.0"):
    failures.append(f"torchvision={torchvision.__version__} expected 0.19.0")
if transformers.__version__ != "4.57.3":
    failures.append(f"transformers={transformers.__version__} expected 4.57.3")
try:
    import peft
except ImportError:
    failures.append("peft is required for native Qwen3-VL LoRA")
else:
    if peft.__version__ != "0.18.0":
        failures.append(f"peft={peft.__version__} expected 0.18.0")
try:
    resolve_native_model_class()
except Exception as exc:
    failures.append(f"native Qwen3-VL model class is unavailable: {exc!r}")
if not payload["moxing_file_api"]:
    failures.append("Huawei moxing-framework with mox.file is required")
if not payload["npu_available"]:
    failures.append("NPU runtime is unavailable")
required_npus = int(os.environ["NPROC_PER_NODE"])
if payload["npu_count"] < required_npus:
    failures.append(f"visible NPUs={payload['npu_count']} expected at least {required_npus}")
if failures:
    raise SystemExit("DI-like smoke preflight failed: " + "; ".join(failures))
PY

if command -v npu-smi >/dev/null 2>&1; then
  echo "[di-like-smoke] initial npu-smi snapshot"
  npu-smi info || true
else
  echo "[di-like-smoke] WARNING: npu-smi is unavailable; periodic memory snapshots are disabled."
fi

STOP_FILE="${LOG_ROOT}/.stop_npu_monitor"
rm -f "${STOP_FILE}"
monitor_npu_memory() {
  while [ ! -f "${STOP_FILE}" ]; do
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if command -v npu-smi >/dev/null 2>&1; then
      npu-smi info || true
    fi
    sleep "${NPU_MONITOR_SECONDS}"
  done
}
monitor_npu_memory >"${NPU_MEMORY_LOG}" 2>&1 &
MONITOR_PID=$!

stop_monitor() {
  touch "${STOP_FILE}"
  kill "${MONITOR_PID}" >/dev/null 2>&1 || true
  wait "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap stop_monitor EXIT INT TERM

set +e
OUTPUT_URL="${OUTPUT_URL}" \
RUN_ID="${RUN_ID}" \
OBS_CACHE="${OBS_CACHE}" \
LOCAL_MODEL_SAVE_ROOT="${LOCAL_MODEL_SAVE_ROOT}" \
DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
DATASET_OBS_PATH="${DATASET_OBS_PATH}" \
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
DATASET_PATH="${DATASET_PATH}" \
DATASET_INSPECT_MAX_SAMPLES="${DATASET_INSPECT_MAX_SAMPLES}" \
DATASET_IMAGE_CHECKS_PER_SPLIT="${DATASET_IMAGE_CHECKS_PER_SPLIT}" \
INSTALL_DEPS="${INSTALL_DEPS}" \
ENABLE_MOXING_UPGRADE="${ENABLE_MOXING_UPGRADE}" \
REUSE_LOCAL_ASSETS="${REUSE_LOCAL_ASSETS}" \
SAVE_BEST_TRAIN_LOSS="${SAVE_BEST_TRAIN_LOSS}" \
SAVE_BEST_INFER_INDEX="${SAVE_BEST_INFER_INDEX}" \
SWANLAB_ENABLE="${SWANLAB_ENABLE}" \
VERIFY_LORA_GRADIENTS="${VERIFY_LORA_GRADIENTS}" \
EXPECTED_NUM_IMAGES=3 \
EXPECTED_PYTHON_MAJOR_MINOR=3.11 \
EXPECTED_TORCH_PREFIX=2.4.0 \
EXPECTED_TORCH_NPU_PREFIX=2.4.0 \
EXPECTED_TORCHVISION_PREFIX=0.19.0 \
EXPECTED_TRANSFORMERS_VERSION=4.57.3 \
EXPECTED_PEFT_VERSION=0.18.0 \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
LOGGING_STEPS="${LOGGING_STEPS}" \
bash "${FORMAL_SCRIPT}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

stop_monitor
trap - EXIT INT TERM

FINAL_OUTPUT="${OUTPUT_URL%/}/${RUN_ID}"
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "============================================================" >&2
  echo "DI-LIKE SMOKE FAILED (exit=${TRAIN_EXIT})" >&2
  echo "Training log:   ${TRAIN_LOG}" >&2
  echo "NPU memory log: ${NPU_MEMORY_LOG}" >&2
  grep -Ein 'out of memory|failed to allocate memory|EL0004|acl.*memory|killed|SIGKILL|traceback|error:' "${TRAIN_LOG}" | tail -n 80 >&2 || true
  echo "============================================================" >&2
  exit "${TRAIN_EXIT}"
fi

if [ ! -d "${FINAL_OUTPUT}" ]; then
  echo "ERROR: final saved output was not produced: ${FINAL_OUTPUT}" >&2
  exit 1
fi
if ! grep -Eq "('loss'|\"loss\")" "${TRAIN_LOG}"; then
  echo "ERROR: no training loss was found in ${TRAIN_LOG}" >&2
  exit 1
fi
if ! grep -Fq "DI_throughput:" "${TRAIN_LOG}"; then
  echo "ERROR: required DI_throughput log line is missing." >&2
  exit 1
fi
if ! grep -Eq 'DI_throughput: (0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?) samples/s/npu' "${TRAIN_LOG}"; then
  echo "ERROR: no positive per-NPU DI_throughput measurement was found." >&2
  exit 1
fi
if ! grep -Eq "\[native-multimodal-input\].*images_per_sample=\[3(, 3)*\].*total_images=[1-9][0-9]* processed_image_grids=[1-9][0-9]*" "${TRAIN_LOG}"; then
  echo "ERROR: the smoke log does not prove that all three ordered images reached native Qwen3-VL." >&2
  exit 1
fi
if ! grep -Fq "Native gradient checkpointing: use_reentrant=False (visual LoRA)" "${TRAIN_LOG}"; then
  echo "ERROR: non-reentrant visual-LoRA checkpointing was not confirmed in the training log." >&2
  exit 1
fi
if ! grep -Eq 'Native Qwen3-VL LoRA targets: language=[1-9][0-9]*, visual_attention=[1-9][0-9]*, merger=[1-9][0-9]*' "${TRAIN_LOG}"; then
  echo "ERROR: resolved language, visual-attention, and merger LoRA targets were not all confirmed." >&2
  exit 1
fi
if ! grep -Eq '\[native-lora-gradient-check\].*language=nonzero.*vision=nonzero.*merger=nonzero' "${TRAIN_LOG}"; then
  echo "ERROR: one or more LoRA groups never produced a finite non-zero gradient." >&2
  exit 1
fi
CHECKPOINT_DIR="${FINAL_OUTPUT}/checkpoint-${MAX_STEPS}"
if [ ! -f "${CHECKPOINT_DIR}/adapter_config.json" ]; then
  echo "ERROR: LoRA adapter config is missing: ${CHECKPOINT_DIR}/adapter_config.json" >&2
  exit 1
fi
if ! find "${CHECKPOINT_DIR}" -maxdepth 1 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \) -print -quit | grep -q .; then
  echo "ERROR: LoRA adapter weights are missing below ${CHECKPOINT_DIR}" >&2
  exit 1
fi
for required_file in trainer_state.json optimizer.pt scheduler.pt; do
  if [ ! -f "${CHECKPOINT_DIR}/${required_file}" ]; then
    echo "ERROR: resumable checkpoint artifact is missing: ${CHECKPOINT_DIR}/${required_file}" >&2
    exit 1
  fi
done
CHECKPOINT_DIR="${CHECKPOINT_DIR}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import torch

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
config_path = checkpoint_dir / "adapter_config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
targets = config.get("target_modules") or []
if isinstance(targets, str):
    targets = [targets]

def group_for_target_name(name):
    """Classify PEFT-minimized target suffixes from adapter_config.json."""
    lowered = str(name).lower()
    if "merger" in lowered or "linear_fc" in lowered:
        return "merger"
    if lowered == "qkv" or lowered.endswith(".qkv") or lowered.endswith("attn.proj"):
        return "vision"
    return "language"

target_counts = {group: 0 for group in ("language", "vision", "merger")}
for name in targets:
    target_counts[group_for_target_name(name)] += 1
missing_targets = [group for group, count in target_counts.items() if count <= 0]
if missing_targets:
    raise SystemExit(
        f"Native LoRA targets are missing groups {missing_targets!r} from {config_path}: {targets}"
    )

safe_path = checkpoint_dir / "adapter_model.safetensors"
bin_path = checkpoint_dir / "adapter_model.bin"
if safe_path.is_file():
    from safetensors.torch import load_file

    state = load_file(str(safe_path), device="cpu")
elif bin_path.is_file():
    try:
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(bin_path), map_location="cpu")
else:
    raise SystemExit(f"Adapter weights are missing below {checkpoint_dir}")

updated_lora_b = {group: 0.0 for group in ("language", "vision", "merger")}

def group_for_state_name(name):
    """Classify full adapter state keys, with minimized-name fallback."""
    lowered = f".{str(name).lower()}."
    if "merger" in lowered or "linear_fc" in lowered:
        return "merger"
    if ".visual." in lowered:
        return "vision"
    return group_for_target_name(name)

for name, tensor in state.items():
    if "lora_b" not in name.lower() or not torch.is_tensor(tensor):
        continue
    updated_lora_b[group_for_state_name(name)] += float(tensor.detach().float().abs().sum().item())
missing_updates = [group for group, magnitude in updated_lora_b.items() if magnitude <= 0.0]
if missing_updates:
    raise SystemExit(
        "Saved LoRA-B weights did not change from zero for groups "
        f"{missing_updates!r}: {updated_lora_b!r}"
    )
print(
    "[native-lora-checkpoint] "
    f"targets={target_counts} updated_lora_b={updated_lora_b}",
    flush=True,
)
PY

FINAL_OUTPUT="${FINAL_OUTPUT}" TRAIN_LOG="${TRAIN_LOG}" NPU_MEMORY_LOG="${NPU_MEMORY_LOG}" \
MAX_STEPS="${MAX_STEPS}" PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE}" "${PYTHON}" - <<'PY'
import json
import os
import time
from pathlib import Path

output = Path(os.environ["FINAL_OUTPUT"])
artifacts = [
    path.name
    for path in output.iterdir()
    if path.name.startswith("adapter_model")
    or path.name.startswith("model")
    or path.name.startswith("pytorch_model")
    or path.name.startswith("non_lora_trainables")
    or path.name.startswith("checkpoint-")
]
if not artifacts:
    raise SystemExit(f"No final model/checkpoint artifact was found under {output}")
payload = {
    "status": "passed",
    "max_steps": int(os.environ["MAX_STEPS"]),
    "per_device_train_batch_size": int(os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"]),
    "target_global_batch_size": int(os.environ["TARGET_GLOBAL_BATCH_SIZE"]),
    "final_output": str(output),
    "train_log": os.environ["TRAIN_LOG"],
    "npu_memory_log": os.environ["NPU_MEMORY_LOG"],
    "artifacts": sorted(artifacts),
    "completed_unix_time": time.time(),
}
summary = output / "DI_LIKE_SMOKE_SUCCESS.json"
summary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
print(f"[di-like-smoke-success] {json.dumps(payload, ensure_ascii=True)}", flush=True)
PY

echo "============================================================"
echo "DI-LIKE SMOKE PASSED"
echo "Final output:   ${FINAL_OUTPUT}"
echo "Training log:   ${TRAIN_LOG}"
echo "NPU memory log: ${NPU_MEMORY_LOG}"
echo "Formal DI keeps the same HCCL DDP LoRA path and derives accumulation from world size."
