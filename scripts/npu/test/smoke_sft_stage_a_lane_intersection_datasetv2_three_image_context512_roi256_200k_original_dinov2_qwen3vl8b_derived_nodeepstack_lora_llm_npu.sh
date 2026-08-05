#!/usr/bin/env bash
set -euo pipefail

# Single-node DI-like smoke for matched-200k experiment A.
# It reuses the formal DI launcher and only overrides runtime knobs.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_original_dinov2_qwen3vl8b_derived_nodeepstack_lora_llm_npu.sh"
cd "${REPO_ROOT}"

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
RUN_ID=${RUN_ID:-datasetv2_three_image_context512_roi256_200k_dinov2_qwen3vl8b_derived_smoke_$(date -u +%Y%m%d_%H%M%S)}
SMOKE_ROOT=${SMOKE_ROOT:-${OBS_CACHE}/outputs/datasetv2_three_image_context512_roi256_200k_dinov2_qwen3vl8b_derived_smoke}
OUTPUT_URL=${OUTPUT_URL:-${SMOKE_ROOT}/completed}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-${SMOKE_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${SMOKE_ROOT}/logs/${RUN_ID}}
TRAIN_LOG=${TRAIN_LOG:-${LOG_ROOT}/train.log}
NPU_MEMORY_LOG=${NPU_MEMORY_LOG:-${LOG_ROOT}/npu_smi.log}
NPU_MONITOR_SECONDS=${NPU_MONITOR_SECONDS:-10}

DATASET_OBS_PATH=${DATASET_OBS_PATH:?DATASET_OBS_PATH is required}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/datasets/rawlane_pose_three_image_context512_roi256_800k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/datasets/rawlane_pose_three_image_context512_roi256_800k_extract}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/rawlane_pose_three_image_context512_roi256_800k}
DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-5000}
DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}

INSTALL_DEPS=${INSTALL_DEPS:-False}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False}
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}

mkdir -p "${OUTPUT_URL}" "${LOCAL_MODEL_SAVE_ROOT}" "${LOG_ROOT}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"

echo "============================================================"
echo "Experiment A: matched-200k three-image context512/ROI256 DINOv2 + Qwen3-VL-8B-derived LLM LoRA smoke"
echo "Repo:             ${REPO_ROOT}"
echo "Python:           ${PYTHON}"
echo "Visible NPUs:     ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Topology:         1 node x ${NPROC_PER_NODE} NPUs"
echo "Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Global target:    ${TARGET_GLOBAL_BATCH_SIZE}"
echo "Max steps:        ${MAX_STEPS}"
echo "Checkpoint step:  ${SAVE_STEPS}"
echo "DINO input:       518"
echo "Distributed:      HCCL DDP (LLM LoRA; no DeepSpeed)"
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
import transformers

payload = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
    "moxing_file_api": hasattr(moxing, "file"),
}
print(f"[di-like-smoke-preflight] {json.dumps(payload, ensure_ascii=True)}", flush=True)
failures = []
if sys.version_info[:2] != (3, 11):
    failures.append(f"python={platform.python_version()} expected 3.11.x")
if not torch.__version__.startswith("2.7.1"):
    failures.append(f"torch={torch.__version__} expected 2.7.1")
if not torch_npu.__version__.startswith("2.7.1"):
    failures.append(f"torch_npu={torch_npu.__version__} expected 2.7.1")
if transformers.__version__ != "4.56.2":
    failures.append(f"transformers={transformers.__version__} expected 4.56.2")
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
if ! grep -Eq "\[multimodal-input\].*images_per_sample=3.*vision_batch=" "${TRAIN_LOG}"; then
  echo "ERROR: the smoke log does not prove that all three ordered images reached the vision tower." >&2
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
if [ ! -s "${CHECKPOINT_DIR}/non_lora_trainables.bin" ]; then
  echo "ERROR: trained DINOv2/projector weights are missing: ${CHECKPOINT_DIR}/non_lora_trainables.bin" >&2
  exit 1
fi

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
