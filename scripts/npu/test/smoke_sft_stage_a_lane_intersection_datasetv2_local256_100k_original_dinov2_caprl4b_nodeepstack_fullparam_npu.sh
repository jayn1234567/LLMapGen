#!/usr/bin/env bash
set -euo pipefail

# Single-node DI-like smoke for Dataset V2 local256 100k with original DINOv2 + CapRL.
# It reuses the formal DI launcher and only overrides runtime knobs.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_local256_100k_original_dinov2_caprl4b_nodeepstack_npu.sh"
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

MAX_STEPS=${MAX_STEPS:-20}
NUM_EPOCHS=${NUM_EPOCHS:-100}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
SAVE_STEPS=${SAVE_STEPS:-${MAX_STEPS}}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
LOGGING_STEPS=${LOGGING_STEPS:-1}
EVAL_STEPS=${EVAL_STEPS:-10}
EVAL_SAMPLE_LIMIT=${EVAL_SAMPLE_LIMIT:-64}

OBS_CACHE=${OBS_CACHE:-/cache/jn}
RUN_ID=${RUN_ID:-datasetv2_local256_100k_original_dinov2_caprl4b_eval_di_like_smoke_$(date -u +%Y%m%d_%H%M%S)}
SMOKE_ROOT=${SMOKE_ROOT:-${OBS_CACHE}/outputs/datasetv2_local256_100k_original_dinov2_caprl4b_eval_di_like_smoke}
OUTPUT_URL=${OUTPUT_URL:-${SMOKE_ROOT}/completed}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-${SMOKE_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${SMOKE_ROOT}/logs/${RUN_ID}}
TRAIN_LOG=${TRAIN_LOG:-${LOG_ROOT}/train.log}
NPU_MEMORY_LOG=${NPU_MEMORY_LOG:-${LOG_ROOT}/npu_smi.log}
NPU_MONITOR_SECONDS=${NPU_MONITOR_SECONDS:-10}

DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/datasets/local256_100k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/datasets/local256_100k_extract}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/local256}
DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-5000}
DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}

INSTALL_DEPS=${INSTALL_DEPS:-False}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False}
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}
ENABLE_EVAL=${ENABLE_EVAL:-True}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-True}
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}

mkdir -p "${OUTPUT_URL}" "${LOCAL_MODEL_SAVE_ROOT}" "${LOG_ROOT}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"

echo "============================================================"
echo "Dataset V2 local256 100k original-DINOv2 + CapRL-4B eval-best DI-like smoke"
echo "Repo:             ${REPO_ROOT}"
echo "Python:           ${PYTHON}"
echo "Visible NPUs:     ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Topology:         1 node x ${NPROC_PER_NODE} NPUs"
echo "Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Eval batch/NPU:   ${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "Global target:    ${TARGET_GLOBAL_BATCH_SIZE}"
echo "Max steps:        ${MAX_STEPS}"
echo "Checkpoint step:  ${SAVE_STEPS}"
echo "Eval cadence:     every ${EVAL_STEPS} steps on ${EVAL_SAMPLE_LIMIT} fixed samples"
echo "DINO input:       518"
echo "DeepSpeed:        scripts/deepspeed_zero3.json"
echo "Output root:      ${OUTPUT_URL}/${RUN_ID}"
echo "Training log:     ${TRAIN_LOG}"
echo "NPU memory log:   ${NPU_MEMORY_LOG}"
echo "============================================================"

NPROC_PER_NODE="${NPROC_PER_NODE}" "${PYTHON}" - <<'PY'
import json
import os
import platform
import sys

import deepspeed
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
    "deepspeed": deepspeed.__version__,
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
if deepspeed.__version__ != "0.14.4":
    failures.append(f"deepspeed={deepspeed.__version__} expected 0.14.4")
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
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
DATASET_PATH="${DATASET_PATH}" \
DATASET_INSPECT_MAX_SAMPLES="${DATASET_INSPECT_MAX_SAMPLES}" \
DATASET_IMAGE_CHECKS_PER_SPLIT="${DATASET_IMAGE_CHECKS_PER_SPLIT}" \
INSTALL_DEPS="${INSTALL_DEPS}" \
ENABLE_MOXING_UPGRADE="${ENABLE_MOXING_UPGRADE}" \
REUSE_LOCAL_ASSETS="${REUSE_LOCAL_ASSETS}" \
ENABLE_EVAL="${ENABLE_EVAL}" \
EVAL_STEPS="${EVAL_STEPS}" \
EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT}" \
SAVE_BEST_TRAIN_LOSS="${SAVE_BEST_TRAIN_LOSS}" \
SAVE_BEST_EVAL_LOSS="${SAVE_BEST_EVAL_LOSS}" \
SAVE_BEST_INFER_INDEX="${SAVE_BEST_INFER_INDEX}" \
SWANLAB_ENABLE="${SWANLAB_ENABLE}" \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE}" \
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
if ! grep -Eq "('eval_loss'|\"eval_loss\")" "${TRAIN_LOG}"; then
  echo "ERROR: no eval loss was found in ${TRAIN_LOG}" >&2
  exit 1
fi
BEST_EVAL_METADATA=""
for BEST_EVAL_ROOT in "${FINAL_OUTPUT}/eval_best_candidates" "${FINAL_OUTPUT}/eval_best"; do
  if [ -d "${BEST_EVAL_ROOT}" ]; then
    BEST_EVAL_METADATA=$(find "${BEST_EVAL_ROOT}" -type f -name best_eval_loss.json -print -quit 2>/dev/null || true)
  fi
  if [ -n "${BEST_EVAL_METADATA}" ]; then
    break
  fi
done
if [ -z "${BEST_EVAL_METADATA}" ]; then
  echo "ERROR: eval-best checkpoint metadata was not produced below ${FINAL_OUTPUT}/eval_best_candidates or eval_best" >&2
  exit 1
fi
echo "[di-like-smoke] eval-best metadata: ${BEST_EVAL_METADATA}"
if ! grep -Fq "DI_throughput:" "${TRAIN_LOG}"; then
  echo "ERROR: required DI_throughput log line is missing." >&2
  exit 1
fi

FINAL_OUTPUT="${FINAL_OUTPUT}" TRAIN_LOG="${TRAIN_LOG}" NPU_MEMORY_LOG="${NPU_MEMORY_LOG}" \
BEST_EVAL_METADATA="${BEST_EVAL_METADATA}" EVAL_STEPS="${EVAL_STEPS}" EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT}" \
MAX_STEPS="${MAX_STEPS}" PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE}" \
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE}" "${PYTHON}" - <<'PY'
import json
import os
import time
from pathlib import Path

output = Path(os.environ["FINAL_OUTPUT"])
artifacts = [
    path.name
    for path in output.iterdir()
    if path.name.startswith("model")
    or path.name.startswith("pytorch_model")
    or path.name.startswith("checkpoint-")
]
if not artifacts:
    raise SystemExit(f"No final model/checkpoint artifact was found under {output}")
payload = {
    "status": "passed",
    "max_steps": int(os.environ["MAX_STEPS"]),
    "per_device_train_batch_size": int(os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"]),
    "per_device_eval_batch_size": int(os.environ["PER_DEVICE_EVAL_BATCH_SIZE"]),
    "target_global_batch_size": int(os.environ["TARGET_GLOBAL_BATCH_SIZE"]),
    "final_output": str(output),
    "train_log": os.environ["TRAIN_LOG"],
    "npu_memory_log": os.environ["NPU_MEMORY_LOG"],
    "eval_steps": int(os.environ["EVAL_STEPS"]),
    "eval_sample_limit": int(os.environ["EVAL_SAMPLE_LIMIT"]),
    "best_eval_metadata": os.environ["BEST_EVAL_METADATA"],
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
echo "On 4 DI nodes, ZeRO-3 partitions model states across 32 ranks,"
echo "so per-NPU model-state memory should be lower than this 8-rank test."
