#!/usr/bin/env bash
set -euo pipefail

# Single-node Ascend smoke for the stratified 200k recipe. It deliberately
# saves checkpoint-2 so checkpoint serialization is exercised before exit.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
FORMAL_SCRIPT="${REPO_ROOT}/scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_local256_200k_stratified_original_dinov2_caprl4b_nodeepstack_npu.sh"
cd "${REPO_ROOT}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
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
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=$(${PYTHON} - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind(("127.0.0.1", 0))
    print(server.getsockname()[1])
PY
)
fi

MAX_STEPS=${MAX_STEPS:-3}
NUM_EPOCHS=${NUM_EPOCHS:-100}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
SAVE_STEPS=${SAVE_STEPS:-2}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LOGGING_STEPS=${LOGGING_STEPS:-1}

SUBSET_TARGET_SAMPLES=${SUBSET_TARGET_SAMPLES:-200000}
SUBSET_DIFFICULTY_RATIOS=${SUBSET_DIFFICULTY_RATIOS:-easy=0.30,medium=0.3560290909,hard=0.2439709091,very_hard=0.10}
SUBSET_SEED=${SUBSET_SEED:-42}

OBS_CACHE=${OBS_CACHE:-/cache/jn}
RUN_ID=${RUN_ID:-datasetv2_local256_200k_stratified_checkpoint_smoke_$(date -u +%Y%m%d_%H%M%S)}
SMOKE_ROOT=${SMOKE_ROOT:-${OBS_CACHE}/outputs/datasetv2_local256_200k_stratified_checkpoint_smoke}
OUTPUT_URL=${OUTPUT_URL:-${SMOKE_ROOT}/completed}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-${SMOKE_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${SMOKE_ROOT}/logs/${RUN_ID}}
TRAIN_LOG=${TRAIN_LOG:-${LOG_ROOT}/train.log}
NPU_MEMORY_LOG=${NPU_MEMORY_LOG:-${LOG_ROOT}/npu_smi.log}
NPU_MONITOR_SECONDS=${NPU_MONITOR_SECONDS:-5}

DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/data/local256.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/data/local256_extract}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/local256}
DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-5000}
DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-8}

INSTALL_DEPS=${INSTALL_DEPS:-False}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False}
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}
SAVE_BEST_TRAIN_LOSS=False
SAVE_BEST_EVAL_LOSS=False
SAVE_BEST_INFER_INDEX=False
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}

mkdir -p "${OUTPUT_URL}" "${LOCAL_MODEL_SAVE_ROOT}" "${LOG_ROOT}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"

echo "============================================================"
echo "Dataset V2 local256 stratified-200k checkpoint-save smoke"
echo "Repo:             ${REPO_ROOT}"
echo "Python:           ${PYTHON}"
echo "Visible NPUs:     ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Topology:         1 node x ${NPROC_PER_NODE} NPUs"
echo "Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Global target:    ${TARGET_GLOBAL_BATCH_SIZE}"
echo "Max steps:        ${MAX_STEPS}"
echo "Checkpoint step:  ${SAVE_STEPS}"
echo "Eval:             disabled"
echo "Subset:           ${SUBSET_TARGET_SAMPLES}, ${SUBSET_DIFFICULTY_RATIOS}, seed=${SUBSET_SEED}"
echo "Final output:     ${OUTPUT_URL}/${RUN_ID}"
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
import numpy
import torch
import torch_npu
import transformers

payload = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "deepspeed": deepspeed.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
    "moxing_file_api": hasattr(moxing, "file"),
}
print(f"[checkpoint-smoke-preflight] {json.dumps(payload, ensure_ascii=True)}", flush=True)
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
if numpy.__version__ != "1.26.4":
    failures.append(f"numpy={numpy.__version__} expected 1.26.4")
if not payload["moxing_file_api"]:
    failures.append("Huawei moxing-framework with mox.file is required")
if not payload["npu_available"]:
    failures.append("NPU runtime is unavailable")
required_npus = int(os.environ["NPROC_PER_NODE"])
if payload["npu_count"] < required_npus:
    failures.append(f"visible NPUs={payload['npu_count']} expected at least {required_npus}")
if failures:
    raise SystemExit("Checkpoint smoke preflight failed: " + "; ".join(failures))
PY

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
fi

STOP_FILE="${LOG_ROOT}/.stop_npu_monitor"
rm -f "${STOP_FILE}"
monitor_npu_memory() {
  while [ ! -f "${STOP_FILE}" ]; do
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    npu-smi info || true
    sleep "${NPU_MONITOR_SECONDS}"
  done
}

if command -v npu-smi >/dev/null 2>&1; then
  monitor_npu_memory >"${NPU_MEMORY_LOG}" 2>&1 &
  MONITOR_PID=$!
else
  MONITOR_PID=""
  : >"${NPU_MEMORY_LOG}"
fi

stop_monitor() {
  touch "${STOP_FILE}"
  if [ -n "${MONITOR_PID}" ]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" >/dev/null 2>&1 || true
  fi
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
SUBSET_TARGET_SAMPLES="${SUBSET_TARGET_SAMPLES}" \
SUBSET_DIFFICULTY_RATIOS="${SUBSET_DIFFICULTY_RATIOS}" \
SUBSET_SEED="${SUBSET_SEED}" \
INSTALL_DEPS="${INSTALL_DEPS}" \
ENABLE_MOXING_UPGRADE="${ENABLE_MOXING_UPGRADE}" \
REUSE_LOCAL_ASSETS="${REUSE_LOCAL_ASSETS}" \
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
EXPECTED_CHECKPOINT="${FINAL_OUTPUT}/checkpoint-${SAVE_STEPS}"
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "============================================================" >&2
  echo "CHECKPOINT SMOKE FAILED (exit=${TRAIN_EXIT})" >&2
  echo "Training log:   ${TRAIN_LOG}" >&2
  echo "NPU memory log: ${NPU_MEMORY_LOG}" >&2
  grep -Ein 'out of memory|failed to allocate memory|EL0004|munmap_chunk|invalid pointer|SIGABRT|traceback|error:' "${TRAIN_LOG}" | tail -n 100 >&2 || true
  echo "============================================================" >&2
  exit "${TRAIN_EXIT}"
fi

if [ ! -d "${EXPECTED_CHECKPOINT}" ]; then
  echo "ERROR: expected checkpoint was not produced: ${EXPECTED_CHECKPOINT}" >&2
  find "${FINAL_OUTPUT}" -maxdepth 2 -type d -print >&2 || true
  exit 1
fi
if [ ! -f "${EXPECTED_CHECKPOINT}/trainer_state.json" ]; then
  echo "ERROR: trainer_state.json is missing: ${EXPECTED_CHECKPOINT}" >&2
  exit 1
fi
MODEL_STATE_COUNT=$(find "${EXPECTED_CHECKPOINT}" -type f -name '*model_states.pt' | wc -l)
OPTIM_STATE_COUNT=$(find "${EXPECTED_CHECKPOINT}" -type f -name '*optim_states.pt' | wc -l)
if [ "${MODEL_STATE_COUNT}" -lt "${NPROC_PER_NODE}" ]; then
  echo "ERROR: incomplete ZeRO model-state checkpoint: found=${MODEL_STATE_COUNT}, expected>=${NPROC_PER_NODE}" >&2
  exit 1
fi
if [ "${OPTIM_STATE_COUNT}" -lt "${NPROC_PER_NODE}" ]; then
  echo "ERROR: incomplete ZeRO optimizer-state checkpoint: found=${OPTIM_STATE_COUNT}, expected>=${NPROC_PER_NODE}" >&2
  exit 1
fi
if ! grep -Eq "('loss'|\"loss\")" "${TRAIN_LOG}"; then
  echo "ERROR: no training loss was found in ${TRAIN_LOG}" >&2
  exit 1
fi
if grep -Eq "('eval_loss'|\"eval_loss\")" "${TRAIN_LOG}"; then
  echo "ERROR: eval loss was unexpectedly computed by the no-eval recipe." >&2
  exit 1
fi
if ! grep -Fq "DI_throughput:" "${TRAIN_LOG}"; then
  echo "ERROR: required DI_throughput log line is missing." >&2
  exit 1
fi

FINAL_OUTPUT="${FINAL_OUTPUT}" EXPECTED_CHECKPOINT="${EXPECTED_CHECKPOINT}" \
TRAIN_LOG="${TRAIN_LOG}" NPU_MEMORY_LOG="${NPU_MEMORY_LOG}" \
MODEL_STATE_COUNT="${MODEL_STATE_COUNT}" OPTIM_STATE_COUNT="${OPTIM_STATE_COUNT}" \
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "status": "passed",
    "final_output": os.environ["FINAL_OUTPUT"],
    "checkpoint": os.environ["EXPECTED_CHECKPOINT"],
    "model_state_shards": int(os.environ["MODEL_STATE_COUNT"]),
    "optimizer_state_shards": int(os.environ["OPTIM_STATE_COUNT"]),
    "train_log": os.environ["TRAIN_LOG"],
    "npu_memory_log": os.environ["NPU_MEMORY_LOG"],
    "eval_enabled": False,
}
summary = Path(os.environ["FINAL_OUTPUT"]) / "CHECKPOINT_SMOKE_SUCCESS.json"
summary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"[checkpoint-smoke-success] {json.dumps(payload, ensure_ascii=True)}", flush=True)
PY

echo "============================================================"
echo "CHECKPOINT SMOKE PASSED"
echo "Checkpoint:       ${EXPECTED_CHECKPOINT}"
echo "Model shards:     ${MODEL_STATE_COUNT}"
echo "Optimizer shards: ${OPTIM_STATE_COUNT}"
echo "Training log:     ${TRAIN_LOG}"
echo "NPU memory log:   ${NPU_MEMORY_LOG}"
echo "============================================================"
