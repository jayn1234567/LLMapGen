#!/usr/bin/env bash
# Restore all local assets and run an 8-NPU discrete-coordinate SFT smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ENV_NAME="${SOURCE_ENV_NAME:-mapgen}"
SOURCE_ENV_PREFIX="${SOURCE_ENV_PREFIX:-}"
ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/mllm-coordtokens-npu-py311}"
RECREATE_ENV="${RECREATE_ENV:-false}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-false}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-20}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-8}"

RECOVERY_ROOT="${RECOVERY_ROOT:-/cache/jn/coordtoken_sft_recovery}"
ASSET_ROOT="${ASSET_ROOT:-${RECOVERY_ROOT}/assets}"
WORK_ROOT="${WORK_ROOT:-${RECOVERY_ROOT}/work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RECOVERY_ROOT}/outputs}"
RUN_ID="${RUN_ID:-coordtokens_npu_smoke_$(date -u +%Y%m%d_%H%M%S)}"

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES)$ ]]
}

echo "============================================================"
echo "[coordtoken-recovery] repo:          ${REPO_ROOT}"
echo "[coordtoken-recovery] source env:    ${SOURCE_ENV_PREFIX:-${SOURCE_ENV_NAME}}"
echo "[coordtoken-recovery] target env:    ${ENV_DIR}"
echo "[coordtoken-recovery] assets:        ${ASSET_ROOT}"
echo "[coordtoken-recovery] outputs:       ${OUTPUT_ROOT}/${RUN_ID}"
echo "[coordtoken-recovery] devices:       ${ASCEND_RT_VISIBLE_DEVICES}"
echo "[coordtoken-recovery] smoke steps:   ${MAX_STEPS}"
echo "[coordtoken-recovery] batch:         per_device=${PER_DEVICE_TRAIN_BATCH_SIZE}, target_global=${TARGET_GLOBAL_BATCH_SIZE}"
echo "============================================================"

mkdir -p "${RECOVERY_ROOT}" "${ASSET_ROOT}" "${WORK_ROOT}" "${OUTPUT_ROOT}"
AVAILABLE_KB="$(df -Pk "${RECOVERY_ROOT}" | awk 'NR==2 {print $4}')"
REQUIRED_KB=$(( MIN_FREE_GB * 1024 * 1024 ))
echo "[coordtoken-recovery] disk status:"
df -h "${RECOVERY_ROOT}"
if [ "${AVAILABLE_KB}" -lt "${REQUIRED_KB}" ]; then
  echo "[coordtoken-recovery] insufficient free space: require ${MIN_FREE_GB} GiB under ${RECOVERY_ROOT}." >&2
  echo "[coordtoken-recovery] lower MIN_FREE_GB only after confirming room for dataset extraction and ZeRO-3 checkpoints." >&2
  exit 2
fi

if ! bool_enabled "${SKIP_ENV_SETUP}"; then
  SOURCE_ENV_NAME="${SOURCE_ENV_NAME}" \
  SOURCE_ENV_PREFIX="${SOURCE_ENV_PREFIX}" \
  ENV_DIR="${ENV_DIR}" \
  RECREATE_ENV="${RECREATE_ENV}" \
  REQUIRE_NPU=true \
  bash scripts/npu/setup/create_mllm_coordtokens_npu_env_from_mapgen.sh
fi

ACTIVATE_SCRIPT="${ENV_DIR}/activate_mllm_coordtokens_npu.sh"
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "[coordtoken-recovery] activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

NPROC_PER_NODE="${NPROC_PER_NODE}" python - <<'PY'
import json
import os
import torch
import torch_npu

requested = int(os.environ["NPROC_PER_NODE"])
result = {
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "visible_npu_count": int(torch.npu.device_count()),
    "requested_processes": requested,
}
print("[coordtoken-recovery] NPU preflight=" + json.dumps(result, ensure_ascii=True))
if not result["npu_available"]:
    raise SystemExit("NPU is not available after environment activation.")
if result["visible_npu_count"] < requested:
    raise SystemExit(
        f"Need {requested} visible NPUs, got {result['visible_npu_count']}. "
        "Check ASCEND_RT_VISIBLE_DEVICES."
    )
PY

DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH:-${ASSET_ROOT}/datasets/data_lane_intersection_samples_norm_33w_empty_patch.zip}"
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT:-${ASSET_ROOT}/datasets/data_lane_intersection_samples_norm_33w_empty_patch_extracted}"
QWEN_PATH="${QWEN_PATH:-${ASSET_ROOT}/models/CapRL-Qwen3VL-4B}"
LOCAL_MODEL_SAVE_ROOT="${LOCAL_MODEL_SAVE_ROOT:-${WORK_ROOT}/training_outputs}"

FORMAL_SCRIPT="scripts/npu/train/train_sft_stage_a_lane_intersection_jjh33w_latest_private_dinov2_last2_caprl4b_coordtokens_nodeepstack_npu.sh"
if [ ! -f "${FORMAL_SCRIPT}" ]; then
  echo "[coordtoken-recovery] formal training script not found: ${FORMAL_SCRIPT}" >&2
  exit 2
fi

echo "[coordtoken-recovery] downloading/reusing dataset and model assets, then launching training"
OUTPUT_URL="${OUTPUT_ROOT}" \
RUN_ID="${RUN_ID}" \
OBS_CACHE="${ASSET_ROOT}" \
DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
QWEN_PATH="${QWEN_PATH}" \
LOCAL_MODEL_SAVE_ROOT="${LOCAL_MODEL_SAVE_ROOT}" \
REUSE_LOCAL_ASSETS=True \
INSTALL_DEPS=False \
ENABLE_MOXING_UPGRADE=False \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${MASTER_PORT:-6067}" \
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
NUM_EPOCHS=1 \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${MAX_STEPS}" \
SAVE_TOTAL_LIMIT=1 \
LOGGING_STEPS=1 \
ENABLE_EVAL=False \
SAVE_BEST_TRAIN_LOSS=False \
SAVE_BEST_INFER_INDEX=False \
SWANLAB_ENABLE=False \
DATASET_INSPECT_MAX_SAMPLES="${DATASET_INSPECT_MAX_SAMPLES:-2000}" \
DATASET_IMAGE_CHECKS_PER_SPLIT="${DATASET_IMAGE_CHECKS_PER_SPLIT:-16}" \
bash "${FORMAL_SCRIPT}"

FINAL_OUTPUT="${OUTPUT_ROOT}/${RUN_ID}"
if [ ! -d "${FINAL_OUTPUT}" ]; then
  echo "[coordtoken-recovery] training returned successfully but output is missing: ${FINAL_OUTPUT}" >&2
  exit 3
fi

FINAL_OUTPUT="${FINAL_OUTPUT}" EXPECTED_MAX_STEPS="${MAX_STEPS}" python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

root = Path(os.environ["FINAL_OUTPUT"])
expected_step = int(os.environ["EXPECTED_MAX_STEPS"])
checkpoint = root / f"checkpoint-{expected_step}"
if not checkpoint.is_dir():
    candidates = sorted(
        (path for path in root.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not candidates:
        raise SystemExit(f"No checkpoint-* directory found below {root}")
    checkpoint = candidates[-1]

trainer_state = checkpoint / "trainer_state.json"
if not trainer_state.is_file():
    raise SystemExit(f"Missing trainer state: {trainer_state}")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
tokens = [f"<{value}>" for value in range(1001)]
token_ids = tokenizer.convert_tokens_to_ids(tokens)
if len(set(token_ids)) != 1001:
    raise SystemExit("Discrete coordinate tokens do not map to 1001 unique token ids.")
unk_id = getattr(tokenizer, "unk_token_id", None)
if unk_id is not None and any(token_id == unk_id for token_id in token_ids):
    raise SystemExit("At least one discrete coordinate token maps to the unknown token.")

state = json.loads(trainer_state.read_text(encoding="utf-8"))
global_step = int(state.get("global_step", -1))
if global_step < expected_step:
    raise SystemExit(f"Training stopped at step {global_step}, expected at least {expected_step}.")

report = {
    "status": "passed",
    "checkpoint": str(checkpoint),
    "global_step": global_step,
    "coordinate_token_mode": "angle",
    "coordinate_token_count": len(tokens),
    "coordinate_token_id_min": min(token_ids),
    "coordinate_token_id_max": max(token_ids),
}
(root / "COORDTOKEN_SMOKE_SUCCESS.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)
print("[coordtoken-recovery] validation=" + json.dumps(report, ensure_ascii=True))
PY

echo "DI_throughput: 0.00 samples/s/npu"
echo "[coordtoken-recovery] PASS: ${FINAL_OUTPUT}/COORDTOKEN_SMOKE_SUCCESS.json"
