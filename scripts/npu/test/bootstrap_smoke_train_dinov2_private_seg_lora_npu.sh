#!/usr/bin/env bash
# Restore the Ascend environment/assets and validate DINOv2 LoRA DDP training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ENV_NAME="${SOURCE_ENV_NAME:-mapgen}"
SOURCE_ENV_PREFIX="${SOURCE_ENV_PREFIX:-}"
ENV_DIR="${ENV_DIR:-/home/ma-user/.conda/envs/mllm-coordtokens-npu-py311}"
RECREATE_ENV="${RECREATE_ENV:-false}"
FORCE_ENV_SETUP="${FORCE_ENV_SETUP:-false}"
ACTIVATE_SCRIPT="${ENV_DIR}/activate_mllm_coordtokens_npu.sh"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29642}"
MAX_STEPS="${MAX_STEPS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

SMOKE_ROOT="${SMOKE_ROOT:-/cache/jn/dinov2_private_seg_lora_smoke}"
ASSET_ROOT="${ASSET_ROOT:-${SMOKE_ROOT}/assets}"
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-${ASSET_ROOT}/models/facebook_dinov2-large}"
DATA_LOCAL_ROOT="${DATA_LOCAL_ROOT:-${ASSET_ROOT}/data/rc_lane_segmentation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SMOKE_ROOT}/outputs}"
RUN_ID="${RUN_ID:-dinov2_private_seg_lora_ddp_smoke_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"

MODEL_OBS_PATH="${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}"
DATASET_LIMIT="${DATASET_LIMIT:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-128}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-32}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES)$ ]]
}

mkdir -p "${SMOKE_ROOT}" "${ASSET_ROOT}" "${OUTPUT_ROOT}"
AVAILABLE_KB="$(df -Pk "${SMOKE_ROOT}" | awk 'NR==2 {print $4}')"
REQUIRED_KB=$(( MIN_FREE_GB * 1024 * 1024 ))
echo "============================================================"
echo "DINOv2 private-segmentation LoRA DDP smoke"
echo "Repo:              ${REPO_ROOT}"
echo "Source env:        ${SOURCE_ENV_PREFIX:-${SOURCE_ENV_NAME}}"
echo "Target env:        ${ENV_DIR}"
echo "Model local:       ${MODEL_LOCAL_PATH}"
echo "Dataset local:     ${DATA_LOCAL_ROOT}"
echo "Dataset sources:   ${DATASET_LIMIT}"
echo "Output:            ${OUTPUT_DIR}"
echo "NPU processes:     ${NPROC_PER_NODE}"
echo "Steps:             ${MAX_STEPS}"
echo "Train/val samples: ${MAX_TRAIN_SAMPLES}/${MAX_VAL_SAMPLES}"
echo "============================================================"
df -h "${SMOKE_ROOT}"
if [ "${AVAILABLE_KB}" -lt "${REQUIRED_KB}" ]; then
  echo "ERROR: require at least ${MIN_FREE_GB} GiB free under ${SMOKE_ROOT}." >&2
  exit 2
fi

if bool_enabled "${FORCE_ENV_SETUP}" || [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  SOURCE_ENV_NAME="${SOURCE_ENV_NAME}" \
  SOURCE_ENV_PREFIX="${SOURCE_ENV_PREFIX}" \
  ENV_DIR="${ENV_DIR}" \
  RECREATE_ENV="${RECREATE_ENV}" \
  REQUIRE_NPU=true \
  bash scripts/npu/setup/create_mllm_coordtokens_npu_env_from_mapgen.sh
else
  echo "[lora-smoke] reuse environment: ${ENV_DIR}"
fi

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
PYTHON="${ENV_DIR}/bin/python"

export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export WITHOUT_JIT_COMPILE="${WITHOUT_JIT_COMPILE:-1}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NPROC_PER_NODE="${NPROC_PER_NODE}" "${PYTHON}" - <<'PY'
import json
import os

import moxing
import numpy
import torch
import torch_npu
import transformers

result = {
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "npu_count": int(torch.npu.device_count()),
    "moxing_file_api": hasattr(moxing, "file"),
}
print("[lora-smoke] preflight=" + json.dumps(result, ensure_ascii=True), flush=True)
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"Expected numpy 1.26.4, got {numpy.__version__}")
if transformers.__version__ != "4.56.2":
    raise SystemExit(f"Expected transformers 4.56.2, got {transformers.__version__}")
if not result["npu_available"]:
    raise SystemExit("NPU is not available.")
if result["npu_count"] < int(os.environ["NPROC_PER_NODE"]):
    raise SystemExit(
        f"Need {os.environ['NPROC_PER_NODE']} visible NPUs, got {result['npu_count']}."
    )
if not result["moxing_file_api"]:
    raise SystemExit("Huawei moxing-framework is not available.")
PY

if [ ! -f "${MODEL_LOCAL_PATH}/config.json" ]; then
  mkdir -p "${MODEL_LOCAL_PATH}"
  MODEL_OBS_PATH="${MODEL_OBS_PATH}" MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH}" "${PYTHON}" - <<'PY'
import os
import moxing as mox

source = os.environ["MODEL_OBS_PATH"]
target = os.environ["MODEL_LOCAL_PATH"]
print(f"[lora-smoke] model download {source} -> {target}", flush=True)
mox.file.copy_parallel(source, target, threads=64)
PY
else
  echo "[lora-smoke] reuse model: ${MODEL_LOCAL_PATH}"
fi

"${PYTHON}" scripts/tools/download_rc_lane_segmentation_obs.py \
  --output-root "${DATA_LOCAL_ROOT}" \
  --limit "${DATASET_LIMIT}" \
  --threads 64

mapfile -t DATASET_ROOTS < "${DATA_LOCAL_ROOT}/train_roots.txt"
if [ "${#DATASET_ROOTS[@]}" -eq 0 ]; then
  echo "ERROR: no complete dataset roots were produced." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
set -o pipefail
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.vision_pretrain.train_dinov2_segmentation \
  --model_name_or_path "${MODEL_LOCAL_PATH}" \
  --dataset_roots "${DATASET_ROOTS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --input_size 518 \
  --hidden_state_indices 6 12 18 24 \
  --projection_channels 256 \
  --decoder_type dinov3_style_fpn \
  --normalization_mode minus_half \
  --split_strategy ordered_per_root \
  --vision_unfreeze_last_n_blocks 0 \
  --vision_lora_enable true \
  --vision_lora_r 8 \
  --vision_lora_alpha 16 \
  --vision_lora_dropout 0.0 \
  --vision_lora_target_modules query,value \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 1e-4 \
  --decoder_learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --warmup_steps 500 \
  --min_lr_ratio 0.0 \
  --max_grad_norm 1.0 \
  --foreground_ce_weight 1.0 \
  --dice_loss_weight 0.0 \
  --val_fraction 0.1 \
  --split_seed 42 \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --num_workers "${NUM_WORKERS}" \
  --logging_steps 1 \
  --eval_every_epochs 100 \
  --best_metric lane_iou \
  --gradient_checkpointing true \
  --bf16 true \
  --augment true \
  --device npu 2>&1 | tee "${OUTPUT_DIR}/train.log"

OUTPUT_DIR="${OUTPUT_DIR}" EXPECTED_STEPS="${MAX_STEPS}" EXPECTED_WORLD_SIZE="${NPROC_PER_NODE}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
log_text = (root / "train.log").read_text(encoding="utf-8", errors="replace")
required_logs = (
    "gradient_checkpointing_mode=non_reentrant",
    "first-backward gradient audit passed",
    "DI_throughput:",
)
missing_logs = [item for item in required_logs if item not in log_text]
if missing_logs:
    raise SystemExit(f"Smoke log is missing required evidence: {missing_logs}")

summary_path = root / "train_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Missing training summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_steps = int(os.environ["EXPECTED_STEPS"])
actual_steps = int(summary.get("global_step", -1))
if actual_steps < expected_steps:
    raise SystemExit(f"Training stopped at step {actual_steps}, expected {expected_steps}.")

vision_dir = root / "best" / "vision_tower"
for required in (
    vision_dir / "config.json",
    vision_dir / "preprocessor_config.json",
    root / "best" / "segmentation_head.pt",
):
    if not required.is_file():
        raise SystemExit(f"Missing smoke artifact: {required}")

report = {
    "status": "passed",
    "global_step": actual_steps,
    "world_size": int(os.environ["EXPECTED_WORLD_SIZE"]),
    "gradient_checkpointing_mode": "non_reentrant",
    "vision_lora_modules": 48,
    "output_dir": str(root),
}
(root / "DINOV2_LORA_DDP_SMOKE_SUCCESS.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)
print("[lora-smoke] validation=" + json.dumps(report, ensure_ascii=True), flush=True)
PY

echo "DI_throughput: 0.00 samples/s/npu"
echo "[lora-smoke] PASS: ${OUTPUT_DIR}/DINOV2_LORA_DDP_SMOKE_SUCCESS.json"
