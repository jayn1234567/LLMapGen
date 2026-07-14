#!/usr/bin/env bash
set -euo pipefail

# End-to-end DINOv2 private-segmentation validation for an Ascend host or DI.
# It downloads OBS assets, trains on 8 NPUs, evaluates, exports the best HF
# vision tower, reloads it through the production MLLM wrapper, and uploads it.

echo "[di-like-entry] reached DINOv2 private segmentation test script"
echo "[di-like-entry] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
STRICT_VERSION_CHECK=${STRICT_VERSION_CHECK:-True}
OBS_CACHE=${OBS_CACHE:-/cache/jn}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
MODEL_LOCAL_PATH=${MODEL_LOCAL_PATH:-${OBS_CACHE}/models/facebook_dinov2-large}
DATA_LOCAL_ROOT=${DATA_LOCAL_ROOT:-${OBS_CACHE}/data/rc_lane_segmentation_di_like}
DATASET_LIMIT=${DATASET_LIMIT:-1}

MAX_STEPS=${MAX_STEPS:-100}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-100}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-4096}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-256}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MASTER_PORT=${MASTER_PORT:-29630}

if [ -n "${MA_VJ_NAME:-}" ]; then
  NNODES=${NNODES:-${MA_NUM_HOSTS}}
  NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX}}
  NPROC_PER_NODE=${NPROC_PER_NODE:-${MA_NUM_GPUS}}
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
  DEFAULT_RUN_ID=$(printf '%s' "${MA_VJ_NAME}" | tr -c 'A-Za-z0-9_.-' '_')
else
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  DEFAULT_RUN_ID=dinov2_private_seg_di_like_$(date -u +%Y%m%d_%H%M%S)
fi
RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}
OUTPUT_DIR=${OUTPUT_DIR:-${OBS_CACHE}/outputs/${RUN_ID}}
OUTPUT_URL=${OUTPUT_URL:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_private_seg_di_like}
CLOUD_OUTPUT_PATH=${CLOUD_OUTPUT_PATH:-${OUTPUT_URL%/}/${RUN_ID}}
UPLOAD_TO_OBS=${UPLOAD_TO_OBS:-True}
VERIFY_ONLY=${VERIFY_ONLY:-False}

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "DINOv2 private segmentation DI-like validation"
echo "Repo:             ${REPO_ROOT}"
echo "Python:           ${PYTHON}"
echo "Model OBS:        ${MODEL_OBS_PATH}"
echo "Model local:      ${MODEL_LOCAL_PATH}"
echo "Dataset local:    ${DATA_LOCAL_ROOT}"
echo "Dataset count:    ${DATASET_LIMIT}"
echo "Output local:     ${OUTPUT_DIR}"
echo "Output OBS:       ${CLOUD_OUTPUT_PATH}"
echo "Upload:           ${UPLOAD_TO_OBS}"
echo "Verify only:      ${VERIFY_ONLY}"
echo "Topology:         nnodes=${NNODES} node_rank=${NODE_RANK} nproc=${NPROC_PER_NODE}"
echo "Rendezvous:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Max steps:        ${MAX_STEPS}"
echo "Epoch ceiling:    ${NUM_TRAIN_EPOCHS}"
echo "Train samples:    ${MAX_TRAIN_SAMPLES}"
echo "Validation:       ${MAX_VAL_SAMPLES}"
echo "Effective batch:  $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))"
echo "============================================================"

STRICT_VERSION_CHECK="${STRICT_VERSION_CHECK}" "${PYTHON}" - <<'PY'
import os
import platform
import sys

import moxing as mox
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
    "moxing_file_api": hasattr(mox, "file"),
}
print(f"[di-like-preflight] {payload}", flush=True)
if not payload["npu_available"]:
    raise SystemExit("NPU is not available.")
if not payload["moxing_file_api"]:
    raise SystemExit("The imported moxing package does not provide mox.file.")
strict = os.environ.get("STRICT_VERSION_CHECK", "True").lower() in {"1", "true", "yes", "on"}
if strict:
    failures = []
    if sys.version_info[:2] != (3, 11):
        failures.append(f"python={platform.python_version()} expected 3.11.x")
    if not torch.__version__.startswith("2.7.1"):
        failures.append(f"torch={torch.__version__} expected 2.7.1")
    if not torch_npu.__version__.startswith("2.7.1"):
        failures.append(f"torch_npu={torch_npu.__version__} expected 2.7.1")
    if transformers.__version__ != "4.56.2":
        failures.append(f"transformers={transformers.__version__} expected 4.56.2")
    if failures:
        raise SystemExit("DI-like version check failed: " + "; ".join(failures))
PY

if [[ ! "${VERIFY_ONLY}" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]] && [ ! -f "${MODEL_LOCAL_PATH}/config.json" ]; then
  mkdir -p "${MODEL_LOCAL_PATH}"
  MODEL_OBS_PATH="${MODEL_OBS_PATH}" MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH}" "${PYTHON}" - <<'PY'
import os
import moxing as mox

source = os.environ["MODEL_OBS_PATH"]
target = os.environ["MODEL_LOCAL_PATH"]
print(f"[di-like-model-download] {source} -> {target}", flush=True)
mox.file.copy_parallel(source, target, threads=64)
PY
elif [[ ! "${VERIFY_ONLY}" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]]; then
  echo "[di-like-model-download] reuse ${MODEL_LOCAL_PATH}"
fi

if [[ "${VERIFY_ONLY}" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]]; then
  echo "[di-like-train] skipped; reusing completed output ${OUTPUT_DIR}"
else
  "${PYTHON}" scripts/tools/download_rc_lane_segmentation_obs.py \
    --output-root "${DATA_LOCAL_ROOT}" \
    --limit "${DATASET_LIMIT}" \
    --threads 64

  mapfile -t DATASET_ROOTS < "${DATA_LOCAL_ROOT}/train_roots.txt"
  if [ "${#DATASET_ROOTS[@]}" -eq 0 ]; then
    echo "ERROR: no dataset train roots were produced."
    exit 1
  fi

  TORCHRUN_ARGS=(
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )

  set -o pipefail
  "${PYTHON}" -m torch.distributed.run \
    "${TORCHRUN_ARGS[@]}" \
    -m mllm.vision_pretrain.train_dinov2_segmentation \
    --model_name_or_path "${MODEL_LOCAL_PATH}" \
    --dataset_roots "${DATASET_ROOTS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --input_size 518 \
    --hidden_state_indices 6 12 18 24 \
    --projection_channels 256 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 5e-6 \
    --decoder_learning_rate 1e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --foreground_ce_weight 1.0 \
    --dice_loss_weight 0.5 \
    --val_fraction 0.1 \
    --split_seed 42 \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --logging_steps 1 \
    --eval_every_epochs 1 \
    --gradient_checkpointing true \
    --bf16 true \
    --augment true \
    --device npu 2>&1 | tee "${OUTPUT_DIR}/di_like_train.log"
fi

if [ "${NODE_RANK}" -eq 0 ]; then
  VISION_TOWER_DIR="${OUTPUT_DIR}/best/vision_tower"
  if ! grep -Fq "DI_throughput:" "${OUTPUT_DIR}/di_like_train.log"; then
    echo "ERROR: DI_throughput was not printed by the training/evaluation job."
    exit 1
  fi
  if ! grep -Fq "[dinov2-seg] eval=" "${OUTPUT_DIR}/di_like_train.log"; then
    echo "ERROR: the DI-like job did not complete an evaluation pass."
    exit 1
  fi
  for required_path in \
    "${OUTPUT_DIR}/train_summary.json" \
    "${OUTPUT_DIR}/best/metrics.json" \
    "${OUTPUT_DIR}/best/segmentation_head.pt" \
    "${VISION_TOWER_DIR}/config.json" \
    "${VISION_TOWER_DIR}/preprocessor_config.json"; do
    if [ ! -f "${required_path}" ]; then
      echo "ERROR: expected artifact was not produced: ${required_path}"
      exit 1
    fi
  done

  "${PYTHON}" scripts/tools/verify_dinov2_vision_tower.py \
    --vision-tower "${VISION_TOWER_DIR}" \
    --device npu \
    --input-size 518 \
    --select-layer -2 \
    --expected-tokens 1369 \
    --expected-hidden-size 1024 \
    --expected-num-layers 24 \
    --output-json "${OUTPUT_DIR}/best/vision_tower_verify.json"

  RUN_ID="${RUN_ID}" OUTPUT_DIR="${OUTPUT_DIR}" CLOUD_OUTPUT_PATH="${CLOUD_OUTPUT_PATH}" MAX_STEPS="${MAX_STEPS}" "${PYTHON}" - <<'PY'
import json
import os
import time
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
summary = json.loads((output_dir / "train_summary.json").read_text(encoding="utf-8"))
expected_steps = int(os.environ["MAX_STEPS"])
if int(summary.get("global_step", -1)) != expected_steps:
    raise SystemExit(
        f"Training stopped at global_step={summary.get('global_step')}, expected {expected_steps}."
    )
payload = {
    "status": "passed",
    "run_id": os.environ["RUN_ID"],
    "local_output": str(output_dir),
    "cloud_output": os.environ["CLOUD_OUTPUT_PATH"],
    "completed_unix_time": time.time(),
    "vision_tower": str(output_dir / "best" / "vision_tower"),
    "global_step": int(summary["global_step"]),
    "best_mean_iou": summary.get("best_mean_iou"),
}
(output_dir / "DI_LIKE_SUCCESS.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
)
print(f"[di-like-success] {json.dumps(payload, ensure_ascii=True)}", flush=True)
PY

  if [[ "${UPLOAD_TO_OBS}" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}" CLOUD_OUTPUT_PATH="${CLOUD_OUTPUT_PATH}" "${PYTHON}" - <<'PY'
import os
import moxing as mox

source = os.environ["OUTPUT_DIR"]
target = os.environ["CLOUD_OUTPUT_PATH"]
print(f"[di-like-upload] {source} -> {target}", flush=True)
mox.file.copy_parallel(source, target, threads=64)
required = (
    f"{target}/DI_LIKE_SUCCESS.json",
    f"{target}/best/vision_tower/config.json",
    f"{target}/best/vision_tower/preprocessor_config.json",
    f"{target}/best/vision_tower/model.safetensors",
    f"{target}/best/vision_tower_verify.json",
)
missing = [path for path in required if not mox.file.exists(path)]
if missing:
    raise SystemExit(f"OBS round-trip verification failed; missing: {missing}")
print(f"[di-like-upload] verified {len(required)} required OBS artifacts", flush=True)
PY
  else
    echo "[di-like-upload] skipped because UPLOAD_TO_OBS=${UPLOAD_TO_OBS}"
  fi

  echo "============================================================"
  echo "DI-like DINOv2 segmentation validation PASSED"
  echo "Local vision tower: ${VISION_TOWER_DIR}"
  echo "OBS output:         ${CLOUD_OUTPUT_PATH}"
  echo "============================================================"
fi
