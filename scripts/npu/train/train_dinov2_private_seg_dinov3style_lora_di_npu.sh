#!/usr/bin/env bash
set -euo pipefail

# Formal DI recipe for private-road DINOv2-L segmentation that mirrors the
# pasted private DINOv3 LoRA setup: q/v LoRA in the vision backbone, FPN-style
# segmentation head, image/255 - 0.5 normalization, and ordered 90/10 split per
# source dataset. The exported best/vision_tower merges LoRA into a normal
# HuggingFace DINOv2 checkpoint, so Jiangjihua's DINOv2 route can load it
# directly as VISION_TOWER.
# This file is self-contained: it prepares the DI Python environment, downloads
# all configured OBS datasets, launches distributed NPU training, validates the
# exported HF vision tower, and uploads the complete run to OUTPUT_URL.

echo "[di-entry] reached formal DINOv2 private segmentation DINOv3-style LoRA training script"
echo "[di-entry] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

source_if_exists() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    return
  fi
  local nounset_was_on=0
  case "$-" in
    *u*) nounset_was_on=1; set +u ;;
  esac
  export ZSH_VERSION="${ZSH_VERSION:-}"
  source "${path}"
  if [ "${nounset_was_on}" = "1" ]; then
    set -u
  fi
}

bool_enabled() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|YES|on|ON)$ ]]
}

source_if_exists "/usr/local/Ascend/ascend-toolkit/set_env.sh"
source_if_exists "/usr/local/Ascend/nnal/atb/set_env.sh"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
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
INSTALL_DEPS=${INSTALL_DEPS:-True}
STRICT_VERSION_CHECK=${STRICT_VERSION_CHECK:-True}
MOXING_WHL_OBS_PATH=${MOXING_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}
MOXING_WHL_LOCAL_PATH=${MOXING_WHL_LOCAL_PATH:-/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl}
TORCH_NPU_WHL_OBS_PATH=${TORCH_NPU_WHL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
TORCH_NPU_WHL_LOCAL_PATH=${TORCH_NPU_WHL_LOCAL_PATH:-/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}

OBS_CACHE=${OBS_CACHE:-/cache/llmapgen_dinov2_seg}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
MODEL_LOCAL_PATH=${MODEL_LOCAL_PATH:-${OBS_CACHE}/models/facebook_dinov2-large}
DATA_LOCAL_ROOT=${DATA_LOCAL_ROOT:-${OBS_CACHE}/data/rc_lane_segmentation}
DATASET_LIMIT=${DATASET_LIMIT:-0}
EXPECTED_FULL_DATASET_COUNT=16

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-20}
MAX_STEPS=${MAX_STEPS:--1}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-0}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-64}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
VISION_LEARNING_RATE=${VISION_LEARNING_RATE:-1e-4}
DECODER_LEARNING_RATE=${DECODER_LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
WARMUP_STEPS=${WARMUP_STEPS:-500}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
FOREGROUND_CE_WEIGHT=${FOREGROUND_CE_WEIGHT:-1.0}
DICE_LOSS_WEIGHT=${DICE_LOSS_WEIGHT:-0.0}
HIDDEN_STATE_INDICES=${HIDDEN_STATE_INDICES:-"6 12 18 24"}
DECODER_TYPE=${DECODER_TYPE:-dinov3_style_fpn}
NORMALIZATION_MODE=${NORMALIZATION_MODE:-minus_half}
SPLIT_STRATEGY=${SPLIT_STRATEGY:-ordered_per_root}
VISION_UNFREEZE_LAST_N_BLOCKS=${VISION_UNFREEZE_LAST_N_BLOCKS:-0}
VISION_LORA_ENABLE=${VISION_LORA_ENABLE:-True}
VISION_LORA_R=${VISION_LORA_R:-8}
VISION_LORA_ALPHA=${VISION_LORA_ALPHA:-16}
VISION_LORA_DROPOUT=${VISION_LORA_DROPOUT:-0.0}
VISION_LORA_TARGET_MODULES=${VISION_LORA_TARGET_MODULES:-query,value}
BEST_METRIC=${BEST_METRIC:-lane_iou}
VAL_FRACTION=${VAL_FRACTION:-0.1}
SPLIT_SEED=${SPLIT_SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-8}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_EVERY_EPOCHS=${EVAL_EVERY_EPOCHS:-1}
MASTER_PORT=${MASTER_PORT:-29640}

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
  DEFAULT_RUN_ID=dinov2_private_seg_dinov3style_lora_$(date -u +%Y%m%d_%H%M%S)
fi
RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-/cache/local_model_save_path}
OUTPUT_DIR=${OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT}/${RUN_ID}}
if [ -z "${OUTPUT_URL:-}" ]; then
  OUTPUT_URL=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_private_seg_dinov3style_lora
fi
CLOUD_OUTPUT_PATH=${CLOUD_OUTPUT_PATH:-${OUTPUT_URL%/}/${RUN_ID}}
DINOV2_REGISTRY_OBS_ROOT=${DINOV2_REGISTRY_OBS_ROOT:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_private_seg_dinov3style_lora}
DINOV2_REGISTRY_RUN_PATH=${DINOV2_REGISTRY_RUN_PATH:-${DINOV2_REGISTRY_OBS_ROOT%/}/${RUN_ID}}

TOTAL_DEVICES=$((NNODES * NPROC_PER_NODE))
MICRO_GLOBAL_BATCH=$((TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_GLOBAL_BATCH - 1) / MICRO_GLOBAL_BATCH ))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
  GRADIENT_ACCUMULATION_STEPS=1
fi
EFFECTIVE_GLOBAL_BATCH_SIZE=$((MICRO_GLOBAL_BATCH * GRADIENT_ACCUMULATION_STEPS))

echo "============================================================"
echo "Formal DINOv2 private segmentation: DINOv3-style LoRA"
echo "Repo:             ${REPO_ROOT}"
echo "Python:           ${PYTHON}"
echo "Install deps:     ${INSTALL_DEPS}"
echo "Model OBS:        ${MODEL_OBS_PATH}"
echo "Model local:      ${MODEL_LOCAL_PATH}"
echo "Dataset local:    ${DATA_LOCAL_ROOT}"
echo "Dataset limit:    ${DATASET_LIMIT} (0 means all ${EXPECTED_FULL_DATASET_COUNT})"
echo "Output local:     ${OUTPUT_DIR}"
echo "Output OBS:       ${CLOUD_OUTPUT_PATH}"
echo "Registry OBS:     ${DINOV2_REGISTRY_RUN_PATH}"
echo "Topology:         nnodes=${NNODES} node_rank=${NODE_RANK} nproc=${NPROC_PER_NODE}"
echo "Rendezvous:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Epochs:           ${NUM_TRAIN_EPOCHS}"
echo "Max steps:        ${MAX_STEPS} (-1 means epoch schedule)"
echo "Sample limits:    train=${MAX_TRAIN_SAMPLES}, val=${MAX_VAL_SAMPLES} (0 means all)"
echo "Per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Grad accumulation:${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch:  ${EFFECTIVE_GLOBAL_BATCH_SIZE}"
echo "Vision LR:        ${VISION_LEARNING_RATE}"
echo "Decoder LR:       ${DECODER_LEARNING_RATE}"
echo "Decoder:          ${DECODER_TYPE}"
echo "Hidden states:    ${HIDDEN_STATE_INDICES}"
echo "Normalization:    ${NORMALIZATION_MODE}"
echo "Split strategy:   ${SPLIT_STRATEGY}"
echo "Vision LoRA:      enable=${VISION_LORA_ENABLE} r=${VISION_LORA_R} alpha=${VISION_LORA_ALPHA} targets=${VISION_LORA_TARGET_MODULES}"
echo "Best metric:      ${BEST_METRIC}"
echo "============================================================"

if bool_enabled "${INSTALL_DEPS}"; then
  "${PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Formal DI recipe requires Python 3.11, got {sys.version}")
PY
  MOXING_WHL_OBS_PATH="${MOXING_WHL_OBS_PATH}" \
  MOXING_WHL_LOCAL_PATH="${MOXING_WHL_LOCAL_PATH}" \
  TORCH_NPU_WHL_OBS_PATH="${TORCH_NPU_WHL_OBS_PATH}" \
  TORCH_NPU_WHL_LOCAL_PATH="${TORCH_NPU_WHL_LOCAL_PATH}" \
  USE_MEMARTS=0 "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import moxing as mox

for source_key, target_key in (
    ("MOXING_WHL_OBS_PATH", "MOXING_WHL_LOCAL_PATH"),
    ("TORCH_NPU_WHL_OBS_PATH", "TORCH_NPU_WHL_LOCAL_PATH"),
):
    source = os.environ[source_key]
    target = os.environ[target_key]
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    print(f"[di-deps] download {source} -> {target}", flush=True)
    mox.file.copy(source, target)
PY

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  "${PYTHON}" -m pip install --upgrade pip setuptools==75.8.0 wheel
  "${PYTHON}" -m pip install torch==2.7.1 torchvision==0.22.1
  "${PYTHON}" -m pip install --force-reinstall --no-deps "${TORCH_NPU_WHL_LOCAL_PATH}"
  "${PYTHON}" -m pip uninstall -y moxing moxing-framework >/dev/null 2>&1 || true
  "${PYTHON}" -m pip install "${MOXING_WHL_LOCAL_PATH}"
  "${PYTHON}" -m pip install \
    transformers==4.56.2 \
    'tokenizers>=0.22.0,<0.23.0' \
    huggingface-hub==0.36.2 \
    'safetensors>=0.4.3' \
    'Pillow>=10.0.0' \
    'tqdm>=4.66' \
    'requests>=2.31,<3' \
    'packaging>=23' \
    timm==1.0.27 \
    numpy==1.26.4 \
    protobuf==4.25.7 \
    'scipy>=1.10,<2' \
    'psutil>=5.9' \
    'absl-py>=2.0' \
    'attrs>=23.0' \
    'cloudpickle>=3.0' \
    'decorator>=5.1' \
    'ml-dtypes>=0.4,<1' \
    'tornado>=6.3'
  "${PYTHON}" -m pip install setuptools==75.8.0 numpy==1.26.4 protobuf==4.25.7
fi

export MOX_PROFILE=1
export MOX_RECORD_OBS=1

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
print(f"[di-preflight] {payload}", flush=True)
if not payload["npu_available"]:
    raise SystemExit("NPU is not available after dependency setup.")
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
        raise SystemExit("Formal DI version check failed: " + "; ".join(failures))
PY

if [ ! -f "${MODEL_LOCAL_PATH}/config.json" ]; then
  mkdir -p "${MODEL_LOCAL_PATH}"
  MODEL_OBS_PATH="${MODEL_OBS_PATH}" MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH}" "${PYTHON}" - <<'PY'
import os
import moxing as mox

source = os.environ["MODEL_OBS_PATH"]
target = os.environ["MODEL_LOCAL_PATH"]
print(f"[di-model-download] {source} -> {target}", flush=True)
mox.file.copy_parallel(source, target, threads=64)
PY
else
  echo "[di-model-download] reuse ${MODEL_LOCAL_PATH}"
fi

"${PYTHON}" scripts/tools/download_rc_lane_segmentation_obs.py \
  --output-root "${DATA_LOCAL_ROOT}" \
  --limit "${DATASET_LIMIT}" \
  --threads 128

mapfile -t DATASET_ROOTS < "${DATA_LOCAL_ROOT}/train_roots.txt"
if [ "${#DATASET_ROOTS[@]}" -eq 0 ]; then
  echo "ERROR: no dataset train roots were produced."
  exit 1
fi
if [ "${DATASET_LIMIT}" -eq 0 ] && [ "${#DATASET_ROOTS[@]}" -ne "${EXPECTED_FULL_DATASET_COUNT}" ]; then
  echo "ERROR: expected ${EXPECTED_FULL_DATASET_COUNT} complete datasets, got ${#DATASET_ROOTS[@]}."
  exit 1
fi
read -r -a HIDDEN_STATE_INDEX_ARGS <<< "${HIDDEN_STATE_INDICES}"

mkdir -p "${OUTPUT_DIR}"
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
  --hidden_state_indices "${HIDDEN_STATE_INDEX_ARGS[@]}" \
  --projection_channels 256 \
  --decoder_type "${DECODER_TYPE}" \
  --normalization_mode "${NORMALIZATION_MODE}" \
  --split_strategy "${SPLIT_STRATEGY}" \
  --vision_unfreeze_last_n_blocks "${VISION_UNFREEZE_LAST_N_BLOCKS}" \
  --vision_lora_enable "${VISION_LORA_ENABLE}" \
  --vision_lora_r "${VISION_LORA_R}" \
  --vision_lora_alpha "${VISION_LORA_ALPHA}" \
  --vision_lora_dropout "${VISION_LORA_DROPOUT}" \
  --vision_lora_target_modules "${VISION_LORA_TARGET_MODULES}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${VISION_LEARNING_RATE}" \
  --decoder_learning_rate "${DECODER_LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --foreground_ce_weight "${FOREGROUND_CE_WEIGHT}" \
  --dice_loss_weight "${DICE_LOSS_WEIGHT}" \
  --val_fraction "${VAL_FRACTION}" \
  --split_seed "${SPLIT_SEED}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --num_workers "${NUM_WORKERS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_every_epochs "${EVAL_EVERY_EPOCHS}" \
  --best_metric "${BEST_METRIC}" \
  --gradient_checkpointing true \
  --bf16 true \
  --augment true \
  --device npu 2>&1 | tee "${OUTPUT_DIR}/train.log"

if [ "${NODE_RANK}" -eq 0 ]; then
  VISION_TOWER_DIR="${OUTPUT_DIR}/best/vision_tower"
  for required_path in \
    "${OUTPUT_DIR}/train_summary.json" \
    "${OUTPUT_DIR}/best/metrics.json" \
    "${OUTPUT_DIR}/best/segmentation_head.pt" \
    "${VISION_TOWER_DIR}/config.json" \
    "${VISION_TOWER_DIR}/preprocessor_config.json"; do
    if [ ! -f "${required_path}" ]; then
      echo "ERROR: expected formal-training artifact was not produced: ${required_path}"
      exit 1
    fi
  done
  if ! grep -Fq "DI_throughput:" "${OUTPUT_DIR}/train.log"; then
    echo "ERROR: DI_throughput was not printed by formal training."
    exit 1
  fi

  "${PYTHON}" scripts/tools/verify_dinov2_vision_tower.py \
    --vision-tower "${VISION_TOWER_DIR}" \
    --device npu \
    --input-size 518 \
    --select-layer -2 \
    --expected-tokens 1369 \
    --expected-hidden-size 1024 \
    --expected-num-layers 24 \
    --output-json "${OUTPUT_DIR}/best/vision_tower_verify.json"

  RUN_ID="${RUN_ID}" OUTPUT_DIR="${OUTPUT_DIR}" CLOUD_OUTPUT_PATH="${CLOUD_OUTPUT_PATH}" DINOV2_REGISTRY_RUN_PATH="${DINOV2_REGISTRY_RUN_PATH}" "${PYTHON}" - <<'PY'
import json
import os
import time
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
summary = json.loads((output_dir / "train_summary.json").read_text(encoding="utf-8"))
payload = {
    "status": "passed",
    "run_id": os.environ["RUN_ID"],
    "local_output": str(output_dir),
    "cloud_output": os.environ["CLOUD_OUTPUT_PATH"],
    "registry_output": os.environ["DINOV2_REGISTRY_RUN_PATH"],
    "completed_unix_time": time.time(),
    "global_step": summary.get("global_step"),
    "best_metric": summary.get("best_metric"),
    "best_metric_value": summary.get("best_metric_value"),
    "best_mean_iou": summary.get("best_mean_iou"),
    "vision_tower": str(output_dir / "best" / "vision_tower"),
}
(output_dir / "DI_TRAIN_SUCCESS.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
)
print(f"[di-train-success] {json.dumps(payload, ensure_ascii=True)}", flush=True)
PY

  OUTPUT_DIR="${OUTPUT_DIR}" CLOUD_OUTPUT_PATH="${CLOUD_OUTPUT_PATH}" "${PYTHON}" - <<'PY'
import os
import moxing as mox

source = os.environ["OUTPUT_DIR"]
target = os.environ["CLOUD_OUTPUT_PATH"]
print(f"[di-upload] {source} -> {target}", flush=True)
mox.file.copy_parallel(source, target, threads=128)
required = (
    f"{target}/DI_TRAIN_SUCCESS.json",
    f"{target}/train_summary.json",
    f"{target}/best/metrics.json",
    f"{target}/best/vision_tower/config.json",
    f"{target}/best/vision_tower/preprocessor_config.json",
    f"{target}/best/vision_tower/model.safetensors",
    f"{target}/best/vision_tower_verify.json",
)
missing = [path for path in required if not mox.file.exists(path)]
if missing:
    raise SystemExit(f"Formal DI OBS upload verification failed; missing: {missing}")
print(f"[di-upload] verified {len(required)} required OBS artifacts", flush=True)
PY

  OUTPUT_DIR="${OUTPUT_DIR}" CLOUD_OUTPUT_PATH="${CLOUD_OUTPUT_PATH}" DINOV2_REGISTRY_RUN_PATH="${DINOV2_REGISTRY_RUN_PATH}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import moxing as mox

source = Path(os.environ["OUTPUT_DIR"])
cloud_target = os.environ["CLOUD_OUTPUT_PATH"].rstrip("/")
registry_target = os.environ["DINOV2_REGISTRY_RUN_PATH"].rstrip("/")
if registry_target == cloud_target:
    print(f"[dinov2-registry] reuse primary upload {registry_target}", flush=True)
else:
    print(f"[dinov2-registry] publish verified best vision tower -> {registry_target}", flush=True)
    mox.file.copy_parallel(
        str(source / "best" / "vision_tower"),
        f"{registry_target}/best/vision_tower",
        threads=128,
    )
    for relative in (
        "train_summary.json",
        "best/metrics.json",
        "best/vision_tower_verify.json",
    ):
        mox.file.copy(str(source / relative), f"{registry_target}/{relative}")
    # Publish the success marker last so downstream jobs never select a partial run.
    mox.file.copy(
        str(source / "DI_TRAIN_SUCCESS.json"),
        f"{registry_target}/DI_TRAIN_SUCCESS.json",
    )

required = (
    f"{registry_target}/DI_TRAIN_SUCCESS.json",
    f"{registry_target}/train_summary.json",
    f"{registry_target}/best/metrics.json",
    f"{registry_target}/best/vision_tower/config.json",
    f"{registry_target}/best/vision_tower/preprocessor_config.json",
    f"{registry_target}/best/vision_tower/model.safetensors",
    f"{registry_target}/best/vision_tower_verify.json",
)
missing = [path for path in required if not mox.file.exists(path)]
if missing:
    raise SystemExit(f"DINOv2 registry publish verification failed; missing: {missing}")
print(f"[dinov2-registry] verified {len(required)} registry artifacts", flush=True)
PY

  echo "============================================================"
  echo "Formal DI DINOv2 DINOv3-style LoRA segmentation training PASSED"
  echo "Best vision tower: ${VISION_TOWER_DIR}"
  echo "OBS output:        ${CLOUD_OUTPUT_PATH}"
  echo "Registry run:      ${DINOV2_REGISTRY_RUN_PATH}"
  echo "============================================================"
fi
