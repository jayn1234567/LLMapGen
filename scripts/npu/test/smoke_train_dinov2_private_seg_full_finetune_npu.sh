#!/usr/bin/env bash
set -euo pipefail

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
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29620}
OBS_CACHE=${OBS_CACHE:-/cache/jn}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
MODEL_LOCAL_PATH=${MODEL_LOCAL_PATH:-${OBS_CACHE}/models/facebook_dinov2-large}
DATA_LOCAL_ROOT=${DATA_LOCAL_ROOT:-${OBS_CACHE}/data/rc_lane_segmentation_smoke}
DATASET_LIMIT=${DATASET_LIMIT:-1}
MAX_STEPS=${MAX_STEPS:-20}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-512}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-64}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
RUN_ID=${RUN_ID:-dinov2_private_seg_fullft_smoke_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OBS_CACHE}/outputs/${RUN_ID}}

echo "============================================================"
echo "DINOv2 private segmentation full-finetune smoke"
echo "Repo:          ${REPO_ROOT}"
echo "Python:        ${PYTHON}"
echo "Model OBS:     ${MODEL_OBS_PATH}"
echo "Model local:   ${MODEL_LOCAL_PATH}"
echo "Dataset local: ${DATA_LOCAL_ROOT}"
echo "Dataset count: ${DATASET_LIMIT}"
echo "Output:        ${OUTPUT_DIR}"
echo "NPUs:          ${NPROC_PER_NODE}"
echo "Max steps:     ${MAX_STEPS}"
echo "============================================================"

"${PYTHON}" - <<'PY'
import torch
import transformers
import moxing as mox
import torch_npu

print(f"[preflight] torch={torch.__version__}")
print(f"[preflight] torch_npu={torch_npu.__version__}")
print(f"[preflight] transformers={transformers.__version__}")
print(f"[preflight] npu_available={torch.npu.is_available()} count={torch.npu.device_count()}")
if not torch.npu.is_available():
    raise SystemExit("NPU is not available.")
if not hasattr(mox, "file"):
    raise SystemExit("moxing does not provide mox.file.")
PY

if [ ! -f "${MODEL_LOCAL_PATH}/config.json" ]; then
  mkdir -p "${MODEL_LOCAL_PATH}"
  MODEL_OBS_PATH="${MODEL_OBS_PATH}" MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH}" "${PYTHON}" - <<'PY'
import os
import moxing as mox

source = os.environ["MODEL_OBS_PATH"]
target = os.environ["MODEL_LOCAL_PATH"]
print(f"[model-download] {source} -> {target}", flush=True)
mox.file.copy_parallel(source, target, threads=64)
PY
fi

"${PYTHON}" scripts/tools/download_rc_lane_segmentation_obs.py \
  --output-root "${DATA_LOCAL_ROOT}" \
  --limit "${DATASET_LIMIT}" \
  --threads 64

mapfile -t DATASET_ROOTS < "${DATA_LOCAL_ROOT}/train_roots.txt"
if [ "${#DATASET_ROOTS[@]}" -eq 0 ]; then
  echo "ERROR: no dataset train roots were produced."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
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
  --num_train_epochs 10 \
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
  --eval_every_epochs 100 \
  --gradient_checkpointing true \
  --bf16 true \
  --augment true \
  --device npu

echo "[smoke] completed"
echo "[smoke] Jiangjihua-compatible vision tower: ${OUTPUT_DIR}/best/vision_tower"
