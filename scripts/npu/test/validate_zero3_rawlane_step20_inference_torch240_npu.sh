#!/usr/bin/env bash
set -euo pipefail

# Download all four node-local ZeRO-3 shard sets, merge checkpoint-20 on CPU,
# and run a small real-data NPU inference to prove the checkpoint is usable.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: torch-2.4 inference activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

SHARDED_RUN_OBS_ROOT=${SHARDED_RUN_OBS_ROOT:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/31/09ff56affc6c410ab2cdd27628ac6124/output/ma-job-8cdd0024-64ca-4628-a450-7fb5b84b1ea2}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-20}
EXPECTED_NODES=${EXPECTED_NODES:-4}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-32}

LOCAL_RUN_ROOT=${LOCAL_RUN_ROOT:-/cache/jn/checkpoints/rawlane550k_zero3_step20_validation}
MERGED_CHECKPOINT_DIR=${MERGED_CHECKPOINT_DIR:-${LOCAL_RUN_ROOT}/merged_${CHECKPOINT_NAME}}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/rawlane_local256_550k/rawlane_local256_550k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/rawlane_local256_550k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/rawlane_local256_550k_extract}
DATASET_ROOT=${DATASET_ROOT:-${DATASET_EXTRACT_ROOT}/rawlane_local256_550k}

VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}

MAX_SAMPLES=${MAX_SAMPLES:-8}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
RUN_ID=${RUN_ID:-rawlane550k_zero3_${CHECKPOINT_NAME}_load_infer_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
SMOKE_JSONL=${SMOKE_JSONL:-${OUTPUT_ROOT}/smoke_eval_${MAX_SAMPLES}.jsonl}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export MLLM_LOG_RANK0_ONLY=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}

mkdir -p "${LOCAL_RUN_ROOT}/zero_shards" "${OUTPUT_ROOT}" \
  "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}" "$(dirname "${VISION_TOWER}")"

echo "============================================================"
echo "ZeRO-3 Raw-Lane checkpoint load/inference validation"
echo "OBS run:       ${SHARDED_RUN_OBS_ROOT}"
echo "Checkpoint:    ${CHECKPOINT_NAME}"
echo "Local shards:  ${LOCAL_RUN_ROOT}/zero_shards"
echo "Merged model:  ${MERGED_CHECKPOINT_DIR}"
echo "Samples:       ${MAX_SAMPLES}"
echo "Visible NPU:   ${ASCEND_RT_VISIBLE_DEVICES}"
echo "============================================================"

python - <<'PY'
import deepspeed
import moxing
import torch
import torch_npu

print(
    "[preflight] "
    f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
    f"deepspeed={deepspeed.__version__} npu_available={torch.npu.is_available()}"
)
if not torch.npu.is_available():
    raise SystemExit("NPU is unavailable in the active torch-2.4 environment")
if not hasattr(moxing, "file"):
    raise SystemExit("Huawei moxing-framework with mox.file is required")
PY

for node_rank in $(seq 0 $((EXPECTED_NODES - 1))); do
  local_node_dir=${LOCAL_RUN_ROOT}/zero_shards/node_${node_rank}
  local_checkpoint_dir=${local_node_dir}/${CHECKPOINT_NAME}
  complete_marker=${local_checkpoint_dir}/.obs_download_complete
  if [ -f "${complete_marker}" ]; then
    echo "[shard-download] reuse node_${node_rank}/${CHECKPOINT_NAME}"
    continue
  fi
  mkdir -p "${local_node_dir}" "${local_checkpoint_dir}"
  obs_node_root=${SHARDED_RUN_OBS_ROOT%/}/zero_shards/node_${node_rank}
  echo "[shard-download] ${obs_node_root}/${CHECKPOINT_NAME} -> ${local_checkpoint_dir}"
  OBS_NODE_ROOT="${obs_node_root}" LOCAL_NODE_DIR="${local_node_dir}" CHECKPOINT_NAME="${CHECKPOINT_NAME}" python - <<'PY'
import os
import moxing as mox

obs_node = os.environ["OBS_NODE_ROOT"]
local_node = os.environ["LOCAL_NODE_DIR"]
checkpoint = os.environ["CHECKPOINT_NAME"]
mox.file.copy(f"{obs_node}/zero3_shard_layout.json", f"{local_node}/zero3_shard_layout.json")
mox.file.copy_parallel(f"{obs_node}/{checkpoint}", f"{local_node}/{checkpoint}")
PY
  touch "${complete_marker}"
done

LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT}" CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
EXPECTED_NODES="${EXPECTED_NODES}" EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LOCAL_RUN_ROOT"])
checkpoint_name = os.environ["CHECKPOINT_NAME"]
expected_nodes = int(os.environ["EXPECTED_NODES"])
expected_world_size = int(os.environ["EXPECTED_WORLD_SIZE"])
optimizer_shards = []
summary = {"nodes": {}}

for rank in range(expected_nodes):
    node = root / "zero_shards" / f"node_{rank}"
    layout_path = node / "zero3_shard_layout.json"
    checkpoint = node / checkpoint_name
    if not layout_path.is_file() or not checkpoint.is_dir():
        raise SystemExit(f"Missing node {rank} layout/checkpoint under {node}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if int(layout["expected_nodes"]) != expected_nodes:
        raise SystemExit(f"Unexpected expected_nodes in {layout_path}: {layout}")
    if int(layout["expected_world_size"]) != expected_world_size:
        raise SystemExit(f"Unexpected world size in {layout_path}: {layout}")
    node_optim = sorted(checkpoint.rglob("*optim_states.pt"))
    node_model = sorted(checkpoint.rglob("*model_states.pt"))
    optimizer_shards.extend(node_optim)
    summary["nodes"][str(rank)] = {
        "optimizer_shards": len(node_optim),
        "model_state_files": len(node_model),
    }

if len(optimizer_shards) != expected_world_size:
    raise SystemExit(
        f"Incomplete ZeRO checkpoint: optimizer shards={len(optimizer_shards)}, "
        f"expected={expected_world_size}"
    )
if len({path.name for path in optimizer_shards}) != expected_world_size:
    raise SystemExit("ZeRO optimizer shard names are not globally unique across nodes")

summary["total_optimizer_shards"] = len(optimizer_shards)
summary["status"] = "complete"
print("[shard-validation] " + json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY

if [ ! -s "${MERGED_CHECKPOINT_DIR}/pytorch_model.bin" ] || [ ! -f "${MERGED_CHECKPOINT_DIR}/config.json" ]; then
  echo "[zero3-merge] merging ${CHECKPOINT_NAME}; this is CPU/RAM/disk intensive"
  bash scripts/tools/merge_zero3_multinode_checkpoint.sh \
    "${LOCAL_RUN_ROOT}" "${CHECKPOINT_NAME}" "${MERGED_CHECKPOINT_DIR}"
else
  echo "[zero3-merge] reuse ${MERGED_CHECKPOINT_DIR}"
fi

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "[vision-download] ${VISION_TOWER_OBS_PATH} -> ${VISION_TOWER}"
  VISION_TOWER_OBS_PATH="${VISION_TOWER_OBS_PATH}" VISION_TOWER="${VISION_TOWER}" python - <<'PY'
import os
import moxing as mox

mox.file.copy_parallel(os.environ["VISION_TOWER_OBS_PATH"], os.environ["VISION_TOWER"])
PY
fi

if [ ! -f "${DATASET_ROOT}/phase_a/eval.jsonl" ] || [ ! -d "${DATASET_ROOT}/images" ]; then
  if [ ! -s "${DATASET_ARCHIVE_PATH}" ]; then
    echo "[dataset-download] ${DATASET_OBS_PATH} -> ${DATASET_ARCHIVE_PATH}"
    DATASET_OBS_PATH="${DATASET_OBS_PATH}" DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" python - <<'PY'
import os
import moxing as mox

mox.file.copy(os.environ["DATASET_OBS_PATH"], os.environ["DATASET_ARCHIVE_PATH"])
PY
  fi
  echo "[dataset-extract] ${DATASET_ARCHIVE_PATH} -> ${DATASET_EXTRACT_ROOT}"
  tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}"
fi

if [ ! -f "${DATASET_ROOT}/phase_a/eval.jsonl" ]; then
  DATASET_ROOT=$(python - "${DATASET_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path

roots = [path.parent.parent for path in Path(sys.argv[1]).rglob("phase_a/eval.jsonl")]
roots = sorted(set(roots))
if len(roots) != 1:
    raise SystemExit(f"Unable to resolve one Raw-Lane dataset root: {roots}")
print(roots[0])
PY
  )
fi
EVAL_JSONL=${DATASET_ROOT}/phase_a/eval.jsonl

python - "${EVAL_JSONL}" "${SMOKE_JSONL}" "${MAX_SAMPLES}" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
limit = int(sys.argv[3])
records = []
with source.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(line)
            if len(records) >= limit:
                break
if len(records) != limit:
    raise SystemExit(f"Expected {limit} smoke records, found {len(records)} in {source}")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("".join(records), encoding="utf-8")
print(f"[smoke-data] wrote {len(records)} records to {target}")
PY

INFERENCE_ROOT=${OUTPUT_ROOT}/${CHECKPOINT_NAME}/inference
mkdir -p "${INFERENCE_ROOT}/json"
MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${MERGED_CHECKPOINT_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type dinov2 \
  --input_image_size 518 \
  --disable_deepstack \
  --test-json "${SMOKE_JSONL}" \
  --num-samples "${MAX_SAMPLES}" \
  --image-folder "${DATASET_ROOT}" \
  --prompt-mode dataset \
  --map-task lane_intersection \
  --patch-size 256 \
  --coord-mode auto \
  --coord-range 1000 \
  --conv-template conv_qwen_3_Dinov2_huawei \
  --output-dir "${INFERENCE_ROOT}" \
  --sample-json-dir "${INFERENCE_ROOT}/json" \
  --output-json "${INFERENCE_ROOT}/summary.json" \
  --temperature 0.0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --per-device-infer-batch-size "${PER_DEVICE_INFER_BATCH_SIZE}" \
  --eval-centerline \
  --eval-output-json "${INFERENCE_ROOT}/eval.json"

SAMPLE_RESULTS=$(find "${INFERENCE_ROOT}/json" -type f -name '*.json' | wc -l)
if [ ! -s "${INFERENCE_ROOT}/summary.json" ] || [ "${SAMPLE_RESULTS}" -lt 1 ]; then
  echo "ERROR: inference did not produce a usable summary/sample result" >&2
  exit 1
fi

echo "DI_throughput: 0.00 samples/s/npu"
echo "============================================================"
echo "ZERO-3 CHECKPOINT LOAD + INFERENCE PASSED"
echo "Merged checkpoint: ${MERGED_CHECKPOINT_DIR}"
echo "Inference summary: ${INFERENCE_ROOT}/summary.json"
echo "Inference metrics: ${INFERENCE_ROOT}/eval.json"
echo "Sample JSON files: ${SAMPLE_RESULTS}"
echo "============================================================"
