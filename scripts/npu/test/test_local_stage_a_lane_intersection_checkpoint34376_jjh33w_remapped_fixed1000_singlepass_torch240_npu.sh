#!/usr/bin/env bash
set -euo pipefail

# Remap the fixed JJH33W sample identities to the current local256 prompt/GT,
# run all 1000 records in one 8-NPU pass, then report per-bucket and total metrics.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ENV_DIR}/activate_mllm_infer_torch240.sh
DATASET_PATH=${DATASET_PATH:-/cache/jn/data/local256_extract/local256}
REFERENCE_SPLIT_ROOT=${REFERENCE_SPLIT_ROOT:-/cache/jn/eval_sets/jjh33w_1000_e300_m300_h300_vh100_seed42_v1}
REMAPPED_SPLIT_ROOT=${REMAPPED_SPLIT_ROOT:-/cache/jn/eval_sets/jjh33w_1000_remapped_local256_current_prompt_v1}
VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: reusable DINOv2 model is incomplete: ${VISION_TOWER}" >&2
  exit 2
fi
if [ ! -d "${DATASET_PATH}/phase_a" ]; then
  echo "ERROR: current local256 dataset was not found: ${DATASET_PATH}" >&2
  exit 2
fi

RUN_ID=${RUN_ID:-checkpoint34376_jjh33w_remapped_fixed1000_singlepass_$(date -u +%Y%m%d_%H%M%S)}
RUNTIME_ROOT=${RUNTIME_ROOT:-/cache/jn/fresh_assets/${RUN_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
CHECKPOINT_DIR=${RUNTIME_ROOT}/checkpoint-34376
INFERENCE_ROOT=${OUTPUT_ROOT}/checkpoint-34376/single_pass
METRICS_ROOT=${OUTPUT_ROOT}/checkpoint-34376/by_difficulty

for target in "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"; do
  case "${target}" in
    /cache/jn/*) ;;
    *)
      echo "ERROR: refusing recursive cleanup outside /cache/jn: ${target}" >&2
      exit 2
      ;;
  esac
done
rm -rf "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"
mkdir -p "${CHECKPOINT_DIR}" "${INFERENCE_ROOT}/json" "${METRICS_ROOT}" "${REMAPPED_SPLIT_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export MLLM_LOG_RANK0_ONLY=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

echo "============================================================"
echo "Fixed reference: ${REFERENCE_SPLIT_ROOT}"
echo "Current dataset: ${DATASET_PATH}"
echo "Remapped set:    ${REMAPPED_SPLIT_ROOT}"
echo "Reused DINOv2:   ${VISION_TOWER}"
echo "Checkpoint OBS:  ${CHECKPOINT_OBS_PATH}"
echo "Output:          ${OUTPUT_ROOT}"
echo "Mode:            one 8-NPU inference pass for all 1000 samples"
echo "============================================================"

# Exact image identities and original difficulty labels come from JJH33W.
# Prompt, output schema, GT, metadata, and image path come from current local256.
TARGET_SPLITS=()
for split in eval test train; do
  if [ -f "${DATASET_PATH}/phase_a/${split}.jsonl" ] || [ -f "${DATASET_PATH}/${split}.jsonl" ]; then
    TARGET_SPLITS+=("${split}")
  fi
done
if [ "${#TARGET_SPLITS[@]}" -eq 0 ]; then
  echo "ERROR: no eval/test/train JSONL found under ${DATASET_PATH}" >&2
  exit 2
fi
echo "[single-pass-eval] target splits used for exact-ID mapping: ${TARGET_SPLITS[*]}"

python scripts/tools/remap_fixed_eval_to_dataset.py \
  --reference-dir "${REFERENCE_SPLIT_ROOT}" \
  --target-dataset-root "${DATASET_PATH}" \
  --output-dir "${REMAPPED_SPLIT_ROOT}" \
  --target-phase phase_a \
  --scan-target-splits "${TARGET_SPLITS[@]}" \
  --allowed-target-splits "${TARGET_SPLITS[@]}" \
  --patch-size 256 \
  --ground-truth-source target \
  --min-output-match-ratio 1.0 \
  --require-all

python - "${REMAPPED_SPLIT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 100}
seen = set()
for name, count in expected.items():
    path = root / f"{name}.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(records) != count:
        raise SystemExit(f"Expected {count} remapped {name} records, found {len(records)}")
    for record in records:
        sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
        if not sample_id or sample_id in seen:
            raise SystemExit(f"Invalid or duplicate remapped sample id: {sample_id!r}")
        seen.add(sample_id)
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise SystemExit(f"Current prompt/GT conversations missing for: {sample_id}")
if len(seen) != 1000:
    raise SystemExit(f"Expected exactly 1000 remapped records, found {len(seen)}")
print(f"[single-pass-eval] validated current-format fixed set: {len(seen)} records")
PY

echo "[single-pass-eval] downloading checkpoint -> ${CHECKPOINT_DIR}"
python - "${CHECKPOINT_OBS_PATH}" "${CHECKPOINT_DIR}" <<'PY'
import sys
import moxing as mox

source, destination = sys.argv[1:3]
mox.file.copy_parallel(source, destination)
print(f"[single-pass-eval] checkpoint downloaded: {destination}")
PY

python - "${CHECKPOINT_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = {
    "model.safetensors", "model.safetensors.index.json",
    "pytorch_model.bin", "pytorch_model.bin.index.json",
    "adapter_model.safetensors", "adapter_model.bin",
}
found = sorted(path.name for path in root.iterdir() if path.is_file() and path.name in names)
if not found:
    raise SystemExit(f"No supported model weights found in downloaded checkpoint: {root}")
print(f"[single-pass-eval] checkpoint weights: {found}")
PY

MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}

echo "[single-pass-eval] starting one distributed generation pass on port ${MASTER_PORT}"
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type dinov2 \
  --input_image_size 518 \
  --disable_deepstack \
  --test-json "${REMAPPED_SPLIT_ROOT}/all_selected.jsonl" \
  --num-samples 0 \
  --image-folder "${DATASET_PATH}" \
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
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --eval-meter-per-pixel 0.2 \
  --eval-buffer-size 1.0 \
  --eval-match-threshold 0.33 \
  --eval-intersection-iou-threshold 0.5 \
  --eval-centerline \
  --eval-output-json "${INFERENCE_ROOT}/eval.json"

python scripts/tools/split_single_pass_eval_by_difficulty.py \
  --summary-json "${INFERENCE_ROOT}/summary.json" \
  --split-root "${REMAPPED_SPLIT_ROOT}" \
  --output-root "${METRICS_ROOT}" \
  --expected-counts easy=300,medium=300,hard=300,very_hard=100 \
  --meter-per-pixel 0.2 \
  --buffer-size 1.0 \
  --match-threshold 0.33 \
  --intersection-iou-threshold 0.5

VIS_LIMIT=${VIS_LIMIT:-0}
if [ "${VIS_LIMIT}" -gt 0 ]; then
  python scripts/tools/visualize_centerline.py \
    --input-dir "${INFERENCE_ROOT}" \
    --image-folder "${DATASET_PATH}" \
    --output-dir "${OUTPUT_ROOT}/checkpoint-34376/viz" \
    --map-task lane_intersection \
    --max-samples "${VIS_LIMIT}" \
    --no-eval-centerline \
    --skip-whole-map-viz
fi

echo "============================================================"
echo "SINGLE-PASS FIXED-1000 EVALUATION COMPLETE"
echo "Combined inference: ${INFERENCE_ROOT}/summary.json"
echo "Combined metrics:   ${METRICS_ROOT}/all_selected/eval.json"
echo "Difficulty metrics: ${METRICS_ROOT}/{easy,medium,hard,very_hard}/eval.json"
echo "Mapping report:     ${REMAPPED_SPLIT_ROOT}/mapping_report.json"
echo "============================================================"
