#!/usr/bin/env bash
set -euo pipefail

# Convert the fixed JJH33W records to the current prompt contract without
# matching them against Dataset V2. Run all 1000 records in one 8-NPU pass,
# then report per-bucket and aggregate geometry metrics.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ENV_DIR}/activate_mllm_infer_torch240.sh
REFERENCE_SPLIT_ROOT=${REFERENCE_SPLIT_ROOT:-/cache/jn/eval_sets/jjh33w_1000_e300_m300_h300_vh100_seed42_v1}
PROMPT_TEMPLATE_JSONL=${PROMPT_TEMPLATE_JSONL:-}
CONVERTED_SPLIT_ROOT=${CONVERTED_SPLIT_ROOT:-/cache/jn/eval_sets/jjh33w_1000_current_prompt_legacy_gt_v1}
VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/}

LEGACY_DATASET_OBS_PATH=${LEGACY_DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}
LEGACY_DATA_CACHE=${LEGACY_DATA_CACHE:-/cache/jn/data/jjh33w_fixed_eval}
LEGACY_DATA_ARCHIVE=${LEGACY_DATA_ARCHIVE:-${LEGACY_DATA_CACHE}/data_line_samples_33w.zip}
LEGACY_DATA_EXTRACT_ROOT=${LEGACY_DATA_EXTRACT_ROOT:-${LEGACY_DATA_CACHE}/extract}
LEGACY_DATASET_ROOT=${LEGACY_DATASET_ROOT:-}

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
if [ -n "${PROMPT_TEMPLATE_JSONL}" ] && [ ! -f "${PROMPT_TEMPLATE_JSONL}" ]; then
  echo "ERROR: current Dataset V2 prompt template JSONL was not found: ${PROMPT_TEMPLATE_JSONL}" >&2
  exit 2
fi

RUN_ID=${RUN_ID:-checkpoint34376_jjh33w_current_prompt_legacy_gt_singlepass_$(date -u +%Y%m%d_%H%M%S)}
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
mkdir -p "${CHECKPOINT_DIR}" "${INFERENCE_ROOT}/json" "${METRICS_ROOT}" "${LEGACY_DATA_CACHE}"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
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
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
  set -u
fi

# The fixed JSONL stores relative legacy image paths. Cache the matching old
# image tree once; subsequent runs reuse it.
if [ -z "${LEGACY_DATASET_ROOT}" ]; then
  LEGACY_DATASET_ROOT=$(python - "${LEGACY_DATA_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [root / "data_line_samples_33w", root]
candidates.extend(path for path in root.rglob("data_line_samples_33w") if path.is_dir()) if root.is_dir() else None
for candidate in candidates:
    if (candidate / "images").is_dir():
        print(candidate)
        raise SystemExit(0)
PY
  )
fi

if [ -z "${LEGACY_DATASET_ROOT}" ] || [ ! -d "${LEGACY_DATASET_ROOT}/images" ]; then
  if [ ! -s "${LEGACY_DATA_ARCHIVE}" ]; then
    echo "[legacy-data] downloading ${LEGACY_DATASET_OBS_PATH} -> ${LEGACY_DATA_ARCHIVE}"
    python - "${LEGACY_DATASET_OBS_PATH}" "${LEGACY_DATA_ARCHIVE}" <<'PY'
import sys
import moxing as mox

mox.file.copy(sys.argv[1], sys.argv[2])
PY
  else
    echo "[legacy-data] reuse archive: ${LEGACY_DATA_ARCHIVE}"
  fi
  case "${LEGACY_DATA_EXTRACT_ROOT}" in
    /cache/jn/*) ;;
    *)
      echo "ERROR: refusing extraction cleanup outside /cache/jn: ${LEGACY_DATA_EXTRACT_ROOT}" >&2
      exit 2
      ;;
  esac
  rm -rf "${LEGACY_DATA_EXTRACT_ROOT}"
  mkdir -p "${LEGACY_DATA_EXTRACT_ROOT}"
  echo "[legacy-data] extracting -> ${LEGACY_DATA_EXTRACT_ROOT}"
  python - "${LEGACY_DATA_ARCHIVE}" "${LEGACY_DATA_EXTRACT_ROOT}" <<'PY'
import shutil
import sys

shutil.unpack_archive(sys.argv[1], sys.argv[2])
PY
  LEGACY_DATASET_ROOT=$(python - "${LEGACY_DATA_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [root / "data_line_samples_33w", root]
candidates.extend(path for path in root.rglob("data_line_samples_33w") if path.is_dir())
for candidate in candidates:
    if (candidate / "images").is_dir():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(f"Unable to resolve legacy dataset root below {root}")
PY
  )
fi

echo "============================================================"
echo "Fixed reference:  ${REFERENCE_SPLIT_ROOT}"
echo "Prompt template:  ${PROMPT_TEMPLATE_JSONL:-<built-in Dataset V2 local256-550k prompt>}"
echo "Legacy images:    ${LEGACY_DATASET_ROOT}"
echo "Converted set:    ${CONVERTED_SPLIT_ROOT}"
echo "Reused DINOv2:    ${VISION_TOWER}"
echo "Checkpoint OBS:   ${CHECKPOINT_OBS_PATH}"
echo "Output:           ${OUTPUT_ROOT}"
echo "Visible NPUs:     ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Mode:             one ${NPROC_PER_NODE}-NPU inference pass for all 1000 samples"
echo "============================================================"

CONVERT_ARGS=(
  --reference-dir "${REFERENCE_SPLIT_ROOT}"
  --output-dir "${CONVERTED_SPLIT_ROOT}"
  --expected-counts easy=300,medium=300,hard=300,very_hard=100
  --require-norm1000
  --image-source-root "${LEGACY_DATASET_ROOT}"
  --materialize-images copy
  --missing-image-policy skip
)
if [ -n "${PROMPT_TEMPLATE_JSONL}" ]; then
  CONVERT_ARGS+=(--prompt-template-jsonl "${PROMPT_TEMPLATE_JSONL}")
fi
python scripts/tools/convert_legacy_fixed_eval_schema.py "${CONVERT_ARGS[@]}"

python - "${CONVERTED_SPLIT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

split_root = Path(sys.argv[1])
image_root = split_root
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 100}
seen = set()
missing_images = []
actual = {}
for name in expected:
    path = split_root / f"{name}.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    actual[name] = len(records)
    for record in records:
        sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
        if not sample_id or sample_id in seen:
            raise SystemExit(f"Invalid or duplicate converted sample id: {sample_id!r}")
        seen.add(sample_id)
        image = record.get("image", record.get("images", ""))
        if isinstance(image, list):
            image = image[0] if image else ""
        image_path = Path(str(image))
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.is_file():
            missing_images.append((sample_id, str(image_path)))
if missing_images:
    raise SystemExit(f"Missing {len(missing_images)} legacy images; examples={missing_images[:5]}")
if not seen:
    raise SystemExit("Converted fixed evaluation set is empty")
print(f"[single-pass-eval] validated converted fixed set and images: {len(seen)} records; buckets={actual}")
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
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type dinov2 \
  --input_image_size 518 \
  --disable_deepstack \
  --test-json "${CONVERTED_SPLIT_ROOT}/all_selected.jsonl" \
  --num-samples 0 \
  --image-folder "${CONVERTED_SPLIT_ROOT}" \
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
  --split-root "${CONVERTED_SPLIT_ROOT}" \
  --output-root "${METRICS_ROOT}" \
  --expected-counts "" \
  --meter-per-pixel 0.2 \
  --buffer-size 1.0 \
  --match-threshold 0.33 \
  --intersection-iou-threshold 0.5

VIS_LIMIT=${VIS_LIMIT:-0}
if [ "${VIS_LIMIT}" -gt 0 ]; then
  python scripts/tools/visualize_centerline.py \
    --input-dir "${INFERENCE_ROOT}" \
    --image-folder "${CONVERTED_SPLIT_ROOT}" \
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
echo "Conversion report:  ${CONVERTED_SPLIT_ROOT}/conversion_report.json"
echo "Type policy:        missing legacy semantic types are skipped"
echo "============================================================"
