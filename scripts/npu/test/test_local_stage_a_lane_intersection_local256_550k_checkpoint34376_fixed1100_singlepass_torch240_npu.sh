#!/usr/bin/env bash
set -euo pipefail

# Build and persist one deterministic 1100-sample Dataset V2 evaluation set,
# then evaluate checkpoint-34376 in a single distributed generation pass.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256/local256.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/local256.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/local256_extract}
DATASET_ROOT=${DATASET_ROOT:-${DATASET_EXTRACT_ROOT}/local256}
EVAL_SOURCE_JSONL=${EVAL_SOURCE_JSONL:-${DATASET_ROOT}/phase_a/eval.jsonl}

FIXED_EVAL_ROOT=${FIXED_EVAL_ROOT:-/cache/jn/eval_sets/datasetv2_local256_550k_fixed1100_e300_m300_h300_vh200_seed42_v1}
FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}
FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}
REBUILD_FIXED_EVAL=${REBUILD_FIXED_EVAL:-False}

VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/local256_550k_checkpoint34376}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}}

RUN_LABEL=${RUN_LABEL:-local256_550k_checkpoint34376_fixed1100}
RUN_ID=${RUN_ID:-${RUN_LABEL}_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
INFERENCE_ROOT=${OUTPUT_ROOT}/${CHECKPOINT_NAME}/single_pass
METRICS_ROOT=${OUTPUT_ROOT}/${CHECKPOINT_NAME}/by_difficulty

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

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

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_cache_path() {
  case "$1" in
    /cache/jn/*) ;;
    *)
      echo "ERROR: expected a path below /cache/jn, got: $1" >&2
      exit 2
      ;;
  esac
}

has_checkpoint_weights() {
  local root=$1
  [ -f "${root}/model.safetensors" ] || \
    [ -f "${root}/model.safetensors.index.json" ] || \
    [ -f "${root}/pytorch_model.bin" ] || \
    [ -f "${root}/pytorch_model.bin.index.json" ] || \
    [ -f "${root}/adapter_model.safetensors" ] || \
    [ -f "${root}/adapter_model.bin" ]
}

require_cache_path "${DATASET_ARCHIVE_PATH}"
require_cache_path "${DATASET_EXTRACT_ROOT}"
require_cache_path "${FIXED_EVAL_ROOT}"
require_cache_path "${CHECKPOINT_DIR}"
require_cache_path "${OUTPUT_ROOT}"

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: reusable DINOv2 model is incomplete: ${VISION_TOWER}" >&2
  exit 2
fi

if [ ! -f "${EVAL_SOURCE_JSONL}" ] || [ ! -d "${DATASET_ROOT}/images" ]; then
  mkdir -p "$(dirname "${DATASET_ARCHIVE_PATH}")" "${DATASET_EXTRACT_ROOT}"
  if [ ! -s "${DATASET_ARCHIVE_PATH}" ]; then
    echo "[dataset] downloading ${DATASET_OBS_PATH} -> ${DATASET_ARCHIVE_PATH}"
    python - "${DATASET_OBS_PATH}" "${DATASET_ARCHIVE_PATH}" <<'PY'
import sys
import moxing as mox

mox.file.copy(sys.argv[1], sys.argv[2])
PY
  else
    echo "[dataset] reuse archive: ${DATASET_ARCHIVE_PATH}"
  fi
  echo "[dataset] extracting missing files below ${DATASET_EXTRACT_ROOT}"
  tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}"
fi

if [ ! -f "${EVAL_SOURCE_JSONL}" ]; then
  echo "ERROR: Dataset V2 evaluation JSONL not found: ${EVAL_SOURCE_JSONL}" >&2
  exit 2
fi
if [ ! -d "${DATASET_ROOT}/images" ]; then
  echo "ERROR: Dataset V2 image root not found: ${DATASET_ROOT}/images" >&2
  exit 2
fi

validate_fixed_eval() {
  python - "${FIXED_EVAL_ROOT}" "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

split_root = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected = {"easy": 300, "medium": 300, "hard": 300, "very_hard": 200}
seen = set()

for difficulty, expected_count in expected.items():
    path = split_root / f"{difficulty}.jsonl"
    if not path.is_file():
        raise SystemExit(1)
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
            if not sample_id or sample_id in seen:
                raise SystemExit(f"Invalid or duplicate sample id at {path}:{line_number}: {sample_id!r}")
            seen.add(sample_id)
            image = record.get("image", record.get("images", ""))
            if isinstance(image, list):
                image = image[0] if image else ""
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            if not image_path.is_file():
                raise SystemExit(f"Missing fixed-eval image for {sample_id}: {image_path}")
            count += 1
    if count != expected_count:
        raise SystemExit(f"Expected {expected_count} records in {path}, found {count}")

all_path = split_root / "all_selected.jsonl"
manifest_path = split_root / "manifest.jsonl"
summary_path = split_root / "summary.json"
if not all_path.is_file() or not manifest_path.is_file() or not summary_path.is_file():
    raise SystemExit(1)
all_count = sum(1 for line in all_path.open(encoding="utf-8") if line.strip())
if all_count != 1100 or len(seen) != 1100:
    raise SystemExit(f"Expected 1100 fixed records, found all={all_count}, unique={len(seen)}")
print(f"[fixed-eval] validated persistent set: {split_root} ({len(seen)} unique samples)")
PY
}

if validate_fixed_eval; then
  echo "[fixed-eval] reusing the exact saved evaluation set"
else
  if [ -e "${FIXED_EVAL_ROOT}" ] && ! is_true "${REBUILD_FIXED_EVAL}"; then
    echo "ERROR: fixed evaluation directory exists but failed validation: ${FIXED_EVAL_ROOT}" >&2
    echo "Set REBUILD_FIXED_EVAL=True only if you intentionally want to replace its identity." >&2
    exit 2
  fi

  BUILD_ROOT="${FIXED_EVAL_ROOT}.building.$$"
  require_cache_path "${BUILD_ROOT}"
  rm -rf "${BUILD_ROOT}"
  mkdir -p "${BUILD_ROOT}"
  echo "[fixed-eval] selecting deterministic samples from ${EVAL_SOURCE_JSONL}"
  python scripts/tools/build_difficulty_eval_splits.py \
    --input-jsonl "${EVAL_SOURCE_JSONL}" \
    --output-dir "${BUILD_ROOT}" \
    --samples-per-difficulty 0 \
    --samples-per-difficulty-spec "${FIXED_EVAL_COUNTS}" \
    --difficulties easy medium hard very_hard \
    --seed "${FIXED_EVAL_SEED}" \
    --coord-mode auto \
    --coord-range 1000

  rm -rf "${FIXED_EVAL_ROOT}"
  mv "${BUILD_ROOT}" "${FIXED_EVAL_ROOT}"
  validate_fixed_eval

  python - "${FIXED_EVAL_ROOT}" "${EVAL_SOURCE_JSONL}" "${FIXED_EVAL_SEED}" "${FIXED_EVAL_COUNTS}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

paths = [source]
paths.extend(root / f"{name}.jsonl" for name in ("easy", "medium", "hard", "very_hard"))
paths.extend([root / "all_selected.jsonl", root / "manifest.jsonl"])
identity = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_jsonl": str(source),
    "seed": int(sys.argv[3]),
    "requested_counts": sys.argv[4],
    "total_samples": 1100,
    "sha256": {str(path): sha256(path) for path in paths},
    "reuse_policy": "Do not rebuild. Reuse these exact JSONL files for every comparison model.",
}
(root / "fixed_eval_identity.json").write_text(
    json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(identity, ensure_ascii=False, indent=2))
PY
fi

if ! has_checkpoint_weights "${CHECKPOINT_DIR}"; then
  mkdir -p "${CHECKPOINT_DIR}"
  echo "[checkpoint] downloading ${CHECKPOINT_OBS_PATH} -> ${CHECKPOINT_DIR}"
  python - "${CHECKPOINT_OBS_PATH}" "${CHECKPOINT_DIR}" <<'PY'
import sys
import moxing as mox

mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
else
  echo "[checkpoint] reuse ${CHECKPOINT_DIR}"
fi
if ! has_checkpoint_weights "${CHECKPOINT_DIR}"; then
  echo "ERROR: no supported model weights found in ${CHECKPOINT_DIR}" >&2
  exit 2
fi

mkdir -p "${INFERENCE_ROOT}/json" "${METRICS_ROOT}"

echo "============================================================"
echo "Dataset root:      ${DATASET_ROOT}"
echo "Fixed eval set:    ${FIXED_EVAL_ROOT}"
echo "Fixed counts:      ${FIXED_EVAL_COUNTS} (total=1100)"
echo "Fixed seed:        ${FIXED_EVAL_SEED}"
echo "Checkpoint:        ${CHECKPOINT_DIR}"
echo "DINOv2:            ${VISION_TOWER}"
echo "Output:            ${OUTPUT_ROOT}"
echo "Visible NPUs:      ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Inference mode:    one ${NPROC_PER_NODE}-NPU pass for all 1100 samples"
echo "============================================================"

MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}

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
  --test-json "${FIXED_EVAL_ROOT}/all_selected.jsonl" \
  --num-samples 0 \
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
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --eval-meter-per-pixel 0.2 \
  --eval-buffer-size 1.0 \
  --eval-match-threshold 0.33 \
  --eval-centerline \
  --eval-output-json "${INFERENCE_ROOT}/eval.json"

python scripts/tools/split_single_pass_eval_by_difficulty.py \
  --summary-json "${INFERENCE_ROOT}/summary.json" \
  --split-root "${FIXED_EVAL_ROOT}" \
  --output-root "${METRICS_ROOT}" \
  --expected-counts "${FIXED_EVAL_COUNTS}" \
  --meter-per-pixel 0.2 \
  --buffer-size 1.0 \
  --match-threshold 0.33 \
  --intersection-iou-threshold 0.5

VIS_LIMIT=${VIS_LIMIT:-0}
if [ "${VIS_LIMIT}" -gt 0 ]; then
  python scripts/tools/visualize_centerline.py \
    --input-dir "${INFERENCE_ROOT}" \
    --image-folder "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/${CHECKPOINT_NAME}/viz" \
    --map-task lane_intersection \
    --max-samples "${VIS_LIMIT}" \
    --no-eval-centerline \
    --skip-whole-map-viz
fi

echo "============================================================"
echo "FIXED-1100 EVALUATION COMPLETE"
echo "Saved eval set:     ${FIXED_EVAL_ROOT}"
echo "Eval identity:      ${FIXED_EVAL_ROOT}/fixed_eval_identity.json"
echo "Combined summary:   ${INFERENCE_ROOT}/summary.json"
echo "Combined metrics:   ${METRICS_ROOT}/all_selected/eval.json"
echo "Difficulty metrics: ${METRICS_ROOT}/{easy,medium,hard,very_hard}/eval.json"
echo "Future models must reuse: ${FIXED_EVAL_ROOT}/all_selected.jsonl"
echo "============================================================"
