#!/usr/bin/env bash
set -euo pipefail

# RC end-to-end inference:
# raw *_inter.tif -> context512/center-ROI256 JSONL -> raw per-patch MLLM JSON.
# The downstream DI main.sh accepts the raw JSON directory as INFER_RESULT and
# performs its own infer_result_format.py conversion before rule evaluation.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}

E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
E2E_WORK_ROOT=${E2E_WORK_ROOT:-/cache/jn/e2e_eval}
E2E_ARCHIVE_PATH=${E2E_ARCHIVE_PATH:-${E2E_WORK_ROOT}/e2e_data.zip}
E2E_RAW_ROOT=${E2E_RAW_ROOT:-${E2E_WORK_ROOT}/raw_e2e_data}
E2E_VIEW_MODE=${E2E_VIEW_MODE:-context512_roi256}
E2E_TARGET_SIZE=${E2E_TARGET_SIZE:-256}
E2E_CONTEXT_SIZE=${E2E_CONTEXT_SIZE:-512}
E2E_PROMPT_PROFILE=${E2E_PROMPT_PROFILE:-current}
E2E_INPUT_RASTER=${E2E_INPUT_RASTER:-inter}
E2E_DATASET_ROOT=${E2E_DATASET_ROOT:-${E2E_WORK_ROOT}/e2e_data_${E2E_VIEW_MODE}}
REBUILD_E2E_DATASET=${REBUILD_E2E_DATASET:-False}
BLACK_RATIO_THRESHOLD=${BLACK_RATIO_THRESHOLD:-1.0}
VALIDATE_RASTER_ALIGNMENT=${VALIDATE_RASTER_ALIGNMENT:-True}
RASTER_ALIGNMENT_REPORT=${RASTER_ALIGNMENT_REPORT:-${E2E_WORK_ROOT}/raster_alignment_report.json}

CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-12504}
CHECKPOINT_OBS_PATH=${CHECKPOINT_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/24/4f735c63da7a4f86829b26246143e219/output/ma-job-81341482-55b8-4c28-887b-0e4166776561/checkpoint-12504/}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/context512_roi256_checkpoint12504}
CHECKPOINT_POINTER_FILE=${CHECKPOINT_POINTER_FILE:-${CHECKPOINT_CACHE_ROOT}/.${CHECKPOINT_NAME}.validated_dir}
if [ -z "${CHECKPOINT_DIR:-}" ] && [ -s "${CHECKPOINT_POINTER_FILE}" ]; then
  CHECKPOINT_DIR=$(head -n 1 "${CHECKPOINT_POINTER_FILE}")
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}}
REFRESH_CHECKPOINT=${REFRESH_CHECKPOINT:-False}

VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}

RUN_ID=${RUN_ID:-rc_e2e_context512_roi256_checkpoint12504_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
INFERENCE_ROOT=${OUTPUT_ROOT}/inference
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${INFERENCE_ROOT}/json}
INFER_RESULT_OBS_PATH=${INFER_RESULT_OBS_PATH:-}

NUM_SAMPLES=${NUM_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}
MAX_TIFS=${MAX_TIFS:-0}
MAX_PATCHES=${MAX_PATCHES:-0}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

has_extracted_e2e_data() {
  [ -f "${E2E_RAW_ROOT}/.extract_complete" ] || \
    find "${E2E_RAW_ROOT}" -type d -name rc_one_patch_release -print -quit 2>/dev/null | grep -q .
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

validate_checkpoint_weights() {
  python scripts/tools/validate_checkpoint_files.py --checkpoint-dir "$1"
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
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

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export MLLM_LOG_RANK0_ONLY=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}

mkdir -p "${E2E_WORK_ROOT}" "${CHECKPOINT_DIR}" "$(dirname "${VISION_TOWER}")" "${RAW_RESULT_DIR}"

if has_extracted_e2e_data; then
  echo "[e2e] reuse extracted raw data: ${E2E_RAW_ROOT}"
else
  if [ ! -s "${E2E_ARCHIVE_PATH}" ]; then
    echo "[e2e] downloading ${E2E_DATA_OBS_PATH} -> ${E2E_ARCHIVE_PATH}"
    python - "${E2E_DATA_OBS_PATH}" "${E2E_ARCHIVE_PATH}" <<'PY'
import sys
import moxing as mox

mox.file.copy(sys.argv[1], sys.argv[2])
PY
  else
    echo "[e2e] reuse archive: ${E2E_ARCHIVE_PATH}"
  fi

  echo "[e2e] extracting ${E2E_ARCHIVE_PATH} -> ${E2E_RAW_ROOT}"
  mkdir -p "${E2E_RAW_ROOT}"
  python - "${E2E_ARCHIVE_PATH}" "${E2E_RAW_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with zipfile.ZipFile(archive) as handle:
    handle.extractall(destination)
(destination / ".extract_complete").write_text("ok\n", encoding="utf-8")
PY
fi

if is_true "${VALIDATE_RASTER_ALIGNMENT}"; then
  echo "[e2e] validating inter/lane TIF pixel-grid alignment"
  python scripts/tools/validate_rc_e2e_raster_alignment.py \
    --input-root "${E2E_RAW_ROOT}" \
    --patch-size "${E2E_TARGET_SIZE}" \
    --output-json "${RASTER_ALIGNMENT_REPORT}"
else
  echo "[e2e] WARNING: raster alignment validation is disabled"
fi

if is_true "${REBUILD_E2E_DATASET}" || [ ! -s "${E2E_DATASET_ROOT}/infer.jsonl" ]; then
  echo "[e2e] building ${E2E_VIEW_MODE} inference dataset"
  python scripts/tools/prepare_rc_e2e_inference_dataset.py \
    --input-root "${E2E_RAW_ROOT}" \
    --output-root "${E2E_DATASET_ROOT}" \
    --view-mode "${E2E_VIEW_MODE}" \
    --target-size "${E2E_TARGET_SIZE}" \
    --context-size "${E2E_CONTEXT_SIZE}" \
    --stride "${E2E_TARGET_SIZE}" \
    --coord-range 1000 \
    --prompt-profile "${E2E_PROMPT_PROFILE}" \
    --input-raster "${E2E_INPUT_RASTER}" \
    --black-ratio-threshold "${BLACK_RATIO_THRESHOLD}" \
    --include-intersections \
    --max-tifs "${MAX_TIFS}" \
    --max-patches "${MAX_PATCHES}"
else
  echo "[e2e] reuse inference dataset: ${E2E_DATASET_ROOT}"
fi

if [ ! -s "${E2E_DATASET_ROOT}/infer.jsonl" ]; then
  echo "ERROR: E2E inference JSONL was not produced." >&2
  exit 2
fi

python - "${E2E_DATASET_ROOT}/dataset_summary.json" "${E2E_INPUT_RASTER}" "${E2E_PROMPT_PROFILE}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
if not summary_path.is_file():
    raise FileNotFoundError(f"E2E dataset summary not found: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_raster = sys.argv[2]
expected_prompt = sys.argv[3]
if summary.get("input_raster", "inter") != expected_raster:
    raise ValueError(
        f"Cached E2E dataset input_raster={summary.get('input_raster')!r}, expected {expected_raster!r}. "
        "Set REBUILD_E2E_DATASET=True or use a separate E2E_DATASET_ROOT."
    )
if summary.get("prompt_profile") != expected_prompt:
    raise ValueError(
        f"Cached E2E dataset prompt_profile={summary.get('prompt_profile')!r}, expected {expected_prompt!r}. "
        "Set REBUILD_E2E_DATASET=True or use a separate E2E_DATASET_ROOT."
    )
print(f"[e2e] dataset profile passed: input_raster={expected_raster} prompt={expected_prompt}")
PY

checkpoint_ready=False
if ! is_true "${REFRESH_CHECKPOINT}" && has_checkpoint_weights "${CHECKPOINT_DIR}"; then
  echo "[e2e] validating cached checkpoint: ${CHECKPOINT_DIR}"
  if validate_checkpoint_weights "${CHECKPOINT_DIR}"; then
    checkpoint_ready=True
  else
    echo "WARNING: cached checkpoint is incomplete or corrupt; a fresh copy will be downloaded." >&2
  fi
fi

if ! is_true "${checkpoint_ready}"; then
  if [ -e "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}.validated_$(date -u +%Y%m%d_%H%M%S)"
  fi
  echo "[e2e] downloading checkpoint ${CHECKPOINT_OBS_PATH} -> ${CHECKPOINT_DIR}"
  python - "${CHECKPOINT_OBS_PATH}" "${CHECKPOINT_DIR}" <<'PY'
import sys
import moxing as mox

mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
  echo "[e2e] validating downloaded checkpoint: ${CHECKPOINT_DIR}"
  if ! validate_checkpoint_weights "${CHECKPOINT_DIR}"; then
    echo "ERROR: downloaded checkpoint is incomplete or corrupt: ${CHECKPOINT_DIR}" >&2
    echo "ERROR: verify the checkpoint files in OBS before retrying: ${CHECKPOINT_OBS_PATH}" >&2
    exit 2
  fi
  mkdir -p "${CHECKPOINT_CACHE_ROOT}"
  printf '%s\n' "${CHECKPOINT_DIR}" > "${CHECKPOINT_POINTER_FILE}"
else
  echo "[e2e] reuse validated checkpoint: ${CHECKPOINT_DIR}"
fi

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "[e2e] downloading DINOv2 ${VISION_TOWER_OBS_PATH} -> ${VISION_TOWER}"
  python - "${VISION_TOWER_OBS_PATH}" "${VISION_TOWER}" <<'PY'
import sys
import moxing as mox

mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
else
  echo "[e2e] reuse DINOv2: ${VISION_TOWER}"
fi
if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: DINOv2 config.json not found below ${VISION_TOWER}" >&2
  exit 2
fi

MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}

echo "============================================================"
echo "RC E2E data:       ${E2E_DATASET_ROOT}"
echo "Checkpoint:        ${CHECKPOINT_DIR}"
echo "DINOv2:            ${VISION_TOWER}"
echo "Output:            ${OUTPUT_ROOT}"
echo "Visible NPUs:      ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Per-device batch:  ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Dataset view:      ${E2E_VIEW_MODE}"
echo "Prompt profile:    ${E2E_PROMPT_PROFILE}"
echo "Input raster:      ${E2E_INPUT_RASTER}"
echo "Target frame:      ${E2E_TARGET_SIZE}x${E2E_TARGET_SIZE}, norm1000"
echo "Source image:      ${E2E_CONTEXT_SIZE}x${E2E_CONTEXT_SIZE} -> DINOv2 518x518"
echo "============================================================"
echo "DI_throughput: 0.00 samples/s/npu"

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
  --test-json "${E2E_DATASET_ROOT}/infer.jsonl" \
  --num-samples "${NUM_SAMPLES}" \
  --image-folder "${E2E_DATASET_ROOT}" \
  --prompt-mode dataset \
  --map-task lane_intersection \
  --patch-size "${E2E_TARGET_SIZE}" \
  --coord-mode auto \
  --coord-range 1000 \
  --conv-template conv_qwen_3_Dinov2_huawei \
  --output-dir "${INFERENCE_ROOT}" \
  --sample-json-dir "${RAW_RESULT_DIR}" \
  --output-json "${INFERENCE_ROOT}/summary.json" \
  --temperature 0.0 \
  --per-device-infer-batch-size "${PER_DEVICE_INFER_BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python - "${RAW_RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
paths = sorted(source.glob("*.json"))
if not paths:
    raise FileNotFoundError(f"No per-patch inference JSON files found below {source}")

errors = []
for path in paths:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("image"), str) or not payload["image"]:
            errors.append(f"{path.name}: missing image")
        if not isinstance(payload.get("prediction_json"), str):
            errors.append(f"{path.name}: missing prediction_json")
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")
if errors:
    raise ValueError("Invalid raw inference JSON output: " + "; ".join(errors[:10]))
print(f"[e2e] validated raw per-patch inference JSON files: {len(paths)}")
PY

if [ -n "${INFER_RESULT_OBS_PATH}" ]; then
  echo "[e2e] uploading raw inference JSON ${RAW_RESULT_DIR} -> ${INFER_RESULT_OBS_PATH}"
  python - "${RAW_RESULT_DIR}" "${INFER_RESULT_OBS_PATH}" <<'PY'
import sys
import moxing as mox

mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
fi

echo "============================================================"
echo "RC E2E INFERENCE COMPLETE"
echo "Inference summary: ${INFERENCE_ROOT}/summary.json"
echo "Raw result dir:    ${RAW_RESULT_DIR}"
if [ -n "${INFER_RESULT_OBS_PATH}" ]; then
  echo "DI main.sh input:   INFER_RESULT=${INFER_RESULT_OBS_PATH}"
else
  echo "Upload this directory to OBS, then set its OBS URI as INFER_RESULT in DI main.sh."
fi
echo "============================================================"
