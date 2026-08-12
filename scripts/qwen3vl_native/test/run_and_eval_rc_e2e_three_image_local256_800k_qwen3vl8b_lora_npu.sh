#!/usr/bin/env bash
set -euo pipefail

# Native Qwen3-VL-8B three-image local256 E2E inference and original RC
# all/low/high evaluation. CHECKPOINT_OBS_PATH is intentionally required
# because the 800k training result is supplied after this entry is prepared.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
NPU_TEST_DIR="${REPO_ROOT}/scripts/npu/test"

CHECKPOINT_OBS_PATH=${1:-${CHECKPOINT_OBS_PATH:-}}
: "${CHECKPOINT_OBS_PATH:?Set CHECKPOINT_OBS_PATH to the 800k three-image native Qwen3-VL-8B LoRA checkpoint OBS directory}"
CHECKPOINT_NAME=${CHECKPOINT_NAME:-$(basename "${CHECKPOINT_OBS_PATH%/}")}
CHECKPOINT_CACHE_ROOT=${CHECKPOINT_CACHE_ROOT:-/cache/jn/checkpoints/native_qwen3vl_three_image_local256_800k}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}}
REFRESH_CHECKPOINT=${REFRESH_CHECKPOINT:-False}

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-native-qwen3vl-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_native_qwen3vl_torch240.sh}
CREATE_ENV_IF_MISSING=${CREATE_ENV_IF_MISSING:-True}

MODEL_OBS_ROOT=${MODEL_OBS_ROOT:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
QWEN3VL_MODEL_NAME=${QWEN3VL_MODEL_NAME:-Qwen3-VL-8B-Instruct}
QWEN3VL_OBS_PATH=${QWEN3VL_OBS_PATH:-${MODEL_OBS_ROOT}/${QWEN3VL_MODEL_NAME}/}
QWEN3VL_PATH=${QWEN3VL_PATH:-/cache/jn/model/${QWEN3VL_MODEL_NAME}}
REFRESH_BASE_MODEL=${REFRESH_BASE_MODEL:-False}

E2E_DATA_OBS_PATH=${E2E_DATA_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/s00008810/RCDATA/E2E_eval/e2e_data.zip}
E2E_ARCHIVE_PATH=${E2E_ARCHIVE_PATH:-/cache/jn/e2e_eval/e2e_data.zip}
REFRESH_E2E_ARCHIVE=${REFRESH_E2E_ARCHIVE:-False}

RUN_ID=${RUN_ID:-three_image_local256_800k_native_qwen3vl8b_lora_${CHECKPOINT_NAME}_e2e_$(date -u +%Y%m%d_%H%M%S)}
RUN_WORK_ROOT=${RUN_WORK_ROOT:-/cache/jn/e2e_eval/native_qwen3vl_runs/${RUN_ID}}
E2E_DATA_ROOT=${E2E_DATA_ROOT:-${RUN_WORK_ROOT}/e2e_data}
INFERENCE_DATASET_ROOT=${INFERENCE_DATASET_ROOT:-${RUN_WORK_ROOT}/three_image_local256_dataset}
SHARD_JSONL_ROOT=${SHARD_JSONL_ROOT:-${RUN_WORK_ROOT}/inference_shards}
ACTIVE_INFER_JSONL=${ACTIVE_INFER_JSONL:-${SHARD_JSONL_ROOT}/selected.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
SHARD_OUTPUT_ROOT=${SHARD_OUTPUT_ROOT:-${OUTPUT_ROOT}/inference/shards}
INFERENCE_ROOT=${INFERENCE_ROOT:-${OUTPUT_ROOT}/inference}
RAW_RESULT_DIR=${RAW_RESULT_DIR:-${INFERENCE_ROOT}/json}
POSTPROCESS_ROOT=${POSTPROCESS_ROOT:-${OUTPUT_ROOT}/postprocess}
PATCH_REFERENCE_JSONL=${PATCH_REFERENCE_JSONL:-${POSTPROCESS_ROOT}/gt_presence.jsonl}
PATCH_REFERENCE_REPORT=${PATCH_REFERENCE_REPORT:-${POSTPROCESS_ROOT}/gt_presence_report.json}
FILTERED_PREDICTION_DIR=${FILTERED_PREDICTION_DIR:-${POSTPROCESS_ROOT}/predictions}
FILTER_REPORT=${FILTER_REPORT:-${POSTPROCESS_ROOT}/filter_report.json}

ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-1}
NUM_SAMPLES=${NUM_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
LOAD_STAGGER_SECONDS=${LOAD_STAGGER_SECONDS:-10}
BLACK_RATIO_THRESHOLD=${BLACK_RATIO_THRESHOLD:-1.0}
REBUILD_E2E_DATASET=${REBUILD_E2E_DATASET:-True}

# This keeps historical E2E comparisons aligned. It is a GT-assisted oracle
# diagnostic and can be disabled for an unassisted production metric.
GT_EMPTY_SUPPRESSION=${GT_EMPTY_SUPPRESSION:-True}
RULE_WORKERS=${RULE_WORKERS:-16}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
EXPECTED_E2E_SCENES=${EXPECTED_E2E_SCENES:-110}
RESULT_OBS_PATH=${RESULT_OBS_PATH:-}

# Reuse the raw model JSON for intersection evaluation. The lane GT-empty
# filtered copy must not be reused because it removes every category in a
# centerline-empty patch, including valid intersection predictions.
RUN_INTERSECTION_E2E=${RUN_INTERSECTION_E2E:-True}
INTERSECTION_COLLAPSE_TYPE_TO_ONE=${INTERSECTION_COLLAPSE_TYPE_TO_ONE:-False}
INTERSECTION_EVAL_ONLY_TYPE1=${INTERSECTION_EVAL_ONLY_TYPE1:-False}
INTERSECTION_GT_EMPTY_SUPPRESSION=${INTERSECTION_GT_EMPTY_SUPPRESSION:-False}
INTERSECTION_EVAL_VIS_FLAG=${INTERSECTION_EVAL_VIS_FLAG:-False}
INTERSECTION_QUERY_NAME=${INTERSECTION_QUERY_NAME:-output_llm_intersection_jn}
INTERSECTION_RESULT_ROOT=${INTERSECTION_RESULT_ROOT:-${OUTPUT_ROOT}/intersection_pipeline_metrics}
INTERSECTION_RUN_WORK_ROOT=${INTERSECTION_RUN_WORK_ROOT:-${RUN_WORK_ROOT}/intersection_original_pipeline}
INTERSECTION_ENGINE_EXTRACT_ROOT=${INTERSECTION_ENGINE_EXTRACT_ROOT:-${INTERSECTION_RUN_WORK_ROOT}/original_engine_grid256}

DEFAULT_SYSTEM_PROMPT=$'You are a road-map reconstruction assistant designed to process BEV (Bird\'s Eye View) images generated from LiDAR data.\nPredict the complete road map from the current patch in the BEV image.\nReturn only valid JSON in the required schema.\nDo not output markdown fences or extra explanation.\nKeep all coordinates in the patch-local coordinate system.'
SYSTEM_PROMPT=${SYSTEM_PROMPT:-${DEFAULT_SYSTEM_PROMPT}}

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

safe_source() {
  local source_path=$1
  set +u
  # shellcheck disable=SC1090
  source "${source_path}"
  set -u
}

download_obs_directory() {
  local source=$1
  local destination=$2
  local sentinel=$3
  if [ -s "${destination}/${sentinel}" ]; then
    echo "[native-three-image-e2e] reuse ${destination}"
    return
  fi
  SOURCE="${source}" DESTINATION="${destination}" python - <<'PY'
import os
import moxing as mox

print(f"[native-three-image-e2e] download {os.environ['SOURCE']} -> {os.environ['DESTINATION']}", flush=True)
mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
}

has_native_base_model() {
  local root=$1
  [ -s "${root}/config.json" ] && \
    { [ -s "${root}/preprocessor_config.json" ] || [ -s "${root}/processor_config.json" ]; } && \
    { [ -s "${root}/model.safetensors" ] || [ -s "${root}/model.safetensors.index.json" ] || \
      [ -s "${root}/pytorch_model.bin" ] || [ -s "${root}/pytorch_model.bin.index.json" ]; }
}

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  if ! is_true "${CREATE_ENV_IF_MISSING}"; then
    echo "ERROR: native Qwen3-VL environment is missing: ${ACTIVATE_SCRIPT}" >&2
    exit 2
  fi
  echo "[native-three-image-e2e] creating isolated native Qwen3-VL Torch-2.4 environment"
  ENV_DIR="${ENV_DIR}" RECREATE_ENV=False REQUIRE_NPU=True \
    bash "${REPO_ROOT}/scripts/npu/setup/create_mllm_native_qwen3vl_torch240_npu_env_from_infer.sh"
fi
safe_source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

if ! python -c "import rasterio" >/dev/null 2>&1; then
  echo "[native-three-image-e2e] installing one-time E2E raster dependency"
  python -m pip install "rasterio==1.4.4"
fi

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  safe_source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WITHOUT_JIT_COMPILE=${WITHOUT_JIT_COMPILE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}

python - <<'PY'
import json
import peft
import rasterio
import torch
import torch_npu
import transformers

versions = {
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "rasterio": rasterio.__version__,
    "npu_available": bool(torch.npu.is_available()),
}
print(json.dumps(versions, indent=2))
if transformers.__version__ != "4.57.3":
    raise SystemExit(f"Expected transformers=4.57.3, got {transformers.__version__}")
if peft.__version__ != "0.18.0":
    raise SystemExit(f"Expected peft=0.18.0, got {peft.__version__}")
if not torch.npu.is_available():
    raise SystemExit("NPU is unavailable in the native Qwen3-VL inference environment")
PY

mkdir -p "$(dirname "${E2E_ARCHIVE_PATH}")" "${RUN_WORK_ROOT}" "${OUTPUT_ROOT}" \
  "${CHECKPOINT_CACHE_ROOT}" "$(dirname "${QWEN3VL_PATH}")"

echo "============================================================"
echo "NATIVE QWEN3-VL-8B THREE-IMAGE LOCAL256 800K E2E"
echo "Checkpoint OBS:     ${CHECKPOINT_OBS_PATH}"
echo "Native base OBS:    ${QWEN3VL_OBS_PATH}"
echo "E2E OBS:            ${E2E_DATA_OBS_PATH}"
echo "Run work root:      ${RUN_WORK_ROOT}"
echo "Output root:        ${OUTPUT_ROOT}"
echo "Visible NPUs:       ${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPU processes:      ${NPROC_PER_NODE}"
echo "Per-device batch:   ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Images per sample:  3 (clean BEV, Raw-Lane, Pose)"
echo "View/coordinates:   local256 / norm1000"
echo "Generation cap:     ${MAX_NEW_TOKENS}"
echo "GT-empty suppression:${GT_EMPTY_SUPPRESSION}"
echo "Evaluations:        original all + low + high"
echo "Intersection E2E:   ${RUN_INTERSECTION_E2E} (reuse raw inference JSON)"
echo "Intersection types: only_type1=${INTERSECTION_EVAL_ONLY_TYPE1} collapse=${INTERSECTION_COLLAPSE_TYPE_TO_ONE}"
echo "Intersection oracle:${INTERSECTION_GT_EMPTY_SUPPRESSION}"
echo "============================================================"

echo "[native-three-image-e2e] stage 1/7: prepare clean E2E source"
if is_true "${REFRESH_E2E_ARCHIVE}" || [ ! -s "${E2E_ARCHIVE_PATH}" ]; then
  rm -f "${E2E_ARCHIVE_PATH}"
  SOURCE="${E2E_DATA_OBS_PATH}" DESTINATION="${E2E_ARCHIVE_PATH}" python - <<'PY'
import os
import moxing as mox

print(f"download {os.environ['SOURCE']} -> {os.environ['DESTINATION']}", flush=True)
mox.file.copy(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
else
  echo "[native-three-image-e2e] reuse E2E archive: ${E2E_ARCHIVE_PATH}"
fi

python scripts/tools/prepare_rc_e2e_original_run_data.py \
  --archive "${E2E_ARCHIVE_PATH}" \
  --destination "${E2E_DATA_ROOT}" \
  --allowed-root /cache/jn/e2e_eval/native_qwen3vl_runs

echo "[native-three-image-e2e] stage 2/7: build aligned local256 image triplets"
if is_true "${REBUILD_E2E_DATASET}" || [ ! -s "${INFERENCE_DATASET_ROOT}/infer.jsonl" ]; then
  python scripts/tools/prepare_rc_e2e_three_image_local256_dataset.py \
    --input-root "${E2E_DATA_ROOT}" \
    --output-root "${INFERENCE_DATASET_ROOT}" \
    --patch-size 256 \
    --stride 256 \
    --coord-range 1000 \
    --black-ratio-threshold "${BLACK_RATIO_THRESHOLD}"
else
  echo "[native-three-image-e2e] reuse inference dataset: ${INFERENCE_DATASET_ROOT}"
fi

python - "${INFERENCE_DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
expected_roles = ["bev_road_structure", "pv_camera_raw_lane", "historical_vehicle_trajectory"]
if summary.get("view_mode") != "local256" or summary.get("patch_size") != 256:
    raise SystemExit(f"Unexpected E2E geometry contract: {summary}")
if summary.get("num_images_per_sample") != 3 or summary.get("input_image_roles") != expected_roles:
    raise SystemExit(f"Unexpected three-image role contract: {summary}")
if summary.get("prompt_contract_version") != "three_image_roles_concise_v2":
    raise SystemExit(f"Unexpected prompt contract: {summary}")
if int(summary.get("patch_count", 0)) <= 0:
    raise SystemExit("Three-image E2E dataset contains no inference patches")
with (root / "infer.jsonl").open(encoding="utf-8") as handle:
    first = json.loads(next(line for line in handle if line.strip()))
if len(first.get("images") or []) != 3 or first["conversations"][0]["value"].count("<image>") != 3:
    raise SystemExit("First E2E record does not contain three ordered images and prompt tokens")
print(f"[native-three-image-e2e] dataset preflight passed: patches={summary['patch_count']}")
PY

echo "[native-three-image-e2e] stage 3/7: prepare native base and LoRA checkpoint"
if is_true "${REFRESH_BASE_MODEL}" || ! has_native_base_model "${QWEN3VL_PATH}"; then
  if [ -e "${QWEN3VL_PATH}" ]; then
    QWEN3VL_PATH="${QWEN3VL_PATH}.validated_$(date -u +%Y%m%d_%H%M%S)"
  fi
fi
download_obs_directory "${QWEN3VL_OBS_PATH}" "${QWEN3VL_PATH}" config.json
if is_true "${REFRESH_CHECKPOINT}" && [ -e "${CHECKPOINT_DIR}" ]; then
  CHECKPOINT_DIR="${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}.validated_$(date -u +%Y%m%d_%H%M%S)"
fi
if [ ! -s "${CHECKPOINT_DIR}/adapter_config.json" ] || \
   { [ ! -s "${CHECKPOINT_DIR}/adapter_model.safetensors" ] && [ ! -s "${CHECKPOINT_DIR}/adapter_model.bin" ]; }; then
  if [ -e "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_CACHE_ROOT}/${CHECKPOINT_NAME}.validated_$(date -u +%Y%m%d_%H%M%S)"
  fi
  download_obs_directory "${CHECKPOINT_OBS_PATH}" "${CHECKPOINT_DIR}" adapter_config.json
else
  echo "[native-three-image-e2e] reuse LoRA checkpoint: ${CHECKPOINT_DIR}"
fi

python - "${CHECKPOINT_DIR}" "${QWEN3VL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
base = Path(sys.argv[2])
adapter_config = checkpoint / "adapter_config.json"
weights = [checkpoint / "adapter_model.safetensors", checkpoint / "adapter_model.bin"]
base_weights = [
    base / "model.safetensors",
    base / "model.safetensors.index.json",
    base / "pytorch_model.bin",
    base / "pytorch_model.bin.index.json",
]
processor_files = [base / "preprocessor_config.json", base / "processor_config.json"]
missing = [str(path) for path in (adapter_config, base / "config.json") if not path.is_file()]
if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
    missing.append(str(weights[0]) + "|" + str(weights[1]))
if not any(path.is_file() and path.stat().st_size > 0 for path in base_weights):
    missing.append("native base model weights")
if not any(path.is_file() and path.stat().st_size > 0 for path in processor_files):
    missing.append("native base processor config")
if missing:
    raise FileNotFoundError(f"Incomplete native Qwen3-VL LoRA checkpoint/base: {missing}")
payload = json.loads(adapter_config.read_text(encoding="utf-8"))
if not payload.get("base_model_name_or_path"):
    raise ValueError(f"adapter_config has no base_model_name_or_path: {adapter_config}")
print(json.dumps({
    "checkpoint": str(checkpoint),
    "base": str(base),
    "peft_type": payload.get("peft_type"),
    "base_model_name_or_path": payload.get("base_model_name_or_path"),
}, indent=2))
PY

echo "[native-three-image-e2e] stage 4/7: split and run independent NPU inference shards"
IFS=',' read -r -a DEVICE_IDS <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [ "${#DEVICE_IDS[@]}" -ne "${NPROC_PER_NODE}" ]; then
  echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE}, but visible devices=${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 2
fi

rm -rf "${SHARD_JSONL_ROOT}" "${SHARD_OUTPUT_ROOT}" "${RAW_RESULT_DIR}"
mkdir -p "${SHARD_JSONL_ROOT}" "${SHARD_OUTPUT_ROOT}" "${INFERENCE_ROOT}/logs"
python scripts/tools/split_jsonl_for_inference.py \
  --input-jsonl "${INFERENCE_DATASET_ROOT}/infer.jsonl" \
  --output-root "${SHARD_JSONL_ROOT}" \
  --num-shards "${NPROC_PER_NODE}" \
  --num-samples "${NUM_SAMPLES}"

pids=()
for rank in "${!DEVICE_IDS[@]}"; do
  device=$(echo "${DEVICE_IDS[$rank]}" | xargs)
  shard_name=$(printf 'shard_%05d' "${rank}")
  shard_jsonl="${SHARD_JSONL_ROOT}/${shard_name}.jsonl"
  shard_output="${SHARD_OUTPUT_ROOT}/${shard_name}"
  shard_log="${INFERENCE_ROOT}/logs/${shard_name}.log"
  mkdir -p "${shard_output}"
  (
    export ASCEND_RT_VISIBLE_DEVICES="${device}"
    export ASCEND_VISIBLE_DEVICES="${device}"
    export NPU_VISIBLE_DEVICES="${device}"
    python -m mllm.native_qwen3vl.infer \
      --model-name-or-path "${CHECKPOINT_DIR}" \
      --model-base "${QWEN3VL_PATH}" \
      --test-json "${shard_jsonl}" \
      --image-folder "${INFERENCE_DATASET_ROOT}" \
      --output-dir "${shard_output}" \
      --phase phase_a \
      --map-task lane_intersection \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --temperature 0.0 \
      --coord-mode auto \
      --coord-range 1000 \
      --default-patch-size 256 \
      --per-device-infer-batch-size "${PER_DEVICE_INFER_BATCH_SIZE}" \
      --device npu:0 \
      --bf16 \
      --include-intersections \
      --system-prompt "${SYSTEM_PROMPT}" \
      --skip-eval \
      --skip-visualize
  ) >"${shard_log}" 2>&1 &
  pids+=("$!")
  echo "[native-three-image-e2e] launched rank=${rank} physical_npu=${device} pid=$! log=${shard_log}"
  if [ "${LOAD_STAGGER_SECONDS}" -gt 0 ] && [ "${rank}" -lt "$((NPROC_PER_NODE - 1))" ]; then
    sleep "${LOAD_STAGGER_SECONDS}"
  fi
done

failed=0
for rank in "${!pids[@]}"; do
  if ! wait "${pids[$rank]}"; then
    echo "ERROR: native inference shard ${rank} failed; tail follows" >&2
    tail -n 120 "${INFERENCE_ROOT}/logs/$(printf 'shard_%05d' "${rank}").log" >&2 || true
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi

for rank in "${!DEVICE_IDS[@]}"; do
  shard_log="${INFERENCE_ROOT}/logs/$(printf 'shard_%05d' "${rank}").log"
  throughput_line=$(grep 'DI_throughput:' "${shard_log}" | tail -n 1 || true)
  if [ -z "${throughput_line}" ]; then
    echo "ERROR: shard ${rank} did not report DI_throughput: ${shard_log}" >&2
    exit 1
  fi
  echo "[native-three-image-e2e] rank=${rank} ${throughput_line}"
done

python scripts/tools/merge_native_qwen3vl_inference_shards.py \
  --infer-jsonl "${ACTIVE_INFER_JSONL}" \
  --shard-root "${SHARD_OUTPUT_ROOT}" \
  --output-dir "${INFERENCE_ROOT}" \
  --prediction-dir "${RAW_RESULT_DIR}" \
  --reset
python scripts/tools/verify_e2e_inference_completeness.py \
  --infer-jsonl "${ACTIVE_INFER_JSONL}" \
  --prediction-dir "${RAW_RESULT_DIR}" \
  --output-json "${INFERENCE_ROOT}/completeness.json"

echo "[native-three-image-e2e] stage 5/7: optional lane GT-empty suppression and centerline evaluation"
if is_true "${GT_EMPTY_SUPPRESSION}"; then
  mkdir -p "${POSTPROCESS_ROOT}"
  python scripts/tools/build_rc_e2e_patch_gt_presence.py \
    --raw-e2e-root "${E2E_DATA_ROOT}" \
    --prediction-dir "${RAW_RESULT_DIR}" \
    --output-jsonl "${PATCH_REFERENCE_JSONL}" \
    --report-json "${PATCH_REFERENCE_REPORT}" \
    --patch-size 256 \
    --require-all
  python scripts/tools/suppress_e2e_predictions_without_patch_gt.py \
    --eval-jsonl "${PATCH_REFERENCE_JSONL}" \
    --prediction-dir "${RAW_RESULT_DIR}" \
    --output-dir "${FILTERED_PREDICTION_DIR}" \
    --report-json "${FILTER_REPORT}" \
    --reset \
    --strict
  EVALUATED_PREDICTIONS="${FILTERED_PREDICTION_DIR}"
else
  EVALUATED_PREDICTIONS="${RAW_RESULT_DIR}"
fi

PREDICTION_CACHE="${EVALUATED_PREDICTIONS}" \
REUSE_PREDICTIONS=True \
E2E_DATA_SOURCE=raw_direct \
E2E_RAW_ROOT="${E2E_DATA_ROOT}" \
E2E_DATA_ROOT="${E2E_DATA_ROOT}" \
RUN_ID="${RUN_ID}_original_pipeline" \
RUN_WORK_ROOT="${RUN_WORK_ROOT}/original_pipeline" \
RESULT_ROOT="${OUTPUT_ROOT}/original_pipeline_metrics" \
RESULT_OBS_PATH="" \
PREDICTION_COORD_SCALE=0.256 \
RUN_FORMAT_STEP=True \
RUN_RULE_STEP=True \
RUN_ALL_EVAL=True \
RUN_LOW_EVAL=True \
RUN_HIGH_EVAL=True \
EVAL_SIMPLIFY_PATH=False \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
FILL_MISSING_SCENE_PREDICTIONS=True \
FAIL_ON_INVALID_PREDICTIONS=False \
INSTALL_ENGINE_DEPS=True \
REUSE_ENGINE_ARCHIVE=True \
RESET_EXISTING_MODEL_OUTPUTS=False \
RULE_WORKERS="${RULE_WORKERS}" \
UPLOAD_RESULTS=False \
bash "${NPU_TEST_DIR}/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"

echo "[native-three-image-e2e] stage 6/7: original RC intersection evaluation"
if is_true "${RUN_INTERSECTION_E2E}"; then
  PREDICTION_DIR="${RAW_RESULT_DIR}" \
  E2E_DATA_ROOT="${E2E_DATA_ROOT}" \
  QUERY_NAME="${INTERSECTION_QUERY_NAME}" \
  RUN_ID="${RUN_ID}_intersection_original_pipeline" \
  RESULT_ROOT="${INTERSECTION_RESULT_ROOT}" \
  RUN_WORK_ROOT="${INTERSECTION_RUN_WORK_ROOT}" \
  ENGINE_EXTRACT_ROOT="${INTERSECTION_ENGINE_EXTRACT_ROOT}" \
  WINDOW_SIZE=256 \
  INTERSECTION_STRIDE=256 \
  ORIGINAL_E2E_LANE_GRID_SIZE=256 \
  PREDICTION_COORD_SCALE=0.256 \
  COORD_RANGE=1000 \
  COLLAPSE_INTERSECTION_TYPE_TO_ONE="${INTERSECTION_COLLAPSE_TYPE_TO_ONE}" \
  EVAL_INTERSECTION_ONLY_TYPE1="${INTERSECTION_EVAL_ONLY_TYPE1}" \
  SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION="${INTERSECTION_GT_EMPTY_SUPPRESSION}" \
  E2E_USE_RAW_ROOT_DIRECTLY=True \
  EVAL_VIS_FLAG="${INTERSECTION_EVAL_VIS_FLAG}" \
  EXPECTED_E2E_SCENES="${EXPECTED_E2E_SCENES}" \
  INSTALL_ENGINE_DEPS=False \
  UPLOAD_RESULTS=False \
  bash "${NPU_TEST_DIR}/eval_local512_predictions_original_intersection_e2e_npu.sh"
else
  echo "[native-three-image-e2e] skip intersection E2E: RUN_INTERSECTION_E2E=${RUN_INTERSECTION_E2E}"
fi

echo "[native-three-image-e2e] stage 7/7: optional OBS upload"
if [ -n "${RESULT_OBS_PATH}" ]; then
  SOURCE="${OUTPUT_ROOT}" DESTINATION="${RESULT_OBS_PATH}" python - <<'PY'
import os
import moxing as mox

mox.file.copy_parallel(os.environ["SOURCE"], os.environ["DESTINATION"])
PY
fi

echo "============================================================"
echo "NATIVE QWEN3-VL THREE-IMAGE LOCAL256 E2E COMPLETE"
echo "Checkpoint:       ${CHECKPOINT_DIR}"
echo "Three-image set:  ${INFERENCE_DATASET_ROOT}"
echo "Raw predictions:  ${RAW_RESULT_DIR}"
echo "Evaluated preds:  ${EVALUATED_PREDICTIONS}"
echo "Completeness:     ${INFERENCE_ROOT}/completeness.json"
echo "All roads:        ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_all"
echo "Low roads:        ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_low"
echo "High roads:       ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_high"
if is_true "${RUN_INTERSECTION_E2E}"; then
  echo "Intersections:    ${INTERSECTION_RESULT_ROOT}/eval_result_all"
  echo "Intersection log: ${INTERSECTION_RESULT_ROOT}/logs/03_eval_all.log"
fi
if is_true "${GT_EMPTY_SUPPRESSION}"; then
  echo "Oracle audit:     ${FILTER_REPORT}"
  echo "WARNING: GT-empty suppression is an oracle diagnostic."
fi
if [ -n "${RESULT_OBS_PATH}" ]; then
  echo "Result OBS:       ${RESULT_OBS_PATH}"
fi
echo "============================================================"
