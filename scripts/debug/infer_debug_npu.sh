#!/usr/bin/env bash
set -euo pipefail

# Local Ascend NPU inference + patch visualization + whole-map visualization.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:-dinov2}
DATASET_PHASE=${DATASET_PHASE:-phase_a}
MAP_TASK=${MAP_TASK:-lane}
DEBUG_RUN_NAME=${DEBUG_RUN_NAME:-local_debug}

DATASET_ROOT=${DATASET_ROOT:-/cache/data/data_line_samples_33w}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_ROOT}}
DEBUG_DATA_ROOT=${DEBUG_DATA_ROOT:-${REPO_ROOT}/checkpoints/debug_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/debug}

DINOV2_PATH=${DINOV2_PATH:-/cache/jjh/checkpoints/facebook_dinov2-large}
DINOV3_PATH=${DINOV3_PATH:-/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}

case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER="${DINOV2_PATH}"
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    ;;
  dinov3)
    VISION_TOWER="${DINOV3_PATH}"
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
  *) echo "ERROR: VISION_BACKBONE must be dinov2 or dinov3"; exit 1 ;;
esac
case "${DATASET_PHASE}" in
  phase_a|phase_b) ;;
  *) echo "ERROR: DATASET_PHASE must be phase_a or phase_b"; exit 1 ;;
esac
case "${MAP_TASK}" in
  lane|lane_intersection) ;;
  *) echo "ERROR: MAP_TASK must be lane or lane_intersection"; exit 1 ;;
esac

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export MLLM_SUPPRESS_NONZERO_STDERR=${MLLM_SUPPRESS_NONZERO_STDERR:-0}

TRAIN_LIMIT=${TRAIN_LIMIT:-16}
EVAL_LIMIT=${EVAL_LIMIT:-4}
TEST_LIMIT=${TEST_LIMIT:-4}
SAMPLE_SEED=${SAMPLE_SEED:-42}
python scripts/debug/sample_debug_jsonl.py \
  --dataset-root "${DATASET_ROOT}" \
  --phase "${DATASET_PHASE}" \
  --output-root "${DEBUG_DATA_ROOT}" \
  --train-limit "${TRAIN_LIMIT}" \
  --eval-limit "${EVAL_LIMIT}" \
  --test-limit "${TEST_LIMIT}" \
  --seed "${SAMPLE_SEED}"

TEST_JSON=${TEST_JSON:-${DEBUG_DATA_ROOT}/${DATASET_PHASE}/test.jsonl}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
if [ -z "${CHECKPOINT_DIR}" ]; then
  if CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
      --output-dir "${TRAIN_OUTPUT_DIR}" \
      --best-name eval_best \
      --best-name best \
      --allow-direct 2>/dev/null); then
    :
  else
    CHECKPOINT_DIR=$(python - "${TRAIN_OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
candidates = []
for path in root.glob("checkpoint-*"):
    if path.is_dir():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No checkpoint found under {root}")
print(sorted(candidates)[-1][1])
PY
)
  fi
fi

for path in "${CHECKPOINT_DIR}" "${VISION_TOWER}" "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done

OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/infer_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
JSON_DIR="${OUTPUT_DIR}/json"
PATCH_VIZ_DIR="${OUTPUT_DIR}/viz"
WHOLE_MAP_VIZ_DIR="${OUTPUT_DIR}/whole_map_viz"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
MERGED_GLOBAL_JSON="${OUTPUT_DIR}/merged_global.json"
EVAL_JSON="${OUTPUT_DIR}/eval.json"
mkdir -p "${OUTPUT_DIR}" "${JSON_DIR}" "${PATCH_VIZ_DIR}" "${WHOLE_MAP_VIZ_DIR}"

COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6062}

echo "Inference debug:"
echo "  checkpoint=${CHECKPOINT_DIR}"
echo "  phase=${DATASET_PHASE} map_task=${MAP_TASK} vision=${VISION_BACKBONE}"
echo "  test=${TEST_JSON}"
echo "  output=${OUTPUT_DIR}"

if [ "${DATASET_PHASE}" = "phase_b" ]; then
  INCLUDE_INTERSECTION_ARGS=()
  if [ "${MAP_TASK}" = "lane_intersection" ]; then
    INCLUDE_INTERSECTION_ARGS=(--include-intersections)
  fi
  python scripts/tools/infer_centerline_state_update.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack \
    --patch-json "${TEST_JSON}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-json "${SUMMARY_JSON}" \
    --output-dir "${JSON_DIR}" \
    --sample-json-dir "${JSON_DIR}" \
    --merged-output-json "${MERGED_GLOBAL_JSON}" \
    --whole-map-viz-dir "${WHOLE_MAP_VIZ_DIR}" \
    --conv-template conv_qwen_3_Dinov2_huawei \
    --device "${DEVICE:-npu:0}" \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    "${INCLUDE_INTERSECTION_ARGS[@]}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    --eval-centerline \
    --eval-output-json "${EVAL_JSON}"
else
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/tools/infer_centerline_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack \
    --test-json "${TEST_JSON}" \
    --num-samples "${NUM_TEST_SAMPLES}" \
    --image-folder "${IMAGE_FOLDER}" \
    --prompt-mode dataset \
    --map-task "${MAP_TASK}" \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template conv_qwen_3_Dinov2_huawei \
    --output-dir "${OUTPUT_DIR}" \
    --sample-json-dir "${JSON_DIR}" \
    --output-json "${SUMMARY_JSON}" \
    --temperature 0.0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --eval-centerline \
    --eval-output-json "${EVAL_JSON}"

  python - "${OUTPUT_DIR}" "${SUMMARY_JSON}" <<'PY'
import glob
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
summary = Path(sys.argv[2])
rank_files = sorted(glob.glob(str(out / "summary_rank*.json")))
if not rank_files:
    raise SystemExit(0)
records = []
for name in rank_files:
    text = Path(name).read_text(encoding="utf-8-sig").strip()
    if not text:
        continue
    payload = json.loads(text) if text[0] == "[" else [json.loads(line) for line in text.splitlines() if line.strip()]
    records.extend(payload if isinstance(payload, list) else [payload])
records.sort(key=lambda item: item.get("idx", item.get("record_id", "")))
summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Merged {len(records)} records into {summary}")
PY
fi

python scripts/tools/visualize_centerline.py \
  --input-dir "${OUTPUT_DIR}" \
  --image-folder "${IMAGE_FOLDER}" \
  --output-dir "${PATCH_VIZ_DIR}" \
  --eval-output-json "${EVAL_JSON}" \
  --whole-map-viz-dir "${WHOLE_MAP_VIZ_DIR}"

echo "Inference debug finished:"
echo "  summary=${SUMMARY_JSON}"
echo "  sample_json_dir=${JSON_DIR}"
echo "  patch_viz_dir=${PATCH_VIZ_DIR}"
echo "  eval_json=${EVAL_JSON}"
echo "  whole_map_viz_dir=${WHOLE_MAP_VIZ_DIR}"
