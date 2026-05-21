#!/usr/bin/env bash
set -euo pipefail

# Common NPU inference launcher for explicit stage/task/vision wrappers.
#
# Wrapper-selected parameters:
#   VISION_BACKBONE=dinov2|dinov3
#   DATASET_PHASE=phase_a|phase_b
#   MAP_TASK=lane|lane_intersection
#
# Required local paths to fill before running:
#   CHECKPOINT_DIR: trained checkpoint directory. Examples:
#     /cache/unimapgen_v2/train_output/sft_phase_a_lane_dinov2_qwen3vl8b_nodeepstack/best
#     /cache/unimapgen_v2/train_output/grpo_phase_b_lane_dinov3_qwen3vl8b_nodeepstack/final_merged
#   DATASET_PATH: dataset root containing phase_a/phase_b test.jsonl and patch images.
#     Default: /cache/unimapgen_v2/dataset
#   IMAGE_FOLDER: root for image paths in JSONL. Usually same as DATASET_PATH.
#   VISION_TOWER: local DINO checkpoint path. Defaults are set from VISION_BACKBONE.
#   OUTPUT_DIR: output root for this inference run.
#
# Output layout:
#   OUTPUT_DIR/summary.json           normal inference summary
#   OUTPUT_DIR/json/                  per-sample/per-patch JSON
#   OUTPUT_DIR/viz/                   single-patch comparison PNGs
#   OUTPUT_DIR/eval.json              line metric JSON with table string
#   OUTPUT_DIR/whole_map_viz/         stitched whole-map PNGs for A and B
#
# Stage behavior:
#   phase_a uses normal independent patch inference.
#   phase_b uses state-update inference, where left/top hints come from previous predictions.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:?set VISION_BACKBONE to dinov2 or dinov3}
DATASET_PHASE=${DATASET_PHASE:?set DATASET_PHASE to phase_a or phase_b}
MAP_TASK=${MAP_TASK:?set MAP_TASK to lane or lane_intersection}

case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER=${VISION_TOWER:-/cache/jjh/checkpoints/facebook_dinov2-large}
    INPUT_IMAGE_SIZE_ARGS=()
    ;;
  dinov3)
    VISION_TOWER=${VISION_TOWER:-/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
    INPUT_IMAGE_SIZE_ARGS=(--input_image_size "${INPUT_IMAGE_SIZE:-512}")
    ;;
  *) echo "ERROR: unsupported VISION_BACKBONE=${VISION_BACKBONE}"; exit 1 ;;
esac
case "${DATASET_PHASE}" in
  phase_a|phase_b) ;;
  *) echo "ERROR: unsupported DATASET_PHASE=${DATASET_PHASE}"; exit 1 ;;
esac
case "${MAP_TASK}" in
  lane|lane_intersection) ;;
  *) echo "ERROR: unsupported MAP_TASK=${MAP_TASK}"; exit 1 ;;
esac

DATASET_PATH=${DATASET_PATH:-/cache/unimapgen_v2/dataset}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/cache/unimapgen_v2/train_output/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
BEST_CHECKPOINT_NAME=${BEST_CHECKPOINT_NAME:-eval_best} # eval_best first; set to best for train-loss best.
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
if [ -z "${CHECKPOINT_DIR}" ]; then
  CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --best-name "${BEST_CHECKPOINT_NAME}" \
    --best-name best \
    --allow-direct)
fi
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/cache/unimapgen_v2/infer_output/${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack/${RUN_ID}}
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0} # 0 means all rows.
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

TEST_JSON=${TEST_JSON:-}
if [ -z "${TEST_JSON}" ]; then
  if [ -f "${DATASET_PATH}/${DATASET_PHASE}/test.jsonl" ]; then
    TEST_JSON="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
  else
    TEST_JSON="${DATASET_PATH}/test.jsonl"
  fi
fi

for path in "${CHECKPOINT_DIR}" "${VISION_TOWER}" "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done

JSON_DIR="${OUTPUT_DIR}/json"
PATCH_VIZ_DIR="${OUTPUT_DIR}/viz"
WHOLE_MAP_VIZ_DIR="${OUTPUT_DIR}/whole_map_viz"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
MERGED_GLOBAL_JSON="${OUTPUT_DIR}/merged_global.json"
EVAL_JSON="${OUTPUT_DIR}/eval.json"
mkdir -p "${OUTPUT_DIR}" "${JSON_DIR}" "${PATCH_VIZ_DIR}" "${WHOLE_MAP_VIZ_DIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export MLLM_SUPPRESS_NONZERO_STDERR=${MLLM_SUPPRESS_NONZERO_STDERR:-0}

echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "VISION_TOWER=${VISION_TOWER}"
echo "DATASET_PHASE=${DATASET_PHASE}"
echo "MAP_TASK=${MAP_TASK}"
echo "TEST_JSON=${TEST_JSON}"
echo "IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "COORD_MODE=${COORD_MODE} COORD_RANGE=${COORD_RANGE}"

if [ "${DATASET_PHASE}" = "phase_b" ]; then
  INCLUDE_INTERSECTION_ARGS=()
  if [ "${MAP_TASK}" = "lane_intersection" ]; then
    INCLUDE_INTERSECTION_ARGS=(--include-intersections)
  fi
  python scripts/tools/infer_centerline_state_update.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --vision_tower "${VISION_TOWER}" \
    "${INPUT_IMAGE_SIZE_ARGS[@]}" \
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
    "${INPUT_IMAGE_SIZE_ARGS[@]}" \
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

output_dir = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
rank_files = sorted(glob.glob(str(output_dir / "summary_rank*.json")))
if not rank_files:
    raise SystemExit(0)
merged = []
for path in rank_files:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        merged.extend(payload)
    else:
        merged.append(payload)
merged.sort(key=lambda item: item.get("idx", item.get("record_id", "")))
summary_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Merged {len(rank_files)} rank summaries into {summary_json}")
PY
fi

python scripts/tools/visualize_centerline.py \
  --input-dir "${OUTPUT_DIR}" \
  --image-folder "${IMAGE_FOLDER}" \
  --output-dir "${PATCH_VIZ_DIR}" \
  --eval-output-json "${EVAL_JSON}" \
  --whole-map-viz-dir "${WHOLE_MAP_VIZ_DIR}"

echo "Inference outputs:"
echo "  summary:        ${SUMMARY_JSON}"
echo "  sample json dir:${JSON_DIR}"
echo "  patch viz dir:  ${PATCH_VIZ_DIR}"
echo "  eval json:      ${EVAL_JSON}"
echo "  whole map dir:  ${WHOLE_MAP_VIZ_DIR}"
