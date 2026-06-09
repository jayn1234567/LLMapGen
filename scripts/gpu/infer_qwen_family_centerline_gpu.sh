#!/usr/bin/env bash
set -euo pipefail

# Generic GPU patch inference for Qwen-family multimodal checkpoints:
# Qwen2, Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.5-MoE, and Qwen3-VL-derived LLMs.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/debug_runs/qwen3_nodeepstack_smoke/checkpoint-final}
IMAGE=${IMAGE:-}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/av2_patch_256_fullimage_cutflag_test_v2}
TEST_JSON=${TEST_JSON:-data/debug_phase_a_lane20/test.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-10}
SAMPLE_OFFSET=${SAMPLE_OFFSET:-0}
PROMPT_MODE=${PROMPT_MODE:-dataset}
MAP_TASK=${MAP_TASK:-lane}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
CONV_TEMPLATE=${CONV_TEMPLATE:-conv_qwen_3_Dinov2_huawei}
OUTPUT_JSON=${OUTPUT_JSON:-outputs/qwen_family_infer/predictions.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen_family_infer/samples}
DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}

VISION_TOWER=${VISION_TOWER:-}
MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-}
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-}
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-}
MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-}
MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-}
MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-}
MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-}
MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-}
MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-}
MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-}
MULTI_VISION_ROUTER_TEMPERATURE=${MULTI_VISION_ROUTER_TEMPERATURE:-}
MULTI_VISION_ROUTER_HIDDEN_RATIO=${MULTI_VISION_ROUTER_HIDDEN_RATIO:-}
MULTI_VISION_ROUTER_USE_DIFF=${MULTI_VISION_ROUTER_USE_DIFF:-}
MULTI_VISION_DROPOUT=${MULTI_VISION_DROPOUT:-}

EXTRA_ARGS=()
[ -n "${IMAGE}" ] && EXTRA_ARGS+=(--image "${IMAGE}")
[ -n "${VISION_TOWER}" ] && EXTRA_ARGS+=(--vision_tower "${VISION_TOWER}")
[ -n "${MM_VISION_TOWER_TYPE}" ] && EXTRA_ARGS+=(--mm_vision_tower_type "${MM_VISION_TOWER_TYPE}")
[ -n "${INPUT_IMAGE_SIZE}" ] && EXTRA_ARGS+=(--input_image_size "${INPUT_IMAGE_SIZE}")
[ -n "${DISABLE_DEEPSTACK}" ] && [[ "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && EXTRA_ARGS+=(--disable_deepstack)
[ -n "${DEEPSTACK_VISUAL_INDEXES}" ] && EXTRA_ARGS+=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
[ -n "${VISION_LAYER_FUSION_INDEXES}" ] && EXTRA_ARGS+=(--vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES})
[ -n "${VISION_LAYER_FUSION_TYPE}" ] && EXTRA_ARGS+=(--vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}")
[ -n "${MULTI_VISION_TOWERS}" ] && EXTRA_ARGS+=(--multi_vision_towers "${MULTI_VISION_TOWERS}")
[ -n "${MULTI_VISION_TOWER_TYPES}" ] && EXTRA_ARGS+=(--multi_vision_tower_types "${MULTI_VISION_TOWER_TYPES}")
[ -n "${MULTI_VISION_INPUT_IMAGE_SIZES}" ] && EXTRA_ARGS+=(--multi_vision_input_image_sizes "${MULTI_VISION_INPUT_IMAGE_SIZES}")
[ -n "${MULTI_VISION_PRIMARY_INDEX}" ] && EXTRA_ARGS+=(--multi_vision_primary_index "${MULTI_VISION_PRIMARY_INDEX}")
[ -n "${MULTI_VISION_HIDDEN_SIZE}" ] && EXTRA_ARGS+=(--multi_vision_hidden_size "${MULTI_VISION_HIDDEN_SIZE}")
[ -n "${MULTI_VISION_TARGET_GRID}" ] && EXTRA_ARGS+=(--multi_vision_target_grid "${MULTI_VISION_TARGET_GRID}")
[ -n "${MULTI_VISION_FUSION}" ] && EXTRA_ARGS+=(--multi_vision_fusion "${MULTI_VISION_FUSION}")
[ -n "${MULTI_VISION_ROUTER_TEMPERATURE}" ] && EXTRA_ARGS+=(--multi_vision_router_temperature "${MULTI_VISION_ROUTER_TEMPERATURE}")
[ -n "${MULTI_VISION_ROUTER_HIDDEN_RATIO}" ] && EXTRA_ARGS+=(--multi_vision_router_hidden_ratio "${MULTI_VISION_ROUTER_HIDDEN_RATIO}")
[ -n "${MULTI_VISION_ROUTER_USE_DIFF}" ] && EXTRA_ARGS+=(--multi_vision_router_use_diff "${MULTI_VISION_ROUTER_USE_DIFF}")
[ -n "${MULTI_VISION_DROPOUT}" ] && EXTRA_ARGS+=(--multi_vision_dropout "${MULTI_VISION_DROPOUT}")

echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Test JSON:  ${TEST_JSON:-<single image>}"
echo "Image:      ${IMAGE:-<from test_json>}"
echo "Template:   ${CONV_TEMPLATE}"
echo "Map task:   ${MAP_TASK}"
echo "Device:     ${DEVICE}"
echo "Output:     ${OUTPUT_JSON}"
echo "============================================================"

python scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --image-folder "${IMAGE_FOLDER}" \
  --test-json "${TEST_JSON}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-offset "${SAMPLE_OFFSET}" \
  --prompt-mode "${PROMPT_MODE}" \
  --map-task "${MAP_TASK}" \
  --patch-size 256 \
  --coord-mode "${COORD_MODE}" \
  --coord-range "${COORD_RANGE}" \
  --conv-template "${CONV_TEMPLATE}" \
  --device "${DEVICE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --output-json "${OUTPUT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
