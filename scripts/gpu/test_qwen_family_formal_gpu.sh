#!/usr/bin/env bash
set -euo pipefail

# Formal GPU patch inference/eval entrypoint for Qwen-family multimodal checkpoints.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-}
if [[ -z "${CHECKPOINT_DIR}" && -n "${TRAIN_OUTPUT_DIR}" ]]; then
  CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --best-name infer_best \
    --best-name eval_best \
    --best-name best \
    --allow-direct)
fi
[ -n "${CHECKPOINT_DIR}" ] || { echo "Set CHECKPOINT_DIR or TRAIN_OUTPUT_DIR"; exit 1; }
[ -d "${CHECKPOINT_DIR}" ] || { echo "Checkpoint not found: ${CHECKPOINT_DIR}"; exit 1; }

VISION_BACKBONE=${VISION_BACKBONE:-dinov2}
case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER=${VISION_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov2}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    ;;
  dinov3)
    VISION_TOWER=${VISION_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook/dinov3-vitl16-pretrain-lvd1689m}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov3}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
  multi_moe|multi_vision_moe|dual_dino_moe)
    DINO_V2_TOWER=${DINO_V2_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large}
    DINO_V3_TOWER=${DINO_V3_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook/dinov3-vitl16-pretrain-lvd1689m}
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINO_V2_TOWER},${DINO_V3_TOWER}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov2,dinov3}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,512}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-1}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-softmax_router}
    VISION_TOWER=${VISION_TOWER:-${MULTI_VISION_TOWERS}}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-multi_moe}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
  dinov2_siglip_concat|dinov2_siglip|dinosiglip_v2)
    DINO_V2_TOWER=${DINO_V2_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large}
    SIGLIP_TOWER=${SIGLIP_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/google_siglip-large-patch16-384}
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINO_V2_TOWER},${SIGLIP_TOWER}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov2,siglip}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,384}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-0}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-concat_projector}
    VISION_TOWER=${VISION_TOWER:-${MULTI_VISION_TOWERS}}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-multi_concat}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
  dinov3_siglip_concat|dinov3_siglip|dinosiglip_v3)
    DINO_V3_TOWER=${DINO_V3_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook/dinov3-vitl16-pretrain-lvd1689m}
    SIGLIP_TOWER=${SIGLIP_TOWER:-/media/q/data2/jjh/project/MLLM_project/checkpoints/google_siglip-large-patch16-384}
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINO_V3_TOWER},${SIGLIP_TOWER}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov3,siglip}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,384}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-0}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-concat_projector}
    VISION_TOWER=${VISION_TOWER:-${MULTI_VISION_TOWERS}}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-multi_concat}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
  *)
    echo "Unsupported VISION_BACKBONE=${VISION_BACKBONE}; expected dinov2, dinov3, multi_moe, dinov2_siglip_concat, or dinov3_siglip_concat"
    exit 1
    ;;
esac

FLOW_PHASE=${FLOW_PHASE:-phase_a_lane}
case "${FLOW_PHASE}" in
  phase_a|phase_a_lane)
    MAP_TASK=${MAP_TASK:-lane}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_a_lane20/test.jsonl}
    ;;
  phase_a_lane_intersection)
    MAP_TASK=${MAP_TASK:-lane_intersection}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_a_lane_intersection20/test.jsonl}
    ;;
  phase_b|phase_b_lane)
    MAP_TASK=${MAP_TASK:-lane}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_b_lane20/test.jsonl}
    ;;
  phase_b_lane_intersection)
    MAP_TASK=${MAP_TASK:-lane_intersection}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_b_lane_intersection20/test.jsonl}
    ;;
  *)
    echo "Unsupported FLOW_PHASE=${FLOW_PHASE}; expected phase_a_lane, phase_a_lane_intersection, phase_b_lane, or phase_b_lane_intersection"
    exit 1
    ;;
esac

IMAGE_FOLDER=${IMAGE_FOLDER:-data/av2_patch_256_fullimage_cutflag_test_v2}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/formal_eval/$(basename "${CHECKPOINT_DIR}")}
OUTPUT_JSON=${OUTPUT_JSON:-${OUTPUT_DIR}/summary.json}
VERSION=${VERSION:-conv_qwen_3_Dinov2_huawei}
DEVICE=${DEVICE:-cuda}
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
SAMPLE_OFFSET=${SAMPLE_OFFSET:-0}
PROMPT_MODE=${PROMPT_MODE:-dataset}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-}
RUN_EVAL_CENTERLINE=${RUN_EVAL_CENTERLINE:-True}

[ -f "${TEST_JSONL}" ] || { echo "Test JSONL not found: ${TEST_JSONL}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

VISION_ARGS=(
  --vision_tower "${VISION_TOWER}"
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}"
  --input_image_size "${INPUT_IMAGE_SIZE}"
)
if [[ -n "${DISABLE_DEEPSTACK}" && "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  VISION_ARGS+=(--disable_deepstack)
fi
if [[ -n "${DEEPSTACK_VISUAL_INDEXES}" ]]; then
  VISION_ARGS+=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi
if [[ -n "${VISION_LAYER_FUSION_INDEXES}" ]]; then
  VISION_ARGS+=(--vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES})
fi
if [[ -n "${VISION_LAYER_FUSION_TYPE}" ]]; then
  VISION_ARGS+=(--vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}")
fi
if [[ "${MM_VISION_TOWER_TYPE}" == "multi_moe" || "${MM_VISION_TOWER_TYPE}" == "multi_concat" ]]; then
  VISION_ARGS+=(
    --multi_vision_towers "${MULTI_VISION_TOWERS}"
    --multi_vision_tower_types "${MULTI_VISION_TOWER_TYPES}"
    --multi_vision_input_image_sizes "${MULTI_VISION_INPUT_IMAGE_SIZES}"
    --multi_vision_primary_index "${MULTI_VISION_PRIMARY_INDEX}"
    --multi_vision_hidden_size "${MULTI_VISION_HIDDEN_SIZE}"
    --multi_vision_target_grid "${MULTI_VISION_TARGET_GRID}"
    --multi_vision_fusion "${MULTI_VISION_FUSION}"
  )
fi

EVAL_ARGS=()
if [[ "${RUN_EVAL_CENTERLINE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(--eval-centerline)
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Formal GPU inference: ${FLOW_PHASE} ${MAP_TASK} ${VISION_BACKBONE}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Vision:     ${VISION_TOWER}"
echo "Type:       ${MM_VISION_TOWER_TYPE}"
echo "Fusion:     ${MULTI_VISION_FUSION:-single}"
echo "Layer fuse: ${VISION_LAYER_FUSION_INDEXES:-off} (${VISION_LAYER_FUSION_TYPE:-checkpoint/default})"
echo "Test:       ${TEST_JSONL}"
echo "Output:     ${OUTPUT_JSON}"
echo "============================================================"

python scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "${VISION_ARGS[@]}" \
  --test-json "${TEST_JSONL}" \
  --image-folder "${IMAGE_FOLDER}" \
  --num-samples "${NUM_TEST_SAMPLES}" \
  --sample-offset "${SAMPLE_OFFSET}" \
  --prompt-mode "${PROMPT_MODE}" \
  --map-task "${MAP_TASK}" \
  --patch-size 256 \
  --coord-mode "${COORD_MODE}" \
  --coord-range "${COORD_RANGE}" \
  --conv-template "${VERSION}" \
  --device "${DEVICE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_JSON}" \
  "${EVAL_ARGS[@]}"

echo "Formal GPU inference finished: ${OUTPUT_JSON}"
