#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU SFT smoke for Qwen3VL + single-tower/Multi-MoE/Multi-Concat + no DeepStack.
# Covers both lane-only patch recognition and lane+intersection state-update
# data formats. This is a real GPU runtime check, not a syntax-only script.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-unimapgen}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

GPU_IDS=${GPU_IDS:-0,2}
NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29670}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/media/q/data2/jjh/project/MLLM_project/outputs/test_qwen3vl}
VISION_BACKBONE=${VISION_BACKBONE:-dinov2}  # dinov2, dinov3, multi_moe, dinov2_siglip_concat, or dinov3_siglip_concat

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

FLOW_PHASE=${FLOW_PHASE:-phase_a}  # phase_a, phase_a_lane_intersection, or phase_b
case "${FLOW_PHASE}" in
  phase_a)
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_a_lane20/train.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_a_lane20/test.jsonl}
    MAP_TASK=${MAP_TASK:-lane}
    RUN_STATE_UPDATE=${RUN_STATE_UPDATE:-False}
    ;;
  phase_a_lane_intersection)
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_a_lane_intersection20/train.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_a_lane_intersection20/test.jsonl}
    MAP_TASK=${MAP_TASK:-lane_intersection}
    RUN_STATE_UPDATE=${RUN_STATE_UPDATE:-False}
    ;;
  phase_b)
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_b_lane_intersection20/train.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_b_lane_intersection20/test.jsonl}
    MAP_TASK=${MAP_TASK:-lane_intersection}
    RUN_STATE_UPDATE=${RUN_STATE_UPDATE:-True}
    ;;
  *)
    echo "Unsupported FLOW_PHASE=${FLOW_PHASE}; expected phase_a, phase_a_lane_intersection, or phase_b"
    exit 1
    ;;
esac

IMAGE_FOLDER=${IMAGE_FOLDER:-data/av2_patch_256_fullimage_cutflag_test_v2}
VERSION=${VERSION:-conv_qwen_3_Dinov2_huawei}
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}

LORA_ENABLE=${LORA_ENABLE:-True}
LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
if [[ "${LORA_ENABLE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  TRAIN_MODE=${TRAIN_MODE:-lora}
  UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-False}
else
  TRAIN_MODE=${TRAIN_MODE:-full}
  UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}
fi

MAX_STEPS=${MAX_STEPS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-1e-6}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-1e-6}
MM_VISION_FUSION_LR=${MM_VISION_FUSION_LR:-${MM_PROJECTOR_LR}}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
BF16=${BF16:-True}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384}
NUM_INFER_SAMPLES=${NUM_INFER_SAMPLES:-1}

RUN_ID=${RUN_ID:-${FLOW_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_${TRAIN_MODE}_gpu02}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/debug_runs/full_flow_${RUN_ID}}
INFER_DIR=${INFER_DIR:-${OUTPUT_DIR}/infer_patch}
STATE_DIR=${STATE_DIR:-${OUTPUT_DIR}/state_update}

# SwanLab monitoring. Keep the API key outside the script:
#   export SWANLAB_API_KEY=...
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_debug_${FLOW_PHASE}_${MAP_TASK}_${VISION_BACKBONE}}
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-sft_debug}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_debug_${FLOW_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_${TRAIN_MODE}}
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,debug,${FLOW_PHASE},${MAP_TASK},${VISION_BACKBONE},${TRAIN_MODE}}
SWANLAB_MODE=${SWANLAB_MODE:-}
export SWANLAB_API_KEY

[ -d "${MODEL_NAME_OR_PATH}" ] || { echo "Model not found: ${MODEL_NAME_OR_PATH}"; exit 1; }
if [[ "${MM_VISION_TOWER_TYPE}" == "multi_moe" || "${MM_VISION_TOWER_TYPE}" == "multi_concat" ]]; then
  IFS=',' read -r -a _vision_tower_paths <<< "${MULTI_VISION_TOWERS}"
  for _vision_tower_path in "${_vision_tower_paths[@]}"; do
    [ -d "${_vision_tower_path}" ] || { echo "Vision tower not found: ${_vision_tower_path}"; exit 1; }
  done
else
  [ -d "${VISION_TOWER}" ] || { echo "Vision tower not found: ${VISION_TOWER}"; exit 1; }
fi
[ -f "${TRAIN_JSONL}" ] || { echo "Train JSONL not found: ${TRAIN_JSONL}"; exit 1; }
[ -f "${TEST_JSONL}" ] || { echo "Test JSONL not found: ${TEST_JSONL}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

TRAIN_VISION_ARGS=(
  --vision_tower "${VISION_TOWER}"
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}"
  --input_image_size "${INPUT_IMAGE_SIZE}"
)
INFER_VISION_ARGS=(
  --vision_tower "${VISION_TOWER}"
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}"
  --input_image_size "${INPUT_IMAGE_SIZE}"
  --disable_deepstack
)
if [[ -n "${VISION_LAYER_FUSION_INDEXES}" ]]; then
  TRAIN_VISION_ARGS+=(
    --vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES}
    --vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}"
  )
  INFER_VISION_ARGS+=(
    --vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES}
    --vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}"
  )
fi
if [[ "${MM_VISION_TOWER_TYPE}" == "multi_moe" || "${MM_VISION_TOWER_TYPE}" == "multi_concat" ]]; then
  TRAIN_VISION_ARGS+=(
    --multi_vision_towers "${MULTI_VISION_TOWERS}"
    --multi_vision_tower_types "${MULTI_VISION_TOWER_TYPES}"
    --multi_vision_input_image_sizes "${MULTI_VISION_INPUT_IMAGE_SIZES}"
    --multi_vision_primary_index "${MULTI_VISION_PRIMARY_INDEX}"
    --multi_vision_hidden_size "${MULTI_VISION_HIDDEN_SIZE}"
    --multi_vision_target_grid "${MULTI_VISION_TARGET_GRID}"
    --multi_vision_fusion "${MULTI_VISION_FUSION}"
  )
  INFER_VISION_ARGS+=(
    --multi_vision_towers "${MULTI_VISION_TOWERS}"
    --multi_vision_tower_types "${MULTI_VISION_TOWER_TYPES}"
    --multi_vision_input_image_sizes "${MULTI_VISION_INPUT_IMAGE_SIZES}"
    --multi_vision_primary_index "${MULTI_VISION_PRIMARY_INDEX}"
    --multi_vision_hidden_size "${MULTI_VISION_HIDDEN_SIZE}"
    --multi_vision_target_grid "${MULTI_VISION_TARGET_GRID}"
    --multi_vision_fusion "${MULTI_VISION_FUSION}"
  )
fi

echo "============================================================"
echo "SFT smoke: ${FLOW_PHASE} ${MAP_TASK} ${VISION_BACKBONE} ${TRAIN_MODE}"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "Model:     ${MODEL_NAME_OR_PATH}"
echo "ViT:       ${VISION_TOWER}"
echo "ViT type:  ${MM_VISION_TOWER_TYPE}"
echo "Fusion:    ${MULTI_VISION_FUSION:-single}"
echo "Layer fusion: ${VISION_LAYER_FUSION_INDEXES:-off} (${VISION_LAYER_FUSION_TYPE})"
echo "Input:     ${INPUT_IMAGE_SIZE}"
echo "Train:     ${TRAIN_JSONL}"
echo "Output:    ${OUTPUT_DIR}"
echo "LoRA:      ${LORA_ENABLE}"
echo "Unfreeze ViT: ${UNFREEZE_MM_VISION_TOWER}"
echo "SwanLab:   ${SWANLAB_ENABLE}, project=${SWANLAB_PROJECT}, group=${SWANLAB_GROUP}, job=${SWANLAB_JOB_TYPE}, exp=${SWANLAB_EXPERIMENT_NAME}"
echo "============================================================"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_sft \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --version "${VERSION}" \
  "${TRAIN_VISION_ARGS[@]}" \
  --disable_deepstack "${DISABLE_DEEPSTACK}" \
  --mm_vision_select_layer -2 \
  --mm_vision_select_feature patch \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
  --data_path "${TRAIN_JSONL}" \
  --image_folder "${IMAGE_FOLDER}" \
  --image_aspect_ratio pad \
  --output_dir "${OUTPUT_DIR}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_target_scope "${LORA_TARGET_SCOPE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --mm_vision_fusion_lr "${MM_VISION_FUSION_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_steps "${MAX_STEPS}" \
  --warmup_steps 0 \
  --lr_scheduler_type constant \
  --bf16 "${BF16}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps 1 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --use_hf_progress_bar True \
  --report_to none \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_workspace "${SWANLAB_WORKSPACE}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type "${SWANLAB_JOB_TYPE}" \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --tf32 False \
  --ddp_find_unused_parameters False \
  --deepspeed "${DEEPSPEED_CONFIG}"

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
  CHECKPOINT_DIR=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
fi
[ -n "${CHECKPOINT_DIR}" ] && [ -d "${CHECKPOINT_DIR}" ] || { echo "No checkpoint found under ${OUTPUT_DIR}"; exit 1; }

python scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "${INFER_VISION_ARGS[@]}" \
  --test-json "${TEST_JSONL}" \
  --image-folder "${IMAGE_FOLDER}" \
  --num-samples "${NUM_INFER_SAMPLES}" \
  --prompt-mode dataset \
  --map-task "${MAP_TASK}" \
  --patch-size 256 \
  --coord-mode "${COORD_MODE}" \
  --coord-range "${COORD_RANGE}" \
  --conv-template "${VERSION}" \
  --device cuda \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0 \
  --output-dir "${INFER_DIR}" \
  --output-json "${INFER_DIR}/summary.json" \
  --eval-centerline

if [[ "${RUN_STATE_UPDATE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  python scripts/tools/infer_centerline_state_update.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    "${INFER_VISION_ARGS[@]}" \
    --patch-json "${TEST_JSONL}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-json "${STATE_DIR}/summary.json" \
    --output-dir "${STATE_DIR}/patches" \
    --conv-template "${VERSION}" \
    --device cuda \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --include-intersections \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --eval-centerline
fi

echo "SFT smoke finished: ${CHECKPOINT_DIR}"
