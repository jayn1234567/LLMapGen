#!/usr/bin/env bash
set -euo pipefail

# Formal GPU SFT entrypoint for Qwen-family LLMs plus single/multi ViT recipes.
# Named recipe scripts in this folder set environment variables and dispatch here.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

GPU_IDS=${GPU_IDS:-0}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29670}
USE_TORCHRUN=${USE_TORCHRUN:-True}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero2.json}

MODEL_FAMILY=${MODEL_FAMILY:-qwen3vl}  # qwen3vl, qwen3, qwen3_5, or custom.
QWEN3VL_MODEL_PATH=${QWEN3VL_MODEL_PATH:-/media/q/data2/jjh/project/MLLM_project/checkpoints/qwen/Qwen3-VL-2B-Instruct}
QWEN3_MODEL_PATH=${QWEN3_MODEL_PATH:-${QWEN3VL_MODEL_PATH}}
QWEN3_5_MODEL_PATH=${QWEN3_5_MODEL_PATH:-/media/q/data2/jjh/project/MLLM_project/checkpoints/qwen/Qwen3.5-4B-Instruct}
case "${MODEL_FAMILY}" in
  qwen3vl)
    MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${QWEN3VL_MODEL_PATH}}
    ;;
  qwen3)
    MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${QWEN3_MODEL_PATH}}
    ;;
  qwen3_5|qwen3.5)
    MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${QWEN3_5_MODEL_PATH}}
    ;;
  custom)
    MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH for MODEL_FAMILY=custom}
    ;;
  *)
    echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}; expected qwen3vl, qwen3, qwen3_5, or custom"
    exit 1
    ;;
esac

VISION_BACKBONE=${VISION_BACKBONE:-dinov2}  # dinov2, dinov3, multi_moe, dinov2_siglip_concat, dinov3_siglip_concat.
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
    DATASET_PHASE=phase_a
    MAP_TASK=${MAP_TASK:-lane}
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_a_lane20/train.jsonl}
    EVAL_JSONL=${EVAL_JSONL:-data/debug_phase_a_lane20/eval.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_a_lane20/test.jsonl}
    ;;
  phase_a_lane_intersection)
    DATASET_PHASE=phase_a
    MAP_TASK=${MAP_TASK:-lane_intersection}
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_a_lane_intersection20/train.jsonl}
    EVAL_JSONL=${EVAL_JSONL:-data/debug_phase_a_lane_intersection20/eval.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_a_lane_intersection20/test.jsonl}
    ;;
  phase_b|phase_b_lane)
    DATASET_PHASE=phase_b
    MAP_TASK=${MAP_TASK:-lane}
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_b_lane20/train.jsonl}
    EVAL_JSONL=${EVAL_JSONL:-data/debug_phase_b_lane20/eval.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_b_lane20/test.jsonl}
    ;;
  phase_b_lane_intersection)
    DATASET_PHASE=phase_b
    MAP_TASK=${MAP_TASK:-lane_intersection}
    TRAIN_JSONL=${TRAIN_JSONL:-data/debug_phase_b_lane_intersection20/train.jsonl}
    EVAL_JSONL=${EVAL_JSONL:-data/debug_phase_b_lane_intersection20/eval.jsonl}
    TEST_JSONL=${TEST_JSONL:-data/debug_phase_b_lane_intersection20/test.jsonl}
    ;;
  *)
    echo "Unsupported FLOW_PHASE=${FLOW_PHASE}; expected phase_a_lane, phase_a_lane_intersection, phase_b_lane, or phase_b_lane_intersection"
    exit 1
    ;;
esac

IMAGE_FOLDER=${IMAGE_FOLDER:-data/av2_patch_256_fullimage_cutflag_test_v2}
VERSION=${VERSION:-conv_qwen_3_Dinov2_huawei}
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}

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

NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:-}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-5e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}
MM_VISION_FUSION_LR=${MM_VISION_FUSION_LR:-${MM_PROJECTOR_LR}}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
BF16=${BF16:-True}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
SAVE_STEPS=${SAVE_STEPS:-400}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}
LOGGING_STEPS=${LOGGING_STEPS:-10}
ENABLE_EVAL=${ENABLE_EVAL:-False}
EVAL_STEPS=${EVAL_STEPS:-400}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-0}
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-False}
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-5}

RUN_ID=${RUN_ID:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_${MODEL_FAMILY}_${TRAIN_MODE}_gpu}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/formal_runs/${RUN_ID}}

SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}}
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-sft}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-${RUN_ID}}
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},${MODEL_FAMILY},${TRAIN_MODE}}
SWANLAB_MODE=${SWANLAB_MODE:-offline}
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
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
mkdir -p "${OUTPUT_DIR}"

TRAIN_VISION_ARGS=(
  --vision_tower "${VISION_TOWER}"
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}"
  --input_image_size "${INPUT_IMAGE_SIZE}"
)
if [[ "${DISABLE_DEEPSTACK}" =~ ^(0|false|False|FALSE|no|NO)$ && -n "${DEEPSTACK_VISUAL_INDEXES}" ]]; then
  TRAIN_VISION_ARGS+=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi
if [[ -n "${VISION_LAYER_FUSION_INDEXES}" ]]; then
  TRAIN_VISION_ARGS+=(
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
fi

TRAIN_LENGTH_ARGS=(--num_train_epochs "${NUM_EPOCHS}")
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_LENGTH_ARGS+=(--max_steps "${MAX_STEPS}")
fi

EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  [ -f "${EVAL_JSONL}" ] || { echo "Eval JSONL not found: ${EVAL_JSONL}"; exit 1; }
  EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")
  EVAL_ARGS=(
    --eval_data_path "${EVAL_JSONL}"
    --eval_image_folder "${IMAGE_FOLDER}"
    "${EVAL_STRATEGY_ARG}" steps
    --eval_steps "${EVAL_STEPS}"
    --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}"
    --best_eval_loss_dir eval_best
  )
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" && ! "${DEEPSPEED_CONFIG}" =~ ^(0|none|None|NONE|false|False|FALSE)$ ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

LAUNCH_CMD=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m mllm.train.train_sft)
if [[ "${USE_TORCHRUN}" =~ ^(0|false|False|FALSE|no|NO)$ ]]; then
  LAUNCH_CMD=(python -m mllm.train.train_sft)
fi

echo "============================================================"
echo "Formal GPU SFT: ${DATASET_PHASE} ${MAP_TASK} ${VISION_BACKBONE} ${MODEL_FAMILY}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "Launcher:   ${LAUNCH_CMD[*]}"
echo "Model:      ${MODEL_NAME_OR_PATH}"
echo "Vision:     ${VISION_TOWER}"
echo "Type:       ${MM_VISION_TOWER_TYPE}"
echo "Fusion:     ${MULTI_VISION_FUSION:-single}"
echo "Layer fuse: ${VISION_LAYER_FUSION_INDEXES:-off} (${VISION_LAYER_FUSION_TYPE})"
echo "DeepStack disabled: ${DISABLE_DEEPSTACK}, indexes=${DEEPSTACK_VISUAL_INDEXES:-auto}"
echo "Train:      ${TRAIN_JSONL}"
echo "Output:     ${OUTPUT_DIR}"
echo "LoRA:       ${LORA_ENABLE}, scope=${LORA_TARGET_SCOPE}"
echo "============================================================"

"${LAUNCH_CMD[@]}" \
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
  "${EVAL_ARGS[@]}" \
  --sample_seed 42 \
  --image_aspect_ratio pad \
  --bf16 "${BF16}" \
  --output_dir "${OUTPUT_DIR}" \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --mm_vision_fusion_lr "${MM_VISION_FUSION_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
  --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
  --best_train_loss_dir best \
  --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
  --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_target_scope "${LORA_TARGET_SCOPE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
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
  "${DEEPSPEED_ARGS[@]}"

RUN_INFER_AFTER_TRAIN=${RUN_INFER_AFTER_TRAIN:-False}
if [[ "${RUN_INFER_AFTER_TRAIN}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
    --output-dir "${OUTPUT_DIR}" \
    --best-name infer_best \
    --best-name eval_best \
    --best-name best \
    --allow-direct)
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  FLOW_PHASE="${FLOW_PHASE}" \
  MAP_TASK="${MAP_TASK}" \
  VISION_BACKBONE="${VISION_BACKBONE}" \
  TEST_JSONL="${TEST_JSONL}" \
  OUTPUT_DIR="${OUTPUT_DIR}/infer_patch" \
  VISION_TOWER="${VISION_TOWER}" \
  MM_VISION_TOWER_TYPE="${MM_VISION_TOWER_TYPE}" \
  INPUT_IMAGE_SIZE="${INPUT_IMAGE_SIZE}" \
  DISABLE_DEEPSTACK="${DISABLE_DEEPSTACK}" \
  DEEPSTACK_VISUAL_INDEXES="${DEEPSTACK_VISUAL_INDEXES}" \
  VISION_LAYER_FUSION_INDEXES="${VISION_LAYER_FUSION_INDEXES}" \
  VISION_LAYER_FUSION_TYPE="${VISION_LAYER_FUSION_TYPE}" \
  MULTI_VISION_TOWERS="${MULTI_VISION_TOWERS:-}" \
  MULTI_VISION_TOWER_TYPES="${MULTI_VISION_TOWER_TYPES:-}" \
  MULTI_VISION_INPUT_IMAGE_SIZES="${MULTI_VISION_INPUT_IMAGE_SIZES:-}" \
  MULTI_VISION_PRIMARY_INDEX="${MULTI_VISION_PRIMARY_INDEX:-}" \
  MULTI_VISION_HIDDEN_SIZE="${MULTI_VISION_HIDDEN_SIZE:-}" \
  MULTI_VISION_TARGET_GRID="${MULTI_VISION_TARGET_GRID:-}" \
  MULTI_VISION_FUSION="${MULTI_VISION_FUSION:-}" \
  bash "${SCRIPT_DIR}/test_qwen_family_formal_gpu.sh"
fi

echo "Formal GPU SFT finished: ${OUTPUT_DIR}"
