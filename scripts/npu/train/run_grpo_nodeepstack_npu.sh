#!/usr/bin/env bash
set -euo pipefail

# Common GRPO launcher for explicit NPU flow wrappers.
#
# Current project GRPO is the formal vLLM prompt-embedding architecture:
#   HF multimodal actor -> prompt embeddings -> vLLM text decoder rollout.
# vLLM is CUDA-oriented, so this NPU entry fails fast by default instead of
# silently starting a run that cannot complete on Ascend. Set
# GRPO_ENABLE_CUDA_VLLM_FROM_NPU_SCRIPT=True only if this script is being used
# on a CUDA host while keeping the same flow naming convention.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:?set VISION_BACKBONE to dinov2 or dinov3}
DATASET_PHASE=${DATASET_PHASE:?set DATASET_PHASE to phase_a or phase_b}
MAP_TASK=${MAP_TASK:?set MAP_TASK to lane or lane_intersection}

case "${VISION_BACKBONE}" in
  dinov2|dinov3) ;;
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

if [[ "${GRPO_ENABLE_CUDA_VLLM_FROM_NPU_SCRIPT:-False}" != "True" ]]; then
  cat <<EOF
ERROR: current GRPO backend requires CUDA/vLLM and is not supported on Ascend NPU.

Requested flow:
  DATASET_PHASE=${DATASET_PHASE}
  MAP_TASK=${MAP_TASK}
  VISION_BACKBONE=${VISION_BACKBONE}

Use the GPU GRPO scripts for real training, or set
GRPO_ENABLE_CUDA_VLLM_FROM_NPU_SCRIPT=True only when this wrapper is launched
on a CUDA host for naming compatibility.
EOF
  exit 2
fi

case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER=${VISION_TOWER:-/cache/jjh/checkpoints/facebook_dinov2-large}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov2}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    ;;
  dinov3)
    VISION_TOWER=${VISION_TOWER:-/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov3}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
esac

DATASET_PATH=${DATASET_PATH:-/cache/unimapgen_v2/dataset}
if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
  DATA_PATH=${DATA_PATH:-${DATASET_PATH}/${DATASET_PHASE}/train.jsonl}
else
  DATA_PATH=${DATA_PATH:-${DATASET_PATH}/train.jsonl}
fi
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-/cache/unimapgen_v2/train_output/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack/best}
OUTPUT_DIR=${OUTPUT_DIR:-/cache/unimapgen_v2/train_output/grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_ENABLE=${SWANLAB_ENABLE:-True}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-mllm-grpo-${DATASET_PHASE}-${MAP_TASK}-${VISION_BACKBONE}-nodeepstack}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
SWANLAB_TAGS=${SWANLAB_TAGS:-grpo,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},qwen3vl8b,nodeepstack}

python -m mllm.train.train_grpo \
  --model_name_or_path "${SFT_CHECKPOINT}" \
  --version conv_qwen_3_Dinov2_huawei \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
  --input_image_size "${INPUT_IMAGE_SIZE}" \
  --disable_deepstack True \
  --tokenizer_use_fast False \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --image_aspect_ratio pad \
  --map_task "${MAP_TASK}" \
  --coord_mode auto \
  --coord_range 1000 \
  --output_dir "${OUTPUT_DIR}" \
  --rollout_backend vllm_prompt_embeds \
  --actor_num_gpus "${ACTOR_NUM_GPUS:-1}" \
  --rollout_num_gpus "${ROLLOUT_NUM_GPUS:-1}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.70}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-2048}" \
  --vllm_enforce_eager "${VLLM_ENFORCE_EAGER:-True}" \
  --num_generations "${NUM_GENERATIONS:-2}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.9}" \
  --kl_beta "${KL_BETA:-0.02}" \
  --clip_range 0.2 \
  --reward_format_weight "${REWARD_FORMAT_WEIGHT:-0.08}" \
  --reward_centerline_instance_weight "${REWARD_CENTERLINE_INSTANCE_WEIGHT:-0.37}" \
  --reward_centerline_length_weight "${REWARD_CENTERLINE_LENGTH_WEIGHT:-0.45}" \
  --reward_cut_type_weight "${REWARD_CUT_TYPE_WEIGHT:-0.05}" \
  --reward_cut_continuity_weight "${REWARD_CUT_CONTINUITY_WEIGHT:-0.05}" \
  --reward_intersection_weight "${REWARD_INTERSECTION_WEIGHT:-0.0}" \
  --lora_enable True \
  --lora_target_scope "${LORA_TARGET_SCOPE:-llm}" \
  --lora_r "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --learning_rate "${LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --max_steps "${MAX_STEPS:-100}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --bf16 True \
  --model_max_length "${MODEL_MAX_LENGTH:-4096}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE:-}"

