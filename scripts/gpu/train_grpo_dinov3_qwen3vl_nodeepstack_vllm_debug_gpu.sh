#!/usr/bin/env bash
set -euo pipefail

# Formal GRPO debug path:
#   actor(HF multimodal policy) -> prompt embeddings -> vLLM text-decoder rollout
#   -> infer_index reward -> GRPO update -> adapter checkpoint + final merged checkpoint.
#
# This script intentionally does not use HF-local generate as the rollout path.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Paths. Keep these edited in the script for reproducible debug runs.
SFT_CHECKPOINT=${SFT_CHECKPOINT:-outputs/debug_runs/dinov3_sft_512_gpu_smoke_20260519/checkpoint-1}
VISION_BACKBONE=${VISION_BACKBONE:-dinov3}  # dinov2 or dinov3
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
  *)
    echo "Unsupported VISION_BACKBONE=${VISION_BACKBONE}; expected dinov2 or dinov3"
    exit 1
    ;;
esac
DATA_PATH=${DATA_PATH:-data/debug_phase_a_lane20/train.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/av2_patch_256_fullimage_cutflag_test_v2}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/debug_runs/grpo_dinov3_qwen3vl_nodeepstack_vllm_gpu}

# Task switch. Current default is lane. Change to lane_intersection only after
# preparing corresponding lane+intersection RL data.
MAP_TASK=${MAP_TASK:-lane}

# Rollout/training controls.
MAX_STEPS=${MAX_STEPS:-2}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
KL_BETA=${KL_BETA:-0.02}
LR=${LR:-1e-6}
REWARD_INTERSECTION_WEIGHT=${REWARD_INTERSECTION_WEIGHT:-0.0}

# Ray role placement. With CUDA_VISIBLE_DEVICES=0,2 this requests one visible
# GPU for the actor and one visible GPU for vLLM rollout.
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-1}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.70}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-True}

# SwanLab monitoring. Keep the API key outside the script:
#   export SWANLAB_API_KEY=...
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-mllm-grpo-debug-${MAP_TASK}-${VISION_BACKBONE}}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-grpo_${VISION_BACKBONE}_${MAP_TASK}_debug}
SWANLAB_TAGS=${SWANLAB_TAGS:-grpo,debug,${VISION_BACKBONE},${MAP_TASK}}
SWANLAB_MODE=${SWANLAB_MODE:-}
export SWANLAB_API_KEY

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
  --actor_num_gpus "${ACTOR_NUM_GPUS}" \
  --rollout_num_gpus "${ROLLOUT_NUM_GPUS}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --vllm_enforce_eager "${VLLM_ENFORCE_EAGER}" \
  --num_generations "${NUM_GENERATIONS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --kl_beta "${KL_BETA}" \
  --clip_range 0.2 \
  --reward_format_weight 0.08 \
  --reward_centerline_instance_weight 0.37 \
  --reward_centerline_length_weight 0.45 \
  --reward_cut_type_weight 0.05 \
  --reward_cut_continuity_weight 0.05 \
  --reward_intersection_weight "${REWARD_INTERSECTION_WEIGHT}" \
  --lora_enable True \
  --lora_target_scope llm \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --learning_rate "${LR}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --max_steps "${MAX_STEPS}" \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 2 \
  --bf16 True \
  --model_max_length 2048 \
  --dataloader_num_workers 0 \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}"
