#!/usr/bin/env bash
set -euo pipefail

# GRPO debug launcher with the same A/B and DINOv2/DINOv3 naming as the NPU
# flow. Current project GRPO uses CUDA/vLLM prompt-embedding rollout, so this
# script fails fast on pure Ascend unless explicitly allowed on a CUDA host.

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
    MM_VISION_TOWER_TYPE=dinov2
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    ;;
  dinov3)
    VISION_TOWER="${DINOV3_PATH}"
    MM_VISION_TOWER_TYPE=dinov3
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

if [[ "${GRPO_ENABLE_CUDA_VLLM_FROM_NPU_SCRIPT:-False}" != "True" ]]; then
  cat <<EOF
ERROR: current GRPO backend requires CUDA/vLLM and is not supported as pure Ascend NPU training.

Requested debug flow:
  DATASET_PHASE=${DATASET_PHASE}
  MAP_TASK=${MAP_TASK}
  VISION_BACKBONE=${VISION_BACKBONE}

Run this script only on a CUDA/vLLM host and set:
  GRPO_ENABLE_CUDA_VLLM_FROM_NPU_SCRIPT=True
EOF
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

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

DATA_PATH="${DEBUG_DATA_ROOT}/${DATASET_PHASE}/train.jsonl"
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-}
if [ -z "${SFT_CHECKPOINT}" ]; then
  if SFT_CHECKPOINT=$(python scripts/tools/resolve_best_checkpoint.py \
      --output-dir "${SFT_OUTPUT_DIR}" \
      --best-name eval_best \
      --best-name best \
      --allow-direct 2>/dev/null); then
    :
  else
    SFT_CHECKPOINT=$(python - "${SFT_OUTPUT_DIR}" <<'PY'
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
    raise SystemExit(f"No SFT checkpoint found under {root}")
print(sorted(candidates)[-1][1])
PY
)
  fi
fi

OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
mkdir -p "${OUTPUT_DIR}"

for path in "${DATA_PATH}" "${IMAGE_FOLDER}" "${VISION_TOWER}" "${SFT_CHECKPOINT}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done
python -c "import vllm; print('vllm', getattr(vllm, '__version__', 'unknown'))"

if [ "${MAP_TASK}" = "lane_intersection" ]; then
  REWARD_INTERSECTION_WEIGHT=${REWARD_INTERSECTION_WEIGHT:-0.10}
else
  REWARD_INTERSECTION_WEIGHT=${REWARD_INTERSECTION_WEIGHT:-0.0}
fi

SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_GROUP=${SWANLAB_GROUP:-debug_grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}}
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-debug_grpo}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-debug_grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}}
SWANLAB_TAGS=${SWANLAB_TAGS:-debug,grpo,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},nodeepstack}
SWANLAB_MODE=${SWANLAB_MODE:-}
export SWANLAB_API_KEY

echo "GRPO debug:"
echo "  phase=${DATASET_PHASE} map_task=${MAP_TASK} vision=${VISION_BACKBONE}"
echo "  sft_checkpoint=${SFT_CHECKPOINT}"
echo "  data=${DATA_PATH}"
echo "  output=${OUTPUT_DIR}"

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
  --clip_range "${CLIP_RANGE:-0.2}" \
  --reward_format_weight "${REWARD_FORMAT_WEIGHT:-0.08}" \
  --reward_centerline_instance_weight "${REWARD_CENTERLINE_INSTANCE_WEIGHT:-0.37}" \
  --reward_centerline_length_weight "${REWARD_CENTERLINE_LENGTH_WEIGHT:-0.45}" \
  --reward_cut_type_weight "${REWARD_CUT_TYPE_WEIGHT:-0.05}" \
  --reward_cut_continuity_weight "${REWARD_CUT_CONTINUITY_WEIGHT:-0.05}" \
  --reward_intersection_weight "${REWARD_INTERSECTION_WEIGHT}" \
  --lora_enable True \
  --lora_target_scope "${LORA_TARGET_SCOPE:-llm}" \
  --lora_r "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --learning_rate "${LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --max_steps "${MAX_STEPS:-2}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-1}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --bf16 True \
  --model_max_length "${MODEL_MAX_LENGTH:-4096}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_workspace "${SWANLAB_WORKSPACE}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type "${SWANLAB_JOB_TYPE}" \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}"

echo "GRPO debug finished: ${OUTPUT_DIR}"
