#!/usr/bin/env bash
set -euo pipefail

# Ascend NPU GRPO debug launcher with the same A/B and DINOv2/DINOv3 naming as
# the formal NPU flow. It uses vLLM-Ascend prompt-embedding rollout.

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

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-npu}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export INF_NAN_MODE_ENABLE=1
export WITHOUT_JIT_COMPILE=1
export COMBINED_ENABLE=1

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
python -c "import vllm; print('vllm', getattr(vllm, '__version__', 'unknown')); import vllm_ascend; print('vllm_ascend imported')"

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
SWANLAB_MODE=${SWANLAB_MODE:-}          # Empty = SwanLab default cloud behavior; use offline/local/disabled when needed.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlab} # Local SwanLab files, beside checkpoint-* and merged/.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}  # Optional private SwanLab API host.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}  # Optional private SwanLab web host.
export SWANLAB_API_KEY

echo "GRPO debug:"
echo "  phase=${DATASET_PHASE} map_task=${MAP_TASK} vision=${VISION_BACKBONE}"
echo "  sft_checkpoint=${SFT_CHECKPOINT}"
echo "  data=${DATA_PATH}"
echo "  output=${OUTPUT_DIR}"
echo "  actor_npu=${ACTOR_NPU_DEVICES:-0} rollout_npu=${ROLLOUT_NPU_DEVICES:-1}"
echo "  swanlab=${SWANLAB_ENABLE} project=${SWANLAB_PROJECT} group=${SWANLAB_GROUP} mode=${SWANLAB_MODE} logdir=${SWANLAB_LOG_DIR}"
echo "  swanlab_url api=${SWANLAB_API_HOST:-default} web=${SWANLAB_WEB_HOST:-default}"

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
  --device_backend npu \
  --actor_npu_devices "${ACTOR_NPU_DEVICES:-0}" \
  --rollout_npu_devices "${ROLLOUT_NPU_DEVICES:-1}" \
  --actor_num_cpus "${ACTOR_NUM_CPUS:-4}" \
  --rollout_num_cpus "${ROLLOUT_NUM_CPUS:-4}" \
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
  --swanlab_mode "${SWANLAB_MODE}" \
  --swanlab_log_dir "${SWANLAB_LOG_DIR}" \
  --swanlab_api_host "${SWANLAB_API_HOST}" \
  --swanlab_web_host "${SWANLAB_WEB_HOST}"

echo "GRPO debug finished: ${OUTPUT_DIR}"
