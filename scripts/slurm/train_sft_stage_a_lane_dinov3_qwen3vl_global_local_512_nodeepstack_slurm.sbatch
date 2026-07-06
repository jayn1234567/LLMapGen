#!/usr/bin/env bash
#SBATCH --job-name=gl512_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

# Submit with any cluster-specific resources, for example:
# sbatch --partition <partition> --gres <accelerator-request> scripts/slurm/train_sft_stage_a_lane_dinov3_qwen3vl_global_local_512_nodeepstack_slurm.sbatch

REPO_ROOT=${REPO_ROOT:-/mingli01/project/jn}
cd "${REPO_ROOT}"
mkdir -p logs/slurm

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-fastvlm}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
RUN_MODE=${RUN_MODE:-smoke_then_full}  # smoke, full, or smoke_then_full

MODEL_ROOT=${MODEL_ROOT:-/mingli01/data/checkpoints}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${MODEL_ROOT}/Qwen3-VL-8B-Instruct}
VISION_TOWER=${VISION_TOWER:-${MODEL_ROOT}/facebook_dinov3-vitl16-pretrain-lvd1689m}
MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov3}

DATASET_ROOT=${DATASET_ROOT:-/mingli01/data/outputs/data_line_samples_33w}
DATASET_PHASE=${DATASET_PHASE:-phase_a}
TRAIN_JSON=${TRAIN_JSON:-${DATASET_ROOT}/${DATASET_PHASE}/train.jsonl}
EVAL_JSON=${EVAL_JSON:-${DATASET_ROOT}/${DATASET_PHASE}/eval.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_ROOT}}

OUTPUT_ROOT=${OUTPUT_ROOT:-/mingli01/data/outputs/global_local_512}
WORK_DATA_ROOT=${WORK_DATA_ROOT:-${OUTPUT_ROOT}/prepared_${RUN_ID}}
SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR:-${OUTPUT_ROOT}/smoke_${RUN_ID}}
FULL_OUTPUT_DIR=${FULL_OUTPUT_DIR:-${OUTPUT_ROOT}/stage_a_lane_dinov3_qwen3vl_gl512_nodeepstack_e6_${RUN_ID}}

INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
PATCH_SIZE=${PATCH_SIZE:-512}
STRIDE=${STRIDE:-512}
CONTEXT_SIZE=${CONTEXT_SIZE:-1024}
CONTEXT_RESIZED_SIZE=${CONTEXT_RESIZED_SIZE:-512}
CONTEXT_IMAGE_KEY=${CONTEXT_IMAGE_KEY:-context_image}

NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29600}
DDP_BACKEND=${DDP_BACKEND:-hccl}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
  GRADIENT_ACCUMULATION_STEPS=1
fi

LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}
NUM_EPOCHS=${NUM_EPOCHS:-6}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-400}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}
LOGGING_STEPS=${LOGGING_STEPS:-10}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}

for path in "${MODEL_NAME_OR_PATH}" "${VISION_TOWER}" "${TRAIN_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

prepare_context_jsonl() {
  local input_json="$1"
  local output_json="$2"
  local max_samples="$3"
  local context_dir="$4"
  local args=(
    scripts/data/add_global_local_context.py
    --input-jsonl "${input_json}"
    --output-jsonl "${output_json}"
    --image-folder "${IMAGE_FOLDER}"
    --context-dir "${context_dir}"
    --context-image-key "${CONTEXT_IMAGE_KEY}"
    --context-size "${CONTEXT_SIZE}"
    --output-size "${CONTEXT_RESIZED_SIZE}"
    --patch-size "${PATCH_SIZE}"
    --stride "${STRIDE}"
    --overwrite
  )
  if [ "${max_samples}" -gt 0 ]; then
    args+=(--max-samples "${max_samples}")
  fi
  python "${args[@]}"
}

run_sft() {
  local train_json="$1"
  local output_dir="$2"
  local max_steps="$3"
  local epochs="$4"
  local swanlab_enable="$5"

  mkdir -p "${output_dir}"
  local extra_steps=()
  if [ "${max_steps}" -gt 0 ]; then
    extra_steps+=(--max_steps "${max_steps}")
  fi

  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_qwen \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version conv_qwen_3_Dinov2_huawei \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower True \
    --disable_deepstack True \
    --mm_use_view_type_embedding True \
    --mm_view_type_embedding_count 2 \
    --data_path "${train_json}" \
    --image_folder "${IMAGE_FOLDER}" \
    --context_image_key "${CONTEXT_IMAGE_KEY}" \
    --require_context_image True \
    --sample_seed 42 \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${output_dir}" \
    --num_train_epochs "${epochs}" \
    "${extra_steps[@]}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_best_train_loss True \
    --best_train_loss_start_step 5000 \
    --best_train_loss_dir best \
    --best_checkpoint_save_mode rotating_create_only \
    --best_checkpoint_keep_limit 5 \
    --use_hf_progress_bar True \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --swanlab_enable "${swanlab_enable}" \
    --swanlab_project unimapgen_v3 \
    --swanlab_experiment_name "stage_a_lane_dinov3_qwen3vl_gl512_nodeepstack_${RUN_ID}" \
    --swanlab_group sft_stage_a_lane_global_local_512 \
    --swanlab_job_type sft \
    --swanlab_tags sft,phase_a,lane,dinov3,qwen3vl8b,nodeepstack,global_local_512 \
    --swanlab_mode offline \
    --swanlab_log_dir "${output_dir}/swanlab" \
    --ddp_find_unused_parameters False \
    --ddp_backend "${DDP_BACKEND}" \
    --deepspeed "${DEEPSPEED_CONFIG}"
}

if [[ "${RUN_MODE}" == "smoke" || "${RUN_MODE}" == "smoke_then_full" ]]; then
  SMOKE_JSON="${WORK_DATA_ROOT}/smoke/train_gl512_smoke.jsonl"
  prepare_context_jsonl "${TRAIN_JSON}" "${SMOKE_JSON}" "${SMOKE_MAX_SAMPLES:-32}" "context_1024_resize512_smoke/${RUN_ID}"
  echo "Starting global-local smoke training: ${SMOKE_JSON}"
  run_sft "${SMOKE_JSON}" "${SMOKE_OUTPUT_DIR}" "${SMOKE_MAX_STEPS:-5}" 1 False
  echo "Smoke training finished successfully."
fi

if [[ "${RUN_MODE}" == "full" || "${RUN_MODE}" == "smoke_then_full" ]]; then
  FULL_TRAIN_JSON="${WORK_DATA_ROOT}/full/train_gl512.jsonl"
  prepare_context_jsonl "${TRAIN_JSON}" "${FULL_TRAIN_JSON}" 0 "context_1024_resize512/${RUN_ID}/train"
  if [ -e "${EVAL_JSON}" ]; then
    FULL_EVAL_JSON="${WORK_DATA_ROOT}/full/eval_gl512.jsonl"
    prepare_context_jsonl "${EVAL_JSON}" "${FULL_EVAL_JSON}" 0 "context_1024_resize512/${RUN_ID}/eval"
  fi
  echo "Starting global-local full training: ${FULL_TRAIN_JSON}"
  run_sft "${FULL_TRAIN_JSON}" "${FULL_OUTPUT_DIR}" 0 "${NUM_EPOCHS}" True
  echo "Full training finished successfully: ${FULL_OUTPUT_DIR}"
fi
