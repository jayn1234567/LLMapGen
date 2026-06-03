#!/usr/bin/env bash
set -euo pipefail

# Local Ascend NPU SFT smoke/debug launcher.
# It samples a tiny split from DATASET_ROOT and writes checkpoints under this
# repository's checkpoints/debug directory.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:-dinov2}  # dinov2, dinov3, multi_moe, dinov2_siglip_concat, or dinov3_siglip_concat
DATASET_PHASE=${DATASET_PHASE:-phase_a}     # phase_a or phase_b
MAP_TASK=${MAP_TASK:-lane}                  # lane or lane_intersection
DEBUG_RUN_NAME=${DEBUG_RUN_NAME:-local_debug}

DATASET_ROOT=${DATASET_ROOT:-/cache/data/data_line_samples_33w}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_ROOT}}
DEBUG_DATA_ROOT=${DEBUG_DATA_ROOT:-${REPO_ROOT}/checkpoints/debug_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/debug}

QWEN3VL_PATH=${QWEN3VL_PATH:-/cache/jjh/checkpoints/Qwen3-VL-8B-Instruct}
DINOV2_PATH=${DINOV2_PATH:-/cache/jjh/checkpoints/facebook_dinov2-large}
DINOV3_PATH=${DINOV3_PATH:-/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
SIGLIP_PATH=${SIGLIP_PATH:-/cache/jjh/checkpoints/google_siglip-large-patch16-384}
REQUIRED_VISION_TOWERS=()

case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER="${DINOV2_PATH}"
    MM_VISION_TOWER_TYPE=dinov2
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    REQUIRED_VISION_TOWERS=("${VISION_TOWER}")
    ;;
  dinov3)
    VISION_TOWER="${DINOV3_PATH}"
    MM_VISION_TOWER_TYPE=dinov3
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    REQUIRED_VISION_TOWERS=("${VISION_TOWER}")
    ;;
  multi_moe|multi_vision_moe|dual_dino_moe)
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINOV2_PATH},${DINOV3_PATH}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov2,dinov3}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,512}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-1}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-softmax_router}
    VISION_TOWER="${MULTI_VISION_TOWERS}"
    MM_VISION_TOWER_TYPE=multi_moe
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    IFS=',' read -r -a REQUIRED_VISION_TOWERS <<< "${MULTI_VISION_TOWERS}"
    ;;
  dinov2_siglip_concat|dinov2_siglip|dinosiglip_v2)
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINOV2_PATH},${SIGLIP_PATH}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov2,siglip}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,384}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-0}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-concat_projector}
    VISION_TOWER="${MULTI_VISION_TOWERS}"
    MM_VISION_TOWER_TYPE=multi_concat
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    IFS=',' read -r -a REQUIRED_VISION_TOWERS <<< "${MULTI_VISION_TOWERS}"
    ;;
  dinov3_siglip_concat|dinov3_siglip|dinosiglip_v3)
    MULTI_VISION_TOWERS=${MULTI_VISION_TOWERS:-${DINOV3_PATH},${SIGLIP_PATH}}
    MULTI_VISION_TOWER_TYPES=${MULTI_VISION_TOWER_TYPES:-dinov3,siglip}
    MULTI_VISION_INPUT_IMAGE_SIZES=${MULTI_VISION_INPUT_IMAGE_SIZES:-512,384}
    MULTI_VISION_PRIMARY_INDEX=${MULTI_VISION_PRIMARY_INDEX:-0}
    MULTI_VISION_HIDDEN_SIZE=${MULTI_VISION_HIDDEN_SIZE:-1024}
    MULTI_VISION_TARGET_GRID=${MULTI_VISION_TARGET_GRID:-32}
    MULTI_VISION_FUSION=${MULTI_VISION_FUSION:-concat_projector}
    VISION_TOWER="${MULTI_VISION_TOWERS}"
    MM_VISION_TOWER_TYPE=multi_concat
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    IFS=',' read -r -a REQUIRED_VISION_TOWERS <<< "${MULTI_VISION_TOWERS}"
    ;;
  *) echo "ERROR: VISION_BACKBONE must be dinov2, dinov3, multi_moe, dinov2_siglip_concat, or dinov3_siglip_concat"; exit 1 ;;
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

if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi
MASTER_PORT=${MASTER_PORT:-6060}

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export INF_NAN_MODE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export WITHOUT_JIT_COMPILE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE
export COMBINED_ENABLE=1
export OMP_NUM_THREADS=1
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
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

TRAIN_PATH="${DEBUG_DATA_ROOT}/${DATASET_PHASE}/train.jsonl"
EVAL_PATH="${DEBUG_DATA_ROOT}/${DATASET_PHASE}/eval.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${DEBUG_RUN_NAME}/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}"
mkdir -p "${OUTPUT_DIR}"

for path in "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}" "${QWEN3VL_PATH}" "${REQUIRED_VISION_TOWERS[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done

TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-8}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
total_devices=$(( NNODES * NPROC_PER_NODE ))
micro_batch=$(( total_devices * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + micro_batch - 1) / micro_batch ))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
  GRADIENT_ACCUMULATION_STEPS=1
fi

MAX_STEPS=${MAX_STEPS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}
MM_VISION_FUSION_LR=${MM_VISION_FUSION_LR:-${MM_PROJECTOR_LR}}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-1}
EVAL_STEPS=${EVAL_STEPS:-1}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
ENABLE_EVAL=${ENABLE_EVAL:-True}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-1}
BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best}
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-True}
BEST_EVAL_LOSS_DIR=${BEST_EVAL_LOSS_DIR:-eval_best}
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-1}

SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_GROUP=${SWANLAB_GROUP:-debug_sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}}
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-debug_sft}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-debug_sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}}
SWANLAB_TAGS=${SWANLAB_TAGS:-debug,sft,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},nodeepstack}
SWANLAB_MODE=${SWANLAB_MODE:-}                             # Empty = SwanLab default cloud behavior; use offline/local/disabled when needed.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlab}  # Local SwanLab files, beside checkpoint-* and best dirs.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}                     # Optional private SwanLab API host.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}                     # Optional private SwanLab web host.
export SWANLAB_API_KEY

EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")
EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(
    --eval_data_path "${EVAL_PATH}"
    --eval_image_folder "${IMAGE_FOLDER}"
    "${EVAL_STRATEGY_ARG}" steps
    --eval_steps "${EVAL_STEPS}"
    --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}"
    --best_eval_loss_dir "${BEST_EVAL_LOSS_DIR}"
  )
fi

VISION_ARGS=(
  --vision_tower "${VISION_TOWER}"
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}"
  --input_image_size "${INPUT_IMAGE_SIZE}"
)
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

echo "SFT debug:"
echo "  phase=${DATASET_PHASE} map_task=${MAP_TASK} vision=${VISION_BACKBONE}"
echo "  vision_tower=${VISION_TOWER}"
echo "  vision_type=${MM_VISION_TOWER_TYPE} fusion=${MULTI_VISION_FUSION:-single}"
echo "  train=${TRAIN_PATH}"
echo "  image_folder=${IMAGE_FOLDER}"
echo "  output=${OUTPUT_DIR}"
echo "  devices=${total_devices} target_batch=${TARGET_GLOBAL_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS}"
echo "  swanlab=${SWANLAB_ENABLE} project=${SWANLAB_PROJECT} group=${SWANLAB_GROUP} mode=${SWANLAB_MODE} logdir=${SWANLAB_LOG_DIR}"
echo "  swanlab_url api=${SWANLAB_API_HOST:-default} web=${SWANLAB_WEB_HOST:-default}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  --model_name_or_path "${QWEN3VL_PATH}" \
  --version conv_qwen_3_Dinov2_huawei \
  "${VISION_ARGS[@]}" \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower True \
  --disable_deepstack True \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --sample_seed "${SAMPLE_SEED}" \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
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
  --best_train_loss_dir "${BEST_TRAIN_LOSS_DIR}" \
  --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
  --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
  --use_hf_progress_bar True \
  --logging_steps 1 \
  --report_to none \
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
  --swanlab_web_host "${SWANLAB_WEB_HOST}" \
  --ddp_find_unused_parameters False \
  --ddp_backend hccl \
  --deepspeed "${DEEPSPEED_CONFIG}"

echo "SFT debug finished: ${OUTPUT_DIR}"
python scripts/tools/resolve_best_checkpoint.py \
  --output-dir "${OUTPUT_DIR}" \
  --best-name eval_best \
  --best-name best \
  --allow-direct || true
