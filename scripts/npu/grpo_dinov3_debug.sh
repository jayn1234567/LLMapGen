#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NPU (Ascend) GRPO debug training script
# Qwen3-VL-8B LLM + DINOv3-L + no DeepStack | DDP, no DeepSpeed/ZeRO
# ============================================================
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "Current working path: ${PWD}"

# ====================== NPU environment ======================
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ====================== distributed parameters ======================
# no-ZeRO DDP: every NPU holds a full policy model and, when KL_BETA > 0, a full reference model.
# Override NPUS and NPROC_PER_NODE together for smaller debug runs.
if [[ -z "${MA_VJ_NAME:-}" ]]; then
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
    NPUS=${NPUS:-0,1,2,3,4,5,6,7}
    NPROC_PER_NODE=${NPROC_PER_NODE:-8}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
    NNODES=${NNODES:-$MA_NUM_HOSTS}
    NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
    NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}
    MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
    NPUS=${NPUS:-}
fi

MASTER_PORT=${MASTER_PORT:-6061}
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-local_grpo_qwen3vl_dinov3_nozero}
if [ -n "${NPUS}" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
fi

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES:-<platform default>}"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== HCCL & NPU settings ======================
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}

export CUDA_MAX_DEVICE_LOCK_RETRY=10
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

# ====================== local paths ======================
OBS_CACHE=${OBS_CACHE:-/cache}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-/cache/unimapgen_v2/train_output/sft_qwen3vl_dinov3/checkpoint-1}
DINOV3_PATH=${DINOV3_PATH:-${OBS_CACHE}/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}

DATASET_PHASE=${DATASET_PHASE:-phase_a}
DATASET_PATH=${DATASET_PATH:-/cache/unimapgen_v2/dataset}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
OUTPUT_PATH=${OUTPUT_PATH:-/cache/unimapgen_v2/train_output/grpo_qwen3vl_dinov3_nozero_debug}

if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
    TRAIN_PATH=${TRAIN_PATH:-${DATASET_PATH}/${DATASET_PHASE}/train.jsonl}
else
    TRAIN_PATH=${TRAIN_PATH:-${DATASET_PATH}/train.jsonl}
fi

for path in "${TRAIN_PATH}" "${IMAGE_FOLDER}" "${DINOV3_PATH}" "${SFT_CHECKPOINT}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: path does not exist: ${path}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ====================== GRPO debug parameters ======================
VERSION=${VERSION:-conv_qwen_3_Dinov2_huawei}
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}

TRAINING_BRANCH=${TRAINING_BRANCH:-auto_lane}
MAP_TASK=${MAP_TASK:-lane}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}

NUM_GENERATIONS=${NUM_GENERATIONS:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1.0}
KL_BETA=${KL_BETA:-0.02}
CLIP_RANGE=${CLIP_RANGE:-0.2}

REWARD_FORMAT_WEIGHT=${REWARD_FORMAT_WEIGHT:-0.08}
REWARD_CENTERLINE_INSTANCE_WEIGHT=${REWARD_CENTERLINE_INSTANCE_WEIGHT:-0.37}
REWARD_CENTERLINE_LENGTH_WEIGHT=${REWARD_CENTERLINE_LENGTH_WEIGHT:-0.45}
REWARD_CUT_TYPE_WEIGHT=${REWARD_CUT_TYPE_WEIGHT:-0.05}
REWARD_CUT_CONTINUITY_WEIGHT=${REWARD_CUT_CONTINUITY_WEIGHT:-0.05}

LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MAX_STEPS=${MAX_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-10}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
BF16=${BF16:-True}

echo "============================================================"
echo "GRPO DINOv3 debug: no DeepSpeed/ZeRO, DDP/HCCL when NPROC_PER_NODE > 1"
echo "SFT:          ${SFT_CHECKPOINT}"
echo "ViT:          ${DINOV3_PATH}"
echo "Train:        ${TRAIN_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output:       ${OUTPUT_PATH}"
echo "KL beta:      ${KL_BETA}"
echo "Sampling:     generations=${NUM_GENERATIONS}, max_new_tokens=${MAX_NEW_TOKENS}, temperature=${TEMPERATURE}, top_p=${TOP_P}"
echo "============================================================"

DDP_ARGS=()
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    DDP_ARGS=(--ddp_find_unused_parameters False --ddp_backend hccl)
fi

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_grpo \
    --model_name_or_path "${SFT_CHECKPOINT}" \
    --version "${VERSION}" \
    --vision_tower "${DINOV3_PATH}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack True \
    --data_path "${TRAIN_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio pad \
    --training_branch "${TRAINING_BRANCH}" \
    --map_task "${MAP_TASK}" \
    --coord_mode "${COORD_MODE}" \
    --coord_range "${COORD_RANGE}" \
    --output_dir "${OUTPUT_PATH}" \
    --grpo_backend custom \
    --num_generations "${NUM_GENERATIONS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --kl_beta "${KL_BETA}" \
    --clip_range "${CLIP_RANGE}" \
    --reward_format_weight "${REWARD_FORMAT_WEIGHT}" \
    --reward_centerline_instance_weight "${REWARD_CENTERLINE_INSTANCE_WEIGHT}" \
    --reward_centerline_length_weight "${REWARD_CENTERLINE_LENGTH_WEIGHT}" \
    --reward_cut_type_weight "${REWARD_CUT_TYPE_WEIGHT}" \
    --reward_cut_continuity_weight "${REWARD_CUT_CONTINUITY_WEIGHT}" \
    --lora_enable True \
    --lora_target_scope "${LORA_TARGET_SCOPE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_steps "${MAX_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to none \
    "${DDP_ARGS[@]}"

echo "=== GRPO DINOv3 no-ZeRO debug finished ==="
