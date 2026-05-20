#!/usr/bin/env bash
set -euo pipefail

# GRPO lane-only LoRA finetuning without DeepSpeed ZeRO.
# Default target: multi-NPU DDP on 60GB Ascend NPUs, Qwen3-VL-8B LLM + DINOv3-L, no DeepStack.
# This avoids the NPU ZeRO3 + HF generate synced_gpus path.
# Note: DDP does not shard model memory. Each NPU holds a full policy model and
# a full frozen reference model when KL_BETA > 0.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# ---------- Runtime ----------
# With no ZeRO, every process holds a full model copy. Use fewer NPUs for debug
# by overriding NPUS and NPROC_PER_NODE together.
NPUS=${NPUS:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29641}

# ---------- Paths ----------
# Use an SFT checkpoint as the GRPO starting policy.
SFT_CHECKPOINT=${SFT_CHECKPOINT:-/cache/unimapgen_v2/train_output/sft_qwen3vl_dinov3/checkpoint-1}
VISION_TOWER=${VISION_TOWER:-/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
TRAIN_JSONL=${TRAIN_JSONL:-/cache/unimapgen_v2/dataset/phase_a/train.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-/cache/unimapgen_v2/dataset}
OUTPUT_DIR=${OUTPUT_DIR:-/cache/unimapgen_v2/train_output/grpo_qwen3vl_dinov3_nozero}

# ---------- Model ----------
VERSION=${VERSION:-conv_qwen_3_Dinov2_huawei}
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}

# ---------- GRPO ----------
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

# Reward weights. Centerline metrics use infer_index line matching.
REWARD_FORMAT_WEIGHT=${REWARD_FORMAT_WEIGHT:-0.08}
REWARD_CENTERLINE_INSTANCE_WEIGHT=${REWARD_CENTERLINE_INSTANCE_WEIGHT:-0.37}
REWARD_CENTERLINE_LENGTH_WEIGHT=${REWARD_CENTERLINE_LENGTH_WEIGHT:-0.45}
REWARD_CUT_TYPE_WEIGHT=${REWARD_CUT_TYPE_WEIGHT:-0.05}
REWARD_CUT_CONTINUITY_WEIGHT=${REWARD_CUT_CONTINUITY_WEIGHT:-0.05}

# ---------- LoRA ----------
LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# ---------- Optimization ----------
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MAX_STEPS=${MAX_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-20}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
BF16=${BF16:-True}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

for path in "${SFT_CHECKPOINT}" "${VISION_TOWER}" "${TRAIN_JSONL}" "${IMAGE_FOLDER}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: path does not exist: ${path}"
        exit 1
    fi
done

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
mkdir -p "${OUTPUT_DIR}"

LAUNCHER=(python -m mllm.train.train_grpo)
DDP_ARGS=()
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    LAUNCHER=(
        torchrun
        --nproc_per_node "${NPROC_PER_NODE}"
        --master_addr "${MASTER_ADDR}"
        --master_port "${MASTER_PORT}"
        -m mllm.train.train_grpo
    )
    DDP_ARGS=(--ddp_backend hccl --ddp_find_unused_parameters False)
fi

echo "============================================================"
echo "GRPO NPU lane no-ZeRO: DINOv3 + Qwen3VL + LoRA + no DeepStack"
echo "NPUs:         ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Processes:    ${NPROC_PER_NODE}"
echo "SFT:          ${SFT_CHECKPOINT}"
echo "ViT:          ${VISION_TOWER}"
echo "Train JSONL:  ${TRAIN_JSONL}"
echo "Output:       ${OUTPUT_DIR}"
echo "DeepSpeed:    disabled"
echo "DDP:          $([ "${NPROC_PER_NODE}" -gt 1 ] && echo "enabled/hccl" || echo "disabled")"
echo "Sampling:     num_generations=${NUM_GENERATIONS}, max_new_tokens=${MAX_NEW_TOKENS}, temperature=${TEMPERATURE}, top_p=${TOP_P}"
echo "KL beta:      ${KL_BETA} (loads a frozen reference model when > 0)"
echo "LoRA:         scope=${LORA_TARGET_SCOPE}, r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "============================================================"

"${LAUNCHER[@]}" \
    --model_name_or_path "${SFT_CHECKPOINT}" \
    --version "${VERSION}" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack "${DISABLE_DEEPSTACK}" \
    --data_path "${TRAIN_JSONL}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio pad \
    --training_branch "${TRAINING_BRANCH}" \
    --map_task "${MAP_TASK}" \
    --coord_mode "${COORD_MODE}" \
    --coord_range "${COORD_RANGE}" \
    --output_dir "${OUTPUT_DIR}" \
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
    --remove_unused_columns False \
    --report_to none \
    "${DDP_ARGS[@]}"

echo "=== GRPO no-ZeRO training finished ==="
