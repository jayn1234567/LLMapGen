#!/usr/bin/env bash
set -euo pipefail

# Local NPU GRPO smoke test: DINOv2 + Qwen3VL + no DeepStack + lane.
# This script does not install dependencies and does not use OBS.
# Edit the parameter block below to match the local NPU conda environment and paths.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

# ---------- Local environment ----------
CONDA_SH="/home/q/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="fastvlm"                       # Must contain torch_npu, transformers, deepspeed, peft.
NPU_IDS="0"                               # For multi-NPU debug, edit to "0,1" and NPROC_PER_NODE=2.
NPROC_PER_NODE=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29671

# ---------- Model paths ----------
# Use a stable SFT checkpoint as the GRPO starting policy, not the base Qwen3VL checkpoint.
SFT_CHECKPOINT="/media/q/data2/jjh/project/MLLM_project/outputs/test_qwen3vl"
VISION_TOWER="/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large"

# ---------- Data paths ----------
# If these files are missing, the script builds them from the local debug SFT JSONL.
TRAIN_JSONL="data/grpo_debug_lane20/train.jsonl"
TEST_JSONL="data/grpo_debug_lane20/test.jsonl"
IMAGE_FOLDER="data/av2_patch_256_fullimage_cutflag_test_v2"
OUTPUT_DIR="outputs/grpo_debug_lane_dinov2_qwen3vl_nodeepstack_local_npu"
INFER_DIR="${OUTPUT_DIR}/infer_test"

# ---------- Model config ----------
VERSION="conv_qwen_3_Dinov2_huawei"
INPUT_IMAGE_SIZE=518
DISABLE_DEEPSTACK=True
MODEL_MAX_LENGTH=2048
TRAINING_BRANCH="auto_lane"
MAP_TASK="lane"
COORD_MODE=auto
COORD_RANGE=1000

# ---------- GRPO sampling / reward ----------
NUM_GENERATIONS=2
MAX_NEW_TOKENS=128
TEMPERATURE=0.7
TOP_P=0.9
# Keep KL disabled for the smallest single-NPU smoke run, so we do not load a
# second reference model. For LoRA ZeRO3 multi-NPU debug, this can be set >0.
KL_BETA=0.0
CLIP_RANGE=0.2

REWARD_FORMAT_WEIGHT=0.08
REWARD_CENTERLINE_INSTANCE_WEIGHT=0.37
REWARD_CENTERLINE_LENGTH_WEIGHT=0.45
REWARD_CUT_TYPE_WEIGHT=0.05
REWARD_CUT_CONTINUITY_WEIGHT=0.05

# ---------- LoRA ----------
LORA_TARGET_SCOPE="llm"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# ---------- Optimization ----------
MAX_STEPS=1
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-6
BF16=True
LOGGING_STEPS=1
SAVE_STEPS=1
SAVE_TOTAL_LIMIT=1
DEEPSPEED_CONFIG=""                      # Empty means no DeepSpeed for the smallest local smoke.

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

python -c "import torch, torch_npu; assert hasattr(torch, 'npu') and torch.npu.is_available(), 'torch_npu is installed but NPU is not available'; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_count', torch.npu.device_count())"

if [ ! -f "${TRAIN_JSONL}" ] || [ ! -f "${TEST_JSONL}" ]; then
    python scripts/gpu/build_grpo_debug_data.py --limit 20 --test-count 4
fi

[ -d "${SFT_CHECKPOINT}" ] || { echo "SFT checkpoint not found: ${SFT_CHECKPOINT}"; exit 1; }
[ -d "${VISION_TOWER}" ] || { echo "Vision tower not found: ${VISION_TOWER}"; exit 1; }
[ -f "${TRAIN_JSONL}" ] || { echo "Train JSONL not found: ${TRAIN_JSONL}"; exit 1; }
[ -f "${TEST_JSONL}" ] || { echo "Test JSONL not found: ${TEST_JSONL}"; exit 1; }
[ -d "${IMAGE_FOLDER}" ] || { echo "Image folder not found: ${IMAGE_FOLDER}"; exit 1; }

export ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MLLM_LOG_RANK0_ONLY=1
mkdir -p "${OUTPUT_DIR}"

DEEPSPEED_ARGS=()
if [ -n "${DEEPSPEED_CONFIG}" ]; then
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}" --ddp_find_unused_parameters False --ddp_backend hccl)
fi

echo "============================================================"
echo "GRPO local NPU lane debug: DINOv2 + Qwen3VL + no DeepStack"
echo "NPUs:       ${ASCEND_RT_VISIBLE_DEVICES} (${NPROC_PER_NODE} process)"
echo "Model:      ${SFT_CHECKPOINT}"
echo "ViT:        ${VISION_TOWER}"
echo "Train:      ${TRAIN_JSONL}"
echo "Output:     ${OUTPUT_DIR}"
echo "KL beta:    ${KL_BETA}"
echo "DeepSpeed:  ${DEEPSPEED_CONFIG:-disabled}"
echo "============================================================"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m mllm.train.train_grpo \
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
    --max_steps "${MAX_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bf16 "${BF16}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --remove_unused_columns False \
    --report_to none \
    "${DEEPSPEED_ARGS[@]}"

python scripts/infer_centerline_checkpoint.py \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoint-1" \
    --vision_tower "${VISION_TOWER}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --disable_deepstack \
    --test-json "${TEST_JSONL}" \
    --image-folder "${IMAGE_FOLDER}" \
    --num-samples 2 \
    --prompt-mode dataset \
    --map-task lane \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template "${VERSION}" \
    --device npu \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --output-dir "${INFER_DIR}" \
    --output-json "${INFER_DIR}/summary.json" \
    --eval-centerline

echo "Local NPU GRPO debug finished."
