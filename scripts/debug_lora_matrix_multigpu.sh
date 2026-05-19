#!/usr/bin/env bash
set -u

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/..")
cd "${REPO_ROOT}" || exit 1

GPU_IDS=${GPU_IDS:-1,2}
NPROC_PER_NODE=$(python - <<PY
print(len("${GPU_IDS}".split(",")))
PY
)
BASE_PORT=${BASE_PORT:-29610}

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}

QWEN2_PATH=${QWEN2_PATH:-checkpoints/llava-fastvithd_1.5b_stage2}
QWEN3VL_PATH=${QWEN3VL_PATH:-checkpoints/qwen/Qwen3-VL-2B-Instruct}
DINOV2_PATH=${DINOV2_PATH:-checkpoints/facebook_dinov2-large}
DINOV3_PATH=${DINOV3_PATH:-checkpoints/facebook/dinov3-vitl16-pretrain-lvd1689m}
TRAIN_JSON=${TRAIN_JSON:-data/train.jsonl}
TEST_JSON=${TEST_JSON:-data/test.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
OUTPUT_ROOT=${OUTPUT_ROOT:-/tmp/mllm_lora_matrix_multigpu_debug}

INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
MAX_STEPS=${MAX_STEPS:-2}
TRAIN_SAMPLE_LIMIT=${TRAIN_SAMPLE_LIMIT:-8}
NUM_INFER_SAMPLES=${NUM_INFER_SAMPLES:-2}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-1536}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-"6 12 18 23"}

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}

mkdir -p "${OUTPUT_ROOT}"

echo "repo: ${REPO_ROOT}"
echo "gpus: ${GPU_IDS}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "output_root: ${OUTPUT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

FAILED_CASES=()
PASSED_CASES=()
CASE_INDEX=0

require_file() {
    local path="$1"
    local desc="$2"
    if [ ! -f "${path}" ]; then
        echo "missing ${desc}: ${path}" >&2
        return 1
    fi
    return 0
}

require_grep() {
    local pattern="$1"
    local path="$2"
    local desc="$3"
    if ! grep -Eq "${pattern}" "${path}"; then
        echo "missing ${desc}: ${pattern} in ${path}" >&2
        return 1
    fi
    return 0
}

run_case() {
    local case_name="$1"
    local model_path="$2"
    local conv_template="$3"
    local vision_path="$4"
    local deepstack_mode="$5"

    CASE_INDEX=$((CASE_INDEX + 1))
    local train_port=$((BASE_PORT + CASE_INDEX * 2))
    local infer_port=$((train_port + 1))
    local case_dir="${OUTPUT_ROOT}/${case_name}"
    local train_dir="${case_dir}/train_lora"
    local infer_dir="${case_dir}/infer"
    local train_log="${case_dir}/train.log"
    local infer_log="${case_dir}/infer.log"

    rm -rf "${case_dir}"
    mkdir -p "${train_dir}" "${infer_dir}"

    local deepstack_train_args=()
    if [ "${deepstack_mode}" = "on" ]; then
        deepstack_train_args=(--disable_deepstack False --deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
    else
        deepstack_train_args=(--disable_deepstack True)
    fi

    echo "========== CASE ${CASE_INDEX}: ${case_name} =========="
    echo "model=${model_path}"
    echo "vision=${vision_path}"
    echo "deepstack=${deepstack_mode}"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr=127.0.0.1 \
        --master_port="${train_port}" \
        -m mllm.train.train_qwen \
        --model_name_or_path "${model_path}" \
        --version "${conv_template}" \
        --vision_tower "${vision_path}" \
        --input_image_size "${INPUT_IMAGE_SIZE}" \
        --mm_vision_select_layer -2 \
        --mm_projector_type mlp2x_gelu \
        "${deepstack_train_args[@]}" \
        --data_path "${TRAIN_JSON}" \
        --image_folder "${IMAGE_FOLDER}" \
        --train_sample_limit "${TRAIN_SAMPLE_LIMIT}" \
        --sample_seed 42 \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_dir "${train_dir}" \
        --num_train_epochs 1 \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --mm_projector_lr 1e-5 \
        --weight_decay 0.0 \
        --warmup_steps 0 \
        --lr_scheduler_type constant \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --remove_unused_columns false \
        --save_strategy no \
        --logging_steps 1 \
        --report_to none \
        --ddp_find_unused_parameters False \
        --lora_enable True \
        --lora_r 8 \
        --lora_alpha 16 >"${train_log}" 2>&1
    local train_status=$?
    if [ ${train_status} -ne 0 ]; then
        echo "CASE ${case_name} train failed"
        tail -n 80 "${train_log}"
        FAILED_CASES+=("${case_name}:train")
        return
    fi

    require_file "${train_dir}/adapter_model.safetensors" "LoRA adapter" || { FAILED_CASES+=("${case_name}:adapter"); return; }
    require_file "${train_dir}/non_lora_trainables.bin" "non-LoRA trainables" || { FAILED_CASES+=("${case_name}:non_lora"); return; }
    require_grep "DI_throughput: [0-9.]+ tokens/s/npu" "${train_dir}/train_metrics.log" "DI throughput" || { FAILED_CASES+=("${case_name}:throughput"); return; }
    require_grep "global_step: ${MAX_STEPS}" "${train_dir}/train_metrics.log" "final global step" || { FAILED_CASES+=("${case_name}:global_step"); return; }

    if [ "${deepstack_mode}" = "on" ]; then
        require_grep "DeepStack \\(real injection\\) enabled" "${train_log}" "DeepStack enabled train log" || { FAILED_CASES+=("${case_name}:deepstack_train"); return; }
    else
        if grep -q "DeepStack (real injection) enabled" "${train_log}"; then
            echo "DeepStack unexpectedly enabled in ${case_name}" >&2
            FAILED_CASES+=("${case_name}:deepstack_disabled")
            return
        fi
    fi

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr=127.0.0.1 \
        --master_port="${infer_port}" \
        scripts/infer_centerline_checkpoint.py \
        --checkpoint-dir "${train_dir}" \
        --vision_tower "${vision_path}" \
        --input_image_size "${INPUT_IMAGE_SIZE}" \
        --test-json "${TEST_JSON}" \
        --num-samples "${NUM_INFER_SAMPLES}" \
        --image-folder "${IMAGE_FOLDER}" \
        --prompt-mode dataset \
        --conv-template "${conv_template}" \
        --output-dir "${infer_dir}" \
        --output-json "${infer_dir}/summary.json" \
        --temperature 0.0 \
        --max-new-tokens 8 >"${infer_log}" 2>&1
    local infer_status=$?
    if [ ${infer_status} -ne 0 ]; then
        echo "CASE ${case_name} infer failed"
        tail -n 100 "${infer_log}"
        FAILED_CASES+=("${case_name}:infer")
        return
    fi

    require_grep "Loaded [0-9]+ non-LoRA trainable tensors from LoRA checkpoint" "${infer_log}" "LoRA non-LoRA load" || { FAILED_CASES+=("${case_name}:lora_infer_load"); return; }
    if [ "${deepstack_mode}" = "on" ]; then
        require_grep "DeepStack \\(real injection\\) enabled" "${infer_log}" "DeepStack enabled infer log" || { FAILED_CASES+=("${case_name}:deepstack_infer"); return; }
    fi

    DEBUG_CASE_DIR="${case_dir}" DEBUG_NPROC="${NPROC_PER_NODE}" DEBUG_NUM_INFER_SAMPLES="${NUM_INFER_SAMPLES}" DEBUG_CONV="${conv_template}" python - <<'PY'
import json
import os
from pathlib import Path

case_dir = Path(os.environ["DEBUG_CASE_DIR"])
nproc = int(os.environ["DEBUG_NPROC"])
num_infer_samples = int(os.environ["DEBUG_NUM_INFER_SAMPLES"])
conv = os.environ["DEBUG_CONV"]
infer_dir = case_dir / "infer"
summary_files = sorted(infer_dir.glob("summary_rank*.json"))
sample_files = sorted(p for p in infer_dir.glob("rank*_*.json") if not p.name.startswith("summary_"))
assert len(summary_files) == nproc, f"expected {nproc} summary_rank files, got {len(summary_files)}"
expected_sample_files = min(num_infer_samples, nproc) if num_infer_samples > 0 else nproc
assert len(sample_files) >= expected_sample_files, f"expected at least {expected_sample_files} rank sample files, got {len(sample_files)}"
for path in summary_files:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    for item in data:
        assert item["input_token_len"] > 0
        assert item["output_token_len"] > 0
        assert item["conv_template"] == conv
        assert "<image>" in item["prompt"]
print("matrix_case_outputs_ok")
PY
    local output_status=$?
    if [ ${output_status} -ne 0 ]; then
        FAILED_CASES+=("${case_name}:outputs")
        return
    fi

    PASSED_CASES+=("${case_name}")
    echo "CASE ${case_name} passed"
}

run_case "qwen2p5_dinov2_deepstack_on"  "${QWEN2_PATH}"   "conv_qwen_2_Dinov2_huawei" "${DINOV2_PATH}" "on"
run_case "qwen2p5_dinov2_deepstack_off" "${QWEN2_PATH}"   "conv_qwen_2_Dinov2_huawei" "${DINOV2_PATH}" "off"
run_case "qwen2p5_dinov3_deepstack_on"  "${QWEN2_PATH}"   "conv_qwen_2_Dinov2_huawei" "${DINOV3_PATH}" "on"
run_case "qwen2p5_dinov3_deepstack_off" "${QWEN2_PATH}"   "conv_qwen_2_Dinov2_huawei" "${DINOV3_PATH}" "off"
run_case "qwen3vl2b_dinov2_deepstack_on"  "${QWEN3VL_PATH}" "conv_qwen_3_Dinov2_huawei" "${DINOV2_PATH}" "on"
run_case "qwen3vl2b_dinov2_deepstack_off" "${QWEN3VL_PATH}" "conv_qwen_3_Dinov2_huawei" "${DINOV2_PATH}" "off"
run_case "qwen3vl2b_dinov3_deepstack_on"  "${QWEN3VL_PATH}" "conv_qwen_3_Dinov2_huawei" "${DINOV3_PATH}" "on"
run_case "qwen3vl2b_dinov3_deepstack_off" "${QWEN3VL_PATH}" "conv_qwen_3_Dinov2_huawei" "${DINOV3_PATH}" "off"

echo "========== MATRIX SUMMARY =========="
printf 'passed: %s\n' "${PASSED_CASES[@]:-none}"
if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    printf 'failed: %s\n' "${FAILED_CASES[@]}"
    exit 1
fi
echo "all matrix cases passed"
echo "output_root: ${OUTPUT_ROOT}"
