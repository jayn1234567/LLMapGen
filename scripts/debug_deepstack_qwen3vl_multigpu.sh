#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/..")
cd "${REPO_ROOT}"

GPU_IDS=${GPU_IDS:-1,2}
NPROC_PER_NODE=$(python - <<PY
print(len("${GPU_IDS}".split(",")))
PY
)
MASTER_PORT=${MASTER_PORT:-29531}

CONDA_SH=${CONDA_SH:-/home/q/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-fastvlm}

QWEN3VL_PATH=${QWEN3VL_PATH:-checkpoints/qwen/Qwen3-VL-2B-Instruct}
DINOV3_PATH=${DINOV3_PATH:-checkpoints/facebook/dinov3-vitl16-pretrain-lvd1689m}
INFER_CHECKPOINT_DIR=${INFER_CHECKPOINT_DIR:-outputs/test_dinov3_qwen3vl}
TRAIN_JSON=${TRAIN_JSON:-data/train.jsonl}
TEST_JSON=${TEST_JSON:-data/test.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
CONV_TEMPLATE=${CONV_TEMPLATE:-conv_qwen_3_Dinov2_huawei}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-"6 12 18 23"}
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-224}
MAX_STEPS=${MAX_STEPS:-2}
TRAIN_SAMPLE_LIMIT=${TRAIN_SAMPLE_LIMIT:-4}
NUM_INFER_SAMPLES=${NUM_INFER_SAMPLES:-2}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-1536}
OUTPUT_ROOT=${OUTPUT_ROOT:-/tmp/mllm_deepstack_qwen3vl_multigpu_debug}

TRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/train_lora"
INFER_OUTPUT_DIR="${OUTPUT_ROOT}/infer"
LOG_DIR="${OUTPUT_ROOT}/logs"
SHAPE_LOG="${LOG_DIR}/shape_probe.log"
TRAIN_LOG="${LOG_DIR}/train.log"
INFER_LOG="${LOG_DIR}/infer.log"

mkdir -p "${TRAIN_OUTPUT_DIR}" "${INFER_OUTPUT_DIR}" "${LOG_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LLAVA_LOG_RANK0_ONLY=${LLAVA_LOG_RANK0_ONLY:-1}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}

echo "repo: ${REPO_ROOT}"
echo "gpus: ${GPU_IDS}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "output_root: ${OUTPUT_ROOT}"
echo "qwen3vl: ${QWEN3VL_PATH}"
echo "dinov3: ${DINOV3_PATH}"
echo "infer_checkpoint: ${INFER_CHECKPOINT_DIR}"
echo "deepstack_indexes: ${DEEPSTACK_VISUAL_INDEXES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

require_file() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo "ERROR missing file: ${path}" >&2
        exit 1
    fi
}

require_grep() {
    local pattern="$1"
    local path="$2"
    local desc="$3"
    if ! grep -Eq "${pattern}" "${path}"; then
        echo "ERROR missing ${desc}: ${pattern} in ${path}" >&2
        exit 1
    fi
}

echo "========== shape probe =========="
DEBUG_INFER_CHECKPOINT_DIR="${INFER_CHECKPOINT_DIR}" \
DEBUG_DINOV3_PATH="${DINOV3_PATH}" \
DEBUG_TEST_JSON="${TEST_JSON}" \
DEBUG_IMAGE_FOLDER="${IMAGE_FOLDER}" \
DEBUG_CONV_TEMPLATE="${CONV_TEMPLATE}" \
DEBUG_DEEPSTACK_VISUAL_INDEXES="${DEEPSTACK_VISUAL_INDEXES}" \
DEBUG_INPUT_IMAGE_SIZE="${INPUT_IMAGE_SIZE}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
python - <<'PY' 2>&1 | tee "${SHAPE_LOG}"
import json
import os
from pathlib import Path

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from scripts.infer_centerline_checkpoint import _load_full_finetune_model, build_prompt

checkpoint_dir = Path(os.environ["DEBUG_INFER_CHECKPOINT_DIR"])
vision_tower = os.environ["DEBUG_DINOV3_PATH"]
test_json = Path(os.environ["DEBUG_TEST_JSON"])
image_folder = Path(os.environ["DEBUG_IMAGE_FOLDER"])
conv_template = os.environ["DEBUG_CONV_TEMPLATE"]
deepstack_indexes = [int(x) for x in os.environ["DEBUG_DEEPSTACK_VISUAL_INDEXES"].split()]
input_image_size = int(os.environ["DEBUG_INPUT_IMAGE_SIZE"])

tokenizer, model, image_processor = _load_full_finetune_model(
    checkpoint_dir,
    "cuda:0",
    config_overrides={
        "mm_vision_tower": vision_tower,
        "vision_tower": vision_tower,
        "input_image_size": input_image_size,
        "disable_deepstack": False,
        "deepstack_visual_indexes": deepstack_indexes,
    },
)
model.eval()

with test_json.open("r", encoding="utf-8") as f:
    first = f.read(1)
    f.seek(0)
    record = json.load(f)[0] if first == "[" else json.loads(next(line for line in f if line.strip()))

image_path = (image_folder / record["image"]).resolve()
image = Image.open(image_path).convert("RGB")
prompt_text = record.get("conversations", [{"value": "<image>"}])[0]["value"]
prompt = build_prompt(prompt_text, conv_template)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=model.device)
images = process_images([image], image_processor, model.config)
vt = model.get_vision_tower()
if isinstance(images, list):
    images = [img.to(device=vt.device, dtype=vt.dtype) for img in images]
else:
    images = images.to(device=vt.device, dtype=vt.dtype)

with torch.no_grad():
    _, _, new_attention_mask, _, inputs_embeds, _, visual_pos_mask, deepstack_embeds = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            images=images,
            image_sizes=[image.size],
        )
    )

visual_tokens = int(visual_pos_mask.sum().item())
print(f"DEBUG_SHAPE input_ids {tuple(input_ids.shape)}")
print(f"DEBUG_SHAPE inputs_embeds {tuple(inputs_embeds.shape)}")
print(f"DEBUG_SHAPE attention_mask {tuple(new_attention_mask.shape)}")
print(f"DEBUG_SHAPE visual_pos_mask {tuple(visual_pos_mask.shape)} visual_tokens {visual_tokens}")
print(f"DEBUG_SHAPE deepstack_count {len(deepstack_embeds)} expected {len(deepstack_indexes)}")

assert len(deepstack_embeds) == len(deepstack_indexes), "deepstack count mismatch"
assert visual_tokens > 0, "no visual tokens found"
for idx, tensor in enumerate(deepstack_embeds):
    print(f"DEBUG_SHAPE deepstack_{idx} {tuple(tensor.shape)}")
    assert tensor.ndim == 2, "deepstack tensor must be flattened token-aligned"
    assert tensor.shape[0] == visual_tokens, "deepstack token count must match visual_pos_mask"
    assert tensor.shape[1] == inputs_embeds.shape[-1], "deepstack hidden size must match LLM hidden size"

print("DEBUG_CHECK deepstack_count_ok")
print("DEBUG_CHECK visual_token_alignment_ok")
print("DEBUG_CHECK hidden_size_alignment_ok")
PY

require_grep "DEBUG_CHECK deepstack_count_ok" "${SHAPE_LOG}" "DeepStack feature count check"
require_grep "DEBUG_CHECK visual_token_alignment_ok" "${SHAPE_LOG}" "visual token alignment check"
require_grep "DEBUG_CHECK hidden_size_alignment_ok" "${SHAPE_LOG}" "hidden size alignment check"

echo "========== 2-GPU training smoke =========="
rm -rf "${TRAIN_OUTPUT_DIR}"
mkdir -p "${TRAIN_OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    -m llava.train.train_qwen \
    --model_name_or_path "${QWEN3VL_PATH}" \
    --version "${CONV_TEMPLATE}" \
    --vision_tower "${DINOV3_PATH}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES} \
    --data_path "${TRAIN_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --train_sample_limit "${TRAIN_SAMPLE_LIMIT}" \
    --sample_seed 42 \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
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
    --lora_alpha 16 2>&1 | tee "${TRAIN_LOG}"

require_grep "DeepStack \\(real injection\\) enabled" "${TRAIN_LOG}" "DeepStack enabled training log"
require_grep "Using DINOv3 input image size: ${INPUT_IMAGE_SIZE}" "${TRAIN_LOG}" "DINOv3 image size training log"
require_grep "DI_throughput: [0-9.]+ tokens/s/npu" "${TRAIN_OUTPUT_DIR}/train_metrics.log" "DI throughput metric"
require_file "${TRAIN_OUTPUT_DIR}/adapter_model.safetensors"
require_file "${TRAIN_OUTPUT_DIR}/non_lora_trainables.bin"
require_file "${TRAIN_OUTPUT_DIR}/train_metrics.log"
if [ "$(grep -c "trainable params:" "${TRAIN_LOG}")" -ne 1 ]; then
    echo "ERROR expected exactly one rank0 trainable-params print" >&2
    exit 1
fi
if grep -Eq "\\[rank[1-9]\\]" "${TRAIN_LOG}"; then
    echo "ERROR non-rank0 traceback/log prefix appeared in training log" >&2
    exit 1
fi

echo "========== 2-GPU inference smoke =========="
rm -rf "${INFER_OUTPUT_DIR}"
mkdir -p "${INFER_OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr=127.0.0.1 \
    --master_port="$((MASTER_PORT + 1))" \
    scripts/infer_centerline_checkpoint.py \
    --checkpoint-dir "${INFER_CHECKPOINT_DIR}" \
    --vision_tower "${DINOV3_PATH}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    --deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES} \
    --test-json "${TEST_JSON}" \
    --num-samples "${NUM_INFER_SAMPLES}" \
    --image-folder "${IMAGE_FOLDER}" \
    --prompt-mode dataset \
    --conv-template "${CONV_TEMPLATE}" \
    --output-dir "${INFER_OUTPUT_DIR}" \
    --output-json "${INFER_OUTPUT_DIR}/summary.json" \
    --temperature 0.0 \
    --max-new-tokens 16 2>&1 | tee "${INFER_LOG}"

require_grep "DeepStack \\(real injection\\) enabled" "${INFER_LOG}" "DeepStack enabled inference log"
require_grep "Loaded [0-9]+/[0-9]+ multimodal checkpoint tensors after vision tower init" "${INFER_LOG}" "multimodal checkpoint reload"

DEBUG_INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR}" \
DEBUG_NPROC_PER_NODE="${NPROC_PER_NODE}" \
python - <<'PY'
import glob
import json
import os
from pathlib import Path

out_dir = Path(os.environ["DEBUG_INFER_OUTPUT_DIR"])
nproc = int(os.environ["DEBUG_NPROC_PER_NODE"])
summary_files = sorted(out_dir.glob("summary_rank*.json"))
sample_files = sorted(p for p in out_dir.glob("rank*_*.json") if not p.name.startswith("summary_"))
print(f"DEBUG_INFER summary_files {len(summary_files)}")
print(f"DEBUG_INFER sample_files {len(sample_files)}")
assert len(summary_files) == nproc, f"expected {nproc} summary_rank*.json files"
assert len(sample_files) >= nproc, "expected rank-prefixed per-sample output files"
for path in summary_files:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{path} must contain a list"
    for item in data:
        assert item["input_token_len"] > 0
        assert item["output_token_len"] > 0
        assert "<image>" in item["prompt"]
        assert item["conv_template"] == "conv_qwen_3_Dinov2_huawei"
print("DEBUG_CHECK inference_rank_outputs_ok")
print("DEBUG_CHECK prompt_image_token_ok")
PY

echo "========== debug passed =========="
echo "shape_log: ${SHAPE_LOG}"
echo "train_log: ${TRAIN_LOG}"
echo "infer_log: ${INFER_LOG}"
echo "train_output: ${TRAIN_OUTPUT_DIR}"
echo "infer_output: ${INFER_OUTPUT_DIR}"
