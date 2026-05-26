#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# Standalone NPU (Ascend) SFT Stage-B training script.
# Qwen3-VL-8B + DINOv2/DINOv3 + no DeepStack.
#
# This script does not dispatch through another shell launcher. It is intended
# for Stage B fine-tuning initialized from a Stage A checkpoint.
#
# Key parameters to edit:
#   VISION_BACKBONE=dinov2|dinov3
#   MAP_TASK=lane|lane_intersection
#   STAGE_A_CHECKPOINT_OBS_PATH=obs://.../infer_best_candidates/...
#     or
#   STAGE_A_CHECKPOINT_PATH=/cache/.../checkpoint-or-best-dir
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "Current working path: ${PWD}"

# ====================== user-editable experiment block ======================
VISION_BACKBONE=${VISION_BACKBONE:-dinov3}  # dinov2 or dinov3.
DATASET_PHASE=phase_b                       # Fixed Stage B.
MAP_TASK=${MAP_TASK:-lane}                  # lane or lane_intersection.

# Stage A initialization. Provide either an exact checkpoint/best directory or
# a training output root. If a root is provided, infer_best -> eval_best -> best
# is resolved automatically after download.
STAGE_A_CHECKPOINT_OBS_PATH=${STAGE_A_CHECKPOINT_OBS_PATH:-}
STAGE_A_CHECKPOINT_PATH=${STAGE_A_CHECKPOINT_PATH:-}
REQUIRE_STAGE_A_CHECKPOINT=${REQUIRE_STAGE_A_CHECKPOINT:-True}

# Main cloud paths.
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}

# Batch and training recipe.
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-2}  # Stage B usually starts from Stage A, so default is shorter.
LR=${LR:-1e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-1e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-500}
SAMPLE_SEED=${SAMPLE_SEED:-42}

# Best checkpoint policies.
ENABLE_EVAL=${ENABLE_EVAL:-True}
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-False}
BEST_EVAL_LOSS_DIR=${BEST_EVAL_LOSS_DIR:-eval_best}
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-True}
BEST_INFER_INDEX_DIR=${BEST_INFER_INDEX_DIR:-infer_best}
BEST_INFER_INDEX_METRIC=${BEST_INFER_INDEX_METRIC:-length_f1}
BEST_INFER_INDEX_NUM_SAMPLES=${BEST_INFER_INDEX_NUM_SAMPLES:-0}  # 0 means full eval.
BEST_INFER_INDEX_EVAL_STEPS=${BEST_INFER_INDEX_EVAL_STEPS:-${SAVE_STEPS}}
BEST_INFER_INDEX_MAX_NEW_TOKENS=${BEST_INFER_INDEX_MAX_NEW_TOKENS:-2048}
BEST_INFER_INDEX_DEVICE=${BEST_INFER_INDEX_DEVICE:-auto}
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-1}

# Train-loss best is usually less useful than infer_index for Stage B.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000}
BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best}

# SwanLab.
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_stage_b_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-sft}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_33w_stage_b_${MAP_TASK}_${VISION_BACKBONE}_from_stage_a_qwen3vl8b_nodeepstack}
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,33w,stage_b,${MAP_TASK},${VISION_BACKBONE},from_stage_a,qwen3vl8b,nodeepstack}
SWANLAB_MODE=${SWANLAB_MODE:-}
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}
export SWANLAB_API_KEY

# ====================== validate options ======================
case "${VISION_BACKBONE}" in
  dinov2|dinov3) ;;
  *) echo "ERROR: VISION_BACKBONE must be dinov2 or dinov3, got ${VISION_BACKBONE}"; exit 1 ;;
esac
case "${MAP_TASK}" in
  lane|lane_intersection) ;;
  *) echo "ERROR: MAP_TASK must be lane or lane_intersection, got ${MAP_TASK}"; exit 1 ;;
esac

# ====================== NPU environment ======================
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}

workerID=$(echo "$HOSTNAME" | awk -F'-' '{print $(NF-1)"-"$NF}')
echo "workerID=${workerID}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -d /usr/local/Ascend/ascend-toolkit ]; then
  sudo chmod -R 777 /usr/local/Ascend/ascend-toolkit/ || true
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ====================== moxing upgrade ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> changing moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
pip uninstall moxing-framework -y
pip cache purge
pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
export MOX_PROFILE=1
export MOX_RECORD_OBS=1
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> moxing change finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== dependencies ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing dependencies >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pip install torch==2.7.1
pip install torch_npu==2.7.1rc1
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

pip install "sentencepiece>=0.1.99"
pip install "tiktoken>=0.7.0"
pip install "transformers==4.56.2"
pip install "tokenizers>=0.22.0,<0.23.0"
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install "safetensors>=0.4.3"
pip install packaging
pip install "Pillow>=10.0.0"
pip install torchvision==0.22.1
pip install shortuuid
pip install "peft>=0.10.0"
pip install pydantic
pip install 'markdown2[all]'
pip install 'numpy>=1.26'
pip install 'scikit-learn>=1.2'
pip install 'gradio>=5.0'
pip install requests
pip install uvicorn
pip install fastapi
pip install 'einops>=0.6'
pip install 'einops-exts>=0.0.4'
pip install 'timm>=0.9.0'
pip install 'opencv-python-headless>=4.8.0'
pip install 'loguru>=0.7.0'
pip install 'shapely>=2.0.0'
pip install 'geopandas>=0.14.0'
pip install 'rasterio>=1.3.0'
pip install 'pyproj>=3.6.0'
pip install 'fiona>=1.9.0'
pip install wandb
pip install swanlab
pip install "huggingface-hub==0.36.2" --force-reinstall
pip install urllib3==1.26.15

echo "========== key deps =========="
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
echo "==============================="

# ====================== distributed parameters ======================
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-localhost}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi

MASTER_PORT=${MASTER_PORT:-6060}
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-stage_b_from_stage_a_${VISION_BACKBONE}_${MAP_TASK}}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== HCCL & NPU settings ======================
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

# ====================== output management ======================
CLUSTER_SAVE=${OUTPUT_URL:-}
if [ -z "${CLUSTER_SAVE}" ]; then
  CLUSTER_SAVE=/cache/unimapgen_v2/train_output
fi
OSB_SHARE_PATH="${CLUSTER_SAVE}"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}
LOCAL_MODEL_SAVE_PATH="${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}"
mkdir -p "${LOCAL_MODEL_SAVE_PATH}"

if [[ "${NODE_RANK}" == "0" ]]; then
  OUTPUT_PATH=${OUTPUT_PATH:-"${OSB_SHARE_PATH%/}/${RUN_ID}"}
else
  OUTPUT_PATH=${OUTPUT_PATH:-$LOCAL_MODEL_SAVE_PATH}
fi
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab}

echo "Run id: ${RUN_ID}"
echo "Output path: ${OUTPUT_PATH}"

# ====================== model and dataset paths ======================
Qwen3VL_PATH=${Qwen3VL_PATH:-${OBS_CACHE}/checkpoints/Qwen3-VL-8B-Instruct}
case "${VISION_BACKBONE}" in
  dinov2)
    VISION_OBS_NAME=facebook_dinov2-large
    VISION_TOWER_PATH=${DINOV2_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov2-large}
    INPUT_IMAGE_SIZE_ARGS=()
    BEST_INFER_INPUT_IMAGE_SIZE_ARGS=()
    ;;
  dinov3)
    VISION_OBS_NAME=facebook_dinov3-vitl16-pretrain-lvd1689m
    VISION_TOWER_PATH=${DINOV3_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    INPUT_IMAGE_SIZE_ARGS=(--input_image_size "${INPUT_IMAGE_SIZE}")
    BEST_INFER_INPUT_IMAGE_SIZE_ARGS=(--best_infer_index_input_image_size "${INPUT_IMAGE_SIZE}")
    ;;
esac

DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/data_line_samples_33w}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}

# ====================== download base model, vision tower, dataset ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading base model and vision tower >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_OBS_NAME}', '${VISION_TOWER_PATH}')"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/Qwen3-VL-8B-Instruct', '${Qwen3VL_PATH}')"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
if [ ! -d "${DATASET_PATH}" ]; then
  python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
  mkdir -p "${DATASET_EXTRACT_ROOT}"
  unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
fi

if [ ! -d "${DATASET_PATH}" ]; then
  echo "ERROR: Expected dataset directory not found: ${DATASET_PATH}"
  ls -l "${DATASET_EXTRACT_ROOT}" || true
  exit 1
fi

# ====================== download and resolve Stage A checkpoint ======================
if [ -n "${STAGE_A_CHECKPOINT_OBS_PATH}" ]; then
  STAGE_A_CHECKPOINT_PATH=${STAGE_A_CHECKPOINT_PATH:-${OBS_CACHE}/stage_a_checkpoint_${RUN_ID}}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading Stage A checkpoint >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "STAGE_A_CHECKPOINT_OBS_PATH=${STAGE_A_CHECKPOINT_OBS_PATH}"
  echo "STAGE_A_CHECKPOINT_PATH=${STAGE_A_CHECKPOINT_PATH}"
  python -c "import moxing as mox; mox.file.copy_parallel('${STAGE_A_CHECKPOINT_OBS_PATH}', '${STAGE_A_CHECKPOINT_PATH}')"
fi

if [ -n "${STAGE_A_CHECKPOINT_PATH}" ] && [ -d "${STAGE_A_CHECKPOINT_PATH}" ]; then
  if RESOLVED_STAGE_A=$(python scripts/tools/resolve_best_checkpoint.py \
      --output-dir "${STAGE_A_CHECKPOINT_PATH}" \
      --best-name infer_best \
      --best-name eval_best \
      --best-name best \
      --allow-direct 2>/dev/null); then
    INIT_MODEL_PATH="${RESOLVED_STAGE_A}"
  else
    INIT_MODEL_PATH="${STAGE_A_CHECKPOINT_PATH}"
  fi
else
  if [[ "${REQUIRE_STAGE_A_CHECKPOINT}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    echo "ERROR: Stage B requires Stage A checkpoint. Set STAGE_A_CHECKPOINT_OBS_PATH or STAGE_A_CHECKPOINT_PATH."
    exit 1
  fi
  INIT_MODEL_PATH="${Qwen3VL_PATH}"
fi

# ====================== dataset paths ======================
if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
  TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
  EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
  TEST_PATH="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
else
  TRAIN_PATH="${DATASET_PATH}/train.jsonl"
  EVAL_PATH="${DATASET_PATH}/eval.jsonl"
  TEST_PATH="${DATASET_PATH}/test.jsonl"
fi
EVAL_IMAGE_FOLDER="${IMAGE_FOLDER}"

for path in "${TRAIN_PATH}" "${EVAL_PATH}" "${TEST_PATH}" "${IMAGE_FOLDER}" "${VISION_TOWER_PATH}" "${INIT_MODEL_PATH}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done

# ====================== batch size ======================
total_gpus=$(( NNODES * NPROC_PER_NODE ))
micro_batch=$(( total_gpus * PER_DEVICE_TRAIN_BATCH_SIZE ))
gradient_accumulation_steps=$(( (TARGET_GLOBAL_BATCH_SIZE + micro_batch - 1) / micro_batch ))
if [ "${gradient_accumulation_steps}" -lt 1 ]; then
  gradient_accumulation_steps=1
fi
ACTUAL_GLOBAL_BATCH_SIZE=$(( micro_batch * gradient_accumulation_steps ))

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ====================== training parameters ======================
MM_VISION_SELECT_LAYER=${MM_VISION_SELECT_LAYER:--2}
MM_PROJECTOR_TYPE=${MM_PROJECTOR_TYPE:-mlp2x_gelu}
UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
USE_HF_PROGRESS_BAR=${USE_HF_PROGRESS_BAR:-True}
EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")

EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(
    --eval_data_path "${EVAL_PATH}"
    --eval_image_folder "${EVAL_IMAGE_FOLDER}"
    "${EVAL_STRATEGY_ARG}" steps
    --eval_steps "${EVAL_STEPS}"
    --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}"
    --best_eval_loss_dir "${BEST_EVAL_LOSS_DIR}"
  )
fi

echo "============================================================"
echo "Stage:      ${DATASET_PHASE}"
echo "Task:       ${MAP_TASK}"
echo "Backbone:   ${VISION_BACKBONE}"
echo "Init model: ${INIT_MODEL_PATH}"
echo "Base Qwen:  ${Qwen3VL_PATH}"
echo "ViT:        ${VISION_TOWER_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "Train:      ${TRAIN_PATH}"
echo "Eval:       ${EVAL_PATH}"
echo "Output:     ${OUTPUT_PATH}"
echo "Batch:      target=${TARGET_GLOBAL_BATCH_SIZE}, per_device=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_acc=${gradient_accumulation_steps}, actual=${ACTUAL_GLOBAL_BATCH_SIZE}"
echo "LR:         llm=${LR}, projector=${MM_PROJECTOR_LR}, vision=${MM_VISION_TOWER_LR}"
echo "Infer best: ${SAVE_BEST_INFER_INDEX}, metric=${BEST_INFER_INDEX_METRIC}, samples=${BEST_INFER_INDEX_NUM_SAMPLES}, eval_steps=${BEST_INFER_INDEX_EVAL_STEPS}"
echo "============================================================"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  --model_name_or_path "${INIT_MODEL_PATH}" \
  --version conv_qwen_3_Dinov2_huawei \
  --vision_tower "${VISION_TOWER_PATH}" \
  "${INPUT_IMAGE_SIZE_ARGS[@]}" \
  --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
  --mm_projector_type "${MM_PROJECTOR_TYPE}" \
  --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
  --disable_deepstack True \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --sample_seed "${SAMPLE_SEED}" \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --learning_rate "${LR}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers 4 \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
  --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
  --best_train_loss_dir "${BEST_TRAIN_LOSS_DIR}" \
  --save_best_infer_index "${SAVE_BEST_INFER_INDEX}" \
  --best_infer_index_dir "${BEST_INFER_INDEX_DIR}" \
  --best_infer_index_metric "${BEST_INFER_INDEX_METRIC}" \
  --best_infer_index_phase "${DATASET_PHASE}" \
  --best_infer_index_eval_data_path "${EVAL_PATH}" \
  --best_infer_index_image_folder "${EVAL_IMAGE_FOLDER}" \
  --best_infer_index_vision_tower "${VISION_TOWER_PATH}" \
  "${BEST_INFER_INPUT_IMAGE_SIZE_ARGS[@]}" \
  --best_infer_index_conv_template conv_qwen_3_Dinov2_huawei \
  --best_infer_index_map_task "${MAP_TASK}" \
  --best_infer_index_num_samples "${BEST_INFER_INDEX_NUM_SAMPLES}" \
  --best_infer_index_eval_steps "${BEST_INFER_INDEX_EVAL_STEPS}" \
  --best_infer_index_max_new_tokens "${BEST_INFER_INDEX_MAX_NEW_TOKENS}" \
  --best_infer_index_device "${BEST_INFER_INDEX_DEVICE}" \
  --best_infer_index_work_dir "${OUTPUT_PATH}/infer_index_eval" \
  --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
  --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
  --use_hf_progress_bar "${USE_HF_PROGRESS_BAR}" \
  --logging_steps "${LOGGING_STEPS}" \
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

echo "=== Stage B training finished ==="
