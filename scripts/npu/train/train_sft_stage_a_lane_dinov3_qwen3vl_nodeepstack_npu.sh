#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU SFT training
# Fixed recipe: phase_a | lane-only centerline | dinov3 + Qwen3-VL-8B | no DeepStack
# This file is self-contained and does not call another project .sh file.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

DATASET_PHASE=phase_a
MAP_TASK=lane
VISION_BACKBONE=dinov3
VISION_TOWER_NAME=facebook_dinov3-vitl16-pretrain-lvd1689m
MM_VISION_TOWER_TYPE=dinov3
INPUT_IMAGE_SIZE=512

echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "Recipe: ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
# ====================== cloud paths ======================
# OUTPUT_URL is injected by the cloud training platform.
# Keep the reference-script convention: mirror it into OSB_SHARE_PATH,
# then place cloud outputs under OSB_SHARE_PATH/RUN_ID.
CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="${CLUSTER_SAVE}"
echo "System defined obs share path: ${OSB_SHARE_PATH}"

# Cloud training mounts can be create-only. Rank0 writes to a fresh cloud run dir;
# nonzero ranks write to local cache to avoid cross-rank overwrite/rename issues.
# Unique run id. Override it when all nodes must share a fixed output folder.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
# Local cache root on the NPU worker. Models, dataset zip, and temp outputs are stored here.
OBS_CACHE=${OBS_CACHE:-/cache}
# OBS directory that contains Qwen3-VL and DINO checkpoints.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
# OBS zip path of the prepared dataset. The zip should contain phase_a/phase_b jsonl and images.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}
# Directory name expected after unzipping DATASET_OBS_PATH.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}

# Local DINO vision tower path after downloading from MODEL_OBS_PATH.
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}
# Local path for the downloaded dataset zip.
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
# Local root used to unzip the dataset.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
# Final local dataset directory. Override only if the dataset is already extracted.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/data_line_samples_33w}
# Image root passed to training/inference. Usually the same as DATASET_PATH.
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
# Cloud output directory for rank0 SFT checkpoints.
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}
# Local checkpoint root for nonzero ranks and temporary model saves.
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}
# Local run-specific checkpoint directory. This is created with mkdir -p.
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}
# Local Qwen3-VL base model path used by Stage-A SFT.
QWEN_PATH=${QWEN_PATH:-${OBS_CACHE}/checkpoints/Qwen3-VL-8B-Instruct}

# ====================== training params ======================
# Main knobs for this SFT recipe. Edit values here instead of passing one-off shell prefixes.
# TARGET_GLOBAL_BATCH_SIZE is the effective batch across all nodes and NPUs.
# SAVE_STEPS/SAVE_TOTAL_LIMIT control regular checkpoint-* retention.
# BEST_* options control train-loss, eval-loss, or infer-index best checkpoint folders.
# Target effective global batch size across all nodes, NPUs, and grad accumulation.
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
# Micro batch size on each NPU process.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
# Number of SFT epochs. Stage-A usually uses more epochs than Stage-B.
NUM_EPOCHS=${NUM_EPOCHS:-3}
# Base learning rate for the LLM and default trainable parameters.
LR=${LR:-2e-5}
# Learning rate for the multimodal projector.
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
# Learning rate for the DINO vision tower; keep lower than LLM LR for full-param training.
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}
# Weight decay passed to Trainer. Current recipe keeps it disabled.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
# Warmup ratio for the LR scheduler.
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
# Max text sequence length including prompt, image token, and generated coordinate text.
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
# Save a regular checkpoint-* every this many optimizer steps.
SAVE_STEPS=${SAVE_STEPS:-500}
# Keep this many latest regular checkpoint-* directories.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
# Console/SwanLab logging interval in optimizer steps.
LOGGING_STEPS=${LOGGING_STEPS:-10}
# Eval-loss interval in optimizer steps when ENABLE_EVAL=True.
EVAL_STEPS=${EVAL_STEPS:-500}
# DeepSpeed config file. Current NPU SFT recipe uses ZeRO3.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
# Whether to run eval loss during SFT training.
ENABLE_EVAL=${ENABLE_EVAL:-True}
# Whether to save the best eval-loss checkpoint under eval_best.
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-True}
# Whether to save the best train-loss checkpoint under best.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
# Ignore train-loss best saving before this step.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000}
# Whether to run inference-index evaluation during training and save infer_best.
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}
# Metric name used to select infer_best, for example length_f1.
BEST_INFER_INDEX_METRIC=${BEST_INFER_INDEX_METRIC:-length_f1}
# Number of eval samples for infer-index best. 0 means full eval set.
BEST_INFER_INDEX_NUM_SAMPLES=${BEST_INFER_INDEX_NUM_SAMPLES:-0}
# Best checkpoint save mode. rotating_create_only avoids overwrite/rename on create-only filesystems.
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}
# How many best checkpoint candidates to keep in each best folder.
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-1}

# Enable SwanLab logging for this run.
SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}
# SwanLab API key. Override from the platform env if needed.
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
# SwanLab project name.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
# SwanLab group name for related experiments.
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_phase_a_lane_dinov3_nodeepstack}
# SwanLab experiment display name.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_phase_a_lane_dinov3_qwen3vl8b_nodeepstack}
# SwanLab comma-separated tags.
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,phase_a,lane,dinov3,qwen3vl8b,nodeepstack}
# SwanLab mode. Use offline if the cloud cannot reach SwanLab.
SWANLAB_MODE=${SWANLAB_MODE:-}
# SwanLab private deployment API host, if used.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}
# SwanLab private deployment web host, if used.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}
# ====================== Ascend environment ======================
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Whether this script installs Python dependencies before running.
INSTALL_DEPS=${INSTALL_DEPS:-True}
# Whether to replace the platform moxing package with the required wheel.
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}
# vLLM version used by GRPO rollout scripts.
VLLM_VERSION=${VLLM_VERSION:-0.9.2}
# vLLM-Ascend version used by GRPO rollout scripts.
VLLM_ASCEND_VERSION=${VLLM_ASCEND_VERSION:-0.9.2rc1}

if [[ "${ENABLE_MOXING_UPGRADE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
  pip uninstall moxing-framework -y
  pip cache purge
  pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
  export MOX_PROFILE=1
  export MOX_RECORD_OBS=1
fi

if [[ "${INSTALL_DEPS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  pip install torch==2.7.1 torch_npu==2.7.1rc1
  python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
  pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl
  pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "transformers==4.56.2" "tokenizers>=0.22.0,<0.23.0"
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless>=4.8.0'
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15

fi
# ====================== distributed setup ======================
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
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-sft_phase_a_lane_dinov3_${RUN_ID}}
mkdir -p "${LOCAL_MODEL_SAVE_PATH}"
if [[ "${NODE_RANK}" == "0" ]]; then
  OUTPUT_PATH="${CLOUD_OUTPUT_PATH}"
else
  OUTPUT_PATH="${LOCAL_MODEL_SAVE_PATH}"
fi
echo "Run id: ${RUN_ID}"
echo "Output path: ${OUTPUT_PATH}"
# Local SwanLab log directory.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab}

# ====================== downloads ======================
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/Qwen3-VL-8B-Instruct', '${QWEN_PATH}')"
INIT_MODEL_PATH="${QWEN_PATH}"
TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
TEST_PATH="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
for path in "${INIT_MODEL_PATH}" "${VISION_TOWER}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
  GRADIENT_ACCUMULATION_STEPS=1
fi

EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")
EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(
    --eval_data_path "${EVAL_PATH}"
    --eval_image_folder "${IMAGE_FOLDER}"
    "${EVAL_STRATEGY_ARG}" steps
    --eval_steps "${EVAL_STEPS}"
    --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}"
    --best_eval_loss_dir eval_best
  )
fi

echo "============================================================"
echo "Recipe:       ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
echo "Init model:   ${INIT_MODEL_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Train:        ${TRAIN_PATH}"
echo "Eval:         ${EVAL_PATH}"
echo "Output:       ${OUTPUT_PATH}"
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
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
  --input_image_size "${INPUT_IMAGE_SIZE}" \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower True \
  --disable_deepstack True \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --sample_seed 42 \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
  --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
  --best_train_loss_dir best \
  --save_best_infer_index "${SAVE_BEST_INFER_INDEX}" \
  --best_infer_index_dir infer_best \
  --best_infer_index_metric "${BEST_INFER_INDEX_METRIC}" \
  --best_infer_index_phase "${DATASET_PHASE}" \
  --best_infer_index_eval_data_path "${EVAL_PATH}" \
  --best_infer_index_image_folder "${IMAGE_FOLDER}" \
  --best_infer_index_vision_tower "${VISION_TOWER}" \
  --best_infer_index_input_image_size "${INPUT_IMAGE_SIZE}" \
  --best_infer_index_conv_template conv_qwen_3_Dinov2_huawei \
  --best_infer_index_map_task "${MAP_TASK}" \
  --best_infer_index_num_samples "${BEST_INFER_INDEX_NUM_SAMPLES}" \
  --best_infer_index_eval_steps "${SAVE_STEPS}" \
  --best_infer_index_max_new_tokens 2048 \
  --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
  --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
  --use_hf_progress_bar True \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to none \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type sft \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --swanlab_log_dir "${SWANLAB_LOG_DIR}" \
  --swanlab_api_host "${SWANLAB_API_HOST}" \
  --swanlab_web_host "${SWANLAB_WEB_HOST}" \
  --ddp_find_unused_parameters False \
  --ddp_backend hccl \
  --deepspeed "${DEEPSPEED_CONFIG}"
