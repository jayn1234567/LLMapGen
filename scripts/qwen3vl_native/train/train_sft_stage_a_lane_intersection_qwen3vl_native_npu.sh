#!/usr/bin/env bash

# ============================================================
# NPU SFT training
# Fixed recipe: phase_a | lane+intersection | native Qwen3-VL full architecture
# This launcher does not use the project DINO/DeepStack vision path.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative imports.
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"                # Required cloud output root provided by ModelArts.

DATASET_PHASE=phase_a                                                             # Dataset stage: phase_a patch construction.
MAP_TASK=lane_intersection                                                        # Task type: lane or lane_intersection.
MODEL_RECIPE=qwen3vl_native                                                       # Native Qwen3-VL visual+LLM architecture.
TRAIN_VARIANT=full                                                                # Full native model SFT.

CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root injected by the platform.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias used by existing project scripts.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id for local and OBS outputs.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local worker cache root.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS root for model assets.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_lane_intersection_samples_norm_33w_empty_patch.zip}  # Prepared dataset zip.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_lane_intersection_samples_norm_33w_empty_patch}  # Directory expected after unzip.

QWEN3VL_MODEL_NAME=${QWEN3VL_MODEL_NAME:-Qwen3-VL-8B-Instruct}                    # Native Qwen3-VL checkpoint directory name under MODEL_OBS_PATH.
QWEN3VL_OBS_PATH=${QWEN3VL_OBS_PATH:-${MODEL_OBS_PATH}/${QWEN3VL_MODEL_NAME}}     # OBS path for native Qwen3-VL base checkpoint.
QWEN3VL_PATH=${QWEN3VL_PATH:-${OBS_CACHE}/checkpoints/${QWEN3VL_MODEL_NAME}}      # Local native Qwen3-VL checkpoint path.
REPLACE_PATCH_EMBED_CONV3D_WITH_LINEAR=${REPLACE_PATCH_EMBED_CONV3D_WITH_LINEAR:-True}  # Run native Qwen3-VL patch_embed Conv3d through an equivalent Linear path to avoid NPU Conv3D backward format errors.
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}          # Local dataset zip path.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Local extraction root.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}         # Extracted dataset root.
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}                                     # Image root passed to native train.
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}                                   # Final OBS/cloud output path.
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}      # Local save root.
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}  # Per-run local output dir.

TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}                         # Desired global batch size.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}                     # Per-NPU micro batch size.
NUM_EPOCHS=${NUM_EPOCHS:-5}                                                       # Training epochs.
LR=${LR:-2e-5}                                                                    # Native Qwen3-VL full-model learning rate.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}                                                 # Weight decay.
WARMUP_RATIO=${WARMUP_RATIO:-0.03}                                                # Warmup ratio.
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}                                        # Max text sequence length.
SAVE_STEPS=${SAVE_STEPS:-500}                                                     # Regular checkpoint interval.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}                                          # Regular checkpoint keep limit.
LOGGING_STEPS=${LOGGING_STEPS:-10}                                                # Logging interval.
EVAL_STEPS=${EVAL_STEPS:-500}                                                     # Eval interval if ENABLE_EVAL=True.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}                # DeepSpeed config for full model SFT.
ENABLE_EVAL=${ENABLE_EVAL:-False}                                                 # Whether to run eval loss.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}                                # Save best train-loss native checkpoints.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-5000}                    # Best train-loss starts after this step.
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-5}                       # Best checkpoint keep limit.

SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}                                           # Enable SwanLab logging; native baseline defaults to disabled.
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}                # SwanLab API key; override on platform if needed.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v9}                                  # SwanLab project.
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${TRAIN_VARIANT}}  # SwanLab group.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${TRAIN_VARIANT}}  # SwanLab experiment.
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,${DATASET_PHASE},${MAP_TASK},${MODEL_RECIPE},${TRAIN_VARIANT},unimapgen_v9}  # SwanLab tags.
SWANLAB_MODE=${SWANLAB_MODE:-offline}                                             # SwanLab mode.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}                                            # Optional private API host.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}                                            # Optional private web host.

export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend toolkit root.
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend custom OPP root.
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend OPP path.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}                             # Gloo network interface.
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}                                 # Tensor-parallel service interface.
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}                             # HCCL network interface.
export CUDA_DEVICE_MAX_CONNECTIONS=1                                              # Ascend compatibility flag.
export HCCL_WHITELIST_DISABLE=1                                                   # Disable HCCL whitelist.
export HCCL_CONNECT_TIMEOUT=7200                                                  # HCCL connect timeout.
export HCCL_EXEC_TIMEOUT=7200                                                     # HCCL exec timeout.
export HCCL_IF_BASE_PORT=64000                                                    # HCCL base port.
export INF_NAN_MODE_ENABLE=1                                                      # Inf/NaN handling.
export HCCL_ASYNC_ERROR_HANDLING=0                                                # Async error handling switch.
export WITHOUT_JIT_COMPILE=1                                                      # Disable JIT compile path.
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE                                        # HCCL compatibility switch.
export COMBINED_ENABLE=1                                                          # Ascend combined op switch.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}                                      # CPU threads per process.
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}                              # Rank0-only project logs.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # Disable tokenizer parallel warnings.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # Project imports.

INSTALL_DEPS=${INSTALL_DEPS:-True}                                                # Install dependencies on managed NPU images.
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}                              # Upgrade moxing wheel.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers>=5.7.0"}                     # Native Qwen3-VL-capable transformers build.
TOKENIZERS_SPEC=${TOKENIZERS_SPEC:-"tokenizers>=0.22.0"}                          # Tokenizers version aligned with transformers.
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
  pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "${TRANSFORMERS_SPEC}" "${TOKENIZERS_SPEC}" "qwen-vl-utils>=0.0.10"
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless>=4.8.0'
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15
fi

if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}                                                             # Distributed node count.
  NODE_RANK=${NODE_RANK:-0}                                                       # Rank of this node.
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}                                             # NPU processes per node.
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}                                           # Rendezvous master.
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}                                                 # Distributed node count.
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}                                          # Rank of this node.
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}                                  # NPU processes per node.
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}                              # Rendezvous master.
fi
MASTER_PORT=${MASTER_PORT:-6060}                                                  # Rendezvous port.
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-sft_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${RUN_ID}}  # Rendezvous id.

mkdir -p "${LOCAL_MODEL_SAVE_PATH}" "${DATASET_EXTRACT_ROOT}"
OUTPUT_PATH="${LOCAL_MODEL_SAVE_PATH}"                                            # Trainer output dir.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab}                        # SwanLab local log dir.

if [ ! -e "${QWEN3VL_PATH}/config.json" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${QWEN3VL_OBS_PATH}', '${QWEN3VL_PATH}')"
fi
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"                         # Training JSONL.
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"                           # Eval JSONL.
for path in "${QWEN3VL_PATH}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then echo "ERROR: required path not found: ${path}"; exit 1; fi
done

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))                                      # Total NPU workers.
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))                    # Global micro batch.
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))  # Derived accumulation.
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then GRADIENT_ACCUMULATION_STEPS=1; fi

EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")  # Compatible eval arg name.
EVAL_ARGS=()                                                                      # Optional eval args.
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(--eval_data_path "${EVAL_PATH}" --eval_image_folder "${IMAGE_FOLDER}" "${EVAL_STRATEGY_ARG}" steps --eval_steps "${EVAL_STEPS}")
fi

echo "============================================================"
echo "Recipe:       ${DATASET_PHASE} | ${MAP_TASK} | ${MODEL_RECIPE}"
echo "Init model:   ${QWEN3VL_PATH}"
echo "Patch embed:  linearized_conv3d=${REPLACE_PATCH_EMBED_CONV3D_WITH_LINEAR}"
echo "Train:        ${TRAIN_PATH}"
echo "Eval:         ${EVAL_PATH}"
echo "Output:       ${OUTPUT_PATH}"
echo "Cloud output: ${CLOUD_OUTPUT_PATH}"
echo "============================================================"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.native_qwen3vl.train_sft \
  --model_name_or_path "${QWEN3VL_PATH}" \
  --replace_patch_embed_conv3d_with_linear "${REPLACE_PATCH_EMBED_CONV3D_WITH_LINEAR}" \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
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

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then echo "Training failed with exit code ${TRAIN_EXIT}"; exit "${TRAIN_EXIT}"; fi

if [[ "${NODE_RANK}" == "0" ]]; then
  if [ -e "${CLOUD_OUTPUT_PATH}" ]; then echo "ERROR: cloud output path already exists: ${CLOUD_OUTPUT_PATH}"; exit 1; fi
  echo "Moving rank0 local output to cloud output: ${OUTPUT_PATH} -> ${CLOUD_OUTPUT_PATH}"
  mv "${OUTPUT_PATH}" "${CLOUD_OUTPUT_PATH}"
  echo "Final cloud output path: ${CLOUD_OUTPUT_PATH}"
else
  echo "Non-master node ${NODE_RANK}: skip cloud output move."
fi
