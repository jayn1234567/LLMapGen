#!/usr/bin/env bash

# ============================================================
# NPU SFT training
# Fixed recipe: phase_b | lane+intersection | native Qwen3-VL full architecture
# Stage-B starts from a Stage-A native Qwen3-VL checkpoint.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative imports.
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"                # Required cloud output root provided by ModelArts.

DATASET_PHASE=phase_b                                                             # Dataset stage: phase_b state update.
MAP_TASK=lane_intersection                                                        # Task type.
MODEL_RECIPE=qwen3vl_native                                                       # Native Qwen3-VL visual+LLM architecture.
TRAIN_VARIANT=full                                                                # Full native model SFT.

CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias used by existing scripts.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local cache root.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS model root.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_lane_intersection_samples_norm_33w_empty_patch.zip}  # Dataset zip.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_lane_intersection_samples_norm_33w_empty_patch}  # Extracted dataset directory.

QWEN3VL_MODEL_NAME=${QWEN3VL_MODEL_NAME:-Qwen3-VL-8B-Instruct}                    # Base model name, used only when ALLOW_BASE_FOR_STAGE_B=True.
QWEN3VL_OBS_PATH=${QWEN3VL_OBS_PATH:-${MODEL_OBS_PATH}/${QWEN3VL_MODEL_NAME}}     # Base model OBS path.
QWEN3VL_PATH=${QWEN3VL_PATH:-${OBS_CACHE}/checkpoints/${QWEN3VL_MODEL_NAME}}      # Local base model path.
STAGE_A_CHECKPOINT_OBS_PATH=${STAGE_A_CHECKPOINT_OBS_PATH:-}                      # OBS Stage-A checkpoint/output root to continue from.
STAGE_A_CHECKPOINT_DIR=${STAGE_A_CHECKPOINT_DIR:-}                                # Local Stage-A checkpoint/output root to continue from.
ALLOW_BASE_FOR_STAGE_B=${ALLOW_BASE_FOR_STAGE_B:-False}                           # True allows Stage-B to start from base Qwen3-VL for debugging only.
STAGE_A_DOWNLOAD_DIR=${STAGE_A_DOWNLOAD_DIR:-${OBS_CACHE}/stage_a_native_${RUN_ID}}  # Local Stage-A download root.

DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}          # Local dataset zip.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Dataset extract root.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}         # Extracted dataset root.
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}                                     # Image root.
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}                                   # Final cloud output path.
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}      # Local save root.
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}  # Local output dir.

TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}                         # Desired global batch size.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}                     # Per-NPU micro batch.
NUM_EPOCHS=${NUM_EPOCHS:-5}                                                       # Training epochs.
LR=${LR:-2e-5}                                                                    # Native model learning rate.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}                                                 # Weight decay.
WARMUP_RATIO=${WARMUP_RATIO:-0.03}                                                # Warmup ratio.
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}                                        # Max sequence length.
SAVE_STEPS=${SAVE_STEPS:-500}                                                     # Checkpoint interval.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}                                          # Regular checkpoint keep limit.
LOGGING_STEPS=${LOGGING_STEPS:-10}                                                # Logging interval.
EVAL_STEPS=${EVAL_STEPS:-500}                                                     # Eval interval.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}                # DeepSpeed config.
ENABLE_EVAL=${ENABLE_EVAL:-False}                                                 # Whether to run eval loss.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}                                # Save best train-loss checkpoints.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-5000}                    # Best train-loss start step.
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-5}                       # Best checkpoint keep limit.

SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}                                           # Enable SwanLab logging; native baseline defaults to disabled.
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}                # SwanLab API key.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v9}                                  # SwanLab project.
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${TRAIN_VARIANT}}  # SwanLab group.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${TRAIN_VARIANT}_from_stage_a}  # SwanLab experiment.
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,${DATASET_PHASE},${MAP_TASK},${MODEL_RECIPE},${TRAIN_VARIANT},from_stage_a,unimapgen_v9}  # SwanLab tags.
SWANLAB_MODE=${SWANLAB_MODE:-offline}                                             # SwanLab mode.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}                                            # Optional private API host.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}                                            # Optional private web host.

export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend toolkit root.
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend custom OPP root.
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend OPP path.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}                             # Gloo interface.
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}                                 # TP interface.
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}                             # HCCL interface.
export CUDA_DEVICE_MAX_CONNECTIONS=1                                              # Ascend compatibility flag.
export HCCL_WHITELIST_DISABLE=1                                                   # Disable HCCL whitelist.
export HCCL_CONNECT_TIMEOUT=7200                                                  # HCCL connect timeout.
export HCCL_EXEC_TIMEOUT=7200                                                     # HCCL exec timeout.
export HCCL_IF_BASE_PORT=64000                                                    # HCCL base port.
export INF_NAN_MODE_ENABLE=1                                                      # Inf/NaN handling.
export HCCL_ASYNC_ERROR_HANDLING=0                                                # Async error handling.
export WITHOUT_JIT_COMPILE=1                                                      # Disable JIT compile.
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE                                        # HCCL compatibility switch.
export COMBINED_ENABLE=1                                                          # Ascend combined op switch.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}                                      # CPU threads.
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}                              # Rank0 logs.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # Tokenizer warning control.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # Project imports.

INSTALL_DEPS=${INSTALL_DEPS:-True}                                                # Install dependencies.
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}                              # Upgrade moxing.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers>=5.7.0"}                     # Native Qwen3-VL-capable transformers.
TOKENIZERS_SPEC=${TOKENIZERS_SPEC:-"tokenizers>=0.22.0"}                          # Tokenizers version.
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

resolve_training_checkpoint() {
  python - "$1" <<'PYRESOLVE'
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")):
    print(root)
    raise SystemExit(0)
cmd = [sys.executable, "scripts/tools/resolve_best_checkpoint.py", "--output-dir", str(root), "--best-name", "infer_best", "--best-name", "eval_best", "--best-name", "best", "--allow-direct"]
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
if result.returncode == 0 and result.stdout.strip():
    print(result.stdout.strip())
    raise SystemExit(0)
checkpoints = []
for path in root.glob("checkpoint-*"):
    if path.is_dir():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        checkpoints.append((step, path))
if checkpoints:
    print(sorted(checkpoints)[-1][1])
    raise SystemExit(0)
raise SystemExit(f"cannot resolve a usable checkpoint under: {root}")
PYRESOLVE
}

if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; NPROC_PER_NODE=${NPROC_PER_NODE:-8}; MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}; NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}; NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}; MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi
MASTER_PORT=${MASTER_PORT:-6060}                                                  # Rendezvous port.
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-sft_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${RUN_ID}}  # Rendezvous id.

mkdir -p "${LOCAL_MODEL_SAVE_PATH}" "${DATASET_EXTRACT_ROOT}" "${STAGE_A_DOWNLOAD_DIR}"
OUTPUT_PATH="${LOCAL_MODEL_SAVE_PATH}"                                            # Trainer output dir.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab}                        # SwanLab local log dir.

python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

if [ -n "${STAGE_A_CHECKPOINT_DIR}" ]; then
  INIT_MODEL_PATH=$(resolve_training_checkpoint "${STAGE_A_CHECKPOINT_DIR}")       # Local Stage-A checkpoint selected for Stage-B.
elif [ -n "${STAGE_A_CHECKPOINT_OBS_PATH}" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${STAGE_A_CHECKPOINT_OBS_PATH}', '${STAGE_A_DOWNLOAD_DIR}')"
  INIT_MODEL_PATH=$(resolve_training_checkpoint "${STAGE_A_DOWNLOAD_DIR}")         # Downloaded Stage-A checkpoint selected for Stage-B.
elif [[ "${ALLOW_BASE_FOR_STAGE_B}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  if [ ! -e "${QWEN3VL_PATH}/config.json" ]; then
    python -c "import moxing as mox; mox.file.copy_parallel('${QWEN3VL_OBS_PATH}', '${QWEN3VL_PATH}')"
  fi
  INIT_MODEL_PATH="${QWEN3VL_PATH}"                                                # Debug-only fallback.
else
  echo "ERROR: set STAGE_A_CHECKPOINT_OBS_PATH or STAGE_A_CHECKPOINT_DIR for Stage-B native training."
  exit 1
fi

TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"                         # Training JSONL.
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"                           # Eval JSONL.
for path in "${INIT_MODEL_PATH}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then echo "ERROR: required path not found: ${path}"; exit 1; fi
done

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then GRADIENT_ACCUMULATION_STEPS=1; fi

EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")
EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(--eval_data_path "${EVAL_PATH}" --eval_image_folder "${IMAGE_FOLDER}" "${EVAL_STRATEGY_ARG}" steps --eval_steps "${EVAL_STEPS}")
fi

echo "============================================================"
echo "Recipe:       ${DATASET_PHASE} | ${MAP_TASK} | ${MODEL_RECIPE}"
echo "Init model:   ${INIT_MODEL_PATH}"
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
  --model_name_or_path "${INIT_MODEL_PATH}" \
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
