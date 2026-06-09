#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NPU SFT training
# Fixed recipe: phase_b | lane+intersection | DINOv3 no-DeepStack with LLM LoRA | Qwen3.5 text LLM
# This file is self-contained and only downloads the model assets required by this recipe.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative script and Python imports.
# Platform I/O and recipe metadata are declared first so cloud jobs can be audited quickly.
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"                # Required cloud output root provided by the platform.

# Recipe identity: fixed task, visual architecture, model family, and train variant.
DATASET_PHASE=phase_b                                                             # Dataset stage: phase_a for patch inference, phase_b for state update.
MAP_TASK=lane_intersection                                                        # Task type: lane or lane_intersection.
VISION_RECIPE=dinov3                                                              # Fixed visual architecture recipe encoded by this script name.
MODEL_FAMILY=qwen3_5                                                              # LLM family selector: qwen3, qwen3_5, or qwen3vl.
MODEL_LABEL=qwen3_5                                                               # Short model label used in run names, logs, and output paths.
TRAIN_VARIANT=lora_llm                                                            # Training variant: full parameters or LLM LoRA.

case "${MAP_TASK}" in lane|lane_intersection) ;; *) echo "ERROR: MAP_TASK must be lane or lane_intersection"; exit 1 ;; esac
case "${MODEL_FAMILY}" in qwen3|qwen3_5) ;; *) echo "ERROR: MODEL_FAMILY must be qwen3 or qwen3_5"; exit 1 ;; esac
case "${TRAIN_VARIANT}" in full|lora_llm) ;; *) echo "ERROR: TRAIN_VARIANT must be full or lora_llm"; exit 1 ;; esac

# Cloud and local storage roots. Outputs are staged locally before OBS upload.
CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root injected by the training platform.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias of the platform output root used by existing scripts.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id for local cache and cloud output folders.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local worker cache root for models, datasets, checkpoints, and outputs.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS directory that stores model and vision checkpoint assets.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}  # OBS zip path for the prepared UniMapGen dataset.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}                       # Dataset directory name expected after the zip is extracted.

# Vision assets for this recipe. Scripts only download the towers declared below.
DINO_V3_TOWER_NAME=${DINO_V3_TOWER_NAME:-facebook_dinov3-vitl16-pretrain-lvd1689m}  # DINOv3 checkpoint directory name under MODEL_OBS_PATH.
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${DINO_V3_TOWER_NAME}}      # Vision tower path passed to the model loader. Multi-vision uses a comma list.
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-${MODEL_OBS_PATH}/${DINO_V3_TOWER_NAME}}  # OBS path for the single vision tower used by this recipe.
MM_VISION_TOWER_TYPE=dinov3                                                       # Model-side vision tower type: dinov2, dinov3, multi_moe, or multi_concat.
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}                                         # Image size fed to the vision encoder; DINOv3 recipes usually use 512.
REQUIRED_VISION_TOWERS=("${VISION_TOWER}")                                        # Local vision tower paths that must exist before launch.


# Local dataset and output paths for this run.
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}          # Local path for the downloaded dataset zip.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Local directory where the dataset zip is extracted.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}         # Extracted dataset root containing phase_a and phase_b folders.
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}                                     # Image root passed to training or inference. Usually DATASET_PATH.
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}                                   # Final cloud output directory for training artifacts.
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}      # Local root for training outputs before cloud upload.
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}  # Per-run local training output directory.
# Continuation checkpoint inputs. OBS paths are downloaded, local paths are used directly.
STAGE_A_CHECKPOINT_OBS_PATH=${STAGE_A_CHECKPOINT_OBS_PATH:-}                      # Stage-B training: OBS root of the Stage-A checkpoint to continue from.
STAGE_A_CHECKPOINT_PATH=${STAGE_A_CHECKPOINT_PATH:-}                              # Stage-B training: local Stage-A checkpoint root.
STAGE_A_DOWNLOAD_DIR=${STAGE_A_DOWNLOAD_DIR:-${OBS_CACHE}/stage_a_checkpoint_${RUN_ID}}  # Local download directory for the Stage-A checkpoint.

# Visual fusion controls. Empty layer lists disable the optional path.
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}                                      # True disables DeepStack residual injection; False allows it when indexes are set.
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}                            # ViT layers for DeepStack residual injection, for example 6 12 18 23.
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}                      # ViT layers fused into the main visual stream; empty disables direct fusion.
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}                        # Fusion mode: mean, sum, learned_weighted; aliases: weighted, softmax_weighted.

# Main runtime parameters and hyperparameters.
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}                         # Desired global batch size used to derive gradient accumulation.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}                     # Micro batch size per NPU process.
NUM_EPOCHS=${NUM_EPOCHS:-5}                                                       # Number of SFT training epochs.
LR=${LR:-2e-5}                                                                    # Base learning rate for LLM and default trainable parameters.
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}                                          # Learning rate for the multimodal projector.
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}                                    # Learning rate for trainable vision tower parameters.
MM_VISION_FUSION_LR=${MM_VISION_FUSION_LR:-${MM_PROJECTOR_LR}}                    # Learning rate for multi-vision adapters, router, or concat projector.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}                                                 # Weight decay used by the trainer.
WARMUP_RATIO=${WARMUP_RATIO:-0.03}                                                # Warmup ratio for the cosine learning-rate schedule.
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}                                        # Maximum text sequence length.
SAVE_STEPS=${SAVE_STEPS:-500}                                                     # Checkpoint save interval in optimizer steps.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}                                          # Maximum number of regular checkpoints to keep.
LOGGING_STEPS=${LOGGING_STEPS:-10}                                                # Training log interval in optimizer steps.
EVAL_STEPS=${EVAL_STEPS:-500}                                                     # Evaluation interval when ENABLE_EVAL is true.

ENABLE_EVAL=${ENABLE_EVAL:-False}                                                 # Whether to run Trainer evaluation during training.
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-False}                                 # Whether to save a best checkpoint by eval loss.
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}                                # Whether to save a best checkpoint by training loss.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-5000}                    # Step threshold before best-train-loss checkpointing starts.
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}                             # Whether to run inference-based best checkpoint selection.
BEST_INFER_INDEX_METRIC=${BEST_INFER_INDEX_METRIC:-length_f1}                     # Metric used for inference-based best checkpoint selection.
BEST_INFER_INDEX_NUM_SAMPLES=${BEST_INFER_INDEX_NUM_SAMPLES:-0}                   # Number of eval samples for best-infer selection; 0 uses all.
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}      # Save policy for best checkpoint directories.
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-5}                       # Number of best checkpoint candidates to keep.

# LoRA switch: LoRA scripts train adapters on LLM modules; full scripts keep LoRA disabled.
if [ "${TRAIN_VARIANT}" = "lora_llm" ]; then
  LORA_ENABLE=${LORA_ENABLE:-True}                                                # Enable LoRA adapters for this run.
  LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}                                     # LoRA target scope, usually llm.
  LORA_R=${LORA_R:-8}                                                             # LoRA rank.
  LORA_ALPHA=${LORA_ALPHA:-16}                                                    # LoRA alpha scaling value.
  LORA_DROPOUT=${LORA_DROPOUT:-0.05}                                              # LoRA dropout probability.
  LORA_BIAS=${LORA_BIAS:-none}                                                    # LoRA bias handling mode.
  UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}                      # Whether the vision tower is trainable.
else
  LORA_ENABLE=${LORA_ENABLE:-False}                                               # Enable LoRA adapters for this run.
  LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}                                     # LoRA target scope, usually llm.
  LORA_R=${LORA_R:-8}                                                             # LoRA rank.
  LORA_ALPHA=${LORA_ALPHA:-16}                                                    # LoRA alpha scaling value.
  LORA_DROPOUT=${LORA_DROPOUT:-0.05}                                              # LoRA dropout probability.
  LORA_BIAS=${LORA_BIAS:-none}                                                    # LoRA bias handling mode.
  UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}                      # Whether the vision tower is trainable.
fi

# Experiment logging metadata.
SWANLAB_ENABLE=${SWANLAB_ENABLE:-True}                                            # Enable SwanLab experiment logging.
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}                # SwanLab API key. Override from platform env when possible.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}                                  # SwanLab project name.
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}}  # SwanLab group name for related runs.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}}  # SwanLab experiment display name.
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,${DATASET_PHASE},${MAP_TASK},${VISION_RECIPE},${MODEL_LABEL},${TRAIN_VARIANT},unimapgen_v9}  # Comma-separated SwanLab tags for filtering runs.
SWANLAB_MODE=${SWANLAB_MODE:-offline}                                             # SwanLab mode, for example offline on restricted cloud networks.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}                                            # Optional SwanLab private API host.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}                                            # Optional SwanLab private web host.

# Ascend and HCCL runtime environment for NPU jobs.
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend toolkit root.
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend custom operator package root.
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend operator package path.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}                             # Network interface used by Gloo rendezvous.
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}                                 # Network interface used by tensor-parallel services.
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}                             # Network interface used by HCCL communication.
export CUDA_DEVICE_MAX_CONNECTIONS=1                                              # NPU compatibility setting used by Ascend PyTorch jobs.
export HCCL_WHITELIST_DISABLE=1                                                   # Disable HCCL whitelist checks on managed clusters.
export HCCL_CONNECT_TIMEOUT=7200                                                  # HCCL connection timeout in seconds.
export HCCL_EXEC_TIMEOUT=7200                                                     # HCCL execution timeout in seconds.
export HCCL_IF_BASE_PORT=64000                                                    # Base port for HCCL communication.
export INF_NAN_MODE_ENABLE=1                                                      # Enable Inf/NaN handling in Ascend runtime.
export HCCL_ASYNC_ERROR_HANDLING=0                                                # HCCL async error handling switch.
export WITHOUT_JIT_COMPILE=1                                                      # Disable JIT compile path for more stable NPU startup.
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE                                        # Disable HCCL FFTS operator base mode for compatibility.
export COMBINED_ENABLE=1                                                          # Ascend combined-operator switch used by the NPU runtime.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}                                      # CPU thread count per process.
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}                              # Limit project logs to rank 0 when set.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # Disable tokenizer worker parallelism warnings.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # Ensure project modules are importable.

# Dependency installation for managed NPU images. Set INSTALL_DEPS=False on prebuilt images.
INSTALL_DEPS=${INSTALL_DEPS:-True}                                                # Whether the script installs Python dependencies before launch.
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}                              # Whether to replace platform moxing with the pinned wheel.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers>=5.7.0"}                     # Transformers package spec; Qwen3.5 scripts may require newer versions.
TOKENIZERS_SPEC=${TOKENIZERS_SPEC:-"tokenizers>=0.22.0"}                          # Tokenizers package spec aligned with TRANSFORMERS_SPEC.
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
  pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "${TRANSFORMERS_SPEC}" "${TOKENIZERS_SPEC}"
  pip install accelerate==1.6.0 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless>=4.8.0'
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15
fi

# Helper functions for parsing checkpoint lists and resolving best/direct checkpoint roots.
resolve_training_checkpoint() {
  python - "$1" <<'PYRESOLVE'
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin")):
    print(root)
    raise SystemExit(0)
cmd = [sys.executable, "scripts/tools/resolve_best_checkpoint.py", "--output-dir", str(root), "--best-name", "infer_best", "--best-name", "eval_best", "--best-name", "best", "--best-name", "best_reward", "--allow-direct"]
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

# Distributed topology: local defaults or ModelArts-provided node metadata.
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}                                                             # Distributed node count.
  NODE_RANK=${NODE_RANK:-0}                                                       # Rank of this node in the distributed job.
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}                                             # Number of NPU worker processes on each node.
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}                                           # Distributed rendezvous master address.
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}                                                 # Distributed node count.
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}                                          # Rank of this node in the distributed job.
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}                                  # Number of NPU worker processes on each node.
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}                              # Distributed rendezvous master address.
fi
MASTER_PORT=${MASTER_PORT:-6060}                                                  # Distributed rendezvous master port.
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}_${RUN_ID}}  # Unique rendezvous id for this distributed run.
mkdir -p "${LOCAL_MODEL_SAVE_PATH}"
OUTPUT_PATH="${LOCAL_MODEL_SAVE_PATH}"                                            # Actual output_dir passed to the trainer.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab}                        # Local SwanLab log directory.

# Download recipe-specific assets and the dataset, then verify required local paths.
if [ ! -e "${VISION_TOWER}/config.json" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${VISION_TOWER_OBS_PATH}', '${VISION_TOWER}')"
fi
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

if [ -n "${STAGE_A_CHECKPOINT_OBS_PATH}" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${STAGE_A_CHECKPOINT_OBS_PATH}', '${STAGE_A_DOWNLOAD_DIR}')"
  CHECKPOINT_INPUT_PATH="${STAGE_A_DOWNLOAD_DIR}"
elif [ -n "${STAGE_A_CHECKPOINT_PATH}" ]; then
  CHECKPOINT_INPUT_PATH="${STAGE_A_CHECKPOINT_PATH}"
else
  echo "ERROR: set STAGE_A_CHECKPOINT_OBS_PATH or STAGE_A_CHECKPOINT_PATH for Stage-B SFT."
  exit 1
fi
INIT_MODEL_PATH=$(resolve_training_checkpoint "${CHECKPOINT_INPUT_PATH}")         # Initial model path passed to train_qwen. Stage B uses the Stage-A checkpoint.
if [ "${TRAIN_VARIANT}" = "lora_llm" ] && compgen -G "${INIT_MODEL_PATH}/adapter_model*" > /dev/null; then
  echo "ERROR: Stage-B lora_llm expects a full Stage-A checkpoint. Adapter-only Stage-A LoRA continuation is not supported by this train entrypoint."
  exit 1
fi

TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"                         # Training JSONL path for the selected dataset phase.
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"                           # Evaluation JSONL path for the selected dataset phase.
# Fail early if any required model, dataset, image, or vision asset is missing.
for path in "${INIT_MODEL_PATH}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}" "${REQUIRED_VISION_TOWERS[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

# Derive gradient accumulation from the requested global batch size.
TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))                                      # Total number of NPU processes across all nodes.
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))                    # Global micro-batch size before gradient accumulation.
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))  # Derived accumulation steps to reach TARGET_GLOBAL_BATCH_SIZE.
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then GRADIENT_ACCUMULATION_STEPS=1; fi

# Build optional eval and vision argument arrays for the Python entrypoint.
EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")  # Transformers-compatible eval argument name for the installed version.
EVAL_ARGS=()                                                                      # Optional Trainer eval arguments, populated only when ENABLE_EVAL is true.
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(--eval_data_path "${EVAL_PATH}" --eval_image_folder "${IMAGE_FOLDER}" "${EVAL_STRATEGY_ARG}" steps --eval_steps "${EVAL_STEPS}" --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}" --best_eval_loss_dir eval_best)  # Optional Trainer eval arguments, populated only when ENABLE_EVAL is true.
fi
# Vision CLI arguments are assembled once to keep train/test launches consistent.
VISION_ARGS=(--vision_tower "${VISION_TOWER}" --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" --input_image_size "${INPUT_IMAGE_SIZE}")  # Model vision arguments shared by train or inference launch commands.

if [ -n "${VISION_LAYER_FUSION_INDEXES}" ]; then
  VISION_ARGS+=(--vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES} --vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}")
fi
if [[ ! "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ && -n "${DEEPSTACK_VISUAL_INDEXES}" ]]; then
  VISION_ARGS+=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi
BEST_INFER_VISION_TOWER="${VISION_TOWER}"                                         # Vision tower value passed to best-infer checkpoint selection.

# Print the resolved run configuration before the expensive launch.
echo "============================================================"
echo "Recipe:       ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_RECIPE} | ${MODEL_FAMILY} | ${TRAIN_VARIANT}"
echo "Init model:   ${INIT_MODEL_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Vision type:  ${MM_VISION_TOWER_TYPE} fusion=${MULTI_VISION_FUSION:-single}"
echo "DeepStack disabled: ${DISABLE_DEEPSTACK}, indexes=${DEEPSTACK_VISUAL_INDEXES:-auto}"
echo "Layer fusion: ${VISION_LAYER_FUSION_INDEXES:-off} (${VISION_LAYER_FUSION_TYPE})"
echo "LoRA:         enable=${LORA_ENABLE}, scope=${LORA_TARGET_SCOPE}, r=${LORA_R}"
echo "Train:        ${TRAIN_PATH}"
echo "Eval:         ${EVAL_PATH}"
echo "Output:       ${OUTPUT_PATH}"
echo "Cloud output: ${CLOUD_OUTPUT_PATH}"
echo "============================================================"

# Launch the recipe entrypoint. Training uses HCCL/DDP and full SFT may add DeepSpeed.
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  --model_name_or_path "${INIT_MODEL_PATH}" \
  --version conv_qwen_3_Dinov2_huawei \
  "${VISION_ARGS[@]}" \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
  --disable_deepstack "${DISABLE_DEEPSTACK}" \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --sample_seed 42 \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_target_scope "${LORA_TARGET_SCOPE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_bias "${LORA_BIAS}" \
  --num_train_epochs "${NUM_EPOCHS}" \
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
  --best_infer_index_vision_tower "${BEST_INFER_VISION_TOWER}" \
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
  --ddp_backend hccl

# Rank 0 moves final training artifacts to the platform cloud output path.
if [[ "${NODE_RANK}" == "0" ]]; then
  if [ -e "${CLOUD_OUTPUT_PATH}" ]; then
    echo "ERROR: cloud output path already exists, refusing to overwrite: ${CLOUD_OUTPUT_PATH}"
    exit 1
  fi
  echo "Moving rank0 local output to cloud output: ${OUTPUT_PATH} -> ${CLOUD_OUTPUT_PATH}"
  mv "${OUTPUT_PATH}" "${CLOUD_OUTPUT_PATH}"
  echo "Final cloud output path: ${CLOUD_OUTPUT_PATH}"
else
  echo "Non-master node ${NODE_RANK}: skip cloud output move."
fi
