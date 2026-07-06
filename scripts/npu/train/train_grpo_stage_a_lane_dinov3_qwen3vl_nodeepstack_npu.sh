#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU GRPO training
# Fixed recipe: phase_a | lane-only centerline | dinov3 + Qwen3-VL-8B | no DeepStack
# This file is self-contained and does not call another project .sh file.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative script and Python imports.
# Platform I/O and recipe metadata are declared first so cloud jobs can be audited quickly.
cd "${REPO_ROOT}"

# Recipe identity: fixed task, visual architecture, model family, and train variant.
DATASET_PHASE=phase_a                                                             # Dataset stage: phase_a for patch inference, phase_b for state update.
MAP_TASK=lane                                                                     # Task type: lane or lane_intersection.
VISION_BACKBONE=dinov3                                                            # Visual backbone selector used by the generic multi-vision launcher.
# Vision assets for this recipe. Scripts only download the towers declared below.
VISION_TOWER_NAME=facebook_dinov3-vitl16-pretrain-lvd1689m                        # Single vision tower directory name under MODEL_OBS_PATH.
MM_VISION_TOWER_TYPE=dinov3                                                       # Model-side vision tower type: dinov2, dinov3, multi_moe, or multi_concat.
INPUT_IMAGE_SIZE=512                                                              # Image size fed to the vision encoder; DINOv3 recipes usually use 512.

echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "Recipe: ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
# ====================== cloud paths ======================
# OUTPUT_URL is injected by the cloud training platform.
# Keep the reference-script convention: mirror OUTPUT_URL into OSB_SHARE_PATH.
# Cloud and local storage roots. Outputs are staged locally before OBS upload.
CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root injected by the training platform.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias of the platform output root used by existing scripts.
echo "System defined obs share path: ${OSB_SHARE_PATH}"

# GRPO writes to local cache first, then uploads the complete run dir to OBS.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id for local cache and cloud output folders.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local worker cache root for models, datasets, checkpoints, and outputs.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS directory that stores model and vision checkpoint assets.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}  # OBS zip path for the prepared UniMapGen dataset.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}                       # Dataset directory name expected after the zip is extracted.

# Continuation checkpoint inputs. OBS paths are downloaded, local paths are used directly.
SFT_CHECKPOINT_OBS_PATH=${SFT_CHECKPOINT_OBS_PATH:-}                              # GRPO: OBS root of the SFT checkpoint used as initial policy.
SFT_CHECKPOINT_PATH=${SFT_CHECKPOINT_PATH:-}                                      # GRPO: local SFT checkpoint root used when OBS path is empty.
SFT_DOWNLOAD_DIR=${SFT_DOWNLOAD_DIR:-${OBS_CACHE}/sft_checkpoint_${RUN_ID}}       # GRPO: local download directory for the SFT checkpoint.
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}       # Vision tower path passed to the model loader. Multi-vision uses a comma list.
# Local dataset and output paths for this run.
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}          # Local path for the downloaded dataset zip.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Local directory where the dataset zip is extracted.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/data_line_samples_33w}       # Extracted dataset root containing phase_a and phase_b folders.
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}                                     # Image root passed to training or inference. Usually DATASET_PATH.
LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-${OBS_CACHE}/grpo_phase_a_lane_dinov3_output_${RUN_ID}}  # Per-run local GRPO output directory.
CLOUD_OUTPUT_DIR=${GRPO_RESULT_OBS:-${OSB_SHARE_PATH%/}/${RUN_ID}}                # Final cloud output directory for inference or GRPO results.

# ====================== GRPO params ======================
# Main knobs for this GRPO recipe. Edit values here instead of passing one-off shell prefixes.
# ACTOR_NPU_DEVICES and ROLLOUT_NPU_DEVICES select the training actor and rollout engine devices.
# KL_BETA controls reference-policy regularization; reward weights are task-specific.
# Main runtime parameters and hyperparameters.
ACTOR_NPU_DEVICES=${ACTOR_NPU_DEVICES:-0}                                         # NPU ids used by the GRPO actor trainer.
ROLLOUT_NPU_DEVICES=${ROLLOUT_NPU_DEVICES:-1}                                     # NPU ids used by the GRPO rollout engine.
NUM_GENERATIONS=${NUM_GENERATIONS:-2}                                             # Number of completions sampled per prompt for GRPO.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}                                             # Maximum number of generated tokens per sample.
KL_BETA=${KL_BETA:-0.02}                                                          # GRPO KL penalty weight against the reference policy.
MAX_STEPS=${MAX_STEPS:-100}                                                       # Maximum optimizer steps for GRPO.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}                     # Micro batch size per NPU process.
LR=${LR:-1e-6}                                                                    # Base learning rate for LLM and default trainable parameters.
REWARD_INTERSECTION_WEIGHT=${REWARD_INTERSECTION_WEIGHT:-0.0}                     # Extra reward weight for intersection fields. Use 0 for lane-only.

# Experiment logging metadata.
SWANLAB_ENABLE=${SWANLAB_ENABLE:-True}                                            # Enable SwanLab experiment logging.
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}                # SwanLab API key. Override from platform env when possible.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}                                  # SwanLab project name.
SWANLAB_GROUP=${SWANLAB_GROUP:-grpo_phase_a_lane_dinov3_nodeepstack}              # SwanLab group name for related runs.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-grpo_phase_a_lane_dinov3_qwen3vl8b_nodeepstack}  # SwanLab experiment display name.
SWANLAB_TAGS=${SWANLAB_TAGS:-grpo,phase_a,lane,dinov3,qwen3vl8b,nodeepstack,vllm-ascend}  # Comma-separated SwanLab tags for filtering runs.
SWANLAB_MODE=${SWANLAB_MODE:-}                                                    # SwanLab mode, for example offline on restricted cloud networks.
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${LOCAL_OUTPUT_DIR}/swanlab}                   # Local SwanLab log directory.
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}                                            # Optional SwanLab private API host.
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}                                            # Optional SwanLab private web host.
# ====================== Ascend environment ======================
# Ascend and HCCL runtime environment for NPU jobs.
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend toolkit root.
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend custom operator package root.
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend operator package path.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
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
VLLM_VERSION=${VLLM_VERSION:-0.9.2}                                               # vLLM version used by GRPO rollout workers.
VLLM_ASCEND_VERSION=${VLLM_ASCEND_VERSION:-0.9.2rc1}                              # vLLM-Ascend version used by GRPO rollout workers.

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
  pip install "ray[default]==2.55.1"
  pip install "vllm==${VLLM_VERSION}" "vllm-ascend==${VLLM_ASCEND_VERSION}"
  pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl
fi
export VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-npu}                              # Device backend target for vLLM rollout.

# Download recipe-specific assets and the dataset, then verify required local paths.
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}" "${LOCAL_OUTPUT_DIR}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"

if [ -n "${SFT_CHECKPOINT_OBS_PATH}" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${SFT_CHECKPOINT_OBS_PATH}', '${SFT_DOWNLOAD_DIR}')"
  CHECKPOINT_INPUT_PATH="${SFT_DOWNLOAD_DIR}"
elif [ -n "${SFT_CHECKPOINT_PATH}" ]; then
  CHECKPOINT_INPUT_PATH="${SFT_CHECKPOINT_PATH}"
else
  echo "ERROR: set SFT_CHECKPOINT_OBS_PATH or SFT_CHECKPOINT_PATH for GRPO."
  exit 1
fi
SFT_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors")):
    print(root)
    raise SystemExit(0)
cmd = [
    sys.executable,
    "scripts/tools/resolve_best_checkpoint.py",
    "--output-dir",
    str(root),
    "--best-name",
    "infer_best",
    "--best-name",
    "eval_best",
    "--best-name",
    "best",
    "--best-name",
    "best_reward",
    "--allow-direct",
]
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
PY
)
DATA_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"                          # GRPO training JSONL path.
# Fail early if any required model, dataset, image, or vision asset is missing.
for path in "${SFT_CHECKPOINT}" "${VISION_TOWER}" "${DATA_PATH}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

# Print the resolved run configuration before the expensive launch.
echo "============================================================"
echo "Recipe:        ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
echo "SFT checkpoint:${SFT_CHECKPOINT}"
echo "Train:         ${DATA_PATH}"
echo "Local output:  ${LOCAL_OUTPUT_DIR}"
echo "Run id:        ${RUN_ID}"
echo "Cloud output:  ${CLOUD_OUTPUT_DIR}"
echo "============================================================"

# Launch the recipe entrypoint. Training uses HCCL/DDP and full SFT may add DeepSpeed.
python -m mllm.train.train_grpo \
  --model_name_or_path "${SFT_CHECKPOINT}" \
  --version conv_qwen_3_Dinov2_huawei \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
  --input_image_size "${INPUT_IMAGE_SIZE}" \
  --disable_deepstack True \
  --tokenizer_use_fast False \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --image_aspect_ratio pad \
  --map_task "${MAP_TASK}" \
  --coord_mode auto \
  --coord_range 1000 \
  --output_dir "${LOCAL_OUTPUT_DIR}" \
  --rollout_backend vllm_prompt_embeds \
  --device_backend npu \
  --actor_npu_devices "${ACTOR_NPU_DEVICES}" \
  --rollout_npu_devices "${ROLLOUT_NPU_DEVICES}" \
  --actor_num_cpus "${ACTOR_NUM_CPUS:-4}" \
  --rollout_num_cpus "${ROLLOUT_NUM_CPUS:-4}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.70}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-2048}" \
  --vllm_enforce_eager "${VLLM_ENFORCE_EAGER:-True}" \
  --num_generations "${NUM_GENERATIONS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.9}" \
  --kl_beta "${KL_BETA}" \
  --clip_range "${CLIP_RANGE:-0.2}" \
  --reward_format_weight "${REWARD_FORMAT_WEIGHT:-0.08}" \
  --reward_centerline_instance_weight "${REWARD_CENTERLINE_INSTANCE_WEIGHT:-0.37}" \
  --reward_centerline_length_weight "${REWARD_CENTERLINE_LENGTH_WEIGHT:-0.45}" \
  --reward_cut_type_weight "${REWARD_CUT_TYPE_WEIGHT:-0.05}" \
  --reward_cut_continuity_weight "${REWARD_CUT_CONTINUITY_WEIGHT:-0.05}" \
  --reward_intersection_weight "${REWARD_INTERSECTION_WEIGHT}" \
  --lora_enable True \
  --lora_target_scope "${LORA_TARGET_SCOPE:-llm}" \
  --lora_r "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --max_steps "${MAX_STEPS}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --bf16 True \
  --model_max_length "${MODEL_MAX_LENGTH:-4096}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type grpo \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --swanlab_log_dir "${SWANLAB_LOG_DIR}" \
  --swanlab_api_host "${SWANLAB_API_HOST}" \
  --swanlab_web_host "${SWANLAB_WEB_HOST}"

python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_DIR}', '${CLOUD_OUTPUT_DIR}')"
