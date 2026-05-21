#!/usr/bin/env bash
set -euo pipefail

# Common Ascend NPU GRPO launcher for explicit stage/task/vision wrappers.
#
# Architecture:
#   multimodal actor on Ascend NPU -> prompt embeddings -> vLLM-Ascend rollout
#   -> map reward -> GRPO LoRA update -> adapter / merged checkpoint export.
#
# Required wrapper variables:
#   VISION_BACKBONE=dinov2|dinov3
#   DATASET_PHASE=phase_a|phase_b
#   MAP_TASK=lane|lane_intersection
#
# Main local paths:
#   DATASET_PATH=/cache/unimapgen_v2/dataset
#   SFT_CHECKPOINT=/cache/unimapgen_v2/train_output/sft_.../best
#   OUTPUT_DIR=/cache/unimapgen_v2/train_output/grpo_...
#
# Cloud mode:
#   If OUTPUT_URL is set, this script installs runtime deps, downloads dataset /
#   DINO / SFT checkpoint from OBS, trains GRPO locally, and uploads OUTPUT_DIR.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

VISION_BACKBONE=${VISION_BACKBONE:?set VISION_BACKBONE to dinov2 or dinov3}
DATASET_PHASE=${DATASET_PHASE:?set DATASET_PHASE to phase_a or phase_b}
MAP_TASK=${MAP_TASK:?set MAP_TASK to lane or lane_intersection}

case "${VISION_BACKBONE}" in
  dinov2|dinov3) ;;
  *) echo "ERROR: unsupported VISION_BACKBONE=${VISION_BACKBONE}"; exit 1 ;;
esac
case "${DATASET_PHASE}" in
  phase_a|phase_b) ;;
  *) echo "ERROR: unsupported DATASET_PHASE=${DATASET_PHASE}"; exit 1 ;;
esac
case "${MAP_TASK}" in
  lane|lane_intersection) ;;
  *) echo "ERROR: unsupported MAP_TASK=${MAP_TASK}"; exit 1 ;;
esac

CLOUD_MODE=False
if [ -n "${OUTPUT_URL:-}" ]; then
  CLOUD_MODE=True
fi
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "CLOUD_MODE=${CLOUD_MODE}"
echo "RUN_ID=${RUN_ID}"

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
export VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-npu}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ====================== dependencies ======================
INSTALL_DEPS=${INSTALL_DEPS:-${CLOUD_MODE}}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-${CLOUD_MODE}}
TORCH_NPU_OBS_WHL=${TORCH_NPU_OBS_WHL:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
VLLM_VERSION=${VLLM_VERSION:-0.9.2}
VLLM_ASCEND_VERSION=${VLLM_ASCEND_VERSION:-0.9.2rc1}

if [[ "${ENABLE_MOXING_UPGRADE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> changing moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
  pip uninstall moxing-framework -y
  pip cache purge
  pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
  export MOX_PROFILE=1
  export MOX_RECORD_OBS=1
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> moxing change finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
fi

if [[ "${INSTALL_DEPS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing GRPO NPU dependencies >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

  pip install torch==2.7.1
  pip install torch_npu==2.7.1rc1
  python -c "import moxing as mox; mox.file.copy_parallel('${TORCH_NPU_OBS_WHL}', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
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
  pip install 'scipy>=1.10'
  pip install 'scikit-learn>=1.2'
  pip install requests
  pip install uvicorn
  pip install fastapi
  pip install 'einops>=0.6'
  pip install 'einops-exts>=0.0.4'
  pip install 'timm>=0.9.0'
  pip install 'opencv-python-headless>=4.8.0'
  pip install 'loguru>=0.7.0'
  pip install 'shapely>=2.0.0'
  pip install wandb
  pip install swanlab
  pip install "huggingface-hub==0.36.2" --force-reinstall
  pip install urllib3==1.26.15
  pip install "ray[default]==2.55.1"
  pip install "vllm==${VLLM_VERSION}" "vllm-ascend==${VLLM_ASCEND_VERSION}"
  pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

  echo "========== key deps =========="
  python -c "import torch; print('torch', torch.__version__)"
  python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
  python -c "import transformers; print('transformers', transformers.__version__)"
  python -c "import vllm; print('vllm', getattr(vllm, '__version__', 'unknown'))"
  python -c "import vllm_ascend; print('vllm_ascend imported')"
  echo "==============================="
fi

# ====================== paths and cloud downloads ======================
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-MLLM20260427_rc_jjh}
SFT_CHECKPOINT_OBS=${SFT_CHECKPOINT_OBS:-${TRAINED_CHECKPOINT_OBS:-${CHECKPOINT_OBS:-}}}

case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER_NAME=${VISION_TOWER_NAME:-facebook_dinov2-large}
    VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov2}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
    ;;
  dinov3)
    VISION_TOWER_NAME=${VISION_TOWER_NAME:-facebook_dinov3-vitl16-pretrain-lvd1689m}
    VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}
    MM_VISION_TOWER_TYPE=${MM_VISION_TOWER_TYPE:-dinov3}
    INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}
    ;;
esac

if [[ "${CLOUD_MODE}" == "True" ]]; then
  DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
  DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
  DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}
  IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}

  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading vision tower >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"

  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
  mkdir -p "${DATASET_EXTRACT_ROOT}"
  unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
  if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: expected dataset directory not found: ${DATASET_PATH}"
    ls -la "${DATASET_EXTRACT_ROOT}"
    exit 1
  fi

  if [ -z "${SFT_CHECKPOINT:-}" ]; then
    if [ -z "${SFT_CHECKPOINT_OBS}" ]; then
      echo "ERROR: cloud GRPO requires SFT_CHECKPOINT_OBS/TRAINED_CHECKPOINT_OBS/CHECKPOINT_OBS."
      exit 1
    fi
    SFT_CHECKPOINT_LOCAL=${SFT_CHECKPOINT_LOCAL:-${OBS_CACHE}/sft_checkpoint_${RUN_ID}}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading SFT checkpoint from ${SFT_CHECKPOINT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    python -c "import moxing as mox; mox.file.copy_parallel('${SFT_CHECKPOINT_OBS}', '${SFT_CHECKPOINT_LOCAL}')"
    SFT_CHECKPOINT="${SFT_CHECKPOINT_LOCAL}"
  fi
else
  DATASET_PATH=${DATASET_PATH:-/cache/unimapgen_v2/dataset}
  IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
fi

if [ -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
  DATA_PATH=${DATA_PATH:-${DATASET_PATH}/${DATASET_PHASE}/train.jsonl}
else
  DATA_PATH=${DATA_PATH:-${DATASET_PATH}/train.jsonl}
fi

if [ -z "${SFT_CHECKPOINT:-}" ]; then
  SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-/cache/unimapgen_v2/train_output/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
  SFT_CHECKPOINT=$(python scripts/tools/resolve_best_checkpoint.py \
    --output-dir "${SFT_OUTPUT_DIR}" \
    --best-name eval_best \
    --best-name best \
    --allow-direct)
fi

if [ ! -f "${SFT_CHECKPOINT}/model.safetensors" ] && [ ! -f "${SFT_CHECKPOINT}/adapter_model.safetensors" ] && [ ! -f "${SFT_CHECKPOINT}/pytorch_model.bin" ]; then
  if RESOLVED_SFT=$(python scripts/tools/resolve_best_checkpoint.py \
      --output-dir "${SFT_CHECKPOINT}" \
      --best-name eval_best \
      --best-name best \
      --allow-direct 2>/dev/null); then
    SFT_CHECKPOINT="${RESOLVED_SFT}"
  fi
fi

if [[ "${CLOUD_MODE}" == "True" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-${OBS_CACHE}/grpo_output_${RUN_ID}}
else
  OUTPUT_DIR=${OUTPUT_DIR:-/cache/unimapgen_v2/train_output/grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
fi
mkdir -p "${OUTPUT_DIR}"

for path in "${DATA_PATH}" "${IMAGE_FOLDER}" "${VISION_TOWER}" "${SFT_CHECKPOINT}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done

# ====================== GRPO placement ======================
# Actor computes multimodal embeddings. Rollout runs vLLM-Ascend. They can use
# different NPU ids on multi-card machines. For one-card debug, set both to 0.
ACTOR_NPU_DEVICES=${ACTOR_NPU_DEVICES:-0}
ROLLOUT_NPU_DEVICES=${ROLLOUT_NPU_DEVICES:-1}
ACTOR_NUM_CPUS=${ACTOR_NUM_CPUS:-4}
ROLLOUT_NUM_CPUS=${ROLLOUT_NUM_CPUS:-4}
RAY_ADDRESS=${RAY_ADDRESS:-}
RAY_ARGS=()
if [ -n "${RAY_ADDRESS}" ]; then
  RAY_ARGS=(--ray_address "${RAY_ADDRESS}")
fi

if [ "${MAP_TASK}" = "lane_intersection" ]; then
  REWARD_INTERSECTION_WEIGHT=${REWARD_INTERSECTION_WEIGHT:-0.10}
else
  REWARD_INTERSECTION_WEIGHT=${REWARD_INTERSECTION_WEIGHT:-0.0}
fi

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}
SWANLAB_ENABLE=${SWANLAB_ENABLE:-True}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v3}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_GROUP=${SWANLAB_GROUP:-grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack}
SWANLAB_JOB_TYPE=${SWANLAB_JOB_TYPE:-grpo}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-grpo_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
SWANLAB_TAGS=${SWANLAB_TAGS:-grpo,${DATASET_PHASE},${MAP_TASK},${VISION_BACKBONE},qwen3vl8b,nodeepstack,vllm-ascend}

echo "============================================================"
echo "GRPO NPU:"
echo "  phase=${DATASET_PHASE} map_task=${MAP_TASK} vision=${VISION_BACKBONE}"
echo "  data=${DATA_PATH}"
echo "  image_folder=${IMAGE_FOLDER}"
echo "  sft_checkpoint=${SFT_CHECKPOINT}"
echo "  vision_tower=${VISION_TOWER}"
echo "  output=${OUTPUT_DIR}"
echo "  actor_npu=${ACTOR_NPU_DEVICES} rollout_npu=${ROLLOUT_NPU_DEVICES}"
echo "  vllm=${VLLM_VERSION} vllm_ascend=${VLLM_ASCEND_VERSION}"
echo "============================================================"

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
  --output_dir "${OUTPUT_DIR}" \
  --rollout_backend vllm_prompt_embeds \
  --device_backend npu \
  "${RAY_ARGS[@]}" \
  --actor_npu_devices "${ACTOR_NPU_DEVICES}" \
  --rollout_npu_devices "${ROLLOUT_NPU_DEVICES}" \
  --actor_num_cpus "${ACTOR_NUM_CPUS}" \
  --rollout_num_cpus "${ROLLOUT_NUM_CPUS}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.70}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-2048}" \
  --vllm_enforce_eager "${VLLM_ENFORCE_EAGER:-True}" \
  --num_generations "${NUM_GENERATIONS:-2}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.9}" \
  --kl_beta "${KL_BETA:-0.02}" \
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
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --learning_rate "${LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --max_steps "${MAX_STEPS:-100}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --bf16 True \
  --model_max_length "${MODEL_MAX_LENGTH:-4096}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_workspace "${SWANLAB_WORKSPACE}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type "${SWANLAB_JOB_TYPE}" \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE:-}"

if [[ "${CLOUD_MODE}" == "True" ]]; then
  GRPO_RESULT_OBS=${GRPO_RESULT_OBS:-${OUTPUT_URL%/}/grpo_results_${RUN_ID}}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Uploading GRPO results to ${GRPO_RESULT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy_parallel('${OUTPUT_DIR}', '${GRPO_RESULT_OBS}')"
  echo "GRPO results saved to ${GRPO_RESULT_OBS}"
fi
