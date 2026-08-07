#!/usr/bin/env bash

# ============================================================
# DI/NPU LoRA recipe for the released local256 Raw-Lane + Pose 800k dataset.
# All three ordered images are processed by the native Qwen3-VL-8B visual
# tower and multimodal merger before entering its native text LLM.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative imports.
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"                # Required cloud output root provided by ModelArts.

echo "[di-entry] reached local256-800k three-image native-Qwen3VL-8B LoRA launcher"
echo "[di-entry] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"
echo "DI_throughput: 0.00 samples/s/npu"

DATASET_PHASE=phase_a                                                             # Dataset stage: phase_a patch construction.
MAP_TASK=lane_intersection                                                        # Task type: lane or lane_intersection.
MODEL_RECIPE=qwen3vl_native                                                       # Native Qwen3-VL visual+LLM architecture.
TRAIN_VARIANT=lora                                                               # PEFT LoRA on language, visual attention, and native merger.

CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root injected by the platform.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias used by existing project scripts.
if [ -n "${MA_VJ_NAME:-}" ]; then
  DEFAULT_RUN_ID=$(printf '%s' "${MA_VJ_NAME}" | tr -c 'A-Za-z0-9_.-' '_')
else
  DEFAULT_RUN_ID=datasetv2_three_image_local256_800k_native_qwen3vl8b_lora_$(date -u +%Y%m%d_%H%M%S)
fi
RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}                                               # Unique run id for local and OBS outputs.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local worker cache root.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS root for model assets.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_rawpos/local256_rawlane_pose_800k.tar}  # Released local256 three-image 800k TAR.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-local256_rawlane_pose_800k}                  # Preferred root after extraction.
EXPECTED_DATASET_VARIANTS=${EXPECTED_DATASET_VARIANTS:-local256_rawlane_pose_800k,local256_rawlane_pose,rawlane_pose_three_image_local256_800k}  # Released identity plus historical aliases.

QWEN3VL_MODEL_NAME=${QWEN3VL_MODEL_NAME:-Qwen3-VL-8B-Instruct}                    # Native Qwen3-VL checkpoint directory name under MODEL_OBS_PATH.
QWEN3VL_OBS_PATH=${QWEN3VL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/Qwen3-VL-8B-Instruct/}  # Verified native Qwen3-VL-8B checkpoint.
QWEN3VL_PATH=${QWEN3VL_PATH:-${OBS_CACHE}/checkpoints/${QWEN3VL_MODEL_NAME}}      # Local native Qwen3-VL checkpoint path.
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.tar}  # Local dataset archive path.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Local extraction root.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}         # Extracted dataset root.
IMAGE_FOLDER=${IMAGE_FOLDER:-}                                                    # Defaults after dynamic dataset-root resolution.
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}                                   # Final OBS/cloud output path.
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/local_model_save_path}      # Local save root.
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}  # Per-run local output dir.

TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}                         # Desired global batch size.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}                     # Per-NPU micro batch; accumulation preserves global 128.
NUM_EPOCHS=${NUM_EPOCHS:-8}                                                       # Formal SFT epochs.
MAX_STEPS=${MAX_STEPS:--1}                                                        # Positive values are reserved for smoke runs.
LR=${LR:-2e-4}                                                                    # Native Qwen3-VL language LoRA learning rate.
VISION_LORA_LR=${VISION_LORA_LR:-${LR}}                                           # Match the language LoRA LR by default.
MERGER_LORA_LR=${MERGER_LORA_LR:-${LR}}                                           # Match the language LoRA LR by default.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}                                                 # Weight decay.
WARMUP_RATIO=${WARMUP_RATIO:-0.03}                                                # Warmup ratio.
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}                                        # Native Qwen3-VL uses fewer visual tokens than DINOv2 A.
EXPECTED_SOURCE_TRAIN_SAMPLES=${EXPECTED_SOURCE_TRAIN_SAMPLES:-800000}            # Released source-package train count.
TRAIN_SAMPLE_LIMIT=${TRAIN_SAMPLE_LIMIT:-0}                                       # Zero consumes all 800k records without resampling.
SAMPLE_SEED=${SAMPLE_SEED:-42}                                                    # Used only when a positive subset override is requested.
SAVE_STEPS=${SAVE_STEPS:-1000}                                                    # Regular checkpoint interval.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}                                          # Regular checkpoint keep limit.
LOGGING_STEPS=${LOGGING_STEPS:-10}                                                # Logging interval.
ENABLE_EVAL=False                                                                 # Keep the architecture comparison focused on SFT throughput and quality.
SAVE_BEST_TRAIN_LOSS=False                                                        # Regular adapter checkpoints only.
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-5000}                    # Best train-loss starts after this step.
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-5}                       # Best checkpoint keep limit.
LORA_ENABLE=True                                                                  # Language-model LoRA.
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-none}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
VISION_LORA_ENABLE=True                                                           # Adapt native visual attention without full-tower optimizer states.
VISION_LORA_TARGET_MODULES=${VISION_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,qkv,proj}
MERGER_LORA_ENABLE=True                                                           # Adapt every Linear layer in the native multimodal merger with LoRA.
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}
DATASET_INSPECT_STRICT=${DATASET_INSPECT_STRICT:-True}
DATASET_INSPECT_MAX_SAMPLES=${DATASET_INSPECT_MAX_SAMPLES:-0}
DATASET_IMAGE_CHECKS_PER_SPLIT=${DATASET_IMAGE_CHECKS_PER_SPLIT:-64}
DATASET_ALLOWED_TARGET_LANE_TYPES=${DATASET_ALLOWED_TARGET_LANE_TYPES:-"common right_turn waiting_area bus_lane main_auxiliary_connector other"}
DATASET_ALLOWED_TARGET_INTERSECTION_TYPES=${DATASET_ALLOWED_TARGET_INTERSECTION_TYPES:-"common t_intersection small_untyped t_lane_change_area other"}
RAW_LANE_PROMPT_TEXT=${RAW_LANE_PROMPT_TEXT:-"second image is a lane image predicted by a PV camera model"}
POSE_PROMPT_TEXT=${POSE_PROMPT_TEXT:-"third image is a historical vehicle-trajectory image"}
DEFAULT_SYSTEM_PROMPT=$'You are a road-map reconstruction assistant designed to process BEV (Bird\'s Eye View) images generated from LiDAR data.\nPredict the complete road map from the current patch in the BEV image.\nReturn only valid JSON in the required schema.\nDo not output markdown fences or extra explanation.\nKeep all coordinates in the patch-local coordinate system.'
SYSTEM_PROMPT=${SYSTEM_PROMPT:-${DEFAULT_SYSTEM_PROMPT}}

SWANLAB_ENABLE=${SWANLAB_ENABLE:-False}                                           # Enable SwanLab logging; native baseline defaults to disabled.
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}                                      # Optional SwanLab key; keep secrets in the DI environment.
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v9}                                  # SwanLab project.
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_phase_a_lane_intersection_datasetv2_three_image_local256_800k_native_qwen3vl8b}  # Native local256 group.
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-native_qwen3vl8b_lora_three_image_local256_800k}  # Experiment name.
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,lora,phase_a,lane_intersection,datasetv2,three_image,local256,800k,native_qwen3vl8b}  # Tags.
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
export MLLM_LOG_NATIVE_MULTI_IMAGE_SHAPE=${MLLM_LOG_NATIVE_MULTI_IMAGE_SHAPE:-1}  # Prove all three images reach the native processor.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # Disable tokenizer parallel warnings.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # Project imports.

INSTALL_DEPS=${INSTALL_DEPS:-True}                                                # Install dependencies on managed NPU images.
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}                              # Upgrade moxing wheel.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers==4.57.3"}                    # Native Qwen3-VL with a Torch-2.4-compatible DTensor guard.
TOKENIZERS_SPEC=${TOKENIZERS_SPEC:-"tokenizers>=0.22.0"}                          # Tokenizers version aligned with transformers.
PEFT_SPEC=${PEFT_SPEC:-"peft==0.18.0"}                                            # LoRA version compatible with the pinned Transformers 4.x build.
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
  pip install accelerate==1.6.0 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "${PEFT_SPEC}" pydantic 'markdown2[all]' numpy==1.26.4 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' opencv-python-headless==4.11.0.86
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab protobuf==4.25.7 urllib3==1.26.15
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
if [ ! -e "${QWEN3VL_PATH}/config.json" ]; then
  echo "ERROR: native Qwen3-VL model download is incomplete: ${QWEN3VL_PATH}"
  exit 1
fi

if [[ "${REUSE_LOCAL_ASSETS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && [ -s "${DATASET_ARCHIVE_PATH}" ]; then
  echo "[dataset-download] reuse ${DATASET_ARCHIVE_PATH}"
else
  mkdir -p "$(dirname "${DATASET_ARCHIVE_PATH}")"
  python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ARCHIVE_PATH}')"
fi
if [ ! -s "${DATASET_ARCHIVE_PATH}" ]; then
  echo "ERROR: dataset archive is missing or empty: ${DATASET_ARCHIVE_PATH}"
  exit 1
fi

if [[ "${REUSE_LOCAL_ASSETS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && \
   find "${DATASET_EXTRACT_ROOT}" -type f -path '*/phase_a/train.jsonl' -print -quit | grep -q .; then
  echo "[dataset-extract] reuse ${DATASET_EXTRACT_ROOT}"
else
  mkdir -p "${DATASET_EXTRACT_ROOT}"
  case "${DATASET_ARCHIVE_PATH}" in
    *.tar|*.tar.gz|*.tgz) tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}" ;;
    *.zip) unzip -q -o "${DATASET_ARCHIVE_PATH}" -d "${DATASET_EXTRACT_ROOT}" ;;
    *) echo "ERROR: unsupported dataset archive: ${DATASET_ARCHIVE_PATH}"; exit 1 ;;
  esac
fi

if [ ! -f "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" ]; then
  DATASET_PATH=$(python - "${DATASET_EXTRACT_ROOT}" "${DATASET_DIR_NAME}" "${DATASET_PHASE}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
preferred = extract_root / sys.argv[2]
phase = sys.argv[3]
roots = []
for candidate in (preferred, extract_root):
    if (candidate / phase / "train.jsonl").is_file():
        roots.append(candidate)
for train_path in extract_root.rglob("train.jsonl"):
    if train_path.parent.name != phase or "__MACOSX" in train_path.parts:
        continue
    root = train_path.parent.parent
    if root not in roots:
        roots.append(root)
if len(roots) != 1:
    preview = "\n".join(str(root) for root in roots[:20]) or "<none>"
    raise SystemExit(f"Unable to resolve exactly one dataset root below {extract_root}:\n{preview}")
print(roots[0])
PY
  )
fi
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl"
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl"
[ -f "${EVAL_PATH}" ] || EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/val.jsonl"
for path in "${QWEN3VL_PATH}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then echo "ERROR: required path not found: ${path}"; exit 1; fi
done

python - "${DATASET_PATH}" "${EXPECTED_DATASET_VARIANTS}" "${EXPECTED_SOURCE_TRAIN_SAMPLES}" "${TRAIN_SAMPLE_LIMIT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_variants = {item.strip() for item in sys.argv[2].split(",") if item.strip()}
expected_source_records = int(sys.argv[3])
sample_limit = int(sys.argv[4])
info_path = root / "dataset_info.json"
if not info_path.is_file():
    raise SystemExit(f"Three-image dataset metadata not found: {info_path}")
info = json.loads(info_path.read_text(encoding="utf-8"))
actual_variant = info.get("dataset_variant") or info.get("active_variant")
if actual_variant not in expected_variants:
    raise SystemExit(f"Expected one of dataset variants {sorted(expected_variants)!r}, found {actual_variant!r}")
overlay = info.get("input_overlay") or {}
multi = info.get("multi_image_input") or {}
expected_roles = ["bev_road_structure", "pv_camera_raw_lane", "historical_vehicle_trajectory"]
expected_prompt_contract = "three_image_roles_concise_v2"
if overlay.get("raw_lane_overlay") is not False:
    raise SystemExit("dataset_info.json must keep Raw-Lane separate from the clean BEV")
if overlay.get("raw_lane_separate_image") is not True:
    raise SystemExit("dataset_info.json does not enable the separate Raw-Lane input")
if int(multi.get("num_images_per_sample", 0)) != 3:
    raise SystemExit(f"Invalid three-image metadata: {multi!r}")
metadata_roles = multi.get("image_roles")
metadata_order = multi.get("image_order")
if metadata_roles is not None and metadata_order is not None and list(metadata_roles) != list(metadata_order):
    raise SystemExit(f"Conflicting three-image role metadata: {multi!r}")
resolved_roles = metadata_roles if metadata_roles is not None else metadata_order
if list(resolved_roles or []) != expected_roles:
    raise SystemExit(f"Unexpected three-image role order: {multi!r}")
metadata_prompt_contract = info.get("three_image_prompt_contract_version")
if metadata_prompt_contract not in (None, "", expected_prompt_contract):
    raise SystemExit(
        f"Unexpected three-image prompt contract {metadata_prompt_contract!r}; "
        f"expected {expected_prompt_contract!r}"
    )
validation_path = root / "three_image_validation.json"
if not validation_path.is_file():
    raise SystemExit(f"Builder validation marker is missing: {validation_path}")
validation = json.loads(validation_path.read_text(encoding="utf-8"))
if validation.get("status") != "passed":
    raise SystemExit(f"Builder validation did not pass: {validation!r}")
validation_prompt_contract = validation.get("prompt_contract_version")
if validation_prompt_contract not in (None, "", expected_prompt_contract):
    raise SystemExit(f"Builder validation uses a stale prompt contract: {validation!r}")
if not metadata_prompt_contract and not validation_prompt_contract:
    print(
        "[three-image-preflight] prompt contract version metadata is missing; "
        "record-level prompt inspection will enforce the concise three-image contract"
    )
train_path = root / "phase_a" / "train.jsonl"
records = sum(1 for line in train_path.open("r", encoding="utf-8") if line.strip())
if records != expected_source_records:
    raise SystemExit(f"Expected {expected_source_records} source records, found {records}")
if sample_limit < 0 or sample_limit > records:
    raise SystemExit(f"Invalid sample limit {sample_limit}; source contains {records}")
selected = records if sample_limit == 0 else sample_limit
print(f"[three-image-preflight] variant={actual_variant} source={records} selected={selected}")
PY

INSPECT_ARGS=(
  --dataset-root "${DATASET_PATH}"
  --phase "${DATASET_PHASE}"
  --expected-image-size 256
  --coord-min 0
  --coord-max 1000
  --image-checks-per-split "${DATASET_IMAGE_CHECKS_PER_SPLIT}"
  --require-three-image-rawlane-pose
  --forbid-lane-type 3
  --require-centerline-type-field
  --require-intersection-type-field
  --forbid-intersection-subtype-field
  --require-taxonomy-prompt
  --required-prompt-text "${RAW_LANE_PROMPT_TEXT}"
  --required-prompt-text "${POSE_PROMPT_TEXT}"
  --report "${OUTPUT_PATH}/dataset_inspection_rank${NODE_RANK}.json"
)
for lane_type in ${DATASET_ALLOWED_TARGET_LANE_TYPES}; do INSPECT_ARGS+=(--allowed-centerline-type "${lane_type}"); done
for intersection_type in ${DATASET_ALLOWED_TARGET_INTERSECTION_TYPES}; do INSPECT_ARGS+=(--allowed-intersection-type "${intersection_type}"); done
if [ "${DATASET_INSPECT_MAX_SAMPLES}" -gt 0 ]; then INSPECT_ARGS+=(--max-samples-per-split "${DATASET_INSPECT_MAX_SAMPLES}"); fi
if [[ "${DATASET_INSPECT_STRICT}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then INSPECT_ARGS+=(--strict); fi
python scripts/tools/inspect_lane_intersection_training_dataset.py "${INSPECT_ARGS[@]}"

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))                                      # Total NPU workers.
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))                    # Global micro batch.
if [ $(( TARGET_GLOBAL_BATCH_SIZE % MICRO_BATCH )) -ne 0 ]; then
  echo "ERROR: TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE} is not divisible by global micro batch ${MICRO_BATCH}."
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$(( TARGET_GLOBAL_BATCH_SIZE / MICRO_BATCH ))

echo "============================================================"
echo "Recipe:       ${DATASET_PHASE} | ${MAP_TASK} | ${MODEL_RECIPE}"
echo "Init model:   ${QWEN3VL_PATH}"
echo "Architecture: native Qwen3-VL-8B vision tower + native multimodal merger + text LLM"
echo "Train mode:   LLM + vision-attention + native-merger LoRA r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Train:        ${TRAIN_PATH}"
if [ "${TRAIN_SAMPLE_LIMIT}" -eq 0 ]; then
  echo "Train subset: all ${EXPECTED_SOURCE_TRAIN_SAMPLES} records, resampled=False"
else
  echo "Train subset: ${TRAIN_SAMPLE_LIMIT}/${EXPECTED_SOURCE_TRAIN_SAMPLES}, seed=${SAMPLE_SEED}"
fi
echo "System prompt: matched to conv_qwen_3_Dinov2_huawei"
echo "Images:       3 x local256; native Qwen3-VL dynamic processor"
echo "Sequence:     max_length=${MODEL_MAX_LENGTH}"
echo "Batch:        per_device=${PER_DEVICE_TRAIN_BATCH_SIZE}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, effective=$((MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS))"
echo "LRs:          language_lora=${LR}, vision_lora=${VISION_LORA_LR}, merger_lora=${MERGER_LORA_LR}"
echo "Eval loss:    disabled"
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
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --train_sample_limit "${TRAIN_SAMPLE_LIMIT}" \
  --sample_seed "${SAMPLE_SEED}" \
  --system_prompt "${SYSTEM_PROMPT}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_bias "${LORA_BIAS}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --vision_lora_enable "${VISION_LORA_ENABLE}" \
  --vision_lora_target_modules "${VISION_LORA_TARGET_MODULES}" \
  --merger_lora_enable "${MERGER_LORA_ENABLE}" \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --vision_lora_learning_rate "${VISION_LORA_LR}" \
  --merger_lora_learning_rate "${MERGER_LORA_LR}" \
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
  --ddp_backend hccl

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
