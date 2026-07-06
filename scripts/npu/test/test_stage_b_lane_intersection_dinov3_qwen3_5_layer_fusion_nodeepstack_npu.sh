#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NPU inference
# Fixed recipe: phase_b | lane+intersection | DINOv3 direct ViT layer fusion, no DeepStack | Qwen3.5 text LLM
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
VISION_RECIPE=dinov3_layer_fusion                                                 # Fixed visual architecture recipe encoded by this script name.
MODEL_FAMILY=qwen3_5                                                              # LLM family selector: qwen3, qwen3_5, or qwen3vl.
MODEL_LABEL=qwen3_5                                                               # Short model label used in run names, logs, and output paths.
TRAIN_VARIANT=full                                                                # Training variant: full parameters or LLM LoRA.

case "${MAP_TASK}" in lane|lane_intersection) ;; *) echo "ERROR: MAP_TASK must be lane or lane_intersection"; exit 1 ;; esac
case "${MODEL_FAMILY}" in qwen3|qwen3_5) ;; *) echo "ERROR: MODEL_FAMILY must be qwen3 or qwen3_5"; exit 1 ;; esac

# Cloud and local storage roots. Outputs are staged locally before OBS upload.
CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root injected by the training platform.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias of the platform output root used by existing scripts.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id for local cache and cloud output folders.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local worker cache root for models, datasets, checkpoints, and outputs.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS directory that stores model and vision checkpoint assets.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}  # OBS zip path for the prepared UniMapGen dataset.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}                       # Dataset directory name expected after the zip is extracted.
CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}                                      # Comma, semicolon, or newline separated OBS checkpoint roots to evaluate.
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}                                              # Comma, semicolon, or newline separated local checkpoint roots to evaluate.

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
TEST_JSON=${TEST_JSON:-${DATASET_PATH}/${DATASET_PHASE}/test.jsonl}               # Inference JSONL path for the selected dataset phase.
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/checkpoints_${RUN_ID}}  # Local root used to download checkpoint candidates from OBS.
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-${OBS_CACHE}/test_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}_${RUN_ID}}  # Per-run local inference output root.
CLOUD_OUTPUT_DIR=${TEST_RESULT_OBS:-${OSB_SHARE_PATH%/}/test_results_${RUN_ID}}   # Final cloud output directory for inference or GRPO results.
# Main runtime parameters and hyperparameters.
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}                                           # Number of test samples to run; 0 means the full split.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}                                            # Maximum number of generated tokens per sample.
COORD_MODE=${COORD_MODE:-auto}                                                    # Coordinate mode: auto reads meta.coord_mode, or force norm1000 or pixel.
COORD_RANGE=${COORD_RANGE:-1000}                                                  # Coordinate range for normalized labels, normally 1000.

# Visual fusion controls. Empty layer lists disable the optional path.
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}                                      # True disables DeepStack residual injection; False allows it when indexes are set.
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-}                            # ViT layers for DeepStack residual injection, for example 6 12 18 23.
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-"6 12 18 23"}          # ViT layers fused into the main visual stream; empty disables direct fusion.
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}                        # Fusion mode: mean, sum, learned_weighted; aliases: weighted, softmax_weighted.

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
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless>=4.8.0'
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15
fi

# Helper functions for parsing checkpoint lists and resolving best/direct checkpoint roots.
read_list() {
  python - "$1" <<'PYREAD'
import re
import sys
for item in re.split("[,;" + chr(10) + "]+", sys.argv[1] or ""):
    item = item.strip()
    if item:
        print(item)
PYREAD
}

safe_label() {
  python - "$1" <<'PYLABEL'
import re
import sys
value = sys.argv[1].strip().rstrip("/") or "checkpoint"
label = re.sub(r"[^A-Za-z0-9._-]+", "_", value.split("/")[-1]).strip("._-")
print(label or "checkpoint")
PYLABEL
}

resolve_checkpoint() {
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
export RDZV_ID=${RDZV_ID:-test_${DATASET_PHASE}_${MAP_TASK}_${VISION_RECIPE}_${MODEL_LABEL}_${TRAIN_VARIANT}_${RUN_ID}}  # Unique rendezvous id for this distributed run.

# Download recipe-specific assets and the dataset, then verify required local paths.
if [ ! -e "${VISION_TOWER}/config.json" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${VISION_TOWER_OBS_PATH}', '${VISION_TOWER}')"
fi
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}" "${CHECKPOINT_DOWNLOAD_ROOT}" "${LOCAL_OUTPUT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"


# Build the checkpoint evaluation list from OBS roots or local directories.
CHECKPOINT_ITEMS=()                                                               # Resolved checkpoint paths that will be evaluated.
CHECKPOINT_LABELS=()                                                              # Display labels paired with CHECKPOINT_ITEMS.
if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
  while IFS= read -r obs_item; do
    label=$(safe_label "${obs_item}")
    local_dir="${CHECKPOINT_DOWNLOAD_ROOT}/${label}"
    python -c "import moxing as mox; mox.file.copy_parallel('${obs_item}', '${local_dir}')"
    resolved=$(resolve_checkpoint "${local_dir}")
    CHECKPOINT_ITEMS+=("${resolved}")
    CHECKPOINT_LABELS+=("${label}")
  done < <(read_list "${CHECKPOINT_OBS_LIST}")
elif [ -n "${CHECKPOINT_DIRS}" ]; then
  while IFS= read -r local_item; do
    resolved=$(resolve_checkpoint "${local_item}")
    CHECKPOINT_ITEMS+=("${resolved}")
    CHECKPOINT_LABELS+=("$(safe_label "${local_item}")")
  done < <(read_list "${CHECKPOINT_DIRS}")
else
  echo "ERROR: set CHECKPOINT_OBS_LIST or CHECKPOINT_DIRS for inference."
  exit 1
fi

# Fail early if any required model, dataset, image, or vision asset is missing.
for path in "${TEST_JSON}" "${IMAGE_FOLDER}" "${REQUIRED_VISION_TOWERS[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

# Vision CLI arguments are assembled once to keep train/test launches consistent.
VISION_ARGS=(--vision_tower "${VISION_TOWER}" --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" --input_image_size "${INPUT_IMAGE_SIZE}")  # Model vision arguments shared by train or inference launch commands.
if [[ "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  VISION_ARGS+=(--disable_deepstack)
fi

if [ -n "${VISION_LAYER_FUSION_INDEXES}" ]; then
  VISION_ARGS+=(--vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES} --vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}")
fi
if [[ ! "${DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ && -n "${DEEPSTACK_VISUAL_INDEXES}" ]]; then
  VISION_ARGS+=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
fi

# Run one checkpoint through inference, evaluation, visualization, and table printing.
run_one_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_label="$2"
  local output_dir="$3"
  local json_dir="${output_dir}/json"
  local patch_viz_dir="${output_dir}/viz"
  local whole_map_viz_dir="${output_dir}/whole_map_viz"
  local summary_json="${output_dir}/summary.json"
  local merged_global_json="${output_dir}/merged_global.json"
  local eval_json="${output_dir}/eval.json"
  mkdir -p "${json_dir}" "${patch_viz_dir}" "${whole_map_viz_dir}"
  echo "============================================================"
  echo "Infer:      ${checkpoint_label}"
  echo "Checkpoint: ${checkpoint_dir}"
  echo "Recipe:     ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_RECIPE} | ${MODEL_FAMILY} | ${TRAIN_VARIANT}"
  echo "Output:     ${output_dir}"
  echo "============================================================"
  if [ "${DATASET_PHASE}" = "phase_a" ]; then
    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      scripts/tools/infer_centerline_checkpoint.py \
      --checkpoint-dir "${checkpoint_dir}" \
      "${VISION_ARGS[@]}" \
      --test-json "${TEST_JSON}" \
      --num-samples "${NUM_TEST_SAMPLES}" \
      --image-folder "${IMAGE_FOLDER}" \
      --prompt-mode dataset \
      --map-task "${MAP_TASK}" \
      --patch-size 256 \
      --coord-mode "${COORD_MODE}" \
      --coord-range "${COORD_RANGE}" \
      --conv-template conv_qwen_3_Dinov2_huawei \
      --output-dir "${output_dir}" \
      --sample-json-dir "${json_dir}" \
      --output-json "${summary_json}" \
      --temperature 0.0 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --eval-centerline \
      --eval-output-json "${eval_json}"
  else
    INCLUDE_INTERSECTION_ARGS=()                                                  # Stage-B inference flag list for lane+intersection outputs.
    if [ "${MAP_TASK}" = "lane_intersection" ]; then
      INCLUDE_INTERSECTION_ARGS+=(--include-intersections)
    fi
    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      scripts/tools/infer_centerline_state_update.py \
      --checkpoint-dir "${checkpoint_dir}" \
      "${VISION_ARGS[@]}" \
      --patch-json "${TEST_JSON}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-json "${summary_json}" \
      --output-dir "${json_dir}" \
      --sample-json-dir "${json_dir}" \
      --merged-output-json "${merged_global_json}" \
      --whole-map-viz-dir "${whole_map_viz_dir}" \
      --conv-template conv_qwen_3_Dinov2_huawei \
      --device "${DEVICE:-auto}" \
      --patch-size 256 \
      --coord-mode "${COORD_MODE}" \
      --coord-range "${COORD_RANGE}" \
      "${INCLUDE_INTERSECTION_ARGS[@]}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --temperature 0.0 \
      --eval-centerline \
      --eval-output-json "${eval_json}" \
      --distributed-by-tile \
      --distributed-merge-timeout "${DISTRIBUTED_MERGE_TIMEOUT:-7200}"
  fi
  if [ "${NODE_RANK}" -ne 0 ]; then
    return 0
  fi
  python scripts/tools/visualize_centerline.py \
    --input-dir "${output_dir}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${patch_viz_dir}" \
    --eval-output-json "${eval_json}" \
    --whole-map-viz-dir "${whole_map_viz_dir}"
  if [ -f "${eval_json}" ]; then
    python - "${eval_json}" <<'PYTABLE'
import json
import sys
from pathlib import Path
from infer_index.line_eval import format_eval_table
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
payload = payload.get('summary', payload) if isinstance(payload, dict) else payload
print(payload.get('table') if isinstance(payload, dict) and payload.get('table') else format_eval_table(payload))
PYTABLE
  fi
}

# Evaluate each requested checkpoint, separating output folders when multiple are provided.
for index in "${!CHECKPOINT_ITEMS[@]}"; do
  label="${CHECKPOINT_LABELS[$index]}"
  checkpoint="${CHECKPOINT_ITEMS[$index]}"
  if [ "${#CHECKPOINT_ITEMS[@]}" -gt 1 ]; then
    output_dir="${LOCAL_OUTPUT_ROOT}/${index}_${label}"
  else
    output_dir="${LOCAL_OUTPUT_ROOT}"
  fi
  run_one_checkpoint "${checkpoint}" "${label}" "${output_dir}"
done

# Rank 0 uploads the complete local result tree to OBS.
if [ "${NODE_RANK}" -eq 0 ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_ROOT}', '${CLOUD_OUTPUT_DIR}')"
  echo "Inference results uploaded to ${CLOUD_OUTPUT_DIR}"
fi
