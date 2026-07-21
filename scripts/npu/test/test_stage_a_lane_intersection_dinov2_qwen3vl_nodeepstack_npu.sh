#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU inference
# Fixed recipe: phase_a | lane + intersection | dinov2 + Qwen3-VL-8B | no DeepStack
# This file is self-contained and does not call another project .sh file.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative script and Python imports.
# Platform I/O and recipe metadata are declared first so cloud jobs can be audited quickly.
cd "${REPO_ROOT}"

# Recipe identity: fixed task, visual architecture, model family, and train variant.
DATASET_PHASE=phase_a                                                             # Dataset stage: phase_a for patch inference, phase_b for state update.
MAP_TASK=lane_intersection                                                        # Task type: lane or lane_intersection.
VISION_BACKBONE=dinov2                                                            # Visual backbone selector used by the generic multi-vision launcher.
# Vision assets for this recipe. Scripts only download the towers declared below.
VISION_TOWER_NAME=facebook_dinov2-large                                           # Single vision tower directory name under MODEL_OBS_PATH.
MM_VISION_TOWER_TYPE=dinov2                                                       # Model-side vision tower type: dinov2, dinov3, multi_moe, or multi_concat.
INPUT_IMAGE_SIZE=518                                                              # Image size fed to the vision encoder; DINOv3 recipes usually use 512.

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

# Inference writes local files first, then rank0 uploads the complete result dir.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id for local cache and cloud output folders.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local worker cache root for models, datasets, checkpoints, and outputs.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS directory that stores model and vision checkpoint assets.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}  # OBS zip path for the prepared UniMapGen dataset.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}                       # Dataset directory name expected after the zip is extracted.

CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}                                      # Comma, semicolon, or newline separated OBS checkpoint roots to evaluate.
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}                                              # Comma, semicolon, or newline separated local checkpoint roots to evaluate.
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}       # Vision tower path passed to the model loader. Multi-vision uses a comma list.
# Local dataset and output paths for this run.
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.archive}}  # Local archive path; zip/tar/tar.gz are detected by content.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Local directory where the dataset archive is extracted.
DATASET_PATH=${DATASET_PATH:-}                                                    # Optional extracted dataset root; auto-resolved when empty.
IMAGE_FOLDER=${IMAGE_FOLDER:-}                                                    # Optional image-root override; defaults to the resolved DATASET_PATH.
TEST_JSON=${TEST_JSON:-}                                                         # Optional test JSONL override; test/val/eval is auto-resolved when empty.
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/checkpoints_${RUN_ID}}  # Local root used to download checkpoint candidates from OBS.
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-${OBS_CACHE}/test_phase_a_lane_intersection_dinov2_output_${RUN_ID}}  # Per-run local inference output root.
CLOUD_OUTPUT_DIR=${TEST_RESULT_OBS:-${OSB_SHARE_PATH%/}/test_results_${RUN_ID}}   # Final cloud output directory for inference or GRPO results.
UPLOAD_RESULTS=${UPLOAD_RESULTS:-True}                                            # Set false for local Ascend evaluation that should keep results on disk only.
REUSE_LOCAL_ASSETS=${REUSE_LOCAL_ASSETS:-True}                                   # Reuse downloaded vision/data assets and extracted data when present.

# ====================== inference params ======================
# CHECKPOINT_OBS_LIST or CHECKPOINT_DIRS can contain one or multiple checkpoints.
# NUM_TEST_SAMPLES=0 means run the full test jsonl.
# Patch json, visualization, metrics, and stitched maps are written locally first, then uploaded.
# Main runtime parameters and hyperparameters.
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}                                           # Number of test samples to run; 0 means the full split.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}                                            # Maximum number of generated tokens per sample.
COORD_MODE=${COORD_MODE:-auto}                                                    # Coordinate mode: auto reads meta.coord_mode, or force norm1000 or pixel.
COORD_RANGE=${COORD_RANGE:-1000}                                                  # Coordinate range for normalized labels, normally 1000.
DIFFICULTY_EVAL=${DIFFICULTY_EVAL:-False}                                         # Split TEST_JSON into geometry-based difficulty buckets before inference.
DIFFICULTIES=${DIFFICULTIES:-easy,medium,hard,very_hard}                          # Difficulty buckets evaluated independently.
DIFFICULTY_SAMPLES_PER_BUCKET=${DIFFICULTY_SAMPLES_PER_BUCKET:-300}               # Balanced sample count per bucket; 0 evaluates every eligible sample.
DIFFICULTY_SAMPLES_PER_BUCKET_SPEC=${DIFFICULTY_SAMPLES_PER_BUCKET_SPEC:-easy=300,medium=300,hard=300,very_hard=100}  # Optional per-bucket sample counts.
DIFFICULTY_VIS_LIMIT=${DIFFICULTY_VIS_LIMIT:-50}                                  # Maximum patch comparisons rendered per bucket; 0 renders all.
DIFFICULTY_SEED=${DIFFICULTY_SEED:-42}                                            # Stable reservoir-sampling seed.
DIFFICULTY_INCLUDE_EMPTY=${DIFFICULTY_INCLUDE_EMPTY:-False}                       # Include empty patches in easy; false focuses metrics on road geometry.
DIFFICULTY_SPLIT_NAME=${DIFFICULTY_SPLIT_NAME:-${DATASET_DIR_NAME}_${DATASET_PHASE}_${MAP_TASK}_seed${DIFFICULTY_SEED}}  # Stable eval-set name used when DIFFICULTY_SPLIT_ROOT is not set.
DIFFICULTY_SPLIT_ROOT=${DIFFICULTY_SPLIT_ROOT:-${OBS_CACHE}/fixed_eval_splits/${DIFFICULTY_SPLIT_NAME}}  # Generated/reused per-difficulty JSONL files and manifest.
DIFFICULTY_REUSE_SPLITS=${DIFFICULTY_REUSE_SPLITS:-True}                          # Reuse existing JSONL files under DIFFICULTY_SPLIT_ROOT for comparable eval.
DIFFICULTY_REBUILD_SPLITS=${DIFFICULTY_REBUILD_SPLITS:-False}                     # Force rebuilding fixed eval JSONL files even if they already exist.
DIFFICULTY_TOTAL_EVAL=${DIFFICULTY_TOTAL_EVAL:-True}                              # Merge per-bucket predictions and compute one aggregate metric.
DIFFICULTY_TOTAL_LABEL=${DIFFICULTY_TOTAL_LABEL:-all_selected}                    # Output folder name for the aggregate selected-sample metrics.
EVAL_METER_PER_PIXEL=${EVAL_METER_PER_PIXEL:-0.2}                                 # Jiangjihua line-eval meter-per-pixel setting.
EVAL_BUFFER_SIZE=${EVAL_BUFFER_SIZE:-1.0}                                         # Jiangjihua line-eval buffer size.
EVAL_MATCH_THRESHOLD=${EVAL_MATCH_THRESHOLD:-0.33}                                # Jiangjihua line-eval matching threshold.
CHECKPOINT_DEEPSTACK_MODE=${CHECKPOINT_DEEPSTACK_MODE:-disabled}                  # disabled preserves this recipe; auto trusts checkpoint config.
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

fi
# Helper functions for parsing checkpoint lists and resolving best/direct checkpoint roots.
read_list() {
  python - "$1" <<'PY'
import re
import sys

for item in re.split("[,;" + chr(10) + "]+", sys.argv[1] or ""):
    item = item.strip()
    if item:
        print(item)
PY
}

safe_label() {
  python - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip().rstrip("/") or "checkpoint"
label = re.sub(r"[^A-Za-z0-9._-]+", "_", value.split("/")[-1]).strip("._-")
print(label or "checkpoint")
PY
}
# ====================== distributed setup ======================
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
if [ "${NNODES}" -eq 1 ] && [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind(("127.0.0.1", 0))
    print(server.getsockname()[1])
PY
)
  echo "Auto-selected free rendezvous port: ${MASTER_PORT}"
else
  MASTER_PORT=${MASTER_PORT:-6060}                                                # Shared multi-node rendezvous port.
fi
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-test_phase_a_lane_intersection_dinov2_${RUN_ID}}        # Unique rendezvous id for this distributed run.
# Download recipe-specific assets and the dataset, then verify required local paths.
if [[ "${REUSE_LOCAL_ASSETS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && [ -f "${VISION_TOWER}/config.json" ]; then
  echo "[assets] reusing vision tower: ${VISION_TOWER}"
else
  python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
fi
if [[ "${REUSE_LOCAL_ASSETS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && [ -s "${DATASET_ARCHIVE_PATH}" ]; then
  echo "[assets] reusing dataset archive: ${DATASET_ARCHIVE_PATH}"
else
  mkdir -p "$(dirname "${DATASET_ARCHIVE_PATH}")"
  python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ARCHIVE_PATH}')"
fi
mkdir -p "${DATASET_EXTRACT_ROOT}" "${CHECKPOINT_DOWNLOAD_ROOT}" "${LOCAL_OUTPUT_ROOT}"
if [[ "${REUSE_LOCAL_ASSETS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && \
   [ -n "$(find "${DATASET_EXTRACT_ROOT}" -type f \( -name test.jsonl -o -name val.jsonl \) -print -quit 2>/dev/null)" ]; then
  echo "[assets] reusing extracted dataset: ${DATASET_EXTRACT_ROOT}"
else
  python - "${DATASET_ARCHIVE_PATH}" "${DATASET_EXTRACT_ROOT}" <<'PY'
import sys
import tarfile
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
if zipfile.is_zipfile(archive):
    print(f"[dataset] extracting zip: {archive} -> {output}", flush=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output)
elif tarfile.is_tarfile(archive):
    print(f"[dataset] extracting tar: {archive} -> {output}", flush=True)
    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(output)
else:
    raise SystemExit(f"Unsupported dataset archive (expected zip/tar/tar.gz): {archive}")
PY
fi

if [ -z "${DATASET_PATH}" ]; then
  DATASET_PATH=$(python - "${DATASET_EXTRACT_ROOT}" "${DATASET_DIR_NAME}" "${DATASET_PHASE}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
preferred = str(sys.argv[2]).strip()
phase = str(sys.argv[3]).strip()
candidates = []
if preferred:
    candidates.append(extract_root / preferred)
candidates.append(extract_root)
candidates.extend(path.parent.parent for path in extract_root.rglob(f"{phase}/test.jsonl"))
candidates.extend(path.parent for path in extract_root.rglob("test.jsonl"))
candidates.extend(path.parent for path in extract_root.rglob("val.jsonl"))

seen = set()
for candidate in candidates:
    candidate = candidate.resolve()
    if candidate in seen:
        continue
    seen.add(candidate)
    if (candidate / phase / "test.jsonl").is_file() or (candidate / "test.jsonl").is_file():
        print(candidate)
        raise SystemExit(0)
    if (candidate / phase / "val.jsonl").is_file() or (candidate / "val.jsonl").is_file():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(f"Unable to resolve a dataset root below {extract_root}")
PY
  )
fi
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
if [ -z "${TEST_JSON}" ]; then
  for candidate in \
    "${DATASET_PATH}/${DATASET_PHASE}/test.jsonl" \
    "${DATASET_PATH}/test.jsonl" \
    "${DATASET_PATH}/${DATASET_PHASE}/val.jsonl" \
    "${DATASET_PATH}/val.jsonl" \
    "${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl" \
    "${DATASET_PATH}/eval.jsonl"; do
    if [ -f "${candidate}" ]; then
      TEST_JSON="${candidate}"
      break
    fi
  done
fi
echo "Run id: ${RUN_ID}"
echo "Dataset archive: ${DATASET_ARCHIVE_PATH}"
echo "Dataset root: ${DATASET_PATH}"
echo "Test JSONL: ${TEST_JSON}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Local output root: ${LOCAL_OUTPUT_ROOT}"
echo "Cloud output dir: ${CLOUD_OUTPUT_DIR}"

# Build the checkpoint evaluation list from OBS roots or local directories.
CHECKPOINT_ITEMS=()                                                               # Resolved checkpoint paths that will be evaluated.
CHECKPOINT_LABELS=()                                                              # Display labels paired with CHECKPOINT_ITEMS.
if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
  while IFS= read -r obs_item; do
    label=$(safe_label "${obs_item}")
    local_dir="${CHECKPOINT_DOWNLOAD_ROOT}/${label}"
    python -c "import moxing as mox; mox.file.copy_parallel('${obs_item}', '${local_dir}')"
    CHECKPOINT_INPUT_PATH="${local_dir}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'                  # Checkpoint directory selected by the resolver.
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
    CHECKPOINT_ITEMS+=("${RESOLVED_CHECKPOINT}")
    CHECKPOINT_LABELS+=("${label}")
  done < <(read_list "${CHECKPOINT_OBS_LIST}")
elif [ -n "${CHECKPOINT_DIRS}" ]; then
  while IFS= read -r local_item; do
    CHECKPOINT_INPUT_PATH="${local_item}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'                  # Checkpoint directory selected by the resolver.
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
    CHECKPOINT_ITEMS+=("${RESOLVED_CHECKPOINT}")
    CHECKPOINT_LABELS+=("$(safe_label "${local_item}")")
  done < <(read_list "${CHECKPOINT_DIRS}")
else
  echo "ERROR: set CHECKPOINT_OBS_LIST or CHECKPOINT_DIRS for inference."
  exit 1
fi

for path in "${VISION_TOWER}" "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

# Build deterministic balanced evaluation files. Difficulty scoring is always
# performed on a normalized 0..1000 geometry grid, including legacy pixel GT.
TEST_JSON_ITEMS=()
TEST_JSON_LABELS=()
if [[ "${DIFFICULTY_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  difficulty_args=(
    --input-jsonl "${TEST_JSON}"
    --output-dir "${DIFFICULTY_SPLIT_ROOT}"
    --samples-per-difficulty "${DIFFICULTY_SAMPLES_PER_BUCKET}"
    --seed "${DIFFICULTY_SEED}"
    --coord-mode "${COORD_MODE}"
    --coord-range "${COORD_RANGE}"
  )
  if [ -n "${DIFFICULTY_SAMPLES_PER_BUCKET_SPEC}" ]; then
    difficulty_args+=(--samples-per-difficulty-spec "${DIFFICULTY_SAMPLES_PER_BUCKET_SPEC}")
  fi
  if [[ "${DIFFICULTY_INCLUDE_EMPTY}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    difficulty_args+=(--include-empty)
  fi
  requested_difficulties=()
  while IFS= read -r difficulty; do
    requested_difficulties+=("${difficulty}")
  done < <(read_list "${DIFFICULTIES}")
  reuse_existing_splits=0
  if [[ "${DIFFICULTY_REUSE_SPLITS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && \
     ! [[ "${DIFFICULTY_REBUILD_SPLITS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    reuse_existing_splits=1
    for difficulty in "${requested_difficulties[@]}"; do
      split_json="${DIFFICULTY_SPLIT_ROOT}/${difficulty}.jsonl"
      if [ ! -s "${split_json}" ]; then
        reuse_existing_splits=0
        break
      fi
    done
  fi
  if [ "${reuse_existing_splits}" -eq 1 ]; then
    echo "[difficulty-splits] reusing fixed evaluation splits from ${DIFFICULTY_SPLIT_ROOT}"
  else
    difficulty_args+=(--difficulties "${requested_difficulties[@]}")
    if ! python scripts/tools/build_difficulty_eval_splits.py "${difficulty_args[@]}"; then
      echo "ERROR: failed to build difficulty evaluation splits from ${TEST_JSON}."
      exit 1
    fi
  fi
  for difficulty in "${requested_difficulties[@]}"; do
    split_json="${DIFFICULTY_SPLIT_ROOT}/${difficulty}.jsonl"
    if [ ! -s "${split_json}" ]; then
      echo "WARNING: skip empty difficulty split: ${split_json}"
      continue
    fi
    TEST_JSON_ITEMS+=("${split_json}")
    TEST_JSON_LABELS+=("${difficulty}")
  done
else
  TEST_JSON_ITEMS+=("${TEST_JSON}")
  TEST_JSON_LABELS+=("all")
fi

if [ "${#TEST_JSON_ITEMS[@]}" -eq 0 ]; then
  echo "ERROR: no non-empty evaluation JSONL was produced."
  exit 1
fi

# Run one checkpoint through inference, evaluation, visualization, and table printing.
run_one_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_label="$2"
  local output_dir="$3"
  local test_json="$4"
  local difficulty_label="$5"
  local json_dir="${output_dir}/json"
  local patch_viz_dir="${output_dir}/viz"
  local whole_map_viz_dir="${output_dir}/whole_map_viz"
  local summary_json="${output_dir}/summary.json"
  local merged_global_json="${output_dir}/merged_global.json"
  local eval_json="${output_dir}/eval.json"
  mkdir -p "${json_dir}" "${patch_viz_dir}" "${whole_map_viz_dir}"
  echo "Infer ${checkpoint_label}: ${checkpoint_dir}"
  deepstack_args=()
  case "${CHECKPOINT_DEEPSTACK_MODE}" in
    disabled|disable|off|false|False|FALSE|0)
      deepstack_args+=(--disable_deepstack)
      ;;
    auto|checkpoint)
      ;;
    *)
      echo "ERROR: unsupported CHECKPOINT_DEEPSTACK_MODE=${CHECKPOINT_DEEPSTACK_MODE}; use disabled or auto."
      return 1
      ;;
  esac
# Launch the recipe entrypoint. Training uses HCCL/DDP and full SFT may add DeepSpeed.
if ! torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/tools/infer_centerline_checkpoint.py \
    --checkpoint-dir "${checkpoint_dir}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
    --input_image_size "${INPUT_IMAGE_SIZE}" \
    "${deepstack_args[@]}" \
    --test-json "${test_json}" \
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
    --eval-meter-per-pixel "${EVAL_METER_PER_PIXEL}" \
    --eval-buffer-size "${EVAL_BUFFER_SIZE}" \
    --eval-match-threshold "${EVAL_MATCH_THRESHOLD}" \
    --eval-centerline \
    --eval-output-json "${eval_json}"; then
    echo "ERROR: inference failed for ${checkpoint_label}/${difficulty_label}."
    return 1
  fi
  if [ "${NODE_RANK}" -ne 0 ]; then
    return 0
  fi
  if [ ! -s "${summary_json}" ]; then
    echo "ERROR: inference completed without a non-empty summary: ${summary_json}"
    return 1
  fi
  visualize_args=(
      --input-dir "${output_dir}"
      --image-folder "${IMAGE_FOLDER}"
      --output-dir "${patch_viz_dir}"
      --map-task "${MAP_TASK}"
      --eval-output-json "${eval_json}"
      --eval-meter-per-pixel "${EVAL_METER_PER_PIXEL}"
      --eval-buffer-size "${EVAL_BUFFER_SIZE}"
      --eval-match-threshold "${EVAL_MATCH_THRESHOLD}"
  )
  if [[ "${DIFFICULTY_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    visualize_args+=(--max-samples "${DIFFICULTY_VIS_LIMIT}" --no-eval-centerline --skip-whole-map-viz)
  else
    visualize_args+=(--max-samples 0 --whole-map-viz-dir "${whole_map_viz_dir}")
  fi
  python scripts/tools/visualize_centerline.py "${visualize_args[@]}"
  if [ -f "${eval_json}" ]; then
    python - "${eval_json}" <<'PY'
import json
import sys
from pathlib import Path
from infer_index.line_eval import format_eval_table
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
payload = payload.get('summary', payload) if isinstance(payload, dict) else payload
print(payload.get('table') if isinstance(payload, dict) and payload.get('table') else format_eval_table(payload))
PY
  fi
}

write_aggregate_difficulty_eval() {
  local checkpoint_label="$1"
  local checkpoint_root="${LOCAL_OUTPUT_ROOT}/${checkpoint_label}"
  local aggregate_dir="${checkpoint_root}/${DIFFICULTY_TOTAL_LABEL}"
  mkdir -p "${aggregate_dir}"
  python - \
    "${checkpoint_root}" \
    "${aggregate_dir}" \
    "${DIFFICULTIES}" \
    "${EVAL_METER_PER_PIXEL}" \
    "${EVAL_BUFFER_SIZE}" \
    "${EVAL_MATCH_THRESHOLD}" <<'PY'
import json
import re
import sys
from pathlib import Path

repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from infer_index.line_eval import evaluate_lane_intersection_records, print_lane_intersection_eval_tables


def read_list(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;\n]+", text or "") if item.strip()]


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("patch_results", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported summary payload: {path}")


checkpoint_root = Path(sys.argv[1])
aggregate_dir = Path(sys.argv[2])
difficulties = read_list(sys.argv[3])
meter_per_pixel = float(sys.argv[4])
buffer_size = float(sys.argv[5])
match_threshold = float(sys.argv[6])

records = []
source_counts = {}
source_files = []
for difficulty in difficulties:
    summary_path = checkpoint_root / difficulty / "summary.json"
    if not summary_path.is_file():
        print(f"[aggregate-eval] skip missing split summary: {summary_path}", flush=True)
        continue
    split_records = load_records(summary_path)
    for record in split_records:
        record.setdefault("difficulty_eval_bucket", difficulty)
    records.extend(split_records)
    source_counts[difficulty] = len(split_records)
    source_files.append(str(summary_path))

if not records:
    raise SystemExit(f"No per-difficulty inference summaries found under {checkpoint_root}")

summary_path = aggregate_dir / "summary.json"
summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

map_eval = evaluate_lane_intersection_records(
    records,
    meter_per_pixel=meter_per_pixel,
    buffer_size=buffer_size,
    match_threshold=match_threshold,
)
eval_summary = {
    "centerline_eval": map_eval["lane"],
    "intersection_eval": map_eval["intersection"],
    "lane_intersection_eval": map_eval["lane_intersection"],
    "map_eval": map_eval,
    "aggregate": {
        "source_counts": source_counts,
        "num_records": len(records),
        "source_files": source_files,
    },
}
eval_path = aggregate_dir / "eval.json"
eval_path.write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[aggregate-eval] wrote {len(records)} records -> {summary_path}", flush=True)
print(f"[aggregate-eval] eval -> {eval_path}", flush=True)
print_lane_intersection_eval_tables(eval_summary["map_eval"])
print(json.dumps({"aggregate_eval_json": str(eval_path), "aggregate_eval": eval_summary}, ensure_ascii=False), flush=True)
PY
}

# Evaluate each requested checkpoint, separating output folders when multiple are provided.
for index in "${!CHECKPOINT_ITEMS[@]}"; do
  label="${CHECKPOINT_LABELS[$index]}"
  checkpoint="${CHECKPOINT_ITEMS[$index]}"
  for test_index in "${!TEST_JSON_ITEMS[@]}"; do
    test_json="${TEST_JSON_ITEMS[$test_index]}"
    test_label="${TEST_JSON_LABELS[$test_index]}"
    if [[ "${DIFFICULTY_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
      output_dir="${LOCAL_OUTPUT_ROOT}/${label}/${test_label}"
    elif [ "${#CHECKPOINT_ITEMS[@]}" -gt 1 ]; then
      output_dir="${LOCAL_OUTPUT_ROOT}/${index}_${label}"
    else
      output_dir="${LOCAL_OUTPUT_ROOT}"
    fi
    if ! run_one_checkpoint "${checkpoint}" "${label}/${test_label}" "${output_dir}" "${test_json}" "${test_label}"; then
      echo "ERROR: evaluation aborted at checkpoint=${label}, split=${test_label}."
      exit 1
    fi
  done
  if [[ "${DIFFICULTY_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && \
     [[ "${DIFFICULTY_TOTAL_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    write_aggregate_difficulty_eval "${label}"
  fi
done

# Rank 0 uploads the complete local result tree to OBS.
if [ "${NODE_RANK}" -eq 0 ] && [[ "${UPLOAD_RESULTS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_ROOT}', '${CLOUD_OUTPUT_DIR}')"
  echo "Inference results uploaded to ${CLOUD_OUTPUT_DIR}"
elif [ "${NODE_RANK}" -eq 0 ]; then
  echo "Inference results kept locally at ${LOCAL_OUTPUT_ROOT}"
fi
