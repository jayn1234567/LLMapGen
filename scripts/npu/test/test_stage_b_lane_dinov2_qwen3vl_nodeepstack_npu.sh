#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU inference
# Fixed recipe: phase_b | lane-only centerline | dinov2 + Qwen3-VL-8B | no DeepStack
# This file is self-contained and does not call another project .sh file.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

DATASET_PHASE=phase_b
MAP_TASK=lane
VISION_BACKBONE=dinov2
VISION_TOWER_NAME=facebook_dinov2-large
MM_VISION_TOWER_TYPE=dinov2
INPUT_IMAGE_SIZE=518

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

# Inference writes local files first, then rank0 uploads the complete result dir.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}

CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/data_line_samples_33w}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
TEST_JSON=${TEST_JSON:-${DATASET_PATH}/${DATASET_PHASE}/test.jsonl}
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/checkpoints_${RUN_ID}}
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-${OBS_CACHE}/test_phase_b_lane_dinov2_output_${RUN_ID}}
CLOUD_OUTPUT_DIR=${TEST_RESULT_OBS:-${OSB_SHARE_PATH%/}/test_results_${RUN_ID}}

# ====================== inference params ======================
# CHECKPOINT_OBS_LIST or CHECKPOINT_DIRS can contain one or multiple checkpoints.
# NUM_TEST_SAMPLES=0 means run the full test jsonl.
# Patch json, visualization, metrics, and stitched maps are written locally first, then uploaded.
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
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

INSTALL_DEPS=${INSTALL_DEPS:-True}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}
VLLM_VERSION=${VLLM_VERSION:-0.9.2}
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
export RDZV_ID=${RDZV_ID:-test_phase_b_lane_dinov2_${RUN_ID}}
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}" "${CHECKPOINT_DOWNLOAD_ROOT}" "${LOCAL_OUTPUT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
echo "Run id: ${RUN_ID}"
echo "Local output root: ${LOCAL_OUTPUT_ROOT}"
echo "Cloud output dir: ${CLOUD_OUTPUT_DIR}"

CHECKPOINT_ITEMS=()
CHECKPOINT_LABELS=()
if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
  while IFS= read -r obs_item; do
    label=$(safe_label "${obs_item}")
    local_dir="${CHECKPOINT_DOWNLOAD_ROOT}/${label}"
    python -c "import moxing as mox; mox.file.copy_parallel('${obs_item}', '${local_dir}')"
    CHECKPOINT_INPUT_PATH="${local_dir}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'
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
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'
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
  echo "Infer ${checkpoint_label}: ${checkpoint_dir}"
INCLUDE_INTERSECTION_ARGS=()
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/tools/infer_centerline_state_update.py \
  --checkpoint-dir "${checkpoint_dir}" \
  --vision_tower "${VISION_TOWER}" \
  --input_image_size "${INPUT_IMAGE_SIZE}" \
  --disable_deepstack \
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

if [ "${NODE_RANK}" -eq 0 ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_ROOT}', '${CLOUD_OUTPUT_DIR}')"
  echo "Inference results uploaded to ${CLOUD_OUTPUT_DIR}"
fi
