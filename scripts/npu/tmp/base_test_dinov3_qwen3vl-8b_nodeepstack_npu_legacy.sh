#!/usr/bin/env bash
# set -euo pipefail

# Standalone NPU inference script for a fixed stage/task/vision recipe.
#
# Cloud mode:
#   When OUTPUT_URL is set by the NPU platform, this script installs the same
#   runtime deps as the NPU training scripts, downloads dataset / DINO / trained
#   checkpoints from OBS, runs inference, prints the line-eval table, then uploads
#   outputs to OUTPUT_URL/test_results_${RUN_ID}.
#
# Local mode:
#   When OUTPUT_URL is absent, this script uses existing local paths:
#   DATASET_PATH, IMAGE_FOLDER, VISION_TOWER and CHECKPOINT_DIRS/CHECKPOINT_DIR/TRAIN_OUTPUT_DIR.
#
# Checkpoint selection:
#   CHECKPOINT_DIRS: comma/semicolon/newline separated local checkpoint dirs.
#   CHECKPOINT_OBS_LIST: comma/semicolon/newline separated full OBS checkpoint dirs.
#   TRAINED_CHECKPOINT_OBS + CHECKPOINT_NAMES: one OBS training output root plus
#     relative checkpoint dirs, for example:
#       CHECKPOINT_NAMES=checkpoint-500,eval_best_candidates,best_candidates,merged
#     eval_best_candidates/best_candidates are resolved locally to the latest
#     successful candidate with _SUCCESS.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

# ====================== standalone default selector ======================
VISION_BACKBONE=${VISION_BACKBONE:-dinov3}  # dinov2 or dinov3.
DATASET_PHASE=${DATASET_PHASE:-phase_a}     # phase_a or phase_b.
MAP_TASK=${MAP_TASK:-lane}                  # lane or lane_intersection.
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-512}

# ====================== editable cloud/test defaults ======================
# Dataset OBS zip and the directory name after unzip.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}

# DINO/Qwen base model OBS root. The selected DINO folder is downloaded from this root.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}

# Single or multiple weights. Use one pattern:
#   CHECKPOINT_OBS_LIST=obs://.../checkpoint-500,obs://.../eval_best_candidates
#   TRAINED_CHECKPOINT_OBS=obs://.../train_output CHECKPOINT_NAMES=checkpoint-500,eval_best_candidates,best_candidates,infer_best_candidates,merged
#   CHECKPOINT_DIRS=/cache/.../checkpoint-500,/cache/.../eval_best_candidates
CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}
TRAINED_CHECKPOINT_OBS=${TRAINED_CHECKPOINT_OBS:-}
CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-}
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}

# Output and sampling. NUM_TEST_SAMPLES=0 means all test rows.
OUTPUT_DIR=${OUTPUT_DIR:-}
TEST_RESULT_OBS=${TEST_RESULT_OBS:-}
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
INSTALL_DEPS=${INSTALL_DEPS:-}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-}
# ==========================================================================

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

read_list() {
  python - "$1" <<'PY'
import re
import sys

value = sys.argv[1]
for item in re.split(r"[,;\n]+", value or ""):
    item = item.strip()
    if item:
        print(item)
PY
}

safe_label() {
  python - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip().rstrip("/")
if not value:
    value = "checkpoint"
label = value.split("/")[-1]
label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")
print(label or "checkpoint")
PY
}

resolve_checkpoint_dir() {
  local root="$1"
  if [ -f "${root}/model.safetensors" ] || [ -f "${root}/pytorch_model.bin" ]; then
    echo "${root}"
    return 0
  fi

  local candidate_root_resolved
  candidate_root_resolved=$(python - "${root}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
if not root.is_dir() or not root.name.endswith("_candidates"):
    raise SystemExit(0)

def step_from_name(name: str) -> int:
    marker = "_step-"
    if marker not in name:
        return -1
    tail = name.split(marker, 1)[1]
    digits = []
    for char in tail:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else -1

def load_step(path: Path) -> int:
    for name in ("best_infer_index.json", "best_eval_loss.json", "best_train_loss.json", "best_reward.json"):
        metadata_path = path / name
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        for key in ("best_infer_index_step", "best_eval_loss_step", "best_train_loss_step", "best_reward_step", "global_step"):
            if key in metadata:
                try:
                    return int(metadata[key])
                except Exception:
                    pass
    return step_from_name(path.name)

candidates = []
for path in root.iterdir():
    if path.is_dir() and (path / "_SUCCESS").is_file():
        candidates.append((load_step(path), path))
if candidates:
    print(sorted(candidates, key=lambda item: (item[0], str(item[1])))[-1][1])
PY
)
  if [ -n "${candidate_root_resolved}" ]; then
    echo "${candidate_root_resolved}"
    return 0
  fi

  local resolved
  if resolved=$(python scripts/tools/resolve_best_checkpoint.py \
      --output-dir "${root}" \
      --best-name infer_best \
      --best-name eval_best \
      --best-name best \
      --best-name best_reward \
      --allow-direct 2>/dev/null); then
    echo "${resolved}"
    return 0
  fi

  resolved=$(python - "${root}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = []
for path in root.glob("checkpoint-*"):
    if path.is_dir():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        candidates.append((step, path))
if candidates:
    print(sorted(candidates)[-1][1])
PY
)
  if [ -n "${resolved}" ]; then
    echo "${resolved}"
    return 0
  fi

  echo "${root}"
}

CHECKPOINT_ITEMS=()
CHECKPOINT_LABELS=()
add_checkpoint_item() {
  local checkpoint_root="$1"
  local raw_label="$2"
  local resolved
  resolved=$(resolve_checkpoint_dir "${checkpoint_root}")
  CHECKPOINT_ITEMS+=("${resolved}")
  CHECKPOINT_LABELS+=("$(safe_label "${raw_label:-${resolved}}")")
}

download_checkpoint_from_obs() {
  local obs_path="$1"
  local raw_label="$2"
  local label
  label=$(safe_label "${raw_label:-${obs_path}}")
  DOWNLOADED_CHECKPOINT_DIR="${CHECKPOINT_DOWNLOAD_ROOT}/${label}"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading checkpoint ${obs_path} -> ${DOWNLOADED_CHECKPOINT_DIR} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy_parallel('${obs_path}', '${DOWNLOADED_CHECKPOINT_DIR}')"
}

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

# ====================== NPU environment ======================
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ====================== dependencies ======================
INSTALL_DEPS=${INSTALL_DEPS:-${CLOUD_MODE}}
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-${CLOUD_MODE}}
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
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing dependencies >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

  pip install torch==2.7.1
  pip install torch_npu==2.7.1rc1
  python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
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
  pip install 'gradio>=5.0'
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

  echo "========== key deps =========="
  python -c "import torch; print('torch', torch.__version__)"
  python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
  python -c "import transformers; print('transformers', transformers.__version__)"
  python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
  echo "==============================="
fi

# ====================== distributed parameters ======================
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
export RDZV_ID=${RDZV_ID:-infer_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_${RUN_ID}}

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
export OMP_NUM_THREADS=1
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export MLLM_SUPPRESS_NONZERO_STDERR=${MLLM_SUPPRESS_NONZERO_STDERR:-0}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== paths and downloads ======================
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_line_samples_33w}
TRAINED_CHECKPOINT_OBS=${TRAINED_CHECKPOINT_OBS:-${CHECKPOINT_OBS:-}}
CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-}
CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/checkpoints_${RUN_ID}}

case "${VISION_BACKBONE}" in
  dinov2)
    VISION_TOWER_NAME=${VISION_TOWER_NAME:-facebook_dinov2-large}
    VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}
    INPUT_IMAGE_SIZE_ARGS=()
    ;;
  dinov3)
    VISION_TOWER_NAME=${VISION_TOWER_NAME:-facebook_dinov3-vitl16-pretrain-lvd1689m}
    VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/${VISION_TOWER_NAME}}
    INPUT_IMAGE_SIZE_ARGS=(--input_image_size "${INPUT_IMAGE_SIZE:-512}")
    ;;
esac

if [[ "${CLOUD_MODE}" == "True" ]]; then
  DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}
  DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}
  DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/data_line_samples_33w}
  IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}

  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading vision tower >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"

  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
  mkdir -p "${DATASET_EXTRACT_ROOT}"
  unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
  if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: expected dataset directory not found: ${DATASET_PATH}"
    echo "Dataset extract root contents:"
    ls -la "${DATASET_EXTRACT_ROOT}"
    exit 1
  fi

  if [ -z "${CHECKPOINT_DIRS}" ] && [ -z "${CHECKPOINT_DIR:-}" ]; then
    mkdir -p "${CHECKPOINT_DOWNLOAD_ROOT}"
    if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
      while IFS= read -r obs_item; do
        [ -n "${obs_item}" ] || continue
        download_checkpoint_from_obs "${obs_item}" "${obs_item}"
        add_checkpoint_item "${DOWNLOADED_CHECKPOINT_DIR}" "${obs_item}"
      done < <(read_list "${CHECKPOINT_OBS_LIST}")
    elif [ -n "${TRAINED_CHECKPOINT_OBS}" ]; then
      if [ -n "${CHECKPOINT_NAMES}" ]; then
        while IFS= read -r checkpoint_name; do
          [ -n "${checkpoint_name}" ] || continue
          case "${checkpoint_name}" in
            .|./)
              download_checkpoint_from_obs "${TRAINED_CHECKPOINT_OBS}" "train_output"
              ;;
            obs://*)
              download_checkpoint_from_obs "${checkpoint_name}" "${checkpoint_name}"
              ;;
            *)
              download_checkpoint_from_obs "${TRAINED_CHECKPOINT_OBS%/}/${checkpoint_name}" "${checkpoint_name}"
              ;;
          esac
          add_checkpoint_item "${DOWNLOADED_CHECKPOINT_DIR}" "${checkpoint_name}"
        done < <(read_list "${CHECKPOINT_NAMES}")
      else
        download_checkpoint_from_obs "${TRAINED_CHECKPOINT_OBS}" "${TRAINED_CHECKPOINT_OBS}"
        CHECKPOINT_DIR="${DOWNLOADED_CHECKPOINT_DIR}"
      fi
    else
      echo "ERROR: set CHECKPOINT_OBS_LIST, or TRAINED_CHECKPOINT_OBS/CHECKPOINT_OBS, or CHECKPOINT_DIRS/CHECKPOINT_DIR."
      exit 1
    fi
  fi
else
  DATASET_PATH=${DATASET_PATH:-/cache/unimapgen_v2/dataset}
  IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}
fi

TEST_JSON=${TEST_JSON:-}
if [ -z "${TEST_JSON}" ]; then
  if [ -f "${DATASET_PATH}/${DATASET_PHASE}/test.jsonl" ]; then
    TEST_JSON="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl"
  else
    TEST_JSON="${DATASET_PATH}/test.jsonl"
  fi
fi

if [ "${#CHECKPOINT_ITEMS[@]}" -eq 0 ]; then
  if [ -n "${CHECKPOINT_DIRS}" ]; then
    while IFS= read -r checkpoint_dir_item; do
      [ -n "${checkpoint_dir_item}" ] || continue
      add_checkpoint_item "${checkpoint_dir_item}" "${checkpoint_dir_item}"
    done < <(read_list "${CHECKPOINT_DIRS}")
  elif [ -n "${CHECKPOINT_DIR:-}" ]; then
    add_checkpoint_item "${CHECKPOINT_DIR}" "${CHECKPOINT_DIR}"
  fi
fi

if [ "${#CHECKPOINT_ITEMS[@]}" -eq 0 ]; then
  TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/cache/unimapgen_v2/train_output/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
  BEST_CHECKPOINT_NAME=${BEST_CHECKPOINT_NAME:-infer_best}
  CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --best-name "${BEST_CHECKPOINT_NAME}" \
    --best-name eval_best \
    --best-name best \
    --allow-direct)
  add_checkpoint_item "${CHECKPOINT_DIR}" "${BEST_CHECKPOINT_NAME}"
fi

if [ "${#CHECKPOINT_ITEMS[@]}" -eq 0 ]; then
  echo "ERROR: no checkpoint directories resolved."
  exit 1
fi

for path in "${VISION_TOWER}" "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
    exit 1
  fi
done
for checkpoint_path in "${CHECKPOINT_ITEMS[@]}"; do
  if [ ! -e "${checkpoint_path}" ]; then
    echo "ERROR: checkpoint path missing: ${checkpoint_path}"
    exit 1
  fi
done

NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
if [[ "${CLOUD_MODE}" == "True" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-${OBS_CACHE}/test_output_${RUN_ID}}
else
  OUTPUT_DIR=${OUTPUT_DIR:-/cache/unimapgen_v2/infer_output/${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack/${RUN_ID}}
fi

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
  mkdir -p "${output_dir}" "${json_dir}" "${patch_viz_dir}" "${whole_map_viz_dir}"

  echo "============================================================"
  echo "CHECKPOINT_LABEL=${checkpoint_label}"
  echo "CHECKPOINT_DIR=${checkpoint_dir}"
  echo "VISION_TOWER=${VISION_TOWER}"
  echo "DATASET_PHASE=${DATASET_PHASE}"
  echo "MAP_TASK=${MAP_TASK}"
  echo "TEST_JSON=${TEST_JSON}"
  echo "IMAGE_FOLDER=${IMAGE_FOLDER}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "COORD_MODE=${COORD_MODE} COORD_RANGE=${COORD_RANGE}"
  echo "NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES}"
  echo "============================================================"

  if [ "${DATASET_PHASE}" = "phase_b" ]; then
    INCLUDE_INTERSECTION_ARGS=()
    if [ "${MAP_TASK}" = "lane_intersection" ]; then
      INCLUDE_INTERSECTION_ARGS=(--include-intersections)
    fi
    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      scripts/tools/infer_centerline_state_update.py \
      --checkpoint-dir "${checkpoint_dir}" \
      --vision_tower "${VISION_TOWER}" \
      "${INPUT_IMAGE_SIZE_ARGS[@]}" \
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
  else
    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      scripts/tools/infer_centerline_checkpoint.py \
      --checkpoint-dir "${checkpoint_dir}" \
      --vision_tower "${VISION_TOWER}" \
      "${INPUT_IMAGE_SIZE_ARGS[@]}" \
      --disable_deepstack \
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

    if [ "${NODE_RANK}" -eq 0 ]; then
      python - "${output_dir}" "${summary_json}" <<'PY'
import glob
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
rank_files = sorted(glob.glob(str(output_dir / "summary_rank*.json")))
if not rank_files:
    raise SystemExit(0)
merged = []
for path in rank_files:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        continue
    try:
        payload = json.loads(text)
        merged.extend(payload if isinstance(payload, list) else [payload])
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                merged.append(json.loads(line))
merged.sort(key=lambda item: item.get("idx", item.get("record_id", "")))
summary_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Merged {len(rank_files)} rank summaries into {summary_json}, records={len(merged)}")
PY
    fi
  fi

  if [ "${NODE_RANK}" -ne 0 ]; then
    echo "Skip visualization/upload for ${checkpoint_label} on non-master node ${NODE_RANK}"
    return 0
  fi

  python scripts/tools/visualize_centerline.py \
    --input-dir "${output_dir}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${patch_viz_dir}" \
    --eval-output-json "${eval_json}" \
    --whole-map-viz-dir "${whole_map_viz_dir}"

  if [ -f "${eval_json}" ]; then
    echo "================ Final Centerline Eval Table: ${checkpoint_label} ================"
    python - "${eval_json}" <<'PY'
import json
import sys
from pathlib import Path
from infer_index.line_eval import format_eval_table

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
    payload = payload["summary"]
table = payload.get("table") if isinstance(payload, dict) else None
print(table if table else format_eval_table(payload))
PY
    echo "============================================================="
  fi

  echo "Inference outputs for ${checkpoint_label}:"
  echo "  summary:         ${summary_json}"
  echo "  sample json dir: ${json_dir}"
  echo "  patch viz dir:   ${patch_viz_dir}"
  echo "  eval json:       ${eval_json}"
  echo "  whole map dir:   ${whole_map_viz_dir}"
}

echo "Resolved checkpoints:"
for index in "${!CHECKPOINT_ITEMS[@]}"; do
  echo "  [$index] ${CHECKPOINT_LABELS[$index]} -> ${CHECKPOINT_ITEMS[$index]}"
done

CHECKPOINT_COUNT=${  # CHECKPOINT_ITEMS[@]}
for index in "${!CHECKPOINT_ITEMS[@]}"; do
  current_label="${CHECKPOINT_LABELS[$index]}"
  current_checkpoint="${CHECKPOINT_ITEMS[$index]}"
  if [ "${CHECKPOINT_COUNT}" -gt 1 ]; then
    current_output_dir="${OUTPUT_DIR}/${index}_${current_label}"
  else
    current_output_dir="${OUTPUT_DIR}"
  fi
  run_one_checkpoint "${current_checkpoint}" "${current_label}" "${current_output_dir}"
done

if [ "${NODE_RANK}" -ne 0 ]; then
  echo "Skip final upload on non-master node ${NODE_RANK}"
  exit 0
fi

if [[ "${CLOUD_MODE}" == "True" ]]; then
  TEST_RESULT_OBS=${TEST_RESULT_OBS:-${OUTPUT_URL%/}/test_results_${RUN_ID}}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Uploading results to ${TEST_RESULT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy_parallel('${OUTPUT_DIR}', '${TEST_RESULT_OBS}')"
  echo "Results saved to ${TEST_RESULT_OBS}"
fi
