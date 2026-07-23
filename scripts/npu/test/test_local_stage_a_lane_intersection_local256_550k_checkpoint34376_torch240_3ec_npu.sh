#!/usr/bin/env bash
set -euo pipefail

# Fresh-asset local Ascend evaluation using the known-good 3ec535b inference
# implementation and the isolated torch/torch-npu 2.4.0 environment.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ENV_DIR}/activate_mllm_infer_torch240.sh

if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: environment activation script not found: ${ACTIVATE_SCRIPT}" >&2
  echo "Run scripts/npu/setup/create_mllm_infer_torch240_npu_env_from_mapgen.sh first." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:-checkpoint34376_torch240_3ec_$(date -u +%Y%m%d_%H%M%S)}
RUNTIME_ROOT=${RUNTIME_ROOT:-/cache/jn/fresh_assets/${RUN_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}

for target in "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"; do
  case "${target}" in
    /cache/jn/*) ;;
    *)
      echo "ERROR: refusing recursive cleanup outside /cache/jn: ${target}" >&2
      exit 2
      ;;
  esac
done
rm -rf "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"
mkdir -p "${RUNTIME_ROOT}" "${OUTPUT_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export MASTER_ADDR=127.0.0.1
unset MASTER_PORT

export RUN_ID
export OUTPUT_URL=${OUTPUT_URL:-${OUTPUT_ROOT}}
export LOCAL_OUTPUT_ROOT=${OUTPUT_ROOT}
export OBS_CACHE=${RUNTIME_ROOT}
export MODEL_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints
export VISION_TOWER=${RUNTIME_ROOT}/models/facebook_dinov2-large
export CHECKPOINT_OBS_LIST=obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/18/2260c16d83414dea8b663282962413ba/output/ma-job-bb9b7ed9-4bc2-4f55-a72a-25219f865069/checkpoint-34376/
export CHECKPOINT_DOWNLOAD_ROOT=${RUNTIME_ROOT}/checkpoints
export DATASET_OBS_PATH=obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256/local256.tar
export DATASET_ARCHIVE_PATH=${RUNTIME_ROOT}/dataset/local256.tar
export DATASET_EXTRACT_ROOT=${RUNTIME_ROOT}/dataset/local256_extract
export DATASET_DIR_NAME=local256
export TEST_JSON=${DATASET_EXTRACT_ROOT}/local256/phase_a/eval.jsonl
export UPLOAD_RESULTS=False
export REUSE_LOCAL_ASSETS=False

export DIFFICULTY_EVAL=True
export DIFFICULTY_SAMPLES_PER_BUCKET_SPEC=easy=300,medium=300,hard=300,very_hard=100
export DIFFICULTY_SPLIT_ROOT=${RUNTIME_ROOT}/difficulty_fixed1000
export DIFFICULTY_REUSE_SPLITS=False
export DIFFICULTY_REBUILD_SPLITS=True
export DIFFICULTY_INCLUDE_EMPTY=False
export DIFFICULTY_VIS_LIMIT=50
export DIFFICULTY_TOTAL_EVAL=True
export NUM_TEST_SAMPLES=0
export MAX_NEW_TOKENS=2048
export CHECKPOINT_DEEPSTACK_MODE=disabled

export INSTALL_DEPS=False
export ENABLE_MOXING_UPGRADE=False

echo "[torch240-infer] python: $(python -c 'import sys; print(sys.executable)')"
echo "[torch240-infer] runtime root: ${RUNTIME_ROOT}"
echo "[torch240-infer] output root: ${OUTPUT_ROOT}"
python - <<'PY'
import cv2
import filelock
import torch
import torch_npu
import torchvision
print(
    "[torch240-infer] versions: "
    f"filelock={filelock.__version__}, opencv={cv2.__version__}, "
    f"torch={torch.__version__}, torch_npu={torch_npu.__version__}, "
    f"torchvision={torchvision.__version__}, npu_count={torch.npu.device_count()}"
)
PY

exec bash "${SCRIPT_DIR}/test_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh"
