#!/usr/bin/env bash
# set -euo pipefail

# DI one-node/eight-NPU evaluation wrapper for the known-good 3ec535b
# DINOv2 + Qwen3-VL inference path. All runtime assets are downloaded from OBS.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

echo "[di-entry] reached ${SCRIPT_PATH}"
echo "[di-entry] repo root: ${REPO_ROOT}"
echo "[di-entry] UTC time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -z "${OUTPUT_URL:-}" ]; then
  echo "ERROR: OUTPUT_URL was not injected by DI."
  exit 1
fi

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

PYTHON=${PYTHON:-python}
RUN_ID=${RUN_ID:-checkpoint34376_3ec_di_$(date -u +%Y%m%d_%H%M%S)}
RUNTIME_ROOT=${RUNTIME_ROOT:-/cache/llmapgen_eval_${RUN_ID}}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-/cache/llmapgen_eval_bootstrap}

TORCH_NPU_WHL_OBS=${TORCH_NPU_WHL_OBS:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
MOXING_WHL_OBS=${MOXING_WHL_OBS:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}
TORCH_NPU_WHL_LOCAL=${BOOTSTRAP_ROOT}/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl
MOXING_WHL_LOCAL=${BOOTSTRAP_ROOT}/moxing_framework-2.3.8-py2.py3-none-any.whl

mkdir -p "${RUNTIME_ROOT}" "${BOOTSTRAP_ROOT}"

echo "[di-entry] python: $(${PYTHON} -c 'import sys; print(sys.executable, sys.version.replace(chr(10), " "))')"
echo "[di-entry] downloading bootstrap wheels"
USE_MEMARTS=0 "${PYTHON}" - "${TORCH_NPU_WHL_OBS}" "${TORCH_NPU_WHL_LOCAL}" "${MOXING_WHL_OBS}" "${MOXING_WHL_LOCAL}" <<'PY'
import sys
import moxing as mox

for source, target in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    print(f"[di-entry] download {source} -> {target}", flush=True)
    mox.file.copy(source, target)
PY

echo "[di-entry] installing pinned inference dependencies"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
"${PYTHON}" -m pip install --upgrade pip setuptools==75.8.0 wheel
"${PYTHON}" -m pip install torch==2.7.1 torchvision==0.22.1
"${PYTHON}" -m pip install --force-reinstall --no-deps "${TORCH_NPU_WHL_LOCAL}"
"${PYTHON}" -m pip uninstall -y moxing moxing-framework >/dev/null 2>&1 || true
"${PYTHON}" -m pip install "${MOXING_WHL_LOCAL}"
"${PYTHON}" -m pip install \
  transformers==4.56.2 \
  'tokenizers>=0.22.0,<0.23.0' \
  huggingface-hub==0.36.2 \
  accelerate==1.6.0 \
  deepspeed==0.14.4 \
  peft==0.19.1 \
  'safetensors>=0.4.3' \
  sentencepiece==0.2.1 \
  tiktoken==0.13.0 \
  'Pillow>=10.0.0' \
  shortuuid==1.0.13 \
  'pydantic>=2.0' \
  'markdown2[all]' \
  packaging \
  requests \
  einops==0.8.2 \
  einops-exts==0.0.4 \
  timm==1.0.27 \
  numpy==1.26.4 \
  opencv-python-headless==4.11.0.86 \
  protobuf==4.25.7 \
  'scipy>=1.10,<2' \
  'scikit-learn>=1.2' \
  shapely==2.1.2 \
  loguru==0.7.3
"${PYTHON}" -m pip install setuptools==75.8.0 numpy==1.26.4 protobuf==4.25.7

export MOX_PROFILE=1
export MOX_RECORD_OBS=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}

"${PYTHON}" - <<'PY'
import json
import torch
import torch_npu

info = {
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": torch.npu.is_available(),
    "npu_count": torch.npu.device_count(),
}
print("[di-preflight] " + json.dumps(info), flush=True)
if not info["npu_available"] or info["npu_count"] < 8:
    raise SystemExit("DI evaluation requires one node with eight visible NPUs.")
PY

export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1
unset MASTER_PORT

export RUN_ID
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
export LOCAL_OUTPUT_ROOT=${RUNTIME_ROOT}/results
export TEST_RESULT_OBS=${OUTPUT_URL%/}/checkpoint34376_3ec_eval_${RUN_ID}
export UPLOAD_RESULTS=True
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

# Dependencies are prepared above; keep the restored 3ec535b inference launcher unchanged.
export INSTALL_DEPS=False
export ENABLE_MOXING_UPGRADE=False

echo "[di-entry] topology: NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[di-entry] runtime root: ${RUNTIME_ROOT}"
echo "[di-entry] result OBS: ${TEST_RESULT_OBS}"
echo "DI_throughput: 0.01 samples/s/npu"

exec bash "${SCRIPT_DIR}/test_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh"
