#!/usr/bin/env bash
set -euo pipefail

# DI one-node/eight-NPU evaluation for the local512 oracle-intersection-prompt model.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

echo "[di-entry] reached ${SCRIPT_PATH}"
echo "[di-entry] repo=${REPO_ROOT} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"
echo "DI_throughput: 0.00 samples/s/npu"

if [ -z "${OUTPUT_URL:-}" ]; then
  echo "ERROR: OUTPUT_URL was not injected by DI." >&2
  exit 2
fi

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
  set -u
fi

PYTHON=${PYTHON:-python}
RUN_ID=${RUN_ID:-local512_intersection_prompt_checkpoint12504_fixed1100_$(date -u +%Y%m%d_%H%M%S)}
RUNTIME_ROOT=${RUNTIME_ROOT:-/cache/jn/di_eval_${RUN_ID}}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-/cache/jn/di_eval_bootstrap}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUNTIME_ROOT}/results}

TORCH_NPU_WHL_OBS=${TORCH_NPU_WHL_OBS:-obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl}
MOXING_WHL_OBS=${MOXING_WHL_OBS:-obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl}
TORCH_NPU_WHL_LOCAL=${BOOTSTRAP_ROOT}/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl
MOXING_WHL_LOCAL=${BOOTSTRAP_ROOT}/moxing_framework-2.3.8-py2.py3-none-any.whl

mkdir -p "${RUNTIME_ROOT}" "${BOOTSTRAP_ROOT}"
echo "[di-entry] python=$(${PYTHON} -c 'import sys; print(sys.executable, sys.version.replace(chr(10), " "))')"

USE_MEMARTS=0 "${PYTHON}" - "${TORCH_NPU_WHL_OBS}" "${TORCH_NPU_WHL_LOCAL}" "${MOXING_WHL_OBS}" "${MOXING_WHL_LOCAL}" <<'PY'
import sys
import moxing as mox

for source, target in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    print(f"[di-bootstrap] {source} -> {target}", flush=True)
    mox.file.copy(source, target)
PY

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
"${PYTHON}" -m pip install --upgrade pip setuptools==75.8.0 wheel
"${PYTHON}" -m pip install torch==2.7.1 torchvision==0.22.1
"${PYTHON}" -m pip install --force-reinstall --no-deps "${TORCH_NPU_WHL_LOCAL}"
"${PYTHON}" -m pip uninstall -y moxing moxing-framework >/dev/null 2>&1 || true
"${PYTHON}" -m pip install "${MOXING_WHL_LOCAL}"
"${PYTHON}" -m pip install \
  transformers==4.56.2 'tokenizers>=0.22.0,<0.23.0' huggingface-hub==0.36.2 \
  accelerate==1.6.0 deepspeed==0.14.4 peft==0.19.1 safetensors sentencepiece==0.2.1 \
  tiktoken==0.13.0 Pillow shortuuid pydantic 'markdown2[all]' packaging requests \
  einops==0.8.2 einops-exts==0.0.4 timm==1.0.27 numpy==1.26.4 \
  opencv-python-headless==4.11.0.86 protobuf==4.25.7 'scipy>=1.10,<2' \
  'scikit-learn>=1.2' shapely==2.1.2 loguru==0.7.3
"${PYTHON}" -m pip install setuptools==75.8.0 numpy==1.26.4 protobuf==4.25.7

export MOX_PROFILE=1
export MOX_RECORD_OBS=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

"${PYTHON}" - <<'PY'
import json
import os
import torch
import torch_npu

required_devices = int(os.environ.get("NPROC_PER_NODE", "1"))
info = {
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": torch.npu.is_available(),
    "npu_count": torch.npu.device_count(),
}
print("[di-preflight] " + json.dumps(info), flush=True)
if not info["npu_available"] or info["npu_count"] < required_devices:
    raise SystemExit(
        f"DI evaluation requested {required_devices} processes but only "
        f"{info['npu_count']} NPUs are visible."
    )
PY

VISION_TOWER=${VISION_TOWER:-${RUNTIME_ROOT}/models/facebook_dinov2-large}
VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
if [ ! -f "${VISION_TOWER}/config.json" ]; then
  mkdir -p "${VISION_TOWER}"
  echo "[vision] downloading ${VISION_TOWER_OBS_PATH} -> ${VISION_TOWER}"
  "${PYTHON}" - "${VISION_TOWER_OBS_PATH}" "${VISION_TOWER}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
fi
if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "ERROR: DINOv2 download is incomplete: ${VISION_TOWER}" >&2
  exit 2
fi

export SKIP_ENV_ACTIVATION=True
export VISION_TOWER
export RUN_ID
export RUN_LABEL=${RUN_LABEL:-local512_intersection_prompt_checkpoint12504_fixed1100}
export OUTPUT_ROOT
export DATASET_OBS_PATH=obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local512/local512_550k.tar
export DATASET_ARCHIVE_PATH=${RUNTIME_ROOT}/dataset/local512_550k.tar
export DATASET_EXTRACT_ROOT=${RUNTIME_ROOT}/dataset/local512_extract
export DATASET_ROOT=${DATASET_EXTRACT_ROOT}/local512
export EVAL_SOURCE_JSONL=${DATASET_ROOT}/phase_a/eval.jsonl
export FIXED_EVAL_ROOT=${RUNTIME_ROOT}/eval_sets/local512_fixed1100_seed42_v1
export TRANSFORMED_FIXED_EVAL_ROOT=${RUNTIME_ROOT}/eval_sets/local512_fixed1100_intersection_prompt_v1
export FIXED_EVAL_COUNTS=easy=300,medium=300,hard=300,very_hard=200
export FIXED_EVAL_SEED=42
export REBUILD_FIXED_EVAL=False
export EVAL_RECORD_TRANSFORM=intersection_prompt
export INFERENCE_MAP_TASK=lane
export CHECKPOINT_NAME=checkpoint-12504
export CHECKPOINT_OBS_PATH=obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/27/6c0d9e3afbb34751abb22ba915363096/output/ma-job-7f369f87-998d-410b-b205-316ba192fa5a/checkpoint-12504/
export CHECKPOINT_CACHE_ROOT=${RUNTIME_ROOT}/checkpoints/local512_intersection_prompt
export PATCH_SIZE=512
export PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-2}
export VIS_LIMIT=${VIS_LIMIT:-50}

echo "[di-entry] checkpoint=${CHECKPOINT_OBS_PATH}"
echo "[di-entry] fixed_eval=local512 eval, seed=42, counts=${FIXED_EVAL_COUNTS}"
echo "[di-entry] prompt=oracle current-patch intersections; target/evaluation=centerline only"
echo "[di-entry] output=${OUTPUT_ROOT}"

bash "${SCRIPT_DIR}/test_local_stage_a_lane_intersection_prompt_local512_checkpoint12504_fixed1100_torch240_npu.sh"

mkdir -p \
  "${OUTPUT_ROOT}/evaluation_set/source_local512_fixed1100" \
  "${OUTPUT_ROOT}/evaluation_set/intersection_prompt_fixed1100"
cp -a "${FIXED_EVAL_ROOT}/." "${OUTPUT_ROOT}/evaluation_set/source_local512_fixed1100/"
cp -a "${TRANSFORMED_FIXED_EVAL_ROOT}/." "${OUTPUT_ROOT}/evaluation_set/intersection_prompt_fixed1100/"

PUBLISH_ROOT=${OUTPUT_URL%/}/${RUN_ID}
echo "[di-publish] ${OUTPUT_ROOT} -> ${PUBLISH_ROOT}"
if [[ "${PUBLISH_ROOT}" == obs://* ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}" "${PUBLISH_ROOT}" <<'PY'
import sys
import moxing as mox
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
PY
else
  mkdir -p "${PUBLISH_ROOT}"
  cp -a "${OUTPUT_ROOT}/." "${PUBLISH_ROOT}/"
fi

echo "DI_throughput: 0.00 samples/s/npu"
echo "[di-entry] evaluation complete: ${PUBLISH_ROOT}"
