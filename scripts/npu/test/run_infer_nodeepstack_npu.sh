#!/usr/bin/env bash
set -euo pipefail

# Common NPU inference launcher for explicit stage/task/vision wrappers.
#
# Cloud mode:
#   When OUTPUT_URL is set by the NPU platform, this script installs the same
#   runtime deps as the NPU training scripts, downloads dataset / DINO / trained
#   checkpoint from OBS, runs inference, prints the line-eval table, then uploads
#   outputs to OUTPUT_URL/test_results_${RUN_ID}.
#
# Local mode:
#   When OUTPUT_URL is absent, this script uses existing local paths:
#   DATASET_PATH, IMAGE_FOLDER, VISION_TOWER and CHECKPOINT_DIR/TRAIN_OUTPUT_DIR.

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
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip}
DATASET_DIR_NAME=${DATASET_DIR_NAME:-MLLM20260427_rc_jjh}
TRAINED_CHECKPOINT_OBS=${TRAINED_CHECKPOINT_OBS:-${CHECKPOINT_OBS:-}}

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
    echo "Dataset extract root contents:"
    ls -la "${DATASET_EXTRACT_ROOT}"
    exit 1
  fi

  if [ -z "${CHECKPOINT_DIR:-}" ]; then
    if [ -z "${TRAINED_CHECKPOINT_OBS}" ]; then
      echo "ERROR: set TRAINED_CHECKPOINT_OBS or CHECKPOINT_OBS to the trained checkpoint/output OBS path."
      exit 1
    fi
    CHECKPOINT_LOCAL=${CHECKPOINT_LOCAL:-${OBS_CACHE}/train_output_${RUN_ID}}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading checkpoint from ${TRAINED_CHECKPOINT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    python -c "import moxing as mox; mox.file.copy_parallel('${TRAINED_CHECKPOINT_OBS}', '${CHECKPOINT_LOCAL}')"
    CHECKPOINT_DIR="${CHECKPOINT_LOCAL}"
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

if [ -z "${CHECKPOINT_DIR:-}" ]; then
  TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/cache/unimapgen_v2/train_output/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_qwen3vl8b_nodeepstack}
  BEST_CHECKPOINT_NAME=${BEST_CHECKPOINT_NAME:-eval_best}
  CHECKPOINT_DIR=$(python scripts/tools/resolve_best_checkpoint.py \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --best-name "${BEST_CHECKPOINT_NAME}" \
    --best-name best \
    --allow-direct)
fi

if [ ! -f "${CHECKPOINT_DIR}/model.safetensors" ] && [ ! -f "${CHECKPOINT_DIR}/pytorch_model.bin" ]; then
  if RESOLVED_CHECKPOINT=$(python scripts/tools/resolve_best_checkpoint.py \
      --output-dir "${CHECKPOINT_DIR}" \
      --best-name eval_best \
      --best-name best \
      --best-name best_reward \
      --allow-direct 2>/dev/null); then
    CHECKPOINT_DIR="${RESOLVED_CHECKPOINT}"
  else
    LATEST_CHECKPOINT=$(python - "${CHECKPOINT_DIR}" <<'PY'
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
    if [ -n "${LATEST_CHECKPOINT}" ]; then
      CHECKPOINT_DIR="${LATEST_CHECKPOINT}"
    fi
  fi
fi

for path in "${CHECKPOINT_DIR}" "${VISION_TOWER}" "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path missing: ${path}"
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

JSON_DIR="${OUTPUT_DIR}/json"
PATCH_VIZ_DIR="${OUTPUT_DIR}/viz"
WHOLE_MAP_VIZ_DIR="${OUTPUT_DIR}/whole_map_viz"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
MERGED_GLOBAL_JSON="${OUTPUT_DIR}/merged_global.json"
EVAL_JSON="${OUTPUT_DIR}/eval.json"
mkdir -p "${OUTPUT_DIR}" "${JSON_DIR}" "${PATCH_VIZ_DIR}" "${WHOLE_MAP_VIZ_DIR}"

echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "VISION_TOWER=${VISION_TOWER}"
echo "DATASET_PHASE=${DATASET_PHASE}"
echo "MAP_TASK=${MAP_TASK}"
echo "TEST_JSON=${TEST_JSON}"
echo "IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "COORD_MODE=${COORD_MODE} COORD_RANGE=${COORD_RANGE}"
echo "NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES}"

# ====================== inference ======================
if [ "${DATASET_PHASE}" = "phase_b" ]; then
  INCLUDE_INTERSECTION_ARGS=()
  if [ "${MAP_TASK}" = "lane_intersection" ]; then
    INCLUDE_INTERSECTION_ARGS=(--include-intersections)
  fi
  python scripts/tools/infer_centerline_state_update.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --vision_tower "${VISION_TOWER}" \
    "${INPUT_IMAGE_SIZE_ARGS[@]}" \
    --disable_deepstack \
    --patch-json "${TEST_JSON}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-json "${SUMMARY_JSON}" \
    --output-dir "${JSON_DIR}" \
    --sample-json-dir "${JSON_DIR}" \
    --merged-output-json "${MERGED_GLOBAL_JSON}" \
    --whole-map-viz-dir "${WHOLE_MAP_VIZ_DIR}" \
    --conv-template conv_qwen_3_Dinov2_huawei \
    --device "${DEVICE:-npu:0}" \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    "${INCLUDE_INTERSECTION_ARGS[@]}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    --eval-centerline \
    --eval-output-json "${EVAL_JSON}"
else
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/tools/infer_centerline_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
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
    --output-dir "${OUTPUT_DIR}" \
    --sample-json-dir "${JSON_DIR}" \
    --output-json "${SUMMARY_JSON}" \
    --temperature 0.0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --eval-centerline \
    --eval-output-json "${EVAL_JSON}"

  if [ "${NODE_RANK}" -eq 0 ]; then
    python - "${OUTPUT_DIR}" "${SUMMARY_JSON}" <<'PY'
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
  echo "Skip visualization/upload on non-master node ${NODE_RANK}"
  exit 0
fi

python scripts/tools/visualize_centerline.py \
  --input-dir "${OUTPUT_DIR}" \
  --image-folder "${IMAGE_FOLDER}" \
  --output-dir "${PATCH_VIZ_DIR}" \
  --eval-output-json "${EVAL_JSON}" \
  --whole-map-viz-dir "${WHOLE_MAP_VIZ_DIR}"

if [ -f "${EVAL_JSON}" ]; then
  echo "================ Final Centerline Eval Table ================"
  python - "${EVAL_JSON}" <<'PY'
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

echo "Inference outputs:"
echo "  summary:         ${SUMMARY_JSON}"
echo "  sample json dir: ${JSON_DIR}"
echo "  patch viz dir:   ${PATCH_VIZ_DIR}"
echo "  eval json:       ${EVAL_JSON}"
echo "  whole map dir:   ${WHOLE_MAP_VIZ_DIR}"

if [[ "${CLOUD_MODE}" == "True" ]]; then
  TEST_RESULT_OBS=${TEST_RESULT_OBS:-${OUTPUT_URL%/}/test_results_${RUN_ID}}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Uploading results to ${TEST_RESULT_OBS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python -c "import moxing as mox; mox.file.copy_parallel('${OUTPUT_DIR}', '${TEST_RESULT_OBS}')"
  echo "Results saved to ${TEST_RESULT_OBS}"
fi
