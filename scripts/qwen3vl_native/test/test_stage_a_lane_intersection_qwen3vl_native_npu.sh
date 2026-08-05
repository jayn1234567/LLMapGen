#!/usr/bin/env bash

# ============================================================
# NPU inference/eval
# Fixed recipe: phase_a | lane+intersection | native Qwen3-VL full architecture
# Produces result JSON, aggregate eval summaries, and visualization images.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # Absolute path of this launcher.
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # Directory that contains this launcher.
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # Project root used for relative imports.
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"                # Required cloud output root provided by ModelArts.

DATASET_PHASE=phase_a                                                             # Dataset stage.
MAP_TASK=lane_intersection                                                        # Task type.
MODEL_RECIPE=qwen3vl_native                                                       # Native Qwen3-VL architecture.

CLUSTER_SAVE=${OUTPUT_URL}                                                        # Cloud output root.
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # Alias used by existing scripts.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}                                       # Unique run id.
OBS_CACHE=${OBS_CACHE:-/cache}                                                    # Local cache root.
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # OBS model root.
QWEN3VL_MODEL_NAME=${QWEN3VL_MODEL_NAME:-Qwen3-VL-8B-Instruct}                   # Native base checkpoint name.
QWEN3VL_OBS_PATH=${QWEN3VL_OBS_PATH:-${MODEL_OBS_PATH}/${QWEN3VL_MODEL_NAME}}     # Native base checkpoint OBS path.
QWEN3VL_PATH=${QWEN3VL_PATH:-${OBS_CACHE}/checkpoints/${QWEN3VL_MODEL_NAME}}      # Local native base checkpoint.
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_lane_intersection_samples_norm_33w_empty_patch.zip}  # Dataset zip.
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_lane_intersection_samples_norm_33w_empty_patch}  # Extracted dataset directory.
CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}                                      # OBS checkpoint roots, separated by comma/semicolon/newline.
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}                                              # Local checkpoint roots, separated by comma/semicolon/newline.
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/native_qwen3vl_ckpts_${RUN_ID}}  # Local checkpoint download root.

DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset_${RUN_ID}.zip}          # Local dataset zip.
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset_extract_${RUN_ID}}  # Dataset extract root.
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}         # Extracted dataset root.
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}                                     # Image root.
TEST_JSON=${TEST_JSON:-${DATASET_PATH}/${DATASET_PHASE}/test.jsonl}               # Test JSONL.
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-${OBS_CACHE}/test_${DATASET_PHASE}_${MAP_TASK}_${MODEL_RECIPE}_${RUN_ID}}  # Local result root.
CLOUD_OUTPUT_DIR=${TEST_RESULT_OBS:-${OSB_SHARE_PATH%/}/test_results_${RUN_ID}}   # Cloud result directory.

NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}                                           # 0 means full test split.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}                                            # Generation limit.
COORD_MODE=${COORD_MODE:-auto}                                                    # auto, pixel, or norm1000.
COORD_RANGE=${COORD_RANGE:-1000}                                                  # Normalized coord range.
DEFAULT_PATCH_SIZE=${DEFAULT_PATCH_SIZE:-512}                                     # Fallback patch size.
EVAL_METER_PER_PIXEL=${EVAL_METER_PER_PIXEL:-0.2}                                 # Metric conversion.
EVAL_BUFFER_SIZE=${EVAL_BUFFER_SIZE:-1.0}                                         # Line matching buffer.
EVAL_MATCH_THRESHOLD=${EVAL_MATCH_THRESHOLD:-0.33}                                # Matching threshold.
SKIP_VISUALIZE=${SKIP_VISUALIZE:-False}                                           # Disable per-sample visualization when True.
SKIP_WHOLE_MAP_VIZ=${SKIP_WHOLE_MAP_VIZ:-False}                                   # Disable stitched whole-map visualization when True.
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}                                                  # Optional system prompt; set it to the training value for matched comparisons.

export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend toolkit root.
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend custom OPP root.
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend OPP path.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}                             # Gloo interface.
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}                                 # TP interface.
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}                             # HCCL interface.
export CUDA_DEVICE_MAX_CONNECTIONS=1                                              # Ascend compatibility flag.
export HCCL_WHITELIST_DISABLE=1                                                   # Disable HCCL whitelist.
export HCCL_CONNECT_TIMEOUT=7200                                                  # HCCL connect timeout.
export HCCL_EXEC_TIMEOUT=7200                                                     # HCCL exec timeout.
export HCCL_IF_BASE_PORT=64000                                                    # HCCL base port.
export INF_NAN_MODE_ENABLE=1                                                      # Inf/NaN handling.
export WITHOUT_JIT_COMPILE=1                                                      # Disable JIT compile path.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}                                      # CPU threads.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # Tokenizer warnings.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # Project imports.

INSTALL_DEPS=${INSTALL_DEPS:-True}                                                # Install dependencies.
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-True}                              # Upgrade moxing.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers>=5.7.0"}                     # Native Qwen3-VL-capable transformers.
TOKENIZERS_SPEC=${TOKENIZERS_SPEC:-"tokenizers>=0.22.0"}                          # Tokenizers version.
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
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2' 'shapely>=2.0.0' 'opencv-python-headless>=4.8.0' loguru
  pip install "peft>=0.10.0"
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
print(re.sub(r"[^A-Za-z0-9._-]+", "_", value.split("/")[-1]).strip("._-") or "checkpoint")
PY
}

resolve_checkpoint() {
  python - "$1" <<'PY'
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "adapter_config.json")):
    print(root)
    raise SystemExit(0)
cmd = [sys.executable, "scripts/tools/resolve_best_checkpoint.py", "--output-dir", str(root), "--best-name", "infer_best", "--best-name", "eval_best", "--best-name", "best", "--allow-direct"]
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
}

mkdir -p "${DATASET_EXTRACT_ROOT}" "${CHECKPOINT_DOWNLOAD_ROOT}" "${LOCAL_OUTPUT_ROOT}"
if [ ! -f "${QWEN3VL_PATH}/config.json" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${QWEN3VL_OBS_PATH}', '${QWEN3VL_PATH}')"
fi
if [ ! -f "${QWEN3VL_PATH}/config.json" ]; then
  echo "ERROR: native Qwen3-VL base model is incomplete: ${QWEN3VL_PATH}"
  exit 1
fi
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
for path in "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then echo "ERROR: required path not found: ${path}"; exit 1; fi
done

CHECKPOINT_ITEMS=()
CHECKPOINT_LABELS=()
if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
  while IFS= read -r obs_item; do
    label=$(safe_label "${obs_item}")
    local_dir="${CHECKPOINT_DOWNLOAD_ROOT}/${label}"
    python -c "import moxing as mox; mox.file.copy_parallel('${obs_item}', '${local_dir}')"
    CHECKPOINT_ITEMS+=("$(resolve_checkpoint "${local_dir}")")
    CHECKPOINT_LABELS+=("${label}")
  done < <(read_list "${CHECKPOINT_OBS_LIST}")
elif [ -n "${CHECKPOINT_DIRS}" ]; then
  while IFS= read -r local_item; do
    CHECKPOINT_ITEMS+=("$(resolve_checkpoint "${local_item}")")
    CHECKPOINT_LABELS+=("$(safe_label "${local_item}")")
  done < <(read_list "${CHECKPOINT_DIRS}")
else
  echo "ERROR: set CHECKPOINT_OBS_LIST or CHECKPOINT_DIRS for native Qwen3-VL inference."
  exit 1
fi

for idx in "${!CHECKPOINT_ITEMS[@]}"; do
  ckpt="${CHECKPOINT_ITEMS[$idx]}"
  label="${CHECKPOINT_LABELS[$idx]}"
  out_dir="${LOCAL_OUTPUT_ROOT}/${label}"
  extra_args=()
  if [[ "${SKIP_VISUALIZE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then extra_args+=(--skip-visualize); fi
  if [[ "${SKIP_WHOLE_MAP_VIZ}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then extra_args+=(--skip-whole-map-viz); fi
  if [ -n "${SYSTEM_PROMPT}" ]; then extra_args+=(--system-prompt "${SYSTEM_PROMPT}"); fi
  echo "============================================================"
  echo "Native Qwen3-VL checkpoint: ${ckpt}"
  echo "Output: ${out_dir}"
  echo "============================================================"
  python -m mllm.native_qwen3vl.infer \
    --model-name-or-path "${ckpt}" \
    --model-base "${QWEN3VL_PATH}" \
    --test-json "${TEST_JSON}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${out_dir}" \
    --phase "${DATASET_PHASE}" \
    --map-task "${MAP_TASK}" \
    --num-samples "${NUM_TEST_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --default-patch-size "${DEFAULT_PATCH_SIZE}" \
    --eval-meter-per-pixel "${EVAL_METER_PER_PIXEL}" \
    --eval-buffer-size "${EVAL_BUFFER_SIZE}" \
    --eval-match-threshold "${EVAL_MATCH_THRESHOLD}" \
    --bf16 \
    --include-intersections \
    "${extra_args[@]}"
done

python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_ROOT}', '${CLOUD_OUTPUT_DIR}')"
echo "Final native Qwen3-VL inference output: ${CLOUD_OUTPUT_DIR}"
