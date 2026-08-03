#!/usr/bin/env bash
set -euo pipefail

# Download a four-node ZeRO-3 final checkpoint, merge it into a regular full
# checkpoint, then run one fixed-1100 Raw-Lane local256 evaluation pass.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

ENV_DIR=${ENV_DIR:-/home/ma-user/.conda/envs/mllm-infer-torch240-py311}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-${ENV_DIR}/activate_mllm_infer_torch240.sh}
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
  echo "ERROR: inference activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

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

SHARDED_RUN_OBS_ROOT=${SHARDED_RUN_OBS_ROOT:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/31/3fce4c245d294c20a99be5699e5269cc/output/ma-job-04702ef4-047f-4b37-baa6-cc996720a92b}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-global_step34376}
EXPECTED_NODES=${EXPECTED_NODES:-4}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-32}

LOCAL_RUN_ROOT=${LOCAL_RUN_ROOT:-/cache/jn/checkpoints/rawlane550k_zero3_globalstep34376}
MERGED_CHECKPOINT_DIR=${MERGED_CHECKPOINT_DIR:-${LOCAL_RUN_ROOT}/merged_${CHECKPOINT_NAME}}

DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/rawlane_local256_550k/rawlane_local256_550k.tar}
DATASET_ARCHIVE_PATH=${DATASET_ARCHIVE_PATH:-/cache/jn/data/rawlane_local256_550k.tar}
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-/cache/jn/data/rawlane_local256_550k_extract}
FIXED_EVAL_ROOT=${FIXED_EVAL_ROOT:-/cache/jn/eval_sets/rawlane_local256_550k_fixed1100_e300_m300_h300_vh200_seed42_v1}

VISION_TOWER_OBS_PATH=${VISION_TOWER_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large}
VISION_TOWER=${VISION_TOWER:-/cache/jn/model/facebook_dinov2-large}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-2}
VIS_LIMIT=${VIS_LIMIT:-50}
RUN_ID=${RUN_ID:-rawlane550k_zero3_globalstep34376_fixed1100_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOCAL_RUN_ROOT}/zero_shards" "$(dirname "${DATASET_ARCHIVE_PATH}")" \
  "${DATASET_EXTRACT_ROOT}" "$(dirname "${VISION_TOWER}")" "${OUTPUT_ROOT}"

echo "============================================================"
echo "Raw-Lane 550k ZeRO-3 merge + fixed-1100 evaluation"
echo "OBS run:       ${SHARDED_RUN_OBS_ROOT}"
echo "Checkpoint:    ${CHECKPOINT_NAME}"
echo "Expected:      nodes=${EXPECTED_NODES}, world_size=${EXPECTED_WORLD_SIZE}"
echo "Local shards:  ${LOCAL_RUN_ROOT}/zero_shards"
echo "Merged model:  ${MERGED_CHECKPOINT_DIR}"
echo "Visible NPUs:  ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Output:        ${OUTPUT_ROOT}"
echo "============================================================"

python - <<'PY'
import deepspeed
import moxing
import torch
import torch_npu

print(
    "[preflight] "
    f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
    f"deepspeed={deepspeed.__version__} npu_available={torch.npu.is_available()}"
)
if not torch.npu.is_available():
    raise SystemExit("NPU is unavailable in the active environment")
if not hasattr(moxing, "file"):
    raise SystemExit("Huawei moxing-framework with mox.file is required")
PY

for node_rank in $(seq 0 $((EXPECTED_NODES - 1))); do
  local_node_dir=${LOCAL_RUN_ROOT}/zero_shards/node_${node_rank}
  local_checkpoint_dir=${local_node_dir}/${CHECKPOINT_NAME}
  complete_marker=${local_checkpoint_dir}/.obs_download_complete
  if [ -f "${complete_marker}" ]; then
    echo "[shard-download] reuse node_${node_rank}/${CHECKPOINT_NAME}"
    continue
  fi
  mkdir -p "${local_node_dir}" "${local_checkpoint_dir}"
  obs_node_root=${SHARDED_RUN_OBS_ROOT%/}/zero_shards/node_${node_rank}
  echo "[shard-download] ${obs_node_root}/${CHECKPOINT_NAME} -> ${local_checkpoint_dir}"
  OBS_NODE_ROOT="${obs_node_root}" LOCAL_NODE_DIR="${local_node_dir}" CHECKPOINT_NAME="${CHECKPOINT_NAME}" python - <<'PY'
import os
from pathlib import Path

import moxing as mox

obs_node = os.environ["OBS_NODE_ROOT"].rstrip("/")
local_node = Path(os.environ["LOCAL_NODE_DIR"])
checkpoint = os.environ["CHECKPOINT_NAME"]
mox.file.copy_parallel(f"{obs_node}/{checkpoint}", str(local_node / checkpoint))

root_files = (
    "zero3_shard_layout.json", "zero_to_fp32.py", "latest", "config.json",
    "generation_config.json", "trainer_state.json", "training_args.bin",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja", "vocab.json", "merges.txt",
    "tokenizer.model", "preprocessor_config.json", "args.json",
    "qwen_multimodal_checkpoint.json", "rc_dinov2_centerline_json_modules.pt",
    "rc_dinov2_centerline_json_modules.pth",
)
for name in root_files:
    source = f"{obs_node}/{name}"
    if mox.file.exists(source):
        mox.file.copy(source, str(local_node / name))
PY

  if [ ! -f "${local_node_dir}/zero3_shard_layout.json" ]; then
    python - "${local_node_dir}/zero3_shard_layout.json" "${EXPECTED_NODES}" "${node_rank}" "${EXPECTED_WORLD_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_nodes = int(sys.argv[2])
node_rank = int(sys.argv[3])
expected_world_size = int(sys.argv[4])
path.write_text(
    json.dumps(
        {
            "format": "deepspeed_zero3_node_shards",
            "expected_nodes": expected_nodes,
            "node_rank": node_rank,
            "nproc_per_node": expected_world_size // expected_nodes,
            "expected_world_size": expected_world_size,
            "layout_inferred_after_download": True,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
  fi
  touch "${complete_marker}"
done

if ! find "${LOCAL_RUN_ROOT}/zero_shards" -maxdepth 2 -name zero_to_fp32.py -type f -print -quit | grep -q .; then
  converter=$(python - <<'PY'
from pathlib import Path
import deepspeed.utils.zero_to_fp32 as converter

print(Path(converter.__file__).resolve())
PY
  )
  cp -f "${converter}" "${LOCAL_RUN_ROOT}/zero_shards/node_0/zero_to_fp32.py"
  echo "[zero3-merge] using converter from installed DeepSpeed: ${converter}"
fi

if [ ! -s "${MERGED_CHECKPOINT_DIR}/pytorch_model.bin" ]; then
  echo "[zero3-merge] merging on CPU; this requires substantial RAM and disk space"
  bash scripts/tools/merge_zero3_multinode_checkpoint.sh \
    "${LOCAL_RUN_ROOT}" "${CHECKPOINT_NAME}" "${MERGED_CHECKPOINT_DIR}"
else
  echo "[zero3-merge] reuse ${MERGED_CHECKPOINT_DIR}/pytorch_model.bin"
fi

if [ ! -f "${MERGED_CHECKPOINT_DIR}/config.json" ]; then
  echo "ERROR: merged checkpoint is missing config.json: ${MERGED_CHECKPOINT_DIR}" >&2
  exit 2
fi

if [ ! -f "${VISION_TOWER}/config.json" ]; then
  echo "[vision-download] ${VISION_TOWER_OBS_PATH} -> ${VISION_TOWER}"
  VISION_TOWER_OBS_PATH="${VISION_TOWER_OBS_PATH}" VISION_TOWER="${VISION_TOWER}" python - <<'PY'
import os
import moxing as mox

mox.file.copy_parallel(os.environ["VISION_TOWER_OBS_PATH"], os.environ["VISION_TOWER"])
PY
fi

if [ ! -s "${DATASET_ARCHIVE_PATH}" ]; then
  echo "[dataset-download] ${DATASET_OBS_PATH} -> ${DATASET_ARCHIVE_PATH}"
  DATASET_OBS_PATH="${DATASET_OBS_PATH}" DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" python - <<'PY'
import os
import moxing as mox

mox.file.copy(os.environ["DATASET_OBS_PATH"], os.environ["DATASET_ARCHIVE_PATH"])
PY
fi
if ! find "${DATASET_EXTRACT_ROOT}" -type f -path '*/phase_a/eval.jsonl' -print -quit | grep -q .; then
  echo "[dataset-extract] ${DATASET_ARCHIVE_PATH} -> ${DATASET_EXTRACT_ROOT}"
  tar -xf "${DATASET_ARCHIVE_PATH}" -C "${DATASET_EXTRACT_ROOT}"
fi

DATASET_ROOT=$(python - "${DATASET_EXTRACT_ROOT}" <<'PY'
import sys
from pathlib import Path

extract_root = Path(sys.argv[1]).resolve()
candidates = []
for eval_jsonl in extract_root.rglob("phase_a/eval.jsonl"):
    root = eval_jsonl.parent.parent
    if (root / "images").is_dir():
        candidates.append(root)
if len(candidates) != 1:
    raise SystemExit(f"Expected exactly one extracted dataset root, found: {candidates}")
print(candidates[0])
PY
)
echo "[dataset] resolved root: ${DATASET_ROOT}"

SKIP_ENV_ACTIVATION=True \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}" \
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
VIS_LIMIT="${VIS_LIMIT}" \
DATASET_OBS_PATH="${DATASET_OBS_PATH}" \
DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH}" \
DATASET_EXTRACT_ROOT="${DATASET_EXTRACT_ROOT}" \
DATASET_ROOT="${DATASET_ROOT}" \
EVAL_SOURCE_JSONL="${DATASET_ROOT}/phase_a/eval.jsonl" \
FIXED_EVAL_ROOT="${FIXED_EVAL_ROOT}" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
CHECKPOINT_DIR="${MERGED_CHECKPOINT_DIR}" \
CHECKPOINT_CACHE_ROOT="${LOCAL_RUN_ROOT}" \
VISION_TOWER="${VISION_TOWER}" \
RUN_LABEL="rawlane550k_zero3_globalstep34376_fixed1100" \
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
bash scripts/npu/test/test_local_stage_a_lane_intersection_local256_550k_checkpoint34376_fixed1100_singlepass_torch240_npu.sh

echo "============================================================"
echo "ZERO-3 MERGE + FIXED-1100 EVALUATION COMPLETE"
echo "Merged checkpoint: ${MERGED_CHECKPOINT_DIR}"
echo "Combined metrics:  ${OUTPUT_ROOT}/${CHECKPOINT_NAME}/by_difficulty/all_selected/eval.json"
echo "Difficulty metrics:${OUTPUT_ROOT}/${CHECKPOINT_NAME}/by_difficulty/{easy,medium,hard,very_hard}/eval.json"
echo "Visualizations:    ${OUTPUT_ROOT}/${CHECKPOINT_NAME}/viz"
echo "============================================================"
