#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${1:?"usage: merge_zero3_multinode_checkpoint.sh RUN_ROOT CHECKPOINT_NAME [OUTPUT_DIR]"}
CHECKPOINT_NAME=${2:?"checkpoint name is required, for example checkpoint-1000"}
RUN_ROOT=$(readlink -f "${RUN_ROOT}")
OUTPUT_DIR=${3:-${RUN_ROOT}/merged_${CHECKPOINT_NAME}}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(readlink -f "${OUTPUT_DIR}")

SHARD_ROOT=${RUN_ROOT}/zero_shards
if [ ! -d "${SHARD_ROOT}" ]; then
  echo "ERROR: zero_shards directory not found: ${SHARD_ROOT}" >&2
  exit 1
fi

ASSEMBLED_DIR=$(mktemp -d "${OUTPUT_DIR}/.assembled_zero3.XXXXXX")
cleanup() {
  rm -rf -- "${ASSEMBLED_DIR}"
}
trap cleanup EXIT

NODE_COUNT=0
EXPECTED_NODES=
EXPECTED_WORLD_SIZE=
DIRECT_GLOBAL_STEP=False
for node_dir in "${SHARD_ROOT}"/node_*; do
  [ -d "${node_dir}" ] || continue
  checkpoint_dir=${node_dir}/${CHECKPOINT_NAME}
  if [ ! -d "${checkpoint_dir}" ]; then
    echo "ERROR: ${CHECKPOINT_NAME} is missing from ${node_dir}" >&2
    exit 1
  fi
  NODE_COUNT=$((NODE_COUNT + 1))
  layout=${node_dir}/zero3_shard_layout.json
  if [ ! -f "${layout}" ]; then
    echo "ERROR: shard layout metadata is missing: ${layout}" >&2
    exit 1
  fi
  read -r node_expected_nodes node_expected_world_size < <(
    python - "${layout}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(payload["expected_nodes"]), int(payload["expected_world_size"]))
PY
  )
  if [ -z "${EXPECTED_NODES}" ]; then
    EXPECTED_NODES=${node_expected_nodes}
    EXPECTED_WORLD_SIZE=${node_expected_world_size}
  elif [ "${EXPECTED_NODES}" -ne "${node_expected_nodes}" ] || [ "${EXPECTED_WORLD_SIZE}" -ne "${node_expected_world_size}" ]; then
    echo "ERROR: inconsistent ZeRO shard layout metadata in ${layout}" >&2
    exit 1
  fi
  # A regular Trainer checkpoint stores global_step* and zero_to_fp32.py below
  # checkpoint-*. A final DeepSpeed save can instead expose global_step*
  # directly below node_*. Preserve the wrapper expected by zero_to_fp32.py in
  # both cases while combining globally unique rank shard names.
  if [[ "${CHECKPOINT_NAME}" == global_step* ]]; then
    DIRECT_GLOBAL_STEP=True
    mkdir -p "${ASSEMBLED_DIR}/${CHECKPOINT_NAME}"
    cp -an "${checkpoint_dir}/." "${ASSEMBLED_DIR}/${CHECKPOINT_NAME}/"
    printf '%s\n' "${CHECKPOINT_NAME}" > "${ASSEMBLED_DIR}/latest"
  else
    cp -an "${checkpoint_dir}/." "${ASSEMBLED_DIR}/"
  fi

  if [ "${NODE_COUNT}" -eq 1 ]; then
    for name in \
      zero_to_fp32.py latest config.json generation_config.json trainer_state.json training_args.bin \
      tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json \
      chat_template.jinja vocab.json merges.txt tokenizer.model preprocessor_config.json \
      args.json qwen_multimodal_checkpoint.json \
      rc_dinov2_centerline_json_modules.pt rc_dinov2_centerline_json_modules.pth; do
      if [ -f "${node_dir}/${name}" ]; then
        cp -f "${node_dir}/${name}" "${ASSEMBLED_DIR}/${name}"
      fi
    done
    if [[ "${CHECKPOINT_NAME}" == global_step* ]]; then
      printf '%s\n' "${CHECKPOINT_NAME}" > "${ASSEMBLED_DIR}/latest"
    fi
  fi
done

if [ "${NODE_COUNT}" -eq 0 ]; then
  echo "ERROR: no node_* shard directories found under ${SHARD_ROOT}" >&2
  exit 1
fi
if [ "${NODE_COUNT}" -ne "${EXPECTED_NODES}" ]; then
  echo "ERROR: incomplete multi-node checkpoint: found ${NODE_COUNT}/${EXPECTED_NODES} node shard directories" >&2
  exit 1
fi

CONVERTER=${ASSEMBLED_DIR}/zero_to_fp32.py
if [ ! -f "${CONVERTER}" ]; then
  echo "ERROR: zero_to_fp32.py is missing after shard assembly" >&2
  exit 1
fi

OPTIM_SHARDS=$(find "${ASSEMBLED_DIR}" -type f -name '*optim_states.pt' | wc -l)
MODEL_SHARDS=$(find "${ASSEMBLED_DIR}" -type f -name '*model_states.pt' | wc -l)
echo "[zero3-merge] nodes=${NODE_COUNT} optimizer_shards=${OPTIM_SHARDS} model_shards=${MODEL_SHARDS} direct_global_step=${DIRECT_GLOBAL_STEP}"
if [ "${OPTIM_SHARDS}" -lt "${EXPECTED_WORLD_SIZE}" ]; then
  echo "ERROR: shard assembly is incomplete: optimizer_shards=${OPTIM_SHARDS}, expected_world_size=${EXPECTED_WORLD_SIZE}" >&2
  exit 1
fi

python "${CONVERTER}" "${ASSEMBLED_DIR}" "${OUTPUT_DIR}/pytorch_model.bin"

for name in \
  config.json generation_config.json trainer_state.json training_args.bin \
  tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json \
  chat_template.jinja vocab.json merges.txt tokenizer.model preprocessor_config.json \
  args.json qwen_multimodal_checkpoint.json \
  rc_dinov2_centerline_json_modules.pt rc_dinov2_centerline_json_modules.pth; do
  if [ -f "${ASSEMBLED_DIR}/${name}" ]; then
    cp -f "${ASSEMBLED_DIR}/${name}" "${OUTPUT_DIR}/${name}"
  fi
done

python - "${OUTPUT_DIR}/pytorch_model.bin" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"Merged model was not produced: {path}")
print(f"[zero3-merge] merged model bytes={path.stat().st_size}")
PY

trap - EXIT
cleanup
echo "[zero3-merge] done: ${OUTPUT_DIR}"
