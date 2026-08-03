#!/usr/bin/env bash
set -euo pipefail

# Merge the four-node RawLane-550k ZeRO-3 global_step34376 checkpoint and run
# the same fresh-OBS, GT-empty-suppressed original all/low/high E2E protocol
# used by the RawLane-200k comparison.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

SHARDED_RUN_OBS_ROOT=${SHARDED_RUN_OBS_ROOT:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/31/3fce4c245d294c20a99be5699e5269cc/output/ma-job-04702ef4-047f-4b37-baa6-cc996720a92b}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-global_step34376}
EXPECTED_NODES=${EXPECTED_NODES:-4}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-32}

LOCAL_RUN_ROOT=${LOCAL_RUN_ROOT:-/cache/jn/checkpoints/rawlane550k_zero3_globalstep34376}
MERGED_CHECKPOINT_DIR=${MERGED_CHECKPOINT_DIR:-${LOCAL_RUN_ROOT}/merged_${CHECKPOINT_NAME}}
MERGED_CHECKPOINT_NAME=$(basename "${MERGED_CHECKPOINT_DIR}")

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}

RUN_ID=${RUN_ID:-rawlane550k_zero3_globalstep34376_gt_empty_fresh_obs_e2e_$(date +%Y%m%d_%H%M%S)}
FRESH_RUN_ROOT=${FRESH_RUN_ROOT:-/cache/jn/e2e_eval/fresh_obs_runs/${RUN_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cache/jn/outputs/${RUN_ID}}
EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}
RULE_WORKERS=${RULE_WORKERS:-16}

echo "============================================================"
echo "RAWLANE-550K ZERO-3 GLOBAL-STEP34376 FULL E2E"
echo "OBS shards:       ${SHARDED_RUN_OBS_ROOT}"
echo "Shard checkpoint: ${CHECKPOINT_NAME}"
echo "Local shards:     ${LOCAL_RUN_ROOT}/zero_shards"
echo "Merged checkpoint:${MERGED_CHECKPOINT_DIR}"
echo "Visible NPUs:     ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Per-device batch: ${PER_DEVICE_INFER_BATCH_SIZE}"
echo "Fresh E2E root:   ${FRESH_RUN_ROOT}"
echo "Output:           ${OUTPUT_ROOT}"
echo "Protocol:         fresh OBS + GT-empty suppression + all/low/high"
echo "============================================================"

echo "[rawlane550k-zero3-e2e] stage 1/2: download and merge four-node ZeRO-3 shards"
MERGE_ONLY=True \
SHARDED_RUN_OBS_ROOT="${SHARDED_RUN_OBS_ROOT}" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
EXPECTED_NODES="${EXPECTED_NODES}" \
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE}" \
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT}" \
MERGED_CHECKPOINT_DIR="${MERGED_CHECKPOINT_DIR}" \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}" \
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}" \
bash "${SCRIPT_DIR}/test_local_rawlane550k_zero3_globalstep34376_merge_eval_torch240_npu.sh"

if [ ! -s "${MERGED_CHECKPOINT_DIR}/pytorch_model.bin" ]; then
  echo "ERROR: merged model was not produced: ${MERGED_CHECKPOINT_DIR}/pytorch_model.bin" >&2
  exit 2
fi
if [ ! -f "${MERGED_CHECKPOINT_DIR}/config.json" ]; then
  echo "ERROR: merged model config is missing: ${MERGED_CHECKPOINT_DIR}/config.json" >&2
  exit 2
fi

echo "[rawlane550k-zero3-e2e] stage 2/2: fresh RawLane inference and original E2E evaluation"
RUN_ID="${RUN_ID}" \
FRESH_RUN_ROOT="${FRESH_RUN_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CHECKPOINT_NAME="${MERGED_CHECKPOINT_NAME}" \
CHECKPOINT_CACHE_ROOT="${LOCAL_RUN_ROOT}" \
CHECKPOINT_OBS_PATH="${SHARDED_RUN_OBS_ROOT}" \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES}" \
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
PER_DEVICE_INFER_BATCH_SIZE="${PER_DEVICE_INFER_BATCH_SIZE}" \
EXPECTED_E2E_SCENES=110 \
EVAL_VIS_FLAG="${EVAL_VIS_FLAG}" \
RULE_WORKERS="${RULE_WORKERS}" \
bash "${SCRIPT_DIR}/eval_rawlane200k_checkpoint12504_gt_empty_fresh_obs_original_e2e_npu.sh"

echo "============================================================"
echo "RAWLANE-550K ZERO-3 E2E COMPLETE"
echo "Merged checkpoint: ${MERGED_CHECKPOINT_DIR}"
echo "Fresh E2E data:    ${FRESH_RUN_ROOT}/e2e_data"
echo "Inference JSON:    ${OUTPUT_ROOT}/inference/json"
echo "Suppression report:${OUTPUT_ROOT}/gt_oracle_suppression_report.json"
echo "All roads:         ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_all"
echo "Low roads:         ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_low"
echo "High roads:        ${OUTPUT_ROOT}/original_pipeline_metrics/eval_result_high"
echo "============================================================"
