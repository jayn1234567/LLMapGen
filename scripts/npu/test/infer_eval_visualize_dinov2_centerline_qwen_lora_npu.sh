#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-0}}"

DI_THROUGHPUT_VALUE="${DI_THROUGHPUT_VALUE:-1.00}"
echo "DI_throughput: ${DI_THROUGHPUT_VALUE} samples/s/npu"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cache/jn/checkpoint-29610}"
RUN_ROOT="${RUN_ROOT:-}"
RUN_ARGS_JSON="${RUN_ARGS_JSON:-}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/cache/jn/model/Qwen3-8B}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
DINOV2_MODEL_NAME_OR_PATH="${DINOV2_MODEL_NAME_OR_PATH:-/cache/jn/model/dinov2-large}"
VISUAL_ENCODER_CHECKPOINT_PATH="${VISUAL_ENCODER_CHECKPOINT_PATH:-/cache/jn/model/dinov2seg_bridge/dinov2_centerline_assets_qwen3_8b/visual_encoder_checkpoint.pt}"
BRIDGE_MODULES_STATE_PATH="${BRIDGE_MODULES_STATE_PATH:-/cache/jn/model/dinov2seg_bridge/dinov2_centerline_assets_qwen3_8b/bridge_modules_state.pt}"
TRAINROOT="${TRAINROOT:-/cache/jn/prepared_lane_intersection_trainroot}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-npu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
VIS_LIMIT="${VIS_LIMIT:-64}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
ENCODER_INPUT_PAD_SIZE="${ENCODER_INPUT_PAD_SIZE:-518}"
MAP_TASK="${MAP_TASK:-lane_intersection}"
CATEGORIES="${CATEGORIES:-centerline,intersection}"
METER_PER_PIXEL="${METER_PER_PIXEL:-0.2}"
LINE_WIDTH_PX="${LINE_WIDTH_PX:-6}"
JIANGJIHUA_BUFFER_SIZE="${JIANGJIHUA_BUFFER_SIZE:-1.0}"
JIANGJIHUA_MATCH_THRESHOLD="${JIANGJIHUA_MATCH_THRESHOLD:-0.33}"
AUTO_INSTALL_EVAL_DEPS="${AUTO_INSTALL_EVAL_DEPS:-true}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://repo.huaweicloud.com/repository/pypi/simple/}"
INFER_HEARTBEAT_SECONDS="${INFER_HEARTBEAT_SECONDS:-60}"
SHARD_LOG_TAIL_LINES="${SHARD_LOG_TAIL_LINES:-120}"

RUN_NAME="${RUN_NAME:-$(basename "${CHECKPOINT_DIR}")_${SPLIT}_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/cache/jn/outputs/infer_eval_visualize_${RUN_NAME}}"
PRED_JSONL="${PRED_JSONL:-${OUTPUT_DIR}/predictions.jsonl}"
PRED_SUMMARY_JSON="${PRED_SUMMARY_JSON:-${OUTPUT_DIR}/predict_summary.json}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/shards}"

mkdir -p "${OUTPUT_DIR}"

PREDICT_ARGS=(
  scripts/predict_dinov2_centerline.py
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}"
  --dinov2-model-name-or-path "${DINOV2_MODEL_NAME_OR_PATH}"
  --visual-encoder-checkpoint-path "${VISUAL_ENCODER_CHECKPOINT_PATH}"
  --bridge-modules-state-path "${BRIDGE_MODULES_STATE_PATH}"
  --trainroot "${TRAINROOT}"
  --split "${SPLIT}"
  --output-jsonl "${PRED_JSONL}"
  --summary-json "${PRED_SUMMARY_JSON}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --image-size "${IMAGE_SIZE}"
  --encoder-input-pad-size "${ENCODER_INPUT_PAD_SIZE}"
  --map-task "${MAP_TASK}"
)

if [ -n "${RUN_ROOT}" ]; then
  PREDICT_ARGS+=(--run-root "${RUN_ROOT}")
fi
if [ -n "${RUN_ARGS_JSON}" ]; then
  PREDICT_ARGS+=(--run-args-json "${RUN_ARGS_JSON}")
fi
if [ "${LOCAL_FILES_ONLY}" = "true" ]; then
  PREDICT_ARGS+=(--local-files-only)
fi

echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Run root:   ${RUN_ROOT:-<auto>}"
echo "Args json:  ${RUN_ARGS_JSON:-<auto>}"
echo "Qwen:       ${MODEL_NAME_OR_PATH}"
echo "DINOv2:     ${DINOV2_MODEL_NAME_OR_PATH}"
echo "Visual ckpt:${VISUAL_ENCODER_CHECKPOINT_PATH}"
echo "Bridge:     ${BRIDGE_MODULES_STATE_PATH}"
echo "Trainroot:  ${TRAINROOT}"
echo "Split:      ${SPLIT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Device:     ${DEVICE}, visible=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPROC:      ${NPROC_PER_NODE}"
echo "============================================================"

if [ "${NPROC_PER_NODE}" -le 1 ]; then
  python "${PREDICT_ARGS[@]}" --device "${DEVICE}"
else
  IFS=',' read -r -a VISIBLE_DEVICE_ITEMS <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  if [ "${#VISIBLE_DEVICE_ITEMS[@]}" -lt "${NPROC_PER_NODE}" ]; then
    echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE}, but ASCEND_RT_VISIBLE_DEVICES has only ${#VISIBLE_DEVICE_ITEMS[@]} entries: ${ASCEND_RT_VISIBLE_DEVICES}" >&2
    exit 1
  fi
  rm -rf "${SHARD_DIR}"
  mkdir -p "${SHARD_DIR}"
  PIDS=()
  SHARD_LOGS=()
  for ((SHARD_INDEX=0; SHARD_INDEX<NPROC_PER_NODE; SHARD_INDEX++)); do
    DEVICE_ID="${VISIBLE_DEVICE_ITEMS[${SHARD_INDEX}]}"
    SHARD_JSONL="${SHARD_DIR}/predictions_shard_${SHARD_INDEX}.jsonl"
    SHARD_SUMMARY="${SHARD_DIR}/predict_summary_shard_${SHARD_INDEX}.json"
    SHARD_LOG="${SHARD_DIR}/predict_shard_${SHARD_INDEX}.log"
    echo "[multi-infer] start shard ${SHARD_INDEX}/${NPROC_PER_NODE} on visible device ${DEVICE_ID}"
    (
      export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
      export ASCEND_VISIBLE_DEVICES="${DEVICE_ID}"
      export NPU_VISIBLE_DEVICES="${DEVICE_ID}"
      python "${PREDICT_ARGS[@]}" \
        --output-jsonl "${SHARD_JSONL}" \
        --summary-json "${SHARD_SUMMARY}" \
        --num-shards "${NPROC_PER_NODE}" \
        --shard-index "${SHARD_INDEX}" \
        --device "${DEVICE}"
    ) >"${SHARD_LOG}" 2>&1 &
    PIDS+=("$!")
    SHARD_LOGS+=("${SHARD_LOG}")
  done

  echo "[multi-infer] shard stdout/stderr is redirected to: ${SHARD_DIR}/predict_shard_*.log"
  echo "[multi-infer] main process will print a heartbeat every ${INFER_HEARTBEAT_SECONDS}s while waiting."
  (
    while true; do
      sleep "${INFER_HEARTBEAT_SECONDS}"
      RUNNING=0
      for PID in "${PIDS[@]}"; do
        if kill -0 "${PID}" >/dev/null 2>&1; then
          RUNNING=$((RUNNING + 1))
        fi
      done
      echo "[multi-infer] waiting: running=${RUNNING}/${NPROC_PER_NODE}; logs=${SHARD_DIR}"
      for LOG_PATH in "${SHARD_LOGS[@]}"; do
        if [ -s "${LOG_PATH}" ]; then
          echo "[multi-infer] $(basename "${LOG_PATH}") last line: $(tail -n 1 "${LOG_PATH}")"
        fi
      done
    done
  ) &
  HEARTBEAT_PID="$!"

  FAILED=0
  for SHARD_INDEX in "${!PIDS[@]}"; do
    PID="${PIDS[${SHARD_INDEX}]}"
    if wait "${PID}"; then
      echo "[multi-infer] shard ${SHARD_INDEX} completed"
    else
      echo "[multi-infer] shard ${SHARD_INDEX} failed; log=${SHARD_LOGS[${SHARD_INDEX}]}" >&2
      FAILED=1
    fi
  done
  kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  if [ "${FAILED}" -ne 0 ]; then
    echo "ERROR: at least one inference shard failed. Logs are under ${SHARD_DIR}" >&2
    for LOG_PATH in "${SHARD_LOGS[@]}"; do
      echo "==================== tail -n ${SHARD_LOG_TAIL_LINES} ${LOG_PATH} ====================" >&2
      if [ -f "${LOG_PATH}" ]; then
        tail -n "${SHARD_LOG_TAIL_LINES}" "${LOG_PATH}" >&2 || true
      else
        echo "missing log file" >&2
      fi
    done
    exit 1
  fi

  python - "${SHARD_DIR}" "${PRED_JSONL}" "${PRED_SUMMARY_JSON}" "${NPROC_PER_NODE}" <<'PY'
import json
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
output_jsonl = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
nproc = int(sys.argv[4])
output_jsonl.parent.mkdir(parents=True, exist_ok=True)

total = 0
parse_ok = 0
summaries = []
with output_jsonl.open("w", encoding="utf-8") as out:
    for shard_index in range(nproc):
        shard_path = shard_dir / f"predictions_shard_{shard_index}.jsonl"
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with shard_path.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    out.write(line)
                    total += 1
        shard_summary_path = shard_dir / f"predict_summary_shard_{shard_index}.json"
        if shard_summary_path.is_file():
            payload = json.loads(shard_summary_path.read_text(encoding="utf-8"))
            summaries.append(payload)
            parse_ok += int(payload.get("parse_ok", 0) or 0)

summary = {
    "mode": "multi_shard_inference",
    "num_shards": nproc,
    "output_jsonl": str(output_jsonl),
    "num_rows": total,
    "parse_ok": parse_ok,
    "parse_ok_rate": (parse_ok / total) if total else 0.0,
    "shard_summaries": summaries,
}
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
fi

if [ "${AUTO_INSTALL_EVAL_DEPS}" = "true" ]; then
  python - <<'PY'
import importlib.util
import os
import subprocess
import sys

missing = []
for module_name, package_name in (("shapely", "shapely"), ("scipy", "scipy")):
    if importlib.util.find_spec(module_name) is None:
        missing.append(package_name)
if missing:
    index_url = os.environ.get("PIP_INDEX_URL", "http://repo.huaweicloud.com/repository/pypi/simple/")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-i",
        index_url,
        "--trusted-host",
        "repo.huaweicloud.com",
        *missing,
    ]
    print("[eval-deps] installing: " + " ".join(missing), flush=True)
    subprocess.check_call(cmd)
else:
    print("[eval-deps] shapely/scipy already available", flush=True)
PY
fi

python scripts/tools/eval_visualize_dinov2_centerline_predictions.py \
  --pred-jsonl "${PRED_JSONL}" \
  --trainroot "${TRAINROOT}" \
  --out-dir "${OUTPUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --map-task "${MAP_TASK}" \
  --categories "${CATEGORIES}" \
  --meter-per-pixel "${METER_PER_PIXEL}" \
  --line-width-px "${LINE_WIDTH_PX}" \
  --jiangjihua-buffer-size "${JIANGJIHUA_BUFFER_SIZE}" \
  --jiangjihua-match-threshold "${JIANGJIHUA_MATCH_THRESHOLD}" \
  --vis-limit "${VIS_LIMIT}"

echo "============================================================"
echo "DI_throughput: ${DI_THROUGHPUT_VALUE} samples/s/npu"
echo "Done."
echo "Prediction JSONL: ${PRED_JSONL}"
echo "Prediction summary: ${PRED_SUMMARY_JSON}"
echo "Eval summary: ${OUTPUT_DIR}/eval_visualization_summary.json"
echo "Jiangjihua eval: ${OUTPUT_DIR}/eval_jiangjihua.json"
echo "Official eval: ${OUTPUT_DIR}/eval_official.json"
echo "Engineering eval: ${OUTPUT_DIR}/eval_engineering.json"
echo "Visualization sheet: ${OUTPUT_DIR}/visualization/prediction_overlay_sheet.png"
echo "============================================================"
