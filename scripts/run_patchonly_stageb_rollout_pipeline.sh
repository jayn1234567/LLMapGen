#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LF_ENV="${LF_ENV:-$ROOT/.envs/llamafactory-cu128}"
PY_ENV="${PY_ENV:-$ROOT/.envs/unimapgen-gpu}"
LOG_DIR="$ROOT/outputs/logs"

PATCH_ONLY_PID_FILE="${PATCH_ONLY_PID_FILE:-$LOG_DIR/patch_only_100img_train_20260316.pid}"
STAGEB_CFG="${STAGEB_CFG:-$ROOT/configs/llamafactory_paper16_stageb_from_patchonly_mixture/qwen2_5vl_3b_lora_sft.yaml}"
STAGEB_LOG="${STAGEB_LOG:-$LOG_DIR/stageb_from_patchonly_mixture_20260316.log}"
STAGEB_PID_FILE="${STAGEB_PID_FILE:-$LOG_DIR/stageb_from_patchonly_mixture_20260316.pid}"
STAGEB_OUTPUT="${STAGEB_OUTPUT:-$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_stageb_from_patchonly_mixture_lora}"

ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-$ROOT/scripts/rollout_predict_qwen2_5vl_from_raw_family_manifest.py}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$ROOT/outputs/rollout_eval_stageb_from_patchonly_16fam}"
ROLLOUT_LOG="${ROLLOUT_LOG:-$LOG_DIR/rollout_eval_stageb_from_patchonly_16fam_20260316.log}"

BASE_MODEL="${BASE_MODEL:-/mnt/data/project/jn/UniMapGen/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct}"
FAMILY_MANIFEST="${FAMILY_MANIFEST:-$ROOT/outputs/paper16_family_manifest_100img.jsonl}"
ANN_JSON="${ANN_JSON:-/mnt/data/data1/OpenSateMap/annotrainval20.json}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_DIR/pipeline_patchonly_stageb_rollout_20260316.log}"

mkdir -p "$LOG_DIR"

require_executable() {
  local exe_path="$1"
  local name="$2"
  if [[ ! -x "$exe_path" ]]; then
    echo "[ERROR] Missing $name executable: $exe_path" >&2
    exit 1
  fi
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"
}

wait_for_pid_finish() {
  local pid_file="$1"
  local name="$2"
  if [[ ! -f "$pid_file" ]]; then
    log "$name pid file not found, assuming the stage is already finished."
    return 0
  fi
  local pid
  pid="$(tr -d '
 ' < "$pid_file" || true)"
  if [[ -z "$pid" ]]; then
    log "$name pid file is empty, assuming the stage is already finished."
    return 0
  fi
  log "Waiting for $name to finish. pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "$name finished. pid=$pid"
}

start_stageb() {
  if [[ -f "$STAGEB_OUTPUT/trainer_state.json" || -d "$STAGEB_OUTPUT/checkpoint-200" ]]; then
    log "Stage B output already exists, skipping retrain: $STAGEB_OUTPUT"
    return 0
  fi
  log "Starting Stage B training from patch-only adapter."
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    "$LF_ENV/bin/llamafactory-cli" train "$STAGEB_CFG"
  ) > "$STAGEB_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$STAGEB_PID_FILE"
  log "Stage B started. pid=$pid"
  wait "$pid"
  log "Stage B training finished."
}

run_rollout() {
  log "Starting rollout evaluation."
  rm -rf "$ROLLOUT_OUTPUT"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    "$PY_ENV/bin/python" "$ROLLOUT_SCRIPT"       --ann-json "$ANN_JSON"       --family-manifest "$FAMILY_MANIFEST"       --output-root "$ROLLOUT_OUTPUT"       --split train       --max-families 16       --base-model "$BASE_MODEL"       --adapter "$STAGEB_OUTPUT"       --engine custom       --device cuda:0       --max-new-tokens 2048       --use-patch-only-prompt-when-empty       --export-visualizations
  ) > "$ROLLOUT_LOG" 2>&1
  log "Rollout evaluation finished. output=$ROLLOUT_OUTPUT"
}

main() {
  require_executable "$LF_ENV/bin/llamafactory-cli" "llamafactory-cli"
  require_executable "$PY_ENV/bin/python" "python"
  log "Pipeline supervisor started."
  wait_for_pid_finish "$PATCH_ONLY_PID_FILE" "patch-only training"
  start_stageb
  run_rollout
  log "Pipeline supervisor finished successfully."
}

main "$@"
