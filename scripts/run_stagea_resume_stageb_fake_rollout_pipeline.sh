#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/project/jn/UniMapGen"
LF_ENV="$ROOT/.envs/llamafactory-cu128"
PY_ENV="$ROOT/.envs/unimapgen-gpu"
ROLLOUT_ENV="$ROOT/.envs/llamafactory-cu128"
LOG_DIR="$ROOT/outputs/logs"

ANN_JSON="/mnt/data/data1/OpenSateMap/annotrainval20.json"
FAMILY_MANIFEST="$ROOT/outputs/paper16_family_manifest_100img.jsonl"

EXPORT_SCRIPT="$ROOT/scripts/export_llamafactory_state_sft_from_raw_family_manifest.py"
EXPORT_OUTPUT="$ROOT/outputs/paper16_sft_100img_system_paper_serialized_neighborfix_fake_mixture"
EXPORT_LOG="$LOG_DIR/paper16_fake_mixture_100img_export_20260317.log"

STAGEA_CFG="$ROOT/configs/llamafactory_paper16_patch_only_100img_system/qwen2_5vl_3b_lora_sft.yaml"
STAGEA_RUNTIME_CFG="$LOG_DIR/stagea_patch_only_resume_20260317.yaml"
STAGEA_LOG="$LOG_DIR/stagea_patch_only_resume_20260317.log"
STAGEA_PID_FILE="$LOG_DIR/stagea_patch_only_resume_20260317.pid"
STAGEA_OUTPUT="$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora"

STAGEB_TEMPLATE_CFG="$ROOT/configs/llamafactory_paper16_stageb_from_patchonly_fake_mixture/qwen2_5vl_3b_lora_sft.yaml"
STAGEB_RUNTIME_CFG="$LOG_DIR/stageb_from_patchonly_fake_mixture_20260317.yaml"
STAGEB_LOG="$LOG_DIR/stageb_from_patchonly_fake_mixture_20260317.log"
STAGEB_PID_FILE="$LOG_DIR/stageb_from_patchonly_fake_mixture_20260317.pid"
STAGEB_OUTPUT="$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_stageb_from_patchonly_fake_mixture_lora"

ROLLOUT_SCRIPT="$ROOT/scripts/rollout_predict_qwen2_5vl_from_raw_family_manifest.py"
ROLLOUT_OUTPUT="$ROOT/outputs/rollout_eval_stageb_fake_mixture_gated_16fam"
ROLLOUT_LOG="$LOG_DIR/rollout_eval_stageb_fake_mixture_gated_20260317.log"

BASE_MODEL="$ROOT/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct"
PIPELINE_LOG="$LOG_DIR/pipeline_stagea_resume_stageb_fake_rollout_20260317.log"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"
}

latest_checkpoint() {
  local base_dir="$1"
  find "$base_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

resolve_stagea_adapter() {
  if [[ -f "$STAGEA_OUTPUT/adapter_config.json" ]]; then
    echo "$STAGEA_OUTPUT"
    return 0
  fi
  latest_checkpoint "$STAGEA_OUTPUT"
}

export_fake_mixture_dataset() {
  if [[ -f "$EXPORT_OUTPUT/dataset_info.json" ]]; then
    log "Fake-state mixture dataset already exists, skipping export: $EXPORT_OUTPUT"
    return 0
  fi
  log "Exporting fake-state mixture 100img dataset."
  (
    "$PY_ENV/bin/python" "$EXPORT_SCRIPT" \
      --ann-json "$ANN_JSON" \
      --family-manifest "$FAMILY_MANIFEST" \
      --output-root "$EXPORT_OUTPUT" \
      --splits train \
      --use-system-prompt
  ) > "$EXPORT_LOG" 2>&1
  log "Fake-state mixture dataset export finished."
}

resume_stagea_if_needed() {
  if [[ -f "$STAGEA_OUTPUT/adapter_config.json" ]]; then
    log "Stage A already finished, adapter root exists: $STAGEA_OUTPUT"
    return 0
  fi
  local ckpt
  ckpt="$(latest_checkpoint "$STAGEA_OUTPUT")"
  if [[ -z "$ckpt" ]]; then
    log "No Stage A checkpoint found under $STAGEA_OUTPUT"
    exit 1
  fi
  log "Resuming Stage A from checkpoint: $ckpt"
  cp "$STAGEA_CFG" "$STAGEA_RUNTIME_CFG"
  {
    echo ""
    echo "resume_from_checkpoint: $ckpt"
    echo "overwrite_output_dir: false"
  } >> "$STAGEA_RUNTIME_CFG"
  (
    export CUDA_VISIBLE_DEVICES=0
    "$LF_ENV/bin/llamafactory-cli" train "$STAGEA_RUNTIME_CFG"
  ) > "$STAGEA_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$STAGEA_PID_FILE"
  log "Stage A resume started. pid=$pid"
  wait "$pid"
  log "Stage A resume finished."
}

start_stageb() {
  if [[ -f "$STAGEB_OUTPUT/adapter_config.json" ]]; then
    log "New Stage B output already exists, skipping retrain: $STAGEB_OUTPUT"
    return 0
  fi
  local stagea_adapter
  stagea_adapter="$(resolve_stagea_adapter)"
  if [[ -z "$stagea_adapter" ]]; then
    log "Unable to resolve a finished Stage A adapter/checkpoint."
    exit 1
  fi
  log "Starting new Stage B from Stage A adapter: $stagea_adapter"
  sed "s#^adapter_name_or_path:.*#adapter_name_or_path: $stagea_adapter#g" "$STAGEB_TEMPLATE_CFG" > "$STAGEB_RUNTIME_CFG"
  (
    export CUDA_VISIBLE_DEVICES=0
    "$LF_ENV/bin/llamafactory-cli" train "$STAGEB_RUNTIME_CFG"
  ) > "$STAGEB_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$STAGEB_PID_FILE"
  log "Stage B started. pid=$pid"
  wait "$pid"
  log "Stage B finished."
}

run_rollout() {
  local stagea_adapter
  stagea_adapter="$(resolve_stagea_adapter)"
  if [[ -z "$stagea_adapter" ]]; then
    log "Unable to resolve Stage A adapter for rollout."
    exit 1
  fi
  if [[ ! -f "$STAGEB_OUTPUT/adapter_config.json" ]]; then
    log "Stage B root adapter not found, cannot run rollout: $STAGEB_OUTPUT"
    exit 1
  fi
  log "Starting gated rollout evaluation."
  rm -rf "$ROLLOUT_OUTPUT"
  (
    export CUDA_VISIBLE_DEVICES=0
    "$ROLLOUT_ENV/bin/python" "$ROLLOUT_SCRIPT" \
      --ann-json "$ANN_JSON" \
      --family-manifest "$FAMILY_MANIFEST" \
      --output-root "$ROLLOUT_OUTPUT" \
      --split train \
      --max-families 16 \
      --base-model "$BASE_MODEL" \
      --adapter "$STAGEB_OUTPUT" \
      --patch-only-adapter "$stagea_adapter" \
      --engine custom \
      --device cuda:0 \
      --max-new-tokens 2048 \
      --use-patch-only-prompt-when-empty \
      --enable-handoff-gating \
      --export-visualizations
  ) > "$ROLLOUT_LOG" 2>&1
  log "Rollout evaluation finished. output=$ROLLOUT_OUTPUT"
}

main() {
  log "Pipeline started."
  export_fake_mixture_dataset
  resume_stagea_if_needed
  start_stageb
  run_rollout
  log "Pipeline finished successfully."
}

main "$@"
