#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/project/jn/UniMapGen
source scripts/activate_unimapgen_gpu_env.sh
export PYTHONUNBUFFERED=1
export USE_TF=0
export USE_FLAX=0
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CKPT=outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type/best.pt

python -m unimapgen.eval_qwen_map \
  --config configs/qwen_dinov2_map_serialization_av2_partial_state_line_type.yaml \
  --checkpoint "${CKPT}" \
  --split val

python -m unimapgen.predict_qwen_state_scan \
  --config configs/qwen_dinov2_map_serialization_av2_partial_state_line_type.yaml \
  --checkpoint "${CKPT}" \
  --split val \
  --output outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type/predictions_state_scan.json

python -m unimapgen.eval_opensatmap_official \
  --config configs/qwen_dinov2_map_serialization_av2_partial_state_line_type.yaml \
  --prediction_json outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type/predictions_state_scan.json \
  --output outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type/official_metrics_state_scan.json
