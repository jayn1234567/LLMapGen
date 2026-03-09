#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/project/jn/UniMapGen

CONFIG="configs/qwen_dinov2_map_serialization_smoke.yaml"
OUT_DIR="outputs/qwen_dinov2_map_serialization_smoke"

python -m unimapgen.train_qwen_map --config "${CONFIG}"
python -m unimapgen.eval_qwen_map --config "${CONFIG}" --checkpoint "${OUT_DIR}/latest.pt" --split val --max_samples 1
python -m unimapgen.predict_qwen_map --config "${CONFIG}" --checkpoint "${OUT_DIR}/latest.pt" --split val --max_samples 1 --output "${OUT_DIR}/predictions.json"
