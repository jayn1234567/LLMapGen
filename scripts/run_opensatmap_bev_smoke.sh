#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CFG="configs/unimapgen_opensatmap_bev_smoke.yaml"
OUT_DIR="outputs/unimapgen_opensatmap_bev_smoke"

echo "[Check] OpenSatMap data readiness..."
python -m unimapgen.check_data --config "$CFG"

echo "[Train] OpenSatMap BEV smoke..."
python -m unimapgen.train --config "$CFG"

echo "[Eval] OpenSatMap BEV smoke..."
python -m unimapgen.eval --config "$CFG" --checkpoint "${OUT_DIR}/latest.pt"

echo "[Predict] OpenSatMap BEV smoke..."
python -m unimapgen.predict \
  --config "$CFG" \
  --checkpoint "${OUT_DIR}/latest.pt" \
  --split val \
  --max_samples 8 \
  --output outputs/opensatmap_bev_smoke_preds.json

echo "Done."
