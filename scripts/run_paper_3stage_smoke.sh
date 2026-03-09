#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[Check] Stage1 config data readiness..."
python -m unimapgen.check_data --config configs/unimapgen_paper_stage1_sft_smoke.yaml

echo "[Stage1] Training..."
python -m unimapgen.train --config configs/unimapgen_paper_stage1_sft_smoke.yaml

echo "[Stage2] Training..."
python -m unimapgen.train --config configs/unimapgen_paper_stage2_align_smoke.yaml

echo "[Stage3] Training..."
python -m unimapgen.train --config configs/unimapgen_paper_stage3_state_smoke.yaml

echo "[Stage3] Eval..."
python -m unimapgen.eval \
  --config configs/unimapgen_paper_stage3_state_smoke.yaml \
  --checkpoint outputs/unimapgen_paper_stage3_state_smoke/latest.pt

echo "[Stage3] Global state scan..."
python -m unimapgen.infer_state_scan \
  --config configs/unimapgen_paper_stage3_state_smoke.yaml \
  --checkpoint outputs/unimapgen_paper_stage3_state_smoke/latest.pt \
  --split val \
  --scene_limit 1 \
  --max_patches_per_scene 8 \
  --output outputs/paper_stage3_state_scan.json

echo "Done."
