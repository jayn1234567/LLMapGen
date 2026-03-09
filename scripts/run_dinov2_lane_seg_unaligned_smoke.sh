#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/dinov2_lane_seg_unaligned_smoke.yaml"
OUT_DIR="outputs/dinov2_lane_seg_unaligned_smoke"
PRED_DIR="${OUT_DIR}/pred_masks"
PRED_JSON="${OUT_DIR}/predictions.json"

echo "[1/3] Train"
python -m unimapgen.train_lane_seg --config "${CONFIG}"

if [[ -f "${OUT_DIR}/best.pt" ]]; then
  CKPT="${OUT_DIR}/best.pt"
else
  CKPT="${OUT_DIR}/latest.pt"
fi
echo "Use checkpoint: ${CKPT}"

echo "[2/3] Eval"
python -m unimapgen.eval_lane_seg --config "${CONFIG}" --checkpoint "${CKPT}"

echo "[3/3] Predict"
python -m unimapgen.predict_lane_seg \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --max_samples 4 \
  --threshold 0.5 \
  --output_dir "${PRED_DIR}" \
  --output_json "${PRED_JSON}"

echo "Done. Output dir: ${OUT_DIR}"
