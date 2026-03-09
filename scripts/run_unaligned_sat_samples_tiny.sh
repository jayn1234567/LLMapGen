#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/unimapgen_unaligned_sat_samples_tiny.yaml"
OUT_DIR="outputs/unimapgen_unaligned_sat_samples_tiny"
PRED_JSON="${OUT_DIR}/preds_val.json"

echo "[1/4] Check data"
python -m unimapgen.check_data --config "${CONFIG}" --max_scan_per_split 256

echo "[2/4] Train"
python -m unimapgen.train --config "${CONFIG}"

if [[ -f "${OUT_DIR}/best.pt" ]]; then
  CKPT="${OUT_DIR}/best.pt"
elif [[ -f "${OUT_DIR}/latest.pt" ]]; then
  CKPT="${OUT_DIR}/latest.pt"
else
  CKPT="$(ls -1 "${OUT_DIR}"/*.pt | sort | tail -n 1)"
fi
echo "Use checkpoint: ${CKPT}"

echo "[3/4] Eval"
python -m unimapgen.eval --config "${CONFIG}" --checkpoint "${CKPT}"

echo "[4/4] Predict"
python -m unimapgen.predict --config "${CONFIG}" --checkpoint "${CKPT}" --output "${PRED_JSON}"

echo "Done. Output dir: ${OUT_DIR}"
