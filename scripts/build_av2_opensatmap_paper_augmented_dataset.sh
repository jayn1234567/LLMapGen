#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/project/jn/UniMapGen

python scripts/build_av2_opensatmap_paper_augmented_dataset.py \
  --crop-root "${AV2_ALIGNED_CROP_ROOT:-/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix}" \
  --output-root "${AV2_PAPER_AUG_ROOT:-/mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_paper_aug_partial}" \
  --opensatmap-root "${OPENSATMAP_ROOT:-/mnt/data/data1/OpenSateMap}"
