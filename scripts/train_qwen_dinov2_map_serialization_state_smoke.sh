#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/project/jn/UniMapGen
source scripts/activate_unimapgen_gpu_env.sh

python -m unimapgen.train_qwen_map \
  --config configs/qwen_dinov2_map_serialization_state_smoke.yaml
