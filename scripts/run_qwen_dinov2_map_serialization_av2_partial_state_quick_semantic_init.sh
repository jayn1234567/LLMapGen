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

python -m unimapgen.check_data \
  --config configs/qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.yaml

python -m unimapgen.train_qwen_map \
  --config configs/qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.yaml
