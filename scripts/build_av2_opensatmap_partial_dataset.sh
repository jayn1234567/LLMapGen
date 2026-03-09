#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/project/jn/UniMapGen

python scripts/build_av2_opensatmap_partial_dataset.py "$@"
