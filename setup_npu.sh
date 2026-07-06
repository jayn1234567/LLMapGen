#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# NPU environment setup for this project
# Run: bash setup_npu.sh
# ============================================================

echo "=== Setting up NPU environment ==="

# ---- NPU toolkit ----
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# ---- proxy (unset if behind firewall) ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# ---- torch + torch_npu ----
# Pick the wheel matching your python version (example for cp311, aarch64):
#   pip install torch==2.7.1
#   pip install torch_npu==2.7.1rc1
# Or install from local whl:
#   pip install /path/to/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

pip install torch==2.7.1 2>/dev/null || echo "[WARN] torch not installed from PyPI, expecting pre-installed"
pip install torch_npu==2.7.1rc1 2>/dev/null || echo "[WARN] torch_npu not installed, expecting pre-installed"

# ---- transformers ----
pip install transformers==4.48.3

# ---- other deps ----
pip install tokenizers==0.22.1
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install safetensors
pip install packaging
pip install Pillow

# ---- project (editable install) ----
cd "$(dirname "$0")"
pip install -e . 2>/dev/null || pip install -e . --no-deps

echo "=== Setup done ==="
echo "Verify: python -c 'import torch; print(torch.npu.is_available())'"
