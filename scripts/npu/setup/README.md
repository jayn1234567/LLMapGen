# NPU Environment Setup

Create a clean Python 3.11 Ascend/NPU conda environment:

```bash
cd /cache/jn/MLLM_project-unimapgen_v7
bash scripts/npu/setup/create_mllm_npu_py311_env.sh
```

Activate it:

```bash
source /home/ma-user/.conda/envs/mllm-npu-py311/activate_mllm_npu.sh
python -V
which python
which torchrun
```

Recreate from scratch:

```bash
FORCE_RECREATE=true bash scripts/npu/setup/create_mllm_npu_py311_env.sh
```

Useful overrides:

```bash
ENV_NAME=mllm-npu-py311
ENV_PREFIX=/home/ma-user/.conda/envs/mllm-npu-py311
PYTHON_VERSION=3.11
CONDA_CHANNEL=http://192.168.214.30:8088/repository/conda-proxy/main
```

The script installs `torch==2.7.1`, the pinned Ascend `torch_npu`
wheel, `transformers==4.56.2`, `accelerate==1.6.0`, `deepspeed==0.14.4`,
`moxing-framework`, and the project in editable mode with `--no-deps`.
