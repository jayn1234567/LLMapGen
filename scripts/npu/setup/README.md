# NPU Environment Setup

## Native Qwen3-VL Training On The Local Ascend Server

```text
create_mllm_native_qwen3vl_torch240_npu_env_from_infer.sh
```

This script clones the proven local
`/home/ma-user/.conda/envs/mllm-infer-torch240-py311` environment into
`/home/ma-user/.conda/envs/mllm-native-qwen3vl-torch240-py311`. It keeps the
local CANN-compatible Torch stack unchanged (`torch=2.4.0`,
`torch_npu=2.4.0`, `torchvision=0.19.0`) and installs the newer Transformers,
PEFT, processor, and image dependencies required by native Qwen3-VL-8B.

The source inference environment is never modified. Set `SOURCE_ENV_NAME` or
`SOURCE_ENV_PREFIX` when the proven Torch 2.4 environment is stored elsewhere.
Set `RECREATE_ENV=true` only when the isolated target environment should be
rebuilt. The generated activation script is:

```text
/home/ma-user/.conda/envs/mllm-native-qwen3vl-torch240-py311/activate_mllm_native_qwen3vl_torch240.sh
```

This local environment recipe does not alter formal DI jobs. Formal DI launchers
install and verify their separate Torch 2.7 / torch-npu 2.7 runtime.

## Other Setup Scripts

| Script | Purpose |
|---|---|
| `create_mllm_infer_torch240_npu_env_from_mapgen.sh` | Build the stable local Torch 2.4 inference environment. |
| `create_mllm_coordtokens_npu_env_from_mapgen.sh` | Build the coordinate-token experiment environment. |
| `create_mllm_dinov2_seg_npu_env.sh` | Build the DINO segmentation-pretraining environment. |
| `create_rc_dataset_v2_env.sh` | Build the Dataset V2 preparation environment. |
