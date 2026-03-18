# LLaMAFactory Config For Stage B (`patch-only` -> `state mixture`)

This folder contains the second-stage LLaMAFactory training config:

- Stage A adapter:
  - `/mnt/data/project/jn/UniMapGen/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora`
- Stage B dataset:
  - `/mnt/data/project/jn/UniMapGen/outputs/paper16_sft_100img_system_paper_serialized_neighborfix_mixture`

Files:

- `dataset_info.json`: dataset registry for LLaMAFactory.
- `qwen2_5vl_3b_lora_sft.yaml`: second-stage LoRA SFT config.

Design:

- Stage A teaches the model to predict roads from the current `896 x 896` patch alone.
- Stage B continues training from the Stage A LoRA adapter and introduces `neighborfix + state mixture`.
- The goal is to learn cross-patch continuity without making the model over-rely on perfect state traces.

Notes:

- This config uses `adapter_name_or_path` to continue training from the Stage A LoRA weights.
- The Stage B dataset already contains mixed state modes:
  - `empty`
  - `no_state`
  - `weak_state`
  - `full_state`
- `image_max_pixels` is set to `896 * 896 = 802816`.
- The Stage B learning rate is lower (`5e-5`) than the patch-only base stage to make continuation more stable.

Suggested command:

```bash
llamafactory-cli train /mnt/data/project/jn/UniMapGen/configs/llamafactory_paper16_stageb_from_patchonly_mixture/qwen2_5vl_3b_lora_sft.yaml
```

If you want to start a brand-new second-stage adapter while keeping Stage A frozen as the loaded base adapter, add:

```yaml
create_new_adapter: true
```

Default local base model path:

- `/mnt/data/project/jn/UniMapGen/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct`
