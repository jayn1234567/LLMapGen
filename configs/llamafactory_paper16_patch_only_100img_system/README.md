# LLaMAFactory Config For `paper16_patch_only_100img_system`

This folder contains the LLaMAFactory training config for the dataset:

- `/mnt/data/project/jn/UniMapGen/outputs/paper16_patch_only_100img_system`

Files:

- `dataset_info.json`: dataset registry for LLaMAFactory.
- `qwen2_5vl_3b_lora_sft.yaml`: Qwen2.5-VL-3B LoRA SFT config.

Notes:

- This is the patch-only baseline: the model only sees the current `896 x 896` satellite patch.
- There is no previous-state input in the user message.
- The target is `full_patch_map`, not ownership-only supervision.
- The dataset uses OpenAI-style ShareGPT messages plus an `images` column.
- The training samples use the current structured serialization rules:
  - `paper_structured`
  - `canonical_cut_then_origin`
  - `first_point_distance_to_patch_origin`
  - `equal_distance`
- `image_max_pixels` is set to `896 * 896 = 802816` to preserve the current patch resolution.
- If this causes OOM, reduce `image_max_pixels` first, then reduce `cutoff_len` or LoRA target scope.

Suggested command:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train /mnt/data/project/jn/UniMapGen/configs/llamafactory_paper16_patch_only_100img_system/qwen2_5vl_3b_lora_sft.yaml
```

Default local model path:

- `/mnt/data/project/jn/UniMapGen/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct`
