# Script Naming

Top-level `scripts/` keeps the current NPU full-parameter train/test entrypoints:

- `train_full_*`: train LLM, ViT, projector, and DeepStack mergers.
- `test_full_*`: cloud inference/eval for the corresponding full-parameter checkpoint.
- `debug.sh`: local NPU DINOv3 smoke training with no OBS transfer and no dependency installation.

Subdirectories keep non-full or local platform-specific scripts:

- `scripts/npu/`: NPU training scripts that freeze one side of the model.
- `scripts/gpu/`: GPU training/inference/visualization utilities.

Training mode in filenames:

- `llm_align_*_freeze-vit`: freeze the ViT, train the LLM plus alignment/projector/DeepStack modules.
- `vit_align_*_freeze-llm`: freeze the LLM, train the ViT plus alignment/projector/DeepStack modules.
- `full`: train all model components.

DINO type and platform are explicit:

- `dinov2` or `dinov3` identifies the vision tower family.
- `_npu` or `_gpu` identifies the target platform.

Common DINOv3 scripts do not pass `--dino_variant`; the code infers DINO type from `mm_vision_tower_type` or the `vision_tower` path. They also avoid hard-coded `--input_image_size` unless a specific experiment needs an override.
