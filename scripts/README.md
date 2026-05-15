# Script Naming

Top-level `scripts/` keeps the current NPU full-parameter train/test entrypoints:

- `train_full_*_deepstack_npu.sh`: train LLM, ViT, projector, and DeepStack mergers.
- `train_full_*_no-deepstack_npu.sh`: train LLM, ViT, and projector with DeepStack disabled.
- `train_full_*_train_best_npu.sh`: train with DeepStack and maintain a best checkpoint by lowest training loss.
- `train_full_*_eval_best_npu.sh`: train with DeepStack, run a validation set by steps, and maintain a best checkpoint by lowest eval loss.
- `test_full_*`: cloud inference/eval for the corresponding full-parameter checkpoint.
- `debug.sh`: local NPU DINOv3 smoke training with no OBS transfer and no dependency installation.

Subdirectories keep non-full or local platform-specific scripts:

- `scripts/npu/`: NPU training scripts that freeze one side of the model.
- `scripts/gpu/`: GPU training/inference/visualization utilities.

Training mode in filenames:

- `llm_align_*_freeze-vit`: freeze the ViT, train the LLM plus alignment/projector/DeepStack modules.
- `vit_align_*_freeze-llm`: freeze the LLM, train the ViT plus alignment/projector/DeepStack modules.
- `full`: train all model components.
- `deepstack`: enable DeepStack during training.
- `no-deepstack`: disable DeepStack during training.
- `train_best`: copy the best training-loss checkpoint to `BEST_TRAIN_LOSS_DIR` (default `best`).
- `eval_best`: copy the best validation-loss checkpoint to `BEST_EVAL_LOSS_DIR` (default `eval_best`).
- `ckpt3200`: starts from the local/cloud checkpoint-3200 variant instead of the base Qwen3-VL checkpoint.

DINO type and platform are explicit:

- `dinov2` or `dinov3` identifies the vision tower family.
- `_npu` or `_gpu` identifies the target platform.

Common DINOv3 scripts do not pass `--dino_variant`; the code infers DINO type from `mm_vision_tower_type` or the `vision_tower` path. They also avoid hard-coded `--input_image_size` unless a specific experiment needs an override.

DeepStack and gradient checkpointing are intended to work together. Training scripts keep `GRADIENT_CHECKPOINTING=True` by default, including DeepStack runs. Inference scripts do not hard-code DeepStack settings; they recover DeepStack enabled/disabled state from the checkpoint config unless an override is passed.

Qwen multimodal checkpoints write `qwen_multimodal_checkpoint.json`. `llava_checkpoint.json` is treated as legacy metadata only. Inference refuses to use the old generic LLaVA loader for directories that contain full model weights, because that route can silently skip Qwen projector, ViT, or DeepStack tensors.

Best checkpoint behavior:

- Normal full training scripts do not run eval and do not maintain best-loss directories by default.
- `train_full_dinov3_qwen3vl-8b_deepstack_train_best_npu.sh` sets `SAVE_BEST_TRAIN_LOSS=True` and writes the current best training-loss checkpoint to `output/best/` by default. Override with `BEST_TRAIN_LOSS_START_STEP` and `BEST_TRAIN_LOSS_DIR`.
- `train_full_dinov3_qwen3vl-8b_deepstack_eval_best_npu.sh` requires an eval jsonl through `EVAL_PATH` (defaulting to the dataset `test.jsonl` when present), evaluates every `EVAL_STEPS`, and writes the current best eval-loss checkpoint to `output/eval_best/` by default. Override with `EVAL_IMAGE_FOLDER`, `EVAL_STEPS`, and `BEST_EVAL_LOSS_DIR`.
- Both best directories are copied from a normal `checkpoint-*` directory after that checkpoint is fully saved, and include `config.json`, `model.safetensors`, optimizer/scheduler state, and `qwen_multimodal_checkpoint.json`.

Distributed logging defaults to `LLAVA_LOG_RANK0_ONLY=1`, so normal stdout logs are printed by global rank 0 only. Error tracebacks on stderr are kept for nonzero ranks unless `LLAVA_SUPPRESS_NONZERO_STDERR=1` is set.
