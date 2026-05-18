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

Common DINOv3 scripts infer DINO type from checkpoint metadata, `mm_vision_tower_type`, or the `vision_tower` path. They also avoid hard-coded `--input_image_size` unless a specific experiment needs an override.

DeepStack and gradient checkpointing are intended to work together. Training scripts keep `GRADIENT_CHECKPOINTING=True` by default, including DeepStack runs. Inference scripts do not hard-code DeepStack settings; they recover DeepStack enabled/disabled state from the checkpoint config unless an override is passed.

The Python training entry defaults to DeepStack disabled. Only scripts explicitly named `*_deepstack_*` enable it by passing both `--disable_deepstack False` and `--deepstack_visual_indexes ...`. Other align/debug scripts keep `DEEPSTACK_VISUAL_INDEXES` empty by default; set it only when you intentionally want DeepStack.

Qwen multimodal checkpoints write `qwen_multimodal_checkpoint.json`. `llava_checkpoint.json` is treated as legacy metadata only. Inference refuses to use the old generic LLaVA loader for directories that contain full model weights, because that route can silently skip Qwen projector, ViT, or DeepStack tensors.

Best checkpoint behavior:

- Normal full training scripts do not run eval and do not maintain best-loss directories by default.
- `train_full_dinov3_qwen3vl-8b_deepstack_train_best_npu.sh` sets train-loss best checkpointing internally and writes the current best training-loss checkpoint to `output/best/` by default. Edit `BEST_TRAIN_LOSS_START_STEP` and `BEST_TRAIN_LOSS_DIR` inside the script when needed.
- `train_full_dinov3_qwen3vl-8b_deepstack_eval_best_npu.sh` uses the validation jsonl configured inside the script, evaluates every `EVAL_STEPS`, and writes the current best eval-loss checkpoint to `output/eval_best/` by default. Edit `EVAL_PATH`, `EVAL_IMAGE_FOLDER`, `EVAL_STEPS`, and `BEST_EVAL_LOSS_DIR` inside the script when needed.
- Both best directories are copied from a normal `checkpoint-*` directory after that checkpoint is fully saved, and include `config.json`, `model.safetensors`, optimizer/scheduler state, and `qwen_multimodal_checkpoint.json`.

Centerline geometry evaluation:

- The project uses `infer_index/line_eval.py` for centerline metrics: LineString buffer IoU plus Hungarian matching, reporting instance-level and length-level precision/recall/F1.
- `scripts/infer_centerline_checkpoint.py --eval-centerline` and `scripts/infer_centerline_state_update.py --eval-centerline` write these metrics with the inference summary.
- `scripts/visualize_centerline.py` automatically prints and saves `centerline_eval.json` after visualization when ground truth is present. Use `--no-eval-centerline` to disable it.
- The default metric scale is `--eval-meter-per-pixel 0.2`, matching `infer_index/param.py`.

Distributed logging defaults to `LLAVA_LOG_RANK0_ONLY=1`, so normal stdout logs are printed by global rank 0 only. Error tracebacks on stderr are kept for nonzero ranks unless `LLAVA_SUPPRESS_NONZERO_STDERR=1` is set.

Full training scripts keep the Hugging Face tqdm progress bar enabled by default and write full logs to `train_metrics.log`, `eval_metrics.log`, and `checkpoint_events.log`. In tqdm mode, `DI_throughput` is also printed with the step metric line through `tqdm.write(...)`.

Do not pass experiment knobs as one-off shell prefixes. Edit the parameter block inside the target script. The main block contains comments for:

- `TARGET_GLOBAL_BATCH_SIZE`, `PER_DEVICE_TRAIN_BATCH_SIZE`: total batch control.
- `LR`, `MM_PROJECTOR_LR`, `WEIGHT_DECAY`, `WARMUP_STEPS`, `LR_SCHEDULER_TYPE`: optimizer schedule.
- `NUM_EPOCHS`, `MODEL_MAX_LENGTH`, `SAVE_STEPS`, `LOGGING_STEPS`: training length and logging.
- `DEEPSTACK_VISUAL_INDEXES`, `DISABLE_DEEPSTACK`: DeepStack on/off and selected ViT layers.
- `SAVE_BEST_TRAIN_LOSS`, `BEST_TRAIN_LOSS_START_STEP`, `BEST_TRAIN_LOSS_DIR`: train-loss best checkpoint.
- `SAVE_BEST_EVAL_LOSS`, `EVAL_STEPS`, `BEST_EVAL_LOSS_DIR`: eval-loss best checkpoint in eval-best scripts.
- `USE_HF_PROGRESS_BAR`: console progress style.
