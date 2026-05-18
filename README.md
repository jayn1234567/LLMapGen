# MLLM_project

BEV road centerline reconstruction VLM based on a LLaVA-style multimodal stack.

Current working branch:

```text
unimapgen
```

## Stable Baseline

Baseline before reinforcement-learning work:

```text
tag:    baseline-before-rl-20260518
commit: 9ba4d59a2cd532bdf0fa050e7e11899af8d9edca
branch: unimapgen
note:   Stable supervised-training/inference baseline before adding RL/DPO/GRPO experiments.
```

Rollback or branch from this baseline:

```bash
git switch -c rl-debug-baseline baseline-before-rl-20260518
```

## Current Capabilities

This branch supports:

- Qwen2.5 / Qwen3-VL language backbones.
- DINOv2 / DINOv3 vision towers.
- Qwen3-VL-style DeepStack visual injection.
- Training with or without DeepStack.
- Full-parameter and LoRA training.
- Per-module learning rates for projector and vision tower.
- DeepSpeed ZeRO2/ZeRO3 training.
- Checkpoint metadata for Qwen multimodal checkpoints.
- Best checkpoint maintenance by training loss or eval loss.
- LoRA and full-parameter checkpoint loading for inference.
- Inference that recovers DeepStack settings from checkpoint metadata.
- Centerline geometric evaluation with buffer-IoU + Hungarian matching.
- Visualization of ground truth vs prediction after inference.
- Rank-0-only clean training logs with `DI_throughput: ... tokens/s/npu`.
- UniMapGen-style 256 patch state-update data flow for centerline and intersection prediction.

## Documentation

- Main architecture, training, inference, logging, and validation notes:
  [docs/qwen3vl_dinov3_deepstack.md](docs/qwen3vl_dinov3_deepstack.md)
- Script naming and placement:
  [scripts/README.md](scripts/README.md)
- UniMapGen reproduction plan:
  [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md)
- AV2 4096-to-256 patch processing notes:
  [DATASET_PATCH_PROCESSING.md](DATASET_PATCH_PROCESSING.md)
- State-update handover notes:
  [HANDOVER_STATE_UPDATE.md](HANDOVER_STATE_UPDATE.md)

## Main NPU Scripts

Full-parameter training:

```bash
bash scripts/train_full_dinov2_qwen3vl-8b_deepstack_npu.sh
bash scripts/train_full_dinov2_qwen3vl-8b_no-deepstack_npu.sh
bash scripts/train_full_dinov3_qwen3vl-8b_deepstack_npu.sh
bash scripts/train_full_dinov3_qwen3vl-8b_no-deepstack_npu.sh
```

The normal training scripts do not run validation and do not maintain best-loss checkpoints unless explicitly enabled.

Best checkpoint variants:

```bash
# Maintain output/best/ by lowest training loss after BEST_TRAIN_LOSS_START_STEP.
bash scripts/train_full_dinov3_qwen3vl-8b_deepstack_train_best_npu.sh

# Run a separate validation set every EVAL_STEPS and maintain output/eval_best/
# by lowest eval_loss.
bash scripts/train_full_dinov3_qwen3vl-8b_deepstack_eval_best_npu.sh
```

Do not pass experiment knobs as one-off shell prefixes. Edit the parameter block inside the target script instead, especially batch size, LR, epoch/step count, DeepStack, and best-checkpoint settings.

## Training Parameters

Most training scripts are shell wrappers around:

```bash
python -m llava.train.train_qwen
```

Core model/data parameters:

| Parameter | Purpose |
|---|---|
| `--model_name_or_path` | Qwen/Qwen3-VL base model or an existing checkpoint. |
| `--vision_tower` | DINOv2/DINOv3 vision tower path. |
| `--mm_vision_tower_type` | Optional explicit type: `dinov2` or `dinov3`. Usually inferred from metadata/path. |
| `--data_path` | Train json/jsonl path. |
| `--image_folder` | Root directory for train images. |
| `--eval_data_path` | Eval json/jsonl path, only needed when running eval. |
| `--eval_image_folder` | Root directory for eval images. |
| `--train_sample_limit` / `--eval_sample_limit` | Debug limits for small smoke runs. |

DeepStack parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `--disable_deepstack` | `True` | Raw Python entry disables DeepStack unless explicitly enabled. |
| `--deepstack_visual_indexes 6 12 18 23` | unset | ViT layers used for DeepStack. Fixed DeepStack scripts pass this explicitly. |
| `--input_image_size` | inferred | Override DINO input size. DINOv2-L defaults to 518, DINOv3-L/B to 224. |

Optimization parameters:

| Parameter | Purpose |
|---|---|
| `--learning_rate` | Main optimizer LR. |
| `--mm_projector_lr` | Optional separate LR for `mm_projector`. |
| `--mm_vision_tower_lr` | Optional separate LR for the vision tower. |
| `--weight_decay` | AdamW weight decay. |
| `--num_train_epochs` / `--max_steps` | Epoch-based or step-based training length. |
| `--per_device_train_batch_size` | Per-card batch size. Total batch is per-card batch x cards x gradient accumulation. |
| `--gradient_accumulation_steps` | Accumulates gradients to reach the intended total batch size. |
| `--lr_scheduler_type` | Scheduler, usually `cosine` or `constant` for debug. |
| `--warmup_ratio` / `--warmup_steps` | Warmup configuration. |
| `--gradient_checkpointing True` | Recommended for large full-parameter runs. |
| `--deepspeed scripts/deepspeed_zero2.json` | ZeRO2 training. |
| `--deepspeed scripts/deepspeed_zero3.json` | ZeRO3 training with gathered saves. |

LoRA parameters:

| Parameter | Purpose |
|---|---|
| `--lora_enable True` | Enable LoRA training. |
| `--lora_r` | LoRA rank. |
| `--lora_alpha` | LoRA alpha. |
| `--lora_dropout` | LoRA dropout. |

Best checkpoint parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `--save_best_train_loss` | `False` | Copy lower train-loss checkpoint to best dir. |
| `--best_train_loss_start_step` | `0` | Ignore train loss before this step. |
| `--best_train_loss_dir` | `best` | Output directory for best train-loss checkpoint. |
| `--save_best_eval_loss` | `False` | Copy lower eval-loss checkpoint to eval best dir. |
| `--best_eval_loss_dir` | `eval_best` | Output directory for best eval-loss checkpoint. |
| `--eval_strategy steps` | off by default | Required for eval-loss checkpointing. |
| `--eval_steps` | unset | Eval interval. Keep `save_steps` compatible with `eval_steps` if using HF best-model logic. |

Logging/output parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `--use_hf_progress_bar` | `False` in raw entry | Keep Hugging Face tqdm progress bar. Full scripts set it to `True`. |
| `--logging_steps` | script-specific | Metric logging interval. |
| `--save_steps` | script-specific | Normal checkpoint interval. |
| `--output_dir` | required | Run output directory. |

## Inference And Evaluation

Single/checkpoint inference:

```bash
python scripts/infer_centerline_checkpoint.py \
  --checkpoint-dir outputs/my_run/checkpoint-1000 \
  --test-json data/test.jsonl \
  --image-folder data/images \
  --prompt-mode dataset \
  --conv-template conv_qwen_3_Dinov2_huawei \
  --output-dir outputs/my_run/infer \
  --output-json outputs/my_run/infer/summary.json \
  --eval-centerline
```

State-update patch inference:

```bash
python scripts/infer_centerline_state_update.py \
  --checkpoint-dir outputs/my_run/best \
  --patch-json data/test.jsonl \
  --image-folder data/images \
  --output-json outputs/my_run/state_update_summary.json \
  --eval-centerline
```

Visualization with metrics:

```bash
python scripts/visualize_centerline.py \
  --input-dir outputs/my_run/infer \
  --image-folder data/images
```

When ground truth exists, visualization writes and prints `centerline_eval.json`. The metric backend is `infer_index/line_eval.py`, using LineString buffer IoU plus Hungarian matching. The default scale is `--eval-meter-per-pixel 0.2`.

Full-checkpoint testing:

```bash
bash scripts/test_full_dinov2_qwen3vl-8b_npu.sh
bash scripts/test_full_dinov3_qwen3vl-8b_npu.sh
```

The test scripts do not need a manual DeepStack flag. They infer whether DeepStack is enabled from the checkpoint configuration.

## Validation

The latest lightweight GPU matrix passed on two GPUs:

```text
qwen2.5 + dinov2 + deepstack on/off
qwen2.5 + dinov3 + deepstack on/off
qwen3vl-2b + dinov2 + deepstack on/off
qwen3vl-2b + dinov3 + deepstack on/off
```

The key `qwen3vl + dinov3 + deepstack` path was also checked separately:

```text
Loaded 754/754 model tensors from full-finetune checkpoint
DeepStack enabled
shape alignment passed
2-GPU training passed
2-GPU inference passed
no UNEXPECTED vision-weight load warning
```

## Training Console Log

The training console prints only step metrics, for example:

```text
time: 2026-05-13 15:43:14  global_step: 1  epoch: 1  loss: 1.23  learning_rate: 2e-05  DI_throughput: 12716.48 tokens/s/npu
```

Final runtime summaries and checkpoint events are written to log files only:

```text
train_metrics.log
eval_metrics.log
checkpoint_events.log
```
