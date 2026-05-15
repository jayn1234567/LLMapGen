# MLLM_project

BEV road centerline reconstruction VLM based on a LLaVA-style multimodal stack.

Current working branch:

```text
unimapgen
```

This branch supports:

- Qwen2.5 / Qwen3-VL language backbones.
- DINOv2 / DINOv3 vision towers.
- Qwen3-VL-style DeepStack visual injection.
- Training with or without DeepStack.
- LoRA and full-parameter checkpoint loading.
- Inference that recovers DeepStack settings from checkpoint metadata.
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

Useful environment overrides:

```bash
SAVE_BEST_TRAIN_LOSS=True
BEST_TRAIN_LOSS_START_STEP=3000
BEST_TRAIN_LOSS_DIR=best
EVAL_PATH=/path/to/eval.jsonl
EVAL_IMAGE_FOLDER=/path/to/images
EVAL_STEPS=300
BEST_EVAL_LOSS_DIR=eval_best
```

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
