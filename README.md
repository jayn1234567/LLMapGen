# Generic MLLM Framework

Generic Qwen-centered multimodal training and inference framework. The current
project task is BEV road centerline / intersection reconstruction, but framework
code is kept task-neutral where possible.

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
- 256 patch state-update data flow for centerline and intersection prediction.
- GRPO-style RL finetuning with map-task rewards for `lane` and `lane_intersection`.

## Latest Flow Audit

2026-05-19 audit covered both LoRA and full-parameter paths:

- LoRA ZeRO3 SFT on GPU0,2: train 1 step, save checkpoint, reload checkpoint, run inference.
- Full-parameter ZeRO3 SFT on GPU0,2: train 1 step, save `model.safetensors`, reload checkpoint, run inference.
- State-update inference: real model-prediction path and GT dry-run path both checked.
- GRPO ZeRO3 LoRA: lane and lane+intersection debug scripts use adapter-disabled reference KL with `KL_BETA=0.02`.

Fixes from this audit:

- Inference now decodes completion tokens only when `generate()` returns prompt+completion, so prompt JSON is not parsed as prediction.
- LoRA inference now loads compatible base checkpoint tensors, including projector/vision tensors when the configured vision tower matches.
- LoRA saves now include tokenizer files in both SFT and GRPO paths.
- `lora_bias=lora_only` state extraction was fixed.
- GRPO ZeRO3 LoRA now supports reference KL by temporarily disabling LoRA adapters on the same DeepSpeed-wrapped policy. Full-parameter ZeRO3 still raises early when `kl_beta > 0`.
- Debug/test scripts pass `--map-task lane` or `--map-task lane_intersection` explicitly.

## Documentation

- Project file tree and where to start:
  [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- Main architecture, training, inference, logging, and validation notes:
  [docs/qwen3vl_dinov3_deepstack.md](docs/qwen3vl_dinov3_deepstack.md)
- Script naming and placement:
  [scripts/README.md](scripts/README.md)
- Project reproduction plan:
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
bash scripts/npu/train_sft_dinov2_qwen3vl-8b_nodeepstack_npu.sh
bash scripts/npu/train_sft_dinov2_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh
bash scripts/npu/train_full_dinov3_qwen3vl-8b_deepstack_npu.sh
bash scripts/npu/train_full_dinov3_qwen3vl-8b_no-deepstack_npu.sh
```

The normal training scripts do not run validation and do not maintain best-loss checkpoints unless explicitly enabled.
The `no-deepstack` training scripts are standalone scripts. They pass
`--disable_deepstack True` directly and do not delegate to the corresponding
`deepstack` script.

Current full-parameter Qwen3VL-8B + DINO recipe:

| Setting | Value |
|---|---:|
| Global batch | 128 |
| Per-device batch | 4 |
| Epochs | 6 |
| Learning rate | 2e-5 |
| Projector LR | 2e-5 |
| Weight decay | 0.0 |
| Scheduler | cosine |
| Warmup | ratio 0.03 |

At global batch 128, 110k samples for 6 epochs is about 5156 optimizer steps
and 155 warmup steps. 330k samples for 6 epochs is about 15469 optimizer steps
and 465 warmup steps. The scripts compute gradient accumulation from the target
global batch and print the actual global batch at startup.

For the first 330k-sample run, prefer
`scripts/npu/train_sft_dinov2_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh`.
It uses 3 epochs and separate module LRs: LLM `2e-5`, projector `2e-5`,
vision tower `2e-6`. It evaluates during training and copies the lowest
`eval_loss` checkpoint to `eval_best/`.

Best checkpoint variants:

```bash
# Maintain output/best/ by lowest training loss after BEST_TRAIN_LOSS_START_STEP.
bash scripts/npu/train_full_dinov3_qwen3vl-8b_deepstack_train_best_npu.sh

# Run a separate validation set every EVAL_STEPS and maintain output/eval_best/
# by lowest eval_loss.
bash scripts/npu/train_full_dinov3_qwen3vl-8b_deepstack_eval_best_npu.sh
```

Do not pass experiment knobs as one-off shell prefixes. Edit the parameter block inside the target script instead, especially batch size, LR, epoch/step count, DeepStack, and best-checkpoint settings.

## Training Parameters

Most training scripts are shell wrappers around:

```bash
python -m mllm.train.train_qwen
```

New SFT scripts can also use the neutral alias:

```bash
python -m mllm.train.train_sft
```

`config.json` keeps the base language model identity in `model_type`; for
example Qwen3 checkpoints save `model_type: "qwen3"`, not a project-specific
name. Multimodal framework details are stored in `qwen_multimodal_checkpoint.json`
and normal config fields such as `mm_vision_tower`, `mm_vision_tower_type`,
`deepstack_visual_indexes`, and `input_image_size`.

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
| `--input_image_size` | inferred | Override DINO input size. DINOv2-L defaults to 518; DINOv3 registry defaults to 224, while project DINOv3 scripts pass 512 for 1024 visual tokens on 256x256 BEV patches. |

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

LoRA module selection:

| Parameter | Purpose |
|---|---|
| `--lora_target_scope` | Comma-separated module groups: `llm`, `projector`, `vision`, `deepstack`, `all`. |
| `--lora_target_modules` | Optional exact module-name override. If set, scope auto-detection is skipped. |
| `--lora_exclude_modules` | Comma-separated filters to exclude, defaulting to `lm_head,embed_tokens`. |

## GRPO

Chinese details are in `docs/grpo_中文说明.md`.

Two NPU GRPO script templates are provided:

```bash
bash scripts/npu/train_grpo_dinov2_qwen3vl-8b_lora_nodeepstack_npu.sh
bash scripts/npu/train_grpo_dinov3_qwen3vl-8b_lora_nodeepstack_auto_lane_npu.sh
bash scripts/npu/train_grpo_dinov3_qwen3vl-8b_lora_nodeepstack_auto_lane_intersection_npu.sh
```

GPU debug scripts for the current priority path, DINOv2 + Qwen3VL + no DeepStack:

```bash
bash scripts/gpu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_gpu.sh
bash scripts/gpu/train_grpo_debug_lane_intersection_dinov2_qwen3vl_nodeepstack_gpu.sh
bash scripts/gpu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_deepspeed_gpu.sh
bash scripts/gpu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh
bash scripts/gpu/train_grpo_debug_lane_intersection_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh
```

The main GRPO entrypoint is:

```bash
python -m mllm.train.train_grpo
```

This is a compatibility wrapper. The actual RL implementation lives in
`mllm/train/rl/grpo.py` and can also be launched as:

```bash
python -m mllm.train.rl.grpo
```

Important script-local parameters:

| Parameter | Purpose |
|---|---|
| `TRAINING_BRANCH` | Optional branch check: `auto_lane`, `auto_lane_intersection`, or one of the four A/B task combinations. |
| `MAP_TASK` | `lane` for centerlines+cut, `lane_intersection` for centerlines+intersection polygons. |
| `NUM_GENERATIONS` | Number of sampled candidates per prompt for group-relative rewards. |
| `KL_BETA` | Frozen SFT reference-model KL penalty weight. |
| `LORA_TARGET_SCOPE` | Which model parts receive LoRA adapters. |
| `REWARD_*_WEIGHT` | Reward component weights for format, centerline, cut, and intersection terms. |

`TRAINING_BRANCH` has two independent axes. Phase A/B controls whether incoming
state hints are present; `lane`/`lane_intersection` controls the output schema.
Valid strict branches are `phase_a_lane`, `phase_b_lane`,
`phase_a_lane_intersection`, and `phase_b_lane_intersection`. The `auto_*`
branches only check task type and are useful for debug JSONL that does not carry
phase metadata. The training entry checks sample metadata when present and fails
fast if a script points at the wrong branch data.

`--grpo_backend custom` is the default production path. `--grpo_backend trl`
checks that TRL is installed, then uses the same image-aware MLLM adapter because
native TRL trainers do not know this project's `images/image_sizes` batch format.

DeepSpeed can be enabled through normal HF arguments, for example
`--deepspeed scripts/deepspeed_zero2.json` or `--deepspeed scripts/deepspeed_zero3.json`.
For custom GRPO ZeRO3 LoRA, no separate frozen reference model is loaded outside
the DeepSpeed engine. The reference policy is the same model with LoRA adapters
temporarily disabled, so ZeRO3 lane and lane+intersection debug scripts can keep
`KL_BETA=0.02`. Full-parameter ZeRO3 still requires `KL_BETA=0.0` until a
separate DeepSpeed-wrapped reference engine is added.

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

## Phase A / Phase B State Update

The patch data flow has two supervised stages:

| Phase | Purpose | Prompt hints |
|---|---|---|
| A | Single patch recognition, used to learn centerline/intersection JSON format and local geometry. | Incoming traces/intersections are empty. |
| B | State-update stitching, used to train with previous patch context. | Incoming lane traces and intersection hints are filled from left/top neighbors when available. |

Debug SFT scripts for DINOv2 + Qwen3VL + no DeepStack + ZeRO3:

```bash
bash scripts/gpu/train_sft_debug_phase_a_lane_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh
bash scripts/gpu/train_sft_debug_phase_b_lane_intersection_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh
```

Debug data builders:

```bash
python scripts/gpu/build_ab_debug_data.py --limit 20 --test-count 4
```

Generated debug files include:

```text
data/debug_phase_a_lane20/{train,test}.jsonl
data/debug_phase_b_lane20/{train,test}.jsonl
data/debug_phase_a_lane_intersection20/{train,test}.jsonl
data/debug_phase_b_lane_intersection20/{train,test}.jsonl
```

`scripts/infer_centerline_state_update.py` must use model predictions as the
next patch state during normal inference. For engineering verification only,
`--dry-run-prompts` can replay ground truth JSON to confirm stitching and hint
generation without depending on model quality.

## Coordinate Convention

Newly generated A/B datasets keep patch images at their original patch size
(`256x256` by default) but store JSON coordinates in `coord_mode=norm1000`.
That means model-visible points use a normalized `0..1000` grid over the
original patch, independent of whether DINOv2 resizes the image to 518 or a
future vision tower uses another input size.

Important behavior:

- Data processing defaults to `--coord-mode norm1000 --coord-range 1000`.
- Phase B incoming left/top hints use the same coordinate mode as targets; hints may be negative or above 1000 because they come from neighboring patches.
- Model outputs are parsed in the dataset coordinate mode, then converted back to pixel coordinates for state-update stitching, visualization, and `infer_index/line_eval.py`.
- Inference and test scripts expose `COORD_MODE=auto` and `COORD_RANGE=1000`; `auto` reads `meta.coord_mode` from JSONL and remains compatible with old pixel datasets.

Conversion formula:

```text
x_norm = round(x_pixel / (patch_width  - 1) * coord_range)
y_norm = round(y_pixel / (patch_height - 1) * coord_range)

x_pixel = round(x_norm / coord_range * (patch_width  - 1))
y_pixel = round(y_norm / coord_range * (patch_height - 1))
```

For the default `256x256` patch and `coord_range=1000`, `[0,0]` maps to
`[0,0]`, `[255,255]` maps to `[1000,1000]`, and `[128,128]` maps to
approximately `[502,502]`. Assistant targets and parsed model outputs are
clamped to the valid in-patch range; Phase B incoming hints are not clamped.

## Inference And Evaluation

Single/checkpoint inference:

```bash
python scripts/infer_centerline_checkpoint.py \
  --checkpoint-dir outputs/my_run/checkpoint-1000 \
  --test-json data/test.jsonl \
  --image-folder data/images \
  --prompt-mode dataset \
  --coord-mode auto \
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
  --coord-mode auto \
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
bash scripts/npu/test_dinov2_qwen3vl-8b_nodeepstack_npu.sh
bash scripts/npu/test_full_dinov3_qwen3vl-8b_npu.sh
```

The test scripts do not need a manual DeepStack flag. They infer whether DeepStack is enabled from the checkpoint configuration.
They infer directly on the dataset's prebuilt `test.jsonl`; `eval.jsonl` is produced during data processing at raw-sample level. `NUM_TEST_SAMPLES=0` means all final-test rows; use a positive value only for a debug subset.

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
