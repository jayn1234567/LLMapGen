# Qwen3-VL + DINOv2/DINOv3 + DeepStack Notes

This document records the current behavior of branch `unimapgen`.

## Scope

The branch is for BEV road centerline reconstruction with:

- Qwen2.5 or Qwen3-VL as the language model.
- DINOv2 or DINOv3 as the vision tower.
- Optional DeepStack visual injection.
- Full-parameter or LoRA training.
- Checkpoint-driven inference that avoids manual DeepStack flags.
- Best checkpoint maintenance by training loss or eval loss.
- Centerline geometry evaluation after inference/visualization.
- UniMapGen-style patch/state-update inference for centerline and intersection maps.

## Model Data Flow

Main visual path:

```text
image
  -> DINOv2/DINOv3 ViT
  -> selected main ViT layer
  -> mm_projector
  -> replace <image> token positions in LLM embeddings
  -> LLM decoder
```

DeepStack path:

```text
selected intermediate ViT layers
  -> independent DeepStack merger MLPs
  -> token-aligned visual residuals
  -> injected into early LLM decoder layers at image-token positions
```

The main visual embedding remains the normal image-token embedding. DeepStack adds extra visual residuals inside decoder layers; it does not append extra text tokens.

For the default DINOv3-L setup:

```text
DINOv3 layer 23 -> main visual feature -> mm_projector -> image-token embeddings
DINOv3 layers 6, 12, 18, 23 -> DeepStack mergers -> LLM early-layer residual injection
```

Each DeepStack merger is independent:

```text
LayerNorm(vit_dim) -> Linear(vit_dim, llm_dim) -> GELU -> Linear(llm_dim, llm_dim)
```

## DeepStack And Gradient Checkpointing

DeepStack injection is implemented through the decoder forward path, not through fragile forward hooks.

The intended combination is:

```text
DeepStack enabled
gradient_checkpointing True
DeepSpeed ZeRO3 supported
```

This avoids the previous failure modes:

- different tensor counts between checkpoint forward and recomputation
- backward-through-graph-a-second-time errors
- hook order issues under recomputation

## DINO Detection

The code prefers explicit metadata when available and otherwise infers the vision tower from the path.

Common paths:

```text
dinov2                  -> DINOv2
dinov3-vitl16           -> DINOv3 ViT-L
dinov3-vitb16           -> DINOv3 ViT-B
```

Default image sizes:

```text
DINOv2-L      518
DINOv3-L      224
DINOv3-B      224
```

Training and inference derive the DINO type from checkpoint metadata, `mm_vision_tower_type`, or the `vision_tower` path. Vision tower paths should include a recognizable DINO key or alias such as `dinov3-vitl16`.

## Training Modes

Full-parameter NPU scripts:

```bash
bash scripts/train_full_dinov2_qwen3vl-8b_deepstack_npu.sh
bash scripts/train_full_dinov2_qwen3vl-8b_no-deepstack_npu.sh
bash scripts/train_full_dinov3_qwen3vl-8b_deepstack_npu.sh
bash scripts/train_full_dinov3_qwen3vl-8b_no-deepstack_npu.sh
```

Non-full training scripts are under:

```text
scripts/npu/
scripts/gpu/
```

Filename meaning:

```text
full             train LLM, ViT, projector, and DeepStack modules
deepstack        enable DeepStack
no-deepstack     disable DeepStack
freeze-vit       freeze ViT, train LLM plus alignment modules
freeze-llm       freeze LLM, train ViT plus alignment modules
dinov2/dinov3    selected vision tower family
```

The raw Python training entry defaults to DeepStack disabled. DeepStack fixed scripts explicitly pass `--disable_deepstack False` together with `--deepstack_visual_indexes ...`; non-DeepStack align scripts leave `DEEPSTACK_VISUAL_INDEXES` empty unless the caller overrides it.

## Training Parameter Reference

The canonical entrypoint is:

```bash
python -m llava.train.train_qwen
```

Model and data:

```text
--model_name_or_path          Qwen/Qwen3-VL base model or existing checkpoint.
--vision_tower                DINOv2/DINOv3 checkpoint path.
--mm_vision_tower_type        Optional explicit dinov2/dinov3 override.
--version                     Conversation template, e.g. conv_qwen_3_Dinov2_huawei.
--data_path                   Training json/jsonl path.
--image_folder                Training image root.
--eval_data_path              Eval json/jsonl path when using eval.
--eval_image_folder           Eval image root.
--train_sample_limit          Debug-only sample cap.
--eval_sample_limit           Debug-only eval sample cap.
```

DeepStack and vision:

```text
--disable_deepstack           Defaults to True in the raw Python entry.
--deepstack_visual_indexes    ViT layers for DeepStack, e.g. 6 12 18 23.
--input_image_size            Optional override; otherwise inferred from DINO type.
--mm_vision_select_layer      Main ViT feature layer.
--mm_projector_type           Usually mlp2x_gelu.
--unfreeze_mm_vision_tower    Train/freeze the vision tower.
```

Optimization:

```text
--learning_rate               Main LR.
--mm_projector_lr             Optional projector LR.
--mm_vision_tower_lr          Optional vision tower LR.
--weight_decay                AdamW weight decay.
--num_train_epochs            Epoch-based training length.
--max_steps                   Step-based training length, overrides epochs.
--per_device_train_batch_size Per-card batch size.
--gradient_accumulation_steps Total batch multiplier.
--lr_scheduler_type           cosine/constant/etc.
--warmup_ratio                Warmup fraction.
--gradient_checkpointing      Recommended True for large runs.
--bf16                        Preferred dtype on supported GPU/NPU.
--deepspeed                   DeepSpeed config path.
```

LoRA:

```text
--lora_enable True
--lora_r 8
--lora_alpha 16
--lora_dropout 0.05
```

Best checkpoint controls:

```text
--save_best_train_loss True
--best_train_loss_start_step 3000
--best_train_loss_dir best
--save_best_eval_loss True
--best_eval_loss_dir eval_best
--eval_strategy steps
--eval_steps 300
```

Best checkpoints are copied from fully written `checkpoint-*` directories. Under DeepSpeed, the best directory is the checkpoint directory copy, not a separate manual weight merge. LoRA best checkpoints include `adapter_model.safetensors`, `non_lora_trainables.bin`, `config.json`, and `qwen_multimodal_checkpoint.json`.

Logging:

```text
--use_hf_progress_bar True    Keep tqdm progress bar; full scripts default to this.
--logging_steps              Metric interval.
--save_steps                 Normal checkpoint interval.
--output_dir                 Run output directory.
```

The current Transformers version in `fastvlm` uses `--eval_strategy`, not `--evaluation_strategy`. The eval-best script detects this automatically.

## Checkpoint Metadata

Every normal checkpoint and final output writes:

```text
qwen_multimodal_checkpoint.json
config.json
```

The metadata includes:

```text
mm_vision_tower
vision_tower
mm_vision_tower_type
input_image_size
deepstack_visual_indexes
disable_deepstack
bundled_vision_tower
```

For DeepStack runs, `deepstack_visual_indexes` is synchronized from the actual runtime vision tower, not only from CLI arguments. This matters because the vision tower builder may fill default layers such as `[6, 12, 18, 23]`.

## Inference Behavior

Inference does not need the user to manually say whether a checkpoint used DeepStack.

The loader checks checkpoint metadata and config files, including:

```text
qwen_multimodal_checkpoint.json
config.json
adapter_config.json
non_lora_trainables.bin
model.safetensors / pytorch_model.bin
```

Behavior:

- DeepStack checkpoint -> inference enables DeepStack automatically.
- No-DeepStack checkpoint -> inference disables DeepStack automatically.
- LoRA checkpoint -> loads adapter weights plus `non_lora_trainables.bin`.
- Full checkpoint -> loads full tensors directly.

`llava_checkpoint.json` is treated as legacy metadata only. Qwen checkpoints should use the Qwen multimodal loading path.

## Inference And Evaluation

Checkpoint inference:

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

State-update inference:

```bash
python scripts/infer_centerline_state_update.py \
  --checkpoint-dir outputs/my_run/best \
  --patch-json data/test.jsonl \
  --image-folder data/images \
  --output-json outputs/my_run/state_update_summary.json \
  --eval-centerline
```

Visualization:

```bash
python scripts/visualize_centerline.py \
  --input-dir outputs/my_run/infer \
  --image-folder data/images
```

When records contain `ground_truth`, visualization automatically writes and prints `centerline_eval.json`. Use `--no-eval-centerline` to disable this.

The metric backend is `infer_index/line_eval.py`:

```text
LineString buffer IoU
Hungarian matching
instance precision/recall/F1
length precision/recall/F1
valid format ratio
```

The default metric scale is `--eval-meter-per-pixel 0.2`, matching `infer_index/param.py`.

## Test Scripts

NPU full-checkpoint test scripts:

```bash
bash scripts/test_full_dinov2_qwen3vl-8b_npu.sh
bash scripts/test_full_dinov3_qwen3vl-8b_npu.sh
```

They only distinguish DINOv2 vs DINOv3. They do not distinguish DeepStack vs no-DeepStack, because that is recovered from checkpoint metadata.

## Logging

Normal stdout is rank-0 only by default:

```bash
LLAVA_LOG_RANK0_ONLY=1
```

Direct `python -m llava.train.train_qwen` runs print compact step metrics by default:

```text
time: 2026-05-13 15:43:14  global_step: 1  epoch: 1  loss: 1.23  learning_rate: 2e-05  DI_throughput: 12716.48 tokens/s/npu
```

Full training scripts set `USE_HF_PROGRESS_BAR=True` in the script parameter block, so they keep the Hugging Face tqdm progress bar by default. In that mode, the custom metric callback writes `train_metrics.log` and `eval_metrics.log`, and also prints the step metric line with `DI_throughput` through `tqdm.write(...)`. Edit `USE_HF_PROGRESS_BAR=False` inside the script if compact step lines without tqdm are preferred.

Compact mode does not print Hugging Face dict logs such as:

```text
{'loss': ...}
```

The console also does not print checkpoint events or final runtime summaries. These are written to files:

```text
train_metrics.log
eval_metrics.log
checkpoint_events.log
```

`train_metrics.log` contains both step metrics and final runtime summaries. `checkpoint_events.log` contains `train_begin`, `save`, and `train_end` events.

## UNEXPECTED Load Warnings

There are two different cases.

Expected case:

```text
qwen2.5 base checkpoint: checkpoints/llava-fastvithd_1.5b_stage2
runtime vision tower: DINOv2 or DINOv3
```

This old base checkpoint contains FastViT / old projector tensors. When switching the runtime vision tower to DINO, those old visual keys can appear as `UNEXPECTED`. This does not mean the DINO checkpoint failed to load.

Problem case:

```text
Qwen3-VL + DINOv3 trained checkpoint
inference reports many DINO vision_tower tensors as UNEXPECTED
or does not report compatible tensor loading
```

That would indicate the wrong loading route or metadata. The current Qwen3-VL + DINOv3 path was checked and reports:

```text
Loaded 754/754 model tensors from full-finetune checkpoint
```

No DINO vision-weight `UNEXPECTED` warning appeared in that path.

## Validation

The latest lightweight two-GPU matrix passed:

```text
qwen2.5 + dinov2 + deepstack on
qwen2.5 + dinov2 + deepstack off
qwen2.5 + dinov3 + deepstack on
qwen2.5 + dinov3 + deepstack off
qwen3vl-2b + dinov2 + deepstack on
qwen3vl-2b + dinov2 + deepstack off
qwen3vl-2b + dinov3 + deepstack on
qwen3vl-2b + dinov3 + deepstack off
```

Validation script:

```bash
GPU_IDS=1,2 \
OUTPUT_ROOT=/tmp/mllm_lora_matrix_multigpu_debug_latest2 \
MAX_STEPS=1 \
TRAIN_SAMPLE_LIMIT=2 \
NUM_INFER_SAMPLES=2 \
MODEL_MAX_LENGTH=1536 \
bash scripts/debug_lora_matrix_multigpu.sh
```

Key Qwen3-VL + DINOv3 + DeepStack validation:

```bash
GPU_IDS=1,2 \
OUTPUT_ROOT=/tmp/mllm_qwen3vl_dinov3_deepstack_no_brace_latest \
MAX_STEPS=1 \
TRAIN_SAMPLE_LIMIT=2 \
NUM_INFER_SAMPLES=2 \
MODEL_MAX_LENGTH=1536 \
bash scripts/debug_deepstack_qwen3vl_multigpu.sh
```

Observed checks:

```text
Loaded 754/754 model tensors from full-finetune checkpoint
DEBUG_CHECK deepstack_count_ok
DEBUG_CHECK visual_token_alignment_ok
DEBUG_CHECK hidden_size_alignment_ok
DEBUG_CHECK inference_rank_outputs_ok
DEBUG_CHECK prompt_image_token_ok
```

## Operational Notes

- Use `scripts/deepspeed_zero3.json` when training needs ZeRO3 and gathered weights during save.
- Keep `gradient_checkpointing True` enabled by default.
- Use no-DeepStack scripts only for ablation or when intentionally training a base ViT-projector-LLM model.
- For NPU jobs, prefer enough cards/nodes so ZeRO3 save can gather weights without memory pressure.
- If a run fails before the first training step, `train_metrics.log` is still created at `train_begin`.
