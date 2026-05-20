# RL Roadmap

RL is a post-training stack that starts from an SFT checkpoint and SFT inference
outputs. It does not change the stable SFT, data processing, inference,
visualization, or state-update code paths.

## Target Architecture

```text
SFT checkpoint + SFT inference summary
    -> hard-sample pool
    -> actor worker computes multimodal prompt embeddings
    -> vLLM text-decoder rollout from prompt embeddings
    -> reward worker using infer_index geometry metrics
    -> GRPO actor update
    -> adapter checkpoint + final merged checkpoint
```

The formal rollout backend is `vllm_prompt_embeds`. HF-local generation is not a
training backend and must not be used as evidence that the RL architecture works.

The first supported training mode is no-DeepStack + LLM LoRA. DeepStack needs
layer-level visual residual injection, so it cannot be represented by prompt
embeddings alone.

## Current Priority

Continue SFT training first. When the dataset grows, missing or under-predicted
centerlines should be treated as a supervised-learning baseline issue before RL
is used. RL should start after the SFT checkpoint can mostly produce valid JSON
and reasonable line topology.

Use SFT inference summaries to identify hard samples for RL. Do not start RL
from a model that still has broad format failures, because invalid outputs only
receive the invalid reward and provide weak learning signal.

## Task Selection

Use `--map_task` to choose what the reward/parser should optimize:

- `lane`: current default. Reward only centerlines, cut type, and cut continuity.
- `lane_intersection`: future lane + intersection task. Intersection reward is
  enabled only in this mode.

There is no RL-specific Stage A / Stage B naming. Dataset names can stay as they
are for SFT/debug compatibility, but RL behavior is controlled by `--map_task`.

## Hard-Sample Pool

Build a pool from SFT inference summaries:

```bash
python scripts/rl/build_hard_pool.py \
  --summary outputs/my_sft_infer/summary.json \
  --source-jsonl data/my_dataset/train.jsonl \
  --output-dir data/rl_pool/lane \
  --map-task lane
```

The output keeps the SFT JSONL shape:

```text
id
image
meta
conversations[0]  prompt
conversations[1]  ground truth
```

RL audit data is stored under `meta.rl_pool`, including bucket, source summary,
parse status, centerline F1, reward components, and prediction preview.

For the current missing-line failure mode, add a future hard-sample bucket for
under-predicted centerlines. The bucket should be selected from inference/eval
records where the output is parseable but line coverage is low, for example low
`instance_recall`, low `length_recall`, or `pred_line_num < gt_line_num`.

## Reward Direction

The current reward already uses the same `infer_index.line_eval` matcher as
post-inference evaluation and includes `instance_f1` plus `length_f1`. This
penalizes missing lines because recall drops when matched predicted lines cover
only part of the ground truth.

If under-prediction remains after stronger SFT, add explicit recall/coverage
components to the GRPO reward:

- `centerline_instance_recall`: reward matching more GT line instances.
- `centerline_length_recall`: reward covering more GT centerline length.
- `under_prediction_penalty`: penalize parseable predictions where
  `pred_line_num < gt_line_num`.

Keep precision/F1 terms in the reward when adding recall terms. A pure count or
recall reward can push the model to hallucinate extra lines, so recall should be
balanced by precision, matched-length quality, and cut-continuity checks.

## vLLM Text Export

vLLM receives prompt embeddings, so it only needs the Qwen text decoder weights.
The GRPO entry exports this automatically to `output_dir/vllm_text_model` when
`--vllm_model_path` is not provided. Manual export:

```bash
python scripts/rl/export_text_decoder_for_vllm.py \
  --checkpoint outputs/my_sft/checkpoint-1000 \
  --output-dir outputs/my_sft/checkpoint-1000-vllm-text
```

This export supports no-DeepStack checkpoints only.

## Environment

Use the dedicated `unimapgen` conda environment for RL. The local GPU debug path
has been validated with:

- `torch==2.7.0+cu126`
- `vllm==0.9.2`
- `ray==2.55.1`
- `transformers==4.56.2`
- `huggingface-hub==0.36.2`

`vllm==0.9.2` is used because this branch requires prompt-embedding rollout and
online LoRA support. Older vLLM versions without prompt embeddings are not
suitable for the formal GRPO path.

## Training

Debug example:

```bash
bash scripts/gpu/train_grpo_dinov3_qwen3vl_nodeepstack_vllm_debug_gpu.sh
```

Important parameters:

| Parameter | Purpose |
|---|---|
| `--rollout_backend vllm_prompt_embeds` | Required formal rollout path. |
| `--map_task lane` | Current lane-only reward/parser mode. |
| `--map_task lane_intersection` | Future lane+intersection mode. |
| `--vllm_model_path` | Optional pre-exported text-decoder checkpoint. |
| `--actor_num_gpus` | Ray GPU allocation for HF actor training. |
| `--rollout_num_gpus` | Ray GPU allocation for vLLM rollout. |
| `--num_generations` | GRPO group size; must be at least 2. |
| `--kl_beta` | KL penalty against the adapter-disabled SFT reference. |

## Checkpoints

GRPO saves LoRA checkpoints for resume and final inference export:

- `checkpoint-*` and `final`: adapter checkpoint plus non-LoRA trainables and
  multimodal metadata.
- `best_reward/`: best adapter checkpoint by mean reward.
- `merged/`: final SFT+LoRA merged full checkpoint, written after training.

Use manual merge when exporting a specific adapter checkpoint:

```bash
python scripts/rl/export_merged_lora_checkpoint.py \
  --adapter-checkpoint outputs/my_grpo/best_reward \
  --model-base outputs/my_sft/checkpoint-1000 \
  --vision-tower /path/to/dinov3-vitl16 \
  --input-image-size 512 \
  --output-dir outputs/my_grpo/best_reward_merged \
  --bf16
```
