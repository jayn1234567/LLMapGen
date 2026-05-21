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

## Current Implementation

The code implements a Ray-style separation of roles:

| Role | Code | Responsibility |
|---|---|---|
| Entry | `mllm/train/train_grpo.py` | Parse RL args, reject unsupported modes, start coordinator. |
| Coordinator | `mllm/rl/grpo_trainer.py` | Launch Ray actor/rollout/reward workers and drive the GRPO loop. |
| Actor | `mllm/rl/grpo_trainer.py::ActorWorker` | Load multimodal policy, compute DINO/projector/Qwen prompt embeddings, update trainable weights. |
| Rollout | `mllm/rl/rollout.py::VLLMPromptEmbedRolloutWorker` | Use vLLM `enable_prompt_embeds=True` to sample text completions from actor-provided prompt embeddings. |
| Reward | `mllm/rl/grpo_trainer.py::RewardWorker` | Score completions using map JSON parsing plus metric-aligned rewards. |
| Reward metric | `mllm/reward/map_reward.py` | Convert normalized coordinates to pixels and call `infer_index.line_eval.evaluate_records()`. |
| Export | `mllm/rl/export.py` | Export Qwen text decoder for vLLM and merge LoRA checkpoints into full checkpoints. |

One training step is:

1. Actor reads SFT-format JSONL and images, then computes multimodal prompt
   embeddings from the current policy.
2. If LoRA is enabled, actor saves a temporary runtime LoRA adapter so vLLM can
   sample from the current policy state.
3. vLLM samples `num_generations` completions for each prompt from prompt
   embeddings.
4. Reward worker parses each completion and computes reward components.
5. Actor rebuilds full prompt+completion sequences, computes old/current log
   probabilities, normalizes advantages within each prompt group, and applies a
   clipped GRPO/PPO-style loss.
6. If `kl_beta > 0`, LoRA actors compute reference logprobs by temporarily
   disabling the adapter, so the reference is the SFT base policy.
7. The actor saves adapter checkpoints, `best_reward/`, and finally exports a
   `merged/` full checkpoint when `export_merged_checkpoints=True`.

This keeps the project-specific vision side in the HF actor and uses vLLM only
for high-throughput text-decoder rollout. It avoids writing a full custom vLLM
multimodal model plugin while still using the rollout backend intended for
large-scale RL.

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

Current reward components:

| Component | Default weight | Notes |
|---|---:|---|
| `format` | 0.08 | Valid task JSON receives format credit. Invalid JSON receives `invalid_reward`. |
| `centerline_instance_f1` | 0.37 | Main instance-level metric from `infer_index.line_eval`. |
| `centerline_length_f1` | 0.45 | Main length-coverage metric from `infer_index.line_eval`. |
| `cut_type` | 0.05 | Checks predicted `cut/inside` endpoint labels against GT order. |
| `cut_continuity` | 0.05 | Checks cut endpoints lie on patch boundary after pixel conversion. |
| `intersection` | 0.0 by default | Forced to zero for `--map_task lane`; enabled only for lane+intersection tasks. |

For lane-only training, intersection reward does not affect optimization. The
parser/reward mode is selected by `--map_task lane`. When switching to
`--map_task lane_intersection`, set a nonzero `--reward_intersection_weight`
only after the SFT model can produce stable lane+intersection JSON.

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
suitable for the formal GRPO path. Ascend NPU runs use the same backend through
vLLM-Ascend; the NPU scripts install `vllm-ascend==0.9.2rc1` and force reinstall
the OBS `torch_npu-2.7.1.dev20250724` wheel used by the SFT NPU scripts.

## Training

Debug example:

```bash
bash scripts/gpu/train_grpo_dinov3_qwen3vl_nodeepstack_vllm_debug_gpu.sh
```

Important parameters:

| Parameter | Purpose |
|---|---|
| `--rollout_backend vllm_prompt_embeds` | Required formal rollout path. |
| `--device_backend auto/cuda/npu` | Ray worker device placement. Use `npu` for vLLM-Ascend. |
| `--map_task lane` | Current lane-only reward/parser mode. |
| `--map_task lane_intersection` | Future lane+intersection mode. |
| `--vllm_model_path` | Optional pre-exported text-decoder checkpoint. |
| `--actor_num_gpus` | Ray GPU allocation for HF actor training. |
| `--rollout_num_gpus` | Ray GPU allocation for vLLM rollout. |
| `--actor_npu_devices` | Ascend visible devices for the HF actor worker, for example `0`. |
| `--rollout_npu_devices` | Ascend visible devices for vLLM-Ascend rollout, for example `1` or `1,2`. |
| `--num_generations` | GRPO group size; must be at least 2. |
| `--kl_beta` | KL penalty against the adapter-disabled SFT reference. |
| `--swanlab_enable True` | Enable SwanLab tracking for GRPO config and scalar metrics. |
| `--swanlab_project` | Shared project name. Current scripts default to `unimapgen_v3`. |
| `--swanlab_workspace` | Optional SwanLab workspace/org. |
| `--swanlab_experiment_name` | Run name. |
| `--swanlab_group` | Group for related runs, for example `grpo_phase_a_lane_dinov3_nodeepstack`. |
| `--swanlab_job_type` | Job category, usually `grpo` or `grpo_debug`. |

Coordinate parameters:

| Parameter | Purpose |
|---|---|
| `--coord_mode auto` | Reads `meta.coord_mode` from SFT JSONL. New datasets use `norm1000`. |
| `--coord_range 1000` | Normalized coordinate range when `coord_mode=norm1000`. |
| `--patch_size 256` | Original patch size used for pixel conversion and boundary checks. |
| `--meter_per_pixel 0.2` | Scale used by `infer_index` length/buffer metrics. |

The model may generate normalized `0..1000` coordinates, but reward geometry is
computed after conversion back to patch pixels. This matches visualization and
post-inference evaluation.

SwanLab monitoring:

- `mllm.train.train_grpo` initializes one SwanLab run in the coordinator.
- `grpo_run_config.json` content is also sent as the SwanLab config.
- Keep GRPO runs under `SWANLAB_PROJECT=unimapgen_v3`; separate phase/task,
  backbone, and debug/formal runs with `SWANLAB_GROUP`, `SWANLAB_JOB_TYPE`, and
  tags.
- Each GRPO step logs `reward_mean`, `reward_min`, `reward_max`, `loss`,
  `policy_loss`, `approx_kl`, `clip_fraction`, `action_tokens`, and `lr`.
- Reward diagnostics are also logged. `reward/parse_ok_rate` tracks format
  validity; `reward_component/*` contains parseable component averages such as
  instance/length precision, recall, and F1; `reward_count/*` contains
  GT/pred/matched line counts, missing/extra line counts, under-prediction rate,
  and intersection counts; `rollout/completion_tokens_*` and `reward/group_*`
  monitor generation length and GRPO group reward variance.
- Final checkpoint information is logged after `final/` and `merged/` are saved.

## Checkpoints

GRPO saves LoRA checkpoints for resume and final inference export:

- `checkpoint-*` and `final`: adapter checkpoint plus non-LoRA trainables and
  multimodal metadata.
- `best_reward/`: best adapter checkpoint by mean reward.
- `merged/`: final SFT+LoRA merged full checkpoint, written after training.

Recommended continuation policy:

- Use `merged/` when you want a single self-contained checkpoint for inference,
  SFT continuation, or another RL run.
- Use adapter checkpoints (`checkpoint-*`, `final`, `best_reward/`) when you
  want to resume exactly from an RL adapter. The adapter path must contain
  `adapter_config.json`; if the base checkpoint cannot be recovered from that
  file, pass `--model_base`.
- SFT LoRA continuation does not depend on the checkpoint directory name. It is
  controlled by `--lora_enable True`.

Verified smoke path:

```text
SFT checkpoint -> GRPO LoRA -> merged/ -> SFT LoRA continuation -> inference/eval table
```

The verified run was a tiny GPU0,2 smoke test, not a quality claim. It confirms
the checkpoint/load/save path, not that the reward has improved the model.

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
