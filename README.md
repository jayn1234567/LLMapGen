# Generic MLLM Framework

Generic Qwen-centered multimodal training and inference framework. The current
project task is BEV road centerline / intersection reconstruction, but framework
code is kept task-neutral where possible.

Current working branch:

```text
unimapgen_v7
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
- Multi-vision MoE fusion for two or more vision towers, including DINOv2+DINOv3 token-level routing.
- Qwen3-VL-style DeepStack visual injection.
- Training with or without DeepStack.
- Full-parameter and LoRA training.
- Per-module learning rates for projector and vision tower.
- DeepSpeed ZeRO2/ZeRO3 training.
- Checkpoint metadata for Qwen multimodal checkpoints.
- Best checkpoint maintenance by training loss or eval loss.
- LoRA and full-parameter checkpoint loading for inference.
- Single-file and sharded checkpoint resolution/loading, including
  `model.safetensors.index.json` and `model-00001-of-00004.safetensors` style
  full-model shards.
- Inference that recovers DeepStack settings from checkpoint metadata.
- Centerline geometric evaluation with buffer-IoU + Hungarian matching.
- Visualization of ground truth vs prediction after inference.
- Evaluation-table printing after inference/visualization using the same
  `infer_index` metric format.
- Rank-0-only clean training logs with `DI_throughput: ... tokens/s/npu`.
- 256 patch state-update data flow for centerline and intersection prediction.
- Post-SFT GRPO training with Ray + vLLM prompt-embedding rollout, LoRA actor
  updates, metric-based rewards, adapter checkpoints, and merged full-checkpoint
  export.
- Optional SwanLab experiment tracking for SFT and GRPO hyperparameters, run
  config, and scalar metrics.

## Project Structure

The active package is `mllm/`. Names such as `llava` are legacy compatibility
history unless a script explicitly imports them.

```text
.
├── README.md                         # Main overview, recommended workflows, and current verified behavior.
├── AGENTS.md                         # Maintainer/agent working notes and branch conventions.
├── pyproject.toml                    # Python package metadata.
├── configs/                          # Shared DeepSpeed configs used by training scripts.
├── data_process/                     # Raw BEV image/GeoJSON/tar processing into SFT JSONL.
├── docs/                             # Detailed design docs: project structure, RL, DeepStack, handover.
├── infer_index/                      # Centerline metric backend: buffer-IoU + Hungarian matching.
├── mllm/                             # Active multimodal framework: model, SFT, RL, reward, coordinate utilities.
├── scripts/                          # Runnable data, train, inference, visualization, export, debug entrypoints.
├── model_export/                     # Model-component export helpers.
├── data/                             # Local debug/sample datasets; generated data, not framework source.
├── outputs/                          # Local experiment outputs; generated artifacts.
└── test_*.py                         # Lightweight legacy/debug smoke tests.
```

More detail is maintained in [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Latest Flow Audit

2026-05-20 GPU audit covered SFT, GRPO, inference, and state-update paths on
GPU0,2:

- Phase-A lane and lane+intersection debug data follow the same JSONL structure
  as generated data.
- DINOv2 and DINOv3 no-DeepStack SFT smoke paths were run for LoRA and
  full-parameter modes.
- LoRA ZeRO3 SFT on GPU0,2: train 1 step, save checkpoint, reload checkpoint,
  run patch inference.
- Full-parameter ZeRO3 SFT on GPU0,2: train 1 step, save full checkpoint,
  reload checkpoint, run patch inference.
- DINOv3 phase-B lane+intersection LoRA path was checked with state-update
  inference using model predictions as state.
- GRPO with Ray + vLLM prompt embeddings was run from SFT checkpoints for
  lane-only and lane+intersection debug data. Outputs include `final/`,
  `best_reward_candidates/`, and `merged/`.
- A GRPO `merged/` full checkpoint was then used as `MODEL_NAME_OR_PATH` for a
  second SFT LoRA smoke run on GPU0,2. The path name did not contain `lora`, and
  `--lora_enable True` still correctly added LoRA adapters.

Fixes from this audit:

- Inference now decodes completion tokens only when `generate()` returns prompt+completion, so prompt JSON is not parsed as prediction.
- LoRA inference now loads compatible base checkpoint tensors, including projector/vision tensors when the configured vision tower matches.
- LoRA saves now include tokenizer files in SFT paths.
- `lora_bias=lora_only` state extraction was fixed.
- Debug/test scripts pass `--map-task lane` or `--map-task lane_intersection` explicitly.
- Checkpoint config saving handles both dict-like and object-like configs.
- Inference, state-update inference, and visualization print the standard
  `Line Evaluation Results` table when centerline ground truth is available.

## Documentation

- Project file tree and where to start:
  [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- Main architecture, training, inference, logging, and validation notes:
  [docs/qwen3vl_dinov3_deepstack.md](docs/qwen3vl_dinov3_deepstack.md)
- Script naming and placement:
  [scripts/README.md](scripts/README.md)
- RL post-training roadmap:
  [docs/RL_ROADMAP.md](docs/RL_ROADMAP.md)
- Project reproduction plan:
  [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md)
- AV2 4096-to-256 patch processing notes:
  [DATASET_PATCH_PROCESSING.md](DATASET_PATCH_PROCESSING.md)
- State-update handover notes:
  [HANDOVER_STATE_UPDATE.md](HANDOVER_STATE_UPDATE.md)

## Main NPU Scripts

Full-parameter training:

```bash
bash scripts/npu/train/train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/train/train_sft_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/train/train_sft_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/train/train_sft_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh
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
`scripts/npu/train/train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`.
It uses 3 epochs and separate module LRs: LLM `2e-5`, projector `2e-5`,
vision tower `2e-6`. It evaluates during training and saves the lowest
`eval_loss` checkpoint directly to `eval_best_candidates/`; set `SAVE_BEST_INFER_INDEX=True` to
also run generation-based `infer_index` evaluation and keep the best
`length_f1` checkpoint in `infer_best_candidates/`.

Best checkpoint variants are controlled in the current NPU script parameter
blocks, not by legacy tmp wrappers. Normal `checkpoint-*` saving follows
`SAVE_STEPS` and `SAVE_TOTAL_LIMIT`. Train-loss best uses
`SAVE_BEST_TRAIN_LOSS`, `BEST_TRAIN_LOSS_START_STEP`, and
`BEST_TRAIN_LOSS_DIR`. Eval-loss best uses `ENABLE_EVAL`,
`SAVE_BEST_EVAL_LOSS`, `EVAL_STEPS`, and `BEST_EVAL_LOSS_DIR`. Infer-index
best uses `SAVE_BEST_INFER_INDEX`, `BEST_INFER_INDEX_METRIC`,
`BEST_INFER_INDEX_NUM_SAMPLES`, `BEST_INFER_INDEX_EVAL_STEPS`, and
`BEST_INFER_INDEX_DIR`. `BEST_INFER_INDEX_NUM_SAMPLES=0` uses the full eval
set and is the default for best-checkpoint selection.

Best checkpoints are create-only candidates: a metric improvement saves the
current model directly under `best_candidates/`, `eval_best_candidates/`, or
`infer_best_candidates/` and writes `_SUCCESS` last. They do not create a normal
`checkpoint-*` first, and the rotation path only deletes older validated
candidate directories with `rm -rf`; no rename or replace operation is used.

Do not pass experiment knobs as one-off shell prefixes. Edit the parameter block inside the target script instead, especially batch size, LR, epoch/step count, DeepStack, and best-checkpoint settings.

## Training Parameters

Most training shell scripts launch:

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
| `--vision_tower` | Single vision tower path, or comma-separated paths for multi-vision recipes. |
| `--mm_vision_tower_type` | Optional explicit type: `dinov2`, `dinov3`, `siglip`, `multi_moe`, or `multi_concat`. Usually inferred from metadata/path for single towers. |
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

Multi-vision recipes:

There are three clear visual-backbone families in scripts:

| Recipe | `mm_vision_tower_type` | `multi_vision_fusion` | Meaning |
|---|---|---|---|
| Original single tower | `dinov2`, `dinov3`, or `siglip` | unset | One encoder feeds the normal `mm_projector`. |
| Dynamic visual MoE | `multi_moe` | `softmax_router` | Per-token router learns how much to trust each encoder. |
| Prismatic-style concat | `multi_concat` | `concat_projector` | Align token grids, concatenate encoder channels, then project back before `mm_projector`. |

`multi_moe` and `multi_concat` both use the same extensible multi-vision tower. For DINO+SigLIP, the forward path also resizes and re-normalizes the already processed image tensor per expert, because DINO and SigLIP use different image statistics/input sizes.

| Parameter | Purpose |
|---|---|
| `--vision_tower path_a,path_b` or `--multi_vision_towers path_a,path_b` | Expert vision tower paths. Any tower supported by `build_vision_tower` can be used. |
| `--multi_vision_tower_types dinov2,dinov3` | Optional per-expert type override, e.g. `dinov2,siglip` or `dinov3,siglip`. |
| `--multi_vision_input_image_sizes 512,384` | Optional per-expert input sizes. Useful for DINO+SigLIP. |
| `--multi_vision_target_grid 32` | Align all experts to a square token grid before fusion. Defaults to the smallest expert grid. |
| `--multi_vision_hidden_size` | Fused hidden size before `mm_projector`. Defaults to the largest expert hidden size. |
| `--multi_vision_primary_index 0` | Which expert supplies the shared image processor. |
| `--multi_vision_fusion softmax_router` | Dynamic MoE fusion; each token gets a softmax weight over experts. |
| `--multi_vision_fusion concat_projector` | Static concat fusion; closest to the Prismatic DINO+SigLIP idea. |

MoE example:

```bash
--mm_vision_tower_type multi_moe \
--vision_tower "${DINO2_PATH},${DINO3_PATH}" \
--multi_vision_tower_types dinov2,dinov3 \
--input_image_size 512 \
--multi_vision_target_grid 32 \
--multi_vision_hidden_size 1024 \
--multi_vision_primary_index 1 \
--mm_vision_fusion_lr 2e-5
```

DINOv3 + SigLIP concat example:

```bash
--mm_vision_tower_type multi_concat \
--vision_tower "${DINO3_PATH},${SIGLIP_PATH}" \
--multi_vision_tower_types dinov3,siglip \
--multi_vision_input_image_sizes 512,384 \
--multi_vision_fusion concat_projector \
--multi_vision_target_grid 32 \
--multi_vision_hidden_size 1024 \
--multi_vision_primary_index 0 \
--mm_vision_fusion_lr 2e-5
```

Optimization parameters:

| Parameter | Purpose |
|---|---|
| `--learning_rate` | Main optimizer LR. |
| `--mm_projector_lr` | Optional separate LR for `mm_projector`. |
| `--mm_vision_fusion_lr` | Optional separate LR for multi-vision adapters/router/post-fusion/concat-projector layers. |
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

SwanLab monitoring:

| Parameter | Purpose |
|---|---|
| `--swanlab_enable True` | Enable SwanLab logging. Default scripts keep it off unless changed in the script parameter block. |
| `--swanlab_project` | Project name. Main scripts default to the shared project `unimapgen_v3`. |
| `--swanlab_workspace` | Optional SwanLab workspace/org. Leave empty for the account default. |
| `--swanlab_experiment_name` | Run name, usually including SFT/GRPO, data scale, phase, task, backbone, and DeepStack mode. |
| `--swanlab_group` | Group related runs inside `unimapgen_v3`, for example `sft_phase_a_lane_dinov2_nodeepstack` or `grpo_phase_b_lane_intersection_dinov3_nodeepstack`. |
| `--swanlab_job_type` | Job category for filtering, for example `sft`, `grpo`, `sft_debug`, or `grpo_debug`. |
| `--swanlab_tags` | Comma-separated run tags. |
| `--swanlab_mode` | Optional SwanLab mode, for example `cloud`, `offline`, `local`, or `disabled`. Leave empty for SwanLab default. |
| `--swanlab_log_dir` | Local SwanLab file directory. Current SFT scripts default to `${OUTPUT_PATH}/swanlab`; GRPO scripts default to `${OUTPUT_DIR}/swanlab`. |
| `--swanlab_api_host` / `--swanlab_web_host` | Optional private SwanLab server API and web URLs. Current scripts expose them as `SWANLAB_API_HOST` and `SWANLAB_WEB_HOST`. |

The provided SFT/GRPO scripts define `SWANLAB_API_KEY`, `SWANLAB_PROJECT`,
`SWANLAB_GROUP`, `SWANLAB_JOB_TYPE`, `SWANLAB_EXPERIMENT_NAME`,
`SWANLAB_MODE`, and `SWANLAB_LOG_DIR` in their parameter blocks. Keep
`SWANLAB_PROJECT=unimapgen_v3` to compare all runs in one project; use
group/job type/tags to separate SFT, GRPO, stage/task, backbone, and
debug/formal runs. For offline recording, set `SWANLAB_ENABLE=True` and
`SWANLAB_MODE=offline` or `local` inside the target script. Local SwanLab files
are written under the run output directory, beside `checkpoint-*`, `eval_best*`,
`best*`, `best_reward_candidates/`, and `merged/` directories. For private deployment, set
`SWANLAB_API_HOST` and `SWANLAB_WEB_HOST` in the same script parameter block.

## RL Post-Training

RL is a separate post-training stack and does not change the stable SFT/data
processing path:

```text
SFT checkpoint + SFT inference summary
    -> hard-sample pool
    -> actor worker computes multimodal prompt embeddings
    -> vLLM text-decoder rollout
    -> reward worker using infer_index metrics
    -> GRPO actor update
    -> adapter checkpoint + final merged checkpoint
```

Keep the SFT entrypoint as `python -m mllm.train.train_qwen` or
`python -m mllm.train.train_sft`. GRPO uses `python -m mllm.train.train_grpo`
with `--rollout_backend vllm_prompt_embeds`.

Current GRPO implementation:

- Actor role: loads the multimodal SFT checkpoint, runs DINO/projector/Qwen
  multimodal prompt preparation, and trains the policy.
- Rollout role: uses vLLM with `enable_prompt_embeds=True` to decode from the
  actor-computed multimodal prompt embeddings. vLLM only needs the Qwen text
  decoder export. CUDA runs use upstream vLLM; Ascend runs use vLLM-Ascend with
  the same `vllm_prompt_embeds` backend.
- Reward role: parses generated JSON and ground truth, converts normalized
  coordinates back to pixels, then reuses `infer_index.line_eval` for the main
  centerline geometry score.
- Update rule: grouped completions are scored, advantages are normalized within
  each prompt group, and the actor is updated with a clipped GRPO/PPO-style
  objective plus optional KL to the adapter-disabled SFT reference.
- Checkpoints: `checkpoint-*` and `final/` are adapter checkpoints, `best_reward_candidates/`
  tracks highest mean reward, and `merged/` is the exported full checkpoint for
  inference or later SFT/RL continuation.

Current supported RL mode is no-DeepStack + LoRA, with `lora_target_scope=llm`
as the safest default. DeepStack is intentionally rejected by the vLLM
prompt-embedding path because layer-level visual residual injection cannot be
represented as text-decoder prompt embeddings.

Current training priority is still SFT. If larger data still produces missing
or under-predicted centerlines, collect those parseable low-recall cases from
SFT inference summaries and use them later as RL hard samples. The future GRPO
reward should add explicit instance/length recall and under-prediction
penalties while keeping precision/F1 terms to avoid hallucinated extra lines.

RL task selection:

| Parameter | Purpose |
|---|---|
| `--map_task lane` | Current lane-only reward/parser mode. Intersection reward is forced off. |
| `--map_task lane_intersection` | Future lane+intersection reward/parser mode. |
| `--rollout_backend vllm_prompt_embeds` | Required formal rollout path. HF-local generation is not a training backend. |
| `--device_backend auto/cuda/npu` | Device placement for Ray actor/rollout workers. NPU uses vLLM-Ascend. |
| `--actor_npu_devices` | `ASCEND_RT_VISIBLE_DEVICES` for the actor worker, such as `0`. |
| `--rollout_npu_devices` | `ASCEND_RT_VISIBLE_DEVICES` for the vLLM-Ascend rollout worker, such as `1` or `1,2`. |
| `--vllm_model_path` | Optional pre-exported text-decoder checkpoint. If unset, GRPO exports one from the SFT checkpoint. |

GRPO SwanLab logging records the full run config plus step metrics such as
`reward_mean`, `loss`, `policy_loss`, `approx_kl`, `clip_fraction`,
`action_tokens`, and learning rate. It also logs reward diagnostics:
`reward/parse_ok_rate`, `reward_component/*` for parseable completions,
`reward_count/*` for GT/pred/matched line counts and under-prediction signals,
`rollout/completion_tokens_*`, and `reward/group_*` group-variance statistics.

LoRA parameters:

| Parameter | Purpose |
|---|---|
| `--lora_enable True` | Enable LoRA training. |
| `--lora_r` | LoRA rank. |
| `--lora_alpha` | LoRA alpha. |
| `--lora_dropout` | LoRA dropout. |
| `--lora_target_scope llm` | First supported RL mode; vLLM online LoRA rollout is LLM-only. |

GRPO checkpoint outputs:

| Output | Purpose |
|---|---|
| `checkpoint-*` / `final` | Adapter checkpoint for resume. |
| `best_reward_candidates/` | Successful best-reward adapter candidates. The latest `_SUCCESS` candidate is the current best mean reward. |
| `merged/` | Final SFT+LoRA merged full checkpoint for inference or second-stage training. |

LoRA path behavior:

- SFT training enters LoRA mode by `--lora_enable True`; the checkpoint path name
  does not need to contain `lora`.
- Inference and RL adapter loading should identify adapters by
  `adapter_config.json`.
- For a self-contained continuation path, prefer GRPO/SFT `merged/` as
  `--model_name_or_path`. Adapter-only checkpoints can also be used, but they
  need a valid `adapter_config.json` and base-model reference.

RL environment:

Use the dedicated `unimapgen` conda environment. The GPU debug path has been
validated with `torch==2.7.0+cu126`, `vllm==0.9.2`, `ray==2.55.1`,
`transformers==4.56.2`, and `huggingface-hub==0.36.2`.

Ascend NPU GRPO uses the same formal vLLM prompt-embedding rollout path through
vLLM-Ascend. The NPU scripts install `torch==2.7.1`, force reinstall the OBS
`torch_npu-2.7.1.dev20250724` wheel, and install `vllm==0.9.2` plus
`vllm-ascend==0.9.2rc1`.

Best checkpoint parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `--save_best_train_loss` | `False` | Save lower train-loss checkpoint directly to `best_candidates/`. |
| `--best_train_loss_start_step` | `0` | Ignore train loss before this step. |
| `--best_train_loss_dir` | `best` | Logical best name; rotating mode writes `best_candidates/`. |
| `--save_best_eval_loss` | `False` | Save lower eval-loss checkpoint directly to `eval_best_candidates/`. |
| `--best_eval_loss_dir` | `eval_best` | Logical best name; rotating mode writes `eval_best_candidates/`. |
| `--save_best_infer_index` | `False` | Run generation eval at eval steps and save the best infer_index checkpoint directly. |
| `--best_infer_index_metric` | `length_f1` | Metric from `infer_index/line_eval.py`; higher is better by default. |
| `--best_infer_index_dir` | `infer_best` | Logical best name; rotating mode writes `infer_best_candidates/`. |
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

Stage/task selection is different in the two inference tools:

- Stage A uses `scripts/tools/infer_centerline_checkpoint.py`; pass
  `--map-task lane` or `--map-task lane_intersection`.
- Stage B uses `scripts/tools/infer_centerline_state_update.py`; lane-only mode
  omits `--include-intersections`, while lane+intersection mode passes
  `--include-intersections`.

Formal multi-vision NPU entrypoints:

```bash
# SFT train examples
bash scripts/npu/train/train_sft_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/train/train_sft_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/train/train_sft_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh

# Test examples
bash scripts/npu/test/test_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/test/test_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/test/test_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh
```

Each multi-vision recipe has stage A/B and lane/lane_intersection variants. The
formal SFT multi-vision defaults are: eval off, best train loss on, `SAVE_STEPS=400`,
`SAVE_TOTAL_LIMIT=15`, `BEST_CHECKPOINT_KEEP_LIMIT=5`, `LR=2e-5`,
`NUM_EPOCHS=5`, SwanLab enabled with `SWANLAB_MODE=offline`.

Local debug entrypoints:

```bash
bash scripts/debug/train_sft_debug_npu.sh
bash scripts/debug/infer_debug_npu.sh
bash scripts/debug/train_grpo_debug_npu.sh
bash scripts/debug/run_debug_full_flow_npu.sh
```

The local NPU debug scripts sample a tiny split from
`/cache/data/data_line_samples_33w` into `checkpoints/debug_data/` and write
outputs under `checkpoints/debug/`. Set `DATASET_PHASE`, `MAP_TASK`, and
`VISION_BACKBONE` to switch phase/task/backbone. Supported debug backbones include
`dinov2`, `dinov3`, `multi_moe`, `dinov2_siglip_concat`, and
`dinov3_siglip_concat`. See
`scripts/debug/README.md` for the full command matrix.

SFT debug saves normal `checkpoint-*` by `SAVE_STEPS`/`SAVE_TOTAL_LIMIT`, keeps
`eval_best_candidates/` by default, and can enable best train loss with
`SAVE_BEST_TRAIN_LOSS=True`. GRPO debug uses vLLM-Ascend and tracks
`best_reward_candidates/` plus `merged/`. When SwanLab offline/local mode is enabled in a
debug script, local logs go to `checkpoints/debug/<run>/.../swanlab/`, which is
the same output directory level as those checkpoints.

`scripts/tools/infer_centerline_state_update.py` must use model predictions as the
next patch state during normal inference. For engineering verification only,
`--dry-run-prompts` can replay ground truth JSON to confirm stitching and hint
generation without depending on model quality.

Dataset patch retention is phase-aware. The raw image is still masked with
`patch_tif/0_edit_poly.tif` before patch generation, matching the legacy
`DatasetCreator` behavior, so fully black masked patches are skipped. After that,
empty-target downsampling is applied only where it is safe: `phase_a/train` uses
`--max-empty-ratio` by default, while `phase_a/eval`, `phase_a/test`, and all
`phase_b` splits keep every non-black masked patch by default. Keep
`--phase-b-max-empty-ratio -1` for normal B-stage training/eval/test so left/top
state-update chains and stitched maps remain complete.

## Coordinate Convention

Newly generated A/B datasets keep patch images at their original patch size
(`256x256` by default) but store JSON coordinates in `coord_mode=norm1000`.
That means model-visible points use a normalized `0..1000` grid over the
original patch, independent of whether DINOv2 resizes the image to 518 or a
future vision tower uses another input size.

Important behavior:

- Data processing defaults to `--coord-mode norm1000 --coord-range 1000`.
- Phase B incoming left/top hints use the same coordinate mode as targets; hints may be negative or above 1000 because they come from neighboring patches.
- Model outputs are parsed in the dataset coordinate mode, then converted back to pixel coordinates for state-update stitching, visualization, reward scoring, and `infer_index/line_eval.py`.
- Inference and test scripts expose `COORD_MODE=auto` and `COORD_RANGE=1000`; `auto` reads `meta.coord_mode` from JSONL and remains compatible with old pixel datasets.
- The metric result is therefore a pixel/meter-space evaluation, not a
  normalized-coordinate distance. `meter_per_pixel` controls the final conversion
  used by `infer_index`.

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
python scripts/tools/infer_centerline_checkpoint.py \
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
python scripts/tools/infer_centerline_state_update.py \
  --checkpoint-dir outputs/my_run/best \
  --patch-json data/test.jsonl \
  --image-folder data/images \
  --output-json outputs/my_run/state_update_summary.json \
  --coord-mode auto \
  --eval-centerline
```

For multi-card B-stage inference, use `--distributed-by-tile` under `torchrun`.
The script shards complete `tile_id` groups, so all patches from the same raw
sample stay on one rank and left/top state dependencies are preserved. Each rank
writes `summary_rank*.json`; rank0 merges them into the final `summary.json`,
`merged_global.json`, `eval.json`, and `whole_map_viz/` outputs. The NPU
`scripts/npu/test/test_stage_b_*_npu.sh` scripts enable this path by default.

Visualization with metrics:

```bash
python scripts/tools/visualize_centerline.py \
  --input-dir outputs/my_run/infer \
  --image-folder data/images
```

When ground truth exists, inference/visualization writes metric JSON and prints
the same table format from `infer_index.LineEvalRes.show_res()`, for example:

```text
==========================================================
                 Line Evaluation Results
==========================================================
Metric             Precision    Recall       F1
----------------------------------------------------------
Instance Level     0.0000       0.0000       0.0000
Length Level       0.0000       0.0000       0.0000
==========================================================
格式合法的推理结果占比: 0.0000(0/1)
```

The metric backend is `infer_index/line_eval.py`, using LineString buffer IoU
plus Hungarian matching. The default scale is `--eval-meter-per-pixel 0.2`.

Full-checkpoint testing:

```bash
bash scripts/npu/test/test_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh
bash scripts/npu/test/test_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh
```

The test scripts do not need a manual DeepStack flag. They infer whether DeepStack is enabled from the checkpoint configuration.
They infer directly on the dataset's prebuilt `test.jsonl`; `eval.jsonl` is produced during data processing at raw-sample level. `NUM_TEST_SAMPLES=0` means all final-test rows; use a positive value only for a debug subset.
The current NPU stage test scripts write `summary.json`, per-sample JSON under `json/`, single-patch PNGs under `viz/`, metrics in `eval.json`, and stitched whole-map PNGs under `whole_map_viz/`.

Checkpoint paths can point to a direct checkpoint directory, a normal
`checkpoint-*`, or a best-candidate root. The resolver accepts LoRA adapters,
single-file full checkpoints, and standard sharded full checkpoints:

```text
adapter_model.safetensors
model.safetensors
pytorch_model.bin
model.safetensors.index.json + model-00001-of-00004.safetensors ...
pytorch_model.bin.index.json + pytorch_model-00001-of-00004.bin ...
```

For training continuation through Transformers `from_pretrained`, keep the
index JSON with the shard files. The custom inference loader can also read bare
`model-*-of-*` shard files as a fallback, but the indexed HF format is the
recommended checkpoint layout.

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
