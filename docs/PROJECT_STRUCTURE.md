# Project Structure

This file is the quick map for new contributors. The active framework package is
`mllm/`; names such as `llava` are legacy compatibility/history unless a script
explicitly imports them.

## Top-Level Tree

```text
.
├── README.md                         # Main overview, verified flows, train/RL/infer usage.
├── AGENTS.md                         # Maintainer working notes and branch conventions.
├── pyproject.toml                    # Python package metadata and dependency hints.
├── configs/                          # Shared DeepSpeed config templates.
├── data_process/                     # Raw BEV/GeoJSON/tar -> normalized SFT JSONL.
├── docs/                             # Detailed design docs and handover notes.
├── infer_index/                      # Official line metric implementation used by eval/reward.
├── mllm/                             # Active framework code: model, SFT, RL, rewards, coordinates.
├── scripts/                          # Executable train/infer/eval/visualize/export/debug scripts.
├── model_export/                     # Model conversion/export helpers.
├── data/                             # Local debug/sample datasets; generated artifacts.
├── outputs/                          # Local training/inference outputs; generated artifacts.
├── test_*.py                         # Lightweight legacy/debug smoke tests.
└── UniMapGen.pdf                     # Local paper/reference artifact when present; not committed.
```

## Core Framework: `mllm/`

```text
mllm/
├── __init__.py                       # Lazy package exports; avoids importing model deps for utility-only imports.
├── constants.py                      # Shared token/constants.
├── conversation.py                   # Qwen/LLaVA-style conversation templates.
├── coord_utils.py                    # Pixel <-> norm1000 conversion and coord metadata helpers.
├── mm_utils.py                       # Image/token multimodal helper functions.
├── model/                            # Model loading and multimodal architecture.
├── reward/                           # Map JSON parser plus metric-aligned reward functions.
├── rl/                               # Ray + vLLM GRPO stack and export utilities.
├── train/                            # SFT training entrypoints and Trainer customizations.
└── serve/                            # Legacy/optional serving utilities.
```

Important subdirectories:

- `mllm/model/builder.py`: checkpoint loading for full, LoRA, and multimodal checkpoints.
- `mllm/model/llava_arch.py`: multimodal feature preparation, image token replacement, and DeepStack wiring.
- `mllm/model/language_model/`: Qwen2/Qwen3 multimodal language model wrappers.
- `mllm/model/multimodal_encoder/`: DINOv2/DINOv3/CLIP vision tower builders and encoders.
- `mllm/model/multimodal_projector/`: vision-to-LLM projector builders.
- `mllm/train/train_qwen.py`: main SFT training implementation.
- `mllm/train/train_sft.py`: SFT wrapper entrypoint.
- `mllm/train/llava_trainer.py`: Trainer subclass, checkpoint save behavior, grouped LR, best-loss callbacks.
- `mllm/train/checkpoint_metadata.py`: Qwen multimodal checkpoint metadata sync/write helpers.
- `mllm/reward/map_schema.py`: JSON schema extraction and task-specific validation for `lane` / `lane_intersection`.
- `mllm/reward/map_reward.py`: reward function that converts model coordinates to pixels and reuses `infer_index.line_eval`.
- `mllm/rl/config.py`: GRPO/RL dataclass arguments.
- `mllm/rl/data_pool.py`: hard-sample pool builder from SFT inference summaries.
- `mllm/rl/dataset.py`: SFT JSONL reader reused by RL training.
- `mllm/rl/grpo_trainer.py`: Ray-style GRPO coordinator, HF actor, vLLM rollout worker, and reward worker.
- `mllm/rl/rollout.py`: prompt-embedding rollout contracts and vLLM wrapper.
- `mllm/rl/modeling.py`: policy loading, LoRA/full-train scope selection, optimizer, checkpoint save.
- `mllm/rl/export.py`: text-decoder export for vLLM and LoRA merged-checkpoint export.
- `mllm/rl/schemas.py`: shared post-training data structures for pool/audit outputs.

## Data Processing: `data_process/`

```text
data_process/
├── README_DATA_PROCESSING.md          # Data format, A/B phase, cut, coordinate docs.
├── build_lane_dataset.py              # Generate lane-only A/B SFT JSONL.
├── build_lane_intersection_dataset.py # Generate lane+intersection A/B SFT JSONL.
├── state_update_dataset_common.py     # Shared raw sample discovery, patching, cut, hint generation.
├── debug_intersection_parse.py        # Inspect lane/intersection GeoJSON parsing for a raw sample.
├── convert_cloud_format.py            # Cloud-format conversion utilities.
├── dataset_creator.py                 # Older dataset creation code kept for reference/compatibility.
├── main.py / unzip.py / validate.py   # Legacy or small utility scripts.
└── sample.jsonl                       # Small reference sample.
```

Current generated datasets use:

- `phase_a/{train,eval,test}.jsonl`: single-patch recognition data with empty hints.
- `phase_b/{train,eval,test}.jsonl`: state-update data with left/top incoming hints.
- `meta_*.jsonl`: raw pixel-coordinate patch metadata for auditing/debugging.
- `coord_mode=norm1000` by default for SFT JSONL; raw meta rows remain patch-pixel oriented.

## Inference And Metrics

```text
infer_index/
├── line_eval.py                       # Centerline buffer-IoU + Hungarian matching metrics.
├── utils.py                           # JSON extraction and matching helpers.
├── eval_report_format.py              # Metric result dataclasses.
└── param.py                           # Metric defaults such as meter-per-pixel.
```

Main inference/visualization files:

- `scripts/tools/infer_centerline_checkpoint.py`: checkpoint inference over one image or JSONL.
- `scripts/tools/infer_centerline_state_update.py`: row-major patch inference using previous predicted left/top state.
- `scripts/tools/visualize_centerline.py`: GT vs prediction patch visualization and metric output.
- `scripts/tools/visualize_state_update_global.py`: merged global map visualization.
- `scripts/tools/summarize_centerline_eval.py`: standalone metric summary over inference JSON.

Inference summaries keep both coordinate spaces when needed:

- `prediction_json`: raw model-coordinate output.
- `prediction_json_pixel`: pixel-converted output for visualization, stitching, and metrics.
- `ground_truth_pixel`: pixel-converted GT when the dataset is normalized.

Metric printing:

- `infer_centerline_checkpoint.py`, `infer_centerline_state_update.py`, and
  `visualize_centerline.py` write metric JSON and print the table produced by
  `LineEvalRes.show_res()`.
- The metric is computed in pixel/meter space. Normalized model coordinates are
  converted to pixels before `infer_index.line_eval.py` runs.

## Scripts

```text
scripts/
├── README.md                          # Script naming, modes, and parameter docs.
├── deepspeed_zero*.json               # Script-local DeepSpeed configs.
├── train_full_*.sh                    # Top-level full-parameter NPU entrypoints.
├── test_full_*.sh                     # Top-level NPU inference/eval entrypoints.
├── data/                              # Dataset split and eval/test helper utilities.
├── rl/                                # RL hard-pool and checkpoint export command-line tools.
├── gpu/                               # Local GPU smoke/debug scripts for SFT/GRPO/inference.
└── npu/                               # NPU cloud training/testing scripts.
```

Script naming rules:

- `full`: train all model components.
- `llm_align_*_freeze-vit`: freeze vision tower, train LLM/projector/related modules.
- `vit_align_*_freeze-llm`: freeze LLM, train vision/projector/related modules.
- `deepstack` / `no-deepstack`: whether DeepStack is enabled.
- `phase_a` / `phase_b`: supervised patch-recognition vs state-update data.
- `_gpu` / `_npu`: target runtime platform.

## Docs

```text
docs/
├── PROJECT_STRUCTURE.md               # This file.
├── RL_ROADMAP.md                       # Post-training RL migration plan.
├── qwen3vl_dinov3_deepstack.md        # Architecture, checkpoint, train/infer details.
└── 交接文档.md                         # Legacy handover notes.
```

Root-level historical docs may also exist, for example `HANDOVER_STATE_UPDATE.md`,
`DATASET_PATCH_PROCESSING.md`, or `REPRODUCTION_PLAN.md`.

## Runtime Output Directories

- `data/`: local debug datasets and small generated examples.
- `outputs/`: local training/inference outputs.
- `checkpoints/`: local model checkpoints when present.

These directories are experiment artifacts, not framework source. Large datasets
and model weights should stay out of normal code review unless explicitly needed.

## Where To Start

For data generation, read:

1. `data_process/README_DATA_PROCESSING.md`
2. `data_process/build_lane_dataset.py`
3. `data_process/build_lane_intersection_dataset.py`

For SFT training, read:

1. `README.md`
2. `scripts/README.md`
3. `mllm/train/train_qwen.py`
4. `mllm/train/llava_trainer.py`

For inference/evaluation, read:

1. `scripts/tools/infer_centerline_checkpoint.py`
2. `scripts/tools/infer_centerline_state_update.py`
3. `infer_index/line_eval.py`
4. `scripts/tools/visualize_centerline.py`

For RL work, start from `docs/RL_ROADMAP.md`, `scripts/rl/build_hard_pool.py`,
and `mllm/train/train_grpo.py`. The formal rollout path is vLLM prompt
embeddings; HF-local generation is not a training backend.

Current tested RL continuation path:

1. Train or choose an SFT checkpoint.
2. Run GRPO with `python -m mllm.train.train_grpo` and
   `--rollout_backend vllm_prompt_embeds`.
3. Use GRPO `merged/` as a self-contained full checkpoint for inference or for
   later SFT/RL continuation.
4. If continuing with LoRA SFT, `--lora_enable True` controls adapter creation;
   the directory name does not need to contain `lora`.
