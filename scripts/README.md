# Script Naming

Top-level `scripts/` now keeps only shared configs and this README. Runnable
scripts live under platform/purpose folders:

- `scripts/npu/train/`: current self-contained NPU SFT/GRPO training entrypoints.
- `scripts/npu/test/`: current self-contained NPU inference/eval entrypoints.
- `scripts/gpu/`: local GPU debug, smoke-test, and inference scripts.
- `scripts/tools/`: Python tool implementations.
- `scripts/data/`: dataset helper scripts.
- `scripts/rl/`: RL data/export helper scripts.
- `scripts/tmp/` and `scripts/npu/tmp/`: old root-level wrappers or legacy NPU scripts kept for reference.

NPU scripts are grouped by purpose:

- `scripts/npu/train/`: explicit NPU training entrypoints for SFT/GRPO, stage A/B, lane-only/lane+intersection, and DINOv2/DINOv3.
- `scripts/npu/test/`: explicit NPU inference entrypoints for the same stage/task/vision matrix.
- `scripts/npu/tmp/`: older or non-current NPU scripts kept for reference instead of being deleted.
- `scripts/gpu/`: GPU training/inference/visualization utilities.
- `scripts/rl/`: post-training RL utilities that do not change SFT scripts.
- Use Python tools directly from `scripts/tools/`, for example
  `python scripts/tools/infer_centerline_checkpoint.py` and
  `python scripts/tools/visualize_centerline.py`.

NPU train entrypoints:

- SFT lane-only:
  - `scripts/npu/train/train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_sft_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_sft_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_sft_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
- SFT lane+intersection:
  - `scripts/npu/train/train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_sft_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_sft_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_sft_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh`
- Standalone Stage-B-from-Stage-A SFT:
  - `scripts/npu/train/train_sft_stage_b_from_stage_a_qwen3vl_nodeepstack_npu.sh`
  - This script is also self-contained. Set `VISION_BACKBONE`, `MAP_TASK`, and either `STAGE_A_CHECKPOINT_OBS_PATH` or `STAGE_A_CHECKPOINT_PATH` inside the script before launch.
- GRPO scripts use the same stage/task/vision naming:
  - `scripts/npu/train/train_grpo_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_grpo_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_grpo_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/train/train_grpo_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
  - the four `*_lane_intersection_*` GRPO scripts mirror the SFT names.

NPU test entrypoints mirror the same matrix under `scripts/npu/test/`, for example:

- `scripts/npu/test/test_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
- `scripts/npu/test/test_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh`

Each concrete train/test script is now standalone. For example,
`scripts/npu/test/test_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
contains its own editable parameter block plus the full download, inference,
visualization, metric, and upload flow. Edit the concrete stage/task/backbone
script you plan to launch; it no longer dispatches through a second shell
launcher.

Local path inputs:

- `CHECKPOINT_DIRS`: comma/semicolon/newline separated checkpoint directories to evaluate.
- `CHECKPOINT_DIR`: single checkpoint directory fallback.
- `TRAIN_OUTPUT_DIR`: local training output root; if checkpoint dirs are not set, the resolver tries eval-best, train-best, then normal checkpoints.
- `DATASET_PATH`: dataset root with `phase_a/phase_b/test.jsonl` and images.
- `IMAGE_FOLDER`: image root, usually the same as `DATASET_PATH`.
- `VISION_TOWER`: local DINOv2/DINOv3 checkpoint path.
- `OUTPUT_DIR`: result root.

Checkpoint inputs may be direct checkpoint directories, normal `checkpoint-*`
folders, or best-candidate roots. The shared resolver supports LoRA adapters,
single-file full checkpoints, and standard HF sharded full checkpoints:

```text
adapter_model.safetensors
adapter_model.bin
model.safetensors
pytorch_model.bin
model.safetensors.index.json + model-00001-of-00004.safetensors ...
pytorch_model.bin.index.json + pytorch_model-00001-of-00004.bin ...
```

Keep the index JSON beside shard files when a checkpoint will be used for
training continuation. The inference loader has a fallback for bare
`model-*-of-*` shards, but the indexed format is the reliable path for
Transformers `from_pretrained`.

Cloud/OBS inputs:

- `DATASET_OBS_PATH`: OBS dataset zip path. Current NPU train/test defaults use `obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_line_samples_33w.zip`.
- `DATASET_DIR_NAME`: directory name expected after unzip, currently `data_line_samples_33w`.
- `DATASET_PATH`: by default `${DATASET_EXTRACT_ROOT}/data_line_samples_33w` after unzip.
- `MODEL_OBS_PATH`: OBS root containing DINO model directories.
- `CHECKPOINT_OBS_LIST`: comma/semicolon/newline separated full OBS checkpoint dirs.
- `TRAINED_CHECKPOINT_OBS`: one OBS training output root.
- `CHECKPOINT_NAMES`: relative weight dirs under `TRAINED_CHECKPOINT_OBS`, for example `checkpoint-500,eval_best_candidates,best_candidates,infer_best_candidates,best_reward_candidates,merged`.

When multiple checkpoints are given, each checkpoint writes a separate subfolder
under `OUTPUT_DIR`, named by checkpoint index and label. `infer_best_candidates`,
`eval_best_candidates`, `best_candidates`, and `best_reward_candidates` are
resolved after download to the latest successful candidate containing `_SUCCESS`.

Inference outputs are intentionally separated:

- `summary.json`: normal inference summary.
- `json/`: per-sample or per-patch JSON files.
- `viz/`: single-patch comparison PNGs.
- `eval.json`: `infer_index.line_eval` metrics plus table text.
- `whole_map_viz/`: stitched whole-map PNGs for both stage A and stage B.

For `phase_b`, NPU inference uses `torchrun` and shards complete `tile_id`
groups across ranks. Each rank processes whole raw-sample maps in row-major
order, writes `summary_rank*.json`, and rank0 merges them into `summary.json`
before patch visualization, whole-map stitching, and `eval.json`. This keeps
left/top state-update dependencies inside each original map intact while still
using multiple cards.

Stage A and Stage B expose task selection through different inference tools:

- Stage A uses `scripts/tools/infer_centerline_checkpoint.py` and passes
  `--map-task lane` or `--map-task lane_intersection`.
- Stage B uses `scripts/tools/infer_centerline_state_update.py`; lane-only
  scripts do not pass `--include-intersections`, while lane+intersection scripts
  pass that boolean flag.

The concrete SFT train scripts now contain the full 33w no-DeepStack NPU recipe after pinning `DATASET_PHASE` and `MAP_TASK`. The concrete GRPO train scripts contain the formal vLLM prompt-embedding rollout path on Ascend through vLLM-Ascend; they are not HF-local generation scripts and no longer require a CUDA compatibility escape hatch. The NPU GRPO script installs/uses the NPU RL stack and separates actor and rollout devices with `ACTOR_NPU_DEVICES` and `ROLLOUT_NPU_DEVICES`.

Training mode in filenames:

- `llm_align_*_freeze-vit`: freeze the ViT, train the LLM plus alignment/projector/DeepStack modules.
- `vit_align_*_freeze-llm`: freeze the LLM, train the ViT plus alignment/projector/DeepStack modules.
- `full`: train all model components.
- `deepstack`: enable DeepStack during training.
- `no-deepstack`: disable DeepStack during training.
- `train_best`: keep the best training-loss checkpoint under `best_candidates/`.
- `eval_best`: keep the best validation-loss checkpoint under `eval_best_candidates/`.
- `phase_a`: supervised patch-recognition data without incoming state hints.
- `phase_b`: supervised state-update data with left/top incoming lane and intersection hints.
- `ckpt3200`: starts from the local/cloud checkpoint-3200 variant instead of the base Qwen3-VL checkpoint.

DINO type and platform are explicit:

- `dinov2` or `dinov3` identifies the vision tower family.
- `_npu` or `_gpu` identifies the target platform.

Common DINOv3 scripts infer DINO type from checkpoint metadata, `mm_vision_tower_type`, or the `vision_tower` path. For this BEV task they set `INPUT_IMAGE_SIZE=512` inside the scripts: 256x256 patches are resized to 512x512, and DINOv3 patch16 produces 32x32 = 1024 visual tokens.

The GPU smoke script also supports `VISION_BACKBONE=multi_moe`, which builds a
multi-vision router over comma-separated `MULTI_VISION_TOWERS` and forwards the
same multi-vision parameters into both training and inference.

DeepStack and gradient checkpointing are intended to work together. Training scripts keep `GRADIENT_CHECKPOINTING=True` by default, including DeepStack runs. Inference scripts do not hard-code DeepStack settings; they recover DeepStack enabled/disabled state from the checkpoint config unless an override is passed.

The Python training entry defaults to DeepStack disabled. Only scripts explicitly named `*_deepstack_*` enable it by passing both `--disable_deepstack False` and `--deepstack_visual_indexes ...`. Other align/debug scripts keep `DEEPSTACK_VISUAL_INDEXES` empty by default; set it only when you intentionally want DeepStack.

No-DeepStack training scripts are kept independent from DeepStack scripts. They pass `--disable_deepstack True` directly and should not be implemented as wrappers around `*_deepstack_*`.

Qwen multimodal checkpoints write `qwen_multimodal_checkpoint.json`. `llava_checkpoint.json` is treated as legacy metadata only. Inference refuses to use the old generic multimodal loader for directories that contain full model weights, because that route can silently skip Qwen projector, ViT, or DeepStack tensors.

`config.json` keeps the real base language-model type in `model_type`, for example `qwen2` or `qwen3`. Framework-specific multimodal fields live in normal config fields and `qwen_multimodal_checkpoint.json`; do not encode project names in `model_type`.

Best checkpoint behavior:

- Current SFT cloud scripts use the dataset's prebuilt raw-sample-level split: `train.jsonl`, `eval.jsonl`, and `test.jsonl`; they no longer split eval from test at runtime.
- NPU test scripts infer directly on the prebuilt `test.jsonl`. `NUM_TEST_SAMPLES=0` means run all final-test rows; set a positive value only for a quick smoke subset.
- Normal full training scripts keep `ENABLE_EVAL=False` and do not maintain best-loss directories unless the script enables the relevant `SAVE_BEST_*` flag.
- `SAVE_BEST_INFER_INDEX=True` runs generation-based `infer_index` evaluation at eval steps and keeps the best metric checkpoint under `infer_best_candidates/`. The default metric is `length_f1`, which is usually more aligned with missing/unaligned centerlines than `eval_loss`.
- `BEST_CHECKPOINT_SAVE_MODE=rotating_create_only` is the current NPU cloud default. A new best creates a unique directory under `infer_best_candidates/`, `best_candidates/`, or `eval_best_candidates/`, writes the model directly into that directory, writes metadata, then writes `_SUCCESS` last.
- `BEST_CHECKPOINT_KEEP_LIMIT=1` keeps only the latest successful best candidate by default. Because the code saves only when the metric improves, the largest step among `_SUCCESS` candidates is the current best.
- This mode never renames or replaces files in the output mount. It only creates the new best candidate and then deletes older successful candidates, which matches cloud mounts that allow create/delete but not rename/replace.
- Regular `checkpoint-*` rotation still uses `SAVE_TOTAL_LIMIT` and deletes old normal checkpoints with a validated `rm -rf`; current NPU SFT scripts default it to 10.
- Best candidate directories are not copied from normal `checkpoint-*` directories. Train-loss, eval-loss, and infer-index best checkpoints are saved directly to their own candidate directories, so they do not create an extra normal checkpoint when the best metric improves.
- NPU inference scripts resolve best checkpoints with `scripts/tools/resolve_best_checkpoint.py`. By default they try `infer_best_candidates/` first, then `eval_best_candidates/`, then `best_candidates/`, and only accept candidate directories with `_SUCCESS`.

NPU cloud output behavior:

- Formal NPU scripts follow the cloud reference output convention:
  `CLUSTER_SAVE=${OUTPUT_URL}` and `OSB_SHARE_PATH="${CLUSTER_SAVE}"`.
- SFT scripts train into local `LOCAL_MODEL_SAVE_PATH` on every node. After
  successful training, only `NODE_RANK=0` moves its local run directory to
  `CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}`. This keeps checkpoint
  rotation on local `/cache` instead of the ModelArts OBS-mounted output path.
- GRPO and test scripts write to `LOCAL_OUTPUT_DIR` / `LOCAL_OUTPUT_ROOT` first,
  then upload the complete run directory to `${OSB_SHARE_PATH}` or the explicit
  `GRPO_RESULT_OBS` / `TEST_RESULT_OBS` override.
- Main script parameters are documented as inline comments beside the variable,
  so edit the parameter block inside each concrete script instead of passing
  experiment knobs as one-off shell prefixes.

SwanLab logging:

- SFT and GRPO scripts define `SWANLAB_ENABLE`, `SWANLAB_PROJECT`, `SWANLAB_GROUP`, `SWANLAB_JOB_TYPE`, `SWANLAB_EXPERIMENT_NAME`, `SWANLAB_MODE`, and `SWANLAB_LOG_DIR` in their parameter blocks.
- Leave `SWANLAB_MODE` empty for SwanLab default cloud behavior. Set it to `offline` or `local` when the NPU job cannot upload during training, or `disabled` to suppress SwanLab runtime logging.
- Local SwanLab files are stored next to the training checkpoints: SFT uses `${OUTPUT_PATH}/swanlab`; GRPO uses `${OUTPUT_DIR}/swanlab`.
- Offline/local modes skip `swanlab.login(...)`; cloud mode still uses `SWANLAB_API_KEY` when it is set.
- For private SwanLab deployment, set `SWANLAB_API_HOST` and `SWANLAB_WEB_HOST` in the same script parameter block. They are passed to `swanlab.login(host=..., web_host=...)` and exported as `SWANLAB_API_HOST` / `SWANLAB_WEB_HOST`.

Centerline geometry evaluation:

- The project uses `infer_index/line_eval.py` for centerline metrics: LineString buffer IoU plus Hungarian matching, reporting instance-level and length-level precision/recall/F1.
- `scripts/tools/infer_centerline_checkpoint.py --eval-centerline` and `scripts/tools/infer_centerline_state_update.py --eval-centerline` write these metrics to `eval.json` by default; pass `--eval-output-json` to override.
- `scripts/tools/visualize_centerline.py` automatically prints and saves `eval.json` after visualization when ground truth is present. Use `--no-eval-centerline` to disable it.
- The saved metrics include scalar JSON fields and a `table` string matching the console table.
- The default metric scale is `--eval-meter-per-pixel 0.2`, matching `infer_index/param.py`.
- New data uses `coord_mode=norm1000` by default. Inference/test scripts keep `COORD_MODE=auto` and `COORD_RANGE=1000`, so JSONL metadata controls whether labels are normalized or legacy pixels.
- Inference summaries keep raw model-coordinate JSON in `prediction_json` and write pixel-converted JSON to `prediction_json_pixel`. Visualization, state-update stitching, and line metrics use the pixel-converted fields.

Phase A / Phase B debug flow:

- Build small A/B data with `python scripts/gpu/build_ab_debug_data.py --limit 20 --test-count 4`.
- Phase A JSONL clears incoming hints and is used for single-patch recognition smoke tests.
- Phase B JSONL keeps generated left/top continuity hints and is used for state-update smoke tests.
- `scripts/tools/infer_centerline_state_update.py` uses predictions as the next state in normal inference. The `--dry-run-prompts` mode is only for GT replay checks of stitching logic.
- B-stage state-update inference writes per-patch JSONs to `--output-dir`/`--sample-json-dir` and stitched whole-map images to `whole_map_viz/` by default. Use `--whole-map-viz-dir` to choose a separate directory, or `--skip-whole-map-viz` to disable it.
- Current GPU ZeRO3 SFT smoke scripts cover Phase A lane patch inference and Phase B lane+intersection patch inference plus state-update inference.

RL notes:

- GRPO uses the formal vLLM prompt-embedding rollout path. The actor computes
  DINO/projector prompt embeddings with the HF multimodal model; vLLM runs the
  Qwen text decoder rollout from those embeddings.
- Run RL scripts from the dedicated `unimapgen` conda environment. The verified
  local GPU stack is `torch==2.7.0+cu126`, `vllm==0.9.2`, `ray==2.55.1`,
  `transformers==4.56.2`, and `huggingface-hub==0.36.2`.
- Use `scripts/gpu/train_grpo_dinov3_qwen3vl_nodeepstack_vllm_debug_gpu.sh`
  for local GPU debug. It is not an HF-local generation script.
- The RL task is selected by `MAP_TASK` / `--map_task`: `lane` for current
  centerline-only RL, `lane_intersection` for future centerline+intersection RL.
- The first supported RL mode is no-DeepStack + LLM LoRA. DeepStack is blocked
  because prompt embeddings cannot represent layer-level visual residual
  injection.
- Build an initial hard-sample pool with `python scripts/rl/build_hard_pool.py`.
- Export a vLLM text-decoder checkpoint manually with
  `python scripts/rl/export_text_decoder_for_vllm.py` when you do not want the
  training entry to create `output_dir/vllm_text_model`.
- Export a specific LoRA adapter to a full merged checkpoint with
  `python scripts/rl/export_merged_lora_checkpoint.py`.

Distributed logging defaults to `MLLM_LOG_RANK0_ONLY=1`, so normal stdout logs are printed by global rank 0 only. Error tracebacks on stderr are kept for nonzero ranks unless `MLLM_SUPPRESS_NONZERO_STDERR=1` is set.

Full training scripts keep the Hugging Face tqdm progress bar enabled by default and write full logs to `train_metrics.log`, `eval_metrics.log`, and `checkpoint_events.log`. In tqdm mode, `DI_throughput` is also printed with the step metric line through `tqdm.write(...)`.

Do not pass experiment knobs as one-off shell prefixes. Edit the parameter block inside the target script. The main block contains comments for:

- `TARGET_GLOBAL_BATCH_SIZE`, `PER_DEVICE_TRAIN_BATCH_SIZE`: total batch control.
- `LR`, `MM_PROJECTOR_LR`, `MM_VISION_TOWER_LR`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `LR_SCHEDULER_TYPE`: optimizer schedule.
- `NUM_EPOCHS`, `MODEL_MAX_LENGTH`, `SAVE_STEPS`, `LOGGING_STEPS`: training length and logging.

Current full-parameter Qwen3VL-8B + DINO scripts use global batch 128,
per-device batch 4, LR 2e-5, projector LR 2e-5, 6 epochs, weight decay 0.0,
cosine scheduler, and warmup ratio 0.03. With batch 128 and 6 epochs this is
about 5156 optimizer steps for 110k samples and 15469 steps for 330k samples;
the 0.03 warmup ratio corresponds to about 155 and 465 warmup steps.

The 330k first-run recipe script uses 3 epochs instead of 6 and sets
`MM_VISION_TOWER_LR=2e-6`, so DINO is updated more conservatively than the LLM
and projector. It enables eval-by-steps and writes the best eval-loss checkpoint
to `eval_best_candidates/`.
- `DEEPSTACK_VISUAL_INDEXES`, `DISABLE_DEEPSTACK`: DeepStack on/off and selected ViT layers.
- `SAVE_BEST_TRAIN_LOSS`, `BEST_TRAIN_LOSS_START_STEP`, `BEST_TRAIN_LOSS_DIR`: train-loss best checkpoint.
- `SAVE_BEST_EVAL_LOSS`, `EVAL_STEPS`, `BEST_EVAL_LOSS_DIR`: eval-loss best checkpoint in eval-best scripts.
- `SAVE_BEST_INFER_INDEX`, `BEST_INFER_INDEX_METRIC`, `BEST_INFER_INDEX_NUM_SAMPLES`, `BEST_INFER_INDEX_EVAL_STEPS`, `BEST_INFER_INDEX_DIR`: generation/infer-index best checkpoint. Current NPU SFT scripts wire this to the prebuilt `eval.jsonl`; `BEST_INFER_INDEX_NUM_SAMPLES=0` means use the full eval set and is the default for best selection.
- `BEST_CHECKPOINT_SAVE_MODE`, `BEST_CHECKPOINT_KEEP_LIMIT`: best checkpoint materialization policy. Use `rotating_create_only` on NPU cloud mounts that allow create/delete but not rename/replace.
- `USE_HF_PROGRESS_BAR`: console progress style.
- `COORD_MODE`, `COORD_RANGE`: coordinate parsing for inference and metrics. Keep `COORD_MODE=auto` for datasets generated by `data_process`; override only when testing legacy pixel JSONL.
- `mllm/model/builder.py` also parses string boolean overrides such as `"False"` correctly and falls back from fast tokenizer to slow tokenizer if a fast backend initialization fails.
