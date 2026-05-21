# Script Naming

Top-level `scripts/` keeps the current NPU full-parameter train/test entrypoints:

- `train_full_*_deepstack_npu.sh`: train LLM, ViT, projector, and DeepStack mergers.
- `train_full_*_no-deepstack_npu.sh`: standalone train script for LLM, ViT, and projector with DeepStack disabled; it does not delegate to the DeepStack script.
- `train_sft_dinov2_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh`: 330k-sample no-DeepStack full-parameter SFT recipe for DINOv2 + Qwen3VL-8B.
- `train_sft_dinov3_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh`: 330k-sample no-DeepStack full-parameter SFT recipe for DINOv3 + Qwen3VL-8B; it sets `INPUT_IMAGE_SIZE=512`.
- `scripts/npu/train_sft_dinov2_qwen3vl-8b_nodeepstack_npu.sh`: current SFT cloud entry for DINOv2 + Qwen3VL + no DeepStack.
- `scripts/npu/train_sft_dinov2_qwen3vl-8b_nodeepstack_33w_evalbest_npu.sh`: same DINOv2 330k recipe kept under the NPU subdirectory for compatibility.
- `scripts/npu/test_dinov2_qwen3vl-8b_nodeepstack_npu.sh`: current NPU test/infer entry; reads the prebuilt final `test.jsonl`.
- `train_full_*_train_best_npu.sh`: train with DeepStack and maintain a best checkpoint by lowest training loss.
- `train_full_*_eval_best_npu.sh`: train with DeepStack, run a validation set by steps, and maintain a best checkpoint by lowest eval loss.
- `scripts/gpu/train_sft_debug_phase_a_*_zero3_gpu.sh`: local GPU SFT Phase A smoke tests with empty incoming hints.
- `scripts/gpu/train_sft_debug_phase_b_*_zero3_gpu.sh`: local GPU SFT Phase B smoke tests with incoming hints and state-update inference.
- `test_full_*`: cloud inference/eval for the corresponding full-parameter checkpoint.
- `debug.sh`: local NPU DINOv3 smoke training with no OBS transfer and no dependency installation.

Subdirectories keep non-full or local platform-specific scripts:

- `scripts/npu/`: NPU training scripts that freeze one side of the model.
- `scripts/npu/flows/`: explicit NPU flow entrypoints for SFT/GRPO, stage A/B, lane-only/lane+intersection, and DINOv2/DINOv3.
- `scripts/gpu/`: GPU training/inference/visualization utilities.
- `scripts/rl/`: post-training RL utilities that do not change SFT scripts.
- `scripts/tools/`: Python tool implementations. Root-level `scripts/*.py` files are compatibility wrappers, so existing commands like `python scripts/infer_centerline_checkpoint.py` still work.

NPU flow entrypoints:

- SFT lane-only:
  - `scripts/npu/flows/train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_sft_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_sft_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_sft_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
- SFT lane+intersection:
  - `scripts/npu/flows/train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_sft_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_sft_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_sft_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh`
- GRPO wrappers use the same stage/task/vision naming:
  - `scripts/npu/flows/train_grpo_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_grpo_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_grpo_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
  - `scripts/npu/flows/train_grpo_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh`
  - the four `*_lane_intersection_*` GRPO wrappers mirror the SFT names.

The SFT flow wrappers call the existing 33w no-DeepStack NPU recipes after setting `DATASET_PHASE` and `MAP_TASK`. The GRPO flow wrappers are named and parameterized consistently, but fail fast on pure Ascend NPU because the current supported GRPO backend is CUDA/vLLM prompt-embedding rollout. Only set `GRPO_ENABLE_CUDA_VLLM_FROM_NPU_SCRIPT=True` when deliberately launching the same wrapper on a CUDA host for naming compatibility.

Training mode in filenames:

- `llm_align_*_freeze-vit`: freeze the ViT, train the LLM plus alignment/projector/DeepStack modules.
- `vit_align_*_freeze-llm`: freeze the LLM, train the ViT plus alignment/projector/DeepStack modules.
- `full`: train all model components.
- `deepstack`: enable DeepStack during training.
- `no-deepstack`: disable DeepStack during training.
- `train_best`: copy the best training-loss checkpoint to `BEST_TRAIN_LOSS_DIR` (default `best`).
- `eval_best`: copy the best validation-loss checkpoint to `BEST_EVAL_LOSS_DIR` (default `eval_best`).
- `phase_a`: supervised patch-recognition data without incoming state hints.
- `phase_b`: supervised state-update data with left/top incoming lane and intersection hints.
- `ckpt3200`: starts from the local/cloud checkpoint-3200 variant instead of the base Qwen3-VL checkpoint.

DINO type and platform are explicit:

- `dinov2` or `dinov3` identifies the vision tower family.
- `_npu` or `_gpu` identifies the target platform.

Common DINOv3 scripts infer DINO type from checkpoint metadata, `mm_vision_tower_type`, or the `vision_tower` path. For this BEV task they set `INPUT_IMAGE_SIZE=512` inside the scripts: 256x256 patches are resized to 512x512, and DINOv3 patch16 produces 32x32 = 1024 visual tokens.

DeepStack and gradient checkpointing are intended to work together. Training scripts keep `GRADIENT_CHECKPOINTING=True` by default, including DeepStack runs. Inference scripts do not hard-code DeepStack settings; they recover DeepStack enabled/disabled state from the checkpoint config unless an override is passed.

The Python training entry defaults to DeepStack disabled. Only scripts explicitly named `*_deepstack_*` enable it by passing both `--disable_deepstack False` and `--deepstack_visual_indexes ...`. Other align/debug scripts keep `DEEPSTACK_VISUAL_INDEXES` empty by default; set it only when you intentionally want DeepStack.

No-DeepStack training scripts are kept independent from DeepStack scripts. They pass `--disable_deepstack True` directly and should not be implemented as wrappers around `*_deepstack_*`.

Qwen multimodal checkpoints write `qwen_multimodal_checkpoint.json`. `llava_checkpoint.json` is treated as legacy metadata only. Inference refuses to use the old generic multimodal loader for directories that contain full model weights, because that route can silently skip Qwen projector, ViT, or DeepStack tensors.

`config.json` keeps the real base language-model type in `model_type`, for example `qwen2` or `qwen3`. Framework-specific multimodal fields live in normal config fields and `qwen_multimodal_checkpoint.json`; do not encode project names in `model_type`.

Best checkpoint behavior:

- Current SFT cloud scripts use the dataset's prebuilt raw-sample-level split: `train.jsonl`, `eval.jsonl`, and `test.jsonl`; they no longer split eval from test at runtime.
- NPU test scripts infer directly on the prebuilt `test.jsonl`. `NUM_TEST_SAMPLES=0` means run all final-test rows; set a positive value only for a quick smoke subset.
- Normal full training scripts keep `ENABLE_EVAL=False` and do not maintain best-loss directories unless the script enables the relevant `SAVE_BEST_*` flag.
- `train_full_dinov3_qwen3vl-8b_deepstack_train_best_npu.sh` sets train-loss best checkpointing internally and writes the current best training-loss checkpoint to `output/best/` by default. Edit `BEST_TRAIN_LOSS_START_STEP` and `BEST_TRAIN_LOSS_DIR` inside the script when needed.
- `train_full_dinov3_qwen3vl-8b_deepstack_eval_best_npu.sh` uses the validation jsonl configured inside the script, evaluates every `EVAL_STEPS`, and writes the current best eval-loss checkpoint to `output/eval_best/` by default. Edit `EVAL_PATH`, `EVAL_IMAGE_FOLDER`, `EVAL_STEPS`, and `BEST_EVAL_LOSS_DIR` inside the script when needed.
- Both best directories are copied from a normal `checkpoint-*` directory after that checkpoint is fully saved, and include `config.json`, `model.safetensors`, optimizer/scheduler state, and `qwen_multimodal_checkpoint.json`.

Centerline geometry evaluation:

- The project uses `infer_index/line_eval.py` for centerline metrics: LineString buffer IoU plus Hungarian matching, reporting instance-level and length-level precision/recall/F1.
- `scripts/infer_centerline_checkpoint.py --eval-centerline` and `scripts/infer_centerline_state_update.py --eval-centerline` write these metrics to `eval.json` by default; pass `--eval-output-json` to override.
- `scripts/visualize_centerline.py` automatically prints and saves `eval.json` after visualization when ground truth is present. Use `--no-eval-centerline` to disable it.
- The saved metrics include scalar JSON fields and a `table` string matching the console table.
- The default metric scale is `--eval-meter-per-pixel 0.2`, matching `infer_index/param.py`.
- New data uses `coord_mode=norm1000` by default. Inference/test scripts keep `COORD_MODE=auto` and `COORD_RANGE=1000`, so JSONL metadata controls whether labels are normalized or legacy pixels.
- Inference summaries keep raw model-coordinate JSON in `prediction_json` and write pixel-converted JSON to `prediction_json_pixel`. Visualization, state-update stitching, and line metrics use the pixel-converted fields.

Phase A / Phase B debug flow:

- Build small A/B data with `python scripts/gpu/build_ab_debug_data.py --limit 20 --test-count 4`.
- Phase A JSONL clears incoming hints and is used for single-patch recognition smoke tests.
- Phase B JSONL keeps generated left/top continuity hints and is used for state-update smoke tests.
- `scripts/infer_centerline_state_update.py` uses predictions as the next state in normal inference. The `--dry-run-prompts` mode is only for GT replay checks of stitching logic.
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
to `eval_best/`.
- `DEEPSTACK_VISUAL_INDEXES`, `DISABLE_DEEPSTACK`: DeepStack on/off and selected ViT layers.
- `SAVE_BEST_TRAIN_LOSS`, `BEST_TRAIN_LOSS_START_STEP`, `BEST_TRAIN_LOSS_DIR`: train-loss best checkpoint.
- `SAVE_BEST_EVAL_LOSS`, `EVAL_STEPS`, `BEST_EVAL_LOSS_DIR`: eval-loss best checkpoint in eval-best scripts.
- `USE_HF_PROGRESS_BAR`: console progress style.
- `COORD_MODE`, `COORD_RANGE`: coordinate parsing for inference and metrics. Keep `COORD_MODE=auto` for datasets generated by `data_process`; override only when testing legacy pixel JSONL.
- `mllm/model/builder.py` also parses string boolean overrides such as `"False"` correctly and falls back from fast tokenizer to slow tokenizer if a fast backend initialization fails.
