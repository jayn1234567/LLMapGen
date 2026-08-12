# Native Qwen3-VL Scripts

## Three-Image Local256 800k LoRA E2E

```text
test/run_and_eval_rc_e2e_three_image_local256_800k_qwen3vl8b_lora_npu.sh
```

This Ascend entry evaluates a completed checkpoint from the native Qwen3-VL-8B
three-image local256 800k recipe. `CHECKPOINT_OBS_PATH` is required because the
training output URI is job-specific. It downloads the matching native
`Qwen3-VL-8B-Instruct` base, validates the PEFT adapter, and builds each E2E
record from three aligned source rasters in this fixed order:

1. clean masked BEV from `inter_patch_tif/0_inter.tif`;
2. binary Raw-Lane image from `patch_tif/0_lane.tif`; and
3. binary Pose image from `patch_tif/0_pose.tif`.

All three inputs are masked by `patch_tif/0_edit_poly.tif` and cropped on the
same 256-pixel grid. The builder uses the exact
`three_image_roles_concise_v2` prompt contract from training and fails before
model loading if an auxiliary raster is missing or misaligned.

Inference defaults to physical NPUs `2,3,4,5,6,7`. It launches one independent
native 8B process per NPU. `PER_DEVICE_INFER_BATCH_SIZE` controls the number of
three-image samples passed to one native `generate()` call on each NPU; the
default is `1`, and it can be raised when device memory allows. The launcher
verifies one-to-one patch completeness, then runs the original RC all/low/high
road evaluation. `GT_EMPTY_SUPPRESSION=True` is the default for
continuity with recent project comparisons, but it uses ground truth and must
be reported as an oracle diagnostic. It builds only a lightweight GT-presence
reference and does not calculate repository patch metrics. Set the switch to
`False` for unassisted metrics.

After centerline all/low/high completes, the launcher automatically reuses the
same raw per-patch inference JSON for one original whole-map intersection
evaluation. It does not run the model a second time and does not reuse the
centerline GT-empty-filtered JSON, because that copy can remove valid
intersection predictions from centerline-empty patches. Polygon coordinates
are restored with the local256 `256/1000` scale and 256-pixel row/column grid,
then written to the original RC `inter512/tif_512_256` compatibility layout and
merged into per-scene `Intersection.geojson`. All GT and prediction types are
evaluated by default. Use `INTERSECTION_COLLAPSE_TYPE_TO_ONE=True` for a
geometry-only diagnostic, `INTERSECTION_GT_EMPTY_SUPPRESSION=True` for the
separately labelled GT-assisted intersection oracle, or
`RUN_INTERSECTION_E2E=False` to skip the intersection stage.

After training finishes, run:

```bash
CHECKPOINT_OBS_PATH="obs://.../checkpoint-N/" \
bash scripts/qwen3vl_native/test/run_and_eval_rc_e2e_three_image_local256_800k_qwen3vl8b_lora_npu.sh
```

The isolated native environment is created only when its activation script is
absent. Existing environments are reused without reinstalling packages.

## Three-Image Local256 800k LoRA

```text
train/train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_qwen3vl8b_lora_npu.sh
```

This is the native-Qwen3-VL comparison for the released Raw-Lane + Pose
three-image local256 dataset. The clean BEV, Raw-Lane, and Pose images are all
encoded by the original Qwen3-VL-8B vision tower and native multimodal merger.
LoRA is applied separately to the language model, visual attention, and merger.
Visual LoRA uses non-reentrant gradient checkpointing so the frozen native
vision backbone does not detach the checkpointed LoRA branches from autograd.
It does not load DINOv2, the CapRL-derived text-only LLM, or the project
`mm_projector`.

The formal defaults are all 800,000 train records, eight epochs, sequence
length 4096, per-device batch 4, target global batch 128, BF16, gradient
checkpointing, learning rate `2e-4` for all three LoRA groups, no eval-loss
pass, and ordinary non-ZeRO adapter checkpoints every 1,000 steps. The verified
base model is downloaded from:

```text
obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/Qwen3-VL-8B-Instruct/
```

Both the local smoke and formal DI recipe pin `transformers==4.57.3` and
`peft==0.18.0`. Transformers 4.57.3 includes native Qwen3-VL while retaining a
Torch-version guard around DTensor imports, so it can be imported with the
local Torch 2.4 NPU runtime.

The JSONL user prompt is preserved verbatim. Preflight requires the ordered
roles `bev_road_structure`, `pv_camera_raw_lane`, and
`historical_vehicle_trajectory`, and rejects stale verbose prompts that describe
white-line rendering.

### Local Ascend smoke before DI

Create the isolated local Torch 2.4 environment once, then run the five-step
single-node smoke:

```bash
bash scripts/npu/setup/create_mllm_native_qwen3vl_torch240_npu_env_from_infer.sh
bash scripts/npu/test/smoke_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_native_qwen3vl8b_lora_npu.sh
```

The smoke defaults to eight visible NPUs and per-device batch 4. Override
`ASCEND_RT_VISIBLE_DEVICES`, `NPROC_PER_NODE`, model/data paths, or cache paths
when the local server layout differs. A pass requires all of the following:

- exactly three ordered images reach the native processor for every sampled
  record;
- language, visual-attention, and native-merger LoRA targets all resolve;
- every LoRA group produces a finite non-zero gradient;
- loss and positive per-NPU `DI_throughput` are logged;
- `checkpoint-5` contains a resumable Trainer state and PEFT adapter; and
- saved LoRA-B weights changed from zero in all three groups.

After the smoke passes, launch the formal DI job with the same training entry:

```bash
bash scripts/qwen3vl_native/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_qwen3vl8b_lora_npu.sh
```

DI injects `OUTPUT_URL` and topology variables. The formal job installs and
verifies its separate Torch 2.7 / torch-npu 2.7 stack, runs the full dataset
preflight, derives gradient accumulation from the actual world size, and trains
all 800,000 records for eight epochs. To resume an interrupted ordinary PEFT
run, set `RESUME_FROM_CHECKPOINT` to a downloaded `checkpoint-N` directory;
the directory must include adapter weights, `trainer_state.json`, `optimizer.pt`,
and `scheduler.pt`.

## Three-Image Context512/ROI256 800k LoRA

```text
train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_qwen3vl8b_lora_npu.sh
```

This is the full-data context counterpart of the validated local256 recipe.
Its only model-data-flow change is the input view: every sample contains three
512x512 context images, while supervision and norm1000 coordinates remain
relative to the central 256x256 ROI `[128,128,384,384)`. It uses the same
native Qwen3-VL-8B base, language/visual-attention/merger LoRA groups, LoRA
hyperparameters, sequence length 4096, per-device batch 4, target global batch
128, eight epochs, non-reentrant gradient checkpointing, and ordinary PEFT
checkpoints as the local256 recipe.

The formal preflight scans all 800,000 records and blocks training if image
size, context size, target size, ROI, view mode, prompt, taxonomy, or three-image
order is stale. The matching five-step Ascend smoke is:

```bash
bash scripts/npu/test/smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_native_qwen3vl8b_lora_npu.sh
```

Because 512x512 native visual inputs produce more visual tokens than local256,
run this smoke before submitting the formal DI job even when the local256 smoke
has already passed.

## Three-Image Context512/ROI256 200k LoRA

```text
train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_qwen3vl8b_lora_npu.sh
```

This recipe uses native Qwen3-VL-8B with three ordered images per sample,
language LoRA, visual-attention LoRA, native-merger LoRA, 4096
maximum sequence length, per-device batch 4, global batch 128, eight epochs,
and no eval-loss pass. Its
matched DINOv2 comparison is documented in
`docs/THREE_IMAGE_CONTEXT512_200K_DINO_VS_NATIVE_QWEN3VL8B.md`.

This directory contains standalone native Qwen3-VL architecture baselines for
UniMapGen. They keep the original Qwen3-VL vision encoder, VL projector,
processor, and language model, and do not use project `vision_tower`,
DINOv2/DINOv3, SigLIP, DeepStack, direct ViT layer fusion, or `mm_projector`
arguments.

The scripts still consume the same UniMapGen `conversations` JSONL format and
use the same Stage A / Stage B task split. Stage A starts from the native
Qwen3-VL base checkpoint. Stage B starts from a Stage-A native checkpoint via
`STAGE_A_CHECKPOINT_OBS_PATH` or `STAGE_A_CHECKPOINT_DIR`.

NPU launchers default to:

- `SWANLAB_ENABLE=False`.
- The original Qwen3-VL patch-embedding Conv3d path.

## Train Scripts

| Script | Purpose |
|---|---|
| `train/train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_qwen3vl8b_lora_npu.sh` | Stage A LoRA on all local256 Raw-Lane + Pose 800k records using the native Qwen3-VL-8B vision tower, merger, and language model. |
| `train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_qwen3vl8b_lora_npu.sh` | Stage A LoRA on all context512/ROI256 Raw-Lane + Pose 800k records with strict center-ROI coordinate checks. |
| `train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_qwen3vl8b_lora_npu.sh` | Matched 200k context512/ROI256 native-Qwen3-VL LoRA comparison. |
| `train/train_sft_stage_a_lane_intersection_qwen3vl_native_npu.sh` | SFT training: Stage A, lane+intersection, native Qwen3-VL architecture. |
| `train/train_sft_stage_b_lane_intersection_qwen3vl_native_npu.sh` | SFT training: Stage B, lane+intersection, native Qwen3-VL architecture, continued from Stage A. |

## Test Scripts

| Script | Purpose |
|---|---|
| `test/run_and_eval_rc_e2e_three_image_local256_800k_qwen3vl8b_lora_npu.sh` | Full three-image local256 native-Qwen3-VL-8B LoRA E2E inference plus original all/low/high evaluation. |
| `test/test_stage_a_lane_intersection_qwen3vl_native_npu.sh` | Inference/eval: Stage A, lane+intersection, native Qwen3-VL architecture; writes eval and visualization outputs. |
| `test/test_stage_b_lane_intersection_qwen3vl_native_npu.sh` | Inference/eval: Stage B, lane+intersection, native Qwen3-VL architecture with sequential state-update incoming hints; writes eval and visualization outputs. |
