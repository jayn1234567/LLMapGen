# Native Qwen3-VL Scripts

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
| `train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_qwen3vl8b_lora_npu.sh` | Matched 200k context512/ROI256 native-Qwen3-VL LoRA comparison. |
| `train/train_sft_stage_a_lane_intersection_qwen3vl_native_npu.sh` | SFT training: Stage A, lane+intersection, native Qwen3-VL architecture. |
| `train/train_sft_stage_b_lane_intersection_qwen3vl_native_npu.sh` | SFT training: Stage B, lane+intersection, native Qwen3-VL architecture, continued from Stage A. |

## Test Scripts

| Script | Purpose |
|---|---|
| `test/test_stage_a_lane_intersection_qwen3vl_native_npu.sh` | Inference/eval: Stage A, lane+intersection, native Qwen3-VL architecture; writes eval and visualization outputs. |
| `test/test_stage_b_lane_intersection_qwen3vl_native_npu.sh` | Inference/eval: Stage B, lane+intersection, native Qwen3-VL architecture with sequential state-update incoming hints; writes eval and visualization outputs. |
