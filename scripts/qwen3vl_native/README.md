# Native Qwen3-VL Scripts

## Three-Image Context512/ROI256 200k LoRA

```text
train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_qwen3vl8b_lora_npu.sh
```

This recipe uses native Qwen3-VL-8B with three ordered images per sample,
language-projection LoRA, frozen native vision, 6144 maximum sequence length,
per-device batch 4, global batch 128, eight epochs, and no eval-loss pass. Its
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
| `train/train_sft_stage_a_lane_intersection_qwen3vl_native_npu.sh` | SFT training: Stage A, lane+intersection, native Qwen3-VL architecture. |
| `train/train_sft_stage_b_lane_intersection_qwen3vl_native_npu.sh` | SFT training: Stage B, lane+intersection, native Qwen3-VL architecture, continued from Stage A. |

## Test Scripts

| Script | Purpose |
|---|---|
| `test/test_stage_a_lane_intersection_qwen3vl_native_npu.sh` | Inference/eval: Stage A, lane+intersection, native Qwen3-VL architecture; writes eval and visualization outputs. |
| `test/test_stage_b_lane_intersection_qwen3vl_native_npu.sh` | Inference/eval: Stage B, lane+intersection, native Qwen3-VL architecture with sequential state-update incoming hints; writes eval and visualization outputs. |
