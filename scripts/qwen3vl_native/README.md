# Native Qwen3-VL Scripts

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
- `REPLACE_PATCH_EMBED_CONV3D_WITH_LINEAR=True`.

`REPLACE_PATCH_EMBED_CONV3D_WITH_LINEAR=True` keeps the Qwen3-VL
`visual.patch_embed.proj.weight` parameter in its original Conv3d shape, but
runs the patch embedding through an equivalent Linear calculation. This avoids
NPU `Conv3DBackpropFilter` format errors while keeping saved checkpoint weights
compatible with native Qwen3-VL loading.

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
