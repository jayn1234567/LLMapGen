# NPU / DI Training Guide

This document describes the initial Ascend NPU / DI-platform launch convention for the minimal DINOv2 centerline route.

The goal is that the training platform only needs:

1. Code repository and branch.
2. Data path.
3. Model path.
4. Startup command.

## Required Environment Variables

Set these in the DI job configuration or before running the launcher:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen-or-Qwen3-CausalLM
DINOV2_MODEL_NAME_OR_PATH=/path/to/dinov2-large
TRAINROOT=/path/to/rc_centerline_trainroot
OUTPUT_DIR=/path/to/output
```

The launchers also accept the shorter aliases `MODEL_PATH`, `DINOV2_PATH`, `DATA_ROOT`, and `OUTPUT_PATH` when a DI platform already uses those names.

Optional:

```bash
PREPARED_TRAINROOT=/path/to/prepared_trainroot
VISUAL_ENCODER_CHECKPOINT_PATH=/path/to/latest.pt
BRIDGE_MODULES_STATE_PATH=/path/to/rc_dinov2_centerline_json_modules.pt
TOKENIZER_NAME_OR_PATH=/path/to/tokenizer
FREEZE_VISION_ENCODER=true
VISION_TRAIN_LAST_N_LAYERS=2
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501
```

## Package Portable DINOv2 Assets

For the current private-data SFT route, copy only the visual side from the
public-data experiments to the Ascend server:

- segmentation-tuned DINOv2 checkpoint
- Qwen3-8B-aligned bridge/projector modules

On the GPU server, create a portable asset directory:

```bash
python scripts/tools/package_dinov2_centerline_assets.py \
  --visual-encoder-checkpoint /mingli01/data/outputs/rc_centerline_seg_dinov2_heatmap_pad518_4gpu_20260422/best.pt \
  --bridge-modules-state /mingli01/data/outputs/stage2_rc_dinov2_caption_grid16_bridgev2_stage1init_qwen3_8b_4gpu_retryfix_20260423 \
  --output-dir /mingli01/project/jn/dinov2_centerline_assets_qwen3_8b \
  --dinov2-model-name-or-path /path/on/ascend/dinov2-large \
  --qwen-model-name-or-path /path/on/ascend/Qwen3-8B \
  --vision-train-last-n-layers 2
```

Copy the output directory to OBS or directly to the Ascend server. It contains:

```text
asset_manifest.json
train_env_template.sh
visual_encoder_checkpoint.pt
bridge_modules_state.pt
```

On the Ascend server:

```bash
source /path/to/dinov2_centerline_assets_qwen3_8b/train_env_template.sh
export DINOV2_MODEL_NAME_OR_PATH=/path/to/dinov2-large
export MODEL_NAME_OR_PATH=/path/to/Qwen3-8B
```

When `VISUAL_ENCODER_CHECKPOINT_PATH` is set, bridge modules do not overwrite
the DINOv2 weights. The bridge file is still used for `visual_norm`,
`visual_projector`, `geometric_position_mlp`, token alignment, and special-token
adapter weights.

## Prepare NPU Python Environment

The NPU image must already contain Ascend driver/CANN. Create a repo-local
Python environment with:

```bash
bash scripts/npu/setup/create_llmapgen_npu_env.sh
source .venv-llmapgen-npu/activate_llmapgen_npu.sh
```

If the DI image requires a different torch/torch-npu/CANN compatibility pair,
override the package specs:

```bash
TORCH_SPEC='torch==2.6.0' \
TORCHVISION_SPEC='torchvision==0.21.0' \
TORCHAUDIO_SPEC='torchaudio==2.6.0' \
TORCH_NPU_SPEC='torch-npu==2.6.0' \
PIP_INDEX_URL='https://your.internal.pypi/simple' \
bash scripts/npu/setup/create_llmapgen_npu_env.sh
```

## Training Command

If the private dataset uses `train.jsonl`, `test.jsonl`, and `img/<group_id>`,
convert it first:

```bash
python scripts/tools/prepare_di_qa_trainroot.py \
  --input-root /cache/dataset_extract \
  --dataset-dir-name my_dataset_dir_name \
  --output-root /cache/prepared_trainroot \
  --train-file train.jsonl \
  --eval-file test.jsonl \
  --image-root img
```

If the dataset is the current `data_line_samples_33w` server/DI layout:

```text
data_line_samples_33w/
  images/train
  images/eval
  images/test
  phase_a/train.jsonl
  phase_a/eval.jsonl
  phase_a/meta_train.jsonl
  phase_a/meta_eval.jsonl
```

convert Phase A with:

```bash
python scripts/tools/prepare_di_qa_trainroot.py \
  --input-root /cache/dataset_extract \
  --dataset-dir-name data_line_samples_33w \
  --phase phase_a \
  --image-root images \
  --output-root /cache/prepared_trainroot
```

This writes `/cache/prepared_trainroot/train.jsonl` and
`/cache/prepared_trainroot/val.jsonl`, and links or copies the `images`
directory depending on `--media-mode`. For datasets whose `dataset_info.json`
declares `coord_mode=norm1000`, `coord_range=1000`, and `patch_size=256`, the
converter automatically scales assistant labels from `0..1000` to the training
coordinate range `0..512`; metadata fallback labels are scaled from `0..255`.
Patches with no valid centerline are kept as `{"lines":[]}` because this is a
valid target for empty road-centerline patches.

Validate the prepared trainroot before launching training:

```bash
python scripts/tools/validate_di_trainroot.py \
  --trainroot /cache/prepared_trainroot \
  --expect-train-count 335506 \
  --expect-val-count 19084
```

The validator exits with a non-zero status if required files are missing, JSONL
rows are malformed, images cannot be resolved in checked samples, record/meta
ids do not align, or coordinates fall outside `0..512`.

```bash
bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
```

For the intended route on DI/NPU, the minimal startup command is:

```bash
source /path/to/.venv-llmapgen-npu/activate_llmapgen_npu.sh
source /path/to/dinov2_centerline_assets_qwen3_8b/train_env_template.sh

export TRAINROOT=/path/to/prepared_trainroot
export OUTPUT_DIR=/path/to/output
export MODEL_NAME_OR_PATH=/path/to/Qwen3-8B
export DINOV2_MODEL_NAME_OR_PATH=/path/to/dinov2-large
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export LEARNING_RATE=2e-5
export NUM_TRAIN_EPOCHS=3
export MAX_STEPS=-1
export SAVE_STEPS=1000
export LOGGING_STEPS=10
export BF16=true

bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
```

The launcher:

- Sets Ascend/HCCL environment variables.
- Imports `torch_npu` when available.
- Uses `torchrun`.
- Passes `--ddp-backend hccl`.
- Runs the same Python training entry as GPU training.

## Inference Command

```bash
CHECKPOINT_DIR=/path/to/output/checkpoint-1000 \
OUTPUT_JSONL=/path/to/pred_val.jsonl \
bash scripts/npu/test/test_dinov2_centerline_qwen_lora_npu.sh
```

The inference launcher also accepts `CKPT_DIR`, `DATA_ROOT`, and `PRED_OUTPUT_JSONL` aliases.

## Checkpoint Contents

A usable training output should keep:

```text
output_dir/
  args.json
  checkpoint-*/
  tokenizer files
  adapter_config.json or model weights
  rc_dinov2_centerline_json_modules.pt
```

Inference recovers model arguments from `args.json` and bridge/vision modules from `rc_dinov2_centerline_json_modules.pt`.

## Current Scope

This is an initial NPU platformization layer for SFT. It does not yet include:

- GRPO/DPO training.
- StageB incoming trace training.
- Multi-node checkpoint merge policy.
- OBS upload/download helpers.

Those can be added after the SFT route is stable on one NPU node.
