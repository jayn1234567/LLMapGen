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
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501
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

```bash
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
