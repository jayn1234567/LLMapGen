# Local NPU Debug Commands

This directory is for small-sample local debug runs based on:

- Dataset root: `/cache/data/data_line_samples_33w`
- Output root: `checkpoints/debug/`
- Sampled JSONL root: `checkpoints/debug_data/`
- Qwen3VL: `/cache/jjh/checkpoints/Qwen3-VL-8B-Instruct`
- DINOv2: `/cache/jjh/checkpoints/facebook_dinov2-large`
- DINOv3: `/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m`

The scripts sample a tiny subset from `phase_a` or `phase_b` and keep image paths pointing to the original dataset root. SFT, inference, and GRPO are local Ascend NPU flows. GRPO uses vLLM-Ascend prompt-embedding rollout, so the active environment must have `vllm`, `vllm-ascend`, `torch==2.7.1`, and the OBS `torch_npu-2.7.1.dev20250724` wheel installed.

## Common Variables

```bash
DATASET_ROOT=/cache/data/data_line_samples_33w
DEBUG_RUN_NAME=local_debug
TRAIN_LIMIT=16
EVAL_LIMIT=4
TEST_LIMIT=4
NPROC_PER_NODE=8
```

`MAP_TASK=lane` predicts centerlines only. Use `MAP_TASK=lane_intersection` when the dataset split contains lane + intersection targets.

## Sample JSONL Only

```bash
python scripts/debug/sample_debug_jsonl.py \
  --dataset-root /cache/data/data_line_samples_33w \
  --phase phase_a \
  --output-root checkpoints/debug_data \
  --train-limit 16 \
  --eval-limit 4 \
  --test-limit 4
```

```bash
python scripts/debug/sample_debug_jsonl.py \
  --dataset-root /cache/data/data_line_samples_33w \
  --phase phase_b \
  --output-root checkpoints/debug_data \
  --train-limit 16 \
  --eval-limit 4 \
  --test-limit 4
```

## SFT Debug

Phase A, lane, DINOv2:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_sft_debug_npu.sh
```

Phase A, lane, DINOv3:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_sft_debug_npu.sh
```

Phase B, lane, DINOv2:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_sft_debug_npu.sh
```

Phase B, lane, DINOv3:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_sft_debug_npu.sh
```

Phase A, lane + intersection:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane_intersection VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_sft_debug_npu.sh
```

```bash
DATASET_PHASE=phase_a MAP_TASK=lane_intersection VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_sft_debug_npu.sh
```

Phase B, lane + intersection:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane_intersection VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_sft_debug_npu.sh
```

```bash
DATASET_PHASE=phase_b MAP_TASK=lane_intersection VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_sft_debug_npu.sh
```

## Inference Debug

The inference script auto-resolves the best SFT checkpoint from:

`checkpoints/debug/${DEBUG_RUN_NAME}/sft_${DATASET_PHASE}_${MAP_TASK}_${VISION_BACKBONE}_nodeepstack`

Outputs:

- `summary.json`
- `json/`
- `viz/`
- `eval.json`
- `whole_map_viz/`
- for phase B: `merged_global.json`

Phase A, lane:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/infer_debug_npu.sh
```

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/infer_debug_npu.sh
```

Phase B, lane:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/infer_debug_npu.sh
```

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/infer_debug_npu.sh
```

Lane + intersection inference:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane_intersection VISION_BACKBONE=dinov2 \
  bash scripts/debug/infer_debug_npu.sh
```

```bash
DATASET_PHASE=phase_a MAP_TASK=lane_intersection VISION_BACKBONE=dinov3 \
  bash scripts/debug/infer_debug_npu.sh
```

```bash
DATASET_PHASE=phase_b MAP_TASK=lane_intersection VISION_BACKBONE=dinov2 \
  bash scripts/debug/infer_debug_npu.sh
```

```bash
DATASET_PHASE=phase_b MAP_TASK=lane_intersection VISION_BACKBONE=dinov3 \
  bash scripts/debug/infer_debug_npu.sh
```

Use a specific checkpoint:

```bash
CHECKPOINT_DIR=/path/to/checkpoint-or-best \
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/infer_debug_npu.sh
```

## GRPO Debug

GRPO imports the SFT checkpoint from the matching SFT output directory unless `SFT_CHECKPOINT` is set.
By default the actor uses `ACTOR_NPU_DEVICES=0` and vLLM-Ascend rollout uses `ROLLOUT_NPU_DEVICES=1`.
For a one-card debug machine, set `ACTOR_NPU_DEVICES=0 ROLLOUT_NPU_DEVICES=0`; this is slower and can OOM on large settings, so keep `MAX_STEPS`, `NUM_GENERATIONS`, and `MAX_NEW_TOKENS` small.

Phase A, lane, DINOv2:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

Phase A, lane, DINOv3:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

Phase B, lane, DINOv2:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

Phase B, lane, DINOv3:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

Lane + intersection GRPO:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane_intersection VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

```bash
DATASET_PHASE=phase_a MAP_TASK=lane_intersection VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

```bash
DATASET_PHASE=phase_b MAP_TASK=lane_intersection VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

```bash
DATASET_PHASE=phase_b MAP_TASK=lane_intersection VISION_BACKBONE=dinov3 \
  bash scripts/debug/train_grpo_debug_npu.sh
```

## One-Command SFT + Inference

Phase A, lane, DINOv2:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```

Phase A, lane, DINOv3:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```

Phase B, lane, DINOv2:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```

Phase B, lane, DINOv3:

```bash
DATASET_PHASE=phase_b MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```

Run GRPO as part of the flow:

```bash
RUN_GRPO=True RUN_GRPO_INFER=True \
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```

## Useful Overrides

Use fewer devices:

```bash
NPROC_PER_NODE=1 DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_sft_debug_npu.sh
```

Use a different output run name:

```bash
DEBUG_RUN_NAME=debug_001 DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov3 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```

Increase the debug sample size:

```bash
TRAIN_LIMIT=64 EVAL_LIMIT=16 TEST_LIMIT=16 \
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/run_debug_full_flow_npu.sh
```
