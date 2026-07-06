# Local NPU Debug Commands

This directory is for small-sample local debug runs based on:

- Dataset root: `/cache/data/data_line_samples_33w`
- Output root: `checkpoints/debug/`
- Sampled JSONL root: `checkpoints/debug_data/`
- Qwen3VL: `/cache/jjh/checkpoints/Qwen3-VL-8B-Instruct`
- DINOv2: `/cache/jjh/checkpoints/facebook_dinov2-large`
- DINOv3: `/cache/jjh/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m`
- SigLIP: `/cache/jjh/checkpoints/google_siglip-large-patch16-384`

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

`VISION_BACKBONE` supports:

| Value | Meaning |
|---|---|
| `dinov2` | Original single DINOv2 tower. |
| `dinov3` | Original single DINOv3 tower. |
| `multi_moe` | DINOv2+DINOv3 dynamic token-level router. |
| `dinov2_siglip_concat` | DINOv2+SigLIP static concat projector. |
| `dinov3_siglip_concat` | DINOv3+SigLIP static concat projector. |

Override `SIGLIP_PATH` if your SigLIP checkpoint is stored elsewhere.

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

Phase A, lane, DINOv2+DINOv3 MoE:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane \
  bash scripts/debug/train_sft_debug_moe_npu.sh
```

Phase A, lane, DINOv2+SigLIP concat:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane \
  bash scripts/debug/train_sft_debug_dinov2_siglip_concat_npu.sh
```

Phase A, lane, DINOv3+SigLIP concat:

```bash
DATASET_PHASE=phase_a MAP_TASK=lane \
  bash scripts/debug/train_sft_debug_dinov3_siglip_concat_npu.sh
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

### Debug Checkpoint Behavior

SFT debug saves normal step checkpoints with the same Trainer policy as formal SFT:

- `SAVE_STEPS=1` by default: save a normal `checkpoint-*` every step.
- `SAVE_TOTAL_LIMIT=3` by default: keep only the newest normal checkpoints.
- `ENABLE_EVAL=True`, `SAVE_BEST_EVAL_LOSS=True`, and `EVAL_STEPS=1` by default: maintain the best eval-loss checkpoint.
- `SAVE_BEST_TRAIN_LOSS=False` by default: best train-loss checkpointing is available but opt-in.

Enable best train-loss debug explicitly:

```bash
SAVE_BEST_TRAIN_LOSS=True BEST_TRAIN_LOSS_START_STEP=1 BEST_TRAIN_LOSS_DIR=best \
DATASET_PHASE=phase_a MAP_TASK=lane VISION_BACKBONE=dinov2 \
  bash scripts/debug/train_sft_debug_npu.sh
```

`BEST_CHECKPOINT_SAVE_MODE=rotating_create_only` and `BEST_CHECKPOINT_KEEP_LIMIT=1`
are the debug defaults, matching the NPU cloud-safe behavior. Best eval candidates
are written under `eval_best_candidates/`; best train-loss candidates are written
under `best_candidates/`. A best candidate is saved directly into its own
directory and is valid only after `_SUCCESS` is written.

The checkpoint resolver used by GRPO and inference tries SFT checkpoints in this order:

1. `eval_best` / `eval_best_candidates`
2. `best` / `best_candidates`
3. newest normal `checkpoint-*`

GRPO debug has its own RL checkpoint policy: it saves adapter checkpoints,
tracks `best_reward_candidates/` by mean reward, and writes `merged/` for
self-contained inference or follow-up training.

### SwanLab Offline Debug

The debug scripts expose SwanLab settings in their parameter blocks. Leave
`SWANLAB_MODE` empty for default cloud behavior. Set `SWANLAB_MODE=offline` or
`local` when the NPU job cannot upload while running. Local SwanLab files are
written to `${OUTPUT_DIR}/swanlab`, beside `checkpoint-*`, `eval_best*`,
`best*`, `best_reward_candidates/`, and `merged/`.

Edit the SwanLab block inside `scripts/debug/train_sft_debug_npu.sh` or
`scripts/debug/train_grpo_debug_npu.sh`:

```bash
SWANLAB_ENABLE=True
SWANLAB_MODE=offline
```

Then run the same debug command matrix below.

For a private SwanLab server, set these in the same script block:

```bash
SWANLAB_API_HOST=http://your-swanlab-api
SWANLAB_WEB_HOST=http://your-swanlab-web
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

```bash
DATASET_PHASE=phase_a MAP_TASK=lane \
  bash scripts/debug/infer_debug_moe_npu.sh
```

```bash
DATASET_PHASE=phase_a MAP_TASK=lane \
  bash scripts/debug/infer_debug_dinov2_siglip_concat_npu.sh
```

```bash
DATASET_PHASE=phase_a MAP_TASK=lane \
  bash scripts/debug/infer_debug_dinov3_siglip_concat_npu.sh
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

GRPO imports the SFT checkpoint from the matching SFT output directory unless `SFT_CHECKPOINT` is set. It uses the same resolver priority as inference: eval-best first, train-loss best second, newest normal checkpoint last.
By default the actor uses `ACTOR_NPU_DEVICES=0` and vLLM-Ascend rollout uses `ROLLOUT_NPU_DEVICES=1`.
For a one-card debug machine, set `ACTOR_NPU_DEVICES=0 ROLLOUT_NPU_DEVICES=0`; this is slower and can OOM on large settings, so keep `MAX_STEPS`, `NUM_GENERATIONS`, and `MAX_NEW_TOKENS` small.
GRPO debug accepts the same `VISION_BACKBONE` values as SFT/inference, including `multi_moe`, `dinov2_siglip_concat`, and `dinov3_siglip_concat`.

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

When `RUN_GRPO_INFER=True`, the flow first tries the latest successful GRPO
`best_reward_candidates/` checkpoint and falls back to `merged/` if no
successful best-reward checkpoint exists.

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
