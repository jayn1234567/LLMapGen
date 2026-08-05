# Three-Image Context512 200k: DINOv2 vs Native Qwen3-VL-8B

## Goal

This is a practical route comparison on the same 200,000 Stage-A records.
Both experiments consume three ordered 512x512 images and supervise only the
center 256x256 ROI in norm1000 coordinates.

Input order and prompt are identical:

```text
<image>
<image>
<image>
The first image is the clean BEV road-structure image.
The second image is a lane image predicted by a PV camera model.
The third image is a historical vehicle-trajectory image.
```

The remaining prompt and assistant JSON schema come directly from the packaged
Dataset V2 JSONL. Neither launcher rewrites them. Both experiments also use the
same `conv_qwen_3_Dinov2_huawei` road-map system instruction; the native route
passes its plain-text equivalent through the Qwen3-VL chat template.

## Controlled Variables

| Setting | Experiment A | Experiment B |
|---|---|---|
| Source data | Same three-image context512/ROI256 800k TAR | Same TAR |
| Selected records | 200,000 | 200,000 |
| Selection | `deterministic_sample_indices(..., seed=42)` | Same helper |
| Text base | CapRL-Qwen3VL-4B-derived text LLM | Native Qwen3-VL-8B text LLM |
| Vision | Original DINOv2-Large, 518 input, layer -2 | Native Qwen3-VL vision tower |
| Visual tokens | 3 x 1369 = 4107 | 3 x 256 = 768 for exact 512x512 inputs |
| Alignment | Trainable `mlp2x_gelu` | Native multimodal merger LoRA |
| Language tuning | LoRA, r=8, alpha=16, dropout=0.05 | Same LoRA settings |
| Vision tuning | Full DINOv2, LR 2e-5 | Visual-attention and merger LoRA, LR 2e-5 |
| Epochs | 8 | 8 |
| Max length | 7168 | 4096 |
| Per-device batch | 4 | 4 |
| Target global batch | 128 | 128 |
| Eval loss | Disabled | Disabled |

Experiment A adapts DINOv2 and its newly attached projector. Experiment B uses
the same LoRA rank, alpha, and dropout on the language model, native visual
attention, and native merger. This is intentionally a practical
system comparison, not an architecture-only ablation: the text-base size and
sequence limits differ. Report trainable parameter counts with final metrics.

## Formal DI Launchers

Experiment A:

```bash
DATASET_OBS_PATH=obs://path/to/rawlane_pose_three_image_context512_roi256_800k.tar \
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
```

Experiment B:

```bash
DATASET_OBS_PATH=obs://path/to/rawlane_pose_three_image_context512_roi256_800k.tar \
bash scripts/qwen3vl_native/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_qwen3vl8b_lora_npu.sh
```

Experiment A expects `CapRL-Qwen3VL-4B`; experiment B expects
`Qwen3-VL-8B-Instruct`. Override their model OBS variables when needed.

## Single-Node Ascend Smoke

Experiment A uses the regular Transformers 4.56.2 MLLM environment:

```bash
DATASET_OBS_PATH=obs://path/to/rawlane_pose_three_image_context512_roi256_800k.tar \
MAX_STEPS=5 \
bash scripts/npu/test/smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
```

Experiment B requires a separate native-Qwen3-VL environment with Transformers
5.7 or newer:

```bash
DATASET_OBS_PATH=obs://path/to/rawlane_pose_three_image_context512_roi256_800k.tar \
MAX_STEPS=5 \
bash scripts/npu/test/smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_native_qwen3vl8b_lora_npu.sh
```

The smoke must print a loss, `DI_throughput`, and a three-image runtime marker,
then produce a reloadable LoRA adapter checkpoint.

## Evaluation Rule

Use the same fixed eval image IDs, prompt, target JSON, generation parameters,
coordinate conversion, and metric thresholds. Report centerline geometry,
intersection IoU/F1, lane type accuracy, intersection type accuracy, valid JSON
ratio, throughput, peak NPU memory, and trainable parameter count together.

Native inference now accepts adapter checkpoints directly when `--model-base`
points to the original Qwen3-VL-8B directory. Pass the same road-map system
instruction through `--system-prompt`; the generic native NPU test wrapper
exposes it as the `SYSTEM_PROMPT` environment variable.
