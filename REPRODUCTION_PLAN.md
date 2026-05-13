# UniMapGen Flow Reproduction Plan

## Purpose

This document consolidates the useful project context currently split across [AGENTS.md](/media/q/data2/jjh/project/unimapgen_mllm/AGENTS.md), [HANDOVER.md](/media/q/data2/jjh/project/unimapgen_mllm/HANDOVER.md), and [configs/数据样本.json](/media/q/data2/jjh/project/unimapgen_mllm/configs/数据样本.json). It defines the current baseline, the agreed reproduction scope, the target training schema, and the minimum code changes needed to reproduce the paper workflow in this repo.

## Current Baseline

- Framework: LLaVA-style multimodal training and inference.
- Vision encoder: `DINOv3` is the current preferred visual backbone.
- LLM side: `Qwen3-VL` checkpoint is used as the source, with the LLM part extracted by [llava/model/qwen3vl_extractor.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/qwen3vl_extractor.py).
- Alignment layer: 2-layer MLP projector.
- Existing branch capability: single-patch BEV road centerline prediction in patch-local coordinates.
- Existing custom architecture: DeepStack visual injection is already implemented and should remain unchanged for the first reproduction phase.
- Existing training framework is already generic enough to train new prompt/output schemas because [llava/train/train_qwen.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/train/train_qwen.py:1020) reads standard `image + conversations` records.

## What We Reproduce First

The first target is not full paper parity. The first target is the paper workflow core adapted to the current project constraints.

### In Scope

- BEV-only training and inference.
- Patch-local coordinate prediction only.
- State-update style conditioning via `incoming traces`.
- `centerline` prediction with `cut|inside` endpoint labels.
- `intersection` prediction as a closed polyline that outlines the road-intersection region.
- Whole-image patch-by-patch inference with global merging outside the model.

### Explicitly Out of Scope for Phase 1

- PV input branch.
- Paper-style discrete vector tokens.
- Paper-style equidistant sampling.
- Full lane attribute system from the paper.
- Full topology benchmark reproduction.
- Backbone changes to `DINOv3`, `Qwen3-VL`, projector, or DeepStack.

## Key Differences From the Paper

- The paper uses equidistant sampling. This repo will keep Douglas sampling.
- The paper discusses discrete vector tokenization. This repo will keep plain JSON text I/O.
- The paper is broader than centerlines. This repo first focuses on `centerline` plus `intersection`.

These are intentional design choices, not implementation gaps.

## Geometry Semantics

### Patch Coordinates

The model always predicts patch-local coordinates. It should never be asked to predict full-image coordinates.

### Endpoint Types

`start_type` and `end_type` are used only for open `centerline` polylines.

- `cut`: the endpoint is clipped by the patch boundary and should be considered for continuation across neighboring patches.
- `inside`: the endpoint is a natural endpoint inside the current patch.

Examples:

- `cut -> cut`: the line passes through the patch and continues outside on both ends.
- `cut -> inside`: the line enters from a boundary and terminates inside the patch.
- `inside -> inside`: the full local line is contained inside the patch.

### Intersection Geometry

`intersection` is represented as a closed polyline outlining the road-intersection region.

`intersection` should not carry `start_type` or `end_type`. Those fields are meaningful only for open `centerline` polylines used in state update.

## Target Training Schema

The training framework remains standard supervised fine-tuning on single patch samples. The model input is still one patch image plus a text prompt. The difference is that the prompt now includes `incoming traces`, and the assistant target uses a richer JSON schema.

### Recommended Record Format

```json
{
  "id": "tile_B_r21_c14",
  "image": "images/train/tile_B_r21_c14.png",
  "meta": {
    "tile_id": "tile_B",
    "row": 21,
    "col": 14,
    "x0": 3584,
    "y0": 5376,
    "patch_size": 256,
    "task_mode": "state_update_centerline_intersection"
  },
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nThis is a 256x256 BEV road patch.\nPredict the road geometry inside this patch only.\n\nIncoming traces JSON:\n[{\"id\":\"L0\",\"side\":\"left\",\"points\":[[-25,126],[-13,126],[-1,126]]},{\"id\":\"T0\",\"side\":\"top\",\"points\":[[128,-25],[128,-13],[128,-1]]}]\n\nEach incoming trace is ordered from the previous patch interior toward the current patch boundary.\nIncoming traces are continuity hints only.\nReturn only valid JSON in this schema:\n{\"lines\":[{\"category\":\"centerline\",\"start_type\":\"cut|inside\",\"end_type\":\"cut|inside\",\"points\":[[x,y],[x,y]]},{\"category\":\"intersection\",\"points\":[[x,y],[x,y]]}]}\nAll output points must be integers inside [0,255]."
    },
    {
      "from": "gpt",
      "value": "{\"lines\":[{\"category\":\"centerline\",\"start_type\":\"cut\",\"end_type\":\"cut\",\"points\":[[0,126],[92,126]]},{\"category\":\"centerline\",\"start_type\":\"cut\",\"end_type\":\"cut\",\"points\":[[127,0],[127,92]]},{\"category\":\"intersection\",\"points\":[[92,92],[164,92],[164,164],[92,164],[92,92]]}]}"
    }
  ]
}
```

### Why This Schema

- It matches the existing `conversations` training pipeline.
- It avoids special token work.
- It keeps `centerline` endpoint semantics explicit.
- It cleanly separates closed `intersection` geometry from open `centerline` geometry.

## Training Strategy

Training does not need recursive state update inside the model. The model still trains on single patch samples.

### Patch Order Contract

The state-update scan order is fixed as row-major:

```text
(0,0) -> (0,1) -> (0,2) -> ...
(1,0) -> (1,1) -> (1,2) -> ...
```

This means the system processes patches from top to bottom, and within each row from left to right.

For a current patch `(row, col)`, the first version may only use already processed neighbors:

- left neighbor: `(row, col - 1)`
- top neighbor: `(row - 1, col)`

Training samples may be shuffled. The important constraint is not training order, but trace construction: every training sample that includes `incoming_traces` must use only GT traces from the same left/top neighbor rule. Inference uses predicted traces from the same left/top neighbor rule.

### Phase A

Train on single patch images with the new output schema but allow `Incoming traces JSON: []`.

Goal:

- Teach the model the new schema.
- Teach the model `centerline` plus `intersection`.
- Teach the model `cut|inside`.

### Phase B

Train on single patch images with teacher-forced `incoming traces` constructed from GT neighboring patches.

Goal:

- Teach the model to use continuity hints.
- Prepare the model for patch-by-patch state update inference.

### Why This Is Enough

The current training pipeline in [llava/train/train_qwen.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/train/train_qwen.py:1063) already reads the prompt text from the sample. As long as the dataset generator writes the right prompt and the right assistant JSON, the backbone code does not need to change.

Recommended metadata fields written into converted samples:

- `scan_order`: `row_major_top_to_bottom_left_to_right`
- `available_neighbors`: `["left", "top"]`
- `train_shuffle_allowed`: `true`
- `trace_source_train`: `gt_left_top_neighbors`
- `trace_source_infer`: `predicted_left_top_neighbors`

## Incoming Traces

### Representation

`incoming traces` stay as plain JSON text in the prompt.

Example:

```json
[
  {"id":"L0","side":"left","points":[[-25,170],[-13,149],[-1,128]]},
  {"id":"T0","side":"top","points":[[96,-42],[84,-22],[72,-1]]}
]
```

Each incoming trace should carry points from one adjacent `centerline` near the shared boundary. Prefer 3 ordered points; if a short clipped line cannot provide 3 points, keep 2 points. Fewer than 2 points should be dropped. The paper passes an explicit direction; in this repo the local direction is represented by these ordered points instead of adding a separate direction field.

Incoming traces should only be extracted from adjacent `centerline` endpoints whose `start_type` or `end_type` is `cut`. A natural original-line endpoint that happens to lie on a patch boundary remains `inside` and must not be propagated as a continuity hint.

### First Version Extraction Rule

For the first implementation, only use traces from already processed neighbors:

- left neighbor
- top neighbor

This matches a left-to-right, top-to-bottom patch scan and keeps the logic simple.

## State Update Design

State update should live outside the model, in the inference orchestration layer.

### Design Rule

- Model: patch image plus prompt in, patch-local JSON out.
- Orchestrator: patch ordering, trace extraction, prompt assembly, local-to-global merging, output aggregation.

### Recommended New Script

- [scripts/infer_centerline_state_update.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_state_update.py)

This script should:

1. Load patch records with row, col, x0, y0 metadata.
2. Sort them in scan order.
3. Maintain a merged global state.
4. Extract `incoming traces` for the current patch from already merged predictions.
5. Build the prompt and call the existing generation path.
6. Parse patch-local output JSON.
7. Convert local coordinates to merged global coordinates for storage only.
8. Write outputs for visualization and downstream evaluation.

### Important Separation

- The model never predicts global coordinates.
- The orchestrator may still store merged lines in global coordinates after local prediction, because that is just geometry bookkeeping.

## Minimum Code Changes

### Files That Can Stay Unchanged in Phase 1

- [llava/model/llava_arch.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/llava_arch.py)
- [llava/model/language_model/llava_qwen3.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/language_model/llava_qwen3.py)
- [llava/model/multimodal_encoder/dinov3_encoder.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/multimodal_encoder/dinov3_encoder.py)
- DeepStack implementation
- projector implementation

### Files That Need Changes

#### 1. Prompt templates

[llava/conversation.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/conversation.py:436)

Added state-update prompt templates for Qwen2/Qwen3:

- `conv_qwen_2_state_update_centerline`
- `conv_qwen_3_state_update_centerline`

These templates:

- explains `incoming traces`
- defines the new JSON schema
- permits `centerline` and `intersection`
- states that coordinates are patch-local integers

#### 2. Inference JSON parser

[scripts/infer_centerline_checkpoint.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_checkpoint.py:220)

The parser has been extended with `parse_map_json` to accept:

- top-level `{"lines": [...]}`
- `centerline` records with `start_type/end_type`
- `intersection` records without endpoint fields
- legacy top-level `CenterLine` lists

#### 3. State-update inference entrypoint

Added:

- [scripts/infer_centerline_state_update.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_state_update.py)

Responsibilities:

- patch ordering
- trace extraction
- prompt construction
- model invocation
- local-to-global merge
- final export

The first version only uses left and top neighbors, matching left-to-right, top-to-bottom scan order.

#### 4. Visualization

[scripts/visualize_centerline.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/visualize_centerline.py)

Drawing code now supports:

- `centerline`
- `intersection`
- legacy list output
- new `{"lines": [...]}` output

#### 5. Dataset generation

This is new work outside the current training code. The training code itself likely does not need structural changes as long as generated samples stay in `image + conversations` format.

## Existing Useful Reference

[configs/数据样本.json](/media/q/data2/jjh/project/unimapgen_mllm/configs/数据样本.json) already contains a useful state-update style sample structure:

- `incoming_traces`
- `target_lines`
- patch metadata
- `cut|inside` supervision

This file should be treated as a local reference for the first dataset schema implementation.

## Data Preparation Scripts

The first dataset conversion entrypoint is:

- [scripts/data/build_sft_dataset.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/data/build_sft_dataset.py)

It supports two modes.

### Legacy centerline conversion

Use this to convert the current repo-style centerline-only records into the new schema with:

- `Incoming traces JSON: []`
- inferred `cut|inside`
- assistant output as top-level `{"lines": ...}`

Example:

```bash
python scripts/data/build_sft_dataset.py \
    legacy-centerline \
    --input data/train.jsonl \
    --output data/train_state_phase_a.jsonl
```

### State-update metadata conversion

Use this when source rows already contain:

- `incoming_traces`
- `target_lines`
- patch metadata such as `patch_row`, `patch_col`, and patch box

Example:

```bash
python scripts/data/build_sft_dataset.py \
    state-update-meta \
    --input meta_train.jsonl \
    --output train_state_phase_b.jsonl
```

The script also supports the local reference file [configs/数据样本.json](/media/q/data2/jjh/project/unimapgen_mllm/configs/数据样本.json) for quick schema checks.

## Project Code Change Summary

Current implementation touches only task-level code and does not modify model internals.

### Data processing

- [scripts/data/build_sft_dataset.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/data/build_sft_dataset.py)

Responsibilities:

- convert legacy centerline samples to the new `{"lines": ...}` schema
- convert state-update metadata rows to SFT records
- write scan-order metadata into every converted sample

### Prompt templates

- [llava/conversation.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/conversation.py)

Added templates:

- `conv_qwen_2_state_update_centerline`
- `conv_qwen_3_state_update_centerline`

### Parser and single-patch inference compatibility

- [scripts/infer_centerline_checkpoint.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_checkpoint.py)

Added `parse_map_json` so existing inference can parse both legacy and new schemas.

### State-update inference

- [scripts/infer_centerline_state_update.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_state_update.py)

Responsibilities:

- sort patches by row-major order
- extract predicted traces from left/top neighbors
- inject traces into current patch prompt
- store patch-local and merged global outputs

### Visualization

- [scripts/visualize_centerline.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/visualize_centerline.py)

Updated to read legacy list outputs and new `{"lines": ...}` outputs, including `intersection`.

## Recommended Implementation Order

1. Generate a small dataset slice with `cut|inside` and `incoming traces`.
2. Train Phase A on the new schema.
3. Train Phase B with teacher-forced traces.
4. Run `scripts/infer_centerline_state_update.py` with a real checkpoint.
5. Validate on a small patch grid with visible patch boundaries.
6. Extend whole-image visualization and evaluation as needed.

## Success Criteria

Phase 1 is successful when all of the following are true:

- The model can output the new JSON schema reliably.
- `centerline` endpoint types are generated correctly on held-out patch samples.
- `intersection` closed polylines can be predicted from annotated training data.
- A patch-by-patch state-update script can run over a patch grid.
- Whole-image merged results are visibly more continuous than naive independent patch inference.

## Notes on Existing Docs

- [AGENTS.md](/media/q/data2/jjh/project/unimapgen_mllm/AGENTS.md) remains the branch-specific operational document.
- [HANDOVER.md](/media/q/data2/jjh/project/unimapgen_mllm/HANDOVER.md) remains a useful quick onboarding note.
- This file is the main planning document for UniMapGen flow reproduction and should be the primary reference for new implementation work.
