# RC Dataset V2 local256 550k Handoff

## Purpose

This document describes the finalized 550k RC Dataset V2 asset and the
constraints for using it with the Jiangjihua CapRL + DINOv2 no-DeepStack
training route. It is the handoff entrypoint for an agent implementing the next
DI training run.

## Canonical OBS asset

Use this exact OBS location. It differs from the older
`yw-ads-training-gy1/.../whu/jn/...` draft path:

```text
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256/
|-- local.tar
|-- build_summary.json
|-- semantic_schema_report.json
|-- split_manifest.json
`-- balance_report.json
```

The training archive is:

```bash
DATASET_OBS_PATH=obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256/local.tar
DATASET_DIR_NAME=local256
```

`local.tar` is approximately 6.76 GiB and stores `local256/` as its top-level
directory. The four JSON files beside the tar are audit/provenance files. The
tar also contains the variant-level `dataset_info.json`, `balance_report.json`,
and `split_manifest.json` needed to identify the extracted dataset.

Do not use the old `local256.tar` OBS URI currently present as the default in
some launchers. Do not run `prepare_di_qa_trainroot.py` or any other conversion
step on this asset: it is already final SFT data.

## Dataset size and splits

| Split | Records | Policy |
|---|---:|---|
| train | 550,000 | balanced, no empty targets |
| eval | 50,925 | complete natural holdout |
| test | 53,796 | complete natural holdout |
| total | 654,721 | |

Raw sample folders are assigned to exactly one split using a stable SHA-256
split. Patches from one raw map do not cross train/eval/test. Eval and test
retain their natural negative patches so false-positive behavior remains
measurable.

The archive layout is:

```text
local256/
|-- dataset_info.json
|-- balance_report.json
|-- split_manifest.json
|-- images/
|   |-- train/<raw-tile-id>/*.png
|   |-- eval/<raw-tile-id>/*.png
|   `-- test/<raw-tile-id>/*.png
`-- phase_a/
    |-- train.jsonl
    |-- eval.jsonl
    |-- test.jsonl
    |-- meta_train.jsonl
    |-- meta_eval.jsonl
    `-- meta_test.jsonl
```

This release contains only `local256` and Phase A. It does not contain
`context512_roi256` or Phase B state-update records.

## Train distribution

The finalized observed distribution is:

| Difficulty | Records | Share |
|---|---:|---:|
| empty | 0 | 0.000% |
| easy | 165,000 | 30.000% |
| medium | 195,816 | 35.603% |
| hard | 134,184 | 24.397% |
| very_hard | 55,000 | 10.000% |

Exactly 165,000 train records (30.000%) contain at least one intersection.
The 30% constraint is global, not per difficulty bucket.

The initial requested difficulty shares were 30% easy, 33% medium, 27% hard,
and 10% very-hard. There were 14,316 fewer unique hard candidates than the
requested 148,500, so the production allocator legally moved those records to
medium. The final data still contains 550,000 unique patch ids and no exact
repeats. This redistribution is intentional and must not be treated as data
corruption.

Difficulty rule version is `geometry_v2_strict_easy_no_cut_score`. Crop-boundary
`cut` endpoints do not by themselves make a sample hard. Difficulty is based on
geometry density and structure, including line/point count, intersections,
forks, cycles, crossings, curvature, sharp turns, and short fragments.

## Images and coordinates

- PNG size on disk: `256x256`.
- View mode: `local256`; the complete image is supervised.
- Coordinate mode: `norm1000`.
- Coordinate range: integer `0..1000`, relative to the 256x256 patch.
- The image is not upscaled in the dataset files.

Coordinate conversion is:

```text
x_norm = round(x_pixel / 255 * 1000)
y_norm = round(y_pixel / 255 * 1000)
x_pixel = round(x_norm / 1000 * 255)
y_pixel = round(y_norm / 1000 * 255)
```

For Jiangjihua's Hugging Face DINOv2-L route, `input_image_size=518` resizes the
256x256 PNG inside the image processor and produces a 37x37 visual-token grid
(1369 patch tokens). This must not trigger coordinate rescaling: targets remain
norm1000 regardless of the visual encoder input size.

## JSONL contract

Every JSONL line is one object with these top-level fields:

```json
{
  "id": "unique_patch_id",
  "image": "images/train/raw_tile_id/unique_patch_id.png",
  "meta": {},
  "conversations": [
    {"from": "human", "value": "<image>\n..."},
    {"from": "gpt", "value": "{\"lines\":[...]}"}
  ]
}
```

The assistant `value` is a JSON string, not a nested object and not Markdown.
Its parsed schema is:

```json
{
  "lines": [
    {
      "category": "centerline",
      "lane_type": "common",
      "start_type": "cut",
      "end_type": "inside",
      "points": [[0, 500], [420, 510]]
    },
    {
      "category": "intersection",
      "intersection_type": "t_intersection",
      "is_cut": false,
      "points": [[100, 100], [900, 100], [900, 900], [100, 100]]
    }
  ]
}
```

Centerlines have at least two points and endpoint types in `cut|inside`.
Intersection polygons have at least four points and are explicitly closed.

## Semantic taxonomy

Every centerline has exactly one `lane_type`:

| Source LaneType | Output `lane_type` |
|---|---|
| 1 | `common` |
| 2 | `right_turn` |
| 3 | excluded completely (U-turn reference line) |
| remaining or missing | `other` |

Every intersection has exactly one `intersection_type`:

| Source `(IntersectionType, IntersectionSubType)` | Output |
|---|---|
| `(1,1)` | `common` |
| `(1,2)` | `t_intersection` |
| `(1,3)` | `small_untyped` |
| `(4,1)` | `t_lane_change_area` |
| remaining or missing | `other` |

The target contains no public `intersection_subtype` field. Training and
inference should emit the textual types above, not numeric source codes.

## Prompt contract

The Phase-A user prompt asks for the complete road map in the current BEV
patch, states the norm1000 coordinate system, requires JSON-only output, and
defines the allowed `lane_type` and `intersection_type` values. It also contains:

```text
Incoming traces JSON:
[]

Incoming intersections JSON:
[]
```

These empty Phase-A fields are intentional and compatible with the existing
conversation/data loader. Do not replace the dataset prompt during the first
controlled training run. Prompt changes should be a separate experiment.

## Generation provenance

The dataset was generated from seven raw RC source roots:

```text
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0902_1935/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0426_1639/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0922_0901/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1013_2100/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1023_2143/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1029_1153/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1120_2889/
```

Native 256 windows are selected first. When needed, additional unique windows
come from a 128-pixel translation grid. These are genuine raster/GeoJSON
recrops whose geometry and endpoint types are recomputed; they are not repeated
or artificially shifted copies of an existing PNG.

The local staged candidates remain under `D:\data\fulldata\staging`, so later
subsets or larger releases can be finalized without downloading the raw source
again. The currently uploaded 550k tar is immutable and should be versioned
rather than edited in place.

## Validation status

The completed audit observed:

- 550,000 train, 50,925 eval, and 53,796 test records.
- Exactly 30% intersection-bearing train records.
- 50 rendered samples from each of easy, medium, hard, and very-hard.
- No format, semantic, coordinate, image-reference, split-leakage, or image-size
  errors were reported.

The first audit showed two difficulty-distribution errors because the original
auditor compared against requested quotas instead of the production
`final_bucket_counts`. That auditor bug was fixed in GitHub `MLLM` commit
`53f077b`; the legal medium/hard redistribution is now read from
`dataset_info.json`. The dataset bytes did not change.

For a local recheck:

```bat
python scripts\tools\validate_visualize_rc_dataset_v2.py --dataset-root "D:\data\fulldata\output\local256" --output-dir "D:\data\fulldata\output\local256_validation" --visualize-per-difficulty 50
```

## Jiangjihua-route integration

Primary original-DINOv2 launcher:

```text
scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_original_dinov2_caprl4b_nodeepstack_npu.sh
```

Its intended architecture is:

```text
original facebook DINOv2-L
  -> selected visual layer -2
  -> mlp2x_gelu projector
  -> CapRL-Qwen3VL-4B-derived text LLM
```

The recipe disables DeepStack and direct visual layer fusion. It uses the
original `facebook_dinov2-large` vision tower under Jiangjihua's checkpoint OBS
root and unfreezes the vision tower, projector, and LLM in the Jiangjihua-style
SFT policy. Ordinary decimal coordinates are the baseline
(`COORDINATE_TOKEN_MODE=none`). Discrete coordinate tokens must be a later
ablation.

The private-data DINOv2 last-2-block recipe remains available as a separate
comparison launcher:

```text
scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_private_dinov2_last2_caprl4b_nodeepstack_npu.sh
```

The launcher currently has an obsolete default dataset URI. The implementing
agent must change or override it with:

```bash
DATASET_OBS_PATH=obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256/local.tar
DATASET_DIR_NAME=local256
```

After extraction, resolve paths as:

```text
DATASET_PATH=<extract-root>/local256
TRAIN_PATH=<DATASET_PATH>/phase_a/train.jsonl
EVAL_PATH=<DATASET_PATH>/phase_a/eval.jsonl
TEST_PATH=<DATASET_PATH>/phase_a/test.jsonl
IMAGE_FOLDER=<DATASET_PATH>
```

The `image` values are already relative to `IMAGE_FOLDER`; do not set it to the
`images/` subdirectory.

The original-DINOv2 launcher downloads
`MODEL_OBS_PATH/facebook_dinov2-large` directly. It does not use the private
DINOv2 segmentation resolver or `DINOV2_TRAIN_OUTPUT_OBS_PATH`.

## Required rollout sequence

1. Pull the latest `MLLM` branch.
2. Update/override the dataset OBS URI above.
3. Run the launcher with `INSPECT_ONLY=True`. It must download, extract, resolve
   `local256`, and pass the semantic/taxonomy/image preflight without training.
4. Confirm the original DINOv2 `facebook_dinov2-large` and CapRL-Qwen3VL-4B assets.
5. Run an 8-NPU smoke with `MAX_STEPS=20`, micro-batch 1, BF16, gradient
   checkpointing, and the same ZeRO configuration intended for DI.
6. Inspect loss, trainable-parameter groups, truncation/token statistics, NPU
   memory, and `DI_throughput` output.
7. Only then launch the formal multi-node run. DI supplies `OUTPUT_URL`; the
   launcher must not hard-code a cloud output destination.

For the first controlled run, do not simultaneously enable DeepStack, visual
layer fusion, coordinate tokens, prompt rewriting, or a different image size.
Those changes would make it impossible to attribute gains or regressions to
the new dataset.

## Agent checklist

- [ ] Uses the new `yw-ads-training-2-gy1` OBS bucket and `local.tar` filename.
- [ ] Does not reconvert or renormalize the dataset.
- [ ] Resolves the archive's `local256/` root correctly.
- [ ] Uses `phase_a/train.jsonl` and `IMAGE_FOLDER=<local256 root>`.
- [ ] Keeps target coordinates as norm1000 while DINOv2 resizes images to 518.
- [ ] Preserves `lane_type` and `intersection_type` in labels and generation.
- [ ] Excludes LaneType 3 and rejects public intersection subtype fields.
- [ ] Runs `INSPECT_ONLY=True` before model downloads/training.
- [ ] Uses the original DINOv2 `facebook_dinov2-large` checkpoint.
- [ ] Starts with ordinary numeric JSON and no DeepStack/layer fusion.
- [ ] Runs a same-recipe NPU smoke before the formal multi-node DI job.
- [ ] Prints `DI_throughput: ... samples/s/npu` during DI training.
