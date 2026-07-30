# RC Dataset V2 Raw-Lane Overlay 550k

## Purpose

This dataset variant adds a white lane overlay from the PV-camera small model
to the BEV input image. The overlay is read from each raw sample folder at:

```text
patch_tif/0_lane.tif
```

The label geometry, train/eval/test split policy, lane/intersection clipping,
norm1000 coordinate rules, and difficulty balancing follow Dataset V2.

## Variants

The builder creates two 550k Stage-A variants:

| Variant | Image on disk | Supervised region | Coordinates |
|---|---:|---|---|
| `rawlane_local256_550k` | `256x256` | full image | norm1000 over the 256 patch |
| `rawlane_context512_roi256_550k` | `512x512` | center ROI `[128,128,384,384)` | norm1000 over the center 256 ROI |

For `rawlane_context512_roi256_550k`, the white lane overlay is rendered over
the whole `512x512` context image, not only inside the center ROI. The target
labels and prompt still require predictions only for the center 256 ROI.

The raw-lane overlay is clipped by the same `patch_tif/0_edit_poly.tif` valid
image mask as the RGB input, so invalid black regions do not become non-empty
patches because of the overlay.

## Semantic Schema

Centerlines include `lane_type` with exactly one of:

```text
common
right_turn
waiting_area
bus_lane
main_auxiliary_connector
other
```

Intersections include `intersection_type` with exactly one of:

```text
common
t_intersection
small_untyped
t_lane_change_area
other
```

Source LaneType 3 and LaneType 22 are filtered out during dataset generation,
but this filtering detail is not written into the model prompt.

## Train Distribution

The raw-lane 550k variants use the same requested difficulty distribution as
the earlier 256 Dataset V2 recipe:

| Difficulty | Ratio | Records |
|---|---:|---:|
| empty | 0% | 0 |
| easy | 30% | 165,000 |
| medium | 33% | 181,500 |
| hard | 27% | 148,500 |
| very_hard | 10% | 55,000 |

The global train intersection target ratio is `30%`, or 165,000 records. This
is a global constraint, not a per-difficulty constraint.

## Prompt

For `rawlane_local256_550k`, Stage A uses:

```text
<image>
Please construct the complete road map in the current BEV (Bird's Eye View) image patch.
Coordinates use a normalized 0-1000 grid over the original 256x256 image patch.
The image also contains a white lane overlay predicted by a PV camera model. Do not copy it blindly when it conflicts with the visible BEV evidence.

Return only valid JSON in the form {"lines":[...]} with no extra explanation.
For every centerline, include "lane_type" with exactly one of: "common" for a regular centerline, "right_turn" for a right-turn-only centerline, "waiting_area" for a waiting-area centerline, "bus_lane" for a bus-lane centerline, "main_auxiliary_connector" for a connector between main and auxiliary roads, or "other" for any remaining lane class.
For every intersection, include "intersection_type" with exactly one of: "common" for a common intersection, "t_intersection" for a T-intersection, "small_untyped" for a small untyped intersection, or "t_lane_change_area" for a T-shaped lane-change area, or "other" for any remaining or unknown intersection class.

Incoming traces JSON:
[]

Incoming intersections JSON:
[]
```

For `rawlane_context512_roi256_550k`, the first coordinate block is replaced by:

```text
The input is a 512x512 context image centered on the target region.
Predict only map elements clipped to the central 256x256 target ROI [128,128,384,384).
Coordinates use a normalized 0-1000 grid over the 256x256 target ROI.
Coordinates are relative to the target ROI, not the full context image.
Do not output geometry that lies only outside the target ROI.
```

The raw-lane overlay sentence and JSON schema instructions remain the same.

## Build Command

After pulling the latest `MLLM` branch, run on the internal Windows machine:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_256_context_windows.py `
  --work-root "D:\data\fulldata_rawlane" `
  --obs-backend obsutil `
  --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" `
  --resume
```

Expected output directories:

```text
D:\data\fulldata_rawlane\output_rawlane_256_context\rawlane_local256_550k
D:\data\fulldata_rawlane\output_rawlane_256_context\rawlane_context512_roi256_550k
```

Expected packages:

```text
D:\data\fulldata_rawlane\packages_rawlane\rawlane_local256_550k.tar
D:\data\fulldata_rawlane\packages_rawlane\rawlane_context512_roi256_550k.tar
```

Use a fresh `work-root` or rebuild old staging roots, because old selective
extraction runs may not have extracted `patch_tif/0_lane.tif`.

## Final 200k Training Assets

The DI recipes consume the already selected 200k packages directly. They do
not resample the records on a DI worker:

| Difficulty | Records |
|---|---:|
| easy | 60,000 |
| medium | 66,000 |
| hard | 54,000 |
| very_hard | 20,000 |

```text
local256:
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_200k_rawlane/local256_200k.tar

context512_roi256:
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256_200k_rawlane/context512_roi256_200k.tar
```

Each launcher requires exactly 200,000 non-empty JSONL records, validates the
Raw-Lane overlay metadata, image size, norm1000 coordinates, semantic type
schema, and the Raw-Lane prompt before downloading model assets.

Both recipes use original DINOv2-L, CapRL-Qwen3VL-4B, no DeepStack,
full-parameter SFT, global batch size 128, and 8 epochs. The local256 recipe
uses micro-batch 4 per NPU. The context512/ROI256 recipe uses micro-batch 2 and
gradient accumulation to keep the same global batch while preserving enough
NPU memory for full-parameter checkpoint saves.

DI entry for local256:

```bash
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_200k_stratified_original_dinov2_caprl4b_nodeepstack_npu.sh
```

DI entry for context512 with center ROI256 supervision:

```bash
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_context512_roi256_200k_stratified_original_dinov2_caprl4b_nodeepstack_npu.sh
```
