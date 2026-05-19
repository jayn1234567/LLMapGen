# State-update Data Processing

This folder contains the current RC raw-data processors for the UniMapGen-style training flow.

## Entrypoints

- `build_lane_dataset.py`
  - Reads `label_check_crop/Lane.geojson`.
  - Generates centerline-only targets.
- `build_lane_intersection_dataset.py`
  - Reads `label_check_crop/Lane.geojson` and `label_check_crop/intersection.geojson`.
  - Generates centerline + intersection targets.
  - By default, raw samples whose intersection GeoJSON has zero features are skipped. Pass `--allow-empty-intersection-files` to keep them as negative samples.

Both scripts accept an input directory that contains `.tar.gz` archives or already extracted sample folders.
Archives are deleted after successful extraction unless `--keep-archives` is passed.

Expected extracted sample layout:

```text
sample_id/
├── label_check_crop/
│   ├── Lane.geojson
│   └── Intersection.geojson or intersection.geojson
├── inter_patch_tif/
│   └── 0_inter.tif
└── patch_tif/
    └── 0_edit_poly.tif
```

## Outputs

The scripts split raw samples before patch generation, then build Phase A and Phase B from the same split:

```text
output_root/
├── split_manifest.json
├── dataset_info.json
├── phase_a/
│   ├── train.jsonl
│   ├── eval.jsonl
│   ├── test.jsonl
│   ├── meta_train.jsonl
│   ├── meta_eval.jsonl
│   └── meta_test.jsonl
└── phase_b/
    ├── train.jsonl
    ├── eval.jsonl
    ├── test.jsonl
    ├── meta_train.jsonl
    ├── meta_eval.jsonl
    └── meta_test.jsonl
```

`phase_a` uses empty incoming hints. `phase_b` uses GT left/top hints from already-processed neighboring patches.
The train/eval/test split unit is the raw sample folder, not a patch, so eval/test patches cannot leak into either Phase A or Phase B training data.
`--train-ratio` controls the raw-sample train share, `--eval-ratio` or `--eval-count` reserves raw samples for eval, and the remaining raw samples become final test.
Images are padded with black pixels to a patch-size multiple before patching; metadata keeps both padded `source_image_size` and `original_source_image_size`.

## Prompt Shape

Prompts stay close to the legacy training data and do not include an explicit output schema.

Phase A lane-only:

```text
<image>
Please construct the complete road map in the current BEV (Bird's Eye View) image patch.
Coordinates use a normalized 0-1000 grid over the original 256x256 image patch.

Incoming traces JSON:
[]
```

Phase B lane + intersection:

```text
<image>
Please construct the complete road map in the current BEV (Bird's Eye View) image patch.
Coordinates use a normalized 0-1000 grid over the original 256x256 image patch.

Incoming traces JSON:
[{"id":"L0","side":"left","points":[[-98,667],[-51,584],[-4,502]]}]

Incoming intersections JSON:
[{"id":"IL0","side":"left","points":[[-4,643]]}]

Each incoming trace has 1 to 3 points. If multiple points are present, they are ordered from the previous patch interior toward the current patch boundary.
Incoming traces are continuity hints only; they may be incomplete or absent.
Each incoming intersection has 1 to 3 boundary points from neighboring patches.
```

## Target Schema

Centerline-only:

```json
{"lines":[{"category":"centerline","start_type":"cut","end_type":"inside","points":[[0,494],[357,549]]}]}
```

Centerline + intersection:

```json
{"lines":[{"category":"centerline","start_type":"cut","end_type":"inside","points":[[0,494],[357,549]]},{"category":"intersection","is_cut":true,"points":[[0,361],[643,361],[643,643],[0,643],[0,361]]}]}
```

Rules:

- `centerline` uses `start_type/end_type = cut|inside`.
- `intersection` uses `is_cut = true|false` and stays a closed polyline.
- SFT JSONL defaults to `coord_mode=norm1000`: all assistant target points are integers in `[0,1000]`, normalized over the original patch. Images are still 256x256 by default.
- `--coord-mode pixel` keeps the legacy `[0, patch_size - 1]` pixel labels when explicitly needed.
- Phase B incoming hints use the same coordinate mode as targets. Left/top hints may be outside the `[0,1000]` range because they come from neighboring patches.
- GeoJSON properties such as `Id`, `IntersectionType`, `IsRegular`, and `IntersectionSubType` are retained in meta outputs, not assistant targets.

Normalization formula:

```text
x_norm = round(x_pixel / (patch_width  - 1) * coord_range)
y_norm = round(y_pixel / (patch_height - 1) * coord_range)

x_pixel = round(x_norm / coord_range * (patch_width  - 1))
y_pixel = round(y_norm / coord_range * (patch_height - 1))
```

For the default `patch_size=256` and `coord_range=1000`:

```text
[0, 0]     -> [0, 0]
[255,255]  -> [1000,1000]
[128,128]  -> [502,502]
```

Target points and parsed model outputs are clamped into the valid in-patch range.
Incoming hints are not clamped, so a left-neighbor pixel point like `[-1,128]`
becomes approximately `[-4,502]`.

## Cut Rules

`centerline` cut is endpoint-level:

- A natural original-line endpoint is `inside`, even if it lies on a patch boundary.
- A new endpoint produced by clipping against the patch bbox is `cut`.
- If one original line creates multiple disconnected clipped segments in a patch, each segment becomes one centerline.

`intersection` cut is region-level:

- `is_cut=false` when the original polygon is fully inside the patch bbox.
- `is_cut=true` when the original polygon is clipped by the patch bbox.
- Polygon points are always emitted as the current patch-local clipped polygon.

## Ordering

Incoming order:

```text
centerline: L0, L1, ... then T0, T1, ...
intersection: IL0, IL1, ... then IT0, IT1, ...
```

Target line order:

```text
1. centerlines continuing from the left
2. centerlines continuing from the top
3. other centerlines
4. intersections continuing from the left
5. intersections continuing from the top
6. other intersections
```

For centerlines that continue an incoming trace, target points start at the current patch boundary and continue inward.
Target points never include out-of-patch hint coordinates such as negative x/y or values above the coordinate range.

## Example Commands

Lane-only:

```bash
python data_process/build_lane_dataset.py \
  --input-root /path/to/train \
  --output-root /path/to/output_lane \
  --patch-size 256 \
  --stride 256 \
  --coord-mode norm1000 \
  --train-ratio 0.9 \
  --eval-ratio 0.05
```

Lane + intersection:

```bash
python data_process/build_lane_intersection_dataset.py \
  --input-root /path/to/train \
  --output-root /path/to/output_lane_intersection \
  --patch-size 256 \
  --stride 256 \
  --coord-mode norm1000 \
  --train-ratio 0.9 \
  --eval-ratio 0.05
```

For a small smoke run:

```bash
python data_process/build_lane_dataset.py \
  --input-root /path/to/train \
  --output-root /tmp/lane_smoke \
  --limit-samples 1 \
  --max-patches-per-sample 20 \
  --coord-mode norm1000 \
  --keep-archives
```

Lane + intersection smoke run:

```bash
python data_process/build_lane_intersection_dataset.py \
  --input-root /path/to/train \
  --output-root /tmp/lane_intersection_smoke \
  --limit-samples 1 \
  --max-patches-per-sample 20 \
  --coord-mode norm1000 \
  --keep-archives
```

Debug whether one extracted raw sample can load and clip `Intersection.geojson`:

```bash
python data_process/debug_intersection_parse.py \
  /path/to/one_sample \
  --patch-size 256 \
  --stride 256
```

Key fields:

- `intersection_geojson exists=True`: the script found `Intersection.geojson` or `intersection.geojson`.
- `loaded_intersection_polygon_count`: number of polygon geometries read from the file.
- `intersection_patch_count`: number of nonblack patches where intersection targets were clipped.
- `intersection_examples`: example patch-local intersection targets.
