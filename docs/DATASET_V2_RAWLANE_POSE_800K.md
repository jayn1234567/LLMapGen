# Dataset V2 Raw-Lane + Pose 800k

## Purpose

The raw-lane + pose build contains two paired Stage-A lane/intersection SFT
datasets: `rawlane_pose_local256_800k` and
`rawlane_pose_context512_roi256_800k`. They keep the existing Dataset V2
geometry, semantic classes, norm1000 coordinates, split policy, and difficulty
selection.

Each sample contains:

1. a BEV road image with `patch_tif/0_lane.tif` rendered as a white PV-camera
   lane overlay;
2. a separate black-background historical vehicle-trajectory image built from
   `patch_tif/0_pose.tif`.

The local variant stores two 256x256 images. The context variant stores two
512x512 context images and supervises only the central 256x256 ROI.

The pose raster is never painted onto the BEV image. Both source rasters are
masked by `patch_tif/0_edit_poly.tif` and cropped at identical coordinates.

## Target Distribution

- train samples: 800,000 unique patch ids
- empty: 0%
- easy: 30%
- medium: 33%
- hard: 27%
- very hard: 10%
- samples containing intersections: 30%
- train stride: 128
- eval/test stride: 256

If a requested difficulty bucket lacks candidates, the existing Dataset V2
redistribution policy fills it with unique candidates from neighboring useful
difficulty buckets. Exact repeats are not introduced.

## Record Format

```json
{
  "id": "...",
  "image": "images/train/.../sample.png",
  "images": [
    "images/train/.../sample.png",
    "pose_images/train/.../sample.png"
  ],
  "meta": {
    "input_image_roles": [
      "bev_road_structure",
      "historical_vehicle_trajectory"
    ],
    "pose_image_source": "patch_tif/0_pose.tif"
  },
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n..."},
    {"from": "gpt", "value": "{\"lines\":[...]}"}
  ]
}
```

The compatibility `image` field remains the primary BEV path. Training code
uses `images` when it is present and checks that its length equals the number
of `<image>` tokens.

## Prompt Addition

```text
<image>
<image>
The first image is the BEV road-structure image.
The second image is a historical vehicle-trajectory image: white lines are historical vehicle trajectories on a black background.
```

The existing raw-lane prompt sentence remains attached to the first image.

## Windows Build

Run from the repository root in the geospatial Dataset V2 conda environment:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_800k_windows.py --work-root "D:\data\fulldata_rawlane_pose" --obsutil-path "C:\path\to\obsutil.exe" --resume
```

The wrapper downloads and stages one source at a time, keeps only required
TIFF/GeoJSON members, globally balances 800k unique training records, validates
every two-image pair, and creates:

```text
D:\data\fulldata_rawlane_pose\output_rawlane_pose_256_context\rawlane_pose_local256_800k
D:\data\fulldata_rawlane_pose\output_rawlane_pose_256_context\rawlane_pose_context512_roi256_800k
D:\data\fulldata_rawlane_pose\packages_rawlane_pose\rawlane_pose_local256_800k.tar
D:\data\fulldata_rawlane_pose\packages_rawlane_pose\rawlane_pose_context512_roi256_800k.tar
```

For comparable evaluation, pass the canonical large-map split manifest:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_800k_windows.py --work-root "D:\data\fulldata_rawlane_pose" --fixed-source-split-manifest "D:\data\fixed_splits\rc_fixed_large_maps_v1.json" --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" --resume
```

See `docs/FIXED_SOURCE_EVAL_SPLIT.md` for manifest creation and leakage checks.

Use `--skip-download --resume` when the raw sources are already present. Use
`--skip-stage --resume` only after all source stage markers have completed.

## Model-Side Contract

`mllm/train/train_qwen.py` reads both image paths, produces a tensor shaped
`[batch, 2, channels, height, width]`, and preserves two image tokens during
prompt preprocessing. `mllm/model/llava_arch.py` maps each image to its own
visual-token sequence in prompt order.

Two full DINO image streams roughly double visual-token memory and sequence
length. Before a formal training run, use a short NPU smoke test and verify
that the configured model maximum length leaves enough room for target JSON.
