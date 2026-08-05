# Context512 ROI Triplet GT Conversion

## Purpose

`scripts/tools/convert_context512_roi_triplet_gt_to_dataset_v2.py` converts an
already cropped context512 dataset into the repository's Stage A Dataset V2
contract. It is a schema conversion and does not resample or rebalance the
source records.

Each source record must contain one clean BEV image, one Pose image, and one
Raw-Lane image. The source JSON lists them as BEV, Pose, Raw-Lane; the output
model order is deliberately changed to:

1. clean BEV road-structure image;
2. Raw-Lane image predicted by the PV-camera model;
3. historical vehicle-trajectory Pose image.

The converter copies or hard-links the three existing images. It does not try
to remove a Raw-Lane overlay from an image. A sampled image-header preflight
requires the source triplets to be 512x512, so a local256 directory with actual
256x256 assets cannot be silently mislabeled as context512.

## Source Contract

The converter discovers GT JSON files recursively. A GT file contains a JSON
list whose records follow this shape:

```json
{
  "id": "tile_id_r13_c9_p03",
  "GT": [
    {"lane": "[{\"category\":1,\"coords\":[[0,0],[255,255]]}]"},
    {"intersection": "[{\"category\":\"1_1\",\"coords\":[...]}]"},
    {"patch_size": 256}
  ],
  "image": [
    "tile_id/r13_c9_p03.png",
    "tile_id/r13_c9_p03_pose.png",
    "tile_id/r13_c9_p03_raw_lane.png"
  ]
}
```

Lane and intersection values are embedded JSON strings. Their coordinates are
0..255 pixels relative to the center 256x256 ROI, not the full 512 image.

## Output Contract

```text
dataset_root/
  images/{train,eval,test}/...
  raw_lane_images/{train,eval,test}/...
  pose_images/{train,eval,test}/...
  phase_a/{train,eval,test}.jsonl
  phase_a/meta_{train,eval,test}.jsonl
  dataset_info.json
  split_manifest.json
  semantic_schema_report.json
  build_summary.json
  conversion_validation.json
```

Coordinates are converted to norm1000 relative to the 256x256 ROI. LaneType 3
and 22 are removed. Other lane and intersection categories use the canonical
Dataset V2 mappings in `data_process/state_update_dataset_common.py`.

The generated prompt contains exactly three `<image>` tokens and uses the
same BEV, Raw-Lane, Pose order as the `images` array.

## Windows Command

For the production OBS source, use the all-in-one downloader, archive
extractor, converter, validator, and packager. The two OBS defaults are already
embedded in this entrypoint:

```text
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModelTrain/sjn_context_512_roi_256/
obs://yw-ncasd-result-gy1/data/RCDataset/BaseModelTrain/sjn_context_512_roi_256/GT_json/
```

Run this as one line from the repository root:

```cmd
python scripts\tools\build_context512_roi_triplet_gt_dataset_v2_from_obs_windows.py --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" --work-root "D:\data\sjn_context512_roi256_three_image_dataset_v2" --obsutil-jobs 16 --archive-workers 8 --resume
```

It creates:

```text
D:\data\sjn_context512_roi256_three_image_dataset_v2\download\source
D:\data\sjn_context512_roi256_three_image_dataset_v2\download\GT_json
D:\data\sjn_context512_roi256_three_image_dataset_v2\extracted_images
D:\data\sjn_context512_roi256_three_image_dataset_v2\output\context512_roi256_three_image_full
D:\data\sjn_context512_roi256_three_image_dataset_v2\packages\context512_roi256_three_image_full.tar
D:\data\sjn_context512_roi256_three_image_dataset_v2\pipeline_summary.json
```

The source OBS directory contains `GT_json` plus TAR packages. TAR, TAR.GZ,
TGZ, and ZIP files are extracted in parallel into isolated subdirectories
under `extracted_images`; packages cannot overwrite one another. Each archive
has an independent completion marker. The download completion markers are
written only after `obsutil` exits successfully. With `--resume`, completed
downloads and archive extractions are reused, already correct hard-linked
images are reused, stale generated images are replaced by source content, and
a current TAR package is not rebuilt.

### Local-Only Conversion

Run from the repository root in the Dataset V2 Python environment:

```cmd
python scripts\tools\convert_context512_roi_triplet_gt_to_dataset_v2.py --input-root "D:\data\fulldata_rawlane_pose_three_image_800k\output_rawlane_pose_three_image_800k\local256" --output-root "D:\data\context512_roi256_three_image_dataset_v2_full" --copy-mode hardlink --image-check-mode sampled --image-check-limit 10000 --resume --package
```

If the GT JSON files are outside `--input-root`, add:

```cmd
--annotation-root "D:\path\to\gt_json_root"
```

If the image tree is outside `--input-root`, add:

```cmd
--image-root "D:\path\to\images"
```

`--copy-mode hardlink` avoids duplicating image bytes when source and output
are on the same NTFS volume; it falls back to a normal copy when a hard link is
not possible. `--package` writes a sibling TAR after conversion validation.

The command succeeds only after `conversion_validation.json` is written with
`status=passed`. Check `build_summary.json` for source count, split count,
difficulty count, ignored LaneType count, and semantic type count.
