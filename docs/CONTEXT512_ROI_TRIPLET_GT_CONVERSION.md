# Context512 ROI Triplet GT Conversion

## Purpose

`scripts/tools/convert_context512_roi_triplet_gt_to_dataset_v2.py` converts an
already cropped context512 dataset into a complete reusable Dataset V2 sample
pool. `scripts/tools/build_balanced_three_image_dataset_v2.py` then creates the
strictly balanced Stage A training dataset used for packaging.

Each source record must contain one clean BEV image, one Pose image, and one
Raw-Lane image. The source JSON lists them as BEV, Pose, Raw-Lane; the output
model order is deliberately changed to:

1. clean BEV road-structure image;
2. Raw-Lane image predicted by the PV-camera model;
3. historical vehicle-trajectory Pose image.

The converter builds one group-directory index at startup, then resolves each
sample from that index instead of recursively searching the extracted tree for
every new group. GT JSON files are parsed once. The converter copies or
hard-links the three existing images. It does not try
to remove a Raw-Lane overlay from an image. A sampled image-header preflight
requires every triplet to have matching dimensions. Full 512x512 triplets are
hard-linked unchanged. The production OBS pipeline skips boundary-clipped
triplets such as 256x512 or 512x256 and records them in
`skipped_samples.jsonl`; it does not pad or resize them. The lower-level
converter retains explicit `--non-512-policy pad` and `error` modes for audits,
but its default is `skip`.

The production OBS wrapper also uses `--missing-triplet-policy skip`. A GT
record whose BEV, Raw-Lane, or Pose file is absent is written to
`skipped_samples.jsonl` with `reason=missing_image_triplet` and does not stop
the remaining conversion. The lower-level converter keeps the default
`--missing-triplet-policy error` for strict source audits.

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
D:\data\sjn_context512_roi256_three_image_dataset_v2\output\context512_roi256_three_image_balanced_800k
D:\data\sjn_context512_roi256_three_image_dataset_v2\packages\context512_roi256_three_image_balanced_800k.tar
D:\data\sjn_context512_roi256_three_image_dataset_v2\pipeline_summary.json
```

The full output is an intermediate sample pool and is retained for reproducible
rebalancing. The packaged 800k training split is sampled without replacement
using exact quotas:

| Stratum | Ratio | 800k count |
|---|---:|---:|
| empty | 5% | 40,000 |
| easy | 25% | 200,000 |
| medium | 33% | 264,000 |
| hard | 27% | 216,000 |
| very_hard | 10% | 80,000 |

Eval and test records are preserved unchanged. The selector never silently
redistributes a shortage. It writes `balance_preflight.json` and fails with the
exact per-bucket deficit.

If only the empty bucket is short, the Windows entrypoint automatically tries
these paired staging roots:

```text
D:\data\fulldata_context512\staging_context512
D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context
```

The first supplies clean context512 images; the second supplies matching
Raw-Lane, Pose, target JSON, and difficulty metadata. Donor samples are matched
by `source_index`, must contain all three valid images, and are excluded when
their ID is already present in the selected train/eval/test records.

The source OBS directory contains `GT_json` plus TAR packages. TAR, TAR.GZ,
TGZ, and ZIP files are extracted in parallel into isolated subdirectories
under `extracted_images`; packages cannot overwrite one another. Each archive
has an independent completion marker. The download completion markers are
written only after `obsutil` exits successfully. With `--resume`, completed
downloads and archive extractions are reused, already correct hard-linked
images are reused, stale generated images are replaced by source content, and
a current TAR package is not rebuilt. Archives containing no files are reported
as `empty` and skipped. Dataset conversion still requires every GT record to
resolve all three images, so a genuinely missing sample remains a hard error
with its exact sample ID.

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

The lower-level converter command creates the complete pool only. Run the
strict selector separately when using local-only conversion:

```cmd
python scripts\tools\build_balanced_three_image_dataset_v2.py --input-root "D:\data\context512_roi256_three_image_dataset_v2_full" --output-root "D:\data\context512_roi256_three_image_balanced_800k" --train-target-samples 800000 --difficulty-ratios "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10" --empty-donor-clean-staging-root "D:\data\fulldata_context512\staging_context512" --empty-donor-aux-staging-root "D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context" --copy-mode hardlink --resume --package
```

`--copy-mode hardlink` avoids duplicating image bytes when source and output
are on the same NTFS volume; it falls back to a normal copy when a hard link is
not possible. `--package` writes a sibling TAR after conversion validation.

The command succeeds only after `conversion_validation.json` is written with
`status=passed`. Check `build_summary.json` for source count, split count,
difficulty count, ignored LaneType count, and semantic type count.
