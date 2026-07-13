# RC Dataset V2: balanced local/context A/B data

This pipeline builds two controlled Stage-A lane + intersection datasets from
the same raw RC samples. It keeps Jiangjihua's TIFF mask, GeoJSON coordinate
transform, lane/intersection clipping, endpoint typing, raw-sample split, and
norm1000 target rules.

## Built-in sources

`scripts/tools/build_rc_dataset_v2_from_obs.py` contains these seven defaults:

1. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0902_1935/`
2. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0426_1639/`
3. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0922_0901/`
4. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1013_2100/`
5. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1023_2143/`
6. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1029_1153/`
7. `obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1120_2889/`

There are seven roots, even though the initial estimate referred to six source
datasets. Duplicate raw sample ids are reported in `split_manifest.json`.
Default duplicate policy is `last`: a later source in the list wins.

## Controlled variants

### `local256`

- PNG on disk: `256x256`.
- Supervised target: the complete image.
- Target coordinates: norm1000 relative to this 256 target.

### `context512_roi256`

- PNG on disk: `512x512`.
- Supervised target: central ROI `[128,128,384,384)` only.
- Target coordinates: norm1000 relative to the central 256 ROI, not the full
  512 image.
- Context beyond the raw image is padded with black pixels. Reflection padding
  is deliberately not used because it can create fake road geometry.
- There is no visible ROI border in this first ablation. The prompt states the
  exact ROI. A border/corner-marker experiment should be separate.

The target geometry, patch ids, raw-sample split, train selection, and repeated
sampling counts are identical between the two variants. Rotation and offset
grids are disabled so visible context is the only A/B variable.

For Jiangjihua's DINOv2-L recipe, the processor resizes the complete input view
to `518x518`. Therefore:

- `local256`: the supervised target occupies the full 37x37 DINOv2 patch grid.
- `context512_roi256`: the supervised target occupies about half the image
  width after resize and roughly an 18x18 central token region; the remaining
  tokens provide surrounding context.

## Train distribution

Default Stage-A train size is 450,000 records:

| Stratum | Ratio | Records |
|---|---:|---:|
| natural empty | 5% | 22,500 |
| easy, non-empty | 30% | 135,000 |
| medium | 30% | 135,000 |
| hard | 25% | 112,500 |
| very hard | 10% | 45,000 |

The target overall intersection share is 38%. If a difficulty bucket has too
few unique samples, the builder repeats records in `train.jsonl`; the PNG is
stored once. Repeated records receive ids such as `...__repeat001` and retain
`meta.base_sample_id`. Disable this behavior with
`--no-oversample-short-buckets` (the build then fails if 450,000 cannot be
filled).

Eval and test retain their natural, complete non-black patch distributions.
Splitting happens by raw sample folder before patch generation, so patches from
one source map cannot leak across train/eval/test.

## One-command OBS build

Run this in the Ascend/DI environment that can import Huawei MoXing:

```bash
python -c "import moxing, geopandas, rasterio, shapely; print('dataset preflight ok')"
```

If the geospatial packages are absent, install `data_process/requirements.txt`
in a data-preparation environment. This is separate from the NPU training
dependency installation; do not replace a working `torch`/`torch_npu` pair just
to build the dataset.

```bash
cd /cache/jn/MLLM_project-unimapgen_v7

python scripts/tools/build_rc_dataset_v2_from_obs.py \
  --work-root /cache/rc_dataset_v2 \
  --output-obs-root obs://YOUR_BUCKET/YOUR_PATH/rc_dataset_v2 \
  --resume
```

`--resume` reuses downloads with a completed marker, keeps existing PNGs, and
reuses completed tar packages. Do not treat a directory without its
`.obs_download_complete.json` marker as a complete download.

The default upload mode creates:

```text
obs://YOUR_BUCKET/YOUR_PATH/rc_dataset_v2/
|-- local256.tar
|-- context512_roi256.tar
|-- build_summary.json
`-- split_manifest.json
```

Each tar contains one top-level directory with the same variant name. Point a
DI experiment at one tar at a time, then set the extracted dataset root to that
top-level directory.

To build locally without OBS operations, put the seven source directories under
the names printed by `--skip-download`, then run:

```bash
python scripts/tools/build_rc_dataset_v2_from_obs.py \
  --work-root /cache/rc_dataset_v2 \
  --skip-download \
  --skip-upload \
  --resume
```

For a small end-to-end smoke build, add `--limit-samples 12` and lower
`--train-target-samples` if oversampling is not desired.

## Output layout

```text
output/
|-- build_summary.json
|-- split_manifest.json
|-- manifests/
|   |-- train_candidates.jsonl
|   |-- train_selection.jsonl
|   `-- balance_report.json
|-- local256/
|   |-- dataset_info.json
|   |-- split_manifest.json
|   |-- images/{train,eval,test}/...
|   `-- phase_a/{train,eval,test}.jsonl
`-- context512_roi256/
    |-- dataset_info.json
    |-- split_manifest.json
    |-- images/{train,eval,test}/...
    `-- phase_a/{train,eval,test}.jsonl
```

This first V2 builder intentionally emits Stage A only. Phase B requires an
unbroken row-major patch lattice for neighbor traces; balancing individual
patches would break that property. A future Phase-B dataset should preserve the
full lattice and apply weighting in the sampler instead of deleting patches.

## Audit files

- `train_candidates.jsonl`: geometry metrics and difficulty tag for every
  unique train candidate.
- `train_selection.jsonl`: selected patch ids and repeat counts.
- `balance_report.json`: requested versus actual counts, unique counts,
  oversampling, and intersection ratio.
- `split_manifest.json`: source roots, duplicate ids, and raw sample split.
