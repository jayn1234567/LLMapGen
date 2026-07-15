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

Lane GeoJSON records are filtered before patch clipping. `LaneType=3` is a
U-turn reference line rather than a supervised road centerline, so it is
excluded from every Dataset V2 variant. `LaneType=2` is retained and recorded
internally as `right_turn`; the public centerline JSON schema remains unchanged.

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
| empty | 0% | 0 |
| easy, non-empty | 35% | 157,500 |
| medium | 30% | 135,000 |
| hard | 25% | 112,500 |
| very hard | 10% | 45,000 |

The target overall intersection share is 30% (135,000 records). This is one
global constraint, not a separate 30% requirement inside every difficulty
bucket. The allocator follows the natural intersection density of each bucket
and prefers unique records before introducing repeats. `balance_report.json`
records available, planned, selected, and repeated intersection counts for
every bucket.

If a difficulty bucket has too few unique samples, the builder repeats records
in `train.jsonl`; the PNG is stored once. Repeated records receive ids such as
`...__repeat001` and retain `meta.base_sample_id`. Disable this behavior with
`--no-oversample-short-buckets` (the build then fails if 450,000 cannot be
filled).

The zero-empty rule applies to train selection. Eval and test retain their
natural, complete non-black patch distributions, including valid negative
patches, so false-positive behavior remains measurable.
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
  --resume
```

The formal Ascend-side entrypoint pins the no-empty, globally 30%-intersection
recipe and uses a versioned OBS destination:

```bash
cd /cache/jn/MLLM_project-unimapgen_v7

WORK_ROOT=/cache/jn/rc_dataset_v2_noempty_i30 \
bash scripts/npu/data/build_rc_dataset_v2_balanced_noempty_i30_npu.sh
```

Set `INSTALL_DATA_DEPS=true` only when the active Python is missing GeoPandas,
Rasterio, Shapely, or the other packages in `data_process/requirements.txt`.
The script uses Huawei MoXing by default and does not initialize NPU devices.
Its default destination is
`obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/rc_dataset_v2_noempty_i30/`.

The default destination is:

```text
obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/rc_dataset_v2/
```

`--resume` reuses downloads with a completed marker, keeps existing PNGs, and
reuses completed tar packages. Do not treat a directory without its
`.obs_download_complete.json` marker as a complete download.

The default upload mode creates:

```text
obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/rc_dataset_v2/
|-- local256.tar
|-- context512_roi256.tar
|-- build_summary.json
|-- split_manifest.json
`-- balance_report.json
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
  oversampling, global intersection plan, and per-bucket natural/selected
  intersection ratios.
- `split_manifest.json`: source roots, duplicate ids, and raw sample split.

## Local Windows build

Use a separate data-preparation environment. PyTorch, torch_npu, Transformers,
and the training environment are not required.

Create or update the Conda environment from PowerShell or Anaconda Prompt:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\tools\setup_rc_dataset_v2_windows.ps1
conda activate rc-dataset-v2
```

The script installs Python 3.11, NumPy 1.26, Pillow, tqdm, GeoPandas, Rasterio,
Shapely, and pyproj. If `conda-forge` is unavailable on the intranet, pass the
company Conda channel with `-Channel YOUR_CHANNEL`.

For local Windows OBS access, use Huawei `obsutil.exe` instead of installing a
package named `moxing` from public PyPI. Add `obsutil.exe` to `PATH`, then use
interactive configuration so AK/SK do not appear in shell history:

```powershell
obsutil config -interactive
obsutil ls obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/
```

First run a local-only structure smoke test from the repository root:

```powershell
python .\scripts\tools\build_rc_dataset_v2_from_obs.py `
  --work-root "D:\rcv2_smoke" `
  --obs-backend obsutil `
  --source-obs-root "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0426_1639/" `
  --limit-samples 12 `
  --train-target-samples 200 `
  --skip-upload `
  --resume
```

`--limit-samples` limits processing after download; it does not limit objects
downloaded from a supplied OBS prefix. The example downloads one task instead
of all seven. For a truly small download, first inspect the task with obsutil or
OBS Browser+, then set `--source-obs-root` to one complete raw-sample subfolder.

After the smoke build finishes, run the full build:

```powershell
python .\scripts\tools\build_rc_dataset_v2_from_obs.py `
  --work-root "D:\rcv2" `
  --obs-backend obsutil `
  --resume `
  --remove-package-after-upload
```

`D:\rcv2` is an example. Use a short path on a large NTFS disk. Avoid FAT32
because each tar can exceed 4 GiB. A short path also reduces the risk of the
Windows 260-character path limit with long raw sample ids. The pipeline stores
seven raw sources, both generated image variants, and temporarily one tar, so
several hundred GiB of free space may be required. The command prints detected
free space before downloading. OBS object names are case-sensitive while
Windows paths are not; review the source task if obsutil reports case-collision
warnings rather than allowing one object to overwrite another silently.

The default OBS upload directory is already set to:

```text
obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/rc_dataset_v2/
```

Use `--obsutil-path "C:\path\to\obsutil.exe"` if it is not on `PATH`. Use
`--obsutil-config "C:\path\to\.obsutilconfig"` when the config file is not in
the current Windows user's home directory.
