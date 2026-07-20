# RC Dataset V2 true-512 build

## Scope

This workflow creates four self-contained Phase-A datasets from the seven raw
RC sources:

| Dataset | Train records | Supervision |
|---|---:|---|
| `local512` | 550,000 | centerlines and intersections |
| `local512` quick subset | 100,000 | centerlines and intersections |
| `local512_intersection_prompt` | 550,000 | intersection GT in user prompt; centerlines only in assistant target |
| `local512_intersection_prompt` quick subset | 100,000 | same oracle-conditioned task |

This is a true `512x512` target patch. It is different from
`context512_roi256`, whose image is 512 but whose supervised center ROI remains
256. Coordinates use norm1000 across the complete 512 target:

```text
[0, 0] pixel -> [0, 0]
[511, 511] pixel -> [1000, 1000]
```

The 100k train split is selected strictly from the completed 550k train split.
Both task variants have identical sample IDs, image paths, split assignments,
difficulty distribution, and intersection distribution. Images are hard-linked
while building when the filesystem supports it.

## Build on Windows

Use the existing Dataset V2 conda environment. The command downloads and
processes one OBS source at a time, deletes each raw source after staging, then
finalizes, audits, visualizes, and packages all four assets.

Run this as one line in `cmd.exe`:

```bat
conda activate rc-dataset-v2-py313 && python scripts\tools\build_rc_dataset_v2_local512_windows.py --work-root "D:\data\fulldata_local512" --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" --archive-workers 16 --train-stride 256 --resume
```

The default train distribution is:

```text
empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10
intersection target ratio=0.30
```

Training uses a 256-pixel stride, so translated 512 windows augment the base
grid without exact duplicate records. Eval and test use a 512-pixel stride and
therefore retain a non-overlapping base grid. If the seven sources do not yield
550k selectable train patches, rerun the same command with
`--train-stride 128 --resume`; stride-aware resume validation will rebuild
incompatible source stages instead of mixing the two geometries.

## Outputs

```text
D:\data\fulldata_local512\
|-- output_550k\
|   |-- local512\
|   `-- local512_intersection_prompt\
|-- output_100k\
|   |-- local512\
|   `-- local512_intersection_prompt\
|-- packages\
|   |-- local512_550k.tar
|   |-- local512_100k.tar
|   |-- local512_intersection_prompt_550k.tar
|   `-- local512_intersection_prompt_100k.tar
|-- filters\
`-- staging_local512\
```

Each dataset contains `phase_a/{train,eval,test}.jsonl`, matching meta JSONL,
all referenced PNG files, `dataset_info.json`, `balance_report.json`, and
`split_manifest.json`. Validation reports and 50 visualizations per difficulty
are written beside each dataset directory.

The four tar files duplicate image bytes even though the unpacked prompt and
standard variants use hard links. Use `--skip-package` when temporary disk space
is tight, then package or upload one variant at a time.

## Oracle intersection task

The `local512_intersection_prompt` user message contains:

```text
Current-patch intersection ground truth JSON:
{"lines":[{"category":"intersection",...}]}
```

Its assistant target contains only centerlines. It must not emit intersection
objects. This experiment is oracle-conditioned: inference and evaluation must
also provide the current patch's ground-truth intersection polygons and types.
It is not directly comparable to image-only inference unless that same oracle
input is supplied.

The builder verifies that prompt intersections round-trip exactly from the
standard target, that no intersection leaks into the assistant target, and that
the standard/prompt variants remain paired row by row.
