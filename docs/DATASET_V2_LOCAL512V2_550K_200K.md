# RC Dataset V2 local512v2

This workflow creates matching `local512v2` and `local256v2` Phase-A datasets.
It also creates a 512-resolution oracle-intersection-prompt variant. All three
use the same train difficulty balance:

```text
empty=0
easy=0.20
medium=0.30
hard=0.30
very_hard=0.20
intersection ratio=0.30
```

The semantic schema, source split, coordinate system, lane/intersection
taxonomy, ignored `LaneType=3/22`, stride, image geometry, and no-duplicate
policy remain the same as the previous true-512 build. The 200k train set is a
strict subset of the completed 550k train set.

The stream builder stages both resolutions before deleting each downloaded raw
source. `local512v2` uses train stride 256; `local256v2` uses train stride 128
to provide enough unique candidates for 550k. Existing verified
`<work-root>/staging_local512` and `<work-root>/staging_local256` shards are
reused when `--resume` is used.

Run in Windows `cmd.exe`:

```bat
conda activate rc-dataset-v2-py313 && python scripts\tools\build_rc_dataset_v2_local512v2_windows.py --work-root "D:\data\fulldata_local512" --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" --archive-workers 16 --train-stride 256 --local256-train-stride 128 --quick-train-target-samples 200000 --resume
```

Outputs:

```text
<work-root>/
|-- output_local512v2_550k/
|   |-- local512v2/
|   `-- local512v2_intersection_prompt/
|-- output_local512v2_200k/
|   |-- local512v2/
|   `-- local512v2_intersection_prompt/
|-- output_local256v2_550k/
|   `-- local256v2/
|-- output_local256v2_200k/
|   `-- local256v2/
`-- packages/
    |-- local512v2_550k.tar
    |-- local512v2_200k.tar
    |-- local512v2_intersection_prompt_550k.tar
    |-- local512v2_intersection_prompt_200k.tar
    |-- local256v2_550k.tar
    `-- local256v2_200k.tar
```

The standard variants predict centerlines and intersections. The
`intersection_prompt` variants put current-patch intersection ground truth in
the user prompt and supervise centerlines only.
