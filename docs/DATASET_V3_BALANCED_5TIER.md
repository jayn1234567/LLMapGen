# RC Dataset V3: five-tier balanced datasets

## Outputs

This workflow reuses completed staging shards and creates four Phase-A assets:

| Dataset | Train samples | Image / supervised area | Assistant target |
|---|---:|---|---|
| `local512v3` | 550,000 | 512 / 512 | centerlines and intersections |
| `local512v3_intersection_prompt` | 550,000 | 512 / 512 | centerlines only; intersection GT is in the user prompt |
| `context512_roi256v3` | 550,000 | 512 / center 256 ROI | centerlines and intersections in the ROI |
| `context512_roi256v3` quick subset | 200,000 | 512 / center 256 ROI | same as above |

The standard and oracle-prompt local512 datasets are row-wise paired. The
200k context train split is a strict subset of the 550k context train split.
There are no empty train samples and no exact duplicate sample IDs.

## Difficulty distribution

Dataset V3 adds `very_easy` below `easy`. The requested train distribution is:

```text
empty=0
very_easy=0.05
easy=0.20
medium=0.30
hard=0.30
very_hard=0.15
intersection target ratio=0.30
```

Exact 550k quotas are 27,500 very-easy, 110,000 easy, 165,000 medium,
165,000 hard, and 82,500 very-hard samples. Exact 200k quotas are 10,000,
40,000, 60,000, 60,000, and 30,000 respectively.

Difficulty is recomputed from the retained geometry rather than trusting the
old staging label. The 512-target and 256-ROI variants use separate
resolution-aware thresholds. The classifier considers line and point counts,
intersection count, forks, loops, crossings, lane-change-like structure,
short fragments, turning angles, and non-common lane types. Excess
`very_easy` candidates are never used to fill shortages in another bucket.

## Comparable evaluation splits

All four datasets must use exactly the same raw full-image IDs for `eval` and
`test`. The builder checks this twice:

1. Before generation, it compares the raw-image split ownership in the two
   staging roots.
2. After generation, it scans the materialized `phase_a/eval.jsonl` and
   `phase_a/test.jsonl` files and compares `meta.raw_sample_id` sets across all
   four datasets.

The second check also verifies that each dataset's manifest matches the JSONL
records. A mismatch stops the build. Patch coordinates and patch counts can
differ between true-512 and context-ROI views, but their source full-image
sets cannot differ.

## Windows command

Run in `cmd.exe` as one line. Point the two staging arguments at the retained
true-512 and context512-ROI256 staging directories:

```bat
conda activate rc-dataset-v2-py313 && python scripts\tools\build_rc_dataset_v3_balanced_windows.py --work-root "D:\data\fulldata" --local512-staging-root "D:\data\fulldata_local512\staging_local512" --context-staging-root "D:\data\fulldata_context512\staging_context512" --visualize-per-difficulty 100 --image-decode-mode sampled --resume
```

If both staging directories are already under the same work root as
`staging_local512` and `staging_context512`, the two explicit staging
arguments may be omitted.

The script first writes candidate visualizations to
`<work-root>/difficulty_audit_v3/{local512,context512_roi256}/viz_by_difficulty`.
It verifies that every difficulty bucket has enough unique candidates for the
exact 550k quota before materializing images. It then builds, validates, and
packages the datasets. Final archives are:

```text
<work-root>/packages_v3/local512v3_550k.tar
<work-root>/packages_v3/local512v3_intersection_prompt_550k.tar
<work-root>/packages_v3/context512_roi256v3_550k.tar
<work-root>/packages_v3/context512_roi256v3_200k.tar
```

The cross-dataset check is saved as
`<work-root>/output_dataset_v3/eval_source_consistency.json`. Re-running the
same command with `--resume` reuses compatible audits and completed outputs.
