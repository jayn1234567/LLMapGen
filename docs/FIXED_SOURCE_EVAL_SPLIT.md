# Fixed Large-Map Evaluation Split

## Goal

All comparable Dataset V2/V3-style releases should reserve the same raw BEV
large maps for evaluation. A split is assigned at `raw_sample_id` level before
patch extraction, so no neighboring patch from an eval/test map can leak into
training.

The fixed manifest stores only explicit eval and test IDs. Every unlisted raw
map is assigned to train. This lets future training sources grow without
silently changing the benchmark.

## Recommended V1 Selection

The recommended path is the one-command orchestrator. It completes bootstrap
staging, freezes the manifest, repartitions that same staging in place during
finalization, validates both views, saves raw lane as an inactive auxiliary
PNG for future three-image ablations, and creates both tar packages. It does
not download, extract, or decode the seven OBS sources a second time:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_800k_fixed_eval_windows.py --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" --resume
```

The fixed 800k entry uses an exact 28% intersection target. The reusable
staging can reach at most 28.188875% after the frozen holdouts and difficulty
quotas are applied, so the former exact 30% request cannot be satisfied with
800k unique records. Use `--intersection-target-ratio` only when intentionally
building a differently specified release.

The individual commands below remain useful for auditing or recovery.

First complete all seven source stages without finalizing a random split:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_800k_windows.py --work-root "D:\data\fulldata_rawlane_pose" --obsutil-path "C:\Users\jWX1497058\Downloads\obsutil_windows_amd64\obsutil_windows_amd64_5.8.3\obsutil.exe" --stage-only --resume
```

Then create a source-balanced
manifest with 14 eval maps and 7 test maps:

```powershell
python scripts\tools\create_fixed_source_split_manifest.py --staging-root "D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context" --output "D:\data\fixed_splits\rc_fixed_large_maps_v1.json" --eval-count 14 --test-count 7 --seed 20260731
```

Selection uses only base-grid patches when estimating each large map. It
balances selection across source datasets and prefers maps whose patch count,
intersection share, and difficulty profile are representative of the complete
staging catalog. The adjacent `*_candidates.jsonl` report records every
candidate and can be audited before freezing V1.

Once V1 is accepted, do not overwrite it. Version a deliberate benchmark
change as `rc_fixed_large_maps_v2.json` instead.

## Build With The Fixed Split

Pass the same manifest to every future build:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_800k_windows.py --work-root "D:\data\fulldata_rawlane_pose_fixed_v1" --reuse-staging-root "D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context" --fixed-source-split-manifest "D:\data\fixed_splits\rc_fixed_large_maps_v1.json" --intersection-target-ratio 0.28 --resume
```

The generic streaming builder and direct Dataset V2 builder expose the same
`--fixed-source-split-manifest` option. This covers local256, local512,
context512_roi256, raw-lane, pose, and future paired views.

For wrappers that do not yet expose the option directly, set it once in the
PowerShell process before building:

```powershell
$env:RC_FIXED_SOURCE_SPLIT_MANIFEST="D:\data\fixed_splits\rc_fixed_large_maps_v1.json"
```

## Enforced Checks

- eval/test IDs must be disjoint;
- all fixed holdout maps must be present by default;
- no fixed eval/test map may appear in train;
- no unlisted map may appear in eval/test;
- every stage and finalized dataset must carry the same manifest SHA-256;
- the output `split_manifest.json` and `dataset_info.json` record the manifest
  ID, file hash, coverage report, and source-level counts.

`--allow-missing-fixed-holdouts` exists only for source-subset smoke tests. Do
not use it for benchmark datasets.

## Existing Staging

Bootstrap stages cut with the hash-ratio policy are reusable through
`--repartition-existing-stages-by-fixed-manifest`. Finalization assigns every
record according to the fixed raw-map manifest and rewrites image paths to the
new split. A fixed eval/test map keeps only canonical base-grid patches, so
stride=128 translated training augmentation cannot leak into evaluation.

When a bootstrap eval/test map becomes fixed train, only its already staged
base-grid candidates are available; missing translated-grid candidates are
not regenerated. This is intentional: it avoids a second raw-data download
and still provides a leakage-free train complement. Finalization fails clearly
if the remaining unique candidates cannot satisfy the requested sample and
difficulty quotas.

Use a new work/output root for the fixed release, but point it at the existing
bootstrap staging root. Final output images are hard-linked where possible, so
they do not duplicate PNG payload bytes on the same NTFS volume. Existing
packaged datasets are fingerprinted and will not be silently reused under a
new manifest.

To derive only a 550k context512/ROI256 release from that same bootstrap
staging, run:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_context512_roi256_550k_from_staging_windows.py --resume
```

This entry has no OBS arguments by design. It performs fixed-manifest
repartitioning, context-only finalization, two-image validation, and tar
packaging without downloading or restaging the seven sources.

## Comparability Boundary

The manifest fixes source geography and prevents leakage. Strict metric
comparison additionally requires the same eval patch geometry, stride,
semantic schema, prompt/output schema, and metric implementation. Compare
local256 models on one frozen local256 benchmark and context512_roi256 models
on one frozen context benchmark; sharing source maps alone does not make those
two different input protocols numerically identical.
