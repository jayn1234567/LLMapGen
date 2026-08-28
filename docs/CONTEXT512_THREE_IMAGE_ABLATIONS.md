# Context512 Three-Image Ablations

`scripts/tools/build_context512_roi256_three_image_ablation_windows.py` works
from an already completed Context512/ROI256 Dataset V2 containing exactly three
images per record:

1. clean BEV road-structure image;
2. separate Raw-Lane image;
3. separate Pose image.

It does not download or extract source archives.

## Neighbor Rotation Replacement

Use `--mode neighbor_rotation` for the stride-256 experiment. The output train
grid is always 256 pixels. By default, rows from a finer source grid (for
example the historical 128-pixel 800k release) whose origin is not aligned to
256 are filtered out; pass `--neighbor-source-grid-policy require` to reject
such an input instead. Each selected train row is replaced in place; no extra
rows are added. The angle is selected from `0,45,135` using:

```text
phase = (grid_x + grid_y) % 3
grid_x = x0 / 256
grid_y = y0 / 256
```

This gives different phases to immediate horizontal and vertical neighbors.
With only three phases, one diagonal direction can share a phase; this is a
compact schedule rather than a claim of eight-neighbor graph coloring. The
BEV, Raw-Lane, and Pose images rotate together. The target
geometry is mapped from ROI-relative norm1000 to context pixels, rotated around
the 512x512 center, clipped to the fixed center 256x256 ROI, and mapped back to
ROI-relative norm1000. Evaluation and test rows are not rotated.

When the input train grid is already stride 256, the output keeps the original
train sample count and IDs. If the input is on a finer grid, the default filter
keeps only origins aligned to 256 and preserves the IDs of that subset. Non-zero
angle rows are local augmentations and have invalid global coordinates; their
source coordinates are retained in `source_global_metadata` for provenance only.
This changes the visual orientation seen by the model, but it does not remove
the physical 50% context-window overlap caused by a 256-pixel stride. Use
`--mode nonoverlap` separately when the goal is a strict 512-pixel grid.

The three image paths remain separate in the output. The clean BEV image uses
the requested interpolation; Raw-Lane and Pose use nearest-neighbor
interpolation so their sparse white pixels are not blurred into geometric
evidence. No Raw-Lane or Pose pixels are composited into the BEV image.

Example:

```powershell
python scripts\tools\build_context512_roi256_three_image_ablation_windows.py --mode neighbor_rotation --input-root "D:\data\rc_dataset_v2_three_image_800k_from_obs\output_three_image_800k\context512_roi256_rawlane_pose_800k" --output-root "D:\data\context512_roi256_three_image_neighbor_rotation_256" --neighbor-angles "0,45,135" --neighbor-grid-stride 256 --neighbor-source-grid-policy filter --copy-mode hardlink --image-resample bilinear --package-path "D:\data\context512_roi256_three_image_neighbor_rotation_256.tar" --progress-every 10000
```

If the completed source was stored under a different name, change only
`--input-root`; it must be the directory containing `dataset_info.json`,
`phase_a/`, `images/`, `raw_lane_images/`, and `pose_images/`.

The builder writes `dataset_info.json`, `balance_report.json`,
`ablation_validation.json`, `build_summary.json`, `build_complete.json`, and a
tar package outside the output directory.

The builder validates every source row before writing output. In the default
`filter` policy it records the number of discarded non-256-aligned train rows
in `balance_report.json`; in `require` policy it fails before writing output
rather than assigning an arbitrary neighbor phase.

## Other Modes

- `nonoverlap` keeps only the strict 512-grid rows. This removes physical
  overlap but reduces the train candidate count.
- `rotation` retains base rows and adds rotated copies. It is an additive
  augmentation and is intentionally different from `neighbor_rotation`.
