# Dataset Patch Processing Notes

## Purpose

This document records how we convert AV2 per-log 4096x4096 BEV images into 256x256 patch samples for the current UniMapGen-style reproduction workflow.

Keep future dataset-processing decisions and bug notes here.

## Input Dataset

Current source dataset:

```text
/media/q/data2/jjh/work/av2_tbv_perlog_fixedscale_rc_full_dropPed_dropIntRoad_dropIntConnector_20260408
```

Important files and folders:

```text
manifest.jsonl          # per-log index
rc_input/               # 4096x4096 RGB BEV input images
centerline_json/        # full-image centerline annotations
structure_json/         # lane divider / road edge structures, not used for current centerline SFT
tile_meta/              # per-tile metadata
```

Each `manifest.jsonl` row points to one 4096 image and its full-image centerline JSON.

## Output Format

The patch script writes:

```text
images/                 # cropped 256x256 patch images
meta.jsonl              # patch metadata + target lines + incoming traces
sft.jsonl               # training records in image + conversations format
summary.json            # processing summary
```

`sft.jsonl` can be used directly by the current LLaVA-style training pipeline.

Each patch record uses patch-local coordinates:

```text
x, y in [0, 255]
```

The model should never be asked to predict full-image coordinates.

## Main Script

```text
scripts/data/split_av2_perlog_to_patches.py
```

Basic command for one full 4096 image:

```bash
python scripts/data/split_av2_perlog_to_patches.py \
  --input-root /media/q/data2/jjh/work/av2_tbv_perlog_fixedscale_rc_full_dropPed_dropIntRoad_dropIntConnector_20260408 \
  --output-root data/av2_patch_256_fullimage_cutflag_test_v2 \
  --limit-logs 1 \
  --keep-empty \
  --with-gt-incoming
```

Small sample command:

```bash
python scripts/data/split_av2_perlog_to_patches.py \
  --input-root /media/q/data2/jjh/work/av2_tbv_perlog_fixedscale_rc_full_dropPed_dropIntRoad_dropIntConnector_20260408 \
  --output-root data/av2_patch_256_sample20_phase_b \
  --max-patches 20 \
  --with-gt-incoming
```

Phase A command, no incoming traces:

```bash
python scripts/data/split_av2_perlog_to_patches.py \
  --input-root /media/q/data2/jjh/work/av2_tbv_perlog_fixedscale_rc_full_dropPed_dropIntRoad_dropIntConnector_20260408 \
  --output-root data/av2_patch_256_sample20_phase_a \
  --max-patches 20
```

## Patch Ordering

Patch order must preserve each 4096 image as a contiguous group.

Within each image, patches are ordered row-major:

```text
(0,0), (0,1), ..., (0,15)
(1,0), (1,1), ..., (1,15)
...
(15,0), ..., (15,15)
```

State-update inference also sorts by:

```text
tile_id -> row -> col
```

This prevents patches from different 4096 images from mixing state.

## centerline cut/inside Rule

`cut/inside` is based on endpoint source, not just endpoint location.

Correct rule:

```text
cut    = endpoint was newly created by clipping the full-image line against a 256 patch boundary
inside = endpoint comes from the original 4096 full-image annotation
```

This matters because an original full-image line endpoint may naturally lie on a 256 patch boundary. That endpoint must stay `inside`; it is not a continuation point.

Bad old rule:

```text
endpoint near x/y 0 or 255 -> cut
```

This is wrong because it mislabels natural endpoints that happen to be on patch boundaries.

## Incoming Trace Rule

State update uses only already processed neighbors:

```text
left neighbor: (row, col - 1)
top neighbor:  (row - 1, col)
```

Each `incoming_trace` comes from one adjacent `centerline`.

Rules:

```text
1. Only use adjacent centerline endpoints whose start_type/end_type is cut.
2. Prefer 3 ordered points near the shared boundary.
3. If 3 points are not available, keep 2 points.
4. Use these ordered points as a direction proxy instead of the paper's explicit direction field.
5. Do not generate traces from inside endpoints, even if the endpoint lies on a patch boundary.
```

Example from a left neighbor:

```json
{
  "id": "L0",
  "side": "left",
  "points": [[-26, 211], [-14, 218], [-1, 226]]
}
```

Coordinates may be negative because they are expressed in the current patch coordinate frame and lie just outside the current patch.

## Current Test Output

Current one-image test dataset:

```text
data/av2_patch_256_fullimage_cutflag_test_v2
```

Verified:

```text
num_patches = 256
patch order = row-major
incoming traces = generated only from cut endpoints
state-update dry-run = passed over all 256 patches
```

Useful dry-run command:

```bash
source /home/q/anaconda3/etc/profile.d/conda.sh
conda activate fastvlm

python scripts/infer_centerline_state_update.py \
  --checkpoint-dir /tmp/nonexistent \
  --patch-json data/av2_patch_256_fullimage_cutflag_test_v2/sft.jsonl \
  --image-folder data/av2_patch_256_fullimage_cutflag_test_v2 \
  --output-json /tmp/fullimage_cutflag_state_update_dry_run.json \
  --output-dir /tmp/fullimage_cutflag_state_update_dry_run_patches \
  --dry-run-prompts
```

## Known Open Items

- Current AV2 patch script uses `centerline_json` only.
- `intersection` schema is supported by the model/data format, but this AV2 source slice currently does not provide closed intersection polygons.
- Full training data generation should be done after validating the one-image and small-sample outputs.
- If later adding overlap stride smaller than 256, revisit patch ordering and cut/inside semantics.
