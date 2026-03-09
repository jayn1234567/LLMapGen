# Paper Patch Augmentation Route (2026-03-07)

## Goal

Add a paper-aligned OpenSatMap patch expansion stage on top of the current
aligned AV2/OpenSatMap 896 crop pipeline, without blocking the current
state-update training line.

## Current status

Implemented:

1. Dataset builder:
   - `scripts/build_av2_opensatmap_paper_augmented_dataset.py`
   - Output format matches current UniMapGen training pipeline:
     - `train/`
     - `val/`
     - `annotations.json`
     - `splits_meta.json`
     - `patch_geometry.json`
     - `manifest.json`
     - `summary.json`

2. Supported augmentation types:
   - `rotation`
   - `overlap`
   - `inclined`
   - `overlap + rotation`
   - `inclined + rotation`

3. Stage-1 style configs/scripts:
   - `configs/qwen_dinov2_map_serialization_av2_paper_stage1_aug.yaml`
   - `configs/qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.yaml`
   - corresponding `run/eval` scripts

## Important constraints

1. `rotation` can run from existing cropped `896x896` patches alone.

2. `overlap` and `inclined crop` require the raw OpenSatMap root:
   - `picuse20trainvaltest/`
   - `GPS_info_all.json`
   - `annotrainval20.json`

3. Current augmented dataset is intended for `stage1/2`-style no-state training.
   Do not directly replace the current state-scan dataset with it.

4. Validation is kept unaugmented by default.

## Default augmentation schedule

Current engineering defaults in the builder are:

- overlap offsets:
  - `(+448, 0)`
  - `(-448, 0)`
  - `(0, +448)`
  - `(0, -448)`
  - four diagonal offsets with the same magnitude
- inclined crop angles:
  - `-15`
  - `+15`
- in-patch rotation:
  - `90`
  - `180`
  - `270`

These defaults are configurable and should be treated as a paper-aligned
starting point, not yet a verified exact reproduction of the authors'
final augmentation schedule.

## Suggested workflow

1. While the current state training is running:
   - keep using `av2_opensatmap_partial_fix` for the state line
   - do not mix augmented patches into that experiment

2. After the current run finishes:
   - build the augmented dataset
   - run `paper_stage1_aug_quick`
   - verify `official_metrics_val.json`

3. After the full crop job finishes:
   - rebuild the augmented dataset from the final complete crop root
   - rerun stage-1 full training
   - only then decide whether to increase image size beyond `224`

## Commands

Rotation-only validation:

```bash
cd /mnt/data/project/jn/UniMapGen
python scripts/build_av2_opensatmap_paper_augmented_dataset.py \
  --crop-root /mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix \
  --output-root /mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_paper_aug_partial \
  --disable-overlap \
  --disable-inclined
```

Full builder once raw OpenSatMap is available:

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/build_av2_opensatmap_paper_augmented_dataset.sh
```

Current local default raw root:

```bash
/mnt/data/data1/OpenSateMap
```

Quick stage-1 training:

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/run_qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.sh
```
