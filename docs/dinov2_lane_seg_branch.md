# DINOv2 Lane Segmentation Branch

## Scope

This branch is intentionally narrower than the full UniMapGen paper pipeline:

- input: satellite image
- backbone: frozen pretrained DINOv2
- head: trainable segmentation decoder
- output: lane mask

It does not use:

- map serialization
- autoregressive decoding
- state update

The purpose is to validate the satellite-feature branch first and keep the engineering path clean before restoring the larger paper pipeline.

## Implementation

Key files:

- `/mnt/data/project/jn/UniMapGen/unimapgen/data/lane_seg_dataset.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/models/dino_lane_seg.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/train_lane_seg.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/eval_lane_seg.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/predict_lane_seg.py`

The backbone path is resolved from the local HuggingFace cache-style directory under:

- `/mnt/data/project/jn/UniMapGen/ckpts/dinov2_vitl14`

## Smoke config

- config: `/mnt/data/project/jn/UniMapGen/configs/dinov2_lane_seg_unaligned_smoke.yaml`
- script: `/mnt/data/project/jn/UniMapGen/scripts/run_dinov2_lane_seg_unaligned_smoke.sh`

Run:

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/run_dinov2_lane_seg_unaligned_smoke.sh
```

## Data

Current smoke test uses the small unaligned sample dataset:

- `/mnt/data/project/jn/UniMapGen/data_samples/unaligned_sat_examples`

Only `lane_line` is treated as positive foreground in this branch.

## Next step

After aligned satellite data is ready, this branch can be reused directly by replacing:

- `data.dataset_root`
- `data.annotation_json`

Then the full UniMapGen reproduction can continue by reintroducing serialized map decoding and state update as separate stages.
