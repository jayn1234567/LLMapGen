# RC Handoff 2026-04-08

Last refreshed: `2026-04-15` (Beijing time)

## 0. Read This First

This document is the current RC handoff baseline.

- Prefer this file over:
  - `docs/rc_handoff_20260405.md`
  - `docs/rc_query_resampler_handoff_20260406.md`
- The older docs are still useful for historical design context, but the execution state, dataset root, and current mainline experiment have all moved forward.
- When any RC dataset rule, supervision target, training launcher, or remote build state changes, update this handoff in the same work turn.
- Current remote login path for the `file_storage01 / mingli` RC work line is:
  - host: `172.29.15.117`
  - user: `mingli`
  - auth: password SSH
  - do **not** confuse this with the older `lenovo@zidao-ai.com:6006` route, which belongs to a different historical workspace lineage
- The old clean end-to-end comparison line is still the `16745-style` single-coordinate-head route on the new AV2 per-log train root.
- However, as of `2026-04-11`, the next implementation priority has shifted again:
  - keep the AV2 per-log RC train root as the canonical data root
  - keep the pure visual structure-seg benchmark results as the visual-quality reference
  - prepare a new two-stage alignment route:
    - `Stage 1`: `DINOv2 -> geometric-pos -> projector -> token-alignment MLP -> light readout -> frozen Qwen text tower` with CLIP-style coarse semantic alignment
    - `Stage 2`: `DINOv2 -> geometric-pos -> projector -> token-alignment MLP -> visual token injection into Qwen` with caption-token fine alignment
  - do **not** start from the older query-resampler end-to-end centerline route when resuming new architecture work

## 1. Current Mainline in One Paragraph

The RC project now has two layers that must be kept separate:

1. the **historical end-to-end baseline**
2. the **next architecture work line**

The stable historical baseline is still:

`AV2 map-only -> per-log fixed-scale 4096 RC -> offset-grid 512 patches -> 24 px centerline resampling -> RC-only Qwen training`

with the clean end-to-end reference model:

`RC patch (512x512) -> ResNet50+FPN -> road neck -> learned query resampler (576) -> visual projector -> Qwen3-4B -> structure tokens + single continuous coord head`

and clean run:

- `16982`

But the next architecture work should **not** continue from that route directly.

The next intended implementation line is now a two-stage route:

`Stage 1`

`RC patch (512x512, center-padded to 518) -> RC-trained DINOv2 -> all patch tokens -> explicit (cx, cy) geometric position encoding -> projector -> token-alignment MLP -> light readout -> CLIP-style coarse semantic alignment against frozen Qwen text embeddings`

`Stage 2`

`RC patch (512x512, center-padded to 518) -> RC-trained DINOv2 -> all patch tokens -> explicit (cx, cy) geometric position encoding -> visual_norm -> projector -> token-alignment MLP -> direct replacement of <vis_patch> embedding slots inside frozen Qwen3-4B -> caption-token fine alignment`

Current design rules:

- `Stage 1` uses natural-language scene+side descriptions, not `Scene=...` schema text
- `Stage 1` uses `top`, `bottom`, `left`, `right` as the only direction words
- `Stage 1` keeps Qwen as a text-only tower; Qwen does **not** receive visual tokens yet
- `Stage 2` is the first stage where visual tokens are actually injected into Qwen
- keep all DINO patch tokens; do **not** add a query resampler in the first implementation
- current accepted `Stage 2` rule is:
  - reuse the `Stage 1` bridge
  - do **not** add a new visual injection module
  - keep the `Qwen` backbone frozen in the first `Stage 2` run
  - but allow only the 2 visual boundary-token rows (`<vis_start>`, `<vis_end>`) to train

## 2. Dataset Pipeline That Is Now Considered Correct

### 2.1 Data Source

- Primary source: `AV2 TBV map-only`
- Do not treat OpenSatMap-aligned tiles as the main scaling route anymore
- The current train root is built from `per-log fixed-scale 4096 RC`

### 2.2 Rendering Rules

The current accepted AV2 RC rendering rule set is:

- drop `ped crossings`
- drop `is_intersection=true road_edges`
- drop `intersection connectors`
- keep the remaining road-structure input geometry
- keep real lane centerlines as centerline target

This is the route behind the current new train root.

### 2.3 Build Chain

Key scripts:

- `scripts/build_av2_tbv_perlog_fixedscale_rc_full.py`
- `scripts/build_rc_perlog_offset_patch_dataset.py`
- `scripts/export_llamafactory_rc_centerline_from_offset_manifest.py`

Key sbatch:

- `build_av2_tbv_perlog_offset_trainroot_full_dropintconnector_20260408.sbatch`

That sbatch runs the three-stage chain:

1. build full per-log `4096 x 4096` RC
2. crop `512 x 512` patches with offset grid
3. export the Qwen/LLaMAFactory train root

### 2.4 Current Canonical Train Root

Remote train root:

- `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`

Current summary:

- train rows: `108,764`
- val rows: `5,645`
- train tiles: `990`
- val tiles: `52`
- total accepted rows: `114,409`
- total accepted tiles: `1,042`
- source mode: `av2_perlog_fixedscale`
- resample step: `24 px`
- this root now also contains precomputed thin multiclass structure masks under:
  - `patches512_offset/seg_structure_multiclass/...`
- `meta_train.jsonl` and `meta_val.jsonl` now carry:
  - `seg_structure_multiclass`

Prompt text inside `dataset_info.json`:

```text
This is a black-background BEV road-structure image.
Predict the road centerlines for this patch from the visible lane_boundary and lane_divider structure.
Return only the raw JSON object.
```

System prompt inside `dataset_info.json`:

```text
You are an expert road-centerline reconstruction assistant for black-background BEV road-structure images.

VISIBLE SEMANTICS:
The visible road-structure classes are lane_boundary, lane_divider, and background.
The image does not show centerlines directly.

TASK DEFINITION:
Your task is to infer the unseen road centerlines strictly from the visible road structure.
1. A centerline is the geometric middle path of one valid drivable corridor.
2. Do not trace lane_boundary or lane_divider themselves.
3. Keep different lanes, branches, and intersecting paths as separate continuous polylines.
4. If a centerline reaches the patch border, terminate it at the visible border.
5. Predict all valid centerlines implied by the visible road structure in the current patch only.

OUTPUT CONSTRAINTS:
1. Return ONLY valid JSON.
2. Do NOT wrap the JSON in markdown fences.
3. Do NOT output explanations or extra text.
4. Use the patch-local coordinate system.
5. All x and y coordinates must be integers between 0 and 512 inclusive.
6. Strictly use this JSON structure:
{"lines":[]}
or
{"lines":[{"points":[[x1,y1],[x2,y2]]}]}
```

### 2.5 Useful Local Sample Artifacts

20 sampled RC patches from the new train root:

- `artifacts/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408_samples20`

The same 20 patches with centerline GT overlaid:

- `artifacts/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408_samples20_overlay_centerline_gt_20260408`

These are the fastest sanity-check assets for a new agent.

### 2.6 `caption_short` Full Root, Scene Review Root, and Current Stage-1 Clean Root

The `caption_short` line is now split into three layers that should not be mixed up:

- the full export root used as the semantic source pool
- the split manual-review root used to collect reviewed scene labels
- the autocorrected clean semantic root actually used for `Stage 1` training

Current full `caption_short` source root:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411`

Current split manual-review root:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411`

Current reviewed-label subset already used by the autocorrector:

- `part_01`
- reviewed labels: `500`
- reviewed label counts:
  - `straight = 390`
  - `intersection-approach = 67`
  - `branching = 20`
  - `complex = 21`
  - `curved = 2`

Current clean semantic root for `Stage 1`:

- `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412`

Current clean-root summary:

- source full root size:
  - train rows seen: `96,006`
  - val rows seen: `5,033`
- autocorrector reviewed OOF accuracy: `0.846`
- accepted threshold:
  - `accept_prob = 0.98`
  - `accept_margin = 0.05`
  - accepted OOF coverage: `0.66`
  - accepted OOF accuracy: `0.9545`
- clean accepted rows used for training:
  - train: `75,948`
  - val: `3,973`
- clean accepted train scene counts:
  - `straight = 67,755`
  - `intersection-approach = 5,288`
  - `branching = 1,251`
  - `complex = 1,079`
  - `curved = 575`

Conflict / low-confidence review pool exported alongside the clean root:

- train unresolved rows:
  - `needs_review_conflict = 17,434`
  - `needs_review_low_confidence = 2,624`
- val unresolved rows:
  - `needs_review_conflict = 942`
  - `needs_review_low_confidence = 118`
- review pool path:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412/review_conflicts`

Important meaning:

- `Stage 1` should now train from the clean semantic root above, not from the old `435`-row reviewed subset
- the full `caption_short` root remains the semantic-source pool and media root
- the split review root remains the place to accumulate additional reviewed scene labels
- unresolved conflict rows should be manually reviewed only from the exported conflict pool, not by re-reading the whole `96,006`
- `Stage 2` will later reuse the same visual bridge for caption-token fine alignment

### 2.7 Precomputed Thin `structure_multiclass` Masks

This correction was added on `2026-04-10` after we found that the earlier
`structure_multiclass` visual benchmark was still too thick.

Root cause:

- the earlier `structure_multiclass` training path did **not** use precomputed thin masks
- it reconstructed labels directly from the rendered RC RGB colors
- this made the targets visually thicker than the old binary `seg_binary` route
- therefore it did **not** match the user requirement of "as thin as the first binary structure benchmark"
Corrected implementation:

- new script:
  - `scripts/precompute_rc_structure_multiclass_masks.py`
- dataset loader update:
  - `unimapgen/data/rc_structure_seg_dataset.py`
- behavior:
  - for `structure_multiclass`, the loader now prefers a precomputed mask path from meta:
    - `seg_structure_multiclass`
  - instead of reconstructing labels from RC RGB at load time

Current accepted thin multiclass label set:

- `background`
- `lane_divider`
- `road_edge`

Important note:

- `ped_edge` is **not** part of this current accepted thin multiclass route
- the canonical AV2 per-log train root already drops ped crossings in the current rendering rule set

Full remote precompute status:

- completed on the canonical root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
- verified counts:
  - train: `108,764 / 108,764` rows now have valid `seg_structure_multiclass`
  - val: `5,645 / 5,645` rows now have valid `seg_structure_multiclass`

Practical meaning:

- any future thin `structure_multiclass` benchmark should use this precomputed-mask route
- do **not** go back to the older RGB-derived `structure_multiclass` target generation when comparing visual quality

## 3. Model Lines and Which One Is the Main One

### 3.1 Current Clean Mainline: `16745-style`

Code:

- model: `unimapgen/models/qwen3_rc_centerline_16745style.py`
- trainer: `scripts/train_qwen3_rc_centerline_cnn_prefix_16745style.py`
- launcher: `stagea_rc_centerline_qwen3_queryresampler_segmeta_16745style_4gpu_newtrainroot_20260408.sbatch`

Important properties:

- `Qwen3-4B`
- `ResNet50+FPN` with local pretrained weights
- freeze policy: `stem + layer1 frozen`, `layer2-4 + FPN trainable`
- `576` learned query tokens
- `24 x 24` visual grid
- `cutoff_len = 8192`
- `LoRA + full embedding tuning`
- single continuous `coord_head`
- dense binary seg aux
- dense centerline heatmap aux

Most important clarification:

- this line keeps the query resampler
- but it does **not** use query tokens to predict the centerline heatmap
- and it does **not** use multi-layer `p4/p3/p2` coordinate heads

This line is the clean comparator right now.

### 3.2 Experimental Side Line: Multicoord + Query Heatmap

Related launcher:

- `stagea_rc_centerline_qwen3_queryresampler_multicoord_queryheatmap_4gpu_q576_newtrainroot_20260408.sbatch`

Current run:

- `16949 | jnrcq5764`

This route is still exploratory.

Do not mix its results with the clean `16745-style` line when making short-term decisions.

### 3.3 Legacy `16745` Reference

Old launcher:

- `stagea_rc_centerline_qwen3_queryresampler_segmeta_4gpu_q576.sbatch`

Old log:

- `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stagea_rc_centerline_qwen3_queryresampler_segmeta_4gpu_q576_16745.out`

Important checkpoint note:

- the exact `epoch 3` checkpoint of `16745` is no longer on disk
- because of checkpoint retention, the nearest retained checkpoint used for later eval is:
  - `/file_storage01/home/mingli/data/outputs/stagea_rc_centerline_qwen3_queryresampler_segmeta_q576_cut8192_4gpu_20260407/checkpoint-59500`

## 4. Completed Experimental Results

### 4.1 Quick Eval on New Val Root

Completed eval job:

- `16965 | jnrce745s`

Setting:

- checkpoint: old `16745` retained `checkpoint-59500`
- dataset: new val root
- only `400` samples
- only samples with `num_lines < 10`
- render `100` visualizations

Output dir:

- `/file_storage01/home/mingli/data/outputs/eval_qwen3_rc_centerline_16745style_ckpt59500_newval_max400_lt10_viz100_4gpu_20260408`

Main metrics:

- `APC@2px = 0.0025`
- `APC@4px = 0.0361`
- `APC@8px = 0.2604`
- `mean_chamfer_px = 26.19`
- `continuity_pred = 0.1174`
- `continuity_gt = 0.2661`
- `continuity_gap = 0.2206`
- `pred_num_lines = 3.6425`
- `gt_num_lines = 3.655`

Interpretation:

- line count is already roughly matched
- the main weakness is still geometry quality and continuity
- the bottleneck is not primarily "how many lines" but "where the lines are and how well they stay connected"
Local pulled visualizations:

- `artifacts/eval_qwen3_rc_centerline_16745style_ckpt59500_newval_max400_lt10_viz100_4gpu_20260408/viz`

### 4.2 Early Training Comparison: New Root vs Old `16745`

Current clean run:

- `16982 | jnrcq576sc`

This run was explicitly cleaned to remove fake multilevel coord-loss logging.

Meaning:

- there is now only one `coord_pred`
- no `coord_pred_l4/l3/l2`
- no fake `coord_point_loss_l4/l3/l2`

Early same-period comparison against old `16745` shows:

- the new root is **slightly better**, not dramatically better
- improvement is mostly in:
  - `base_ce_loss`
  - `coord_point_loss`
  - `coord_reg_mae`
- `seg_loss` is not clearly better

Reference numbers:

First 50 logged steps average:

| metric | old `16745` | new `16982` |
|---|---:|---:|
| loss | 1.9430 | 1.9356 |
| base_ce_loss | 0.3874 | 0.3600 |
| coord_point_loss | 1.4007 | 1.3581 |
| coord_reg_mae | 0.2851 | 0.2766 |
| seg_loss | 0.0338 | 0.0407 |
| centerline_heatmap_loss | 0.0709 | 0.0769 |

Synchronized by current new-run epoch (`~0.03194`):

| metric | old `16745` synced | new `16982` |
|---|---:|---:|
| loss | 1.9065 | 1.8276 |
| base_ce_loss | 0.3742 | 0.3230 |
| coord_point_loss | 1.3768 | 1.2952 |
| coord_reg_mae | 0.2803 | 0.2640 |
| seg_loss | 0.0328 | 0.0372 |
| centerline_heatmap_loss | 0.0693 | 0.0720 |

Last-10-logs average in the same early phase:

| metric | old synced last 10 | new last 10 |
|---|---:|---:|
| loss | 1.3537 | 1.2975 |
| base_ce_loss | 0.1501 | 0.1315 |
| coord_point_loss | 1.0513 | 1.0445 |
| coord_reg_mae | 0.2152 | 0.2139 |
| coord_dir_loss | 0.0804 | 0.0699 |
| seg_loss | 0.0171 | 0.0197 |
| centerline_heatmap_loss | 0.0605 | 0.0518 |

Interpretation:

- the new root does seem to help
- but the gain is modest so far
- the strongest positive signal is on the main geometric terms, not on the dense seg term

### 4.3 Pure Visual Structure-Seg Benchmark Results

This benchmark is now finished far enough to guide the next architecture decision.

#### ResNet50+FPN binary structure benchmark

Reference job:

- `17110`

Best remembered validation point:

- `val_loss ~= 0.1923`
- `val_iou ~= 0.7389`
- `val_dice ~= 0.8471`

Later stable point:

- `val_loss ~= 0.2036`
- `val_iou ~= 0.7324`
- `val_dice ~= 0.8404`

Interpretation:

- strong and stable
- currently the best practical dense RC structure baseline
- enough to confirm that the RC train root and structure-seg task are both valid

#### DINOv2 binary structure benchmark

Reference job:

- `17111`

Best remembered validation point:

- `val_loss ~= 0.2083`
- `val_iou ~= 0.7199`
- `val_dice ~= 0.8346`

Observed later point:

- `val_loss ~= 0.2218`
- `val_iou ~= 0.7056`
- `val_dice ~= 0.8234`

Interpretation:

- DINOv2 does learn RC structure
- but it is weaker and slower than the ResNet50+FPN benchmark under the current segmentation setup
- still worth using as the next **token-native** encoder candidate for feature-to-Qwen alignment

#### Dash-vs-solid benchmark caution

Reference job:

- `17249`

Important caution:

- the later `dash_solid` segmentation task should **not** be compared numerically against the earlier binary structure benchmark as if they were the same task
- label generation and metric definition changed
- the `dash_solid` task is also more directly tied to the rendered RC colors

Practical reading rule:

- use `17110` vs `17111` to compare encoder families under the clean binary protocol
- do not use `17249` to claim that the encoder itself suddenly became much stronger

#### DINOv2 dash-vs-solid attempt on `2026-04-11`

This short attempt should now be treated as superseded.

Launcher:

- `stagea_rc_structure_seg_dinov2_dashsolid_pad518_4gpu_20260411.sbatch`

Remote output dir:

- `/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_dashsolid_pad518_4gpu_20260411`

Submission history:

- `17744`
  - accepted by Slurm first
  - then explicitly cancelled after the user requested the true thin-mask counterpart instead

Important caveat:

- this `dash_solid` supervision path still followed the color-derived label route
- it is not the same as the newer precomputed thin `structure_multiclass` route
- do not use `17744` as the canonical DINOv2 comparison for the corrected thin multiclass labels

#### DINOv2 thin `structure_multiclass` relaunch on `2026-04-11`

This is the real DINO counterpart to the corrected thin-mask ResNet benchmark.

Launcher:

- `stagea_rc_structure_seg_dinov2_structure_multiclass_thinmask_pad518_4gpu_20260411.sbatch`

Remote output dir:

- `/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411`

Submission:

- `17746`

Verified startup:

- `17744` was cancelled
- `17746` entered `RUNNING`
- the job log confirms:
  - canonical data root:
    - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
  - supervision:
    - `structure_multiclass`
  - DINO local weights:
    - `/file_storage01/home/mingli/data/ckpts/dinov2-large`
  - input padding:
    - `512 -> 518`
- current early state:
  - Slurm runtime check showed `17746 RUNNING gpu28`
  - stderr only had the usual Transformers cache deprecation warning

Main settings:

- encoder:
  - `DINOv2 ViT-L/14`
  - input padded from `512` to `518`
  - unfreeze last `12` blocks
- supervision:
  - `structure_multiclass`
  - class weights `0.2,1.0,1.0`
- optimization:
  - batch size `4`
  - head lr `1e-4`
  - backbone lr `1e-5`

Latest snapshot visualization on `2026-04-11`:

- render launcher:
  - `render_rc_structure_seg_predictions_dinov2_structure_multiclass_thinmask_latest_val30_1gpu_20260411.sbatch`
- render job:
  - `17816`
- snapshot checkpoint used:
  - `/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/viz_latest_snapshot_20260411.pt`
- remote render dir:
  - `/file_storage01/home/mingli/data/outputs/render_rc_structure_seg_dinov2_structure_multiclass3_thinmask_snapshot_val30_20260411`
- local pulled dir:
  - `artifacts/render_rc_structure_seg_dinov2_structure_multiclass3_thinmask_snapshot_val30_20260411`
- render subset summary on `30` val samples:
  - `mIoU ~= 0.7125`
  - `foreground_iou ~= 0.7258`
  - `dice ~= 0.8192`
  - `lane_divider_iou ~= 0.7458`
  - `road_edge_iou ~= 0.6793`

#### Thin `structure_multiclass` correction after `17249`

Later visual inspection showed that both:

- `dash_solid`
- and the first RGB-derived `structure_multiclass` attempt

made the line targets appear thicker than the original binary structure baseline.

Therefore:

- the RGB-derived `structure_multiclass` route should be treated as a dead-end diagnostic path
- the corrected thin multiclass route is now:
  - precompute label PNGs from `structure_json`
  - save them as `seg_structure_multiclass`
  - let the dataset loader read those masks directly

Important consequence:

- any future `lane_divider + road_edge` multiclass benchmark must be relaunched on the new precomputed thin masks
- the already rendered `17490` visualizations are useful only as a cautionary reference, not as the final thin-label benchmark

#### Thin `structure_multiclass` ResNet relaunch on `2026-04-10`

This relaunch is now the correct ResNet50 visual benchmark for thin multiclass structure labels.

Launcher:

- `stagea_rc_structure_seg_resnet50_structure_multiclass_thinmask_4gpu_20260410.sbatch`

Remote output dir:

- `/file_storage01/home/mingli/data/outputs/rc_structure_seg_resnet50_structure_multiclass3_thinmask_4gpu_20260410`

Submission history:

- `17599`
  - first submit
  - failed immediately because the job imported `torch` from `~/.local/lib/python3.10/site-packages`
  - error signature:
    - `undefined symbol: ncclCommWindowDeregister`
- `17600`
  - corrected relaunch
  - launcher now forces:
    - `PYTHONNOUSERSITE=1`
    - `unset PYTHONUSERBASE`
    - `unset PYTHONHOME`
    - `python -s`

Meaning:

- if later RC training jobs hit a similar `torch` / `nccl` symbol mismatch on the cluster, first suspect user-site package pollution
- the corrected thin multiclass ResNet benchmark should be referenced from job `17600`, not `17599`

Verified early startup on `17600`:

- the job successfully entered `Epoch 1/20`
- first printed train indicator (around step `20`) was:
  - `loss ~= 1.7310`
  - `iou ~= 0.016`
  - `dice ~= 0.031`
  - `lane_divider_iou ~= 0.008`
  - `road_edge_iou ~= 0.024`
- early warmup then improved quickly:
  - around step `60`: `loss ~= 0.9849`, `iou ~= 0.341`
  - around step `500`: `loss ~= 0.4172`, `iou ~= 0.529`

First full epoch summary from `metrics.jsonl`:

- epoch `1`
- train:
  - `loss ~= 0.3296`
  - `iou ~= 0.6703`
  - `dice ~= 0.7922`
  - `lane_divider_iou ~= 0.6925`
  - `road_edge_iou ~= 0.6481`
- val:
  - `loss ~= 0.2542`
  - `iou ~= 0.7415`
  - `dice ~= 0.8465`
  - `lane_divider_iou ~= 0.7667`
  - `road_edge_iou ~= 0.7163`

Epoch-1 visualization render:

- render job:
  - `17616`
- frozen checkpoint used:
  - `/file_storage01/home/mingli/data/outputs/rc_structure_seg_resnet50_structure_multiclass3_thinmask_4gpu_20260410/epoch1_latest.pt`
- remote render dir:
  - `/file_storage01/home/mingli/data/outputs/render_rc_structure_seg_resnet50_structure_multiclass3_thinmask_epoch1_val30_20260410`
- local pulled dir:
  - `artifacts/render_rc_structure_seg_resnet50_structure_multiclass3_thinmask_epoch1_val30_20260410`
- render subset summary on 30 val samples:
  - `mIoU ~= 0.6944`
  - `foreground_iou ~= 0.7121`
  - `dice ~= 0.8073`
  - `lane_divider_iou ~= 0.7174`
  - `road_edge_iou ~= 0.6715`

## 5. Current Jobs and Recent Validation Runs Worth Knowing

As of the latest refresh on `2026-04-13`:

- `18254 | jnrcjs1`
  - current formal `SFT v1` `8`-GPU run
  - status:
    - `RUNNING`
  - output root:
    - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413`
  - launcher:
    - `stage3_rc_dinov2_centerline_json_sft_lora_8gpu_20260413.sbatch`
  - current verified configuration:
    - `num_train_epochs = 6`
    - uses the final `Stage 2` bridge bundle from:
      - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/rc_dinov2_caption_modules.pt`
    - `cutoff_len = 7168`
    - `LoRA` enabled
    - `freeze_language_model = False`
    - `freeze_vision_encoder = True`
- `18231 | jnrcjs1`
  - previous formal `SFT v1` `8`-GPU run
  - status:
    - `CANCELLED`
  - reason:
    - user requested to switch the formal run from `1 epoch` to `6 epoch`
  - note:
    - it had already entered real training and printed stable non-zero losses before cancellation
- `18230 | rcstg2q2`
  - completed template-fixed `Stage 2` structured quick gate on `100` val samples
  - output root:
    - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/structured_eval_quick100_templatefix_20260413`
  - root-cause conclusion:
    - the earlier empty-output collapse was **not** a bad bridge checkpoint
    - it came from the chat-template handling:
      - the code rejected Qwen3's empty non-thinking `<think> ... </think>` scaffold
      - this made prompt formatting drift away from the training target format
      - generation then fell into free-form `<think>` reasoning, and cleanup stripped most outputs to empty
  - template-fixed metrics:
    - `scene_acc = 0.30`
    - `grid_cell_acc = 0.9633`
    - `macro_f1 = 0.9118`
    - `exact_match = 0.02`
    - `parse_ok_rate = 0.99`
  - follow-up visualization path added on `2026-04-13`:
    - renderer:
      - `scripts/render_stage2_structured_eval_viz.py`
    - rendered output root:
      - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/structured_eval_quick100_templatefix_20260413/viz_input_gt_pred_20260413`
    - panel format:
      - `Input | GT State | Pred State`
    - visualization convention:
      - left panel shows the RC input patch with fixed `8x8` grid
      - middle panel overlays GT `GridStates`
      - right panel overlays predicted `GridStates`
      - red cell borders on the prediction panel mark mismatch cells
    - generated artifacts:
      - `100` per-sample PNG panels
      - `contact_sheet_first16.png`
      - `manifest.json`
  - interpretation:
    - `Stage 2` no longer shows catastrophic empty-output collapse
    - local grid-state alignment quality is strong
    - the main remaining weakness is `scene` classification accuracy, not parse failure
- `18223 | rcjsons1`
  - completed `SFT v1` smoke run for:
    - `DINOv2 + aligned bridge + Qwen LoRA + raw centerline JSON`
  - status:
    - `COMPLETED`
  - output root:
    - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_smoke_1gpu_20260413`
  - verified healthy startup:
    - correct final `Stage 2` bridge bundle is loaded from:
      - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/rc_dinov2_caption_modules.pt`
    - `cutoff_len = 7168`
    - LoRA enabled
    - first stable non-zero logged losses:
      - `0.8088`
      - `0.4998`
      - `0.5719`
      - `0.4458`
  - later stable losses continued around:
    - `0.3630`
    - `0.3918`
    - `0.4214`
    - `0.4192`
  - final smoke summary:
    - `train_loss = 0.4935`
    - `train_runtime ~= 219.6s`
    - the JSON+LoRA `SFT v1` training path is now engineering-validated end-to-end
- `18221 | rcstg2q2`
  - completed final `Stage 2` structured quick gate on `100` val samples
  - output root:
    - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/structured_eval_final_quick100_20260413`
  - metrics:
    - `scene_acc = 0.07`
    - `grid_cell_acc = 0.0755`
    - `macro_f1 = 0.1659`
    - `exact_match = 0.0`
    - `parse_ok_rate = 0.08`
  - key diagnostic:
    - `92 / 100` predictions were empty after decoding / cleanup
    - the small parsed subset can still be locally strong
    - example parsed row achieved:
      - `scene_correct = True`
      - `grid_cell_correct = 59 / 64`
- `18228 | jnrcjs1`
  - first formal `SFT v1` `8`-GPU submission
  - status:
    - `FAILED`
  - failure reason:
    - launcher passed:
      - `--ddp-find-unused-parameters false`
    - but the trainer uses `BooleanOptionalAction`, so the correct flag is:
      - `--no-ddp-find-unused-parameters`
  - action taken:
    - launcher fixed locally and remotely
    - formal run resubmitted as `18231`
- `18080 | jnrcstg2`
  - formal `Stage 2` `1`-epoch `4`-GPU run
  - status:
    - `COMPLETED`
  - uses the save-fixed checkpoint path:
    - `save_safetensors = False`
  - uses the current accepted token-freeze rule:
    - freeze the `Qwen` backbone
    - freeze `DINOv2`
    - do **not** train `<vis_patch>`
    - train only `<vis_start>` and `<vis_end>` token rows
  - output root:
    - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412`
- `18088 | rcstg2tk`
  - `Stage 2` boundary-token strategy smoke verification
  - `COMPLETED`
  - purpose:
    - verify that only `<vis_start>` and `<vis_end>` are selectively trainable
    - verify that `<vis_patch>` remains a pure placeholder that is overwritten by aligned visual embeddings
- `18097 | rcstg2ev`
  - `Stage 2` structured offline-eval smoke
  - `COMPLETED`
  - purpose:
    - verify that the offline structured-eval path now works end-to-end for:
      - `scene_acc`
      - `grid_cell_acc`
      - `macro_f1`
      - `exact_match`
- `18001`
  - completed formal `Stage 1` CLIP-style coarse alignment run
  - output root:
    - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412`
  - this remains the current coarse-alignment initialization source for `Stage 2`
- `16982 | jnrcq576sc`
  - clean `16745-style` end-to-end historical baseline
  - keep as the reference end-to-end comparator, not as the current alignment work line

## 6. What Is Already Settled vs What Is Still Open

### 6.1 Settled

- RC input is the model input, not satellite imagery
- AV2 per-log fixed-scale route is the main dataset route
- patch size is `512 x 512`
- resample step is `24 px`
- current new train root is usable
- `drop intersection connectors` is now part of the accepted data rule
- the clean comparison model should be single-coordinate-head `16745-style`
- the pure visual structure benchmark is already informative enough:
    - `ResNet50+FPN` is the stronger dense-seg benchmark
    - `DINOv2` remains the more natural token-native candidate for the next language-alignment stage
- if a thin multiclass structure benchmark is needed again, the correct supervision source is now settled:
  - use precomputed `seg_structure_multiclass`
  - do **not** derive multiclass labels from RC RGB on the fly
- the current cleaned RC input no longer shows explicit intersection connector geometry
- therefore future language pre-alignment text should **not** talk about:
    - visible centerlines in the input
  - explicit junction boundary lines
- instead, describe what is actually visible:
  - road structure / road branches
  - central gap
  - internal truncation
- the next pre-alignment route should use image-coordinate side words:
  - `top`
  - `bottom`
  - `left`
  - `right`
- do **not** switch to ego-centric `front/rear` wording on the current dataset
- if all DINO patch tokens are forwarded to Qwen in the first pre-align version, the first version does **not** need a query resampler
- the current accepted token-count mode for the alignment route is:
  - center-padded `518 -> 37x37 = 1369` visual tokens
- the current accepted `Stage 2` injection rule is:
  - direct replacement of `<vis_patch>` embedding slots
  - do **not** add a new visual injection module
- the current accepted first `Stage 2` freeze rule is:
  - freeze the `Qwen` backbone
  - freeze `DINOv2`
  - do **not** train `<vis_patch>`
  - train only `<vis_start>` and `<vis_end>` token rows
- the current accepted `Stage 2` checkpoint-save fix is:
  - set `save_safetensors = False`
  - because tied `Qwen` weights between `embed_tokens` and `lm_head` otherwise break safetensors checkpoint save in this training path
- the current accepted Qwen3 chat-template rule is:
  - do **not** reject the empty non-thinking `<think> ... </think>` scaffold emitted by the official Qwen3 chat template
  - keep the same chat-template family for both prompt-only rendering and full target rendering
  - otherwise generation can fall into long `<think>` reasoning text that later gets stripped to empty output
- the current accepted `Stage 2` offline quality check is:
  - run `scripts/eval_qwen3_rc_dinov2_caption_structured.py`
  - record:
    - `scene_acc`
    - `grid_cell_acc`
    - `macro_f1`
    - `exact_match`
  - inspect `predictions.jsonl` manually after metrics

### 6.2 Still Open

- whether the formal `Stage 2` run `18080` reaches a usable quality bar on the held-out `val` split after structured evaluation
- whether the main remaining `Stage 2` failures are mostly:
  - scene confusion
  - background overprediction
  - `lane_boundary` / `lane_divider` / `mix` collapse
- whether the current bridge is heavier than necessary for the first caption-alignment pass
- whether a lighter bridge should become the next ablation before touching the caption schema itself:
  - reduce `token_alignment_hidden_dim`
  - reduce `token_alignment_num_layers`
  - optionally reduce `visual_projector_hidden_dim`
- whether frozen-`Qwen` `Stage 2` is already sufficient, or whether a later follow-up should add limited `LoRA` only after the frozen-backbone line is fully evaluated
- whether the resulting `Stage 2` bridge actually improves later downstream centerline training enough to justify keeping this alignment stage

## 7. Recommended Next Steps

### 7.1 Short-Term, Highest Priority

1. Continue monitoring the current formal `SFT v1` `8`-GPU run `18254` and record:
   - first stable loss range
   - whether checkpoint save stays healthy
   - whether eval/save hooks stay healthy through the first real save interval
2. Treat the template-fixed `Stage 2` gate as **structurally passed** for parsing / local-grid alignment:
   - `parse_ok_rate = 0.99`
   - `grid_cell_acc = 0.9633`
   - `macro_f1 = 0.9118`
3. Treat the main remaining `Stage 2` weakness as `scene` quality, not empty-output collapse:
   - current quick-100 `scene_acc = 0.30`
   - prioritize scene-confusion review before touching the grid schema
4. If later full-val `Stage 2` metrics still show weak `scene` performance, run a lighter bridge ablation before changing the schema:
   - reduce `token_alignment_hidden_dim`
   - reduce `token_alignment_num_layers`
   - optionally reduce `visual_projector_hidden_dim`
5. Only after the frozen-`Qwen` `Stage 2` line is properly evaluated, decide whether to:
   - keep the bridge as-is for downstream warm-start
   - or open a later controlled `LoRA` follow-up
6. Keep the new `SFT v1` JSON+LoRA line as an engineering-validated path, and treat `18254` as the current formal `6`-epoch run on top of the template-fixed input pipeline.
7. Do **not** reopen the unresolved semantic-label conflict pool unless semantic coverage becomes the actual blocker again.

This is the current cleanest next decision point.

### 7.23 Two-Stage CLIP-First Alignment Decision (`2026-04-11`)

The alignment route is now explicitly split into two stages.

`Stage 1`:

- CLIP-style coarse semantic alignment
- text is short natural language
- target semantics are only:
  - `scene`
  - `top / bottom / left / right`
- `Qwen` acts only as a text encoder here
- image and text are aligned in a dual-tower contrastive setup

`Stage 2`:

- caption-token fine alignment
- this is the first stage where aligned visual tokens are injected into `Qwen`
- finer local targets such as grid / pixel-box descriptors belong here, not in Stage 1

Current implementation choice for the shared Stage-1/Stage-2 bridge:

- keep all DINO patch tokens
- use explicit token-center `(cx, cy)` geometry encoding in `[-1, 1]`
- use a token-alignment block implemented as a light residual MLP in the first version
- keep the Stage-1 readout head light and disposable
- the main reusable capacity should live in:
  - geometric position encoding
  - projector
  - token-alignment MLP

Formal execution spec:

- `docs/rc_two_stage_clip_alignment_spec_20260411.md`

### 7.24 Stage-1 Implementation Scaffold Started (`2026-04-11`)

The Stage-1 implementation scaffold is now in place around the agreed
two-stage route.

New local files:

- semantic export:
  - `scripts/export_rc_semantic_align_dataset_view.py`
- semantic dataset:
  - `unimapgen/data/rc_semantic_align_dataset.py`
- Stage-1 model:
  - `unimapgen/models/qwen3_rc_dinov2_clip_align.py`
- Stage-1 trainer:
  - `scripts/train_qwen3_rc_dinov2_clip_align.py`
- Stage-1 launcher:
  - `stagea_rc_dinov2_clip_scene_sides_pad518_4gpu_20260411.sbatch`

Stage-1 implementation rules that must stay fixed in the first pass:

- keep `518 -> 37 x 37 = 1369` tokens
- keep all DINO patch tokens
- use explicit token-center `(cx, cy)` coordinates in `[-1, 1]`
- use a token-alignment block implemented as a light residual `MLP`
- keep the Stage-1 readout head light:
  - mean pooling -> linear projection -> normalized `z_img`
- keep `Qwen` text-only in `Stage 1`
- use grouped / multi-positive CLIP-style loss keyed by `(scene, side_set)`
- set `PYTHONNOUSERSITE=1` in the Stage-1 launcher:
  - this avoids user-site `torch` / `NCCL` collisions seen in interactive remote import checks

The main reusable bridge that must transfer from `Stage 1` to `Stage 2` is:

`DINOv2 tokens -> geometric-pos MLP -> projector -> token-alignment MLP`

The Stage-1 readout head is temporary and should not be treated as the main
carrier of transferable alignment capacity.

Stage-1 semantic export is now also available remotely:

- output root:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_from_captionv3_20260411`
- current summary:
  - train rows: `435`
  - val rows: `0`
  - patch asset link: `patches512_offset` symlink
- example semantic text form:
  - `This is a black-background BEV road-structure patch showing a straight road scene. Visible road structure reaches the bottom and right sides.`

First launcher issue found during initial smoke submission:

- first submitted job: `17844`
- failure mode:
  - inherited template still had `unset PYTHONNOUSERSITE`
  - this let remote user-site `torch` override the env torch and caused a `libtorch_cuda.so` / `ncclCommWindowDeregister` import error before training
- fix applied:
  - remove `unset PYTHONNOUSERSITE`
  - also prefix both `py_compile` and `torch.distributed.run` with `PYTHONNOUSERSITE=1`

Second smoke issue found after the environment fix:

- repaired job: `17845`
- status:
  - training itself started correctly
  - first logged losses reached roughly `2.08 -> 1.57` before the save event
- failure mode:
  - `Trainer` tried to save a safetensors checkpoint at `checkpoint-100`
  - tied weights between `language_model.model.embed_tokens.weight` and `language_model.lm_head.weight` triggered a shared-memory save error
- fix applied:
  - set `save_safetensors=False` in the Stage-1 training script when the argument is supported

### 7.2 If Geometry Is Still the Bottleneck

The next improvement should probably target geometry quality directly, rather than adding more architectural complexity immediately.

Most promising directions:

- stronger line-level continuity / topology supervision
- line-wise or polyline-wise matching loss
- more explicit structure constraints during decoding

Less urgent than that:

- adding back query-generated coarse centerline heatmap
- returning to multi-layer coord heads

### 7.3 Data-Side Follow-Up

If the next clean run still struggles, the most useful data-side follow-up is likely:

- inspect hard failure cases from the new root
- especially intersection-heavy patches
- determine whether some centerline targets are still too dense or too ambiguous for the current sequence format

### 7.4 New Priority Pivot on 2026-04-09: Split the End-to-End Stack

The project direction was updated after the initial end-to-end RC runs:

- do not treat the next step as "keep stacking more heads onto Qwen first"
- first decouple the RC system into stages
- validate the visual encoder by itself before making more end-to-end changes

The intended staged route is now:

1. visual encoder benchmark
2. feature-to-token alignment / resampler study
3. SFT on the full RC serialization route
4. RL only if the SFT route becomes strong enough to justify it

So the immediate highest-priority question is no longer:

- "which end-to-end Qwen variant is best?"

It is now:

- "which visual encoder actually understands RC road structure best?"

### 7.5 Visual Encoder Benchmark Plan for RC

This benchmark was the next clean experimental line and is now considered sufficiently informative.

#### 7.5.1 Goal

Use a pure visual dense prediction task to test whether the encoder has learned RC structure well enough.

The task should **not** be centerline sequence generation yet.

The first target is:

- `structure segmentation` only
- if this branch is resumed, relaunch it on the new thin-mask route first:
  - use `seg_structure_multiclass`
  - do not reuse the earlier RGB-derived `structure_multiclass` benchmark output as the final answer

Do **not** add the centerline heatmap head in the first visual benchmark.

Reason:

- it is the cleanest way to isolate visual encoder quality
- it avoids mixing "road structure understanding" with the harder centerline geometry task
- it makes ResNet vs DINOv2 comparison much easier to interpret

#### 7.5.2 First Benchmark Task Definition

Input:

- RC patch `512 x 512`

Output:

- structure segmentation logits at `512 x 512`

Recommended first label setting:

- binary structure segmentation
- `background`
- `road-structure`

For this first binary task, `road-structure` should include the visible RC structure lines:

- `lane_divider`
- `road_edge`
- `ped_edge`

Do **not** make centerline a separate second head in this first benchmark.

#### 7.5.3 Unified Comparison Protocol

To compare encoders fairly:

- use exactly the same RC train root
- use the same `512 x 512` input
- use the same binary structure GT
- use the same `512 x 512` output resolution
- use the same decoder depth as much as possible
- use the same loss family
- use the same validation/render protocol

The only intended major variable should be the encoder.

#### 7.5.4 ResNet Benchmark Model

Recommended model:

`RC 512x512 -> ResNet50 -> FPN -> fused dense feature 128x128x256 -> lightweight seg decoder -> structure seg head -> logits 512x512x1`

Suggested details:

- backbone: `ResNet50`
- initialization: pretrained weights
- FPN output channels: `256`
- fused feature resolution: `128 x 128`
- seg decoder output resolution: `512 x 512`
- final seg head: `1x1 conv`

Suggested training policy:

- freeze `stem + layer1`
- train `layer2-4 + FPN + seg decoder + seg head`

Suggested loss:

- `BCEWithLogits(pos_weight)`
- plus `Dice`

This should be treated as the strongest practical CNN baseline for RC structure understanding.

#### 7.5.5 DINOv2 Benchmark Model

Recommended model:

`RC 512x512 -> DINOv2 ViT-L/14 -> patch tokens -> reshape to token map -> token adapter -> seg decoder -> structure seg head -> logits 512x512x1`

Important token-count note:

- DINOv2 patch size is `14`
- with `512 x 512` input, patch tokens are `36 x 36 = 1296`
- plus one `CLS` token if the implementation keeps it
- for dense prediction, use the patch tokens and reshape them to a `36 x 36` token map

Suggested details:

- backbone: `DINOv2 ViT-L/14`
- patch tokens only for dense map construction
- token adapter: `1x1 conv / projection -> 256 channels`
- seg decoder output resolution: `512 x 512`

Important training policy update:

- do **not** fully freeze DINOv2
- full freezing was already judged too weak for RC
- first benchmark should unfreeze the **last half** of the backbone blocks

For `ViT-L/14` this means:

- total blocks: `24`
- freeze the first `12`
- unfreeze the last `12`
- also train the final norm, token adapter, seg decoder, and seg head

Suggested LR split:

- smaller LR for unfrozen DINOv2 blocks
- larger LR for adapter/decoder/head

#### 7.5.6 Why the Benchmark Output Should Be 512x512

Earlier RC auxiliary heads in the end-to-end route were effectively supervised at low resolution.

That is acceptable for coarse structure cues, but it is too coarse for diagnosing fine RC geometry quality.

The updated benchmark recommendation is:

- do not stop at `128 x 128`
- produce `512 x 512` structure logits for both encoder families

Important implementation clarification:

- for the ResNet route, it is fine if the FPN fused feature stays at `128 x 128`
- then a lightweight decoder upsamples it to `512 x 512`
- there is no need to rewrite the backbone itself to be natively `512 x 512`

The same logic applies to DINOv2:

- construct a dense token map
- then use a lightweight decoder to reach `512 x 512`

#### 7.5.7 Metrics and Validation Outputs

First-round benchmark metrics:

- IoU
- Dice
- precision
- recall

Required outputs:

- fixed validation split
- a rendered panel for each selected sample:
  - RC input
  - GT binary structure mask
  - predicted structure mask
  - overlay comparison

This benchmark should answer:

- can the encoder recover RC structure at all?
- which encoder gives cleaner line continuity?
- which encoder produces less broken / over-thick / over-merged line structure?

#### 7.5.8 Practical Decision Rule

If `ResNet50+FPN` clearly wins on RC structure segmentation:

- keep ResNet/FPN as the main encoder family for the next RC stages

If `DINOv2` becomes competitive only after partial unfreezing:

- then it is still worth keeping as the main ViT/token candidate for later feature-to-Qwen alignment

But the benchmark should be used to decide that with actual dense prediction results, not intuition alone.

### 7.6 End-to-End RC Mainline vs Visual Benchmark Result

Important handoff clarification:

- the old and current end-to-end RC/Qwen lines still matter as historical baselines
- but they are **not** the next highest-priority implementation target anymore

The visual benchmark has already served its purpose:

- it validated the RC train root
- it showed `ResNet50+FPN` is the stronger dense-seg baseline
- it showed `DINOv2` is still the preferred token-native candidate for feature-to-Qwen alignment

The next highest-priority implementation target is therefore:

- DINOv2-based caption-only pre-alignment

Do not mix this new pre-align route with:

- multicoord/query-heatmap changes
- RL planning
- extra sequence-format changes
- direct return to end-to-end centerline decoding

until the first DINOv2 caption-only bridge is actually running.

### 7.7 New Priority Pivot on 2026-04-10: DINOv2 Caption-Only Pre-Alignment

The next implementation line should now start from the following decision:

- use the RC-trained `DINOv2` encoder as the visual backbone for pre-alignment
- do **not** start from the old ResNet/query-resampler/Qwen centerline generator
- first align visual features to Qwen with a short text task

The first-stage pre-alignment route is now defined as:

`RC patch (512x512, optional model-side pad to 518) -> DINOv2 -> all patch tokens -> projector + grid position embeddings -> frozen Qwen3-4B -> caption_short`

Important details:

- the first version should keep **all** DINO patch tokens
- therefore the first version should **not** use a query resampler
- if input stays at `512 x 512`, token count is:
  - `36 x 36 = 1296`
- if model-side center padding to `518 x 518` is enabled, token count becomes:
  - `37 x 37 = 1369`
- do **not** rebuild the dataset to `518 x 518`
- if the `518` route is used, apply center padding inside the model only

Recommended first training policy:

- freeze `DINOv2`
- freeze `Qwen3-4B`
- train only:
  - `visual_projector`
  - `visual_type_embedding`
  - optional position-scale parameter such as `alpha_pos`

This is intentionally a very pure bridge-learning experiment.

### 7.8 `caption_short` Pre-Alignment Design

The first text target should be **caption only**.

Do not start the first DINOv2 pre-align run with:

- schema text
- count buckets as hard fields
- centerline coordinate generation
- topology JSON

#### 7.8.1 What the caption should talk about

Because the current RC input has no explicit intersection connector geometry and centerlines are target-side virtual geometry, the caption should only describe **visible road structure**.

Allowed concepts:

- road-structure patch
- branches
- parallel branches
- central gap
- internal stopping / stopping before the central gap

Avoid these phrases in the first version:

- `visible centerlines`
- `junction boundary`
- `entry side`
- `exit side`

#### 7.8.2 Suggested coarse scenario set

Use a small coarse label space in caption generation:

- `straight`
- `curved`
- `intersection-approach`
- `branching`
- `complex`

Do not start by forcing:

- `t_junction`
- `crossroad`

because the cleaned RC input often does not expose enough explicit connector geometry to support those finer labels reliably.

#### 7.8.3 Caption templates

Use highly templated short English text.

Suggested template set:

```text
A straight road-structure patch with parallel branches extending from {side_a} to {side_b}.

A curved road-structure patch with branches bending from {side_a} toward {side_b}.

An intersection-approach road-structure patch with visible branches on the {side_list}. The structures stop before the central gap.

A branching road-structure patch with visible branches on the {side_list}.

A complex road-structure patch with road structures on the {side_list}.
```

If coarse branch-count wording is added later, only use:

- `one`
- `two`
- `three`
- `multiple`

Do not use exact centerline counts in the first caption-only version.

#### 7.8.4 Automatic caption generation rules

The caption generator should read the current patch-level geometry from:

- `centerline_json_path`
- `structure_json_path`

with the current dataset source still rooted at:

- `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`

Practical extraction rules:

1. derive `visible sides` from line endpoints close to patch borders
2. derive `straight / curved / branching / intersection-approach` from:
   - number of visible sides
   - whether parallel dominant direction exists
   - whether multiple internal endpoints stop before a shared central empty region
3. use `intersection-approach` when multiple branches are visible and several lines stop internally before a central gap
4. use `straight` when the dominant visible geometry is approximately parallel and continuous across opposite sides
5. use `curved` when one dominant branch bends between two non-opposite sides
6. fall back to `complex` if the patch does not fit the earlier rules reliably

Important implementation note:

- the first version does **not** need perfect linguistic richness
- it only needs stable, low-entropy, visually truthful captions

### 7.9 Immediate Implementation Tasks

The next agent should implement the DINOv2 pre-align route in this order:

1. add a new caption-only dataset builder / formatter for RC pre-alignment
2. add a `no-resampler DINO path` that forwards all DINO patch tokens to the projector
3. support optional model-side `512 -> 518` center padding before DINOv2
4. set `<vis_patch>` count to:
   - `1296` for `512`
   - `1369` for padded `518`
5. start with `cutoff_len = 2048`
6. first train only the projector-side bridge with both DINOv2 and Qwen frozen

### 7.10 `caption_short` Work Completed on `2026-04-10`

The following caption-prealignment preparation work is now complete:

- caption dataset formatter:
  - `unimapgen/data/rc_caption_short_dataset.py`
- caption train-root exporter:
  - `scripts/export_rc_caption_short_dataset_view.py`
- caption quality preview renderer:
  - `scripts/render_rc_caption_short_preview.py`
- export sbatch:
  - `build_rc_caption_short_trainroot_from_existing_20260410.sbatch`
- preview sbatch:
  - `preview_rc_caption_short_quality_20260410.sbatch`

Completed jobs:

- first export attempt:
  - `17510`
  - failed quickly because user-site torch packages under `~/.local` polluted the job environment and triggered an NCCL symbol mismatch
- corrected export:
  - `17513`
  - completed successfully
- first preview attempt:
  - `17551`
  - canceled because the original preview script loaded too much dataset state before sampling and was unnecessarily slow
- corrected preview:
  - `17556`
  - completed successfully

Environment hardening that should be preserved in future sbatch work:

- `export PYTHONNOUSERSITE=1`
- `unset PYTHONPATH || true`
- do not prepend `~/.local` torch site-packages into the runtime path

Preview output dir:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_quality_20260410`

Preview summary:

- rendered panels: `96`
- label balancing:
  - `branching = 20`
  - `complex = 19`
  - `curved = 19`
  - `intersection-approach = 19`
  - `straight = 19`
- missing images: `0`
- samples with structure overlay: `96`
- samples with centerline overlay: `96`

Human QA verdict from the pulled local review set:

- the current caption quality is good enough for the **first alignment-layer training run**
- the captions are intentionally low-entropy and templated, which is desirable for projector-side bootstrap alignment
- `branching`, `straight`, `curved`, and `intersection-approach` look broadly stable
- the weakest class is still `complex`, but it is acceptable as a conservative fallback class in the first run
- this caption root should be treated as a bootstrap alignment target, not as a final rich-caption corpus

Important correction after the first QA round:

- centerlines are **not** visible in the RC input and must not be treated as visible caption semantics
- the first QA render used orange centerline overlay only as a target-side reference aid
- future caption QA preview should default to structure-only overlay unless reference centerlines are explicitly requested
- caption generation should rely on `structure_lines` only, not on centerline fallback

Current caption-quality issues seen in the `input -> caption` QA render:

- the biggest semantic weakness is still the `complex` fallback class
- several `complex` samples are visually closer to `straight` partial parallel corridors than to genuinely complex structure
- `intersection-approach` is sometimes overused for patches that look more like full connected intersections or fragmented multi-road scenes
- very sparse crops can still receive overconfident direction wording even when the visible context is weak
- in short: the current caption set is still acceptable for bootstrap projector alignment, but it should not be treated as final high-quality semantic annotation

Practical caution:

- a small number of previewed samples are very sparse
- in the 96-sample QA pull, `8` samples had `num_structure_lines <= 2`
- if the first alignment run appears noisy, the first easy ablation is to filter out extremely sparse caption samples before retraining

### 7.11 `caption_short` Rule Fixes and Sparse Filtering on `2026-04-10`

The first `96`-panel QA round exposed three main rule weaknesses:

- `straight` was too conservative and let some obvious partial parallel corridors fall into `complex`
- `intersection-approach` was too easy to trigger on busy but already connected structure
- sparse crops could still receive overconfident captions

The following rule-side fixes are now implemented:

- `side support` now uses border-support evidence from visible structure coverage rather than raw endpoint-touching alone
- `straight corridor` detection now has a stronger near-parallel corridor path instead of requiring a naive visible-two-side pattern
- `intersection-approach` is now gated more tightly and explicitly vetoed when center-crossing structure is present
- `complex` has an earlier `loop-like road structures` fallback for clear loop-style patches
- `straight` / `curved` wording now avoids fake travel-direction phrasing such as `from X to Y`

Sparse-filtering support is now also implemented in the exporter:

- exporter flag:
  - `--min-structure-lines`
- kept-row metadata field:
  - `caption_min_structure_lines_filter`
- dropped-row accounting:
  - `drop_reason = sparse_structure`

New export and preview jobs:

- rule-fixed sparse-filtered export:
  - `17595`
- `500`-panel `input -> caption` preview on the new root:
  - `17596`

Current recommended caption pre-alignment root:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_rulefix_sparse3_20260410`

New root summary:

- `min_structure_lines = 3`
- `train_rows = 99,542`
- `val_rows = 5,216`
- `train_drop_counts`:
  - `sparse_structure = 9,222`
- `val_drop_counts`:
  - `sparse_structure = 429`
- `train_caption_labels`:
  - `straight = 77,811`
  - `branching = 13,652`
  - `intersection-approach = 5,345`
  - `curved = 1,592`
  - `complex = 1,142`
- `val_caption_labels`:
  - `straight = 3,992`
  - `branching = 763`
  - `intersection-approach = 291`
  - `curved = 107`
  - `complex = 63`

`500`-panel preview summary on the new root:

- output dir:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_rulefix_sparse3_500_20260410`
- rendered panels:
  - `500`
- label distribution:
  - `straight = 379`
  - `branching = 68`
  - `intersection-approach = 36`
  - `curved = 10`
  - `complex = 7`
- missing images:
  - `0`

Human QA on a random `100`-panel local pull from that `500`-panel preview:

- sampled label distribution:
  - `straight = 83`
  - `branching = 14`
  - `intersection-approach = 2`
  - `curved = 1`
  - `complex = 0`
- overall quality is clearly better than the pre-fix root
- the old failure mode where obvious partial parallel corridors were mislabeled as `complex` is substantially reduced
- `branching` and `intersection-approach` look much more stable than in the first QA round
- the random `100` sample did not include any `complex` rows, so `complex` still needs a targeted class-specific spot check later

Remaining weaknesses after the rule fix:

- the dominant residual error is now `straight` over-generalization on some crossing / connector-heavy scenes
- patches with a strong parallel corridor prior can still absorb nearby crossing structure and get a coarse `straight` caption
- a few sparse-but-kept samples are still semantically thin even after the `min_structure_lines = 3` filter

Practical QA verdict for the new root:

- good enough to start the first DINOv2 -> projector -> frozen Qwen alignment run
- do not treat it as the final caption corpus
- if another data-side cleanup pass is needed, the next highest-value rule change is to add a stronger `straight` veto for obvious crossing / multi-axis connector structure

### 7.12 `straight` Cross-Veto Fix and `complex` Spot Check on `2026-04-10`

After the first `rulefix_sparse3` QA round, one major residual issue was still visible:

- a minority of crossing / connector-heavy scenes were still being absorbed by the fallback `straight` template

The key implementation correction is now in place:

- `straight` no longer relies only on the earlier `parallel corridor` branch
- a new cross-veto now also blocks the looser fallback `straight` route when strong off-axis structure support is present
- in practice this catches multi-axis scenes that still have a dominant corridor prior but clearly contain additional crossing geometry

Files updated for this pass:

- `unimapgen/data/rc_caption_short_dataset.py`
- `scripts/render_rc_caption_short_preview.py`
- `preview_rc_caption_short_quality_20260410.sbatch`

New capability added to preview tooling:

- `--filter-labels`
- this is mainly useful for class-targeted QA such as `complex`-only review

Single-sample regression check on previously problematic examples:

- `IfDKJGa3z4LEFCYj5JSLGEuw9LI4UIlI__Summer_2020__ox256oy000__r02c02`
  - old behavior:
    - `straight`
  - new behavior:
    - `branching`
- `GgEzZgtFZ7xblomeMqUMC0STDZ2Vuo17__Autumn_2020__ox000oy256__r01c04`
  - old behavior:
    - `straight`
  - new behavior:
    - `branching`
- `wy5k4ANsqQSz7VHgaM5W2s5dkAEyKtmJ__Autumn_2020__ox256oy000__r04c00`
  - old behavior:
    - `straight`
  - new behavior:
    - `branching`
- `a3eqK8HdSbNrpUuyFBM3n1jW2nQdMz9g__Spring_2020__ox000oy000__r05c03`
  - old behavior:
    - `straight`
  - new behavior:
    - `complex`

New export / preview jobs:

- cross-veto export:
  - `17611`
- random `500`-panel `input -> caption` preview:
  - `17612`
- `complex`-only `64`-panel preview:
  - `17613`

Current recommended caption pre-alignment root is now:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_rulefix_sparse3_crossveto_20260410`

Cross-veto root summary:

- `min_structure_lines = 3`
- `train_rows = 99,542`
- `val_rows = 5,216`
- `train_drop_counts`:
  - `sparse_structure = 9,222`
- `val_drop_counts`:
  - `sparse_structure = 429`
- `train_caption_labels`:
  - `straight = 63,963`
  - `branching = 19,955`
  - `complex = 8,840`
  - `intersection-approach = 5,345`
  - `curved = 1,439`
- `val_caption_labels`:
  - `straight = 3,329`
  - `branching = 1,063`
  - `complex = 433`
  - `intersection-approach = 291`
  - `curved = 100`

Random `500`-panel preview summary on the cross-veto root:

- output dir:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_rulefix_sparse3_crossveto_500_20260410`
- label distribution:
  - `straight = 307`
  - `branching = 104`
  - `complex = 44`
  - `intersection-approach = 36`
  - `curved = 9`

`complex`-only spot check summary:

- output dir:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_rulefix_sparse3_crossveto_complex_20260410`
- rendered panels:
  - `64`
- missing images:
  - `0`
- human spot-check verdict:
  - the `complex` bucket is no longer just a hiding place for obvious partial parallel corridors
  - most reviewed `complex` samples are genuinely awkward multi-axis / junction-heavy / incomplete-connector scenes where `straight` would be worse
  - some samples are still semantically coarse and could arguably also fit `branching`, but the class now behaves like a reasonable conservative fallback instead of a clear heuristic failure

Residual caution after the cross-veto fix:

- the fix intentionally moves a substantial amount of data mass from `straight` into `branching` / `complex`
- this is directionally desirable for the previously observed error mode, but a later class-balance review is still worthwhile before very long training runs
- there may now be a small opposite-side regression risk where some curvy multi-arm scenes become `branching` instead of `curved`; keep this on the watch list in the next QA pull

### 7.13 Final Execution Spec for Probe-Augmented Caption Pre-Alignment

The next pre-alignment target should use a **structured caption schema** rather than a free-form natural-language sentence.

Final target design:

1. one global scene label
2. one local probe-state sequence

Constant priors such as `black background`, `two visible road classes`, and the downstream
goal of `road-centerline prediction` should be stated in the **system prompt**, not repeated
inside every assistant target.

Final system prompt:

```text
You are a structured BEV road-scene captioning assistant.
The image is a black-background BEV road-structure image with two visible road classes.
The downstream task is road-centerline prediction.
The image itself does not show centerlines directly; use only visible road structure.

You must output:
1. one global scene label
2. one probe-state sequence for the provided probe centers

Allowed Scene labels:
straight, curved, branching, intersection-approach, complex

Allowed ProbeStates labels:
background, lane_boundary, lane_divider, mix

Each probe center corresponds to an 8x8 box centered at that point.
For each probe:
- output background if the 8x8 box contains no road-class pixels
- output lane_boundary if it contains only lane_boundary pixels
- output lane_divider if it contains only lane_divider pixels
- output mix if it contains both lane_boundary and lane_divider pixels

Output format must be exactly:
Scene=<scene_label>
ProbeStates=[state_1,state_2,...,state_16]

Do not add explanations or extra text.
```

Final user prompt:

```text
Predict:
1. Scene
2. ProbeStates in the same order as ProbeCenters

ProbeCenters=[(x1,y1),(x2,y2),...,(x16,y16)]
```

Final assistant target format:

```text
Scene=<scene_label>
ProbeStates=[state_1,state_2,...,state_16]
```

Final scene-label vocabulary:

- `straight`
- `curved`
- `branching`
- `intersection-approach`
- `complex`

Final probe-state vocabulary:

- `background`
- `lane_boundary`
- `lane_divider`
- `mix`

Probe-center policy:

- probe centers should **not** be globally fixed at one universal set of positions
- to avoid teaching the model to overfit a single canonical probe layout, each patch should carry its own `16` probe centers
- probe centers must still be deterministic per sample once exported
- recommended implementation:
  - divide the `512 x 512` patch into a `4 x 4` macro-grid
  - sample one probe center inside each macro-cell with bounded jitter
  - seed the probe layout from a stable sample identifier so the same patch always gets the same `16` centers
- probe order in `ProbeStates` must exactly match the order listed in `ProbeCenters`

Probe-window rule:

- each probe center defines an `8 x 8` box centered at that point
- boxes must be clipped or sampled safely so they remain valid inside the patch boundary

Final probe classification rule:

- if the `8 x 8` box contains no visible road-class pixels:
  - `background`
- if the `8 x 8` box contains only `lane_boundary` pixels:
  - `lane_boundary`
- if the `8 x 8` box contains only `lane_divider` pixels:
  - `lane_divider`
- if the `8 x 8` box contains both `lane_boundary` and `lane_divider` pixels:
  - `mix`

Important detail for `mix`:

- `mix` uses **presence**, not thresholding
- as long as both visible road classes appear in the `8 x 8` box, the probe state is `mix`
- background pixels do not affect whether the probe is labeled `mix`

Why this is the current preferred schema:

- the `Scene=` line preserves coarse topology
- the `ProbeStates=` line injects compact local occupancy / class evidence
- listing `ProbeCenters` in the user prompt avoids tying the model to one single universal probe layout
- the vocabulary is fully closed and easy to QA, count, and debug
- the target remains compact enough for projector-side alignment

Reference example:

User prompt:

```text
Predict:
1. Scene
2. ProbeStates in the same order as ProbeCenters

ProbeCenters=[(71,66),(181,81),(333,59),(429,92),(88,152),(201,176),(309,187),(447,139),(61,309),(190,321),(337,300),(452,350),(95,432),(174,441),(316,415),(438,446)]
```

Assistant target:

```text
Scene=branching
ProbeStates=[background,background,lane_boundary,lane_boundary,background,lane_divider,mix,lane_boundary,background,background,lane_divider,lane_boundary,background,background,background,background]
```

Important caution:

- this schema is a **structured alignment target**, not a final rich-caption corpus
- keep the output deterministic and compact
- if later sequence-length pressure appears, shorten formatting only after preserving both the `Scene=` line and the full probe-state sequence

### 7.14 Current Implementation Status of `scene_probe_states_v1` (`2026-04-10`)

The code path for the new probe-augmented caption schema is now wired through the
main local RC caption tooling.

Updated files:

- `unimapgen/data/rc_caption_short_dataset.py`
- `scripts/export_rc_caption_short_dataset_view.py`
- `scripts/render_rc_caption_short_preview.py`

What is implemented now:

- export uses the new structured assistant target:
  - `Scene=<scene_label>`
  - `ProbeStates=[...]`
- export writes per-sample `ProbeCenters=[...]` into the **user prompt**
- export stores probe metadata into `meta_train.jsonl` / `meta_val.jsonl`:
  - `caption_schema_version = scene_probe_states_v1`
  - `caption_user_prompt`
  - `caption_probe_centers`
  - `caption_probe_states`
  - `caption_probe_box_size`
  - `caption_probe_rows`
  - `caption_probe_cols`
- dataset loading is now backward-safe for this transition:
  - if an older caption root is missing the new schema markers, the loader regenerates
    the `scene_probe_states_v1` target on the fly instead of silently using the old
    free-form natural-language caption
- preview `input_caption` mode now renders:
  - left: RC input patch with numbered probe boxes
  - right: the actual structured `user prompt + assistant target` used for training

Remote smoke validation completed on `2026-04-10`:

- smoke export root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_sceneprobe_smoke_20260410`
- smoke preview root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_sceneprobe_preview_smoke_20260410`
- smoke export result:
  - train rows: `3`
  - val rows: `2`
  - schema fields verified present in the exported sample rows and meta rows
- smoke preview result:
  - `3` `input_caption` panels rendered successfully
  - probe boxes and structured prompt/target card both rendered without missing-image issues

Important implementation note:

- the thin supervision mask still uses the internal class name `road_edge`
- the new caption schema intentionally exposes that class to Qwen as:
  - `lane_boundary`
- treat `lane_boundary` in the caption vocabulary as the textual alias of the thin
  `road_edge` mask class

Current caveat to watch:

- with the current deterministic `4 x 4` jittered macro-grid probe policy, some sparse
  patches will naturally have many `background` probe states
- this is acceptable for the current `v1` bootstrap alignment target
- if later QA shows the local probe signal is too weak, the next upgrade should be:
  - denser probes, or
  - structure-aware probe placement

### 7.15 Small Formal Export for `scene_probe_states_v1` (`2026-04-10`)

Before doing a full caption-root rebuild, a small formal export was completed to check
the actual probe-augmented data distribution first.

Small export root:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_sceneprobe500_20260410`

Preview root:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_sceneprobe500_100_20260410`

Export setting:

- source root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
- schema:
  - `scene_probe_states_v1`
- `min_structure_lines = 3`
- requested subset:
  - first `500` train rows only
- actual kept rows after sparse filtering:
  - `473`
- dropped rows:
  - `27`
  - all due to `sparse_structure`
- val rows:
  - `0`
  - intentionally skipped for this small export

Label distribution on the kept `473` train rows:

- `straight = 310`
- `branching = 88`
- `intersection-approach = 44`
- `complex = 29`
- `curved = 2`

`100`-panel random preview summary:

- rendered panels:
  - `100`
- missing images:
  - `0`
- preview label distribution:
  - `straight = 56`
  - `branching = 22`
  - `intersection-approach = 12`
  - `complex = 9`
  - `curved = 1`

Probe-state distribution over the full kept `473` rows:

- total probes:
  - `7,568`
- `background = 6,705`
- `lane_boundary = 588`
- `mix = 273`
- `lane_divider = 2`

Per-sample non-background probe count:

- min:
  - `0`
- max:
  - `8`
- mean:
  - `1.8245`
- histogram head:
  - `0 -> 105`
  - `1 -> 141`
  - `2 -> 90`
  - `3 -> 64`
  - `4 -> 37`

Important early observation:

- the current probe vocabulary is wired correctly
- but on this `500`-sample check, `lane_divider` probes are extremely rare
- this is not yet a blocker for a small visual QA round
- however, before full-scale alignment training, this should be treated as a real data-signal caution:
  - the current `8 x 8`, `4 x 4` jittered probe policy may under-sample divider pixels heavily on this RC root
  - if later manual QA agrees that the right-side prompt/target format is correct but probe supervision looks too weak, the next data-side upgrade should be probe-placement improvement rather than schema redesign

### 7.16 Local `30`-Panel Spot Check and Root-Cause Finding for Rare `lane_divider` (`2026-04-10`)

After the small `sceneprobe500` export, a targeted local pull and manual spot check was
done to answer one specific question:

- why is `lane_divider` almost absent in the exported probe states?

Local pulled review directory:

- `tmp/caption_preview_inputcaption_sceneprobe500_selected30_20260410`

Contents:

- `30` pulled preview panels
- `manual_spotcheck_selected30.json`
- `summary.json`

Selection policy for the `30` pulled panels:

- `20` `divider_mismatch` samples:
  - thin mask says at least one probe should be `lane_divider`
  - current exported target says there is **no** `lane_divider`
- `5` `all_background_control` samples
- `5` `agreement_control` samples where current export already matches the thin-mask probe result

Manual spot-check conclusion:

- the main failure is real and systematic
- visually, many probes centered on cyan dashed divider structure are currently exported as:
  - `mix`
  - or occasionally `lane_boundary`
- the `all_background` controls look reasonable
- the `agreement_control` samples also look reasonable
- therefore the problem is **not** that all probe states are broken
- the problem is specifically concentrated on `lane_divider`

Key quantitative finding:

- the thin-mask pixel distribution on the same `473` rows is **not** divider-sparse
- non-background thin-mask pixels are:
  - `lane_divider = 1,030,747`
  - `lane_boundary(road_edge) = 1,254,584`
- ratio among non-background pixels:
  - `lane_divider ~= 45.1%`
  - `lane_boundary ~= 54.9%`
- rows containing at least one divider pixel:
  - `451 / 473`

This proves:

- `lane_divider` rarity in exported probe states is **not** caused by the source dataset itself

Critical diagnostic comparison on the same probe centers:

- current exported probe-state totals:
  - `background = 6,705`
  - `lane_boundary = 588`
  - `mix = 273`
  - `lane_divider = 2`
- if the exact same `ProbeCenters` are evaluated directly on the thin
  `seg_structure_multiclass` masks instead:
  - `background = 6,945`
  - `lane_boundary = 344`
  - `lane_divider = 275`
  - `mix = 4`

Practical interpretation:

- the current probe placement is **not** the main reason `lane_divider` disappeared
- the dominant issue is the current probe-state classification path

Confirmed implementation root cause:

- `build_scene_probe_caption -> classify_probe_states_from_image`
- currently calls:
  - `load_segmentation_label_map_from_path(image_path, supervision_mode=\"structure_multiclass\")`
- here `image_path` is the rendered RC RGB patch, **not** the precomputed thin
  `seg_structure_multiclass` label PNG
- therefore probe states are being inferred from RGB color reconstruction instead of the
  canonical thin mask

Observed consequence:

- true `lane_divider` probe boxes are being systematically converted into `mix`
- the most important confusion row over all `473` rows is:
  - true thin-mask `lane_divider` probes:
    - exported as `background = 19`
    - exported as `lane_boundary = 11`
    - exported as `lane_divider = 2`
    - exported as `mix = 243`

Single-sample concrete evidence:

- sample:
  - `01bb304d7bd835f8bbef7086b688e35e__Summer_2019__ox000oy256__r02c02`
- current exported panel shows:
  - `5` probe states as `mix`
- thin-mask evaluation on the same probe centers shows:
  - those same `5` probes are all pure `lane_divider`
- per-probe comparison confirmed that the RGB-derived label map sees:
  - `{1, 2}`
  - while the thin mask sees:
  - `{1}`

Most likely mechanism:

- RC RGB rendering around cyan dashed divider lines introduces color blending / nearest-color
  ambiguity at probe-box scale
- under the current presence-based rule, once both reconstructed classes appear in the
  `8 x 8` RGB-derived box, the probe becomes `mix`
- this collapses many true divider probes into `mix`

Current execution recommendation:

- do **not** launch full-scale `scene_probe_states_v1` alignment training from the current
  RGB-derived probe-state export
- first correct probe-state generation to read the canonical thin
  `seg_structure_multiclass` mask path directly from meta
- after that fix, rerun the small export + preview QA before full export

### 7.17 Switch from Random Probes to Fixed `8x8` Full-Coverage Grid (`2026-04-11`)

After the `7.16` diagnosis, the caption-target design was intentionally changed.

Final decision:

- do **not** continue with random probe supervision as the main alignment target
- switch to a fixed full-coverage grid target:
  - schema:
    - `scene_grid_states_v1`
  - output text:
    - `Scene=<scene_label>`
    - `GridStates=[state_1,...,state_64]`
  - grid:
    - fixed `8 x 8`
    - full coverage over the `512 x 512` patch
    - row-major order from top-left to bottom-right
  - allowed cell labels:
    - `background`
    - `lane_boundary`
    - `lane_divider`
    - `mix`

Important implementation rule:

- `GridStates` must be derived directly from the thin `seg_structure_multiclass` mask
- do **not** infer them back from the RC RGB patch

Code paths updated on `2026-04-11`:

- `unimapgen/data/rc_caption_short_dataset.py`
- `scripts/export_rc_caption_short_dataset_view.py`
- `scripts/render_rc_caption_short_preview.py`

What changed in code:

- `scene_probe_states_v1` is now superseded by `scene_grid_states_v1`
- dataset rebuild logic now regenerates caption targets with fixed-grid `GridStates`
- export meta now stores:
  - `caption_grid_states`
  - `caption_grid_rows`
  - `caption_grid_cols`
  - `caption_grid_count`
  - `caption_grid_order = row_major`
- preview rendering in `input_caption` mode now shows:
  - left:
    - input RC patch with fixed `8 x 8` grid overlay
  - right:
    - the actual prompt/target text used for training

Remote smoke validation completed:

- compile:
  - `py_compile` passed for all three updated files
- smoke export root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_grid8_smoke_train3_val2_20260411`
- smoke preview root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_grid8_preview_inputcaption_smoke3_20260411`

Small formal `train-only` export completed with the new grid schema:

- train-only root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_trainonly500_20260411`
- preview root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_100_20260411`
- source root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
- filter:
  - `min_structure_lines = 3`
- requested train subset:
  - first `500`
- actual kept train rows:
  - `473`
- dropped:
  - `27`
  - all due to `sparse_structure`
- val rows:
  - `0`
  - intentionally skipped in the train-only root

Train label distribution on the kept `473` rows:

- `straight = 310`
- `branching = 88`
- `intersection-approach = 44`
- `complex = 29`
- `curved = 2`

Grid-state distribution over the full kept `473` rows:

- total cells:
  - `30,272`
- `background = 20,843`
- `mix = 4,600`
- `lane_boundary = 3,707`
- `lane_divider = 1,122`

Current interpretation:

- the earlier `lane_divider ~= 0` failure was a probe-generation bug, not a true data absence
- after switching to direct thin-mask `GridStates`, divider supervision returns at a reasonable scale
- the new `8 x 8` grid does introduce many `mix` cells, but it now gives genuine full-image coverage and avoids local-attention bias from sparse probes

Practical recommendation going forward:

- use `scene_grid_states_v1` as the default caption-alignment data route
- keep the older random-probe roots only as historical debugging artifacts
- if later QA says `mix` is too dominant, the next tuning dimension should be grid granularity or cell-labeling policy, not a return to sparse random probes

### 7.18 Local `100`-Panel Grid Preview QA (`2026-04-11`)

After the fixed-grid export landed, the `100`-panel `input_caption` preview was pulled
to the local workstation and manually spot-checked again.

Local pulled full preview dir:

- `tmp/rc_caption_short_preview_inputcaption_grid8_trainonly500_100_20260411`

Local curated QA subset:

- `tmp/caption_preview_inputcaption_grid8_trainonly500_selected16_20260411`

Local written note:

- `tmp/caption_preview_inputcaption_grid8_trainonly500_selected16_20260411/spotcheck_summary.md`

Review scope:

- manually inspected `16` representative panels
- covered:
  - `straight`
  - `branching`
  - `intersection-approach`
  - `complex`
  - `curved`
- intentionally included:
  - high-`mix` panels
  - high-`lane_divider` panels
  - sparse low-signal panels

Quick quantitative snapshot on the full `100`-panel local preview:

- samples containing at least one `lane_divider` cell:
  - `64 / 100`
- samples containing at least one `mix` cell:
  - `91 / 100`
- low-signal samples with `<= 4` non-background cells:
  - `7 / 100`
- high-`mix` samples with `>= 16` `mix` cells:
  - `21 / 100`

Manual QA conclusion:

- the fixed `8 x 8` grid schema is clearly healthier than the previous random-probe route
- `lane_divider` is now visibly present and usually aligned well with dashed cyan structure
- most `GridStates` look believable enough to support the next projector-alignment run
- the remaining noise is mainly:
  - scene-label ambiguity
  - and a small number of ultra-sparse crops

Representative good cases:

- `0001`
- `0005`
- `0007`
- `0017`
- `0047`
- `0088`

These panels show:

- sensible `lane_divider`
- sensible `mix`
- and no sign of the old divider-collapse bug

Representative caution cases:

- `0010`
- `0085`
  - connector-heavy scenes still sometimes remain labeled as `straight`
- `0038`
- `0068`
  - very sparse crops are still weak for scene supervision even though the grid target itself is not obviously wrong

Current recommendation:

- treat `scene_grid_states_v1` as good enough to continue training-side work
- if a cleaner `v2` is needed later, prioritize:
  - stronger sparse filtering
  - and a stricter scene-label veto for connector-heavy pseudo-straight scenes

### 7.19 `straight` Connector Veto Tightening + Ultra-Sparse Filtering (`2026-04-11`)

Based on the `7.18` local grid QA, two concrete data-side fixes were implemented next.

Code changes:

- `unimapgen/data/rc_caption_short_dataset.py`
  - tightened the `straight` veto for connector-heavy corridor scenes
  - widened connector detection from the tight center box to the broader central region
  - added a dedicated connector-branching veto path for pseudo-straight scenes
- `scripts/export_rc_caption_short_dataset_view.py`
  - added grid-based ultra-sparse filtering
  - new export argument:
    - `--min-non-background-grid-cells`
  - current default:
    - `5`

Intended behavior:

- samples with only `1` to `4` non-background grid cells are dropped as ultra-sparse
- some connector-heavy scenes that previously stayed `straight` should now flip to `branching`

New train-only export after the rule update:

- root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_trainonly500_sparse5_connectorveto_v2_20260411`
- preview:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_100_20260411`

Export setting:

- source root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
- `min_structure_lines = 3`
- `min_non_background_grid_cells = 5`
- requested train subset:
  - first `500`

Result:

- kept train rows:
  - `435`
- dropped:
  - `27` due to `sparse_structure`
  - `38` due to `ultra_sparse_grid`
- val rows:
  - `0`

Updated train label distribution:

- `straight = 249`
- `branching = 126`
- `intersection-approach = 44`
- `complex = 14`
- `curved = 2`

Compared with the earlier grid root before these fixes:

- `straight` decreased:
  - `268 -> 249`
- `branching` increased:
  - `98 -> 126`
- ultra-sparse fragments are now explicitly removed instead of kept as weak scene-supervision samples

Concrete sanity checks:

- connector-heavy sample
  - `0262e2af16044492b13ca051d6ab4d80__Spring_2020__ox000oy256__r04c05`
  - changed:
    - `straight -> branching`
- sparse examples
  - `04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy__Spring_2020__ox000oy000__r06c07`
  - `07YOTznatmYypvQYpzviEcU3yGPsyaGg__Spring_2020__ox000oy000__r06c03`
  - now dropped from the exported root by the new ultra-sparse filter

Current recommendation:

- if we need to continue immediately, use this new sparse-filtered connector-veto `v2` grid root as the default caption-alignment root
- if later QA still finds some connector-heavy scenes staying `straight`, keep iterating on the scene heuristic only; the grid-state extraction itself is already in much better shape

### 7.20 Targeted Re-Review of Remaining `straight` Connector-Heavy Samples in `v2` (`2026-04-11`)

After exporting the sparse-filtered connector-veto `v2` root, a second local QA pass was
done, but this time focused only on the remaining `straight` samples that still looked
high-risk.

Local full preview dir:

- `tmp/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_100_20260411`

Local curated straight-review subset:

- `tmp/caption_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_straight_review_20260411`

Local written note:

- `tmp/caption_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_straight_review_20260411/straight_connector_review.md`

Review scope:

- only `straight` samples
- prioritized by:
  - high `mix`
  - relatively high non-background grid coverage
  - visual risk of hidden connector structure

Quick preview stats for the `v2` `100`-panel local preview:

- total `straight` samples:
  - `53`
- average non-background grid cells among `straight`:
  - `16.64`
- average `mix` cells among `straight`:
  - `8.98`
- high-`mix` `straight` samples (`>= 12 mix cells`):
  - `15`

Manual re-review conclusion:

- the `v2` connector veto clearly removed a meaningful portion of the earlier false-positive `straight` cases
- most of the remaining high-risk `straight` samples are still acceptable as `straight`
- remaining leakage now looks limited, not systemic

Most likely remaining false-positive:

- `0098`
  - `0262e2af16044492b13ca051d6ab4d80__Spring_2020__ox000oy000__r02c03`
  - this sample still shows a visible mid-patch connector between two main corridor groups
  - this is the clearest remaining example that still feels more like `branching`

Borderline but acceptable as `straight` after review:

- `0073`
- `0052`
- `0044`
- `0002`
- `0003`
- `0058`
- `0085`

Clearly fine as `straight`:

- `0066`
- `0088`

Current recommendation:

- keep using the current sparse-filtered connector-veto `v2` root as the active baseline
- if another heuristic iteration is desired, focus only on residual center-connector misses like `0098`
- no further change is needed right now on the grid-state side; remaining work is scene-label refinement only

### 7.21 Narrow Center-Connector-Bridge Veto for Residual `0098`-Type Misses (`2026-04-11`)

One final narrow heuristic refinement was added after the `7.20` straight-only review.

Motivation:

- `v2` had already removed most obvious connector-heavy `straight` errors
- but one remaining sample type still stood out:
  - a short internal connector bridge sitting in the central region between two otherwise parallel corridor groups
- the representative miss was:
  - `0098`
  - `0262e2af16044492b13ca051d6ab4d80__Spring_2020__ox000oy000__r02c03`

Implementation:

- added a dedicated narrow `center_bridge_veto` inside:
  - `unimapgen/data/rc_caption_short_dataset.py`
- this veto is intentionally stricter than the broader connector veto:
  - line must touch the central region
  - line must have no border-side support
  - line must be at least `min_support_length`
  - line must show both:
    - non-trivial axis deviation
    - and clear turning
  - the main corridor still needs to look strongly parallel

Direct contrastive check after the patch:

- `0098`
  - changed:
    - `straight -> branching`
- retained as `straight`:
  - `0073`
  - `0044`
  - `0002`

New train-only export after this narrow veto:

- root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_trainonly500_sparse5_connectorveto_v3_20260411`
- preview:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_100_20260411`

Result relative to `v2`:

- kept train rows:
  - unchanged at `435`
- `straight`:
  - `249 -> 248`
- `branching`:
  - `126 -> 127`
- no change to sparse filtering behavior

Current recommendation:

- use this `v3` sparse-filtered connector-veto root as the default caption-alignment root
- at this point, remaining scene-label noise looks small enough that model-side training should resume unless later QA finds a new systematic failure mode

### 7.22 Additional Random `100`-Panel QA on `v3` (`2026-04-11`)

After the narrow `center_bridge_veto` landed, one more independent random `100`-panel QA
pass was run with a different sampling seed to reduce the risk of overfitting the review to
the earlier manually inspected subset.

Remote preview:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_seed314159_100_20260411`

Local full pull:

- `tmp/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_seed314159_100_20260411`

Local curated subset:

- `tmp/caption_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_seed314159_selected11_20260411`

Local written note:

- `tmp/caption_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_seed314159_selected11_20260411/random100_qa_summary.md`

Preview composition:

- `branching = 40`
- `straight = 50`
- `intersection-approach = 6`
- `complex = 3`
- `curved = 1`

Manual review conclusion:

- this second random QA pass did **not** reveal a new systematic problem
- `v3` still looks stable enough to use as the active training baseline
- the narrow `center_bridge_veto` does not appear to over-correct the scene labels

Representative good cases:

- `0006`
- `0016`
- `0082`
- `0053`
- `0017`
- `0096`

Residual mild caution:

- some high-`mix` `straight` samples still look busy:
  - `0061`
  - `0001`
- but after manual review they still look acceptable enough to keep as `straight`
- these read more like corridor-dominant patches with extra side structure than the old `0098`-type center-bridge miss

Residual low-signal caution:

### 7.23 Random `100`-Panel Scene QA on Full `v3` Export (`2026-04-11`)

After the full `v3` export finished (`train = 96,006`, `val = 5,033`), a new independent
random `100`-panel QA pass was rendered from the **full** training root rather than the
earlier `trainonly500` subset. This was important because the smaller reviewed subset was
not large enough to guarantee that the full export had the same scene-label cleanliness.

Remote preview:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_full_sparse5_connectorveto_v3_seed20260411_100_20260411`

Local full pull:

- `tmp/rc_caption_short_preview_inputcaption_grid8_full_sparse5_connectorveto_v3_seed20260411_100_20260411`

Local written note:

- `tmp/rc_caption_short_preview_inputcaption_grid8_full_sparse5_connectorveto_v3_seed20260411_100_20260411/random100_scene_qa_summary.md`

Preview composition:

- `straight = 66`
- `branching = 25`
- `complex = 6`
- `intersection-approach = 2`
- `curved = 1`

Manual review conclusion:

- unlike the earlier `trainonly500` `v3` review, this full-root random `100` pass **did**
  reveal a real residual systematic problem
- the dominant failure mode is still `branching -> straight` over-call on sparse or partial
  corridor fragments
- about `14 / 100` samples looked like clear scene mismatches
- another `3-4 / 100` looked borderline but still arguable
- this means the full `v3` root should **not** yet be treated as scene-clean enough for
  full-scale `Stage 1` semantic alignment without another rule pass

Representative clear mismatches:

- `0003`, `0011`, `0012`, `0044`, `0045`, `0047`, `0055`, `0056`, `0078`, `0079`,
  `0082`, `0092`, `0093`
  - all labeled `branching` but visually read as sparse or clipped straight corridors
- `0021`
  - labeled `straight` but visually contains a crossing / connector-heavy structure

Borderline examples worth keeping in mind:

- `0010`
- `0042`
- `0091`
- `0094`

Practical implication:

- keep the current `trainonly500`-reviewed root as the safer reviewed semantic-source pool
  for now
- do **not** immediately switch `Stage 1` semantic export to the full `101,039`-row root
- the next data-side fix should strengthen the `branching` veto for:
  - single-orientation sparse edge fragments
  - disjoint clipped corridor pieces without a visible junction core
  - right-edge / top-edge partial strips that currently inherit `branching`

- `0035`
- `0055`
- `0040`

These crops remain semantically weak, but they are above the current sparse threshold and do
not yet justify another immediate filtering change.

Current recommendation:

- stop iterating caption heuristics for now
- use the `v3` root as the working pre-alignment data
- let the next round of model-side training and loss behavior decide whether another caption-side rule pass is actually necessary

### 7.25 Split-10 Manual Scene Review Start on Full `train=96,006` (`2026-04-11`)

To make full-root scene cleanup tractable, the full `train.jsonl` from the exported `v3`
caption root was split into `10` aligned review parts.

Important implementation detail:

- the split was redone after catching a bad first attempt
- do **not** split `train.jsonl` and `meta_train.jsonl` independently with shell tools
- the correct split is order-preserving and `id`-aligned between:
  - `train.jsonl`
  - `meta_train.jsonl`

Remote split root:

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411`

Split summary:

- total train rows: `96,006`
- part sizes:
  - `9601, 9601, 9601, 9601, 9601, 9601, 9600, 9600, 9600, 9600`
- each `part_01 ... part_10` contains:
  - `train.jsonl`
  - `meta_train.jsonl`
  - `manual_scene_review.jsonl`

For `part_01`, a first manual-review batch of the first `100` samples was rendered and
reviewed locally before writing results back to the remote part file.

Local preview root:

- `tmp/rc_caption_short_preview_inputcaption_part01_first100_20260411`

Local written review files:

- `tmp/rc_caption_short_preview_inputcaption_part01_first100_20260411/manual_scene_labels_first100.jsonl`
- `tmp/rc_caption_short_preview_inputcaption_part01_first100_20260411/manual_scene_labels_first100_summary.md`

Manual review result for the first `100`:

- reviewed: `100`
- accepted current label: `68`
- corrected: `32`
- manual distribution:
  - `straight = 74`
  - `intersection-approach = 18`
  - `branching = 7`
  - `complex = 1`

Main finding from this first human pass:

- the dominant residual problem is still `branching -> straight` over-call
- many corrected samples were sparse single-corridor or disconnected corridor fragments
- a smaller set were upgraded from `branching` to `intersection-approach`
- one sparse false `complex` sample was downgraded to `straight`
- one busier multi-connector sample was upgraded to `complex`

Remote write-back status:

- the first `100` entries of:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411/part_01/manual_scene_review.jsonl`
  have now been updated in place with:
  - `manual_scene_label`
  - `review_status=reviewed`
  - `review_note`
- backup created before overwrite:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411/part_01/manual_scene_review.pre_first100_20260411.jsonl`

Practical next step:

- continue manual scene review on `part_01` in additional `100`-sample batches
- once `part_01` is complete, move to `part_02`
- keep using the split review root rather than trying to freehand-review the full `96,006`
  rows at once

### 7.26 Scene Autocorrector, Full Auto-Scoring, and Stage-1 Smoke Success (`2026-04-12`)

This is the current state that supersedes the earlier "Stage-1 export exists but has
not yet been proven in training" situation.

New local implementation:

- scene autocorrector trainer / scorer:
  - `scripts/train_rc_scene_autocorrector.py`
- Stage-1 smoke launcher:
  - `stage1_rc_dinov2_clip_align_1gpu_smoke_20260412.sbatch`

What the autocorrector now does:

- reads the full `caption_short` root plus reviewed scene labels
- computes handcrafted scene features from:
  - `analyze_scene_caption_short(...)`
  - `GridStates(8x8)` coverage statistics
- trains a lightweight MLP ensemble with OOF calibration
- exports:
  - clean accepted semantic root
  - full scored files
  - conflict / low-confidence review pool
  - summary json
  - saved autocorrector weights

Important conservative rule added in this version:

- `--min-auto-class-support = 10`
- meaning:
  - classes with too little reviewed support must not be auto-overwritten aggressively
  - this was especially important for `curved`, which only had `2` reviewed labels

Current accepted full-root output:

- clean root:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412`
- summary:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412/summary.json`
- conflict pool:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412/review_conflicts`

Key metrics from the accepted full-root run:

- reviewed labels used: `500`
- reviewed OOF accuracy: `0.846`
- accepted threshold:
  - `accept_prob = 0.98`
  - `accept_margin = 0.05`
- accepted OOF subset:
  - coverage: `0.66`
  - accuracy: `0.9545`
- train decisions:
  - `auto_high_confidence = 75,448`
  - `needs_review_conflict = 17,434`
  - `needs_review_low_confidence = 2,624`
  - `reviewed_manual = 500`
- val decisions:
  - `auto_high_confidence = 3,973`
  - `needs_review_conflict = 942`
  - `needs_review_low_confidence = 118`

Stage-1 smoke training status on the supercomputer:

- first smoke job:
  - `17999`
- observed issue:
  - the job entered the training loop, but `loss` stayed at `0.0`
- root cause:
  - CLIP-style contrastive loss with `per_device_train_batch_size = 1` produces only a
    single positive pair in each step
  - that degenerates to a `1 x 1` softmax and zero contrastive loss
- implication:
  - do **not** use batch size `1` for this objective when validating Stage 1

Corrected smoke job:

- job id:
  - `18000`
- change:
  - `per_device_train_batch_size = 4`
  - `per_device_eval_batch_size = 4`
- same run confirmed:
  - Qwen checkpoint shards loaded
  - DINOv2 loaded
  - RC-trained visual checkpoint loaded with:
    - `missing = 0`
    - `unexpected = 0`
  - training loop entered normally
  - non-zero `loss` and `grad_norm` printed stably

Representative live log lines from `18000`:

- `loss = 1.3933, grad_norm = 1.1041`
- `loss = 1.4004, grad_norm = 2.5182`
- `loss = 1.6162, grad_norm = 5.5060`
- later losses stabilized around `1.37 - 1.40`

Smoke output and logs:

- output root:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_smoke_20260412`
- stdout:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stage1_rc_dinov2_clip_align_smoke_18000.out`
- stderr:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stage1_rc_dinov2_clip_align_smoke_18000.err`

End-of-run smoke result:

- `train_runtime = 14.4948`
- `train_steps_per_second = 3.45`
- `train_loss = 1.394464340209961`

Current conclusion:

- the `Stage 1` DINOv2 -> geometric-pos -> projector -> token-alignment MLP -> light
  readout -> frozen Qwen text-tower route is now implemented and has already been
  verified to enter real training on the supercomputer
- the remaining data-side work is no longer "can Stage 1 train at all"
- the remaining data-side work is to continue manual review only on the exported
  unresolved conflict pool if we want to expand beyond the current clean accepted subset

Follow-up formal run then completed successfully:

- new launcher:
  - `stage1_rc_dinov2_clip_align_4gpu_clean_v1_20260412.sbatch`
- job id:
  - `18001`
- current status:
  - `COMPLETED`
- training root:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412`
- media root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411`
- current formal output root:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412`
- retained outputs:
  - `checkpoint-28000`
  - `checkpoint-28482`
  - `rc_dinov2_clip_align_modules.pt`
- early confirmed live log behavior:
  - non-zero loss printed continuously from about `2.08` down through `1.47`
  - representative early lines:
    - `loss = 2.0837, grad_norm = 1.5500`
    - `loss = 2.0175, grad_norm = 2.3402`
    - `loss = 1.7918, grad_norm = 9.6912`
    - `loss = 1.3982, grad_norm = 16.0070`
- end-of-run result:
  - `global_step = 28,482`
  - `train_loss = 0.8166`
- implication:
  - both the `1`-GPU smoke and the `4`-GPU clean-root formal launcher are now confirmed
    to print non-zero contrastive loss under the current implementation

Offline retrieval evaluation was then added and run on the `val` split:

- eval script:
  - `scripts/eval_qwen3_rc_dinov2_clip_retrieval.py`
- eval launcher:
  - `stage1_rc_dinov2_clip_retrieval_eval_val_1gpu_20260412.sbatch`
- eval job:
  - `18031`
- status:
  - `COMPLETED`
- eval target:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412/val.jsonl`
- eval output dir:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/retrieval_eval_val`
- metrics file:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/retrieval_eval_val/metrics.json`

Current offline `val` retrieval metrics for the aligned model:

- samples:
  - `3,973`
- unique semantic groups:
  - `64`
- image -> text group recall:
  - `R@1 = 0.4395`
  - `R@5 = 0.4722`
  - `R@10 = 0.5454`
- text -> image group recall:
  - `R@1 = 0.7249`
  - `R@5 = 0.9650`
  - `R@10 = 0.9751`
- image -> text top-1 semantics:
  - `scene_acc = 0.6892`
  - `side_set_acc = 0.6313`
- text -> image top-1 semantics:
  - `scene_acc = 0.9607`
  - `side_set_acc = 0.7473`
- similarity margins:
  - image -> text:
    - `margin_mean = -0.0289`
    - `margin_positive_rate = 0.4395`
  - text -> image:
    - `margin_mean = 0.0062`
    - `margin_positive_rate = 0.7249`

Interpretation:

- the aligned model has clearly learned non-trivial semantic structure; this is no longer a
  "loss decreases but retrieval is random" situation
- however the retrieval is still asymmetric:
  - `text -> image` is already fairly strong
  - `image -> text` is only moderate
- in particular, the negative `image -> text` mean margin shows that many images still rank a
  wrong semantic text above the best positive text
- so the current conclusion should be:
  - `Stage 1` is working and useful
  - but it is not yet strong enough to call "fully solved" coarse alignment

### 7.27 Stage-2 Caption Spec Revision and Smoke Success (`2026-04-12`)

`Stage 2` has now been rewritten to match the accepted latest design.

New accepted `Stage 2` rule:

- reuse the `Stage 1` bridge directly
- do **not** add visual type embedding
- do **not** add `alpha_pos`
- do **not** add any extra visual injection module
- keep `Qwen` frozen in the first `Stage 2` run
- keep the `Qwen` backbone frozen in the first `Stage 2` run
- keep `DINOv2` frozen in the first `Stage 2` run
- inject visual tokens only by direct replacement of `<vis_patch>` embedding slots
- allow `<vis_start>` and `<vis_end>` token rows to train

Current formal execution spec:

- `docs/rc_stage2_caption_execution_spec_20260412.md`

Current code paths after the rewrite:

- model:
  - `unimapgen/models/qwen3_rc_dinov2_caption_llava.py`
- trainer:
  - `scripts/train_qwen3_rc_dinov2_caption_llava.py`
- predictor:
  - `scripts/predict_qwen3_rc_dinov2_caption_llava.py`
- formal launcher:
  - `stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412.sbatch`
- smoke launcher:
  - `stage2_rc_dinov2_caption_grid8_stage1init_smoke_1gpu_20260412.sbatch`

Current `Stage 2` caption schema remains:

- `Scene=<scene_label>`
- `GridStates=[state_1,...,state_64]`

Current `Stage 2` warm-start source:

- `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/rc_dinov2_clip_align_modules.pt`

Important implementation fix discovered during smoke:

- the first `Stage 2` smoke failed because `cutoff_len = 2048` truncated the `<vis_patch>` block
- the collator correctly caught this as:
  - expected `1369` visual patch tokens
  - actual truncated count `890`
- current accepted fix:
  - use `cutoff_len = 4096` for `Stage 2`

Successful corrected smoke:

- job id:
  - `18036`
- status:
  - `COMPLETED`
- bridge loading confirmed:
  - `vision_encoder`
  - `visual_norm`
  - `visual_projector`
  - `geometric_position_mlp`
  - `token_alignment`
- trainable params:
  - `57,970,176`
- non-zero caption CE loss printed:
  - `0.7565`
  - `0.7284`
  - `0.5380`
  - `0.7638`
- final smoke train loss:
  - `0.6967`

Current conclusion:

- the revised `Stage 2` code now matches the accepted architecture
- the direct `<vis_patch>` replacement route with frozen `Qwen` backbone, trainable visual
  boundary-token rows, and frozen `DINOv2`
  has already been validated by smoke training

Formal `4`-GPU `Stage 2` training was then submitted from the corrected launcher:

- launcher:
  - `stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412.sbatch`
- job id:
  - `18037`
- current status:
  - `CANCELLED` after the user changed the training target from `3 epoch` to `1 epoch`
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412`
- stdout:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_18037.out`
- stderr:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_18037.err`
- current confirmed live behavior:
  - `Stage 1` bridge warm-start loaded with:
    - `vision_encoder`
    - `visual_norm`
    - `visual_projector`
    - `geometric_position_mlp`
    - `token_alignment`
  - trainable params:
    - `57,970,176`
  - first logged caption losses:
    - `0.5469`
    - `0.5670`
    - `0.5667`
    - `0.5965`
    - `0.5818`

Current implication:

- the formal distributed `Stage 2` launcher has already moved beyond smoke-only validation
- the `4`-GPU run is now in real caption-token training with stable non-zero loss logging

Current active formal `Stage 2` run after the `1 epoch` change:

- launcher:
  - `stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412.sbatch`
- effective launcher default:
  - `NUM_TRAIN_EPOCHS = 1`
- job id:
  - `18038`
- current status at handoff refresh:
  - `FAILED` at the first checkpoint save
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_4gpu_20260412`
- stdout:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_18038.out`
- stderr:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/logs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_18038.err`
- confirmed launch behavior:
  - warm-start modules loaded successfully from the `Stage 1` bridge
  - trainable params remain:
    - `57,970,176`
  - first logged loss:
    - `0.5448`
  - trainer progress bar total steps:
    - `24,002`
  - first online eval before failure:
    - `eval_loss = 0.1845` at `step = 1,000`
  - failure root cause:
    - Hugging Face `Trainer` attempted `safetensors` checkpoint save
    - tied weights between `language_model.model.embed_tokens.weight` and `language_model.lm_head.weight`
      triggered the shared-memory save error

Current implication:

- the `1 epoch` setting itself was correct
- the blocker was checkpoint serialization, not forward/backward training stability

Checkpoint-save fix applied on `2026-04-12`:

- code fix:
  - `scripts/train_qwen3_rc_dinov2_caption_llava.py`
- exact change:
  - set `save_safetensors = False` when supported by local `TrainingArguments`
- verification save-smoke job:
  - `18077`
- verification status:
  - `COMPLETED`
- verification output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_caption_save_smoke_1gpu_20260412`
- verification result:
  - `checkpoint-20/pytorch_model.bin` saved successfully
  - `checkpoint-25/pytorch_model.bin` saved successfully
  - final `rc_dinov2_caption_modules.pt` saved successfully
  - final smoke `train_loss = 0.4334`

Current re-launched formal `Stage 2` run after the save fix:

- job id:
  - `18080`
- current status at handoff refresh:
  - `RUNNING` on `gpu48`
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412`
- current accepted token-freeze rule in this run:
  - freeze the `Qwen` backbone
  - do **not** train `<vis_patch>`
  - train only `<vis_start>` and `<vis_end>` token rows
- verification smoke for the token-freeze rule:
  - job id:
    - `18088`
  - current status at handoff refresh:
    - `COMPLETED`
  - current confirmed log signal:
    - `[qwen3-rc-caption] selective token tuning enabled for visual boundary tokens: tokens=('<vis_start>', '<vis_end>')`
  - current confirmed trainable params after the boundary-token change:
    - `57,975,296`
  - interpretation:
    - relative to the earlier `57,970,176`, the extra `5,120` params are exactly the two
      trainable boundary-token rows:
      - `2 x 2560`

Structured `Stage 2` offline evaluation script was then added:

- script:
  - `scripts/eval_qwen3_rc_dinov2_caption_structured.py`
- output metrics:
  - `scene_acc`
  - `grid_cell_acc`
  - `macro_f1`
  - `exact_match`
- additional outputs:
  - `metrics.json`
  - `predictions.jsonl`

Structured-eval smoke verification:

- job id:
  - `18097`
- status:
  - `COMPLETED`
- checkpoint evaluated:
  - `/file_storage01/home/mingli/data/outputs/stage2_caption_save_smoke_1gpu_20260412`
- eval output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_structured_eval_smoke_8_20260412`
- smoke eval sample count:
  - `8`
- smoke metrics:
  - `scene_acc = 0.1250`
  - `grid_cell_acc = 0.0254`
  - `macro_f1 = 0.0172`
  - `exact_match = 0.0000`
  - `parse_ok_rate = 0.0000`

Interpretation:

- the structured-eval tooling itself is now working end-to-end
- the evaluated checkpoint was only an early save-smoke checkpoint, so the above numbers should
  be treated as tooling verification, not as the target `Stage 2` quality bar

Current next-step queue after the latest updates:

1. wait for formal `Stage 2` job `18080` to finish under the save-fixed + boundary-token strategy
2. run `scripts/eval_qwen3_rc_dinov2_caption_structured.py` on the formal `val` split using the
   final `18080` output root
3. inspect `predictions.jsonl` with a small manual QA pass to see whether errors are mostly:
   - scene confusion
   - malformed outputs / parse failures
   - background overprediction
   - `lane_boundary` / `lane_divider` / `mix` collapse
4. if the latest `Stage 2` eval is basically acceptable and does not show catastrophic parse / collapse
   behavior, start `centerline SFT v1` immediately:
   - keep the aligned `DINOv2 -> visual_norm -> visual_projector -> geometric_position_mlp -> token_alignment`
     bridge
   - switch the centerline target to raw JSON generation with the native `Qwen` vocabulary
   - do **not** use discrete centerline structure tokens such as `<coord_pt>` / `<line>` / `<count_xx>`
   - do **not** use a continuous coordinate-regression head in `SFT v1`
   - enable `LoRA` on `Qwen`
   - use a long context budget directly:
     - `max_seq_length = 7168`
   - use black-background `BEV road-structure image` wording in prompts
   - do **not** mention `RC` in the prompt text
5. if formal `Stage 2` metrics are still weak, run a lighter bridge ablation first:
   - reduce `token_alignment_hidden_dim`
   - reduce `token_alignment_num_layers`
   - optionally reduce `visual_projector_hidden_dim`

### 7.26 Centerline `SFT v1` Decision (`2026-04-13`)

Once the current `Stage 2` alignment eval is no longer showing obvious collapse, the next
centerline route should be a deliberately simple `SFT v1` baseline:

- reuse the aligned `Stage 2` visual bridge
- keep the direct `<vis_patch>` replacement input path
- keep `DINOv2` frozen in the first `SFT v1` run
- train `Qwen` with `LoRA`
- continue training the shared bridge:
  - `visual_norm`
  - `visual_projector`
  - `geometric_position_mlp`
  - `token_alignment`
- keep `<vis_start>` / `<vis_end>` trainable
- keep `<vis_patch>` as a pure placeholder and do **not** train it

Accepted `SFT v1` output rule:

- predict raw JSON directly with the native `Qwen` vocabulary
- do **not** use custom discrete centerline tokens
- do **not** use the older continuous coordinate-regression head

Accepted `SFT v1` prompt wording rule:

- describe the input as a:
  - `black-background BEV road-structure image`
- do **not** use `RC` in prompt text

Accepted `SFT v1` prompt-constraint rule:

- the system prompt must strongly constrain JSON output
- require:
  - JSON only
  - no markdown fences
  - integer coordinates
  - coordinates in `[0, 512]` inclusive
  - exact schema family:
    - `{"lines":[]}`
    - `{"lines":[{"points":[[x1,y1],[x2,y2]]}]}`

Accepted `SFT v1` data / loss rule:

- reuse the existing centerline export root and raw assistant JSON targets
- keep assistant targets minified and deterministic
- compute language-model loss only on the assistant JSON span
- mask:
  - system prompt
  - user prompt
  - all visual placeholder tokens

Accepted `SFT v1` long-context rule:

- do not stay near the shorter `caption` cutoff lengths
- use:
  - `max_seq_length = 7168`

Formal execution spec:

- `docs/rc_centerline_sft_v1_json_lora_spec_20260413.md`

## 8. Fast Paths and Artifacts

### 8.1 Main Code Paths

- `unimapgen/models/qwen3_rc_centerline_16745style.py`
- `scripts/train_qwen3_rc_centerline_cnn_prefix_16745style.py`
- `scripts/eval_qwen3_rc_centerline_checkpoint_16745style.py`
- `unimapgen/data/rc_caption_short_dataset.py`
- `unimapgen/data/rc_semantic_align_dataset.py`
- `unimapgen/models/qwen3_rc_dinov2_clip_align.py`
- `scripts/export_rc_caption_short_dataset_view.py`
- `scripts/export_rc_semantic_align_dataset_view.py`
- `scripts/train_qwen3_rc_dinov2_clip_align.py`
- `scripts/eval_qwen3_rc_dinov2_clip_retrieval.py`
- `scripts/train_qwen3_rc_dinov2_caption_llava.py`
- `scripts/eval_qwen3_rc_dinov2_caption_structured.py`
- `scripts/predict_qwen3_rc_dinov2_caption_llava.py`
- `docs/rc_centerline_sft_v1_json_lora_spec_20260413.md`
- `scripts/train_rc_scene_autocorrector.py`
- `scripts/render_rc_caption_short_preview.py`
- `unimapgen/data/rc_centerline_cnn_prefix_dataset.py`
- `unimapgen/data/rc_structure_seg_dataset.py`
- `scripts/precompute_rc_structure_multiclass_masks.py`
- `unimapgen/models/rc_structure_seg.py`
- `scripts/train_rc_structure_seg.py`
- `scripts/render_rc_structure_seg_predictions.py`
- `stage1_rc_dinov2_clip_align_1gpu_smoke_20260412.sbatch`
- `stage1_rc_dinov2_clip_align_4gpu_clean_v1_20260412.sbatch`
- `stage1_rc_dinov2_clip_retrieval_eval_val_1gpu_20260412.sbatch`
- `stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412.sbatch`
- `docs/rc_handoff_20260408.md`
- `docs/rc_stage2_caption_execution_spec_20260412.md`

### 8.2 Main Data Paths

- new train root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
- full `caption_short` semantic-source root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411`
- split-10 manual scene-review root for the full `v3` train export:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411`
- `part_01` manual review file now containing the running reviewed labels used by the autocorrector:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411/part_01/manual_scene_review.jsonl`
- current clean Stage-1 semantic root:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412`
- current clean-root conflict pool:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412/review_conflicts`
- current clean-root summary:
  - `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412/summary.json`
- Stage-1 smoke output root:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_smoke_20260412`
- current Stage-1 formal 4-GPU output root:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412`
- current Stage-1 retrieval eval output root:
  - `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/retrieval_eval_val`
- current Stage-2 formal 4-GPU output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412`
- current active Stage-2 formal 1-epoch 4-GPU output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_4gpu_20260412`
- current active Stage-2 formal 1-epoch save-fixed 4-GPU output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412`
- Stage-2 checkpoint-save verification root:
  - `/file_storage01/home/mingli/data/outputs/stage2_caption_save_smoke_1gpu_20260412`
- Stage-2 structured-eval smoke output root:
  - `/file_storage01/home/mingli/data/outputs/stage2_structured_eval_smoke_8_20260412`
- current caption QA preview output:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_100_20260411`
- older `435`-row reviewed caption subset kept only as historical reference:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_trainonly500_sparse5_connectorveto_v3_20260411`
- previous sparse-filtered connector-veto `v2` root:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_trainonly500_sparse5_connectorveto_v2_20260411`
- previous sparse-filtered connector-veto `v2` preview:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_100_20260411`
- previous fixed-grid root before sparse5/connector-veto tightening:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_trainonly500_20260411`
- previous fixed-grid preview before sparse5/connector-veto tightening:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_grid8_trainonly500_100_20260411`
- latest smoke export root for the grid schema:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_grid8_smoke_train3_val2_20260411`
- previous rule-fixed sparse-filtered root kept for reference:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_rulefix_sparse3_20260410`
- previous cross-veto natural-language root kept for historical reference:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_rulefix_sparse3_crossveto_20260410`
- older probe-schema root kept only for debugging reference:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_sceneprobe500_20260410`
- older caption root kept only for historical reference and no longer recommended for training:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_20260410`
- updated `500`-panel `input -> caption` QA preview output:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_rulefix_sparse3_500_20260410`
- cross-veto `500`-panel `input -> caption` QA preview output:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_rulefix_sparse3_crossveto_500_20260410`
- cross-veto `complex`-only QA preview output:
  - `/file_storage01/home/mingli/data/outputs/rc_caption_short_preview_inputcaption_rulefix_sparse3_crossveto_complex_20260410`
- quick eval output:
  - `/file_storage01/home/mingli/data/outputs/eval_qwen3_rc_centerline_16745style_ckpt59500_newval_max400_lt10_viz100_4gpu_20260408`
- visual benchmark outputs worth remembering:
  - ResNet binary benchmark:
    - `17110`
  - DINOv2 binary benchmark:
    - `17111`
  - dash-vs-solid benchmark (metric-caution only):
    - `17249`
- corrected thin multiclass supervision now lives inside the canonical root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408/patches512_offset/seg_structure_multiclass`

### 8.3 Local Inspection Assets

- 20 sampled RC patches:
  - `artifacts/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408_samples20`
- 20 sampled RC patches with centerline GT overlay:
  - `artifacts/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408_samples20_overlay_centerline_gt_20260408`
- pulled eval visualizations:
  - `artifacts/eval_qwen3_rc_centerline_16745style_ckpt59500_newval_max400_lt10_viz100_4gpu_20260408/viz`
- pulled caption QA full set:
  - `tmp/caption_preview_review_20260410_full96`
- pulled caption QA representative subsets:
  - `tmp/caption_preview_review_20260410`
  - `tmp/caption_preview_review_20260410_extremes`
- pulled rule-fixed `500`-preview QA assets:
  - `tmp/caption_preview_inputcaption_rulefix_sparse3_500_20260410`
- pulled cross-veto `complex`-only QA assets:
  - `tmp/caption_preview_inputcaption_rulefix_sparse3_crossveto_complex_20260410`
- pulled fixed-grid `100`-preview QA assets:
  - `tmp/rc_caption_short_preview_inputcaption_grid8_trainonly500_100_20260411`
- pulled fixed-grid curated `16`-panel QA subset:
  - `tmp/caption_preview_inputcaption_grid8_trainonly500_selected16_20260411`
- pulled sparse-filtered connector-veto `v2` full preview:
  - `tmp/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_100_20260411`
- pulled sparse-filtered connector-veto `v2` straight-focused review subset:
  - `tmp/caption_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v2_straight_review_20260411`
- pulled sparse-filtered connector-veto `v3` random-seed full preview:
  - `tmp/rc_caption_short_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_seed314159_100_20260411`
- pulled sparse-filtered connector-veto `v3` random-seed curated QA subset:
  - `tmp/caption_preview_inputcaption_grid8_trainonly500_sparse5_connectorveto_v3_seed314159_selected11_20260411`
- pulled full-root sparse-filtered connector-veto `v3` random `100` scene-QA preview:
  - `tmp/rc_caption_short_preview_inputcaption_grid8_full_sparse5_connectorveto_v3_seed20260411_100_20260411`
- pulled `part_01` first `100` manual-scene-review preview:
  - `tmp/rc_caption_short_preview_inputcaption_part01_first100_20260411`
- local written first-`100` manual-scene-review labels:
  - `tmp/rc_caption_short_preview_inputcaption_part01_first100_20260411/manual_scene_labels_first100.jsonl`
- local written first-`100` manual-scene-review summary:
  - `tmp/rc_caption_short_preview_inputcaption_part01_first100_20260411/manual_scene_labels_first100_summary.md`

### 8.4 Remote Access Note

If plain `ssh` login fails on this Windows workstation, do not assume the remote
server is down.

Check:

- `docs/handoff/04_servers_and_environment.md`

Current known-good reminder:

- host: `172.29.15.117`
- user: `mingli`
- use PowerShell `Posh-SSH` fallback on this machine if regular key-based login is failing
- local helper for handoff-doc sync:
  - `scripts/sync_rc_handoff_doc.ps1`
- root cause of the common Windows-side failure:
  - plain `ssh -o BatchMode=yes` can fail here with `Permission denied (publickey,password)` because `~/.ssh/config` exists but there is no usable private key for `mingli@172.29.15.117`
- verified recovery path:
  - first run `Test-NetConnection 172.29.15.117 -Port 22`
  - if the port is reachable, switch to PowerShell `Posh-SSH` with runtime `PSCredential`
  - preferred read-only check:
    - `hostname; pwd; squeue -u mingli -o '%.18i %.20j %.8T %.10M %.9P %.20R'`
  - preferred sync path:
    - run `.\scripts\sync_rc_handoff_doc.ps1`
    - if `-Password` is omitted, the script now prompts interactively instead of requiring a plaintext password on the command line
  - `2026-04-11` note:
    - this recovery path was re-verified successfully on the current Windows workspace
    - the sync helper also needed a small fix in its remote hash check; upload itself had already been working
- canonical login-recovery reference:
  - `docs/handoff/04_servers_and_environment.md`
- if PowerShell policy blocks module loading, use process-scoped execution-policy bypass only for the current shell session
- do not hardcode secrets into the handoff doc

### 8.5 Latest SFT-v1 Inference Viz

As of `2026-04-13`, there is now a dedicated JSON-SFT inference path for the
current centerline `SFT v1` line.

New files:

- `scripts/predict_qwen3_rc_dinov2_centerline_json_sft.py`
- `stage3_rc_dinov2_centerline_json_sft_latest100_viz_1gpu_20260413.sbatch`

What this path does:

- resolves the latest `checkpoint-*` under the active `SFT v1` run root
- rebuilds the `Qwen3 + DINOv2 + bridge + LoRA` model from `args.json`
- loads the intermediate Trainer checkpoint `pytorch_model.bin`
- runs autoregressive raw-JSON generation on the selected split
- writes:
  - `predictions.jsonl`
  - `summary.json`
- then renders side-by-side GT-vs-Pred visualization panels through:
  - `scripts/render_eval_predictions_jsonl.py`

Important implementation notes:

- intermediate `checkpoint-*` dirs from the current `SFT v1` run are **not**
  the final `rc_dinov2_centerline_json_modules.pt` export format
- the inference script therefore reconstructs the model from the saved run args
  and loads the full Trainer `state_dict` from `checkpoint-*/pytorch_model.bin`
- generation now has an early-stop rule:
  - if a complete valid JSON object has already been produced, stop decoding
    even before `eos`
- `predictions.jsonl` is line-buffered and flushes after every sample, so
  partial progress can be inspected while the job is still running

Current visualization run:

- first attempt:
  - job `18327`
  - functionally okay, but canceled and replaced
- current active run:
  - job `18329`
  - launcher:
    - `stage3_rc_dinov2_centerline_json_sft_latest100_viz_1gpu_20260413.sbatch`
  - run root:
    - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413`
  - resolved latest checkpoint at launch:
    - `checkpoint-9000`
  - current viz output root:
    - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413/inference_viz_checkpoint-9000_val100_20260413`

Observed status at the latest check:

- `18329` is `RUNNING`
- `predictions.jsonl` is already being written incrementally
- latest observed partial count:
  - `20 / 100`
- panels are rendered only after prediction finishes, so the `panels/` dir may
  exist before images are populated

Immediate next step for this line:

1. let `18329` finish the full `100`-sample prediction pass
2. inspect `summary.json`
3. review the rendered `panels/`
4. if needed, pull a local subset for manual QA

### 8.x Minimal `rc-llm` Branch Snapshot (`2026-04-13`)

Latest packaging work completed for the current RC LLM mainline:

- the code was reduced to a minimal runnable RC-only branch containing only:
  - `Stage A` structure segmentation
  - `Stage 1` coarse alignment
  - `Stage 2` caption alignment
  - `Stage 3` JSON + LoRA centerline SFT v1
- local minimal snapshot:
  - `C:\Users\Administrator\Desktop\UniMapGen_rc_llm_minimal_20260413`
- current minimal snapshot layout:
  - `train_required/`
  - `eval_optional/`
  - `dataset_examples/`
  - `unimapgen/`
  - `rc_llm_minimal_runnable_usage.md`
  - `rc_llm_scripts_overview.md`
- `dataset_examples/` was added so each retained dataset now has a sample JSON bundle and field-level explanation
- `rc_llm_scripts_overview.md` was added to explain the purpose of each retained training / eval script and the key shared modules
- important logic in the retained training / eval entry scripts and key data/model modules now has Chinese comments added
- the current standalone minimal sync commit on the remote packaging repo is:
  - `c8332cfbcba29b3879bb660b3a2214d6697ea5c8`
- current GitHub minimal branch name remains:
  - `rc-llm`

Important meaning:

- if someone wants the smallest current RC LLM runnable package, use the local snapshot above
- if someone wants the full historical experimentation workspace, stay in the main `UniMapGen` tree instead
- when updating the minimal branch in the future, keep the local minimal snapshot and GitHub `rc-llm` branch aligned in the same work turn

### 8.y Current Prompt State for `Stage 3`

The current accepted `SFT v1` prompt is already landed in:

- `unimapgen/data/rc_centerline_json_sft_dataset.py`

Current system prompt key points:

- describe the input as a `black-background BEV road-structure image`
- explicitly state visible semantics:
  - `lane_boundary`
  - `lane_divider`
  - `background`
- explicitly define what a centerline means
- explicitly forbid tracing `lane_boundary` / `lane_divider` themselves
- explicitly require separate continuous polylines for different lanes / branches
- explicitly require border truncation at the visible patch border
- constrain output to raw JSON only
- constrain coordinates to integer `0..512`

Current user prompt key points:

- keep it short
- mention:
  - `black-background BEV road-structure image`
  - infer centerlines from visible `lane_boundary` and `lane_divider`
  - return raw JSON only

### 8.z Proposed New `Stage 3` 4-GPU Run (Prompt-Updated, Same Architecture)

If starting a fresh 4-GPU run from the current accepted prompt, the recommended configuration is:

- base model:
  - `Qwen3-4B`
- visual encoder:
  - `DINOv2-large`
- visual encoder init:
  - `/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/latest.pt`
- bridge init:
  - `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/rc_dinov2_caption_modules.pt`
- data root:
  - `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`
- model route:
  - `RC patch 512 -> center-pad 518 -> DINOv2 patch tokens -> visual_norm -> visual_projector -> geometric_position_mlp -> token_alignment -> direct <vis_patch> replacement -> Qwen + LoRA -> raw JSON`
- freezing rule:
  - freeze `DINOv2`
  - keep bridge trainable
  - keep `<vis_start>` / `<vis_end>` trainable
  - do not train `<vis_patch>`
  - train `Qwen` through `LoRA`
- LoRA target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- LoRA config:
  - `rank = 32`
  - `alpha = 64`
  - `dropout = 0.05`
- sequence length:
  - `cutoff_len = 7168`
- image config:
  - `image_size = 512`
  - `encoder_input_pad_size = 518`
  - `num_visual_tokens = 1369`
- bridge config:
  - `visual_projector_hidden_dim = 4096`
  - `geometric_mlp_hidden_dim = 512`
  - `token_alignment_hidden_dim = 4096`
  - `token_alignment_num_layers = 2`
  - `token_alignment_dropout = 0.0`
- optimizer / schedule:
  - `lr = 5e-5`
  - `weight_decay = 0.0`
  - `warmup_ratio = 0.03`
- training precision:
  - `bf16`
  - `gradient_checkpointing = True`
- recommended 4-GPU batch recipe:
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 2`
  - effective global batch = `8`
- recommended training length:
  - `num_train_epochs = 6`
- save / eval:
  - `save_strategy = steps`
  - `save_steps = 1000`
  - `evaluation_strategy = steps`
  - `eval_steps = 1000`

Important meaning:

- the current old 4-GPU launcher still uses `gradient_accumulation_steps = 1`, which gives effective global batch `4`
- for a fairer comparison against the current 8-GPU mainline, the new 4-GPU prompt-updated run should use `gradient_accumulation_steps = 2`
- if a new 4-GPU run is opened, use a fresh output root and do not overwrite the current 8-GPU line

Actual new prompt-updated 4-GPU run opened on `2026-04-14`:

- launcher:
  - `stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_4gpu_20260414.sbatch`
- job id:
  - `18406`
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414`
- effective global batch:
  - `4 GPUs x per_device_bs1 x grad_accum2 = 8`
- this launcher embeds the full current accepted prompt explicitly via `--system-prompt` and `--user-prompt`
- this was necessary because the remote repo copy had not yet fully picked up the latest prompt text by default
- startup status:
  - passed `py_compile`
  - passed model / bridge / DINO load
  - reached real training steps
  - first observed logged loss:
    - `2.14`
- current observed node at startup:
  - `gpu12`

### 8.za `18254` Latest Checkpoint Inference-Visualization Follow-up

The historical `Stage 3` 8-GPU run tied to job `18254` must be interpreted carefully:

- job id:
  - `18254`
- launcher family:
  - `stage3_rc_dinov2_centerline_json_sft_lora_8gpu_20260413.sbatch`
- Slurm final state:
  - `CANCELLED`
- but the actually used output root for that run is:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413`
- the latest preserved checkpoint under that root is:
  - `checkpoint-22000`
- earlier preserved checkpoint under that same root:
  - `checkpoint-21000`

Important meaning:

- when the user says "use the latest checkpoint from `18254`", the correct target is currently:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413/checkpoint-22000`
- do **not** accidentally switch to the older partial root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_8gpu_20260413`
  - that older partial root only has:
    - `checkpoint-1000`

Actual inference-visualization follow-up opened on `2026-04-14`:

- visualization launcher:
  - `stage3_rc_dinov2_centerline_json_sft_latest100_viz_1gpu_20260413.sbatch`
- visualization job id:
  - `18636`
- explicit exported `RUN_ROOT`:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413`
- resolved checkpoint at startup:
  - `checkpoint-22000`
- output dir:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_20260413/inference_viz_checkpoint-22000_val100_20260413`
- split:
  - `val`
- requested sample count:
  - `100`

Observed startup status for `18636`:

- passed launcher startup
- loaded `DINOv2` backbone successfully
- loaded RC structure encoder checkpoint successfully
- loaded `visual_norm`, `visual_projector`, `geometric_position_mlp`, and `token_alignment` successfully
- began real JSON generation on `gpu12`

Observed progress snapshot when last checked on `2026-04-14`:

- job state:
  - `RUNNING`
- current written prediction rows:
  - `8 / 100`
- current rendered panels:
  - `0`
- current reason panels are still `0`:
  - this launcher writes `predictions.jsonl` during generation first
  - it renders the visualization panels only after the prediction phase completes

Operational reminder:

- if continuing this follow-up, always check both:
  - `predictions.jsonl` line count
  - `panels/` file count
- do not assume panel export has failed just because `predictions.jsonl` already exists while `panels/` is still empty

### 8.zb `18406` Prompt-v2 Checkpoint-21000 50-Sample Visualization

Because training job `18406` was still running on `2026-04-14`, a fixed-checkpoint visualization launcher was added so the exact evaluated weight would stay stable during inference:

- source training job:
  - `18406`
- source training root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414`
- latest saved checkpoint when this visualization was opened:
  - `checkpoint-21000`
- fixed-checkpoint visualization launcher:
  - `stage3_rc_dinov2_centerline_json_sft_promptv2_ckpt21000_viz50_1gpu_20260414.sbatch`
- visualization job id:
  - `18662`
- visualization output dir:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-21000_val50_promptv2_20260414`

Completed status:

- Slurm state:
  - `COMPLETED`
- sample count:
  - `50`
- rendered panel count:
  - `50`
- parse success:
  - `50 / 50`
- parse success rate:
  - `1.0`
- average GT line count:
  - `8.3`
- average predicted line count:
  - `6.58`

Important meaning:

- this visualization is tied to the prompt-v2 4-GPU line, not the older `18254` 8-GPU line
- this run uses a fixed explicit checkpoint path:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/checkpoint-21000`
- if someone later wants "the newest prompt-v2 checkpoint", they should re-check the training root again instead of assuming `21000` is still latest

### 8.zc `18406` Prompt-v2 Newer Checkpoint-22000 50-Sample Visualization

After the prompt-v2 training line kept running, a newer checkpoint became available and was re-visualized with the same fixed-50 validation setup:

- source training job:
  - `18406`
- source training root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414`
- newer latest checkpoint at re-check time:
  - `checkpoint-22000`
- re-visualization job id:
  - `18680`
- re-visualization output dir:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-22000_val50_promptv2_rerun_20260414`

Completed status:

- Slurm state:
  - `COMPLETED`
- sample count:
  - `50`
- rendered panel count:
  - `50`
- parse success:
  - `49 / 50`
- parse success rate:
  - `0.98`
- average GT line count:
  - `8.3`
- average predicted line count:
  - `6.12`

Important meaning:

- this is the newer prompt-v2 visualization result relative to the earlier `checkpoint-21000` run
- because `avg_pred_lines` dropped from `6.58` to `6.12` while `avg_gt_lines` stayed `8.3`, this rerun should be manually spot-checked for under-prediction or increased empty / truncated outputs
- if comparing prompt-v2 checkpoints manually, prefer looking at:
  - `checkpoint-21000` 50-sample panels
  - `checkpoint-22000` 50-sample panels
  side by side on the same sampled IDs

### 8.zd Historical Qwen3-VL JSONNum Full-Train Root Cleanup

On `2026-04-15`, the old historical full-train Qwen3-VL JSON-numeric training output root was manually cleaned to reduce storage:

- cleaned root:
  - `/file_storage01/home/mingli/data/outputs/llamafactory_qwen3_vl_8b_paper16_patch_only_full_trainval_geomdedup_jsonnum36_full_8gpu_zero2_r15_20260412`
- this root is a training-output root, not a dataset root
- the actual dataset-export root with a similar name is:
  - `/file_storage01/home/mingli/data/paper16_patch_only_full_trainval_geomdedup_system_jsonnum36_qwen3vl_r1_20260411`
  - do **not** confuse the two

Deleted checkpoints:

- `checkpoint-16000`
- `checkpoint-17000`
- `checkpoint-18000`
- `checkpoint-19000`
- `checkpoint-20000`
- `checkpoint-21000`
- `checkpoint-22000`

Retained items:

- `checkpoint-22128`
- root-level final model:
  - `model.safetensors`
- root-level config / tokenizer / trainer metadata files

Observed size change:

- before cleanup:
  - `1.1T`
- after cleanup:
  - `143G`

Important meaning:

- this historical root no longer contains the intermediate checkpoints listed above
- if someone needs to resume or inspect that old line, they should use:
  - `checkpoint-22128`
  - or the root-level `model.safetensors`

### 8.ze 18406 Timeout Cause and Epoch-3 Resume

The prompt-v2 Stage-3 formal 4-GPU training job `18406` did **not** fail due to a model or data error.

Observed stop reason:

- `sacct -j 18406` shows:
  - `TIMEOUT`
- the original launcher had:
  - `#SBATCH -t 24:00:00`

Observed training progress at stop:

- retained checkpoints in:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414`
  - `checkpoint-26000`
  - `checkpoint-27000`
- tail logs from job `18406` show training had already advanced into:
  - `epoch ~= 2.06`

Follow-up action taken on `2026-04-15`:

- updated the formal launcher
  - `/file_storage01/home/mingli/project/jn/UniMapGen/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_4gpu_20260414.sbatch`
  - to support optional:
    - `RESUME_FROM_CHECKPOINT`
- added a dedicated resume wrapper:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_resume_epoch3_4gpu_20260415.sbatch`
- submitted resume job:
  - `18861`

Resume target:

- resume from:
  - `checkpoint-27000`
- continue only until:
  - `num_train_epochs = 3`

Important meaning:

- `18406` should be interpreted as a healthy long run that simply hit wall-clock limit
- the current continuation line for this prompt-v2 Stage-3 root is:
  - job `18861`

### 8.zf Checkpoint-27000 Partial Visualization Recovery from Existing 34 Predictions

The first checkpoint-`27000` visualization job:

- job id:
  - `18850`
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-27000_val50_promptv2_20260415`

Observed final state:

- `18850` was cancelled while the team prioritized GPU for the Stage-3 resume run
- this left:
  - `predictions.jsonl` with `34` rows
  - no `summary.json`
  - no rendered `panels/`

Recovery action taken on `2026-04-15`:

- reused the existing partial predictions file:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-27000_val50_promptv2_20260415/predictions.jsonl`
- ran the render-only script:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/scripts/render_eval_predictions_jsonl.py`
- wrote recovered partial panels to:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-27000_val34_partial_panels_20260415`
- packed archive:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-27000_val34_partial_panels_20260415.tar.gz`

Recovered artifact count:

- rendered panels:
  - `34`

Local download location used in this session:

- extracted folder:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt27000_val34_partial_20260415\inference_viz_checkpoint-27000_val34_partial_panels_20260415`
- downloaded archive:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt27000_val34_partial_20260415\inference_viz_checkpoint-27000_val34_partial_panels_20260415.tar.gz`

### 8.zg Checkpoint-27000 Val100 Visualization Submission

To get a broader manual read on the current prompt-v2 Stage-3 line, a fixed-checkpoint `100`-sample visualization was submitted on `2026-04-15`.

Submission choice:

- use the latest saved checkpoint available at submission time:
  - `checkpoint-27000`
- keep the checkpoint fixed, rather than following the still-running resume line, to avoid checkpoint drift during inference

Submitted job:

- job id:
  - `18867`
- launcher reused:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/stage3_rc_dinov2_centerline_json_sft_promptv2_ckpt21000_viz50_1gpu_20260414.sbatch`
- overridden key runtime args:
  - `CHECKPOINT_DIR=/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/checkpoint-27000`
  - `MAX_SAMPLES=100`

Expected output root:

- `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/inference_viz_checkpoint-27000_val100_promptv2_20260415`

Initial observed state right after submission:

- `squeue -j 18867`:
  - `RUNNING`
- output directory already created:
  - yes

### 8.zh Direct Qwen3-VL RC Centerline JSON Baseline (Visual Tower Trainable)

On `2026-04-15`, a separate native-vision baseline was formalized for direct
comparison against the current `DINOv2 -> Qwen` Stage-3 line.

Purpose:

- test whether native `Qwen3-VL` image understanding alone can already learn RC
  centerline reconstruction
- keep this as a parallel baseline, not a replacement for the current main line

Accepted design summary:

- input route:
  - native `Qwen3-VL` image input
- output route:
  - raw centerline JSON with the native `Qwen` vocabulary
- do **not** use:
  - `DINOv2`
  - the current bridge:
    - `visual_norm`
    - `visual_projector`
    - `geometric_position_mlp`
    - `token_alignment`
  - `<vis_patch>` replacement
  - discrete centerline output tokens
  - continuous coordinate heads

Current first-run training rule:

- model:
  - `Qwen3-VL-8B-Instruct`
- language path:
  - `LoRA`
- visual tower:
  - **trainable**
- multi-modal merger:
  - **trainable**
- coordinates:
  - still patch-local `0..512`

New formal spec:

- local:
  - `C:\Users\Administrator\Desktop\UniMapGen\docs\rc_qwen3vl_direct_centerline_json_spec_20260415.md`
- remote target path:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/docs/rc_qwen3vl_direct_centerline_json_spec_20260415.md`

New training script:

- local:
  - `C:\Users\Administrator\Desktop\UniMapGen\scripts\train_qwen3_vl_direct_rc_centerline_json.py`
- remote target path:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/scripts/train_qwen3_vl_direct_rc_centerline_json.py`

New formal launcher:

- local:
  - `C:\Users\Administrator\Desktop\UniMapGen\stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_8gpu_20260415.sbatch`
- remote target path:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_8gpu_20260415.sbatch`

4-GPU formal launcher added in the same session:

- local:
  - `C:\Users\Administrator\Desktop\UniMapGen\stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_4gpu_20260415.sbatch`
- remote target path:
  - `/file_storage01/home/mingli/project/jn/UniMapGen/stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_4gpu_20260415.sbatch`
- main difference:
  - `4 GPU`
  - `gradient_accumulation_steps = 2`
- intended meaning:
  - keep the effective global batch close to the earlier 8-GPU setting while
    reducing accelerator demand

Launcher defaults:

- GPUs:
  - `8`
- epochs:
  - `3`
- lr:
  - `2e-5`
- cutoff:
  - `8192`
- LoRA:
  - `r=16`
  - `alpha=32`
  - `dropout=0.05`
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_8gpu_20260415`

4-GPU run that was actually submitted first:

- submitted launcher:
  - `stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_4gpu_20260415.sbatch`
- output root:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_qwen3vl_direct_centerline_json_lora_vttrain_4gpu_20260415`
- effective batch choice:
  - `4 GPU`
  - `gradient_accumulation_steps = 2`
  - global batch remains `8`

First 4-GPU submission history:

- first submit:
  - job `18874`
- observed stop cause:
  - missing remote dependency file:
    - `/file_storage01/home/mingli/project/jn/UniMapGen/unimapgen/rc_llm_runtime.py`
- fix applied:
  - synced the missing runtime file to the remote repo
  - re-ran remote import check successfully

Successful 4-GPU retried run:

- active job:
  - `18876`
- observed state after retry:
  - `RUNNING`
- initial model printout confirms:
  - visual tower = trainable
  - merger = trainable
  - `LoRA target_modules = all-linear`
- observed trainable parameter count:
  - `628,882,160`
- observed total parameter count:
  - `8,817,430,256`

Observed first loss prints from job `18876`:

- `loss = 0.6789`, `epoch = 0.0007355`
- `loss = 0.6954`, `epoch = 0.001471`
- `loss = 0.6245`, `epoch = 0.002207`
- `loss = 0.6980`, `epoch = 0.002942`

Important current note:

- training is running and has already entered the real optimization loop
- stderr currently shows repeated processor warnings:
  - `Kwargs passed to processor.__call__ have to be in processor_kwargs dict, not in **kwargs`
- these warnings did **not** block training startup; loss is printing normally

### 8.zi Prompt-v2 Checkpoint-33000 Partial-71 Manual QA

On `2026-04-15`, a partial inference snapshot was taken from the still-running 100-sample visualization job on the current prompt-v2 Stage-3 line:

- source job:
  - `18955`
- source checkpoint:
  - `/file_storage01/home/mingli/data/outputs/stage3_rc_dinov2_centerline_json_sft_lora_promptv2_ga2_6epoch_4gpu_20260414/checkpoint-33000`
- partial rendered sample count:
  - `71`
- local review root:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415`

Manual QA summary:

- simple straight / gently curved / regular multi-lane corridor patches are already mostly good
- border truncation behavior is usually reasonable in simple scenes
- lane count is often correct on clean parallel corridors
- the main remaining failure mode is no longer "cannot draw anything"; it is now "draws extra or topologically wrong centerlines in complex scenes"

Observed positive examples:

- good simple corridor alignment:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\Oe1cbEQhfk2js42iBwTx2QRb23M30Rev__Spring_2020__ox256oy256__r04c04.png`
- good sparse single-corridor behavior:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\qktHtGkRwIYuPrYh89EkSG8rslgdq0Bs__Summer_2020__ox256oy256__r03c05.png`
- good cross-like simple topology:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\9Y4wSZrjgU4CbBwXiC5qDT11Ia2M9MN2__Spring_2020__ox000oy000__r04c03.png`

Observed failure examples:

- connector-heavy diagonal corridor with an unsupported long diagonal hallucination:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\2E8su1EVHLnvOShfJ4HrSSxwH46hNd3E__Summer_2020__ox000oy000__r06c02.png`
- intersection / merge scene with wrong branch continuation and extra crossing lines:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\MM9h1HHQcpGC93cmb8HxEOmfVNAuD3Or__Spring_2020__ox256oy000__r01c03.png`
- side-branch scene with a clear false crossing centerline:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\WY0cVNmhg7LtAs5Eny78Csltv2tbjdsd__Winter_2021__ox256oy256__r03c02.png`
- junction scene where the model collapses topology and draws stray bridge-like lines:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\WYLGLyvfbcBy9e60BRVy80iEVkWooGC6__Summer_2020__ox000oy000__r05c05.png`
- curved / intersecting scene with red branch overshoot:
  - `C:\Users\Administrator\Desktop\UniMapGen\artifacts\stage3_rc_promptv2_ckpt33000_val71_partial_20260415\inference_viz_checkpoint-33000_val71_partial_panels_20260415\jFRjEux1xWEA5iGGplN3oIGeZ2YpVuki__Spring_2020__ox000oy256__r04c05.png`

Failure modes to keep in mind:

- false connector bridges across unrelated corridors
- wrong continuation choice at junctions / merges / splits
- over-prediction of lane count in connector-heavy scenes
- occasional endpoint overshoot or unsupported extension near crop borders
- complex topology is still much weaker than clean straight-corridor geometry

Recommended next optimization order:

1. keep the current prompt-v2 line as the main DINO->Qwen route for now, because simple-scene geometry is already usable
2. add a hard-case replay set for Stage-3 SFT using the failure types above:
   - connector-heavy
   - merge / split
   - T-junction
   - crossing-like local patches
3. strengthen prompt/task constraints so the model is explicitly told:
   - do not invent centerlines without visible corridor support
   - do not connect two branches unless the visible structure clearly supports a valid drivable continuation
   - terminate unsupported lines instead of bridging across empty space
4. for inference, prefer deterministic decoding:
   - greedy or very low-temperature decoding
   - avoid sampling noise on topology-sensitive JSON output
5. add a lightweight geometry sanity filter after decoding:
   - reject or trim lines that cross visible boundaries or jump across unrelated corridors
6. make the next manual QA focused, not random:
   - sample specifically from connector-heavy / junction-rich patches rather than mostly straight corridors

Current judgment:

- this checkpoint is promising for simple corridor reconstruction
- it is **not** yet robust enough to be treated as a topology-safe centerline predictor on complex local road structures
- the next gain is more likely to come from hard-case data / prompt / decoding control than from only training longer on the same easy-heavy distribution

### 8.zj Minimal Continuous-Head Code Drop

On `2026-04-15`, a minimal continuous coordinate-regression code path was organized on top of the current `DINOv2 -> Qwen3-4B` RC line.

Goal:

- keep only the smallest code set needed for a continuous coordinate head
- avoid bringing back the older patch-only / Qwen2.5-VL historical training stack
- stay compatible with the current Stage-3 DINO bridge structure

Minimal files added:

- `unimapgen/data/rc_centerline_continuous_head_dataset.py`
  - converts normal centerline JSON into a placeholder JSON text
  - uses one literal placeholder token per point:
    - `<coord_pt>`
  - aligns each placeholder token with a normalized continuous `(x, y)` target
- `unimapgen/models/qwen3_rc_dinov2_centerline_continuous_head.py`
  - extends the current `Qwen3RCDinoCenterlineJSONSFTModel`
  - adds a minimal `coord_head: hidden -> 2`
  - keeps the existing visual injection path unchanged
- `scripts/train_qwen3_rc_dinov2_centerline_continuous_head.py`
  - adds a minimal custom trainer
  - computes:
    - base LM CE loss
    - `SmoothL1` continuous coordinate loss on `<coord_pt>` positions
  - logs:
    - `coord_reg_loss`
    - `coord_reg_mae`

Current design choice:

- assistant target text is still JSON-shaped
- but concrete numeric coordinates are replaced by placeholder point tokens
- the continuous head regresses the real `(x, y)` values for those placeholder positions

This code drop is currently intended as a clean minimal branch artifact, not yet as the active default training line.

## 9. Prompt for the Next Agent

Use this prompt to continue RC work quickly:

```text
Please first read docs/rc_handoff_20260408.md carefully and treat it as the current RC handoff baseline.

Then note the current priority split:
- the old RC/Qwen end-to-end line is still the historical baseline
- the pure visual encoder benchmark has already produced enough signal to guide the next step
- the current active implementation line is the two-stage DINOv2 -> Qwen alignment route

Keep these references in mind:
- main dataset root: /file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408
- full caption semantic-source root: /file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411
- split review root: /file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_split10_scene_review_20260411
- current clean Stage-1 semantic root: /file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412
- Stage-1 smoke output root: /file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_smoke_20260412
- Stage-1 successful smoke job: 18000
- current Stage-1 formal job: 18001
- current Stage-1 formal output root: /file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412
- current Stage-1 retrieval eval job: 18031
- current Stage-1 retrieval metrics file: /file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/retrieval_eval_val/metrics.json
- current Stage-2 smoke job: 18036
- cancelled Stage-2 formal 3-epoch job: 18037
- failed Stage-2 formal 1-epoch pre-fixsave job: 18038
- failed Stage-2 formal 1-epoch pre-fixsave output root: /file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_4gpu_20260412
- Stage-2 checkpoint-save verification job: 18077
- current active Stage-2 formal 1-epoch save-fixed job: 18080
- Stage-2 boundary-token strategy smoke job: 18088
- Stage-2 structured-eval smoke job: 18097
- next centerline follow-up route once Stage-2 eval is acceptable:
  - `SFT v1 = aligned DINO bridge + Qwen LoRA + raw JSON output + max_seq_length 7168`
- current active Stage-2 formal save-fixed output root: /file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412
- main training job: 16982 (clean 16745-style single-coord-head run)
- reference quick eval: 16965 and its output dir /file_storage01/home/mingli/data/outputs/eval_qwen3_rc_centerline_16745style_ckpt59500_newval_max400_lt10_viz100_4gpu_20260408
- ResNet visual benchmark reference: 17110
- DINOv2 visual benchmark reference: 17111

Important constraints:
- do not treat 16949 multicoord/query-heatmap as the canonical baseline
- do not revert back to the broken fake multi-level coord-loss logging route
- do not use the older 2026-04-05 / 2026-04-06 RC handoff docs as the primary execution guide
- do not describe the RC input as if visible centerlines are directly present in the image
- do not use `front/rear`; stay with `top/bottom/left/right`
- do not claim that dash-vs-solid metrics are directly comparable with the earlier binary structure benchmark
- do not continue with `scene_probe_states_v1` or sparse random-probe captions as the default alignment target
- do not use Stage-1 batch size `1` for the CLIP-style contrastive smoke, because it yields a degenerate `1 x 1` zero-loss setup

Your first tasks should be:
1. treat `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412` as the current active Stage-2 formal output root and monitor job `18080`
2. after `18080` completes, run `scripts/eval_qwen3_rc_dinov2_caption_structured.py` on the formal `val` split and record:
   - `scene_acc`
   - `grid_cell_acc`
   - `macro_f1`
   - `exact_match`
3. inspect the resulting `predictions.jsonl` with a small manual QA pass to determine whether failures are mostly:
   - malformed outputs / parse failures
   - scene confusion
   - background overprediction
   - `lane_boundary` / `lane_divider` / `mix` collapse
4. keep the current accepted Stage-2 token-freeze rule:
   - freeze the `Qwen` backbone
   - freeze `DINOv2`
   - do **not** train `<vis_patch>`
   - train only `<vis_start>` and `<vis_end>` token rows
5. if the latest `Stage 2` eval looks basically acceptable, the next route should be `centerline SFT v1`:
   - use the aligned `DINOv2` bridge as the visual front-end
   - keep direct `<vis_patch>` replacement
   - train `Qwen` with `LoRA`
   - predict raw centerline JSON directly with the native `Qwen` vocabulary
   - do **not** use discrete centerline tokens
   - do **not** use a continuous coordinate-regression head
   - keep prompt wording as:
     - `black-background BEV road-structure image`
   - do **not** mention `RC` in the prompt text
   - use:
     - `max_seq_length = 7168`
   - follow:
     - `docs/rc_centerline_sft_v1_json_lora_spec_20260413.md`
6. if Stage-2 formal metrics remain weak, run a lighter bridge ablation before changing the core schema:
   - reduce `token_alignment_hidden_dim`
   - reduce `token_alignment_num_layers`
   - optionally reduce `visual_projector_hidden_dim`
7. only return to unresolved semantic-label review if Stage-1 semantic coverage becomes the actual blocker again
```
