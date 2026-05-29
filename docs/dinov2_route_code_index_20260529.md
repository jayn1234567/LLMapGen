# DINOv2 Route Code Index 2026-05-29

This branch is the curated code map for the earlier DINOv2 route.  It matches
the project write-up that describes the DINOv2 visual encoder, map
serialization, Douglas-sampled centerline targets, and StageB continuation.

## Route Summary

The DINOv2 route keeps a separate visual feature path and uses Qwen as the
language/JSON decoder.

1. Stage 1 aligns frozen or lightly trainable DINOv2 ViT-L/14 visual tokens
   with Qwen-side text semantics.
2. Stage 2 uses a bridge to inject DINOv2 visual tokens into Qwen and predicts
   structured scene/caption/grid state.
3. Stage 3 predicts road structures or centerlines as JSON.
4. Douglas simplification converts dense centerline ground truth into compact
   polyline targets.
5. StageB adds previous-patch state so the model can continue lines across
   patch boundaries.

## Core Model Files

- `unimapgen/models/qwen3_rc_dinov2_clip_align.py`
- `unimapgen/models/qwen3_rc_dinov2_caption_bridgev2.py`
- `unimapgen/models/qwen3_rc_dinov2_caption_bridgev2_segaux.py`
- `unimapgen/models/qwen3_rc_dinov2_caption_llava.py`
- `unimapgen/models/qwen3_rc_dinov2_centerline_json_sft.py`
- `unimapgen/models/qwen3_rc_dinov2_centerline_continuous_head.py`
- `unimapgen/models/qwen3_rc_dinov2_road_structure_json_sft.py`
- `unimapgen/models/qwen3_rc_dinov2_road_structure_bezier4_json_sft.py`
- `unimapgen/models/rc_structure_seg.py`

## Dataset And Target Files

- `unimapgen/data/rc_semantic_align_dataset.py`
- `unimapgen/data/rc_caption_short_dataset.py`
- `unimapgen/data/rc_caption_short_segaux_dataset.py`
- `unimapgen/data/rc_centerline_json_sft_dataset.py`
- `unimapgen/data/rc_centerline_adaptive_keypoint_json_sft_dataset.py`
- `unimapgen/data/rc_centerline_startend_json_sft_dataset.py`
- `unimapgen/data/rc_centerline_startend_continuous_head_dataset.py`
- `unimapgen/data/rc_centerline_douglas_utils.py`
- `unimapgen/data/rc_road_structure_json_sft_dataset.py`
- `unimapgen/data/rc_road_structure_bezier4_json_sft_dataset.py`
- `unimapgen/data/rc_road_structure_quad4_json_sft_dataset.py`
- `unimapgen/data/rc_structure_seg_dataset.py`

## Stage 1: DINOv2 Alignment

Training/eval entry points:

- `scripts/train_qwen3_rc_dinov2_clip_align.py`
- `scripts/eval_qwen3_rc_dinov2_clip_retrieval.py`
- `stage1_rc_dinov2_clip_align_1gpu_smoke_20260412.sbatch`
- `stage1_rc_dinov2_clip_align_4gpu_clean_v1_20260412.sbatch`
- `stage1_rc_dinov2_clip_retrieval_eval_val_1gpu_20260412.sbatch`

This stage provides a bridge-friendly DINOv2 visual representation before the
JSON generation task.  The important implementation choice is that the DINOv2
visual encoder and Qwen decoder remain explicit modules instead of using the
native Qwen3-VL visual tower.

## Stage 2: Bridge / Map Serialization

Training/eval entry points:

- `scripts/train_qwen3_rc_dinov2_caption_bridgev2.py`
- `scripts/train_qwen3_rc_dinov2_caption_bridgev2_segaux.py`
- `scripts/eval_qwen3_rc_dinov2_caption_bridgev2_structured.py`
- `stage2_rc_dinov2_caption_grid16_bridgev2_stage1init_4gpu_20260419.sbatch`
- `stage2_rc_dinov2_caption_grid16_bridgev2_segaux_stage1init_4gpu_20260420.sbatch`
- `stage2_rc_dinov2_caption_grid16_bridgev2_evalviz100_1gpu_20260420.sbatch`

The bridge-v2 path replaces visual placeholder tokens with projected DINOv2
features.  The seg-aux variant adds a road-structure auxiliary signal to make
visual tokens more geometry-aware.

## Stage 3: JSON Road Structure / Centerline

Centerline JSON:

- `scripts/train_qwen3_rc_dinov2_centerline_json_sft.py`
- `scripts/predict_qwen3_rc_dinov2_centerline_json_sft.py`
- `stage3_rc_dinov2_centerline_json_sft_lora_4gpu_20260413.sbatch`
- `stage3_rc_dinov2_centerline_json_sft_lora_8gpu_20260413.sbatch`
- `stage3_rc_dinov2_centerline_json_sft_lora_6epoch_8gpu_qwen3_8b_lr1e4_nonqwen21176_douglas_best_20260430.sbatch`

Adaptive/start-end variants:

- `scripts/train_qwen3_rc_dinov2_centerline_adaptive_keypoint_json_sft.py`
- `scripts/train_qwen3_rc_dinov2_centerline_adaptive_keypoint_grpo.py`
- `scripts/train_qwen3_rc_dinov2_centerline_startend_json_sft.py`
- `scripts/train_qwen3_rc_dinov2_centerline_startend_continuous_head.py`
- `scripts/predict_qwen3_rc_dinov2_centerline_adaptive_keypoint_json_sft.py`
- `scripts/predict_qwen3_rc_dinov2_centerline_startend_json_sft.py`

Road-structure JSON:

- `scripts/train_qwen3_rc_dinov2_road_structure_json_sft.py`
- `scripts/train_qwen3_rc_dinov2_road_structure_bezier4_json_sft.py`
- `scripts/train_qwen3_rc_dinov2_road_structure_quad4_json_sft.py`
- `scripts/predict_qwen3_rc_dinov2_road_structure_json_sft.py`
- `scripts/predict_qwen3_rc_dinov2_road_structure_bezier4_json_sft.py`
- `scripts/predict_qwen3_rc_dinov2_road_structure_quad4_grouped_json_sft.py`

## Douglas Sampling

Primary files:

- `unimapgen/data/rc_centerline_douglas_utils.py`
- `scripts/rewrite_rc_centerline_trainroot_douglas.py`
- `scripts/rewrite_rc_centerline_trainroot_merge_fragments.py`
- `rewrite_rc_centerline_trainroot_douglas_from_22088_mergefrag_20260430.sbatch`

The production DINOv2 centerline branch used Ramer-Douglas-Peucker sampling
with `douglas_epsilon_px=2.5`.  The merge-fragment step was added after visual
inspection showed that raw patch-level annotations split visually continuous
lines too aggressively.

## StageB Continuation

Earlier satellite/fixed-grid StageB files:

- `unimapgen/dataset_build_refactor/stageb.py`
- `scripts/build_stageb_fixed16_gt_point_angle_dataset.py`
- `scripts/run_stageb_rollout_inference.py`
- `scripts/run_patchonly_stageb_rollout_pipeline.sh`
- `configs/llamafactory_paper16_stageb_fixed16_gt_point_angle_empty10/`
- `configs/llamafactory_paper16_stageb_fixed16_gt_trace3_prevregion_sidehint_wr2_empty10_unseenval/`
- `stageb_fixed16_gt_point_angle_4gpu_e24_lr2e5_unseenval.sbatch`
- `stageb_fixed16_gt_trace3_prevregion_sidehint_wr2_4gpu_e6_lr1e5_unseenval.sbatch`

StageB teaches continuity by giving the current patch a compact previous-state
hint.  In the later RC adaptation, this idea became incoming traces from
neighboring patches, but the root design comes from these fixed-grid StageB
experiments.

## Reference Docs

- `docs/dinov2_l14_map_serialization_pipeline.md`
- `docs/qwen_dinov2_map_serialization_branch.md`
- `docs/dinov2_lane_seg_branch.md`
- `docs/rc_centerline_sft_v1_json_lora_spec_20260413.md`
- `docs/rc_centerline_adaptive_keypoint_json_spec_20260416.md`
- `docs/rc_centerline_grpo_rl_execution_spec_20260418.md`
- `docs/rc_handoff_current_20260419.md`

