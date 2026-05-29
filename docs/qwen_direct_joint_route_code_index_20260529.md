# Direct Qwen RC Centerline + Intersection Route Code Index 2026-05-29

This branch is the curated code map for the direct Qwen3-VL route that predicts
RC road centerlines and intersections.  It matches the project write-up for the
native Qwen visual input, Douglas-sampled centerline targets, joint
centerline/intersection QA, and StageB patch-to-patch continuation.

## Route Summary

The direct Qwen route removes the custom DINOv2 bridge and uses native
Qwen3-VL image processing.

1. Render `structure_cleaned` road structure into full RC images.
2. Cut full images into 256x256 centerline patches or 512x512 joint patches.
3. Build JSON QA targets from cleaned centerline and intersection annotations.
4. Apply Douglas simplification and endpoint/heading merge to centerline GT.
5. Train Qwen3-VL LoRA SFT.
6. For StageB, feed predicted left/top neighbor traces into the current patch.
7. Evaluate with patch visualizations, full4096 rollouts, ChamferAP, and joint
   centerline/intersection metrics.

## Core Training And Inference Files

Centerline-only direct Qwen:

- `scripts/train_qwen3_vl_direct_rc_centerline_json.py`
- `scripts/predict_qwen3_vl_direct_rc_centerline_json.py`
- `scripts/predict_qwen3_vl_direct_rc_centerline_adaptive_keypoint_json.py`
- `scripts/predict_qwen3_vl_direct_rc_centerline_json_dpo_stack.py`

Joint centerline + intersection:

- `scripts/build_rc_joint_patch512_dataset.py`
- `scripts/export_llamafactory_rc_centerline_from_offset_manifest.py`
- `scripts/predict_render_qwen3_vl_direct_rc_joint_json.py`
- `scripts/compare_rc_joint_predictions_metrics.py`
- `scripts/train_qwen3_vl_direct_rc_joint_grpo.py`

Shared data and RL helpers:

- `unimapgen/data/rc_centerline_douglas_utils.py`
- `unimapgen/data/rc_centerline_json_sft_dataset.py`
- `unimapgen/data/rc_centerline_adaptive_keypoint_json_sft_dataset.py`
- `unimapgen/rl/centerline_parse_utils.py`
- `unimapgen/rl/centerline_reward.py`
- `unimapgen/rl/joint_reward.py`

## Full RC Rendering And Patch Export

Key files:

- `scripts/build_rc_full4096_from_cleaned_structure_centerline.py`
- `scripts/rebuild_rc_centerline_trainroot_native256_douglas.py`
- `scripts/rewrite_rc_centerline_trainroot_douglas.py`
- `scripts/rewrite_rc_centerline_trainroot_merge_fragments.py`
- `scripts/render_rc_centerline_trainroot_gt_preview.py`
- `scripts/render_rc_joint_trainroot_gt_preview.py`
- `build_rc_full4096_from_cleaned_structure_centerline_xiuzheng_20260427.sbatch`
- `build_rc_centerline_trainroot_native256_douglas_20260427.sbatch`
- `build_rc_centerline_patch256_rot45_135_full_export_merge_20260428.sbatch`
- `build_rc_submit_joint512_trainroot_20260512.sbatch`

Centerline production used 256x256 patches.  Joint centerline + intersection
production used 512x512 patches so one patch can cover more of each
intersection while still keeping JSON sequence length manageable.

## Douglas And Merge Settings

Primary simplification:

- Ramer-Douglas-Peucker epsilon: `douglas_epsilon_px=2.5`
- Stored in metadata as `target_sampling_mode=douglas`
- Implemented in `unimapgen/data/rc_centerline_douglas_utils.py`

Fragment merge:

- implemented by `scripts/rewrite_rc_centerline_trainroot_merge_fragments.py`
- production naming used `merge6h22`, meaning relaxed endpoint proximity and
  heading compatibility were used to join fragments that are visually one line
- previewed with `scripts/render_rc_centerline_trainroot_gt_preview.py`

The goal is to reduce over-dense straight-line point targets while preserving
curves, merges, and lane changes.  Visual QA showed that the merge step was
needed before formal export.

## Direct Centerline SFT

Representative launchers:

- `stage3_rc_qwen3vl4b_direct_centerline_patch256_douglas_merge6h22_plus_rot45_135_json_purelora_colorprompt_4gpu_e6_lr1e4_clean_dataset_xiuzheng_20260429.sbatch`
- `eval_22092_manual_best_step167500_wrapper_direct_centerline_val100_viz_20260502.sbatch`

Important model/checkpoint context:

- base model: `Qwen3-VL-4B-Instruct`
- best centerline-only LoRA from job `22092`
- best checkpoint path used later:
  `/file_storage01/home/mingli/data/outputs/stage3_rc_qwen3vl4b_direct_centerline_patch256_douglas_merge6h22_plus_rot45_135_json_purelora_colorprompt_4gpu_e6_lr1e4_clean_dataset_xiuzheng_rcstyle_20260429/manual_best_step167500_eval_loss0p234529_20260501`

## Joint Centerline + Intersection SFT

Representative launchers:

- `stage3_rc_qwen3vl4b_direct_joint512_centerline_intersection_json_purelora_8gpu_e6_lr1e4_20260506.sbatch`
- `stage3_rc_qwen3vl4b_direct_joint512_submit_purelora_8gpu_e6_lr1e4_20260512.sbatch`
- `eval_23866_checkpoint9500_joint512_submit_val100_viz_20260512.sbatch`

Important model/checkpoint context:

- base model: `Qwen3-VL-4B-Instruct`
- source trainroot:
  `/file_storage01/home/mingli/data/outputs/rc_submit_20260512_joint512_trainroot_douglas_merge6h22`
- best StageA joint checkpoint used as later warm start:
  `/file_storage01/home/mingli/data/outputs/stage3_rc_qwen3vl4b_direct_joint512_submit_purelora_8gpu_e6_lr1e4_20260512/checkpoint-17000`

The public QA target predicts centerline JSON plus intersection polygons/types.
Global intersection IDs are intentionally hidden from the model-visible target.

## StageB Centerline Continuation

Centerline StageB files:

- `scripts/build_rc_centerline_stageb_trace_trainroot.py`
- `scripts/render_rc_centerline_stageb_trace_gt_preview.py`
- `scripts/rollout_predict_qwen3_vl_direct_rc_stageb_full4096.py`
- `scripts/analyze_stageb_patch_continuity.py`
- `stage3_rc_qwen3vl4b_direct_stageb_trace_from22092best_4gpu_e3_lr1e4_best_20260509.sbatch`
- `eval_stageb_full4096_rollout_3epoch_best_10tiles_20260511.sbatch`

Rules:

- incoming traces come from already processed left/top patches
- each incoming trace keeps at least one point and at most three points
- line endpoints carry compact type labels
- full4096 rollout uses previous patch predictions as state, not GT

## Joint StageB: Centerline + Intersection

Joint StageB files:

- `scripts/build_rc_joint_stageb_trace_trainroot.py`
- `scripts/render_rc_joint_stageb_trace_gt_preview.py`
- `scripts/rollout_predict_qwen3_vl_direct_rc_joint_stageb_full4096.py`
- `build_rc_submit_joint512_stageb_trace_trainroot_20260513.sbatch`
- `stage3_rc_qwen3vl4b_direct_joint512_submit_stageb_trace_purelora_8gpu_e6_lr1e4_20260513.sbatch`
- `eval_stageb_joint_checkpoint4500_val100_viz_20260513.sbatch`
- `eval_stageb_joint_full4096_10tiles_checkpoint4500_20260513.sbatch`

Rules:

- `incoming_context.centerline_traces` gives line continuation hints
- `incoming_context.intersection_traces` gives local polygon points from
  intersections that enter the current patch from left/top neighbors
- model-visible QA does not include global intersection IDs
- `require_centerline_for_intersections=true` avoids impossible positives where
  an intersection target exists without centerline support

## Evaluation And Training Extensions

Evaluation:

- `scripts/eval_centerline_predictions_jsonl.py`
- `scripts/render_centerline_prediction_comparison_jsonl.py`
- `scripts/compare_rc_joint_predictions_metrics.py`
- `scripts/analyze_stageb_patch_continuity.py`

RL / preference experiments:

- `scripts/train_qwen3_vl_direct_rc_joint_grpo.py`
- `unimapgen/rl/joint_reward.py`
- `scripts/build_rc_centerline_dpo_preference_dataset.py`
- `scripts/build_rc_centerline_dpo_from_sft_failures.py`
- `configs/llamafactory_rc_centerline_dpo_patch256_20260528/`
- `llamafactory_rc_centerline_dpo_from22092best_4gpu_20260528.sbatch`
- `llamafactory_rc_centerline_dpo_hardneg_from22092best_4gpu_20260528.sbatch`

## Reference Docs

- `docs/rc_qwen3vl_direct_centerline_json_spec_20260415.md`
- `docs/rc_qwen3vl_direct_centerline_adaptive_keypoint_json_spec_20260419.md`
- `docs/rc_centerline_qwen_design_20260403.md`
- `docs/rc_handoff_current_20260419.md`

