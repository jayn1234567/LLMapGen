# NPU Test Scripts

All scripts in this folder are Ascend/NPU inference and evaluation entrypoints.
They mirror the train recipe matrix, download checkpoints/datasets when needed,
run patch inference, write visualizations and metrics, and upload outputs.

| Script | Purpose |
|---|---|
| `test_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane-only, DINOv2 + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane-only state update, DINOv2 + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane-only, DINOv3 + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane-only state update, DINOv3 + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane+intersection, DINOv2 + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane+intersection state update, DINOv2 + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane+intersection, DINOv3 + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane+intersection state update, DINOv3 + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_npu.sh` | Formal test: Stage A lane-only, DINOv2 + Qwen3-VL, DeepStack plus main-stream layer fusion. |
| `test_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane-only, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane-only, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane+intersection, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane+intersection, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane-only, DINOv2+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane-only, DINOv2+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane+intersection, DINOv2+SigLIP concat + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane+intersection, DINOv2+SigLIP concat + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane-only, DINOv3+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane-only, DINOv3+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `test_stage_a_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage A lane+intersection, DINOv3+SigLIP concat + Qwen3-VL, no DeepStack. |
| `test_stage_b_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal test: Stage B lane+intersection, DINOv3+SigLIP concat + Qwen3-VL, no DeepStack. |
| `test_multivision_qwen3vl_nodeepstack_npu.sh` | Older/general multi-vision NPU test launcher; prefer explicit stage/task recipe scripts when possible. |

Common inputs:

| Variable | Meaning |
|---|---|
| `CHECKPOINT_OBS_LIST` | Comma, semicolon, or newline separated OBS checkpoint roots to download and evaluate. |
| `CHECKPOINT_DIRS` | Comma, semicolon, or newline separated local checkpoint roots. |
| `TRAIN_OUTPUT_DIR` | Local training output root; resolver tries best candidates and normal `checkpoint-*` folders. |
| `DATASET_OBS_PATH` | OBS dataset zip path. |
| `DATASET_PATH` | Local extracted dataset root containing `phase_a`/`phase_b` jsonl files and images. |
| `NUM_TEST_SAMPLES` | Number of samples to evaluate; `0` means full test split. |
| `COORD_MODE` | `auto`, `norm1000`, or `pixel`; `auto` reads `meta.coord_mode`. |
