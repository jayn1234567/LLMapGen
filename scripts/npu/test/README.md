# NPU Test Scripts

All scripts in this folder are Ascend/NPU inference and evaluation entrypoints.
They mirror the train recipe matrix, download checkpoints/datasets when needed,
run patch inference, write visualizations and metrics, and upload outputs.

## Common Inputs

| Variable | Meaning |
|---|---|
| `CHECKPOINT_OBS_LIST` | Comma, semicolon, or newline separated OBS checkpoint roots to download and evaluate. |
| `CHECKPOINT_DIRS` | Comma, semicolon, or newline separated local checkpoint roots. |
| `TRAIN_OUTPUT_DIR` | Local training output root for older scripts that support it. |
| `DATASET_OBS_PATH` | OBS dataset zip path. |
| `DATASET_PATH` | Local extracted dataset root containing `phase_a`/`phase_b` jsonl files and images. |
| `NUM_TEST_SAMPLES` | Number of samples to evaluate; `0` means full test split. |
| `COORD_MODE` | `auto`, `norm1000`, or `pixel`; `auto` reads `meta.coord_mode`. |
| `TRANSFORMERS_SPEC` | Transformers package spec installed by the script. Qwen3.5 defaults to `transformers>=5.7.0`. |
| `TOKENIZERS_SPEC` | Tokenizers package spec. Qwen3.5 scripts keep this open-ended by default to avoid conflicts with newer Transformers. |

## Script Catalog

| Script | Purpose |
|---|---|
| `test_multivision_qwen3vl_nodeepstack_npu.sh` | General older multi-vision Qwen3-VL no-DeepStack inference/eval launcher; prefer explicit stage/task recipe scripts for production. |
| `test_stage_a_lane_dinov2_qwen3_5_deepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | LLM LoRA, inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_deepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | LLM LoRA, inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, DeepStack + ViT layer fusion, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_deepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | LLM LoRA, inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_deepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | LLM LoRA, inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_multi_moe_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_multi_moe_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_multi_moe_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_multi_moe_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |

## Naming Rules

| Name part | Meaning |
|---|---|
| `stage_a` | Patch-level recognition evaluation. |
| `stage_b` | State-update evaluation with left/top incoming hints. |
| `lane` | Centerline-only task. |
| `lane_intersection` | Centerline plus intersection task. |
| `multi_moe` | DINOv2+DINOv3 token-level router fusion. |
| `dinov2_siglip_concat`, `dinov3_siglip_concat` | Prismatic-style DINO+SigLIP static concat fusion. |
| `layer_fusion` | Direct multi-layer ViT feature fusion before the projector. |
| `deepstack` | Per-layer visual residual injection into Qwen decoder layers. |
| `lora_llm` | Inference entry intended for checkpoints trained with LLM LoRA. |
