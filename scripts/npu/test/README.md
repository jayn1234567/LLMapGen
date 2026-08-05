# NPU Test Scripts

## Three-Image 200k Architecture Comparison Smokes

```text
smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_native_qwen3vl8b_lora_npu.sh
```

Both smokes select the same 200k source IDs with seed 42. The native smoke
requires a Qwen3-VL-capable Transformers 5.7+ environment.

All scripts in this folder are Ascend/NPU inference and evaluation entrypoints.
They mirror the train recipe matrix, download checkpoints/datasets when needed,
run patch or state-update inference, write visualizations and metrics, and
upload outputs.

The explicit `smoke_train_*` entrypoint is the exception: it verifies the
private DINOv2 segmentation pretraining path before that visual tower is used
by the SFT recipes.

The Raw-Lane + Pose three-image SFT routes provide two single-node checkpoint
smokes:

```text
smoke_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
```

They run five optimizer steps and save at step 5. A smoke passes only when it
finds training loss, the required `DI_throughput` line, an ordinary checkpoint
artifact, and a runtime log proving `images_per_sample=3`. Supply the matching
dataset TAR through `DATASET_OBS_PATH`; the scripts never guess its URI.

## Comment Style

Formal test scripts keep editable paths and knobs as shell variables near the
top of each file. Each dataset path, checkpoint input, model asset, fusion
switch, inference parameter, and NPU/HCCL runtime variable has an inline comment
beside the assignment. Key workflow sections also have short block comments for
asset download, checkpoint resolution, argument assembly, per-checkpoint
inference/evaluation, visualization, and OBS upload.

## Common Inputs

| Variable | Meaning |
|---|---|
| `CHECKPOINT_OBS_LIST` | Comma, semicolon, or newline separated OBS checkpoint roots to download and evaluate. |
| `CHECKPOINT_DIRS` | Comma, semicolon, or newline separated local checkpoint roots. |
| `DATASET_OBS_PATH` | OBS dataset zip path. |
| `DATASET_PATH` | Local extracted dataset root containing `phase_a`/`phase_b` jsonl files and images. |
| `NUM_TEST_SAMPLES` | Number of samples to evaluate; `0` means full test split. |
| `COORD_MODE` | `auto`, `norm1000`, or `pixel`; `auto` reads `meta.coord_mode`. |
| `TRANSFORMERS_SPEC` | Transformers package spec installed by the script. Qwen3.5 defaults to `transformers>=5.7.0`. |
| `TOKENIZERS_SPEC` | Tokenizers package spec. Qwen3.5 scripts keep this open-ended by default to avoid conflicts with newer Transformers. |

## Qwen Text Matrix

The Qwen3/Qwen3.5 text-LLM test matrix mirrors the train matrix for Stage A and
Stage B, both `lane` and `lane_intersection`, including DINOv2, DINOv3,
DeepStack, direct ViT layer fusion, DINO+SigLIP concat, DINOv2+DINOv3
multi-MoE, and LLM-LoRA checkpoint entrypoints.

Each fixed recipe test script only downloads the vision assets needed by that
recipe and the checkpoints requested through `CHECKPOINT_OBS_LIST` or
`CHECKPOINT_DIRS`. A DINOv2-only script does not declare or download DINOv3 or
SigLIP. LoRA test scripts also prepare the required LoRA base model path:
Stage-A LoRA tests can download the base Qwen/Qwen3-VL model, while Stage-B LoRA
tests require `LORA_BASE_CHECKPOINT_OBS_PATH`, `LORA_BASE_CHECKPOINT_PATH`,
`QWEN_BASE_MODEL_PATH`, or `MODEL_BASE` so the adapter is merged with its
Stage-A base checkpoint.

Stage A calls `scripts/tools/infer_centerline_checkpoint.py` and passes
`--map-task lane` or `--map-task lane_intersection`. Stage B calls
`scripts/tools/infer_centerline_state_update.py`; `lane_intersection` scripts
pass `--include-intersections`, while lane-only scripts do not.

## Native Qwen3-VL Baseline

Native Qwen3-VL inference/eval launchers now live under
`scripts/qwen3vl_native/test/`. They are kept separate from the DINO/SigLIP NPU
test recipe matrix and evaluate checkpoints trained by
`mllm.native_qwen3vl.train_sft`.

## Script Catalog

| Script | Purpose |
|---|---|
| `smoke_train_dinov2_private_seg_full_finetune_npu.sh` | 8-NPU, 20-step smoke test for full-parameter private RC DINOv2 segmentation pretraining and HF vision-tower export. |
| `test_multivision_qwen3vl_nodeepstack_npu.sh` | General older multi-vision Qwen3-VL no-DeepStack inference/eval launcher; prefer explicit stage/task recipe scripts for production. |
| `test_stage_a_lane_dinov2_qwen3_5_deepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_deepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, DeepStack + ViT direct layer fusion, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov3_qwen3_5_deepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_deepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov3_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_deepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_deepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_5_deepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_deepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_intersection_multi_moe_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_intersection_multi_moe_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_a_lane_multi_moe_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_a_lane_multi_moe_qwen3_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `test_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov2_qwen3_5_deepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_deepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov2_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov3_qwen3_5_deepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_deepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov3_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_5_deepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_deepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_5_deepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_deepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | inference/eval: LLM LoRA, Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_intersection_multi_moe_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_intersection_multi_moe_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `test_stage_b_lane_multi_moe_qwen3_5_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `test_stage_b_lane_multi_moe_qwen3_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `test_stage_b_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | inference/eval: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |

## Naming Rules

| Name part | Meaning |
|---|---|
| `stage_a` | Patch-level recognition evaluation. |
| `stage_b` | State-update evaluation with left/top incoming hints. |
| `lane` | Centerline-only task. |
| `lane_intersection` | Centerline plus intersection task. |
| `dinov2`, `dinov3` | Single vision-tower family. |
| `multi_moe` | DINOv2+DINOv3 token-level router fusion. |
| `dinov2_siglip_concat`, `dinov3_siglip_concat` | Prismatic-style DINO+SigLIP static concat fusion. |
| `layer_fusion` | Direct multi-layer ViT feature fusion before the projector. |
| `deepstack` | Per-layer visual residual injection into Qwen decoder layers. |
| `lora_llm` | Inference entry intended for checkpoints trained with LLM LoRA. |
# Full-local512 output on the original 256-grid E2E engine

`eval_local512_550k_checkpoint34376_gt_empty_fresh_obs_original_e2e_npu.sh`
keeps model inference and native patch evaluation at full `512x512`. Before
calling the untouched original road-rule engine, it uses
`adapt_local512_predictions_to_original_e2e_grid.py` to clip every prediction
into four `256x256` child cells, shift geometry into each child-local frame,
renormalize coordinates to `0..1000`, and rename row/column indices to the
engine's fixed `STEP=256` grid. The original formatter must use scale `0.256`;
passing local512 rows with scale `0.512` gives valid patch metrics but invalid
whole-map placement.

# RawLane local256 full E2E evaluation

`run_and_eval_rc_e2e_rawlane_local256_checkpoint12504_full_npu.sh` evaluates
the RawLane-local256 checkpoint by cropping the aligned
`lane_patch_tif/*_lane.tif` image directly. That raster already contains the
RawLane overlay; it must not be overlaid a second time. The matching
`*_inter.tif` remains the reference for the patch grid and black-patch filter.

## Raw-Lane 200k shared fixed-set comparison

`compare_rawlane_local256_200k_vs_context512_roi256_200k_fixed1100_torch240_npu.sh`
evaluates the Raw-Lane local256 and context512/ROI256 checkpoint-12504 models.
Each dataset independently creates and persists its own deterministic holdout
with seed 42 and `easy=300,medium=300,hard=300,very_hard=200`. The two sets do
not require identical sample IDs. Per-difficulty metrics, combined metrics,
visualizations, and `rawlane200k_seed42_independent_eval_summary.json` are
written in one run.
### Raw-Lane 550k multi-node ZeRO-3 final checkpoint

`test_local_rawlane550k_zero3_globalstep34376_merge_eval_torch240_npu.sh`

To merge the same four-node ZeRO-3 checkpoint and run a full fresh-OBS
RawLane local256 end-to-end evaluation with GT-empty suppression and original
all/low/high metrics, use:

```bash
bash scripts/npu/test/eval_rawlane550k_zero3_globalstep34376_gt_empty_fresh_obs_original_e2e_npu.sh
```

The wrapper defaults to NPUs `2,3,4,5,6,7` with inference batch size 32 per
device. The merge/eval helper also accepts `MERGE_ONLY=True` when only the
regular `pytorch_model.bin` checkpoint is required.
downloads all four `zero_shards/node_*` directories for the configured
`global_step*`, validates and merges the 32-rank DeepSpeed checkpoint into a
regular `pytorch_model.bin`, then runs the fixed 1100-sample Raw-Lane local256
evaluation with combined, per-difficulty, and visualization outputs. The merge
is CPU/RAM/disk intensive; all node shards from the same step are required.

### Raw-Lane local256 200k Qwen LoRA

`test_local_rawlane_local256_200k_lora_checkpoint12504_fixed1100_torch240_npu.sh`
downloads the ordinary rank-0 LoRA checkpoint, the CapRL-Qwen3VL-4B base, and
the original DINOv2 tower. It extracts the same Qwen text LLM base used during
training, restores `non_lora_trainables.bin`, merges the adapter, and evaluates
the persistent 1100-sample Raw-Lane local256 comparison set.


### External local256 per-patch predictions

`eval_external_local256_predictions_fresh_obs_original_e2e_npu.sh` consumes an
existing directory of per-patch inference JSON files and runs the original
formatter, center-lane rule engine, and all/low/high evaluators. It downloads a
fresh E2E archive into a run-local directory, does not rerun model inference,
and does not suppress empty-GT patches or clip predictions using ground truth.
Malformed prediction JSON is recorded in `invalid_predictions.json` and counts
as no prediction by default; set `FAIL_ON_INVALID_PREDICTIONS=True` to stop
instead. A repeated run with the same `RUN_ID` reuses a completed extraction.
