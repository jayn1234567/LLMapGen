# NPU Test Scripts

All scripts in this folder are Ascend/NPU inference and evaluation entrypoints.
They mirror the train recipe matrix, download checkpoints/datasets when needed,
run patch or state-update inference, write visualizations and metrics, and
upload outputs.

The explicit `smoke_train_*` entrypoint is the exception: it verifies the
private DINOv2 segmentation pretraining path before that visual tower is used
by the SFT recipes.

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
| `DATASET_OBS_PATH` | OBS dataset archive path. The DINOv2 lane-intersection launcher accepts zip, tar, and tar.gz. |
| `DATASET_PATH` | Local extracted dataset root containing `phase_a`/`phase_b` jsonl files and images. |
| `NUM_TEST_SAMPLES` | Number of samples to evaluate; `0` means full test split. |
| `COORD_MODE` | `auto`, `norm1000`, or `pixel`; `auto` reads `meta.coord_mode`. |
| `REFERENCE_EVAL_SPLIT_ROOT` | Optional old fixed difficulty-set root. Its image identities/buckets are remapped to complete records from the current dataset. |
| `REFERENCE_EVAL_ALLOWED_TARGET_SPLITS` | New-dataset splits allowed in the remapped evaluation. Default `eval,test`; adding `train` preserves more inputs but introduces training leakage. |
| `REFERENCE_EVAL_VERIFY_PIXELS` | Compare decoded RGB pixels after ID/coordinate matching when the old image root is available. |
| `TRANSFORMERS_SPEC` | Transformers package spec installed by the script. Qwen3.5 defaults to `transformers>=5.7.0`. |
| `TOKENIZERS_SPEC` | Tokenizers package spec. Qwen3.5 scripts keep this open-ended by default to avoid conflicts with newer Transformers. |

## Fixed-Input Remapping

`scripts/tools/remap_fixed_eval_to_dataset.py` makes cross-dataset model
comparisons use the same image patches without reusing stale labels. It keeps
the reference set's difficulty membership and order, matches each patch through
the preserved patch id or `tile_id + x0 + y0`, then writes the complete record
from the target dataset. The resulting prompts and targets therefore include
the target release's current `lane_type` and `intersection_type` fields.

The tool scans target train/eval/test to report where every reference moved,
but emits only target eval/test by default. Inspect `mapping_report.json` before
inference. A match found only in target train is deliberately excluded because
evaluating it would leak a sample seen by the new model. Pixel verification is
optional and should be enabled when the old dataset image root is still local.

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
