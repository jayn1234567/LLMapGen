# NPU Train Scripts

All scripts in this folder are Ascend/NPU training entrypoints. Formal recipe
files are self-contained: each one includes its own editable parameter block,
environment setup, dependency install, OBS download, training launch,
checkpoint handling, and upload flow. They do not call another project shell
script.

## Comment Style

Formal train scripts keep editable paths and knobs as shell variables near the
top of each file. Each path, model asset, fusion switch, training hyperparameter,
logging option, and NPU/HCCL runtime variable has an inline comment beside the
assignment. Longer workflow sections also have short block comments, so recipe
validation, asset download, checkpoint resolution, argument assembly, launch,
and upload steps can be scanned without reading every command.

## Current Qwen Text Matrix

The Qwen3/Qwen3.5 text-LLM matrix now covers Stage A and Stage B, both `lane`
and `lane_intersection`, for the following recipes:

| Recipe | Meaning |
|---|---|
| `dinov2_*_nodeepstack` | DINOv2 main visual stream, no DeepStack. |
| `dinov2_*_deepstack` | DINOv2 DeepStack residual injection into early Qwen decoder layers. |
| `dinov2_*_layer_fusion_nodeepstack` | DINOv2 direct ViT multi-layer feature fusion before the projector, no DeepStack. |
| `dinov2_siglip_concat_*_nodeepstack` | DINOv2+SigLIP static concat projector, Prismatic-style, no DeepStack. |
| `dinov2_*_nodeepstack_lora_llm` | DINOv2 no-DeepStack recipe with LoRA on LLM modules. |
| `dinov3_*_nodeepstack` | DINOv3 main visual stream at INPUT_IMAGE_SIZE=512, no DeepStack. |
| `dinov3_*_deepstack` | DINOv3 DeepStack residual injection; defaults to visual layers 6 12 18 23 for ViT-L. |
| `dinov3_*_layer_fusion_nodeepstack` | DINOv3 direct ViT multi-layer feature fusion before the projector, no DeepStack. |
| `dinov3_siglip_concat_*_nodeepstack` | DINOv3+SigLIP static concat projector, Prismatic-style, no DeepStack. |
| `dinov3_*_nodeepstack_lora_llm` | DINOv3 no-DeepStack recipe with LoRA on LLM modules. |
| `multi_moe_*_nodeepstack` | DINOv2+DINOv3 token-level router MoE, no DeepStack. |

Stage-B SFT scripts start from a Stage-A checkpoint. Set either
`STAGE_A_CHECKPOINT_OBS_PATH` for cloud download or `STAGE_A_CHECKPOINT_PATH`
for a local/pre-mounted checkpoint root. The resolver accepts direct checkpoint
folders, `checkpoint-*`, and best-candidate roots such as `best`, `eval_best`,
`infer_best`, and `best_reward`. Stage-B `lora_llm` scripts expect a full
Stage-A checkpoint and then add LLM LoRA for the Stage-B run; adapter-only
Stage-A LoRA continuation is intentionally rejected with a clear error.

Each fixed recipe script only declares and downloads the assets required by
that recipe. For example, a `dinov2_*` script downloads DINOv2 only; it does
not declare or download DINOv3 or SigLIP. `dinov2_siglip_concat_*` downloads
DINOv2 and SigLIP, `dinov3_siglip_concat_*` downloads DINOv3 and SigLIP, and
`multi_moe_*` downloads DINOv2 and DINOv3.

Stage-A train scripts download the base Qwen/Qwen3-VL model plus the recipe's
vision tower assets. Stage-B train scripts do not download the base model;
they download or use the Stage-A checkpoint through `STAGE_A_CHECKPOINT_*` and
continue from that checkpoint.

## Native Qwen3-VL Baseline

Native Qwen3-VL baseline launchers now live under
`scripts/qwen3vl_native/train/`. They are kept separate from the DINO/SigLIP
NPU recipe matrix because they do not use project `vision_tower`,
DINOv2/DINOv3, SigLIP, DeepStack, direct ViT layer fusion, or `mm_projector`
arguments.

`*_lora_llm_npu.sh` scripts use plain torchrun DDP on HCCL by default and do
not install or pass DeepSpeed. Full-parameter SFT scripts still use the
configured DeepSpeed/ZeRO path.

Qwen text LLM overrides:

| Variable | Meaning |
|---|---|
| `QWEN_MODEL_NAME` | Directory name under `MODEL_OBS_PATH`, for example `Qwen3-8B` or `Qwen3.5-4B-Instruct`. |
| `QWEN_MODEL_OBS_PATH` | Full OBS path to the text LLM checkpoint. Overrides `MODEL_OBS_PATH/QWEN_MODEL_NAME`. |
| `QWEN_PATH` | Local path for the downloaded or pre-mounted text LLM checkpoint. If `config.json` exists, download is skipped. |
| `TRANSFORMERS_SPEC` | Transformers package spec installed by the script. Qwen3.5 defaults to `transformers>=5.7.0`. |
| `TOKENIZERS_SPEC` | Tokenizers package spec. Qwen3.5 scripts keep this open-ended by default to avoid conflicts with newer Transformers. |

## Script Catalog

| Script | Purpose |
|---|---|
| `train_grpo_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage a lane dinov2 qwen3vl nodeepstack |
| `train_grpo_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage a lane dinov3 qwen3vl nodeepstack |
| `train_grpo_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage a lane intersection dinov2 qwen3vl nodeepstack |
| `train_grpo_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage a lane intersection dinov3 qwen3vl nodeepstack |
| `train_grpo_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage b lane dinov2 qwen3vl nodeepstack |
| `train_grpo_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage b lane dinov3 qwen3vl nodeepstack |
| `train_grpo_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage b lane intersection dinov2 qwen3vl nodeepstack |
| `train_grpo_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO training: stage b lane intersection dinov3 qwen3vl nodeepstack |
| `train_sft_multivision_qwen3vl_nodeepstack_npu.sh` | General older multi-vision Qwen3-VL no-DeepStack SFT launcher; prefer explicit stage/task recipe scripts for production. |
| `train_sft_stage_a_lane_dinov2_qwen3_5_deepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_deepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_npu.sh` | SFT training: Stage A, lane-only, DINOv2, DeepStack + ViT direct layer fusion, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_oracle_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM; GT intersections are moved into the user prompt as oracle topology priors. |
| `train_sft_stage_a_lane_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_5_deepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_deepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_oracle_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM; GT intersections are moved into the user prompt as oracle topology priors. |
| `train_sft_stage_a_lane_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_5_deepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_deepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_5_deepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_deepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_intersection_multi_moe_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_intersection_multi_moe_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_a_lane_multi_moe_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_a_lane_multi_moe_qwen3_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage A, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_5_deepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_deepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_5_deepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_deepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_5_deepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_deepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_5_deepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_5_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_5_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_deepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_layer_fusion_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, ViT direct layer fusion, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_nodeepstack_lora_llm_npu.sh` | SFT training: LLM LoRA, Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_siglip_concat_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_siglip_concat_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv3+SigLIP concat, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_intersection_multi_moe_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_intersection_multi_moe_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane+intersection, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |
| `train_sft_stage_b_lane_multi_moe_qwen3_5_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3.5 text LLM. |
| `train_sft_stage_b_lane_multi_moe_qwen3_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3 text LLM. |
| `train_sft_stage_b_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | SFT training: Stage B, lane-only, DINOv2+DINOv3 token-router MoE, no DeepStack, Qwen3-VL-derived Qwen3 LLM. |

## Naming Rules

| Name part | Meaning |
|---|---|
| `stage_a` | Patch-level recognition without incoming neighbor state hints. |
| `stage_b` | State-update training with left/top incoming hints. |
| `lane` | Centerline-only task. |
| `lane_intersection` | Centerline plus intersection task. |
| `dinov2`, `dinov3` | Single vision-tower family. |
| `multi_moe` | DINOv2+DINOv3 token-level router fusion. |
| `dinov2_siglip_concat`, `dinov3_siglip_concat` | Prismatic-style DINO+SigLIP static concat fusion. |
| `layer_fusion` | Direct multi-layer ViT feature fusion before the projector. |
| `deepstack` | Per-layer visual residual injection into Qwen decoder layers. |
| `nodeepstack` | Main visual tokens only, no per-layer residual injection. |
| `lora_llm` | LoRA adapters on language-model modules; NPU train scripts use DDP/HCCL without DeepSpeed. |
