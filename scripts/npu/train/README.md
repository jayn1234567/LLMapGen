# NPU Train Scripts

All scripts in this folder are training entrypoints for Ascend/NPU cloud jobs.
Current formal SFT recipe files are self-contained: each one includes its own
editable parameter block, environment setup, dependency install, OBS download,
training launch, checkpoint handling, and upload flow.

| Script | Purpose |
|---|---|
| `train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_dinov2_qwen3_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv2 + pure Qwen3 text LLM, no DeepStack. Defaults to `QWEN_MODEL_NAME=Qwen3-8B`. |
| `train_sft_stage_a_lane_dinov2_qwen3_5_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv2 + Qwen3.5 text LLM, no DeepStack. Defaults to `QWEN_MODEL_NAME=Qwen3.5-4B-Instruct` and `TRANSFORMERS_SPEC=transformers>=5.7.0`. |
| `train_sft_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane-only state update, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane-only state update, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane+intersection, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane+intersection state update, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane+intersection, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane+intersection state update, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_lora_llm_npu.sh` | LoRA-focused SFT: Stage A lane+intersection, DINOv2 + Qwen3-VL, train LLM LoRA/alignment while keeping the run lighter. |
| `train_sft_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_lora_llm_npu.sh` | LoRA-focused SFT: Stage A lane+intersection, DINOv3 + Qwen3-VL. |
| `train_sft_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_npu.sh` | Formal SFT: Stage A lane-only, DINOv2 + Qwen3-VL, DeepStack residual injection plus main-stream ViT layer fusion. |
| `train_sft_stage_a_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane-only, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane+intersection, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_intersection_multi_moe_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane+intersection, DINOv2+DINOv3 token-router MoE + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv2+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane-only, DINOv2+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane+intersection, DINOv2+SigLIP concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_intersection_dinov2_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane+intersection, DINOv2+SigLIP concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane-only, DINOv3+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane-only, DINOv3+SigLIP static concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_a_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage A lane+intersection, DINOv3+SigLIP concat + Qwen3-VL, no DeepStack. |
| `train_sft_stage_b_lane_intersection_dinov3_siglip_concat_qwen3vl_nodeepstack_npu.sh` | Formal SFT: Stage B lane+intersection, DINOv3+SigLIP concat + Qwen3-VL, no DeepStack. |
| `train_sft_multivision_qwen3vl_nodeepstack_npu.sh` | Older/general multi-vision SFT launcher for Qwen3-VL no-DeepStack experiments; prefer the explicit formal recipe files above for production runs. |
| `train_grpo_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage A lane-only, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage B lane-only, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_a_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage A lane-only, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_b_lane_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage B lane-only, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage A lane+intersection, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_b_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage B lane+intersection, DINOv2 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_a_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage A lane+intersection, DINOv3 + Qwen3-VL, no DeepStack. |
| `train_grpo_stage_b_lane_intersection_dinov3_qwen3vl_nodeepstack_npu.sh` | GRPO: Stage B lane+intersection, DINOv3 + Qwen3-VL, no DeepStack. |

Naming rules:

| Name part | Meaning |
|---|---|
| `stage_a` | Patch-level recognition without incoming neighbor state hints. |
| `stage_b` | State-update training with left/top incoming hints. |
| `lane` | Centerline-only task. |
| `lane_intersection` | Centerline plus intersection task. |
| `dinov2`, `dinov3` | Single vision-tower family. |
| `multi_moe` | DINOv2+DINOv3 token-level router fusion. |
| `dinov2_siglip_concat`, `dinov3_siglip_concat` | Prismatic-style DINO+SigLIP static concat fusion. |
| `deepstack_layer_fusion` | DeepStack residual injection plus fused main ViT stream. |
| `nodeepstack` | Main visual tokens only, no per-layer residual injection. |

Qwen text LLM overrides:

| Variable | Meaning |
|---|---|
| `QWEN_MODEL_NAME` | Directory name under `MODEL_OBS_PATH`, for example `Qwen3-8B` or `Qwen3.5-4B-Instruct`. |
| `QWEN_MODEL_OBS_PATH` | Full OBS path to the text LLM checkpoint. Overrides `MODEL_OBS_PATH/QWEN_MODEL_NAME`. |
| `QWEN_PATH` | Local path for the downloaded or pre-mounted text LLM checkpoint. If `config.json` exists, download is skipped. |
| `TRANSFORMERS_SPEC` | Transformers package spec installed by the script. Qwen3.5 needs a version exposing `Qwen3_5*` text classes. |
| `TOKENIZERS_SPEC` | Tokenizers package spec. Qwen3.5 scripts keep this open-ended by default to avoid conflicts with newer Transformers. |
