# GPU Scripts

This folder contains local GPU training, inference, debug, and visualization
entrypoints. Formal scripts are meant for real runs and write under
`outputs/formal_runs` or `outputs/formal_eval` by default. Smoke/debug scripts
keep tiny defaults and are for runtime checks only.

Common overrides:

```bash
GPU_IDS=0 NUM_GPUS=1 CONDA_ENV=fastvlm \
MODEL_NAME_OR_PATH=/path/to/qwen-or-qwen3vl \
CHECKPOINT_DIR=/path/to/checkpoint \
bash scripts/gpu/<script>.sh
```

For Qwen3.5, set `MODEL_NAME_OR_PATH` or `QWEN3_5_MODEL_PATH` to a real
Qwen3.5 text checkpoint. The local tiny Qwen3.5 checkpoint used in smoke tests
only validates code paths and should not be used as a real base model.

| Script | Purpose |
|---|---|
| `train_sft_qwen_family_formal_gpu.sh` | Generic formal GPU SFT launcher for Qwen3-VL-derived Qwen3, pure Qwen3, Qwen3.5, single-DINO, layer-fusion, MoE, and DINO+SigLIP concat recipes. |
| `test_qwen_family_formal_gpu.sh` | Generic formal GPU patch inference/eval launcher for the same Qwen-family and vision-fusion checkpoint matrix. |
| `train_sft_stage_a_lane_dinov2_qwen3_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv2 + pure Qwen3 LLM, no DeepStack. |
| `test_stage_a_lane_dinov2_qwen3_nodeepstack_gpu.sh` | Formal Stage-A lane inference/eval for DINOv2 + pure Qwen3 checkpoints, no DeepStack. |
| `train_sft_stage_a_lane_dinov2_qwen3_5_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv2 + Qwen3.5 text LLM, no DeepStack. |
| `test_stage_a_lane_dinov2_qwen3_5_nodeepstack_gpu.sh` | Formal Stage-A lane inference/eval for DINOv2 + Qwen3.5 checkpoints, no DeepStack. |
| `train_sft_stage_a_lane_dinov2_qwen3vl_layer_fusion_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv2 + Qwen3-VL-derived LLM, main-stream ViT layer fusion, no DeepStack. |
| `test_stage_a_lane_dinov2_qwen3vl_layer_fusion_nodeepstack_gpu.sh` | Formal inference/eval for DINOv2 layer-fusion no-DeepStack checkpoints. |
| `train_sft_stage_a_lane_dinov3_qwen3vl_layer_fusion_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv3 + Qwen3-VL-derived LLM, main-stream ViT layer fusion, no DeepStack. |
| `test_stage_a_lane_dinov3_qwen3vl_layer_fusion_nodeepstack_gpu.sh` | Formal inference/eval for DINOv3 layer-fusion no-DeepStack checkpoints. |
| `train_sft_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_gpu.sh` | Formal Stage-A lane SFT: DINOv2 + Qwen3-VL-derived LLM, DeepStack residual injection plus main-stream layer fusion. |
| `test_stage_a_lane_dinov2_qwen3vl_deepstack_layer_fusion_gpu.sh` | Formal inference/eval for DINOv2 DeepStack + layer-fusion checkpoints. |
| `train_sft_stage_a_lane_multi_moe_qwen3vl_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv2+DINOv3 token-router MoE + Qwen3-VL-derived LLM, no DeepStack. |
| `test_stage_a_lane_multi_moe_qwen3vl_nodeepstack_gpu.sh` | Formal inference/eval for DINOv2+DINOv3 MoE checkpoints. |
| `train_sft_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv2+SigLIP concat projector + Qwen3-VL-derived LLM, no DeepStack. |
| `test_stage_a_lane_dinov2_siglip_concat_qwen3vl_nodeepstack_gpu.sh` | Formal inference/eval for DINOv2+SigLIP concat checkpoints. |
| `train_sft_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_gpu.sh` | Formal Stage-A lane SFT: DINOv3+SigLIP concat projector + Qwen3-VL-derived LLM, no DeepStack. |
| `test_stage_a_lane_dinov3_siglip_concat_qwen3vl_nodeepstack_gpu.sh` | Formal inference/eval for DINOv3+SigLIP concat checkpoints. |
| `infer_qwen_family_centerline_gpu.sh` | Lightweight standalone Qwen-family patch inference wrapper. Supports single-tower, DeepStack, layer fusion, and multi-vision override args. |
| `infer_dinov2_centerline_gpu.sh` | Legacy/local DINOv2 centerline inference helper. |
| `test_full_checkpoint_gpu.sh` | Older cloud-style full-checkpoint GPU inference script with OBS download/upload flow. |
| `train_sft_qwen3vl_nodeepstack_smoke_gpu.sh` | Real-GPU smoke train plus patch inference for Qwen-family no-DeepStack recipes; defaults to one optimizer step. |
| `train_sft_qwen3_nodeepstack_smoke_gpu.sh` | Smoke wrapper for pure Qwen3 + DINO no-DeepStack path. |
| `train_sft_qwen3_5_nodeepstack_smoke_gpu.sh` | Smoke wrapper for Qwen3.5 + DINO no-DeepStack path. Requires a Qwen3.5-capable Transformers environment. |
| `train_sft_qwen3vl_deepstack_smoke_gpu.sh` | Smoke wrapper for Qwen3-VL-derived LLM + DeepStack injection. |
| `train_sft_qwen3vl_deepstack_layer_fusion_smoke_gpu.sh` | Smoke wrapper for DeepStack plus main-stream ViT layer fusion. |
| `train_sft_qwen3vl_nodeepstack_dinov2_layer_fusion_smoke_gpu.sh` | Smoke wrapper for DINOv2 layer fusion without DeepStack. |
| `train_sft_qwen3vl_nodeepstack_dinov3_layer_fusion_smoke_gpu.sh` | Smoke wrapper for DINOv3 layer fusion without DeepStack. |
| `train_sft_qwen3vl_nodeepstack_moe_smoke_gpu.sh` | Smoke wrapper for DINOv2+DINOv3 token-router MoE without DeepStack. |
| `train_sft_qwen3vl_nodeepstack_dinov2_siglip_concat_smoke_gpu.sh` | Smoke wrapper for DINOv2+SigLIP concat without DeepStack. |
| `train_sft_qwen3vl_nodeepstack_dinov3_siglip_concat_smoke_gpu.sh` | Smoke wrapper for DINOv3+SigLIP concat without DeepStack. |
| `train_sft_debug_phase_a_lane_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh` | Older debug Stage-A lane SFT launcher using ZeRO3 and DINOv2 + Qwen3-VL. |
| `train_sft_debug_phase_b_lane_intersection_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh` | Older debug Stage-B lane+intersection SFT launcher using ZeRO3. |
| `train_llm_align_dinov2_qwen2-1.5b_freeze-vit_gpu.sh` | GPU alignment script: DINOv2 frozen, train Qwen2.5-1.5B LLM/alignment modules. |
| `train_llm_align_dinov2_qwen3-8b_freeze-vit_gpu.sh` | GPU alignment script: DINOv2 frozen, train Qwen3-8B LLM/alignment modules. |
| `train_llm_align_dinov2_qwen3vl-2b_freeze-vit_gpu.sh` | GPU alignment script: DINOv2 frozen, train Qwen3-VL-2B-derived LLM/alignment modules. |
| `train_llm_align_dinov2_qwen3vl-8b_freeze-vit_gpu.sh` | GPU alignment script: DINOv2 frozen, train Qwen3-VL-8B-derived LLM/alignment modules. |
| `train_llm_align_dinov3_qwen2-1.5b_freeze-vit_gpu.sh` | GPU alignment script: DINOv3 frozen, train Qwen2.5-1.5B LLM/alignment modules. |
| `train_llm_align_dinov3_qwen3vl-2b_freeze-vit_gpu.sh` | GPU alignment script: DINOv3 frozen, train Qwen3-VL-2B-derived LLM/alignment modules. |
| `train_llm_align_dinov3_qwen3vl-8b_freeze-vit_gpu.sh` | GPU alignment script: DINOv3 frozen, train Qwen3-VL-8B-derived LLM/alignment modules. |
| `train_grpo_dinov3_qwen3vl_nodeepstack_vllm_debug_gpu.sh` | GPU GRPO debug launcher for DINOv3 + Qwen3-VL no-DeepStack with vLLM-style rollout checks. |
| `debug_deepstack_qwen3vl_multigpu.sh` | Multi-GPU DeepStack debug launcher. |
| `debug_lora_matrix_multigpu.sh` | Multi-GPU LoRA target-module/debug launcher. |
| `build_ab_debug_data.py` | Helper to build small Stage-A/Stage-B debug JSONL subsets. |
| `visualize_centerline_compare.py` | Local visualization helper for comparing GT and predicted centerlines. |

Formal fusion presets:

| Recipe | Key env defaults |
|---|---|
| Layer fusion | `VISION_LAYER_FUSION_INDEXES="6 12 18 23"`, `VISION_LAYER_FUSION_TYPE=mean` |
| DeepStack + layer fusion | `DISABLE_DEEPSTACK=False`, `DEEPSTACK_VISUAL_INDEXES="6 12 18 23"` plus layer-fusion args |
| Multi-MoE | `VISION_BACKBONE=multi_moe`, `MULTI_VISION_FUSION=softmax_router` |
| DINO+SigLIP concat | `VISION_BACKBONE=dinov2_siglip_concat` or `dinov3_siglip_concat`, `MULTI_VISION_FUSION=concat_projector` |
