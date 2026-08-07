# NPU Train Scripts

## Three-Image 200k DINO vs Native Qwen3-VL-8B

The controlled comparison is documented in
`docs/THREE_IMAGE_CONTEXT512_200K_DINO_VS_NATIVE_QWEN3VL8B.md`. Experiment A
uses original DINOv2-Large with the existing CapRL-Qwen3VL-4B-derived text LLM:

```text
train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_200k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
```

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

## Raw-Lane local256 550k Ordinary Checkpoints

The formal full-parameter Raw-Lane 550k DI entry is:

```bash
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh
```

It defaults to `CHECKPOINT_SAVE_MODE=original`, `scripts/deepspeed_zero3.json`,
`gather_16bit_weights_on_model_save=true`, and rank0-only publication. The
Trainer synchronizes the NPU and releases unused cache immediately before each
save. Checkpoints therefore appear directly as `checkpoint-*` below the run
root and do not require a later CPU merge from `zero_shards/node_*`.

`ORDINARY_CHECKPOINTS_ONLY=True` is the formal-run guard. It rejects an
inherited `CHECKPOINT_SAVE_MODE=sharded` before training starts. Historical
sharded-checkpoint smoke scripts explicitly set this guard to `False`; they
are compatibility tests and are not the formal DI recipe.

The matching 550k LLM-LoRA entry is:

```bash
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
```

It applies LoRA only to the CapRL-derived text LLM while training the projector
and DINOv2 with ordinary parameters. It uses HCCL DDP without DeepSpeed, writes
ordinary rank0 adapter/non-LoRA checkpoints, and disables training-time eval
loss by default (`ENABLE_EVAL=False`).

Stage-A train scripts download the base Qwen/Qwen3-VL model plus the recipe's
vision tower assets. Stage-B train scripts do not download the base model;
they download or use the Stage-A checkpoint through `STAGE_A_CHECKPOINT_*` and
continue from that checkpoint.

## Raw-Lane + Pose Three-Image 800k

The paired three-image datasets provide matched LoRA and full-parameter
Stage-A recipes for both views:

```text
train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_original_dinov2_caprl4b_nodeepstack_fullparam_npu.sh
train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_original_dinov2_caprl4b_nodeepstack_fullparam_npu.sh
```

Both preserve the main Jiangjihua model baseline: original DINOv2-Large at
518, penultimate-layer patch tokens, `mlp2x_gelu`, CapRL-Qwen3VL-4B-derived
text LLM, and no DeepStack. Qwen uses LLM-only LoRA (`r=8`, `alpha=16`,
dropout `0.05`); the projector and DINOv2 remain ordinary trainable modules.
Each sample provides
three independent inputs in the fixed order clean BEV, Raw-Lane, and Pose.
The strict preflight scans every JSON target and record contract, uniformly
opens all three image roles from each split, and blocks training if the order,
aliases, metadata, or three prompt placeholders disagree.
`multi_image_input.image_roles` is the canonical packaged role-order field;
the historical `image_order` spelling is also accepted, while conflicting
values are rejected. Redundant `raw_lane_image` and `pose_image` aliases are
optional when the ordered three-element `images` list and
`meta.input_image_roles` are valid; the primary `image` alias remains required.

The LoRA recipes default to 8 epochs, LR `2e-4` for Qwen LoRA/projector and
`2e-5` for DINOv2, global batch 128, per-device batch 4, BF16, gradient
checkpointing, HCCL DDP without DeepSpeed, and ordinary rank0 LoRA checkpoints.
The matched full-parameter recipes train Qwen, projector, and DINOv2 at `2e-5`,
use DeepSpeed ZeRO-3, per-device batch 1, global batch 128, and ordinary rank0
full-model checkpoints. On 4 nodes x 8 NPUs, the derived gradient accumulation
is 4. Training-time eval loss is disabled for both variants.
`MODEL_MAX_LENGTH=8192` is intentional: three
DINOv2 streams contribute 4107 visual tokens before prompt and target tokens.
Values below 6144 are rejected instead of silently truncating supervision.

The local256 formal recipe defaults to:

```text
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/local256_rawpos/local256_rawlane_pose_800k.tar
```

It accepts the released metadata name `local256_rawlane_pose_800k` and the
historical three-image aliases. The context512/ROI256 package is intentionally
not accepted by this launcher. The separate context512/ROI256 recipe defaults
to:

```text
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256_rawpos/context512_roi256_rawlane_pose_800k.tar
```

It accepts `context512_roi256_rawlane_pose_800k` and its historical aliases,
and verifies that each of the three inputs is 512x512 while target coordinates
remain relative to the center 256x256 ROI. Do not mix the two views in one run. See
`docs/DATASET_V2_RAWLANE_POSE_THREE_IMAGE_800K.md` for the record and coordinate
contracts.

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

## Type-Clean 512 CapRL Rerun

`train_sft_stage_a_lane_intersection_typeclean512_dinov2_caprl4b_nodeepstack_npu.sh`
reproduces the Jiangjihua v9-best architecture and optimization recipe while
using the 512x512 type-clean lane/intersection dataset. The raw archive is
normalized once with
`scripts/npu/data/prepare_upload_typeclean512_lane_intersection_sft_npu.sh` and
uploaded to the configured `whu/jn/data` OBS directory. The conversion maps
lane type 1 to `common`, type 2 to `right_turn`, drops type 3 U-turn reference
lines, and maps every other or missing value, including values above 20, to
`other`. It maps source intersection pairs `1|1`, `1|2`, `1|3`, and `4|1` to
the semantic output values `common`, `t_intersection`, `small_untyped`, and
`t_lane_change_area`. Stage-A prompts omit the always-empty incoming trace and
intersection sections and state the complete output taxonomy explicitly.

Prepare, validate, package, and upload the dataset once on the Ascend server:

```bash
bash scripts/npu/data/prepare_upload_typeclean512_lane_intersection_sft_npu.sh
```

The DI training launcher downloads the prepared archive. Before model downloads,
it scans every target JSON record, samples images from every split, requires a
complete class field on every centerline and intersection, checks norm1000
coordinates and valid class values, and rejects any remaining centerline
`LaneType=3`. Conversion and inspection reports are retained with the run
artifacts.

Run only the dataset download and preflight with:

```bash
INSPECT_ONLY=True \
bash scripts/npu/train/train_sft_stage_a_lane_intersection_typeclean512_dinov2_caprl4b_nodeepstack_npu.sh
```

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
| `train_sft_stage_a_lane_intersection_typeclean512_dinov2_caprl4b_nodeepstack_npu.sh` | Jiangjihua v9-best rerun: type-clean 512 lane+intersection data, DINOv2, CapRL-Qwen3VL-4B-derived LLM, no DeepStack, with strict dataset preflight. |
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
