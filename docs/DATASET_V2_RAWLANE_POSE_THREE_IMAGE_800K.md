# Dataset V2 Clean BEV + Raw-Lane + Pose 800k

## Definition

This release contains two paired Stage A datasets:

- `rawlane_pose_three_image_local256_800k`;
- `rawlane_pose_three_image_context512_roi256_800k`.

Every sample has three independent model inputs in a fixed order:

1. a clean BEV road-structure image with no Raw-Lane overlay;
2. a black-background Raw-Lane image from `patch_tif/0_lane.tif`;
3. a black-background historical vehicle-trajectory image from
   `patch_tif/0_pose.tif`.

The `context512_roi256` inputs are three aligned 512x512 images. Only the
central 256x256 ROI is supervised, and norm1000 coordinates remain relative
to that ROI.

The current prompt contract is `three_image_roles_concise_v2`. It identifies
the second image as the PV-model lane image and the third image as the
historical vehicle-trajectory image without describing their rendering or
adding copy/conflict instructions. Re-running the Windows builder with
`--resume` rewrites prompts in existing finalized JSONL files, revalidates the
datasets, and refreshes stale TAR packages without regenerating image assets.

## Record Contract

```json
{
  "image": "images/train/...png",
  "images": [
    "images/train/...png",
    "raw_lane_images/train/...png",
    "pose_images/train/...png"
  ],
  "raw_lane_image": "raw_lane_images/train/...png",
  "pose_image": "pose_images/train/...png",
  "meta": {
    "raw_lane_overlay": false,
    "raw_lane_separate_image": true,
    "input_image_roles": [
      "bev_road_structure",
      "pv_camera_raw_lane",
      "historical_vehicle_trajectory"
    ]
  }
}
```

The user prompt contains exactly three `<image>` tokens and describes the
modalities in the same order. It retains the warning that the PV-camera lane
prediction must not be copied blindly when it conflicts with visible BEV
evidence. It does not describe Raw-Lane as an overlay.

## Selection

- train: 800,000 unique patch IDs;
- empty: 5%;
- easy: 25%;
- medium: 33%;
- hard: 27%;
- very hard: 10%;
- fixed-eval intersection target: 28%;
- train stride: 128;
- eval/test stride: 256;
- fixed source split: `D:\data\fixed_splits\rc_fixed_large_maps_v1.json`.

The 28% fixed-eval intersection target matches the feasible unique-record
limit of the existing bootstrap staging. Local and context variants use the
same sample IDs and fixed eval/test large maps.

## No-Download Build

The previous Raw-Lane/Pose staging already contains labels, Raw-Lane masks,
Pose masks, and candidate selection metadata. The older clean Dataset V2
staging supplies the non-overlaid BEV images. The build therefore performs a
staging join by source index and patch ID instead of downloading OBS data or
extracting TIFF archives again.

First stop the old two-image 800k build with `Ctrl+C`. Then run from the
repository root:

```powershell
python scripts\tools\build_rc_dataset_v2_rawlane_pose_three_image_800k_from_staging_windows.py --clean-staging-root "D:\data\fulldata\staging" --clean-context-staging-root "D:\data\fulldata_context512\staging_context512" --aux-staging-root "D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context" --fixed-source-split-manifest "D:\data\fixed_splits\rc_fixed_large_maps_v1.json" --resume
```

`--clean-staging-root` supplies clean local256 images. `--clean-context-staging-root`
supplies clean context512_roi256 images directly. If the latter is omitted or lacks
a source shard, the builder reconstructs that clean 512 context from neighboring
clean local256 tiles without downloading raw TIFF archives. If an individual
local256 candidate is absent from the clean local staging, the builder first crops
the center ROI from the matching clean context image, then falls back to overlapping
clean local256 tiles. This allows older filtered clean stagings to cover a larger
auxiliary candidate set when their spatial image coverage is sufficient.

The preflight blocks the build unless every auxiliary source has a matching
clean local source, a direct or reconstructable clean context view exists,
geometry/stride settings match, the clean marker says `raw_lane_overlay=false`,
and the auxiliary marker confirms saved
Raw-Lane and Pose assets. It never substitutes an overlaid BEV when a clean
image is missing.

## Outputs

```text
D:\data\fulldata_rawlane_pose_three_image_800k\output_rawlane_pose_three_image_800k\rawlane_pose_three_image_local256_800k
D:\data\fulldata_rawlane_pose_three_image_800k\output_rawlane_pose_three_image_800k\rawlane_pose_three_image_context512_roi256_800k
D:\data\fulldata_rawlane_pose_three_image_800k\packages_rawlane_pose_three_image_800k\rawlane_pose_three_image_local256_800k.tar
D:\data\fulldata_rawlane_pose_three_image_800k\packages_rawlane_pose_three_image_800k\rawlane_pose_three_image_context512_roi256_800k.tar
```

Primary images are hard-linked from clean staging when both roots are on the
same NTFS volume. Raw-Lane and Pose files are reused from finalized auxiliary
assets. The script validates image order, file existence, prompt token count,
difficulty counts, intersection ratio, paired local/context IDs, and fixed
split metadata before packaging.

`mllm/train/train_qwen.py` and `mllm/model/llava_arch.py` already accept an
arbitrary number of aligned images and map one visual feature sequence to each
`<image>` token. Three inputs increase visual-token memory relative to the
two-image recipe, so use a short NPU smoke test before formal DI training.

At DINOv2 input size 518, each image contributes 1369 patch tokens. The three
streams therefore contribute 4107 visual tokens before user and assistant
text. The formal recipes use `MODEL_MAX_LENGTH=8192` and reject values below
6144 so the post-expansion multimodal truncation cannot silently remove the
third stream or JSON supervision. The current experiment uses per-device
batch 4 and derives gradient accumulation from the actual DI world size.

The first production experiment uses LLM-only LoRA on the CapRL text model
(`r=8`, `alpha=16`, dropout `0.05`). The projector and DINOv2 tower remain
fully trainable at `2e-4` and `2e-5`, respectively. It uses HCCL DDP without
DeepSpeed and does not calculate eval loss during training.

For model configuration, DI script requirements, memory-sensitive defaults,
and the exact handoff checklist for another Agent, see
`docs/RAWLANE_POSE_THREE_IMAGE_800K_TRAINING_HANDOFF.md`.
