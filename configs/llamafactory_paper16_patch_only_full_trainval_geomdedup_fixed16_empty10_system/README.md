# LLaMAFactory Config For `paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system`

This folder contains the portable Stage A `8 GPU` LLaMAFactory config template for the deduplicated full-trainval fixed16 dataset.

Files:

- `dataset_info.json`: template dataset registry for LLaMAFactory.
- `qwen2_5vl_3b_lora_sft_stdce_8gpu_e10.yaml`: template `8 GPU`, `10 epoch` Stage A config for the fixed16 full-trainval route.

How this folder is used:

- The portable launch script rewrites both `dataset_info.json` and the yaml file into `.runtime/` before training.
- That means you do not need to edit absolute paths in this folder by hand.
- The default portable launcher is:
  - `scripts/launch_stagea_full_trainval_geomdedup_fixed16_train.sh`

Expected default resource layout under the repository root:

- `ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct/`
- `dataset/paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system/`
- `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/`

Dataset notes:

- This dataset comes from the deduplicated full-trainval patch-only root, so exact duplicate `source_image + crop_box` patch geometries were removed before fixed16 expansion.
- The final fixed16 dataset keeps empty boxes at about `10%`.
- Recorded dataset size:
  - train: `209574`
  - val: `68481`

Training notes:

- Because this dataset is much larger than the old `100img fixed16` training set, the default config uses epoch-based save and eval.
- This is the configuration family that produced the frequently referenced LoRA checkpoint:
  - `checkpoint-157182`
