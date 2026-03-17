This config is the Stage B continuation run after the patch-only Stage A model.

It uses the fake-state mixture dataset:

- `/mnt/data/project/jn/UniMapGen/outputs/paper16_sft_100img_system_paper_serialized_neighborfix_fake_mixture`

and writes a separate output directory:

- `/mnt/data/project/jn/UniMapGen/outputs/llamafactory_qwen2_5vl_3b_paper16_stageb_from_patchonly_fake_mixture_lora`

The `adapter_name_or_path` in the YAML is only a placeholder default.
The pipeline script rewrites it to the latest completed Stage A checkpoint before training starts.
