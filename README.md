# unimapgen-v1.0

A minimal, runnable extraction of the two-stage fine-tuning route for UniMapGen paper-style 16-patch training:

- Stage A: patch-only Qwen2.5-VL LoRA SFT
- Stage B: continue from Stage A with neighborfix state mixture
- Rollout: family-wise autoregressive evaluation on 4x4 paper16 patch families

This repository contains code only. Large datasets, model checkpoints, caches, and outputs are intentionally excluded.

## Repository layout

```text
unimapgen-v1.0/
├── configs/
│   ├── llamafactory_paper16_patch_only_100img_system/
│   └── llamafactory_paper16_stageb_from_patchonly_mixture/
├── scripts/
│   ├── build_opensatmap_paper16_family_manifest.py
│   ├── export_llamafactory_patch_only_from_raw_family_manifest.py
│   ├── export_llamafactory_state_sft_from_raw_family_manifest.py
│   ├── rollout_predict_qwen2_5vl_from_raw_family_manifest.py
│   ├── run_patchonly_stageb_rollout_pipeline.sh
│   ├── train_patchonly_stagea.sh
│   └── train_stageb.sh
├── unimapgen/
└── outputs/
```

Recommended local runtime directories after clone:

```text
unimapgen-v1.0/
├── .envs/
│   ├── llamafactory-cu128/
│   └── unimapgen-gpu/
├── ckpts/
│   └── modelscope/Qwen/Qwen2___5-VL-3B-Instruct/
└── outputs/
    ├── paper16_family_manifest_100img.jsonl
    ├── paper16_patch_only_100img_system/
    │   └── train.jsonl
    ├── paper16_sft_100img_system_paper_serialized_neighborfix_mixture/
    │   └── train.jsonl
    ├── llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora/
    └── llamafactory_qwen2_5vl_3b_paper16_stageb_from_patchonly_mixture_lora/
```

## Required external data

Raw OpenSatMap root must contain:

```text
<OpenSatMapRoot>/
├── annotrainval20.json
└── picuse20trainvaltest/
    ├── train/
    └── val/
```

Example server path used during development:

```text
/mnt/data/data1/OpenSateMap
```

## Base model placement

Put the downloaded Qwen2.5-VL-3B-Instruct checkpoint at either:

1. `./ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct`
2. any custom path and export `BASE_MODEL_PATH=/your/path/to/Qwen2___5-VL-3B-Instruct`

The training scripts and rollout scripts read `BASE_MODEL_PATH` if provided.

## Environment variables

Common variables:

```bash
export PROJECT_ROOT=$(pwd)
export BASE_MODEL_PATH=$PROJECT_ROOT/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct
export LF_ENV=$PROJECT_ROOT/.envs/llamafactory-cu128
export PY_ENV=$PROJECT_ROOT/.envs/unimapgen-gpu
```

If you already have a finished Stage A adapter somewhere else:

```bash
export STAGEA_ADAPTER_PATH=/path/to/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora
```

## 1. Build paper16 family manifest

```bash
python scripts/build_opensatmap_paper16_family_manifest.py   --opensatmap-root /path/to/OpenSateMap   --ann-json /path/to/OpenSateMap/annotrainval20.json   --output-manifest outputs/paper16_family_manifest_100img.jsonl   --splits train val
```

## 2. Export Stage A patch-only dataset

```bash
python scripts/export_llamafactory_patch_only_from_raw_family_manifest.py   --ann-json /path/to/OpenSateMap/annotrainval20.json   --family-manifest outputs/paper16_family_manifest_100img.jsonl   --output-root outputs/paper16_patch_only_100img_system   --splits train   --use-system-prompt
```

## 3. Export Stage B state-mixture dataset

```bash
python scripts/export_llamafactory_state_sft_from_raw_family_manifest.py   --ann-json /path/to/OpenSateMap/annotrainval20.json   --family-manifest outputs/paper16_family_manifest_100img.jsonl   --output-root outputs/paper16_sft_100img_system_paper_serialized_neighborfix_mixture   --splits train   --use-system-prompt
```

## 4. Train Stage A

```bash
bash scripts/train_patchonly_stagea.sh
```

Equivalent direct command:

```bash
$LF_ENV/bin/llamafactory-cli train   configs/llamafactory_paper16_patch_only_100img_system/qwen2_5vl_3b_lora_sft.yaml
```

## 5. Train Stage B

```bash
bash scripts/train_stageb.sh
```

Equivalent direct command:

```bash
$LF_ENV/bin/llamafactory-cli train   configs/llamafactory_paper16_stageb_from_patchonly_mixture/qwen2_5vl_3b_lora_sft.yaml
```

## 6. Run rollout evaluation

Minimal single-family smoke test:

```bash
$LF_ENV/bin/python scripts/rollout_predict_qwen2_5vl_from_raw_family_manifest.py   --ann-json /path/to/OpenSateMap/annotrainval20.json   --family-manifest outputs/paper16_family_manifest_100img.jsonl   --output-root outputs/rollout_smoke_1fam   --split train   --max-families 1   --base-model "$BASE_MODEL_PATH"   --adapter outputs/llamafactory_qwen2_5vl_3b_paper16_stageb_from_patchonly_mixture_lora   --engine custom   --device cuda:0   --max-new-tokens 256   --use-patch-only-prompt-when-empty
```

Or use the supervisor script after Stage A and Stage B are ready:

```bash
bash scripts/run_patchonly_stageb_rollout_pipeline.sh
```

## Dataset field reference

See `docs/stagea_stageb_dataset_fields.md` for a field-by-field explanation of the Stage A and Stage B fine-tuning datasets.

## Notes

- This repository does not include datasets, training outputs, or model weights.
- `outputs/` is expected to be created locally by the export and training scripts.
- The current branch only keeps the standard two-stage route. The fake-state branch is intentionally excluded.
