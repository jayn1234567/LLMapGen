# Private RC DINOv2 Segmentation Pretraining

This pipeline trains a non-register Hugging Face `facebook/dinov2-large`
backbone on the paired private RC `images/` and `labels_lane/` datasets. The
backbone is fully unfrozen. The segmentation decoder is training-only and is
not required by UniMapGen SFT.

## Why this pipeline is separate

The existing RC DINOv3 training uses a local Meta DINO implementation, Q/V
LoRA modules, a custom decoder, and `image / 255 - 0.5` normalization. Its `.pt`
files are not directly loadable by `transformers.Dinov2Model.from_pretrained`.

This pipeline uses the same model class, 518 input size, and image processor as
the Jiangjihua DINOv2 VLM route. The best validation checkpoint is exported as:

```text
OUTPUT_DIR/best/
|-- vision_tower/
|   |-- config.json
|   |-- model.safetensors
|   |-- preprocessor_config.json
|   `-- private_seg_metadata.json
|-- segmentation_head.pt
`-- metrics.json
```

Only `best/vision_tower` is passed to later MLLM training.

## Data

The source manifest contains the 16 paired segmentation datasets under:

```text
obs://yw-ads-training-gy1/data/external/personal/q00649977/rc-lane-train-from0425/
```

Each selected local root must contain `images/` and `labels_lane/`, directly or
under `train/`. The split is deterministic and strips `_rNNN_cNNN` suffixes so
patches from the same inferred scene group stay in the same split.

The seven `RCDataset/BaseModel/rc_airflow_task_*` OBS roots are raw Dataset V2
sources. They are not used directly by this segmentation pretraining job.

## Environments

Dataset V2 generation and NPU segmentation training use separate environments.
Do not install GeoPandas/Rasterio into the NPU training environment.

For local Windows Dataset V2 generation, open Anaconda Prompt or PowerShell
with `conda` available and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/tools/setup_rc_dataset_v2_windows.ps1
conda activate rc-dataset-v2
```

For Ascend DINOv2 segmentation training, create the isolated Python 3.11
environment once:

```bash
cd /cache/jn/MLLM_project-unimapgen_v7
bash scripts/npu/setup/create_mllm_dinov2_seg_npu_env.sh
source /home/ma-user/.conda/envs/mllm-dinov2-seg-npu-py311/activate_mllm_dinov2_seg_npu.sh
```

The NPU setup downloads the known Huawei `moxing-framework` and cp311
`torch_npu` wheels from OBS, pins NumPy to 1.26.4, and runs an actual NPU tensor
smoke test. It does not install DeepSpeed, bitsandbytes, SwanLab, or the
geospatial data-building stack.

If the configured Conda channel cannot provide Python 3.11, clone an existing
Python 3.11 Conda environment while still installing this recipe's pinned
packages into the new target:

```bash
CLONE_FROM=/home/ma-user/anaconda3/envs/PyTorch-2.5.1 \
bash scripts/npu/setup/create_mllm_dinov2_seg_npu_env.sh
```

## Ascend smoke test

Activate the environment above, then run:

```bash
cd /cache/jn/MLLM_project-unimapgen_v7
bash scripts/npu/test/smoke_train_dinov2_private_seg_full_finetune_npu.sh
```

The smoke launcher downloads `facebook_dinov2-large`, downloads the first of
the 16 private segmentation datasets, and runs 20 optimizer steps on 8 NPUs.
It does not install or replace Python packages.

Useful overrides:

```bash
NPROC_PER_NODE=8 \
DATASET_LIMIT=1 \
MAX_STEPS=20 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=4 \
OUTPUT_DIR=/cache/jn/outputs/dinov2_private_seg_smoke \
bash scripts/npu/test/smoke_train_dinov2_private_seg_full_finetune_npu.sh
```

The effective global batch in this example is `8 * 2 * 4 = 64`.

## Training defaults

- Input: 518x518
- Patch grid: 37x37, 1369 patch tokens
- Features: hidden states 6, 12, 18, and 24
- Trainable backbone: all parameters except the unused masked-image `mask_token`;
  layer 24 uses `last_hidden_state` so the final DINOv2 LayerNorm is trained
- Vision learning rate: `5e-6`
- Decoder learning rate: `1e-4`
- Loss: weighted cross entropy plus foreground Dice
- Precision: BF16
- Best model: highest validation mean IoU
- Logging: includes `DI_throughput: ... samples/s/npu`

The previous per-device batch size of 48 was for a frozen vision backbone and
must not be reused for full-parameter DINOv2-L training.

## Legacy-head last-12-block recipe

The public-data road-structure segmentation experiment used one final DINOv2
feature map and a deeper spatial decoder rather than learned fusion of layers
6, 12, 18, and 24. The private-data comparison recipe is available as:

```bash
bash scripts/npu/train/train_dinov2_private_seg_legacy_head_last12_di_npu.sh
```

Its defaults intentionally form a separate experiment:

- Feature: hidden state 24 (`last_hidden_state`), no cross-layer fusion
- Decoder: 1x1 projection followed by four `2x upsample + 2x Conv/GN/SiLU`
  stages
- Trainable vision parameters: final 12 transformer blocks and final LayerNorm
- Gradient checkpointing: disabled for the partially frozen backbone
- Vision learning rate: `1e-5`
- Decoder learning rate: `1e-4`
- Weight decay: `0.01`
- Foreground CE weight: `5.0`, equivalent to binary class weights `[0.2, 1.0]`
- Dice loss weight: `1.0`
- Fixed warmup: 500 optimizer steps, followed by cosine decay to zero
- Best checkpoint: highest validation `lane_iou`; background IoU is excluded
  from checkpoint selection

Run the matching single-node Ascend smoke without package installation using:

```bash
bash scripts/npu/test/smoke_train_dinov2_private_seg_legacy_head_last12_npu.sh
```

`HIDDEN_STATE_INDEX=23` can be used for a later penultimate-layer ablation, but
the default of 24 first reproduces the feature consumed by the older successful
segmentation head.
