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

## Ascend smoke test

Activate the existing Python 3.11 NPU environment, then run:

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
- Vision learning rate: `5e-6`
- Decoder learning rate: `1e-4`
- Loss: weighted cross entropy plus foreground Dice
- Precision: BF16
- Best model: highest validation mean IoU
- Logging: includes `DI_throughput: ... samples/s/npu`

The previous per-device batch size of 48 was for a frozen vision backbone and
must not be reused for full-parameter DINOv2-L training.
