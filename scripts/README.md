# Scripts

This directory still contains historical experiment scripts from earlier
UniMapGen/OpenSatMap reproduction attempts. For the `douglas_dino_sft` branch,
the maintained centerline JSON route is:

- `train_dinov2_centerline.py`
- `predict_dinov2_centerline.py`
- `tools/prepare_di_qa_trainroot.py`
- `tools/validate_di_trainroot.py`
- `npu/train/train_dinov2_centerline_qwen_lora_npu.sh`
- `npu/test/test_dinov2_centerline_qwen_lora_npu.sh`

The NPU scripts are self-contained launchers intended for DI-style training
platforms. They read paths and hyperparameters from environment variables, then
call the same Python entrypoints used locally.

Use `tools/prepare_di_qa_trainroot.py` to convert private DI QA datasets into
the trainroot layout expected by `train_dinov2_centerline.py`. It supports both
the older flat layout with `train.jsonl`, `test.jsonl`, and
`img/<group_id>/...png`, and the current server layout with
`phase_a/train.jsonl`, `phase_a/eval.jsonl`, `phase_a/meta_*.jsonl`, and
`images/{train,eval,test}`.

For the current `data_line_samples_33w` layout:

```bash
python scripts/tools/prepare_di_qa_trainroot.py \
  --input-root /cache/dataset_extract \
  --dataset-dir-name data_line_samples_33w \
  --phase phase_a \
  --image-root images \
  --output-root /cache/prepared_trainroot
```

Pass `--dataset-dir-name` when the OBS zip extracts into a variable dataset
directory under the extract root. The default Phase A eval split is written as
`val.jsonl`, which is what the maintained DINOv2 training entry expects. When
`dataset_info.json` says the source labels are `norm1000`, the converter scales
them into the training coordinate range instead of clipping large coordinates.

For the 512 patch `lane_intersection` dataset, keep the same trainroot layout
and let the converter preserve both centerlines and intersection polygons:

```bash
python scripts/tools/prepare_di_qa_trainroot.py \
  --input-root /cache/jjh/data/data_lane_intersection_norm_sample_512_33w \
  --phase phase_a \
  --image-root images \
  --task lane_intersection \
  --output-root /cache/jn/prepared_lane_intersection_trainroot
```

The prepared assistant target remains a single JSON object with a `lines` list.
Centerline entries use `category="centerline"` and intersection polygons use
`category="intersection"` with optional `is_cut`.

After conversion, validate the generated trainroot before launching a DI/NPU
job:

```bash
python scripts/tools/validate_di_trainroot.py \
  --trainroot /cache/prepared_trainroot \
  --expect-train-count 335506 \
  --expect-val-count 19084
```

When training the joint centerline plus intersection task, set `MAP_TASK` so
the launcher selects the matching prompt contract:

```bash
export MAP_TASK=lane_intersection
bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
```

For DI/ModelArts jobs, use the outer self-contained launcher. It keeps the
current Ascend debug paths as defaults and downloads missing inputs from OBS
when the matching `*_OBS_PATH` variables are provided:

```bash
bash scripts/npu/train/train_di_dinov2_centerline_qwen_lora_npu.sh
```

Known local defaults:

```bash
MODEL_NAME_OR_PATH=/cache/jn/model/Qwen3-8B
DINOV2_MODEL_NAME_OR_PATH=/cache/jn/model/dinov2-large
ASSET_DIR=/cache/jn/dinov2seg_bridge/dinov2_centerline_assets_qwen3_8b
TRAINROOT=/cache/jn/prepared_lane_intersection_trainroot
MAP_TASK=lane_intersection
```

In a real DI job, configure these from the platform if the local paths are not
pre-populated:

```bash
OUTPUT_URL=obs://bucket/path/to/output
DATASET_OBS_PATH=obs://bucket/path/to/data_lane_intersection_norm_sample_512_33w.zip
QWEN_MODEL_OBS_PATH=obs://bucket/path/to/Qwen3-8B
DINOV2_MODEL_OBS_PATH=obs://bucket/path/to/dinov2-large
ASSET_OBS_PATH=obs://bucket/path/to/dinov2_centerline_assets_qwen3_8b.tar
```
