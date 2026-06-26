# Scripts

This directory still contains historical experiment scripts from earlier
UniMapGen/OpenSatMap reproduction attempts. For the `douglas_dino_sft` branch,
the maintained centerline JSON route is:

- `train_dinov2_centerline.py`
- `predict_dinov2_centerline.py`
- `tools/prepare_di_qa_trainroot.py`
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
