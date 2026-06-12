# DINOv2 Centerline Tiny Trainroot

This is a tiny synthetic RC-style centerline dataset for smoke-testing the
minimal DINOv2 -> Qwen centerline SFT route.

It is intentionally small enough to commit to GitHub. It is not a benchmark
dataset.

Files:

- `train.jsonl`: 6 training rows
- `meta_train.jsonl`: training metadata and target lines
- `val.jsonl`: 2 validation rows
- `meta_val.jsonl`: validation metadata and target lines
- `images/`: 512x512 PNG inputs

The dataset can be used to verify:

- trainroot loading
- Douglas + merge preparation through `--prepare-trainroot`
- DINOv2 centerline model import and Trainer startup, when valid model weights
  are available

Recommended smoke command:

```bash
python scripts/train_dinov2_centerline.py \
  --model-name-or-path /path/to/Qwen-CausalLM \
  --dinov2-model-name-or-path /path/to/dinov2_vitl14 \
  --trainroot data_samples/dinov2_centerline_tiny_trainroot \
  --prepare-trainroot \
  --output-dir /tmp/dinov2_centerline_tiny_smoke \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --save-strategy no \
  --evaluation-strategy no
```

Notes:

- `--model-name-or-path` should point to a complete Hugging Face compatible
  CausalLM checkpoint. The DINOv2 bridge route uses `AutoModelForCausalLM`.
- This tiny dataset does not include model weights.
