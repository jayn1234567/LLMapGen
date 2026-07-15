# Private DINOv2 Last-2 SFT and Discrete Coordinates

## Jiangjihua reference behavior

The best CAPRL no-DeepStack recipe uses one DINOv2 hidden layer
(`mm_vision_select_layer=-2`) followed by `mlp2x_gelu`. Its default
`VISION_LAYER_FUSION_INDEXES` value is empty, so direct multi-layer visual
feature fusion is disabled. DeepStack is also disabled.

## Private DINOv2 integration

The segmentation job exports a standard Hugging Face DINOv2 tower under:

```text
<segmentation-run-root>/best/vision_tower
```

The Dataset V2 SFT launchers accept either the run root or the complete
`best/vision_tower` URI through `DINOV2_TRAIN_OUTPUT_OBS_PATH`. During SFT,
only DINOv2 transformer blocks 22 and 23 plus the final LayerNorm are
trainable. The embeddings and blocks 0 through 21 remain frozen.

The new recipe selects explicit hidden-state index `24`, which is the final
normalized DINOv2 output. Retaining the reference recipe's `-2` selection would
read a feature produced before the physical tail blocks, leaving those newly
unfrozen blocks without a useful gradient path.

Ordinary numeric JSON:

```bash
DINOV2_TRAIN_OUTPUT_OBS_PATH=obs://path/to/segmentation-run \
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_private_dinov2_last2_caprl4b_nodeepstack_npu.sh
```

Discrete coordinate tokens:

```bash
DINOV2_TRAIN_OUTPUT_OBS_PATH=obs://path/to/segmentation-run \
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_private_dinov2_last2_caprl4b_coordtokens_nodeepstack_npu.sh
```

Both launchers default to the Dataset V2 `local256.tar` package, CapRL-Qwen3VL-4B,
8 epochs, ZeRO-3, no DeepStack, and no visual layer fusion.

## Coordinate token format

When `--coordinate_token_mode angle` is enabled, the tokenizer receives 1001
atomic tokens from `<0>` through `<1000>`. Only numeric values nested under a
`points` key in assistant targets are rewritten:

```text
{"points":[[956,42],[1000,0]]}
```

becomes the training sequence:

```text
{"points":[[<956>,<42>],[<1000>,<0>]]}
```

Inference keeps the added coordinate tokens during decoding, restores them to ordinary JSON
numbers, and only then invokes parsing, Jiangjihua line evaluation, and
visualization. Saved prediction JSON therefore remains compatible with the
existing metric tools.

This optimization reduces generated coordinate tokens. It does not reduce the
1369 DINOv2 visual tokens, so its memory and throughput effect should be
measured rather than assumed. Training logs print the original tokenizer's mean
and maximum token count for decimal coordinates before registering the new
vocabulary.
