# RL Roadmap

The stable SFT, data processing, inference, visualization, and state-update
paths should remain unchanged. RL is a post-training stack that starts from an
SFT checkpoint and SFT inference outputs.

## Target Architecture

```text
SFT checkpoint + SFT inference summary
    -> hard-sample pool
    -> rollout backend
    -> reward workers using infer_index metrics
    -> Ray/verl-style actor training
    -> eval and checkpoint selection
```

## Current Step

The first implemented layer is hard-sample pool construction:

```bash
python scripts/rl/build_hard_pool.py \
  --summary outputs/my_sft_infer/summary.json \
  --source-jsonl data/my_dataset/phase_a/train.jsonl \
  --output-dir data/rl_pool/phase_a_lane \
  --map-task lane
```

The output keeps the SFT JSONL shape:

```text
id
image
meta
conversations[0]  prompt
conversations[1]  ground truth
```

RL-specific audit data is stored under `meta.rl_pool`, including bucket, source
summary, parse status, centerline F1, reward components, and prediction preview.

Pass the same source JSONL that produced the inference summary whenever possible.
If `--source-jsonl` is provided, rows without a source match are skipped by
default so the pool does not accidentally train on formatted chat prompts from
the inference summary. Use `--allow-summary-prompt-fallback` only for manual
debugging.

The default pool is geometry-first. Parse-failure samples are written to
`hard_parse_fail.jsonl` for auditing, but they are not added to the combined
`train.jsonl` unless `--include-parse-fail-in-train` is passed. This matches the
expected SFT regime where output format is already mostly stable and RL should
focus on point alignment, centerline matching, and cut continuity.

## Buckets

- `hard_parse_fail.jsonl`: model output could not be parsed as valid task JSON.
- `hard_zero_match.jsonl`: no centerline matched the ground truth.
- `hard_low_f1.jsonl`: valid output but low centerline F1.
- `hard_cut_error.jsonl`: valid output but weak cut/continuity score.
- `medium.jsonl`: moderate failures.
- `random_keep.jsonl`: normal samples kept to reduce distribution collapse.
- `train.jsonl`: combined pool for future RL training.

## Next Layers

1. Add a rollout interface with HF as the correctness baseline.
2. Add an SGLang multimodal rollout POC for `DINO + adapter/projector + Qwen3VL`.
3. Add reward workers that reuse `mllm/reward` and `infer_index`.
4. Connect actor training through Ray/verl/FSDP2 after rollout correctness is verified.
