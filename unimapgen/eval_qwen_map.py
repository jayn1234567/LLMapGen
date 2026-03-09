import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimapgen.qwen_map_pipeline import (
    build_qwen_map_components,
    build_qwen_map_dataset,
    compute_shift_metrics,
    maybe_load_model_checkpoint,
)
from unimapgen.utils import load_yaml, select_torch_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"[Eval] Loaded config: {args.config}", flush=True)
    print(f"[Eval] Building dataset split={args.split}...", flush=True)
    ds = build_qwen_map_dataset(
        cfg,
        split=str(args.split),
        max_samples=args.max_samples,
        train_augment=False,
    )
    print(f"[Eval] Dataset ready: {len(ds)} samples", flush=True)
    print("[Eval] Building tokenizer/collator/model...", flush=True)
    _, collator, model = build_qwen_map_components(cfg, train_set=ds)
    maybe_load_model_checkpoint(model, args.checkpoint)
    print(f"[Eval] Checkpoint loaded: {args.checkpoint}", flush=True)

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    device = select_torch_device(prefer_cuda=True)
    model.to(device)
    model.eval()
    print(f"[Eval] Device={device}", flush=True)

    total_loss = 0.0
    total_count = 0
    total_correct = 0
    total_tok = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {args.split}", leave=False):
            out = model(
                image=batch["image"].to(device),
                prompt_input_ids=batch["prompt_input_ids"].to(device),
                prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                state_input_ids=batch["state_input_ids"].to(device),
                state_attention_mask=batch["state_attention_mask"].to(device),
                map_input_ids=batch["map_input_ids"].to(device),
                map_attention_mask=batch["map_attention_mask"].to(device),
            )
            total_loss += float(out["loss"].item()) * batch["image"].shape[0]
            total_count += batch["image"].shape[0]
            correct, total = compute_shift_metrics(out["logits"], out["labels"])
            total_correct += correct
            total_tok += total

    result = {
        "split": str(args.split),
        "samples": len(ds),
        "loss": total_loss / max(total_count, 1),
        "token_acc": float(total_correct) / float(max(total_tok, 1)),
    }
    print("[Eval] Finished.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
