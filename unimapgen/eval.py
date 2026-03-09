import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unimapgen.data import build_dataset_from_cfg, collate_fn
from unimapgen.models import build_model_from_cfg
from unimapgen.utils import load_yaml


def build_val_dataset(cfg):
    dcfg = cfg["data"]
    return build_dataset_from_cfg(
        cfg,
        split=dcfg["val_split"],
        max_samples=dcfg.get("max_val_samples"),
        train_augment=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ds = build_val_dataset(cfg)
    pad_id = ds.tokenizer.pad_id
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0 if int(cfg["data"]["num_workers"]) <= 0 else max(1, int(cfg["data"]["num_workers"]) // 2),
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_cfg(cfg, vocab_size=ds.tokenizer.vocab_size, pad_id=pad_id)
    model.load_state_dict(ckpt["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_count = 0
    total_tok = 0
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            pv_images = batch.get("pv_images")
            if pv_images is not None:
                pv_images = pv_images.to(device, non_blocking=True)
            prompt_types = batch.get("prompt_type")
            if prompt_types is not None:
                prompt_types = prompt_types.to(device, non_blocking=True)
            prompt_tokens = batch.get("prompt_tokens")
            if prompt_tokens is not None:
                prompt_tokens = prompt_tokens.to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            logits = model(
                image,
                inp,
                pv_images=pv_images,
                prompt_types=prompt_types,
                prompt_tokens=prompt_tokens,
            )
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            ).view_as(tgt)
            valid = loss_mask & tgt.ne(pad_id)
            loss = (per_tok * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            total_loss += float(loss.item()) * image.shape[0]
            total_count += image.shape[0]
            pred = logits.argmax(dim=-1)
            mask = tgt.ne(pad_id)
            total_tok += int(mask.sum().item())
            total_correct += int((pred.eq(tgt) & mask).sum().item())

    print(f"val_loss={total_loss / max(total_count, 1):.4f}")
    print(f"val_tok_acc={total_correct / max(total_tok, 1):.4f}")


if __name__ == "__main__":
    main()
