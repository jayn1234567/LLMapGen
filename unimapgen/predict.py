import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from unimapgen.data import build_dataset_from_cfg, collate_fn
from unimapgen.models import build_model_from_cfg
from unimapgen.utils import ensure_dir, load_yaml


def to_jsonable(lines):
    out = []
    for line in lines:
        pts = line["points"]
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        out.append(
            {
                "category": line["category"],
                "start_type": line.get("start_type", "start"),
                "end_type": line.get("end_type", "end"),
                "points": pts,
            }
        )
    return out


def build_dataset(cfg, split: str, max_samples: int):
    return build_dataset_from_cfg(
        cfg,
        split=split,
        max_samples=max_samples,
        train_augment=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--output", type=str, default="outputs/predictions_v1.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ds = build_dataset(cfg, split=args.split, max_samples=args.max_samples)
    pad_id = ds.tokenizer.pad_id
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_cfg(cfg, vocab_size=ds.tokenizer.vocab_size, pad_id=pad_id)
    model.load_state_dict(ckpt["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dec = cfg.get("decode", {})
    min_new_tokens = int(args.min_new_tokens if args.min_new_tokens is not None else dec.get("min_new_tokens", 0))
    temperature = float(args.temperature if args.temperature is not None else dec.get("temperature", 1.0))
    top_k = int(args.top_k if args.top_k is not None else dec.get("top_k", 1))
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty is not None else dec.get("repetition_penalty", 1.0)
    )

    outputs = []
    for batch in loader:
        image = batch["image"].to(device)
        pv_images = batch.get("pv_images")
        if pv_images is not None:
            pv_images = pv_images.to(device)
        prompt_types = batch.get("prompt_type")
        if prompt_types is not None:
            prompt_types = prompt_types.to(device)
        prompt_tokens = batch.get("prompt_tokens")
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(device)
        full = batch["tokens"][:, :-1]  # drop final EOS as prompt template
        start_idx = int(batch["loss_mask"][0].nonzero(as_tuple=False)[0].item()) + 1 if batch["loss_mask"][0].any() else 1
        prompt_ids = full[:, :start_idx].to(device)
        pred_ids = model.generate(
            image=image,
            bos_id=ds.tokenizer.bos_id,
            eos_id=ds.tokenizer.eos_id,
            max_new_tokens=max(16, min(int(args.max_new_tokens), int(cfg["serialization"]["max_seq_len"]) - 1)),
            pv_images=pv_images,
            prompt_ids=prompt_ids,
            prompt_types=prompt_types,
            prompt_tokens=prompt_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )[0].detach().cpu().tolist()
        pred_lines = ds.tokenizer.decode_to_lines(pred_ids)
        gt_ids = batch["current_tokens"][0].tolist()
        gt_lines = ds.tokenizer.decode_to_lines(gt_ids)
        outputs.append(
            {
                "token": batch["token_strs"][0],
                "pred_num_lines": len(pred_lines),
                "gt_num_lines": len(gt_lines),
                "pred_lines": to_jsonable(pred_lines),
                "gt_lines": to_jsonable(gt_lines),
            }
        )

    ensure_dir(os.path.dirname(args.output) or ".")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"saved {len(outputs)} samples to {args.output}")


if __name__ == "__main__":
    main()
