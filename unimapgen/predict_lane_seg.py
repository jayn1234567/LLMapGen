import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from unimapgen.data.lane_seg_dataset import lane_seg_collate_fn
from unimapgen.train_lane_seg import build_dataset, build_model
from unimapgen.utils import ensure_dir, load_yaml


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy().astype(np.float32)
    red = np.zeros_like(out)
    red[..., 0] = 255.0
    alpha = mask[..., None].astype(np.float32) * 0.45
    out = out * (1.0 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    split_key = "max_val_samples" if args.split == "val" else "max_train_samples"
    if args.max_samples is not None and args.max_samples > 0:
        cfg["data"][split_key] = int(args.max_samples)
    dataset = build_dataset(cfg, split=args.split, max_samples_key=split_key, train_augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lane_seg_collate_fn,
    )

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ensure_dir(args.output_dir)
    records = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            token = batch["token_strs"][0]
            logits = model(image)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob >= float(args.threshold)).astype(np.uint8)

            img_np = (image[0].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            overlay = overlay_mask(img_np, mask)
            base = token.rsplit(".", 1)[0]
            mask_path = os.path.join(args.output_dir, f"{base}_mask.png")
            overlay_path = os.path.join(args.output_dir, f"{base}_overlay.png")
            Image.fromarray(mask * 255).save(mask_path)
            Image.fromarray(overlay).save(overlay_path)
            records.append(
                {
                    "token": token,
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "mask_positive_pixels": int(mask.sum()),
                    "mean_probability": float(prob.mean()),
                }
            )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"saved_predictions={len(records)}")
    print(f"output_dir={args.output_dir}")
    print(f"output_json={args.output_json}")


if __name__ == "__main__":
    main()
