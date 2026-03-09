import argparse

import torch
from torch.utils.data import DataLoader

from unimapgen.data.lane_seg_dataset import lane_seg_collate_fn
from unimapgen.train_lane_seg import batch_metrics, build_dataset, build_model, dice_loss
from unimapgen.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = build_dataset(cfg, split="val", max_samples_key="max_val_samples", train_augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
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

    bce_weight = float(cfg["train"].get("bce_weight", 1.0))
    dice_weight = float(cfg["train"].get("dice_weight", 1.0))
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(image)
            pos = float(mask.sum().item())
            neg = float(mask.numel() - pos)
            pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            dloss = dice_loss(logits, mask)
            loss = float(bce_weight) * bce + float(dice_weight) * dloss
            iou, f1 = batch_metrics(logits, mask)
            n = image.shape[0]
            total_loss += float(loss.item()) * n
            total_iou += iou * n
            total_f1 += f1 * n
            total_n += n

    print(f"val_loss={total_loss / max(1, total_n):.4f}")
    print(f"val_iou={total_iou / max(1, total_n):.4f}")
    print(f"val_f1={total_f1 / max(1, total_n):.4f}")


if __name__ == "__main__":
    main()
