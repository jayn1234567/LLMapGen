import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimapgen.data.lane_seg_dataset import LaneSegDataset, LaneSegDatasetConfig, lane_seg_collate_fn
from unimapgen.models.dino_lane_seg import DINOv2LaneSeg
from unimapgen.utils import cosine_lr, ensure_dir, load_yaml, set_seed


def build_dataset(cfg, split: str, max_samples_key: str, train_augment: bool):
    dcfg = cfg["data"]
    return LaneSegDataset(
        LaneSegDatasetConfig(
            root_dir=str(dcfg["dataset_root"]),
            ann_json_path=str(dcfg["annotation_json"]),
            split=str(dcfg[f"{split}_split"]),
            image_size=int(dcfg["image_size"]),
            max_samples=dcfg.get(max_samples_key),
            positive_categories=list(dcfg.get("positive_categories", ["lane_line"])),
            mask_line_width=int(dcfg.get("mask_line_width", 5)),
            train_augment=bool(train_augment),
            aug_rot90_prob=float(dcfg.get("aug_rot90_prob", 0.0)) if train_augment else 0.0,
            aug_hflip_prob=float(dcfg.get("aug_hflip_prob", 0.0)) if train_augment else 0.0,
            aug_vflip_prob=float(dcfg.get("aug_vflip_prob", 0.0)) if train_augment else 0.0,
        )
    )


def build_model(cfg):
    mcfg = cfg["model"]
    return DINOv2LaneSeg(
        backbone_path=str(mcfg["backbone_path"]),
        freeze_backbone=bool(mcfg.get("freeze_backbone", True)),
        local_files_only=bool(mcfg.get("local_files_only", True)),
        normalize_input=bool(mcfg.get("normalize_input", True)),
        decoder_dim=int(mcfg.get("decoder_dim", 256)),
    )


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * target, dim=(1, 2, 3))
    denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def batch_metrics(logits: torch.Tensor, target: torch.Tensor):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum((pred + target) > 0, dim=(1, 2, 3)).float()
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    tgt_sum = torch.sum(target, dim=(1, 2, 3))
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    f1 = ((2.0 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)).mean().item()
    return float(iou), float(f1)


def run_val(model, loader, device, bce_weight: float, dice_weight: float):
    model.eval()
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
            bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            dloss = dice_loss(logits, mask)
            loss = float(bce_weight) * bce + float(dice_weight) * dloss
            iou, f1 = batch_metrics(logits, mask)
            n = image.shape[0]
            total_loss += float(loss.item()) * n
            total_iou += iou * n
            total_f1 += f1 * n
            total_n += n
    return total_loss / max(1, total_n), total_iou / max(1, total_n), total_f1 / max(1, total_n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out_dir = str(cfg["output_dir"])
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
    with open(os.path.join(out_dir, "run_meta.txt"), "w", encoding="utf-8") as f:
        f.write("command: " + " ".join(sys.argv) + "\n")
        f.write(f"seed: {cfg.get('seed', 42)}\n")

    train_set = build_dataset(cfg, split="train", max_samples_key="max_train_samples", train_augment=True)
    val_set = build_dataset(cfg, split="val", max_samples_key="max_val_samples", train_augment=False)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lane_seg_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if num_workers <= 0 else max(1, num_workers // 2),
        pin_memory=True,
        collate_fn=lane_seg_collate_fn,
    )

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda")
    epochs = int(cfg["train"]["epochs"])
    total_steps = max(1, epochs * len(train_loader))
    best_val = 1e9
    global_step = 0
    bce_weight = float(cfg["train"].get("bce_weight", 1.0))
    dice_weight = float(cfg["train"].get("dice_weight", 1.0))
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Train samples={len(train_set)} Val samples={len(val_set)}")
    print(f"[Info] Device={device}")
    print(f"[Info] Params trainable={trainable_params}/{total_params}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0.0
        ep_iou = 0.0
        ep_f1 = 0.0
        ep_n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for it, batch in enumerate(pbar, start=1):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            pos = float(mask.sum().item())
            neg = float(mask.numel() - pos)
            pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device)

            lr = cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                base_lr=float(cfg["train"]["lr"]),
                warmup_steps=int(cfg["train"].get("warmup_steps", 0)),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda"):
                logits = model(image)
                bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
                dloss = dice_loss(logits, mask)
                loss = float(bce_weight) * bce + float(dice_weight) * dloss

            scaler.scale(loss).backward()
            if float(cfg["train"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

            iou, f1 = batch_metrics(logits.detach(), mask)
            n = image.shape[0]
            ep_loss += float(loss.item()) * n
            ep_iou += iou * n
            ep_f1 += f1 * n
            ep_n += n
            global_step += 1

            if it % int(cfg["train"].get("log_every", 1)) == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.3f}", lr=f"{lr:.2e}")

        train_loss = ep_loss / max(1, ep_n)
        train_iou = ep_iou / max(1, ep_n)
        train_f1 = ep_f1 / max(1, ep_n)
        val_loss, val_iou, val_f1 = run_val(
            model=model,
            loader=val_loader,
            device=device,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )
        dt = time.time() - t0
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_f1={val_f1:.4f} time={dt:.1f}s"
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_iou": train_iou,
                        "train_f1": train_f1,
                        "val_loss": val_loss,
                        "val_iou": val_iou,
                        "val_f1": val_f1,
                        "epoch_seconds": dt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(state, os.path.join(out_dir, "latest.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(out_dir, "best.pt"))


if __name__ == "__main__":
    main()
