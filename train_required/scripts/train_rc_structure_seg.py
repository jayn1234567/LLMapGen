from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _resolve_repo_root() -> Path:
    # 兼容最小仓库的新目录布局，避免脚本移动后相对导入失效。
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.rc_centerline_cnn_prefix_dataset import RC_DASH_SOLID_CLASS_NAMES  # noqa: E402
from unimapgen.data.rc_centerline_cnn_prefix_dataset import RC_MULTICLASS_CLASS_NAMES  # noqa: E402
from unimapgen.data.rc_centerline_cnn_prefix_dataset import RC_STRUCTURE_MULTICLASS_CLASS_NAMES  # noqa: E402
from unimapgen.data.rc_structure_seg_dataset import (  # noqa: E402
    RCStructureSegDataset,
    RCStructureSegDatasetConfig,
    rc_structure_seg_collate_fn,
)
from unimapgen.models.rc_structure_seg import RCStructureSegModel  # noqa: E402
from unimapgen.utils import cosine_lr, ensure_dir, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure-visual RC structure segmentation benchmark.")
    parser.add_argument("--encoder-type", type=str, required=True, choices=["resnet50_fpn", "dinov2_vitl14"])
    parser.add_argument(
        "--seg-supervision",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "structure_multiclass", "dash_solid"],
    )
    parser.add_argument("--resnet-weights-path", type=str, default="")
    parser.add_argument("--dinov2-model-name-or-path", type=str, default="")
    parser.add_argument("--dinov2-unfreeze-last-n-blocks", type=int, default=12)
    parser.add_argument("--no-dinov2-local-files-only", action="store_true")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--train-meta-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--val-meta-jsonl", type=str, required=True)
    parser.add_argument("--media-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mask-size", type=int, default=512)
    parser.add_argument("--encoder-input-pad-size", type=int, default=0)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--fixed-pos-weight", type=float, default=4.0)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument("--multiclass-class-weights", type=str, default="0.2,1.0,1.0")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-rot90-prob", type=float, default=0.5)
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5)
    parser.add_argument("--aug-vflip-prob", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def num_classes_for_supervision(mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode == "binary":
        return 1
    if mode == "multiclass":
        return len(RC_MULTICLASS_CLASS_NAMES)
    if mode == "structure_multiclass":
        return len(RC_STRUCTURE_MULTICLASS_CLASS_NAMES)
    if mode == "dash_solid":
        return 3
    raise ValueError(f"Unsupported seg supervision mode: {mode}")


def class_names_for_supervision(mode: str) -> list[str]:
    mode = str(mode).strip().lower()
    if mode == "binary":
        return ["background", "foreground"]
    if mode == "multiclass":
        return list(RC_MULTICLASS_CLASS_NAMES)
    if mode == "structure_multiclass":
        return list(RC_STRUCTURE_MULTICLASS_CLASS_NAMES)
    if mode == "dash_solid":
        return list(RC_DASH_SOLID_CLASS_NAMES)
    raise ValueError(f"Unsupported seg supervision mode: {mode}")


def parse_class_weights(spec: str, num_classes: int) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if str(part).strip()]
    if len(values) != int(num_classes):
        raise ValueError(
            f"multiclass_class_weights expects {num_classes} values, got {len(values)} from {spec!r}"
        )
    return values


def build_dataset(args: argparse.Namespace, *, split: str, train_augment: bool) -> RCStructureSegDataset:
    # Stage A 直接读取 canonical RC train/val root，并在训练集上打开几何增强。
    return RCStructureSegDataset(
        RCStructureSegDatasetConfig(
            dataset_jsonl=str(args.train_jsonl if split == "train" else args.val_jsonl),
            dataset_meta_jsonl=str(args.train_meta_jsonl if split == "train" else args.val_meta_jsonl),
            media_dir=str(args.media_dir),
            image_size=int(args.image_size),
            mask_size=int(args.mask_size),
            supervision_mode=str(args.seg_supervision),
            max_samples=(int(args.max_train_samples) if split == "train" else int(args.max_val_samples)) or None,
            train_augment=bool(train_augment),
            aug_rot90_prob=float(args.aug_rot90_prob) if train_augment else 0.0,
            aug_hflip_prob=float(args.aug_hflip_prob) if train_augment else 0.0,
            aug_vflip_prob=float(args.aug_vflip_prob) if train_augment else 0.0,
        )
    )


def build_model(args: argparse.Namespace) -> RCStructureSegModel:
    return RCStructureSegModel(
        encoder_type=str(args.encoder_type),
        output_size=int(args.mask_size),
        decoder_dim=int(args.decoder_dim),
        num_classes=num_classes_for_supervision(str(args.seg_supervision)),
        encoder_input_pad_size=int(args.encoder_input_pad_size),
        resnet_weights_path=str(args.resnet_weights_path),
        dinov2_model_name_or_path=str(args.dinov2_model_name_or_path),
        dinov2_local_files_only=not bool(args.no_dinov2_local_files_only),
        dinov2_unfreeze_last_n_blocks=int(args.dinov2_unfreeze_last_n_blocks),
    )


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * target, dim=(1, 2, 3))
    denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_ids: Sequence[int],
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    losses: list[torch.Tensor] = []
    for class_id in class_ids:
        prob = probs[:, int(class_id)]
        tgt = one_hot[:, int(class_id)]
        inter = torch.sum(prob * tgt, dim=(1, 2))
        denom = torch.sum(prob, dim=(1, 2)) + torch.sum(tgt, dim=(1, 2))
        dice = (2.0 * inter + eps) / (denom + eps)
        losses.append(1.0 - dice)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses, dim=0).mean()


def binary_batch_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum((pred + target) > 0, dim=(1, 2, 3)).float()
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    tgt_sum = torch.sum(target, dim=(1, 2, 3))
    return {
        "iou": float(((inter + 1e-6) / (union + 1e-6)).mean().item()),
        "dice": float(((2.0 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)).mean().item()),
        "precision": float(((inter + 1e-6) / (pred_sum + 1e-6)).mean().item()),
        "recall": float(((inter + 1e-6) / (tgt_sum + 1e-6)).mean().item()),
    }


def multiclass_batch_metrics(logits: torch.Tensor, target: torch.Tensor, class_names: Sequence[str]) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    metrics: dict[str, float] = {}
    class_iou: list[float] = []
    class_dice: list[float] = []
    for class_id in range(1, len(class_names)):
        pred_c = (pred == class_id).float()
        tgt_c = (target == class_id).float()
        inter = torch.sum(pred_c * tgt_c, dim=(1, 2))
        union = torch.sum((pred_c + tgt_c) > 0, dim=(1, 2)).float()
        pred_sum = torch.sum(pred_c, dim=(1, 2))
        tgt_sum = torch.sum(tgt_c, dim=(1, 2))
        iou = float(((inter + 1e-6) / (union + 1e-6)).mean().item())
        dice = float(((2.0 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)).mean().item())
        class_name = str(class_names[class_id])
        metrics[f"{class_name}_iou"] = iou
        metrics[f"{class_name}_dice"] = dice
        class_iou.append(iou)
        class_dice.append(dice)

    pred_fg = (pred > 0).float()
    tgt_fg = (target > 0).float()
    fg_inter = torch.sum(pred_fg * tgt_fg, dim=(1, 2))
    fg_union = torch.sum((pred_fg + tgt_fg) > 0, dim=(1, 2)).float()
    fg_pred_sum = torch.sum(pred_fg, dim=(1, 2))
    fg_tgt_sum = torch.sum(tgt_fg, dim=(1, 2))

    metrics["iou"] = float(sum(class_iou) / max(1, len(class_iou)))
    metrics["dice"] = float(sum(class_dice) / max(1, len(class_dice)))
    metrics["precision"] = float(((fg_inter + 1e-6) / (fg_pred_sum + 1e-6)).mean().item())
    metrics["recall"] = float(((fg_inter + 1e-6) / (fg_tgt_sum + 1e-6)).mean().item())
    return metrics


def build_pos_weight(mask: torch.Tensor, fixed_pos_weight: float, auto_pos_weight: bool) -> torch.Tensor:
    if not bool(auto_pos_weight):
        return torch.tensor([float(fixed_pos_weight)], device=mask.device, dtype=mask.dtype)
    pos = float(mask.sum().item())
    neg = float(mask.numel() - pos)
    value = max(1.0, neg / max(pos, 1.0))
    return torch.tensor([value], device=mask.device, dtype=mask.dtype)


def compute_loss_and_metrics(
    *,
    logits: torch.Tensor,
    mask: torch.Tensor,
    seg_supervision: str,
    main_loss_weight: float,
    dice_weight: float,
    fixed_pos_weight: float,
    auto_pos_weight: bool,
    multiclass_class_weights: Sequence[float],
    class_names: Sequence[str],
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    # 这里统一封装 binary / multiclass / dash-solid 三类监督，方便 Stage A 只维护一套训练主循环。
    mode = str(seg_supervision).strip().lower()
    if mode == "binary":
        pos_weight = build_pos_weight(mask, fixed_pos_weight=fixed_pos_weight, auto_pos_weight=auto_pos_weight)
        main_loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
        dice = binary_dice_loss(logits, mask)
        loss = float(main_loss_weight) * main_loss + float(dice_weight) * dice
        metrics = binary_batch_metrics(logits, mask)
        loss_terms = {
            "main_loss": float(main_loss.item()),
            "dice_loss": float(dice.item()),
        }
        return loss, metrics, loss_terms

    if mode == "dash_solid":
        weight = torch.tensor(list(multiclass_class_weights), device=logits.device, dtype=logits.dtype)
        main_loss = F.cross_entropy(logits, mask.long(), weight=weight)
        dice = multiclass_dice_loss(
            logits,
            mask.long(),
            class_ids=range(1, logits.shape[1]),
        )
        loss = float(main_loss_weight) * main_loss + float(dice_weight) * dice
        metrics = multiclass_batch_metrics(logits, mask.long(), class_names=class_names)
        loss_terms = {
            "main_loss": float(main_loss.item()),
            "dice_loss": float(dice.item()),
        }
        return loss, metrics, loss_terms

    if mode in {"multiclass", "structure_multiclass"}:
        weight = torch.tensor(list(multiclass_class_weights), device=logits.device, dtype=logits.dtype)
        main_loss = F.cross_entropy(logits, mask.long(), weight=weight)
        dice = multiclass_dice_loss(
            logits,
            mask.long(),
            class_ids=range(1, logits.shape[1]),
        )
        loss = float(main_loss_weight) * main_loss + float(dice_weight) * dice
        metrics = multiclass_batch_metrics(logits, mask.long(), class_names=class_names)
        loss_terms = {
            "main_loss": float(main_loss.item()),
            "dice_loss": float(dice.item()),
        }
        return loss, metrics, loss_terms

    raise ValueError(f"Unsupported seg supervision mode: {seg_supervision}")


def metric_keys_for_mode(seg_supervision: str) -> list[str]:
    keys = ["loss", "main_loss", "dice_loss", "iou", "dice", "precision", "recall"]
    mode = str(seg_supervision).strip().lower()
    if mode == "dash_solid":
        keys.extend(["dashed_iou", "dashed_dice", "solid_iou", "solid_dice", "foreground_iou"])
        return keys
    if mode in {"multiclass", "structure_multiclass"}:
        class_names = class_names_for_supervision(mode)
        keys.append("foreground_iou")
        for class_name in class_names[1:]:
            keys.extend([f"{class_name}_iou", f"{class_name}_dice"])
    return keys


def run_val(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seg_supervision: str,
    main_loss_weight: float,
    dice_weight: float,
    fixed_pos_weight: float,
    auto_pos_weight: bool,
    multiclass_class_weights: Sequence[float],
    class_names: Sequence[str],
) -> dict[str, float]:
    # 验证阶段只做前向和指标聚合，不做任何随机增强，保证不同 checkpoint 可直接横向比较。
    model.eval()
    keys = metric_keys_for_mode(seg_supervision)
    sums = {key: 0.0 for key in keys}
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(image)
            loss, metrics, loss_terms = compute_loss_and_metrics(
                logits=logits,
                mask=mask,
                seg_supervision=seg_supervision,
                main_loss_weight=main_loss_weight,
                dice_weight=dice_weight,
                fixed_pos_weight=fixed_pos_weight,
                auto_pos_weight=auto_pos_weight,
                multiclass_class_weights=multiclass_class_weights,
                class_names=class_names,
            )
            n = image.shape[0]
            sums["loss"] += float(loss.item()) * n
            sums["main_loss"] += float(loss_terms["main_loss"]) * n
            sums["dice_loss"] += float(loss_terms["dice_loss"]) * n
            for key, value in metrics.items():
                if key in sums:
                    sums[key] += float(value) * n
            total_n += n
    return {key: value / max(1, total_n) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(str(output_dir))
    (output_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    # 先按监督类型解析类别定义和 loss 权重，再构建 train/val dataloader。
    seg_supervision = str(args.seg_supervision).strip().lower()
    class_names = class_names_for_supervision(seg_supervision)
    multiclass_class_weights = parse_class_weights(
        str(args.multiclass_class_weights),
        num_classes=num_classes_for_supervision(seg_supervision),
    )

    train_set = build_dataset(args, split="train", train_augment=True)
    val_set = build_dataset(args, split="val", train_augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=rc_structure_seg_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers) // 2),
        pin_memory=True,
        collate_fn=rc_structure_seg_collate_fn,
    )

    base_model = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    base_lr = float(args.lr)
    backbone_lr = float(args.backbone_lr) if float(args.backbone_lr) > 0.0 else (
        base_lr if str(args.encoder_type).startswith("resnet") else base_lr * 0.1
    )
    optimizer = torch.optim.AdamW(
        base_model.parameter_groups(backbone_lr=backbone_lr, head_lr=base_lr, weight_decay=float(args.weight_decay))
    )
    model: nn.Module = base_model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[rc-structure-seg] using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(base_model)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
    metrics_path = output_dir / "metrics.jsonl"
    best_val = 1e9
    global_step = 0

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"[rc-structure-seg] train={len(train_set)} val={len(val_set)}", flush=True)
    print(f"[rc-structure-seg] device={device}", flush=True)
    print(f"[rc-structure-seg] seg_supervision={seg_supervision}", flush=True)
    print(f"[rc-structure-seg] class_names={class_names}", flush=True)
    print(f"[rc-structure-seg] encoder_input_pad_size={int(args.encoder_input_pad_size)}", flush=True)
    print(f"[rc-structure-seg] multiclass_class_weights={multiclass_class_weights}", flush=True)
    print(f"[rc-structure-seg] trainable_params={trainable_params} total_params={total_params}", flush=True)
    print(f"[rc-structure-seg] backbone_lr={backbone_lr} head_lr={base_lr}", flush=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.time()
        sums = {key: 0.0 for key in metric_keys_for_mode(seg_supervision)}
        total_n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for it, batch in enumerate(pbar, start=1):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # 学习率按 cosine schedule 更新，并区分 backbone/head 两组参数。
            current_lr = cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                base_lr=base_lr,
                warmup_steps=int(args.warmup_steps),
            )
            backbone_current_lr = backbone_lr if base_lr <= 0 else backbone_lr * (current_lr / base_lr)
            for group_idx, group in enumerate(optimizer.param_groups):
                group["lr"] = backbone_current_lr if group_idx == 0 and len(optimizer.param_groups) > 1 else current_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and device.type == "cuda"):
                logits = model(image)
                # loss 里同时包含主监督项和 dice 项，既约束像素分类也约束整体几何覆盖。
                loss, metrics, loss_terms = compute_loss_and_metrics(
                    logits=logits,
                    mask=mask,
                    seg_supervision=seg_supervision,
                    main_loss_weight=float(args.bce_weight),
                    dice_weight=float(args.dice_weight),
                    fixed_pos_weight=float(args.fixed_pos_weight),
                    auto_pos_weight=bool(args.auto_pos_weight),
                    multiclass_class_weights=multiclass_class_weights,
                    class_names=class_names,
                )

            scaler.scale(loss).backward()
            if float(args.grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in base_model.parameters() if p.requires_grad],
                    float(args.grad_clip_norm),
                )
            scaler.step(optimizer)
            scaler.update()

            n = image.shape[0]
            sums["loss"] += float(loss.item()) * n
            sums["main_loss"] += float(loss_terms["main_loss"]) * n
            sums["dice_loss"] += float(loss_terms["dice_loss"]) * n
            for key, value in metrics.items():
                if key in sums:
                    sums[key] += float(value) * n
            total_n += n
            global_step += 1

            if it % max(1, int(args.log_every)) == 0:
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "iou": f"{metrics.get('iou', 0.0):.3f}",
                    "dice": f"{metrics.get('dice', 0.0):.3f}",
                    "lr": f"{current_lr:.2e}",
                }
                if seg_supervision == "dash_solid":
                    postfix["dash_iou"] = f"{metrics.get('dashed_iou', 0.0):.3f}"
                    postfix["solid_iou"] = f"{metrics.get('solid_iou', 0.0):.3f}"
                elif seg_supervision in {"multiclass", "structure_multiclass"}:
                    for class_name in class_names[1:]:
                        postfix[f"{class_name[:6]}_iou"] = f"{metrics.get(f'{class_name}_iou', 0.0):.3f}"
                pbar.set_postfix(**postfix)

        train_metrics = {key: value / max(1, total_n) for key, value in sums.items()}
        # 每个 epoch 都会落 latest 和当前 best，方便后续 Stage 1 直接接最新视觉编码器权重。
        val_metrics = run_val(
            model=model,
            loader=val_loader,
            device=device,
            seg_supervision=seg_supervision,
            main_loss_weight=float(args.bce_weight),
            dice_weight=float(args.dice_weight),
            fixed_pos_weight=float(args.fixed_pos_weight),
            auto_pos_weight=bool(args.auto_pos_weight),
            multiclass_class_weights=multiclass_class_weights,
            class_names=class_names,
        )
        elapsed = time.time() - t0

        record: dict[str, Any] = {"epoch": epoch}
        for key, value in train_metrics.items():
            record[f"train_{key}"] = value
        for key, value in val_metrics.items():
            record[f"val_{key}"] = value
        record["epoch_seconds"] = elapsed

        print(json.dumps(record, ensure_ascii=False), flush=True)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        state = {
            "epoch": epoch,
            "model": base_model.state_dict(),
            "args": vars(args),
        }
        torch.save(state, str(output_dir / "latest.pt"))
        if float(val_metrics["loss"]) < float(best_val):
            best_val = float(val_metrics["loss"])
            torch.save(state, str(output_dir / "best.pt"))


if __name__ == "__main__":
    main()
