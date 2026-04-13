from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


def _resolve_repo_root() -> Path:
    # 最小仓库中评估脚本放在子目录里，需要动态找到真正的仓库根。
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.rc_centerline_cnn_prefix_dataset import (  # noqa: E402
    RC_DASH_SOLID_CLASS_NAMES,
    RC_MULTICLASS_CLASS_NAMES,
    RC_SEG_CLASS_COLORS,
    RC_STRUCTURE_MULTICLASS_CLASS_NAMES,
)
from unimapgen.data.rc_structure_seg_dataset import (  # noqa: E402
    RCStructureSegDataset,
    RCStructureSegDatasetConfig,
    rc_structure_seg_collate_fn,
)
from unimapgen.models.rc_structure_seg import RCStructureSegModel  # noqa: E402
from unimapgen.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render RC structure segmentation predictions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--dataset-meta-jsonl", type=str, required=True)
    parser.add_argument("--media-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mask-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def normalize_seg_supervision(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {"binary", "multiclass", "structure_multiclass", "dash_solid"}:
        raise ValueError(f"Unsupported seg supervision mode: {mode}")
    return value


def num_classes_for_mode(mode: str) -> int:
    mode = normalize_seg_supervision(mode)
    if mode == "binary":
        return 1
    if mode == "dash_solid":
        return len(RC_DASH_SOLID_CLASS_NAMES)
    if mode == "multiclass":
        return len(RC_MULTICLASS_CLASS_NAMES)
    if mode == "structure_multiclass":
        return len(RC_STRUCTURE_MULTICLASS_CLASS_NAMES)
    raise ValueError(f"Unsupported seg supervision mode: {mode}")


def class_names_for_mode(mode: str) -> list[str]:
    mode = normalize_seg_supervision(mode)
    if mode == "binary":
        return ["background", "foreground"]
    if mode == "dash_solid":
        return list(RC_DASH_SOLID_CLASS_NAMES)
    if mode == "multiclass":
        return list(RC_MULTICLASS_CLASS_NAMES)
    if mode == "structure_multiclass":
        return list(RC_STRUCTURE_MULTICLASS_CLASS_NAMES)
    raise ValueError(f"Unsupported seg supervision mode: {mode}")


def rgb_for_class_name(class_name: str) -> tuple[int, int, int]:
    key = str(class_name).strip().lower()
    if key == "dashed":
        key = "lane_divider"
    elif key == "solid":
        key = "road_edge"
    color = RC_SEG_CLASS_COLORS.get(key, RC_SEG_CLASS_COLORS["background"])
    return tuple(int(round(float(channel) * 255.0)) for channel in color.tolist())


def build_model_from_checkpoint(checkpoint_path: Path) -> tuple[RCStructureSegModel, Dict[str, Any], str]:
    # 直接根据 checkpoint 里保存的训练参数重建模型，避免手工再同步一份结构配置。
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    saved_args = dict(state.get("args", {}))
    seg_supervision = normalize_seg_supervision(str(saved_args.get("seg_supervision", "binary")))
    model = RCStructureSegModel(
        encoder_type=str(saved_args.get("encoder_type", "resnet50_fpn")),
        output_size=int(saved_args.get("mask_size", 512)),
        decoder_dim=int(saved_args.get("decoder_dim", 256)),
        num_classes=num_classes_for_mode(seg_supervision),
        encoder_input_pad_size=int(saved_args.get("encoder_input_pad_size", 0)),
        resnet_weights_path=str(saved_args.get("resnet_weights_path", "")),
        dinov2_model_name_or_path=str(saved_args.get("dinov2_model_name_or_path", "")),
        dinov2_local_files_only=not bool(saved_args.get("no_dinov2_local_files_only", False)),
        dinov2_unfreeze_last_n_blocks=int(saved_args.get("dinov2_unfreeze_last_n_blocks", 12)),
    )
    model.load_state_dict(state["model"], strict=True)
    return model, saved_args, seg_supervision


def binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    inter = float(np.logical_and(pred_bool, gt_bool).sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())
    pred_sum = float(pred_bool.sum())
    gt_sum = float(gt_bool.sum())
    return {
        "iou": inter / union if union > 0.0 else 0.0,
        "dice": (2.0 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0.0 else 0.0,
        "precision": inter / pred_sum if pred_sum > 0.0 else 0.0,
        "recall": inter / gt_sum if gt_sum > 0.0 else 0.0,
    }


def multiclass_metrics(pred_label: np.ndarray, gt_label: np.ndarray, class_names: Sequence[str]) -> Dict[str, float]:
    pred_label = pred_label.astype(np.int64)
    gt_label = gt_label.astype(np.int64)
    out: Dict[str, float] = {}
    class_ious: list[float] = []
    class_dices: list[float] = []
    for class_id, class_name in enumerate(class_names[1:], start=1):
        pred_c = pred_label == class_id
        gt_c = gt_label == class_id
        inter = float(np.logical_and(pred_c, gt_c).sum())
        union = float(np.logical_or(pred_c, gt_c).sum())
        pred_sum = float(pred_c.sum())
        gt_sum = float(gt_c.sum())
        iou = inter / union if union > 0.0 else 0.0
        dice = (2.0 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0.0 else 0.0
        out[f"{class_name}_iou"] = iou
        out[f"{class_name}_dice"] = dice
        class_ious.append(iou)
        class_dices.append(dice)

    pred_fg = pred_label > 0
    gt_fg = gt_label > 0
    fg_inter = float(np.logical_and(pred_fg, gt_fg).sum())
    fg_union = float(np.logical_or(pred_fg, gt_fg).sum())
    fg_pred_sum = float(pred_fg.sum())
    fg_gt_sum = float(gt_fg.sum())
    out["iou"] = float(sum(class_ious) / max(1, len(class_ious)))
    out["dice"] = float(sum(class_dices) / max(1, len(class_dices)))
    out["foreground_iou"] = fg_inter / fg_union if fg_union > 0.0 else 0.0
    out["precision"] = fg_inter / fg_pred_sum if fg_pred_sum > 0.0 else 0.0
    out["recall"] = fg_inter / fg_gt_sum if fg_gt_sum > 0.0 else 0.0
    return out


def to_overlay_binary(base_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    pred_only = np.logical_and(pred_mask.astype(bool), ~gt_mask.astype(bool))
    gt_only = np.logical_and(gt_mask.astype(bool), ~pred_mask.astype(bool))
    both = np.logical_and(pred_mask.astype(bool), gt_mask.astype(bool))

    def blend(mask: np.ndarray, color: Sequence[int], alpha: float) -> None:
        nonlocal out
        mask_f = mask.astype(np.float32)[..., None]
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1.0 - mask_f * alpha) + color_arr * (mask_f * alpha)

    blend(gt_only, color=(80, 255, 120), alpha=0.75)
    blend(pred_only, color=(255, 92, 92), alpha=0.75)
    blend(both, color=(255, 218, 64), alpha=0.90)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def label_to_rgb(label: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, class_name in enumerate(class_names):
        if class_id == 0:
            continue
        out[label == class_id] = np.asarray(rgb_for_class_name(class_name), dtype=np.uint8)
    return out


def overlay_multiclass(
    base_rgb: np.ndarray,
    pred_label: np.ndarray,
    gt_label: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    # 可视化里把 “漏检 / 误检 / 类别错” 用不同颜色拆开，方便快速判断模型偏差类型。
    out = base_rgb.astype(np.float32).copy()
    correct = np.logical_and(pred_label == gt_label, gt_label > 0)
    missed = np.logical_and(pred_label == 0, gt_label > 0)
    spurious = np.logical_and(pred_label > 0, gt_label == 0)
    wrong_class = np.logical_and(pred_label > 0, gt_label > 0) & (pred_label != gt_label)

    def blend(mask: np.ndarray, color: Sequence[int], alpha: float) -> None:
        nonlocal out
        mask_f = mask.astype(np.float32)[..., None]
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1.0 - mask_f * alpha) + color_arr * (mask_f * alpha)

    blend(missed, color=(255, 96, 235), alpha=0.80)
    blend(spurious, color=(255, 150, 64), alpha=0.80)
    blend(wrong_class, color=(255, 72, 72), alpha=0.85)

    for class_id, class_name in enumerate(class_names):
        if class_id == 0:
            continue
        blend(np.logical_and(correct, gt_label == class_id), color=rgb_for_class_name(class_name), alpha=0.90)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def render_panel_binary(
    *,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    sample_id: str,
    metrics: Dict[str, float],
) -> Image.Image:
    base = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    tile = base.shape[0]
    header_h = 48
    panel = Image.new("RGB", (tile * 2, tile * 2 + header_h), color=(18, 20, 28))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    prob_rgb = np.stack([pred_prob * 255.0, np.zeros_like(pred_prob), np.zeros_like(pred_prob)], axis=-1).astype(np.uint8)
    gt_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    gt_rgb[gt_mask.astype(bool)] = np.asarray((72, 255, 120), dtype=np.uint8)
    overlay_rgb = to_overlay_binary(base, pred_mask, gt_mask)

    panel.paste(Image.fromarray(base, mode="RGB"), (0, header_h))
    panel.paste(Image.fromarray(gt_rgb, mode="RGB"), (tile, header_h))
    panel.paste(Image.fromarray(prob_rgb, mode="RGB"), (0, header_h + tile))
    panel.paste(Image.fromarray(overlay_rgb, mode="RGB"), (tile, header_h + tile))

    draw.text((10, 8), sample_id, fill=(245, 245, 245), font=font)
    draw.text(
        (10, 26),
        f"IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text((10, header_h + 8), "Input RC", fill=(245, 245, 245), font=font)
    draw.text((tile + 10, header_h + 8), "GT structure mask", fill=(245, 245, 245), font=font)
    draw.text((10, header_h + tile + 8), "Pred structure prob", fill=(245, 245, 245), font=font)
    draw.text((tile + 10, header_h + tile + 8), "Overlay", fill=(245, 245, 245), font=font)
    return panel


def render_panel_multiclass(
    *,
    image: np.ndarray,
    gt_label: np.ndarray,
    pred_logits: np.ndarray,
    pred_label: np.ndarray,
    sample_id: str,
    metrics: Dict[str, float],
    class_names: Sequence[str],
) -> Image.Image:
    del pred_logits
    base = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    tile = base.shape[0]
    header_h = 64
    panel = Image.new("RGB", (tile * 2, tile * 2 + header_h), color=(18, 20, 28))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    gt_rgb = label_to_rgb(gt_label, class_names)
    pred_rgb = label_to_rgb(pred_label, class_names)
    overlay_rgb = overlay_multiclass(base, pred_label, gt_label, class_names=class_names)

    panel.paste(Image.fromarray(base, mode="RGB"), (0, header_h))
    panel.paste(Image.fromarray(gt_rgb, mode="RGB"), (tile, header_h))
    panel.paste(Image.fromarray(pred_rgb, mode="RGB"), (0, header_h + tile))
    panel.paste(Image.fromarray(overlay_rgb, mode="RGB"), (tile, header_h + tile))

    draw.text((10, 8), sample_id, fill=(245, 245, 245), font=font)
    summary_parts = [f"mIoU={metrics['iou']:.3f}", f"FG-IoU={metrics.get('foreground_iou', 0.0):.3f}"]
    for class_name in class_names[1:]:
        summary_parts.append(f"{class_name}={metrics.get(f'{class_name}_iou', 0.0):.3f}")
    draw.text((10, 26), " ".join(summary_parts[:4]), fill=(245, 245, 245), font=font)
    draw.text(
        (10, 44),
        f"Dice={metrics['dice']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text((10, header_h + 8), "Input RC", fill=(245, 245, 245), font=font)
    draw.text((tile + 10, header_h + 8), "GT labels", fill=(245, 245, 245), font=font)
    draw.text((10, header_h + tile + 8), "Pred labels", fill=(245, 245, 245), font=font)
    draw.text((tile + 10, header_h + tile + 8), "Overlay / errors", fill=(245, 245, 245), font=font)
    return panel


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    panels_dir = output_dir / "panels"
    ensure_dir(str(panels_dir))

    model, saved_args, seg_supervision = build_model_from_checkpoint(checkpoint_path)
    device = torch.device(str(args.device))
    model.to(device)
    model.eval()

    dataset = RCStructureSegDataset(
        RCStructureSegDatasetConfig(
            dataset_jsonl=str(args.dataset_jsonl),
            dataset_meta_jsonl=str(args.dataset_meta_jsonl),
            media_dir=str(args.media_dir),
            image_size=int(args.image_size),
            mask_size=int(args.mask_size),
            supervision_mode=seg_supervision,
            max_samples=int(args.max_samples) or None,
            train_augment=False,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=rc_structure_seg_collate_fn,
    )

    class_names = class_names_for_mode(seg_supervision)
    manifest: list[Dict[str, Any]] = []
    # 逐样本前向并保存 panel，同时把平均指标写进 summary.json，便于后续交接和快速回看。
    for batch in loader:
        image = batch["image"].to(device)
        with torch.no_grad():
            logits = model(image).detach().cpu().numpy()
        images_np = batch["image"].cpu().numpy().transpose(0, 2, 3, 1)
        masks_np = batch["mask"].cpu().numpy()

        for idx, sample_id in enumerate(batch["sample_ids"]):
            row: Dict[str, Any] = {"sample_id": str(sample_id)}
            if seg_supervision == "binary":
                pred_prob = 1.0 / (1.0 + np.exp(-logits[idx, 0]))
                gt_mask = masks_np[idx, 0] > 0.5
                pred_mask = pred_prob >= float(args.threshold)
                metrics = binary_metrics(pred_mask=pred_mask, gt_mask=gt_mask)
                panel = render_panel_binary(
                    image=images_np[idx],
                    gt_mask=gt_mask,
                    pred_prob=pred_prob,
                    pred_mask=pred_mask,
                    sample_id=str(sample_id),
                    metrics=metrics,
                )
            else:
                pred_label = np.argmax(logits[idx], axis=0).astype(np.int64)
                gt_label = masks_np[idx].astype(np.int64)
                metrics = multiclass_metrics(pred_label=pred_label, gt_label=gt_label, class_names=class_names)
                panel = render_panel_multiclass(
                    image=images_np[idx],
                    gt_label=gt_label,
                    pred_logits=logits[idx],
                    pred_label=pred_label,
                    sample_id=str(sample_id),
                    metrics=metrics,
                    class_names=class_names,
                )
            panel_name = f"{sample_id}.png"
            panel.save(panels_dir / panel_name)
            row["panel"] = f"panels/{panel_name}"
            row.update({key: float(value) for key, value in metrics.items()})
            manifest.append(row)

    metric_keys = sorted({key for row in manifest for key in row.keys() if key not in {"sample_id", "panel"}})
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(manifest),
        "threshold": float(args.threshold),
        "seg_supervision": seg_supervision,
        "class_names": class_names,
        "saved_args": saved_args,
        "metrics": {
            key: float(sum(float(row.get(key, 0.0)) for row in manifest) / max(1, len(manifest)))
            for key in metric_keys
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
