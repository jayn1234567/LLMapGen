from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def foreground_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
    valid = targets != ignore_index
    foreground = (targets == 1).to(probabilities.dtype)
    valid_float = valid.to(probabilities.dtype)
    probabilities = probabilities * valid_float
    foreground = foreground * valid_float
    intersection = (probabilities * foreground).sum()
    denominator = probabilities.sum() + foreground.sum()
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    foreground_weight: float = 1.0,
    dice_weight: float = 0.5,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    class_weights = logits.new_tensor([1.0, float(foreground_weight)], dtype=torch.float32)
    cross_entropy = F.cross_entropy(
        logits.float(),
        targets,
        weight=class_weights,
        ignore_index=ignore_index,
    )
    dice = foreground_dice_loss(logits, targets, ignore_index=ignore_index)
    total = cross_entropy + float(dice_weight) * dice
    return total, {"cross_entropy": cross_entropy.detach(), "dice_loss": dice.detach()}


def confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int = 2,
    ignore_index: int = 255,
) -> torch.Tensor:
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    # HCCL does not support float64 AllReduce. Accumulate on device in
    # float32, then metrics_from_confusion promotes the reduced matrix on CPU.
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=logits.device)
    for target_class in range(num_classes):
        target_mask = valid & (targets == target_class)
        for predicted_class in range(num_classes):
            matrix[target_class, predicted_class] = (
                target_mask & (predictions == predicted_class)
            ).sum()
    return matrix


def metrics_from_confusion(matrix: torch.Tensor, eps: float = 1e-12) -> dict[str, Any]:
    matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
    true_positive = matrix.diag()
    target_total = matrix.sum(dim=1)
    prediction_total = matrix.sum(dim=0)
    union = target_total + prediction_total - true_positive
    iou = true_positive / union.clamp_min(eps)
    precision = true_positive / prediction_total.clamp_min(eps)
    recall = true_positive / target_total.clamp_min(eps)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
    valid_iou = union > 0
    mean_iou = iou[valid_iou].mean() if valid_iou.any() else iou.new_tensor(0.0)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(eps)
    return {
        "background_iou": float(iou[0].item()),
        "lane_iou": float(iou[1].item()),
        "mean_iou": float(mean_iou.item()),
        "lane_precision": float(precision[1].item()),
        "lane_recall": float(recall[1].item()),
        "lane_f1": float(f1[1].item()),
        "pixel_accuracy": float(accuracy.item()),
        "confusion_matrix": matrix.to(dtype=torch.int64).cpu().tolist(),
    }
