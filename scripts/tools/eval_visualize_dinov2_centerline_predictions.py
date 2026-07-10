#!/usr/bin/env python3
"""Evaluate and visualize DINOv2 centerline JSONL predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GT_COLORS = {
    "centerline": (40, 220, 80),
    "intersection": (40, 220, 220),
}
PRED_COLORS = {
    "centerline": (255, 70, 70),
    "intersection": (255, 70, 220),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-jsonl", required=True, help="Prediction JSONL from scripts/predict_dinov2_centerline.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trainroot", default="", help="Prepared trainroot, used as default media root.")
    parser.add_argument("--media-dir", default="", help="Directory used to resolve relative image paths.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--coord-range", type=float, default=1000.0, help="Coordinate range for normalized labels, e.g. 1000 for norm1000.")
    parser.add_argument("--gt-coord-mode", choices=["auto", "pixel", "norm1000"], default="auto", help="Coordinate mode for gt_lines before evaluation/visualization.")
    parser.add_argument("--pred-coord-mode", choices=["auto", "pixel", "norm1000"], default="auto", help="Coordinate mode for pred_lines before evaluation/visualization.")
    parser.add_argument("--map-task", choices=["lane", "lane_intersection"], default="lane_intersection")
    parser.add_argument("--categories", default="centerline,intersection")
    parser.add_argument("--meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--jiangjihua-buffer-size", type=float, default=1.0)
    parser.add_argument("--jiangjihua-match-threshold", type=float, default=0.33)
    parser.add_argument("--line-width-px", type=int, default=6)
    parser.add_argument("--engineering-thresholds-px", default="2,4,8")
    parser.add_argument("--mask-iou-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--chamfer-thresholds-m", default="0.9,1.5,3.0,4.5")
    parser.add_argument("--vis-limit", type=int, default=64)
    parser.add_argument("--sheet-cols", type=int, default=2)
    return parser.parse_args()


def parse_csv(text: str, cast=float) -> List[Any]:
    out: List[Any] = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(cast(chunk))
    return out


def read_prediction_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return records
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise TypeError(f"Expected list JSON at {path}, got {type(data)!r}")
        return [item for item in data if isinstance(item, dict)]
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict at {path}:{line_no}, got {type(payload)!r}")
        records.append(payload)
    return records


def normalize_category(value: Any) -> str:
    category = str(value or "centerline").strip().lower().replace("-", "_").replace(" ", "_")
    if category in {"center_line", "centerlines", "center_lines", "line"}:
        return "centerline"
    if category in {"junction", "road_intersection", "crossing_region"}:
        return "intersection"
    return category or "centerline"


def clamp_point(point: Sequence[Any], image_size: int) -> List[int] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    high = int(image_size)
    return [max(0, min(high, int(round(x)))), max(0, min(high, int(round(y))))]


def sanitize_lines(lines: Any, image_size: int) -> List[Dict[str, Any]]:
    if not isinstance(lines, list):
        return []
    out: List[Dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        category = normalize_category(line.get("category", "centerline"))
        points: List[List[int]] = []
        for raw_point in line.get("points", []):
            point = clamp_point(raw_point, image_size)
            if point is not None and (not points or points[-1] != point):
                points.append(point)
        min_points = 3 if category == "intersection" else 2
        if len(points) < min_points:
            continue
        item = dict(line)
        item["category"] = category
        item["points"] = points
        out.append(item)
    return out


def sanitize_records(records: Iterable[Dict[str, Any]], image_size: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["gt_lines"] = sanitize_lines(record.get("gt_lines", []), image_size)
        item["pred_lines"] = sanitize_lines(record.get("pred_lines", []), image_size)
        out.append(item)
    return out


def lines_coord_max(lines: Any) -> float:
    max_value = 0.0
    if not isinstance(lines, list):
        return max_value
    for line in lines:
        if not isinstance(line, dict):
            continue
        points = line.get("points", [])
        if not isinstance(points, (list, tuple)):
            continue
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = abs(float(point[0]))
                y = abs(float(point[1]))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                max_value = max(max_value, x, y)
    return float(max_value)


def trainroot_coord_mode(trainroot: str) -> str:
    root = Path(str(trainroot)).expanduser() if str(trainroot).strip() else None
    if root is None:
        return ""
    info_path = root / "dataset_info.json"
    if not info_path.is_file():
        return ""
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for key in ("coord_mode", "coord_system", "source_dataset_coord_mode"):
        value = str(info.get(key, "")).strip().lower()
        if value:
            return value
    for key in ("coord_max", "coord_range", "source_dataset_coord_range"):
        value = info.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 900.0:
            return "norm1000"
    return ""


def is_norm_mode(mode: str) -> bool:
    return "norm" in str(mode or "").strip().lower()


def convert_norm_point_to_pixel(point: Sequence[Any], image_size: int, coord_range: float) -> List[int] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    high = max(1.0, float(image_size) - 1.0)
    denom = max(float(coord_range), 1e-6)
    return [int(round(x / denom * high)), int(round(y / denom * high))]


def convert_lines_coord_mode(lines: Any, *, mode: str, image_size: int, coord_range: float) -> List[Dict[str, Any]]:
    if not isinstance(lines, list):
        return []
    if str(mode).strip().lower() != "norm1000":
        return [dict(line) if isinstance(line, dict) else line for line in lines]
    converted: List[Dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        item = dict(line)
        points: List[List[int]] = []
        for raw in item.get("points", []):
            point = convert_norm_point_to_pixel(raw, image_size=image_size, coord_range=coord_range)
            if point is not None:
                points.append(point)
        item["points"] = points
        item["coord_mode_before_eval"] = "norm1000"
        item["coord_range_before_eval"] = float(coord_range)
        converted.append(item)
    return converted


def resolve_line_coord_mode(
    *,
    requested: str,
    record: Dict[str, Any],
    field: str,
    dataset_mode: str,
    image_size: int,
) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested != "auto":
        return requested
    explicit = str(record.get(f"{field}_coord_mode", record.get("coord_mode", ""))).strip().lower()
    if explicit:
        if is_norm_mode(explicit):
            return "norm1000"
        if "pixel" in explicit:
            return "pixel"
    # Current norm1000 trainroots write GT in normalized coordinates, while older
    # prediction files did not tag gt_lines. Treat GT from such datasets as norm1000
    # to avoid clamping 0..1000 labels to the 512px image border.
    if field == "gt_lines" and is_norm_mode(dataset_mode):
        return "norm1000"
    max_coord = lines_coord_max(record.get(field, []))
    if max_coord > float(image_size) + 2.0:
        return "norm1000"
    return "pixel"


def convert_record_coordinates(
    records: Iterable[Dict[str, Any]],
    *,
    image_size: int,
    coord_range: float,
    gt_coord_mode: str,
    pred_coord_mode: str,
    dataset_mode: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        gt_mode = resolve_line_coord_mode(
            requested=gt_coord_mode,
            record=item,
            field="gt_lines",
            dataset_mode=dataset_mode,
            image_size=image_size,
        )
        pred_mode = resolve_line_coord_mode(
            requested=pred_coord_mode,
            record=item,
            field="pred_lines",
            dataset_mode="",
            image_size=image_size,
        )
        item["gt_lines"] = convert_lines_coord_mode(
            item.get("gt_lines", []), mode=gt_mode, image_size=image_size, coord_range=coord_range
        )
        item["pred_lines"] = convert_lines_coord_mode(
            item.get("pred_lines", []), mode=pred_mode, image_size=image_size, coord_range=coord_range
        )
        item["eval_coord_modes"] = {"gt_lines": gt_mode, "pred_lines": pred_mode, "dataset": dataset_mode}
        out.append(item)
    return out


def line_points(line: Dict[str, Any]) -> np.ndarray:
    points = np.asarray(line.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return points


def draw_binary_line_mask(points_px: np.ndarray, image_size: int, line_width_px: int) -> np.ndarray:
    image = Image.new("L", (int(image_size), int(image_size)), 0)
    draw = ImageDraw.Draw(image)
    points = np.asarray(points_px, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((int(image_size), int(image_size)), dtype=bool)
    if points.shape[0] == 1:
        x = float(points[0, 0])
        y = float(points[0, 1])
        radius = max(1.0, 0.5 * float(line_width_px))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    else:
        draw.line([tuple(map(float, point)) for point in points], fill=255, width=int(line_width_px))
        radius = max(1.0, 0.5 * float(line_width_px))
        for point in (points[0], points[-1]):
            x = float(point[0])
            y = float(point[1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def semantic_masks(lines: Sequence[Dict[str, Any]], categories: Sequence[str], image_size: int, line_width_px: int) -> Dict[str, np.ndarray]:
    masks = {str(category): np.zeros((int(image_size), int(image_size)), dtype=bool) for category in categories}
    for line in lines:
        category = normalize_category(line.get("category", ""))
        if category not in masks:
            continue
        masks[category] |= draw_binary_line_mask(line_points(line), image_size=image_size, line_width_px=line_width_px)
    return masks


def compute_semantic_miou(
    records: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    image_size: int,
    line_width_px: int,
) -> tuple[float, Dict[str, float]]:
    inter = {str(category): 0.0 for category in categories}
    union = {str(category): 0.0 for category in categories}
    for item in records:
        pred_masks = semantic_masks(item.get("pred_lines", []), categories, image_size, line_width_px)
        gt_masks = semantic_masks(item.get("gt_lines", []), categories, image_size, line_width_px)
        for category in categories:
            pred = pred_masks[str(category)]
            gt = gt_masks[str(category)]
            inter[str(category)] += float(np.logical_and(pred, gt).sum())
            union[str(category)] += float(np.logical_or(pred, gt).sum())
    per_category = {
        str(category): inter[str(category)] / union[str(category)]
        for category in categories
        if union[str(category)] > 0.0
    }
    return (float(np.mean(list(per_category.values()))) if per_category else 0.0), per_category


def build_gt_index(records: Sequence[Dict[str, Any]], categories: Sequence[str]) -> tuple[Dict[str, Dict[int, List[Dict[str, Any]]]], Dict[str, int]]:
    gt_by_category_sample: Dict[str, Dict[int, List[Dict[str, Any]]]] = {str(category): {} for category in categories}
    gt_count = {str(category): 0 for category in categories}
    for sample_idx, item in enumerate(records):
        for gt_idx, line in enumerate(item.get("gt_lines", [])):
            category = normalize_category(line.get("category", ""))
            if category not in gt_by_category_sample:
                continue
            gt_by_category_sample[category].setdefault(sample_idx, []).append(
                {"sample_idx": sample_idx, "gt_idx": gt_idx, "line": line}
            )
            gt_count[category] += 1
    return gt_by_category_sample, gt_count


def line_score(line: Dict[str, Any]) -> tuple[float, bool]:
    for key in ("score", "line_score", "pseudo_score"):
        value = line.get(key)
        if isinstance(value, (int, float)):
            return float(value), False
    return 1.0, True


def build_pred_index(records: Sequence[Dict[str, Any]], categories: Sequence[str]) -> tuple[Dict[str, List[Dict[str, Any]]], str]:
    pred_by_category: Dict[str, List[Dict[str, Any]]] = {str(category): [] for category in categories}
    used_fallback = False
    used_explicit = False
    for sample_idx, item in enumerate(records):
        for pred_idx, line in enumerate(item.get("pred_lines", [])):
            category = normalize_category(line.get("category", ""))
            if category not in pred_by_category:
                continue
            score, is_fallback = line_score(line)
            used_fallback = used_fallback or is_fallback
            used_explicit = used_explicit or (not is_fallback)
            pred_by_category[category].append(
                {"sample_idx": sample_idx, "pred_idx": pred_idx, "score": float(score), "line": line}
            )
    if used_explicit and used_fallback:
        score_mode = "mixed_explicit_and_constant_fallback"
    elif used_explicit:
        score_mode = "explicit_line_scores"
    else:
        score_mode = "constant_1.0_fallback"
    return pred_by_category, score_mode


def mask_iou(pred_line: Dict[str, Any], gt_line: Dict[str, Any], image_size: int, line_width_px: int) -> float:
    pred_mask = draw_binary_line_mask(line_points(pred_line), image_size=image_size, line_width_px=line_width_px)
    gt_mask = draw_binary_line_mask(line_points(gt_line), image_size=image_size, line_width_px=line_width_px)
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    if union <= 0.0:
        return 0.0
    return float(np.logical_and(pred_mask, gt_mask).sum()) / union


def densify_polyline(points: np.ndarray, step: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] <= 1:
        return points
    out = [points[0]]
    max_step = max(float(step), 1e-3)
    for idx in range(points.shape[0] - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        n_steps = max(1, int(np.ceil(seg_len / max_step)))
        for step_idx in range(1, n_steps + 1):
            t = float(step_idx) / float(n_steps)
            out.append(p0 * (1.0 - t) + p1 * t)
    return np.asarray(out, dtype=np.float32)


def chamfer_distance_m(pred_line: Dict[str, Any], gt_line: Dict[str, Any], meter_per_pixel: float, densify_step_m: float = 0.25) -> float:
    pred_points = line_points(pred_line) * float(meter_per_pixel)
    gt_points = line_points(gt_line) * float(meter_per_pixel)
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return 1e6
    pred_points = densify_polyline(pred_points, step=densify_step_m)
    gt_points = densify_polyline(gt_points, step=densify_step_m)
    distances = np.sqrt(((pred_points[:, None, :] - gt_points[None, :, :]) ** 2).sum(axis=-1))
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if int(num_gt) <= 0:
        return 0.0
    tp_cum = np.cumsum(tp, dtype=np.float64)
    fp_cum = np.cumsum(fp, dtype=np.float64)
    recall = tp_cum / max(float(num_gt), 1.0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(mpre.shape[0] - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def eval_ap_threshold(
    gt_by_category_sample: Dict[str, Dict[int, List[Dict[str, Any]]]],
    gt_count: Dict[str, int],
    pred_by_category: Dict[str, List[Dict[str, Any]]],
    categories: Sequence[str],
    match_mode: str,
    threshold: float,
    image_size: int,
    line_width_px: int,
    meter_per_pixel: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for category in categories:
        category = str(category)
        num_gt = int(gt_count.get(category, 0))
        if num_gt <= 0:
            continue
        matched = {
            sample_idx: np.zeros(len(lines), dtype=bool)
            for sample_idx, lines in gt_by_category_sample.get(category, {}).items()
        }
        preds = sorted(
            pred_by_category.get(category, []),
            key=lambda item: (-float(item["score"]), int(item["sample_idx"]), int(item["pred_idx"])),
        )
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for pred_idx, pred in enumerate(preds):
            sample_idx = int(pred["sample_idx"])
            gts = gt_by_category_sample.get(category, {}).get(sample_idx, [])
            best_value = None
            best_gt_idx = -1
            best_match = False
            for gt_idx, gt in enumerate(gts):
                if bool(matched[sample_idx][gt_idx]):
                    continue
                if match_mode == "mask_iou":
                    value = mask_iou(pred["line"], gt["line"], image_size=image_size, line_width_px=line_width_px)
                    better = (best_value is None) or (value > best_value)
                    is_match = value >= float(threshold)
                else:
                    value = chamfer_distance_m(pred["line"], gt["line"], meter_per_pixel=meter_per_pixel)
                    better = (best_value is None) or (value < best_value)
                    is_match = value <= float(threshold)
                if better:
                    best_value = value
                    best_gt_idx = gt_idx
                    best_match = is_match
            if best_gt_idx >= 0 and best_match:
                matched[sample_idx][best_gt_idx] = True
                tp[pred_idx] = 1.0
            else:
                fp[pred_idx] = 1.0
        out[category] = average_precision(tp, fp, num_gt=num_gt)
    return out


def mean_over_present(per_category: Dict[str, float], gt_count: Dict[str, int]) -> float:
    values = [float(value) for key, value in per_category.items() if int(gt_count.get(str(key), 0)) > 0]
    return float(np.mean(values)) if values else 0.0


def evaluate_official_records(
    records: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    image_size: int,
    meter_per_pixel: float,
    line_width_px: int,
    mask_iou_thresholds: Sequence[float],
    chamfer_thresholds_m: Sequence[float],
) -> Dict[str, Any]:
    categories = [str(category) for category in categories]
    miou, miou_per_category = compute_semantic_miou(records, categories, image_size, line_width_px)
    gt_by_category_sample, gt_count = build_gt_index(records, categories)
    pred_by_category, score_mode = build_pred_index(records, categories)

    mask_ap_by_iou: Dict[str, Dict[str, float]] = {}
    for threshold in mask_iou_thresholds:
        key = f"{float(threshold):.2f}"
        mask_ap_by_iou[key] = eval_ap_threshold(
            gt_by_category_sample,
            gt_count,
            pred_by_category,
            categories,
            match_mode="mask_iou",
            threshold=float(threshold),
            image_size=image_size,
            line_width_px=line_width_px,
            meter_per_pixel=meter_per_pixel,
        )

    chamfer_ap_by_threshold_m: Dict[str, Dict[str, float]] = {}
    for threshold in chamfer_thresholds_m:
        key = f"{float(threshold):.1f}"
        chamfer_ap_by_threshold_m[key] = eval_ap_threshold(
            gt_by_category_sample,
            gt_count,
            pred_by_category,
            categories,
            match_mode="chamfer",
            threshold=float(threshold),
            image_size=image_size,
            line_width_px=line_width_px,
            meter_per_pixel=meter_per_pixel,
        )

    apm = float(np.mean([mean_over_present(item, gt_count) for item in mask_ap_by_iou.values()])) if mask_ap_by_iou else 0.0
    result: Dict[str, Any] = {
        "samples": len(records),
        "categories": categories,
        "image_size": int(image_size),
        "meter_per_pixel": float(meter_per_pixel),
        "line_width_px": int(line_width_px),
        "score_mode": score_mode,
        "mIoU": miou,
        "APM": apm,
        "APM50": mean_over_present(mask_ap_by_iou.get("0.50", {}), gt_count),
        "APM75": mean_over_present(mask_ap_by_iou.get("0.75", {}), gt_count),
        "per_category": {
            "mIoU": miou_per_category,
            "APM50": mask_ap_by_iou.get("0.50", {}),
            "APM75": mask_ap_by_iou.get("0.75", {}),
        },
        "mask_ap_by_iou": mask_ap_by_iou,
        "chamfer_ap_by_threshold_m": chamfer_ap_by_threshold_m,
        "gt_count": gt_count,
    }
    for threshold in chamfer_thresholds_m:
        key = f"{float(threshold):.1f}"
        result[f"APC{key}"] = mean_over_present(chamfer_ap_by_threshold_m.get(key, {}), gt_count)
        result["per_category"][f"APC{key}"] = chamfer_ap_by_threshold_m.get(key, {})
    return result


def chamfer_symmetric(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1e6
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def continuity_score(lines: Sequence[Dict[str, Any]], tol: float = 12.0) -> float:
    endpoints: List[tuple[int, np.ndarray]] = []
    for idx, line in enumerate(lines):
        points = line_points(line)
        if len(points) < 2:
            continue
        endpoints.append((idx, points[0]))
        endpoints.append((idx, points[-1]))
    if not endpoints:
        return 0.0
    connected = 0
    for idx, (line_id, point) in enumerate(endpoints):
        best = 1e9
        for other_idx, (other_line_id, other_point) in enumerate(endpoints):
            if idx == other_idx or line_id == other_line_id:
                continue
            best = min(best, float(np.linalg.norm(point - other_point)))
        if best <= float(tol):
            connected += 1
    return connected / max(1, len(endpoints))


def engineering_sample_metrics(pred_lines: Sequence[Dict[str, Any]], gt_lines: Sequence[Dict[str, Any]], thresholds: Sequence[float]) -> Dict[str, float]:
    gt_by_category: Dict[str, List[np.ndarray]] = {}
    pred_by_category: Dict[str, List[np.ndarray]] = {}
    for line in gt_lines:
        gt_by_category.setdefault(normalize_category(line.get("category", "")), []).append(line_points(line))
    for line in pred_lines:
        pred_by_category.setdefault(normalize_category(line.get("category", "")), []).append(line_points(line))

    out: Dict[str, float] = {}
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        categories = set(gt_by_category.keys()) | set(pred_by_category.keys())
        for category in categories:
            gts = gt_by_category.get(category, [])
            preds = pred_by_category.get(category, [])
            used: set[int] = set()
            for pred in preds:
                best_gt = -1
                best_distance = 1e9
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in used:
                        continue
                    distance = chamfer_symmetric(pred, gt)
                    if distance < best_distance:
                        best_distance = distance
                        best_gt = gt_idx
                if best_gt >= 0 and best_distance <= float(threshold):
                    used.add(best_gt)
                    tp += 1
                else:
                    fp += 1
            fn += max(0, len(gts) - len(used))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        out[f"APC@{int(threshold)}px"] = float(f1)

    chamfers: List[float] = []
    for category in set(gt_by_category.keys()) | set(pred_by_category.keys()):
        gts = gt_by_category.get(category, [])
        for pred in pred_by_category.get(category, []):
            if not gts:
                chamfers.append(64.0)
            else:
                chamfers.append(min(chamfer_symmetric(pred, gt) for gt in gts))
    out["mean_chamfer_px"] = float(np.mean(chamfers)) if chamfers else 64.0
    out["continuity_pred"] = continuity_score(pred_lines, tol=12.0)
    out["continuity_gt"] = continuity_score(gt_lines, tol=12.0)
    out["continuity_gap"] = abs(out["continuity_pred"] - out["continuity_gt"])
    out["pred_num_lines"] = float(len(pred_lines))
    out["gt_num_lines"] = float(len(gt_lines))
    return out


def evaluate_engineering_records(records: Sequence[Dict[str, Any]], thresholds: Sequence[float]) -> Dict[str, float]:
    aggregated: Dict[str, List[float]] = {}
    for item in records:
        metrics = engineering_sample_metrics(item.get("pred_lines", []), item.get("gt_lines", []), thresholds)
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in aggregated.items()}


def jiangjihua_category_filter(categories: str | Sequence[str] | None) -> set[str]:
    if categories is None:
        return {"centerline"}
    if isinstance(categories, str):
        text = categories.strip().lower()
        if text in {"all", "lane_intersection", "combined", "*"}:
            return {"centerline", "intersection"}
        if text in {"lane", "centerline", "center_line"}:
            return {"centerline"}
        if text in {"intersection", "junction", "crossing"}:
            return {"intersection"}
        return {normalize_category(text)}
    out: set[str] = set()
    for item in categories:
        out.update(jiangjihua_category_filter(str(item)))
    return out


def jiangjihua_table_text(summary: Dict[str, Any], title: str) -> str:
    samples_num = int(summary.get("samples_num", 0) or 0)
    valid_string_format = int(summary.get("valid_string_format", 0) or 0)
    valid_ratio = valid_string_format / samples_num if samples_num else 0.0
    return "\n".join(
        [
            "=" * 58,
            f"{(' ' + title + ' '):^58}",
            "=" * 58,
            f"{'Metric':<18} {'Precision':<12} {'Recall':<12} {'F1':<12}",
            "-" * 58,
            (
                f"{'Instance Level':<18} "
                f"{float(summary.get('instance_pre', 0.0)):<12.4f} "
                f"{float(summary.get('instance_recall', 0.0)):<12.4f} "
                f"{float(summary.get('instance_f1', 0.0)):<12.4f}"
            ),
            (
                f"{'Length Level':<18} "
                f"{float(summary.get('length_pre', 0.0)):<12.4f} "
                f"{float(summary.get('length_recall', 0.0)):<12.4f} "
                f"{float(summary.get('length_f1', 0.0)):<12.4f}"
            ),
            "=" * 58,
            f"valid prediction format ratio: {valid_ratio:.4f}({valid_string_format}/{samples_num})",
        ]
    )


def record_lines_to_meter_linestrings(
    record: Dict[str, Any],
    line_key: str,
    *,
    categories: str | Sequence[str] | None,
    meter_per_pixel: float,
) -> List[Any]:
    try:
        from shapely.geometry import LineString
    except ImportError as exc:
        raise ImportError("jiangjihua metrics require shapely. Install with: pip install shapely scipy") from exc

    allowed_categories = jiangjihua_category_filter(categories)
    output: List[Any] = []
    for line in record.get(line_key, []):
        if not isinstance(line, dict):
            continue
        if normalize_category(line.get("category", "centerline")) not in allowed_categories:
            continue
        points = line_points(line)
        if points.shape[0] < 2:
            continue
        output.append(LineString((points * float(meter_per_pixel)).tolist()))
    return output


def jiangjihua_line_match_metric(line1: Any, line2: Any, buffer_size: float) -> float:
    poly1 = line1.buffer(float(buffer_size))
    poly2 = line2.buffer(float(buffer_size))
    union_area = poly1.union(poly2).area
    if union_area <= 0.0:
        return 0.0
    return float(poly1.intersection(poly2).area / union_area)


def jiangjihua_hungarian_match(
    gt_lines: Sequence[Any],
    pred_lines: Sequence[Any],
    *,
    buffer_size: float,
    match_threshold: float,
) -> tuple[List[int], List[int]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("jiangjihua metrics require scipy. Install with: pip install shapely scipy") from exc

    num_gt = len(gt_lines)
    num_pred = len(pred_lines)
    if num_gt == 0 or num_pred == 0:
        return [], []

    cost_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for gt_idx, gt_line in enumerate(gt_lines):
        for pred_idx, pred_line in enumerate(pred_lines):
            cost_matrix[gt_idx, pred_idx] = jiangjihua_line_match_metric(gt_line, pred_line, buffer_size)

    gt_indices, pred_indices = linear_sum_assignment(-cost_matrix)
    matched_gt: List[int] = []
    matched_pred: List[int] = []
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if float(cost_matrix[gt_idx, pred_idx]) < float(match_threshold):
            continue
        matched_gt.append(int(gt_idx))
        matched_pred.append(int(pred_idx))
    return matched_gt, matched_pred


def evaluate_jiangjihua_one_record(
    record: Dict[str, Any],
    *,
    categories: str | Sequence[str] | None,
    meter_per_pixel: float,
    buffer_size: float,
    match_threshold: float,
) -> Dict[str, Any]:
    valid_string_format = bool(record.get("parse_ok", True))
    try:
        gt_lines = record_lines_to_meter_linestrings(
            record,
            "gt_lines",
            categories=categories,
            meter_per_pixel=meter_per_pixel,
        )
    except Exception:
        gt_lines = []
        valid_string_format = False
    try:
        if not bool(record.get("parse_ok", True)):
            raise ValueError(record.get("parse_error") or "prediction parse_ok is false")
        pred_lines = record_lines_to_meter_linestrings(
            record,
            "pred_lines",
            categories=categories,
            meter_per_pixel=meter_per_pixel,
        )
    except Exception:
        pred_lines = []
        valid_string_format = False

    matched_gt, _ = jiangjihua_hungarian_match(
        gt_lines,
        pred_lines,
        buffer_size=buffer_size,
        match_threshold=match_threshold,
    )
    return {
        "gt_line_num": len(gt_lines),
        "gt_line_length_sum": float(sum(line.length for line in gt_lines)),
        "pred_line_num": len(pred_lines),
        "pred_line_length_sum": float(sum(line.length for line in pred_lines)),
        "matched_line_num": len(matched_gt),
        "matched_line_length_sum": float(sum(gt_lines[idx].length for idx in matched_gt)),
        "sample_num": 1,
        "valid_string_format": int(bool(valid_string_format)),
    }


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) else 0.0


def summarize_jiangjihua_sample_results(
    sample_results: Sequence[Dict[str, Any]],
    *,
    eval_name: str,
    category_filter: str | Sequence[str] | None,
    meter_per_pixel: float,
    buffer_size: float,
    match_threshold: float,
) -> Dict[str, Any]:
    totals = {
        "gt_line_num": 0,
        "gt_line_length_sum": 0.0,
        "pred_line_num": 0,
        "pred_line_length_sum": 0.0,
        "matched_line_num": 0,
        "matched_line_length_sum": 0.0,
        "sample_num": 0,
        "valid_string_format": 0,
    }
    for item in sample_results:
        for key in totals:
            totals[key] += item.get(key, 0)

    instance_pre = safe_div(totals["matched_line_num"], totals["pred_line_num"])
    instance_recall = safe_div(totals["matched_line_num"], totals["gt_line_num"])
    length_pre = safe_div(totals["matched_line_length_sum"], totals["pred_line_length_sum"])
    length_recall = safe_div(totals["matched_line_length_sum"], totals["gt_line_length_sum"])
    summary = {
        "instance_pre": round(instance_pre, 4),
        "instance_recall": round(instance_recall, 4),
        "instance_f1": round(2 * instance_pre * instance_recall / (instance_pre + instance_recall + 1e-6), 4),
        "length_pre": round(length_pre, 4),
        "length_recall": round(length_recall, 4),
        "length_f1": round(2 * length_pre * length_recall / (length_pre + length_recall + 1e-6), 4),
        "valid_string_format": int(totals["valid_string_format"]),
        "samples_num": int(totals["sample_num"]),
        "backend": "jiangjihua.infer_index.line_eval_compatible",
        "eval_name": eval_name,
        "category_filter": category_filter,
        "meter_per_pixel": float(meter_per_pixel),
        "buffer_size": float(buffer_size),
        "match_threshold": float(match_threshold),
        "raw_totals": totals,
    }
    summary["table"] = jiangjihua_table_text(summary, eval_name)
    return summary


def evaluate_jiangjihua_records(
    records: Sequence[Dict[str, Any]],
    *,
    categories: str | Sequence[str] | None,
    eval_name: str,
    meter_per_pixel: float,
    buffer_size: float,
    match_threshold: float,
    include_samples: bool = False,
) -> Dict[str, Any]:
    sample_results = [
        evaluate_jiangjihua_one_record(
            record,
            categories=categories,
            meter_per_pixel=meter_per_pixel,
            buffer_size=buffer_size,
            match_threshold=match_threshold,
        )
        for record in records
    ]
    summary = summarize_jiangjihua_sample_results(
        sample_results,
        eval_name=eval_name,
        category_filter=categories,
        meter_per_pixel=meter_per_pixel,
        buffer_size=buffer_size,
        match_threshold=match_threshold,
    )
    if include_samples:
        return {"summary": summary, "samples": sample_results}
    return summary


def evaluate_jiangjihua_map_records(
    records: Sequence[Dict[str, Any]],
    *,
    map_task: str,
    meter_per_pixel: float,
    buffer_size: float,
    match_threshold: float,
) -> Dict[str, Any]:
    if str(map_task) == "lane":
        return {
            "line_eval": evaluate_jiangjihua_records(
                records,
                categories="lane",
                eval_name="Line Evaluation Results",
                meter_per_pixel=meter_per_pixel,
                buffer_size=buffer_size,
                match_threshold=match_threshold,
            )
        }
    map_eval = {
        "lane": evaluate_jiangjihua_records(
            records,
            categories="lane",
            eval_name="Lane Evaluation Results",
            meter_per_pixel=meter_per_pixel,
            buffer_size=buffer_size,
            match_threshold=match_threshold,
        ),
        "intersection": evaluate_jiangjihua_records(
            records,
            categories="intersection",
            eval_name="Intersection Evaluation Results",
            meter_per_pixel=meter_per_pixel,
            buffer_size=buffer_size,
            match_threshold=match_threshold,
        ),
        "lane_intersection": evaluate_jiangjihua_records(
            records,
            categories="all",
            eval_name="Lane + Intersection Evaluation Results",
            meter_per_pixel=meter_per_pixel,
            buffer_size=buffer_size,
            match_threshold=match_threshold,
        ),
    }
    return {
        "centerline_eval": map_eval["lane"],
        "intersection_eval": map_eval["intersection"],
        "lane_intersection_eval": map_eval["lane_intersection"],
        "map_eval": map_eval,
    }


def resolve_image_path(record: Dict[str, Any], media_dir: Path | None) -> Path | None:
    image = str(record.get("image", "")).strip().replace("\\", "/")
    if not image:
        return None
    raw_path = Path(image)
    if raw_path.is_absolute() and raw_path.is_file():
        return raw_path
    if media_dir is None:
        return None
    candidates = [
        media_dir / image,
        media_dir / image.lstrip("/"),
        media_dir / "images" / image,
        media_dir / Path(image).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def draw_lines(image: Image.Image, lines: Sequence[Dict[str, Any]], colors: Dict[str, tuple[int, int, int]], width: int) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for line in lines:
        points = line.get("points", [])
        if len(points) < 2:
            continue
        category = normalize_category(line.get("category", "centerline"))
        color = colors.get(category, (255, 255, 255))
        xy = [(float(point[0]), float(point[1])) for point in points]
        draw.line(xy, fill=color, width=int(width), joint="curve")
        radius = max(2, int(width))
        for point in (xy[0], xy[-1]):
            x, y = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def label_panel(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    margin = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    box = (margin, margin, margin + bbox[2] - bbox[0] + 8, margin + bbox[3] - bbox[1] + 8)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((margin + 4, margin + 4), text, fill=(255, 255, 255), font=font)
    return canvas


def make_triptych(base: Image.Image, gt_lines: Sequence[Dict[str, Any]], pred_lines: Sequence[Dict[str, Any]], width: int) -> Image.Image:
    base = base.convert("RGB")
    gt = draw_lines(base, gt_lines, GT_COLORS, width)
    pred = draw_lines(base, pred_lines, PRED_COLORS, width)
    overlay = draw_lines(draw_lines(base, gt_lines, GT_COLORS, width), pred_lines, PRED_COLORS, max(2, int(width) - 1))
    panels = [
        label_panel(gt, "GT"),
        label_panel(pred, "Pred"),
        label_panel(overlay, "GT + Pred"),
    ]
    out = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), (0, 0, 0))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out


def save_visualizations(
    records: Sequence[Dict[str, Any]],
    *,
    media_dir: Path | None,
    out_dir: Path,
    limit: int,
    line_width_px: int,
    sheet_cols: int,
) -> Dict[str, Any]:
    vis_dir = out_dir / "visualization"
    individual_dir = vis_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    skipped_missing_image = 0
    for idx, record in enumerate(records[: max(0, int(limit))], start=1):
        image_path = resolve_image_path(record, media_dir)
        if image_path is None:
            skipped_missing_image += 1
            continue
        base = Image.open(image_path).convert("RGB")
        panel = make_triptych(base, record.get("gt_lines", []), record.get("pred_lines", []), int(line_width_px))
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(record.get("id", idx)))[:120]
        output_path = individual_dir / f"{idx:04d}_{safe_id}.png"
        panel.save(output_path)
        saved.append(str(output_path))

    sheet_paths: List[str] = []
    if saved:
        images = [Image.open(path).convert("RGB") for path in saved]
        cols = max(1, int(sheet_cols))
        rows = int(math.ceil(len(images) / cols))
        cell_w = max(image.width for image in images)
        cell_h = max(image.height for image in images)
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
        for idx, image in enumerate(images):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            sheet.paste(image, (x, y))
        sheet_path = vis_dir / "prediction_overlay_sheet.png"
        sheet.save(sheet_path)
        sheet_paths.append(str(sheet_path))

    manifest = {
        "individual_count": len(saved),
        "individual": saved,
        "sheets": sheet_paths,
        "skipped_missing_image": int(skipped_missing_image),
    }
    (vis_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    pred_jsonl = Path(args.pred_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir_text = str(args.media_dir).strip() or str(args.trainroot).strip()
    media_dir = Path(media_dir_text).expanduser().resolve() if media_dir_text else None

    raw_records = read_prediction_records(pred_jsonl)
    dataset_mode = trainroot_coord_mode(str(args.trainroot))
    coord_records = convert_record_coordinates(
        raw_records,
        image_size=int(args.image_size),
        coord_range=float(args.coord_range),
        gt_coord_mode=str(args.gt_coord_mode),
        pred_coord_mode=str(args.pred_coord_mode),
        dataset_mode=dataset_mode,
    )
    records = sanitize_records(coord_records, image_size=int(args.image_size))
    pred_json = out_dir / "predictions.json"
    pred_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    jiangjihua = evaluate_jiangjihua_map_records(
        records,
        map_task=str(args.map_task),
        meter_per_pixel=float(args.meter_per_pixel),
        buffer_size=float(args.jiangjihua_buffer_size),
        match_threshold=float(args.jiangjihua_match_threshold),
    )
    jiangjihua_path = out_dir / "eval_jiangjihua.json"
    jiangjihua_path.write_text(json.dumps(jiangjihua, ensure_ascii=False, indent=2), encoding="utf-8")

    categories = [str(item) for item in parse_csv(args.categories, str)]
    official = evaluate_official_records(
        records=records,
        categories=categories,
        image_size=int(args.image_size),
        meter_per_pixel=float(args.meter_per_pixel),
        line_width_px=int(args.line_width_px),
        mask_iou_thresholds=parse_csv(args.mask_iou_thresholds, float),
        chamfer_thresholds_m=parse_csv(args.chamfer_thresholds_m, float),
    )
    official_path = out_dir / "eval_official.json"
    official_path.write_text(json.dumps(official, ensure_ascii=False, indent=2), encoding="utf-8")

    engineering = evaluate_engineering_records(records, thresholds=parse_csv(args.engineering_thresholds_px, float))
    engineering_path = out_dir / "eval_engineering.json"
    engineering_path.write_text(json.dumps(engineering, ensure_ascii=False, indent=2), encoding="utf-8")

    vis_manifest = save_visualizations(
        records,
        media_dir=media_dir,
        out_dir=out_dir,
        limit=int(args.vis_limit),
        line_width_px=int(args.line_width_px),
        sheet_cols=int(args.sheet_cols),
    )
    summary = {
        "pred_jsonl": str(pred_jsonl),
        "pred_json": str(pred_json),
        "eval_jiangjihua_json": str(jiangjihua_path),
        "eval_official_json": str(official_path),
        "eval_engineering_json": str(engineering_path),
        "visualization": vis_manifest,
        "num_records": len(records),
        "coord_conversion": {
            "dataset_coord_mode": dataset_mode,
            "gt_coord_mode": str(args.gt_coord_mode),
            "pred_coord_mode": str(args.pred_coord_mode),
            "coord_range": float(args.coord_range),
            "image_size": int(args.image_size),
        },
        "parse_ok_rate": (
            sum(1 for item in records if bool(item.get("parse_ok"))) / max(1, len(records))
        ),
        "primary_metric": "jiangjihua.infer_index.line_eval",
        "jiangjihua": jiangjihua,
        "official": {
            "mIoU": official.get("mIoU"),
            "APM": official.get("APM"),
            "APM50": official.get("APM50"),
            "APM75": official.get("APM75"),
            "APC0.9": official.get("APC0.9"),
            "APC1.5": official.get("APC1.5"),
            "APC3.0": official.get("APC3.0"),
            "APC4.5": official.get("APC4.5"),
        },
        "engineering": engineering,
    }
    summary_path = out_dir / "eval_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
