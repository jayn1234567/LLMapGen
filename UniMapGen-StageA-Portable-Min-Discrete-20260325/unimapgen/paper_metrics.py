import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


MASK_IOU_THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.50, 1.00, 0.05).tolist())
CHAMFER_THRESHOLDS_M = (0.9, 1.5, 3.0, 4.5)


@dataclass
class InstanceRecord:
    sample_id: str
    category: str
    score: float
    mask: np.ndarray
    points: np.ndarray


def normalize_category(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text == "lane line":
        return "lane_line"
    if text == "virtual line":
        return "virtual_line"
    return text.replace(" ", "_")


def _coerce_point(raw: Any) -> List[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        x = float(raw[0])
        y = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return [x, y]


def _to_np_points(line: Dict[str, Any]) -> np.ndarray:
    raw_points = line.get("points", [])
    if isinstance(raw_points, np.ndarray):
        try:
            pts = np.asarray(raw_points, dtype=np.float32)
        except (TypeError, ValueError):
            pts = np.zeros((0, 2), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] == 2:
            return pts
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(raw_points, (list, tuple)):
        if len(raw_points) >= 2 and not isinstance(raw_points[0], (list, tuple, np.ndarray)):
            point = _coerce_point(raw_points)
            return np.asarray([point], dtype=np.float32) if point is not None else np.zeros((0, 2), dtype=np.float32)
        cleaned = []
        for raw in raw_points:
            point = _coerce_point(raw)
            if point is not None:
                cleaned.append(point)
        if cleaned:
            return np.asarray(cleaned, dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32)


def _infer_image_hw(item: Dict[str, Any], fallback_size: int) -> Tuple[int, int]:
    raw = item.get("image_size") or item.get("image_hw")
    if isinstance(raw, int):
        size = max(int(raw), 1)
        return size, size
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        width = max(int(raw[0]), 1)
        height = max(int(raw[1]), 1)
        return width, height
    max_x = 0.0
    max_y = 0.0
    for key in ("pred_lines", "gt_lines"):
        for line in item.get(key, []):
            pts = _to_np_points(line)
            if len(pts) == 0:
                continue
            max_x = max(max_x, float(pts[:, 0].max()))
            max_y = max(max_y, float(pts[:, 1].max()))
    if max_x > 0.0 and max_y > 0.0:
        return int(math.ceil(max_x)) + 1, int(math.ceil(max_y)) + 1
    return int(fallback_size), int(fallback_size)


def _draw_polyline_mask(points: np.ndarray, image_hw: Tuple[int, int], line_width_px: int) -> np.ndarray:
    width, height = image_hw
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    pts = [(float(x), float(y)) for x, y in points.tolist()]
    if len(pts) == 1:
        radius = max(int(line_width_px // 2), 1)
        x, y = pts[0]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    elif len(pts) >= 2:
        draw.line(pts, fill=1, width=int(line_width_px), joint="curve")
        radius = max(int(line_width_px // 2), 1)
        for x, y in (pts[0], pts[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def _semantic_mask(lines: Sequence[Dict[str, Any]], image_hw: Tuple[int, int], category: str, line_width_px: int) -> np.ndarray:
    width, height = image_hw
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    wanted = normalize_category(category)
    for line in lines:
        if normalize_category(line.get("category", "unknown")) != wanted:
            continue
        pts = _to_np_points(line)
        if len(pts) == 0:
            continue
        poly = [(float(x), float(y)) for x, y in pts.tolist()]
        if len(poly) == 1:
            radius = max(int(line_width_px // 2), 1)
            x, y = poly[0]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        else:
            draw.line(poly, fill=1, width=int(line_width_px), joint="curve")
            radius = max(int(line_width_px // 2), 1)
            for x, y in (poly[0], poly[-1]):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def _densify_polyline(points: np.ndarray, step_px: float = 1.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return pts.copy()
    out: List[np.ndarray] = [pts[0]]
    step = max(float(step_px), 1e-3)
    for idx in range(pts.shape[0] - 1):
        p0 = pts[idx]
        p1 = pts[idx + 1]
        seg = p1 - p0
        dist = float(np.linalg.norm(seg))
        if dist <= 1e-6:
            continue
        steps = max(int(math.ceil(dist / step)), 1)
        for step_idx in range(1, steps + 1):
            t = float(step_idx) / float(steps)
            out.append((1.0 - t) * p0 + t * p1)
    return np.asarray(out, dtype=np.float32)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 0.0 if union <= 0.0 else inter / union


def _chamfer_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1e6
    d1 = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    return float(0.5 * (d1.min(axis=1).mean() + d1.min(axis=0).mean()))


def _ap_from_tp_fp(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt <= 0:
        return float("nan")
    tp_cum = np.cumsum(tp, axis=0)
    fp_cum = np.cumsum(fp, axis=0)
    recall = tp_cum / max(float(total_gt), 1.0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall_points = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    precisions = np.zeros_like(recall_points)
    for idx, recall_point in enumerate(recall_points):
        valid = precision[recall >= recall_point]
        precisions[idx] = float(valid.max()) if valid.size else 0.0
    return float(np.mean(precisions))


def _collect_categories(items: Sequence[Dict[str, Any]], categories: Optional[Sequence[str]]) -> List[str]:
    if categories:
        return [normalize_category(item) for item in categories]
    found = set()
    for item in items:
        for key in ("pred_lines", "gt_lines"):
            for line in item.get(key, []):
                found.add(normalize_category(line.get("category", "unknown")))
    return sorted(found) if found else ["road"]


def _line_score(line: Dict[str, Any], item: Dict[str, Any]) -> float:
    for key in ("score", "confidence", "line_score"):
        value = line.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    for key in ("sample_score", "generation_score", "confidence"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 1.0


def _build_instances(
    items: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    line_width_px: int,
    default_image_size: int,
) -> Tuple[List[InstanceRecord], Dict[str, Dict[Tuple[str, str], List[InstanceRecord]]], Dict[str, float], Dict[str, float]]:
    preds: List[InstanceRecord] = []
    gts_by_cat: Dict[str, Dict[Tuple[str, str], List[InstanceRecord]]] = {category: {} for category in categories}
    gt_counts: Dict[str, float] = {category: 0.0 for category in categories}
    pred_counts: Dict[str, float] = {category: 0.0 for category in categories}
    for item_idx, item in enumerate(items):
        sample_id = str(item.get("id") or item.get("sample_id") or item_idx)
        image_hw = _infer_image_hw(item, fallback_size=default_image_size)
        for line in item.get("pred_lines", []):
            category = normalize_category(line.get("category", "unknown"))
            if category not in gts_by_cat:
                continue
            pts = _densify_polyline(_to_np_points(line), step_px=1.0)
            if len(pts) == 0:
                continue
            record = InstanceRecord(
                sample_id=sample_id,
                category=category,
                score=_line_score(line, item),
                mask=_draw_polyline_mask(pts, image_hw=image_hw, line_width_px=line_width_px),
                points=pts,
            )
            preds.append(record)
            pred_counts[category] += 1.0
        for line in item.get("gt_lines", []):
            category = normalize_category(line.get("category", "unknown"))
            if category not in gts_by_cat:
                continue
            pts = _densify_polyline(_to_np_points(line), step_px=1.0)
            if len(pts) == 0:
                continue
            record = InstanceRecord(
                sample_id=sample_id,
                category=category,
                score=1.0,
                mask=_draw_polyline_mask(pts, image_hw=image_hw, line_width_px=line_width_px),
                points=pts,
            )
            gts_by_cat[category].setdefault((sample_id, category), []).append(record)
            gt_counts[category] += 1.0
    preds.sort(key=lambda item: float(item.score), reverse=True)
    return preds, gts_by_cat, pred_counts, gt_counts


def evaluate_prediction_items(
    items: Sequence[Dict[str, Any]],
    meter_per_pixel: float = 0.15,
    line_width_px: int = 6,
    categories: Optional[Sequence[str]] = None,
    default_image_size: int = 896,
) -> Dict[str, float]:
    categories_list = _collect_categories(items, categories)
    preds, gts_by_cat, pred_counts, gt_counts = _build_instances(
        items=items,
        categories=categories_list,
        line_width_px=int(line_width_px),
        default_image_size=int(default_image_size),
    )

    intersections = {category: 0.0 for category in categories_list}
    unions = {category: 0.0 for category in categories_list}
    for item in items:
        image_hw = _infer_image_hw(item, fallback_size=default_image_size)
        for category in categories_list:
            pred_mask = _semantic_mask(item.get("pred_lines", []), image_hw=image_hw, category=category, line_width_px=line_width_px)
            gt_mask = _semantic_mask(item.get("gt_lines", []), image_hw=image_hw, category=category, line_width_px=line_width_px)
            intersections[category] += float(np.logical_and(pred_mask, gt_mask).sum())
            unions[category] += float(np.logical_or(pred_mask, gt_mask).sum())

    out: Dict[str, float] = {}
    ious: List[float] = []
    for category in categories_list:
        union = unions[category]
        if union <= 0.0:
            continue
        iou = intersections[category] / union
        out[f"IoU_{category}"] = float(iou)
        ious.append(float(iou))
    out["mIoU"] = float(np.mean(ious)) if ious else 0.0

    preds_by_cat = {category: [pred for pred in preds if pred.category == category] for category in categories_list}
    ap_mask_means: List[float] = []
    ap_mask_50: List[float] = []
    ap_mask_75: List[float] = []
    ap_chamfer_means: List[float] = []
    ap_chamfer_by_threshold = {threshold: [] for threshold in CHAMFER_THRESHOLDS_M}

    for category in categories_list:
        category_preds = preds_by_cat.get(category, [])
        category_gts = gts_by_cat.get(category, {})
        total_gt = int(gt_counts.get(category, 0.0))
        if total_gt <= 0:
            continue

        tp_masks = np.zeros((len(category_preds), len(MASK_IOU_THRESHOLDS)), dtype=np.float32)
        fp_masks = np.zeros_like(tp_masks)
        tp_chamfer = np.zeros((len(category_preds), len(CHAMFER_THRESHOLDS_M)), dtype=np.float32)
        fp_chamfer = np.zeros_like(tp_chamfer)
        used_mask = {threshold: set() for threshold in MASK_IOU_THRESHOLDS}
        used_chamfer = {threshold: set() for threshold in CHAMFER_THRESHOLDS_M}

        for pred_idx, pred in enumerate(category_preds):
            gt_candidates = category_gts.get((pred.sample_id, category), [])
            mask_ious = [_mask_iou(pred.mask, gt.mask) for gt in gt_candidates]
            chamfers_px = [_chamfer_distance_px(pred.points, gt.points) for gt in gt_candidates]
            chamfers_m = [distance * float(meter_per_pixel) for distance in chamfers_px]

            for threshold_idx, threshold in enumerate(MASK_IOU_THRESHOLDS):
                best_gt_idx = -1
                best_value = -1.0
                for gt_idx, score in enumerate(mask_ious):
                    key = (pred.sample_id, gt_idx)
                    if score > best_value and score >= threshold and key not in used_mask[threshold]:
                        best_value = float(score)
                        best_gt_idx = gt_idx
                if best_gt_idx >= 0:
                    tp_masks[pred_idx, threshold_idx] = 1.0
                    used_mask[threshold].add((pred.sample_id, best_gt_idx))
                else:
                    fp_masks[pred_idx, threshold_idx] = 1.0

            for threshold_idx, threshold in enumerate(CHAMFER_THRESHOLDS_M):
                best_gt_idx = -1
                best_value = float("inf")
                for gt_idx, distance_m in enumerate(chamfers_m):
                    key = (pred.sample_id, gt_idx)
                    if distance_m < best_value and distance_m <= threshold and key not in used_chamfer[threshold]:
                        best_value = float(distance_m)
                        best_gt_idx = gt_idx
                if best_gt_idx >= 0:
                    tp_chamfer[pred_idx, threshold_idx] = 1.0
                    used_chamfer[threshold].add((pred.sample_id, best_gt_idx))
                else:
                    fp_chamfer[pred_idx, threshold_idx] = 1.0

        ap_masks = [_ap_from_tp_fp(tp_masks[:, idx], fp_masks[:, idx], total_gt) for idx in range(tp_masks.shape[1])]
        ap_mask_means.append(float(np.nanmean(ap_masks)) if ap_masks else float("nan"))
        ap_mask_50.append(float(ap_masks[0]))
        ap_mask_75.append(float(ap_masks[5]))

        ap_chamfers = [_ap_from_tp_fp(tp_chamfer[:, idx], fp_chamfer[:, idx], total_gt) for idx in range(tp_chamfer.shape[1])]
        ap_chamfer_means.append(float(np.nanmean(ap_chamfers)) if ap_chamfers else float("nan"))
        for threshold, value in zip(CHAMFER_THRESHOLDS_M, ap_chamfers):
            ap_chamfer_by_threshold[threshold].append(float(value))

    out["APM"] = float(np.nanmean(ap_mask_means)) if ap_mask_means else 0.0
    out["APM_50"] = float(np.nanmean(ap_mask_50)) if ap_mask_50 else 0.0
    out["APM_75"] = float(np.nanmean(ap_mask_75)) if ap_mask_75 else 0.0
    out["APC_mean"] = float(np.nanmean(ap_chamfer_means)) if ap_chamfer_means else 0.0
    for threshold in CHAMFER_THRESHOLDS_M:
        out[f"APC_{threshold}"] = float(np.nanmean(ap_chamfer_by_threshold[threshold])) if ap_chamfer_by_threshold[threshold] else 0.0
    out["pred_num_lines"] = float(np.mean(list(pred_counts.values()))) if pred_counts else 0.0
    out["gt_num_lines"] = float(np.mean(list(gt_counts.values()))) if gt_counts else 0.0
    return out
