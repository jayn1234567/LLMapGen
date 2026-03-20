import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
            pt = _coerce_point(raw_points)
            return np.asarray([pt], dtype=np.float32) if pt is not None else np.zeros((0, 2), dtype=np.float32)
        cleaned = []
        for raw in raw_points:
            pt = _coerce_point(raw)
            if pt is not None:
                cleaned.append(pt)
        if cleaned:
            return np.asarray(cleaned, dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32)


def _infer_image_hw(item: Dict[str, Any], fallback_size: int) -> Tuple[int, int]:
    raw = item.get("image_size") or item.get("image_hw")
    if isinstance(raw, int):
        size = max(int(raw), 1)
        return size, size
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        w = max(int(raw[0]), 1)
        h = max(int(raw[1]), 1)
        return w, h

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
        r = max(int(line_width_px // 2), 1)
        x, y = pts[0]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
    elif len(pts) >= 2:
        draw.line(pts, fill=1, width=int(line_width_px), joint="curve")
        r = max(int(line_width_px // 2), 1)
        for x, y in (pts[0], pts[-1]):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
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
            r = max(int(line_width_px // 2), 1)
            x, y = poly[0]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        else:
            draw.line(poly, fill=1, width=int(line_width_px), joint="curve")
            r = max(int(line_width_px // 2), 1)
            for x, y in (poly[0], poly[-1]):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
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
        for j in range(1, steps + 1):
            t = float(j) / float(steps)
            out.append((1.0 - t) * p0 + t * p1)
    return np.asarray(out, dtype=np.float32)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


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
    for idx, rp in enumerate(recall_points):
        valid = precision[recall >= rp]
        precisions[idx] = float(valid.max()) if valid.size else 0.0
    return float(np.mean(precisions))


def _collect_categories(items: Sequence[Dict[str, Any]], categories: Optional[Sequence[str]]) -> List[str]:
    if categories:
        return [normalize_category(x) for x in categories]
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
    gts_by_cat: Dict[str, Dict[Tuple[str, str], List[InstanceRecord]]] = {cat: {} for cat in categories}
    gt_counts: Dict[str, float] = {cat: 0.0 for cat in categories}
    pred_counts: Dict[str, float] = {cat: 0.0 for cat in categories}
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
            rec = InstanceRecord(
                sample_id=sample_id,
                category=category,
                score=_line_score(line, item),
                mask=_draw_polyline_mask(pts, image_hw=image_hw, line_width_px=line_width_px),
                points=pts,
            )
            preds.append(rec)
            pred_counts[category] += 1.0
        for line in item.get("gt_lines", []):
            category = normalize_category(line.get("category", "unknown"))
            if category not in gts_by_cat:
                continue
            pts = _densify_polyline(_to_np_points(line), step_px=1.0)
            if len(pts) == 0:
                continue
            rec = InstanceRecord(
                sample_id=sample_id,
                category=category,
                score=1.0,
                mask=_draw_polyline_mask(pts, image_hw=image_hw, line_width_px=line_width_px),
                points=pts,
            )
            gts_by_cat[category].setdefault((sample_id, category), []).append(rec)
            gt_counts[category] += 1.0
    preds.sort(key=lambda x: (-x.score, x.sample_id, x.category))
    return preds, gts_by_cat, pred_counts, gt_counts


def _mask_ap_for_category(
    preds: Sequence[InstanceRecord],
    gts_by_sample: Dict[Tuple[str, str], List[InstanceRecord]],
    threshold: float,
) -> float:
    total_gt = sum(len(v) for v in gts_by_sample.values())
    if total_gt <= 0:
        return float("nan")
    matched = {key: np.zeros(len(v), dtype=bool) for key, v in gts_by_sample.items()}
    tp: List[float] = []
    fp: List[float] = []
    for pred in preds:
        key = (pred.sample_id, pred.category)
        gt_list = gts_by_sample.get(key, [])
        best_idx = -1
        best_iou = -1.0
        for idx, gt in enumerate(gt_list):
            if matched[key][idx]:
                continue
            iou = _mask_iou(pred.mask, gt.mask)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= threshold:
            matched[key][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    return _ap_from_tp_fp(np.asarray(tp, dtype=np.float32), np.asarray(fp, dtype=np.float32), total_gt=total_gt)


def _chamfer_ap_for_category(
    preds: Sequence[InstanceRecord],
    gts_by_sample: Dict[Tuple[str, str], List[InstanceRecord]],
    threshold_px: float,
) -> float:
    total_gt = sum(len(v) for v in gts_by_sample.values())
    if total_gt <= 0:
        return float("nan")
    matched = {key: np.zeros(len(v), dtype=bool) for key, v in gts_by_sample.items()}
    tp: List[float] = []
    fp: List[float] = []
    for pred in preds:
        key = (pred.sample_id, pred.category)
        gt_list = gts_by_sample.get(key, [])
        best_idx = -1
        best_dist = 1e9
        for idx, gt in enumerate(gt_list):
            if matched[key][idx]:
                continue
            dist = _chamfer_distance_px(pred.points, gt.points)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0 and best_dist <= threshold_px:
            matched[key][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    return _ap_from_tp_fp(np.asarray(tp, dtype=np.float32), np.asarray(fp, dtype=np.float32), total_gt=total_gt)


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

    intersections = {cat: 0.0 for cat in categories_list}
    unions = {cat: 0.0 for cat in categories_list}
    for item in items:
        image_hw = _infer_image_hw(item, fallback_size=default_image_size)
        for cat in categories_list:
            pred_mask = _semantic_mask(item.get("pred_lines", []), image_hw=image_hw, category=cat, line_width_px=line_width_px)
            gt_mask = _semantic_mask(item.get("gt_lines", []), image_hw=image_hw, category=cat, line_width_px=line_width_px)
            intersections[cat] += float(np.logical_and(pred_mask, gt_mask).sum())
            unions[cat] += float(np.logical_or(pred_mask, gt_mask).sum())

    out: Dict[str, float] = {}
    ious: List[float] = []
    for cat in categories_list:
        union = unions[cat]
        if union <= 0.0:
            continue
        iou = intersections[cat] / union
        out[f"IoU_{cat}"] = float(iou)
        ious.append(float(iou))
    out["mIoU"] = float(np.mean(ious)) if ious else 0.0

    preds_by_cat = {cat: [p for p in preds if p.category == cat] for cat in categories_list}
    ap_mask_means: List[float] = []
    ap_mask_50: List[float] = []
    ap_mask_75: List[float] = []
    for cat in categories_list:
        cat_preds = preds_by_cat[cat]
        cat_gts = gts_by_cat[cat]
        aps = [
            _mask_ap_for_category(cat_preds, cat_gts, threshold=float(th))
            for th in MASK_IOU_THRESHOLDS
        ]
        valid_aps = [x for x in aps if not math.isnan(x)]
        if valid_aps:
            ap_mask_means.append(float(np.mean(valid_aps)))
        ap50 = _mask_ap_for_category(cat_preds, cat_gts, threshold=0.50)
        ap75 = _mask_ap_for_category(cat_preds, cat_gts, threshold=0.75)
        if not math.isnan(ap50):
            ap_mask_50.append(float(ap50))
        if not math.isnan(ap75):
            ap_mask_75.append(float(ap75))
    out["APM"] = float(np.mean(ap_mask_means)) if ap_mask_means else 0.0
    out["APM_50"] = float(np.mean(ap_mask_50)) if ap_mask_50 else 0.0
    out["APM_75"] = float(np.mean(ap_mask_75)) if ap_mask_75 else 0.0

    for threshold_m in CHAMFER_THRESHOLDS_M:
        cat_scores: List[float] = []
        threshold_px = float(threshold_m) / max(float(meter_per_pixel), 1e-6)
        for cat in categories_list:
            ap = _chamfer_ap_for_category(preds_by_cat[cat], gts_by_cat[cat], threshold_px=threshold_px)
            if not math.isnan(ap):
                cat_scores.append(float(ap))
        out[f"APC_{threshold_m:.1f}"] = float(np.mean(cat_scores)) if cat_scores else 0.0

    out["num_categories"] = float(len(categories_list))
    out["num_pred_instances"] = float(sum(pred_counts.values()))
    out["num_gt_instances"] = float(sum(gt_counts.values()))
    out["meter_per_pixel"] = float(meter_per_pixel)
    out["line_width_px"] = float(line_width_px)
    return out


def evaluate_prediction_json(
    prediction_json: str,
    meter_per_pixel: float = 0.15,
    line_width_px: int = 6,
    categories: Optional[Sequence[str]] = None,
    default_image_size: int = 896,
) -> Dict[str, float]:
    with open(prediction_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"prediction json must be a list: {prediction_json}")
    return evaluate_prediction_items(
        items=items,
        meter_per_pixel=meter_per_pixel,
        line_width_px=line_width_px,
        categories=categories,
        default_image_size=default_image_size,
    )


def _parse_categories(raw: str) -> Optional[List[str]]:
    values = [x.strip() for x in str(raw).split(",") if x.strip()]
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction json with UniMapGen paper-style metrics.")
    parser.add_argument("--prediction-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--meter-per-pixel", type=float, default=0.15)
    parser.add_argument("--line-width-px", type=int, default=6)
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument("--default-image-size", type=int, default=896)
    args = parser.parse_args()

    metrics = evaluate_prediction_json(
        prediction_json=args.prediction_json,
        meter_per_pixel=float(args.meter_per_pixel),
        line_width_px=int(args.line_width_px),
        categories=_parse_categories(args.categories),
        default_image_size=int(args.default_image_size),
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if str(args.output_json).strip():
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
