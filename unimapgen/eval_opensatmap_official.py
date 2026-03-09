import argparse
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from unimapgen.qwen_map_pipeline import save_json
from unimapgen.utils import load_yaml


def _parse_float_list(text: str) -> List[float]:
    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(float(chunk))
    return out


def _to_points_px(line: Dict) -> np.ndarray:
    pts = np.asarray(line.get("points", []), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return pts


def _line_score(line: Dict) -> Tuple[float, bool]:
    for key in ("score", "line_score", "pseudo_score"):
        val = line.get(key, None)
        if isinstance(val, (int, float)):
            return float(val), False
    return 1.0, True


def _densify_polyline(points: np.ndarray, step: float) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return arr
    out = [arr[0]]
    max_step = max(float(step), 1e-3)
    for i in range(arr.shape[0] - 1):
        p0 = arr[i]
        p1 = arr[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        n = max(1, int(np.ceil(seg_len / max_step)))
        for k in range(1, n + 1):
            t = float(k) / float(n)
            out.append(p0 * (1.0 - t) + p1 * t)
    return np.asarray(out, dtype=np.float32)


def _draw_line_mask(points_px: np.ndarray, image_size: int, line_width_px: int) -> np.ndarray:
    img = Image.new("L", (int(image_size), int(image_size)), 0)
    draw = ImageDraw.Draw(img)
    pts = np.asarray(points_px, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((int(image_size), int(image_size)), dtype=bool)
    if pts.shape[0] == 1:
        x = float(pts[0, 0])
        y = float(pts[0, 1])
        r = max(1.0, 0.5 * float(line_width_px))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    else:
        draw.line([tuple(map(float, p)) for p in pts], fill=255, width=int(line_width_px))
        r = max(1.0, 0.5 * float(line_width_px))
        for p in (pts[0], pts[-1]):
            x = float(p[0])
            y = float(p[1])
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(img, dtype=np.uint8) > 0


def _semantic_masks(lines: Sequence[Dict], categories: Sequence[str], image_size: int, line_width_px: int) -> Dict[str, np.ndarray]:
    masks = {
        str(cat): np.zeros((int(image_size), int(image_size)), dtype=bool)
        for cat in categories
    }
    for line in lines:
        cat = str(line.get("category", ""))
        if cat not in masks:
            continue
        masks[cat] |= _draw_line_mask(_to_points_px(line), image_size=image_size, line_width_px=line_width_px)
    return masks


def _compute_semantic_miou(items: Sequence[Dict], categories: Sequence[str], image_size: int, line_width_px: int) -> Tuple[float, Dict[str, float]]:
    inter = {str(cat): 0.0 for cat in categories}
    union = {str(cat): 0.0 for cat in categories}
    for item in items:
        pred_masks = _semantic_masks(item.get("pred_lines", []), categories=categories, image_size=image_size, line_width_px=line_width_px)
        gt_masks = _semantic_masks(item.get("gt_lines", []), categories=categories, image_size=image_size, line_width_px=line_width_px)
        for cat in categories:
            pred = pred_masks[str(cat)]
            gt = gt_masks[str(cat)]
            inter[str(cat)] += float(np.logical_and(pred, gt).sum())
            union[str(cat)] += float(np.logical_or(pred, gt).sum())
    per_cat = {}
    for cat in categories:
        if union[str(cat)] <= 0.0:
            continue
        per_cat[str(cat)] = inter[str(cat)] / union[str(cat)]
    miou = float(np.mean(list(per_cat.values()))) if per_cat else 0.0
    return miou, per_cat


def _build_gt_index(items: Sequence[Dict], categories: Sequence[str]) -> Tuple[Dict[str, Dict[int, List[Dict]]], Dict[str, int]]:
    gt_by_cat_sample: Dict[str, Dict[int, List[Dict]]] = {str(cat): {} for cat in categories}
    gt_count = {str(cat): 0 for cat in categories}
    for sample_idx, item in enumerate(items):
        for gt_idx, line in enumerate(item.get("gt_lines", [])):
            cat = str(line.get("category", ""))
            if cat not in gt_by_cat_sample:
                continue
            gt_by_cat_sample[cat].setdefault(sample_idx, []).append(
                {
                    "sample_idx": sample_idx,
                    "gt_idx": gt_idx,
                    "line": line,
                }
            )
            gt_count[cat] += 1
    return gt_by_cat_sample, gt_count


def _build_pred_index(items: Sequence[Dict], categories: Sequence[str]) -> Tuple[Dict[str, List[Dict]], str]:
    pred_by_cat: Dict[str, List[Dict]] = {str(cat): [] for cat in categories}
    used_fallback = False
    used_explicit = False
    for sample_idx, item in enumerate(items):
        for pred_idx, line in enumerate(item.get("pred_lines", [])):
            cat = str(line.get("category", ""))
            if cat not in pred_by_cat:
                continue
            score, is_fallback = _line_score(line)
            used_fallback = used_fallback or is_fallback
            used_explicit = used_explicit or (not is_fallback)
            pred_by_cat[cat].append(
                {
                    "sample_idx": sample_idx,
                    "pred_idx": pred_idx,
                    "score": float(score),
                    "line": line,
                }
            )
    if used_explicit and used_fallback:
        score_mode = "mixed_explicit_and_constant_fallback"
    elif used_explicit:
        score_mode = "explicit_line_scores"
    else:
        score_mode = "constant_1.0_fallback"
    return pred_by_cat, score_mode


def _mask_iou(pred_line: Dict, gt_line: Dict, image_size: int, line_width_px: int) -> float:
    pred_mask = _draw_line_mask(_to_points_px(pred_line), image_size=image_size, line_width_px=line_width_px)
    gt_mask = _draw_line_mask(_to_points_px(gt_line), image_size=image_size, line_width_px=line_width_px)
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _chamfer_distance_m(pred_line: Dict, gt_line: Dict, meter_per_pixel: float, densify_step_m: float = 0.25) -> float:
    pred_pts = _to_points_px(pred_line) * float(meter_per_pixel)
    gt_pts = _to_points_px(gt_line) * float(meter_per_pixel)
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return 1e6
    pred_pts = _densify_polyline(pred_pts, step=densify_step_m)
    gt_pts = _densify_polyline(gt_pts, step=densify_step_m)
    dmat = np.sqrt(((pred_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=-1))
    return float(0.5 * (dmat.min(axis=1).mean() + dmat.min(axis=0).mean()))


def _average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if int(num_gt) <= 0:
        return 0.0
    tp_cum = np.cumsum(tp, dtype=np.float64)
    fp_cum = np.cumsum(fp, dtype=np.float64)
    rec = tp_cum / max(float(num_gt), 1.0)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.shape[0] - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _eval_ap_threshold(
    gt_by_cat_sample: Dict[str, Dict[int, List[Dict]]],
    gt_count: Dict[str, int],
    pred_by_cat: Dict[str, List[Dict]],
    categories: Sequence[str],
    match_mode: str,
    threshold: float,
    image_size: int,
    line_width_px: int,
    meter_per_pixel: float,
) -> Dict[str, float]:
    out = {}
    for cat in categories:
        cat = str(cat)
        num_gt = int(gt_count.get(cat, 0))
        if num_gt <= 0:
            continue
        matched = {
            sample_idx: np.zeros(len(lines), dtype=bool)
            for sample_idx, lines in gt_by_cat_sample.get(cat, {}).items()
        }
        preds = sorted(
            pred_by_cat.get(cat, []),
            key=lambda x: (-float(x["score"]), int(x["sample_idx"]), int(x["pred_idx"])),
        )
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for i, pred in enumerate(preds):
            sample_idx = int(pred["sample_idx"])
            gts = gt_by_cat_sample.get(cat, {}).get(sample_idx, [])
            best_val = None
            best_j = -1
            for j, gt in enumerate(gts):
                if bool(matched[sample_idx][j]):
                    continue
                if match_mode == "mask_iou":
                    val = _mask_iou(pred["line"], gt["line"], image_size=image_size, line_width_px=line_width_px)
                    better = (best_val is None) or (val > best_val)
                    is_match = val >= float(threshold)
                else:
                    val = _chamfer_distance_m(pred["line"], gt["line"], meter_per_pixel=meter_per_pixel)
                    better = (best_val is None) or (val < best_val)
                    is_match = val <= float(threshold)
                if better:
                    best_val = val
                    best_j = j
                    best_match = is_match
            if best_j >= 0 and best_match:
                matched[sample_idx][best_j] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0
        out[cat] = _average_precision(tp=tp, fp=fp, num_gt=num_gt)
    return out


def evaluate_prediction_json(
    prediction_json: str,
    categories: Sequence[str],
    image_size: int,
    meter_per_pixel: float,
    line_width_px: int,
    mask_iou_thresholds: Sequence[float],
    chamfer_thresholds_m: Sequence[float],
) -> Dict:
    with open(prediction_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Prediction json must be a list: {prediction_json}")

    categories = [str(cat) for cat in categories]
    miou, miou_per_cat = _compute_semantic_miou(
        items=items,
        categories=categories,
        image_size=image_size,
        line_width_px=line_width_px,
    )
    gt_by_cat_sample, gt_count = _build_gt_index(items=items, categories=categories)
    pred_by_cat, score_mode = _build_pred_index(items=items, categories=categories)

    mask_ap_per_thr = {}
    for thr in mask_iou_thresholds:
        key = f"{thr:.2f}"
        mask_ap_per_thr[key] = _eval_ap_threshold(
            gt_by_cat_sample=gt_by_cat_sample,
            gt_count=gt_count,
            pred_by_cat=pred_by_cat,
            categories=categories,
            match_mode="mask_iou",
            threshold=float(thr),
            image_size=image_size,
            line_width_px=line_width_px,
            meter_per_pixel=meter_per_pixel,
        )

    chamfer_ap_per_thr = {}
    for thr in chamfer_thresholds_m:
        key = f"{thr:.1f}"
        chamfer_ap_per_thr[key] = _eval_ap_threshold(
            gt_by_cat_sample=gt_by_cat_sample,
            gt_count=gt_count,
            pred_by_cat=pred_by_cat,
            categories=categories,
            match_mode="chamfer",
            threshold=float(thr),
            image_size=image_size,
            line_width_px=line_width_px,
            meter_per_pixel=meter_per_pixel,
        )

    def _mean_over_present(per_cat: Dict[str, float]) -> float:
        vals = [float(v) for k, v in per_cat.items() if int(gt_count.get(str(k), 0)) > 0]
        return float(np.mean(vals)) if vals else 0.0

    apm = float(np.mean([_mean_over_present(x) for x in mask_ap_per_thr.values()])) if mask_ap_per_thr else 0.0
    apm50 = _mean_over_present(mask_ap_per_thr.get("0.50", {}))
    apm75 = _mean_over_present(mask_ap_per_thr.get("0.75", {}))

    result = {
        "prediction_json": prediction_json,
        "samples": len(items),
        "categories": categories,
        "image_size": int(image_size),
        "meter_per_pixel": float(meter_per_pixel),
        "line_width_px": int(line_width_px),
        "score_mode": score_mode,
        "mIoU": miou,
        "APM": apm,
        "APM50": apm50,
        "APM75": apm75,
        "per_category": {
            "mIoU": miou_per_cat,
            "APM50": mask_ap_per_thr.get("0.50", {}),
            "APM75": mask_ap_per_thr.get("0.75", {}),
        },
        "mask_ap_by_iou": mask_ap_per_thr,
        "chamfer_ap_by_threshold_m": chamfer_ap_per_thr,
    }
    for thr in chamfer_thresholds_m:
        key = f"{thr:.1f}"
        result[f"APC{key}"] = _mean_over_present(chamfer_ap_per_thr.get(key, {}))
        result["per_category"][f"APC{key}"] = chamfer_ap_per_thr.get(key, {})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prediction_json", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--line_width_px", type=int, default=6)
    parser.add_argument("--mask_iou_thresholds", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--chamfer_thresholds_m", type=str, default="0.9,1.5,3.0,4.5")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    categories = list(cfg["serialization"]["categories"])
    image_size = int(cfg["data"]["image_size"])
    meter_per_pixel = float(cfg["data"].get("meter_per_pixel", 0.15))

    result = evaluate_prediction_json(
        prediction_json=args.prediction_json,
        categories=categories,
        image_size=image_size,
        meter_per_pixel=meter_per_pixel,
        line_width_px=int(args.line_width_px),
        mask_iou_thresholds=_parse_float_list(args.mask_iou_thresholds),
        chamfer_thresholds_m=_parse_float_list(args.chamfer_thresholds_m),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        save_json(args.output, result)


if __name__ == "__main__":
    main()
