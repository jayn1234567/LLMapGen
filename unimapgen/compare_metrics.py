import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from unimapgen.utils import ensure_dir


def _to_np_points(line: Dict) -> np.ndarray:
    pts = np.asarray(line.get("points", []), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return pts


def _chamfer_symmetric(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1e6
    d1 = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    d2 = d1.T
    return float(d1.min(axis=1).mean() + d2.min(axis=1).mean()) * 0.5


def _continuity_score(lines: List[Dict], tol: float = 12.0) -> float:
    endpoints: List[Tuple[int, np.ndarray]] = []
    for i, line in enumerate(lines):
        pts = _to_np_points(line)
        if len(pts) < 2:
            continue
        endpoints.append((i, pts[0]))
        endpoints.append((i, pts[-1]))
    if len(endpoints) == 0:
        return 0.0

    connected = 0
    for i, (lid, p) in enumerate(endpoints):
        best = 1e9
        for j, (lid2, q) in enumerate(endpoints):
            if i == j or lid == lid2:
                continue
            d = float(np.linalg.norm(p - q))
            if d < best:
                best = d
        if best <= tol:
            connected += 1
    return connected / max(1, len(endpoints))


def _sample_metrics(pred_lines: List[Dict], gt_lines: List[Dict], thresholds: List[float]) -> Dict[str, float]:
    cat_to_gt: Dict[str, List[np.ndarray]] = {}
    cat_to_pred: Dict[str, List[np.ndarray]] = {}
    for line in gt_lines:
        cat = line.get("category", "unknown")
        cat_to_gt.setdefault(cat, []).append(_to_np_points(line))
    for line in pred_lines:
        cat = line.get("category", "unknown")
        cat_to_pred.setdefault(cat, []).append(_to_np_points(line))

    out = {}
    for th in thresholds:
        tp = 0
        fp = 0
        fn = 0
        cats = set(cat_to_gt.keys()) | set(cat_to_pred.keys())
        for cat in cats:
            gts = cat_to_gt.get(cat, [])
            preds = cat_to_pred.get(cat, [])
            used = set()
            for p in preds:
                best_j = -1
                best_d = 1e9
                for j, g in enumerate(gts):
                    if j in used:
                        continue
                    d = _chamfer_symmetric(p, g)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d <= th:
                    used.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += max(0, len(gts) - len(used))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        out[f"APC@{int(th)}px"] = f1

    # Mean chamfer via greedy category matching.
    chamfers = []
    cats = set(cat_to_gt.keys()) | set(cat_to_pred.keys())
    for cat in cats:
        gts = cat_to_gt.get(cat, [])
        preds = cat_to_pred.get(cat, [])
        for p in preds:
            if len(gts) == 0:
                chamfers.append(64.0)
                continue
            chamfers.append(min(_chamfer_symmetric(p, g) for g in gts))
    out["mean_chamfer_px"] = float(np.mean(chamfers)) if chamfers else 64.0
    out["continuity_pred"] = _continuity_score(pred_lines, tol=12.0)
    out["continuity_gt"] = _continuity_score(gt_lines, tol=12.0)
    out["continuity_gap"] = abs(out["continuity_pred"] - out["continuity_gt"])
    out["pred_num_lines"] = float(len(pred_lines))
    out["gt_num_lines"] = float(len(gt_lines))
    return out


def evaluate_prediction_json(path: str, thresholds: List[float]) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    agg: Dict[str, List[float]] = {}
    for item in items:
        m = _sample_metrics(item.get("pred_lines", []), item.get("gt_lines", []), thresholds=thresholds)
        for k, v in m.items():
            agg.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def to_markdown(v1_name: str, v1: Dict[str, float], v2_name: str, v2: Dict[str, float]) -> str:
    keys = [
        "APC@2px",
        "APC@4px",
        "APC@8px",
        "mean_chamfer_px",
        "continuity_pred",
        "continuity_gap",
        "pred_num_lines",
        "gt_num_lines",
    ]
    lines = [
        "| Metric | " + v1_name + " | " + v2_name + " |",
        "|---|---:|---:|",
    ]
    for k in keys:
        a = v1.get(k, float("nan"))
        b = v2.get(k, float("nan"))
        lines.append(f"| {k} | {a:.4f} | {b:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_json", type=str, required=True)
    parser.add_argument("--v2_json", type=str, required=True)
    parser.add_argument("--out_markdown", type=str, default="docs/v1_v2_comparison.md")
    args = parser.parse_args()

    thresholds = [2.0, 4.0, 8.0]
    v1 = evaluate_prediction_json(args.v1_json, thresholds=thresholds)
    v2 = evaluate_prediction_json(args.v2_json, thresholds=thresholds)
    md = to_markdown("v1", v1, "v2", v2)

    ensure_dir(os.path.dirname(args.out_markdown) or ".")
    with open(args.out_markdown, "w", encoding="utf-8") as f:
        f.write("# V1 vs V2 对比（近似指标）\n\n")
        f.write("> 指标为工程近似版，用于迭代对比，不等价于论文官方评测。\n\n")
        f.write(md)
    print(md)
    print(f"saved comparison to {args.out_markdown}")


if __name__ == "__main__":
    main()
