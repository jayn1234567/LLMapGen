#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString


@dataclass
class LineMatchRes:
    gt_line_num: int = 0
    pred_line_num: int = 0
    matched_line_num: int = 0
    gt_line_length_sum: float = 0.0
    pred_line_length_sum: float = 0.0
    matched_line_length_sum: float = 0.0
    valid_string_format: int = 0
    sample_num: int = 0


@dataclass
class LineEvalRes:
    instance_pre: float = 0.0
    instance_recall: float = 0.0
    instance_f1: float = 0.0
    length_pre: float = 0.0
    length_recall: float = 0.0
    length_f1: float = 0.0
    valid_string_format: int = 0
    samples_num: int = 0
    backend: str = ""
    meter_per_pixel: float = 1.0
    buffer_size: float = 1.0
    match_threshold: float = 0.33


def _safe_div(num, den):
    return num / den if den else 0.0


def _round_float_fields(obj):
    for field in fields(obj):
        if field.type is float:
            setattr(obj, field.name, round(getattr(obj, field.name), 4))
    return obj


def extract_json_payload(text: str) -> str:
    text = str(text or "").strip()
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return text
    start = min(starts)
    stack = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[start:idx + 1]
    return text[start:]


def parse_map_lines(text: str, category: str = "centerline"):
    payload = json.loads(extract_json_payload(text))
    if isinstance(payload, dict):
        items = payload.get("lines", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("map payload must be a JSON list or object with lines")

    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_category = str(item.get("category", "centerline")).strip()
        item_category = "centerline" if item_category == "CenterLine" else item_category.lower()
        if item_category != category:
            continue
        points = item.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        clean = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("invalid point format")
            clean.append((float(point[0]), float(point[1])))
        lines.append(clean)
    return lines


def convert_to_meter(lines, meter_per_pixel: float):
    return [[(x * meter_per_pixel, y * meter_per_pixel) for x, y in line] for line in lines]


def line_length(line):
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]))


def _line_iou_shapely(line1, line2, buffer_size):
    geom1 = LineString(line1)
    geom2 = LineString(line2)
    if geom1.is_empty or geom2.is_empty:
        return 0.0
    poly1 = geom1.buffer(buffer_size)
    poly2 = geom2.buffer(buffer_size)
    union_area = poly1.union(poly2).area
    return poly1.intersection(poly2).area / union_area if union_area else 0.0


def line_match_metric(line1, line2, buffer_size):
    return _line_iou_shapely(line1, line2, buffer_size)


def hungarian_match(gt_lines, pred_lines, buffer_size, match_threshold):
    if not gt_lines or not pred_lines:
        return [], []
    scores = np.zeros((len(gt_lines), len(pred_lines)), dtype=np.float64)
    for gt_idx, gt_line in enumerate(gt_lines):
        for pred_idx, pred_line in enumerate(pred_lines):
            scores[gt_idx, pred_idx] = line_match_metric(gt_line, pred_line, buffer_size)
    row_indices, col_indices = linear_sum_assignment(-scores)
    gt_match_indices = []
    pred_match_indices = []
    for gt_idx, pred_idx in zip(row_indices, col_indices):
        if scores[gt_idx, pred_idx] >= match_threshold:
            gt_match_indices.append(int(gt_idx))
            pred_match_indices.append(int(pred_idx))
    return gt_match_indices, pred_match_indices


def generate_line_eval_res(gt_lines, pred_lines, gt_match_indices):
    res = LineMatchRes()
    res.gt_line_num = len(gt_lines)
    res.pred_line_num = len(pred_lines)
    res.gt_line_length_sum = sum(line_length(line) for line in gt_lines)
    res.pred_line_length_sum = sum(line_length(line) for line in pred_lines)
    res.matched_line_num = len(gt_match_indices)
    res.matched_line_length_sum = sum(line_length(gt_lines[idx]) for idx in gt_match_indices)
    return res


def evaluate_one_sample(
    ground_truth: str,
    prediction: str,
    parse_ok: bool = True,
    meter_per_pixel: float = 1.0,
    buffer_size: float = 1.0,
    match_threshold: float = 0.33,
    category: str = "centerline",
):
    valid_string_format = bool(parse_ok)
    pred_lines = []
    try:
        gt_lines = parse_map_lines(ground_truth, category=category)
    except Exception:
        gt_lines = []
        valid_string_format = False

    if parse_ok:
        try:
            pred_lines = parse_map_lines(prediction, category=category)
        except Exception:
            pred_lines = []
            valid_string_format = False

    gt_lines = convert_to_meter(gt_lines, meter_per_pixel)
    pred_lines = convert_to_meter(pred_lines, meter_per_pixel)
    gt_match_indices, _ = hungarian_match(
        gt_lines,
        pred_lines,
        buffer_size=buffer_size,
        match_threshold=match_threshold,
    )
    res = generate_line_eval_res(gt_lines, pred_lines, gt_match_indices)
    res.valid_string_format = 1 if valid_string_format else 0
    res.sample_num = 1
    return res


def summarize_eval(sample_results, meter_per_pixel, buffer_size, match_threshold):
    total = LineMatchRes()
    for one in sample_results:
        for field in fields(LineMatchRes):
            setattr(total, field.name, getattr(total, field.name) + getattr(one, field.name))

    res = LineEvalRes()
    res.instance_pre = _safe_div(total.matched_line_num, total.pred_line_num)
    res.instance_recall = _safe_div(total.matched_line_num, total.gt_line_num)
    res.instance_f1 = _safe_div(2 * res.instance_pre * res.instance_recall, res.instance_pre + res.instance_recall)
    res.length_pre = _safe_div(total.matched_line_length_sum, total.pred_line_length_sum)
    res.length_recall = _safe_div(total.matched_line_length_sum, total.gt_line_length_sum)
    res.length_f1 = _safe_div(2 * res.length_pre * res.length_recall, res.length_pre + res.length_recall)
    res.valid_string_format = total.valid_string_format
    res.samples_num = total.sample_num
    res.backend = "shapely"
    res.meter_per_pixel = meter_per_pixel
    res.buffer_size = buffer_size
    res.match_threshold = match_threshold
    return _round_float_fields(res)


def _prediction_text(record):
    return record.get("prediction_json") or record.get("prediction") or ""


def evaluate_records(
    records,
    meter_per_pixel: float = 1.0,
    buffer_size: float = 1.0,
    match_threshold: float = 0.33,
    category: str = "centerline",
    include_samples: bool = False,
):
    sample_results = []
    sample_payloads = []
    for idx, record in enumerate(records):
        if "ground_truth" not in record:
            continue
        one = evaluate_one_sample(
            record["ground_truth"],
            _prediction_text(record),
            parse_ok=record.get("parse_ok", True),
            meter_per_pixel=meter_per_pixel,
            buffer_size=buffer_size,
            match_threshold=match_threshold,
            category=category,
        )
        sample_results.append(one)
        if include_samples:
            payload = asdict(one)
            payload["idx"] = idx
            payload["record_id"] = record.get("record_id", record.get("id", f"sample_{idx}"))
            sample_payloads.append(payload)

    summary = asdict(summarize_eval(sample_results, meter_per_pixel, buffer_size, match_threshold))
    if include_samples:
        return {"summary": summary, "samples": sample_payloads}
    return summary


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("patch_results"), list):
        return payload["patch_results"]
    if isinstance(payload, list):
        return payload
    raise ValueError("input must be a result list or state-update summary object")


def main():
    parser = argparse.ArgumentParser(description="Evaluate centerline predictions with buffer-IoU Hungarian matching.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--meter-per-pixel", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=0.33)
    parser.add_argument("--category", default="centerline")
    parser.add_argument("--include-samples", action="store_true")
    args = parser.parse_args()

    result = evaluate_records(
        load_records(Path(args.summary_json)),
        meter_per_pixel=args.meter_per_pixel,
        buffer_size=args.buffer_size,
        match_threshold=args.match_threshold,
        category=args.category,
        include_samples=args.include_samples,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
