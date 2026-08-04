"""Patch-level intersection coverage metrics compatible with RC E2E formulas.

This module reproduces the original evaluator's asymmetric coverage rule on
direct patch polygons. It does not reproduce the original whole-map rule
engine that generates ``Intersection.geojson`` from stitched centerlines.
"""

from __future__ import annotations

import json
import math
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from mllm.coord_utils import COORD_MODE_PIXEL, convert_payload_text, record_coord_config


INTERSECTION_ALIASES = {"intersection", "junction", "crossing"}
T_INTERSECTION_ALIASES = {
    "2",
    "1_2",
    "1-2",
    "t",
    "t_intersection",
    "t-intersection",
    "t_junction",
    "t-junction",
}


def _round(value: float) -> float:
    return round(float(value), 4)


def _ratio(numerator: float, denominator: float) -> float:
    # The original RC evaluator reports precision=1 when there are no
    # predictions. Apply the same empty-denominator convention to all ratios.
    return float(numerator) / float(denominator) if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _pixel_payload(
    record: dict[str, Any],
    pixel_keys: tuple[str, ...],
    raw_keys: tuple[str, ...],
) -> Any:
    pixel = _first(record, pixel_keys)
    if pixel is not None:
        return pixel
    raw = _first(record, raw_keys)
    if raw is None:
        return {"lines": []}
    config = record_coord_config(record, default_mode=COORD_MODE_PIXEL)
    if config["coord_mode"] == COORD_MODE_PIXEL:
        return raw
    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return convert_payload_text(
        raw_text,
        config["coord_mode"],
        COORD_MODE_PIXEL,
        config["patch_width"],
        config["patch_height"],
        coord_range=config["coord_range"],
        clamp=True,
    )


def _load_lines(payload: Any) -> list[Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        payload = payload.get("lines", payload.get("road_map", payload.get("centerlines", [])))
    if not isinstance(payload, list):
        raise TypeError("map payload must contain a lines list")
    return payload


def _category(item: dict[str, Any]) -> str:
    return str(item.get("category", "centerline")).strip().lower()


def _intersection_type(item: dict[str, Any]) -> str:
    value = item.get("intersection_type", item.get("IntersectionType", item.get("type")))
    normalized = str(value if value is not None else "").strip().lower().replace(" ", "_")
    return "t_intersection" if normalized in T_INTERSECTION_ALIASES else normalized


def _polygon(item: dict[str, Any]):
    points = item.get("points")
    if not isinstance(points, list) or len(points) < 3:
        return None
    cleaned = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        cleaned.append((x, y))
    if len(set(cleaned)) < 3:
        return None
    if cleaned[0] != cleaned[-1]:
        cleaned.append(cleaned[0])
    geometry = Polygon(cleaned)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty or geometry.area <= 0:
        return None
    return geometry


def _parse_polygons(payload: Any, type_filter: str | None) -> tuple[list[Any], int]:
    polygons = []
    invalid = 0
    for item in _load_lines(payload):
        if not isinstance(item, dict) or _category(item) not in INTERSECTION_ALIASES:
            continue
        if type_filter is not None and _intersection_type(item) != type_filter:
            continue
        geometry = _polygon(item)
        if geometry is None:
            invalid += 1
        else:
            polygons.append(geometry)
    return polygons, invalid


def _coverage_values(items: list[Any], other_union: Any | None) -> list[float]:
    if other_union is None or other_union.is_empty:
        return [0.0] * len(items)
    values = []
    for geometry in items:
        try:
            overlap = geometry.intersection(other_union).area
        except Exception:
            overlap = 0.0
        values.append(float(overlap) / float(geometry.area) if geometry.area else 0.0)
    return values


def _evaluate_subset(
    records: list[dict[str, Any]],
    *,
    coverage_threshold: float,
    type_filter: str | None,
    include_samples: bool,
) -> dict[str, Any]:
    totals = {
        "recalled_num": 0,
        "correct_num": 0,
        "gt_num": 0,
        "pred_num": 0,
        "matched_area": 0.0,
        "gt_total_area": 0.0,
        "pred_total_area": 0.0,
        "invalid_gt_polygon_num": 0,
        "invalid_pred_polygon_num": 0,
        "valid_string_format": 0,
        "samples_num": 0,
    }
    samples = []

    for index, record in enumerate(records):
        if not isinstance(record, dict) or not any(
            key in record for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel")
        ):
            continue
        totals["samples_num"] += 1
        gt_parse_ok = pred_parse_ok = True
        try:
            gt_payload = _pixel_payload(
                record,
                ("labels_pixel", "ground_truth_pixel"),
                ("labels", "ground_truth"),
            )
            gt, invalid_gt = _parse_polygons(gt_payload, type_filter)
        except Exception:
            gt, invalid_gt, gt_parse_ok = [], 0, False
        try:
            if not bool(record.get("parse_ok", True)):
                raise ValueError(record.get("parse_error") or "prediction parse_ok is false")
            pred_payload = _pixel_payload(
                record,
                ("response_pixel", "prediction_json_pixel", "prediction_pixel"),
                ("response", "prediction_json", "prediction"),
            )
            pred, invalid_pred = _parse_polygons(pred_payload, type_filter)
        except Exception:
            pred, invalid_pred, pred_parse_ok = [], 0, False

        if gt_parse_ok and pred_parse_ok:
            totals["valid_string_format"] += 1
        totals["invalid_gt_polygon_num"] += invalid_gt
        totals["invalid_pred_polygon_num"] += invalid_pred

        gt_union = unary_union(gt) if gt else None
        pred_union = unary_union(pred) if pred else None
        gt_coverages = _coverage_values(gt, pred_union)
        pred_coverages = _coverage_values(pred, gt_union)
        recalled = sum(value > coverage_threshold for value in gt_coverages)
        correct = sum(value > coverage_threshold for value in pred_coverages)

        gt_area = float(gt_union.area) if gt_union is not None else 0.0
        pred_area = float(pred_union.area) if pred_union is not None else 0.0
        matched_area = (
            float(gt_union.intersection(pred_union).area)
            if gt_union is not None and pred_union is not None
            else 0.0
        )

        totals["recalled_num"] += recalled
        totals["correct_num"] += correct
        totals["gt_num"] += len(gt)
        totals["pred_num"] += len(pred)
        totals["matched_area"] += matched_area
        totals["gt_total_area"] += gt_area
        totals["pred_total_area"] += pred_area

        if include_samples:
            samples.append(
                {
                    "idx": index,
                    "record_id": record.get("record_id", record.get("id", f"sample_{index}")),
                    "gt_num": len(gt),
                    "pred_num": len(pred),
                    "recalled_num": recalled,
                    "correct_num": correct,
                    "gt_coverages": [_round(value) for value in gt_coverages],
                    "pred_coverages": [_round(value) for value in pred_coverages],
                    "matched_area": _round(matched_area),
                    "gt_total_area": _round(gt_area),
                    "pred_total_area": _round(pred_area),
                    "valid_string_format": bool(gt_parse_ok and pred_parse_ok),
                }
            )

    precision = _ratio(totals["correct_num"], totals["pred_num"])
    recall = _ratio(totals["recalled_num"], totals["gt_num"])
    area_precision = _ratio(totals["matched_area"], totals["pred_total_area"])
    area_recall = _ratio(totals["matched_area"], totals["gt_total_area"])
    summary = {
        "backend": "infer_index.original_e2e_intersection_coverage_patch",
        "eval_name": "Patch Intersection Original-E2E Coverage Formula",
        "scope": "patch_direct_prediction_polygons",
        "type_filter": type_filter or "all_intersections",
        "coverage_threshold": float(coverage_threshold),
        "coverage_comparison": ">",
        "zero_denominator_policy": 1.0,
        "instance_precision": _round(precision),
        "instance_recall": _round(recall),
        "instance_f1": _round(_f1(precision, recall)),
        "area_precision": _round(area_precision),
        "area_recall": _round(area_recall),
        "area_f1": _round(_f1(area_precision, area_recall)),
        **{
            key: _round(value) if isinstance(value, float) else value
            for key, value in totals.items()
        },
    }
    summary["table"] = (
        "Patch Intersection Original-E2E Coverage Formula\n"
        "================================================\n"
        f"type filter:                  {summary['type_filter']}\n"
        f"coverage threshold:           > {coverage_threshold:.4f}\n"
        f"instance precision/recall/F1: {precision:.4f} / {recall:.4f} / {_f1(precision, recall):.4f}\n"
        f"area precision/recall/F1:     {area_precision:.4f} / {area_recall:.4f} / {_f1(area_precision, area_recall):.4f}\n"
        f"recalled/correct/GT/pred:      {totals['recalled_num']} / {totals['correct_num']} / "
        f"{totals['gt_num']} / {totals['pred_num']}\n"
        f"valid prediction format:      {totals['valid_string_format']}/{totals['samples_num']}"
    )
    if include_samples:
        return {"summary": summary, "samples": samples}
    return summary


def evaluate_intersection_coverage_records(
    records: list[dict[str, Any]],
    coverage_threshold: float = 0.5,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Evaluate all and T-intersection patch polygons with RC E2E coverage rules."""
    if not 0.0 <= float(coverage_threshold) <= 1.0:
        raise ValueError("coverage_threshold must be in [0, 1]")
    return {
        "intersection": _evaluate_subset(
            records,
            coverage_threshold=float(coverage_threshold),
            type_filter=None,
            include_samples=include_samples,
        ),
        "t_intersection": _evaluate_subset(
            records,
            coverage_threshold=float(coverage_threshold),
            type_filter="t_intersection",
            include_samples=include_samples,
        ),
        "note": (
            "This reproduces the original asymmetric coverage formula on direct patch polygons. "
            "It does not run whole-map stitching or the RC rule engine that creates Intersection.geojson."
        ),
    }
