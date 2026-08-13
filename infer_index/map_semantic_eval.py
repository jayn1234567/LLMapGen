"""Geometry and semantic-type metrics for lane/intersection map outputs."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from mllm.coord_utils import COORD_MODE_PIXEL, convert_payload_text, record_coord_config

from .utils import _item_category, _iter_line_items, _load_prediction_payload


LANE_TYPES = (
    "common",
    "right_turn",
    "waiting_area",
    "bus_lane",
    "main_auxiliary_connector",
    "other",
)
INTERSECTION_TYPES = (
    "common",
    "t_intersection",
    "small_untyped",
    "t_lane_change_area",
    "other",
)
UNKNOWN_TYPE = "unknown"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 4)


def _first_payload(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _payload_in_pixel_coords(
    record: dict[str, Any],
    pixel_keys: tuple[str, ...],
    raw_keys: tuple[str, ...],
) -> Any:
    pixel_payload = _first_payload(record, pixel_keys)
    if pixel_payload is not None:
        return pixel_payload
    raw_payload = _first_payload(record, raw_keys)
    if raw_payload is None:
        return {"lines": []}
    coord_cfg = record_coord_config(record, default_mode=COORD_MODE_PIXEL)
    if coord_cfg["coord_mode"] == COORD_MODE_PIXEL:
        return raw_payload
    try:
        raw_text = raw_payload if isinstance(raw_payload, str) else json.dumps(raw_payload, ensure_ascii=False)
        return convert_payload_text(
            raw_text,
            coord_cfg["coord_mode"],
            COORD_MODE_PIXEL,
            coord_cfg["patch_width"],
            coord_cfg["patch_height"],
            coord_range=coord_cfg["coord_range"],
            clamp=True,
        )
    except Exception:
        return raw_payload


def _restore_semantic_types(pixel_payload: Any, raw_payload: Any) -> Any:
    """Recover semantic fields stripped by older prediction normalizers."""
    if raw_payload is None:
        return pixel_payload
    try:
        pixel_parsed = _load_prediction_payload(pixel_payload)
        raw_parsed = _load_prediction_payload(raw_payload)
        pixel_items = list(_iter_line_items(pixel_parsed))
        raw_items = list(_iter_line_items(raw_parsed))
    except Exception:
        return pixel_payload
    if not pixel_items or not raw_items:
        return pixel_payload

    restored = [dict(item) if isinstance(item, dict) else item for item in pixel_items]
    for category, field in (("centerline", "lane_type"), ("intersection", "intersection_type")):
        pixel_indexes = [
            index
            for index, item in enumerate(restored)
            if isinstance(item, dict) and _item_category(item) == category
        ]
        raw_category_items = [
            item
            for item in raw_items
            if isinstance(item, dict) and _item_category(item) == category
        ]
        for pixel_index, raw_item in zip(pixel_indexes, raw_category_items):
            if restored[pixel_index].get(field) is not None:
                continue
            value = raw_item.get(field, raw_item.get("type"))
            if value is not None:
                restored[pixel_index][field] = value
    return {"lines": restored}


def _record_payloads(record: dict[str, Any]) -> tuple[Any, Any, bool]:
    ground_truth = _payload_in_pixel_coords(
        record,
        ("labels_pixel", "ground_truth_pixel"),
        ("labels", "ground_truth"),
    )
    prediction = _payload_in_pixel_coords(
        record,
        ("response_pixel", "prediction_json_pixel", "prediction_pixel"),
        ("response", "prediction_json", "prediction"),
    )
    prediction = _restore_semantic_types(
        prediction,
        _first_payload(record, ("prediction", "raw_prediction", "response", "prediction_json")),
    )
    ground_truth = _restore_semantic_types(
        ground_truth,
        _first_payload(record, ("labels", "ground_truth")),
    )
    parse_ok = bool(record.get("parse_ok", True))
    return ground_truth, prediction, parse_ok


def _normalize_type(value: Any, allowed_types: tuple[str, ...]) -> str:
    if value is None:
        return UNKNOWN_TYPE
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rightturn": "right_turn",
        "right_turn_only": "right_turn",
        "t": "t_intersection",
        "t_junction": "t_intersection",
        "t_intersection_area": "t_intersection",
        "small_intersection": "small_untyped",
        "unstructured_intersection": "small_untyped",
        "lane_change_area": "t_lane_change_area",
        "t_lane_change": "t_lane_change_area",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in allowed_types else UNKNOWN_TYPE


def _semantic_type(item: dict[str, Any], field: str, allowed_types: tuple[str, ...]) -> str:
    value = item.get(field)
    if value is None:
        value = item.get("type")
    return _normalize_type(value, allowed_types)


def _clean_points(item: dict[str, Any], minimum: int) -> list[tuple[float, float]]:
    points = item.get("points")
    if not isinstance(points, list):
        return []
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return []
        try:
            result.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return []
    if len(result) < minimum:
        return []
    return result


def _parse_items(payload: Any, category: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        parsed = _load_prediction_payload(payload)
        items = [
            item
            for item in _iter_line_items(parsed)
            if isinstance(item, dict) and _item_category(item) == category
        ]
        return items, True
    except Exception:
        return [], False


def _polygon_from_item(item: dict[str, Any]):
    points = _clean_points(item, minimum=3)
    if not points or len(set(points)) < 3:
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _line_from_item(item: dict[str, Any], meter_per_pixel: float):
    points = _clean_points(item, minimum=2)
    if not points or len(set(points)) < 2:
        return None
    return LineString([(x * meter_per_pixel, y * meter_per_pixel) for x, y in points])


def _polygon_iou(first, second) -> float:
    union_area = first.union(second).area
    return safe_div(first.intersection(second).area, union_area)


def _line_buffer_iou(first, second, buffer_size: float) -> float:
    first_buffer = first.buffer(buffer_size)
    second_buffer = second.buffer(buffer_size)
    union_area = first_buffer.union(second_buffer).area
    return safe_div(first_buffer.intersection(second_buffer).area, union_area)


def _match_by_similarity(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metric: Callable[[Any, Any], float],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    if not ground_truth or not predictions:
        return [], np.zeros((len(ground_truth), len(predictions)), dtype=np.float32)
    matrix = np.zeros((len(ground_truth), len(predictions)), dtype=np.float32)
    for gt_index, gt_item in enumerate(ground_truth):
        for pred_index, pred_item in enumerate(predictions):
            matrix[gt_index, pred_index] = metric(gt_item["geometry"], pred_item["geometry"])
    gt_indices, pred_indices = linear_sum_assignment(-matrix)
    matches = [
        (int(gt_index), int(pred_index), float(matrix[gt_index, pred_index]))
        for gt_index, pred_index in zip(gt_indices, pred_indices)
        if float(matrix[gt_index, pred_index]) >= threshold
    ]
    return matches, matrix


def _f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def _type_metrics(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matches: list[tuple[int, int, float]],
    allowed_types: tuple[str, ...],
) -> dict[str, Any]:
    labels = list(allowed_types) + [UNKNOWN_TYPE]
    gt_counts = Counter(item["semantic_type"] for item in ground_truth)
    pred_counts = Counter(item["semantic_type"] for item in predictions)
    confusion = {gt_type: {pred_type: 0 for pred_type in labels} for gt_type in labels}
    matched_gt_counts = Counter()
    correct_counts = Counter()
    valid_gt_matches = 0
    valid_pred_matches = 0
    correct = 0

    for gt_index, pred_index, _ in matches:
        gt_type = ground_truth[gt_index]["semantic_type"]
        pred_type = predictions[pred_index]["semantic_type"]
        confusion[gt_type][pred_type] += 1
        matched_gt_counts[gt_type] += 1
        if gt_type != UNKNOWN_TYPE:
            valid_gt_matches += 1
        if pred_type != UNKNOWN_TYPE:
            valid_pred_matches += 1
        if gt_type != UNKNOWN_TYPE and pred_type == gt_type:
            correct += 1
            correct_counts[gt_type] += 1

    known_gt_count = sum(gt_counts[label] for label in allowed_types)
    known_pred_count = sum(pred_counts[label] for label in allowed_types)
    per_type = {}
    for label in allowed_types:
        precision = safe_div(correct_counts[label], pred_counts[label])
        recall = safe_div(correct_counts[label], gt_counts[label])
        per_type[label] = {
            "support": int(gt_counts[label]),
            "predictions": int(pred_counts[label]),
            "geometry_matched": int(matched_gt_counts[label]),
            "correct": int(correct_counts[label]),
            "matched_accuracy": _round(safe_div(correct_counts[label], matched_gt_counts[label])),
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(_f1(precision, recall)),
        }

    end_to_end_precision = safe_div(correct, len(predictions))
    end_to_end_recall = safe_div(correct, known_gt_count)
    return {
        "geometry_matched_count": len(matches),
        "typed_gt_count": int(known_gt_count),
        "typed_prediction_count": int(known_pred_count),
        "unknown_gt_count": int(gt_counts[UNKNOWN_TYPE]),
        "unknown_prediction_count": int(pred_counts[UNKNOWN_TYPE]),
        "type_correct_count": int(correct),
        "matched_type_accuracy": _round(safe_div(correct, valid_gt_matches)),
        "prediction_type_coverage_on_matches": _round(safe_div(valid_pred_matches, len(matches))),
        "end_to_end_type_precision": _round(end_to_end_precision),
        "end_to_end_type_recall": _round(end_to_end_recall),
        "end_to_end_type_f1": _round(_f1(end_to_end_precision, end_to_end_recall)),
        "per_type": per_type,
        "confusion_matrix": {
            "labels": labels,
            "rows_gt_columns_prediction": confusion,
        },
    }


def _format_type_table(title: str, metrics: dict[str, Any]) -> str:
    lines = [
        title,
        "=" * len(title),
        "type                 support  pred  matched  correct  match_acc  precision  recall  f1",
    ]
    for label, values in metrics["per_type"].items():
        lines.append(
            f"{label:<20} {values['support']:>7} {values['predictions']:>5} "
            f"{values['geometry_matched']:>8} {values['correct']:>8} "
            f"{values['matched_accuracy']:>10.4f} {values['precision']:>10.4f} "
            f"{values['recall']:>7.4f} {values['f1']:>7.4f}"
        )
    lines.append(
        "overall: matched_type_accuracy={:.4f}, end_to_end_type_f1={:.4f}, "
        "unknown_predictions={}".format(
            metrics["matched_type_accuracy"],
            metrics["end_to_end_type_f1"],
            metrics["unknown_prediction_count"],
        )
    )
    return "\n".join(lines)


def _collect_geometry_items(
    items: list[dict[str, Any]],
    category: str,
    type_field: str,
    allowed_types: tuple[str, ...],
    geometry_builder: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], int]:
    collected = []
    invalid = 0
    for item in items:
        geometry = geometry_builder(item)
        if geometry is None:
            invalid += 1
            continue
        collected.append(
            {
                "geometry": geometry,
                "semantic_type": _semantic_type(item, type_field, allowed_types),
                "item": item,
            }
        )
    return collected, invalid


def evaluate_intersection_iou_records(
    records,
    iou_threshold: float = 0.5,
    include_samples: bool = False,
) -> dict[str, Any]:
    total_gt = []
    total_pred = []
    total_matches: list[tuple[int, int, float]] = []
    sample_payloads = []
    matched_ious = []
    sample_union_ious = []
    total_intersection_area = 0.0
    total_union_area = 0.0
    valid_string_format = 0
    invalid_gt_polygons = 0
    invalid_pred_polygons = 0
    samples_num = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict) or not any(
            key in record for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel")
        ):
            continue
        samples_num += 1
        gt_payload, pred_payload, parse_ok = _record_payloads(record)
        gt_items, gt_parse_ok = _parse_items(gt_payload, "intersection")
        pred_items, pred_parse_ok = _parse_items(pred_payload, "intersection") if parse_ok else ([], False)
        if gt_parse_ok and pred_parse_ok:
            valid_string_format += 1
        gt, invalid_gt = _collect_geometry_items(
            gt_items,
            "intersection",
            "intersection_type",
            INTERSECTION_TYPES,
            _polygon_from_item,
        )
        pred, invalid_pred = _collect_geometry_items(
            pred_items,
            "intersection",
            "intersection_type",
            INTERSECTION_TYPES,
            _polygon_from_item,
        )
        invalid_gt_polygons += invalid_gt
        invalid_pred_polygons += invalid_pred
        matches, _ = _match_by_similarity(gt, pred, _polygon_iou, iou_threshold)
        matched_ious.extend(value for _, _, value in matches)

        gt_offset = len(total_gt)
        pred_offset = len(total_pred)
        total_gt.extend(gt)
        total_pred.extend(pred)
        total_matches.extend((gt_offset + gi, pred_offset + pi, score) for gi, pi, score in matches)

        if gt or pred:
            gt_union = unary_union([item["geometry"] for item in gt]) if gt else None
            pred_union = unary_union([item["geometry"] for item in pred]) if pred else None
            if gt_union is None or pred_union is None:
                intersection_area = 0.0
                non_empty_union = gt_union if gt_union is not None else pred_union
                union_area = non_empty_union.area
            else:
                intersection_area = gt_union.intersection(pred_union).area
                union_area = gt_union.union(pred_union).area
            sample_iou = safe_div(intersection_area, union_area)
            sample_union_ious.append(sample_iou)
            total_intersection_area += intersection_area
            total_union_area += union_area
        else:
            sample_iou = 1.0

        if include_samples:
            sample_payloads.append(
                {
                    "idx": index,
                    "record_id": record.get("record_id", record.get("id", f"sample_{index}")),
                    "gt_polygon_num": len(gt),
                    "pred_polygon_num": len(pred),
                    "matched_polygon_num": len(matches),
                    "sample_union_iou": _round(sample_iou),
                    "matched_ious": [_round(value) for _, _, value in matches],
                    "valid_string_format": bool(gt_parse_ok and pred_parse_ok),
                }
            )

    matched_count = len(total_matches)
    precision = safe_div(matched_count, len(total_pred))
    recall = safe_div(matched_count, len(total_gt))
    type_metrics = _type_metrics(total_gt, total_pred, total_matches, INTERSECTION_TYPES)
    type_metrics["table"] = _format_type_table("Intersection Type Evaluation", type_metrics)
    summary = {
        "backend": "infer_index.polygon_iou",
        "eval_name": "Intersection Polygon IoU Evaluation",
        "iou_threshold": iou_threshold,
        "instance_pre": _round(precision),
        "instance_recall": _round(recall),
        "instance_f1": _round(_f1(precision, recall)),
        "mean_matched_iou": _round(np.mean(matched_ious) if matched_ious else 0.0),
        "mean_sample_union_iou": _round(np.mean(sample_union_ious) if sample_union_ious else 0.0),
        "micro_area_iou": _round(safe_div(total_intersection_area, total_union_area)),
        "gt_polygon_num": len(total_gt),
        "pred_polygon_num": len(total_pred),
        "matched_polygon_num": matched_count,
        "invalid_gt_polygon_num": invalid_gt_polygons,
        "invalid_pred_polygon_num": invalid_pred_polygons,
        "valid_string_format": valid_string_format,
        "samples_num": samples_num,
        "type_accuracy": type_metrics,
    }
    summary["table"] = (
        "Intersection Polygon IoU Evaluation\n"
        "===================================\n"
        f"instance precision/recall/F1: {summary['instance_pre']:.4f} / "
        f"{summary['instance_recall']:.4f} / {summary['instance_f1']:.4f}\n"
        f"mean matched IoU:             {summary['mean_matched_iou']:.4f}\n"
        f"mean sample union IoU:        {summary['mean_sample_union_iou']:.4f}\n"
        f"micro area IoU:               {summary['micro_area_iou']:.4f}\n"
        f"matched polygons:             {matched_count}/{len(total_gt)} GT, {len(total_pred)} predictions\n"
        f"valid prediction format:      {valid_string_format}/{samples_num}"
    )
    if include_samples:
        return {"summary": summary, "samples": sample_payloads}
    return summary


def evaluate_lane_type_records(
    records,
    meter_per_pixel: float = 0.2,
    buffer_size: float = 1.0,
    match_threshold: float = 0.33,
) -> dict[str, Any]:
    total_gt = []
    total_pred = []
    total_matches: list[tuple[int, int, float]] = []
    valid_string_format = 0
    samples_num = 0

    for record in records:
        if not isinstance(record, dict) or not any(
            key in record for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel")
        ):
            continue
        samples_num += 1
        gt_payload, pred_payload, parse_ok = _record_payloads(record)
        gt_items, gt_parse_ok = _parse_items(gt_payload, "centerline")
        pred_items, pred_parse_ok = _parse_items(pred_payload, "centerline") if parse_ok else ([], False)
        if gt_parse_ok and pred_parse_ok:
            valid_string_format += 1
        builder = lambda item: _line_from_item(item, meter_per_pixel)
        gt, _ = _collect_geometry_items(gt_items, "centerline", "lane_type", LANE_TYPES, builder)
        pred, _ = _collect_geometry_items(pred_items, "centerline", "lane_type", LANE_TYPES, builder)
        matches, _ = _match_by_similarity(
            gt,
            pred,
            lambda first, second: _line_buffer_iou(first, second, buffer_size),
            match_threshold,
        )
        gt_offset = len(total_gt)
        pred_offset = len(total_pred)
        total_gt.extend(gt)
        total_pred.extend(pred)
        total_matches.extend((gt_offset + gi, pred_offset + pi, score) for gi, pi, score in matches)

    metrics = _type_metrics(total_gt, total_pred, total_matches, LANE_TYPES)
    metrics.update(
        {
            "backend": "infer_index.line_buffer_iou_type",
            "eval_name": "Lane Type Evaluation",
            "meter_per_pixel": meter_per_pixel,
            "buffer_size": buffer_size,
            "match_threshold": match_threshold,
            "valid_string_format": valid_string_format,
            "samples_num": samples_num,
        }
    )
    metrics["table"] = _format_type_table("Lane Type Evaluation", metrics)
    return metrics
