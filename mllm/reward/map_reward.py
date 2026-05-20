from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any

from infer_index.line_eval import evaluate_records

from mllm.coord_utils import COORD_MODE_PIXEL, DEFAULT_COORD_RANGE, convert_items

from .map_schema import parse_map_json


@dataclass
class MapRewardConfig:
    map_task: str = "lane"
    patch_size: int = 256
    coord_mode: str = COORD_MODE_PIXEL
    coord_range: int = DEFAULT_COORD_RANGE
    invalid_reward: float = -1.0
    format_weight: float = 0.08
    centerline_instance_weight: float = 0.37
    centerline_length_weight: float = 0.45
    cut_type_weight: float = 0.05
    cut_continuity_weight: float = 0.05
    intersection_weight: float = 0.0
    boundary_tolerance: int = 3
    meter_per_pixel: float = 0.2
    buffer_size: float = 1.0
    match_threshold: float = 0.33


def _centerline_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("category", "centerline") == "centerline"]


def _intersection_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("category") == "intersection"]


def _as_line_payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"lines": items}, ensure_ascii=False)


def _endpoint_on_boundary(point: list[int], patch_size: int, tolerance: int) -> bool:
    x, y = point
    high = patch_size - 1
    return x <= tolerance or y <= tolerance or x >= high - tolerance or y >= high - tolerance


def _cut_type_score(gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]) -> float:
    gt_lines = _centerline_items(gt_items)
    pred_lines = _centerline_items(pred_items)
    if not gt_lines:
        return 1.0 if not pred_lines else 0.0
    total = min(len(gt_lines), len(pred_lines))
    if total <= 0:
        return 0.0
    score = 0
    for gt, pred in zip(gt_lines[:total], pred_lines[:total]):
        score += int(gt.get("start_type", "inside") == pred.get("start_type", "inside"))
        score += int(gt.get("end_type", "inside") == pred.get("end_type", "inside"))
    return score / (2 * total)


def _cut_boundary_score(pred_items: list[dict[str, Any]], patch_size: int, tolerance: int) -> float:
    pred_lines = _centerline_items(pred_items)
    checks = []
    for item in pred_lines:
        points = item.get("points") or []
        if len(points) < 2:
            continue
        if item.get("start_type") == "cut":
            checks.append(_endpoint_on_boundary(points[0], patch_size, tolerance))
        if item.get("end_type") == "cut":
            checks.append(_endpoint_on_boundary(points[-1], patch_size, tolerance))
    if not checks:
        return 1.0
    return sum(1 for ok in checks if ok) / len(checks)


def _intersection_basic_score(gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]) -> float:
    gt_count = len(_intersection_items(gt_items))
    pred_count = len(_intersection_items(pred_items))
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    return max(0.0, 1.0 - abs(gt_count - pred_count) / max(gt_count, 1))


def _uses_intersection_reward(map_task: str) -> bool:
    # Phase A/lane runs must not be affected by intersection criteria. The
    # parser also rejects intersection outputs for lane, and this guard keeps
    # the scalar reward aligned with that branch split.
    return str(map_task).strip().lower() in {"lane_intersection", "intersection", "all"}


def compute_map_reward(prediction: str, ground_truth: str, config: MapRewardConfig | None = None) -> dict[str, Any]:
    config = config or MapRewardConfig()
    pred_parse = parse_map_json(
        prediction,
        map_task=config.map_task,
        patch_size=config.patch_size,
        coord_mode=config.coord_mode,
        coord_range=config.coord_range,
    )
    gt_parse = parse_map_json(
        ground_truth,
        map_task=config.map_task,
        patch_size=config.patch_size,
        coord_mode=config.coord_mode,
        coord_range=config.coord_range,
    )
    if not pred_parse.ok or not gt_parse.ok:
        return {
            "reward": config.invalid_reward,
            "parse_ok": False,
            "parse_error": pred_parse.error or gt_parse.error,
            "components": {},
            "config": asdict(config),
        }

    # Reward geometry is computed in pixel coordinates so the matcher, cut
    # boundary check, and downstream visualization all share one metric space.
    pred_items = convert_items(
        pred_parse.items,
        config.coord_mode,
        COORD_MODE_PIXEL,
        config.patch_size,
        config.patch_size,
        coord_range=config.coord_range,
        clamp=True,
    )
    gt_items = convert_items(
        gt_parse.items,
        config.coord_mode,
        COORD_MODE_PIXEL,
        config.patch_size,
        config.patch_size,
        coord_range=config.coord_range,
        clamp=True,
    )

    # Main geometry score: reuse the same infer_index matcher used after
    # inference, so future post-training rewards can optimize the metric we
    # actually care about.
    line_res = evaluate_records(
        [{
            "ground_truth_pixel": _as_line_payload(_centerline_items(gt_items)),
            "prediction_json_pixel": _as_line_payload(_centerline_items(pred_items)),
            "parse_ok": True,
        }],
        meter_per_pixel=config.meter_per_pixel,
        buffer_size=config.buffer_size,
        match_threshold=config.match_threshold,
    )
    instance_f1 = float(line_res.get("instance_f1", 0.0))
    length_f1 = float(line_res.get("length_f1", 0.0))
    cut_type = _cut_type_score(gt_items, pred_items)
    cut_continuity = _cut_boundary_score(pred_items, config.patch_size, config.boundary_tolerance)
    use_intersection = _uses_intersection_reward(config.map_task)
    intersection = _intersection_basic_score(gt_items, pred_items) if use_intersection else 0.0

    components = {
        "format": 1.0,
        "centerline_instance_f1": instance_f1,
        "centerline_length_f1": length_f1,
        "cut_type": cut_type,
        "cut_continuity": cut_continuity,
        "intersection": intersection,
    }
    reward = (
        config.format_weight * components["format"]
        + config.centerline_instance_weight * instance_f1
        + config.centerline_length_weight * length_f1
        + config.cut_type_weight * cut_type
        + config.cut_continuity_weight * cut_continuity
    )
    if use_intersection:
        reward += config.intersection_weight * intersection
    return {
        "reward": float(reward),
        "parse_ok": True,
        "parse_error": None,
        "components": components,
        "config": asdict(config),
    }


def compute_map_rewards(predictions: list[str], ground_truths: list[str], config: MapRewardConfig | None = None):
    return [compute_map_reward(pred, gt, config) for pred, gt in zip(predictions, ground_truths)]
