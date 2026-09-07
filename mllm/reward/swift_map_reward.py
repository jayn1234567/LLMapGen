"""Small, dependency-light map reward for ms-swift GRPO.

The scalar reward intentionally mirrors the project's first GRPO proposal:

    0.75 * line_f1 + 0.20 * coordinate_quality + 0.05 * format_reward

``line_f1`` is the average of instance-level and length-level F1.  Because the
instance term already penalizes both missed and extra predictions, this module
does not add a second count penalty.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from mllm.coord_utils import COORD_MODE_PIXEL, convert_items, record_coord_config
from mllm.reward.map_schema import MapParseResult, parse_map_json


@dataclass(frozen=True)
class SwiftMapRewardConfig:
    map_task: str = "lane_intersection"
    default_patch_size: int = 256
    default_coord_mode: str = "norm1000"
    default_coord_range: int = 1000
    meter_per_pixel: float = 0.2
    buffer_size: float = 1.0
    match_threshold: float = 0.33
    coordinate_sigma_m: float = 0.75
    resample_points: int = 32
    line_f1_weight: float = 0.75
    coordinate_weight: float = 0.20
    format_weight: float = 0.05
    invalid_reward: float = -1.0


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _completion_to_text(completion: Any) -> str:
    """Accept Swift strings as well as OpenAI-style message objects."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "response", "completion"):
            if key in completion:
                return _text(completion[key])
        return _text(completion)
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and str(item.get("role", "")).lower() in {"assistant", "gpt", "model"}:
                return _text(item.get("content", item.get("value", "")))
        if len(completion) == 1:
            return _completion_to_text(completion[0])
    return _text(completion)


def _strict_format_ok(text: str) -> bool:
    """Return whether a completion is exactly the expected JSON envelope.

    The geometry parser is intentionally forgiving so that diagnostics can
    still score a JSON object surrounded by prose.  GRPO's format component,
    however, should teach the policy to emit only the contract consumed by the
    downstream evaluator.
    """
    stripped = str(text or "").strip()
    if not stripped or "```" in stripped:
        return False
    try:
        payload = json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("lines"), list)


def _extract_solution(value: Any) -> str:
    """Unwrap the optional converter envelope while preserving raw GT text."""
    if isinstance(value, dict):
        for key in ("ground_truth", "ground_truth_json", "gt_json", "solution"):
            if key in value:
                return _extract_solution(value[key])
    return _text(value)


def _coord_config(value: Any, config: SwiftMapRewardConfig) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, dict):
        # record_coord_config supplies defaults and normalises aliases.
        row = dict(value)
        result = record_coord_config(
            row,
            default_mode=config.default_coord_mode,
            default_patch_size=config.default_patch_size,
            default_coord_range=config.default_coord_range,
        )
        result.update({k: v for k, v in value.items() if v is not None})
        return result
    return record_coord_config(
        {},
        default_mode=config.default_coord_mode,
        default_patch_size=config.default_patch_size,
        default_coord_range=config.default_coord_range,
    )


def _centerlines(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("category", "centerline") == "centerline"]


def _parse_lines(text: str, cfg: SwiftMapRewardConfig, coord: dict[str, Any]) -> tuple[MapParseResult, list[dict[str, Any]]]:
    parsed = parse_map_json(
        text,
        map_task=coord.get("map_task", cfg.map_task),
        patch_size=int(coord["patch_size"]),
        coord_mode=str(coord["coord_mode"]),
        coord_range=int(coord["coord_range"]),
    )
    if not parsed.ok:
        return parsed, []
    lines = _centerlines(parsed.items)
    if any(len(item.get("points", [])) < 2 for item in lines):
        return MapParseResult(False, [], parsed.payload_text, "centerline needs at least two points"), []
    pixel = convert_items(
        lines,
        str(coord["coord_mode"]),
        COORD_MODE_PIXEL,
        int(coord["patch_width"]),
        int(coord["patch_height"]),
        coord_range=int(coord["coord_range"]),
        clamp=True,
    )
    return parsed, pixel


def _line_string(points: list[list[int]], meter_per_pixel: float):
    # Imported lazily so dataset conversion and plugin discovery do not require
    # Shapely/SciPy just to print help or validate JSON structure.
    from shapely.geometry import LineString

    return LineString([(float(x) * meter_per_pixel, float(y) * meter_per_pixel) for x, y in points])


def _buffered_iou(line_a, line_b, buffer_size: float) -> float:
    union = line_a.buffer(buffer_size).union(line_b.buffer(buffer_size))
    if union.area <= 0:
        return 0.0
    return float(line_a.buffer(buffer_size).intersection(line_b.buffer(buffer_size)).area / union.area)


def _match_lines(gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]], cfg: SwiftMapRewardConfig):
    """Return one-to-one matches and the same aggregate quantities as line_eval."""
    from scipy.optimize import linear_sum_assignment

    gt = [_line_string(item["points"], cfg.meter_per_pixel) for item in gt_items]
    pred = [_line_string(item["points"], cfg.meter_per_pixel) for item in pred_items]
    matrix = [[_buffered_iou(g, p, cfg.buffer_size) for p in pred] for g in gt]
    matched: list[tuple[int, int, float]] = []
    if matrix:
        rows, cols = linear_sum_assignment([[-score for score in row] for row in matrix])
        matched = [
            (int(row), int(col), float(matrix[row][col]))
            for row, col in zip(rows, cols)
            if matrix[row][col] >= cfg.match_threshold
        ]

    gt_lengths = [float(line.length) for line in gt]
    pred_lengths = [float(line.length) for line in pred]
    gt_total = sum(gt_lengths)
    pred_total = sum(pred_lengths)
    matched_gt_length = sum(gt_lengths[row] for row, _, _ in matched)
    matched_count = len(matched)
    exact_empty = not gt and not pred
    instance_precision = 1.0 if exact_empty else (matched_count / len(pred) if pred else 0.0)
    instance_recall = 1.0 if exact_empty else (matched_count / len(gt) if gt else 0.0)
    instance_f1 = _f1(instance_precision, instance_recall)
    length_precision = 1.0 if exact_empty else (matched_gt_length / pred_total if pred_total else 0.0)
    length_recall = 1.0 if exact_empty else (matched_gt_length / gt_total if gt_total else 0.0)
    length_f1 = _f1(length_precision, length_recall)
    return {
        "matches": matched,
        "gt_lines": gt,
        "pred_lines": pred,
        "gt_lengths": gt_lengths,
        "pred_lengths": pred_lengths,
        "gt_line_num": len(gt),
        "pred_line_num": len(pred),
        "matched_line_num": matched_count,
        "gt_line_length_sum": gt_total,
        "pred_line_length_sum": pred_total,
        "matched_line_length_sum": matched_gt_length,
        "instance_precision": instance_precision,
        "instance_recall": instance_recall,
        "instance_f1": instance_f1,
        "length_precision": length_precision,
        "length_recall": length_recall,
        "length_f1": length_f1,
    }


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall + 1e-6)


def _resample(points: list[list[int]], count: int, meter_per_pixel: float) -> list[tuple[float, float]]:
    scaled = [(float(x) * meter_per_pixel, float(y) * meter_per_pixel) for x, y in points]
    if not scaled:
        return []
    if len(scaled) == 1:
        return scaled * max(count, 1)
    cumulative = [0.0]
    for first, second in zip(scaled, scaled[1:]):
        cumulative.append(cumulative[-1] + math.hypot(second[0] - first[0], second[1] - first[1]))
    total = cumulative[-1]
    if total <= 1e-12:
        return [scaled[0]] * max(count, 1)
    result = []
    for index in range(max(count, 1)):
        target = total * index / max(count - 1, 1)
        segment = 1
        while segment < len(cumulative) and cumulative[segment] < target:
            segment += 1
        if segment >= len(cumulative):
            result.append(scaled[-1])
            continue
        left_distance = cumulative[segment - 1]
        right_distance = cumulative[segment]
        fraction = (target - left_distance) / max(right_distance - left_distance, 1e-12)
        first = scaled[segment - 1]
        second = scaled[segment]
        result.append((first[0] + fraction * (second[0] - first[0]), first[1] + fraction * (second[1] - first[1])))
    return result


def _coordinate_quality(match_info: dict[str, Any], gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]], cfg: SwiftMapRewardConfig) -> float:
    if not match_info["matches"]:
        return 1.0 if not gt_items and not pred_items else 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for gt_index, pred_index, _ in match_info["matches"]:
        gt_samples = _resample(gt_items[gt_index]["points"], cfg.resample_points, cfg.meter_per_pixel)
        pred_forward = _resample(pred_items[pred_index]["points"], cfg.resample_points, cfg.meter_per_pixel)
        pred_reverse = list(reversed(pred_forward))
        errors_forward = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(gt_samples, pred_forward)]
        errors_reverse = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(gt_samples, pred_reverse)]
        mean_error = min(
            sum(errors_forward) / max(len(errors_forward), 1),
            sum(errors_reverse) / max(len(errors_reverse), 1),
        )
        quality = math.exp(-0.5 * (mean_error / max(cfg.coordinate_sigma_m, 1e-9)) ** 2)
        weight = match_info["gt_lengths"][gt_index]
        weighted_sum += weight * quality
        weight_total += weight
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def score_one(
    completion: Any,
    solution: Any = None,
    *,
    ground_truth: Any = None,
    coord_config: Any = None,
    map_task: Any = None,
    config: SwiftMapRewardConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SwiftMapRewardConfig()
    target = solution if solution is not None else ground_truth
    target_text = _extract_solution(target)
    coord = _coord_config(coord_config, cfg)
    if map_task:
        coord["map_task"] = str(map_task)
    pred_parse, pred_lines = _parse_lines(_completion_to_text(completion), cfg, coord)
    gt_parse, gt_lines = _parse_lines(target_text, cfg, coord)
    if not pred_parse.ok or not gt_parse.ok:
        return {
            "reward": float(cfg.invalid_reward),
            "format_reward": 0.0,
            "parse_ok": False,
            "prediction_error": pred_parse.error,
            "ground_truth_error": gt_parse.error,
            "line_f1": 0.0,
            "coordinate_quality": 0.0,
            "coord_config": coord,
        }

    prediction_text = _completion_to_text(completion)
    format_reward = 1.0 if _strict_format_ok(prediction_text) else 0.0
    match_info = _match_lines(gt_lines, pred_lines, cfg)
    line_f1 = 0.5 * (match_info["instance_f1"] + match_info["length_f1"])
    coordinate_quality = _coordinate_quality(match_info, gt_lines, pred_lines, cfg)
    reward = (
        cfg.line_f1_weight * line_f1
        + cfg.coordinate_weight * coordinate_quality
        + cfg.format_weight * format_reward
    )
    return {
        "reward": float(reward),
        "format_reward": format_reward,
        "parse_ok": True,
        "prediction_error": None,
        "ground_truth_error": None,
        "line_f1": float(line_f1),
        "coordinate_quality": float(coordinate_quality),
        "instance_f1": float(match_info["instance_f1"]),
        "length_f1": float(match_info["length_f1"]),
        "counts": {
            key: match_info[key]
            for key in ("gt_line_num", "pred_line_num", "matched_line_num")
        },
        "coord_config": coord,
    }


def _batch(value: Any, count: int) -> list[Any]:
    if isinstance(value, list) and len(value) == count:
        return value
    return [value] * count


def compute_swift_map_rewards(
    completions: list[Any],
    solution: Any = None,
    *,
    ground_truth: Any = None,
    coord_config: Any = None,
    map_task: Any = None,
    config: SwiftMapRewardConfig | None = None,
    **_: Any,
) -> list[float]:
    completions = list(completions or [])
    n = len(completions)
    solutions = _batch(solution if solution is not None else ground_truth, n)
    coords = _batch(coord_config, n)
    tasks = _batch(map_task, n)
    return [
        score_one(
            completion,
            solutions[index],
            coord_config=coords[index],
            map_task=tasks[index],
            config=config,
        )["reward"]
        for index, completion in enumerate(completions)
    ]


__all__ = [
    "SwiftMapRewardConfig",
    "compute_swift_map_rewards",
    "score_one",
]
