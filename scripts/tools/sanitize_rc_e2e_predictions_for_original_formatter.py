#!/usr/bin/env python3
"""Prepare safe prediction JSON inputs for the untouched RC E2E formatter."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def clip_segment_to_roi(
    start: list[float],
    end: list[float],
    roi_min: float,
    roi_max: float,
) -> tuple[list[float], list[float]] | None:
    """Clip a segment to the inclusive square ROI with Liang-Barsky."""
    x0, y0 = float(start[0]), float(start[1])
    dx = float(end[0]) - x0
    dy = float(end[1]) - y0
    lower = 0.0
    upper = 1.0
    for direction, distance in (
        (-dx, x0 - roi_min),
        (dx, roi_max - x0),
        (-dy, y0 - roi_min),
        (dy, roi_max - y0),
    ):
        if direction == 0.0:
            if distance < 0.0:
                return None
            continue
        ratio = distance / direction
        if direction < 0.0:
            if ratio > upper:
                return None
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return None
            upper = min(upper, ratio)
    if lower > upper:
        return None
    return (
        [x0 + lower * dx, y0 + lower * dy],
        [x0 + upper * dx, y0 + upper * dy],
    )


def points_equal(first: list[float], second: list[float], tolerance: float = 1e-9) -> bool:
    return abs(float(first[0]) - float(second[0])) <= tolerance and abs(
        float(first[1]) - float(second[1])
    ) <= tolerance


def clip_polyline_to_roi(
    points: list[list[float]],
    roi_min: float,
    roi_max: float,
) -> list[list[list[float]]]:
    """Clip a polyline and return each connected in-ROI fragment."""
    fragments: list[list[list[float]]] = []
    current: list[list[float]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2 and not all(points_equal(current[0], point) for point in current[1:]):
            fragments.append(current)
        current = []

    for start, end in zip(points, points[1:]):
        clipped = clip_segment_to_roi(start, end, roi_min, roi_max)
        if clipped is None:
            flush()
            continue
        clipped_start, clipped_end = clipped
        if points_equal(clipped_start, clipped_end):
            flush()
            continue
        if current and points_equal(current[-1], clipped_start):
            if not points_equal(current[-1], clipped_end):
                current.append(clipped_end)
        else:
            flush()
            current = [clipped_start, clipped_end]
    flush()
    return fragments


def clean_centerline(
    line: dict[str, Any],
    roi_min: float,
    roi_max: float,
) -> tuple[list[dict[str, Any]], int, bool]:
    points = line.get("points")
    if not isinstance(points, list):
        return [], 1, False

    cleaned = []
    invalid_points = 0
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            invalid_points += 1
            continue
        if not valid_number(point[0]) or not valid_number(point[1]):
            invalid_points += 1
            continue
        cleaned.append([point[0], point[1]])
    if len(cleaned) < 2:
        return [], invalid_points, False
    fragments = clip_polyline_to_roi(cleaned, roi_min, roi_max)
    if not fragments:
        return [], invalid_points, True

    results = []
    for fragment in fragments:
        result = dict(line)
        result["points"] = fragment
        results.append(result)
    return results, invalid_points, False


def sanitize_prediction(
    text: Any,
    roi_min: float = 0.0,
    roi_max: float = 1000.0,
) -> tuple[str, dict[str, Any]]:
    stats = {
        "prediction_parse_error": "",
        "invalid_items": 0,
        "invalid_points": 0,
        "dropped_centerlines": 0,
        "dropped_outside_roi_centerlines": 0,
        "clipped_centerlines": 0,
        "output_centerline_fragments": 0,
        "kept_centerlines": 0,
        "changed": False,
    }
    try:
        payload = json.loads(text) if isinstance(text, str) else text
    except Exception as exc:
        stats["prediction_parse_error"] = repr(exc)
        stats["changed"] = True
        return '{"lines":[]}', stats

    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        stats["prediction_parse_error"] = "prediction payload does not contain a lines list"
        stats["changed"] = True
        return '{"lines":[]}', stats

    cleaned_lines = []
    for item in payload["lines"]:
        if not isinstance(item, dict) or "category" not in item:
            stats["invalid_items"] += 1
            stats["changed"] = True
            continue
        if str(item.get("category", "")).strip().lower() != "centerline":
            cleaned_lines.append(item)
            continue
        fragments, invalid_points, outside_roi = clean_centerline(item, roi_min, roi_max)
        stats["invalid_points"] += invalid_points
        if not fragments:
            stats["dropped_centerlines"] += 1
            stats["dropped_outside_roi_centerlines"] += int(outside_roi)
            stats["changed"] = True
            continue
        if invalid_points or len(fragments) != 1 or fragments[0].get("points") != item.get("points"):
            stats["clipped_centerlines"] += 1
            stats["changed"] = True
        stats["kept_centerlines"] += 1
        stats["output_centerline_fragments"] += len(fragments)
        cleaned_lines.extend(fragments)

    return json.dumps({"lines": cleaned_lines}, ensure_ascii=False, separators=(",", ":")), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--roi-min", type=float, default=0.0)
    parser.add_argument("--roi-max", type=float, default=1000.0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.roi_min >= args.roi_max:
        raise ValueError("--roi-min must be smaller than --roi-max")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report_json.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ")
    if output_dir.exists() and args.reset:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    affected = []
    outer_errors = []
    totals = {
        "input_files": len(files),
        "output_files": 0,
        "changed_files": 0,
        "prediction_parse_errors": 0,
        "invalid_items": 0,
        "invalid_points": 0,
        "dropped_centerlines": 0,
        "dropped_outside_roi_centerlines": 0,
        "clipped_centerlines": 0,
        "output_centerline_fragments": 0,
        "kept_centerlines": 0,
    }

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise TypeError("outer prediction payload is not an object")
        except Exception as exc:
            outer_errors.append({"file": str(path), "error": repr(exc)})
            continue

        sanitized_text, stats = sanitize_prediction(
            payload.get("prediction_json"),
            roi_min=args.roi_min,
            roi_max=args.roi_max,
        )
        payload["sanitize_source_parse_ok"] = bool(payload.get("parse_ok", True))
        payload["sanitize_source_parse_error"] = str(payload.get("parse_error") or "")
        payload["prediction_json"] = sanitized_text
        for pixel_key in ("prediction_json_pixel", "response_pixel", "prediction_pixel"):
            payload.pop(pixel_key, None)
        payload["parse_ok"] = True
        payload["parse_error"] = ""
        payload["prediction_roi_clipped"] = True
        destination = output_dir / path.name
        destination.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        if stats["changed"]:
            totals["changed_files"] += 1
            affected.append(
                {
                    "file": str(path),
                    "record_id": str(payload.get("record_id") or payload.get("id") or ""),
                    **stats,
                }
            )

        totals["output_files"] += 1
        totals["prediction_parse_errors"] += int(bool(stats["prediction_parse_error"]))
        for key in (
            "invalid_items",
            "invalid_points",
            "dropped_centerlines",
            "dropped_outside_roi_centerlines",
            "clipped_centerlines",
            "output_centerline_fragments",
            "kept_centerlines",
        ):
            totals[key] += int(stats[key])

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "roi": {"min": args.roi_min, "max": args.roi_max, "inclusive": True},
        "policy": (
            "The original RC E2E project is unchanged. Invalid prediction JSON becomes an empty lines payload; "
            "invalid 2D points are removed; centerlines with fewer than two valid points are removed; centerlines "
            "are geometrically clipped to the normalized ROI; disconnected in-ROI fragments are split into "
            "separate centerlines; centerlines with no ROI intersection are removed. Cached pixel predictions are "
            "removed from the sanitized copies so metrics recompute pixels from the clipped normalized geometry."
        ),
        "totals": totals,
        "outer_errors": outer_errors,
        "affected_records": affected,
    }
    report["complete"] = not outer_errors and totals["input_files"] == totals["output_files"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": totals, "outer_errors": outer_errors, "report": str(report_path)}, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
