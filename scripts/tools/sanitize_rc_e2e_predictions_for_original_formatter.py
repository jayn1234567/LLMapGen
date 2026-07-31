#!/usr/bin/env python3
"""Prepare safe prediction JSON inputs for the untouched RC E2E formatter."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


RELATION_ALL_INSIDE = "all_points_inside"
RELATION_ONE_ENDPOINT_OUTSIDE = "one_endpoint_inside_one_outside"
RELATION_BOTH_OUTSIDE_CROSSING = "both_endpoints_outside_crosses_roi"
RELATION_BOTH_OUTSIDE_NO_INTERSECTION = "both_endpoints_outside_no_roi_intersection"
RELATION_BOTH_INSIDE_LEAVES = "both_endpoints_inside_but_leaves_roi"
ROI_RELATIONS = (
    RELATION_ALL_INSIDE,
    RELATION_ONE_ENDPOINT_OUTSIDE,
    RELATION_BOTH_OUTSIDE_CROSSING,
    RELATION_BOTH_OUTSIDE_NO_INTERSECTION,
    RELATION_BOTH_INSIDE_LEAVES,
)
OUTSIDE_RELATIONS = tuple(relation for relation in ROI_RELATIONS if relation != RELATION_ALL_INSIDE)


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


def point_in_roi(point: list[float], roi_min: float, roi_max: float) -> bool:
    return roi_min <= float(point[0]) <= roi_max and roi_min <= float(point[1]) <= roi_max


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


def classify_polyline_relation(
    points: list[list[float]],
    fragments: list[list[list[float]]],
    roi_min: float,
    roi_max: float,
) -> str:
    inside = [point_in_roi(point, roi_min, roi_max) for point in points]
    if all(inside):
        return RELATION_ALL_INSIDE
    if inside[0] != inside[-1]:
        return RELATION_ONE_ENDPOINT_OUTSIDE
    if not inside[0] and not inside[-1]:
        return RELATION_BOTH_OUTSIDE_CROSSING if fragments else RELATION_BOTH_OUTSIDE_NO_INTERSECTION
    return RELATION_BOTH_INSIDE_LEAVES


def clean_centerline(
    line: dict[str, Any],
    roi_min: float,
    roi_max: float,
) -> tuple[list[dict[str, Any]], int, str | None]:
    points = line.get("points")
    if not isinstance(points, list):
        return [], 1, None

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
        return [], invalid_points, None
    fragments = clip_polyline_to_roi(cleaned, roi_min, roi_max)
    relation = classify_polyline_relation(cleaned, fragments, roi_min, roi_max)
    if not fragments:
        return [], invalid_points, relation

    results = []
    for fragment in fragments:
        result = dict(line)
        result["points"] = fragment
        results.append(result)
    return results, invalid_points, relation


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
        "roi_relation_line_counts": {relation: 0 for relation in ROI_RELATIONS},
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
        fragments, invalid_points, relation = clean_centerline(item, roi_min, roi_max)
        stats["invalid_points"] += invalid_points
        if relation is not None:
            stats["roi_relation_line_counts"][relation] += 1
        if not fragments:
            stats["dropped_centerlines"] += 1
            stats["dropped_outside_roi_centerlines"] += int(
                relation == RELATION_BOTH_OUTSIDE_NO_INTERSECTION
            )
            stats["changed"] = True
            continue
        if invalid_points or len(fragments) != 1 or fragments[0].get("points") != item.get("points"):
            stats["clipped_centerlines"] += 1
            stats["changed"] = True
        stats["kept_centerlines"] += 1
        stats["output_centerline_fragments"] += len(fragments)
        cleaned_lines.extend(fragments)

    return json.dumps({"lines": cleaned_lines}, ensure_ascii=False, separators=(",", ":")), stats


def load_manifest_black_ratios(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Manifest JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        entries = payload.get("patches") or payload.get("records") or payload.get("manifest") or []
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ValueError(f"Manifest must contain a list of patch records: {path}")
    result: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not valid_number(entry.get("black_ratio")):
            continue
        record_id = str(entry.get("record_id") or entry.get("id") or "")
        if record_id:
            result[record_id] = float(entry["black_ratio"])
    return result


def resolve_black_ratio(payload: dict[str, Any], manifest_ratios: dict[str, float]) -> float | None:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    for value in (payload.get("black_ratio"), meta.get("black_ratio")):
        if valid_number(value):
            return float(value)
    record_id = str(payload.get("record_id") or payload.get("id") or "")
    return manifest_ratios.get(record_id)


def black_ratio_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.98:
        return "lt_0.98"
    if value < 1.0:
        return "0.98_to_1.0"
    return "eq_1.0"


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--roi-min", type=float, default=0.0)
    parser.add_argument("--roi-max", type=float, default=1000.0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.roi_min >= args.roi_max:
        raise ValueError("--roi-min must be smaller than --roi-max")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report_json.resolve()
    manifest_path = args.manifest_json.resolve() if args.manifest_json is not None else None
    manifest_ratios = load_manifest_black_ratios(manifest_path)
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
    relation_line_counts: Counter[str] = Counter()
    relation_patch_ids: dict[str, set[str]] = {relation: set() for relation in ROI_RELATIONS}
    bin_patch_ids: dict[str, set[str]] = {
        key: set() for key in ("lt_0.98", "0.98_to_1.0", "eq_1.0", "missing")
    }
    bin_outside_patch_ids: dict[str, set[str]] = {key: set() for key in bin_patch_ids}
    bin_relation_line_counts: dict[str, Counter[str]] = {key: Counter() for key in bin_patch_ids}
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
        record_id = str(payload.get("record_id") or payload.get("id") or path.stem)
        ratio = resolve_black_ratio(payload, manifest_ratios)
        ratio_bin = black_ratio_bin(ratio)
        bin_patch_ids[ratio_bin].add(record_id)
        relation_counts = stats["roi_relation_line_counts"]
        has_outside_line = False
        for relation, count in relation_counts.items():
            relation_line_counts[relation] += int(count)
            bin_relation_line_counts[ratio_bin][relation] += int(count)
            if count:
                relation_patch_ids[relation].add(record_id)
                if relation in OUTSIDE_RELATIONS:
                    has_outside_line = True
        if has_outside_line:
            bin_outside_patch_ids[ratio_bin].add(record_id)

        payload["sanitize_source_parse_ok"] = bool(payload.get("parse_ok", True))
        payload["sanitize_source_parse_error"] = str(payload.get("parse_error") or "")
        payload["prediction_black_ratio"] = ratio
        payload["prediction_black_ratio_bin"] = ratio_bin
        payload["prediction_roi_relation_line_counts"] = relation_counts
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
                    "record_id": record_id,
                    "black_ratio": ratio,
                    "black_ratio_bin": ratio_bin,
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

    outside_line_count = sum(relation_line_counts[relation] for relation in OUTSIDE_RELATIONS)
    outside_patch_ids = set().union(*(relation_patch_ids[relation] for relation in OUTSIDE_RELATIONS))
    sparse_bin = "0.98_to_1.0"
    sparse_outside_line_count = sum(
        bin_relation_line_counts[sparse_bin][relation] for relation in OUTSIDE_RELATIONS
    )
    black_ratio_stats = {}
    for key in bin_patch_ids:
        line_counts = {
            relation: int(bin_relation_line_counts[key][relation]) for relation in ROI_RELATIONS
        }
        bin_outside_lines = sum(line_counts[relation] for relation in OUTSIDE_RELATIONS)
        patch_total = len(bin_patch_ids[key])
        patch_with_outside = len(bin_outside_patch_ids[key])
        black_ratio_stats[key] = {
            "patches_total": patch_total,
            "patches_with_outside_lines": patch_with_outside,
            "patches_with_outside_lines_ratio": safe_ratio(patch_with_outside, patch_total),
            "outside_lines": bin_outside_lines,
            "relation_line_counts": line_counts,
        }

    sparse_patch_incidence = black_ratio_stats[sparse_bin]["patches_with_outside_lines_ratio"]
    normal_patch_incidence = black_ratio_stats["lt_0.98"]["patches_with_outside_lines_ratio"]
    sparse_vs_normal_incidence_ratio = (
        sparse_patch_incidence / normal_patch_incidence if normal_patch_incidence > 0.0 else None
    )

    both_outside_lines = (
        relation_line_counts[RELATION_BOTH_OUTSIDE_CROSSING]
        + relation_line_counts[RELATION_BOTH_OUTSIDE_NO_INTERSECTION]
    )
    both_outside_patch_ids = (
        relation_patch_ids[RELATION_BOTH_OUTSIDE_CROSSING]
        | relation_patch_ids[RELATION_BOTH_OUTSIDE_NO_INTERSECTION]
    )
    roi_relation_stats = {
        "definitions": {
            RELATION_ONE_ENDPOINT_OUTSIDE: "Exactly one of the polyline's first/last endpoints is inside the inclusive ROI.",
            RELATION_BOTH_OUTSIDE_CROSSING: "Both endpoints are outside and at least one polyline segment crosses the ROI.",
            RELATION_BOTH_OUTSIDE_NO_INTERSECTION: "Both endpoints and the full polyline are outside without ROI intersection.",
            RELATION_BOTH_INSIDE_LEAVES: "Both endpoints are inside but at least one intermediate segment leaves the ROI.",
        },
        "line_counts": {relation: int(relation_line_counts[relation]) for relation in ROI_RELATIONS},
        "patch_counts": {relation: len(relation_patch_ids[relation]) for relation in ROI_RELATIONS},
        "requested_summary": {
            "one_endpoint_inside_one_outside_lines": int(
                relation_line_counts[RELATION_ONE_ENDPOINT_OUTSIDE]
            ),
            "one_endpoint_inside_one_outside_patches": len(
                relation_patch_ids[RELATION_ONE_ENDPOINT_OUTSIDE]
            ),
            "both_endpoints_outside_lines": int(both_outside_lines),
            "both_endpoints_outside_patches": len(both_outside_patch_ids),
            "both_endpoints_outside_crosses_roi_lines": int(
                relation_line_counts[RELATION_BOTH_OUTSIDE_CROSSING]
            ),
            "both_endpoints_outside_crosses_roi_patches": len(
                relation_patch_ids[RELATION_BOTH_OUTSIDE_CROSSING]
            ),
            "both_endpoints_outside_no_roi_intersection_lines": int(
                relation_line_counts[RELATION_BOTH_OUTSIDE_NO_INTERSECTION]
            ),
            "both_endpoints_outside_no_roi_intersection_patches": len(
                relation_patch_ids[RELATION_BOTH_OUTSIDE_NO_INTERSECTION]
            ),
            "all_outside_related_lines": int(outside_line_count),
            "all_outside_related_patches": len(outside_patch_ids),
            "black_ratio_0.98_to_1.0_outside_lines": int(sparse_outside_line_count),
            "black_ratio_0.98_to_1.0_share_of_outside_lines": safe_ratio(
                sparse_outside_line_count, outside_line_count
            ),
            "black_ratio_0.98_to_1.0_outside_patches": len(
                bin_outside_patch_ids[sparse_bin]
            ),
            "black_ratio_0.98_to_1.0_share_of_outside_patches": safe_ratio(
                len(bin_outside_patch_ids[sparse_bin]), len(outside_patch_ids)
            ),
            "black_ratio_0.98_to_1.0_outside_patch_incidence": sparse_patch_incidence,
            "black_ratio_lt_0.98_outside_patch_incidence": normal_patch_incidence,
            "black_ratio_0.98_to_1.0_incidence_ratio_vs_lt_0.98": sparse_vs_normal_incidence_ratio,
        },
        "black_ratio_bins": black_ratio_stats,
    }

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "manifest_json": str(manifest_path) if manifest_path is not None else "",
        "manifest_black_ratio_records": len(manifest_ratios),
        "roi": {"min": args.roi_min, "max": args.roi_max, "inclusive": True},
        "policy": (
            "The original RC E2E project is unchanged. Invalid prediction JSON becomes an empty lines payload; "
            "invalid 2D points are removed; centerlines with fewer than two valid points are removed; centerlines "
            "are geometrically clipped to the normalized ROI; disconnected in-ROI fragments are split into "
            "separate centerlines; centerlines with no ROI intersection are removed. Cached pixel predictions are "
            "removed from the sanitized copies so metrics recompute pixels from the clipped normalized geometry."
        ),
        "totals": totals,
        "roi_relation_stats": roi_relation_stats,
        "outer_errors": outer_errors,
        "affected_records": affected,
    }
    report["complete"] = not outer_errors and totals["input_files"] == totals["output_files"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "totals": totals,
                "roi_requested_summary": roi_relation_stats["requested_summary"],
                "black_ratio_bins": roi_relation_stats["black_ratio_bins"],
                "outer_errors": outer_errors,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
