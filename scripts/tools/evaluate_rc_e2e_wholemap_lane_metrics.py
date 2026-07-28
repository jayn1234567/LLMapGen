#!/usr/bin/env python3
"""Evaluate raw RC patch predictions with the official E2E lane metric recipe.

The script reconstructs patch-local predictions in each scene's projected map
frame, conservatively stitches directed lines across patch boundaries, applies
the official high/low road masks, and runs direction-aware Hungarian matching.

This is an official-metric-compatible direct evaluator.  It does not replace
the proprietary center_lane_rule post-processing used by the production E2E
pipeline, so the output records that distinction explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rasterio.warp import transform_geom
from scipy.optimize import linear_sum_assignment
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    shape,
)
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.centerline_eval_metrics import extract_json_payload
from scripts.tools.evaluate_rc_e2e_patch_metrics import (
    build_inter_tif_index,
    expected_lane_tif,
    feature_lane_type,
    find_lane_gt,
    geojson_crs,
    load_prediction_records,
    pixel_prediction,
    record_row_col,
    record_scene_and_tif,
    scene_root_for_inter_tif,
)


@dataclass
class PredSegment:
    line: LineString
    source_key: str
    start_is_boundary: bool
    end_is_boundary: bool


@dataclass
class MetricTotals:
    scenes: int = 0
    gt_lane_num: int = 0
    pred_lane_num: int = 0
    matched_lane_num: int = 0
    gt_length: int = 0
    pred_length: int = 0
    gt_matched_length: int = 0
    pred_matched_length: int = 0

    def add(self, other: "MetricTotals") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-e2e-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--baseline-name", default="gt")
    parser.add_argument("--gt-crs", default="EPSG:4326")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--ignore-lane-types", default="3,4,22")
    parser.add_argument("--lane-buffer-size", type=float, default=2.5)
    parser.add_argument("--lane-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--direction-threshold-deg", type=float, default=10.0)
    parser.add_argument("--stitch-distance", type=float, default=1.0)
    parser.add_argument("--stitch-direction-threshold-deg", type=float, default=20.0)
    parser.add_argument("--boundary-pixel-tolerance", type=float, default=2.0)
    parser.add_argument("--min-mask-segment-length", type=float, default=1.0)
    parser.add_argument("--min-intersection-cut-segment-length", type=float, default=15.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--stitch-patch-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cut-predicted-intersections",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-all",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def parse_int_set(text: str) -> set[int]:
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def iter_lines(geometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        if len(geometry.coords) >= 2 and geometry.length > 0:
            yield geometry
        return
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from iter_lines(part)


def iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        if geometry.area > 0:
            yield geometry
        return
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from iter_polygons(part)


def clean_points(points: Any, minimum: int) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    output: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return []
        if not math.isfinite(x) or not math.isfinite(y):
            return []
        current = (x, y)
        if not output or current != output[-1]:
            output.append(current)
    return output if len(output) >= minimum else []


def payload_items(text: str) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(extract_json_payload(text))
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if isinstance(payload, dict):
        items = payload.get("lines", [])
    elif isinstance(payload, list):
        items = payload
    else:
        return [], f"unsupported prediction payload: {type(payload).__name__}"
    if not isinstance(items, list):
        return [], "prediction lines is not a list"
    return [item for item in items if isinstance(item, dict)], ""


def category(item: dict[str, Any]) -> str:
    value = str(item.get("category", "centerline")).strip().lower()
    return "centerline" if value == "centerline" else value


def is_boundary_point(point: tuple[float, float], patch_size: int, tolerance: float) -> bool:
    maximum = float(patch_size - 1)
    x, y = point
    return x <= tolerance or y <= tolerance or x >= maximum - tolerance or y >= maximum - tolerance


def map_points(transform, row: int, col: int, patch_size: int, points: list[tuple[float, float]]):
    x0 = col * patch_size
    y0 = row * patch_size
    return [transform * (x0 + x, y0 + y) for x, y in points]


def transform_geometry(geometry, source_crs: str, target_crs: str):
    if str(source_crs) == str(target_crs):
        return geometry
    transformed = transform_geom(source_crs, target_crs, geometry.__geo_interface__, precision=-1)
    return shape(transformed)


def load_gt_lines(
    path: Path,
    target_crs: str,
    fallback_source_crs: str,
    ignored_types: set[int],
) -> tuple[list[LineString], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_crs = geojson_crs(payload, fallback_source_crs)
    lines: list[LineString] = []
    filtered: dict[str, int] = defaultdict(int)
    for feature in payload.get("features", []):
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        lane_type = feature_lane_type(properties)
        if lane_type == 0:
            lane_type = 1
        if lane_type in ignored_types:
            filtered[str(lane_type)] += 1
            continue
        geometry = shape(
            transform_geom(source_crs, target_crs, feature["geometry"], precision=-1)
        )
        lines.extend(iter_lines(geometry))
    return lines, dict(filtered)


def load_mask(path: Path, target_crs: str, fallback_source_crs: str):
    if not path.is_file():
        return GeometryCollection()
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_crs = geojson_crs(payload, fallback_source_crs)
    polygons = []
    for feature in payload.get("features", []):
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            continue
        geometry = shape(
            transform_geom(source_crs, target_crs, feature["geometry"], precision=-1)
        )
        polygons.extend(iter_polygons(geometry))
    return unary_union(polygons) if polygons else GeometryCollection()


def endpoint_heading(line: LineString, at_start: bool) -> float | None:
    coords = list(line.coords)
    if at_start:
        pairs = zip(coords[:-1], coords[1:])
    else:
        pairs = zip(reversed(coords[:-1]), reversed(coords[1:]))
    for origin, target in pairs:
        dx = float(target[0]) - float(origin[0])
        dy = float(target[1]) - float(origin[1])
        if math.hypot(dx, dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx)) % 360.0
    return None


def direction_delta(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return 180.0
    return abs((second - first + 180.0) % 360.0 - 180.0)


def line_direction(line: LineString) -> float | None:
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    first = coords[0]
    for last in reversed(coords[1:]):
        dx = float(last[0]) - float(first[0])
        dy = float(last[1]) - float(first[1])
        if math.hypot(dx, dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx)) % 360.0
    return None


def stitch_segments(
    segments: list[PredSegment],
    distance_threshold: float,
    direction_threshold: float,
) -> list[LineString]:
    if len(segments) < 2 or distance_threshold <= 0:
        return [segment.line for segment in segments]

    cell_size = max(distance_threshold, 1e-6)
    start_grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, segment in enumerate(segments):
        if not segment.start_is_boundary:
            continue
        x, y = segment.line.coords[0][:2]
        start_grid[(math.floor(x / cell_size), math.floor(y / cell_size))].append(index)

    candidates: list[tuple[float, float, int, int]] = []
    for left_index, left in enumerate(segments):
        if not left.end_is_boundary:
            continue
        end_x, end_y = left.line.coords[-1][:2]
        cell = (math.floor(end_x / cell_size), math.floor(end_y / cell_size))
        left_heading = endpoint_heading(left.line, at_start=False)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for right_index in start_grid.get((cell[0] + dx, cell[1] + dy), []):
                    if left_index == right_index:
                        continue
                    right = segments[right_index]
                    if left.source_key == right.source_key:
                        continue
                    start_x, start_y = right.line.coords[0][:2]
                    distance = math.hypot(start_x - end_x, start_y - end_y)
                    if distance > distance_threshold:
                        continue
                    delta = direction_delta(left_heading, endpoint_heading(right.line, at_start=True))
                    if delta <= direction_threshold:
                        candidates.append((distance, delta, left_index, right_index))

    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for _, _, left_index, right_index in sorted(candidates):
        if left_index in successor or right_index in predecessor:
            continue
        successor[left_index] = right_index
        predecessor[right_index] = left_index

    chains: list[list[int]] = []
    visited: set[int] = set()
    for index in range(len(segments)):
        if index in predecessor:
            continue
        chain = []
        current = index
        while current not in visited:
            visited.add(current)
            chain.append(current)
            if current not in successor:
                break
            current = successor[current]
        chains.append(chain)
    for index in range(len(segments)):
        if index in visited:
            continue
        chain = []
        current = index
        while current not in visited:
            visited.add(current)
            chain.append(current)
            current = successor.get(current, current)
        chains.append(chain)

    output: list[LineString] = []
    for chain in chains:
        coords: list[tuple[float, float]] = []
        for index in chain:
            part = [(float(x), float(y)) for x, y, *_ in segments[index].line.coords]
            if coords and part and coords[-1] == part[0]:
                part = part[1:]
            coords.extend(part)
        if len(coords) >= 2:
            output.append(LineString(coords))
    return output


def cut_lines_by_intersections(
    lines: list[LineString],
    polygons: list[Polygon],
    minimum_cut_length: float,
) -> list[LineString]:
    if not polygons:
        return lines
    result: list[LineString] = []
    for source in lines:
        segments = [source]
        for polygon in polygons:
            buffered = polygon.buffer(1e-6)
            next_segments: list[LineString] = []
            for segment in segments:
                if segment.within(buffered):
                    continue
                if not segment.crosses(buffered):
                    next_segments.append(segment)
                    continue
                outside = segment.difference(buffered)
                next_segments.extend(
                    part for part in iter_lines(outside) if part.length > minimum_cut_length
                )
            segments = next_segments
        result.extend(segments)
    return result


def keep_inside_mask(lines: list[LineString], mask, minimum_length: float) -> list[LineString]:
    if mask is None or mask.is_empty:
        return []
    output: list[LineString] = []
    for line in lines:
        try:
            clipped = line.intersection(mask)
        except Exception:
            continue
        output.extend(part for part in iter_lines(clipped) if part.length > minimum_length)
    return output


def lane_match_score(
    gt_line: LineString,
    pred_line: LineString,
    buffer_size: float,
    direction_threshold: float,
) -> float:
    if direction_delta(
        line_direction(gt_line),
        line_direction(pred_line),
    ) > direction_threshold:
        return 0.0
    denominator = gt_line.length + pred_line.length
    if denominator <= 0:
        return 0.0
    try:
        gt_overlap = gt_line.intersection(pred_line.buffer(buffer_size)).length
        pred_overlap = pred_line.intersection(gt_line.buffer(buffer_size)).length
    except Exception:
        return 0.0
    return float((gt_overlap + pred_overlap) / denominator)


def match_lanes(
    gt_lines: list[LineString],
    pred_lines: list[LineString],
    buffer_size: float,
    direction_threshold: float,
    overlap_threshold: float,
) -> tuple[list[int], list[int]]:
    if not gt_lines or not pred_lines:
        return [], []
    pair_count = len(gt_lines) * len(pred_lines)
    if pair_count > 50_000_000:
        print(
            f"[e2e-wholemap-eval] WARNING: large Hungarian matrix "
            f"{len(gt_lines)}x{len(pred_lines)} ({pair_count} pairs)",
            flush=True,
        )
    scores = np.zeros((len(gt_lines), len(pred_lines)), dtype=np.float32)
    for gt_index, gt_line in enumerate(gt_lines):
        min_x, min_y, max_x, max_y = gt_line.bounds
        expanded = (
            min_x - buffer_size,
            min_y - buffer_size,
            max_x + buffer_size,
            max_y + buffer_size,
        )
        for pred_index, pred_line in enumerate(pred_lines):
            pred_bounds = pred_line.bounds
            if (
                pred_bounds[2] < expanded[0]
                or pred_bounds[0] > expanded[2]
                or pred_bounds[3] < expanded[1]
                or pred_bounds[1] > expanded[3]
            ):
                continue
            scores[gt_index, pred_index] = lane_match_score(
                gt_line,
                pred_line,
                buffer_size,
                direction_threshold,
            )
    rows, columns = linear_sum_assignment(1.0 - scores)
    matched_gt: list[int] = []
    matched_pred: list[int] = []
    for gt_index, pred_index in zip(rows, columns):
        if scores[gt_index, pred_index] >= overlap_threshold:
            matched_gt.append(int(gt_index))
            matched_pred.append(int(pred_index))
    return matched_gt, matched_pred


def scene_metrics(
    gt_lines: list[LineString],
    pred_lines: list[LineString],
    args: argparse.Namespace,
) -> MetricTotals:
    matched_gt, matched_pred = match_lanes(
        gt_lines,
        pred_lines,
        args.lane_buffer_size,
        args.direction_threshold_deg,
        args.lane_overlap_threshold,
    )
    return MetricTotals(
        scenes=1,
        gt_lane_num=len(gt_lines),
        pred_lane_num=len(pred_lines),
        matched_lane_num=len(matched_gt),
        gt_length=int(sum(line.length for line in gt_lines)),
        pred_length=int(sum(line.length for line in pred_lines)),
        gt_matched_length=int(sum(gt_lines[index].length for index in matched_gt)),
        pred_matched_length=int(sum(pred_lines[index].length for index in matched_pred)),
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) >= 1e-6 else 1.0


def harmonic_mean(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize(totals: MetricTotals) -> dict[str, Any]:
    instance_precision = safe_ratio(totals.matched_lane_num, totals.pred_lane_num)
    instance_recall = safe_ratio(totals.matched_lane_num, totals.gt_lane_num)
    length_precision = safe_ratio(totals.pred_matched_length, totals.pred_length)
    length_recall = safe_ratio(totals.gt_matched_length, totals.gt_length)
    return {
        "instance_precision": round(instance_precision, 6),
        "instance_recall": round(instance_recall, 6),
        "instance_f1": round(harmonic_mean(instance_precision, instance_recall), 6),
        "length_precision": round(length_precision, 6),
        "length_recall": round(length_recall, 6),
        "length_f1": round(harmonic_mean(length_precision, length_recall), 6),
        "raw_totals": totals.__dict__,
    }


def scene_target_crs(records: list[dict[str, Any]], inter_index) -> tuple[str, Path]:
    import rasterio

    first = records[0]
    scene_id, tif_stem = record_scene_and_tif(first)
    lane_tif = expected_lane_tif(inter_index[(scene_id, tif_stem)])
    with rasterio.open(lane_tif) as source:
        if source.crs is None:
            raise ValueError(f"Lane TIF has no CRS: {lane_tif}")
        return source.crs.to_string(), lane_tif


def build_scene_predictions(
    records: list[dict[str, Any]],
    inter_index,
    target_crs: str,
    args: argparse.Namespace,
) -> tuple[list[PredSegment], list[Polygon], list[dict[str, str]]]:
    import rasterio

    raster_cache: dict[Path, tuple[Any, str]] = {}
    segments: list[PredSegment] = []
    polygons: list[Polygon] = []
    errors: list[dict[str, str]] = []
    maximum = float(args.patch_size - 1)

    for record in records:
        record_id = str(record.get("record_id") or record.get("id") or record.get("_prediction_file"))
        scene_id, tif_stem = record_scene_and_tif(record)
        inter_tif = inter_index[(scene_id, tif_stem)]
        lane_tif = expected_lane_tif(inter_tif)
        if lane_tif not in raster_cache:
            with rasterio.open(lane_tif) as source:
                if source.crs is None:
                    raise ValueError(f"Lane TIF has no CRS: {lane_tif}")
                raster_cache[lane_tif] = (source.transform, source.crs.to_string())
        transform, raster_crs = raster_cache[lane_tif]
        row, col = record_row_col(record)
        prediction_text, parse_ok, parse_error = pixel_prediction(
            record,
            args.patch_size,
            args.coord_range,
        )
        if not parse_ok or not prediction_text:
            errors.append({"record_id": record_id, "error": parse_error or "invalid prediction"})
            continue
        items, item_error = payload_items(prediction_text)
        if item_error:
            errors.append({"record_id": record_id, "error": item_error})
            continue

        for item_index, item in enumerate(items):
            item_category = category(item)
            minimum_points = 3 if item_category == "intersection" else 2
            local_points = clean_points(item.get("points"), minimum_points)
            if not local_points:
                continue
            local_points = [
                (min(max(x, 0.0), maximum), min(max(y, 0.0), maximum))
                for x, y in local_points
            ]
            projected_points = map_points(transform, row, col, args.patch_size, local_points)
            if raster_crs != target_crs:
                if item_category == "intersection":
                    geometry = transform_geometry(Polygon(projected_points), raster_crs, target_crs)
                    projected_points = list(geometry.exterior.coords)
                else:
                    geometry = transform_geometry(LineString(projected_points), raster_crs, target_crs)
                    projected_points = list(geometry.coords)

            if item_category == "centerline":
                line = LineString(projected_points)
                if not line.is_empty and line.length > 0:
                    source_key = f"{tif_stem}:{row}:{col}"
                    start_type = str(item.get("start_type") or "").lower()
                    end_type = str(item.get("end_type") or "").lower()
                    segments.append(
                        PredSegment(
                            line=line,
                            source_key=source_key,
                            start_is_boundary=(
                                start_type == "cut"
                                or is_boundary_point(
                                    local_points[0], args.patch_size, args.boundary_pixel_tolerance
                                )
                            ),
                            end_is_boundary=(
                                end_type == "cut"
                                or is_boundary_point(
                                    local_points[-1], args.patch_size, args.boundary_pixel_tolerance
                                )
                            ),
                        )
                    )
            elif item_category == "intersection":
                polygon = Polygon(projected_points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygons.extend(iter_polygons(polygon))
    return segments, polygons, errors


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_e2e_root).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    output_json = Path(args.output_json).resolve()
    ignored_types = parse_int_set(args.ignore_lane_types)

    predictions, json_files_seen = load_prediction_records(prediction_dir)
    if args.max_samples > 0:
        predictions = predictions[: args.max_samples]
    if not predictions:
        raise FileNotFoundError(f"No raw per-patch predictions found below {prediction_dir}")

    inter_index = build_inter_tif_index(raw_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairing_errors: list[dict[str, str]] = []
    for record in predictions:
        try:
            scene_id, tif_stem = record_scene_and_tif(record)
            if (scene_id, tif_stem) not in inter_index:
                raise KeyError(f"inter TIF not found for scene={scene_id} tif={tif_stem}")
            grouped[scene_id].append(record)
        except Exception as exc:
            pairing_errors.append(
                {
                    "record_id": str(record.get("record_id") or record.get("id") or ""),
                    "error": repr(exc),
                }
            )
    if pairing_errors and args.require_all:
        raise RuntimeError(
            f"Unable to pair all prediction records; errors={len(pairing_errors)}\n"
            + json.dumps(pairing_errors[:10], ensure_ascii=False, indent=2)
        )

    aggregate = {mode: MetricTotals() for mode in ("all", "high", "low")}
    by_scene: dict[str, Any] = {}
    prediction_errors: list[dict[str, str]] = []
    filtered_lane_types: dict[str, int] = defaultdict(int)
    total_raw_segments = 0
    total_stitched_lines = 0
    total_predicted_intersections = 0

    for scene_index, (scene_id, records) in enumerate(sorted(grouped.items()), 1):
        target_crs, reference_lane_tif = scene_target_crs(records, inter_index)
        first_scene, first_tif = record_scene_and_tif(records[0])
        scene_root = scene_root_for_inter_tif(inter_index[(first_scene, first_tif)])
        gt_path = find_lane_gt(scene_root, args.baseline_name)
        gt_lines, filtered = load_gt_lines(gt_path, target_crs, args.gt_crs, ignored_types)
        for lane_type, count in filtered.items():
            filtered_lane_types[lane_type] += count

        gt_root = scene_root / args.baseline_name
        high_mask = load_mask(gt_root / "high.geojson", target_crs, args.gt_crs)
        low_mask = load_mask(gt_root / "low.geojson", target_crs, args.gt_crs)
        missing_masks = [
            name for name, mask in (("high", high_mask), ("low", low_mask)) if mask.is_empty
        ]
        if missing_masks and args.require_masks:
            raise FileNotFoundError(
                f"Missing or empty E2E masks for scene {scene_id}: {missing_masks}; gt_root={gt_root}"
            )

        pred_segments, pred_intersections, parse_errors = build_scene_predictions(
            records,
            inter_index,
            target_crs,
            args,
        )
        prediction_errors.extend({"scene_id": scene_id, **item} for item in parse_errors)
        raw_count = len(pred_segments)
        pred_lines = (
            stitch_segments(
                pred_segments,
                args.stitch_distance,
                args.stitch_direction_threshold_deg,
            )
            if args.stitch_patch_lines
            else [segment.line for segment in pred_segments]
        )
        stitched_count = len(pred_lines)
        if args.cut_predicted_intersections:
            gt_lines = cut_lines_by_intersections(
                gt_lines,
                pred_intersections,
                args.min_intersection_cut_segment_length,
            )
            pred_lines = cut_lines_by_intersections(
                pred_lines,
                pred_intersections,
                args.min_intersection_cut_segment_length,
            )

        masks = {
            "all": unary_union([mask for mask in (high_mask, low_mask) if not mask.is_empty]),
            "high": high_mask,
            "low": low_mask,
        }
        scene_result: dict[str, Any] = {
            "patch_records": len(records),
            "invalid_prediction_records": len(parse_errors),
            "raw_prediction_segments": raw_count,
            "stitched_prediction_lines": stitched_count,
            "predicted_intersection_polygons": len(pred_intersections),
            "reference_lane_tif": str(reference_lane_tif),
            "lane_gt": str(gt_path),
            "metrics": {},
        }
        for mode, mask in masks.items():
            masked_gt = keep_inside_mask(gt_lines, mask, args.min_mask_segment_length)
            masked_pred = keep_inside_mask(pred_lines, mask, args.min_mask_segment_length)
            totals = scene_metrics(masked_gt, masked_pred, args)
            aggregate[mode].add(totals)
            scene_result["metrics"][mode] = summarize(totals)
        by_scene[scene_id] = scene_result
        total_raw_segments += raw_count
        total_stitched_lines += stitched_count
        total_predicted_intersections += len(pred_intersections)
        print(
            f"[e2e-wholemap-eval] scene={scene_index}/{len(grouped)} id={scene_id} "
            f"patches={len(records)} raw_lines={raw_count} stitched={stitched_count} "
            f"invalid={len(parse_errors)}",
            flush=True,
        )

    result = {
        "backend": "rc_e2e_lane_pr_compatible.direct_wholemap",
        "compatibility_note": (
            "Uses the supplied E2E lane metric formula and masks, but replaces the external "
            "center_lane_rule post-processing with conservative directed boundary stitching."
        ),
        "parameters": {
            "ignore_lane_types": sorted(ignored_types),
            "lane_buffer_size_m": args.lane_buffer_size,
            "lane_overlap_threshold": args.lane_overlap_threshold,
            "direction_threshold_deg": args.direction_threshold_deg,
            "stitch_patch_lines": args.stitch_patch_lines,
            "stitch_distance_m": args.stitch_distance,
            "stitch_direction_threshold_deg": args.stitch_direction_threshold_deg,
            "cut_predicted_intersections": args.cut_predicted_intersections,
            "minimum_mask_segment_length_m": args.min_mask_segment_length,
            "minimum_intersection_cut_segment_length_m": args.min_intersection_cut_segment_length,
        },
        "input": {
            "raw_e2e_root": str(raw_root),
            "prediction_dir": str(prediction_dir),
            "json_files_seen": json_files_seen,
            "prediction_records": len(predictions),
            "paired_prediction_records": sum(len(records) for records in grouped.values()),
            "scene_count": len(grouped),
        },
        "prediction_quality": {
            "invalid_prediction_records": len(prediction_errors),
            "valid_prediction_ratio": round(
                (len(predictions) - len(prediction_errors)) / len(predictions), 6
            ),
            "raw_prediction_segments": total_raw_segments,
            "stitched_prediction_lines": total_stitched_lines,
            "predicted_intersection_polygons": total_predicted_intersections,
        },
        "metrics": {mode: summarize(totals) for mode, totals in aggregate.items()},
        "by_scene": by_scene,
        "filtered_gt_lane_types": dict(sorted(filtered_lane_types.items())),
        "pairing_errors": pairing_errors,
        "prediction_errors": prediction_errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"[e2e-wholemap-eval] output={output_json}", flush=True)


if __name__ == "__main__":
    main()
