#!/usr/bin/env python3
"""Evaluate raw RC E2E patch predictions against scene-level lane GeoJSON GT."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rasterio.warp import transform_geom
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon, shape
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_index.line_eval import evaluate_records, format_eval_table
from mllm.coord_utils import COORD_MODE_NORM1000, COORD_MODE_PIXEL, convert_payload_text
from scripts.tools.prepare_rc_e2e_inference_dataset import discover_inter_tifs, scene_id_for_tif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-e2e-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-eval-jsonl", default="")
    parser.add_argument("--baseline-name", default="gt")
    parser.add_argument("--gt-crs", default="EPSG:4326")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--buffer-size", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=0.33)
    parser.add_argument("--ignore-lane-types", default="3,4,22")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--require-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when a prediction cannot be paired with raster metadata or lane GT.",
    )
    return parser.parse_args()


def parse_int_set(text: str) -> set[int]:
    values: set[int] = set()
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            values.add(int(item))
    return values


def load_prediction_records(prediction_dir: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    files_seen = 0
    for path in sorted(prediction_dir.rglob("*.json")):
        files_seen += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not any(key in payload for key in ("prediction_json", "prediction_json_pixel", "prediction")):
            continue
        payload["_prediction_file"] = str(path)
        records.append(payload)
    records.sort(key=lambda item: str(item.get("record_id") or item.get("id") or item["_prediction_file"]))

    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        record_id = str(record.get("record_id") or record.get("id") or record["_prediction_file"])
        if record_id in seen:
            duplicates.append(record_id)
        seen.add(record_id)
    if duplicates:
        raise ValueError(f"Duplicate prediction record IDs: {duplicates[:10]}")
    return records, files_seen


def scene_root_for_inter_tif(inter_tif: Path) -> Path:
    for parent in inter_tif.parents:
        if parent.name == "rc_one_patch_release":
            return parent.parent
    raise ValueError(f"Unable to find scene root for {inter_tif}")


def build_inter_tif_index(raw_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in discover_inter_tifs(raw_root):
        key = (scene_id_for_tif(path), path.stem)
        if key in index:
            raise ValueError(f"Duplicate inter TIF key {key}: {index[key]} and {path}")
        index[key] = path
    if not index:
        raise FileNotFoundError(f"No inter_patch_tif/*_inter.tif found below {raw_root}")
    return index


def record_scene_and_tif(record: dict[str, Any]) -> tuple[str, str]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    scene_id = str(meta.get("scene_id") or "")
    tif_stem = str(meta.get("tif_stem") or "")
    image = Path(str(record.get("image") or ""))
    if not tif_stem and image.parent.name:
        tif_stem = image.parent.name
    if not scene_id:
        parts = image.parts
        if "images" in parts:
            idx = parts.index("images")
            if idx + 1 < len(parts):
                scene_id = parts[idx + 1]
    if not scene_id or not tif_stem:
        raise ValueError(
            f"Cannot resolve scene/tif from record {record.get('record_id')}: "
            f"scene={scene_id!r} tif={tif_stem!r} image={str(image)!r}"
        )
    return scene_id, tif_stem


def record_row_col(record: dict[str, Any]) -> tuple[int, int]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    row = record.get("row", meta.get("row", meta.get("patch_row")))
    col = record.get("col", meta.get("col", meta.get("patch_col")))
    if row is not None and col is not None:
        return int(row), int(col)
    stem = Path(str(record.get("image") or "")).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-2]), int(parts[-1])
    raise ValueError(f"Cannot resolve row/col for record {record.get('record_id')}")


def expected_lane_tif(inter_tif: Path) -> Path:
    prefix = inter_tif.stem.removesuffix("_inter")
    return inter_tif.parent.parent / "lane_patch_tif" / f"{prefix}_lane.tif"


def find_lane_gt(scene_root: Path, baseline_name: str) -> Path:
    direct = scene_root / baseline_name / "Lane.geojson"
    if direct.is_file():
        return direct
    suffix_matches = sorted(
        path
        for path in scene_root.rglob("Lane.geojson")
        if path.parent.name == baseline_name or path.parent.name.endswith(baseline_name)
    )
    if suffix_matches:
        return suffix_matches[0]
    fallback = scene_root / "rc_one_patch_release" / "center_line_v2" / "prefabricate_Lane.geojson"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Lane GT not found below scene {scene_root}")


def feature_lane_type(properties: dict[str, Any]) -> int | None:
    for key, value in properties.items():
        if str(key).replace("_", "").lower() == "lanetype":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


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


def geojson_crs(payload: dict[str, Any], fallback: str) -> str:
    crs = payload.get("crs")
    if isinstance(crs, dict):
        properties = crs.get("properties")
        if isinstance(properties, dict) and properties.get("name"):
            return str(properties["name"])
    return fallback


class SceneLaneIndex:
    def __init__(self, gt_path: Path, target_crs: str, source_crs: str, ignored_types: set[int]):
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        actual_source_crs = geojson_crs(payload, source_crs)
        self.lines: list[LineString] = []
        self.filtered_lane_types: dict[int, int] = defaultdict(int)
        for feature in payload.get("features", []):
            if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
                continue
            properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            lane_type = feature_lane_type(properties)
            if lane_type in ignored_types:
                self.filtered_lane_types[int(lane_type)] += 1
                continue
            transformed = transform_geom(
                actual_source_crs,
                target_crs,
                feature["geometry"],
                precision=-1,
            )
            self.lines.extend(iter_lines(shape(transformed)))
        self.tree = STRtree(self.lines) if self.lines else None

    def query(self, polygon: Polygon) -> list[LineString]:
        if self.tree is None:
            return []
        candidates = self.tree.query(polygon)
        if len(candidates) == 0:
            return []
        first = candidates[0]
        if isinstance(first, (numbers.Integral, np.integer)):
            return [self.lines[int(index)] for index in candidates]
        return list(candidates)


def raster_polygon(transform, left: float, top: float, right: float, bottom: float) -> Polygon:
    return Polygon(
        [
            transform * (left, top),
            transform * (right, top),
            transform * (right, bottom),
            transform * (left, bottom),
            transform * (left, top),
        ]
    )


def clean_local_points(line: LineString, inverse_transform, x0: int, y0: int, patch_size: int) -> list[list[float]]:
    maximum = float(patch_size - 1)
    points: list[list[float]] = []
    for map_x, map_y, *_ in line.coords:
        pixel_x, pixel_y = inverse_transform * (float(map_x), float(map_y))
        local_x = min(max(float(pixel_x) - x0, 0.0), maximum)
        local_y = min(max(float(pixel_y) - y0, 0.0), maximum)
        point = [round(local_x, 3), round(local_y, 3)]
        if not points or point != points[-1]:
            points.append(point)
    return points if len(points) >= 2 else []


def patch_ground_truth(
    lane_index: SceneLaneIndex,
    *,
    transform,
    raster_width: int,
    raster_height: int,
    row: int,
    col: int,
    patch_size: int,
) -> list[dict[str, Any]]:
    x0 = col * patch_size
    y0 = row * patch_size
    patch_polygon = raster_polygon(transform, x0, y0, x0 + patch_size, y0 + patch_size)
    raster_coverage = raster_polygon(transform, 0, 0, raster_width, raster_height)
    valid_polygon = patch_polygon.intersection(raster_coverage)
    if valid_polygon.is_empty:
        return []

    inverse_transform = ~transform
    output: list[dict[str, Any]] = []
    for source_line in lane_index.query(valid_polygon):
        try:
            clipped = source_line.intersection(valid_polygon)
        except Exception:
            continue
        for line in iter_lines(clipped):
            points = clean_local_points(line, inverse_transform, x0, y0, patch_size)
            if points:
                output.append({"category": "centerline", "points": points})
    return output


def pixel_prediction(
    record: dict[str, Any],
    patch_size: int,
    coord_range: int,
) -> tuple[str, bool, str]:
    declared_parse_ok = bool(record.get("parse_ok", True))
    declared_parse_error = str(record.get("parse_error") or "")
    pixel = record.get("prediction_json_pixel") or record.get("response_pixel")
    if pixel:
        return str(pixel), declared_parse_ok, declared_parse_error
    # A malformed model response is a valid evaluation outcome. Keep it in the
    # sample set as an empty prediction so it lowers recall and format validity;
    # do not misclassify it as a GT/raster pairing failure.
    if not declared_parse_ok:
        return "", False, declared_parse_error
    raw = record.get("prediction_json") or record.get("prediction") or ""
    if not raw:
        return "", declared_parse_ok, declared_parse_error
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    coord_mode = str(record.get("coord_mode") or meta.get("coord_mode") or COORD_MODE_NORM1000)
    if coord_mode == COORD_MODE_PIXEL:
        return str(raw), declared_parse_ok, declared_parse_error
    try:
        converted = convert_payload_text(
            str(raw),
            coord_mode,
            COORD_MODE_PIXEL,
            patch_size,
            patch_size,
            coord_range=int(record.get("coord_range") or meta.get("coord_range") or coord_range),
            clamp=True,
        )
        return converted, declared_parse_ok, declared_parse_error
    except Exception as exc:
        error = f"prediction coordinate conversion failed: {type(exc).__name__}: {exc}"
        return "", False, error


def raw_totals(details: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "gt_line_num",
        "pred_line_num",
        "matched_line_num",
        "gt_line_length_sum",
        "pred_line_length_sum",
        "matched_line_length_sum",
        "sample_num",
        "valid_string_format",
    )
    return {
        key: sum(sample.get(key, 0) for sample in details.get("samples", []))
        for key in keys
    }


def evaluate_group(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    details = evaluate_records(
        records,
        meter_per_pixel=args.meter_per_pixel,
        buffer_size=args.buffer_size,
        match_threshold=args.match_threshold,
        include_samples=True,
        categories="lane",
        eval_name="RC E2E Patch Lane Evaluation Results",
    )
    summary = details["summary"]
    summary["raw_totals"] = raw_totals(details)
    return summary


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_e2e_root).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    ignored_types = parse_int_set(args.ignore_lane_types)
    predictions, json_files_seen = load_prediction_records(prediction_dir)
    if args.max_samples > 0:
        predictions = predictions[: args.max_samples]
    if not predictions:
        raise FileNotFoundError(f"No raw per-patch prediction JSON found below {prediction_dir}")

    inter_index = build_inter_tif_index(raw_root)
    raster_cache: dict[Path, dict[str, Any]] = {}
    scene_lane_cache: dict[tuple[Path, str], SceneLaneIndex] = {}
    enriched: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    prediction_conversion_errors: list[dict[str, str]] = []
    scene_counts: dict[str, int] = defaultdict(int)

    import rasterio

    for index, record in enumerate(predictions, 1):
        record_id = str(record.get("record_id") or record.get("id") or f"sample_{index}")
        stage = "resolve_record"
        try:
            scene_id, tif_stem = record_scene_and_tif(record)
            stage = "resolve_inter_tif"
            inter_tif = inter_index[(scene_id, tif_stem)]
            lane_tif = expected_lane_tif(inter_tif)
            if not lane_tif.is_file():
                raise FileNotFoundError(f"Lane TIF not found: {lane_tif}")
            stage = "read_lane_tif"
            if lane_tif not in raster_cache:
                with rasterio.open(lane_tif) as source:
                    if source.crs is None:
                        raise ValueError(f"Lane TIF has no CRS: {lane_tif}")
                    raster_cache[lane_tif] = {
                        "transform": source.transform,
                        "crs": source.crs.to_string(),
                        "width": int(source.width),
                        "height": int(source.height),
                    }
            raster = raster_cache[lane_tif]
            scene_root = scene_root_for_inter_tif(inter_tif)
            stage = "resolve_lane_gt"
            gt_path = find_lane_gt(scene_root, args.baseline_name)
            lane_cache_key = (gt_path, raster["crs"])
            stage = "load_lane_gt"
            if lane_cache_key not in scene_lane_cache:
                scene_lane_cache[lane_cache_key] = SceneLaneIndex(
                    gt_path,
                    target_crs=raster["crs"],
                    source_crs=args.gt_crs,
                    ignored_types=ignored_types,
                )
            row, col = record_row_col(record)
            stage = "clip_patch_ground_truth"
            gt_items = patch_ground_truth(
                scene_lane_cache[lane_cache_key],
                transform=raster["transform"],
                raster_width=raster["width"],
                raster_height=raster["height"],
                row=row,
                col=col,
                patch_size=args.patch_size,
            )
            ground_truth_pixel = json.dumps({"lines": gt_items}, ensure_ascii=False, separators=(",", ":"))
            stage = "convert_prediction"
            prediction_pixel, eval_parse_ok, eval_parse_error = pixel_prediction(
                record,
                args.patch_size,
                args.coord_range,
            )
            if eval_parse_error and not eval_parse_ok:
                prediction_conversion_errors.append({
                    "record_id": record_id,
                    "prediction_file": str(record.get("_prediction_file") or ""),
                    "error": eval_parse_error,
                })
            output_record = {
                **record,
                "record_id": record_id,
                "scene_id": scene_id,
                "ground_truth_pixel": ground_truth_pixel,
                "prediction_json_pixel": prediction_pixel,
                "parse_ok": eval_parse_ok,
                "parse_error": eval_parse_error,
                "ground_truth_source": str(gt_path),
                "lane_tif": str(lane_tif),
            }
            enriched.append(output_record)
            scene_counts[scene_id] += 1
        except Exception as exc:
            errors.append({
                "record_id": record_id,
                "prediction_file": str(record.get("_prediction_file") or ""),
                "stage": stage,
                "error": repr(exc),
            })
        if index % 1000 == 0 or index == len(predictions):
            print(
                f"[e2e-patch-eval] processed={index}/{len(predictions)} "
                f"paired={len(enriched)} errors={len(errors)}",
                flush=True,
            )

    if errors:
        error_path = Path(args.output_json).with_name("pairing_errors.json")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[e2e-patch-eval] pairing_errors={error_path}", flush=True)
    if errors and args.require_all:
        preview = json.dumps(errors[:10], ensure_ascii=False, indent=2)
        raise RuntimeError(f"Unable to pair all prediction records; errors={len(errors)}\n{preview}")
    if not enriched:
        raise RuntimeError("No prediction records could be paired with E2E lane GT.")

    global_eval = evaluate_group(enriched, args)
    by_scene: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in enriched:
        grouped[record["scene_id"]].append(record)
    for scene_id, records in sorted(grouped.items()):
        by_scene[scene_id] = evaluate_group(records, args)

    filtered_counts: dict[str, int] = defaultdict(int)
    for lane_index in scene_lane_cache.values():
        for lane_type, count in lane_index.filtered_lane_types.items():
            filtered_counts[str(lane_type)] += count

    result = {
        "metric_scope": "patch-level centerline geometry on inferred E2E target ROIs",
        "metric_note": (
            "This reuses infer_index.line_eval for comparability with prior patch experiments. "
            "It is not the DI whole-map post-processing metric."
        ),
        "config": {
            "raw_e2e_root": str(raw_root),
            "prediction_dir": str(prediction_dir),
            "baseline_name": args.baseline_name,
            "gt_crs_fallback": args.gt_crs,
            "patch_size": args.patch_size,
            "meter_per_pixel": args.meter_per_pixel,
            "buffer_size": args.buffer_size,
            "match_threshold": args.match_threshold,
            "ignored_lane_types": sorted(ignored_types),
        },
        "coverage": {
            "json_files_seen": json_files_seen,
            "prediction_records": len(predictions),
            "evaluated_records": len(enriched),
            "pairing_errors": len(errors),
            "prediction_conversion_errors": len(prediction_conversion_errors),
            "scene_count": len(grouped),
            "records_by_scene": dict(sorted(scene_counts.items())),
            "filtered_gt_features_by_lane_type": dict(sorted(filtered_counts.items())),
            "error_examples": errors[:20],
            "prediction_conversion_error_examples": prediction_conversion_errors[:20],
        },
        "centerline_eval": global_eval,
        "per_scene": by_scene,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_eval_jsonl:
        eval_path = Path(args.output_eval_jsonl)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_path.open("w", encoding="utf-8") as handle:
            for record in enriched:
                record.pop("_prediction_file", None)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(format_eval_table(global_eval, title="RC E2E Patch Lane Evaluation Results"))
    print(f"[e2e-patch-eval] metrics={output_path}")
    if args.output_eval_jsonl:
        print(f"[e2e-patch-eval] eval_jsonl={args.output_eval_jsonl}")


if __name__ == "__main__":
    main()
