#!/usr/bin/env python3
"""Adapt full-local512 predictions to the original E2E engine's 256 grid."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from shapely.geometry import GeometryCollection, LineString, MultiLineString, box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--source-patch-size", type=int, default=512)
    parser.add_argument("--engine-patch-size", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def payload_lines(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("lines", [])
    if not isinstance(value, list):
        raise TypeError("prediction payload must contain a lines list")
    return [item for item in value if isinstance(item, dict)]


def finite_points(value: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(value, list):
        return points
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def iter_lines(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        if not geometry.is_empty and len(geometry.coords) >= 2:
            yield geometry
        return
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from iter_lines(part)


def compact_number(value: float) -> int | float:
    rounded = round(float(value), 6)
    if abs(rounded - round(rounded)) < 1e-6:
        return int(round(rounded))
    return rounded


def points_for_output(
    geometry: LineString,
    *,
    x0: float,
    y0: float,
    engine_patch_size: int,
    coord_range: int,
) -> tuple[list[list[int | float]], list[list[int | float]]]:
    pixel: list[list[int | float]] = []
    normalized: list[list[int | float]] = []
    for x, y in geometry.coords:
        local_x = min(max(float(x) - x0, 0.0), float(engine_patch_size))
        local_y = min(max(float(y) - y0, 0.0), float(engine_patch_size))
        pixel.append([compact_number(local_x), compact_number(local_y)])
        normalized.append(
            [
                compact_number(local_x / engine_patch_size * coord_range),
                compact_number(local_y / engine_patch_size * coord_range),
            ]
        )
    return normalized, pixel


def parse_source_lines(
    record: dict[str, Any],
    *,
    source_patch_size: int,
    coord_range: int,
) -> tuple[list[dict[str, Any]], str]:
    pixel_value = record.get("prediction_json_pixel") or record.get("response_pixel")
    if pixel_value:
        return payload_lines(pixel_value), "pixel"

    if record.get("parse_ok") is False:
        return [], "parse_failure"

    normalized_value = record.get("prediction_json") or record.get("prediction")
    if not normalized_value:
        return [], "empty"
    lines = payload_lines(normalized_value)
    converted: list[dict[str, Any]] = []
    scale = float(source_patch_size) / float(coord_range)
    for item in lines:
        clone = dict(item)
        clone["points"] = [
            [compact_number(x * scale), compact_number(y * scale)]
            for x, y in finite_points(item.get("points"))
        ]
        converted.append(clone)
    return converted, "norm"


def child_identity(
    record: dict[str, Any],
    *,
    child_row: int,
    child_col: int,
) -> tuple[str, str]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    image_text = str(record.get("image") or "").replace("\\", "/")
    image_path = PurePosixPath(image_text)
    scene_id = str(meta.get("scene_id") or "")
    if not scene_id and "images" in image_path.parts:
        image_index = image_path.parts.index("images")
        if image_index + 1 < len(image_path.parts):
            scene_id = image_path.parts[image_index + 1]
    tif_stem = str(meta.get("tif_stem") or image_path.parent.name)
    tif_prefix = str(meta.get("tif_prefix") or tif_stem.split("_", 1)[0])
    if not scene_id or not tif_stem:
        raise ValueError(f"cannot resolve scene/tif identity from {record.get('record_id')!r}")
    child_image = image_path.with_name(f"{child_row}_{child_col}.png").as_posix()
    return f"{scene_id}_{tif_prefix}_{child_row}_{child_col}", child_image


def adapt_record(
    record: dict[str, Any],
    *,
    source_patch_size: int,
    engine_patch_size: int,
    coord_range: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if source_patch_size % engine_patch_size:
        raise ValueError("source patch size must be divisible by engine patch size")
    factor = source_patch_size // engine_patch_size
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    source_row = int(record.get("row", meta.get("row", meta.get("patch_row"))))
    source_col = int(record.get("col", meta.get("col", meta.get("patch_col"))))
    source_lines, source_space = parse_source_lines(
        record,
        source_patch_size=source_patch_size,
        coord_range=coord_range,
    )
    stats: Counter[str] = Counter({f"source_space:{source_space}": 1})
    geometries: list[tuple[dict[str, Any], LineString]] = []
    for item in source_lines:
        if str(item.get("category", "centerline")).strip().lower() != "centerline":
            stats["non_centerline_items_omitted"] += 1
            continue
        points = finite_points(item.get("points"))
        if len(points) < 2:
            stats["invalid_centerline_items_omitted"] += 1
            continue
        try:
            geometry = LineString(points)
        except Exception:
            stats["invalid_centerline_items_omitted"] += 1
            continue
        if geometry.is_empty or geometry.length <= 0:
            stats["invalid_centerline_items_omitted"] += 1
            continue
        geometries.append((item, geometry))
    stats["source_centerline_items"] += len(geometries)

    children: list[dict[str, Any]] = []
    for row_offset in range(factor):
        for col_offset in range(factor):
            child_row = source_row * factor + row_offset
            child_col = source_col * factor + col_offset
            x0 = col_offset * engine_patch_size
            y0 = row_offset * engine_patch_size
            roi = box(x0, y0, x0 + engine_patch_size, y0 + engine_patch_size)
            normalized_items: list[dict[str, Any]] = []
            pixel_items: list[dict[str, Any]] = []
            for item, geometry in geometries:
                try:
                    clipped = geometry.intersection(roi)
                except Exception:
                    stats["intersection_failures"] += 1
                    continue
                for segment in iter_lines(clipped):
                    normalized_points, pixel_points = points_for_output(
                        segment,
                        x0=x0,
                        y0=y0,
                        engine_patch_size=engine_patch_size,
                        coord_range=coord_range,
                    )
                    normalized_item = dict(item)
                    normalized_item["points"] = normalized_points
                    pixel_item = dict(item)
                    pixel_item["points"] = pixel_points
                    normalized_items.append(normalized_item)
                    pixel_items.append(pixel_item)
                    stats["output_centerline_segments"] += 1

            child_id, child_image = child_identity(record, child_row=child_row, child_col=child_col)
            child_meta = dict(meta)
            child_meta.update(
                {
                    "row": child_row,
                    "col": child_col,
                    "patch_row": child_row,
                    "patch_col": child_col,
                    "x0": child_col * engine_patch_size,
                    "y0": child_row * engine_patch_size,
                    "patch_size": engine_patch_size,
                    "pixel_patch_size": engine_patch_size,
                    "patch_width": engine_patch_size,
                    "patch_height": engine_patch_size,
                    "context_size": engine_patch_size,
                    "input_image_size": engine_patch_size,
                    "target_roi_in_image": [0, 0, engine_patch_size, engine_patch_size],
                    "view_mode": "original_e2e_256_adapter",
                    "coord_mode": "norm1000",
                    "coord_system": f"patch_norm{coord_range}",
                    "coord_range": coord_range,
                    "adapted_from_local512_record_id": str(record.get("record_id") or record.get("id") or ""),
                }
            )
            child = dict(record)
            child.update(
                {
                    "record_id": child_id,
                    "id": child_id,
                    "image": child_image,
                    "row": child_row,
                    "col": child_col,
                    "x0": child_col * engine_patch_size,
                    "y0": child_row * engine_patch_size,
                    "meta": child_meta,
                    "coord_mode": "norm1000",
                    "coord_range": coord_range,
                    "patch_size": engine_patch_size,
                    "patch_width": engine_patch_size,
                    "patch_height": engine_patch_size,
                    "prediction_json": json.dumps(
                        {"lines": normalized_items}, ensure_ascii=False, separators=(",", ":")
                    ),
                    "prediction_json_pixel": json.dumps(
                        {"lines": pixel_items}, ensure_ascii=False, separators=(",", ":")
                    ),
                    "parse_ok": bool(record.get("parse_ok", True)),
                    "num_items": len(normalized_items),
                    "lines_local": pixel_items,
                    "lines_local_model": normalized_items,
                    "lines_global": [],
                    "original_e2e_grid_adapter": {
                        "source_patch_size": source_patch_size,
                        "engine_patch_size": engine_patch_size,
                        "source_row": source_row,
                        "source_col": source_col,
                        "child_row_offset": row_offset,
                        "child_col_offset": col_offset,
                    },
                }
            )
            children.append(child)
    stats["output_records"] += len(children)
    return children, stats


def adapt_directory(
    input_dir: Path,
    output_dir: Path,
    report_json: Path,
    *,
    source_patch_size: int,
    engine_patch_size: int,
    coord_range: int,
    reset: bool,
    strict: bool,
) -> dict[str, Any]:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("output directory must differ from input directory")
    if reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.json"):
        stale.unlink()

    files = sorted(input_dir.rglob("*.json"))
    errors: list[dict[str, str]] = []
    duplicate_child_ids: list[str] = []
    seen_child_ids: set[str] = set()
    totals: Counter[str] = Counter()
    source_records = 0
    output_index = 0
    for source_index, path in enumerate(files):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise TypeError("outer payload is not an object")
            if not any(key in record for key in ("prediction_json", "prediction_json_pixel", "prediction")):
                continue
            source_records += 1
            children, stats = adapt_record(
                record,
                source_patch_size=source_patch_size,
                engine_patch_size=engine_patch_size,
                coord_range=coord_range,
            )
            totals.update(stats)
            for child_offset, child in enumerate(children):
                child_id = str(child["record_id"])
                if child_id in seen_child_ids:
                    duplicate_child_ids.append(child_id)
                seen_child_ids.add(child_id)
                child["idx"] = output_index
                destination = output_dir / f"{source_index:06d}_{child_offset}_{child_id}.json"
                destination.write_text(json.dumps(child, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                output_index += 1
        except Exception as exc:
            errors.append({"file": str(path), "error": repr(exc)})

    factor = source_patch_size // engine_patch_size if engine_patch_size else 0
    report = {
        "protocol": (
            "Full local512 predictions are clipped into non-overlapping 256x256 child cells, shifted to each "
            "child's local frame, and normalized back to 0..1000 before the untouched original E2E engine."
        ),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_patch_size": source_patch_size,
        "engine_patch_size": engine_patch_size,
        "coord_range": coord_range,
        "source_json_files_seen": len(files),
        "source_prediction_records": source_records,
        "expected_children_per_source": factor * factor,
        "output_prediction_records": output_index,
        "stats": dict(sorted(totals.items())),
        "duplicate_child_ids": sorted(set(duplicate_child_ids)),
        "errors": errors,
    }
    report["complete"] = (
        source_records > 0
        and output_index == source_records * factor * factor
        and not duplicate_child_ids
        and not errors
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if strict and not report["complete"]:
        raise SystemExit(1)
    return report


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_json = args.report_json.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input prediction directory not found: {input_dir}")
    if args.source_patch_size <= 0 or args.engine_patch_size <= 0 or args.coord_range <= 0:
        raise ValueError("patch sizes and coordinate range must be positive")
    adapt_directory(
        input_dir,
        output_dir,
        report_json,
        source_patch_size=args.source_patch_size,
        engine_patch_size=args.engine_patch_size,
        coord_range=args.coord_range,
        reset=args.reset,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
