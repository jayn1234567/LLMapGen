#!/usr/bin/env python3
"""Suppress patch intersection predictions where original E2E GT has no intersection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import rasterio
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--result-subdir", default="inter512/tif_512_256")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--gt-intersection-type",
        type=int,
        default=1,
        help="Only this original-engine IntersectionType makes a patch GT-positive.",
    )
    parser.add_argument("--minimum-overlap-area", type=float, default=1e-6)
    parser.add_argument("--expected-scenes", type=int, default=0)
    return parser.parse_args()


def validate_relative_subdir(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"result subdirectory must stay below center_line_v2: {value}")
    return path


def polygon_parts(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 0:
            yield geometry
        return
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def gt_intersection_path(scene: Path) -> Path:
    candidates = [scene / "gt" / "Intersection.geojson"]
    candidates.extend(sorted(scene.glob("*gt/Intersection.geojson")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Intersection GT not found below scene {scene}")


def load_gt_union(path: Path, target_crs: Any, intersection_type: int) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    geometries = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        if int(properties.get("IntersectionType", -1)) != intersection_type:
            continue
        geometry_payload = feature.get("geometry")
        if not isinstance(geometry_payload, dict):
            continue
        try:
            geometry = shape(geometry_payload)
            geometry = transform_geometry(transformer.transform, geometry)
        except (TypeError, ValueError):
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        geometries.extend(polygon_parts(geometry))
    return unary_union(geometries) if geometries else GeometryCollection()


def patch_polygon(transform: Any, row: int, col: int, window_size: int, stride: int) -> Polygon:
    x0 = col * stride
    y0 = row * stride
    x1 = x0 + window_size
    y1 = y0 + window_size
    return Polygon(
        [
            transform * (x0, y0),
            transform * (x1, y0),
            transform * (x1, y1),
            transform * (x0, y1),
            transform * (x0, y0),
        ]
    )


def suppress_scene(
    scene: Path,
    *,
    result_subdir: Path,
    window_size: int,
    stride: int,
    gt_intersection_type: int,
    minimum_overlap_area: float,
) -> dict[str, Any]:
    centerline_root = scene / "rc_one_patch_release" / "center_line_v2"
    result_root = centerline_root / result_subdir
    tif_root = centerline_root / "inter_patch_tif"
    counters: Counter[str] = Counter()
    gt_path = gt_intersection_path(scene)

    for tif_result_dir in sorted(path for path in result_root.glob("*_tif_res") if path.is_dir()):
        tif_prefix = tif_result_dir.name.removesuffix("_tif_res")
        tif_path = tif_root / f"{tif_prefix}_inter.tif"
        if not tif_path.is_file():
            raise FileNotFoundError(f"Source intersection TIF not found: {tif_path}")
        with rasterio.open(tif_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Source intersection TIF has no CRS: {tif_path}")
            raster_transform = dataset.transform
            target_crs = dataset.crs
        gt_union = load_gt_union(gt_path, target_crs, gt_intersection_type)

        for patch_path in sorted(tif_result_dir.glob("*.json")):
            pieces = patch_path.stem.split("_")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise ValueError(f"Invalid patch result filename: {patch_path}")
            row, col = int(pieces[0]), int(pieces[1])
            footprint = patch_polygon(raster_transform, row, col, window_size, stride)
            has_gt = (
                not gt_union.is_empty
                and footprint.intersection(gt_union).area > minimum_overlap_area
            )
            payload = json.loads(patch_path.read_text(encoding="utf-8-sig"))
            predictions = payload.get("intersection", [])
            if not isinstance(predictions, list):
                raise TypeError(f"Invalid intersection list: {patch_path}")
            counters["patches"] += 1
            counters["predicted_intersections_before"] += len(predictions)
            if has_gt:
                counters["gt_positive_patches"] += 1
                counters["predicted_intersections_after"] += len(predictions)
                continue
            counters["gt_empty_patches"] += 1
            if predictions:
                counters["suppressed_patches"] += 1
                counters["suppressed_intersections"] += len(predictions)
                payload["intersection"] = []
                patch_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    return {
        "scene_id": scene.name,
        "gt_path": str(gt_path),
        "counts": dict(sorted(counters.items())),
    }


def suppress_all(
    e2e_root: Path,
    report_json: Path,
    *,
    result_subdir: Path,
    window_size: int,
    stride: int,
    gt_intersection_type: int,
    minimum_overlap_area: float,
    expected_scenes: int,
) -> dict[str, Any]:
    e2e_root = e2e_root.resolve()
    if not e2e_root.is_dir():
        raise FileNotFoundError(e2e_root)
    scenes = sorted(
        scene for scene in e2e_root.iterdir()
        if scene.is_dir() and (scene / "rc_one_patch_release").is_dir()
    )
    if expected_scenes > 0 and len(scenes) != expected_scenes:
        raise RuntimeError(f"Expected {expected_scenes} scenes, found {len(scenes)}")

    scene_reports = []
    totals: Counter[str] = Counter()
    for index, scene in enumerate(scenes, start=1):
        scene_report = suppress_scene(
            scene,
            result_subdir=result_subdir,
            window_size=window_size,
            stride=stride,
            gt_intersection_type=gt_intersection_type,
            minimum_overlap_area=minimum_overlap_area,
        )
        scene_reports.append(scene_report)
        totals.update(scene_report["counts"])
        print(
            f"[intersection-gt-oracle] scene={index}/{len(scenes)} id={scene.name} "
            f"counts={scene_report['counts']}",
            flush=True,
        )

    report = {
        "policy": (
            "Diagnostic GT oracle: patch intersection predictions are suppressed when the patch "
            "footprint has no positive-area overlap with original E2E IntersectionType=1 GT."
        ),
        "warning": "This uses ground truth and must not be reported as production model performance.",
        "e2e_root": str(e2e_root),
        "result_subdir": result_subdir.as_posix(),
        "window_size": window_size,
        "stride": stride,
        "gt_intersection_type": gt_intersection_type,
        "minimum_overlap_area": minimum_overlap_area,
        "scene_count": len(scenes),
        "counts": dict(sorted(totals.items())),
        "scenes": scene_reports,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}, indent=2))
    if totals["patches"] <= 0:
        raise RuntimeError(f"No patch results found below {e2e_root / result_subdir}")
    return report


def main() -> None:
    args = parse_args()
    suppress_all(
        args.e2e_root,
        args.report_json,
        result_subdir=validate_relative_subdir(args.result_subdir),
        window_size=args.window_size,
        stride=args.stride,
        gt_intersection_type=args.gt_intersection_type,
        minimum_overlap_area=args.minimum_overlap_area,
        expected_scenes=args.expected_scenes,
    )


if __name__ == "__main__":
    main()
