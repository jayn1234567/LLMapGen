#!/usr/bin/env python3
"""Merge RC inter512 patch polygons into per-scene Intersection.geojson files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import rasterio
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--merge-buffer-meters", type=float, default=0.5)
    parser.add_argument("--result-subdir", default="inter512/tif_512_256")
    parser.add_argument("--query-name", default="output_llm_intersection_jn")
    parser.add_argument("--expected-scenes", type=int, default=0)
    parser.add_argument("--reset-query", action="store_true")
    return parser.parse_args()


def feature_collection(features: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
        },
        "features": features or [],
    }


def validate_relative_subdir(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"result subdirectory must stay below center_line_v2: {path}")
    return path


def validate_query_name(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError(f"query name must be one directory name: {value!r}")
    return value


def polygon_parts(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 1e-6:
            yield geometry
        return
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def valid_polygon(coords: Any) -> Polygon | MultiPolygon | None:
    if not isinstance(coords, list) or len(coords) < 3:
        return None
    try:
        polygon = Polygon(coords)
    except (TypeError, ValueError):
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not isinstance(polygon, (Polygon, MultiPolygon)) or polygon.is_empty or polygon.area <= 1e-6:
        return None
    return polygon


def parse_label(label: Any) -> tuple[int, int]:
    pieces = str(label).split("_")
    if len(pieces) != 2:
        return 0, 0
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError:
        return 0, 0


def tif_result_dirs(result_root: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        prefix = path.name.removesuffix("_tif_res")
        return (int(prefix), path.name) if prefix.isdigit() else (10**9, path.name)

    return sorted((path for path in result_root.glob("*_tif_res") if path.is_dir()), key=key)


def world_polygons_for_scene(
    scene: Path,
    *,
    result_subdir: Path,
    stride: int,
    counters: Counter[str],
) -> tuple[dict[str, list[Polygon]], Any | None]:
    centerline_root = scene / "rc_one_patch_release" / "center_line_v2"
    inter_tif_root = centerline_root / "inter_patch_tif"
    result_root = centerline_root / result_subdir
    grouped: dict[str, list[Polygon]] = {}
    scene_crs = None

    for tif_result_dir in tif_result_dirs(result_root):
        tif_prefix = tif_result_dir.name.removesuffix("_tif_res")
        tif_path = inter_tif_root / f"{tif_prefix}_inter.tif"
        if not tif_path.is_file():
            counters["missing_source_tifs"] += 1
            continue
        with rasterio.open(tif_path) as dataset:
            transform = dataset.transform
            crs = dataset.crs
        if crs is None:
            raise ValueError(f"Source TIF has no CRS: {tif_path}")
        if scene_crs is None:
            scene_crs = crs
        elif scene_crs != crs:
            raise ValueError(f"Scene {scene.name} contains inconsistent TIF CRS values")

        for patch_path in sorted(tif_result_dir.glob("*.json")):
            pieces = patch_path.stem.split("_")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                counters["invalid_patch_names"] += 1
                continue
            row, col = int(pieces[0]), int(pieces[1])
            payload = json.loads(patch_path.read_text(encoding="utf-8-sig"))
            for item in payload.get("intersection", []):
                counters["patch_intersections"] += 1
                local_geometry = valid_polygon(item.get("coords"))
                if local_geometry is None:
                    counters["invalid_patch_polygons"] += 1
                    continue
                label = str(item.get("label") or "0_0")
                for local_polygon in polygon_parts(local_geometry):
                    global_pixels = [
                        (x + col * stride, y + row * stride)
                        for x, y in local_polygon.exterior.coords
                    ]
                    world_coords = [transform * point for point in global_pixels]
                    world_geometry = valid_polygon(world_coords)
                    if world_geometry is None:
                        counters["invalid_world_polygons"] += 1
                        continue
                    for world_polygon in polygon_parts(world_geometry):
                        grouped.setdefault(label, []).append(world_polygon)
                        counters[f"source_label:{label}"] += 1
    return grouped, scene_crs


def merge_group(polygons: list[Polygon], buffer_meters: float) -> list[Polygon]:
    if not polygons:
        return []
    if buffer_meters > 0:
        geometry = unary_union([polygon.buffer(buffer_meters) for polygon in polygons]).buffer(-buffer_meters)
    else:
        geometry = unary_union(polygons)
    return list(polygon_parts(geometry))


def lla_feature(polygon: Polygon, label: str, feature_id: int, transformer: Transformer) -> dict[str, Any]:
    coordinates = []
    for x, y in polygon.exterior.coords:
        lon, lat = transformer.transform(x, y)
        coordinates.append([lon, lat, 0.0])
    intersection_type, intersection_subtype = parse_label(label)
    return {
        "type": "Feature",
        "mode": "None",
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": {
            "IntersectionType": intersection_type,
            "IntersectionSubType": intersection_subtype,
            "IsRegular": False,
            "NonstandardIntersectionType": 0,
            "DataTag": 0,
            "Visibility": 1,
            "Id": str(feature_id),
            "Source": 0,
            "SourceType": 0,
            "RefLane": -1,
        },
    }


def empty_lane_payload(scene: Path) -> dict[str, Any]:
    candidates = sorted(scene.glob("*gt/Lane.geojson")) + sorted(scene.glob("gt/Lane.geojson"))
    if candidates:
        try:
            payload = json.loads(candidates[0].read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["features"] = []
                return payload
        except (OSError, ValueError):
            pass
    return feature_collection()


def build_scene(
    scene: Path,
    *,
    result_subdir: Path,
    query_name: str,
    stride: int,
    merge_buffer_meters: float,
    reset_query: bool,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    grouped, crs = world_polygons_for_scene(
        scene,
        result_subdir=result_subdir,
        stride=stride,
        counters=counters,
    )
    features: list[dict[str, Any]] = []
    if grouped and crs is None:
        raise ValueError(f"Unable to resolve CRS for scene {scene.name}")
    if crs is not None:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        for label in sorted(grouped):
            merged = merge_group(grouped[label], merge_buffer_meters)
            counters[f"merged_label:{label}"] += len(merged)
            for polygon in merged:
                features.append(lla_feature(polygon, label, len(features), transformer))

    query_dir = scene / query_name
    if reset_query and query_dir.is_dir():
        for name in ("Intersection.geojson", "Lane.geojson"):
            path = query_dir / name
            if path.is_file():
                path.unlink()
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / "Intersection.geojson").write_text(
        json.dumps(feature_collection(features), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (query_dir / "Lane.geojson").write_text(
        json.dumps(empty_lane_payload(scene), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "scene_id": scene.name,
        "intersection_features": len(features),
        "query_dir": str(query_dir),
        "counts": dict(sorted(counters.items())),
    }


def build_all(
    e2e_root: Path,
    report_json: Path,
    *,
    result_subdir: Path,
    query_name: str,
    stride: int,
    merge_buffer_meters: float,
    expected_scenes: int,
    reset_query: bool,
) -> dict[str, Any]:
    e2e_root = e2e_root.resolve()
    result_subdir = validate_relative_subdir(result_subdir)
    query_name = validate_query_name(query_name)
    if not e2e_root.is_dir():
        raise FileNotFoundError(e2e_root)
    scenes = sorted(
        scene for scene in e2e_root.iterdir() if scene.is_dir() and (scene / "rc_one_patch_release").is_dir()
    )
    if expected_scenes > 0 and len(scenes) != expected_scenes:
        raise RuntimeError(f"Expected {expected_scenes} scenes, found {len(scenes)} below {e2e_root}")

    scene_reports = []
    for index, scene in enumerate(scenes, 1):
        report = build_scene(
            scene,
            result_subdir=result_subdir,
            query_name=query_name,
            stride=stride,
            merge_buffer_meters=merge_buffer_meters,
            reset_query=reset_query,
        )
        scene_reports.append(report)
        print(
            f"[intersection-geojson] scene={index}/{len(scenes)} id={scene.name} "
            f"features={report['intersection_features']}",
            flush=True,
        )

    report = {
        "e2e_root": str(e2e_root),
        "result_subdir": result_subdir.as_posix(),
        "query_name": query_name,
        "stride": stride,
        "merge_buffer_meters": merge_buffer_meters,
        "scene_count": len(scenes),
        "intersection_features": sum(item["intersection_features"] for item in scene_reports),
        "scenes": scene_reports,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "scenes"}, ensure_ascii=False, indent=2))
    if report["intersection_features"] <= 0:
        raise RuntimeError(
            f"No Intersection.geojson features were generated; inspect {report_json}"
        )
    return report


def main() -> None:
    args = parse_args()
    build_all(
        args.e2e_root,
        args.report_json,
        result_subdir=Path(args.result_subdir),
        query_name=args.query_name,
        stride=args.stride,
        merge_buffer_meters=args.merge_buffer_meters,
        expected_scenes=args.expected_scenes,
        reset_query=args.reset_query,
    )


if __name__ == "__main__":
    main()
