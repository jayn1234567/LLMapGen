#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from state_update_dataset_common import (
    clip_intersections_to_patch,
    clip_lanes_to_patch,
    load_intersection_geometries,
    load_line_geometries,
    pad_image_to_patch_grid,
    read_masked_image,
    required_paths,
    sort_target_lines,
)


def geom_summary(gdf, label, max_rows=5):
    print(f"{label}_crs: {gdf.crs}")
    print(f"{label}_row_count: {len(gdf)}")
    if len(gdf) == 0:
        return
    counts = gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
    print(f"{label}_geom_type_counts: {counts}")
    try:
        print(f"{label}_total_bounds: {list(map(float, gdf.total_bounds))}")
    except Exception as exc:
        print(f"{label}_total_bounds_error: {exc}")
    for idx, geom in enumerate(gdf.geometry.head(max_rows)):
        if geom is None:
            print(f"{label}_row_{idx}: geom=None")
            continue
        print(
            f"{label}_row_{idx}: "
            f"type={geom.geom_type} "
            f"is_empty={geom.is_empty} "
            f"has_z={getattr(geom, 'has_z', False)} "
            f"bounds={list(map(float, geom.bounds)) if not geom.is_empty else []}"
        )


def raw_geojson_summary(path: Path, label, max_features=5):
    print(f"{label}_file_size_bytes: {path.stat().st_size if path.exists() else 'missing'}")
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{label}_json_read_error: {exc}")
        return

    if isinstance(payload, dict):
        print(f"{label}_json_type: {payload.get('type')}")
        features = payload.get("features")
        if isinstance(features, list):
            print(f"{label}_json_feature_count: {len(features)}")
            geom_counts = {}
            for feature in features:
                geom = feature.get("geometry") if isinstance(feature, dict) else None
                geom_type = geom.get("type") if isinstance(geom, dict) else None
                geom_counts[geom_type] = geom_counts.get(geom_type, 0) + 1
            print(f"{label}_json_geometry_type_counts: {geom_counts}")
            for idx, feature in enumerate(features[:max_features]):
                geom = feature.get("geometry") if isinstance(feature, dict) else None
                props = feature.get("properties") if isinstance(feature, dict) else None
                print(
                    f"{label}_json_feature_{idx}: "
                    f"geometry_type={geom.get('type') if isinstance(geom, dict) else None} "
                    f"property_keys={list(props.keys()) if isinstance(props, dict) else None}"
                )
        else:
            print(f"{label}_json_features_field_type: {type(features).__name__}")
    else:
        print(f"{label}_json_root_type: {type(payload).__name__}")


def list_geojson_candidates(sample_root: Path):
    label_dir = sample_root / "label_check_crop"
    print("label_geojson_candidates:")
    if not label_dir.exists():
        print(f"  label_dir_missing: {label_dir}")
        return
    for path in sorted(label_dir.glob("*.geojson")):
        feature_count = "unreadable"
        geometry_counts = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features") if isinstance(payload, dict) else None
            if isinstance(features, list):
                feature_count = len(features)
                for feature in features:
                    geom = feature.get("geometry") if isinstance(feature, dict) else None
                    geom_type = geom.get("type") if isinstance(geom, dict) else None
                    geometry_counts[geom_type] = geometry_counts.get(geom_type, 0) + 1
        except Exception as exc:
            feature_count = f"error:{exc}"
        print(
            f"  {path.name}: size={path.stat().st_size} "
            f"features={feature_count} geometry_types={geometry_counts}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Debug whether one raw sample can load and clip Intersection.geojson into patch-local targets."
    )
    parser.add_argument("sample_root", help="Path to one extracted raw sample folder.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--simplify-tolerance", type=float, default=0.5)
    parser.add_argument("--boundary-tol", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    sample = required_paths(sample_root)

    print(f"sample_root: {sample_root}")
    print(f"lane_geojson: {sample.lane_geojson} exists={sample.lane_geojson.exists()}")
    print(f"intersection_geojson: {sample.intersection_geojson} exists={sample.intersection_geojson.exists()}")
    print(f"image_tiff: {sample.image_tiff} exists={sample.image_tiff.exists()}")
    print(f"mask_tiff: {sample.mask_tiff} exists={sample.mask_tiff.exists()}")
    list_geojson_candidates(sample_root)

    image_arr, meta, transform, crs = read_masked_image(sample.image_tiff, sample.mask_tiff)
    image_arr, original_size = pad_image_to_patch_grid(image_arr, args.patch_size)

    print(f"crs: {crs}")
    print(f"transform: {transform}")
    print(f"original_size: {original_size}")
    print(f"padded_size: {[image_arr.shape[2], image_arr.shape[1]]}")

    if sample.intersection_geojson.exists():
        raw_geojson_summary(sample.intersection_geojson, "raw_intersection")
        raw_intersection_gdf = gpd.read_file(sample.intersection_geojson)
        geom_summary(raw_intersection_gdf, "raw_intersection")
        projected_intersection_gdf = raw_intersection_gdf.to_crs(crs)
        geom_summary(projected_intersection_gdf, "projected_intersection")
        if args.simplify_tolerance > 0:
            simplified_intersection_gdf = projected_intersection_gdf.copy()
            simplified_intersection_gdf["geometry"] = simplified_intersection_gdf.geometry.apply(
                lambda geom: geom.simplify(args.simplify_tolerance, preserve_topology=True)
                if geom is not None else geom
            )
            geom_summary(simplified_intersection_gdf, "simplified_intersection")

    lines = load_line_geometries(sample.lane_geojson, crs, transform, args.simplify_tolerance)
    intersections = load_intersection_geometries(
        sample.intersection_geojson,
        crs,
        transform,
        args.simplify_tolerance,
    )

    print(f"loaded_lane_count: {len(lines)}")
    print(f"loaded_intersection_polygon_count: {len(intersections)}")

    height, width = image_arr.shape[1], image_arr.shape[2]
    patch_count = 0
    nonblack_patch_count = 0
    lane_patch_count = 0
    intersection_patch_count = 0
    both_patch_count = 0
    examples = []

    for y0 in range(0, height - args.patch_size + 1, args.stride):
        for x0 in range(0, width - args.patch_size + 1, args.stride):
            patch_count += 1
            chunk = image_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
            if np.all(chunk == 0):
                continue
            nonblack_patch_count += 1

            lane_lines = clip_lanes_to_patch(lines, transform, x0, y0, args.patch_size)
            inter_lines = clip_intersections_to_patch(
                intersections,
                x0,
                y0,
                args.patch_size,
                transform=transform,
            )
            local_lines = sort_target_lines(
                lane_lines + inter_lines,
                args.patch_size,
                args.boundary_tol,
            )

            if lane_lines:
                lane_patch_count += 1
            if inter_lines:
                intersection_patch_count += 1
            if lane_lines and inter_lines:
                both_patch_count += 1

            if inter_lines and len(examples) < args.max_examples:
                examples.append({
                    "row": y0 // args.stride,
                    "col": x0 // args.stride,
                    "x0": x0,
                    "y0": y0,
                    "num_lane": len(lane_lines),
                    "num_intersection": len(inter_lines),
                    "num_total_target_lines": len(local_lines),
                    "intersections": [
                        {
                            "is_cut": item.get("is_cut"),
                            "points": item.get("points"),
                            "source_properties": item.get("_source_properties", {}),
                        }
                        for item in inter_lines[:3]
                    ],
                })

    print(f"patch_count: {patch_count}")
    print(f"nonblack_patch_count: {nonblack_patch_count}")
    print(f"lane_patch_count: {lane_patch_count}")
    print(f"intersection_patch_count: {intersection_patch_count}")
    print(f"both_patch_count: {both_patch_count}")
    print("intersection_examples:")
    print(json.dumps(examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
