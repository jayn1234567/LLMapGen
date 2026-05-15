#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

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

    image_arr, meta, transform, crs = read_masked_image(sample.image_tiff, sample.mask_tiff)
    image_arr, original_size = pad_image_to_patch_grid(image_arr, args.patch_size)

    print(f"crs: {crs}")
    print(f"transform: {transform}")
    print(f"original_size: {original_size}")
    print(f"padded_size: {[image_arr.shape[2], image_arr.shape[1]]}")

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
