#!/usr/bin/env python3
"""Build RC end-to-end inference JSONL from ``*_inter.tif`` images.

The downstream RC parser identifies every target crop by ``row_col.json`` and
always expects coordinates in a 256x256 local target frame. This tool can save
either that local crop or a 512x512 context image centered on the same target
crop without changing the downstream row/column identity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VIEW_LOCAL256 = "local256"
VIEW_CONTEXT512_ROI256 = "context512_roi256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Extracted E2E root containing *_inter.tif files.")
    parser.add_argument("--output-root", required=True, help="Output root for images, infer.jsonl, and manifest.")
    parser.add_argument(
        "--view-mode",
        choices=(VIEW_LOCAL256, VIEW_CONTEXT512_ROI256),
        default=VIEW_CONTEXT512_ROI256,
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--black-ratio-threshold", type=float, default=0.98)
    parser.add_argument(
        "--include-intersections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the lane+intersection Dataset V2 prompt expected by current checkpoints.",
    )
    parser.add_argument("--max-tifs", type=int, default=0, help="0 processes every discovered TIF.")
    parser.add_argument("--max-patches", type=int, default=0, help="0 keeps every non-black target patch.")
    return parser.parse_args()


def scene_id_for_tif(tif_path: Path) -> str:
    for parent in tif_path.parents:
        if parent.name == "rc_one_patch_release":
            return parent.parent.name
    raise ValueError(f"Unable to find scene ID above rc_one_patch_release: {tif_path}")


def discover_inter_tifs(input_root: Path) -> list[Path]:
    paths = [
        path
        for path in input_root.rglob("*_inter.tif")
        if path.parent.name == "inter_patch_tif"
    ]
    return sorted(paths, key=lambda path: (scene_id_for_tif(path), path.name, str(path)))


def _rasterio_image(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as source:
        channels_first = source.read()
    return np.transpose(channels_first, (1, 2, 0))


def read_tif_rgb(path: Path) -> np.ndarray:
    try:
        array = _rasterio_image(path)
    except (ImportError, ModuleNotFoundError):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Unsupported TIF shape {array.shape}: {path}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 2:
        array = np.concatenate((array, array[:, :, :1]), axis=2)
    elif array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def black_ratio(array: np.ndarray) -> float:
    return float(np.mean(array == 0))


def dataset_v2_prompt(
    *,
    view_mode: str,
    target_size: int,
    context_size: int,
    coord_range: int,
    include_intersections: bool,
) -> str:
    parts = [
        "<image>",
        "Please construct the complete road map in the current BEV (Bird's Eye View) image patch.",
    ]
    if view_mode == VIEW_CONTEXT512_ROI256:
        margin = (context_size - target_size) // 2
        roi = [margin, margin, margin + target_size, margin + target_size]
        parts.extend(
            [
                f"The input is a {context_size}x{context_size} context image centered on the target region.",
                (
                    f"Predict only map elements clipped to the central {target_size}x{target_size} "
                    f"target ROI [{roi[0]},{roi[1]},{roi[2]},{roi[3]})."
                ),
                f"Coordinates use a normalized 0-{coord_range} grid over the {target_size}x{target_size} target ROI.",
                "Coordinates are relative to the target ROI, not the full context image.",
                "Do not output geometry that lies only outside the target ROI.",
            ]
        )
    else:
        parts.append(
            f"Coordinates use a normalized 0-{coord_range} grid over the original "
            f"{target_size}x{target_size} image patch."
        )

    parts.extend(
        [
            "",
            'Return only valid JSON in the form {"lines":[...]} with no extra explanation.',
        (
            'For every centerline, include "lane_type" with exactly one of: '
            '"common" for a regular centerline, "right_turn" for a right-turn-only '
            'centerline, "waiting_area" for a waiting-area centerline, "bus_lane" '
            'for a bus-lane centerline, "main_auxiliary_connector" for a connector '
            'between main and auxiliary roads, or "other" for any remaining lane class.'
        ),
        ]
    )
    if include_intersections:
        parts.append(
            'For every intersection, include "intersection_type" with exactly one of: '
            '"common" for a common intersection, "t_intersection" for a T-intersection, '
            '"small_untyped" for a small untyped intersection, '
            '"t_lane_change_area" for a T-shaped lane-change area, or "other" for any remaining class.'
        )
    parts.extend(["", "Incoming traces JSON:", "[]"])
    if include_intersections:
        parts.extend(["", "Incoming intersections JSON:", "[]"])
    return "\n".join(parts)


def pad_bottom_right(image: np.ndarray, target_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    padded_height = int(math.ceil(height / target_size) * target_size)
    padded_width = int(math.ceil(width / target_size) * target_size)
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded[:height, :width] = image
    return padded


def build_record(
    *,
    scene_id: str,
    tif_path: Path,
    relative_image: Path,
    row: int,
    col: int,
    target_size: int,
    context_size: int,
    view_mode: str,
    coord_range: int,
    source_size: tuple[int, int],
    padded_size: tuple[int, int],
    prompt: str,
) -> dict[str, Any]:
    margin = (context_size - target_size) // 2 if view_mode == VIEW_CONTEXT512_ROI256 else 0
    target_roi = [margin, margin, margin + target_size, margin + target_size]
    tif_stem = tif_path.stem
    tif_prefix = tif_stem.split("_", 1)[0]
    sample_id = f"{scene_id}_{tif_prefix}_{row}_{col}"
    return {
        "id": sample_id,
        "image": relative_image.as_posix(),
        "meta": {
            "scene_id": scene_id,
            "tile_id": f"{scene_id}_{tif_prefix}",
            "tif_stem": tif_stem,
            "tif_prefix": tif_prefix,
            "source_tif": str(tif_path),
            "row": row,
            "col": col,
            "patch_row": row,
            "patch_col": col,
            "x0": col * target_size,
            "y0": row * target_size,
            "patch_size": target_size,
            "pixel_patch_size": target_size,
            "patch_width": target_size,
            "patch_height": target_size,
            "context_size": context_size,
            "input_image_size": context_size,
            "target_roi_in_image": target_roi,
            "view_mode": view_mode,
            "coord_mode": "norm1000",
            "coord_system": f"patch_norm{coord_range}",
            "coord_range": coord_range,
            "source_image_size": list(source_size),
            "padded_source_image_size": list(padded_size),
        },
        "conversations": [{"from": "human", "value": prompt}],
    }


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    target_size = int(args.target_size)
    stride = int(args.stride)
    context_size = target_size if args.view_mode == VIEW_LOCAL256 else int(args.context_size)
    if target_size <= 0 or stride != target_size:
        raise ValueError("RC E2E output requires --stride equal to the positive --target-size.")
    if context_size < target_size or (context_size - target_size) % 2:
        raise ValueError("Context size must be >= target size with an even centered margin.")
    if not 0.0 <= args.black_ratio_threshold <= 1.0:
        raise ValueError("--black-ratio-threshold must be within [0,1].")

    tif_paths = discover_inter_tifs(input_root)
    if args.max_tifs > 0:
        tif_paths = tif_paths[: args.max_tifs]
    if not tif_paths:
        raise FileNotFoundError(f"No *_inter.tif files found below {input_root}")

    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_root / "infer.jsonl"
    manifest_path = output_root / "patch_manifest.json"
    prompt = dataset_v2_prompt(
        view_mode=args.view_mode,
        target_size=target_size,
        context_size=context_size,
        coord_range=args.coord_range,
        include_intersections=args.include_intersections,
    )

    manifest: list[dict[str, Any]] = []
    kept = 0
    skipped_black = 0
    stop = False
    with output_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for tif_index, tif_path in enumerate(tif_paths, 1):
            scene_id = scene_id_for_tif(tif_path)
            image = read_tif_rgb(tif_path)
            source_height, source_width = image.shape[:2]
            padded = pad_bottom_right(image, target_size)
            padded_height, padded_width = padded.shape[:2]
            margin = (context_size - target_size) // 2
            context_canvas = (
                np.pad(padded, ((margin, margin), (margin, margin), (0, 0)), constant_values=0)
                if margin
                else padded
            )
            rows = padded_height // target_size
            cols = padded_width // target_size
            print(
                f"[e2e-data] tif {tif_index}/{len(tif_paths)}: {tif_path} "
                f"source={source_width}x{source_height} grid={rows}x{cols}",
                flush=True,
            )

            for row in range(rows):
                for col in range(cols):
                    y0 = row * target_size
                    x0 = col * target_size
                    target = padded[y0 : y0 + target_size, x0 : x0 + target_size]
                    ratio = black_ratio(target)
                    if ratio > args.black_ratio_threshold:
                        skipped_black += 1
                        continue
                    if margin:
                        model_image = context_canvas[y0 : y0 + context_size, x0 : x0 + context_size]
                    else:
                        model_image = target

                    relative_image = Path("images") / scene_id / tif_path.stem / f"{row}_{col}.png"
                    image_path = output_root / relative_image
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(model_image).save(image_path)

                    record = build_record(
                        scene_id=scene_id,
                        tif_path=tif_path,
                        relative_image=relative_image,
                        row=row,
                        col=col,
                        target_size=target_size,
                        context_size=context_size,
                        view_mode=args.view_mode,
                        coord_range=args.coord_range,
                        source_size=(source_width, source_height),
                        padded_size=(padded_width, padded_height),
                        prompt=prompt,
                    )
                    jsonl_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    manifest.append(
                        {
                            "id": record["id"],
                            "scene_id": scene_id,
                            "tif": tif_path.stem,
                            "patch": f"{row}_{col}.png",
                            "row": row,
                            "col": col,
                            "x0": x0,
                            "y0": y0,
                            "black_ratio": ratio,
                            "image_path": relative_image.as_posix(),
                            "target_roi_in_image": record["meta"]["target_roi_in_image"],
                        }
                    )
                    kept += 1
                    if args.max_patches > 0 and kept >= args.max_patches:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "view_mode": args.view_mode,
        "target_size": target_size,
        "context_size": context_size,
        "stride": stride,
        "coord_mode": "norm1000",
        "coord_range": int(args.coord_range),
        "black_ratio_threshold": float(args.black_ratio_threshold),
        "tif_count": len(tif_paths),
        "patch_count": kept,
        "skipped_black": skipped_black,
        "infer_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_path),
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
