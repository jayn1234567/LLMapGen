#!/usr/bin/env python3
"""Audit patch-ID/count parity with the original RC E2E TIF splitter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.prepare_rc_e2e_inference_dataset import discover_inter_tifs, scene_id_for_tif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-e2e-root", required=True)
    parser.add_argument(
        "--original-input-root",
        default="",
        help="Directory passed as input_root to the original splitter; defaults to raw-e2e-root.",
    )
    parser.add_argument("--current-manifest", required=True)
    parser.add_argument("--black-ratio-threshold", type=float, default=0.98)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def original_discover(input_root: Path) -> list[Path]:
    paths: list[Path] = []
    for id_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        inter_dir = id_dir / "rc_one_patch_release" / "center_line_v2" / "inter_patch_tif"
        if inter_dir.is_dir():
            paths.extend(sorted(inter_dir.glob("*_inter.tif")))
    return paths


def patch_key(tif_path: Path, row: int, col: int) -> str:
    return f"{scene_id_for_tif(tif_path)}/{tif_path.stem}/{row}_{col}"


def pad_channels_first(image: np.ndarray, patch_size: int) -> np.ndarray:
    _, height, width = image.shape
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


def original_patch_array(chunk: np.ndarray) -> np.ndarray:
    if chunk.shape[0] == 1:
        array = np.repeat(chunk, 3, axis=0)
    else:
        array = chunk[:3]
    array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def current_patch_array(chunk: np.ndarray) -> np.ndarray:
    array = np.transpose(chunk, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 2:
        array = np.concatenate((array, array[:, :, :1]), axis=2)
    elif array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def simulate_tif(tif_path: Path, patch_size: int, threshold: float) -> dict[str, Any]:
    with rasterio.open(tif_path) as source:
        image = source.read()
        source_width = int(source.width)
        source_height = int(source.height)
        channel_count = int(source.count)
    padded = pad_channels_first(image, patch_size)
    _, height, width = padded.shape
    rows = height // patch_size
    cols = width // patch_size
    original_keys: set[str] = set()
    current_keys: set[str] = set()
    differing_filter: list[dict[str, Any]] = []

    for row in range(rows):
        for col in range(cols):
            y0 = row * patch_size
            x0 = col * patch_size
            chunk = padded[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
            original_ratio = float(np.mean(original_patch_array(chunk) == 0))
            current_ratio = float(np.mean(current_patch_array(chunk) == 0))
            original_keep = original_ratio <= threshold
            current_keep = current_ratio <= threshold
            key = patch_key(tif_path, row, col)
            if original_keep:
                original_keys.add(key)
            if current_keep:
                current_keys.add(key)
            if original_keep != current_keep:
                differing_filter.append(
                    {
                        "key": key,
                        "original_black_ratio": original_ratio,
                        "current_black_ratio": current_ratio,
                    }
                )
    return {
        "path": str(tif_path),
        "source_size": [source_width, source_height],
        "channels": channel_count,
        "grid_rows_cols": [rows, cols],
        "grid_count": rows * cols,
        "original_keys": original_keys,
        "current_keys": current_keys,
        "filter_differences": differing_filter,
    }


def load_manifest_keys(path: Path) -> tuple[set[str], int, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    keys: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id") or item.get("id") or "")
        tif_stem = str(item.get("tif") or "")
        row = item.get("row")
        col = item.get("col")
        if scene_id and tif_stem and row is not None and col is not None:
            keys.append(f"{scene_id}/{tif_stem}/{int(row)}_{int(col)}")
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    return set(keys), len(payload), duplicates


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_e2e_root).resolve()
    original_input_root = Path(args.original_input_root).resolve() if args.original_input_root else raw_root
    manifest_path = Path(args.current_manifest).resolve()
    recursive_tifs = discover_inter_tifs(raw_root)
    top_level_tifs = original_discover(original_input_root)
    if not recursive_tifs:
        raise FileNotFoundError(f"No recursive *_inter.tif found below {raw_root}")

    recursive_paths = {path.resolve() for path in recursive_tifs}
    original_paths = {path.resolve() for path in top_level_tifs}
    expected_original: set[str] = set()
    expected_current: set[str] = set()
    filter_differences: list[dict[str, Any]] = []
    per_tif: list[dict[str, Any]] = []
    channel_counts: Counter[int] = Counter()
    total_grid_count = 0

    for index, tif_path in enumerate(recursive_tifs, 1):
        result = simulate_tif(tif_path, args.patch_size, args.black_ratio_threshold)
        expected_original.update(result.pop("original_keys"))
        expected_current.update(result.pop("current_keys"))
        filter_differences.extend(result.pop("filter_differences"))
        channel_counts[result["channels"]] += 1
        total_grid_count += result["grid_count"]
        per_tif.append(result)
        if index % 10 == 0 or index == len(recursive_tifs):
            print(f"[e2e-patch-audit] scanned_tifs={index}/{len(recursive_tifs)}", flush=True)

    manifest_keys, manifest_entries, duplicate_manifest_keys = load_manifest_keys(manifest_path)
    report = {
        "raw_e2e_root": str(raw_root),
        "original_input_root": str(original_input_root),
        "current_manifest": str(manifest_path),
        "config": {
            "patch_size": args.patch_size,
            "black_ratio_threshold": args.black_ratio_threshold,
        },
        "counts": {
            "recursive_tifs_current_discovery": len(recursive_tifs),
            "top_level_tifs_original_discovery": len(top_level_tifs),
            "total_padded_grid_patches": total_grid_count,
            "expected_original_kept_same_recursive_tifs": len(expected_original),
            "expected_current_kept": len(expected_current),
            "manifest_entries": manifest_entries,
            "manifest_unique_keys": len(manifest_keys),
            "channel_counts": {str(key): value for key, value in sorted(channel_counts.items())},
        },
        "parity": {
            "same_tif_discovery": recursive_paths == original_paths,
            "same_filtering_on_recursive_tifs": expected_original == expected_current,
            "manifest_matches_current_algorithm": manifest_keys == expected_current,
            "ok": (
                recursive_paths == original_paths
                and expected_original == expected_current
                and manifest_keys == expected_current
                and not duplicate_manifest_keys
            ),
        },
        "differences": {
            "tifs_only_recursive_discovery": [str(path) for path in sorted(recursive_paths - original_paths)[:50]],
            "tifs_only_original_discovery": [str(path) for path in sorted(original_paths - recursive_paths)[:50]],
            "patches_only_original_filter": sorted(expected_original - expected_current)[:100],
            "patches_only_current_filter": sorted(expected_current - expected_original)[:100],
            "manifest_missing_expected": sorted(expected_current - manifest_keys)[:100],
            "manifest_unexpected_extra": sorted(manifest_keys - expected_current)[:100],
            "duplicate_manifest_keys": duplicate_manifest_keys[:100],
            "filter_difference_examples": filter_differences[:100],
        },
        "per_tif": per_tif,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"[e2e-patch-audit] report={output_path}")
    if args.strict and not report["parity"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
