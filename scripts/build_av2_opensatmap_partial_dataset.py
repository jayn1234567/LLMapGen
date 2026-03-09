#!/usr/bin/env python3
"""Build a UniMapGen-readable dataset from AV2+OpenSatMap cropped patches."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=str,
        default="/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix",
        help="Directory containing satellite/ and ground_truth/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_partial_fix",
        help="Output dataset root with train/ val/ annotations.json and geometry metadata.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio on the matched token list.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap after sorting matched tokens. Useful for quick experiments.",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to materialize train/val images in the output dataset root.",
    )
    return parser.parse_args()


def token_sort_key(token: str) -> Tuple[str, int, int, str]:
    parts = str(token).split("__")
    log_id = parts[0] if parts else ""
    timestamp = -1
    frame_id = -1
    for part in parts[1:]:
        if part.isdigit():
            timestamp = int(part)
            continue
        m = re.fullmatch(r"frame(\d+)", part)
        if m:
            frame_id = int(m.group(1))
    return log_id, timestamp, frame_id, str(token)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON must be a dict: {path}")
    return obj


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def replace_path(dst: Path, src: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    dst.symlink_to(src)


def collect_satellite_paths(sat_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sat_dir.rglob("*_satellite.png"):
        out[p.name[: -len("_satellite.png")]] = p
    return out


def collect_gt_paths(gt_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in gt_dir.rglob("*.json"):
        out[p.stem] = p
    return out


def iter_tokens(sat_token_paths: Dict[str, Path], gt_token_paths: Dict[str, Path]) -> Tuple[List[str], List[str], List[str]]:
    sat_tokens = set(sat_token_paths)
    gt_tokens = set(gt_token_paths)
    matched = sorted(set(sat_tokens) & set(gt_tokens), key=token_sort_key)
    sat_only = sorted(set(sat_tokens) - set(gt_tokens), key=token_sort_key)
    gt_only = sorted(set(gt_tokens) - set(sat_tokens), key=token_sort_key)
    return matched, sat_only, gt_only


def build_patch_geometry(token: str, ann: Dict) -> Optional[Dict]:
    gps_center = ann.get("gps_center", {})
    if isinstance(gps_center, dict):
        lon = gps_center.get("lon")
        lat = gps_center.get("lat")
    elif isinstance(gps_center, (list, tuple)) and len(gps_center) >= 2:
        lon, lat = gps_center[0], gps_center[1]
    else:
        return None
    if lon is None or lat is None:
        return None

    crop_box = ann.get("crop_box", {}) or {}
    crop_region = {
        "x_min": int(crop_box.get("x_min", 0)),
        "y_min": int(crop_box.get("y_min", 0)),
        "x_max": int(crop_box.get("x_max", ann.get("image_width", 896))),
        "y_max": int(crop_box.get("y_max", ann.get("image_height", 896))),
        "center_x": int(crop_box.get("center_x", 0)),
        "center_y": int(crop_box.get("center_y", 0)),
        "original_image": str(ann.get("source_image", "")),
        "source_split": str(ann.get("source_split", "")),
        "city": str(ann.get("city", "")),
    }
    return {
        "sample_token": token,
        "gps_center": [float(lon), float(lat)],
        "image_width": int(ann.get("image_width", 896)),
        "image_height": int(ann.get("image_height", 896)),
        "crop_region": crop_region,
    }


def split_tokens(tokens: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    if not tokens:
        return [], []
    if len(tokens) == 1:
        return tokens[:], []
    val_count = max(1, int(round(len(tokens) * float(val_ratio))))
    val_count = min(val_count, len(tokens) - 1)
    train_tokens = tokens[: len(tokens) - val_count]
    val_tokens = tokens[len(tokens) - val_count :]
    return train_tokens, val_tokens


def dump_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sat_dir = input_root / "satellite"
    gt_dir = input_root / "ground_truth"
    if not sat_dir.is_dir():
        raise FileNotFoundError(f"missing satellite dir: {sat_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"missing ground_truth dir: {gt_dir}")

    sat_token_paths = collect_satellite_paths(sat_dir)
    gt_token_paths = collect_gt_paths(gt_dir)
    matched_tokens, sat_only_tokens, gt_only_tokens = iter_tokens(
        sat_token_paths=sat_token_paths,
        gt_token_paths=gt_token_paths,
    )
    if args.max_samples is not None and args.max_samples > 0:
        matched_tokens = matched_tokens[: int(args.max_samples)]
    if not matched_tokens:
        raise RuntimeError("no matched satellite/ground_truth samples found")

    train_tokens, val_tokens = split_tokens(tokens=matched_tokens, val_ratio=float(args.val_ratio))
    split_by_token = {tok: "train" for tok in train_tokens}
    split_by_token.update({tok: "val" for tok in val_tokens})

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    output_root.mkdir(parents=True, exist_ok=True)
    reset_dir(train_dir)
    reset_dir(val_dir)

    annotations: Dict[str, Dict] = {}
    patch_geometry: Dict[str, Dict] = {}
    manifest: Dict[str, Dict] = {}
    skipped_tokens: List[str] = []

    for token in matched_tokens:
        image_name = f"{token}_satellite.png"
        src_img = sat_token_paths[token]
        src_gt = gt_token_paths[token]
        try:
            ann = load_json(src_gt)
        except Exception:
            skipped_tokens.append(token)
            continue

        split = split_by_token[token]
        dst_dir = train_dir if split == "train" else val_dir
        replace_path(dst=dst_dir / image_name, src=src_img, mode=str(args.link_mode))

        annotations[image_name] = ann
        geom = build_patch_geometry(token=token, ann=ann)
        if geom is not None:
            patch_geometry[token] = geom
        manifest[token] = {
            "split": split,
            "image_name": image_name,
            "image_path": str(src_img),
            "gt_path": str(src_gt),
            "source_image": str(ann.get("source_image", "")),
            "source_split": str(ann.get("source_split", "")),
            "city": str(ann.get("city", "")),
            "num_lines": len(ann.get("lines", [])) if isinstance(ann.get("lines", []), list) else 0,
        }

    train_tokens_final = [tok for tok in train_tokens if tok in manifest]
    val_tokens_final = [tok for tok in val_tokens if tok in manifest]
    skipped_after_match = [tok for tok in matched_tokens if tok not in manifest]

    splits_meta = {
        "train_tokens": train_tokens_final,
        "val_tokens": val_tokens_final,
        "test_tokens": [],
        "all_tokens": train_tokens_final + val_tokens_final,
    }
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "note": "This output is a snapshot built from a potentially changing crop directory. Rebuild before training if the source crop job is still running.",
        "link_mode": str(args.link_mode),
        "val_ratio": float(args.val_ratio),
        "max_samples": int(args.max_samples) if args.max_samples is not None else None,
        "num_satellite_files": len(sat_token_paths),
        "num_ground_truth_files": len(gt_token_paths),
        "num_matched_tokens_before_skip": len(matched_tokens),
        "num_train_tokens": len(train_tokens_final),
        "num_val_tokens": len(val_tokens_final),
        "num_patch_geometry": len(patch_geometry),
        "sat_only_tokens": sat_only_tokens,
        "gt_only_tokens": gt_only_tokens,
        "skipped_tokens": skipped_tokens,
        "skipped_after_match": skipped_after_match,
    }

    dump_json(output_root / "annotations.json", annotations)
    dump_json(output_root / "splits_meta.json", splits_meta)
    dump_json(output_root / "patch_geometry.json", patch_geometry)
    dump_json(output_root / "manifest.json", manifest)
    dump_json(output_root / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
