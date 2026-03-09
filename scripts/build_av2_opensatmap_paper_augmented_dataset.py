#!/usr/bin/env python3
"""Build a paper-style augmented OpenSatMap snapshot dataset.

This builder starts from the already aligned 896x896 AV2/OpenSatMap crops and
expands the train split with configurable:
  - overlap crops (requires raw OpenSatMap tiles + annotrainval20.json)
  - inclined crops (requires raw OpenSatMap tiles + annotrainval20.json)
  - rotation augmentation (can run from cropped patches only)

The output keeps the same UniMapGen-readable layout:
  train/ val/ annotations.json splits_meta.json patch_geometry.json manifest.json

Design constraints:
  - val split remains unaugmented by default.
  - stage-3/state training should keep using the unaugmented snapshot; this
    augmented dataset is primarily for paper-aligned stage-1/2 style training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


GPS_PAT = re.compile(r"\{([\-\deE+.]+)\s+([\-\deE+.]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop-root",
        type=str,
        default="/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix",
        help="Aligned crop root containing satellite/ and ground_truth/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_paper_aug_partial",
        help="Output dataset root.",
    )
    parser.add_argument(
        "--opensatmap-root",
        type=str,
        default="",
        help="Optional raw OpenSatMap root containing picuse20trainvaltest/, GPS_info_all.json, annotrainval20.json.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio on the matched base token list.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the matched base token list before splitting.",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to materialize non-generated base images.",
    )
    parser.add_argument(
        "--rotation-angles-deg",
        type=str,
        default="90,180,270",
        help="Comma-separated in-patch rotation augmentation angles.",
    )
    parser.add_argument(
        "--inclined-angles-deg",
        type=str,
        default="-15,15",
        help="Comma-separated source-tile rotated crop angles in degrees.",
    )
    parser.add_argument(
        "--overlap-offsets-px",
        type=str,
        default="448,0;-448,0;0,448;0,-448;448,448;448,-448;-448,448;-448,-448",
        help="Semicolon-separated dx,dy offsets in source-tile pixels for overlap crops.",
    )
    parser.add_argument(
        "--disable-rotation",
        action="store_true",
        help="Disable in-patch rotation augmentation.",
    )
    parser.add_argument(
        "--disable-overlap",
        action="store_true",
        help="Disable overlap crop augmentation.",
    )
    parser.add_argument(
        "--disable-inclined",
        action="store_true",
        help="Disable inclined crop augmentation.",
    )
    parser.add_argument(
        "--allow-val-augmentation",
        action="store_true",
        help="Also augment the val split. Default keeps val unaugmented.",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(float(chunk))
    return out


def parse_offset_pairs(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [x.strip() for x in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bad offset pair: {chunk}")
        out.append((int(round(float(parts[0]))), int(round(float(parts[1])))))
    return out


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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def infer_axis_bounds_from_centers(values: Sequence[float]) -> Tuple[float, float]:
    unique_vals = sorted(set(float(v) for v in values))
    if not unique_vals:
        raise ValueError("Cannot infer bounds from an empty coordinate list.")
    if len(unique_vals) == 1:
        return unique_vals[0], unique_vals[0]
    steps = [b - a for a, b in zip(unique_vals[:-1], unique_vals[1:]) if b > a]
    if not steps:
        return unique_vals[0], unique_vals[-1]
    step = float(np.median(np.asarray(steps, dtype=np.float64)))
    return unique_vals[0] - step * 0.5, unique_vals[-1] + step * 0.5


def parse_gps_string(gps_str: str) -> Tuple[float, float]:
    match = GPS_PAT.search(str(gps_str))
    if not match:
        raise ValueError(f"Cannot parse GPS string: {gps_str}")
    return float(match.group(1)), float(match.group(2))


def load_tile_bounds(opensatmap_root: Path) -> Dict[str, Dict]:
    gps_info_path = opensatmap_root / "GPS_info_all.json"
    if not gps_info_path.is_file():
        return {}
    gps_info = load_json(gps_info_path)
    bounds = {}
    for image_name, tiles in gps_info.items():
        if not isinstance(tiles, list):
            continue
        lons: List[float] = []
        lats: List[float] = []
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            lon, lat = parse_gps_string(tile.get("centerGPS", ""))
            lons.append(lon)
            lats.append(lat)
        if not lons or not lats:
            continue
        bounds[str(image_name)] = {
            "lon_range": infer_axis_bounds_from_centers(lons),
            "lat_range": infer_axis_bounds_from_centers(lats),
        }
    return bounds


def pixel_to_gps(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
) -> Tuple[float, float]:
    x_norm = float(x) / max(float(image_width - 1), 1.0)
    y_norm = 1.0 - float(y) / max(float(image_height - 1), 1.0)
    lon = lon_range[0] + x_norm * (lon_range[1] - lon_range[0])
    lat = lat_range[0] + y_norm * (lat_range[1] - lat_range[0])
    return float(lon), float(lat)


class ImageCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = int(max_items)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = np.asarray(Image.open(path).convert("RGB"))
        self._cache[key] = value
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return value


def safe_axis_aligned_box(center_x: float, center_y: float, patch_size: int, image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
    half = patch_size * 0.5
    x_min = int(round(center_x - half))
    y_min = int(round(center_y - half))
    x_max = x_min + int(patch_size)
    y_max = y_min + int(patch_size)
    if x_min < 0 or y_min < 0 or x_max > int(image_width) or y_max > int(image_height):
        return None
    return x_min, y_min, x_max, y_max


def rotate_points(points: np.ndarray, center_xy: Tuple[float, float], angle_deg: float) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cx = float(center_xy[0])
    cy = float(center_xy[1])
    ang = math.radians(float(angle_deg))
    c = math.cos(ang)
    s = math.sin(ang)
    rel = arr - np.asarray([[cx, cy]], dtype=np.float32)
    rot = np.empty_like(rel)
    rot[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    rot[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    return rot + np.asarray([[cx, cy]], dtype=np.float32)


def transform_lines_to_patch(
    raw_lines: Sequence[Dict],
    patch_size: int,
    point_to_local_fn,
    keep_margin_px: float = 10.0,
) -> List[Dict]:
    out: List[Dict] = []
    for line in raw_lines:
        points = line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        new_points = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            local_xy = point_to_local_fn(float(point[0]), float(point[1]))
            x = float(local_xy[0])
            y = float(local_xy[1])
            if -keep_margin_px <= x <= float(patch_size) + keep_margin_px and -keep_margin_px <= y <= float(patch_size) + keep_margin_px:
                new_points.append(
                    [
                        max(0.0, min(float(patch_size), x)),
                        max(0.0, min(float(patch_size), y)),
                    ]
                )
        if len(new_points) < 2:
            continue
        rec = dict(line)
        rec["points"] = new_points
        out.append(rec)
    return out


def crop_source_patch(
    image_array: np.ndarray,
    raw_lines: Sequence[Dict],
    center_x: float,
    center_y: float,
    patch_size: int,
    angle_deg: float = 0.0,
) -> Optional[Tuple[np.ndarray, List[Dict], Dict]]:
    image_height, image_width = image_array.shape[:2]
    crop_box = safe_axis_aligned_box(
        center_x=center_x,
        center_y=center_y,
        patch_size=int(patch_size),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    if crop_box is None:
        return None
    x_min, y_min, x_max, y_max = crop_box
    if abs(float(angle_deg)) < 1e-6:
        cropped = image_array[y_min:y_max, x_min:x_max].copy()
        lines = transform_lines_to_patch(
            raw_lines=raw_lines,
            patch_size=int(patch_size),
            point_to_local_fn=lambda x, y: (x - float(x_min), y - float(y_min)),
        )
    else:
        pil_img = Image.fromarray(image_array)
        rotated = pil_img.rotate(float(angle_deg), resample=Image.Resampling.BILINEAR, expand=False, center=(float(center_x), float(center_y)))
        cropped = np.asarray(rotated.crop((x_min, y_min, x_max, y_max)).convert("RGB"))
        lines = transform_lines_to_patch(
            raw_lines=raw_lines,
            patch_size=int(patch_size),
            point_to_local_fn=lambda x, y: (
                rotate_points(np.asarray([[x, y]], dtype=np.float32), center_xy=(float(center_x), float(center_y)), angle_deg=float(angle_deg))[0, 0] - float(x_min),
                rotate_points(np.asarray([[x, y]], dtype=np.float32), center_xy=(float(center_x), float(center_y)), angle_deg=float(angle_deg))[0, 1] - float(y_min),
            ),
        )
    meta = {
        "crop_box": {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "center_x": int(round(center_x)),
            "center_y": int(round(center_y)),
        },
        "angle_deg": float(angle_deg),
        "image_width": int(patch_size),
        "image_height": int(patch_size),
    }
    return cropped, lines, meta


def rotate_patch_image_and_ann(image_array: np.ndarray, ann: Dict, angle_deg: float) -> Tuple[np.ndarray, Dict]:
    patch_size = int(ann.get("image_width", image_array.shape[1]))
    center_xy = (0.5 * float(patch_size), 0.5 * float(patch_size))
    pil_img = Image.fromarray(image_array)
    rotated = pil_img.rotate(float(angle_deg), resample=Image.Resampling.BILINEAR, expand=False, center=center_xy)
    out_ann = dict(ann)
    out_lines = []
    for line in ann.get("lines", []):
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        pts_rot = rotate_points(pts, center_xy=center_xy, angle_deg=float(angle_deg))
        pts_rot[:, 0] = np.clip(pts_rot[:, 0], 0.0, float(patch_size))
        pts_rot[:, 1] = np.clip(pts_rot[:, 1], 0.0, float(patch_size))
        rec = dict(line)
        rec["points"] = pts_rot.tolist()
        out_lines.append(rec)
    out_ann["lines"] = out_lines
    return np.asarray(rotated.convert("RGB")), out_ann


def build_patch_geometry(token: str, ann: Dict, extra_meta: Optional[Dict] = None) -> Optional[Dict]:
    gps_center = ann.get("gps_center", {})
    if isinstance(gps_center, dict):
        lon = gps_center.get("lon")
        lat = gps_center.get("lat")
    elif isinstance(gps_center, (list, tuple)) and len(gps_center) >= 2:
        lon, lat = gps_center[0], gps_center[1]
    else:
        lon, lat = None, None
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
    rec = {
        "sample_token": token,
        "gps_center": [float(lon), float(lat)],
        "image_width": int(ann.get("image_width", 896)),
        "image_height": int(ann.get("image_height", 896)),
        "crop_region": crop_region,
    }
    if extra_meta:
        rec["augmentation"] = dict(extra_meta)
    return rec


def write_generated_image(dst: Path, image_array: np.ndarray) -> None:
    Image.fromarray(np.asarray(image_array, dtype=np.uint8)).save(dst)


def make_aug_token(base_token: str, suffix: str) -> str:
    return f"{base_token}__{suffix}"


def copy_ann(ann: Dict) -> Dict:
    return json.loads(json.dumps(ann))


def main() -> None:
    args = parse_args()

    crop_root = Path(args.crop_root)
    output_root = Path(args.output_root)
    opensatmap_root = Path(args.opensatmap_root) if str(args.opensatmap_root).strip() else None

    sat_dir = crop_root / "satellite"
    gt_dir = crop_root / "ground_truth"
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
        raise RuntimeError("no matched base samples found")

    train_tokens, val_tokens = split_tokens(tokens=matched_tokens, val_ratio=float(args.val_ratio))
    split_by_token = {tok: "train" for tok in train_tokens}
    split_by_token.update({tok: "val" for tok in val_tokens})

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    output_root.mkdir(parents=True, exist_ok=True)
    reset_dir(train_dir)
    reset_dir(val_dir)

    rotation_angles = [] if args.disable_rotation else [x for x in parse_float_list(args.rotation_angles_deg) if abs(float(x)) > 1e-6]
    inclined_angles = [] if args.disable_inclined else [x for x in parse_float_list(args.inclined_angles_deg) if abs(float(x)) > 1e-6]
    overlap_offsets = [] if args.disable_overlap else [x for x in parse_offset_pairs(args.overlap_offsets_px) if x != (0, 0)]

    raw_annotations = {}
    tile_bounds = {}
    image_cache = ImageCache(max_items=8)
    raw_ready = False
    if opensatmap_root is not None:
        ann_path = opensatmap_root / "annotrainval20.json"
        split_root = opensatmap_root / "picuse20trainvaltest"
        gps_info_path = opensatmap_root / "GPS_info_all.json"
        raw_ready = ann_path.is_file() and split_root.is_dir() and gps_info_path.is_file()
        if raw_ready:
            raw_annotations = load_json(ann_path)
            tile_bounds = load_tile_bounds(opensatmap_root)

    annotations: Dict[str, Dict] = {}
    patch_geometry: Dict[str, Dict] = {}
    manifest: Dict[str, Dict] = {}
    summary_counts = defaultdict(int)
    summary_counts["num_base_tokens"] = len(matched_tokens)

    for base_token in matched_tokens:
        split = split_by_token[base_token]
        out_dir = train_dir if split == "train" else val_dir
        base_image_name = f"{base_token}_satellite.png"
        src_img = sat_token_paths[base_token]
        base_ann = load_json(gt_token_paths[base_token])
        patch_size = int(base_ann.get("image_width", 896))

        # Base sample.
        replace_path(dst=out_dir / base_image_name, src=src_img, mode=str(args.link_mode))
        annotations[base_image_name] = base_ann
        patch_geometry[base_token] = build_patch_geometry(base_token, base_ann, extra_meta={"augmentation_type": "base"})
        manifest[base_token] = {
            "split": split,
            "image_name": base_image_name,
            "augmentation_type": "base",
            "parent_token": base_token,
            "source_image": str(base_ann.get("source_image", "")),
            "source_split": str(base_ann.get("source_split", "")),
            "city": str(base_ann.get("city", "")),
            "num_lines": len(base_ann.get("lines", [])) if isinstance(base_ann.get("lines", []), list) else 0,
        }
        summary_counts[f"{split}_base"] += 1

        allow_aug = split == "train" or bool(args.allow_val_augmentation)
        if not allow_aug:
            continue

        base_image_arr = None
        if rotation_angles:
            base_image_arr = np.asarray(Image.open(src_img).convert("RGB"))
            for angle in rotation_angles:
                aug_token = make_aug_token(base_token, f"rot{int(round(angle)):03d}")
                aug_image_name = f"{aug_token}_satellite.png"
                rot_img, rot_ann = rotate_patch_image_and_ann(base_image_arr, base_ann, angle_deg=float(angle))
                write_generated_image(out_dir / aug_image_name, rot_img)
                annotations[aug_image_name] = rot_ann
                patch_geometry[aug_token] = build_patch_geometry(
                    aug_token,
                    rot_ann,
                    extra_meta={"augmentation_type": "rotation", "rotation_deg": float(angle), "parent_token": base_token},
                )
                manifest[aug_token] = {
                    "split": split,
                    "image_name": aug_image_name,
                    "augmentation_type": "rotation",
                    "rotation_deg": float(angle),
                    "parent_token": base_token,
                    "source_image": str(base_ann.get("source_image", "")),
                    "source_split": str(base_ann.get("source_split", "")),
                    "city": str(base_ann.get("city", "")),
                    "num_lines": len(rot_ann.get("lines", [])),
                }
                summary_counts[f"{split}_rotation"] += 1

        if not raw_ready:
            continue

        source_image = str(base_ann.get("source_image", ""))
        source_split = str(base_ann.get("source_split", ""))
        raw_image_path = opensatmap_root / "picuse20trainvaltest" / source_split / source_image
        raw_rec = raw_annotations.get(source_image, {})
        raw_lines = raw_rec.get("lines", []) if isinstance(raw_rec, dict) else []
        if not raw_image_path.is_file() or not isinstance(raw_lines, list):
            continue

        crop_box = base_ann.get("crop_box", {}) or {}
        center_x = float(crop_box.get("center_x", 0.0))
        center_y = float(crop_box.get("center_y", 0.0))
        raw_image_arr = image_cache.get(raw_image_path)

        def maybe_attach_gps(ann: Dict) -> Dict:
            out_ann = copy_ann(ann)
            if source_image in tile_bounds and "crop_box" in out_ann:
                bounds = tile_bounds[source_image]
                cb = out_ann["crop_box"]
                lon, lat = pixel_to_gps(
                    x=float(cb.get("center_x", 0.0)),
                    y=float(cb.get("center_y", 0.0)),
                    image_width=int(raw_image_arr.shape[1]),
                    image_height=int(raw_image_arr.shape[0]),
                    lon_range=tuple(bounds["lon_range"]),
                    lat_range=tuple(bounds["lat_range"]),
                )
                out_ann["gps_center"] = {"lat": float(lat), "lon": float(lon)}
            return out_ann

        for dx, dy in overlap_offsets:
            cropped = crop_source_patch(
                image_array=raw_image_arr,
                raw_lines=raw_lines,
                center_x=center_x + float(dx),
                center_y=center_y + float(dy),
                patch_size=patch_size,
                angle_deg=0.0,
            )
            if cropped is None:
                summary_counts["skip_overlap_out_of_bounds"] += 1
                continue
            aug_image, aug_lines, aug_meta = cropped
            aug_token = make_aug_token(base_token, f"ovp_x{dx:+d}_y{dy:+d}")
            aug_image_name = f"{aug_token}_satellite.png"
            aug_ann = {
                "image_width": patch_size,
                "image_height": patch_size,
                "lines": aug_lines,
                "source_image": source_image,
                "source_split": source_split,
                "city": str(base_ann.get("city", "")),
                "crop_box": aug_meta["crop_box"],
            }
            aug_ann = maybe_attach_gps(aug_ann)
            write_generated_image(out_dir / aug_image_name, aug_image)
            annotations[aug_image_name] = aug_ann
            patch_geometry[aug_token] = build_patch_geometry(
                aug_token,
                aug_ann,
                extra_meta={
                    "augmentation_type": "overlap",
                    "shift_dx_px": int(dx),
                    "shift_dy_px": int(dy),
                    "parent_token": base_token,
                },
            )
            manifest[aug_token] = {
                "split": split,
                "image_name": aug_image_name,
                "augmentation_type": "overlap",
                "shift_dx_px": int(dx),
                "shift_dy_px": int(dy),
                "parent_token": base_token,
                "source_image": source_image,
                "source_split": source_split,
                "city": str(base_ann.get("city", "")),
                "num_lines": len(aug_lines),
            }
            summary_counts[f"{split}_overlap"] += 1
            for angle in rotation_angles:
                rot_token = make_aug_token(aug_token, f"rot{int(round(angle)):03d}")
                rot_image_name = f"{rot_token}_satellite.png"
                rot_img, rot_ann = rotate_patch_image_and_ann(aug_image, aug_ann, angle_deg=float(angle))
                write_generated_image(out_dir / rot_image_name, rot_img)
                annotations[rot_image_name] = rot_ann
                patch_geometry[rot_token] = build_patch_geometry(
                    rot_token,
                    rot_ann,
                    extra_meta={
                        "augmentation_type": "overlap_rotation",
                        "rotation_deg": float(angle),
                        "shift_dx_px": int(dx),
                        "shift_dy_px": int(dy),
                        "parent_token": base_token,
                    },
                )
                manifest[rot_token] = {
                    "split": split,
                    "image_name": rot_image_name,
                    "augmentation_type": "overlap_rotation",
                    "rotation_deg": float(angle),
                    "shift_dx_px": int(dx),
                    "shift_dy_px": int(dy),
                    "parent_token": base_token,
                    "source_image": source_image,
                    "source_split": source_split,
                    "city": str(base_ann.get("city", "")),
                    "num_lines": len(rot_ann.get("lines", [])),
                }
                summary_counts[f"{split}_overlap_rotation"] += 1

        for crop_angle in inclined_angles:
            cropped = crop_source_patch(
                image_array=raw_image_arr,
                raw_lines=raw_lines,
                center_x=center_x,
                center_y=center_y,
                patch_size=patch_size,
                angle_deg=float(crop_angle),
            )
            if cropped is None:
                summary_counts["skip_inclined_out_of_bounds"] += 1
                continue
            aug_image, aug_lines, aug_meta = cropped
            aug_token = make_aug_token(base_token, f"inc_{int(round(crop_angle)):03d}")
            aug_image_name = f"{aug_token}_satellite.png"
            aug_ann = {
                "image_width": patch_size,
                "image_height": patch_size,
                "lines": aug_lines,
                "source_image": source_image,
                "source_split": source_split,
                "city": str(base_ann.get("city", "")),
                "crop_box": aug_meta["crop_box"],
            }
            aug_ann = maybe_attach_gps(aug_ann)
            write_generated_image(out_dir / aug_image_name, aug_image)
            annotations[aug_image_name] = aug_ann
            patch_geometry[aug_token] = build_patch_geometry(
                aug_token,
                aug_ann,
                extra_meta={
                    "augmentation_type": "inclined",
                    "crop_angle_deg": float(crop_angle),
                    "parent_token": base_token,
                },
            )
            manifest[aug_token] = {
                "split": split,
                "image_name": aug_image_name,
                "augmentation_type": "inclined",
                "crop_angle_deg": float(crop_angle),
                "parent_token": base_token,
                "source_image": source_image,
                "source_split": source_split,
                "city": str(base_ann.get("city", "")),
                "num_lines": len(aug_lines),
            }
            summary_counts[f"{split}_inclined"] += 1
            for angle in rotation_angles:
                rot_token = make_aug_token(aug_token, f"rot{int(round(angle)):03d}")
                rot_image_name = f"{rot_token}_satellite.png"
                rot_img, rot_ann = rotate_patch_image_and_ann(aug_image, aug_ann, angle_deg=float(angle))
                write_generated_image(out_dir / rot_image_name, rot_img)
                annotations[rot_image_name] = rot_ann
                patch_geometry[rot_token] = build_patch_geometry(
                    rot_token,
                    rot_ann,
                    extra_meta={
                        "augmentation_type": "inclined_rotation",
                        "crop_angle_deg": float(crop_angle),
                        "rotation_deg": float(angle),
                        "parent_token": base_token,
                    },
                )
                manifest[rot_token] = {
                    "split": split,
                    "image_name": rot_image_name,
                    "augmentation_type": "inclined_rotation",
                    "crop_angle_deg": float(crop_angle),
                    "rotation_deg": float(angle),
                    "parent_token": base_token,
                    "source_image": source_image,
                    "source_split": source_split,
                    "city": str(base_ann.get("city", "")),
                    "num_lines": len(rot_ann.get("lines", [])),
                }
                summary_counts[f"{split}_inclined_rotation"] += 1

    train_token_list = sorted([tok for tok, rec in manifest.items() if rec.get("split") == "train"], key=token_sort_key)
    val_token_list = sorted([tok for tok, rec in manifest.items() if rec.get("split") == "val"], key=token_sort_key)
    splits_meta = {
        "train_tokens": train_token_list,
        "val_tokens": val_token_list,
        "test_tokens": [],
        "all_tokens": train_token_list + val_token_list,
    }
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "crop_root": str(crop_root),
        "output_root": str(output_root),
        "opensatmap_root": str(opensatmap_root) if opensatmap_root is not None else "",
        "raw_opensatmap_ready": bool(raw_ready),
        "note": (
            "Val is kept unaugmented by default. This dataset is intended for paper-style stage1/2 augmentation; "
            "keep the unaugmented snapshot for stage3 state-update training."
        ),
        "val_ratio": float(args.val_ratio),
        "max_samples": int(args.max_samples) if args.max_samples is not None else None,
        "link_mode": str(args.link_mode),
        "rotation_angles_deg": rotation_angles,
        "inclined_angles_deg": inclined_angles,
        "overlap_offsets_px": overlap_offsets,
        "sat_only_tokens": sat_only_tokens,
        "gt_only_tokens": gt_only_tokens,
        "counts": dict(summary_counts),
        "num_annotations": len(annotations),
        "num_patch_geometry": len(patch_geometry),
        "num_train_tokens": len(train_token_list),
        "num_val_tokens": len(val_token_list),
    }

    dump_json(output_root / "annotations.json", annotations)
    dump_json(output_root / "splits_meta.json", splits_meta)
    dump_json(output_root / "patch_geometry.json", patch_geometry)
    dump_json(output_root / "manifest.json", manifest)
    dump_json(output_root / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
