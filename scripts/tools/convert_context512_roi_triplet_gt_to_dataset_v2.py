#!/usr/bin/env python3
"""Convert pre-cut context512 ROI triplets and per-group GT JSON to Dataset V2.

Expected source records contain three aligned images in this order:

1. clean BEV context image
2. pose image (``*_pose.png``)
3. raw-lane image (``*_raw_lane.png``)

The converted Dataset V2 records use the model input order clean BEV,
raw-lane, pose. Source GT coordinates are ROI-local pixels and are converted to
norm1000 by the repository's canonical SFT record builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, deque
from pathlib import Path, PurePosixPath

from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import classify_row
from data_process.state_update_dataset_common import (
    ALLOWED_INTERSECTION_TYPES,
    ALLOWED_LANE_TYPES,
    COORD_MODE_NORM1000,
    DEFAULT_COORD_RANGE,
    build_sft_record,
    intersection_type_name,
    lane_type_name,
    semantic_sft_record_counts,
    sort_target_lines,
)
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar


FORMAT_VERSION = "context512_roi256_triplet_gt_to_dataset_v2_v1"
OUTPUT_VARIANT = "context512_roi256_three_image_full"
IMAGE_ROLES = [
    "bev_road_structure",
    "pv_camera_raw_lane",
    "historical_vehicle_trajectory",
]
SPLITS = ("train", "eval", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "eval": "eval",
    "evaluation": "eval",
    "val": "eval",
    "validation": "eval",
    "test": "test",
    "testing": "test",
}
ID_GRID_RE = re.compile(r"_r(?P<row>\d+)_c(?P<col>\d+)_p(?P<patch>\d+)$", re.IGNORECASE)
SOURCE_GROUP_DIR_RE = re.compile(r"^A\d+_", re.IGNORECASE)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="Source dataset root. An images/{train,eval,test} tree is detected automatically.",
    )
    parser.add_argument(
        "--annotation-root",
        default="",
        help="GT JSON root; defaults to --input-root and is searched recursively.",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Triplet image root; defaults to <input-root>/images when present.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--annotation-glob", default="*.json")
    parser.add_argument("--default-split", choices=SPLITS, default="train")
    parser.add_argument("--target-patch-size", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--boundary-tolerance", type=float, default=0.5)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--image-check-mode",
        choices=("sampled", "all", "none"),
        default="sampled",
    )
    parser.add_argument("--image-check-limit", type=int, default=10_000)
    parser.add_argument(
        "--non-512-policy",
        choices=("skip", "pad", "error"),
        default="skip",
        help="How to handle boundary-clipped triplets. Production default: skip.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--package-path", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_item(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_split(value: object) -> str | None:
    text = str(value or "").strip().lower()
    return SPLIT_ALIASES.get(text)


def split_hint_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        normalized = normalize_split(part)
        if normalized:
            return normalized
    for token in re.split(r"[^a-zA-Z]+", path.stem.lower()):
        normalized = normalize_split(token)
        if normalized:
            return normalized
    return None


def load_annotation_records(path: Path) -> list[dict] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON annotation file: {path}: {exc}") from exc
    if isinstance(payload, dict):
        for key in ("samples", "records", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(item, dict) for item in payload):
        return None
    if not any("GT" in item and "image" in item for item in payload):
        return None
    if not all("GT" in item and "image" in item for item in payload):
        raise ValueError(f"annotation file mixes GT records and other objects: {path}")
    return payload


def discover_annotation_files(annotation_root: Path, pattern: str, output_root: Path) -> list[Path]:
    files = []
    output_resolved = output_root.resolve()
    for path in sorted(annotation_root.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        files.append(path)
    if not files:
        raise FileNotFoundError(
            f"no JSON files matching {pattern!r} found under {annotation_root}"
        )
    return files


def parse_embedded_list(value: object, field: str, sample_id: str) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"sample={sample_id} has invalid embedded {field} JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"sample={sample_id} {field} must decode to a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"sample={sample_id} {field} contains non-object entries")
    return value


def flatten_gt(record: dict) -> dict:
    gt = record.get("GT")
    if isinstance(gt, dict):
        return dict(gt)
    if not isinstance(gt, list):
        raise ValueError(f"sample={record.get('id')} GT must be a list or object")
    result = {}
    for item in gt:
        if not isinstance(item, dict):
            raise ValueError(f"sample={record.get('id')} GT contains a non-object entry")
        result.update(item)
    return result


def normalize_pixel_points(
    coords: object,
    patch_size: int,
    sample_id: str,
    label: str,
) -> list[list[int]]:
    if not isinstance(coords, list):
        raise ValueError(f"sample={sample_id} {label}.coords must be a list")
    points = []
    for point in coords:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"sample={sample_id} {label} contains invalid point={point!r}")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"sample={sample_id} {label} contains non-finite point={point!r}")
        if x < 0 or y < 0 or x > patch_size - 1 or y > patch_size - 1:
            raise ValueError(
                f"sample={sample_id} {label} point={point!r} is outside [0,{patch_size - 1}]"
            )
        normalized = [int(round(x)), int(round(y))]
        if not points or normalized != points[-1]:
            points.append(normalized)
    return points


def point_is_cut(point: list[int], patch_size: int, tolerance: float) -> bool:
    limit = patch_size - 1
    return (
        point[0] <= tolerance
        or point[1] <= tolerance
        or point[0] >= limit - tolerance
        or point[1] >= limit - tolerance
    )


def parse_intersection_category(value: object) -> tuple[str, tuple[object, object]]:
    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))]
    if len(numbers) >= 2:
        properties = {
            "IntersectionType": numbers[0],
            "IntersectionSubType": numbers[1],
        }
    elif numbers:
        properties = {"IntersectionType": numbers[0]}
        if numbers[0] == 1:
            properties["IntersectionSubType"] = 1
    else:
        properties = {}
    return intersection_type_name(properties)


def convert_gt(record: dict, boundary_tolerance: float) -> tuple[list[dict], dict]:
    sample_id = str(record.get("id") or "").strip()
    if not sample_id:
        raise ValueError("GT record has no id")
    gt = flatten_gt(record)
    patch_size = int(gt.get("patch_size", 0))
    if patch_size <= 1:
        raise ValueError(f"sample={sample_id} has invalid patch_size={patch_size}")
    lanes = parse_embedded_list(gt.get("lane"), "lane", sample_id)
    intersections = parse_embedded_list(gt.get("intersection"), "intersection", sample_id)
    lines = []
    stats = Counter()
    for index, lane in enumerate(lanes):
        lane_type = lane_type_name(lane.get("category"))
        if lane_type is None:
            stats[f"ignored_lane_type:{lane.get('category')}"] += 1
            continue
        points = normalize_pixel_points(
            lane.get("coords"),
            patch_size,
            sample_id,
            f"lane[{index}]",
        )
        if len(set(map(tuple, points))) < 2:
            stats["dropped_degenerate_lane"] += 1
            continue
        lines.append({
            "category": "centerline",
            "lane_type": lane_type,
            "start_type": "cut" if point_is_cut(points[0], patch_size, boundary_tolerance) else "inside",
            "end_type": "cut" if point_is_cut(points[-1], patch_size, boundary_tolerance) else "inside",
            "points": points,
        })
        stats[f"lane_type:{lane_type}"] += 1
    for index, intersection in enumerate(intersections):
        points = normalize_pixel_points(
            intersection.get("coords"),
            patch_size,
            sample_id,
            f"intersection[{index}]",
        )
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(set(map(tuple, points))) < 3:
            stats["dropped_degenerate_intersection"] += 1
            continue
        is_cut = any(point_is_cut(point, patch_size, boundary_tolerance) for point in points)
        points.append(list(points[0]))
        target_type, source_pair = parse_intersection_category(intersection.get("category"))
        lines.append({
            "category": "intersection",
            "intersection_type": target_type,
            "is_cut": is_cut,
            "points": points,
        })
        stats[f"intersection_type:{target_type}"] += 1
        stats[f"source_intersection_type:{source_pair[0]}_{source_pair[1]}"] += 1
    return sort_target_lines(lines, patch_size, boundary_tolerance), {
        "patch_size": patch_size,
        "stats": dict(stats),
    }


def source_triplet_relatives(record: dict) -> tuple[str, str, str]:
    images = record.get("image")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(f"sample={record.get('id')} image must contain exactly three paths")
    paths = [str(item).replace("\\", "/") for item in images]
    raw_lane = [path for path in paths if Path(path).stem.lower().endswith("_raw_lane")]
    pose = [path for path in paths if Path(path).stem.lower().endswith("_pose")]
    primary = [path for path in paths if path not in raw_lane and path not in pose]
    if len(primary) != 1 or len(raw_lane) != 1 or len(pose) != 1:
        raise ValueError(
            f"sample={record.get('id')} cannot identify BEV/pose/raw-lane paths: {paths!r}"
        )
    return primary[0], raw_lane[0], pose[0]


def candidate_source_paths(
    relative: str,
    split: str,
    image_roots: list[Path],
    annotation_path: Path,
) -> list[Path]:
    rel = Path(*PurePosixPath(relative).parts)
    candidates = []
    for root in image_roots:
        candidates.extend((root / split / rel, root / rel))
    candidates.extend((annotation_path.parent / rel, annotation_path.parent / split / rel))
    unique = []
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def source_group_from_relative(relative: str) -> tuple[str, tuple[str, ...]] | None:
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        if SOURCE_GROUP_DIR_RE.match(part):
            return part, tuple(parts[index + 1:])
    return None


def build_group_directory_index(search_root: Path) -> dict[str, list[Path]]:
    """Index extracted per-group directories once instead of recursively per record."""
    started = time.perf_counter()
    index: dict[str, list[Path]] = {}
    scanned_directories = 0
    for directory, child_names, _file_names in os.walk(search_root):
        scanned_directories += 1
        base = Path(directory)
        kept_children = []
        for child_name in child_names:
            child = base / child_name
            if SOURCE_GROUP_DIR_RE.match(child_name):
                index.setdefault(child_name, []).append(child)
            else:
                kept_children.append(child_name)
        child_names[:] = kept_children
    for directories in index.values():
        directories.sort(key=lambda path: os.path.normcase(str(path)))
    print(
        f"[triplet-gt-convert] indexed image groups={len(index)} "
        f"scanned_directories={scanned_directories} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return index


def resolve_triplet_from_group_index(
    record: dict,
    group_index: dict[str, list[Path]],
    split_hint: str | None,
    default_split: str,
) -> tuple[str, tuple[Path, Path, Path]] | None:
    relatives = source_triplet_relatives(record)
    parsed = [source_group_from_relative(relative) for relative in relatives]
    if any(item is None for item in parsed):
        return None
    groups = {item[0] for item in parsed if item is not None}
    if len(groups) != 1:
        raise ValueError(
            f"sample={record.get('id')} triplet paths use different source groups: {relatives!r}"
        )
    group = next(iter(groups))
    matches = []
    for group_dir in group_index.get(group, []):
        paths = tuple(
            group_dir.joinpath(*item[1])
            for item in parsed
            if item is not None
        )
        if len(paths) == 3 and all(path.is_file() for path in paths):
            split = (
                normalize_split(record.get("split"))
                or split_hint_from_path(group_dir)
                or split_hint
                or default_split
            )
            matches.append((split, paths))
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"[triplet-gt-convert] WARNING duplicate extracted group={group}; "
            f"using first of {len(matches)} complete copies",
            flush=True,
        )
    return matches[0]


def path_ends_with(path: Path, relative: str) -> bool:
    relative_parts = tuple(part.lower() for part in PurePosixPath(relative).parts)
    path_parts = tuple(part.lower() for part in path.parts)
    return len(path_parts) >= len(relative_parts) and path_parts[-len(relative_parts):] == relative_parts


def discover_triplet_root(
    record: dict,
    search_root: Path,
    split_hint: str | None,
    default_split: str,
) -> tuple[str, tuple[Path, Path, Path], Path] | None:
    relatives = source_triplet_relatives(record)
    print(
        f"[triplet-gt-convert] locate nested image root for sample={record.get('id')} "
        f"under {search_root}",
        flush=True,
    )
    queue = deque([(search_root, 0)])
    visited = set()
    while queue:
        base, depth = queue.popleft()
        resolved = str(base.resolve())
        if resolved in visited:
            continue
        visited.add(resolved)
        candidate_roots = [(base, split_hint_from_path(base))]
        candidate_roots.extend((base / split, split) for split in SPLITS)
        for root, inferred_split in candidate_roots:
            paths = tuple(root / Path(*PurePosixPath(relative).parts) for relative in relatives)
            if all(path.is_file() for path in paths):
                inferred_split = inferred_split or split_hint or default_split
                print(
                    f"[triplet-gt-convert] discovered image root={root} split={inferred_split}",
                    flush=True,
                )
                return inferred_split, paths, root
        if depth >= 12 or normalize_split(base.name):
            continue
        try:
            children = [child for child in base.iterdir() if child.is_dir()]
        except OSError:
            continue
        for child in children:
            lowered = child.name.lower()
            if lowered == "gt_json" or SOURCE_GROUP_DIR_RE.match(child.name):
                continue
            queue.append((child, depth + 1))
    return None


def resolve_triplet(
    record: dict,
    annotation_path: Path,
    image_roots: list[Path],
    recursive_search_root: Path,
    split_hint: str | None,
    default_split: str,
    group_index: dict[str, list[Path]] | None = None,
) -> tuple[str, tuple[Path, Path, Path]]:
    if group_index is not None:
        indexed = resolve_triplet_from_group_index(
            record,
            group_index,
            split_hint,
            default_split,
        )
        if indexed is not None:
            return indexed
    primary_rel, raw_lane_rel, pose_rel = source_triplet_relatives(record)
    record_split = normalize_split(record.get("split"))
    preferred = []
    for split in (record_split, split_hint, default_split, *SPLITS):
        if split and split not in preferred:
            preferred.append(split)
    matches = []
    for split in preferred:
        role_matches = []
        for relative in (primary_rel, raw_lane_rel, pose_rel):
            existing = [
                path
                for path in candidate_source_paths(relative, split, image_roots, annotation_path)
                if path.is_file()
            ]
            if not existing:
                role_matches = []
                break
            role_matches.append(existing[0])
        if role_matches:
            matches.append((split, tuple(role_matches)))
            if split in {record_split, split_hint}:
                break
    if not matches:
        discovered = discover_triplet_root(
            record,
            recursive_search_root,
            split_hint,
            default_split,
        )
        if discovered is not None:
            split, paths, discovered_root = discovered
            if discovered_root not in image_roots:
                image_roots.append(discovered_root)
            return split, paths
        raise FileNotFoundError(
            f"sample={record.get('id')} cannot resolve image triplet from "
            f"{source_triplet_relatives(record)!r}; roots={image_roots}; "
            f"recursive_search_root={recursive_search_root}"
        )
    return matches[0]


def output_relatives(sample_id: str, split: str, source_primary: Path) -> tuple[str, str, str]:
    group = sample_id.rsplit("_r", 1)[0] if "_r" in sample_id else source_primary.parent.name
    suffix = source_primary.suffix.lower() or ".png"
    filename = f"{sample_id}{suffix}"
    return (
        (PurePosixPath("images") / split / group / filename).as_posix(),
        (PurePosixPath("raw_lane_images") / split / group / filename).as_posix(),
        (PurePosixPath("pose_images") / split / group / filename).as_posix(),
    )


def materialize_file(source: Path, destination: Path, mode: str, resume: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if not resume:
            raise FileExistsError(f"destination already exists: {destination}")
        try:
            if os.path.samefile(source, destination):
                return "reused_hardlink"
        except OSError:
            pass
        if (
            destination.stat().st_size == source.stat().st_size
            and sha256_file(destination) == sha256_file(source)
        ):
            return "reused_identical"
        destination.unlink()
        replacement = True
    else:
        replacement = False
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    used_mode = mode
    try:
        if mode == "hardlink":
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
                used_mode = "copy_fallback"
        else:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return f"replaced_{used_mode}" if replacement else used_mode


def inspect_triplet_sizes(paths: tuple[Path, Path, Path], sample_id: str) -> tuple[int, int]:
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    if len(set(sizes)) != 1:
        raise ValueError(f"sample={sample_id} triplet image sizes differ: {sizes}")
    return sizes[0]


def validate_source_image_size(
    size: tuple[int, int],
    target_size: int,
    context_size: int,
    sample_id: str,
) -> None:
    width, height = size
    if (
        width < target_size
        or height < target_size
        or width > context_size
        or height > context_size
    ):
        raise ValueError(
            f"sample={sample_id} source images are {size}; each dimension must be "
            f"within [{target_size},{context_size}] before context padding"
        )


def context_padding(
    sample_id: str,
    source_size: tuple[int, int],
    target_size: int,
    context_size: int,
) -> tuple[int, int, int, int]:
    width, height = source_size
    if source_size == (context_size, context_size):
        return 0, 0, 0, 0
    match = ID_GRID_RE.search(sample_id)
    if not match:
        raise ValueError(
            f"sample={sample_id} needs context padding from {source_size}, but its id "
            "does not contain _rN_cN_pN grid coordinates"
        )
    row = int(match.group("row"))
    col = int(match.group("col"))
    margin = (context_size - target_size) // 2
    if width == context_size:
        left = right = 0
    else:
        target_x_in_source = 0 if col == 0 else min(margin, width - target_size)
        left = margin - target_x_in_source
        right = context_size - width - left
    if height == context_size:
        top = bottom = 0
    else:
        target_y_in_source = 0 if row == 0 else min(margin, height - target_size)
        top = margin - target_y_in_source
        bottom = context_size - height - top
    if min(left, top, right, bottom) < 0:
        raise ValueError(
            f"sample={sample_id} cannot align source_size={source_size} to centered "
            f"ROI using padding={(left, top, right, bottom)}"
        )
    return left, top, right, bottom


def materialize_padded_image(
    source: Path,
    destination: Path,
    padding: tuple[int, int, int, int],
    context_size: int,
    resume: bool,
) -> str:
    left, top, _, _ = padding
    with Image.open(source) as opened:
        image = opened.copy()
    canvas = Image.new(image.mode, (context_size, context_size), 0)
    canvas.paste(image, (left, top))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if not resume:
            raise FileExistsError(f"destination already exists: {destination}")
        try:
            with Image.open(destination) as existing:
                identical = (
                    existing.size == canvas.size
                    and existing.mode == canvas.mode
                    and ImageChops.difference(existing, canvas).getbbox() is None
                )
        except (OSError, ValueError):
            identical = False
        if identical:
            return "reused_padded"
        destination.unlink()
        status = "replaced_padded"
    else:
        status = "padded"
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        canvas.save(temporary, format="PNG")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return status


def grid_metadata(sample_id: str, patch_size: int) -> dict:
    match = ID_GRID_RE.search(sample_id)
    if not match:
        return {}
    row = int(match.group("row"))
    col = int(match.group("col"))
    patch_index = int(match.group("patch"))
    x0, y0 = col * patch_size, row * patch_size
    return {
        "patch_row": row,
        "patch_col": col,
        "row": row,
        "col": col,
        "patch_index": patch_index,
        "x0": x0,
        "y0": y0,
        "target_box_full": [x0, y0, x0 + patch_size, y0 + patch_size],
    }


def open_writers(output_root: Path):
    phase_root = output_root / "phase_a"
    phase_root.mkdir(parents=True, exist_ok=True)
    records = {
        split: (phase_root / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in SPLITS
    }
    metadata = {
        split: (phase_root / f"meta_{split}.jsonl").open("w", encoding="utf-8")
        for split in SPLITS
    }
    return records, metadata


def close_writers(*groups) -> None:
    for group in groups:
        for handle in group.values():
            handle.close()


def prepare_output_root(output_root: Path, resume: bool) -> None:
    if output_root.exists() and not resume and any(output_root.iterdir()):
        raise FileExistsError(
            f"output root is not empty: {output_root}. Use --resume or a new --output-root."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def convert_dataset(args: argparse.Namespace) -> dict:
    input_root = Path(args.input_root).expanduser().resolve()
    annotation_root = Path(args.annotation_root).expanduser().resolve() if args.annotation_root else input_root
    if args.image_root:
        image_root = Path(args.image_root).expanduser().resolve()
    elif (input_root / "images").is_dir():
        image_root = input_root / "images"
    else:
        image_root = input_root
    output_root = Path(args.output_root).expanduser().resolve()
    if args.target_patch_size <= 1 or args.context_size < args.target_patch_size:
        raise ValueError("invalid --target-patch-size/--context-size")
    if (args.context_size - args.target_patch_size) % 2:
        raise ValueError("context and target sizes must have an even centered-ROI margin")
    if args.coord_range != DEFAULT_COORD_RANGE:
        raise ValueError(f"--coord-range must be {DEFAULT_COORD_RANGE}")
    prepare_output_root(output_root, args.resume)
    annotation_files = discover_annotation_files(
        annotation_root,
        args.annotation_glob,
        output_root,
    )
    image_roots = list(dict.fromkeys((image_root, input_root / "images", input_root)))
    group_index = build_group_directory_index(input_root)
    record_writers, meta_writers = open_writers(output_root)
    skipped_writer = (output_root / "skipped_samples.jsonl").open("w", encoding="utf-8")
    counts = Counter()
    semantic_counts = Counter()
    conversion_counts = Counter()
    seen_ids = set()
    image_checks = 0
    stopped = False
    try:
        for annotation_path in annotation_files:
            records = load_annotation_records(annotation_path)
            if records is None:
                conversion_counts["ignored_non_gt_json_file"] += 1
                continue
            annotation_split = split_hint_from_path(annotation_path)
            for source_record in records:
                if args.max_samples > 0 and counts["total"] >= args.max_samples:
                    stopped = True
                    break
                sample_id = str(source_record.get("id") or "").strip()
                if not sample_id:
                    raise ValueError(f"record in {annotation_path} has no id")
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate sample id={sample_id}")
                seen_ids.add(sample_id)
                counts["processed"] += 1
                target_lines, gt_info = convert_gt(source_record, args.boundary_tolerance)
                if gt_info["patch_size"] != args.target_patch_size:
                    raise ValueError(
                        f"sample={sample_id} GT patch_size={gt_info['patch_size']}, "
                        f"expected={args.target_patch_size}"
                    )
                split, source_paths = resolve_triplet(
                    source_record,
                    annotation_path,
                    image_roots,
                    input_root,
                    annotation_split,
                    args.default_split,
                    group_index,
                )
                should_check = args.image_check_mode == "all" or (
                    args.image_check_mode == "sampled"
                    and image_checks < args.image_check_limit
                )
                with Image.open(source_paths[0]) as primary_image:
                    source_size = primary_image.size
                if source_size != (args.context_size, args.context_size) and args.non_512_policy == "skip":
                    conversion_counts["skipped_non_512_sample"] += 1
                    conversion_counts[f"skipped_source_size:{source_size[0]}x{source_size[1]}"] += 1
                    write_jsonl_item(skipped_writer, {
                        "id": sample_id,
                        "reason": "non_512_context_image",
                        "source_image_size": list(source_size),
                        "expected_image_size": [args.context_size, args.context_size],
                        "source_image_paths": [str(path) for path in source_paths],
                        "annotation": str(annotation_path),
                    })
                    skipped = conversion_counts["skipped_non_512_sample"]
                    if skipped <= 20 or (
                        args.progress_every > 0 and skipped % args.progress_every == 0
                    ):
                        print(
                            f"[triplet-gt-convert] skip non-512 sample={sample_id} "
                            f"size={source_size} skipped={skipped}",
                            flush=True,
                        )
                    continue
                validate_source_image_size(
                    source_size,
                    args.target_patch_size,
                    args.context_size,
                    sample_id,
                )
                needs_padding = source_size != (args.context_size, args.context_size)
                if needs_padding and args.non_512_policy == "error":
                    raise ValueError(
                        f"sample={sample_id} source images are {source_size}, expected "
                        f"{args.context_size}x{args.context_size}; use "
                        "--non-512-policy skip or pad"
                    )
                if should_check or needs_padding:
                    source_size = inspect_triplet_sizes(source_paths, sample_id)
                    validate_source_image_size(
                        source_size,
                        args.target_patch_size,
                        args.context_size,
                        sample_id,
                    )
                    image_checks += 1
                padding = context_padding(
                    sample_id,
                    source_size,
                    args.target_patch_size,
                    args.context_size,
                )
                primary_rel, raw_lane_rel, pose_rel = output_relatives(
                    sample_id,
                    split,
                    source_paths[0],
                )
                output_paths = (
                    output_root / primary_rel,
                    output_root / raw_lane_rel,
                    output_root / pose_rel,
                )
                for source, destination in zip(source_paths, output_paths):
                    if needs_padding:
                        status = materialize_padded_image(
                            source,
                            destination,
                            padding,
                            args.context_size,
                            args.resume,
                        )
                    else:
                        status = materialize_file(
                            source,
                            destination,
                            args.copy_mode,
                            args.resume,
                        )
                    conversion_counts[status] += 1
                if needs_padding:
                    conversion_counts["padded_sample"] += 1
                tile_id = sample_id.rsplit("_r", 1)[0] if "_r" in sample_id else source_paths[0].parent.name
                margin = (args.context_size - args.target_patch_size) // 2
                meta = {
                    "tile_id": tile_id,
                    "log_id": tile_id,
                    **grid_metadata(sample_id, args.target_patch_size),
                    "utm_zone": source_record.get("utm_zone"),
                    "source_annotation": str(annotation_path),
                    "source_image_paths": [str(path) for path in source_paths],
                    "source_image_size": list(source_size),
                    "source_schema": FORMAT_VERSION,
                    "context_box_semantics": "centered_target_roi",
                    "context_image_size": args.context_size,
                    "context_padding_ltrb": list(padding),
                    "target_roi_in_image": [
                        margin,
                        margin,
                        margin + args.target_patch_size,
                        margin + args.target_patch_size,
                    ],
                    "raw_lane_overlay": False,
                }
                row = {
                    "id": sample_id,
                    "image": primary_rel,
                    "raw_lane_image": raw_lane_rel,
                    "pose_image": pose_rel,
                    "incoming_traces": [],
                    "incoming_intersections": [],
                    "target_lines": target_lines,
                    "meta": meta,
                }
                difficulty = classify_row(
                    row,
                    patch_size=args.target_patch_size,
                    coord_range=args.coord_range,
                )
                sft = build_sft_record(
                    row,
                    args.target_patch_size,
                    True,
                    "a",
                    coord_mode=COORD_MODE_NORM1000,
                    coord_range=args.coord_range,
                    context_size=args.context_size,
                    view_mode="context512_roi256_three_image",
                    raw_lane_overlay=False,
                    pose_second_image=True,
                    save_raw_lane_image=True,
                    raw_lane_separate_image=True,
                )
                sft["pose_image"] = pose_rel
                sft["meta"].update({
                    "difficulty": difficulty["difficulty"],
                    "difficulty_score": difficulty["difficulty_score"],
                    "stratum": difficulty["stratum"],
                    "has_intersection": difficulty["has_intersection"],
                })
                semantic_counts.update(semantic_sft_record_counts(
                    sft,
                    strict=True,
                    require_prompt=True,
                ))
                write_jsonl_item(record_writers[split], sft)
                write_jsonl_item(meta_writers[split], {
                    "id": sample_id,
                    "image": primary_rel,
                    "images": list(sft["images"]),
                    "raw_lane_image": raw_lane_rel,
                    "pose_image": pose_rel,
                    "difficulty": difficulty["difficulty"],
                    "difficulty_score": difficulty["difficulty_score"],
                    "stratum": difficulty["stratum"],
                    "has_intersection": difficulty["has_intersection"],
                    "meta": sft["meta"],
                })
                counts[split] += 1
                counts["total"] += 1
                counts[f"difficulty:{difficulty['stratum']}"] += 1
                counts[f"intersection:{bool(difficulty['has_intersection'])}"] += 1
                conversion_counts.update(gt_info["stats"])
                if args.progress_every > 0 and counts["total"] % args.progress_every == 0:
                    print(
                        f"[triplet-gt-convert] samples={counts['total']} "
                        f"train={counts['train']} eval={counts['eval']} test={counts['test']}",
                        flush=True,
                    )
            if stopped:
                break
    finally:
        close_writers(record_writers, meta_writers)
        skipped_writer.close()
    if counts["total"] <= 0:
        raise ValueError("conversion produced no records")
    return {
        "input_root": str(input_root),
        "annotation_root": str(annotation_root),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "annotation_file_count": len(annotation_files),
        "record_counts": {split: counts[split] for split in SPLITS},
        "total_records": counts["total"],
        "processed_source_records": counts["processed"],
        "skipped_non_512_records": conversion_counts["skipped_non_512_sample"],
        "difficulty_counts": {
            key.split(":", 1)[1]: value
            for key, value in counts.items()
            if key.startswith("difficulty:")
        },
        "intersection_presence_counts": {
            key.split(":", 1)[1]: value
            for key, value in counts.items()
            if key.startswith("intersection:")
        },
        "semantic_counts": dict(semantic_counts),
        "conversion_counts": dict(conversion_counts),
        "image_headers_checked": image_checks,
    }


def write_dataset_metadata(args: argparse.Namespace, summary: dict) -> None:
    output_root = Path(summary["output_root"])
    split_files = {
        split: output_root / "phase_a" / f"{split}.jsonl"
        for split in SPLITS
    }
    split_manifest = {
        "format_version": FORMAT_VERSION,
        "split_policy": "preserve_source_path_or_record_split_else_default",
        "default_split": args.default_split,
        "counts": summary["record_counts"],
        "jsonl_sha256": {
            split: sha256_file(path)
            for split, path in split_files.items()
        },
    }
    dataset_info = {
        "dataset_version": FORMAT_VERSION,
        "variant": OUTPUT_VARIANT,
        "coord_mode": COORD_MODE_NORM1000,
        "coord_range": args.coord_range,
        "target_patch_size": args.target_patch_size,
        "context_size": args.context_size,
        "target_roi_in_image": [
            (args.context_size - args.target_patch_size) // 2,
            (args.context_size - args.target_patch_size) // 2,
            (args.context_size + args.target_patch_size) // 2,
            (args.context_size + args.target_patch_size) // 2,
        ],
        "source_gt_coordinate_system": f"pixel_0_{args.target_patch_size - 1}_relative_to_target_roi",
        "non_512_policy": args.non_512_policy,
        "context_padding_policy": {
            "enabled": args.non_512_policy == "pad",
            "canvas_size": [args.context_size, args.context_size],
            "fill": "black",
            "alignment": "target_roi_centered_using_sample_row_col",
            "target_roi": [
                (args.context_size - args.target_patch_size) // 2,
                (args.context_size - args.target_patch_size) // 2,
                (args.context_size + args.target_patch_size) // 2,
                (args.context_size + args.target_patch_size) // 2,
            ],
        },
        "multi_image_input": {
            "enabled": True,
            "num_images_per_sample": 3,
            "image_roles": list(IMAGE_ROLES),
            "image_order": list(IMAGE_ROLES),
            "source_image_order": [
                "bev_road_structure",
                "historical_vehicle_trajectory",
                "pv_camera_raw_lane",
            ],
        },
        "input_overlay": {
            "raw_lane_overlay": False,
            "raw_lane_separate_image": True,
        },
        "lane_types": sorted(ALLOWED_LANE_TYPES),
        "ignored_source_lane_type_codes": [3, 22],
        "intersection_types": sorted(ALLOWED_INTERSECTION_TYPES),
        "phase": "phase_a",
        "record_counts": summary["record_counts"],
        "difficulty_counts": summary["difficulty_counts"],
        "semantic_counts": summary["semantic_counts"],
    }
    write_json(output_root / "dataset_info.json", dataset_info)
    write_json(output_root / "split_manifest.json", split_manifest)
    write_json(output_root / "semantic_schema_report.json", {
        "status": "passed",
        "lane_types": sorted(ALLOWED_LANE_TYPES),
        "ignored_source_lane_type_codes": [3, 22],
        "intersection_types": sorted(ALLOWED_INTERSECTION_TYPES),
        "semantic_counts": summary["semantic_counts"],
        "conversion_counts": summary["conversion_counts"],
    })
    write_json(output_root / "build_summary.json", {"status": "passed", **summary})


def validate_output(args: argparse.Namespace, summary: dict) -> dict:
    root = Path(summary["output_root"])
    counts = Counter()
    for split in SPLITS:
        path = root / "phase_a" / f"{split}.jsonl"
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                images = record.get("images")
                if not isinstance(images, list) or len(images) != 3:
                    raise ValueError(f"{path}:{line_number} does not contain exactly three images")
                expected_prefixes = (
                    f"images/{split}/",
                    f"raw_lane_images/{split}/",
                    f"pose_images/{split}/",
                )
                for relative, prefix in zip(images, expected_prefixes):
                    if not str(relative).startswith(prefix):
                        raise ValueError(f"{path}:{line_number} invalid image order={images!r}")
                    if not (root / str(relative)).is_file():
                        raise FileNotFoundError(f"{path}:{line_number} missing image={root / str(relative)}")
                prompt = str((record.get("conversations") or [{}])[0].get("value", ""))
                if prompt.count("<image>") != 3:
                    raise ValueError(f"{path}:{line_number} prompt does not contain three image tokens")
                target = json.loads(record["conversations"][1]["value"])
                if not isinstance(target.get("lines"), list):
                    raise ValueError(f"{path}:{line_number} target has no lines list")
                for item in target["lines"]:
                    for point in item.get("points") or []:
                        if not (0 <= int(point[0]) <= args.coord_range and 0 <= int(point[1]) <= args.coord_range):
                            raise ValueError(f"{path}:{line_number} point outside norm1000: {point}")
                counts[split] += 1
    if {split: counts[split] for split in SPLITS} != summary["record_counts"]:
        raise ValueError(
            f"validated counts={dict(counts)} differ from build counts={summary['record_counts']}"
        )
    report = {
        "status": "passed",
        "dataset_root": str(root),
        "record_counts": {split: counts[split] for split in SPLITS},
        "image_order": list(IMAGE_ROLES),
        "coord_mode": COORD_MODE_NORM1000,
        "coord_range": args.coord_range,
        "target_patch_size": args.target_patch_size,
        "context_size": args.context_size,
    }
    write_json(root / "conversion_validation.json", report)
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = convert_dataset(args)
    write_dataset_metadata(args, summary)
    validation = validate_output(args, summary)
    package_path = ""
    if args.package:
        output_root = Path(summary["output_root"])
        package = (
            Path(args.package_path).expanduser().resolve()
            if args.package_path
            else output_root.parent / f"{output_root.name}.tar"
        )
        create_variant_tar(output_root, package, args.resume)
        package_path = str(package)
    result = {
        "status": "passed",
        "summary": summary,
        "validation": validation,
        "package": package_path,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
