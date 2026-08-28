#!/usr/bin/env python3
"""Build context512/ROI256 three-image ablation datasets on Windows.

Three modes are provided:

``nonoverlap``
    Keep only train targets whose source-grid origin satisfies
    ``x0 % 512 == 0`` and ``y0 % 512 == 0``.  Their 512x512 context windows
    are disjoint (up to a shared border), while eval/test are copied unchanged.

``rotation``
    Copy the original train rows and add rotated train rows.  The 512x512
    clean BEV, RawLane, and Pose images are rotated together.  Labels are
    transformed from ROI-relative norm1000 coordinates into context pixels,
    rotated around the context center, clipped back to the central 256x256
    ROI, and converted to ROI-relative norm1000 coordinates.  This is a local
    augmentation: the rotated row is not a physically valid crop at the same
    position in the original large map, so its global x0/y0 metadata is marked
    invalid and it must not be used for whole-map stitching or evaluation.

``neighbor_rotation``
    Keep one row for each stride-256 train grid location and replace the
    input/target at that location with a deterministic local rotation.  When
    the source was built with a finer grid (for example stride 128), the
    default policy filters to rows aligned to the 256 grid before replacing
    them.
    The grid phase ``(grid_x + grid_y) % 3`` selects 0, 45, or 135 degrees,
    so horizontal and vertical neighbors do not use the same orientation.
    With only three phases, one diagonal direction can share a phase; this is
    a deliberate compact schedule.  The three images and the ROI labels are
    transformed together.  Unlike ``rotation``, this mode does not add rows.
    It preserves the train count when the input is already on the 256 grid;
    with a finer source grid, the default filter intentionally reduces the
    count to the aligned rows.

The input must already be a completed three-image context512/ROI256 Dataset V2
with ``phase_a/{train,eval,test}.jsonl`` and matching ``meta_*.jsonl`` files.
No OBS download or source archive extraction is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tarfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from PIL import Image

try:
    from shapely.geometry import GeometryCollection, LineString, MultiLineString
    from shapely.geometry import MultiPolygon, Polygon, box
except ModuleNotFoundError:  # pragma: no cover - reported only for rotation modes.
    GeometryCollection = LineString = MultiLineString = None
    MultiPolygon = Polygon = box = None


SPLITS = ("train", "eval", "test")
CONTEXT_SIZE = 512
TARGET_SIZE = 256
ROI_OFFSET = (CONTEXT_SIZE - TARGET_SIZE) // 2
ROI_BOX = (
    float(ROI_OFFSET),
    float(ROI_OFFSET),
    float(ROI_OFFSET + TARGET_SIZE - 1),
    float(ROI_OFFSET + TARGET_SIZE - 1),
)
COORD_RANGE = 1000
IMAGE_PREFIXES = ("images/", "raw_lane_images/", "pose_images/")
ROTATION_CONTRACT = "context512_roi256_three_image_rotation_v1"
NONOVERLAP_CONTRACT = "context512_roi256_three_image_nonoverlap_v1"
NEIGHBOR_ROTATION_CONTRACT = "context512_roi256_three_image_neighbor_rotation_v1"
IMAGE_ROLES = (
    "bev_road_structure",
    "pv_camera_raw_lane",
    "historical_vehicle_trajectory",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("nonoverlap", "rotation", "neighbor_rotation"),
        required=True,
    )
    parser.add_argument("--input-root", required=True, help="Completed three-image context dataset root.")
    parser.add_argument("--output-root", required=True, help="New ablation dataset root.")
    parser.add_argument("--angles", default="45,135", help="Rotation angles for rotation mode.")
    parser.add_argument(
        "--neighbor-angles",
        default="0,45,135",
        help=(
            "Three angles selected by (grid_x + grid_y) %% 3 in neighbor_rotation mode."
        ),
    )
    parser.add_argument(
        "--neighbor-grid-stride",
        type=int,
        default=256,
        help="Source train grid stride required by neighbor_rotation mode (default: 256).",
    )
    parser.add_argument(
        "--neighbor-source-grid-policy",
        choices=("filter", "require"),
        default="filter",
        help=(
            "How neighbor_rotation handles source train rows not aligned to the 256 grid: "
            "filter them (default) or fail immediately."
        ),
    )
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--image-resample", choices=("nearest", "bilinear", "bicubic"), default="bilinear")
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=4)
    parser.add_argument("--package-path", default="")
    parser.add_argument("--validation-sample-limit", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--rotate-empty", action="store_true", help="Also rotate empty train rows.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
                yield line_number, payload


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def safe_relative_path(value: Any) -> Path:
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"image path must be a non-empty string: {value!r}")
    normalized = value.replace("\\", "/").lstrip("/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise ValueError(f"unsafe relative image path: {value!r}")
    return Path(*pure.parts)


def phase_path(root: Path, split: str, meta: bool = False) -> Path:
    name = f"meta_{split}.jsonl" if meta else f"{split}.jsonl"
    path = root / "phase_a" / name
    if split == "eval" and not path.is_file() and not meta:
        path = root / "phase_a" / "val.jsonl"
    if split == "eval" and not path.is_file() and meta:
        path = root / "phase_a" / "meta_val.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {split} {'meta' if meta else 'record'} JSONL: {path}")
    return path


def iter_paired_records(root: Path, split: str) -> Iterator[tuple[int, dict, dict]]:
    record_path = phase_path(root, split)
    meta_path = phase_path(root, split, meta=True)
    with (
        record_path.open("r", encoding="utf-8-sig") as record_handle,
        meta_path.open("r", encoding="utf-8-sig") as meta_handle,
    ):
        for line_number, pair in enumerate(zip_longest(record_handle, meta_handle), start=1):
            record_line, meta_line = pair
            if record_line is None or meta_line is None:
                raise ValueError(f"record/meta length mismatch: {record_path}:{line_number}")
            if not record_line.strip() and not meta_line.strip():
                continue
            if not record_line.strip() or not meta_line.strip():
                raise ValueError(f"blank record/meta mismatch: {record_path}:{line_number}")
            record = json.loads(record_line)
            meta = json.loads(meta_line)
            if str(record.get("id")) != str(meta.get("id")):
                raise ValueError(
                    f"record/meta id mismatch at {record_path}:{line_number}: "
                    f"{record.get('id')!r} != {meta.get('id')!r}"
                )
            yield line_number, record, meta


def parse_angles(raw: str) -> list[float]:
    result = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        angle = float(item)
        if not math.isfinite(angle) or abs(angle % 360.0) < 1e-9:
            raise ValueError(f"invalid or zero rotation angle: {item!r}")
        result.append(angle % 360.0)
    if not result:
        raise ValueError("--angles must contain at least one non-zero angle")
    return result


def parse_neighbor_angles(raw: str) -> list[float]:
    """Parse the three deterministic phases used by neighbor replacement."""

    result = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        angle = float(item)
        if not math.isfinite(angle):
            raise ValueError(f"invalid neighbor rotation angle: {item!r}")
        result.append(angle % 360.0)
    if len(result) != 3 or len(set(result)) != 3:
        raise ValueError(
            "--neighbor-angles must contain exactly three distinct angles, "
            "for example 0,45,135"
        )
    if 0.0 not in result:
        raise ValueError("--neighbor-angles must include 0 degrees")
    return result


def angle_label(angle: float) -> str:
    value = int(round(angle * 1000))
    return f"rot{value:06d}"


def record_images(record: dict) -> list[str]:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(f"sample={record.get('id')} must contain exactly three images")
    normalized = [str(value).replace("\\", "/") for value in images]
    for value, prefix in zip(normalized, IMAGE_PREFIXES):
        if not value.startswith(prefix):
            raise ValueError(f"sample={record.get('id')} image order/path mismatch: {normalized!r}")
    if str(record.get("image", "")).replace("\\", "/") != normalized[0]:
        raise ValueError(f"sample={record.get('id')} primary image does not match images[0]")
    return normalized


def validate_input_record(root: Path, record: dict, sample_limit_state: list[int], sample_limit: int) -> None:
    images = record_images(record)
    meta = record.get("meta") or {}
    if int(meta.get("context_image_size", CONTEXT_SIZE)) != CONTEXT_SIZE:
        raise ValueError(f"sample={record.get('id')} is not context512: {meta.get('context_image_size')}")
    if int(meta.get("target_size", TARGET_SIZE)) != TARGET_SIZE:
        raise ValueError(f"sample={record.get('id')} target size is not 256: {meta.get('target_size')}")
    if str(meta.get("coord_mode", "norm1000")) != "norm1000":
        raise ValueError(f"sample={record.get('id')} does not use norm1000 coordinates")
    prompt = str((record.get("conversations") or [{}])[0].get("value", ""))
    if prompt.count("<image>") != 3:
        raise ValueError(f"sample={record.get('id')} prompt does not contain three image tokens")
    for relative in images:
        path = root / safe_relative_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"sample={record.get('id')} missing image: {path}")
        if sample_limit > 0 and sample_limit_state[0] < sample_limit:
            with Image.open(path) as image:
                if image.size != (CONTEXT_SIZE, CONTEXT_SIZE):
                    raise ValueError(
                        f"sample={record.get('id')} image size is {image.size}, "
                        f"expected {(CONTEXT_SIZE, CONTEXT_SIZE)}: {path}"
                    )
            if relative == images[-1]:
                sample_limit_state[0] += 1


def extract_xy(meta: dict) -> tuple[int, int]:
    if "x0" in meta and "y0" in meta:
        return int(meta["x0"]), int(meta["y0"])
    target_box = meta.get("target_box_full")
    if isinstance(target_box, (list, tuple)) and len(target_box) == 4:
        return int(target_box[0]), int(target_box[1])
    context_box = meta.get("context_box_full")
    if isinstance(context_box, (list, tuple)) and len(context_box) == 4:
        return int(context_box[0]) + ROI_OFFSET, int(context_box[1]) + ROI_OFFSET
    raise ValueError(
        "record metadata does not expose source target coordinates; expected x0/y0, "
        f"target_box_full, or context_box_full: {meta}"
    )


def is_strict_512_grid(record: dict) -> bool:
    x0, y0 = extract_xy(record.get("meta") or {})
    return x0 % CONTEXT_SIZE == 0 and y0 % CONTEXT_SIZE == 0


def neighbor_grid_coordinates(meta: dict, grid_stride: int) -> tuple[int, int, int, int]:
    """Return source coordinates, grid coordinates, and the deterministic phase."""

    if grid_stride <= 0:
        raise ValueError(f"neighbor grid stride must be positive: {grid_stride}")
    x0, y0 = extract_xy(meta)
    if x0 % grid_stride or y0 % grid_stride:
        raise ValueError(
            "neighbor_rotation requires train origins aligned to the configured "
            f"stride={grid_stride}: x0={x0}, y0={y0}"
        )
    grid_x, grid_y = x0 // grid_stride, y0 // grid_stride
    return x0, y0, grid_x, grid_y


def is_grid_aligned(meta: dict, grid_stride: int) -> bool:
    """Return whether the source target origin belongs to the requested grid."""

    if grid_stride <= 0:
        raise ValueError(f"grid stride must be positive: {grid_stride}")
    x0, y0 = extract_xy(meta)
    return x0 % grid_stride == 0 and y0 % grid_stride == 0


def neighbor_rotation_assignment(
    meta: dict,
    grid_stride: int,
    angles: list[float],
) -> tuple[float, int, int, int]:
    """Assign a stable angle from spatial grid coordinates.

    The default three-phase pattern gives different phases to immediate
    horizontal and vertical neighbors.  With three phases, one diagonal
    direction can share a phase; keeping this calculation independent of JSONL
    order makes the result reproducible after sharding or filtering the
    dataset.
    """

    _x0, _y0, grid_x, grid_y = neighbor_grid_coordinates(meta, grid_stride)
    phase = (grid_x + grid_y) % len(angles)
    return float(angles[phase]), phase, grid_x, grid_y


def conversation_value(record: dict) -> Any:
    for message in reversed(record.get("conversations") or []):
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        if role in {"gpt", "assistant"}:
            return message.get("value", message.get("content"))
    return None


def parse_target_lines(record: dict) -> list[dict]:
    raw = conversation_value(record)
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"sample={record.get('id')} assistant target is not JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        raise ValueError(f"sample={record.get('id')} assistant target has no lines array")
    return payload["lines"]


def is_empty_record(record: dict) -> bool:
    return not any(
        isinstance(item, dict) and str(item.get("category", "")).strip()
        for item in parse_target_lines(record)
    )


def replace_target_lines(record: dict, lines: list[dict]) -> dict:
    result = copy.deepcopy(record)
    target_text = json.dumps({"lines": lines}, ensure_ascii=False, separators=(",", ":"))
    replaced = False
    for message in result.get("conversations", []):
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        if role in {"gpt", "assistant"}:
            if "value" in message or "content" not in message:
                message["value"] = target_text
            else:
                message["content"] = target_text
            replaced = True
    if not replaced:
        result.setdefault("conversations", []).append({"from": "gpt", "value": target_text})
    return result


def rotate_context_point(point: tuple[float, float], angle: float) -> tuple[float, float]:
    center = (CONTEXT_SIZE - 1) / 2.0
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    dx, dy = point[0] - center, point[1] - center
    return (
        center + cosine * dx + sine * dy,
        center - sine * dx + cosine * dy,
    )


def model_to_context(point: list[Any]) -> tuple[float, float]:
    if len(point) < 2:
        raise ValueError(f"invalid point: {point!r}")
    return (
        ROI_OFFSET + float(point[0]) / COORD_RANGE * (TARGET_SIZE - 1),
        ROI_OFFSET + float(point[1]) / COORD_RANGE * (TARGET_SIZE - 1),
    )


def context_to_model(point: tuple[float, float]) -> list[int]:
    x = round((point[0] - ROI_OFFSET) / (TARGET_SIZE - 1) * COORD_RANGE)
    y = round((point[1] - ROI_OFFSET) / (TARGET_SIZE - 1) * COORD_RANGE)
    return [max(0, min(COORD_RANGE, int(x))), max(0, min(COORD_RANGE, int(y)))]


def is_roi_boundary(point: tuple[float, float], tolerance: float = 1e-3) -> bool:
    return any(
        abs(value - edge) <= tolerance
        for value, edge in (
            (point[0], ROI_BOX[0]),
            (point[0], ROI_BOX[2]),
            (point[1], ROI_BOX[1]),
            (point[1], ROI_BOX[3]),
        )
    )


def geometry_line_parts(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        result = []
        for item in geometry.geoms:
            result.extend(geometry_line_parts(item))
        return result
    return []


def geometry_polygon_parts(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        result = []
        for item in geometry.geoms:
            result.extend(geometry_polygon_parts(item))
        return result
    return []


def dedupe_points(points: list[list[int]]) -> list[list[int]]:
    result = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def rotate_centerline(item: dict, angle: float, clip_box: Any) -> list[dict]:
    points = [rotate_context_point(model_to_context(point), angle) for point in item.get("points", [])]
    if len(points) < 2:
        return []
    clipped = LineString(points).intersection(clip_box)
    result = []
    for part in geometry_line_parts(clipped):
        if part.length < 1.0:
            continue
        context_points = list(part.coords)
        model_points = dedupe_points([context_to_model((float(x), float(y))) for x, y in context_points])
        if len(model_points) < 2:
            continue
        result.append({
            "category": "centerline",
            "lane_type": item.get("lane_type", item.get("type", "common")),
            "start_type": "cut" if is_roi_boundary(tuple(context_points[0])) else "inside",
            "end_type": "cut" if is_roi_boundary(tuple(context_points[-1])) else "inside",
            "points": model_points,
        })
    return result


def rotate_intersection(item: dict, angle: float, clip_box: Any) -> list[dict]:
    source_points = item.get("points", [])
    if len(source_points) < 4:
        return []
    context_points = [rotate_context_point(model_to_context(point), angle) for point in source_points]
    try:
        polygon = Polygon(context_points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        clipped = polygon.intersection(clip_box)
    except Exception:
        return []
    result = []
    for part in geometry_polygon_parts(clipped):
        if part.area <= 1.0:
            continue
        ring_context = [(float(x), float(y)) for x, y in part.exterior.coords]
        ring = dedupe_points([context_to_model(point) for point in ring_context])
        if len(ring) >= 2 and ring[0] != ring[-1]:
            ring.append(list(ring[0]))
        if len(ring) < 4:
            continue
        result.append({
            "category": "intersection",
            "intersection_type": item.get("intersection_type", "other"),
            "is_cut": bool(item.get("is_cut")) or any(is_roi_boundary(point) for point in ring_context),
            "points": ring,
        })
    return result


def rotate_target_lines(lines: list[dict], angle: float) -> list[dict]:
    if box is None:
        raise ModuleNotFoundError(
            "shapely is required for rotation mode; activate rc-dataset-v2-py313 "
            "or install shapely in the current environment"
        )
    clip_box = box(*ROI_BOX)
    result = []
    for item in lines:
        category = str(item.get("category", "")).strip().lower()
        if category == "centerline":
            result.extend(rotate_centerline(item, angle, clip_box))
        elif category == "intersection":
            result.extend(rotate_intersection(item, angle, clip_box))
        else:
            raise ValueError(f"unsupported target category for rotation: {category!r}")
    result.sort(key=lambda item: (0 if item.get("category") == "centerline" else 1, item["points"][0][1], item["points"][0][0]))
    return result


def angle_image(source: Path, destination: Path, angle: float, resample: int, compress_level: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        rotated = image.rotate(
            angle,
            resample=resample,
            expand=False,
            center=((CONTEXT_SIZE - 1) / 2.0, (CONTEXT_SIZE - 1) / 2.0),
            fillcolor=(0, 0, 0),
        )
        rotated.save(temporary, format="PNG", compress_level=compress_level)
    temporary.replace(destination)


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    return used_mode


def resample_value(name: str) -> int:
    resampling = getattr(Image, "Resampling", Image)
    return {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
    }[name]


def suffixed_image_path(relative: str, label: str) -> str:
    path = safe_relative_path(relative)
    return (path.parent / f"{path.stem}_{label}{path.suffix or '.png'}").as_posix()


def update_meta(meta_row: dict, record: dict, augmentation: str, angle: float | None = None, source_id: str = "") -> dict:
    result = copy.deepcopy(meta_row)
    result.update({
        "id": record["id"],
        "image": record["image"],
        "images": list(record["images"]),
        "raw_lane_image": record["raw_lane_image"],
        "pose_image": record["pose_image"],
        "meta": dict(record.get("meta") or {}),
        "augmentation": augmentation,
    })
    if angle is not None:
        result["augmentation_angle_degrees"] = angle
        result["augmentation_source_id"] = source_id
        result["augmentation_coordinate_frame"] = "rotated_context_local"
        result["global_coordinates_valid"] = False
    return result


def prepare_base_record(record: dict, meta_row: dict, output_variant: str, augmentation: str = "none") -> tuple[dict, dict]:
    result = copy.deepcopy(record)
    images = record_images(result)
    meta = dict(result.get("meta") or {})
    meta.update({
        "dataset_variant": output_variant,
        "target_size": TARGET_SIZE,
        "context_image_size": CONTEXT_SIZE,
        "target_roi_in_image": [ROI_OFFSET, ROI_OFFSET, ROI_OFFSET + TARGET_SIZE, ROI_OFFSET + TARGET_SIZE],
        "coord_mode": "norm1000",
        "coord_range": COORD_RANGE,
        "input_image_roles": list(IMAGE_ROLES),
        "three_image_input": True,
        "raw_lane_overlay": False,
        "raw_lane_separate_image": True,
        "augmentation": augmentation,
        "global_coordinates_valid": True,
    })
    result["meta"] = meta
    result["images"] = images
    result["image"] = images[0]
    result["raw_lane_image"] = images[1]
    result["pose_image"] = images[2]
    return result, update_meta(meta_row, result, augmentation)


def make_rotated_record(
    record: dict,
    meta_row: dict,
    angle: float,
    output_variant: str,
    *,
    allow_empty_after_clip: bool = False,
) -> tuple[dict | None, dict | None]:
    lines = parse_target_lines(record)
    rotated_lines = rotate_target_lines(lines, angle)
    if not rotated_lines and lines and not allow_empty_after_clip:
        return None, None
    label = angle_label(angle)
    result = replace_target_lines(record, rotated_lines)
    source_id = str(record["id"])
    result["id"] = f"{source_id}_{label}"
    source_images = record_images(record)
    rotated_images = [suffixed_image_path(path, label) for path in source_images]
    result["images"] = rotated_images
    result["image"] = rotated_images[0]
    result["raw_lane_image"] = rotated_images[1]
    result["pose_image"] = rotated_images[2]
    meta = dict(result.get("meta") or {})
    meta.update({
        "dataset_variant": output_variant,
        "target_size": TARGET_SIZE,
        "context_image_size": CONTEXT_SIZE,
        "target_roi_in_image": [ROI_OFFSET, ROI_OFFSET, ROI_OFFSET + TARGET_SIZE, ROI_OFFSET + TARGET_SIZE],
        "coord_mode": "norm1000",
        "coord_range": COORD_RANGE,
        "input_image_roles": list(IMAGE_ROLES),
        "three_image_input": True,
        "raw_lane_overlay": False,
        "raw_lane_separate_image": True,
        "augmentation": "rotation",
        "augmentation_angle_degrees": angle,
        "augmentation_source_id": source_id,
        "augmentation_coordinate_frame": "rotated_context_local",
        "global_coordinates_valid": False,
        "rotation_target_policy": "rotate_source_roi_labels_then_clip_to_fixed_center_roi",
        "rotation_center_pixel": [(CONTEXT_SIZE - 1) / 2.0, (CONTEXT_SIZE - 1) / 2.0],
        "rotation_label_space": "roi_relative_norm1000",
    })
    # The rotated tensor is a local augmentation. Its source x0/y0 are kept
    # only for provenance; they no longer describe the physical map location
    # of the fixed, axis-aligned ROI in the rotated image.
    meta["source_global_metadata"] = {
        key: meta.get(key)
        for key in ("x0", "y0", "target_box_full", "context_box_full", "tile_id", "raw_sample_id")
        if meta.get(key) is not None
    }
    meta["x0"] = None
    meta["y0"] = None
    meta["target_box_full"] = None
    meta["context_box_full"] = None
    result["meta"] = meta
    return result, update_meta(meta_row, result, "rotation", angle, source_id)


def make_neighbor_rotation_record(
    record: dict,
    meta_row: dict,
    angle: float,
    phase: int,
    grid_x: int,
    grid_y: int,
    grid_stride: int,
    output_variant: str,
) -> tuple[dict, dict]:
    """Replace one source row with its spatially assigned rotation.

    The replacement mode deliberately keeps the source sample id and the
    number/order of rows unchanged.  Non-zero rotations are local training
    augmentations, so their physical global coordinates are invalidated while
    the original coordinates remain in ``source_global_metadata``.
    """

    source_id = str(record["id"])
    source_lines = parse_target_lines(record)
    is_zero_angle = abs(angle % 360.0) < 1e-9
    if is_zero_angle:
        result, _ = prepare_base_record(
            record,
            meta_row,
            output_variant,
            "neighbor_rotation",
        )
        applied_angle = 0.0
        target_clipped_to_empty = False
    else:
        rotated, _ = make_rotated_record(
            record,
            meta_row,
            angle,
            output_variant,
            allow_empty_after_clip=True,
        )
        if rotated is None:  # Defensive: allow_empty_after_clip should prevent this.
            raise ValueError(
                f"neighbor rotation unexpectedly dropped sample={source_id} angle={angle}"
            )
        result = rotated
        # Replacement mode has one row per source row; do not create an angle-
        # suffixed sample id as the additive rotation mode does.
        result["id"] = source_id
        applied_angle = float(angle % 360.0)
        target_clipped_to_empty = bool(source_lines) and not parse_target_lines(result)

    source_meta = dict(record.get("meta") or {})
    meta = dict(result.get("meta") or {})
    source_global_metadata = {
        key: source_meta.get(key)
        for key in (
            "x0",
            "y0",
            "target_box_full",
            "context_box_full",
            "tile_id",
            "raw_sample_id",
        )
        if source_meta.get(key) is not None
    }
    meta.update({
        "dataset_variant": output_variant,
        "target_size": TARGET_SIZE,
        "context_image_size": CONTEXT_SIZE,
        "target_roi_in_image": [
            ROI_OFFSET,
            ROI_OFFSET,
            ROI_OFFSET + TARGET_SIZE,
            ROI_OFFSET + TARGET_SIZE,
        ],
        "coord_mode": "norm1000",
        "coord_range": COORD_RANGE,
        "input_image_roles": list(IMAGE_ROLES),
        "three_image_input": True,
        "raw_lane_overlay": False,
        "raw_lane_separate_image": True,
        "augmentation": "neighbor_rotation",
        "augmentation_source_id": source_id,
        "augmentation_angle_degrees": applied_angle,
        "neighbor_rotation_requested_angle_degrees": float(angle % 360.0),
        "neighbor_rotation_applied_angle_degrees": applied_angle,
        "neighbor_rotation_phase": int(phase),
        "neighbor_rotation_grid_stride": int(grid_stride),
        "neighbor_rotation_grid_x": int(grid_x),
        "neighbor_rotation_grid_y": int(grid_y),
        "neighbor_rotation_policy": "replace_one_row_per_stride256_grid_location",
        "neighbor_rotation_phase_formula": "(grid_x + grid_y) % 3",
        "rotation_target_policy": "rotate_source_roi_labels_then_clip_to_fixed_center_roi",
        "rotation_center_pixel": [
            (CONTEXT_SIZE - 1) / 2.0,
            (CONTEXT_SIZE - 1) / 2.0,
        ],
        "rotation_label_space": "roi_relative_norm1000",
        "source_global_metadata": source_global_metadata,
        "global_coordinates_valid": bool(is_zero_angle),
        "target_clipped_to_empty": bool(target_clipped_to_empty),
    })
    if not is_zero_angle:
        meta["x0"] = None
        meta["y0"] = None
        meta["target_box_full"] = None
        meta["context_box_full"] = None
    result["meta"] = meta
    images = record_images(result)
    result["images"] = images
    result["image"] = images[0]
    result["raw_lane_image"] = images[1]
    result["pose_image"] = images[2]

    result_meta = update_meta(meta_row, result, "neighbor_rotation")
    result_meta.update({
        "augmentation_source_id": source_id,
        "augmentation_angle_degrees": applied_angle,
        "neighbor_rotation_requested_angle_degrees": float(angle % 360.0),
        "neighbor_rotation_applied_angle_degrees": applied_angle,
        "neighbor_rotation_phase": int(phase),
        "neighbor_rotation_grid_stride": int(grid_stride),
        "neighbor_rotation_grid_x": int(grid_x),
        "neighbor_rotation_grid_y": int(grid_y),
        "global_coordinates_valid": bool(is_zero_angle),
    })
    return result, result_meta


def materialize_rotated_images(
    input_root: Path,
    output_root: Path,
    source_record: dict,
    rotated_record: dict,
    angle: float,
    image_resample: int,
    compress_level: int,
) -> None:
    source_images = record_images(source_record)
    target_images = record_images(rotated_record)
    angle_image(
        input_root / safe_relative_path(source_images[0]),
        output_root / safe_relative_path(target_images[0]),
        angle,
        image_resample,
        compress_level,
    )
    for source_relative, target_relative in zip(source_images[1:], target_images[1:]):
        angle_image(
            input_root / safe_relative_path(source_relative),
            output_root / safe_relative_path(target_relative),
            angle,
            resample_value("nearest"),
            compress_level,
        )


def validate_output(
    root: Path,
    mode: str,
    sample_limit: int,
    *,
    neighbor_angles: list[float] | None = None,
    neighbor_grid_stride: int = 256,
) -> dict:
    counts = Counter()
    augmentation_counts = Counter()
    neighbor_angle_counts = Counter()
    empty_counts = Counter()
    decoded = 0
    seen_ids = set()
    for split in SPLITS:
        for line_number, record, meta_row in iter_paired_records(root, split):
            sample_id = str(record.get("id"))
            if sample_id in seen_ids:
                raise ValueError(f"duplicate output sample id: {sample_id}")
            seen_ids.add(sample_id)
            images = record_images(record)
            for relative in images:
                path = root / safe_relative_path(relative)
                if not path.is_file():
                    raise FileNotFoundError(f"missing output image at {root / relative}")
                if sample_limit > 0 and decoded < sample_limit:
                    with Image.open(path) as image:
                        if image.size != (CONTEXT_SIZE, CONTEXT_SIZE):
                            raise ValueError(f"output image size={image.size}, expected (512,512): {path}")
            if sample_limit > 0 and decoded < sample_limit:
                decoded += 1
            record_meta = record.get("meta") or {}
            augmentation = str(record_meta.get("augmentation", "none"))
            augmentation_counts[augmentation] += 1
            empty_counts[split] += int(is_empty_record(record))
            counts[split] += 1
            if mode == "nonoverlap" and split == "train" and not is_strict_512_grid(record):
                raise ValueError(f"non-overlap output contains non-512-grid row: {sample_id}")
            if mode == "rotation" and augmentation == "rotation":
                source_id = str(record_meta.get("augmentation_source_id", ""))
                try:
                    angle = float(record_meta["augmentation_angle_degrees"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"rotation row has no valid angle: {sample_id}") from exc
                if sample_id != f"{source_id}_{angle_label(angle)}":
                    raise ValueError(f"invalid rotation id: {sample_id}")
                if meta_row.get("augmentation_source_id") != source_id:
                    raise ValueError(f"record/meta rotation provenance mismatch: {sample_id}")
                if record_meta.get("global_coordinates_valid") is not False:
                    raise ValueError(f"rotation row must invalidate global coordinates: {sample_id}")
            if mode == "neighbor_rotation" and split == "train":
                if neighbor_angles is None:
                    raise ValueError("neighbor_rotation validation requires neighbor_angles")
                try:
                    requested_angle = float(
                        record_meta["neighbor_rotation_requested_angle_degrees"]
                    )
                    applied_angle = float(
                        record_meta["neighbor_rotation_applied_angle_degrees"]
                    )
                    phase = int(record_meta["neighbor_rotation_phase"])
                    grid_x = int(record_meta["neighbor_rotation_grid_x"])
                    grid_y = int(record_meta["neighbor_rotation_grid_y"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"neighbor rotation metadata is incomplete: {sample_id}"
                    ) from exc
                if augmentation != "neighbor_rotation":
                    raise ValueError(
                        f"neighbor_rotation output has unexpected augmentation={augmentation!r}: "
                        f"{sample_id}"
                    )
                source_id = str(record_meta.get("augmentation_source_id", ""))
                if source_id != sample_id:
                    raise ValueError(
                        f"neighbor replacement must preserve source id: {sample_id} != {source_id}"
                    )
                source_meta = record_meta.get("source_global_metadata") or {}
                try:
                    source_x0 = int(source_meta["x0"])
                    source_y0 = int(source_meta["y0"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"neighbor rotation is missing source coordinates: {sample_id}"
                    ) from exc
                if source_x0 // neighbor_grid_stride != grid_x or source_y0 // neighbor_grid_stride != grid_y:
                    raise ValueError(f"neighbor grid provenance mismatch: {sample_id}")
                expected_phase = (grid_x + grid_y) % len(neighbor_angles)
                expected_angle = float(neighbor_angles[expected_phase])
                if phase != expected_phase or not math.isclose(
                    requested_angle,
                    expected_angle,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(f"neighbor rotation phase mismatch: {sample_id}")
                if not math.isclose(
                    applied_angle,
                    requested_angle,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(f"neighbor rotation angle was not applied: {sample_id}")
                expected_global_valid = abs(applied_angle % 360.0) < 1e-9
                if bool(record_meta.get("global_coordinates_valid")) != expected_global_valid:
                    raise ValueError(f"neighbor global-coordinate flag mismatch: {sample_id}")
                neighbor_angle_counts[angle_label(applied_angle)] += 1
                if expected_global_valid:
                    if record_meta.get("x0") != source_x0 or record_meta.get("y0") != source_y0:
                        raise ValueError(f"zero-angle neighbor row lost source coordinates: {sample_id}")
                elif any(record_meta.get(key) is not None for key in ("x0", "y0")):
                    raise ValueError(f"rotated neighbor row retained global coordinates: {sample_id}")
    train_empty_ratio = empty_counts["train"] / counts["train"] if counts["train"] else 0.0
    result = {
        "status": "passed",
        "dataset_root": str(root),
        "mode": mode,
        "split_counts": dict(counts),
        "augmentation_counts": dict(augmentation_counts),
        "neighbor_rotation_angle_counts": dict(neighbor_angle_counts),
        "empty_counts": dict(empty_counts),
        "train_empty_ratio": train_empty_ratio,
        "decoded_sample_triplets": decoded,
    }
    write_json(root / "ablation_validation.json", result)
    return result


def package_dataset(root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = package_path.with_name(package_path.name + ".partial")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, mode="w") as archive:
        archive.add(root, arcname=root.name, recursive=True)
    temporary.replace(package_path)


def build(args: argparse.Namespace) -> dict:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if input_root == output_root:
        raise ValueError("--input-root and --output-root must be different")
    if not input_root.is_dir():
        raise FileNotFoundError(f"input dataset does not exist: {input_root}")
    neighbor_source_grid_policy = getattr(args, "neighbor_source_grid_policy", "filter")
    if args.mode == "rotation":
        angles = parse_angles(args.angles)
        if box is None:
            raise ModuleNotFoundError(
                "rotation mode requires shapely; activate rc-dataset-v2-py313 "
                "or install shapely before running this command"
            )
        neighbor_angles = None
    elif args.mode == "neighbor_rotation":
        neighbor_angles = parse_neighbor_angles(args.neighbor_angles)
        angles = []
        if box is None:
            raise ModuleNotFoundError(
                "neighbor_rotation mode requires shapely; activate rc-dataset-v2-py313 "
                "or install shapely before running this command"
            )
        if args.neighbor_grid_stride != 256:
            raise ValueError(
                "neighbor_rotation currently requires --neighbor-grid-stride 256 "
                "to match the Context512 ROI256 dataset contract"
            )
        if neighbor_source_grid_policy not in {"filter", "require"}:
            raise ValueError(
                "neighbor_source_grid_policy must be 'filter' or 'require'"
            )
    else:
        angles = []
        neighbor_angles = None
    package_path = (
        Path(args.package_path).expanduser().resolve()
        if args.package_path
        else output_root.parent / f"{output_root.name}.tar"
    )
    if not args.skip_package:
        try:
            package_path.relative_to(output_root)
        except ValueError:
            pass
        else:
            raise ValueError("--package-path must be outside --output-root")
    complete = output_root / "build_complete.json"
    if args.resume and complete.is_file():
        marker = read_json(complete)
        if marker.get("mode") != args.mode:
            raise ValueError(
                f"completed output mode={marker.get('mode')!r}, requested mode={args.mode!r}"
            )
        validation = validate_output(
            output_root,
            args.mode,
            args.validation_sample_limit,
            neighbor_angles=neighbor_angles,
            neighbor_grid_stride=args.neighbor_grid_stride,
        )
        if not args.skip_package and not package_path.is_file():
            package_dataset(output_root, package_path)
        return {"status": "reused", "output_root": str(output_root), "validation": validation}
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output exists but is incomplete: {output_root}. Use a new output root or inspect/remove only this generated directory."
        )
    if not (input_root / "dataset_info.json").is_file():
        raise FileNotFoundError(f"input dataset_info.json is missing: {input_root}")
    input_info = read_json(input_root / "dataset_info.json")
    source_train_stride = input_info.get("train_stride")
    if args.mode == "neighbor_rotation":
        if source_train_stride is not None and int(source_train_stride) <= 0:
            raise ValueError(
                f"input dataset has invalid train_stride={source_train_stride!r}"
            )
    output_variant = output_root.name
    output_root.mkdir(parents=True, exist_ok=True)

    # First pass validates the complete source contract without retaining the
    # potentially multi-million-row dataset in memory.
    sample_limit_state = [0]
    source_counts = Counter()
    source_empty_counts = Counter()
    strict_grid_train_samples = 0
    neighbor_aligned_train_samples = 0
    neighbor_filtered_train_samples = 0
    for split in SPLITS:
        for _line_number, record, meta_row in iter_paired_records(input_root, split):
            validate_input_record(input_root, record, sample_limit_state, args.validation_sample_limit)
            source_counts[split] += 1
            source_empty_counts[split] += int(is_empty_record(record))
            if split == "train" and is_strict_512_grid(record):
                strict_grid_train_samples += 1
            if args.mode == "neighbor_rotation" and split == "train":
                record_meta = record.get("meta") or {}
                if is_grid_aligned(record_meta, args.neighbor_grid_stride):
                    neighbor_grid_coordinates(record_meta, args.neighbor_grid_stride)
                    neighbor_aligned_train_samples += 1
                elif neighbor_source_grid_policy == "require":
                    neighbor_grid_coordinates(record_meta, args.neighbor_grid_stride)
                else:
                    neighbor_filtered_train_samples += 1
        print(f"[context-ablation] validated {split}={source_counts[split]}", flush=True)

    if not source_counts["train"]:
        raise ValueError("input dataset has no train rows")

    output_counts = Counter()
    output_empty_counts = Counter()
    output_id_digests = {split: hashlib.sha256() for split in SPLITS}
    link_modes = Counter()
    base_counts = Counter()
    rotated_counts = Counter()
    skipped_rotations = Counter()
    neighbor_counts = Counter()
    if args.mode == "nonoverlap":
        base_counts["selected_train"] = strict_grid_train_samples
        base_counts["filtered_train"] = source_counts["train"] - strict_grid_train_samples
        if not strict_grid_train_samples:
            raise ValueError("strict 512-grid filtering selected no train rows")
    elif args.mode == "neighbor_rotation":
        base_counts["selected_train"] = neighbor_aligned_train_samples
        base_counts["filtered_train"] = neighbor_filtered_train_samples
        if not neighbor_aligned_train_samples:
            raise ValueError(
                "neighbor_rotation found no train rows aligned to the stride-256 grid"
            )
    else:
        base_counts["selected_train"] = source_counts["train"]

    phase_dir = output_root / "phase_a"
    phase_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        record_path = phase_dir / f"{split}.jsonl"
        meta_path = phase_dir / f"meta_{split}.jsonl"
        record_partial = record_path.with_name(record_path.name + ".partial")
        meta_partial = meta_path.with_name(meta_path.name + ".partial")
        record_partial.unlink(missing_ok=True)
        meta_partial.unlink(missing_ok=True)
        with (
            record_partial.open("w", encoding="utf-8", newline="\n") as record_handle,
            meta_partial.open("w", encoding="utf-8", newline="\n") as meta_handle,
        ):
            for ordinal, (_line_number, source_record, source_meta) in enumerate(
                iter_paired_records(input_root, split),
                start=1,
            ):
                if (
                    args.mode == "nonoverlap"
                    and split == "train"
                    and not is_strict_512_grid(source_record)
                ):
                    continue

                if (
                    args.mode == "neighbor_rotation"
                    and split == "train"
                    and not is_grid_aligned(
                        source_record.get("meta") or {},
                        args.neighbor_grid_stride,
                    )
                ):
                    continue

                if args.mode == "neighbor_rotation" and split == "train":
                    angle, phase, grid_x, grid_y = neighbor_rotation_assignment(
                        source_record.get("meta") or {},
                        args.neighbor_grid_stride,
                        neighbor_angles,
                    )
                    prepared, prepared_meta = make_neighbor_rotation_record(
                        source_record,
                        source_meta,
                        angle,
                        phase,
                        grid_x,
                        grid_y,
                        args.neighbor_grid_stride,
                        output_variant,
                    )
                    if abs(angle % 360.0) >= 1e-9:
                        materialize_rotated_images(
                            input_root,
                            output_root,
                            source_record,
                            prepared,
                            angle,
                            resample_value(args.image_resample),
                            args.png_compress_level,
                        )
                    neighbor_counts[angle_label(angle)] += 1
                else:
                    base_augmentation = "base" if args.mode == "rotation" else "none"
                    prepared, prepared_meta = prepare_base_record(
                        source_record,
                        source_meta,
                        output_variant,
                        base_augmentation,
                    )
                for relative in record_images(prepared):
                    # Rotated rows already materialize their three images.
                    if not (
                        args.mode == "neighbor_rotation"
                        and split == "train"
                        and abs(float(prepared["meta"].get("augmentation_angle_degrees", 0.0)) % 360.0) >= 1e-9
                    ):
                        used_mode = link_or_copy(
                            input_root / safe_relative_path(relative),
                            output_root / safe_relative_path(relative),
                            args.copy_mode,
                        )
                        link_modes[used_mode] += 1
                record_handle.write(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n")
                meta_handle.write(json.dumps(prepared_meta, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_counts[split] += 1
                output_empty_counts[split] += int(is_empty_record(prepared))
                output_id_digests[split].update((str(prepared["id"]) + "\n").encode("utf-8"))

                if args.mode == "rotation" and split == "train":
                    source_is_empty = is_empty_record(source_record)
                    if args.rotate_empty or not source_is_empty:
                        for angle in angles:
                            rotated, rotated_meta = make_rotated_record(
                                source_record,
                                source_meta,
                                angle,
                                output_variant,
                            )
                            if rotated is None:
                                skipped_rotations[angle_label(angle)] += 1
                                continue
                            materialize_rotated_images(
                                input_root,
                                output_root,
                                source_record,
                                rotated,
                                angle,
                                resample_value(args.image_resample),
                                args.png_compress_level,
                            )
                            record_handle.write(
                                json.dumps(rotated, ensure_ascii=False, separators=(",", ":")) + "\n"
                            )
                            meta_handle.write(
                                json.dumps(rotated_meta, ensure_ascii=False, separators=(",", ":")) + "\n"
                            )
                            output_counts[split] += 1
                            output_empty_counts[split] += int(is_empty_record(rotated))
                            output_id_digests[split].update(
                                (str(rotated["id"]) + "\n").encode("utf-8")
                            )
                            rotated_counts[angle_label(angle)] += 1

                if args.progress_every > 0 and ordinal % args.progress_every == 0:
                    print(
                        f"[context-ablation] processed {split} source={ordinal} "
                        f"output={output_counts[split]}",
                        flush=True,
                    )
        record_partial.replace(record_path)
        meta_partial.replace(meta_path)
        print(
            f"[context-ablation] wrote {split}: source={source_counts[split]} "
            f"output={output_counts[split]}",
            flush=True,
        )

    if args.mode == "neighbor_rotation" and output_counts["train"] != neighbor_aligned_train_samples:
        raise ValueError(
            "neighbor_rotation must preserve exactly one output row per selected "
            "stride-256 train row: "
            f"selected={neighbor_aligned_train_samples} output={output_counts['train']}"
        )

    if args.mode == "nonoverlap":
        dataset_version = "rc_dataset_v2_context512_roi256_three_image_nonoverlap_v1"
        output_train_stride = 512
        output_angles = []
        overlap_policy = "strict_nonoverlap"
        rotation_target_policy = None
        rotation_coordinate_policy = None
        rotation_empty_rows = False
        rotated_rows_are_additive = False
        neighbor_policy = None
    elif args.mode == "rotation":
        dataset_version = "rc_dataset_v2_context512_roi256_three_image_rotation_v1"
        output_train_stride = input_info.get("train_stride", 256)
        output_angles = list(angles)
        overlap_policy = "rotation_augmented"
        rotation_target_policy = "rotate_source_roi_labels_then_clip_to_fixed_center_roi"
        rotation_coordinate_policy = "local_augmentation_global_x0_y0_invalid"
        rotation_empty_rows = bool(args.rotate_empty)
        rotated_rows_are_additive = True
        neighbor_policy = None
    else:
        dataset_version = NEIGHBOR_ROTATION_CONTRACT
        output_train_stride = args.neighbor_grid_stride
        output_angles = list(neighbor_angles)
        overlap_policy = "stride256_neighbor_rotation_replacement"
        rotation_target_policy = "rotate_source_roi_labels_then_clip_to_fixed_center_roi"
        rotation_coordinate_policy = "local_augmentation_global_x0_y0_invalid"
        rotation_empty_rows = True
        rotated_rows_are_additive = False
        neighbor_policy = {
            "grid_stride": args.neighbor_grid_stride,
            "source_train_stride": source_train_stride,
            "source_grid_policy": neighbor_source_grid_policy,
            "phase_formula": "(grid_x + grid_y) % 3",
            "angles_degrees": list(neighbor_angles),
            "replacement_policy": "one_output_row_per_selected_stride256_train_row",
            "filtered_unaligned_train_rows": neighbor_filtered_train_samples,
            "immediate_neighbor_policy": "horizontal_and_vertical_phases_differ",
        }

    output_info = copy.deepcopy(input_info)
    output_info.update({
        "dataset_version": dataset_version,
        "dataset_variant": output_variant,
        "base_dataset_root": str(input_root),
        "target_patch_size": TARGET_SIZE,
        "patch_size": TARGET_SIZE,
        "context_size": CONTEXT_SIZE,
        "stride": 256,
        "train_stride": output_train_stride,
        "eval_test_stride": input_info.get("eval_test_stride", 256),
        "coord_mode": "norm1000",
        "coord_range": COORD_RANGE,
        "three_image_input": True,
        "multi_image_input": {
            "enabled": True,
            "num_images_per_sample": 3,
            "image_roles": list(IMAGE_ROLES),
            "image_order": list(IMAGE_ROLES),
        },
        "input_overlay": {
            "raw_lane_overlay": False,
            "raw_lane_separate_image": True,
            "raw_lane_image_source": "patch_tif/0_lane.tif",
            "pose_image_source": "patch_tif/0_pose.tif",
        },
        "ablation": {
            "mode": args.mode,
            "nonoverlap_rule": "x0 % 512 == 0 and y0 % 512 == 0" if args.mode == "nonoverlap" else "not_applicable",
            "context_overlap_policy": overlap_policy,
            "angles_degrees": output_angles,
            "rotation_center_pixel": [
                (CONTEXT_SIZE - 1) / 2.0,
                (CONTEXT_SIZE - 1) / 2.0,
            ] if args.mode in {"rotation", "neighbor_rotation"} else None,
            "label_transform": rotation_target_policy,
            "rotation_coordinate_policy": rotation_coordinate_policy,
            "rotation_empty_rows": rotation_empty_rows,
            "rotated_rows_are_additive": rotated_rows_are_additive,
            "neighbor_rotation": neighbor_policy,
            "eval_test_augmentation": "none",
        },
        "record_counts": {split: output_counts[split] for split in SPLITS},
        "source_record_counts": {split: source_counts[split] for split in SPLITS},
        "neighbor_source_grid_policy": neighbor_source_grid_policy if args.mode == "neighbor_rotation" else None,
        "neighbor_aligned_train_samples": neighbor_aligned_train_samples if args.mode == "neighbor_rotation" else None,
        "neighbor_filtered_train_samples": neighbor_filtered_train_samples if args.mode == "neighbor_rotation" else None,
        "link_modes": dict(link_modes),
    })
    write_json(output_root / "dataset_info.json", output_info)
    balance_report = {
        "mode": args.mode,
        "input_train_samples": source_counts["train"],
        "input_train_empty_samples": source_empty_counts["train"],
        "output_train_samples": output_counts["train"],
        "selected_train_samples": base_counts.get("selected_train", 0),
        "strict_512_grid_train_samples": (
            strict_grid_train_samples if args.mode == "nonoverlap" else None
        ),
        "filtered_train_samples": base_counts.get("filtered_train", 0),
        "rotation_counts": dict(rotated_counts),
        "neighbor_rotation_counts": dict(neighbor_counts),
        "neighbor_source_train_stride": source_train_stride if args.mode == "neighbor_rotation" else None,
        "neighbor_source_grid_policy": neighbor_source_grid_policy if args.mode == "neighbor_rotation" else None,
        "neighbor_aligned_train_samples": neighbor_aligned_train_samples if args.mode == "neighbor_rotation" else None,
        "neighbor_filtered_train_samples": neighbor_filtered_train_samples if args.mode == "neighbor_rotation" else None,
        "skipped_rotations": dict(skipped_rotations),
        "train_empty_samples": output_empty_counts["train"],
        "train_empty_ratio": output_empty_counts["train"] / output_counts["train"],
    }
    write_json(output_root / "balance_report.json", balance_report)
    split_manifest = copy.deepcopy(read_json(input_root / "split_manifest.json")) if (input_root / "split_manifest.json").is_file() else {}
    split_manifest.update({
        "dataset_variant": output_variant,
        "base_dataset_root": str(input_root),
        "ablation_mode": args.mode,
        "split_counts": {split: output_counts[split] for split in SPLITS},
        "split_id_sha256": {split: output_id_digests[split].hexdigest() for split in SPLITS},
        "strict_512_grid_rule": "x0 % 512 == 0 and y0 % 512 == 0" if args.mode == "nonoverlap" else None,
        "rotation_angles_degrees": output_angles,
        "neighbor_rotation_policy": neighbor_policy,
        "neighbor_source_grid_policy": neighbor_source_grid_policy if args.mode == "neighbor_rotation" else None,
        "neighbor_filtered_train_samples": neighbor_filtered_train_samples if args.mode == "neighbor_rotation" else None,
    })
    write_json(output_root / "split_manifest.json", split_manifest)
    validation = validate_output(
        output_root,
        args.mode,
        args.validation_sample_limit,
        neighbor_angles=neighbor_angles,
        neighbor_grid_stride=args.neighbor_grid_stride,
    )
    summary = {
        "status": "built",
        "mode": args.mode,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "package_path": str(package_path) if not args.skip_package else "",
        "angles": output_angles,
        "base_counts": dict(base_counts),
        "rotation_counts": dict(rotated_counts),
        "neighbor_rotation_counts": dict(neighbor_counts),
        "skipped_rotations": dict(skipped_rotations),
        "split_counts": {split: output_counts[split] for split in SPLITS},
        "validation": validation,
    }
    write_json(output_root / "build_summary.json", summary)
    if not args.skip_package:
        print(f"[context-ablation] packaging: {package_path}", flush=True)
        package_dataset(output_root, package_path)
    write_json(output_root / "build_complete.json", {
        "status": "passed",
        "mode": args.mode,
        "output_root": str(output_root),
        "package_path": str(package_path) if not args.skip_package else "",
        "train_samples": output_counts["train"],
        "neighbor_source_grid_policy": neighbor_source_grid_policy if args.mode == "neighbor_rotation" else None,
        "neighbor_filtered_train_samples": neighbor_filtered_train_samples if args.mode == "neighbor_rotation" else None,
        "prompt_contract": (
            ROTATION_CONTRACT
            if args.mode == "rotation"
            else NEIGHBOR_ROTATION_CONTRACT
            if args.mode == "neighbor_rotation"
            else NONOVERLAP_CONTRACT
        ),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
