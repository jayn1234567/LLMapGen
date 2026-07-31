#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import math
import random
import shutil
import sys
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, box, shape
except ModuleNotFoundError:
    gpd = None
    rasterio = None
    from_origin = None
    GeometryCollection = LineString = MultiLineString = MultiPolygon = Polygon = box = shape = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.coord_utils import (
    COORD_MODE_NORM1000,
    COORD_MODE_PIXEL,
    DEFAULT_COORD_RANGE,
    coord_system_name,
    normalize_coord_mode,
    pixel_point_to_coord,
)


TASK_TEXT = "Please construct the complete road map in the current BEV (Bird's Eye View) image patch."
INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
SEMANTIC_SCHEMA_VERSION = "lane_intersection_semantic_v3_lane_4_18_25_ignore_3_22"
LANE_TYPE_NAMES = {
    1: "common",
    2: "right_turn",
    4: "waiting_area",
    18: "bus_lane",
    25: "main_auxiliary_connector",
}
ALLOWED_LANE_TYPES = frozenset({*LANE_TYPE_NAMES.values(), "other"})
INTERSECTION_TYPE_BY_SOURCE_PAIR = {
    (1, 1): "common",
    (1, 2): "t_intersection",
    (1, 3): "small_untyped",
    (4, 1): "t_lane_change_area",
}
ALLOWED_INTERSECTION_TYPES = frozenset({*INTERSECTION_TYPE_BY_SOURCE_PAIR.values(), "other"})
IGNORED_LANE_TYPE_CODES = frozenset({3, 22})
DEFAULT_ARCHIVE_WORKERS = 16
ARCHIVE_EXTRACT_MARKER = ".archive_extract_complete.json"
REQUIRED_ARCHIVE_MEMBER_VERSION = "rc_required_v2_with_pose"


def require_geo_dependencies():
    missing = []
    if gpd is None:
        missing.append("geopandas")
    if rasterio is None:
        missing.append("rasterio")
    if Polygon is None:
        missing.append("shapely")
    if missing:
        raise ModuleNotFoundError(
            "Missing geospatial dependencies: "
            + ", ".join(missing)
            + ". Install them in the data-processing environment before running dataset generation."
        )


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    root: Path
    lane_geojson: Path
    intersection_geojson: Path
    image_tiff: Path
    mask_tiff: Path
    raw_lane_tiff: Path
    pose_tiff: Path


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def sample_id_from_root(root: Path) -> str:
    return root.name


def geojson_feature_count(path: Path):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    features = payload.get("features") if isinstance(payload, dict) else None
    return len(features) if isinstance(features, list) else None


def find_geojson(label_dir: Path, preferred_names, stem_keywords):
    if not label_dir.exists():
        return label_dir / preferred_names[0]
    preferred_lower = {name.lower() for name in preferred_names}
    geojson_files = sorted(label_dir.glob("*.geojson"))
    candidates = []
    for path in geojson_files:
        if path.name.lower() in preferred_lower:
            candidates.append(path)
    for path in geojson_files:
        if path in candidates:
            continue
        stem = path.stem.lower()
        if any(keyword in stem for keyword in stem_keywords):
            candidates.append(path)
    if not candidates:
        return label_dir / preferred_names[0]
    nonempty = [path for path in candidates if (geojson_feature_count(path) or 0) > 0]
    if nonempty:
        return nonempty[0]
    return candidates[0]


def required_paths(root: Path) -> RawSample:
    label_dir = root / "label_check_crop"
    return RawSample(
        sample_id=sample_id_from_root(root),
        root=root,
        lane_geojson=find_geojson(label_dir, ("Lane.geojson", "lane.geojson"), ("lane",)),
        intersection_geojson=find_geojson(
            label_dir,
            ("intersection.geojson", "Intersection.geojson"),
            ("intersection",),
        ),
        image_tiff=root / "inter_patch_tif" / "0_inter.tif",
        mask_tiff=root / "patch_tif" / "0_edit_poly.tif",
        raw_lane_tiff=root / "patch_tif" / "0_lane.tif",
        pose_tiff=root / "patch_tif" / "0_pose.tif",
    )


def is_valid_sample_root(root: Path, require_intersection: bool = False) -> bool:
    sample = required_paths(root)
    required = [sample.lane_geojson, sample.image_tiff, sample.mask_tiff]
    if require_intersection:
        required.append(sample.intersection_geojson)
    return all(path.exists() for path in required)


def archive_signature(archive_path: Path) -> dict:
    stat = archive_path.stat()
    return {
        "archive_path": str(archive_path.resolve()),
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
    }


def archive_extract_marker_path(archive_path: Path) -> Path:
    target_dir = archive_path.with_suffix("").with_suffix("")
    return target_dir / ARCHIVE_EXTRACT_MARKER


def write_archive_extract_marker(
    archive_path: Path,
    adopted_existing: bool = False,
    extract_mode: str = "full",
    extracted_files: int | None = None,
    extracted_bytes: int | None = None,
):
    marker_path = archive_extract_marker_path(archive_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_marker = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary_marker.write_text(
        json.dumps(
            {
                **archive_signature(archive_path),
                "completed_at_unix": time.time(),
                "adopted_existing_extraction": adopted_existing,
                "extract_mode": extract_mode,
                "required_member_version": REQUIRED_ARCHIVE_MEMBER_VERSION,
                "extracted_files": extracted_files,
                "extracted_bytes": extracted_bytes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_marker.replace(marker_path)


def archive_extract_is_complete(archive_path: Path, extract_mode: str = "full") -> bool:
    marker_path = archive_extract_marker_path(archive_path)
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        signature_matches = all(
            marker.get(key) == value for key, value in archive_signature(archive_path).items()
        )
        marker_mode = marker.get("extract_mode", "full")
        mode_matches = marker_mode == "full" or marker_mode == extract_mode
        member_version_matches = (
            extract_mode != "required_only"
            or marker.get("required_member_version") == REQUIRED_ARCHIVE_MEMBER_VERSION
        )
        return signature_matches and mode_matches and member_version_matches
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def required_archive_member(member: tarfile.TarInfo) -> bool:
    """Return whether an archive member is consumed by the raw RC builder."""
    if not member.isfile():
        return False
    parts = tuple(part for part in member.name.replace("\\", "/").split("/") if part not in {"", "."})
    if not parts or ".." in parts:
        return False
    lowered = tuple(part.lower() for part in parts)
    if len(lowered) >= 2 and lowered[-2:] == ("inter_patch_tif", "0_inter.tif"):
        return True
    if len(lowered) >= 2 and lowered[-2:] == ("patch_tif", "0_edit_poly.tif"):
        return True
    if len(lowered) >= 2 and lowered[-2:] == ("patch_tif", "0_lane.tif"):
        return True
    if len(lowered) >= 2 and lowered[-2:] == ("patch_tif", "0_pose.tif"):
        return True
    if len(lowered) >= 2 and lowered[-2] == "label_check_crop" and lowered[-1].endswith(".geojson"):
        return True
    return False


def extracted_tree_matches_archive(
    archive_path: Path,
    target_dir: Path,
    extract_mode: str = "full",
) -> bool:
    """Validate a pre-marker extraction so interrupted runs can resume safely."""
    marker_path = archive_extract_marker_path(archive_path)
    if not target_dir.is_dir() or not any(path != marker_path for path in target_dir.iterdir()):
        return False
    target_resolved = target_dir.resolve()
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            matched_files = 0
            for member in tar:
                if extract_mode == "required_only" and not required_archive_member(member):
                    continue
                if member.isfile():
                    matched_files += 1
                if extract_mode == "required_only":
                    normalized_parts = [
                        part
                        for part in member.name.replace("\\", "/").split("/")
                        if part not in {"", "."}
                    ]
                    member_path = target_dir.joinpath(*normalized_parts).resolve()
                else:
                    member_path = (target_dir / member.name).resolve()
                try:
                    member_path.relative_to(target_resolved)
                except ValueError:
                    raise ValueError(f"unsafe archive member path: {member.name}")
                if member.isdir():
                    if not member_path.is_dir():
                        return False
                elif member.isfile():
                    if not member_path.is_file() or member_path.stat().st_size != member.size:
                        return False
                elif not member_path.exists() and not member_path.is_symlink():
                    return False
            if extract_mode == "required_only" and matched_files == 0:
                return False
    except (OSError, tarfile.TarError):
        return False
    return True


def extract_required_archive_members(archive_path: Path, target_dir: Path) -> tuple[int, int]:
    """Stream only the TIFF/mask/GeoJSON files needed by Dataset V2."""
    target_resolved = target_dir.resolve()
    extracted_files = 0
    extracted_bytes = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not required_archive_member(member):
                continue
            normalized_parts = [
                part for part in member.name.replace("\\", "/").split("/") if part not in {"", "."}
            ]
            member_path = target_dir.joinpath(*normalized_parts).resolve()
            try:
                member_path.relative_to(target_resolved)
            except ValueError:
                raise ValueError(f"unsafe archive member path: {member.name}")
            source = tar.extractfile(member)
            if source is None:
                raise tarfile.ExtractError(f"unable to read required archive member: {member.name}")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = member_path.with_name(member_path.name + ".partial")
            try:
                with source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                temporary.replace(member_path)
            finally:
                temporary.unlink(missing_ok=True)
            extracted_files += 1
            extracted_bytes += int(member.size)
    return extracted_files, extracted_bytes


def safe_extract_tar_gz(
    archive_path: Path,
    delete_archive: bool,
    selective: bool = False,
) -> Path:
    target_dir = archive_path.with_suffix("").with_suffix("")
    target_dir.mkdir(parents=True, exist_ok=True)
    extract_mode = "required_only" if selective else "full"
    if archive_extract_is_complete(archive_path, extract_mode=extract_mode):
        if delete_archive and find_sample_roots(target_dir, require_intersection=False):
            archive_path.unlink()
        return target_dir

    if extracted_tree_matches_archive(archive_path, target_dir, extract_mode=extract_mode):
        write_archive_extract_marker(
            archive_path,
            adopted_existing=True,
            extract_mode=extract_mode,
        )
        if delete_archive and find_sample_roots(target_dir, require_intersection=False):
            archive_path.unlink()
        return target_dir

    marker_path = archive_extract_marker_path(archive_path)
    marker_path.unlink(missing_ok=True)
    if selective:
        extracted_files, extracted_bytes = extract_required_archive_members(archive_path, target_dir)
        if extracted_files == 0:
            raise FileNotFoundError(f"archive contains no required RC files: {archive_path}")
    else:
        target_resolved = target_dir.resolve()
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_path = (target_dir / member.name).resolve()
                try:
                    member_path.relative_to(target_resolved)
                except ValueError:
                    raise ValueError(f"unsafe archive member path: {member.name}")
            tar.extractall(path=target_dir)
        extracted_files = None
        extracted_bytes = None

    if selective and not find_sample_roots(target_dir, require_intersection=False):
        raise FileNotFoundError(
            f"selective extraction did not produce a valid RC sample; archive retained: {archive_path}"
        )
    write_archive_extract_marker(
        archive_path,
        extract_mode=extract_mode,
        extracted_files=extracted_files,
        extracted_bytes=extracted_bytes,
    )
    if delete_archive and find_sample_roots(target_dir, require_intersection=False):
        archive_path.unlink()
    return target_dir


def extract_archives(
    input_root: Path,
    delete_archive: bool,
    workers: int = DEFAULT_ARCHIVE_WORKERS,
    selective: bool = False,
):
    archives = sorted(input_root.rglob("*.tar.gz"))
    if not archives:
        print(f"[archive-extract] no .tar.gz archives under {input_root}", flush=True)
        return

    extract_mode = "required_only" if selective else "full"
    pending = [
        archive
        for archive in archives
        if delete_archive or not archive_extract_is_complete(archive, extract_mode=extract_mode)
    ]
    skipped = len(archives) - len(pending)
    worker_count = min(max(1, int(workers)), max(1, len(pending)))
    print(
        f"[archive-extract] root={input_root} archives={len(archives)} "
        f"pending={len(pending)} skipped={skipped} workers={worker_count} mode={extract_mode}",
        flush=True,
    )
    if not pending:
        return

    started_at = time.monotonic()
    report_every = max(1, math.ceil(len(pending) / 20))

    def report_progress(completed: int, archive: Path):
        if completed == 1 or completed == len(pending) or completed % report_every == 0:
            elapsed = time.monotonic() - started_at
            rate = completed / elapsed if elapsed > 0 else 0.0
            print(
                f"[archive-extract] completed={completed}/{len(pending)} "
                f"rate={rate:.2f} archives/s last={archive.name}",
                flush=True,
            )

    if worker_count == 1:
        for completed, archive in enumerate(pending, start=1):
            safe_extract_tar_gz(archive, delete_archive=delete_archive, selective=selective)
            report_progress(completed, archive)
        return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="tar-extract",
    ) as executor:
        future_to_archive = {
            executor.submit(safe_extract_tar_gz, archive, delete_archive, selective): archive
            for archive in pending
        }
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_archive), start=1):
            archive = future_to_archive[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"failed to extract archive: {archive}") from exc
            report_progress(completed, archive)


def find_sample_roots(input_root: Path, require_intersection: bool = False):
    roots = set()
    for inter_dir in input_root.rglob("inter_patch_tif"):
        root = inter_dir.parent
        if is_valid_sample_root(root, require_intersection=require_intersection):
            roots.add(root)
    return sorted(roots, key=lambda path: str(path))


def discover_samples(
    input_root: Path,
    include_intersections: bool,
    delete_archives: bool,
    limit_samples=None,
    require_intersection_features: bool = False,
    archive_workers: int = DEFAULT_ARCHIVE_WORKERS,
    selective_archive_extract: bool = False,
):
    extract_archives(
        input_root,
        delete_archive=delete_archives,
        workers=archive_workers,
        selective=selective_archive_extract,
    )
    # Lane+intersection training still needs lane-only negative examples:
    # an empty or missing Intersection.geojson means "no intersection target",
    # not "invalid raw sample". Only lane/image/mask are required here.
    roots = find_sample_roots(input_root, require_intersection=False)
    if require_intersection_features:
        roots = [
            root for root in roots
            if (geojson_feature_count(required_paths(root).intersection_geojson) or 0) > 0
        ]
    samples = [required_paths(root) for root in roots]
    samples = sorted(samples, key=lambda sample: (sample.sample_id, str(sample.root)))
    if limit_samples is not None:
        samples = samples[:limit_samples]
    return samples


def intersection_availability(samples: list[RawSample]) -> dict:
    with_file = 0
    nonempty = 0
    empty = 0
    missing = 0
    for sample in samples:
        if not sample.intersection_geojson.exists():
            missing += 1
            continue
        with_file += 1
        if (geojson_feature_count(sample.intersection_geojson) or 0) > 0:
            nonempty += 1
        else:
            empty += 1
    return {
        "with_intersection_file": with_file,
        "nonempty_intersection_file": nonempty,
        "empty_intersection_file": empty,
        "missing_intersection_file": missing,
    }


def split_samples(samples, train_ratio: float, eval_ratio: float, eval_count: int, seed: int):
    ordered = list(samples)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, [], []
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("--train-ratio must be in (0, 1)")
    if eval_ratio < 0:
        raise ValueError("--eval-ratio must be >= 0")
    if eval_count < 0 and train_ratio + eval_ratio >= 1:
        raise ValueError("--train-ratio + --eval-ratio must be < 1 so test samples remain")

    total = len(ordered)
    wants_eval = eval_count > 0 or (eval_count < 0 and eval_ratio > 0)
    min_holdout = 1 + (1 if wants_eval else 0)
    if total <= min_holdout:
        min_holdout = min(1, total - 1)

    n_train = int(total * train_ratio)
    n_train = max(1, min(total - min_holdout, n_train))
    remaining = total - n_train

    if eval_count >= 0:
        n_eval = eval_count
    else:
        n_eval = int(math.ceil(total * eval_ratio))
    n_eval = max(0, n_eval)
    if wants_eval:
        n_eval = max(1, n_eval)
    # Keep at least one raw sample for the final held-out test split whenever possible.
    max_eval = remaining - 1 if remaining > 1 else 0
    n_eval = min(n_eval, max_eval)

    train_samples = ordered[:n_train]
    eval_samples = ordered[n_train:n_train + n_eval]
    test_samples = ordered[n_train + n_eval:]
    return train_samples, eval_samples, test_samples


def apply_raw_lane_overlay(
    image: np.ndarray,
    raw_lane: np.ndarray,
    threshold: float = 0.0,
    value: int = 255,
) -> np.ndarray:
    if raw_lane.ndim == 2:
        raw_lane = raw_lane[np.newaxis, :, :]
    if raw_lane.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"raw lane shape {raw_lane.shape[-2:]} does not match image shape {image.shape[-2:]}"
        )
    lane_mask = (raw_lane > threshold).any(axis=0)
    if not lane_mask.any():
        return image
    output = image.copy()
    if output.shape[0] >= 3:
        output[:3, lane_mask] = value
    else:
        output[:, lane_mask] = value
    return output


def read_masked_image(
    image_path: Path,
    mask_path: Path,
    raw_lane_path: Path | None = None,
    raw_lane_overlay: bool = False,
    require_raw_lane: bool = False,
    raw_lane_threshold: float = 0.0,
):
    with rasterio.open(image_path) as src:
        image = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
    with rasterio.open(mask_path) as src:
        mask = src.read()
    mask_any = (mask > 0).any(axis=0, keepdims=True)
    image = np.where(mask_any, image, 0)
    if raw_lane_overlay:
        if raw_lane_path is None or not raw_lane_path.exists():
            if require_raw_lane:
                raise FileNotFoundError(f"raw lane TIFF not found: {raw_lane_path}")
        else:
            with rasterio.open(raw_lane_path) as src:
                raw_lane = src.read()
            raw_lane = np.where(mask_any, raw_lane, 0)
            image = apply_raw_lane_overlay(image, raw_lane, threshold=raw_lane_threshold)
    return image, meta, transform, crs


def read_masked_binary_image(
    image_path: Path,
    mask_path: Path,
    threshold: float = 0.0,
) -> np.ndarray:
    """Read a mask-like TIFF as a black RGB image with white positive pixels."""
    if not image_path.is_file():
        raise FileNotFoundError(f"auxiliary TIFF not found: {image_path}")
    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(mask_path) as src:
        valid_mask = src.read()
    if image.shape[-2:] != valid_mask.shape[-2:]:
        raise ValueError(
            f"auxiliary TIFF shape {image.shape[-2:]} does not match mask shape "
            f"{valid_mask.shape[-2:]}: {image_path}"
        )
    positive = (image > threshold).any(axis=0)
    valid = (valid_mask > 0).any(axis=0)
    output = np.zeros((3, image.shape[-2], image.shape[-1]), dtype=np.uint8)
    output[:, positive & valid] = 255
    return output


def image_chunk_to_pil(chunk: np.ndarray) -> Image.Image:
    if chunk.shape[0] == 1:
        arr = np.repeat(chunk, 3, axis=0)
    else:
        arr = chunk[:3]
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pad_image_to_patch_grid(image: np.ndarray, patch_size: int):
    _, height, width = image.shape
    pad_height = (-height) % patch_size
    pad_width = (-width) % patch_size
    if pad_height == 0 and pad_width == 0:
        return image, [width, height]
    padded = np.pad(
        image,
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )
    return padded, [width, height]


def centered_target_roi(target_size: int, context_size: int):
    """Return the target ROI inside a centered square context image."""
    if target_size <= 0 or context_size <= 0:
        raise ValueError("target_size and context_size must be positive")
    if context_size < target_size:
        raise ValueError("context_size must be greater than or equal to target_size")
    if (context_size - target_size) % 2:
        raise ValueError("context_size - target_size must be even for a centered target ROI")
    margin = (context_size - target_size) // 2
    return [margin, margin, margin + target_size, margin + target_size]


def extract_centered_context(
    image: np.ndarray,
    target_x0: int,
    target_y0: int,
    target_size: int,
    context_size: int,
):
    """Extract a centered context crop and use black padding outside the source image."""
    roi = centered_target_roi(target_size, context_size)
    context_x0 = int(target_x0) - roi[0]
    context_y0 = int(target_y0) - roi[1]
    _, source_height, source_width = image.shape
    result = np.zeros((image.shape[0], context_size, context_size), dtype=image.dtype)

    source_x0 = max(0, context_x0)
    source_y0 = max(0, context_y0)
    source_x1 = min(source_width, context_x0 + context_size)
    source_y1 = min(source_height, context_y0 + context_size)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return result

    dest_x0 = source_x0 - context_x0
    dest_y0 = source_y0 - context_y0
    dest_x1 = dest_x0 + (source_x1 - source_x0)
    dest_y1 = dest_y0 + (source_y1 - source_y0)
    result[:, dest_y0:dest_y1, dest_x0:dest_x1] = image[:, source_y0:source_y1, source_x0:source_x1]
    return result


def coord_to_pixel(coord, inverse_transform):
    x, y = float(coord[0]), float(coord[1])
    px, py = inverse_transform * (x, y)
    return [float(px), float(py)]


def line_to_pixel_coords(line: LineString, inverse_transform):
    return [coord_to_pixel(coord, inverse_transform) for coord in line.coords]


def polygon_to_pixel_polygon(poly: Polygon, inverse_transform):
    exterior = [coord_to_pixel(coord, inverse_transform) for coord in poly.exterior.coords]
    interiors = [
        [coord_to_pixel(coord, inverse_transform) for coord in ring.coords]
        for ring in poly.interiors
    ]
    return Polygon(exterior, interiors)


def normalize_lane_type_code(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return str(value)


def lane_type_name(value):
    lane_type_code = normalize_lane_type_code(value)
    if lane_type_code in IGNORED_LANE_TYPE_CODES:
        return None
    return LANE_TYPE_NAMES.get(lane_type_code, "other")


def normalized_property_key(value):
    return "".join(char for char in str(value).lower() if char.isalnum())


def source_property(properties, *names):
    normalized_names = {normalized_property_key(name) for name in names}
    for key, value in (properties or {}).items():
        if normalized_property_key(key) in normalized_names:
            return value
    return None


def row_source_property(row, *names):
    if hasattr(row, "items"):
        return source_property(dict(row.items()), *names)
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return None


def intersection_type_name(properties):
    raw_type = source_property(properties, "IntersectionType", "intersection_type")
    raw_subtype = source_property(properties, "IntersectionSubType", "intersection_subtype")
    main_type = normalize_lane_type_code(raw_type)
    subtype = normalize_lane_type_code(raw_subtype)

    # Some producers omit subtype=1 from type-4 records.
    if main_type == 4 and subtype is None:
        subtype = 1
    source_pair = (main_type, subtype)
    target_type = INTERSECTION_TYPE_BY_SOURCE_PAIR.get(source_pair)
    if target_type is None and main_type == 3:
        # Compatibility with producers that encode the small-untyped class as
        # a standalone type 3 instead of the canonical pair 1-3.
        target_type = "small_untyped"
    if target_type is None:
        target_type = "other"
    return target_type, source_pair


def load_line_geometries(path: Path, crs, transform, simplify_tolerance: float):
    if not path.exists():
        return []
    gdf = gpd.read_file(path).to_crs(crs)
    lines = []
    for index, row in gdf.iterrows():
        raw_lane_type = row_source_property(row, "LaneType", "lane_type")
        source_lane_type = lane_type_name(raw_lane_type)
        if source_lane_type is None:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        geoms = []
        if isinstance(geom, LineString):
            geoms = [geom]
        elif isinstance(geom, MultiLineString):
            geoms = list(geom.geoms)
        for part_idx, line in enumerate(geoms):
            if len(line.coords) >= 2:
                lines.append({
                    "category": "centerline",
                    "geometry": line,
                    "_source_line_index": int(index),
                    "_source_part_index": part_idx,
                    "_source_lane_type": source_lane_type,
                    "_source_lane_type_code": normalize_lane_type_code(raw_lane_type),
                })
    return lines


def geojson_crs_name(payload):
    crs_info = payload.get("crs") if isinstance(payload, dict) else None
    if not isinstance(crs_info, dict):
        return "EPSG:4326"
    properties = crs_info.get("properties")
    if not isinstance(properties, dict):
        return "EPSG:4326"
    name = properties.get("name")
    if not name:
        return "EPSG:4326"
    if str(name).upper().endswith("CRS84"):
        return "EPSG:4326"
    return name


def read_geojson_features_as_gdf(path: Path, dst_crs):
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list) or not features:
        return gpd.GeoDataFrame(geometry=[], crs=dst_crs)

    rows = []
    geometries = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        rows.append(dict(properties))
        geometries.append(geom)

    src_crs = geojson_crs_name(payload)
    fallback_gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs=src_crs)
    if fallback_gdf.empty:
        return fallback_gdf.to_crs(dst_crs)
    return fallback_gdf.to_crs(dst_crs)


def load_intersection_geometries(path: Path, crs, transform, simplify_tolerance: float):
    if not path.exists():
        return []
    gdf = gpd.read_file(path).to_crs(crs)
    if len(gdf) == 0:
        gdf = read_geojson_features_as_gdf(path, crs)
    polygons = []
    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        geoms = polygon_parts(geom)
        properties = {
            key: value
            for key, value in row.items()
            if key != "geometry" and not (isinstance(value, float) and np.isnan(value))
        }
        intersection_type, source_type_pair = intersection_type_name(properties)
        for part_idx, poly in enumerate(geoms):
            if not poly.is_empty and poly.area > 0:
                polygons.append({
                    "geometry": poly,
                    "intersection_type": intersection_type,
                    "source_type_pair": source_type_pair,
                    "source_properties": properties,
                    "source_index": int(index),
                    "source_part_index": part_idx,
                })
    return polygons


def patch_window_polygon(transform, x0, y0, patch_size):
    return Polygon([
        transform * (x0, y0),
        transform * (x0 + patch_size, y0),
        transform * (x0 + patch_size, y0 + patch_size),
        transform * (x0, y0 + patch_size),
        transform * (x0, y0),
    ])


def patch_window_transform(transform, x0, y0):
    return from_origin(
        transform.xoff + x0 * transform.a,
        transform.yoff + y0 * transform.e,
        transform.a,
        abs(transform.e),
    )


def map_coord_to_local_point(coord, window_transform, patch_size):
    px, py = ~window_transform * (float(coord[0]), float(coord[1]))
    x = clamp(abs(int(round(px))), 0, patch_size - 1)
    y = clamp(abs(int(round(py))), 0, patch_size - 1)
    return [x, y]


def map_coord_to_local_float_point(coord, window_transform):
    px, py = ~window_transform * (float(coord[0]), float(coord[1]))
    return [abs(float(px)), abs(float(py))]


def round_local_float_point(point, patch_size):
    x = clamp(int(round(point[0])), 0, patch_size - 1)
    y = clamp(int(round(point[1])), 0, patch_size - 1)
    return [x, y]


def sample_local_points_by_distance(points, sample_distance_px: float):
    if sample_distance_px is None or sample_distance_px <= 0 or len(points) < 2:
        return points

    segment_lengths = []
    total_length = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        dist = math.hypot(dx, dy)
        segment_lengths.append(dist)
        total_length += dist

    if total_length <= 0:
        return points

    sample_distances = []
    current_distance = 0.0
    while current_distance < total_length:
        sample_distances.append(current_distance)
        current_distance += sample_distance_px
    if not sample_distances or not math.isclose(sample_distances[-1], total_length):
        sample_distances.append(total_length)

    out = []
    segment_idx = 0
    distance_before_segment = 0.0
    for target_distance in sample_distances:
        while (
            segment_idx < len(segment_lengths) - 1
            and distance_before_segment + segment_lengths[segment_idx] < target_distance
        ):
            distance_before_segment += segment_lengths[segment_idx]
            segment_idx += 1

        p0 = points[segment_idx]
        p1 = points[segment_idx + 1]
        segment_length = segment_lengths[segment_idx]
        if segment_length <= 0:
            out.append([float(p0[0]), float(p0[1])])
            continue
        t = (target_distance - distance_before_segment) / segment_length
        out.append([
            float(p0[0]) + (float(p1[0]) - float(p0[0])) * t,
            float(p0[1]) + (float(p1[1]) - float(p0[1])) * t,
        ])
    return out


def line_parts(geom):
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for sub in geom.geoms:
            parts.extend(line_parts(sub))
        return parts
    return []


def endpoint_type_from_map_line(original_line, clipped_endpoint, tol=1e-6):
    point = np.array([float(clipped_endpoint[0]), float(clipped_endpoint[1])])
    original_start = np.array([float(original_line.coords[0][0]), float(original_line.coords[0][1])])
    original_end = np.array([float(original_line.coords[-1][0]), float(original_line.coords[-1][1])])
    if np.linalg.norm(point - original_start) <= tol or np.linalg.norm(point - original_end) <= tol:
        return "inside"
    return "cut"


def clip_lanes_to_patch(lines, transform, x0, y0, patch_size, line_sample_distance_px=0.0):
    window_polygon = patch_window_polygon(transform, x0, y0, patch_size)
    window_transform = patch_window_transform(transform, x0, y0)
    results = []
    for idx, line in enumerate(lines):
        geom = line["geometry"]
        if not geom.intersects(window_polygon):
            continue
        clipped = geom.intersection(window_polygon)
        for part_idx, clipped_line in enumerate(line_parts(clipped)):
            if clipped_line.is_empty or len(clipped_line.coords) < 2:
                continue
            local_float_points = [
                map_coord_to_local_float_point(coord, window_transform)
                for coord in clipped_line.coords
            ]
            local_float_points = sample_local_points_by_distance(local_float_points, line_sample_distance_px)
            local_points = [
                round_local_float_point(point, patch_size)
                for point in local_float_points
            ]
            local_points = dedupe_points(local_points)
            if len(local_points) < 2:
                continue
            if np.linalg.norm(np.array(local_points[0]) - np.array(local_points[-1])) < 1:
                continue
            global_pixel_points = [[point[0] + x0, point[1] + y0] for point in local_points]
            results.append({
                "category": "centerline",
                "lane_type": line.get("_source_lane_type", "other"),
                "start_type": endpoint_type_from_map_line(geom, clipped_line.coords[0]),
                "end_type": endpoint_type_from_map_line(geom, clipped_line.coords[-1]),
                "points": local_points,
                "_source_line_index": line.get("_source_line_index", idx),
                "_source_part_index": line.get("_source_part_index", part_idx),
                "_source_lane_type": line.get("_source_lane_type", "common"),
                "_source_lane_type_code": line.get("_source_lane_type_code"),
                "_source_points": global_pixel_points,
                "_patch_x0": x0,
                "_patch_y0": y0,
            })
    return results


def region_code(x, y, xmin, ymin, xmax, ymax):
    code = INSIDE
    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP
    return code


def clip_segment(p0, p1, xmin, ymin, xmax, ymax):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    c0 = region_code(x0, y0, xmin, ymin, xmax, ymax)
    c1 = region_code(x1, y1, xmin, ymin, xmax, ymax)
    p0_cut = False
    p1_cut = False

    while True:
        if not (c0 | c1):
            return {"p0": [x0, y0], "p1": [x1, y1], "p0_cut": p0_cut, "p1_cut": p1_cut}
        if c0 & c1:
            return None

        out = c0 or c1
        if out & TOP:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif out & BOTTOM:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif out & RIGHT:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin

        if out == c0:
            x0, y0 = x, y
            p0_cut = True
            c0 = region_code(x0, y0, xmin, ymin, xmax, ymax)
        else:
            x1, y1 = x, y
            p1_cut = True
            c1 = region_code(x1, y1, xmin, ymin, xmax, ymax)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def round_local_point(point, x0, y0, patch_size):
    x = clamp(int(round(point[0] - x0)), 0, patch_size - 1)
    y = clamp(int(round(point[1] - y0)), 0, patch_size - 1)
    return [x, y]


def local_point_with_cut(point, is_cut, x0, y0, patch_size):
    return {"point": round_local_point(point, x0, y0, patch_size), "cut": bool(is_cut)}


def dedupe_points(points):
    out = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    return out


def dedupe_flagged_points(items):
    out = []
    for item in items:
        if out and item["point"] == out[-1]["point"]:
            out[-1]["cut"] = out[-1]["cut"] or item["cut"]
            continue
        out.append(item)
    return out


def clip_polyline_to_patch(line, x0, y0, patch_size, source_line_index=None):
    points = line.get("points") or []
    if len(points) < 2:
        return []
    xmin, ymin = x0, y0
    xmax, ymax = x0 + patch_size - 1, y0 + patch_size - 1
    clipped_lines = []
    current = []

    for p0, p1 in zip(points[:-1], points[1:]):
        clipped = clip_segment(p0, p1, xmin, ymin, xmax, ymax)
        if clipped is None:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = []
            continue

        fp0 = local_point_with_cut(clipped["p0"], clipped["p0_cut"], x0, y0, patch_size)
        fp1 = local_point_with_cut(clipped["p1"], clipped["p1_cut"], x0, y0, patch_size)
        if fp0["point"] == fp1["point"]:
            continue

        if current and current[-1]["point"] == fp0["point"]:
            current[-1]["cut"] = current[-1]["cut"] or fp0["cut"]
            current.append(fp1)
        else:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = [fp0, fp1]

        if clipped["p1_cut"]:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = []

    if len(current) >= 2:
        clipped_lines.append(current)

    results = []
    for flagged_pts in clipped_lines:
        flagged_pts = dedupe_flagged_points(flagged_pts)
        if len(flagged_pts) < 2:
            continue
        pts = [item["point"] for item in flagged_pts]
        results.append({
            "category": "centerline",
            "start_type": "cut" if flagged_pts[0]["cut"] else "inside",
            "end_type": "cut" if flagged_pts[-1]["cut"] else "inside",
            "points": pts,
            "_source_line_index": source_line_index if source_line_index is not None else line.get("_source_line_index"),
            "_source_points": points,
            "_patch_x0": x0,
            "_patch_y0": y0,
        })
    return results


def polygon_parts(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for sub in geom.geoms:
            parts.extend(polygon_parts(sub))
        return parts
    return []


def local_ring_points(poly: Polygon, x0, y0, patch_size, window_transform=None):
    if window_transform is None:
        points = [round_local_point(point, x0, y0, patch_size) for point in poly.exterior.coords]
    else:
        points = [map_coord_to_local_point(point, window_transform, patch_size) for point in poly.exterior.coords]
    points = dedupe_points(points)
    if len(points) >= 2 and points[0] != points[-1]:
        points.append(points[0])
    return points


def clip_intersections_to_patch(intersections, x0, y0, patch_size, transform=None):
    if transform is None:
        bbox = box(x0, y0, x0 + patch_size - 1, y0 + patch_size - 1)
        window_transform = None
    else:
        bbox = patch_window_polygon(transform, x0, y0, patch_size)
        window_transform = patch_window_transform(transform, x0, y0)
    results = []
    for idx, item in enumerate(intersections):
        geom = item["geometry"]
        if not geom.intersects(bbox):
            continue
        clipped = geom.intersection(bbox)
        is_cut = not geom.difference(bbox).is_empty
        for part_idx, poly in enumerate(polygon_parts(clipped)):
            if poly.is_empty or poly.area <= 0:
                continue
            pts = local_ring_points(poly, x0, y0, patch_size, window_transform=window_transform)
            if len(pts) < 4:
                continue
            results.append({
                "category": "intersection",
                "intersection_type": item.get("intersection_type"),
                "is_cut": bool(is_cut),
                "points": pts,
                "_source_intersection_index": item.get("source_index", idx),
                "_source_part_index": item.get("source_part_index", part_idx),
                "_source_properties": item.get("source_properties", {}),
                "_source_type_pair": item.get("source_type_pair"),
                "_patch_x0": x0,
                "_patch_y0": y0,
            })
    return results


def is_near(value, target, tol):
    return abs(float(value) - float(target)) <= tol


def squared_distance(a, b):
    return (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2


def shift_neighbor_point_to_current(point, side, patch_size):
    x, y = int(round(point[0])), int(round(point[1]))
    if side == "left":
        return [x - patch_size, y]
    if side == "top":
        return [x, y - patch_size]
    return [x, y]


def source_trace_points(line, side, boundary_local, boundary_at_end, patch_size, max_points):
    source_points = line.get("_source_points") or []
    if len(source_points) < 2:
        return []
    patch_x0 = line.get("_patch_x0", 0)
    patch_y0 = line.get("_patch_y0", 0)
    boundary_global = [boundary_local[0] + patch_x0, boundary_local[1] + patch_y0]
    nearest_idx = min(range(len(source_points)), key=lambda idx: squared_distance(source_points[idx], boundary_global))
    if boundary_at_end:
        start_idx = max(0, nearest_idx - (max_points - 1))
        selected = source_points[start_idx:nearest_idx + 1]
        if not selected or squared_distance(selected[-1], boundary_global) > 1.0:
            selected = selected[-(max_points - 1):] + [boundary_global]
    else:
        end_idx = min(len(source_points), nearest_idx + max_points)
        selected = source_points[nearest_idx:end_idx]
        if not selected or squared_distance(selected[0], boundary_global) > 1.0:
            selected = [boundary_global] + selected[1:max_points]
        selected = list(reversed(selected))
    local_points = [[point[0] - patch_x0, point[1] - patch_y0] for point in selected[-max_points:]]
    return [shift_neighbor_point_to_current(point, side, patch_size) for point in local_points]


def make_trace_from_line(line, side, patch_size, max_points, boundary_tol):
    points = line.get("points") or []
    if len(points) < 2:
        return None
    if side == "left":
        boundary = lambda p: p[0] >= patch_size - 1 - boundary_tol
    elif side == "top":
        boundary = lambda p: p[1] >= patch_size - 1 - boundary_tol
    else:
        return None

    if line.get("end_type") == "cut" and boundary(points[-1]):
        trace_points = source_trace_points(line, side, points[-1], True, patch_size, max_points)
        if not trace_points:
            trace_points = [shift_neighbor_point_to_current(point, side, patch_size) for point in points[-max_points:]]
    elif line.get("start_type") == "cut" and boundary(points[0]):
        trace_points = source_trace_points(line, side, points[0], False, patch_size, max_points)
        if not trace_points:
            trace_points = [shift_neighbor_point_to_current(point, side, patch_size) for point in reversed(points[:max_points])]
    else:
        return None
    trace_points = dedupe_points(trace_points[-max_points:])
    if not trace_points:
        return None
    return {"side": side, "points": trace_points}


def build_incoming_traces(patch_lines_by_rc, row, col, patch_size, max_traces, trace_points, boundary_tol):
    traces = []
    for side, lines in [
        ("left", patch_lines_by_rc.get((row, col - 1), [])),
        ("top", patch_lines_by_rc.get((row - 1, col), [])),
    ]:
        side_count = 0
        for line in lines:
            trace = make_trace_from_line(line, side, patch_size, trace_points, boundary_tol)
            if trace is None:
                continue
            trace["id"] = f"{'L' if side == 'left' else 'T'}{side_count}"
            traces.append(trace)
            side_count += 1
            if side_count >= max_traces:
                break
    return traces


def boundary_points_for_intersection(intersection, side, patch_size, boundary_tol, max_points):
    if not intersection.get("is_cut"):
        return []
    points = dedupe_points(intersection.get("points") or [])
    if side == "left":
        selected = [point for point in points if is_near(point[0], patch_size - 1, boundary_tol)]
    elif side == "top":
        selected = [point for point in points if is_near(point[1], patch_size - 1, boundary_tol)]
    else:
        return []
    selected = dedupe_points(selected)
    if len(selected) > max_points:
        if max_points == 1:
            selected = [selected[len(selected) // 2]]
        else:
            step = (len(selected) - 1) / (max_points - 1)
            selected = [selected[round(i * step)] for i in range(max_points)]
    return [shift_neighbor_point_to_current(point, side, patch_size) for point in selected]


def build_incoming_intersections(patch_lines_by_rc, row, col, patch_size, max_hints, hint_points, boundary_tol):
    hints = []
    for side, lines in [
        ("left", patch_lines_by_rc.get((row, col - 1), [])),
        ("top", patch_lines_by_rc.get((row - 1, col), [])),
    ]:
        side_count = 0
        for line in lines:
            if line.get("category") != "intersection":
                continue
            points = boundary_points_for_intersection(line, side, patch_size, boundary_tol, hint_points)
            if not points:
                continue
            prefix = "IL" if side == "left" else "IT"
            hints.append({"id": f"{prefix}{side_count}", "side": side, "points": points})
            side_count += 1
            if side_count >= max_hints:
                break
    return hints


def line_side_priority(line, patch_size, boundary_tol):
    points = line.get("points") or []
    if line.get("category") != "centerline" or len(points) < 2:
        return 2, line
    left_start = line.get("start_type") == "cut" and is_near(points[0][0], 0, boundary_tol)
    left_end = line.get("end_type") == "cut" and is_near(points[-1][0], 0, boundary_tol)
    top_start = line.get("start_type") == "cut" and is_near(points[0][1], 0, boundary_tol)
    top_end = line.get("end_type") == "cut" and is_near(points[-1][1], 0, boundary_tol)
    if left_start or left_end:
        return 0, orient_line_from_endpoint(line, "start" if left_start else "end")
    if top_start or top_end:
        return 1, orient_line_from_endpoint(line, "start" if top_start else "end")
    return 2, line


def orient_line_from_endpoint(line, endpoint):
    if endpoint == "start":
        return line
    reversed_line = dict(line)
    reversed_line["points"] = list(reversed(line["points"]))
    reversed_line["start_type"], reversed_line["end_type"] = line["end_type"], line["start_type"]
    if "_source_points" in line:
        reversed_line["_source_points"] = list(reversed(line["_source_points"]))
    return reversed_line


def intersection_side_priority(line, patch_size, boundary_tol):
    points = line.get("points") or []
    if line.get("category") != "intersection" or not line.get("is_cut"):
        return 5
    if any(is_near(point[0], 0, boundary_tol) for point in points):
        return 3
    if any(is_near(point[1], 0, boundary_tol) for point in points):
        return 4
    return 5


def public_line(line):
    return {key: value for key, value in line.items() if not key.startswith("_")}


def coord_description(coord_mode: str, coord_range: int, patch_size: int) -> str:
    mode = normalize_coord_mode(coord_mode)
    if mode == COORD_MODE_NORM1000:
        return f"Coordinates use a normalized 0-{coord_range} grid over the original {patch_size}x{patch_size} image patch."
    return f"Coordinates use original patch pixel coordinates in [0,{patch_size - 1}]."


def convert_points_to_model_coord(points, patch_size: int, coord_mode: str, coord_range: int, clamp: bool):
    return [
        pixel_point_to_coord(
            point,
            patch_size,
            patch_size,
            coord_mode=coord_mode,
            coord_range=coord_range,
            clamp=clamp,
        )
        for point in points
    ]


def public_line_in_model_coord(line, patch_size: int, coord_mode: str, coord_range: int):
    item = public_line(line)
    item["points"] = convert_points_to_model_coord(
        item.get("points") or [],
        patch_size,
        coord_mode,
        coord_range,
        clamp=True,
    )
    return item


def trace_in_model_coord(trace, patch_size: int, coord_mode: str, coord_range: int):
    item = dict(trace)
    item["points"] = convert_points_to_model_coord(
        item.get("points") or [],
        patch_size,
        coord_mode,
        coord_range,
        clamp=False,
    )
    return item


def sort_target_lines(lines, patch_size, boundary_tol):
    ordered = []
    for idx, line in enumerate(lines):
        if line.get("category") == "centerline":
            priority, oriented = line_side_priority(line, patch_size, boundary_tol)
        else:
            priority, oriented = intersection_side_priority(line, patch_size, boundary_tol), line
        pts = oriented.get("points") or [[999999, 999999]]
        first = pts[0]
        ordered.append((priority, first[1], first[0], idx, oriented))
    return [item[-1] for item in sorted(ordered)]


def make_prompt(include_intersections: bool, incoming_traces, incoming_intersections=None, phase="a",
                coord_mode: str = COORD_MODE_NORM1000, coord_range: int = DEFAULT_COORD_RANGE,
                patch_size: int = 256, context_size=None, incoming_trace_point_spacing_px=None,
                incoming_intersections_full_polygon: bool = False, raw_lane_overlay: bool = False,
                pose_second_image: bool = False):
    context_size = patch_size if context_size is None else int(context_size)
    target_roi = centered_target_roi(patch_size, context_size)
    trace_json = json.dumps(incoming_traces, ensure_ascii=False, separators=(",", ":"))
    parts = ["<image>"]
    if pose_second_image:
        parts.extend([
            "<image>",
            "The first image is the BEV road-structure image.",
            (
                "The second image is a historical vehicle-trajectory image: white lines are "
                "historical vehicle trajectories on a black background."
            ),
        ])
    parts.append(TASK_TEXT)
    if context_size > patch_size:
        if normalize_coord_mode(coord_mode) == COORD_MODE_NORM1000:
            context_coord_description = (
                f"Coordinates use a normalized 0-{coord_range} grid over the {patch_size}x{patch_size} target ROI."
            )
        else:
            context_coord_description = (
                f"Coordinates use target-ROI pixel coordinates in [0,{patch_size - 1}]."
            )
        parts.extend([
            f"The input is a {context_size}x{context_size} context image centered on the target region.",
            (
                f"Predict only map elements clipped to the central {patch_size}x{patch_size} target ROI "
                f"[{target_roi[0]},{target_roi[1]},{target_roi[2]},{target_roi[3]})."
            ),
            context_coord_description,
            "Coordinates are relative to the target ROI, not the full context image.",
            "Do not output geometry that lies only outside the target ROI.",
        ])
    else:
        parts.append(coord_description(coord_mode, coord_range, patch_size))
    if raw_lane_overlay:
        parts.append(
            "The image also contains a white lane overlay predicted by a PV camera model. "
            "Do not copy it blindly when it conflicts with the visible BEV evidence."
        )
    parts.extend([
        "",
        'Return only valid JSON in the form {"lines":[...]} with no extra explanation.',
        (
            'For every centerline, include "lane_type" with exactly one of: '
            '"common" for a regular centerline, "right_turn" for a right-turn-only '
            'centerline, "waiting_area" for a waiting-area centerline, "bus_lane" '
            'for a bus-lane centerline, "main_auxiliary_connector" for a connector '
            'between main and auxiliary roads, or "other" for any remaining lane class.'
        ),
    ])
    if include_intersections:
        parts.append(
            'For every intersection, include "intersection_type" with exactly one of: '
            '"common" for a common intersection, "t_intersection" for a T-intersection, '
            '"small_untyped" for a small untyped intersection, or '
            '"t_lane_change_area" for a T-shaped lane-change area, or "other" '
            'for any remaining or unknown intersection class.'
        )
    parts.extend(["", "Incoming traces JSON:", trace_json])
    if include_intersections:
        inter_json = json.dumps(incoming_intersections or [], ensure_ascii=False, separators=(",", ":"))
        parts.extend(["", "Incoming intersections JSON:", inter_json])
    if phase == "b":
        if incoming_trace_point_spacing_px is None:
            trace_rule = (
                "Each incoming trace has 1 to 3 points. If multiple points are present, they are ordered from "
                "the previous patch interior toward the current patch boundary."
            )
        else:
            spacing = int(round(float(incoming_trace_point_spacing_px)))
            trace_rule = (
                f"Each incoming trace has exactly 3 points sampled about {spacing} pixels apart in the previous "
                "patch, ordered from the previous patch interior toward the current patch boundary."
            )
        parts.extend([
            "",
            trace_rule,
            "Incoming traces are continuity hints only; they may be incomplete or absent.",
        ])
        if include_intersections:
            if incoming_intersections_full_polygon:
                parts.append(
                    "Each incoming intersection contains the full polygon from a neighboring left/top patch, "
                    "shifted into the current patch coordinate frame."
                )
            else:
                parts.append("Each incoming intersection has 1 to 3 boundary points from neighboring patches.")
    return "\n".join(parts)


def build_sft_record(row, patch_size, include_intersections, phase, coord_mode=COORD_MODE_NORM1000,
                     coord_range=DEFAULT_COORD_RANGE, context_size=None, view_mode=None,
                     incoming_trace_point_spacing_px=None, incoming_intersections_full_polygon: bool = False,
                     raw_lane_overlay: bool = False, pose_second_image: bool = False,
                     save_raw_lane_image: bool = False):
    coord_mode = normalize_coord_mode(coord_mode)
    context_size = patch_size if context_size is None else int(context_size)
    target_roi = centered_target_roi(patch_size, context_size)
    incoming_traces = row["incoming_traces"] if phase == "b" else []
    incoming_intersections = row.get("incoming_intersections", []) if phase == "b" else []
    incoming_traces = [
        trace_in_model_coord(trace, patch_size, coord_mode, coord_range)
        for trace in incoming_traces
    ]
    incoming_intersections = [
        trace_in_model_coord(hint, patch_size, coord_mode, coord_range)
        for hint in incoming_intersections
    ]
    target_lines = [
        public_line_in_model_coord(line, patch_size, coord_mode, coord_range)
        for line in row["target_lines"]
    ]
    prompt = make_prompt(
        include_intersections,
        incoming_traces,
        incoming_intersections,
        phase=phase,
        coord_mode=coord_mode,
        coord_range=coord_range,
        patch_size=patch_size,
        context_size=context_size,
        incoming_trace_point_spacing_px=incoming_trace_point_spacing_px,
        incoming_intersections_full_polygon=incoming_intersections_full_polygon,
        raw_lane_overlay=raw_lane_overlay,
        pose_second_image=pose_second_image,
    )
    meta = dict(row["meta"])
    meta.update({
        "scan_order": "row_major_top_to_bottom_left_to_right",
        "available_neighbors": ["left", "top"],
        "train_shuffle_allowed": True,
        "trace_source_train": "gt_left_top_neighbors" if phase == "b" else "none",
        "trace_source_infer": "predicted_left_top_neighbors",
        "phase": f"phase_{phase}",
        "coord_mode": coord_mode,
        "coord_range": coord_range,
        "coord_system": coord_system_name(coord_mode, patch_size, coord_range),
        "pixel_patch_size": patch_size,
        "patch_width": patch_size,
        "patch_height": patch_size,
        "target_size": patch_size,
        "context_image_size": context_size,
        "target_roi_in_image": target_roi,
        "view_mode": view_mode or ("local" if context_size == patch_size else "context_center_roi"),
    })
    if incoming_trace_point_spacing_px is not None:
        meta["incoming_trace_point_spacing_px"] = float(incoming_trace_point_spacing_px)
        meta["incoming_trace_point_count"] = 3
    if incoming_intersections_full_polygon:
        meta["incoming_intersections_full_polygon"] = True
    if raw_lane_overlay:
        meta["raw_lane_overlay"] = True
        meta["raw_lane_overlay_source"] = "patch_tif/0_lane.tif"
    if save_raw_lane_image:
        raw_lane_image = str(row.get("raw_lane_image") or "")
        if not raw_lane_image:
            raise ValueError(f"sample {row.get('id')} is missing raw_lane_image")
        meta["raw_lane_auxiliary_image"] = True
        meta["raw_lane_image_source"] = "patch_tif/0_lane.tif"
        meta["raw_lane_image_role"] = "pv_camera_raw_lane"
    if pose_second_image:
        pose_image = str(row.get("pose_image") or "")
        if not pose_image:
            raise ValueError(f"sample {row.get('id')} is missing pose_image")
        meta["input_image_roles"] = ["bev_road_structure", "historical_vehicle_trajectory"]
        meta["pose_image_source"] = "patch_tif/0_pose.tif"
    if include_intersections:
        meta["intersection_hint_source_train"] = "gt_left_top_neighbors" if phase == "b" else "none"
    target_text = json.dumps({"lines": target_lines}, ensure_ascii=False, separators=(",", ":"))
    record = {
        "id": row["id"],
        "image": row["image"],
        "meta": meta,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target_text},
        ],
    }
    if pose_second_image:
        record["images"] = [row["image"], row["pose_image"]]
    if save_raw_lane_image:
        record["raw_lane_image"] = row["raw_lane_image"]
    return record


def target_has_lines(row):
    return bool(row.get("target_lines"))


def cap_empty_rows(rows, max_empty_ratio):
    if max_empty_ratio is None or max_empty_ratio < 0:
        return rows
    nonempty = [row for row in rows if target_has_lines(row)]
    empty = [row for row in rows if not target_has_lines(row)]
    if not nonempty:
        return []
    max_empty = int((max_empty_ratio / max(1e-8, 1.0 - max_empty_ratio)) * len(nonempty))
    return nonempty + empty[:max_empty]


def process_sample(
    sample: RawSample,
    output_root: Path,
    split_name: str,
    include_intersections: bool,
    args,
    write_images: bool = True,
    max_empty_ratio=-1.0,
):
    pose_second_image = bool(getattr(args, "pose_second_image", False))
    save_raw_lane_image = bool(getattr(args, "save_raw_lane_image", False))
    if pose_second_image and not sample.pose_tiff.is_file():
        raise FileNotFoundError(f"pose TIFF not found: {sample.pose_tiff}")
    if save_raw_lane_image and not sample.raw_lane_tiff.is_file():
        raise FileNotFoundError(f"raw lane TIFF not found: {sample.raw_lane_tiff}")
    image_arr, meta, transform, crs = read_masked_image(
        sample.image_tiff,
        sample.mask_tiff,
        sample.raw_lane_tiff,
        raw_lane_overlay=bool(getattr(args, "raw_lane_overlay", False)),
        require_raw_lane=bool(getattr(args, "require_raw_lane", False)),
        raw_lane_threshold=float(getattr(args, "raw_lane_threshold", 0.0)),
    )
    image_arr, original_image_size = pad_image_to_patch_grid(image_arr, args.patch_size)
    lines = load_line_geometries(sample.lane_geojson, crs, transform, args.simplify_tolerance)
    intersections = []
    if include_intersections and sample.intersection_geojson.exists():
        intersections = load_intersection_geometries(sample.intersection_geojson, crs, transform, args.simplify_tolerance)

    _, height, width = image_arr.shape
    patch_lines_by_rc = {}
    patch_source_meta = {}
    for y0 in range(0, height - args.patch_size + 1, args.stride):
        for x0 in range(0, width - args.patch_size + 1, args.stride):
            chunk = image_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
            nonblack_pixel_count = int(np.count_nonzero(np.any(chunk != 0, axis=0)))
            if nonblack_pixel_count == 0:
                continue
            row = y0 // args.stride
            col = x0 // args.stride
            local_lines = []
            local_lines.extend(clip_lanes_to_patch(
                lines,
                transform,
                x0,
                y0,
                args.patch_size,
                line_sample_distance_px=args.line_sample_distance_px,
            ))
            if include_intersections:
                local_lines.extend(clip_intersections_to_patch(intersections, x0, y0, args.patch_size, transform=transform))
            local_lines = sort_target_lines(local_lines, args.patch_size, args.boundary_tol)
            patch_lines_by_rc[(row, col)] = local_lines
            patch_source_meta[(row, col)] = {
                "intersection_sources": [
                    {
                        "is_cut": line.get("is_cut"),
                        "source_properties": line.get("_source_properties", {}),
                    }
                    for line in local_lines
                    if line.get("category") == "intersection"
                ]
            }

    rows = []
    patch_count = 0
    for (row, col), local_lines in sorted(patch_lines_by_rc.items()):
        if args.max_patches_per_sample is not None and patch_count >= args.max_patches_per_sample:
            break
        x0 = col * args.stride
        y0 = row * args.stride
        patch_id = f"{sample.sample_id}_r{row:03d}_c{col:03d}"
        rel_image = Path("images") / split_name / sample.sample_id / f"{patch_id}.png"
        rel_pose_image = Path("pose_images") / split_name / sample.sample_id / f"{patch_id}.png"
        rel_raw_lane_image = Path("raw_lane_images") / split_name / sample.sample_id / f"{patch_id}.png"

        incoming_traces = build_incoming_traces(
            patch_lines_by_rc, row, col, args.patch_size,
            args.max_traces_per_side, args.trace_points, args.boundary_tol,
        )
        incoming_intersections = []
        if include_intersections:
            incoming_intersections = build_incoming_intersections(
                patch_lines_by_rc, row, col, args.patch_size,
                args.max_intersections_per_side, args.intersection_hint_points, args.boundary_tol,
            )

        meta_payload = {
            "tile_id": sample.sample_id,
            "log_id": sample.sample_id,
            "patch_row": row,
            "patch_col": col,
            "row": row,
            "col": col,
            "x0": x0,
            "y0": y0,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "source_image_size": [width, height],
            "original_source_image_size": original_image_size,
            "coord_system": f"patch_local_{args.patch_size}",
            "task_mode": "state_update_centerline_intersection" if include_intersections else "state_update_centerline",
            "raw_sample_root": str(sample.root),
            "target_patch_nonblack_pixel_ratio": (
                nonblack_pixel_count / float(args.patch_size * args.patch_size)
            ),
            "raw_lane_overlay": bool(getattr(args, "raw_lane_overlay", False)),
            "raw_lane_overlay_source": "patch_tif/0_lane.tif" if bool(getattr(args, "raw_lane_overlay", False)) else "none",
        }
        record = {
            "id": patch_id,
            "image": str(rel_image),
            "tile_id": sample.sample_id,
            "patch_row": row,
            "patch_col": col,
            "base_patch_box_full": [x0, y0, x0 + args.patch_size, y0 + args.patch_size],
            "incoming_traces": incoming_traces,
            "incoming_intersections": incoming_intersections,
            "target_lines": [public_line(line) for line in local_lines],
            "meta": meta_payload,
            **patch_source_meta[(row, col)],
        }
        if pose_second_image:
            record["pose_image"] = str(rel_pose_image)
            record["meta"]["pose_second_image"] = True
            record["meta"]["pose_image_source"] = "patch_tif/0_pose.tif"
        if save_raw_lane_image:
            record["raw_lane_image"] = str(rel_raw_lane_image)
            record["meta"]["raw_lane_auxiliary_image"] = True
            record["meta"]["raw_lane_image_source"] = "patch_tif/0_lane.tif"
        rows.append(record)
        patch_count += 1
    rows = cap_empty_rows(rows, max_empty_ratio)
    if write_images:
        pose_arr = None
        if pose_second_image:
            pose_arr = read_masked_binary_image(
                sample.pose_tiff,
                sample.mask_tiff,
                threshold=float(getattr(args, "pose_threshold", 0.0)),
            )
            pose_arr, _ = pad_image_to_patch_grid(pose_arr, args.patch_size)
        raw_lane_arr = None
        if save_raw_lane_image:
            raw_lane_arr = read_masked_binary_image(
                sample.raw_lane_tiff,
                sample.mask_tiff,
                threshold=float(getattr(args, "raw_lane_threshold", 0.0)),
            )
            raw_lane_arr, _ = pad_image_to_patch_grid(raw_lane_arr, args.patch_size)
        for row in rows:
            x0 = row["meta"]["x0"]
            y0 = row["meta"]["y0"]
            out_image = output_root / row["image"]
            out_image.parent.mkdir(parents=True, exist_ok=True)
            chunk = image_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
            image_chunk_to_pil(chunk).save(out_image)
            if pose_arr is not None:
                out_pose = output_root / row["pose_image"]
                out_pose.parent.mkdir(parents=True, exist_ok=True)
                pose_chunk = pose_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
                image_chunk_to_pil(pose_chunk).save(out_pose)
            if raw_lane_arr is not None:
                out_raw_lane = output_root / row["raw_lane_image"]
                out_raw_lane.parent.mkdir(parents=True, exist_ok=True)
                raw_lane_chunk = raw_lane_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
                image_chunk_to_pil(raw_lane_chunk).save(out_raw_lane)
    return rows


def empty_ratio_for_phase_split(args, phase: str, split_name: str):
    if phase == "a":
        value = getattr(args, f"phase_a_{split_name}_max_empty_ratio")
        if value is None:
            return args.max_empty_ratio
        return value
    return args.phase_b_max_empty_ratio


def validate_rows(rows, include_intersections, patch_size, require_semantic_types=False):
    errors = []
    for row in rows:
        for line in row.get("target_lines", []):
            category = line.get("category")
            points = line.get("points") or []
            for point in points:
                if len(point) != 2 or not all(isinstance(v, int) for v in point):
                    errors.append(f"{row['id']}: invalid point {point}")
                elif not (0 <= point[0] < patch_size and 0 <= point[1] < patch_size):
                    errors.append(f"{row['id']}: out-of-range point {point}")
            if category == "centerline":
                if line.get("start_type") not in {"cut", "inside"} or line.get("end_type") not in {"cut", "inside"}:
                    errors.append(f"{row['id']}: invalid centerline endpoint type")
                if require_semantic_types and line.get("lane_type") not in ALLOWED_LANE_TYPES:
                    errors.append(
                        f"{row['id']}: centerline missing/invalid lane_type={line.get('lane_type')!r}"
                    )
            elif category == "intersection":
                if include_intersections and not isinstance(line.get("is_cut"), bool):
                    errors.append(f"{row['id']}: intersection missing boolean is_cut")
                if len(points) < 4 or points[0] != points[-1]:
                    errors.append(f"{row['id']}: intersection is not closed")
                if require_semantic_types and line.get("intersection_type") not in ALLOWED_INTERSECTION_TYPES:
                    errors.append(
                        f"{row['id']}: intersection missing/unsupported intersection_type="
                        f"{line.get('intersection_type')!r}, source_pair={line.get('_source_type_pair')!r}"
                    )
            else:
                errors.append(f"{row['id']}: unsupported category {category}")
        for trace in row.get("incoming_traces", []):
            if len(trace.get("points", [])) < 1:
                errors.append(f"{row['id']}: centerline trace has no points")
        for hint in row.get("incoming_intersections", []):
            if len(hint.get("points", [])) < 1:
                errors.append(f"{row['id']}: intersection hint has no points")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"dataset validation failed with {len(errors)} errors:\n{preview}")


def semantic_target_counts(lines, sample_id="<unknown>", strict=True):
    counts = Counter()
    errors = []
    for line_index, line in enumerate(lines or []):
        category = line.get("category")
        if category == "centerline":
            lane_type = line.get("lane_type")
            counts[f"lane_type:{lane_type}"] += 1
            if lane_type not in ALLOWED_LANE_TYPES:
                errors.append(f"line[{line_index}] invalid lane_type={lane_type!r}")
        elif category == "intersection":
            intersection_type = line.get("intersection_type")
            counts[f"intersection_type:{intersection_type}"] += 1
            if intersection_type not in ALLOWED_INTERSECTION_TYPES:
                errors.append(
                    f"line[{line_index}] invalid intersection_type={intersection_type!r}"
                )
        else:
            errors.append(f"line[{line_index}] unsupported category={category!r}")
    if strict and errors:
        raise ValueError(
            f"semantic target validation failed for sample={sample_id}: " + "; ".join(errors[:20])
        )
    return counts


def semantic_sft_record_counts(record, strict=True, require_prompt=False):
    conversations = record.get("conversations") or []
    if require_prompt:
        human = next(
            (
                str(item.get("value", ""))
                for item in conversations
                if str(item.get("from", "")).lower() in {"human", "user"}
            ),
            "",
        )
        missing_prompt_fields = [
            field for field in ("lane_type", "intersection_type") if field not in human
        ]
        if missing_prompt_fields:
            raise ValueError(
                f"sample={record.get('id')} prompt is missing semantic field instructions: "
                f"{missing_prompt_fields}"
            )
    assistant = next(
        (
            item.get("value")
            for item in conversations
            if str(item.get("from", "")).lower() in {"gpt", "assistant"}
        ),
        None,
    )
    if isinstance(assistant, str):
        try:
            assistant = json.loads(assistant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"sample={record.get('id')} assistant target is invalid JSON") from exc
    if not isinstance(assistant, dict) or not isinstance(assistant.get("lines"), list):
        raise ValueError(f"sample={record.get('id')} assistant target has no lines array")
    return semantic_target_counts(
        assistant["lines"],
        sample_id=str(record.get("id", "<unknown>")),
        strict=strict,
    )


def build_dataset(include_intersections: bool, args):
    require_geo_dependencies()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    samples = discover_samples(
        input_root,
        include_intersections=include_intersections,
        delete_archives=not args.keep_archives,
        limit_samples=args.limit_samples,
        require_intersection_features=False,
        archive_workers=args.archive_workers,
    )
    if not samples:
        raise FileNotFoundError(f"no valid samples found under {input_root}")
    discovered_intersection_availability = intersection_availability(samples) if include_intersections else {}

    eligible_samples = []
    dropped_empty_sample_ids = []
    for sample in tqdm(samples, desc="precheck raw samples", unit="sample"):
        preview_rows = process_sample(
            sample,
            output_root,
            "_precheck",
            include_intersections,
            args,
            write_images=False,
            max_empty_ratio=-1.0,
        )
        if preview_rows:
            eligible_samples.append(sample)
        else:
            dropped_empty_sample_ids.append(sample.sample_id)
    if not eligible_samples:
        raise ValueError(
            f"all {len(samples)} discovered raw samples produced zero usable patches; "
            "check lane/intersection labels, masks, --max-empty-ratio, and --max-patches-per-sample."
        )

    train_samples, eval_samples, test_samples = split_samples(
        eligible_samples,
        args.train_ratio,
        args.eval_ratio,
        args.eval_count,
        args.split_seed,
    )
    wants_eval = args.eval_count > 0 or (args.eval_count < 0 and args.eval_ratio > 0)
    if not args.allow_empty_splits and (not train_samples or not test_samples or (wants_eval and not eval_samples)):
        raise ValueError(
            "empty split after filtering zero-patch raw samples: "
            f"train={len(train_samples)}, eval={len(eval_samples)}, test={len(test_samples)}. "
            "Increase --limit-samples, lower/disable eval, or pass --allow-empty-splits for a format-only smoke run."
        )

    split_manifest = {
        "split_unit": "raw_sample_folder",
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "discovered_raw_sample_count": len(samples),
        "eligible_raw_sample_count": len(eligible_samples),
        "dropped_empty_raw_sample_count": len(dropped_empty_sample_ids),
        "dropped_empty_raw_sample_ids": dropped_empty_sample_ids,
        "actual_train_ratio": len(train_samples) / len(eligible_samples),
        "actual_eval_ratio": len(eval_samples) / len(eligible_samples),
        "actual_test_ratio": len(test_samples) / len(eligible_samples),
        "split_seed": args.split_seed,
        "eval_split_unit": "raw_sample_folder",
        "eval_count": args.eval_count,
        "include_intersections": include_intersections,
        "intersection_files_required": False,
        "intersection_availability": discovered_intersection_availability,
        "coord_mode": args.coord_mode,
        "coord_range": args.coord_range,
        "empty_patch_filtering": {
            "legacy_max_empty_ratio": args.max_empty_ratio,
            "phase_a_train_max_empty_ratio": empty_ratio_for_phase_split(args, "a", "train"),
            "phase_a_eval_max_empty_ratio": empty_ratio_for_phase_split(args, "a", "eval"),
            "phase_a_test_max_empty_ratio": empty_ratio_for_phase_split(args, "a", "test"),
            "phase_b_max_empty_ratio": empty_ratio_for_phase_split(args, "b", "train"),
        },
        "point_generation": {
            "simplify_tolerance": args.simplify_tolerance,
            "line_sample_distance_px": args.line_sample_distance_px,
        },
        "allow_empty_splits": args.allow_empty_splits,
        "train_ids": [sample.sample_id for sample in train_samples],
        "eval_ids": [sample.sample_id for sample in eval_samples],
        "test_ids": [sample.sample_id for sample in test_samples],
    }
    write_json(output_root / "split_manifest.json", split_manifest)

    split_full_rows = {}
    for split_name, split_samples_list in [("train", train_samples), ("eval", eval_samples), ("test", test_samples)]:
        rows = []
        for sample in tqdm(split_samples_list, desc=f"build {split_name} patches", unit="sample"):
            rows.extend(process_sample(
                sample,
                output_root,
                split_name,
                include_intersections,
                args,
                max_empty_ratio=-1.0,
            ))
        validate_rows(rows, include_intersections, args.patch_size)
        split_full_rows[split_name] = rows

    phase_split_rows = {}
    for phase in ["a", "b"]:
        phase_dir = output_root / f"phase_{phase}"
        for split_name, rows in tqdm(list(split_full_rows.items()), desc=f"write phase_{phase} jsonl", unit="split"):
            max_empty_ratio = empty_ratio_for_phase_split(args, phase, split_name)
            phase_rows = cap_empty_rows(rows, max_empty_ratio)
            validate_rows(phase_rows, include_intersections, args.patch_size)
            phase_split_rows[(phase, split_name)] = phase_rows
            sft_rows = [
                build_sft_record(
                    row,
                    args.patch_size,
                    include_intersections,
                    phase,
                    coord_mode=args.coord_mode,
                    coord_range=args.coord_range,
                    raw_lane_overlay=bool(getattr(args, "raw_lane_overlay", False)),
                    pose_second_image=bool(getattr(args, "pose_second_image", False)),
                    save_raw_lane_image=bool(getattr(args, "save_raw_lane_image", False)),
                )
                for row in phase_rows
            ]
            write_jsonl(phase_dir / f"{split_name}.jsonl", sft_rows)
            write_jsonl(phase_dir / f"meta_{split_name}.jsonl", phase_rows)

    info = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "task": "lane_intersection" if include_intersections else "lane_only",
        "num_discovered_raw_samples": len(samples),
        "num_raw_samples": len(eligible_samples),
        "num_dropped_empty_raw_samples": len(dropped_empty_sample_ids),
        "num_train_raw_samples": len(train_samples),
        "num_eval_raw_samples": len(eval_samples),
        "num_test_raw_samples": len(test_samples),
        "num_train_patches": len(split_full_rows["train"]),
        "num_eval_patches": len(split_full_rows["eval"]),
        "num_test_patches": len(split_full_rows["test"]),
        "phase_a_num_train_patches": len(phase_split_rows[("a", "train")]),
        "phase_a_num_eval_patches": len(phase_split_rows[("a", "eval")]),
        "phase_a_num_test_patches": len(phase_split_rows[("a", "test")]),
        "phase_b_num_train_patches": len(phase_split_rows[("b", "train")]),
        "phase_b_num_eval_patches": len(phase_split_rows[("b", "eval")]),
        "phase_b_num_test_patches": len(phase_split_rows[("b", "test")]),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "coord_mode": args.coord_mode,
        "coord_range": args.coord_range,
        "coord_system": coord_system_name(args.coord_mode, args.patch_size, args.coord_range),
        "max_empty_ratio": args.max_empty_ratio,
        "empty_patch_filtering": split_manifest["empty_patch_filtering"],
        "point_generation": split_manifest["point_generation"],
        "eval_ratio": args.eval_ratio,
        "eval_count": args.eval_count,
        "intersection_files_required": False,
        "allow_empty_intersection_files": True if include_intersections else args.allow_empty_intersection_files,
        "intersection_availability": discovered_intersection_availability,
        "allow_empty_splits": args.allow_empty_splits,
        "phase_a_train_jsonl": str(output_root / "phase_a" / "train.jsonl"),
        "phase_a_eval_jsonl": str(output_root / "phase_a" / "eval.jsonl"),
        "phase_a_test_jsonl": str(output_root / "phase_a" / "test.jsonl"),
        "phase_b_train_jsonl": str(output_root / "phase_b" / "train.jsonl"),
        "phase_b_eval_jsonl": str(output_root / "phase_b" / "eval.jsonl"),
        "phase_b_test_jsonl": str(output_root / "phase_b" / "test.jsonl"),
    }
    write_json(output_root / "dataset_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def add_common_args(parser):
    parser.add_argument("--input-root", required=True, help="Directory containing raw sample folders or .tar.gz archives.")
    parser.add_argument("--output-root", required=True, help="Output dataset directory.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--coord-mode", choices=[COORD_MODE_PIXEL, COORD_MODE_NORM1000], default=COORD_MODE_NORM1000)
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.05, help="Raw-sample ratio reserved for eval before patch generation; test gets the remaining samples.")
    parser.add_argument("--eval-count", type=int, default=-1, help="Explicit raw-sample eval count; -1 uses eval-ratio.")
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=0.1,
        help=(
            "Legacy empty-target patch cap. It now applies only to phase_a/train "
            "unless --phase-a-train-max-empty-ratio is set. Use -1 to keep all non-black patches."
        ),
    )
    parser.add_argument(
        "--phase-a-train-max-empty-ratio",
        type=float,
        default=None,
        help="Empty-target patch cap for phase_a/train. Default: use --max-empty-ratio.",
    )
    parser.add_argument(
        "--phase-a-eval-max-empty-ratio",
        type=float,
        default=-1.0,
        help="Empty-target patch cap for phase_a/eval. -1 keeps all non-black patches for complete evaluation maps.",
    )
    parser.add_argument(
        "--phase-a-test-max-empty-ratio",
        type=float,
        default=-1.0,
        help="Empty-target patch cap for phase_a/test. -1 keeps all non-black patches for complete stitched maps.",
    )
    parser.add_argument(
        "--phase-b-max-empty-ratio",
        type=float,
        default=-1.0,
        help="Empty-target patch cap for all phase_b splits. Keep -1 so state-update neighbor chains and stitched maps stay complete.",
    )
    parser.add_argument("--boundary-tol", type=float, default=1.0)
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.0,
        help="Douglas-Peucker simplify tolerance. Default 0 disables simplification.",
    )
    parser.add_argument(
        "--line-sample-distance-px",
        type=float,
        default=0.0,
        help=(
            "Sample clipped lane target polylines at this fixed patch-pixel distance. "
            "0 keeps the original clipped vertices only."
        ),
    )
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument("--intersection-hint-points", type=int, default=3)
    parser.add_argument("--max-traces-per-side", type=int, default=8)
    parser.add_argument("--max-intersections-per-side", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--max-patches-per-sample", type=int, default=None)
    parser.add_argument(
        "--allow-empty-intersection-files",
        action="store_true",
        help=(
            "Deprecated no-op. Lane+intersection mode now always keeps raw samples "
            "with empty or missing Intersection.geojson and treats them as lane-only targets."
        ),
    )
    parser.add_argument(
        "--allow-empty-splits",
        action="store_true",
        help="Allow empty eval/test JSONL outputs. Intended only for one-sample format smoke tests.",
    )
    parser.add_argument("--keep-archives", action="store_true", help="Do not delete .tar.gz archives after successful extraction.")
    parser.add_argument(
        "--archive-workers",
        type=int,
        default=DEFAULT_ARCHIVE_WORKERS,
        help="Number of independent .tar.gz archives to extract concurrently.",
    )


def run_cli(include_intersections: bool, description: str):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    args = parser.parse_args()
    build_dataset(include_intersections, args)
