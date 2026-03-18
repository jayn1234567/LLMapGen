import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_PROMPT_TEMPLATE = """<image>
Please construct the road map in the current patch.
The previous state contains road segments cut from already processed neighboring patches.
Continue these cut segments when needed and also predict all road segments inside the current patch.
Previous state:
{state_json}"""


@dataclass
class PatchRecord:
    token: str
    image_name: str
    image_path: Path
    split: str
    source_split: str
    source_image: str
    augmentation_type: str
    crop_box: Dict[str, float]
    center_x: float
    center_y: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: int
    height: int
    raw_local_lines: List[np.ndarray]
    annotation_num_lines: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-style 4096-source / 896-crop state-update SFT data."
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--ann-json", type=str, default=None)
    parser.add_argument("--manifest-json", type=str, default=None)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--allowed-augmentations", type=str, nargs="+", default=["base"])
    parser.add_argument("--exclude-rotations", action="store_true")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--cluster-tol-px", type=float, default=8.0)
    parser.add_argument("--resample-step-px", type=float, default=12.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--trace-points", type=int, default=8)
    parser.add_argument("--accepted-categories", type=str, nargs="+", default=["lane_line", "virtual_line", "curb"])
    parser.add_argument("--output-category", type=str, default="road")
    parser.add_argument("--family-manifest-out", type=str, default=None)
    parser.add_argument("--family-manifest-in", type=str, default=None)
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--strict-local-alignment", action="store_true")
    parser.add_argument("--export-alignment-visualizations", action="store_true")
    parser.add_argument("--export-state-visualizations", action="store_true")
    parser.add_argument("--viz-patch-ids", type=int, nargs="+", default=[])
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--system-prompt-file", type=str, default="")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-json at {path}")
    return data


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_opensatmap_category(name: str) -> str:
    value = str(name).strip().lower()
    if value == "lane line":
        return "lane_line"
    if value == "virtual line":
        return "virtual_line"
    if value == "curb":
        return "curb"
    return value.replace(" ", "_")


def token_key(name: str) -> str:
    value = str(name)
    if value.endswith("_satellite.png"):
        return value[: -len("_satellite.png")]
    if value.endswith("_satellite.jpg"):
        return value[: -len("_satellite.jpg")]
    return Path(value).stem


def is_rotation_variant(token: str) -> bool:
    return bool(re.search(r"__rot(090|180|270)$", str(token)))


def dedup_points(points: Sequence[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - out[-1])) > float(eps):
            out.append(arr[idx])
    return np.asarray(out, dtype=np.float32)


def clamp_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(width - 1))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(height - 1))
    return arr


def simplify_for_json(points_xy: np.ndarray, width: int, height: int) -> List[List[int]]:
    arr = clamp_points(points_xy, width=width, height=height)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    arr = np.rint(arr).astype(np.int32)
    arr = dedup_points(arr.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in arr.tolist()]


def resample_polyline(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return pts[:1]
    step = max(float(step_px), 1.0)
    n = max(2, int(math.floor(total / step)) + 1)
    if max_points is not None:
        n = min(n, int(max_points))
    target = np.linspace(0.0, total, n, dtype=np.float32)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    sampled: List[np.ndarray] = []
    for dist in target:
        j = int(np.searchsorted(cum, dist, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t0 = float(cum[j])
        t1 = float(cum[j + 1])
        ratio = 0.0 if t1 <= t0 else (float(dist) - t0) / (t1 - t0)
        sampled.append(pts[j] * (1.0 - ratio) + pts[j + 1] * ratio)
    return dedup_points(sampled)


def point_in_rect(point_xy: np.ndarray, rect: Tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps)


def clip_segment_liang_barsky(
    p0: np.ndarray,
    p1: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_min, y_min, x_max, y_max = rect
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    p = [-dx, dx, -dy, dy]
    q = [float(p0[0] - x_min), float(x_max - p0[0]), float(p0[1] - y_min), float(y_max - p0[1])]
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-8:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t
    c0 = np.asarray([p0[0] + u1 * dx, p0[1] + u1 * dy], dtype=np.float32)
    c1 = np.asarray([p0[0] + u2 * dx, p0[1] + u2 * dy], dtype=np.float32)
    return c0, c1


def clip_polyline_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return []
    pieces: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for idx in range(pts.shape[0] - 1):
        p0 = pts[idx]
        p1 = pts[idx + 1]
        clipped = clip_segment_liang_barsky(p0, p1, rect)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
            continue
        c0, c1 = clipped
        if not current:
            current = [c0, c1]
        else:
            if float(np.linalg.norm(current[-1] - c0)) <= 1e-3:
                current.append(c1)
            else:
                if len(current) >= 2:
                    pieces.append(dedup_points(current))
                current = [c0, c1]
        if not point_in_rect(p1, rect):
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
    if len(current) >= 2:
        pieces.append(dedup_points(current))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    if abs(x - x_min) <= tol_px:
        return "left"
    if abs(y - y_min) <= tol_px:
        return "top"
    if abs(x - x_max) <= tol_px:
        return "right"
    if abs(y - y_max) <= tol_px:
        return "bottom"
    return None


def cluster_axis(values: Sequence[float], tol_px: float) -> List[float]:
    if not values:
        return []
    ordered = sorted(float(v) for v in values)
    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= float(tol_px):
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(sum(group) / len(group)) for group in clusters]


def nearest_cluster_index(value: float, clusters: Sequence[float]) -> int:
    if not clusters:
        raise ValueError("clusters must be non-empty")
    best_idx = 0
    best_dist = abs(float(value) - float(clusters[0]))
    for idx in range(1, len(clusters)):
        dist = abs(float(value) - float(clusters[idx]))
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def collect_local_lines_from_annotation(
    ann: Dict,
    actual_width: int,
    actual_height: int,
    accepted_categories: Sequence[str],
    strict_local_alignment: bool,
) -> List[np.ndarray]:
    ann_width = int(ann.get("image_width", actual_width))
    ann_height = int(ann.get("image_height", actual_height))
    if bool(strict_local_alignment) and (ann_width != actual_width or ann_height != actual_height):
        raise ValueError(
            f"Annotation/image size mismatch: ann=({ann_width},{ann_height}) image=({actual_width},{actual_height})"
        )
    if ann_width != actual_width or ann_height != actual_height:
        sx = float(max(1, actual_width - 1)) / float(max(1, ann_width - 1))
        sy = float(max(1, actual_height - 1)) / float(max(1, ann_height - 1))
    else:
        sx = 1.0
        sy = 1.0

    cat_set = set(str(x) for x in accepted_categories)
    out: List[np.ndarray] = []
    for rec in ann.get("lines", []):
        cat = normalize_opensatmap_category(rec.get("category", ""))
        if cat_set and cat not in cat_set:
            continue
        arr = np.asarray(rec.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        pts = np.stack([arr[:, 0] * sx, arr[:, 1] * sy], axis=-1)
        pts = clamp_points(pts, width=actual_width, height=actual_height)
        pts = dedup_points(pts)
        if pts.shape[0] >= 2:
            out.append(pts)
    return out


def load_patch_records(
    dataset_root: Path,
    split: str,
    annotations: Dict,
    manifest: Dict,
    allowed_augmentations: Sequence[str],
    exclude_rotations: bool,
    accepted_categories: Sequence[str],
    strict_local_alignment: bool,
) -> List[PatchRecord]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    allowed = set(str(x) for x in allowed_augmentations)
    records: List[PatchRecord] = []
    for image_path in sorted(x for x in split_dir.iterdir() if x.is_file()):
        image_name = image_path.name
        ann = annotations.get(image_name)
        if not isinstance(ann, dict):
            continue
        stem = token_key(image_name)
        meta = manifest.get(stem, {})
        aug = str(meta.get("augmentation_type", ""))
        if allowed and aug not in allowed:
            continue
        if bool(exclude_rotations) and is_rotation_variant(stem):
            continue

        crop_box = ann.get("crop_box", {})
        if not isinstance(crop_box, dict):
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        raw_local_lines = collect_local_lines_from_annotation(
            ann=ann,
            actual_width=int(width),
            actual_height=int(height),
            accepted_categories=accepted_categories,
            strict_local_alignment=bool(strict_local_alignment),
        )
        records.append(
            PatchRecord(
                token=stem,
                image_name=image_name,
                image_path=image_path,
                split=split,
                source_split=str(ann.get("source_split", meta.get("source_split", split))),
                source_image=str(ann.get("source_image", meta.get("source_image", ""))),
                augmentation_type=aug,
                crop_box=dict(crop_box),
                center_x=float(crop_box.get("center_x")),
                center_y=float(crop_box.get("center_y")),
                x_min=float(crop_box.get("x_min")),
                y_min=float(crop_box.get("y_min")),
                x_max=float(crop_box.get("x_max")),
                y_max=float(crop_box.get("y_max")),
                width=int(width),
                height=int(height),
                raw_local_lines=raw_local_lines,
                annotation_num_lines=int(len(ann.get("lines", []))),
            )
        )
    return records


def build_candidate_families(
    records: Sequence[PatchRecord],
    grid_size: int,
    cluster_tol_px: float,
) -> List[Dict]:
    by_source: Dict[Tuple[str, str, str], List[PatchRecord]] = {}
    for record in records:
        key = (record.split, record.source_split, record.source_image)
        by_source.setdefault(key, []).append(record)

    families: List[Dict] = []
    for (split, source_split, source_image), source_records in sorted(by_source.items()):
        x_clusters = cluster_axis([x.center_x for x in source_records], tol_px=cluster_tol_px)
        y_clusters = cluster_axis([x.center_y for x in source_records], tol_px=cluster_tol_px)
        if len(x_clusters) < grid_size or len(y_clusters) < grid_size:
            continue

        cell_map: Dict[Tuple[int, int], PatchRecord] = {}
        for record in source_records:
            xi = nearest_cluster_index(record.center_x, x_clusters)
            yi = nearest_cluster_index(record.center_y, y_clusters)
            current = cell_map.get((yi, xi))
            if current is None or str(record.token) < str(current.token):
                cell_map[(yi, xi)] = record

        family_idx = 0
        for row0 in range(0, len(y_clusters) - grid_size + 1):
            for col0 in range(0, len(x_clusters) - grid_size + 1):
                patches: List[Dict] = []
                complete = True
                for local_row in range(grid_size):
                    for local_col in range(grid_size):
                        rec = cell_map.get((row0 + local_row, col0 + local_col))
                        if rec is None:
                            complete = False
                            break
                        patches.append(
                            {
                                "patch_id": int(local_row * grid_size + local_col),
                                "row": int(local_row),
                                "col": int(local_col),
                                "token": rec.token,
                                "image_name": rec.image_name,
                                "crop_box": {
                                    "x_min": rec.x_min,
                                    "y_min": rec.y_min,
                                    "x_max": rec.x_max,
                                    "y_max": rec.y_max,
                                    "center_x": rec.center_x,
                                    "center_y": rec.center_y,
                                },
                            }
                        )
                    if not complete:
                        break
                if not complete:
                    continue
                family_id = f"{Path(source_image).stem}__r{row0:02d}_c{col0:02d}_g{family_idx:03d}"
                family_idx += 1
                families.append(
                    {
                        "family_id": family_id,
                        "split": split,
                        "source_split": source_split,
                        "source_image": source_image,
                        "grid_size": grid_size,
                        "row0": int(row0),
                        "col0": int(col0),
                        "x_clusters": [float(x) for x in x_clusters[col0 : col0 + grid_size]],
                        "y_clusters": [float(y) for y in y_clusters[row0 : row0 + grid_size]],
                        "patches": patches,
                    }
                )
    return families


def load_family_manifest(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sort_lines(lines: List[Dict]) -> List[Dict]:
    return sorted(
        lines,
        key=lambda item: (
            float(item.get("points", [[1e9, 1e9]])[0][1]),
            float(item.get("points", [[1e9, 1e9]])[0][0]),
            str(item.get("source_patch", "")),
        ),
    )


def build_patch_ownership_rect_global(
    family_records: Sequence[PatchRecord],
    patch_id: int,
    grid_size: int,
) -> Tuple[float, float, float, float]:
    record = family_records[patch_id]
    row = patch_id // grid_size
    col = patch_id % grid_size
    left = record.x_min
    right = record.x_max
    top = record.y_min
    bottom = record.y_max
    if col > 0:
        left_neighbor = family_records[patch_id - 1]
        left = 0.5 * (float(record.center_x) + float(left_neighbor.center_x))
    if col < grid_size - 1:
        right_neighbor = family_records[patch_id + 1]
        right = 0.5 * (float(record.center_x) + float(right_neighbor.center_x))
    if row > 0:
        top_neighbor = family_records[patch_id - grid_size]
        top = 0.5 * (float(record.center_y) + float(top_neighbor.center_y))
    if row < grid_size - 1:
        bottom_neighbor = family_records[patch_id + grid_size]
        bottom = 0.5 * (float(record.center_y) + float(bottom_neighbor.center_y))
    return float(left), float(top), float(right), float(bottom)


def build_target_lines_for_patch(
    record: PatchRecord,
    ownership_rect_global: Tuple[float, float, float, float],
    output_category: str,
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    local_rect = (
        float(ownership_rect_global[0] - record.x_min),
        float(ownership_rect_global[1] - record.y_min),
        float(ownership_rect_global[2] - record.x_min),
        float(ownership_rect_global[3] - record.y_min),
    )
    lines: List[Dict] = []
    offset = np.asarray([record.x_min, record.y_min], dtype=np.float32)[None, :]
    for local_line in record.raw_local_lines:
        global_line = np.asarray(local_line, dtype=np.float32) + offset
        pieces = clip_polyline_to_rect(global_line, ownership_rect_global)
        for piece in pieces:
            local_piece = piece - offset
            local_piece = clamp_points(local_piece, width=record.width, height=record.height)
            local_piece = resample_polyline(local_piece, step_px=resample_step_px)
            if local_piece.ndim != 2 or local_piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(local_piece[0], rect=local_rect, tol_px=boundary_tol_px)
            end_side = point_boundary_side(local_piece[-1], rect=local_rect, tol_px=boundary_tol_px)
            lines.append(
                {
                    "category": output_category,
                    "start_type": "cut" if start_side is not None else "start",
                    "end_type": "cut" if end_side is not None else "end",
                    "points": simplify_for_json(local_piece, width=record.width, height=record.height),
                }
            )
    return sort_lines([x for x in lines if len(x.get("points", [])) >= 2])


def extract_state_lines(
    patch_id: int,
    target_lines: Sequence[Dict],
    grid_size: int,
    ownership_rect_global: Tuple[float, float, float, float],
    patch_record: PatchRecord,
    trace_points: int,
    output_category: str,
    boundary_tol_px: float,
) -> List[Dict]:
    row = int(patch_id // grid_size)
    col = int(patch_id % grid_size)
    left_neighbor = patch_id - 1 if col > 0 else None
    top_neighbor = patch_id - grid_size if row > 0 else None
    local_rect = (
        float(ownership_rect_global[0] - patch_record.x_min),
        float(ownership_rect_global[1] - patch_record.y_min),
        float(ownership_rect_global[2] - patch_record.x_min),
        float(ownership_rect_global[3] - patch_record.y_min),
    )

    state_lines: List[Dict] = []
    for line in target_lines:
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        endpoints = [
            ("start", pts[0], pts[:trace_points]),
            ("end", pts[-1], pts[::-1][:trace_points]),
        ]
        for _, endpoint, trace in endpoints:
            side = point_boundary_side(endpoint, rect=local_rect, tol_px=boundary_tol_px)
            if side == "left" and left_neighbor is not None:
                trace_json = simplify_for_json(trace, width=patch_record.width, height=patch_record.height)
                if len(trace_json) >= 2:
                    state_lines.append(
                        {
                            "source_patch": int(left_neighbor),
                            "category": output_category,
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": trace_json,
                        }
                    )
            elif side == "top" and top_neighbor is not None:
                trace_json = simplify_for_json(trace, width=patch_record.width, height=patch_record.height)
                if len(trace_json) >= 2:
                    state_lines.append(
                        {
                            "source_patch": int(top_neighbor),
                            "category": output_category,
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": trace_json,
                        }
                    )
    seen = set()
    deduped: List[Dict] = []
    for line in sort_lines(state_lines):
        key = (int(line["source_patch"]), tuple((int(p[0]), int(p[1])) for p in line["points"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def save_alignment_visualization(record: PatchRecord, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with Image.open(record.image_path) as img:
        patch = img.convert("RGB")
    draw = ImageDraw.Draw(patch)
    for line in record.raw_local_lines:
        pts = [tuple(int(round(v)) for v in p.tolist()) for p in line]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), radius=3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), radius=3)
    patch.save(out_path)


def save_state_visualization(
    record: PatchRecord,
    ownership_rect_global: Tuple[float, float, float, float],
    target_lines: Sequence[Dict],
    state_lines: Sequence[Dict],
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    with Image.open(record.image_path) as img:
        patch = img.convert("RGB")
    draw = ImageDraw.Draw(patch)
    local_rect = (
        float(ownership_rect_global[0] - record.x_min),
        float(ownership_rect_global[1] - record.y_min),
        float(ownership_rect_global[2] - record.x_min),
        float(ownership_rect_global[3] - record.y_min),
    )
    draw.rectangle(local_rect, outline=(255, 240, 0), width=2)
    for line in target_lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), radius=3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), radius=3)
    for line in state_lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 140, 40), width=4)
            draw_endpoint(draw, pts[0], (255, 90, 0), radius=4)
            draw_endpoint(draw, pts[-1], (255, 90, 0), radius=4)
    patch.save(out_path)


def to_message_record(image_rel_path: str, state_lines: Sequence[Dict], target_lines: Sequence[Dict], sample_id: str) -> Dict:
    return to_message_record_with_system(
        image_rel_path=image_rel_path,
        state_lines=state_lines,
        target_lines=target_lines,
        sample_id=sample_id,
        system_prompt="",
    )


def to_message_record_with_system(
    image_rel_path: str,
    state_lines: Sequence[Dict],
    target_lines: Sequence[Dict],
    sample_id: str,
    system_prompt: str,
) -> Dict:
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": DEFAULT_PROMPT_TEMPLATE.format(state_json=state_json)})
    messages.append({"role": "assistant", "content": target_json})
    return {
        "id": sample_id,
        "messages": messages,
        "images": [image_rel_path],
    }


def export_split(
    split: str,
    families: Sequence[Dict],
    records_by_token: Dict[str, PatchRecord],
    output_root: Path,
    max_families_per_split: int,
    output_category: str,
    resample_step_px: float,
    boundary_tol_px: float,
    trace_points: int,
    export_alignment_visualizations: bool,
    export_state_visualizations: bool,
    viz_patch_ids: Sequence[int],
    system_prompt: str,
) -> Dict[str, int]:
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    family_count = 0
    for family in families:
        if str(family.get("split")) != split:
            continue
        family_count += 1
        if max_families_per_split > 0 and family_count > max_families_per_split:
            break

        patch_items = sorted(list(family.get("patches", [])), key=lambda x: int(x.get("patch_id", 0)))
        family_records = [records_by_token[str(item["token"])] for item in patch_items]
        grid_size = int(family.get("grid_size", 4))
        for item in patch_items:
            patch_id = int(item["patch_id"])
            patch_record = records_by_token[str(item["token"])]
            ownership_rect_global = build_patch_ownership_rect_global(family_records=family_records, patch_id=patch_id, grid_size=grid_size)
            target_lines = build_target_lines_for_patch(
                record=patch_record,
                ownership_rect_global=ownership_rect_global,
                output_category=output_category,
                resample_step_px=resample_step_px,
                boundary_tol_px=boundary_tol_px,
            )
            state_lines = extract_state_lines(
                patch_id=patch_id,
                target_lines=target_lines,
                grid_size=grid_size,
                ownership_rect_global=ownership_rect_global,
                patch_record=patch_record,
                trace_points=trace_points,
                output_category=output_category,
                boundary_tol_px=boundary_tol_px,
            )

            image_rel = Path("images") / split / str(family["family_id"]) / patch_record.image_name
            out_image = output_root / image_rel
            ensure_dir(out_image.parent)
            shutil.copy2(patch_record.image_path, out_image)

            if bool(export_alignment_visualizations) and (not viz_patch_ids or patch_id in viz_patch_ids):
                save_alignment_visualization(
                    record=patch_record,
                    out_path=output_root / "alignment_visualizations" / split / str(family["family_id"]) / f"p{patch_id:02d}_alignment.png",
                )
            if bool(export_state_visualizations) and (not viz_patch_ids or patch_id in viz_patch_ids):
                save_state_visualization(
                    record=patch_record,
                    ownership_rect_global=ownership_rect_global,
                    target_lines=target_lines,
                    state_lines=state_lines,
                    out_path=output_root / "state_visualizations" / split / str(family["family_id"]) / f"p{patch_id:02d}_state.png",
                )

            sample_id = f"{family['family_id']}_p{patch_id:02d}"
            rows.append(
                to_message_record_with_system(
                    image_rel_path=image_rel.as_posix(),
                    state_lines=state_lines,
                    target_lines=target_lines,
                    sample_id=sample_id,
                    system_prompt=system_prompt,
                )
            )
            meta_rows.append(
                {
                    "id": sample_id,
                    "split": split,
                    "family_id": family["family_id"],
                    "source_image": family["source_image"],
                    "source_split": family["source_split"],
                    "patch_id": patch_id,
                    "row": int(item["row"]),
                    "col": int(item["col"]),
                    "scan_index": patch_id,
                    "image": image_rel.as_posix(),
                    "token": patch_record.token,
                    "crop_box": {
                        "x_min": patch_record.x_min,
                        "y_min": patch_record.y_min,
                        "x_max": patch_record.x_max,
                        "y_max": patch_record.y_max,
                        "center_x": patch_record.center_x,
                        "center_y": patch_record.center_y,
                    },
                    "ownership_rect_global": [float(x) for x in ownership_rect_global],
                    "history_patch_ids": list(range(patch_id)),
                    "state_source_patch_ids": sorted({int(x["source_patch"]) for x in state_lines}),
                    "num_state_lines": len(state_lines),
                    "num_target_lines": len(target_lines),
                    "alignment_mode": "strict_local_patch_coords",
                    "state_mode": "cut_traces_midline_handoff",
                    "target_mode": "ownership_region_map",
                    "target_coord_system": "patch_local_896",
                    "has_system_prompt": bool(str(system_prompt).strip()),
                    "state_lines": state_lines,
                    "target_lines": target_lines,
                }
            )
    count_main = write_jsonl(output_root / f"{split}.jsonl", rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", meta_rows)
    return {"families": family_count if max_families_per_split <= 0 else min(family_count, max_families_per_split), "samples": count_main, "meta_samples": count_meta}


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    ann_json = Path(args.ann_json).resolve() if args.ann_json else dataset_root / "annotations.json"
    manifest_json = Path(args.manifest_json).resolve() if args.manifest_json else dataset_root / "manifest.json"
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    annotations = load_json(ann_json)
    manifest = load_json(manifest_json)

    all_records: List[PatchRecord] = []
    for split in args.splits:
        all_records.extend(
            load_patch_records(
                dataset_root=dataset_root,
                split=str(split),
                annotations=annotations,
                manifest=manifest,
                allowed_augmentations=[str(x) for x in args.allowed_augmentations],
                exclude_rotations=bool(args.exclude_rotations),
                accepted_categories=[str(x) for x in args.accepted_categories],
                strict_local_alignment=bool(args.strict_local_alignment),
            )
        )
    records_by_token = {record.token: record for record in all_records}

    if args.family_manifest_in:
        families = load_family_manifest(Path(args.family_manifest_in).resolve())
    else:
        families = build_candidate_families(
            records=all_records,
            grid_size=int(args.grid_size),
            cluster_tol_px=float(args.cluster_tol_px),
        )
    if args.family_manifest_out:
        write_jsonl(Path(args.family_manifest_out).resolve(), families)

    system_prompt = ""
    if str(args.system_prompt_file).strip():
        system_prompt = Path(args.system_prompt_file).resolve().read_text(encoding="utf-8").strip()
    elif str(args.system_prompt).strip():
        system_prompt = str(args.system_prompt).strip()
    elif bool(args.use_system_prompt):
        system_prompt = (
            "You are a road-map reconstruction assistant for satellite-image patches.\n\n"
            "Your task is to predict the road map of the current image patch from:\n"
            "1. the satellite image patch, and\n"
            "2. the previous map state provided in the prompt.\n\n"
            "The previous state contains road segments that were cut from already processed neighboring patches.\n"
            "You must use these cut traces to maintain cross-patch continuity whenever they enter the current patch.\n\n"
            "Requirements:\n"
            "- Predict the road map for the current patch.\n"
            "- Continue incoming cut traces when appropriate.\n"
            "- Preserve geometric continuity across patch boundaries.\n"
            "- Do not add explanations, commentary, or extra text.\n"
            "- Return only valid JSON in the required schema.\n"
            "- Do not output markdown fences.\n"
            "- Do not omit required fields.\n"
            "- Keep all coordinates in the patch-local coordinate system used by the sample.\n\n"
            "If no previous state is provided, start from the current patch image only.\n"
            "If a road segment is truncated by the current patch boundary, mark the endpoint consistently with the schema."
        )

    summary: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        summary[str(split)] = export_split(
            split=str(split),
            families=families,
            records_by_token=records_by_token,
            output_root=output_root,
            max_families_per_split=int(args.max_families_per_split),
            output_category=str(args.output_category),
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
            trace_points=int(args.trace_points),
            export_alignment_visualizations=bool(args.export_alignment_visualizations),
            export_state_visualizations=bool(args.export_state_visualizations),
            viz_patch_ids=[int(x) for x in args.viz_patch_ids],
            system_prompt=system_prompt,
        )

    dataset_info = {
        "dataset_name": "unimapgen_paper4096_state_sft",
        "version": "v0",
        "source_dataset_root": str(dataset_root),
        "source_annotations": str(ann_json),
        "source_manifest": str(manifest_json),
        "grid_size": [int(args.grid_size), int(args.grid_size)],
        "num_patches_per_family": int(args.grid_size * args.grid_size),
        "allowed_augmentations": [str(x) for x in args.allowed_augmentations],
        "exclude_rotations": bool(args.exclude_rotations),
        "family_builder": {"mode": "clustered_sliding_window", "cluster_tol_px": float(args.cluster_tol_px)},
        "alignment_mode": "strict_local_patch_coords" if bool(args.strict_local_alignment) else "scale_if_needed",
        "handoff_mode": "midline_between_adjacent_patch_centers",
        "state_mode": "cut_traces_midline_handoff",
        "target_mode": "ownership_region_map",
        "coord_system": "patch_local_896",
        "use_system_prompt": bool(system_prompt),
        "system_prompt": system_prompt,
        "category_set": [str(args.output_category)],
        "accepted_categories": [str(x) for x in args.accepted_categories],
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "export_alignment_visualizations": bool(args.export_alignment_visualizations),
        "export_state_visualizations": bool(args.export_state_visualizations),
        "viz_patch_ids": [int(x) for x in args.viz_patch_ids],
        "num_candidate_families": len(families),
        "summary": summary,
    }
    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    for split, info in summary.items():
        print(f"[{split}] families={info['families']} samples={info['samples']} meta={info['meta_samples']}")
    print(f"Candidate families: {len(families)}")
    print(f"Saved dataset to {output_root}")


if __name__ == "__main__":
    main()
