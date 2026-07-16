#!/usr/bin/env python3
"""Tag and visualize hard BEV map samples from UniMapGen JSONL data.

The script uses geometry-only heuristics, so it can run on private datasets
without model inference. It reads the target JSON from conversation records,
assigns hard-case tags, writes a JSONL report, and saves visual overlays for
manual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.map_visualization import resolve_image_path as resolve_map_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing phase_a/phase_b and images/.")
    parser.add_argument("--phase", default="phase_a", choices=["phase_a", "phase_b"])
    parser.add_argument("--split", default="train", choices=["train", "eval", "test"])
    parser.add_argument("--jsonl", default="", help="Override JSONL path. Defaults to dataset-root/phase/split.jsonl.")
    parser.add_argument("--image-folder", default="", help="Image root. Defaults to dataset-root.")
    parser.add_argument("--output-dir", default="", help="Output dir. Defaults to dataset-root/hard_case_report_phase_split.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--visualize-top-k", type=int, default=80)
    parser.add_argument("--visualize-random-k", type=int, default=0)
    parser.add_argument(
        "--visualize-per-difficulty",
        type=int,
        default=0,
        help="Randomly visualize this many samples for each requested difficulty; 0 disables stratified output.",
    )
    parser.add_argument(
        "--visualize-difficulties",
        nargs="+",
        default=["easy", "medium", "hard", "very_hard"],
        choices=["easy", "medium", "hard", "very_hard"],
        help="Difficulty buckets used by --visualize-per-difficulty.",
    )
    parser.add_argument("--coord-mode", default="auto", choices=["auto", "pixel", "norm1000"])
    parser.add_argument("--coord-range", type=float, default=1000.0)
    parser.add_argument("--junction-tol", type=float, default=36.0, help="Node snapping tolerance in normalized coords.")
    parser.add_argument("--intersection-tol", type=float, default=16.0, help="Line intersection tolerance in normalized coords.")
    parser.add_argument("--dense-line-threshold", type=int, default=8)
    parser.add_argument("--dense-point-threshold", type=int, default=34)
    parser.add_argument("--long-total-length-threshold", type=float, default=3600.0)
    parser.add_argument("--many-cut-threshold", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=0.0, help="Only visualize samples with score >= this value.")
    parser.add_argument("--include-empty", action="store_true", help="Also visualize empty/easy samples if selected.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def role_of(message: dict[str, Any]) -> str:
    return str(message.get("from", message.get("role", ""))).lower()


def value_of(message: dict[str, Any]) -> Any:
    return message.get("value", message.get("content", ""))


def conversations(record: dict[str, Any]) -> list[dict[str, Any]]:
    conv = record.get("conversations") or record.get("messages") or []
    return [item for item in conv if isinstance(item, dict)] if isinstance(conv, list) else []


def extract_target_payload(record: dict[str, Any]) -> dict[str, Any]:
    for message in conversations(record):
        if role_of(message) in {"gpt", "assistant"}:
            value = value_of(message)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                text = value.strip()
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list):
                        return {"lines": parsed}
                except Exception:
                    start = min([idx for idx in (text.find("{"), text.find("[")) if idx >= 0], default=-1)
                    if start >= 0:
                        try:
                            parsed = json.loads(text[start:])
                            if isinstance(parsed, dict):
                                return parsed
                            if isinstance(parsed, list):
                                return {"lines": parsed}
                        except Exception:
                            pass
    return {"lines": []}


def normalize_lines(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        return [item for item in payload["lines"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def clean_points(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, (list, tuple)):
        return []
    out = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def category_of(item: dict[str, Any]) -> str:
    cat = str(item.get("category", "centerline")).strip().lower()
    if cat == "centerline" or cat == "center_line":
        return "centerline"
    if cat in {"intersection", "junction", "crossroad"}:
        return "intersection"
    return cat or "centerline"


def line_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


def point_sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def unit(v: tuple[float, float]) -> tuple[float, float] | None:
    n = norm(v)
    if n < 1e-6:
        return None
    return v[0] / n, v[1] / n


def line_endpoint_tangent(points: list[tuple[float, float]], which: str) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    if which == "start":
        return unit(point_sub(points[1], points[0]))
    return unit(point_sub(points[-2], points[-1]))


def cluster_endpoints(
    centerlines: list[dict[str, Any]],
    tol: float,
) -> tuple[list[tuple[float, float]], dict[tuple[int, str], int], dict[int, list[tuple[int, str]]]]:
    nodes: list[tuple[float, float]] = []
    endpoint_to_node: dict[tuple[int, str], int] = {}
    node_members: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for idx, line in enumerate(centerlines):
        pts = clean_points(line.get("points"))
        if len(pts) < 2:
            continue
        for which, point in (("start", pts[0]), ("end", pts[-1])):
            best_idx = None
            best_dist = float("inf")
            for node_idx, node in enumerate(nodes):
                dist = math.dist(point, node)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = node_idx
            if best_idx is None or best_dist > tol:
                best_idx = len(nodes)
                nodes.append(point)
            else:
                old = nodes[best_idx]
                members = len(node_members[best_idx])
                nodes[best_idx] = ((old[0] * members + point[0]) / (members + 1), (old[1] * members + point[1]) / (members + 1))
            endpoint_to_node[(idx, which)] = best_idx
            node_members[best_idx].append((idx, which))
    return nodes, endpoint_to_node, node_members


def graph_cycle_count(num_nodes: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cycles = 0
    for a, b in edges:
        if a == b:
            cycles += 1
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            cycles += 1
        else:
            parent[rb] = ra
    return cycles


def is_t_junction(node_idx: int, node_members: dict[int, list[tuple[int, str]]], centerlines: list[dict[str, Any]]) -> bool:
    members = node_members.get(node_idx, [])
    if len(members) != 3:
        return False
    dirs = []
    for line_idx, which in members:
        pts = clean_points(centerlines[line_idx].get("points"))
        tangent = line_endpoint_tangent(pts, which)
        if tangent is not None:
            dirs.append(tangent)
    if len(dirs) != 3:
        return False
    # A rough T has one pair nearly opposite and the third roughly orthogonal to that pair.
    for i in range(3):
        for j in range(i + 1, 3):
            opposite = dot(dirs[i], dirs[j]) < -0.72
            if not opposite:
                continue
            k = ({0, 1, 2} - {i, j}).pop()
            if abs(dot(dirs[i], dirs[k])) < 0.55 and abs(dot(dirs[j], dirs[k])) < 0.55:
                return True
    return False


def segment_intersection(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    if o1 == 0 and on_seg(a, c, b):
        return True
    if o2 == 0 and on_seg(a, d, b):
        return True
    if o3 == 0 and on_seg(c, a, d):
        return True
    if o4 == 0 and on_seg(c, b, d):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def count_crossings(centerlines: list[dict[str, Any]], endpoint_tol: float) -> int:
    segments: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    for idx, line in enumerate(centerlines):
        pts = clean_points(line.get("points"))
        for j in range(1, len(pts)):
            if math.dist(pts[j - 1], pts[j]) > 1e-6:
                segments.append((pts[j - 1], pts[j], idx))
    total = 0
    for i in range(len(segments)):
        a, b, line_i = segments[i]
        for j in range(i + 1, len(segments)):
            c, d, line_j = segments[j]
            if line_i == line_j:
                continue
            # Ignore shared endpoints; those are handled by graph degree.
            if min(math.dist(a, c), math.dist(a, d), math.dist(b, c), math.dist(b, d)) <= endpoint_tol:
                continue
            if segment_intersection(a, b, c, d):
                total += 1
    return total


def point_to_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ab = point_sub(b, a)
    denom = dot(ab, ab)
    if denom <= 1e-9:
        return math.dist(p, a)
    t = max(0.0, min(1.0, dot(point_sub(p, a), ab) / denom))
    proj = (a[0] + t * ab[0], a[1] + t * ab[1])
    return math.dist(p, proj)


def near_line_interior(point: tuple[float, float], target: list[tuple[float, float]], tol: float) -> bool:
    if len(target) < 2:
        return False
    if min(math.dist(point, target[0]), math.dist(point, target[-1])) < tol:
        return False
    return any(point_to_segment_distance(point, target[i - 1], target[i]) <= tol for i in range(1, len(target)))


def lane_change_like_count(centerlines: list[dict[str, Any]]) -> int:
    lengths = [line_length(clean_points(item.get("points"))) for item in centerlines]
    if not lengths:
        return 0
    median_len = sorted(lengths)[len(lengths) // 2]
    count = 0
    for idx, line in enumerate(centerlines):
        pts = clean_points(line.get("points"))
        if len(pts) < 2:
            continue
        length = lengths[idx]
        if length > max(260.0, 0.55 * median_len):
            continue
        start, end = pts[0], pts[-1]
        near_start = []
        near_end = []
        for other_idx, other in enumerate(centerlines):
            if other_idx == idx:
                continue
            other_pts = clean_points(other.get("points"))
            if near_line_interior(start, other_pts, 42.0):
                near_start.append(other_idx)
            if near_line_interior(end, other_pts, 42.0):
                near_end.append(other_idx)
        if near_start and near_end and set(near_start) != set(near_end):
            count += 1
    return count


def infer_coord_mode(lines: list[dict[str, Any]], image_size: tuple[int, int] | None, requested: str) -> str:
    if requested != "auto":
        return requested
    max_xy = 0.0
    for item in lines:
        for x, y in clean_points(item.get("points")):
            max_xy = max(max_xy, abs(x), abs(y))
    if image_size and max_xy > max(image_size) + 8:
        return "norm1000"
    if max_xy > 512:
        return "norm1000"
    return "pixel"


def record_patch_size(record: dict[str, Any], image_size: tuple[int, int] | None) -> tuple[int, int]:
    if image_size is not None:
        return max(2, int(image_size[0])), max(2, int(image_size[1]))
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    width = meta.get("patch_width") or meta.get("pixel_patch_size") or meta.get("patch_size") or 256
    height = meta.get("patch_height") or meta.get("pixel_patch_size") or meta.get("patch_size") or 256
    try:
        return max(2, int(width)), max(2, int(height))
    except (TypeError, ValueError):
        return 256, 256


def normalize_lines_for_metrics(
    lines: list[dict[str, Any]],
    coord_mode: str,
    patch_size: tuple[int, int],
    coord_range: float,
) -> list[dict[str, Any]]:
    """Put geometry on the normalized metric grid without changing the source record."""
    if coord_mode != "pixel":
        return lines
    width, height = patch_size
    max_x = max(1.0, float(width - 1))
    max_y = max(1.0, float(height - 1))
    normalized = []
    for item in lines:
        clone = dict(item)
        points = []
        for x, y in clean_points(item.get("points")):
            points.append(
                [
                    max(0.0, min(coord_range, x / max_x * coord_range)),
                    max(0.0, min(coord_range, y / max_y * coord_range)),
                ]
            )
        clone["points"] = points
        normalized.append(clone)
    return normalized


def to_pixel(point: tuple[float, float], image_size: tuple[int, int], coord_mode: str, coord_range: float) -> tuple[int, int]:
    x, y = point
    if coord_mode == "norm1000":
        width, height = image_size
        x = x / coord_range * (width - 1)
        y = y / coord_range * (height - 1)
    return int(round(x)), int(round(y))


def resolve_record_image(record: dict[str, Any], dataset_root: Path, image_folder: Path) -> Path | None:
    raw = record.get("image") or record.get("images") or record.get("image_path") or ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if not raw:
        return None
    try:
        return resolve_map_image_path(str(raw), image_folder, record)
    except Exception:
        p = Path(str(raw))
        return p if p.is_absolute() else dataset_root / p


def sample_metrics(record: dict[str, Any], image_size: tuple[int, int] | None, args: argparse.Namespace) -> dict[str, Any]:
    payload = extract_target_payload(record)
    lines = normalize_lines(payload)
    coord_mode = infer_coord_mode(lines, image_size, args.coord_mode)
    metric_lines = normalize_lines_for_metrics(
        lines,
        coord_mode=coord_mode,
        patch_size=record_patch_size(record, image_size),
        coord_range=float(args.coord_range),
    )
    centerlines = [
        item
        for item in metric_lines
        if category_of(item) == "centerline" and len(clean_points(item.get("points"))) >= 2
    ]
    intersections = [
        item
        for item in metric_lines
        if category_of(item) == "intersection" and len(clean_points(item.get("points"))) >= 3
    ]
    lengths = [line_length(clean_points(item.get("points"))) for item in centerlines]
    total_length = float(sum(lengths))
    point_count = sum(len(clean_points(item.get("points"))) for item in metric_lines)
    cut_count = sum(
        int(str(item.get("start_type", "")).lower() == "cut") + int(str(item.get("end_type", "")).lower() == "cut")
        for item in centerlines
    )
    nodes, endpoint_to_node, node_members = cluster_endpoints(centerlines, args.junction_tol)
    edges = []
    for idx, _line in enumerate(centerlines):
        a = endpoint_to_node.get((idx, "start"))
        b = endpoint_to_node.get((idx, "end"))
        if a is not None and b is not None:
            edges.append((a, b))
    degrees = {node_idx: len(members) for node_idx, members in node_members.items()}
    max_degree = max(degrees.values(), default=0)
    fork_nodes = [node_idx for node_idx, degree in degrees.items() if degree >= 3]
    t_nodes = [node_idx for node_idx in fork_nodes if is_t_junction(node_idx, node_members, centerlines)]
    cycle_count = graph_cycle_count(len(nodes), edges)
    crossing_count = count_crossings(centerlines, args.intersection_tol)
    lane_change_count = lane_change_like_count(centerlines)
    closed_intersections = sum(
        1
        for item in intersections
        if len(clean_points(item.get("points"))) >= 4
        and math.dist(clean_points(item.get("points"))[0], clean_points(item.get("points"))[-1]) <= args.junction_tol
    )

    tags: list[str] = []
    score = 0.0
    if not centerlines and not intersections:
        tags.append("empty_patch")
        score += 0.2
    if intersections:
        tags.append("intersection")
        score += 1.8 + min(len(intersections), 3) * 0.25
    if fork_nodes:
        tags.append("multi_fork")
        score += 2.2 + min(len(fork_nodes), 4) * 0.35
    if t_nodes:
        tags.append("t_intersection_like")
        score += 1.4
    if cycle_count:
        tags.append("cycle_or_loop")
        score += 1.5 + min(cycle_count, 3) * 0.25
    if lane_change_count:
        tags.append("lane_change_like")
        score += 1.2 + min(lane_change_count, 3) * 0.25
    if crossing_count:
        tags.append("crossing_lines")
        score += 1.0 + min(crossing_count, 4) * 0.2
    if len(centerlines) >= args.dense_line_threshold:
        tags.append("dense_lines")
        score += 1.0
    if point_count >= args.dense_point_threshold:
        tags.append("many_points")
        score += 0.8
    if total_length >= args.long_total_length_threshold:
        tags.append("long_total_length")
        score += 0.6
    if cut_count >= args.many_cut_threshold:
        tags.append("many_cut_edges")
        score += 0.5
    if closed_intersections:
        tags.append("closed_intersection_polygon")
        score += 0.4
    if len(tags) == 0:
        tags.append("plain")

    if score >= 5.0:
        difficulty = "very_hard"
    elif score >= 3.0:
        difficulty = "hard"
    elif score >= 1.2:
        difficulty = "medium"
    else:
        difficulty = "easy"
    oversample_weight = 1.0
    if difficulty == "medium":
        oversample_weight = 1.5
    elif difficulty == "hard":
        oversample_weight = 2.5
    elif difficulty == "very_hard":
        oversample_weight = 4.0

    return {
        "id": record.get("id", record.get("sample_id")),
        "image": record.get("image", record.get("images")),
        "coord_mode": coord_mode,
        "line_count": len(metric_lines),
        "centerline_count": len(centerlines),
        "intersection_count": len(intersections),
        "point_count": point_count,
        "total_centerline_length": round(total_length, 3),
        "cut_endpoint_count": cut_count,
        "graph_node_count": len(nodes),
        "max_endpoint_degree": max_degree,
        "fork_node_count": len(fork_nodes),
        "t_junction_like_count": len(t_nodes),
        "cycle_count": cycle_count,
        "crossing_count": crossing_count,
        "lane_change_like_count": lane_change_count,
        "closed_intersection_count": closed_intersections,
        "tags": tags,
        "difficulty": difficulty,
        "difficulty_score": round(score, 3),
        "oversample_weight": oversample_weight,
        "_payload": payload,
    }


def safe_name(value: Any) -> str:
    text = str(value or "sample")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sample"


def draw_overlay(
    record: dict[str, Any],
    metrics: dict[str, Any],
    image_path: Path | None,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    if image_path is not None and image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (256, 256), (20, 20, 20))
    payload = metrics.get("_payload") or {"lines": []}
    lines = normalize_lines(payload)
    coord_mode = infer_coord_mode(lines, image.size, args.coord_mode)
    draw = ImageDraw.Draw(image)
    colors = {
        "centerline": (0, 255, 80),
        "intersection": (255, 210, 0),
        "other": (0, 170, 255),
    }
    for item in lines:
        pts_raw = clean_points(item.get("points"))
        if not pts_raw:
            continue
        pts = [to_pixel(point, image.size, coord_mode, args.coord_range) for point in pts_raw]
        cat = category_of(item)
        color = colors.get(cat, colors["other"])
        width = 4 if cat == "intersection" else 3
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        else:
            draw.line(pts, fill=color, width=width)
            for idx, (x, y) in enumerate(pts):
                radius = 4 if idx in (0, len(pts) - 1) else 2
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    title_h = 78
    canvas = Image.new("RGB", (image.width, image.height + title_h), (0, 0, 0))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    title = f"{metrics.get('difficulty')} score={metrics.get('difficulty_score')} weight={metrics.get('oversample_weight')} id={metrics.get('id')}"
    tags = ",".join(metrics.get("tags") or [])
    stats = (
        f"lines={metrics.get('centerline_count')} inter={metrics.get('intersection_count')} "
        f"fork={metrics.get('fork_node_count')} T={metrics.get('t_junction_like_count')} "
        f"cycle={metrics.get('cycle_count')} lc={metrics.get('lane_change_like_count')}"
    )
    draw.text((8, 6), title[:140], fill=(255, 255, 255), font=font)
    draw.text((8, 31), tags[:160], fill=(255, 220, 80), font=small)
    draw.text((8, 52), stats[:160], fill=(180, 220, 255), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 260, cols: int = 4) -> None:
    images = []
    for path in image_paths:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        img = img.resize((thumb_width, max(1, int(round(img.height * scale)))))
        images.append(img)
    if not images:
        return
    rows = math.ceil(len(images) / cols)
    cell_h = max(img.height for img in images)
    sheet = Image.new("RGB", (thumb_width * cols, cell_h * rows), (10, 10, 10))
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        sheet.paste(img, (col * thumb_width, row * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def select_by_difficulty(
    reports: list[dict[str, Any]],
    difficulties: list[str],
    per_difficulty: int,
    seed: int,
    include_empty: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Select a deterministic random sample from each difficulty bucket."""
    if per_difficulty <= 0:
        return {}
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for difficulty in difficulties:
        bucket = [
            item
            for item in reports
            if item.get("difficulty") == difficulty
            and (include_empty or item.get("tags") != ["empty_patch"])
        ]
        if len(bucket) > per_difficulty:
            bucket = rng.sample(bucket, per_difficulty)
        bucket.sort(key=lambda item: int(item.get("index", 0)))
        selected[difficulty] = bucket
    return selected


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    jsonl_path = Path(args.jsonl) if args.jsonl else dataset_root / args.phase / f"{args.split}.jsonl"
    image_folder = Path(args.image_folder) if args.image_folder else dataset_root
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / f"hard_case_report_{args.phase}_{args.split}"
    viz_dir = output_dir / "viz"
    rows = load_jsonl(jsonl_path, args.max_samples)
    print(f"[hard-tags] loaded {len(rows)} samples from {jsonl_path}")

    reports = []
    for idx, record in enumerate(rows):
        image_path = resolve_record_image(record, dataset_root, image_folder)
        image_size = None
        if image_path is not None and image_path.exists():
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception:
                image_size = None
        metrics = sample_metrics(record, image_size, args)
        metrics["index"] = idx
        metrics["split"] = args.split
        metrics["phase"] = args.phase
        metrics["image_exists"] = bool(image_path and image_path.exists())
        metrics["image_size"] = list(image_size) if image_size else None
        metrics["image_resolved_path"] = str(image_path) if image_path else ""
        reports.append(metrics)

    reports.sort(key=lambda item: (item["difficulty_score"], item["centerline_count"], item["point_count"]), reverse=True)
    report_rows = [{k: v for k, v in item.items() if k != "_payload"} for item in reports]
    dump_jsonl(output_dir / "sample_tags.jsonl", report_rows)

    tag_counter = Counter(tag for item in reports for tag in item["tags"])
    difficulty_counter = Counter(item["difficulty"] for item in reports)
    summary = {
        "dataset_root": str(dataset_root),
        "jsonl": str(jsonl_path),
        "image_folder": str(image_folder),
        "num_samples": len(reports),
        "tag_counts": dict(tag_counter.most_common()),
        "difficulty_counts": dict(difficulty_counter),
        "top_samples": report_rows[: min(30, len(report_rows))],
        "thresholds": {
            "junction_tol": args.junction_tol,
            "intersection_tol": args.intersection_tol,
            "dense_line_threshold": args.dense_line_threshold,
            "dense_point_threshold": args.dense_point_threshold,
            "long_total_length_threshold": args.long_total_length_threshold,
            "many_cut_threshold": args.many_cut_threshold,
        },
    }
    dump_json(output_dir / "summary.json", summary)

    selected = [
        item
        for item in reports
        if item["difficulty_score"] >= args.min_score and (args.include_empty or item["tags"] != ["empty_patch"])
    ][: args.visualize_top_k]
    viz_paths = []
    for rank, metrics in enumerate(selected):
        record = rows[metrics["index"]]
        image_path = Path(metrics["image_resolved_path"]) if metrics.get("image_resolved_path") else None
        out_path = viz_dir / f"{rank:04d}_{safe_name(metrics.get('difficulty'))}_{safe_name(metrics.get('id'))}.png"
        draw_overlay(record, metrics, image_path, out_path, args)
        viz_paths.append(out_path)
    make_contact_sheet(viz_paths[: min(len(viz_paths), 48)], output_dir / "contact_sheet_top.png")

    stratified = select_by_difficulty(
        reports,
        difficulties=list(dict.fromkeys(args.visualize_difficulties)),
        per_difficulty=args.visualize_per_difficulty,
        seed=args.seed,
        include_empty=args.include_empty,
    )
    stratified_counts = {}
    stratified_root = output_dir / "viz_by_difficulty"
    for difficulty, bucket in stratified.items():
        difficulty_dir = stratified_root / difficulty
        difficulty_paths = []
        for rank, metrics in enumerate(bucket):
            record = rows[metrics["index"]]
            image_path = Path(metrics["image_resolved_path"]) if metrics.get("image_resolved_path") else None
            score = str(metrics.get("difficulty_score", 0)).replace(".", "p")
            out_path = difficulty_dir / (
                f"{rank:03d}_index-{metrics['index']:07d}_score-{score}_{safe_name(metrics.get('id'))}.png"
            )
            draw_overlay(record, metrics, image_path, out_path, args)
            difficulty_paths.append(out_path)
        contact_path = output_dir / f"contact_sheet_{difficulty}.png"
        make_contact_sheet(difficulty_paths, contact_path)
        stratified_counts[difficulty] = len(difficulty_paths)
        print(
            f"[hard-tags] {difficulty}: {len(difficulty_paths)} images -> {difficulty_dir}; "
            f"contact -> {contact_path}"
        )

    print(json.dumps({k: summary[k] for k in ("num_samples", "tag_counts", "difficulty_counts")}, ensure_ascii=False, indent=2))
    if stratified_counts:
        print(f"[hard-tags] stratified counts: {json.dumps(stratified_counts, ensure_ascii=False)}")
    print(f"[hard-tags] sample tags: {output_dir / 'sample_tags.jsonl'}")
    print(f"[hard-tags] summary:     {output_dir / 'summary.json'}")
    print(f"[hard-tags] viz dir:     {viz_dir}")
    print(f"[hard-tags] contact:     {output_dir / 'contact_sheet_top.png'}")


if __name__ == "__main__":
    main()
