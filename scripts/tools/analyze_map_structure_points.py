#!/usr/bin/env python3
"""Analyze fork and turn points in BEV map JSONL samples.

This script complements tag_hard_map_samples.py. The existing hard-sample
script reports counts and difficulty tags; this one also writes the positions
of graph fork nodes and polyline turn points so the dataset can be audited or
sampled by explicit road-structure features.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.tag_hard_map_samples import (
    DEFAULT_CURVED_LINE_TURN_THRESHOLD,
    DEFAULT_SHARP_TURN_THRESHOLD,
    category_of,
    clean_points,
    cluster_endpoints,
    dot,
    extract_target_payload,
    infer_coord_mode,
    is_t_junction,
    line_length,
    normalize_lines,
    normalize_lines_for_metrics,
    point_sub,
    record_patch_size,
    resolve_record_image,
    to_pixel,
    unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing phase_a/phase_b and images/.")
    parser.add_argument("--phase", default="phase_a", choices=["phase_a", "phase_b"])
    parser.add_argument("--split", default="train", choices=["train", "eval", "val", "test"])
    parser.add_argument("--jsonl", default="", help="Override JSONL path. Defaults to dataset-root/phase/split.jsonl.")
    parser.add_argument("--image-folder", default="", help="Image root. Defaults to dataset-root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 scans the whole JSONL.")
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--coord-mode", default="auto", choices=["auto", "pixel", "norm1000"])
    parser.add_argument("--coord-range", type=float, default=1000.0)
    parser.add_argument("--junction-tol", type=float, default=36.0, help="Endpoint snapping tolerance in metric coords.")
    parser.add_argument(
        "--turn-point-threshold",
        type=float,
        default=20.0,
        help="Interior polyline point with turn angle >= this value is reported as a bend point.",
    )
    parser.add_argument(
        "--curved-line-turn-threshold",
        type=float,
        default=DEFAULT_CURVED_LINE_TURN_THRESHOLD,
        help="A centerline whose accumulated turn exceeds this value is a curved line.",
    )
    parser.add_argument(
        "--sharp-turn-threshold",
        type=float,
        default=DEFAULT_SHARP_TURN_THRESHOLD,
        help="Interior turn angle threshold for sharp turns.",
    )
    parser.add_argument("--visualize-top-k", type=int, default=0)
    parser.add_argument("--visualize-random-k", type=int, default=0)
    parser.add_argument("--visualize-only-structured", action="store_true", help="Skip plain samples in random viz.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_record_lines(path: Path, max_samples: int, progress_every: int):
    with path.open("r", encoding="utf-8") as handle:
        count = 0
        for line in handle:
            if not line.strip():
                continue
            yield count, json.loads(line)
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"[structure-points] processed {count} records from {path}", flush=True)
            if max_samples and count >= max_samples:
                break


def point_to_output(point: tuple[float, float], patch_size: tuple[int, int], coord_range: float) -> dict[str, int]:
    px, py = to_pixel(point, patch_size, "norm1000", coord_range)
    return {
        "x": int(round(point[0])),
        "y": int(round(point[1])),
        "pixel_x": int(px),
        "pixel_y": int(py),
    }


def segment_projection(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, tuple[float, float]]:
    segment = point_sub(end, start)
    denom = dot(segment, segment)
    if denom <= 1e-9:
        return math.dist(point, start), 0.0, start
    t = max(0.0, min(1.0, dot(point_sub(point, start), segment) / denom))
    projected = (start[0] + t * segment[0], start[1] + t * segment[1])
    return math.dist(point, projected), t, projected


def segment_intersection_point(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> tuple[float, float] | None:
    if max(min(a[0], b[0]), min(c[0], d[0])) > min(max(a[0], b[0]), max(c[0], d[0])):
        return None
    if max(min(a[1], b[1]), min(c[1], d[1])) > min(max(a[1], b[1]), max(c[1], d[1])):
        return None

    bax, bay = b[0] - a[0], b[1] - a[1]
    dcx, dcy = d[0] - c[0], d[1] - c[1]
    denom = bax * dcy - bay * dcx
    if abs(denom) <= 1e-8:
        return None
    acx, acy = c[0] - a[0], c[1] - a[1]
    t = (acx * dcy - acy * dcx) / denom
    u = (acx * bay - acy * bax) / denom
    eps = 1e-6
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return (a[0] + t * bax, a[1] + t * bay)
    return None


def merge_members(existing: list[dict[str, Any]], new_members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {
        (
            item.get("line_index"),
            item.get("endpoint", ""),
            item.get("segment_index", None),
            item.get("point_index", None),
        )
        for item in existing
    }
    for member in new_members:
        key = (
            member.get("line_index"),
            member.get("endpoint", ""),
            member.get("segment_index", None),
            member.get("point_index", None),
        )
        if key not in seen:
            existing.append(member)
            seen.add(key)
    return existing


def add_branch_candidate(
    candidates: list[dict[str, Any]],
    point: tuple[float, float],
    branch_kind: str,
    degree: int,
    members: list[dict[str, Any]],
    merge_tol: float,
    t_like: bool = False,
) -> None:
    best_idx = None
    best_dist = float("inf")
    for idx, candidate in enumerate(candidates):
        dist = math.dist(point, candidate["point"])
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_idx is not None and best_dist <= merge_tol:
        candidate = candidates[best_idx]
        candidate["point"] = (
            (candidate["point"][0] + point[0]) / 2.0,
            (candidate["point"][1] + point[1]) / 2.0,
        )
        candidate["degree"] = max(candidate["degree"], degree)
        candidate["t_like"] = bool(candidate["t_like"] or t_like)
        if branch_kind not in candidate["kinds"]:
            candidate["kinds"].append(branch_kind)
        merge_members(candidate["members"], members)
        return
    candidates.append(
        {
            "point": point,
            "degree": int(degree),
            "t_like": bool(t_like),
            "kinds": [branch_kind],
            "members": list(members),
        }
    )


def line_segments(centerlines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    for line_index, line in enumerate(centerlines):
        points = clean_points(line.get("points"))
        for point_index in range(1, len(points)):
            start = points[point_index - 1]
            end = points[point_index]
            if math.dist(start, end) <= 1e-6:
                continue
            segments.append(
                {
                    "line_index": line_index,
                    "segment_index": point_index - 1,
                    "start": start,
                    "end": end,
                }
            )
    return segments


def branch_point_details(
    centerlines: list[dict[str, Any]],
    patch_size: tuple[int, int],
    coord_range: float,
    junction_tol: float,
) -> list[dict[str, Any]]:
    nodes, _endpoint_to_node, node_members = cluster_endpoints(centerlines, junction_tol)
    candidates: list[dict[str, Any]] = []
    for node_index, members in sorted(node_members.items()):
        if len(members) < 3:
            continue
        add_branch_candidate(
            candidates,
            nodes[node_index],
            "endpoint_cluster",
            len(members),
            [
                {
                    "line_index": int(line_index),
                    "endpoint": str(endpoint),
                }
                for line_index, endpoint in members
            ],
            merge_tol=junction_tol * 0.5,
            t_like=is_t_junction(node_index, node_members, centerlines),
        )

    segments = line_segments(centerlines)
    centerline_points = [clean_points(line.get("points")) for line in centerlines]
    for node_index, members in sorted(node_members.items()):
        endpoint_lines = {line_index for line_index, _ in members}
        point = nodes[node_index]
        for line_index, points in enumerate(centerline_points):
            if line_index in endpoint_lines or len(points) < 3:
                continue
            for point_index, interior_point in enumerate(points[1:-1], start=1):
                if math.dist(point, interior_point) > junction_tol:
                    continue
                add_branch_candidate(
                    candidates,
                    interior_point,
                    "endpoint_on_vertex",
                    len(members) + 2,
                    [
                        {
                            "line_index": int(line_idx),
                            "endpoint": str(endpoint),
                        }
                        for line_idx, endpoint in members
                    ]
                    + [
                        {
                            "line_index": int(line_index),
                            "endpoint": "interior_vertex",
                            "point_index": int(point_index),
                        }
                    ],
                    merge_tol=junction_tol * 0.5,
                    t_like=(len(members) + 2 == 3),
                )
        for segment in segments:
            line_index = segment["line_index"]
            if line_index in endpoint_lines:
                continue
            distance, t, projected = segment_projection(point, segment["start"], segment["end"])
            if distance > junction_tol or not (0.05 <= t <= 0.95):
                continue
            add_branch_candidate(
                candidates,
                projected,
                "endpoint_on_line",
                len(members) + 2,
                [
                    {
                        "line_index": int(line_idx),
                        "endpoint": str(endpoint),
                    }
                    for line_idx, endpoint in members
                ]
                + [
                    {
                        "line_index": int(line_index),
                        "endpoint": "interior",
                        "segment_index": int(segment["segment_index"]),
                    }
                ],
                merge_tol=junction_tol * 0.5,
                t_like=(len(members) + 2 == 3),
            )

    for left_idx, left in enumerate(segments):
        for right in segments[left_idx + 1 :]:
            if left["line_index"] == right["line_index"]:
                continue
            if min(
                math.dist(left["start"], right["start"]),
                math.dist(left["start"], right["end"]),
                math.dist(left["end"], right["start"]),
                math.dist(left["end"], right["end"]),
            ) <= junction_tol:
                continue
            point = segment_intersection_point(left["start"], left["end"], right["start"], right["end"])
            if point is None:
                continue
            add_branch_candidate(
                candidates,
                point,
                "line_crossing",
                4,
                [
                    {
                        "line_index": int(left["line_index"]),
                        "endpoint": "interior",
                        "segment_index": int(left["segment_index"]),
                    },
                    {
                        "line_index": int(right["line_index"]),
                        "endpoint": "interior",
                        "segment_index": int(right["segment_index"]),
                    },
                ],
                merge_tol=junction_tol * 0.5,
                t_like=False,
            )

    output = []
    for idx, candidate in enumerate(sorted(candidates, key=lambda item: (item["point"][1], item["point"][0]))):
        out = point_to_output(candidate["point"], patch_size, coord_range)
        kinds = sorted(candidate["kinds"])
        out.update(
            {
                "node_index": idx,
                "degree": int(candidate["degree"]),
                "branch_type": kinds[0] if len(kinds) == 1 else "+".join(kinds),
                "kinds": kinds,
                "t_like": bool(candidate["t_like"]),
                "members": candidate["members"],
            }
        )
        output.append(out)
    return output


def turn_angle(prev_pt: tuple[float, float], point: tuple[float, float], next_pt: tuple[float, float]) -> float | None:
    incoming = unit(point_sub(point, prev_pt))
    outgoing = unit(point_sub(next_pt, point))
    if incoming is None or outgoing is None:
        return None
    cosine = max(-1.0, min(1.0, dot(incoming, outgoing)))
    return math.degrees(math.acos(cosine))


def line_turn_details(
    centerlines: list[dict[str, Any]],
    patch_size: tuple[int, int],
    coord_range: float,
    turn_point_threshold: float,
    sharp_turn_threshold: float,
    curved_line_turn_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    turn_points: list[dict[str, Any]] = []
    curved_lines: list[dict[str, Any]] = []
    for line_index, line in enumerate(centerlines):
        points = clean_points(line.get("points"))
        angles = []
        reported_turn_indexes = []
        for point_index in range(1, len(points) - 1):
            angle = turn_angle(points[point_index - 1], points[point_index], points[point_index + 1])
            if angle is None:
                continue
            angles.append((point_index, angle))
            if angle >= turn_point_threshold:
                reported_turn_indexes.append(point_index)
                out = point_to_output(points[point_index], patch_size, coord_range)
                out.update(
                    {
                        "line_index": line_index,
                        "point_index": point_index,
                        "angle": round(float(angle), 3),
                        "is_sharp": bool(angle >= sharp_turn_threshold),
                    }
                )
                turn_points.append(out)
        total_turn = sum(angle for _, angle in angles)
        max_turn = max((angle for _, angle in angles), default=0.0)
        if total_turn >= curved_line_turn_threshold:
            start = point_to_output(points[0], patch_size, coord_range) if points else {}
            end = point_to_output(points[-1], patch_size, coord_range) if points else {}
            curved_lines.append(
                {
                    "line_index": line_index,
                    "point_count": len(points),
                    "length": round(float(line_length(points)), 3),
                    "total_turn": round(float(total_turn), 3),
                    "max_turn": round(float(max_turn), 3),
                    "turn_point_indexes": reported_turn_indexes,
                    "start": start,
                    "end": end,
                }
            )
    return turn_points, curved_lines


def analyze_record(record: dict[str, Any], source_index: int, args: argparse.Namespace) -> dict[str, Any]:
    payload = extract_target_payload(record)
    lines = normalize_lines(payload)
    coord_mode = infer_coord_mode(lines, None, args.coord_mode)
    patch_size = record_patch_size(record, None)
    metric_lines = normalize_lines_for_metrics(
        lines,
        coord_mode=coord_mode,
        patch_size=patch_size,
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

    fork_points = branch_point_details(
        centerlines=centerlines,
        patch_size=patch_size,
        coord_range=float(args.coord_range),
        junction_tol=float(args.junction_tol),
    )

    turn_points, curved_lines = line_turn_details(
        centerlines=centerlines,
        patch_size=patch_size,
        coord_range=float(args.coord_range),
        turn_point_threshold=float(args.turn_point_threshold),
        sharp_turn_threshold=float(args.sharp_turn_threshold),
        curved_line_turn_threshold=float(args.curved_line_turn_threshold),
    )
    sharp_turn_count = sum(1 for item in turn_points if item["is_sharp"])
    structure_score = (
        len(fork_points) * 4.0
        + sharp_turn_count * 2.0
        + max(0, len(turn_points) - sharp_turn_count) * 0.7
        + len(curved_lines) * 1.2
        + len(intersections) * 1.5
    )
    return {
        "source_index": source_index,
        "id": record.get("id", record.get("sample_id")),
        "image": record.get("image", record.get("images")),
        "coord_mode": coord_mode,
        "coord_range": args.coord_range,
        "patch_width": patch_size[0],
        "patch_height": patch_size[1],
        "centerline_count": len(centerlines),
        "intersection_count": len(intersections),
        "fork_node_count": len(fork_points),
        "t_junction_like_count": sum(1 for item in fork_points if item["t_like"]),
        "turn_point_count": len(turn_points),
        "sharp_turn_count": sharp_turn_count,
        "curved_line_count": len(curved_lines),
        "structure_score": round(structure_score, 3),
        "fork_points": fork_points,
        "turn_points": turn_points,
        "curved_lines": curved_lines,
    }


def safe_name(value: Any) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "sample")).strip("_") or "sample"


def draw_structure_overlay(
    record: dict[str, Any],
    report: dict[str, Any],
    image_path: Path | None,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    if image_path is not None and image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (report.get("patch_width", 256), report.get("patch_height", 256)), (20, 20, 20))
    payload = extract_target_payload(record)
    lines = normalize_lines(payload)
    coord_mode = infer_coord_mode(lines, image.size, args.coord_mode)
    draw = ImageDraw.Draw(image)
    target_roi = None
    meta = record.get("meta")
    if isinstance(meta, dict):
        raw_roi = meta.get("target_roi_in_image")
        if (
            isinstance(raw_roi, list)
            and len(raw_roi) == 4
            and all(isinstance(value, int) for value in raw_roi)
            and 0 <= raw_roi[0] < raw_roi[2] <= image.width
            and 0 <= raw_roi[1] < raw_roi[3] <= image.height
        ):
            target_roi = tuple(raw_roi)
            if target_roi != (0, 0, image.width, image.height):
                draw.rectangle(
                    [target_roi[0], target_roi[1], target_roi[2] - 1, target_roi[3] - 1],
                    outline=(255, 80, 80),
                    width=2,
                )
    for item in lines:
        points = clean_points(item.get("points"))
        if not points:
            continue
        pts = [to_pixel(point, image.size, coord_mode, float(args.coord_range), target_roi) for point in points]
        color = (0, 240, 80) if category_of(item) == "centerline" else (255, 210, 0)
        width = 3 if category_of(item) == "centerline" else 4
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        else:
            draw.line(pts, fill=color, width=width)

    for fork in report.get("fork_points", []):
        x, y = to_pixel((float(fork["x"]), float(fork["y"])), image.size, "norm1000", float(args.coord_range), target_roi)
        color = (255, 40, 40) if not fork.get("t_like") else (255, 0, 220)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=color, width=3)
        draw.text((x + 8, y - 8), f"F{fork.get('degree')}", fill=color)

    for turn in report.get("turn_points", []):
        x, y = to_pixel((float(turn["x"]), float(turn["y"])), image.size, "norm1000", float(args.coord_range), target_roi)
        color = (255, 120, 0) if turn.get("is_sharp") else (0, 220, 255)
        draw.rectangle([x - 4, y - 4, x + 4, y + 4], outline=color, width=2)

    header_h = 86
    canvas = Image.new("RGB", (image.width, image.height + header_h), (0, 0, 0))
    canvas.paste(image, (0, header_h))
    header = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    title = (
        f"id={report.get('id')} score={report.get('structure_score')} "
        f"fork={report.get('fork_node_count')} turn={report.get('turn_point_count')} "
        f"sharp={report.get('sharp_turn_count')} curved={report.get('curved_line_count')}"
    )
    legend = "green=centerline yellow=intersection red=fork magenta=T-like cyan=turn orange=sharp"
    header.text((8, 8), title[:160], fill=(255, 255, 255), font=font)
    header.text((8, 36), legend, fill=(210, 230, 255), font=small)
    header.text((8, 58), str(report.get("image"))[:180], fill=(220, 220, 180), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def maybe_push_top(top_heap: list[tuple[float, int, dict[str, Any], dict[str, Any]]], limit: int, score: float, source_index: int, record: dict[str, Any], report: dict[str, Any]) -> None:
    if limit <= 0:
        return
    item = (score, source_index, record, report)
    if len(top_heap) < limit:
        heapq.heappush(top_heap, item)
    elif item[:2] > top_heap[0][:2]:
        heapq.heapreplace(top_heap, item)


def maybe_push_random(
    reservoir: list[tuple[int, dict[str, Any], dict[str, Any]]],
    limit: int,
    seen: int,
    source_index: int,
    record: dict[str, Any],
    report: dict[str, Any],
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append((source_index, record, report))
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = (source_index, record, report)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_folder = Path(args.image_folder) if args.image_folder else dataset_root
    jsonl_path = Path(args.jsonl) if args.jsonl else dataset_root / args.phase / f"{args.split}.jsonl"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "structure_points.jsonl"
    top_heap: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    random_reservoir: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    rng = random.Random(args.seed)
    random_seen = 0
    counts = Counter()
    total = 0

    with report_path.open("w", encoding="utf-8") as out:
        for source_index, record in load_record_lines(jsonl_path, args.max_samples, args.progress_every):
            report = analyze_record(record, source_index, args)
            out.write(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
            total += 1
            counts["samples"] += 1
            if report["fork_node_count"]:
                counts["samples_with_forks"] += 1
            if report["turn_point_count"]:
                counts["samples_with_turn_points"] += 1
            if report["sharp_turn_count"]:
                counts["samples_with_sharp_turns"] += 1
            if report["curved_line_count"]:
                counts["samples_with_curved_lines"] += 1
            counts["fork_points"] += report["fork_node_count"]
            counts["turn_points"] += report["turn_point_count"]
            counts["sharp_turns"] += report["sharp_turn_count"]
            counts["curved_lines"] += report["curved_line_count"]
            maybe_push_top(top_heap, args.visualize_top_k, float(report["structure_score"]), source_index, record, report)
            structured = bool(report["fork_node_count"] or report["turn_point_count"] or report["curved_line_count"])
            if structured or not args.visualize_only_structured:
                random_seen += 1
                maybe_push_random(
                    random_reservoir,
                    args.visualize_random_k,
                    random_seen,
                    source_index,
                    record,
                    report,
                    rng,
                )

    viz_records: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    for _, source_index, record, report in sorted(top_heap, reverse=True):
        viz_records.append(("top", source_index, record, report))
    for source_index, record, report in random_reservoir:
        viz_records.append(("random", source_index, record, report))

    viz_dir = output_dir / "viz"
    for prefix, source_index, record, report in viz_records:
        image_path = resolve_record_image(record, dataset_root, image_folder)
        out_name = f"{prefix}_{source_index:08d}_{safe_name(report.get('id'))}.png"
        draw_structure_overlay(record, report, image_path, viz_dir / out_name, args)

    summary = {
        "input_jsonl": str(jsonl_path),
        "dataset_root": str(dataset_root),
        "image_folder": str(image_folder),
        "output_dir": str(output_dir),
        "num_samples": total,
        "counts": dict(counts),
        "thresholds": {
            "junction_tol": args.junction_tol,
            "turn_point_threshold": args.turn_point_threshold,
            "sharp_turn_threshold": args.sharp_turn_threshold,
            "curved_line_turn_threshold": args.curved_line_turn_threshold,
            "coord_mode": args.coord_mode,
            "coord_range": args.coord_range,
        },
        "report_jsonl": str(report_path),
        "viz_dir": str(viz_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[structure-points] report: {report_path}", flush=True)
    print(f"[structure-points] summary: {output_dir / 'summary.json'}", flush=True)
    print(f"[structure-points] viz: {viz_dir}", flush=True)


if __name__ == "__main__":
    main()
