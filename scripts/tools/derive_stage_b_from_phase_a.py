#!/usr/bin/env python3
"""Derive Phase B SFT records from a completed Phase A dataset.

The target samples come from ``phase_a``.  Incoming left/top continuity hints can
be looked up from retained staging records so neighbors do not need to be part
of the final balanced training subset.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import write_jsonl_item
from data_process.build_dataset_v2_staged import (
    STAGE_MARKER,
    build_sample_owners,
    discover_stage_roots,
    iter_jsonl as iter_stage_jsonl,
)
from data_process.state_update_dataset_common import (
    COORD_MODE_NORM1000,
    DEFAULT_COORD_RANGE,
    SEMANTIC_SCHEMA_VERSION,
    build_sft_record,
    is_near,
    sort_target_lines,
)
from mllm.coord_utils import coord_point_to_pixel, normalize_coord_mode


SPLITS = ("train", "eval", "test")
DEFAULT_TRACE_SPACING_PX = 50.0
DEFAULT_TRACE_POINT_COUNT = 3


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Completed dataset variant root.")
    parser.add_argument("--phase-a", default="phase_a")
    parser.add_argument("--phase-b", default="phase_b")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument(
        "--staging-root",
        default="",
        help="Optional staging root used to find full left/top neighbor records.",
    )
    parser.add_argument("--staging-variant", default="local512")
    parser.add_argument("--duplicate-policy", choices=["last", "first", "error"], default="last")
    parser.add_argument("--trace-spacing-px", type=float, default=DEFAULT_TRACE_SPACING_PX)
    parser.add_argument("--trace-point-count", type=int, default=DEFAULT_TRACE_POINT_COUNT)
    parser.add_argument("--boundary-tol", type=float, default=2.0)
    parser.add_argument("--max-traces-per-side", type=int, default=8)
    parser.add_argument("--max-intersections-per-side", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resolve_split_path(phase_root: Path, split: str, *, meta: bool = False) -> Path | None:
    stem = f"meta_{split}" if meta else split
    candidates = [phase_root / f"{stem}.jsonl"]
    if split == "eval":
        candidates.append(phase_root / ("meta_val.jsonl" if meta else "val.jsonl"))
    return next((path for path in candidates if path.is_file()), None)


def assistant_payload(record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations") or []
    for message in reversed(conversations):
        if str(message.get("from", "")).lower() in {"gpt", "assistant"}:
            text = str(message.get("value", "")).strip()
            return json.loads(text) if text else {"lines": []}
    return {"lines": []}


def record_patch_config(record: dict[str, Any]) -> tuple[int, str, int, int]:
    meta = record.get("meta") or {}
    patch_size = int(
        meta.get("pixel_patch_size")
        or meta.get("target_size")
        or meta.get("patch_size")
        or meta.get("patch_width")
        or 512
    )
    context_size = int(meta.get("context_image_size") or patch_size)
    coord_mode = normalize_coord_mode(meta.get("coord_mode") or meta.get("coord_system") or COORD_MODE_NORM1000)
    coord_range = int(meta.get("coord_range") or DEFAULT_COORD_RANGE)
    return patch_size, coord_mode, coord_range, context_size


def record_grid_key(record: dict[str, Any]) -> tuple[str, int, int]:
    meta = record.get("meta") or {}
    raw_sample_id = str(
        meta.get("raw_sample_id")
        or meta.get("tile_id")
        or meta.get("log_id")
        or meta.get("sample_id")
        or ""
    ).strip()
    if not raw_sample_id:
        sample_id = str(record.get("id", ""))
        marker = sample_id.rfind("_r")
        raw_sample_id = sample_id[:marker] if marker >= 0 else sample_id
    row = int(meta.get("row", meta.get("patch_row", 0)))
    col = int(meta.get("col", meta.get("patch_col", 0)))
    return raw_sample_id, row, col


def line_model_to_pixel(line: dict[str, Any], patch_size: int, coord_mode: str, coord_range: int) -> dict[str, Any]:
    converted = dict(line)
    converted["points"] = [
        coord_point_to_pixel(
            point,
            patch_size,
            patch_size,
            coord_mode=coord_mode,
            coord_range=coord_range,
            clamp=True,
        )
        for point in converted.get("points", [])
    ]
    return converted


def record_target_lines_pixel(record: dict[str, Any]) -> list[dict[str, Any]]:
    patch_size, coord_mode, coord_range, _ = record_patch_config(record)
    payload = assistant_payload(record)
    return [
        line_model_to_pixel(line, patch_size, coord_mode, coord_range)
        for line in payload.get("lines", [])
        if isinstance(line, dict)
    ]


def polyline_length(points: list[list[int]]) -> float:
    total = 0.0
    for left, right in zip(points, points[1:]):
        total += math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
    return total


def interpolate_polyline(points: list[list[int]], distance_from_start: float) -> list[int]:
    if not points:
        return [0, 0]
    if len(points) == 1:
        return [int(round(points[0][0])), int(round(points[0][1]))]
    remaining = max(0.0, float(distance_from_start))
    for left, right in zip(points, points[1:]):
        lx, ly = float(left[0]), float(left[1])
        rx, ry = float(right[0]), float(right[1])
        segment = math.hypot(rx - lx, ry - ly)
        if segment <= 1e-8:
            continue
        if remaining <= segment:
            ratio = remaining / segment
            return [int(round(lx + (rx - lx) * ratio)), int(round(ly + (ry - ly) * ratio))]
        remaining -= segment
    return [int(round(points[-1][0])), int(round(points[-1][1]))]


def sample_boundary_trace_points(
    points: list[list[int]],
    *,
    boundary_at_start: bool,
    spacing_px: float = DEFAULT_TRACE_SPACING_PX,
    point_count: int = DEFAULT_TRACE_POINT_COUNT,
) -> list[list[int]]:
    """Return points ordered from neighbor interior toward the shared boundary."""

    if point_count <= 0:
        return []
    if len(points) < 2:
        return [points[0]] * point_count if points else []
    total = polyline_length(points)
    if total <= 1e-8:
        return [[int(round(points[0][0])), int(round(points[0][1]))] for _ in range(point_count)]
    max_span = spacing_px * max(point_count - 1, 0)
    if total < max_span and point_count > 1:
        if boundary_at_start:
            distances = [total * (point_count - 1 - index) / (point_count - 1) for index in range(point_count)]
        else:
            distances = [total * index / (point_count - 1) for index in range(point_count)]
    elif boundary_at_start:
        distances = [spacing_px * (point_count - 1 - index) for index in range(point_count)]
    else:
        distances = [total - spacing_px * (point_count - 1 - index) for index in range(point_count)]
    return [interpolate_polyline(points, distance) for distance in distances]


def shift_points_to_current(points: list[list[int]], side: str, patch_size: int) -> list[list[int]]:
    shifted = []
    for point in points:
        x, y = int(round(point[0])), int(round(point[1]))
        if side == "left":
            shifted.append([x - patch_size, y])
        elif side == "top":
            shifted.append([x, y - patch_size])
        else:
            shifted.append([x, y])
    return shifted


def shared_boundary_touch(line: dict[str, Any], side: str, patch_size: int, boundary_tol: float) -> str | None:
    points = line.get("points") or []
    if len(points) < 2:
        return None
    if side == "left":
        on_boundary = lambda point: is_near(point[0], patch_size - 1, boundary_tol)
    elif side == "top":
        on_boundary = lambda point: is_near(point[1], patch_size - 1, boundary_tol)
    else:
        return None
    start_on = on_boundary(points[0])
    end_on = on_boundary(points[-1])
    if line.get("end_type") == "cut" and end_on:
        return "end"
    if line.get("start_type") == "cut" and start_on:
        return "start"
    if end_on:
        return "end"
    if start_on:
        return "start"
    return None


def make_incoming_trace_from_line(
    line: dict[str, Any],
    side: str,
    patch_size: int,
    spacing_px: float,
    point_count: int,
    boundary_tol: float,
) -> dict[str, Any] | None:
    if line.get("category") != "centerline":
        return None
    boundary = shared_boundary_touch(line, side, patch_size, boundary_tol)
    if boundary is None:
        return None
    sampled = sample_boundary_trace_points(
        line.get("points") or [],
        boundary_at_start=(boundary == "start"),
        spacing_px=spacing_px,
        point_count=point_count,
    )
    if len(sampled) != point_count:
        return None
    return {"side": side, "points": shift_points_to_current(sampled, side, patch_size)}


def make_incoming_intersection_from_line(
    line: dict[str, Any],
    side: str,
    patch_size: int,
    boundary_tol: float,
) -> dict[str, Any] | None:
    if line.get("category") != "intersection":
        return None
    points = line.get("points") or []
    if len(points) < 4:
        return None
    if side == "left":
        touches = any(is_near(point[0], patch_size - 1, boundary_tol) for point in points)
    elif side == "top":
        touches = any(is_near(point[1], patch_size - 1, boundary_tol) for point in points)
    else:
        touches = False
    if not touches:
        return None
    shifted = shift_points_to_current(points, side, patch_size)
    if shifted and shifted[0] != shifted[-1]:
        shifted.append(list(shifted[0]))
    return {
        "side": side,
        "category": "intersection",
        "intersection_type": line.get("intersection_type", "other"),
        "is_cut": bool(line.get("is_cut", True)),
        "points": shifted,
    }


def build_incoming_from_neighbor(
    neighbor_lines: list[dict[str, Any]] | None,
    side: str,
    patch_size: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traces = []
    intersections = []
    if not neighbor_lines:
        return traces, intersections
    for line in neighbor_lines:
        trace = make_incoming_trace_from_line(
            line,
            side,
            patch_size,
            args.trace_spacing_px,
            args.trace_point_count,
            args.boundary_tol,
        )
        if trace is not None and len(traces) < args.max_traces_per_side:
            trace["id"] = f"{'L' if side == 'left' else 'T'}{len(traces)}"
            traces.append(trace)
        inter = make_incoming_intersection_from_line(line, side, patch_size, args.boundary_tol)
        if inter is not None and len(intersections) < args.max_intersections_per_side:
            inter["id"] = f"{'IL' if side == 'left' else 'IT'}{len(intersections)}"
            intersections.append(inter)
        if len(traces) >= args.max_traces_per_side and len(intersections) >= args.max_intersections_per_side:
            break
    return traces, intersections


def collect_needed_neighbor_keys(phase_a_path: Path) -> tuple[int, set[tuple[str, int, int]]]:
    needed = set()
    count = 0
    for _, record in iter_jsonl(phase_a_path):
        raw_sample_id, row, col = record_grid_key(record)
        needed.add((raw_sample_id, row, col - 1))
        needed.add((raw_sample_id, row - 1, col))
        count += 1
    return count, needed


def build_selected_lookup(phase_a_path: Path, needed_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    lookup = {}
    for _, record in iter_jsonl(phase_a_path):
        key = record_grid_key(record)
        if key in needed_keys:
            lookup[key] = record_target_lines_pixel(record)
    return lookup


def build_stage_lookup(
    staging_root: Path,
    variant: str,
    split: str,
    needed_keys: set[tuple[str, int, int]],
    duplicate_policy: str,
    progress_every: int,
) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    stage_roots = discover_stage_roots(staging_root)
    sample_owner, _ = build_sample_owners(stage_roots, duplicate_policy)
    lookup = {}
    seen = 0
    for stage_root in stage_roots:
        marker = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        source_index = int(marker["source_index"])
        index_iter = iter_stage_jsonl(stage_root / "records" / f"{split}.index.jsonl")
        sft_iter = iter_stage_jsonl(stage_root / "records" / variant / f"{split}.jsonl")
        for (_, index_item), (_, record) in zip(index_iter, sft_iter):
            seen += 1
            if progress_every and seen % progress_every == 0:
                print(
                    f"[phase-b] scan staging {split}: seen={seen} matched={len(lookup)}/{len(needed_keys)}",
                    flush=True,
                )
            if sample_owner.get(str(index_item["raw_sample_id"])) != source_index:
                continue
            if str(record.get("id")) != str(index_item.get("id")):
                raise ValueError(f"staging index/SFT mismatch in {stage_root}: {index_item.get('id')} vs {record.get('id')}")
            key = record_grid_key(record)
            if key not in needed_keys or key in lookup:
                continue
            lookup[key] = record_target_lines_pixel(record)
            if len(lookup) >= len(needed_keys):
                break
        if len(lookup) >= len(needed_keys):
            break
    return lookup


def merge_meta_row(base_meta_row: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_meta_row or {})
    merged["id"] = record["id"]
    merged["image"] = record["image"]
    merged["meta"] = record.get("meta", {})
    return merged


def derive_split(dataset_root: Path, split: str, args: argparse.Namespace) -> dict[str, Any]:
    phase_a_root = dataset_root / args.phase_a
    phase_b_root = dataset_root / args.phase_b
    phase_a_path = resolve_split_path(phase_a_root, split)
    if phase_a_path is None:
        raise FileNotFoundError(f"phase_a {split} JSONL not found under {phase_a_root}")
    output_path = phase_b_root / f"{split}.jsonl"
    meta_output_path = phase_b_root / f"meta_{split}.jsonl"
    source_count, needed_keys = collect_needed_neighbor_keys(phase_a_path)
    if args.resume and output_path.is_file() and count_jsonl(output_path) == source_count:
        print(f"[phase-b] reuse completed {split}: {output_path}", flush=True)
        return {"records": source_count, "reused": True}
    if output_path.exists() and not args.overwrite:
        raise ValueError(f"phase_b split already exists; pass --overwrite or --resume: {output_path}")

    if args.staging_root:
        neighbor_lookup = build_stage_lookup(
            Path(args.staging_root).expanduser().resolve(),
            args.staging_variant,
            split,
            needed_keys,
            args.duplicate_policy,
            args.progress_every,
        )
    else:
        neighbor_lookup = build_selected_lookup(phase_a_path, needed_keys)

    phase_b_root.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_meta = meta_output_path.with_suffix(meta_output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    temporary_meta.unlink(missing_ok=True)

    source_meta_path = resolve_split_path(phase_a_root, split, meta=True)
    meta_iter = iter_jsonl(source_meta_path) if source_meta_path is not None else None
    stats = Counter()
    with temporary.open("w", encoding="utf-8") as out_handle, temporary_meta.open("w", encoding="utf-8") as meta_handle:
        for line_number, source_record in iter_jsonl(phase_a_path):
            base_meta_row = None
            if meta_iter is not None:
                try:
                    _, base_meta_row = next(meta_iter)
                except StopIteration as exc:
                    raise ValueError(f"meta file ended before {phase_a_path}:{line_number}") from exc
                if str(base_meta_row.get("id")) != str(source_record.get("id")):
                    raise ValueError(
                        f"phase_a/meta order mismatch for {split}: {base_meta_row.get('id')} vs {source_record.get('id')}"
                    )
            patch_size, coord_mode, coord_range, context_size = record_patch_config(source_record)
            raw_sample_id, row, col = record_grid_key(source_record)
            incoming_traces = []
            incoming_intersections = []
            for side, key in (
                ("left", (raw_sample_id, row, col - 1)),
                ("top", (raw_sample_id, row - 1, col)),
            ):
                traces, intersections = build_incoming_from_neighbor(neighbor_lookup.get(key), side, patch_size, args)
                incoming_traces.extend(traces)
                incoming_intersections.extend(intersections)
            target_lines = sort_target_lines(record_target_lines_pixel(source_record), patch_size, args.boundary_tol)
            row_payload = {
                "id": source_record["id"],
                "image": source_record["image"],
                "incoming_traces": incoming_traces,
                "incoming_intersections": incoming_intersections,
                "target_lines": target_lines,
                "meta": dict(source_record.get("meta") or {}),
            }
            derived = build_sft_record(
                row_payload,
                patch_size,
                include_intersections=True,
                phase="b",
                coord_mode=coord_mode,
                coord_range=coord_range,
                context_size=context_size,
                view_mode=(source_record.get("meta") or {}).get("view_mode"),
                incoming_trace_point_spacing_px=args.trace_spacing_px,
                incoming_intersections_full_polygon=True,
                raw_lane_overlay=bool((source_record.get("meta") or {}).get("raw_lane_overlay", False)),
            )
            write_jsonl_item(out_handle, derived)
            meta_row = merge_meta_row(base_meta_row, derived)
            meta_row["incoming_trace_count"] = len(incoming_traces)
            meta_row["incoming_intersection_count"] = len(incoming_intersections)
            write_jsonl_item(meta_handle, meta_row)
            stats["records"] += 1
            stats["incoming_traces"] += len(incoming_traces)
            stats["incoming_intersections"] += len(incoming_intersections)
            if any(len(trace.get("points") or []) != args.trace_point_count for trace in incoming_traces):
                stats["bad_trace_point_count"] += 1
            if args.progress_every and stats["records"] % args.progress_every == 0:
                print(
                    f"[phase-b] write {split}: records={stats['records']} "
                    f"traces={stats['incoming_traces']} intersections={stats['incoming_intersections']}",
                    flush=True,
                )

    if meta_iter is not None:
        try:
            next(meta_iter)
            raise ValueError(f"meta file has extra rows after {split}: {source_meta_path}")
        except StopIteration:
            pass
    if stats["bad_trace_point_count"]:
        raise ValueError(f"{split} produced traces without exactly {args.trace_point_count} points")
    temporary.replace(output_path)
    temporary_meta.replace(meta_output_path)
    return dict(stats)


def update_dataset_info(dataset_root: Path, split_stats: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    info_path = dataset_root / "dataset_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = {}
    phases = set(info.get("available_phases") or [])
    phases.update([args.phase_a, args.phase_b])
    info["available_phases"] = sorted(phases)
    info["semantic_schema_version"] = info.get("semantic_schema_version") or SEMANTIC_SCHEMA_VERSION
    info["phase_b_generation"] = {
        "phase_a": args.phase_a,
        "phase_b": args.phase_b,
        "trace_point_count": args.trace_point_count,
        "trace_spacing_px": args.trace_spacing_px,
        "incoming_intersections": "full_neighbor_polygon",
        "neighbor_source": str(args.staging_root).strip() or "selected_phase_a_records",
        "staging_variant": args.staging_variant if str(args.staging_root).strip() else "",
        "split_stats": split_stats,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.trace_point_count != DEFAULT_TRACE_POINT_COUNT:
        raise ValueError("This Phase B recipe requires --trace-point-count 3.")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    phase_b_root = dataset_root / args.phase_b
    if phase_b_root.exists() and args.overwrite:
        shutil.rmtree(phase_b_root)
    split_stats = {}
    for split in args.splits:
        print(f"[phase-b] deriving split={split} dataset={dataset_root}", flush=True)
        split_stats[split] = derive_split(dataset_root, split, args)
    update_dataset_info(dataset_root, split_stats, args)
    print(json.dumps({
        "status": "passed",
        "dataset_root": str(dataset_root),
        "phase_b": str(phase_b_root),
        "splits": split_stats,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
