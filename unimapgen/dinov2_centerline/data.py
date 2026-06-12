"""Data preparation helpers for the minimal DINOv2 centerline route.

The production DINOv2 centerline branch used two target-cleaning steps before
SFT:

1. Ramer-Douglas-Peucker simplification, default epsilon 2.5 px.
2. Endpoint/heading fragment merge, default 6 px and 22.5 degrees.

This module keeps those steps together so the public training entry can expose
a single ``--prepare-trainroot`` switch instead of several experiment scripts.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from unimapgen.data.rc_centerline_douglas_utils import (
    dedup_points,
    merge_target_line_fragments,
    ramer_douglas_peucker,
    simplify_for_json,
    sort_lines,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def extract_assistant_lines(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    for message in reversed(list(record.get("messages", []))):
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        try:
            payload = json.loads(str(message.get("content", "")).strip())
        except json.JSONDecodeError:
            return []
        raw_lines = payload.get("lines", []) if isinstance(payload, dict) else []
        return raw_lines if isinstance(raw_lines, list) else []
    return []


def simplify_line(
    line: Dict[str, Any],
    *,
    patch_size: int,
    epsilon_px: float,
) -> Dict[str, Any] | None:
    points = np.asarray(line.get("points", []), dtype=np.float32)
    points = dedup_points(points)
    if points.ndim != 2 or points.shape[0] < 2:
        return None

    first = points[0].copy()
    last = points[-1].copy()
    points = ramer_douglas_peucker(points, epsilon_px=float(epsilon_px))
    points = dedup_points(points)
    if points.ndim != 2 or points.shape[0] < 2:
        points = np.asarray([first, last], dtype=np.float32)

    points_json = simplify_for_json(points, patch_size=int(patch_size))
    if len(points_json) < 2:
        return None
    return {
        "category": str(line.get("category", "centerline")),
        "start_type": str(line.get("start_type", "")),
        "end_type": str(line.get("end_type", "")),
        "points": points_json,
    }


def simplify_lines(
    lines: Any,
    *,
    patch_size: int,
    epsilon_px: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not isinstance(lines, list):
        lines = []
    out: List[Dict[str, Any]] = []
    before_points = 0
    after_points = 0
    for raw_line in lines:
        if not isinstance(raw_line, dict):
            continue
        raw_points = raw_line.get("points", [])
        before_points += int(len(raw_points) if isinstance(raw_points, list) else 0)
        simplified = simplify_line(raw_line, patch_size=int(patch_size), epsilon_px=float(epsilon_px))
        if simplified is None:
            continue
        after_points += int(len(simplified.get("points", [])))
        out.append(simplified)
    return sort_lines(out), {
        "line_count_before_douglas": int(len(lines)),
        "line_count_after_douglas": int(len(out)),
        "point_count_before_douglas": int(before_points),
        "point_count_after_douglas": int(after_points),
    }


def merge_lines(
    lines: Sequence[Dict[str, Any]],
    *,
    patch_size: int,
    endpoint_tol_px: float,
    heading_tol_deg: float,
) -> List[Dict[str, Any]]:
    return merge_target_line_fragments(
        list(lines),
        patch_size=int(patch_size),
        endpoint_tol_px=float(endpoint_tol_px),
        heading_tol_deg=float(heading_tol_deg),
    )


def update_assistant_payload(record: Dict[str, Any], target_lines: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    messages = list(record.get("messages", []))
    assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if str(messages[idx].get("role", "")).strip().lower() == "assistant":
            assistant_idx = idx
            break
    if assistant_idx is None:
        raise ValueError(f"record has no assistant message: {record.get('id', '<unknown>')}")

    try:
        payload = json.loads(str(messages[assistant_idx].get("content", "")).strip())
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["lines"] = list(target_lines)
    messages[assistant_idx] = {
        **messages[assistant_idx],
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }
    return {**record, "messages": messages}


def _record_top_media_dirs(records: Sequence[Dict[str, Any]]) -> List[str]:
    names = set()
    for record in records:
        for image_rel in record.get("images", []):
            parts = Path(str(image_rel).replace("\\", "/")).parts
            if parts:
                names.add(str(parts[0]))
    return sorted(names)


def link_or_copy_media_dirs(input_root: Path, output_root: Path, records: Sequence[Dict[str, Any]]) -> str:
    modes: List[str] = []
    for name in _record_top_media_dirs(records):
        src = input_root / name
        dst = output_root / name
        if not src.exists():
            modes.append(f"{name}:missing_source")
            continue
        if dst.exists() or dst.is_symlink():
            modes.append(f"{name}:already_exists")
            continue
        try:
            dst.symlink_to(src, target_is_directory=True)
            modes.append(f"{name}:symlink")
        except OSError:
            shutil.copytree(src, dst)
            modes.append(f"{name}:copytree")
    return ",".join(modes) if modes else "no_media_dirs"


def rewrite_split(
    *,
    split_name: str,
    input_root: Path,
    output_root: Path,
    patch_size: int,
    douglas_epsilon_px: float,
    merge_endpoint_tol_px: float,
    merge_heading_tol_deg: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    records_path = input_root / f"{split_name}.jsonl"
    meta_path = input_root / f"meta_{split_name}.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"missing split records: {records_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing split metadata: {meta_path}")

    records = load_jsonl(records_path)
    meta_rows = load_jsonl(meta_path)
    if len(records) != len(meta_rows):
        raise ValueError(f"{split_name}: record/meta length mismatch: {len(records)} vs {len(meta_rows)}")

    stats: Dict[str, Any] = {
        "rows": len(records),
        "rows_changed_by_douglas": 0,
        "rows_changed_by_merge": 0,
        "line_count_before_total": 0,
        "line_count_after_douglas_total": 0,
        "line_count_after_merge_total": 0,
        "point_count_before_total": 0,
        "point_count_after_douglas_total": 0,
    }

    out_records: List[Dict[str, Any]] = []
    out_meta_rows: List[Dict[str, Any]] = []
    target_by_id: Dict[str, List[Dict[str, Any]]] = {}

    for idx, meta in enumerate(meta_rows, start=1):
        sample_id = str(meta.get("id", ""))
        source_lines = meta.get("target_lines", [])
        if not isinstance(source_lines, list) or not source_lines:
            source_lines = extract_assistant_lines(records[idx - 1])

        simplified, douglas_stats = simplify_lines(
            source_lines,
            patch_size=int(patch_size),
            epsilon_px=float(douglas_epsilon_px),
        )
        merged = merge_lines(
            simplified,
            patch_size=int(patch_size),
            endpoint_tol_px=float(merge_endpoint_tol_px),
            heading_tol_deg=float(merge_heading_tol_deg),
        )
        target_by_id[sample_id] = merged

        before_lines = int(douglas_stats["line_count_before_douglas"])
        after_douglas_lines = int(douglas_stats["line_count_after_douglas"])
        after_merge_lines = int(len(merged))
        before_points = int(douglas_stats["point_count_before_douglas"])
        after_points = int(douglas_stats["point_count_after_douglas"])

        stats["line_count_before_total"] += before_lines
        stats["line_count_after_douglas_total"] += after_douglas_lines
        stats["line_count_after_merge_total"] += after_merge_lines
        stats["point_count_before_total"] += before_points
        stats["point_count_after_douglas_total"] += after_points
        if before_lines != after_douglas_lines or before_points != after_points:
            stats["rows_changed_by_douglas"] += 1
        if after_douglas_lines != after_merge_lines:
            stats["rows_changed_by_merge"] += 1

        updated_meta = dict(meta)
        updated_meta["target_lines"] = merged
        updated_meta["num_target_lines"] = after_merge_lines
        updated_meta["target_sampling_mode"] = "douglas_merge"
        updated_meta["douglas_epsilon_px"] = float(douglas_epsilon_px)
        updated_meta["target_merge_endpoint_tol_px"] = float(merge_endpoint_tol_px)
        updated_meta["target_merge_heading_tol_deg"] = float(merge_heading_tol_deg)
        updated_meta["target_lines_before_douglas"] = before_lines
        updated_meta["target_lines_after_douglas"] = after_douglas_lines
        updated_meta["target_lines_after_merge"] = after_merge_lines
        updated_meta["target_points_before_douglas"] = before_points
        updated_meta["target_points_after_douglas"] = after_points
        out_meta_rows.append(updated_meta)

        if idx == 1 or idx % 5000 == 0 or idx == len(meta_rows):
            print(
                json.dumps(
                    {
                        "stage": "prepare_dinov2_centerline_trainroot",
                        "split": split_name,
                        "index": idx,
                        "rows_total": len(meta_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for record in records:
        sample_id = str(record.get("id", ""))
        if sample_id not in target_by_id:
            raise KeyError(f"{split_name}: missing prepared target for record id {sample_id}")
        out_records.append(update_assistant_payload(record, target_by_id[sample_id]))

    write_jsonl(output_root / f"{split_name}.jsonl", out_records)
    write_jsonl(output_root / f"meta_{split_name}.jsonl", out_meta_rows)
    return out_records, out_meta_rows, stats


def summarize_meta(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "tiles": len({str(row.get("tile_id", "")) for row in rows}),
        "cities": dict(Counter(str(row.get("city", "")) for row in rows)),
        "source_modes": dict(Counter(str(row.get("source_mode", "")) for row in rows)),
    }


def prepare_trainroot(
    *,
    input_root: Path,
    output_root: Path,
    splits: Sequence[str] = ("train", "val"),
    patch_size: int = 512,
    douglas_epsilon_px: float = 2.5,
    merge_endpoint_tol_px: float = 6.0,
    merge_heading_tol_deg: float = 22.5,
    link_media_dirs: bool = True,
) -> Dict[str, Any]:
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if input_root == output_root:
        raise ValueError("input_root and output_root must be different for safe trainroot preparation.")
    ensure_dir(output_root)

    source_info_path = input_root / "dataset_info.json"
    source_info = load_json(source_info_path) if source_info_path.is_file() else {}

    split_summaries: Dict[str, Any] = {}
    all_records: List[Dict[str, Any]] = []
    all_meta: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in splits:
        records, meta_rows, stats = rewrite_split(
            split_name=str(split_name),
            input_root=input_root,
            output_root=output_root,
            patch_size=int(patch_size),
            douglas_epsilon_px=float(douglas_epsilon_px),
            merge_endpoint_tol_px=float(merge_endpoint_tol_px),
            merge_heading_tol_deg=float(merge_heading_tol_deg),
        )
        all_records.extend(records)
        all_meta[str(split_name)] = meta_rows
        split_summaries[str(split_name)] = {
            "meta": summarize_meta(meta_rows),
            "target_stats": stats,
        }

    media_link_mode = "disabled"
    if bool(link_media_dirs):
        media_link_mode = link_or_copy_media_dirs(input_root, output_root, all_records)

    dataset_info = dict(source_info) if isinstance(source_info, dict) else {}
    dataset_info.update(
        {
            "target_sampling_mode": "douglas_merge",
            "patch_size": int(patch_size),
            "douglas_epsilon_px": float(douglas_epsilon_px),
            "target_merge_endpoint_tol_px": float(merge_endpoint_tol_px),
            "target_merge_heading_tol_deg": float(merge_heading_tol_deg),
            "source_trainroot": str(input_root),
            "prepared_trainroot": str(output_root),
            "media_link_mode": media_link_mode,
            "splits": split_summaries,
        }
    )
    write_json(output_root / "dataset_info.json", dataset_info)
    return dataset_info


__all__ = [
    "prepare_trainroot",
    "rewrite_split",
    "simplify_lines",
    "merge_lines",
    "load_jsonl",
    "write_jsonl",
]

