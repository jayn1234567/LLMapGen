#!/usr/bin/env python3
"""Validate a prepared DINOv2 centerline trainroot.

This script is read-only. It checks that the output of
prepare_di_qa_trainroot.py matches the layout consumed by
scripts/train_dinov2_centerline.py:

    trainroot/
      train.jsonl
      val.jsonl
      meta_train.jsonl
      meta_val.jsonl
      images/
      dataset_info.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True, help="Prepared trainroot directory.")
    parser.add_argument("--coord-max", type=int, default=512, help="Maximum valid output coordinate.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to validate. Default: train,val.")
    parser.add_argument("--expect-train-count", type=int, default=0)
    parser.add_argument("--expect-val-count", type=int, default=0)
    parser.add_argument(
        "--check-images",
        choices=["none", "sampled", "all"],
        default="sampled",
        help="Check image paths. sampled checks the first N and then every stride-th row.",
    )
    parser.add_argument("--image-check-limit", type=int, default=2000)
    parser.add_argument("--image-check-stride", type=int, default=5000)
    parser.add_argument("--max-errors", type=int, default=30)
    parser.add_argument("--quiet-valid-samples", action="store_true")
    return parser.parse_args()


class Issues:
    def __init__(self, max_errors: int) -> None:
        self.max_errors = max(1, int(max_errors))
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.error_count = 0
        self.warning_count = 0

    def error(self, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < self.max_errors:
            self.errors.append(str(message))

    def warning(self, message: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < self.max_errors:
            self.warnings.append(str(message))


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{num_bytes}B"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def iter_jsonl(path: Path, issues: Issues) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        issues.error(f"missing jsonl: {path}")
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                issues.error(f"{path.name}:{line_no}: json decode error: {exc}")
                continue
            if not isinstance(payload, dict):
                issues.error(f"{path.name}:{line_no}: expected dict row, got {type(payload).__name__}")
                continue
            yield line_no, payload


def parse_assistant_json(text: Any, context: str, issues: Issues) -> Dict[str, Any]:
    try:
        payload = json.loads(str(text).strip())
    except Exception as exc:
        issues.error(f"{context}: assistant content is not valid JSON: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.error(f"{context}: assistant JSON must be a dict")
        return {}
    return payload


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def validate_lines(lines: Any, *, coord_max: int, context: str, issues: Issues) -> Dict[str, int]:
    stats = {
        "line_count": 0,
        "empty_samples": 0,
        "point_count": 0,
        "invalid_lines": 0,
        "coord_out_of_range": 0,
    }
    if lines is None:
        stats["empty_samples"] = 1
        return stats
    if not isinstance(lines, list):
        issues.error(f"{context}: lines/target_lines must be a list, got {type(lines).__name__}")
        stats["invalid_lines"] += 1
        return stats
    if not lines:
        stats["empty_samples"] = 1
        return stats

    for line_idx, line in enumerate(lines):
        if not isinstance(line, dict):
            issues.error(f"{context}: line[{line_idx}] must be dict, got {type(line).__name__}")
            stats["invalid_lines"] += 1
            continue
        points = line.get("points", line.get("point", []))
        if not isinstance(points, list) or len(points) < 2:
            issues.error(f"{context}: line[{line_idx}] has fewer than 2 points")
            stats["invalid_lines"] += 1
            continue
        stats["line_count"] += 1
        for point_idx, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) < 2 or not is_number(point[0]) or not is_number(point[1]):
                issues.error(f"{context}: line[{line_idx}].points[{point_idx}] is not numeric xy")
                stats["invalid_lines"] += 1
                continue
            x = float(point[0])
            y = float(point[1])
            stats["point_count"] += 1
            if x < 0 or y < 0 or x > float(coord_max) or y > float(coord_max):
                stats["coord_out_of_range"] += 1
                issues.error(f"{context}: coordinate out of range [0,{coord_max}]: [{x},{y}]")
    return stats


def get_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("messages", [])
    return raw if isinstance(raw, list) else []


def get_assistant_content(record: Dict[str, Any]) -> str:
    for message in reversed(get_messages(record)):
        if isinstance(message, dict) and str(message.get("role", "")).strip().lower() == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def get_user_count(record: Dict[str, Any]) -> int:
    count = 0
    for message in get_messages(record):
        if isinstance(message, dict) and str(message.get("role", "")).strip().lower() == "user":
            count += 1
    return count


def should_check_image(index: int, *, mode: str, limit: int, stride: int) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if index <= max(0, int(limit)):
        return True
    return int(stride) > 0 and index % int(stride) == 0


def image_size(path: Path) -> Tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                return tuple(int(v) for v in struct.unpack(">II", header[16:24]))
            if header[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    if b != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if marker in {
                        b"\xc0",
                        b"\xc1",
                        b"\xc2",
                        b"\xc3",
                        b"\xc5",
                        b"\xc6",
                        b"\xc7",
                        b"\xc9",
                        b"\xca",
                        b"\xcb",
                        b"\xcd",
                        b"\xce",
                        b"\xcf",
                    }:
                        length = struct.unpack(">H", f.read(2))[0]
                        data = f.read(length - 2)
                        h, w = struct.unpack(">HH", data[1:5])
                        return int(w), int(h)
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    length_bytes = f.read(2)
                    if len(length_bytes) != 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]
                    f.seek(length - 2, os.SEEK_CUR)
    except OSError:
        return None
    return None


def add_line_stats(target: Dict[str, int], stats: Dict[str, int]) -> None:
    for key, value in stats.items():
        target[key] = int(target.get(key, 0)) + int(value)


def scan_records(
    *,
    trainroot: Path,
    split: str,
    path: Path,
    coord_max: int,
    image_mode: str,
    image_limit: int,
    image_stride: int,
    issues: Issues,
) -> Dict[str, Any]:
    ids: Set[str] = set()
    duplicate_ids = 0
    image_checked = 0
    image_missing = 0
    image_dims: Counter[str] = Counter()
    line_stats = {
        "line_count": 0,
        "empty_samples": 0,
        "point_count": 0,
        "invalid_lines": 0,
        "coord_out_of_range": 0,
    }
    role_counter: Counter[str] = Counter()
    first_record_preview: Dict[str, Any] = {}
    rows = 0

    for rows, (line_no, record) in enumerate(iter_jsonl(path, issues), start=1):
        sample_id = str(record.get("id", "")).strip()
        if not sample_id:
            issues.error(f"{path.name}:{line_no}: missing id")
        elif sample_id in ids:
            duplicate_ids += 1
            issues.error(f"{path.name}:{line_no}: duplicate id {sample_id}")
        else:
            ids.add(sample_id)

        images = record.get("images", [])
        if not isinstance(images, list) or not images or not str(images[0]).strip():
            issues.error(f"{path.name}:{line_no}: missing non-empty images list")
        elif should_check_image(rows, mode=image_mode, limit=image_limit, stride=image_stride):
            rel_image = str(images[0]).replace("\\", "/").lstrip("/")
            image_path = trainroot / rel_image
            image_checked += 1
            if not image_path.is_file():
                image_missing += 1
                issues.error(f"{path.name}:{line_no}: image not found: {rel_image}")
            else:
                dims = image_size(image_path)
                image_dims[str(dims) if dims else "unknown"] += 1

        messages = get_messages(record)
        if not messages:
            issues.error(f"{path.name}:{line_no}: messages is empty or missing")
        for message in messages:
            if isinstance(message, dict):
                role_counter[str(message.get("role", "")).strip().lower()] += 1
        if get_user_count(record) <= 0:
            issues.error(f"{path.name}:{line_no}: no user message")
        assistant_text = get_assistant_content(record)
        if not assistant_text:
            issues.error(f"{path.name}:{line_no}: no assistant message")
            payload = {}
        else:
            payload = parse_assistant_json(assistant_text, f"{path.name}:{line_no}", issues)
        add_line_stats(
            line_stats,
            validate_lines(payload.get("lines", []), coord_max=coord_max, context=f"{path.name}:{line_no}", issues=issues),
        )

        if not first_record_preview:
            first_record_preview = {
                "id": sample_id,
                "images": images[:1] if isinstance(images, list) else images,
                "assistant": payload,
            }

    return {
        "path": str(path),
        "size": human_size(path.stat().st_size) if path.is_file() else "missing",
        "rows": rows,
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids,
        "ids": ids,
        "roles": dict(role_counter),
        "image_checked": image_checked,
        "image_missing": image_missing,
        "image_dims_sampled": dict(image_dims),
        "target_stats": line_stats,
        "first_record_preview": first_record_preview,
    }


def scan_meta(*, split: str, path: Path, coord_max: int, issues: Issues) -> Dict[str, Any]:
    ids: Set[str] = set()
    duplicate_ids = 0
    line_stats = {
        "line_count": 0,
        "empty_samples": 0,
        "point_count": 0,
        "invalid_lines": 0,
        "coord_out_of_range": 0,
    }
    patch_sizes: Counter[str] = Counter()
    coord_max_values: Counter[str] = Counter()
    rows = 0
    first_meta_preview: Dict[str, Any] = {}

    for rows, (line_no, meta) in enumerate(iter_jsonl(path, issues), start=1):
        sample_id = str(meta.get("id", "")).strip()
        if not sample_id:
            issues.error(f"{path.name}:{line_no}: missing id")
        elif sample_id in ids:
            duplicate_ids += 1
            issues.error(f"{path.name}:{line_no}: duplicate id {sample_id}")
        else:
            ids.add(sample_id)
        if not str(meta.get("image", "")).strip():
            issues.error(f"{path.name}:{line_no}: missing image")
        patch_sizes[str(meta.get("patch_size", ""))] += 1
        coord_max_values[str(meta.get("coord_max", ""))] += 1
        add_line_stats(
            line_stats,
            validate_lines(
                meta.get("target_lines", []),
                coord_max=coord_max,
                context=f"{path.name}:{line_no}",
                issues=issues,
            ),
        )
        if not first_meta_preview:
            first_meta_preview = {
                "id": sample_id,
                "image": meta.get("image", ""),
                "target_lines": meta.get("target_lines", []),
                "patch_size": meta.get("patch_size", ""),
                "coord_max": meta.get("coord_max", ""),
            }

    return {
        "path": str(path),
        "size": human_size(path.stat().st_size) if path.is_file() else "missing",
        "rows": rows,
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids,
        "ids": ids,
        "patch_sizes": dict(patch_sizes),
        "coord_max_values": dict(coord_max_values),
        "target_stats": line_stats,
        "first_meta_preview": first_meta_preview,
    }


def expected_count_for_split(args: argparse.Namespace, split: str) -> int:
    if split == "train":
        return int(args.expect_train_count)
    if split in {"val", "eval"}:
        return int(args.expect_val_count)
    return 0


def split_to_paths(trainroot: Path, split: str) -> Tuple[Path, Path]:
    if split == "val":
        return trainroot / "val.jsonl", trainroot / "meta_val.jsonl"
    return trainroot / f"{split}.jsonl", trainroot / f"meta_{split}.jsonl"


def main() -> None:
    args = parse_args()
    trainroot = Path(args.trainroot).expanduser().resolve()
    issues = Issues(max_errors=int(args.max_errors))
    if not trainroot.is_dir():
        raise FileNotFoundError(f"Trainroot not found: {trainroot}")

    dataset_info = load_json(trainroot / "dataset_info.json")
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits cannot be empty")

    required = ["dataset_info.json"]
    for split in splits:
        records_path, meta_path = split_to_paths(trainroot, split)
        required.append(records_path.name)
        required.append(meta_path.name)
    required.append("images")

    path_status: Dict[str, str] = {}
    for name in required:
        path = trainroot / name
        if path.exists():
            path_status[name] = "dir" if path.is_dir() else f"file:{human_size(path.stat().st_size)}"
        else:
            path_status[name] = "missing"
            issues.error(f"missing required path: {path}")

    split_summaries: Dict[str, Any] = {}
    for split in splits:
        records_path, meta_path = split_to_paths(trainroot, split)
        record_summary = scan_records(
            trainroot=trainroot,
            split=split,
            path=records_path,
            coord_max=int(args.coord_max),
            image_mode=str(args.check_images),
            image_limit=int(args.image_check_limit),
            image_stride=int(args.image_check_stride),
            issues=issues,
        )
        meta_summary = scan_meta(split=split, path=meta_path, coord_max=int(args.coord_max), issues=issues)

        record_ids = record_summary.pop("ids")
        meta_ids = meta_summary.pop("ids")
        missing_meta = sorted(list(record_ids - meta_ids))[:10]
        missing_record = sorted(list(meta_ids - record_ids))[:10]
        if missing_meta:
            issues.error(f"{split}: records without meta ids, first={missing_meta}")
        if missing_record:
            issues.error(f"{split}: meta without record ids, first={missing_record}")
        if len(record_ids) != len(meta_ids):
            issues.error(f"{split}: unique id count mismatch: records={len(record_ids)} meta={len(meta_ids)}")
        if record_summary["rows"] != meta_summary["rows"]:
            issues.error(f"{split}: row count mismatch: records={record_summary['rows']} meta={meta_summary['rows']}")

        expected = expected_count_for_split(args, split)
        if expected > 0 and int(record_summary["rows"]) != expected:
            issues.error(f"{split}: expected {expected} rows, got {record_summary['rows']}")

        if bool(args.quiet_valid_samples):
            record_summary.pop("first_record_preview", None)
            meta_summary.pop("first_meta_preview", None)
        split_summaries[split] = {
            "records": record_summary,
            "meta": meta_summary,
            "id_alignment": {
                "records_unique": len(record_ids),
                "meta_unique": len(meta_ids),
                "missing_meta_examples": missing_meta,
                "missing_record_examples": missing_record,
            },
        }

    summary = {
        "trainroot": str(trainroot),
        "coord_max": int(args.coord_max),
        "path_status": path_status,
        "dataset_info": {
            "patch_size": dataset_info.get("patch_size"),
            "coord_max": dataset_info.get("coord_max"),
            "assistant_coord_source_max": dataset_info.get("assistant_coord_source_max"),
            "meta_coord_source_max": dataset_info.get("meta_coord_source_max"),
            "source_dataset_coord_mode": dataset_info.get("source_dataset_coord_mode"),
            "source_dataset_coord_range": dataset_info.get("source_dataset_coord_range"),
            "source_dataset_patch_size": dataset_info.get("source_dataset_patch_size"),
            "allow_empty_lines": dataset_info.get("allow_empty_lines"),
            "media_mode_result": dataset_info.get("media_mode_result"),
        },
        "splits": split_summaries,
        "issue_counts": {
            "errors": issues.error_count,
            "warnings": issues.warning_count,
        },
        "errors": issues.errors,
        "warnings": issues.warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues.error_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
