#!/usr/bin/env python3
"""Count QA samples in a prepared LLMapGen trainroot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True, help="Prepared trainroot directory.")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to count. Default: train,val,test.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="Parse every JSONL row and fail on invalid JSON.",
    )
    parser.add_argument(
        "--count-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also count image files under trainroot/images. Default: true.",
    )
    return parser.parse_args()


def split_to_paths(trainroot: Path, split: str) -> tuple[Path, Path]:
    if split == "val":
        return trainroot / "val.jsonl", trainroot / "meta_val.jsonl"
    return trainroot / f"{split}.jsonl", trainroot / f"meta_{split}.jsonl"


def count_jsonl(path: Path, *, strict: bool) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if strict:
                try:
                    json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL row: {path}:{line_no}: {exc}") from exc
            count += 1
    return count


def count_images(root: Path) -> Dict[str, Any]:
    images_root = root / "images"
    result: Dict[str, Any] = {
        "root": str(images_root),
        "exists": images_root.is_dir(),
        "total": 0,
        "by_extension": {},
    }
    if not images_root.is_dir():
        return result
    by_ext: Dict[str, int] = {}
    total = 0
    for path in images_root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        total += 1
        by_ext[ext] = by_ext.get(ext, 0) + 1
    result["total"] = total
    result["by_extension"] = dict(sorted(by_ext.items()))
    return result


def load_dataset_info(trainroot: Path) -> Dict[str, Any]:
    path = trainroot / "dataset_info.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_error": f"invalid json: {path}"}


def summarize(trainroot: Path, splits: Iterable[str], *, strict: bool, include_images: bool) -> Dict[str, Any]:
    split_summaries: Dict[str, Any] = {}
    total_records = 0
    total_meta = 0
    mismatches: List[str] = []

    for split in splits:
        records_path, meta_path = split_to_paths(trainroot, split)
        records_count = count_jsonl(records_path, strict=strict)
        meta_count = count_jsonl(meta_path, strict=strict)
        exists = records_path.is_file()
        meta_exists = meta_path.is_file()
        total_records += records_count
        total_meta += meta_count
        mismatch = bool(exists and meta_exists and records_count != meta_count)
        if mismatch:
            mismatches.append(split)
        split_summaries[split] = {
            "records_path": str(records_path),
            "records_exists": exists,
            "records": records_count,
            "meta_path": str(meta_path),
            "meta_exists": meta_exists,
            "meta": meta_count,
            "meta_records_match": not mismatch,
        }

    summary: Dict[str, Any] = {
        "trainroot": str(trainroot),
        "dataset_info": load_dataset_info(trainroot),
        "splits": split_summaries,
        "total_records": total_records,
        "total_meta": total_meta,
        "mismatch_splits": mismatches,
    }
    if include_images:
        summary["images"] = count_images(trainroot)
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("TRAINROOT SAMPLE COUNT")
    print(f"trainroot: {summary['trainroot']}")
    dataset_info = summary.get("dataset_info") or {}
    if dataset_info:
        interesting = {
            key: dataset_info.get(key)
            for key in (
                "task",
                "patch_size",
                "coord_max",
                "source_dataset_coord_mode",
                "source_dataset_coord_range",
            )
            if key in dataset_info
        }
        if interesting:
            print(f"dataset_info: {json.dumps(interesting, ensure_ascii=False)}")
    print()
    print("split           records        meta   status")
    print("-------------- ----------- ----------- ----------------")
    for split, item in summary["splits"].items():
        status = []
        if not item["records_exists"]:
            status.append("missing records")
        if not item["meta_exists"]:
            status.append("missing meta")
        if item["records_exists"] and item["meta_exists"] and not item["meta_records_match"]:
            status.append("records/meta mismatch")
        if not status:
            status.append("ok")
        print(f"{split:<14} {item['records']:>11} {item['meta']:>11} {', '.join(status)}")
    print("-------------- ----------- ----------- ----------------")
    print(f"{'TOTAL':<14} {summary['total_records']:>11} {summary['total_meta']:>11}")
    images = summary.get("images")
    if images:
        print()
        print(f"images: {images['total']} files under {images['root']}")
        if images.get("by_extension"):
            print(f"image_extensions: {json.dumps(images['by_extension'], ensure_ascii=False)}")
    if summary.get("mismatch_splits"):
        print()
        print(f"WARNING: records/meta mismatch in splits: {', '.join(summary['mismatch_splits'])}")


def main() -> None:
    args = parse_args()
    trainroot = Path(args.trainroot).expanduser().resolve()
    if not trainroot.is_dir():
        raise FileNotFoundError(f"trainroot not found: {trainroot}")
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits cannot be empty")
    summary = summarize(
        trainroot,
        splits,
        strict=bool(args.strict_jsonl),
        include_images=bool(args.count_images),
    )
    print_summary(summary)
    if str(args.output_json).strip():
        output_path = Path(str(args.output_json)).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
