#!/usr/bin/env python3
"""Compute target-ROI black ratios for an existing E2E inference JSONL."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"JSONL line {line_number} is not an object: {path}")
            records.append(payload)
    if not records:
        raise ValueError(f"Inference JSONL is empty: {path}")
    return records


def resolve_image_path(record: dict[str, Any], image_root: Path) -> Path:
    raw = str(record.get("image") or record.get("image_path") or "").strip()
    if not raw:
        raise ValueError(f"Record has no image path: {record.get('id') or record.get('record_id')}")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [image_root / path]
    if "images" in path.parts:
        candidates.append(image_root / Path(*path.parts[path.parts.index("images") :]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve image {raw!r} below {image_root}")


def target_roi(record: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    raw = record.get("target_roi_in_image", meta.get("target_roi_in_image"))
    if raw is None:
        return 0, 0, width, height
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"Invalid target_roi_in_image: {raw!r}")
    x0, y0, x1, y1 = (int(value) for value in raw)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Target ROI {raw!r} is outside image size {width}x{height}")
    return x0, y0, x1, y1


def compute_record(record: dict[str, Any], image_root: Path) -> dict[str, Any]:
    record_id = str(record.get("record_id") or record.get("id") or "").strip()
    if not record_id:
        raise ValueError("Inference record has no record_id/id")
    image_path = resolve_image_path(record, image_root)
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"))
    x0, y0, x1, y1 = target_roi(record, array.shape[1], array.shape[0])
    target = array[y0:y1, x0:x1]
    return {
        "id": record_id,
        "black_ratio": float(np.mean(target == 0)),
        "image_path": str(image_path),
        "target_roi_in_image": [x0, y0, x1, y1],
    }


def build_manifest(
    infer_jsonl: Path,
    image_root: Path,
    output_json: Path,
    workers: int,
) -> dict[str, Any]:
    records = load_records(infer_jsonl)
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    def process(record: dict[str, Any]) -> dict[str, Any]:
        return compute_record(record, image_root)

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        for index, item in enumerate(executor.map(process, records), start=1):
            if item["id"] in seen:
                raise ValueError(f"Duplicate inference record ID: {item['id']}")
            seen.add(item["id"])
            results.append(item)
            if index % 1000 == 0 or index == len(records):
                print(f"[black-ratio-manifest] processed={index}/{len(records)}", flush=True)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_json)

    ratios = [item["black_ratio"] for item in results]
    summary = {
        "infer_jsonl": str(infer_jsonl),
        "image_root": str(image_root),
        "output_json": str(output_json),
        "records": len(results),
        "black_ratio_lt_0.98": sum(value < 0.98 for value in ratios),
        "black_ratio_ge_0.98_lt_1.0": sum(0.98 <= value < 1.0 for value in ratios),
        "black_ratio_eq_1.0": sum(value == 1.0 for value in ratios),
        "black_ratio_min": min(ratios),
        "black_ratio_max": max(ratios),
    }
    summary_path = output_json.with_name(f"{output_json.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    infer_jsonl = args.infer_jsonl.resolve()
    image_root = args.image_root.resolve()
    output_json = args.output_json.resolve()
    if not infer_jsonl.is_file():
        raise FileNotFoundError(f"Inference JSONL not found: {infer_jsonl}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    build_manifest(infer_jsonl, image_root, output_json, args.workers)


if __name__ == "__main__":
    main()
