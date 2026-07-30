#!/usr/bin/env python3
"""Filter E2E prediction JSON files using patch-manifest black ratios."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def record_id(payload: dict[str, Any]) -> str:
    value = payload.get("record_id") or payload.get("id")
    return str(value or "").strip()


def load_manifest(path: Path) -> tuple[dict[str, float], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Patch manifest must be a JSON list: {path}")

    ratios: dict[str, float] = {}
    ids: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"Manifest item {index} is not an object")
        item_id = record_id(item)
        if not item_id:
            raise ValueError(f"Manifest item {index} has no id")
        if "black_ratio" not in item:
            raise ValueError(f"Manifest item {index} has no black_ratio: {item_id}")
        ids.append(item_id)
        ratios[item_id] = float(item["black_ratio"])

    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    return ratios, duplicates


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-black-ratio", type=float, default=0.0)
    parser.add_argument("--max-black-ratio", type=float, default=0.98)
    parser.add_argument("--min-exclusive", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest_json.resolve()
    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Patch manifest not found: {manifest_path}")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    if output_dir == prediction_dir:
        raise ValueError("Output directory must differ from prediction directory")
    if not 0.0 <= args.min_black_ratio <= args.max_black_ratio <= 1.0:
        raise ValueError("Black-ratio bounds must satisfy 0 <= min <= max <= 1")

    if output_dir.exists() and args.reset:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.json"):
        stale_path.unlink()

    ratios, duplicate_manifest_ids = load_manifest(manifest_path)
    prediction_files = sorted(prediction_dir.glob("*.json"))
    seen_ids: list[str] = []
    missing_manifest_ids: list[str] = []
    invalid_outer_json: list[dict[str, str]] = []
    selected = 0
    excluded = 0
    selected_ratios: list[float] = []
    excluded_ratios: list[float] = []

    for path in prediction_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            item_id = record_id(payload)
            if not item_id:
                raise ValueError("missing record_id/id")
        except Exception as exc:
            invalid_outer_json.append({"file": str(path), "error": repr(exc)})
            continue

        seen_ids.append(item_id)
        ratio = ratios.get(item_id)
        if ratio is None:
            missing_manifest_ids.append(item_id)
            continue
        lower_selected = ratio > args.min_black_ratio if args.min_exclusive else ratio >= args.min_black_ratio
        if lower_selected and ratio <= args.max_black_ratio:
            link_or_copy(path, output_dir / path.name)
            selected += 1
            selected_ratios.append(ratio)
        else:
            excluded += 1
            excluded_ratios.append(ratio)

    duplicate_prediction_ids = sorted(key for key, count in Counter(seen_ids).items() if count > 1)
    missing_prediction_ids = sorted(set(ratios) - set(seen_ids))
    report = {
        "manifest_json": str(manifest_path),
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "selection": (
            f"{args.min_black_ratio} {'<' if args.min_exclusive else '<='} "
            f"black_ratio <= {args.max_black_ratio}"
        ),
        "manifest_records": len(ratios),
        "prediction_files": len(prediction_files),
        "selected_prediction_files": selected,
        "excluded_prediction_files": excluded,
        "selected_black_ratio_range": (
            [min(selected_ratios), max(selected_ratios)] if selected_ratios else []
        ),
        "excluded_black_ratio_range": (
            [min(excluded_ratios), max(excluded_ratios)] if excluded_ratios else []
        ),
        "duplicate_manifest_ids": duplicate_manifest_ids,
        "duplicate_prediction_ids": duplicate_prediction_ids,
        "missing_manifest_ids": sorted(set(missing_manifest_ids)),
        "missing_prediction_ids": missing_prediction_ids,
        "invalid_outer_json": invalid_outer_json,
    }
    report["complete"] = not any(
        report[key]
        for key in (
            "duplicate_manifest_ids",
            "duplicate_prediction_ids",
            "missing_manifest_ids",
            "missing_prediction_ids",
            "invalid_outer_json",
        )
    )
    report_path = output_dir.parent / f"{output_dir.name}_filter_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
