#!/usr/bin/env python3
"""Suppress predictions for E2E patches that contain no centerline ground truth."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def payload_lines(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict) and isinstance(value.get("lines"), list):
        return [item for item in value["lines"] if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def centerline_count(value: Any) -> int:
    return sum(
        str(item.get("category", "centerline")).strip().lower() == "centerline"
        and isinstance(item.get("points"), list)
        and len(item["points"]) >= 2
        for item in payload_lines(value)
    )


def load_gt_presence(path: Path) -> tuple[dict[str, int], list[str]]:
    result: dict[str, int] = {}
    duplicates: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Evaluation line {line_number} is not an object: {path}")
            record_id = str(record.get("record_id") or record.get("id") or "").strip()
            if not record_id:
                raise ValueError(f"Evaluation line {line_number} has no record ID: {path}")
            if record_id in result:
                duplicates.append(record_id)
            explicit_count = record.get("ground_truth_centerline_count")
            if explicit_count is not None:
                count = int(explicit_count)
                if count < 0:
                    raise ValueError(
                        f"Evaluation line {line_number} has a negative ground-truth count: {count}"
                    )
                result[record_id] = count
            else:
                gt_value = record.get("ground_truth_pixel", record.get("ground_truth", record.get("labels")))
                result[record_id] = centerline_count(gt_value)
    if not result:
        raise ValueError(f"Evaluation JSONL is empty: {path}")
    return result, sorted(set(duplicates))


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def suppress_predictions(
    eval_jsonl: Path,
    prediction_dir: Path,
    output_dir: Path,
    report_json: Path,
    *,
    reset: bool,
    strict: bool,
) -> dict[str, Any]:
    if reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.json"):
        stale.unlink()

    gt_counts, duplicate_gt_ids = load_gt_presence(eval_jsonl)
    seen: set[str] = set()
    duplicate_prediction_ids: list[str] = []
    missing_gt_ids: list[str] = []
    invalid_outer_json: list[dict[str, str]] = []
    kept_gt_positive_patches = 0
    suppressed_gt_empty_patches = 0
    suppressed_centerlines = 0
    suppressed_all_lines = 0

    prediction_files = sorted(prediction_dir.glob("*.json"))
    for path in prediction_files:
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise TypeError("outer prediction payload is not an object")
            record_id = str(record.get("record_id") or record.get("id") or "").strip()
            if not record_id:
                raise ValueError("missing record_id/id")
        except Exception as exc:
            invalid_outer_json.append({"file": str(path), "error": repr(exc)})
            continue

        if record_id in seen:
            duplicate_prediction_ids.append(record_id)
        seen.add(record_id)
        gt_count = gt_counts.get(record_id)
        if gt_count is None:
            missing_gt_ids.append(record_id)
            continue

        destination = output_dir / path.name
        if gt_count > 0:
            link_or_copy(path, destination)
            kept_gt_positive_patches += 1
            continue

        prediction_lines = payload_lines(record.get("prediction_json"))
        suppressed_centerlines += centerline_count(record.get("prediction_json"))
        suppressed_all_lines += len(prediction_lines)
        record["gt_oracle_patch_suppressed"] = True
        record["gt_oracle_ground_truth_centerline_count"] = 0
        record["gt_oracle_suppressed_prediction_line_count"] = len(prediction_lines)
        record["prediction_json"] = '{"lines":[]}'
        for key in ("prediction_json_pixel", "response_pixel", "prediction_pixel"):
            record.pop(key, None)
        record["parse_ok"] = True
        record["parse_error"] = ""
        destination.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        suppressed_gt_empty_patches += 1

    missing_prediction_ids = sorted(set(gt_counts) - seen)
    report = {
        "policy": (
            "GT-oracle diagnostic: patches with zero centerline ground-truth instances retain their "
            "prediction record but prediction_json is replaced by an empty lines payload before the "
            "original formatter and whole-map post-processing."
        ),
        "warning": "This uses ground truth to suppress false positives and is not a production metric.",
        "eval_jsonl": str(eval_jsonl),
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "gt_records": len(gt_counts),
        "prediction_files": len(prediction_files),
        "kept_gt_positive_patches": kept_gt_positive_patches,
        "suppressed_gt_empty_patches": suppressed_gt_empty_patches,
        "suppressed_centerlines": suppressed_centerlines,
        "suppressed_all_lines": suppressed_all_lines,
        "duplicate_gt_ids": duplicate_gt_ids,
        "duplicate_prediction_ids": sorted(set(duplicate_prediction_ids)),
        "missing_gt_ids": sorted(set(missing_gt_ids)),
        "missing_prediction_ids": missing_prediction_ids,
        "invalid_outer_json": invalid_outer_json,
    }
    report["complete"] = not any(
        report[key]
        for key in (
            "duplicate_gt_ids",
            "duplicate_prediction_ids",
            "missing_gt_ids",
            "missing_prediction_ids",
            "invalid_outer_json",
        )
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if strict and not report["complete"]:
        raise SystemExit(1)
    return report


def main() -> None:
    args = parse_args()
    eval_jsonl = args.eval_jsonl.resolve()
    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_json = args.report_json.resolve()
    if not eval_jsonl.is_file():
        raise FileNotFoundError(f"Evaluation JSONL not found: {eval_jsonl}")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    if output_dir == prediction_dir:
        raise ValueError("Output directory must differ from prediction directory")
    suppress_predictions(
        eval_jsonl,
        prediction_dir,
        output_dir,
        report_json,
        reset=args.reset,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
