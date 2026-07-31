#!/usr/bin/env python3
"""Prepare safe prediction JSON inputs for the untouched RC E2E formatter."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def clean_centerline(line: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    points = line.get("points")
    if not isinstance(points, list):
        return None, 1

    cleaned = []
    invalid_points = 0
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            invalid_points += 1
            continue
        if not valid_number(point[0]) or not valid_number(point[1]):
            invalid_points += 1
            continue
        cleaned.append([point[0], point[1]])
    if len(cleaned) < 2:
        return None, invalid_points

    result = dict(line)
    result["points"] = cleaned
    return result, invalid_points


def sanitize_prediction(text: Any) -> tuple[str, dict[str, Any]]:
    stats = {
        "prediction_parse_error": "",
        "invalid_items": 0,
        "invalid_points": 0,
        "dropped_centerlines": 0,
        "kept_centerlines": 0,
        "changed": False,
    }
    try:
        payload = json.loads(text) if isinstance(text, str) else text
    except Exception as exc:
        stats["prediction_parse_error"] = repr(exc)
        stats["changed"] = True
        return '{"lines":[]}', stats

    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        stats["prediction_parse_error"] = "prediction payload does not contain a lines list"
        stats["changed"] = True
        return '{"lines":[]}', stats

    cleaned_lines = []
    for item in payload["lines"]:
        if not isinstance(item, dict) or "category" not in item:
            stats["invalid_items"] += 1
            stats["changed"] = True
            continue
        if str(item.get("category", "")).strip().lower() != "centerline":
            cleaned_lines.append(item)
            continue
        cleaned, invalid_points = clean_centerline(item)
        stats["invalid_points"] += invalid_points
        if cleaned is None:
            stats["dropped_centerlines"] += 1
            stats["changed"] = True
            continue
        if invalid_points or cleaned.get("points") != item.get("points"):
            stats["changed"] = True
        stats["kept_centerlines"] += 1
        cleaned_lines.append(cleaned)

    return json.dumps({"lines": cleaned_lines}, ensure_ascii=False, separators=(",", ":")), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report_json.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ")
    if output_dir.exists() and args.reset:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    affected = []
    outer_errors = []
    totals = {
        "input_files": len(files),
        "output_files": 0,
        "changed_files": 0,
        "prediction_parse_errors": 0,
        "invalid_items": 0,
        "invalid_points": 0,
        "dropped_centerlines": 0,
        "kept_centerlines": 0,
    }

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("outer prediction payload is not an object")
        except Exception as exc:
            outer_errors.append({"file": str(path), "error": repr(exc)})
            continue

        sanitized_text, stats = sanitize_prediction(payload.get("prediction_json"))
        payload["prediction_json"] = sanitized_text
        destination = output_dir / path.name
        if stats["changed"]:
            destination.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            totals["changed_files"] += 1
            affected.append(
                {
                    "file": str(path),
                    "record_id": str(payload.get("record_id") or payload.get("id") or ""),
                    **stats,
                }
            )
        else:
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)

        totals["output_files"] += 1
        totals["prediction_parse_errors"] += int(bool(stats["prediction_parse_error"]))
        for key in ("invalid_items", "invalid_points", "dropped_centerlines", "kept_centerlines"):
            totals[key] += int(stats[key])

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policy": (
            "The original RC E2E project is unchanged. Invalid prediction JSON becomes an empty lines payload; "
            "invalid 2D points are removed; centerlines with fewer than two valid points are removed."
        ),
        "totals": totals,
        "outer_errors": outer_errors,
        "affected_records": affected,
    }
    report["complete"] = not outer_errors and totals["input_files"] == totals["output_files"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": totals, "outer_errors": outer_errors, "report": str(report_path)}, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
