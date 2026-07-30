#!/usr/bin/env python3
"""Verify that E2E inference outputs match the input JSONL one-to-one."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _record_id(record: dict[str, Any]) -> str:
    value = record.get("record_id") or record.get("id")
    if value is None and isinstance(record.get("meta"), dict):
        value = record["meta"].get("sample_id") or record["meta"].get("tile_id")
    return str(value or "").strip()


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def verify(infer_jsonl: Path, prediction_dir: Path) -> dict[str, Any]:
    expected_ids: list[str] = []
    input_errors: list[str] = []
    with infer_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = _record_id(record)
                if not record_id:
                    raise ValueError("missing id")
                expected_ids.append(record_id)
            except Exception as exc:
                input_errors.append(f"line {line_number}: {exc}")

    prediction_files = sorted(prediction_dir.glob("*.json"))
    actual_ids: list[str] = []
    prediction_errors: list[str] = []
    missing_prediction_field: list[str] = []
    for path in prediction_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record_id = _record_id(payload)
            if not record_id:
                raise ValueError("missing record_id/id")
            actual_ids.append(record_id)
            if not isinstance(payload.get("prediction_json"), str):
                missing_prediction_field.append(record_id)
        except Exception as exc:
            prediction_errors.append(f"{path.name}: {exc}")

    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    summary = {
        "infer_jsonl": str(infer_jsonl),
        "prediction_dir": str(prediction_dir),
        "expected_records": len(expected_ids),
        "prediction_files": len(prediction_files),
        "valid_prediction_files": len(actual_ids),
        "unique_expected_ids": len(expected_set),
        "unique_prediction_ids": len(actual_set),
        "duplicate_expected_ids": _duplicates(expected_ids),
        "duplicate_prediction_ids": _duplicates(actual_ids),
        "missing_prediction_ids": sorted(expected_set - actual_set),
        "unexpected_prediction_ids": sorted(actual_set - expected_set),
        "input_errors": input_errors,
        "prediction_errors": prediction_errors,
        "missing_prediction_json": sorted(missing_prediction_field),
    }
    summary["complete"] = not any(
        summary[key]
        for key in (
            "duplicate_expected_ids",
            "duplicate_prediction_ids",
            "missing_prediction_ids",
            "unexpected_prediction_ids",
            "input_errors",
            "prediction_errors",
            "missing_prediction_json",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-jsonl", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not args.infer_jsonl.is_file():
        raise FileNotFoundError(f"Inference JSONL not found: {args.infer_jsonl}")
    if not args.prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {args.prediction_dir}")

    summary = verify(args.infer_jsonl, args.prediction_dir)
    if args.expected_count > 0 and summary["expected_records"] != args.expected_count:
        summary["complete"] = False
        summary["expected_count_error"] = {
            "required": args.expected_count,
            "found": summary["expected_records"],
        }

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
