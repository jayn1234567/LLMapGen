#!/usr/bin/env python3
"""Validate that a streamed inference JSONL covers the requested input set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for index, record in enumerate(payload):
                if not isinstance(record, dict):
                    raise ValueError(f"Record {index} in {path} is not an object")
                yield record
            return

        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record at line {line_number} in {path} is not an object")
            yield record


def record_id(record: dict[str, Any]) -> str:
    for key in ("id", "sample_id", "record_id"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-jsonl", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--expected-start-index", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--dataset-root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest: dict[str, Any] = {
        "status": "failed",
        "expected_jsonl": str(args.expected_jsonl.resolve()),
        "prediction_jsonl": str(args.prediction_jsonl.resolve()),
        "checkpoint_dir": args.checkpoint_dir,
        "dataset_root": args.dataset_root,
        "expected_start_index": args.expected_start_index,
        "expected_count_argument": args.expected_count,
        "expected_count": 0,
        "prediction_count": 0,
        "parse_ok_count": 0,
        "parse_error_count": 0,
        "duplicate_expected_ids": [],
        "duplicate_prediction_ids": [],
        "mismatch_count": 0,
        "mismatch_examples": [],
        "prediction_sha256": "",
    }
    errors: list[str] = []
    expected_ids: set[str] = set()
    prediction_ids: set[str] = set()

    try:
        expected_iter = iter_records(args.expected_jsonl)
        prediction_iter = iter_records(args.prediction_jsonl)
        expected_count = 0
        prediction_count = 0
        while True:
            try:
                expected = next(expected_iter)
                expected_done = False
            except StopIteration:
                expected_done = True
            try:
                prediction = next(prediction_iter)
                prediction_done = False
            except StopIteration:
                prediction_done = True

            if expected_done and prediction_done:
                break

            expected_index = args.expected_start_index + expected_count
            if expected_done:
                errors.append(f"extra prediction row at expected index {expected_index}")
                prediction_count += 1
                if isinstance(prediction, dict):
                    prediction_index = prediction.get("idx")
                    if isinstance(prediction_index, int) and prediction_index != expected_index:
                        errors.append(
                            f"prediction idx mismatch at extra row: {prediction_index} != {expected_index}"
                        )
                continue
            if prediction_done:
                errors.append(f"missing prediction row at expected index {expected_index}")
                expected_count += 1
                continue

            expected_count += 1
            prediction_count += 1
            expected_id = record_id(expected)
            prediction_id = record_id(prediction)
            if expected_id:
                if expected_id in expected_ids:
                    manifest["duplicate_expected_ids"].append(expected_id)
                expected_ids.add(expected_id)
            if prediction_id:
                if prediction_id in prediction_ids:
                    manifest["duplicate_prediction_ids"].append(prediction_id)
                prediction_ids.add(prediction_id)

            prediction_index = prediction.get("idx")
            mismatch = []
            if not isinstance(prediction_index, int) or prediction_index != expected_index:
                mismatch.append(f"idx={prediction_index!r}, expected={expected_index}")
            if expected_id != prediction_id:
                mismatch.append(f"record_id={prediction_id!r}, expected={expected_id!r}")
            if mismatch:
                manifest["mismatch_count"] += 1
                if len(manifest["mismatch_examples"]) < 20:
                    manifest["mismatch_examples"].append(
                        {"index": expected_index, "details": mismatch}
                    )

            if prediction.get("parse_ok") is True:
                manifest["parse_ok_count"] += 1
            else:
                manifest["parse_error_count"] += 1

        manifest["expected_count"] = expected_count
        manifest["prediction_count"] = prediction_count
        if args.expected_count > 0 and expected_count != args.expected_count:
            errors.append(
                f"expected input count {expected_count} does not equal requested {args.expected_count}"
            )
        if prediction_count != expected_count:
            errors.append(
                f"prediction count {prediction_count} does not equal input count {expected_count}"
            )
        if manifest["duplicate_expected_ids"] or manifest["duplicate_prediction_ids"]:
            errors.append("duplicate record ids detected")
        if manifest["mismatch_count"]:
            errors.append(f"{manifest['mismatch_count']} row identity mismatches detected")

        digest = hashlib.sha256()
        with args.prediction_jsonl.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["prediction_sha256"] = digest.hexdigest()
    except Exception as exc:  # noqa: BLE001 - manifest must be emitted on any validation failure.
        errors.append(f"validator exception: {type(exc).__name__}: {exc}")

    manifest["errors"] = errors
    manifest["status"] = "complete" if not errors else "failed"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
