#!/usr/bin/env python3
"""Merge native-Qwen3-VL shard summaries and write per-patch E2E JSON."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("id") or "").strip()


def _safe_name(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return rendered or "sample"


def merge_shards(
    infer_jsonl: Path,
    shard_root: Path,
    output_dir: Path,
    prediction_dir: Path,
    reset: bool = False,
) -> dict[str, Any]:
    expected = [
        json.loads(line)
        for line in infer_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_ids = [_record_id(record) for record in expected]
    if not all(expected_ids):
        raise ValueError(f"Input JSONL contains records without id: {infer_jsonl}")
    expected_duplicates = sorted(
        value for value, count in Counter(expected_ids).items() if count > 1
    )
    if expected_duplicates:
        raise ValueError(f"Input JSONL contains duplicate ids: {expected_duplicates[:10]}")

    shard_files = sorted(shard_root.glob("shard_*/summary.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard summary.json files found below {shard_root}")
    results: list[dict[str, Any]] = []
    for path in shard_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Shard summary must be a JSON list: {path}")
        results.extend(item for item in payload if isinstance(item, dict))

    actual_ids = [_record_id(result) for result in results]
    duplicates = sorted(value for value, count in Counter(actual_ids).items() if value and count > 1)
    result_by_id = {_record_id(result): result for result in results if _record_id(result)}
    missing = sorted(set(expected_ids) - set(result_by_id))
    unexpected = sorted(set(result_by_id) - set(expected_ids))
    if duplicates or missing or unexpected:
        raise ValueError(
            "Native inference shards are incomplete: "
            f"duplicates={duplicates[:10]} missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    ordered = [result_by_id[record_id] for record_id in expected_ids]
    output_dir.mkdir(parents=True, exist_ok=True)
    if reset and prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(ordered):
        path = prediction_dir / f"{index:06d}_{_safe_name(_record_id(result))}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.jsonl").open("w", encoding="utf-8") as handle:
        for result in ordered:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    parse_ok = sum(bool(result.get("parse_ok")) for result in ordered)
    report = {
        "infer_jsonl": str(infer_jsonl),
        "shard_root": str(shard_root),
        "shard_summaries": [str(path) for path in shard_files],
        "output_dir": str(output_dir),
        "prediction_dir": str(prediction_dir),
        "expected_records": len(expected_ids),
        "merged_records": len(ordered),
        "prediction_files": len(list(prediction_dir.glob("*.json"))),
        "parse_ok": parse_ok,
        "parse_ok_rate": parse_ok / len(ordered) if ordered else 0.0,
        "complete": True,
    }
    (output_dir / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-jsonl", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    print(json.dumps(merge_shards(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
