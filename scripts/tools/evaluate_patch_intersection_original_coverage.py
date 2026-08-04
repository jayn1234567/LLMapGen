#!/usr/bin/env python3
"""Evaluate patch intersections with the original RC E2E coverage formula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_index.intersection_coverage_eval import evaluate_intersection_coverage_records


def read_json_array_or_lines(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("patch_results", "results", "records", "samples", "summary"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Expected a JSON array/JSONL or an object containing patch result records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Inference summary.json, summary.jsonl, or eval JSONL.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--include-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    records = records_from_payload(read_json_array_or_lines(input_path))
    records = [
        record
        for record in records
        if any(key in record for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel"))
    ]
    if not records:
        raise ValueError(f"No patch records with ground truth found in {input_path}")

    metrics = evaluate_intersection_coverage_records(
        records,
        coverage_threshold=args.coverage_threshold,
        include_samples=args.include_samples,
    )
    output = {
        "input_json": str(input_path),
        "num_records": len(records),
        "intersection_original_coverage_eval": metrics["intersection"],
        "t_intersection_original_coverage_eval": metrics["t_intersection"],
        "note": metrics["note"],
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    intersection = metrics["intersection"]
    t_intersection = metrics["t_intersection"]
    if args.include_samples:
        intersection = intersection["summary"]
        t_intersection = t_intersection["summary"]
    print(intersection["table"])
    print()
    print(t_intersection["table"])
    print(f"[patch-intersection-coverage] records={len(records)} output={output_path}")


if __name__ == "__main__":
    main()
