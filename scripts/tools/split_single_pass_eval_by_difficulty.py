#!/usr/bin/env python3
"""Split one inference summary into fixed difficulty buckets and evaluate each.

The model is run once on ``all_selected.jsonl``. This tool then uses the
corresponding fixed JSONL files to recover bucket membership without another
model load or generation pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIFFICULTIES = ("easy", "medium", "hard", "very_hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--split-root",
        required=True,
        help="Directory containing remapped easy/medium/hard/very_hard JSONL files.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--expected-counts", default="easy=300,medium=300,hard=300,very_hard=100")
    parser.add_argument("--meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--buffer-size", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=0.33)
    parser.add_argument("--intersection-iou-threshold", type=float, default=0.5)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(payload)
    return records


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("record_id", record.get("id", record.get("sample_id", "")))).strip()


def parse_expected_counts(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, raw_count = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid expected-count item: {item!r}")
        result[name.strip()] = int(raw_count)
    return result


def build_eval(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    from infer_index.line_eval import evaluate_lane_intersection_records

    map_eval = evaluate_lane_intersection_records(
        records,
        meter_per_pixel=args.meter_per_pixel,
        buffer_size=args.buffer_size,
        match_threshold=args.match_threshold,
        intersection_iou_threshold=args.intersection_iou_threshold,
    )
    return {
        "centerline_eval": map_eval["lane"],
        "intersection_eval": map_eval["intersection"],
        "lane_intersection_eval": map_eval["lane_intersection"],
        "lane_type_eval": map_eval["lane_type"],
        "intersection_type_eval": map_eval["intersection_type"],
        "map_eval": map_eval,
    }


def write_bucket(
    output_root: Path,
    name: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> Path:
    bucket_root = output_root / name
    bucket_root.mkdir(parents=True, exist_ok=True)
    summary_path = bucket_root / "summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = build_eval(records, args)
    evaluation["single_pass_split"] = metadata
    eval_path = bucket_root / "eval.json"
    eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    return eval_path


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    split_root = Path(args.split_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_payload = read_json(summary_path)
    if not isinstance(summary_payload, list):
        raise ValueError(f"Inference summary must be a JSON list: {summary_path}")
    results = [item for item in summary_payload if isinstance(item, dict)]
    result_ids = [record_id(item) for item in results]
    missing_result_ids = [index for index, value in enumerate(result_ids) if not value]
    if missing_result_ids:
        raise ValueError(f"Inference results without record_id at indexes: {missing_result_ids[:10]}")
    duplicate_result_ids = sorted(name for name, count in Counter(result_ids).items() if count > 1)
    if duplicate_result_ids:
        raise ValueError(f"Duplicate record_id values in inference summary: {duplicate_result_ids[:10]}")
    results_by_id = dict(zip(result_ids, results))

    expected_counts = parse_expected_counts(args.expected_counts)
    bucket_ids: dict[str, list[str]] = {}
    id_to_bucket: dict[str, str] = {}
    for difficulty in args.difficulties:
        split_path = split_root / f"{difficulty}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"Remapped difficulty split not found: {split_path}")
        ids = [record_id(record) for record in read_jsonl(split_path)]
        if any(not value for value in ids):
            raise ValueError(f"Empty sample ID found in {split_path}")
        expected = expected_counts.get(difficulty)
        if expected is not None and len(ids) != expected:
            raise ValueError(f"Expected {expected} records in {split_path}, found {len(ids)}")
        for sample_id in ids:
            previous = id_to_bucket.get(sample_id)
            if previous is not None:
                raise ValueError(f"Sample {sample_id} appears in both {previous} and {difficulty}")
            id_to_bucket[sample_id] = difficulty
        bucket_ids[difficulty] = ids

    expected_ids = set(id_to_bucket)
    actual_ids = set(results_by_id)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if (missing or unexpected) and not args.allow_incomplete:
        raise ValueError(
            "Single-pass inference does not match the fixed set exactly: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"missing_examples={missing[:5]}, unexpected_examples={unexpected[:5]}"
        )

    report: dict[str, Any] = {
        "summary_json": str(summary_path),
        "split_root": str(split_root),
        "output_root": str(output_root),
        "expected_total": len(expected_ids),
        "inference_total": len(results),
        "matched_total": len(expected_ids & actual_ids),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "bucket_counts": {},
        "eval_jsons": {},
    }

    combined_records: list[dict[str, Any]] = []
    for difficulty in args.difficulties:
        records = []
        for sample_id in bucket_ids[difficulty]:
            result = results_by_id.get(sample_id)
            if result is None:
                continue
            annotated = dict(result)
            annotated["difficulty_eval_bucket"] = difficulty
            records.append(annotated)
        combined_records.extend(records)
        metadata = {
            "difficulty": difficulty,
            "num_records": len(records),
            "source_summary_json": str(summary_path),
            "source_split_jsonl": str(split_root / f"{difficulty}.jsonl"),
        }
        eval_path = write_bucket(output_root, difficulty, records, args, metadata)
        report["bucket_counts"][difficulty] = len(records)
        report["eval_jsons"][difficulty] = str(eval_path)

    all_metadata = {
        "difficulty": "all_selected",
        "num_records": len(combined_records),
        "source_summary_json": str(summary_path),
        "bucket_counts": report["bucket_counts"],
    }
    all_eval_path = write_bucket(output_root, "all_selected", combined_records, args, all_metadata)
    report["eval_jsons"]["all_selected"] = str(all_eval_path)
    report["all_selected_count"] = len(combined_records)
    report_path = output_root / "single_pass_split_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    from infer_index.line_eval import print_lane_intersection_eval_tables

    for difficulty in (*args.difficulties, "all_selected"):
        eval_payload = read_json(Path(report["eval_jsons"][difficulty]))
        print(f"\n[single-pass-eval] {difficulty}: {report['bucket_counts'].get(difficulty, len(combined_records))}")
        print_lane_intersection_eval_tables(eval_payload["map_eval"])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
