#!/usr/bin/env python3
"""Rebuild eval.json from an existing inference summary.json.

This is useful when a difficulty evaluation run was interrupted after
all_selected/summary.json was written but before all_selected/eval.json was
created. It does not reload the model, run inference, or render images.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="",
        help="Inference output directory containing summary.json or summary.jsonl.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Explicit summary path. Overrides --input-dir.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Where to write eval json. Defaults to input-dir/eval.json or summary sibling eval.json.",
    )
    parser.add_argument(
        "--map-task",
        choices=["lane", "lane_intersection", "auto"],
        default="lane_intersection",
        help="Use lane_intersection for centerline + intersection metrics.",
    )
    parser.add_argument("--meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--buffer-size", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=0.33)
    parser.add_argument("--intersection-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Also include per-sample metric payloads in eval.json. Larger but useful for debugging.",
    )
    return parser.parse_args()


def read_json_array_or_lines(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("patch_results", "results", "records", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        summary = payload.get("summary")
        if isinstance(summary, list):
            return [item for item in summary if isinstance(item, dict)]
    raise ValueError("Unsupported summary payload. Expected a JSON list or a dict with patch_results/results/records.")


def record_has_ground_truth(record: dict[str, Any]) -> bool:
    return any(key in record for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel"))


def record_has_intersection(record: dict[str, Any]) -> bool:
    payloads = [
        record.get("ground_truth"),
        record.get("labels"),
        record.get("ground_truth_pixel"),
        record.get("labels_pixel"),
        record.get("prediction_json"),
        record.get("prediction_json_pixel"),
        record.get("response"),
        record.get("response_pixel"),
    ]
    for payload in payloads:
        if isinstance(payload, str) and "intersection" in payload.lower():
            return True
        if isinstance(payload, dict):
            text = json.dumps(payload, ensure_ascii=False).lower()
            if "intersection" in text:
                return True
    return False


def resolve_summary_path(args: argparse.Namespace) -> Path:
    if args.summary_json:
        path = Path(args.summary_json)
        if not path.is_file():
            raise FileNotFoundError(f"summary json not found: {path}")
        return path
    if not args.input_dir:
        raise ValueError("Either --input-dir or --summary-json is required.")
    input_dir = Path(args.input_dir)
    for name in ("summary.json", "summary.jsonl"):
        path = input_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"summary.json/summary.jsonl not found under: {input_dir}")


def resolve_output_path(args: argparse.Namespace, summary_path: Path) -> Path:
    if args.output_json:
        return Path(args.output_json)
    if args.input_dir:
        return Path(args.input_dir) / "eval.json"
    return summary_path.with_name("eval.json")


def install_tqdm_fallback() -> None:
    try:
        import tqdm  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("tqdm")
        module.tqdm = lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []
        sys.modules["tqdm"] = module


def main() -> None:
    args = parse_args()
    summary_path = resolve_summary_path(args)
    output_path = resolve_output_path(args, summary_path)

    payload = read_json_array_or_lines(summary_path)
    records = records_from_payload(payload)
    records = [record for record in records if record_has_ground_truth(record)]
    if not records:
        raise ValueError(f"No evaluable records with ground truth found in {summary_path}")

    install_tqdm_fallback()
    from infer_index.line_eval import (
        evaluate_lane_intersection_records,
        evaluate_records,
        print_eval_table,
        print_lane_intersection_eval_tables,
    )

    eval_kwargs = {
        "meter_per_pixel": args.meter_per_pixel,
        "buffer_size": args.buffer_size,
        "match_threshold": args.match_threshold,
        "intersection_iou_threshold": args.intersection_iou_threshold,
        "include_samples": args.include_samples,
    }
    map_task = args.map_task
    if map_task == "auto":
        map_task = "lane_intersection" if any(record_has_intersection(record) for record in records) else "lane"

    if map_task == "lane_intersection":
        map_eval = evaluate_lane_intersection_records(records, **eval_kwargs)
        eval_summary = {
            "centerline_eval": map_eval["lane"],
            "intersection_eval": map_eval["intersection"],
            "lane_intersection_eval": map_eval["lane_intersection"],
            "lane_type_eval": map_eval["lane_type"],
            "intersection_type_eval": map_eval["intersection_type"],
            "map_eval": map_eval,
            "aggregate": {
                "num_records": len(records),
                "summary_json": str(summary_path),
            },
        }
    else:
        eval_summary = evaluate_records(records, **eval_kwargs)
        if isinstance(eval_summary, dict):
            eval_summary.setdefault("aggregate", {})
            eval_summary["aggregate"].update({"num_records": len(records), "summary_json": str(summary_path)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rebuild-eval] records: {len(records)}")
    print(f"[rebuild-eval] summary: {summary_path}")
    print(f"[rebuild-eval] eval:    {output_path}")
    if map_task == "lane_intersection":
        print_lane_intersection_eval_tables(eval_summary["map_eval"])
    else:
        print_eval_table(eval_summary)
    print(json.dumps({"eval_json": str(output_path), "num_records": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
