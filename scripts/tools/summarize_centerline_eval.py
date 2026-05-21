#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_index.line_eval import evaluate_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--buffer-size", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=0.33)
    args = parser.parse_args()

    records = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    total = len(records)
    parse_ok = sum(1 for x in records if x.get("parse_ok"))
    non_empty = sum(1 for x in records if x.get("prediction", "").strip())
    avg_items = (sum(x.get("num_items", 0) for x in records) / total) if total else 0.0

    summary = {
        "summary_json": args.summary_json,
        "total": total,
        "parse_ok": parse_ok,
        "parse_ok_rate": (parse_ok / total) if total else 0.0,
        "non_empty": non_empty,
        "non_empty_rate": (non_empty / total) if total else 0.0,
        "avg_num_items": avg_items,
        "line_eval": evaluate_records(
            records,
            meter_per_pixel=args.meter_per_pixel,
            buffer_size=args.buffer_size,
            match_threshold=args.match_threshold,
        ),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
