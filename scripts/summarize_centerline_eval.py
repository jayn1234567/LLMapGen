#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
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
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
