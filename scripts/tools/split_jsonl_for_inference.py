#!/usr/bin/env python3
"""Split a JSONL deterministically into round-robin inference shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_jsonl(input_jsonl: Path, output_root: Path, num_shards: int, num_samples: int = 0) -> dict:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    lines = [line for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if num_samples > 0:
        lines = lines[:num_samples]
    if not lines:
        raise ValueError(f"No records found in {input_jsonl}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_jsonl = output_root / "selected.jsonl"
    selected_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = []
    for shard_rank in range(num_shards):
        selected = lines[shard_rank::num_shards]
        path = output_root / f"shard_{shard_rank:05d}.jsonl"
        path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
        counts.append(len(selected))
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "num_records": len(lines),
        "num_shards": num_shards,
        "shard_counts": counts,
        "selected_jsonl": str(selected_jsonl),
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(split_jsonl(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
