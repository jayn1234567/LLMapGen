#!/usr/bin/env python3
"""Create deterministic easy/medium/hard/very-hard JSONL evaluation splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.tag_hard_map_samples import (
    DEFAULT_CURVED_LINE_TURN_THRESHOLD,
    DEFAULT_EASY_MAX_CENTERLINES,
    DEFAULT_EASY_MAX_POINTS,
    DEFAULT_EASY_MAX_SINGLE_TURN,
    DEFAULT_EASY_MAX_TOTAL_TURN,
    DEFAULT_HARD_SCORE_THRESHOLD,
    DEFAULT_SHARP_TURN_THRESHOLD,
    DEFAULT_SHORT_LINE_THRESHOLD,
    DEFAULT_VERY_HARD_SCORE_THRESHOLD,
    DIFFICULTY_RULE_VERSION,
    sample_metrics,
)


DIFFICULTIES = ("easy", "medium", "hard", "very_hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-difficulty", type=int, default=300, help="0 keeps every eligible sample.")
    parser.add_argument(
        "--samples-per-difficulty-spec",
        default="",
        help=(
            "Optional per-bucket counts like "
            "easy=300,medium=300,hard=300,very_hard=100. "
            "Values override --samples-per-difficulty for listed buckets."
        ),
    )
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--include-empty", action="store_true", help="Keep empty patches in the easy bucket.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-samples", type=int, default=0)
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm1000"], default="auto")
    parser.add_argument("--coord-range", type=float, default=1000.0)
    parser.add_argument("--junction-tol", type=float, default=36.0)
    parser.add_argument("--intersection-tol", type=float, default=16.0)
    parser.add_argument("--dense-line-threshold", type=int, default=8)
    parser.add_argument("--dense-point-threshold", type=int, default=34)
    parser.add_argument("--long-total-length-threshold", type=float, default=3600.0)
    parser.add_argument("--many-cut-threshold", type=int, default=6)
    parser.add_argument("--short-line-threshold", type=float, default=DEFAULT_SHORT_LINE_THRESHOLD)
    parser.add_argument(
        "--curved-line-turn-threshold",
        type=float,
        default=DEFAULT_CURVED_LINE_TURN_THRESHOLD,
    )
    parser.add_argument("--sharp-turn-threshold", type=float, default=DEFAULT_SHARP_TURN_THRESHOLD)
    parser.add_argument("--easy-max-centerlines", type=int, default=DEFAULT_EASY_MAX_CENTERLINES)
    parser.add_argument("--easy-max-points", type=int, default=DEFAULT_EASY_MAX_POINTS)
    parser.add_argument("--easy-max-total-turn", type=float, default=DEFAULT_EASY_MAX_TOTAL_TURN)
    parser.add_argument("--easy-max-single-turn", type=float, default=DEFAULT_EASY_MAX_SINGLE_TURN)
    parser.add_argument("--hard-score-threshold", type=float, default=DEFAULT_HARD_SCORE_THRESHOLD)
    parser.add_argument(
        "--very-hard-score-threshold",
        type=float,
        default=DEFAULT_VERY_HARD_SCORE_THRESHOLD,
    )
    return parser.parse_args()


def parse_samples_per_difficulty_spec(text: str) -> dict[str, int]:
    limits: dict[str, int] = {}
    for item in re.split(r"[,;\s]+", str(text or "")):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --samples-per-difficulty-spec item: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty in --samples-per-difficulty-spec: {name!r}")
        count = int(value)
        if count < 0:
            raise ValueError(f"Sample count must be >= 0 for difficulty {name!r}: {count}")
        limits[name] = count
    return limits


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.samples_per_difficulty < 0:
        raise ValueError("--samples-per-difficulty must be >= 0")

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    requested = list(dict.fromkeys(args.difficulties))
    per_difficulty_limits = {name: args.samples_per_difficulty for name in requested}
    for name, value in parse_samples_per_difficulty_spec(args.samples_per_difficulty_spec).items():
        if name in per_difficulty_limits:
            per_difficulty_limits[name] = value
    selected: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {name: [] for name in requested}
    eligible_seen = Counter()
    source_counts = Counter()
    excluded_empty = 0
    total = 0
    rngs = {name: random.Random(args.seed + DIFFICULTIES.index(name) * 1009) for name in requested}

    with input_path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if args.max_source_samples and total >= args.max_source_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = sample_metrics(record, image_size=None, args=args)
            difficulty = str(metrics["difficulty"])
            source_counts[difficulty] += 1
            total += 1
            if difficulty not in selected:
                continue
            if not args.include_empty and metrics.get("tags") == ["empty_patch"]:
                excluded_empty += 1
                continue

            eligible_seen[difficulty] += 1
            entry = (source_index, record, metrics)
            bucket = selected[difficulty]
            limit = per_difficulty_limits[difficulty]
            if limit == 0 or len(bucket) < limit:
                bucket.append(entry)
            else:
                replacement = rngs[difficulty].randrange(eligible_seen[difficulty])
                if replacement < limit:
                    bucket[replacement] = entry

    manifest_rows = []
    all_selected_rows = []
    selected_counts = {}
    for difficulty in requested:
        bucket = sorted(selected[difficulty], key=lambda item: item[0])
        bucket_records = [record for _, record, _ in bucket]
        write_jsonl(output_dir / f"{difficulty}.jsonl", bucket_records)
        all_selected_rows.extend(bucket_records)
        selected_counts[difficulty] = len(bucket)
        for source_index, record, metrics in bucket:
            manifest_rows.append(
                {
                    "source_index": source_index,
                    "difficulty": difficulty,
                    "difficulty_score": metrics.get("difficulty_score"),
                    "tags": metrics.get("tags"),
                    "id": record.get("id", record.get("sample_id")),
                    "image": record.get("image", record.get("images")),
                    "centerline_count": metrics.get("centerline_count"),
                    "intersection_count": metrics.get("intersection_count"),
                    "coord_mode": metrics.get("coord_mode"),
                }
            )
    manifest_rows.sort(key=lambda item: (requested.index(item["difficulty"]), item["source_index"]))
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "all_selected.jsonl", all_selected_rows)

    summary = {
        "difficulty_rule_version": DIFFICULTY_RULE_VERSION,
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "source_samples": total,
        "source_difficulty_counts": dict(source_counts),
        "eligible_counts": {name: int(eligible_seen[name]) for name in requested},
        "selected_counts": selected_counts,
        "samples_per_difficulty": args.samples_per_difficulty,
        "samples_per_difficulty_spec": args.samples_per_difficulty_spec,
        "samples_per_difficulty_by_bucket": per_difficulty_limits,
        "all_selected_jsonl": str(output_dir / "all_selected.jsonl"),
        "all_selected_count": len(all_selected_rows),
        "include_empty": bool(args.include_empty),
        "excluded_empty_samples": excluded_empty,
        "seed": args.seed,
        "coord_mode": args.coord_mode,
        "coord_range": args.coord_range,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for difficulty in requested:
        print(f"[difficulty-splits] {difficulty}: {output_dir / (difficulty + '.jsonl')}")


if __name__ == "__main__":
    main()
