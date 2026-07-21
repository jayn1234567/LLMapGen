#!/usr/bin/env python3
"""Build a deterministic, exact-ratio difficulty subset from an SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.tag_hard_map_samples import (  # noqa: E402
    DIFFICULTY_RULE_VERSION,
    sample_metrics,
)


DIFFICULTIES = ("easy", "medium", "hard", "very_hard")
DEFAULT_RATIOS = "easy=0.30,medium=0.3560290909,hard=0.2439709091,very_hard=0.10"
DIFFICULTY_ARGS = SimpleNamespace(
    coord_mode="norm1000",
    coord_range=1000.0,
    junction_tol=36.0,
    intersection_tol=16.0,
    dense_line_threshold=8,
    dense_point_threshold=34,
    long_total_length_threshold=3600.0,
    many_cut_threshold=6,
    short_line_threshold=90.0,
    curved_line_turn_threshold=45.0,
    sharp_turn_threshold=60.0,
    easy_max_centerlines=3,
    easy_max_points=16,
    easy_max_total_turn=120.0,
    easy_max_single_turn=60.0,
    hard_score_threshold=2.5,
    very_hard_score_threshold=5.5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--target-samples", type=int, default=200_000)
    parser.add_argument("--difficulty-ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--reuse-if-valid", action="store_true")
    return parser.parse_args()


def parse_ratios(spec: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid ratio item: {item!r}")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty bucket: {name!r}")
        ratios[name] = float(raw_value)
    missing = [name for name in DIFFICULTIES if name not in ratios]
    if missing:
        raise ValueError(f"Missing difficulty ratios: {missing}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Difficulty ratios must be non-negative.")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Difficulty ratios must sum to a positive value.")
    return {name: ratios[name] / total for name in DIFFICULTIES}


def allocate_quotas(target_samples: int, ratios: dict[str, float]) -> dict[str, int]:
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")
    raw = {name: target_samples * ratios[name] for name in DIFFICULTIES}
    quotas = {name: int(raw[name]) for name in DIFFICULTIES}
    remaining = target_samples - sum(quotas.values())
    order = sorted(DIFFICULTIES, key=lambda name: (raw[name] - quotas[name], name), reverse=True)
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def default_classifier(record: dict[str, Any]) -> tuple[str, bool]:
    # Keep these values aligned with data_process.build_dataset_v2.DIFFICULTY_ARGS
    # so a subset is bucketed exactly like the finalized 550k build.
    metrics = sample_metrics(record, image_size=None, args=DIFFICULTY_ARGS)
    tags = set(metrics.get("tags") or ())
    return str(metrics["difficulty"]), "empty_patch" in tags


def count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def summary_matches(
    summary_path: Path,
    input_path: Path,
    output_path: Path,
    target_samples: int,
    ratios: dict[str, float],
    seed: int,
) -> bool:
    if not summary_path.is_file() or not output_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stat = input_path.stat()
        return bool(
            summary.get("status") == "complete"
            and summary.get("source_size") == stat.st_size
            and summary.get("source_mtime_ns") == stat.st_mtime_ns
            and summary.get("target_samples") == target_samples
            and summary.get("difficulty_ratios") == ratios
            and summary.get("seed") == seed
            and summary.get("selected_counts") == allocate_quotas(target_samples, ratios)
            and count_nonempty_lines(output_path) == target_samples
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def build_subset(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    target_samples: int,
    ratios: dict[str, float],
    seed: int,
    progress_every: int = 50_000,
    classifier: Callable[[dict[str, Any]], tuple[str, bool]] = default_classifier,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output JSONL paths must differ.")

    quotas = allocate_quotas(target_samples, ratios)
    reservoirs: dict[str, list[tuple[int, int]]] = {name: [] for name in DIFFICULTIES}
    candidate_counts: Counter[str] = Counter()
    rngs = {
        name: random.Random(seed + (index + 1) * 1_000_003)
        for index, name in enumerate(DIFFICULTIES)
    }
    total_records = 0
    excluded_empty = 0

    with input_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            total_records += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at non-empty record {total_records}: {input_path}") from exc
            difficulty, is_empty = classifier(record)
            if is_empty:
                excluded_empty += 1
                continue
            if difficulty not in reservoirs:
                raise ValueError(f"Unsupported difficulty {difficulty!r} at record {total_records}")

            candidate_counts[difficulty] += 1
            seen = candidate_counts[difficulty]
            bucket = reservoirs[difficulty]
            quota = quotas[difficulty]
            entry = (total_records, offset)
            if len(bucket) < quota:
                bucket.append(entry)
            else:
                replacement = rngs[difficulty].randrange(seen)
                if replacement < quota:
                    bucket[replacement] = entry

            if progress_every > 0 and total_records % progress_every == 0:
                print(
                    "[stratified-subset] "
                    f"scanned={total_records} candidates={dict(candidate_counts)} empty={excluded_empty}",
                    flush=True,
                )

    shortages = {
        name: {"required": quotas[name], "available": candidate_counts[name]}
        for name in DIFFICULTIES
        if candidate_counts[name] < quotas[name]
    }
    if shortages:
        raise ValueError(f"Insufficient samples for requested difficulty quotas: {shortages}")

    selected = sorted(
        (record_number, offset, difficulty)
        for difficulty, bucket in reservoirs.items()
        for record_number, offset in bucket
    )
    if len(selected) != target_samples:
        raise RuntimeError(f"Selected {len(selected)} records, expected {target_samples}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    selected_counts: Counter[str] = Counter()
    try:
        with input_path.open("rb") as source, temp_output.open("wb") as destination:
            for _, offset, difficulty in selected:
                source.seek(offset)
                raw_line = source.readline()
                if not raw_line:
                    raise RuntimeError(f"Unable to reread selected offset {offset} from {input_path}")
                destination.write(raw_line)
                digest.update(raw_line)
                selected_counts[difficulty] += 1
        temp_output.replace(output_path)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    source_stat = input_path.stat()
    summary: dict[str, Any] = {
        "status": "complete",
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_records": total_records,
        "excluded_empty_records": excluded_empty,
        "candidate_counts": {name: candidate_counts[name] for name in DIFFICULTIES},
        "target_samples": target_samples,
        "difficulty_ratios": ratios,
        "selected_counts": {name: selected_counts[name] for name in DIFFICULTIES},
        "seed": seed,
        "difficulty_rule_version": DIFFICULTY_RULE_VERSION,
        "output_sha256": digest.hexdigest(),
    }
    temp_summary = summary_path.with_name(f".{summary_path.name}.tmp.{os.getpid()}")
    temp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_summary.replace(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.difficulty_ratios)
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    if args.reuse_if_valid and summary_matches(
        summary_path,
        args.input_jsonl.resolve(),
        args.output_jsonl.resolve(),
        args.target_samples,
        ratios,
        args.seed,
    ):
        print(f"[stratified-subset] reuse valid subset: {args.output_jsonl}", flush=True)
        return
    summary = build_subset(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        summary_path=summary_path,
        target_samples=args.target_samples,
        ratios=ratios,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
