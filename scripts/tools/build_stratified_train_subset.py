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
    parser.add_argument(
        "--shortage-policy",
        choices=("error", "redistribute"),
        default="error",
        help="Fail on a bucket shortage, or fill the deficit from selected donor buckets.",
    )
    parser.add_argument(
        "--shortage-fill-buckets",
        default="medium,hard",
        help="Comma-separated donor buckets used by the redistribute shortage policy.",
    )
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


def parse_fill_buckets(spec: str) -> tuple[str, ...]:
    buckets: list[str] = []
    for raw_name in spec.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in DIFFICULTIES:
            raise ValueError(f"Unknown shortage fill bucket: {name!r}")
        if name not in buckets:
            buckets.append(name)
    if not buckets:
        raise ValueError("shortage-fill-buckets must contain at least one difficulty bucket.")
    return tuple(buckets)


def resolve_selected_quotas(
    requested: dict[str, int],
    candidate_counts: dict[str, int],
    target_samples: int,
    ratios: dict[str, float],
    shortage_policy: str,
    shortage_fill_buckets: tuple[str, ...],
) -> dict[str, int]:
    shortages = {
        name: {"required": requested[name], "available": int(candidate_counts.get(name, 0))}
        for name in DIFFICULTIES
        if int(candidate_counts.get(name, 0)) < requested[name]
    }
    if shortages and shortage_policy == "error":
        raise ValueError(f"Insufficient samples for requested difficulty quotas: {shortages}")

    selected = {
        name: min(requested[name], int(candidate_counts.get(name, 0)))
        for name in DIFFICULTIES
    }
    deficit = target_samples - sum(selected.values())
    if deficit <= 0:
        return selected

    additions = {name: 0 for name in DIFFICULTIES}
    stages = [
        shortage_fill_buckets,
        tuple(name for name in DIFFICULTIES if name not in shortage_fill_buckets),
    ]
    for stage in stages:
        while deficit > 0:
            eligible = [
                name
                for name in stage
                if selected[name] < int(candidate_counts.get(name, 0))
            ]
            if not eligible:
                break
            positive_weights = {name: ratios[name] for name in eligible if ratios[name] > 0}
            if positive_weights:
                chosen = min(
                    positive_weights,
                    key=lambda name: ((additions[name] + 1) / positive_weights[name], name),
                )
            else:
                chosen = min(eligible, key=lambda name: (additions[name], name))
            selected[chosen] += 1
            additions[chosen] += 1
            deficit -= 1
        if deficit == 0:
            break

    if deficit:
        raise ValueError(
            f"Insufficient non-empty samples after shortage redistribution: missing {deficit} of {target_samples}."
        )
    return selected


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
    shortage_policy: str,
    shortage_fill_buckets: tuple[str, ...],
    seed: int,
) -> bool:
    if not summary_path.is_file() or not output_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stat = input_path.stat()
        candidate_counts = {
            name: int((summary.get("candidate_counts") or {}).get(name, 0))
            for name in DIFFICULTIES
        }
        expected_counts = resolve_selected_quotas(
            allocate_quotas(target_samples, ratios),
            candidate_counts,
            target_samples,
            ratios,
            shortage_policy,
            shortage_fill_buckets,
        )
        return bool(
            summary.get("status") == "complete"
            and summary.get("source_size") == stat.st_size
            and summary.get("source_mtime_ns") == stat.st_mtime_ns
            and summary.get("target_samples") == target_samples
            and summary.get("difficulty_ratios") == ratios
            and summary.get("shortage_policy") == shortage_policy
            and summary.get("shortage_fill_buckets") == list(shortage_fill_buckets)
            and summary.get("seed") == seed
            and summary.get("selected_counts") == expected_counts
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
    shortage_policy: str,
    shortage_fill_buckets: tuple[str, ...],
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

    requested_quotas = allocate_quotas(target_samples, ratios)
    reservoir_capacities = {
        name: target_samples if shortage_policy == "redistribute" else requested_quotas[name]
        for name in DIFFICULTIES
    }
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
            quota = reservoir_capacities[difficulty]
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

    selected_quotas = resolve_selected_quotas(
        requested_quotas,
        candidate_counts,
        target_samples,
        ratios,
        shortage_policy,
        shortage_fill_buckets,
    )
    if selected_quotas != requested_quotas:
        print(
            "[stratified-subset] redistributed quotas "
            f"requested={requested_quotas} selected={selected_quotas} "
            f"fill_buckets={list(shortage_fill_buckets)}",
            flush=True,
        )

    selected_buckets: dict[str, list[tuple[int, int]]] = {}
    for index, name in enumerate(DIFFICULTIES):
        bucket = list(reservoirs[name])
        if shortage_policy == "redistribute":
            random.Random(seed + (index + 1) * 9_000_011).shuffle(bucket)
        selected_buckets[name] = bucket[: selected_quotas[name]]

    selected = sorted(
        (record_number, offset, difficulty)
        for difficulty, bucket in selected_buckets.items()
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
        "requested_counts": requested_quotas,
        "selected_counts": {name: selected_counts[name] for name in DIFFICULTIES},
        "shortage_policy": shortage_policy,
        "shortage_fill_buckets": list(shortage_fill_buckets),
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
    shortage_fill_buckets = parse_fill_buckets(args.shortage_fill_buckets)
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    if args.reuse_if_valid and summary_matches(
        summary_path,
        args.input_jsonl.resolve(),
        args.output_jsonl.resolve(),
        args.target_samples,
        ratios,
        args.shortage_policy,
        shortage_fill_buckets,
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
        shortage_policy=args.shortage_policy,
        shortage_fill_buckets=shortage_fill_buckets,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
