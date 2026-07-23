#!/usr/bin/env python3
"""Reclassify staged local512 records and visualize each difficulty bucket.

This tool works entirely from Dataset V2 staging artifacts. It does not read the
raw RC resource archives and it never modifies staging files.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import DIFFICULTY_ARGS
from data_process.difficulty_profiles import (
    DIFFICULTY_PROFILE_VERSION,
    FIVE_TIER_BUCKETS,
    LOCAL512_PROFILE,
    classify_metrics,
    resolution_aware_score,
    resolve_difficulty_profile,
)
from data_process.build_dataset_v2_staged import (
    STAGE_MARKER,
    build_sample_owners,
    discover_stage_roots,
)
from scripts.tools.tag_hard_map_samples import (
    draw_overlay,
    make_contact_sheet,
    safe_name,
    sample_metrics,
)


BUCKETS = FIVE_TIER_BUCKETS
RULE_VERSION = DIFFICULTY_PROFILE_VERSION
DEFAULT_CAPS = LOCAL512_PROFILE.caps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--variant", default="local512")
    parser.add_argument("--split", default="train", choices=["train", "eval", "test"])
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--profile", default="local512_profile_a")
    parser.add_argument("--coord-range", type=float, default=1000.0)
    parser.add_argument("--candidate-jsonl", default="", help="Optional JSONL whose ids limit the audit pool.")
    parser.add_argument("--duplicate-policy", default="last", choices=["first", "last", "error"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--visualize-per-difficulty", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--no-sample-report", action="store_true")
    return parser.parse_args()


def read_candidate_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id") or row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"candidate row has no id at {path}:{line_number}")
            ids.add(sample_id)
    return ids


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def iter_stage_records(
    stage_roots: list[Path],
    sample_owner: dict[str, int],
    variant: str,
    split: str,
    candidate_ids: set[str] | None,
) -> Iterator[tuple[Path, dict[str, Any], dict[str, Any]]]:
    for stage_root in stage_roots:
        marker = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        if variant not in marker.get("variants", []):
            continue
        source_index = int(marker["source_index"])
        index_path = stage_root / "records" / f"{split}.index.jsonl"
        sft_path = stage_root / "records" / variant / f"{split}.jsonl"
        if not index_path.is_file() or not sft_path.is_file():
            raise FileNotFoundError(f"stage record pair missing: {index_path} / {sft_path}")
        index_iter = iter_jsonl(index_path)
        sft_iter = iter_jsonl(sft_path)
        row_number = 0
        while True:
            index_row = next(index_iter, None)
            record = next(sft_iter, None)
            if index_row is None and record is None:
                break
            row_number += 1
            if index_row is None or record is None:
                raise ValueError(f"index/SFT row count mismatch under {stage_root} at row {row_number}")
            patch_id = str(index_row.get("id", ""))
            record_id = str(record.get("id", record.get("sample_id", "")))
            if patch_id != record_id:
                raise ValueError(
                    f"index/SFT id mismatch under {stage_root} row {row_number}: {patch_id!r} != {record_id!r}"
                )
            if sample_owner.get(str(index_row.get("raw_sample_id"))) != source_index:
                continue
            if candidate_ids is not None and patch_id not in candidate_ids:
                continue
            yield stage_root, index_row, record


def resolve_image_path(stage_root: Path, variant: str, record: dict[str, Any]) -> Path | None:
    raw = record.get("images", record.get("image", ""))
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return stage_root / "variants" / variant / path


def compact_metrics(metrics: dict[str, Any], stage_root: Path, image_path: Path | None) -> dict[str, Any]:
    keys = (
        "id", "difficulty", "difficulty_score", "centerline_count", "intersection_count",
        "point_count", "fork_node_count", "cycle_count", "crossing_count",
        "lane_change_like_count", "short_fragment_count", "total_turn_degrees",
        "non_common_lane_count", "max_turn_degrees", "tags", "difficulty_score_components",
    )
    result = {key: metrics.get(key) for key in keys}
    result.update({
        "difficulty_rule_version": RULE_VERSION,
        "stage_root": str(stage_root),
        "image_path": str(image_path) if image_path else "",
    })
    return result


def update_reservoir(
    reservoir: list[dict[str, Any]],
    item: dict[str, Any],
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = item


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * fraction) - 1))
    return float(sorted_values[position])


def threshold_bucket_counts(scores: list[float], thresholds: list[float]) -> dict[str, int]:
    ordered = sorted(scores)
    i0 = bisect.bisect_right(ordered, thresholds[0])
    i1 = bisect.bisect_right(ordered, thresholds[1])
    i2 = bisect.bisect_right(ordered, thresholds[2])
    i3 = bisect.bisect_right(ordered, thresholds[3])
    return {
        "very_easy": i0,
        "easy": i1 - i0,
        "medium": i2 - i1,
        "hard": i3 - i2,
        "very_hard": len(ordered) - i3,
    }


def render_visualizations(
    samples: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    coord_range: float,
) -> dict[str, int]:
    draw_args = argparse.Namespace(coord_mode="norm1000", coord_range=coord_range)
    rendered_counts = {}
    for difficulty in BUCKETS:
        paths = []
        difficulty_dir = output_dir / "viz_by_difficulty" / difficulty
        for rank, item in enumerate(samples[difficulty]):
            record = item["record"]
            metrics = item["metrics"]
            image_path = item["image_path"]
            output_path = difficulty_dir / f"{rank:03d}_{safe_name(metrics.get('id'))}.png"
            draw_overlay(record, metrics, image_path, output_path, draw_args)
            paths.append(output_path)
        for page_index in range(0, len(paths), 25):
            make_contact_sheet(
                paths[page_index:page_index + 25],
                output_dir / f"contact_sheet_{difficulty}_{page_index // 25 + 1:02d}.png",
                cols=5,
            )
        rendered_counts[difficulty] = len(paths)
    return rendered_counts


def main() -> None:
    args = parse_args()
    staging_root = Path(args.staging_root)
    output_dir = Path(args.output_dir) if args.output_dir else staging_root / "difficulty_512_threshold_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_roots = discover_stage_roots(staging_root)
    if not stage_roots:
        raise FileNotFoundError(f"no {STAGE_MARKER} found under {staging_root}")
    sample_owner, duplicate_events = build_sample_owners(stage_roots, args.duplicate_policy)
    candidate_path = Path(args.candidate_jsonl) if str(args.candidate_jsonl).strip() else None
    candidate_ids = read_candidate_ids(candidate_path)
    profile = resolve_difficulty_profile(args.profile, args.patch_size)

    metric_args = copy.copy(DIFFICULTY_ARGS)
    metric_args.coord_mode = "norm1000"
    metric_args.coord_range = args.coord_range
    metric_args.easy_max_centerlines = profile.score_free_centerlines
    metric_args.easy_max_points = profile.score_free_points

    counts = Counter()
    tags = Counter()
    scores = []
    seen_by_bucket = Counter()
    rngs = {name: random.Random(args.seed + index * 1009) for index, name in enumerate(BUCKETS)}
    reservoirs = {name: [] for name in BUCKETS}
    sample_report_path = output_dir / "sample_metrics.jsonl"
    report_handle = None if args.no_sample_report else sample_report_path.open("w", encoding="utf-8")
    scanned = 0
    missing_images = 0
    try:
        for stage_root, _index_row, record in iter_stage_records(
            stage_roots,
            sample_owner,
            args.variant,
            args.split,
            candidate_ids,
        ):
            metrics = sample_metrics(record, (args.patch_size, args.patch_size), metric_args)
            resolution_aware_score(metrics, profile)
            difficulty = classify_metrics(metrics, profile)
            if difficulty == "empty":
                counts["empty"] += 1
                continue
            metrics["difficulty"] = difficulty
            metrics["difficulty_rule_version"] = RULE_VERSION
            metrics["oversample_weight"] = {
                "very_easy": 0.5,
                "easy": 1.0,
                "medium": 1.5,
                "hard": 2.5,
                "very_hard": 4.0,
            }[difficulty]
            image_path = resolve_image_path(stage_root, args.variant, record)
            if image_path is None or not image_path.is_file():
                missing_images += 1
            counts[difficulty] += 1
            seen_by_bucket[difficulty] += 1
            tags.update(metrics.get("tags") or [])
            scores.append(float(metrics["difficulty_score"]))
            compact = compact_metrics(metrics, stage_root, image_path)
            if report_handle is not None:
                report_handle.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
            update_reservoir(
                reservoirs[difficulty],
                {"record": record, "metrics": metrics, "image_path": image_path},
                seen_by_bucket[difficulty],
                args.visualize_per_difficulty,
                rngs[difficulty],
            )
            scanned += 1
            if args.progress_every and scanned % args.progress_every == 0:
                print(f"[difficulty-512-audit] scanned={scanned} counts={dict(counts)}", flush=True)
            if args.max_samples and scanned >= args.max_samples:
                break
    finally:
        if report_handle is not None:
            report_handle.close()

    quantile_thresholds = [percentile(sorted(scores), value) for value in (0.05, 0.25, 0.55, 0.85)]
    summary = {
        "status": "ok",
        "difficulty_rule_version": RULE_VERSION,
        "staging_root": str(staging_root),
        "variant": args.variant,
        "split": args.split,
        "patch_size": args.patch_size,
        "candidate_jsonl": str(candidate_path) if candidate_path else "",
        "stage_count": len(stage_roots),
        "duplicate_raw_sample_events": duplicate_events,
        "scanned_nonempty": scanned,
        "difficulty_counts": {name: counts.get(name, 0) for name in BUCKETS},
        "empty_count": counts.get("empty", 0),
        "difficulty_ratios": {
            name: round(counts.get(name, 0) / max(1, scanned), 6) for name in BUCKETS
        },
        "missing_images": missing_images,
        "difficulty_profile": profile.to_dict(),
        "target_5_20_30_30_15_score_quantiles": {
            "very_easy_max_score": quantile_thresholds[0],
            "easy_max_score": quantile_thresholds[1],
            "medium_max_score": quantile_thresholds[2],
            "hard_max_score": quantile_thresholds[3],
            "score_only_bucket_counts_with_ties": threshold_bucket_counts(scores, quantile_thresholds),
        },
        "tag_counts": dict(tags.most_common()),
        "sample_report": "" if args.no_sample_report else str(sample_report_path),
    }
    rendered_counts = render_visualizations(reservoirs, output_dir, args.coord_range)
    summary["visualization_counts"] = rendered_counts
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[difficulty-512-audit] summary: {summary_path}", flush=True)
    print(f"[difficulty-512-audit] visualizations: {output_dir / 'viz_by_difficulty'}", flush=True)


if __name__ == "__main__":
    main()
