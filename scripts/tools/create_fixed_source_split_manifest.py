#!/usr/bin/env python3
"""Select representative raw BEV maps for a reusable fixed eval/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2_staged import (
    STAGE_MARKER,
    build_sample_owners,
    discover_stage_roots,
    iter_jsonl,
)
from data_process.fixed_source_splits import FORMAT_VERSION, SPLIT_POLICY, assignment_manifest_id


DIFFICULTIES = ("very_easy", "easy", "medium", "hard", "very_hard")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-count", type=int, default=14)
    parser.add_argument("--test-count", type=int, default=7)
    parser.add_argument("--min-base-patches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--duplicate-policy", choices=["last", "first", "error"], default="last")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def stable_hash(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def collect_candidates(stage_roots: list[Path], owners: dict[str, int]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    seen_patches: set[str] = set()
    for stage_root in stage_roots:
        marker = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        source_index = int(marker["source_index"])
        source_uri = str(marker.get("source_uri") or marker.get("input_root") or stage_root)
        for split in ("train", "eval", "test"):
            for _, row in iter_jsonl(stage_root / "records" / f"{split}.index.jsonl"):
                raw_id = str(row["raw_sample_id"])
                if owners.get(raw_id) != source_index:
                    continue
                if str(row.get("grid_kind") or "base") != "base":
                    continue
                patch_id = str(row["id"])
                patch_key = f"{source_index}\0{patch_id}"
                if patch_key in seen_patches:
                    continue
                seen_patches.add(patch_key)
                item = stats.setdefault(raw_id, {
                    "raw_sample_id": raw_id,
                    "source_index": source_index,
                    "source_uri": source_uri,
                    "base_patch_count": 0,
                    "intersection_patch_count": 0,
                    "difficulty_counts": Counter(),
                })
                item["base_patch_count"] += 1
                item["intersection_patch_count"] += int(bool(row.get("has_intersection")))
                item["difficulty_counts"][str(row.get("difficulty") or row.get("stratum") or "medium")] += 1
    candidates = []
    for item in stats.values():
        count = int(item["base_patch_count"])
        item["intersection_ratio"] = item["intersection_patch_count"] / max(1, count)
        item["difficulty_counts"] = dict(item["difficulty_counts"])
        candidates.append(item)
    return candidates


def target_profile(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    patch_count = sum(item["base_patch_count"] for item in candidates)
    intersections = sum(item["intersection_patch_count"] for item in candidates)
    difficulty = Counter()
    for item in candidates:
        difficulty.update(item["difficulty_counts"])
    counts = sorted(item["base_patch_count"] for item in candidates)
    median = counts[len(counts) // 2] if counts else 0
    return {
        "base_patch_count": patch_count,
        "median_base_patches_per_raw_sample": median,
        "intersection_ratio": intersections / max(1, patch_count),
        "difficulty_ratios": {
            name: difficulty[name] / max(1, patch_count) for name in DIFFICULTIES
        },
    }


def add_representativeness_scores(candidates: list[dict[str, Any]], profile: dict[str, Any]) -> None:
    median = max(1, int(profile["median_base_patches_per_raw_sample"]))
    for item in candidates:
        count = max(1, int(item["base_patch_count"]))
        difficulty = item["difficulty_counts"]
        difficulty_distance = sum(
            abs(difficulty.get(name, 0) / count - profile["difficulty_ratios"][name])
            for name in DIFFICULTIES
        )
        item["representativeness_score"] = (
            abs(math.log(count / median))
            + 2.0 * abs(item["intersection_ratio"] - profile["intersection_ratio"])
            + difficulty_distance
        )


def select_source_balanced(
    candidates: list[dict[str, Any]],
    count: int,
    seed: int,
    excluded: set[str],
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("split counts must be non-negative")
    available = [item for item in candidates if item["raw_sample_id"] not in excluded]
    if count > len(available):
        raise ValueError(f"requested {count} fixed maps but only {len(available)} candidates remain")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in available:
        groups[(item["source_index"], item["source_uri"])].append(item)
    for source, values in groups.items():
        values.sort(key=lambda item: (
            item["representativeness_score"],
            stable_hash(seed, item["raw_sample_id"]),
        ))
    source_order = sorted(groups, key=lambda source: stable_hash(seed, f"{source[0]}\0{source[1]}"))
    selected = []
    while len(selected) < count:
        progressed = False
        for source in source_order:
            if not groups[source]:
                continue
            selected.append(groups[source].pop(0))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("source-balanced selection exhausted candidates unexpectedly")
    return selected


def aggregate_selection(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(item["source_index"]) for item in items)
    patch_count = sum(item["base_patch_count"] for item in items)
    intersections = sum(item["intersection_patch_count"] for item in items)
    difficulty = Counter()
    for item in items:
        difficulty.update(item["difficulty_counts"])
    return {
        "raw_sample_count": len(items),
        "base_patch_count": patch_count,
        "intersection_ratio": intersections / max(1, patch_count),
        "source_counts": dict(source_counts),
        "difficulty_ratios": {
            name: difficulty[name] / max(1, patch_count) for name in DIFFICULTIES
        },
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.eval_count <= 0:
        raise ValueError("--eval-count must be positive")
    if args.test_count < 0:
        raise ValueError("--test-count must be non-negative")
    staging_root = Path(args.staging_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_path}")
    stage_roots = discover_stage_roots(staging_root)
    owners, collisions = build_sample_owners(stage_roots, args.duplicate_policy)
    all_candidates = collect_candidates(stage_roots, owners)
    candidates = [item for item in all_candidates if item["base_patch_count"] >= args.min_base_patches]
    if len(candidates) < args.eval_count + args.test_count:
        raise ValueError(
            f"only {len(candidates)} eligible large maps for "
            f"eval={args.eval_count} and test={args.test_count}"
        )
    profile = target_profile(candidates)
    add_representativeness_scores(candidates, profile)
    eval_items = select_source_balanced(candidates, args.eval_count, args.seed, set())
    eval_ids = {item["raw_sample_id"] for item in eval_items}
    test_items = select_source_balanced(candidates, args.test_count, args.seed + 1, eval_ids)
    test_ids = {item["raw_sample_id"] for item in test_items}
    manifest_id = assignment_manifest_id(eval_ids, test_ids)
    selected_by_id = {item["raw_sample_id"]: item for item in eval_items + test_items}
    payload = {
        "format_version": FORMAT_VERSION,
        "manifest_id": manifest_id,
        "split_unit": "raw_sample_id",
        "split_policy": SPLIT_POLICY,
        "unknown_source_policy": "train",
        "raw_sample_ids_by_split": {
            "eval": sorted(eval_ids),
            "test": sorted(test_ids),
        },
        "selected_sources": {
            raw_id: {
                "source_index": selected_by_id[raw_id]["source_index"],
                "source_uri": selected_by_id[raw_id]["source_uri"],
                "base_patch_count": selected_by_id[raw_id]["base_patch_count"],
                "intersection_ratio": selected_by_id[raw_id]["intersection_ratio"],
                "representativeness_score": selected_by_id[raw_id]["representativeness_score"],
            }
            for raw_id in sorted(selected_by_id)
        },
        "selection": {
            "seed": args.seed,
            "eval_count": args.eval_count,
            "test_count": args.test_count,
            "min_base_patches": args.min_base_patches,
            "source_balanced": True,
            "candidate_raw_sample_count": len(candidates),
            "target_profile": profile,
            "eval_profile": aggregate_selection(eval_items),
            "test_profile": aggregate_selection(test_items),
            "duplicate_policy": args.duplicate_policy,
            "duplicate_raw_sample_events": collisions,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_path = output_path.with_name(output_path.stem + "_candidates.jsonl")
    with candidate_path.open("w", encoding="utf-8") as handle:
        split_by_id = {**{item: "eval" for item in eval_ids}, **{item: "test" for item in test_ids}}
        for item in sorted(candidates, key=lambda row: (row["source_index"], row["raw_sample_id"])):
            row = dict(item)
            row["selected_split"] = split_by_id.get(item["raw_sample_id"], "train")
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "status": "passed",
        "manifest": str(output_path),
        "candidate_report": str(candidate_path),
        "manifest_id": manifest_id,
        "eval": aggregate_selection(eval_items),
        "test": aggregate_selection(test_items),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
