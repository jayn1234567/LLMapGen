#!/usr/bin/env python3
"""Verify that Dataset V3 variants use identical eval/test raw source images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_split_sources(dataset_root: Path) -> dict[str, set[str]]:
    manifest_path = dataset_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_ids = manifest.get("raw_sample_ids_by_split")
    if not isinstance(split_ids, dict):
        raise ValueError(f"manifest has no raw_sample_ids_by_split: {manifest_path}")
    result = {}
    for split in ("eval", "test"):
        values = split_ids.get(split)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"manifest has invalid {split} raw sample ids: {manifest_path}")
        result[split] = set(values)
    return result


def load_materialized_sources(dataset_root: Path) -> dict[str, set[str]]:
    result = {}
    for split in ("eval", "test"):
        jsonl_path = dataset_root / "phase_a" / f"{split}.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"dataset split not found: {jsonl_path}")
        source_ids = set()
        with jsonl_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                meta = record.get("meta") or {}
                raw_sample_id = str(meta.get("raw_sample_id") or meta.get("tile_id") or "").strip()
                if not raw_sample_id:
                    raise ValueError(f"missing meta.raw_sample_id at {jsonl_path}:{line_number}")
                source_ids.add(raw_sample_id)
        result[split] = source_ids
    return result


def digest(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_dataset_roots(dataset_roots: list[Path]) -> dict:
    if len(dataset_roots) < 2:
        raise ValueError("at least two --dataset-root values are required")
    manifest_sources = {str(root): load_split_sources(root) for root in dataset_roots}
    actual_sources = {str(root): load_materialized_sources(root) for root in dataset_roots}
    reference_root = str(dataset_roots[0])
    reference_manifest = manifest_sources[reference_root]
    reference_actual = actual_sources[reference_root]
    comparisons = {}
    passed = True
    for root, split_sources in manifest_sources.items():
        split_report = {}
        for split in ("eval", "test"):
            actual_split_sources = actual_sources[root][split]
            manifest_missing = reference_manifest[split] - split_sources[split]
            manifest_extra = split_sources[split] - reference_manifest[split]
            actual_missing = reference_actual[split] - actual_split_sources
            actual_extra = actual_split_sources - reference_actual[split]
            manifest_matches_actual = split_sources[split] == actual_split_sources
            exact = not any((
                manifest_missing,
                manifest_extra,
                actual_missing,
                actual_extra,
            )) and manifest_matches_actual
            passed = passed and exact
            split_report[split] = {
                "exact_match": exact,
                "manifest_raw_sample_count": len(split_sources[split]),
                "materialized_raw_sample_count": len(actual_split_sources),
                "materialized_sha256": digest(actual_split_sources),
                "manifest_matches_materialized": manifest_matches_actual,
                "manifest_missing_count": len(manifest_missing),
                "manifest_extra_count": len(manifest_extra),
                "materialized_missing_count": len(actual_missing),
                "materialized_extra_count": len(actual_extra),
                "missing_examples": sorted(manifest_missing | actual_missing)[:20],
                "extra_examples": sorted(manifest_extra | actual_extra)[:20],
            }
        comparisons[root] = split_report
    return {
        "status": "passed" if passed else "failed",
        "reference_dataset_root": reference_root,
        "dataset_count": len(dataset_roots),
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    roots = [Path(value).expanduser().resolve() for value in args.dataset_root]
    report = verify_dataset_roots(roots)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "passed":
        raise ValueError(f"eval/test raw source consistency check failed: {output_path}")
    print(f"[dataset-v3-split-check] passed: {output_path}", flush=True)


if __name__ == "__main__":
    main()
