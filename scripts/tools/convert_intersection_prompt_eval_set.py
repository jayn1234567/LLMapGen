#!/usr/bin/env python3
"""Convert a fixed lane+intersection evaluation set to oracle-intersection prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.derive_intersection_prompt_dataset import (
    PROMPT_MARKER,
    extract_prompt_intersections,
    transform_record,
)


DIFFICULTIES = ("easy", "medium", "hard", "very_hard")
JSONL_NAMES = (*DIFFICULTIES, "all_selected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-variant", default="local512_intersection_prompt_fixed_eval")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(record: dict) -> str:
    return str(record.get("id", record.get("sample_id", record.get("record_id", "")))).strip()


def transform_jsonl(source: Path, destination: Path, dataset_variant: str) -> dict:
    counts = Counter()
    ids: list[str] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            transformed, centerlines, intersections = transform_record(
                record,
                dataset_variant=dataset_variant,
            )
            sample_id = record_id(transformed)
            if not sample_id:
                raise ValueError(f"Missing sample id at {source}:{line_number}")
            prompt = transformed["conversations"][0]["value"]
            if PROMPT_MARKER not in prompt or extract_prompt_intersections(prompt) != intersections:
                raise ValueError(f"Intersection prompt round-trip failed for sample={sample_id}")
            if any(str(item.get("category", "")).lower() != "centerline" for item in centerlines):
                raise ValueError(f"Non-centerline assistant target remains for sample={sample_id}")
            writer.write(compact_json(transformed) + "\n")
            ids.append(sample_id)
            counts["records"] += 1
            counts["centerlines"] += len(centerlines)
            counts["prompt_intersections"] += len(intersections)
            if intersections:
                counts["records_with_intersections"] += 1
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample ids in {source}")
    return {"counts": dict(counts), "ids": ids, "sha256": sha256(destination)}


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if input_root == output_root:
        raise ValueError("input and output roots must differ")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Fixed evaluation root not found: {input_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    reports = {}
    for name in JSONL_NAMES:
        source = input_root / f"{name}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Required fixed-eval split not found: {source}")
        reports[name] = transform_jsonl(
            source,
            output_root / source.name,
            args.dataset_variant,
        )

    concatenated_ids = [sample_id for name in DIFFICULTIES for sample_id in reports[name]["ids"]]
    if concatenated_ids != reports["all_selected"]["ids"]:
        raise ValueError("all_selected.jsonl is not the ordered concatenation of the four difficulty splits")

    for name in ("manifest.jsonl", "summary.json", "fixed_eval_identity.json"):
        source = input_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)

    report = {
        "status": "passed",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_variant": args.dataset_variant,
        "task_mode": "centerline_conditioned_on_gt_intersections",
        "assistant_target": "centerline_only",
        "evaluation_policy": "centerline_geometry_only; oracle intersections are model inputs, not predictions",
        "splits": {
            name: {key: value for key, value in payload.items() if key != "ids"}
            for name, payload in reports.items()
        },
    }
    (output_root / "intersection_prompt_conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
