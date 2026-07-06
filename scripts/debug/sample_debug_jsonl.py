#!/usr/bin/env python3
"""Build a small deterministic debug split from an existing phase dataset.

The script only samples JSONL records. It does not copy images because current
records usually store image paths relative to the original dataset root.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PHASE_ALIASES = {
    "phase_a": ("phase_a", "stage_a", "a"),
    "phase_b": ("phase_b", "stage_b", "b"),
}


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        for key in ("records", "results", "patch_results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported JSON structure in {path}")
    return [item for item in payload if isinstance(item, dict)]


def phase_dir(dataset_root: Path, phase: str) -> Path:
    aliases = PHASE_ALIASES.get(phase, (phase,))
    for alias in aliases:
        candidate = dataset_root / alias
        if candidate.is_dir():
            return candidate
    return dataset_root / phase


def split_source(dataset_root: Path, phase_root: Path, split: str) -> Path | None:
    candidates = [
        phase_root / f"{split}.jsonl",
        phase_root / f"{split}.json",
        dataset_root / f"{split}.jsonl",
        dataset_root / f"{split}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def sample_records(records: list[dict[str, Any]], limit: int, seed: int, split: str) -> list[dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    rng = random.Random(f"{seed}:{split}")
    indices = list(range(len(records)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [records[index] for index in keep]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/cache/data/data_line_samples_33w")
    parser.add_argument("--phase", choices=("phase_a", "phase_b"), default="phase_a")
    parser.add_argument("--output-root", default="checkpoints/debug_data")
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=4)
    parser.add_argument("--test-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    phase_root = phase_dir(dataset_root, args.phase)
    output_phase_root = output_root / args.phase
    limits = {"train": args.train_limit, "eval": args.eval_limit, "test": args.test_limit}

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not phase_root.exists():
        raise FileNotFoundError(f"phase directory not found: {phase_root}")

    sources: dict[str, Path | None] = {
        split: split_source(dataset_root, phase_root, split) for split in ("train", "eval", "test")
    }
    if sources["train"] is None:
        raise FileNotFoundError(f"train split not found under {phase_root} or {dataset_root}")
    if sources["eval"] is None:
        sources["eval"] = sources["test"]
    if sources["test"] is None:
        sources["test"] = sources["eval"]

    manifest: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "phase": args.phase,
        "output_root": str(output_root),
        "splits": {},
    }

    for split in ("train", "eval", "test"):
        source = sources[split]
        if source is None:
            raise FileNotFoundError(f"{split} split not found")
        records = load_records(source)
        sampled = sample_records(records, limits[split], args.seed, split)
        target = output_phase_root / f"{split}.jsonl"
        write_jsonl(target, sampled)
        manifest["splits"][split] = {
            "source": str(source),
            "source_count": len(records),
            "target": str(target),
            "target_count": len(sampled),
            "limit": limits[split],
        }

    manifest_path = output_phase_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
