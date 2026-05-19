#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"{path} is JSON but not a list")
            return payload
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def choose_eval_count(total: int, ratio: float, count: int | None) -> int:
    if total <= 1:
        return 0
    if count is not None and count >= 0:
        eval_count = count
    else:
        eval_count = int(round(total * ratio))
    eval_count = max(0, eval_count)
    # Keep at least one final-test record whenever possible.
    return min(eval_count, total - 1)


def split_rows(rows: list[dict[str, Any]], ratio: float, count: int | None, seed: int):
    eval_count = choose_eval_count(len(rows), ratio, count)
    if eval_count <= 0:
        return rows, []
    rng = random.Random(seed)
    eval_indices = set(rng.sample(range(len(rows)), eval_count))
    test_rows = [row for idx, row in enumerate(rows) if idx not in eval_indices]
    eval_rows = [row for idx, row in enumerate(rows) if idx in eval_indices]
    return test_rows, eval_rows


def split_one(test_json: Path, output_test: Path, output_eval: Path, ratio: float, count: int | None, seed: int):
    if test_json.resolve() == output_test.resolve():
        backup = output_test.with_name("test_full.jsonl")
        if not backup.exists():
            backup.write_text(test_json.read_text(encoding="utf-8"), encoding="utf-8")
        test_json = backup
    rows = load_json_or_jsonl(test_json)
    test_rows, eval_rows = split_rows(rows, ratio=ratio, count=count, seed=seed)
    write_jsonl(output_test, test_rows)
    write_jsonl(output_eval, eval_rows)
    manifest = {
        "source_test_json": str(test_json),
        "output_test_json": str(output_test),
        "output_eval_json": str(output_eval),
        "source_count": len(rows),
        "test_count": len(test_rows),
        "eval_count": len(eval_rows),
        "eval_ratio": ratio,
        "eval_count_requested": count,
        "seed": seed,
    }
    manifest_path = output_eval.with_name("eval_split_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def split_dataset_root(dataset_root: Path, phases: list[str], ratio: float, count: int | None, seed: int):
    manifests = []
    for phase in phases:
        phase_dir = dataset_root / phase
        test_json = phase_dir / "test_full.jsonl"
        if not test_json.exists():
            test_json = phase_dir / "test.jsonl"
        if not test_json.exists():
            continue
        output_test = phase_dir / "test.jsonl"
        output_eval = phase_dir / "eval.jsonl"
        if test_json == output_test:
            backup = phase_dir / "test_full.jsonl"
            if not backup.exists():
                backup.write_text(test_json.read_text(encoding="utf-8"), encoding="utf-8")
            test_json = backup
        manifests.append(split_one(test_json, output_test, output_eval, ratio, count, seed))

        meta_test = phase_dir / "meta_test.jsonl"
        if meta_test.exists():
            meta_source = phase_dir / "meta_test_full.jsonl"
            if not meta_source.exists():
                meta_source.write_text(meta_test.read_text(encoding="utf-8"), encoding="utf-8")
            split_one(
                meta_source,
                phase_dir / "meta_test.jsonl",
                phase_dir / "meta_eval.jsonl",
                ratio,
                count,
                seed,
            )
    return manifests


def main():
    parser = argparse.ArgumentParser(description="Split eval records out of an existing test JSON/JSONL.")
    parser.add_argument("--test-json", default="", help="Source test.jsonl. Use with --output-test/--output-eval.")
    parser.add_argument("--output-test", default="", help="Final test JSONL after eval records are removed.")
    parser.add_argument("--output-eval", default="", help="Eval JSONL split from source test.")
    parser.add_argument("--dataset-root", default="", help="Dataset root containing phase_a/phase_b directories.")
    parser.add_argument("--phases", nargs="*", default=["phase_a", "phase_b"])
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--eval-count", type=int, default=-1, help="Explicit eval count; -1 means use ratio.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    explicit_count = None if args.eval_count < 0 else args.eval_count
    if args.dataset_root:
        manifests = split_dataset_root(Path(args.dataset_root), args.phases, args.eval_ratio, explicit_count, args.seed)
        if not manifests:
            raise FileNotFoundError(f"No test jsonl found under {args.dataset_root} for phases {args.phases}")
        print(json.dumps({"splits": manifests}, ensure_ascii=False, indent=2))
        return

    if not args.test_json or not args.output_test or not args.output_eval:
        raise ValueError("Use either --dataset-root or all of --test-json/--output-test/--output-eval.")
    manifest = split_one(
        Path(args.test_json),
        Path(args.output_test),
        Path(args.output_eval),
        args.eval_ratio,
        explicit_count,
        args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
