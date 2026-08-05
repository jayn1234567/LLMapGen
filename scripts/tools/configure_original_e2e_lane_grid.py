#!/usr/bin/env python3
"""Configure a run-local original RC E2E engine for a lane-result grid size."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, required=True, choices=(256, 512))
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lane_parser_block(source: str) -> tuple[int, int]:
    match = re.search(r"^class LaneNNParser\b", source, flags=re.MULTILINE)
    if match is None:
        raise ValueError("LaneNNParser class not found")
    next_class = re.search(r"^class \w", source[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_class.start() if next_class else len(source)
    return match.start(), end


def configure_engine(engine_root: Path, patch_size: int) -> dict[str, object]:
    candidates = []
    for path in sorted(engine_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "class LaneNNParser" in source:
            candidates.append((path, source))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one original LaneNNParser below {engine_root}, found {len(candidates)}: "
            f"{[str(path) for path, _ in candidates]}"
        )

    path, source = candidates[0]
    start, end = lane_parser_block(source)
    block = source[start:end]
    crop_count = len(re.findall(r"self\.CROP_SIZE\s*=\s*\d+", block))
    step_count = len(re.findall(r"self\.STEP\s*=\s*\d+", block))
    if crop_count != 1 or step_count != 1:
        raise RuntimeError(
            f"Unexpected LaneNNParser grid assignments in {path}: CROP_SIZE={crop_count}, STEP={step_count}"
        )
    updated_block = re.sub(r"self\.CROP_SIZE\s*=\s*\d+", f"self.CROP_SIZE = {patch_size}", block)
    updated_block = re.sub(r"self\.STEP\s*=\s*\d+", f"self.STEP = {patch_size}", updated_block)
    updated = source[:start] + updated_block + source[end:]
    changed = updated != source
    if changed:
        path.write_text(updated, encoding="utf-8")

    return {
        "engine_root": str(engine_root),
        "parser_path": str(path),
        "patch_size": patch_size,
        "formatter_coordinate_scale": patch_size / 1000.0,
        "changed": changed,
        "source_sha256_before": digest(source),
        "source_sha256_after": digest(updated),
        "policy": (
            "Run-local grid adaptation only: LaneNNParser CROP_SIZE and STEP are set to the same "
            "local patch size as the formatter output. The shared original-engine cache must not be modified."
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.engine_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"engine root not found: {root}")
    report = configure_engine(root, args.patch_size)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
