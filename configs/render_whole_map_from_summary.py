#!/usr/bin/env python3
"""Render whole-map GT/Pred visualizations directly from old inference summary.

This does not generate single-patch visualizations. It reads summary.json/jsonl
from a previous inference output and stitches the original patch images plus
GT/pred lines into whole-map PNGs.

Example:
  python configs/render_whole_map_from_summary.py \
    --summary /path/to/infer_output/summary.json \
    --image-folder /path/to/dataset \
    --output-dir /path/to/infer_output/whole_map_viz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.map_visualization import render_whole_map_visualizations


def load_json_array_or_jsonl(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    if content.startswith("{") and content.endswith("}"):
        payload = json.loads(content)
        if isinstance(payload, dict) and isinstance(payload.get("patch_results"), list):
            return payload["patch_results"]
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return payload["results"]
        return [payload]

    if content.startswith("[") and content.endswith("]"):
        payload = json.loads(content)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return payload

    rows = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary.json or summary.jsonl from inference.")
    parser.add_argument("--image-folder", required=True, help="Dataset/image root used to resolve patch image paths.")
    parser.add_argument("--output-dir", default="", help="Output dir. Defaults to summary parent / whole_map_viz.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    image_folder = Path(args.image_folder)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "whole_map_viz"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not image_folder.exists():
        raise FileNotFoundError(image_folder)

    records = load_json_array_or_jsonl(summary_path)
    if not records:
        raise RuntimeError(f"No records found in {summary_path}")

    rendered = render_whole_map_visualizations(records, image_folder, output_dir)
    print(json.dumps({
        "summary": str(summary_path),
        "image_folder": str(image_folder),
        "output_dir": str(output_dir),
        "num_records": len(records),
        "rendered": rendered,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
