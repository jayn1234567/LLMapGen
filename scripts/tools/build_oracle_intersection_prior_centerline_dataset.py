#!/usr/bin/env python3
"""Build a centerline-only trainroot with oracle intersection priors in the prompt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = """You are an expert road-centerline reconstruction assistant for black-background BEV road-structure images.

VISIBLE SEMANTICS:
The visible road-structure classes are lane_boundary, lane_divider, and background.
lane_boundary is rendered as bright green road-edge or lane-boundary lines.
lane_divider is rendered as cyan / light-blue lane separator markings, often dashed.
background is black.
The image does not show centerlines directly.

TASK DEFINITION:
Predict road centerlines for the current 512x512 RC patch.
The user message provides oracle intersection polygons in patch-local coordinates.
Use those intersection priors as topology anchors when deciding where centerlines meet, split, or terminate.

RULES:
1. Do not trace lane_boundary or lane_divider themselves.
2. Keep different lanes, branches, and intersecting paths as separate continuous centerline polylines.
3. If a centerline reaches the patch border, terminate it at the visible border.
4. Predict only geometry visible in the current patch.
5. Use the oracle intersection polygons only as input priors; do not output them.
6. This dataset uses the native 512x512 patch-local coordinate system.

OUTPUT CONSTRAINTS:
1. Return ONLY valid JSON.
2. Do NOT wrap the JSON in markdown fences.
3. Do NOT output explanations or extra text.
4. All x and y coordinates must be integers between 0 and 511 inclusive.
5. Strictly use this JSON structure:
{"lines":[]}
or
{"lines":[{"points":[[x1,y1],[x2,y2]]}]}"""


USER_TEMPLATE = """This is a native 512x512 black-background BEV road-structure image.

Oracle intersection polygons for this patch are provided below. Coordinates are patch-local integers in [0, 511].
{prior_json}

Use these oracle intersection polygons as topology priors.
Predict ONLY the road centerlines implied by the visible lane_boundary and lane_divider structure.
Do not output intersection polygons.
Return only the raw JSON object with a "lines" array."""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def extract_message(row: Dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)
    return ""


def parse_assistant_json(row: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_message(row, "assistant")
    if not text:
        return {"lines": [], "intersections": []}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"lines": [], "intersections": []}
    if not isinstance(value, dict):
        return {"lines": [], "intersections": []}
    lines = value.get("lines")
    intersections = value.get("intersections")
    return {
        "lines": lines if isinstance(lines, list) else [],
        "intersections": intersections if isinstance(intersections, list) else [],
    }


def convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
    target = parse_assistant_json(row)
    prior_json = json.dumps(
        {"intersections": target["intersections"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "id": row.get("id", ""),
        "images": row.get("images", []),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(prior_json=prior_json)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"lines": target["lines"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
    }


def copy_optional(src: Path, dst: Path, name: str) -> None:
    src_path = src / name
    dst_path = dst / name
    if not src_path.exists() and not src_path.is_symlink():
        return
    if dst_path.exists() or dst_path.is_symlink():
        if dst_path.is_dir() and not dst_path.is_symlink():
            shutil.rmtree(dst_path)
        else:
            dst_path.unlink()
    if src_path.is_symlink():
        os.symlink(os.readlink(src_path), dst_path)
    elif src_path.is_dir():
        os.symlink(str(src_path), dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    with_lines = 0
    with_intersections = 0
    line_count = 0
    intersection_count = 0
    for row in rows:
        target = parse_assistant_json(row)
        lines = target["lines"]
        intersections = target["intersections"]
        if lines:
            with_lines += 1
            line_count += len(lines)
        if intersections:
            with_intersections += 1
            intersection_count += len(intersections)
    return {
        "rows": len(rows),
        "rows_with_lines": with_lines,
        "rows_with_intersections": with_intersections,
        "line_count": line_count,
        "intersection_count": intersection_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-trainroot", type=Path, required=True)
    parser.add_argument("--output-trainroot", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src_trainroot.resolve()
    out = args.output_trainroot.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "src_trainroot": str(src),
        "output_trainroot": str(out),
        "splits": {},
    }
    for split in ("train", "val"):
        src_jsonl = src / f"{split}.jsonl"
        if not src_jsonl.is_file():
            continue
        rows = load_jsonl(src_jsonl)
        converted = [convert_row(row) for row in rows]
        write_jsonl(out / f"{split}.jsonl", converted)
        summary["splits"][split] = summarize(rows)
        meta_name = f"meta_{split}.jsonl"
        if (src / meta_name).is_file():
            shutil.copy2(src / meta_name, out / meta_name)

    for name in ("dataset_info.json", "split_summary.json", "patches512_joint"):
        copy_optional(src, out, name)

    (out / "oracle_prior_dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
