from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def coord_high(meta: dict) -> int:
    if str(meta.get("coord_mode", "")).lower() == "norm1000" or "norm" in str(meta.get("coord_system", "")).lower():
        return int(meta.get("coord_range", 1000))
    return int(meta.get("patch_size", 256)) - 1


def coord_text(meta: dict) -> str:
    high = coord_high(meta)
    if high == 1000:
        return "Coordinates use a normalized 0-1000 grid over the original image patch."
    return f"Coordinates use original patch pixel coordinates in [0,{high}]."


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_incoming_traces(prompt: str) -> str:
    match = re.search(r"Incoming traces JSON:\n(.*?)\n\n", prompt, flags=re.S)
    if not match:
        return "[]"
    return match.group(1).strip() or "[]"


def base_prompt(traces: str, include_intersections: bool, meta: dict) -> str:
    parts = [
        "<image>",
        "Please construct the complete road map in the current BEV (Bird's Eye View) image patch.",
        coord_text(meta),
        "",
        "Incoming traces JSON:",
        traces,
    ]
    if include_intersections:
        parts.extend(["", "Incoming intersections JSON:", "[]"])
    parts.extend([
        "",
        "Each incoming trace has 1 to 3 points. If multiple points are present, they are ordered from the previous patch interior toward the current patch boundary.",
        "Incoming traces are continuity hints only; they may be incomplete or absent.",
    ])
    if include_intersections:
        parts.append("Each incoming intersection has 1 to 3 boundary points from neighboring patches.")
    return "\n".join(parts)


def clamp(value: int, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(value)))


def synthesize_intersection(lines: list[dict], index: int, meta: dict) -> dict:
    high = coord_high(meta)
    points = []
    for line in lines:
        if line.get("category", "centerline") == "centerline":
            points.extend(line.get("points") or [])
    if points:
        cx = round(sum(point[0] for point in points) / len(points))
        cy = round(sum(point[1] for point in points) / len(points))
    else:
        cx, cy = high // 2, high // 2
    radius = max(4, round(high * (0.07 + (index % 3) * 0.015)))
    polygon = [
        [clamp(cx - radius, high=high), clamp(cy - radius // 2, high=high)],
        [clamp(cx - radius // 2, high=high), clamp(cy - radius, high=high)],
        [clamp(cx + radius, high=high), clamp(cy - radius // 2, high=high)],
        [clamp(cx + radius // 2, high=high), clamp(cy + radius, high=high)],
        [clamp(cx - radius, high=high), clamp(cy - radius // 2, high=high)],
    ]
    is_cut = any(x in (0, high) or y in (0, high) for x, y in polygon)
    return {"category": "intersection", "is_cut": bool(is_cut), "points": polygon}


def parse_target(sample: dict) -> list[dict] | None:
    conversations = sample.get("conversations") or []
    if len(conversations) < 2:
        return None
    try:
        payload = json.loads(conversations[1]["value"])
    except Exception:
        return None
    lines = payload.get("lines")
    if not isinstance(lines, list):
        return None
    centerlines = [
        item for item in lines
        if isinstance(item, dict) and item.get("category", "centerline") == "centerline"
    ]
    return centerlines or None


def build_record(sample: dict, phase: str, task: str, index: int) -> dict | None:
    centerlines = parse_target(sample)
    if not centerlines:
        return None
    conversations = sample.get("conversations") or []
    source_prompt = conversations[0].get("value", "") if conversations else ""
    traces = "[]" if phase == "phase_a" else extract_incoming_traces(source_prompt)
    include_intersections = task == "lane_intersection"
    lines = list(centerlines)
    meta = dict(sample.get("meta") or {})
    if include_intersections:
        lines.append(synthesize_intersection(centerlines, index, meta))

    meta.update({
        "debug_dataset": True,
        "debug_phase": phase,
        "map_task": task,
        "intersection_source": "synthetic" if include_intersections else "none",
    })
    return {
        "id": f"{sample.get('id', index)}__{phase}_{task}",
        "image": sample["image"],
        "meta": meta,
        "conversations": [
            {"from": "human", "value": base_prompt(traces, include_intersections, meta)},
            {"from": "gpt", "value": json.dumps({"lines": lines}, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build_rows(source_rows: list[dict], phase: str, task: str, limit: int) -> list[dict]:
    rows = []
    # For Phase B smoke, prefer samples that actually contain incoming traces.
    candidates = source_rows
    if phase == "phase_b":
        candidates = sorted(
            source_rows,
            key=lambda row: extract_incoming_traces((row.get("conversations") or [{}])[0].get("value", "")) == "[]",
        )
    for sample in candidates:
        record = build_record(sample, phase, task, len(rows))
        if record is not None:
            rows.append(record)
        if len(rows) >= limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build A/B stage debug datasets from existing patch SFT data.")
    parser.add_argument("--source-jsonl", default="data/av2_patch_256_fullimage_cutflag_test_v2/sft.jsonl")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=4)
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.source_jsonl))
    output_root = Path(args.output_root)
    for phase in ("phase_a", "phase_b"):
        for task in ("lane", "lane_intersection"):
            rows = build_rows(source_rows, phase, task, args.limit)
            if len(rows) < args.limit:
                raise RuntimeError(f"Only built {len(rows)} rows for {phase}/{task}, expected {args.limit}")
            test_count = min(args.test_count, len(rows))
            train_rows = rows[:-test_count]
            test_rows = rows[-test_count:]
            out_dir = output_root / f"debug_{phase}_{task}20"
            write_jsonl(out_dir / "train.jsonl", train_rows)
            write_jsonl(out_dir / "test.jsonl", test_rows)
            manifest = {
                "phase": phase,
                "task": task,
                "source_jsonl": args.source_jsonl,
                "image_folder": "data/av2_patch_256_fullimage_cutflag_test_v2",
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "total_count": len(rows),
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {out_dir}: train={len(train_rows)} test={len(test_rows)}")


if __name__ == "__main__":
    main()
