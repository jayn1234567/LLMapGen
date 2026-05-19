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


def incoming_traces(prompt: str) -> str:
    match = re.search(r"Incoming traces JSON:\n(.*?)\n\n", prompt, flags=re.S)
    if not match:
        return "[]"
    return match.group(1).strip() or "[]"


def lane_prompt(traces: str, meta: dict) -> str:
    return (
        "<image>\n"
        "Please construct the complete road map in the current BEV (Bird's Eye View) image patch.\n"
        f"{coord_text(meta)}\n"
        "Predict the road centerlines inside this patch only.\n\n"
        f"Incoming traces JSON:\n{traces}\n\n"
        "Each incoming trace is ordered from the previous patch interior toward the current patch boundary.\n"
        "Incoming traces are continuity hints only; they may be incomplete or absent.\n"
        "Use this JSON schema for centerlines:\n"
        "{\"lines\":[{\"category\":\"centerline\",\"start_type\":\"cut|inside\",\"end_type\":\"cut|inside\",\"points\":[[x,y],[x,y]]}]}"
    )


def lane_intersection_prompt(traces: str, meta: dict) -> str:
    return (
        "<image>\n"
        "Please construct the complete road map in the current BEV (Bird's Eye View) image patch.\n"
        f"{coord_text(meta)}\n"
        "Predict road centerlines and intersection polygons inside this patch only.\n\n"
        f"Incoming traces JSON:\n{traces}\n\n"
        "Each incoming trace is ordered from the previous patch interior toward the current patch boundary.\n"
        "Incoming traces are continuity hints only; they may be incomplete or absent.\n"
        "Use this JSON schema for centerlines and intersections:\n"
        "{\"lines\":[{\"category\":\"centerline\",\"start_type\":\"cut|inside\",\"end_type\":\"cut|inside\",\"points\":[[x,y],[x,y]]},{\"category\":\"intersection\",\"is_cut\":false,\"points\":[[x,y],[x,y]]}]}"
    )


def clamp(value: int, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(value)))


def synthesize_intersection(lines: list[dict], index: int, meta: dict) -> dict:
    high = coord_high(meta)
    points = []
    for line in lines:
        points.extend(line.get("points") or [])
    if not points:
        cx, cy = high // 2, high // 2
    else:
        cx = round(sum(pt[0] for pt in points) / len(points))
        cy = round(sum(pt[1] for pt in points) / len(points))
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


def build_record(sample: dict, task: str, index: int) -> dict | None:
    conversations = sample.get("conversations") or []
    if len(conversations) < 2:
        return None
    try:
        gt_payload = json.loads(conversations[1]["value"])
    except Exception:
        return None
    centerlines = [
        item for item in gt_payload.get("lines", [])
        if isinstance(item, dict) and item.get("category", "centerline") == "centerline"
    ]
    if not centerlines:
        return None

    traces = incoming_traces(conversations[0].get("value", ""))
    meta = dict(sample.get("meta") or {})
    if task == "lane":
        prompt = lane_prompt(traces, meta)
        lines = centerlines
    elif task == "lane_intersection":
        prompt = lane_intersection_prompt(traces, meta)
        lines = [*centerlines, synthesize_intersection(centerlines, index, meta)]
    else:
        raise ValueError(f"Unsupported task: {task}")

    meta.update({
        "debug_dataset": True,
        "map_task": task,
        "intersection_source": "synthetic" if task == "lane_intersection" else "none",
    })
    return {
        "id": f"{sample.get('id', index)}__grpo_debug_{task}",
        "image": sample["image"],
        "meta": meta,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps({"lines": lines}, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build_task(source_rows: list[dict], task: str, limit: int) -> list[dict]:
    rows = []
    for sample in source_rows:
        record = build_record(sample, task, len(rows))
        if record is not None:
            rows.append(record)
        if len(rows) >= limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build small GRPO debug datasets from existing patch SFT data.")
    parser.add_argument("--source-jsonl", default="data/av2_patch_256_fullimage_cutflag_test_v2/sft.jsonl")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=4)
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.source_jsonl))
    output_root = Path(args.output_root)
    for task, dirname in (
        ("lane", "grpo_debug_lane20"),
        ("lane_intersection", "grpo_debug_lane_intersection20"),
    ):
        rows = build_task(source_rows, task, args.limit)
        if len(rows) < args.limit:
            raise RuntimeError(f"Only built {len(rows)} rows for {task}, expected {args.limit}")
        test_count = min(args.test_count, len(rows))
        train_rows = rows[:-test_count]
        test_rows = rows[-test_count:]
        out_dir = output_root / dirname
        write_jsonl(out_dir / "train.jsonl", train_rows)
        write_jsonl(out_dir / "test.jsonl", test_rows)
        manifest = {
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
