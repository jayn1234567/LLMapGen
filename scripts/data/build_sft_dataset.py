#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_boundary_point(point, patch_size: int, border_tol: float):
    x, y = point
    hi = patch_size - 1
    return (
        x <= border_tol
        or y <= border_tol
        or x >= hi - border_tol
        or y >= hi - border_tol
    )


def infer_endpoint_type(points, patch_size: int, border_tol: float):
    if not points:
        return "inside", "inside"
    start_type = "cut" if is_boundary_point(points[0], patch_size, border_tol) else "inside"
    end_type = "cut" if is_boundary_point(points[-1], patch_size, border_tol) else "inside"
    return start_type, end_type


def clamp_int_point(point):
    return [int(round(point[0])), int(round(point[1]))]


def normalize_line(line, patch_size: int, border_tol: float):
    category = str(line.get("category", "centerline")).strip().lower()
    points = [clamp_int_point(pt) for pt in line.get("points", []) if isinstance(pt, list) and len(pt) == 2]
    if category == "centerline":
        start_type = line.get("start_type")
        end_type = line.get("end_type")
        if start_type not in {"cut", "inside"} or end_type not in {"cut", "inside"}:
            start_type, end_type = infer_endpoint_type(points, patch_size, border_tol)
        return {
            "category": "centerline",
            "start_type": start_type,
            "end_type": end_type,
            "points": points,
        }
    if category == "intersection":
        return {
            "category": "intersection",
            "points": points,
        }
    return None


def make_user_prompt(patch_size: int, incoming_traces):
    traces_json = json.dumps(incoming_traces, ensure_ascii=False, separators=(",", ":"))
    schema = (
        "{\"lines\":["
        "{\"category\":\"centerline\",\"start_type\":\"cut|inside\",\"end_type\":\"cut|inside\",\"points\":[[x,y],[x,y]]},"
        "{\"category\":\"intersection\",\"points\":[[x,y],[x,y]]}"
        "]}"
    )
    return (
        f"<image>\nThis is a {patch_size}x{patch_size} BEV road patch.\n"
        "Predict the road geometry inside this patch only.\n\n"
        f"Incoming traces JSON:\n{traces_json}\n\n"
        "Each incoming trace is ordered from the previous patch interior toward the current patch boundary.\n"
        "Incoming traces are continuity hints only; they may be incomplete or absent.\n"
        "Return only valid JSON in this schema:\n"
        f"{schema}\n"
        f"All output points must be integers inside [0,{patch_size - 1}]."
    )


def build_record(record_id, image, patch_size: int, incoming_traces, lines, meta=None):
    lines_payload = [line for line in lines if line is not None]
    meta_payload = dict(meta or {})
    meta_payload.setdefault("scan_order", "row_major_top_to_bottom_left_to_right")
    meta_payload.setdefault("available_neighbors", ["left", "top"])
    meta_payload.setdefault("train_shuffle_allowed", True)
    meta_payload.setdefault("trace_source_train", "gt_left_top_neighbors")
    meta_payload.setdefault("trace_source_infer", "predicted_left_top_neighbors")
    return {
        "id": record_id,
        "image": image,
        "meta": meta_payload,
        "conversations": [
            {
                "from": "human",
                "value": make_user_prompt(patch_size, incoming_traces),
            },
            {
                "from": "gpt",
                "value": json.dumps({"lines": lines_payload}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def convert_legacy_rows(rows, patch_size: int, border_tol: float):
    converted = []
    for row in rows:
        if "image" not in row or "conversations" not in row or len(row["conversations"]) < 2:
            continue
        try:
            raw_lines = json.loads(row["conversations"][1]["value"])
        except Exception:
            continue
        lines = [normalize_line(line, patch_size, border_tol) for line in raw_lines]
        meta = dict(row.get("meta") or {})
        converted.append(
            build_record(
                record_id=row.get("id", Path(row["image"]).stem),
                image=row["image"],
                patch_size=patch_size,
                incoming_traces=[],
                lines=lines,
                meta=meta,
            )
        )
    return converted


def iter_meta_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        rows = []
        for sample in payload["samples"]:
            meta_row = sample.get("meta_from_meta_train_jsonl")
            if meta_row:
                rows.append(meta_row)
        return rows
    if isinstance(payload, dict):
        return [payload]
    return []


def convert_state_update_rows(rows, patch_size: int, border_tol: float):
    converted = []
    for row in iter_meta_rows(rows):
        image = row.get("image")
        target_lines = row.get("target_lines")
        if not image or not isinstance(target_lines, list):
            continue
        incoming_traces = row.get("incoming_traces") or []
        lines = [normalize_line(line, patch_size, border_tol) for line in target_lines]
        meta = {
            "tile_id": row.get("tile_id"),
            "city": row.get("city"),
            "row": row.get("patch_row"),
            "col": row.get("patch_col"),
            "patch_size": patch_size,
            "coord_system": row.get("coord_system", f"patch_local_{patch_size}"),
            "task_mode": row.get("target_mode", "state_update_centerline_intersection"),
        }
        if "base_patch_box_full4096" in row:
            box = row["base_patch_box_full4096"]
            if isinstance(box, list) and len(box) == 4:
                meta["x0"] = int(box[0])
                meta["y0"] = int(box[1])
        converted.append(
            build_record(
                record_id=row.get("id", Path(image).stem),
                image=image,
                patch_size=patch_size,
                incoming_traces=incoming_traces,
                lines=lines,
                meta=meta,
            )
        )
    return converted


def main():
    parser = argparse.ArgumentParser(description="Build SFT datasets for centerline/intersection training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    legacy = subparsers.add_parser("legacy-centerline", help="Convert legacy centerline-only records.")
    legacy.add_argument("--input", required=True)
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--patch-size", type=int, default=256)
    legacy.add_argument("--border-tol", type=float, default=1.0)

    state = subparsers.add_parser("state-update-meta", help="Convert state-update metadata rows to SFT records.")
    state.add_argument("--input", required=True)
    state.add_argument("--output", required=True)
    state.add_argument("--patch-size", type=int, default=256)
    state.add_argument("--border-tol", type=float, default=1.0)

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_json_or_jsonl(input_path)

    if args.command == "legacy-centerline":
        converted = convert_legacy_rows(rows, args.patch_size, args.border_tol)
    else:
        converted = convert_state_update_rows(rows, args.patch_size, args.border_tol)

    dump_jsonl(output_path, converted)
    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "num_records": len(converted),
        "command": args.command,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
