#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image

from build_sft_dataset import build_record


INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8


def region_code(x, y, xmin, ymin, xmax, ymax):
    code = INSIDE
    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP
    return code


def clip_segment(p0, p1, xmin, ymin, xmax, ymax):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    c0 = region_code(x0, y0, xmin, ymin, xmax, ymax)
    c1 = region_code(x1, y1, xmin, ymin, xmax, ymax)
    p0_cut = False
    p1_cut = False

    while True:
        if not (c0 | c1):
            return {
                "p0": [x0, y0],
                "p1": [x1, y1],
                "p0_cut": p0_cut,
                "p1_cut": p1_cut,
            }
        if c0 & c1:
            return None

        out = c0 or c1
        if out & TOP:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif out & BOTTOM:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif out & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin

        if out == c0:
            x0, y0 = x, y
            p0_cut = True
            c0 = region_code(x0, y0, xmin, ymin, xmax, ymax)
        else:
            x1, y1 = x, y
            p1_cut = True
            c1 = region_code(x1, y1, xmin, ymin, xmax, ymax)


def round_local_point(point, x0, y0, patch_size):
    x = int(round(point[0] - x0))
    y = int(round(point[1] - y0))
    return [max(0, min(patch_size - 1, x)), max(0, min(patch_size - 1, y))]


def local_point_with_cut(point, is_cut, x0, y0, patch_size):
    return {
        "point": round_local_point(point, x0, y0, patch_size),
        "cut": bool(is_cut),
    }


def is_same_point(a, b):
    return int(round(a[0])) == int(round(b[0])) and int(round(a[1])) == int(round(b[1]))


def dedupe_points(points):
    out = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    return out


def dedupe_flagged_points(items):
    out = []
    for item in items:
        if out and item["point"] == out[-1]["point"]:
            out[-1]["cut"] = out[-1]["cut"] or item["cut"]
            continue
        out.append(item)
    return out


def public_line(line):
    return {key: value for key, value in line.items() if not key.startswith("_")}


def clip_polyline_to_patch(line, x0, y0, patch_size, border_tol, source_line_index=None):
    points = line.get("points") or []
    if len(points) < 2:
        return []

    xmin, ymin = x0, y0
    xmax, ymax = x0 + patch_size - 1, y0 + patch_size - 1
    category = str(line.get("category", "centerline")).lower()
    clipped_lines = []
    current = []

    for p0, p1 in zip(points[:-1], points[1:]):
        clipped = clip_segment(p0, p1, xmin, ymin, xmax, ymax)
        if clipped is None:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = []
            continue

        fp0 = local_point_with_cut(clipped["p0"], clipped["p0_cut"], x0, y0, patch_size)
        fp1 = local_point_with_cut(clipped["p1"], clipped["p1_cut"], x0, y0, patch_size)
        if fp0["point"] == fp1["point"]:
            continue

        if current and current[-1]["point"] == fp0["point"]:
            current[-1]["cut"] = current[-1]["cut"] or fp0["cut"]
            current.append(fp1)
        else:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = [fp0, fp1]

        if clipped["p1_cut"]:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = []

    if len(current) >= 2:
        clipped_lines.append(current)

    results = []
    for flagged_pts in clipped_lines:
        flagged_pts = dedupe_flagged_points(flagged_pts)
        if len(flagged_pts) < 2:
            continue
        pts = [item["point"] for item in flagged_pts]
        if category == "intersection":
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            if len(pts) >= 4:
                results.append({"category": "intersection", "points": pts})
            continue
        results.append({
            "category": "centerline",
            "start_type": "cut" if flagged_pts[0]["cut"] else "inside",
            "end_type": "cut" if flagged_pts[-1]["cut"] else "inside",
            "points": pts,
            "_source_line_index": source_line_index,
            "_source_points": points,
            "_patch_x0": x0,
            "_patch_y0": y0,
        })
    return results


def squared_distance(a, b):
    return (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2


def shift_neighbor_point_to_current(point, side, patch_size):
    x, y = int(round(point[0])), int(round(point[1]))
    if side == "left":
        return [x - patch_size, y]
    if side == "top":
        return [x, y - patch_size]
    return [x, y]


def source_trace_points(line, side, boundary_local, boundary_at_end, patch_size, max_points):
    source_points = line.get("_source_points") or []
    if len(source_points) < 2:
        return []

    patch_x0 = line.get("_patch_x0", 0)
    patch_y0 = line.get("_patch_y0", 0)
    boundary_global = [boundary_local[0] + patch_x0, boundary_local[1] + patch_y0]
    nearest_idx = min(range(len(source_points)), key=lambda idx: squared_distance(source_points[idx], boundary_global))

    if boundary_at_end:
        start_idx = max(0, nearest_idx - (max_points - 1))
        selected = source_points[start_idx:nearest_idx + 1]
        if not selected or squared_distance(selected[-1], boundary_global) > 1.0:
            selected = selected[-(max_points - 1):] + [boundary_global]
    else:
        end_idx = min(len(source_points), nearest_idx + max_points)
        selected = source_points[nearest_idx:end_idx]
        if not selected or squared_distance(selected[0], boundary_global) > 1.0:
            selected = [boundary_global] + selected[1:max_points]
        selected = list(reversed(selected))

    local_points = [[point[0] - patch_x0, point[1] - patch_y0] for point in selected[-max_points:]]
    return [shift_neighbor_point_to_current(point, side, patch_size) for point in local_points]


def make_trace_from_line(line, side, patch_size, max_points, boundary_tol):
    if line.get("category") != "centerline":
        return None
    points = line.get("points") or []
    if len(points) < 2:
        return None

    if side == "left":
        boundary = lambda p: p[0] >= patch_size - 1 - boundary_tol
    elif side == "top":
        boundary = lambda p: p[1] >= patch_size - 1 - boundary_tol
    else:
        return None

    if line.get("end_type") == "cut" and boundary(points[-1]):
        trace_points = source_trace_points(line, side, points[-1], True, patch_size, max_points)
        if not trace_points:
            trace_points = [shift_neighbor_point_to_current(point, side, patch_size) for point in points[-max_points:]]
    elif line.get("start_type") == "cut" and boundary(points[0]):
        trace_points = source_trace_points(line, side, points[0], False, patch_size, max_points)
        if not trace_points:
            trace_points = [shift_neighbor_point_to_current(point, side, patch_size) for point in reversed(points[:max_points])]
    else:
        return None

    trace_points = dedupe_points(trace_points[-max_points:])
    if not trace_points:
        return None
    return {"side": side, "points": trace_points}


def build_incoming_traces(patch_lines_by_rc, row, col, patch_size, max_traces, trace_points, boundary_tol):
    traces = []
    candidates = [
        ("left", patch_lines_by_rc.get((row, col - 1), [])),
        ("top", patch_lines_by_rc.get((row - 1, col), [])),
    ]
    for side, lines in candidates:
        side_count = 0
        for line in lines:
            trace = make_trace_from_line(line, side, patch_size, trace_points, boundary_tol)
            if trace is None:
                continue
            trace["id"] = f"{side[0].upper()}{side_count}"
            traces.append(trace)
            side_count += 1
            if side_count >= max_traces:
                break
    return traces


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_manifest(manifest_path, limit):
    with manifest_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def should_keep_patch(lines, keep_empty):
    return keep_empty or bool(lines)


def process_record(record, input_root, output_root, patch_size, stride, border_tol,
                   keep_empty, with_gt_incoming, max_traces_per_side, trace_points,
                   max_patches=None):
    image_path = input_root / record["rc_input_path"]
    line_path = input_root / record["centerline_json_path"]
    image = Image.open(image_path).convert("RGB")
    payload = load_json(line_path)
    lines = payload.get("lines", [])
    width, height = image.size
    tile_id = record["tile_id"]
    city = record.get("city", "unknown")

    patch_rows = []
    patch_lines_by_rc = {}
    for y0 in range(0, height - patch_size + 1, stride):
        for x0 in range(0, width - patch_size + 1, stride):
            row = y0 // stride
            col = x0 // stride
            local_lines = []
            for line_idx, line in enumerate(lines):
                category = str(line.get("category", "centerline")).lower()
                if category not in {"centerline", "intersection"}:
                    continue
                local_lines.extend(clip_polyline_to_patch(line, x0, y0, patch_size, border_tol, line_idx))
            if should_keep_patch(local_lines, keep_empty):
                patch_lines_by_rc[(row, col)] = local_lines

    for patch_idx, ((row, col), local_lines) in enumerate(sorted(patch_lines_by_rc.items())):
        if max_patches is not None and patch_idx >= max_patches:
            break
        x0 = col * stride
        y0 = row * stride
        patch_name = f"{tile_id}_r{row:02d}_c{col:02d}.png"
        rel_image = Path("images") / city / patch_name
        out_image = output_root / rel_image
        out_image.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x0, y0, x0 + patch_size, y0 + patch_size)).save(out_image)

        incoming = []
        if with_gt_incoming:
            incoming = build_incoming_traces(
                patch_lines_by_rc, row, col, patch_size,
                max_traces_per_side, trace_points, border_tol,
            )

        patch_id = f"{tile_id}_r{row:02d}_c{col:02d}"
        meta = {
            "tile_id": tile_id,
            "log_id": record.get("log_id"),
            "city": city,
            "patch_row": row,
            "patch_col": col,
            "row": row,
            "col": col,
            "x0": x0,
            "y0": y0,
            "patch_size": patch_size,
            "stride": stride,
            "source_image_size": [width, height],
            "coord_system": f"patch_local_{patch_size}",
            "task_mode": "state_update_centerline_intersection",
        }
        patch_rows.append({
            "id": patch_id,
            "image": str(rel_image),
            "tile_id": tile_id,
            "city": city,
            "patch_row": row,
            "patch_col": col,
            "base_patch_box_full4096": [x0, y0, x0 + patch_size, y0 + patch_size],
            "incoming_traces": incoming,
            "target_lines": [public_line(line) for line in local_lines],
            "meta": meta,
        })
    return patch_rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split AV2 per-log 4096x4096 maps into patch-local 256 SFT records.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--border-tol", type=float, default=1.0)
    parser.add_argument("--limit-logs", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--with-gt-incoming", action="store_true")
    parser.add_argument("--max-traces-per-side", type=int, default=8)
    parser.add_argument(
        "--trace-points",
        type=int,
        default=3,
        help="Maximum adjacent-line points per incoming trace. Default 3; one-point boundary anchors are kept.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = input_root / args.manifest
    meta_rows = []
    sft_rows = []
    num_logs = 0

    for record in iter_manifest(manifest_path, args.limit_logs):
        if args.max_patches is not None and len(meta_rows) >= args.max_patches:
            break
        num_logs += 1
        remaining = None
        if args.max_patches is not None:
            remaining = args.max_patches - len(meta_rows)
        rows = process_record(
            record=record,
            input_root=input_root,
            output_root=output_root,
            patch_size=args.patch_size,
            stride=args.stride,
            border_tol=args.border_tol,
            keep_empty=args.keep_empty,
            with_gt_incoming=args.with_gt_incoming,
            max_traces_per_side=args.max_traces_per_side,
            trace_points=args.trace_points,
            max_patches=remaining,
        )
        meta_rows.extend(rows)
        for row in rows:
            sft_rows.append(build_record(
                record_id=row["id"],
                image=row["image"],
                patch_size=args.patch_size,
                incoming_traces=row["incoming_traces"],
                lines=row["target_lines"],
                meta=row["meta"],
            ))

    write_jsonl(output_root / "meta.jsonl", meta_rows)
    write_jsonl(output_root / "sft.jsonl", sft_rows)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_logs": num_logs,
        "num_patches": len(meta_rows),
        "max_patches": args.max_patches,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "with_gt_incoming": args.with_gt_incoming,
        "keep_empty": args.keep_empty,
        "meta_jsonl": str(output_root / "meta.jsonl"),
        "sft_jsonl": str(output_root / "sft.jsonl"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
