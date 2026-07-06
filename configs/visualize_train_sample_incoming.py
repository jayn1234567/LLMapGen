#!/usr/bin/env python3
"""Standalone visualizer for one Stage-B training JSONL sample.

This script intentionally does not import project modules. It is meant for
quick dataset sanity checks on another machine.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "centerline": (0, 220, 80),
    "intersection": (255, 190, 0),
    "trace": (0, 190, 255),
    "intersection_hint": (255, 80, 220),
    "border": (255, 255, 255),
    "text": (255, 255, 255),
    "background": (18, 18, 18),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def safe_name(text: str) -> str:
    text = str(text or "sample")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "sample"


def record_row_col(record: dict[str, Any]) -> tuple[int, int]:
    meta = record.get("meta") or {}
    row = meta.get("row", meta.get("patch_row", record.get("row", record.get("patch_row", 0))))
    col = meta.get("col", meta.get("patch_col", record.get("col", record.get("patch_col", 0))))
    return int(row), int(col)


def record_tile_id(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    return str(meta.get("tile_id") or meta.get("log_id") or record.get("tile_id") or "")


def select_record(records: list[dict[str, Any]], args) -> dict[str, Any]:
    if args.id:
        for record in records:
            if str(record.get("id", "")) == args.id:
                return record
        raise ValueError(f"sample id not found: {args.id}")
    if args.image_contains:
        for record in records:
            if args.image_contains in str(record.get("image", "")):
                return record
        raise ValueError(f"no sample image contains: {args.image_contains}")
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"--index out of range: {args.index}, total={len(records)}")
    return records[args.index]


def build_record_index(records: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    result = {}
    for record in records:
        row, col = record_row_col(record)
        result[(record_tile_id(record), row, col)] = record
    return result


def find_json_after_label(text: str, label: str, default):
    start = text.find(label)
    if start < 0:
        return default
    tail = text[start + len(label):].lstrip()
    try:
        payload, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return default
    return payload


def human_prompt(record: dict[str, Any]) -> str:
    for message in record.get("conversations") or []:
        if message.get("from") == "human":
            return str(message.get("value", ""))
    return ""


def assistant_text(record: dict[str, Any]) -> str:
    for message in reversed(record.get("conversations") or []):
        if message.get("from") in {"gpt", "assistant"}:
            return str(message.get("value", ""))
    return ""


def parse_payload(text: str):
    if not text:
        return {"lines": []}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"lines": []}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"lines": payload}
    return {"lines": []}


def normalize_lines(payload) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("lines", [])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def coord_cfg(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta") or {}
    patch_size = int(meta.get("patch_size") or meta.get("pixel_patch_size") or 256)
    width = int(meta.get("patch_width") or patch_size)
    height = int(meta.get("patch_height") or patch_size)
    return {
        "mode": str(meta.get("coord_mode") or "pixel").lower(),
        "range": int(meta.get("coord_range") or 1000),
        "width": width,
        "height": height,
    }


def finite_xy(point) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def point_to_pixel(point, cfg: dict[str, Any]) -> list[int] | None:
    xy = finite_xy(point)
    if xy is None:
        return None
    x, y = xy
    if cfg["mode"] in {"norm1000", "patch_norm1000", "normalized"}:
        x = x / cfg["range"] * (cfg["width"] - 1)
        y = y / cfg["range"] * (cfg["height"] - 1)
    return [int(round(x)), int(round(y))]


def points_to_pixel(points, cfg: dict[str, Any]) -> list[list[int]]:
    if not isinstance(points, list):
        return []
    converted = []
    for point in points:
        xy = point_to_pixel(point, cfg)
        if xy is not None:
            converted.append(xy)
    return converted


def line_points_to_pixel(line: dict[str, Any], cfg: dict[str, Any]) -> list[list[int]]:
    return points_to_pixel(line.get("points") or [], cfg)


def hints_to_pixel(hints, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    if not isinstance(hints, list):
        return result
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        result.append({
            "id": str(hint.get("id", "")),
            "side": str(hint.get("side", "")),
            "points": points_to_pixel(hint.get("points") or [], cfg),
        })
    return result


def resolve_image_path(record: dict[str, Any], image_root: Path) -> Path | None:
    raw = str(record.get("image") or record.get("image_path") or "")
    if not raw:
        return None
    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    candidates.extend([
        image_root / path,
        image_root / path.name,
    ])
    parts = list(path.parts)
    for marker in ("images", "img"):
        if marker in parts:
            idx = parts.index(marker)
            candidates.append(image_root / Path(*parts[idx:]))
            if idx + 1 < len(parts):
                candidates.append(image_root / Path(*parts[idx + 1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def load_patch_image(record: dict[str, Any] | None, image_root: Path, width: int, height: int) -> tuple[Image.Image, str]:
    if record is None:
        return Image.new("RGB", (width, height), (45, 45, 45)), "missing record"
    image_path = resolve_image_path(record, image_root)
    if image_path and image_path.exists():
        return Image.open(image_path).convert("RGB"), str(image_path)
    return Image.new("RGB", (width, height), (55, 35, 35)), f"missing image: {image_path}"


def get_font(size: int = 18):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[list[int]], offset: tuple[int, int], color, width: int, radius: int):
    shifted = [(x + offset[0], y + offset[1]) for x, y in points]
    if len(shifted) >= 2:
        draw.line(shifted, fill=color, width=width, joint="curve")
    for x, y in shifted:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0))


def draw_hint_points(draw: ImageDraw.ImageDraw, hint: dict[str, Any], offset: tuple[int, int], color, radius: int, font):
    points = hint.get("points") or []
    draw_polyline(draw, points, offset, color, width=2, radius=radius)
    if points:
        x, y = points[-1]
        label = hint.get("id") or hint.get("side") or "hint"
        draw.text((x + offset[0] + 6, y + offset[1] + 6), label, fill=color, font=font)


def patch_canvas(image: Image.Image, title: str, margin: int) -> tuple[Image.Image, tuple[int, int], ImageDraw.ImageDraw]:
    font = get_font(18)
    title_h = 34
    canvas = Image.new("RGB", (image.width + margin * 2, image.height + margin * 2 + title_h), COLORS["background"])
    offset = (margin, margin + title_h)
    canvas.paste(image, offset)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [offset[0], offset[1], offset[0] + image.width - 1, offset[1] + image.height - 1],
        outline=COLORS["border"],
        width=2,
    )
    draw.text((10, 8), title, fill=COLORS["text"], font=font)
    return canvas, offset, draw


def draw_current_panel(record, image_root: Path, traces_px, intersections_px, margin: int) -> Image.Image:
    cfg = coord_cfg(record)
    image, image_note = load_patch_image(record, image_root, cfg["width"], cfg["height"])
    row, col = record_row_col(record)
    canvas, offset, draw = patch_canvas(image, f"current r{row} c{col}", margin)
    font = get_font(15)
    lines = normalize_lines(parse_payload(assistant_text(record)))
    for line in lines:
        category = str(line.get("category", "centerline")).lower()
        color = COLORS["intersection"] if category == "intersection" else COLORS["centerline"]
        draw_polyline(draw, line_points_to_pixel(line, cfg), offset, color, width=3, radius=3)
    for hint in traces_px:
        draw_hint_points(draw, hint, offset, COLORS["trace"], radius=4, font=font)
    for hint in intersections_px:
        draw_hint_points(draw, hint, offset, COLORS["intersection_hint"], radius=5, font=font)
    if image_note.startswith("missing"):
        draw.text((10, canvas.height - 24), image_note, fill=(255, 120, 120), font=font)
    return canvas


def shift_hint_to_neighbor(hint: dict[str, Any], side: str, width: int, height: int) -> dict[str, Any]:
    shifted = []
    for x, y in hint.get("points") or []:
        if side == "left":
            shifted.append([x + width, y])
        elif side == "top":
            shifted.append([x, y + height])
    return {"id": hint.get("id", ""), "side": side, "points": shifted}


def draw_neighbor_panel(title: str, neighbor_record, image_root: Path, hints, width: int, height: int, margin: int, color) -> Image.Image:
    image, image_note = load_patch_image(neighbor_record, image_root, width, height)
    canvas, offset, draw = patch_canvas(image, title, margin)
    font = get_font(15)
    for hint in hints:
        draw_hint_points(draw, hint, offset, color, radius=5, font=font)
    if image_note.startswith("missing"):
        draw.text((10, canvas.height - 24), image_note, fill=(255, 120, 120), font=font)
    return canvas


def combine_panels(panels: list[Image.Image]) -> Image.Image:
    if not panels:
        return Image.new("RGB", (1, 1), COLORS["background"])
    gap = 12
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), COLORS["background"])
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def write_debug_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one training JSONL sample and its injected Stage-B neighbor hints.")
    parser.add_argument("--jsonl", required=True, help="Training JSONL path.")
    parser.add_argument("--image-root", required=True, help="Root used to resolve record['image'].")
    parser.add_argument("--output-dir", default="configs/train_sample_incoming_viz")
    parser.add_argument("--id", default="", help="Exact record id to visualize.")
    parser.add_argument("--index", type=int, default=0, help="Record index when --id is not set.")
    parser.add_argument("--image-contains", default="", help="Select first record whose image path contains this text.")
    parser.add_argument("--margin", type=int, default=90, help="Canvas margin so negative incoming points are visible.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"no records found in {jsonl_path}")
    record = select_record(records, args)
    record_index = build_record_index(records)

    cfg = coord_cfg(record)
    row, col = record_row_col(record)
    tile_id = record_tile_id(record)
    prompt = human_prompt(record)
    traces = find_json_after_label(prompt, "Incoming traces JSON:", [])
    intersections = find_json_after_label(prompt, "Incoming intersections JSON:", [])
    traces_px = hints_to_pixel(traces, cfg)
    intersections_px = hints_to_pixel(intersections, cfg)

    current_panel = draw_current_panel(record, image_root, traces_px, intersections_px, args.margin)

    left_record = record_index.get((tile_id, row, col - 1))
    top_record = record_index.get((tile_id, row - 1, col))
    left_hints = [shift_hint_to_neighbor(hint, "left", cfg["width"], cfg["height"]) for hint in traces_px + intersections_px if hint.get("side") == "left"]
    top_hints = [shift_hint_to_neighbor(hint, "top", cfg["width"], cfg["height"]) for hint in traces_px + intersections_px if hint.get("side") == "top"]

    panels = [current_panel]
    prefix = safe_name(str(record.get("id") or f"r{row}_c{col}"))
    current_path = output_dir / f"{prefix}_current.png"
    current_panel.save(current_path)

    left_path = None
    top_path = None
    if left_hints or left_record is not None:
        left_panel = draw_neighbor_panel(
            f"left neighbor r{row} c{col - 1} injected points only",
            left_record,
            image_root,
            left_hints,
            cfg["width"],
            cfg["height"],
            args.margin,
            COLORS["trace"],
        )
        left_path = output_dir / f"{prefix}_left_neighbor_injected.png"
        left_panel.save(left_path)
        panels.append(left_panel)
    if top_hints or top_record is not None:
        top_panel = draw_neighbor_panel(
            f"top neighbor r{row - 1} c{col} injected points only",
            top_record,
            image_root,
            top_hints,
            cfg["width"],
            cfg["height"],
            args.margin,
            COLORS["trace"],
        )
        top_path = output_dir / f"{prefix}_top_neighbor_injected.png"
        top_panel.save(top_path)
        panels.append(top_panel)

    overview_path = output_dir / f"{prefix}_overview.png"
    combine_panels(panels).save(overview_path)

    debug = {
        "record_id": record.get("id"),
        "tile_id": tile_id,
        "row": row,
        "col": col,
        "coord": cfg,
        "incoming_traces_raw": traces,
        "incoming_intersections_raw": intersections,
        "incoming_traces_pixel": traces_px,
        "incoming_intersections_pixel": intersections_px,
        "left_neighbor_found": left_record is not None,
        "top_neighbor_found": top_record is not None,
        "outputs": {
            "current": str(current_path),
            "left_neighbor": str(left_path) if left_path else "",
            "top_neighbor": str(top_path) if top_path else "",
            "overview": str(overview_path),
        },
    }
    debug_path = output_dir / f"{prefix}_debug.json"
    write_debug_json(debug_path, debug)
    print(json.dumps(debug["outputs"] | {"debug": str(debug_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
