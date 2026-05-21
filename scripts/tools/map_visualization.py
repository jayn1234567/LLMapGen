#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_json_maybe(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {}


def normalize_lines(payload):
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        return payload["lines"]
    if isinstance(payload, list):
        return payload
    return []


def offset_lines(lines, dx, dy):
    shifted = []
    for line in lines:
        out = dict(line)
        out["points"] = [[int(round(x + dx)), int(round(y + dy))] for x, y in line.get("points", [])]
        shifted.append(out)
    return shifted


def draw_map_lines(image: Image.Image, payload, centerline_color: tuple, intersection_color: tuple, width: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for item in normalize_lines(payload):
        points = item.get("points") or []
        xy_points = [(int(pt[0]), int(pt[1])) for pt in points if isinstance(pt, list) and len(pt) >= 2]
        if not xy_points:
            continue
        category = str(item.get("category", "centerline")).lower()
        color = intersection_color if category == "intersection" else centerline_color
        for idx in range(len(xy_points) - 1):
            draw.line([xy_points[idx], xy_points[idx + 1]], fill=color, width=width)
        for x, y in xy_points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    return image


def add_title(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 40), "black")
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill="white", font=font)
    return canvas


def resolve_image_path(raw_path: str, image_folder: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    candidate = image_folder / image_path
    if candidate.exists():
        return candidate
    fallback = image_folder / image_path.name
    if fallback.exists():
        return fallback
    return image_path


def record_tile_id(record: dict) -> str:
    meta = record.get("meta") or {}
    return str(record.get("tile_id") or meta.get("tile_id") or "tile")


def record_patch_shape(record: dict) -> tuple[int, int]:
    patch_size = int(record.get("patch_size", (record.get("meta") or {}).get("patch_size", 256)))
    return (
        int(record.get("patch_width", (record.get("meta") or {}).get("patch_width", patch_size))),
        int(record.get("patch_height", (record.get("meta") or {}).get("patch_height", patch_size))),
    )


def record_origin(record: dict) -> tuple[int, int]:
    meta = record.get("meta") or {}
    patch_width, patch_height = record_patch_shape(record)
    row = int(record.get("row", meta.get("row", meta.get("patch_row", 0))))
    col = int(record.get("col", meta.get("col", meta.get("patch_col", 0))))
    x0 = int(record.get("x0", meta.get("x0", col * patch_width)))
    y0 = int(record.get("y0", meta.get("y0", row * patch_height)))
    return x0, y0


def local_payload_lines(record: dict, pixel_keys: tuple[str, ...], raw_keys: tuple[str, ...] = ()) -> list:
    for key in pixel_keys:
        text = record.get(key)
        if text:
            return normalize_lines(load_json_maybe(text))
    for key in raw_keys:
        text = record.get(key)
        if text:
            return normalize_lines(load_json_maybe(text))
    return []


def prediction_lines_for_record(record: dict, origin_x: int, origin_y: int) -> list:
    if record.get("lines_global"):
        return offset_lines(record.get("lines_global") or [], -origin_x, -origin_y)
    x0, y0 = record_origin(record)
    local_lines = local_payload_lines(record, ("prediction_json_pixel", "response_pixel", "prediction_pixel"), ("prediction_json", "response", "prediction"))
    return offset_lines(local_lines, x0 - origin_x, y0 - origin_y)


def gt_lines_for_record(record: dict, origin_x: int, origin_y: int) -> list:
    x0, y0 = record_origin(record)
    local_lines = local_payload_lines(record, ("ground_truth_pixel", "labels_pixel"), ("ground_truth", "labels"))
    return offset_lines(local_lines, x0 - origin_x, y0 - origin_y)


def render_whole_map_visualizations(patch_results, image_folder: Path, output_dir: Path):
    """Render one stitched BEV canvas per tile from patch-level inference results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for record in patch_results:
        grouped.setdefault(record_tile_id(record), []).append(record)

    rendered = []
    for tile_id, records in sorted(grouped.items()):
        if not records:
            continue
        origin_x = min(record_origin(record)[0] for record in records)
        origin_y = min(record_origin(record)[1] for record in records)
        max_x = max(record_origin(record)[0] + record_patch_shape(record)[0] for record in records)
        max_y = max(record_origin(record)[1] + record_patch_shape(record)[1] for record in records)
        width = max(1, max_x - origin_x)
        height = max(1, max_y - origin_y)

        background = Image.new("RGB", (width, height), "black")
        for record in records:
            image_path = resolve_image_path(record.get("image", ""), image_folder)
            if not image_path.exists():
                continue
            patch_width, patch_height = record_patch_shape(record)
            patch = Image.open(image_path).convert("RGB")
            if patch.size != (patch_width, patch_height):
                patch = patch.resize((patch_width, patch_height))
            x0, y0 = record_origin(record)
            background.paste(patch, (x0 - origin_x, y0 - origin_y))

        gt_lines = []
        pred_lines = []
        for record in records:
            gt_lines.extend(gt_lines_for_record(record, origin_x, origin_y))
            pred_lines.extend(prediction_lines_for_record(record, origin_x, origin_y))

        gt_canvas = draw_map_lines(background.copy(), {"lines": gt_lines}, (0, 255, 0), (255, 255, 0))
        pred_canvas = draw_map_lines(background.copy(), {"lines": pred_lines}, (255, 0, 0), (0, 128, 255))
        gt_panel = add_title(gt_canvas, f"{tile_id} Ground Truth")
        pred_panel = add_title(pred_canvas, f"{tile_id} Prediction")
        compare = Image.new("RGB", (gt_panel.width + pred_panel.width + 10, gt_panel.height), "black")
        compare.paste(gt_panel, (0, 0))
        compare.paste(pred_panel, (gt_panel.width + 10, 0))

        safe_tile = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tile_id)
        gt_path = output_dir / f"{safe_tile}_ground_truth.png"
        pred_path = output_dir / f"{safe_tile}_prediction.png"
        compare_path = output_dir / f"{safe_tile}_compare.png"
        gt_canvas.save(gt_path)
        pred_canvas.save(pred_path)
        compare.save(compare_path)
        rendered.append({
            "tile_id": tile_id,
            "origin": [origin_x, origin_y],
            "size": [width, height],
            "ground_truth": str(gt_path),
            "prediction": str(pred_path),
            "compare": str(compare_path),
        })
    return rendered
