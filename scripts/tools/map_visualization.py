#!/usr/bin/env python3
import json
import math
import re
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


def coerce_xy_point(point):
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return int(round(x)), int(round(y))


def sanitize_points(points):
    if not isinstance(points, (list, tuple)):
        return []
    xy_points = []
    for point in points:
        xy = coerce_xy_point(point)
        if xy is not None:
            xy_points.append(xy)
    return xy_points


def count_invalid_geometry(payload):
    invalid_lines = 0
    invalid_points = 0
    for item in normalize_lines(payload):
        if not isinstance(item, dict):
            invalid_lines += 1
            continue
        points = item.get("points")
        if points in (None, []):
            continue
        if not isinstance(points, (list, tuple)):
            invalid_lines += 1
            continue
        invalid_points += sum(1 for point in points if coerce_xy_point(point) is None)
    return invalid_lines, invalid_points


def offset_lines(lines, dx, dy):
    shifted = []
    for line in normalize_lines(lines):
        if not isinstance(line, dict):
            continue
        xy_points = sanitize_points(line.get("points") or [])
        if not xy_points:
            continue
        out = dict(line)
        out["points"] = [[int(round(x + dx)), int(round(y + dy))] for x, y in xy_points]
        shifted.append(out)
    return shifted


def draw_map_lines(image: Image.Image, payload, centerline_color: tuple, intersection_color: tuple, width: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for item in normalize_lines(payload):
        if not isinstance(item, dict):
            continue
        xy_points = sanitize_points(item.get("points") or [])
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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
ROW_COL_RE = re.compile(r"(?:^|[_-])r(?P<row>\d+)[_-]?c(?P<col>\d+)(?:[_-]|$)", re.IGNORECASE)


def _path_suffix_after_marker(path: Path, marker: str) -> Path | None:
    parts = path.parts
    lowered = [part.lower() for part in parts]
    if marker not in lowered:
        return None
    idx = lowered.index(marker)
    if idx >= len(parts) - 1:
        return None
    return Path(*parts[idx:])


def _parse_row_col_from_text(text: str) -> tuple[int | None, int | None]:
    match = ROW_COL_RE.search(text)
    if not match:
        return None, None
    return int(match.group("row")), int(match.group("col"))


def _record_row_col(record: dict) -> tuple[int, int]:
    meta = record.get("meta") or {}
    for row_key, col_key in (
        ("row", "col"),
        ("patch_row", "patch_col"),
    ):
        if row_key in record and col_key in record:
            return int(record[row_key]), int(record[col_key])
        if row_key in meta and col_key in meta:
            return int(meta[row_key]), int(meta[col_key])
    for key in ("image", "image_path", "record_id", "id"):
        value = record.get(key)
        if value:
            row, col = _parse_row_col_from_text(str(value))
            if row is not None and col is not None:
                return row, col
    return 0, 0


def _tile_id_from_text(text: str) -> str | None:
    value = Path(text).stem
    match = ROW_COL_RE.search(value)
    if match:
        value = value[: match.start()].rstrip("_-")
    if value:
        return value
    return None


def _candidate_image_paths(raw_path: str, image_folder: Path, record: dict | None = None) -> list[Path]:
    image_path = Path(raw_path)
    candidates = []
    if image_path.is_absolute() and image_path.exists():
        return [image_path]
    if raw_path:
        candidates.append(image_folder / image_path)
        for marker in ("images", "img"):
            suffix = _path_suffix_after_marker(image_path, marker)
            if suffix is not None:
                candidates.append(image_folder / suffix)
        candidates.append(image_folder / image_path.name)

    if record:
        meta = record.get("meta") or {}
        tile_id = record_tile_id(record)
        row, col = _record_row_col(record)
        city = str(record.get("city") or meta.get("city") or "")
        names = [
            f"{tile_id}_r{row:02d}_c{col:02d}.png",
            f"{tile_id}_r{row}_c{col}.png",
            f"r{row}_c{col}.png",
            f"r{row}_c{col}_p00.png",
            f"r{row}_c{col}_p01.png",
            f"r{row:02d}_c{col:02d}.png",
        ]
        for name in names:
            if city:
                candidates.append(image_folder / "images" / city / name)
                candidates.append(image_folder / city / name)
            candidates.append(image_folder / "images" / name)
            candidates.append(image_folder / "img" / tile_id / name)
            candidates.append(image_folder / tile_id / name)

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped or [image_path]


def build_image_basename_index(image_folder: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in image_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name, []).append(path)
    return index


def resolve_image_path(raw_path: str, image_folder: Path, record: dict | None = None, image_index: dict[str, list[Path]] | None = None) -> Path:
    candidates = _candidate_image_paths(raw_path, image_folder, record)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if image_index is not None:
        raw_name = Path(raw_path).name if raw_path else ""
        lookup_names = [raw_name] if raw_name else []
        if record:
            tile_id = record_tile_id(record)
            row, col = _record_row_col(record)
            lookup_names.extend([
                f"{tile_id}_r{row:02d}_c{col:02d}.png",
                f"{tile_id}_r{row}_c{col}.png",
                f"r{row}_c{col}.png",
                f"r{row}_c{col}_p00.png",
            ])
        for name in lookup_names:
            matches = image_index.get(name) or []
            if len(matches) == 1:
                return matches[0]
    return candidates[0]


def record_tile_id(record: dict) -> str:
    meta = record.get("meta") or {}
    value = record.get("tile_id") or meta.get("tile_id") or meta.get("log_id")
    if value:
        return str(value)
    for key in ("image", "image_path"):
        raw = record.get(key)
        if raw:
            path = Path(str(raw))
            parsed = _tile_id_from_text(str(raw))
            if parsed:
                return parsed
            parent = path.parent.name
            grandparent = path.parent.parent.name.lower()
            if parent and grandparent == "img":
                return path.parent.name
            if parent and parent not in (".", "") and grandparent != "images":
                return parent
    for key in ("record_id", "id"):
        raw = record.get(key)
        if raw:
            parsed = _tile_id_from_text(str(raw))
            if parsed:
                return parsed
    return "tile"


def record_patch_shape(record: dict) -> tuple[int, int]:
    patch_size = int(record.get("patch_size", (record.get("meta") or {}).get("patch_size", 256)))
    return (
        int(record.get("patch_width", (record.get("meta") or {}).get("patch_width", patch_size))),
        int(record.get("patch_height", (record.get("meta") or {}).get("patch_height", patch_size))),
    )


def record_origin(record: dict) -> tuple[int, int]:
    meta = record.get("meta") or {}
    patch_width, patch_height = record_patch_shape(record)
    row, col = _record_row_col(record)
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


def render_whole_map_visualizations(patch_results, image_folder: Path, output_dir: Path, *, scan_image_folder: bool = False):
    """Render one stitched BEV canvas per tile from patch-level inference results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for record in patch_results:
        grouped.setdefault(record_tile_id(record), []).append(record)

    rendered = []
    image_index = None
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
        num_images = 0
        missing_images = []
        for record in records:
            image_path = resolve_image_path(record.get("image", "") or record.get("image_path", ""), image_folder, record, image_index)
            if scan_image_folder and not image_path.exists():
                if image_index is None:
                    image_index = build_image_basename_index(image_folder)
                image_path = resolve_image_path(record.get("image", "") or record.get("image_path", ""), image_folder, record, image_index)
            if not image_path.exists():
                missing_images.append({
                    "record_id": str(record.get("record_id") or record.get("id") or ""),
                    "image": str(record.get("image") or record.get("image_path") or ""),
                    "resolved": str(image_path),
                })
                continue
            patch_width, patch_height = record_patch_shape(record)
            patch = Image.open(image_path).convert("RGB")
            if patch.size != (patch_width, patch_height):
                patch = patch.resize((patch_width, patch_height))
            x0, y0 = record_origin(record)
            background.paste(patch, (x0 - origin_x, y0 - origin_y))
            num_images += 1

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
            "num_records": len(records),
            "num_images": num_images,
            "num_missing_images": len(missing_images),
            "missing_images": missing_images[:20],
            "origin": [origin_x, origin_y],
            "size": [width, height],
            "ground_truth": str(gt_path),
            "prediction": str(pred_path),
            "compare": str(compare_path),
        })
    return rendered
