from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


PANEL_BG = (18, 20, 28)
GT_COLOR = (48, 160, 255)
PRED_COLOR = (255, 80, 80)
STATE_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT and predicted centerlines from a predictions.jsonl file.")
    parser.add_argument("--predictions-jsonl", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--line-width", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def sanitize_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return safe or "sample"


def _coerce_point(raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        x = int(round(float(raw[0])))
        y = int(round(float(raw[1])))
    except (TypeError, ValueError):
        return None
    return x, y


def sanitize_lines(lines: Any) -> List[List[tuple[int, int]]]:
    # 这里只保留真正可画的 polyline，顺手过滤掉坏点、单点线和异常结构。
    if not isinstance(lines, list):
        return []
    cleaned: List[List[tuple[int, int]]] = []
    for item in lines:
        if not isinstance(item, dict):
            continue
        raw_points = item.get("points", [])
        if not isinstance(raw_points, (list, tuple)):
            continue
        points = [_coerce_point(raw) for raw in raw_points]
        polyline = [point for point in points if point is not None]
        if len(polyline) >= 2:
            cleaned.append(polyline)
    return cleaned


def draw_polylines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[Sequence[tuple[int, int]]],
    *,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    for line in lines:
        if len(line) >= 2:
            draw.line(line, fill=color, width=int(line_width), joint="curve")


def render_panel(
    *,
    image: Image.Image,
    title: str,
    lines: Sequence[Sequence[tuple[int, int]]],
    state_lines: Sequence[Sequence[tuple[int, int]]],
    line_color: tuple[int, int, int],
    line_width: int,
) -> Image.Image:
    # 一个 panel 只做一件事：把底图、state 参考线和 GT/Pred 折线叠在一起，方便肉眼对比。
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if state_lines:
        draw_polylines(draw, state_lines, color=STATE_COLOR, line_width=max(1, int(line_width) - 1))
    draw_polylines(draw, lines, color=line_color, line_width=int(line_width))

    header_h = 56
    panel = Image.new("RGB", (canvas.width, canvas.height + header_h), color=PANEL_BG)
    panel.paste(canvas, (0, header_h))
    header_draw = ImageDraw.Draw(panel)
    header_draw.text((12, 10), title, fill=(245, 245, 245), font=get_font(18))
    header_draw.text(
        (12, 32),
        f"lines={len(lines)} state_lines={len(state_lines)}",
        fill=(220, 220, 220),
        font=get_font(13),
    )
    return panel


def stack_panels(panels: Iterable[Image.Image]) -> Image.Image:
    images = list(panels)
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def main() -> None:
    args = parse_args()
    predictions_jsonl = Path(args.predictions_jsonl).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(predictions_jsonl)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    saved = 0
    # 逐条读取 predictions.jsonl，按 “GT | Pred” 双栏方式输出，方便快速抽看模型几何质量。
    for row in rows:
        sample_id = str(row.get("id", "sample"))
        rel_image = str(row.get("image", "")).strip()
        image_path = (dataset_root / rel_image).resolve()
        if not image_path.is_file():
            continue

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        gt_lines = sanitize_lines(row.get("gt_lines", []))
        pred_lines = sanitize_lines(row.get("pred_lines", []))
        state_lines = sanitize_lines(row.get("state_lines", []))

        gt_panel = render_panel(
            image=image,
            title=f"{sample_id} | GT",
            lines=gt_lines,
            state_lines=state_lines,
            line_color=GT_COLOR,
            line_width=int(args.line_width),
        )
        pred_panel = render_panel(
            image=image,
            title=f"{sample_id} | Pred",
            lines=pred_lines,
            state_lines=state_lines,
            line_color=PRED_COLOR,
            line_width=int(args.line_width),
        )
        combined = stack_panels([gt_panel, pred_panel])
        combined.save(output_dir / f"{sanitize_name(sample_id)}.png")
        saved += 1

    print(json.dumps({"saved": saved, "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
