#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def load_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canvas_size(summary, explicit_size=None):
    if explicit_size:
        return explicit_size
    max_x = max_y = 0
    for result in summary.get("patch_results", []):
        x0 = int(result.get("x0", 0))
        y0 = int(result.get("y0", 0))
        max_x = max(max_x, x0 + int(result.get("patch_size", 256)))
        max_y = max(max_y, y0 + int(result.get("patch_size", 256)))
    for line in summary.get("merged_global", {}).get("lines", []):
        for x, y in line.get("points", []):
            max_x = max(max_x, int(x) + 1)
            max_y = max(max_y, int(y) + 1)
    return max(max_x, 1), max(max_y, 1)


def draw_lines(image, lines, centerline_color, intersection_color, width):
    draw = ImageDraw.Draw(image)
    for line in lines:
        points = [(int(x), int(y)) for x, y in line.get("points", [])]
        if len(points) < 2:
            continue
        category = str(line.get("category", "centerline")).lower()
        color = intersection_color if category == "intersection" else centerline_color
        for p0, p1 in zip(points[:-1], points[1:]):
            draw.line([p0, p1], fill=color, width=width)
        for x, y in points:
            r = max(2, width)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return image


def draw_patch_grid(image, summary, color=(80, 80, 80)):
    draw = ImageDraw.Draw(image)
    seen = set()
    for result in summary.get("patch_results", []):
        x0 = int(result.get("x0", 0))
        y0 = int(result.get("y0", 0))
        patch_size = int(result.get("patch_size", 256))
        key = (x0, y0, patch_size)
        if key in seen:
            continue
        seen.add(key)
        draw.rectangle([x0, y0, x0 + patch_size - 1, y0 + patch_size - 1], outline=color, width=1)


def main():
    parser = argparse.ArgumentParser(description="Visualize merged global state-update predictions.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--background", default="black")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--draw-grid", action="store_true")
    parser.add_argument("--canvas-width", type=int, default=0)
    parser.add_argument("--canvas-height", type=int, default=0)
    args = parser.parse_args()

    summary = load_summary(Path(args.summary_json))
    explicit_size = None
    if args.canvas_width > 0 and args.canvas_height > 0:
        explicit_size = (args.canvas_width, args.canvas_height)
    size = canvas_size(summary, explicit_size)
    image = Image.new("RGB", size, args.background)
    if args.draw_grid:
        draw_patch_grid(image, summary)
    draw_lines(
        image,
        summary.get("merged_global", {}).get("lines", []),
        centerline_color=(255, 64, 64),
        intersection_color=(0, 180, 255),
        width=args.width,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
