"""可视化 helper。

当前主链入口脚本已经不再依赖这些函数，但为了以后需要做 QA 图，
这里仍保留基础画图能力。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

from PIL import Image, ImageDraw

from .common import ensure_dir


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    """在图上画一个端点圆点。"""
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[int]],
    color: Tuple[int, int, int],
    width: int = 3,
) -> None:
    """在图上画一条折线。"""
    pts = [tuple(int(v) for v in point[:2]) for point in points]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=int(width))


def save_patch_lines_visualization(
    *,
    image: Image.Image,
    lines: Sequence[dict],
    out_path: Path,
    border_color: Tuple[int, int, int] = (255, 0, 180),
) -> None:
    """把 patch 和其中的多条线画成一张 QA 图并保存。"""
    ensure_dir(out_path.parent)
    panel = image.convert("RGB")
    draw = ImageDraw.Draw(panel)
    patch_size = int(panel.size[0])
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=border_color, width=2)
    for line in lines:
        pts = [tuple(int(v) for v in point[:2]) for point in line.get("points", [])]
        if len(pts) < 2:
            continue
        draw.line(pts, fill=(40, 220, 255), width=3)
        draw_endpoint(draw, pts[0], (0, 180, 220), 3)
        draw_endpoint(draw, pts[-1], (0, 180, 220), 3)
    panel.save(out_path)
