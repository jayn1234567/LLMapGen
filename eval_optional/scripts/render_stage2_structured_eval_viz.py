#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


GRID_ROWS = 8
GRID_COLS = 8
PATCH_SIZE = 512

STATE_COLORS = {
    "background": (28, 30, 38, 120),
    "lane_boundary": (80, 190, 255, 150),
    "lane_divider": (255, 176, 72, 150),
    "mix": (220, 104, 255, 150),
    "__invalid__": (255, 64, 64, 180),
}
TEXT_ABBR = {
    "background": "BG",
    "lane_boundary": "BND",
    "lane_divider": "DIV",
    "mix": "MIX",
    "__invalid__": "INV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage-2 structured-eval panels: input + GT states + Pred states.")
    parser.add_argument("--predictions-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--grid-rows", type=int, default=GRID_ROWS)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--contact-sheet-count", type=int, default=16)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return safe or "sample"


def grid_cell_bounds(
    *,
    row_idx: int,
    col_idx: int,
    patch_size: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    patch = max(1, int(patch_size))
    x0 = (int(col_idx) * patch) // cols
    x1 = (((int(col_idx) + 1) * patch) // cols) - 1
    y0 = (int(row_idx) * patch) // rows
    y1 = (((int(row_idx) + 1) * patch) // rows) - 1
    return int(x0), int(x1), int(y0), int(y1)


def get_font(size: int = 14) -> ImageFont.ImageFont:
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


def load_patch_image(image_path: Path, tile_size: int) -> tuple[Image.Image, bool]:
    # 如果原图缺失，就直接生成一张占位图，这样整批渲染不会因为个别坏样本中断。
    if not image_path.is_file():
        missing = Image.new("RGB", (int(tile_size), int(tile_size)), color=(20, 22, 28))
        draw = ImageDraw.Draw(missing)
        font = get_font(16)
        draw.text((16, 16), "missing image", fill=(255, 96, 96), font=font)
        draw.text((16, 40), image_path.name, fill=(220, 220, 220), font=font)
        return missing, True
    with Image.open(image_path) as img:
        patch = img.convert("RGB").resize((int(tile_size), int(tile_size)), Image.Resampling.BILINEAR)
    return patch, False


def draw_grid_lines(canvas: Image.Image, *, grid_rows: int, grid_cols: int, color: tuple[int, int, int], width: int = 1) -> None:
    draw = ImageDraw.Draw(canvas)
    tile_size = int(canvas.size[0])
    for row in range(1, int(grid_rows)):
        y = int(round(tile_size * float(row) / float(grid_rows)))
        draw.line((0, y, tile_size, y), fill=color, width=int(width))
    for col in range(1, int(grid_cols)):
        x = int(round(tile_size * float(col) / float(grid_cols)))
        draw.line((x, 0, x, tile_size), fill=color, width=int(width))


def render_state_overlay(
    *,
    patch: Image.Image,
    grid_states: Sequence[str],
    mismatch_indices: set[int],
    tile_size: int,
    patch_size: int,
    grid_rows: int,
    grid_cols: int,
    title: str,
    scene_text: str,
    footer_text: str,
) -> Image.Image:
    # GT / Pred 都画成统一的 8x8 网格热力图；预测错的格子额外用红框强调。
    base = patch.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = get_font(14)
    small_font = get_font(12)
    scale = float(tile_size) / float(max(1, int(patch_size)))

    for row_idx in range(int(grid_rows)):
        for col_idx in range(int(grid_cols)):
            cell_idx = (row_idx * int(grid_cols)) + col_idx
            state = str(grid_states[cell_idx]).strip().lower() if cell_idx < len(grid_states) else "__invalid__"
            color = STATE_COLORS.get(state, STATE_COLORS["__invalid__"])
            x0, x1, y0, y1 = grid_cell_bounds(
                row_idx=row_idx,
                col_idx=col_idx,
                patch_size=int(patch_size),
                grid_rows=int(grid_rows),
                grid_cols=int(grid_cols),
            )
            sx0 = int(round(x0 * scale))
            sx1 = int(round((x1 + 1) * scale))
            sy0 = int(round(y0 * scale))
            sy1 = int(round((y1 + 1) * scale))
            draw.rectangle((sx0, sy0, sx1, sy1), fill=color)
            border_color = (255, 72, 72, 255) if cell_idx in mismatch_indices else (255, 255, 255, 96)
            border_width = 3 if cell_idx in mismatch_indices else 1
            draw.rectangle((sx0, sy0, sx1, sy1), outline=border_color, width=border_width)
            idx_label = f"{cell_idx + 1:02d}"
            draw.rectangle((sx0 + 2, sy0 + 2, sx0 + 28, sy0 + 16), fill=(0, 0, 0, 160))
            draw.text((sx0 + 4, sy0 + 3), idx_label, fill=(255, 255, 255), font=small_font)
            abbr = TEXT_ABBR.get(state, "UNK")
            draw.text((sx0 + 4, sy0 + 22), abbr, fill=(255, 255, 255), font=font)

    merged = Image.alpha_composite(base, overlay).convert("RGB")
    draw_grid_lines(merged, grid_rows=int(grid_rows), grid_cols=int(grid_cols), color=(240, 240, 240), width=1)

    panel = Image.new("RGB", (tile_size, tile_size + 96), color=(18, 20, 28))
    panel.paste(merged, (0, 96))
    draw_panel = ImageDraw.Draw(panel)
    draw_panel.text((10, 10), title, fill=(245, 245, 245), font=get_font(18))
    draw_panel.text((10, 34), scene_text, fill=(235, 235, 235), font=get_font(14))
    draw_panel.text((10, 56), footer_text, fill=(235, 235, 235), font=get_font(14))
    return panel


def render_input_panel(
    *,
    patch: Image.Image,
    tile_size: int,
    sample_id: str,
    stats_text: str,
) -> Image.Image:
    canvas = patch.convert("RGB").copy()
    draw_grid_lines(canvas, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, color=(255, 210, 128), width=1)
    panel = Image.new("RGB", (tile_size, tile_size + 96), color=(18, 20, 28))
    panel.paste(canvas, (0, 96))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), "Input", fill=(245, 245, 245), font=get_font(18))
    draw.text((10, 34), sample_id, fill=(235, 235, 235), font=get_font(13))
    draw.text((10, 56), stats_text, fill=(235, 235, 235), font=get_font(14))
    return panel


def stack_three(left: Image.Image, middle: Image.Image, right: Image.Image) -> Image.Image:
    width = int(left.size[0] + middle.size[0] + right.size[0])
    height = int(max(left.size[1], middle.size[1], right.size[1]))
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in (left, middle, right):
        canvas.paste(image, (offset, 0))
        offset += int(image.size[0])
    return canvas


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path, *, columns: int = 4) -> None:
    paths = [path for path in image_paths if path.is_file()]
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    tile_w, tile_h = images[0].size
    cols = max(1, int(columns))
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        canvas.paste(image, (col * tile_w, row * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    predictions_jsonl = Path(args.predictions_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(predictions_jsonl)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    # 输出既保留逐样本三栏图，也额外生成 contact sheet，便于先粗扫再挑重点样本。
    saved_paths: List[Path] = []
    for row in rows:
        sample_id = str(row.get("id", "sample"))
        image_path = Path(str(row.get("image_path", ""))).resolve()
        patch, image_missing = load_patch_image(image_path, int(args.tile_size))

        gt_states = list(row.get("gt_grid_states", []))
        pred_states = list(row.get("pred_grid_states", []))
        mismatch_indices = {
            idx
            for idx, (gt_state, pred_state) in enumerate(zip(gt_states, pred_states))
            if str(gt_state) != str(pred_state)
        }

        input_panel = render_input_panel(
            patch=patch,
            tile_size=int(args.tile_size),
            sample_id=sample_id,
            stats_text=(
                f"scene_correct={str(bool(row.get('scene_correct', False))).lower()} "
                f"cell_correct={int(row.get('grid_cell_correct', 0))}/{int(row.get('grid_cell_total', 64))} "
                f"parse_ok={str(bool(row.get('pred_parse_ok', False))).lower()} "
                f"missing={str(bool(image_missing)).lower()}"
            ),
        )
        gt_panel = render_state_overlay(
            patch=patch,
            grid_states=gt_states,
            mismatch_indices=set(),
            tile_size=int(args.tile_size),
            patch_size=int(args.patch_size),
            grid_rows=int(args.grid_rows),
            grid_cols=int(args.grid_cols),
            title="GT State",
            scene_text=f"scene={str(row.get('gt_scene', ''))}",
            footer_text="white grid only",
        )
        pred_panel = render_state_overlay(
            patch=patch,
            grid_states=pred_states,
            mismatch_indices=mismatch_indices,
            tile_size=int(args.tile_size),
            patch_size=int(args.patch_size),
            grid_rows=int(args.grid_rows),
            grid_cols=int(args.grid_cols),
            title="Pred State",
            scene_text=f"scene={str(row.get('pred_scene', ''))}",
            footer_text=f"mismatch_cells={len(mismatch_indices)} red_border=error",
        )
        combined = stack_three(input_panel, gt_panel, pred_panel)
        output_path = output_dir / f"{sanitize_name(sample_id)}.png"
        combined.save(output_path)
        saved_paths.append(output_path)

    manifest = {
        "predictions_jsonl": str(predictions_jsonl),
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_saved": len(saved_paths),
        "contact_sheet_count": min(len(saved_paths), int(args.contact_sheet_count)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_count = min(len(saved_paths), int(args.contact_sheet_count))
    if contact_count > 0:
        build_contact_sheet(saved_paths[:contact_count], output_dir / "contact_sheet_first16.png", columns=4)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
