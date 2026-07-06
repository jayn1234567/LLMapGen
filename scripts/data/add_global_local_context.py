#!/usr/bin/env python3
"""Add 1024-context-resized-to-512 image fields for global-local SFT.

The script works from existing patch images: rows are grouped by tile/log,
stitched into a temporary canvas with their patch offsets, then each target
patch gets a context crop centered on that patch. The model still predicts in
the target local patch coordinate system.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


VIEW_PROMPT = (
    "You are given two BEV road-structure views in the visual input.\n"
    "View 1 is a surrounding context crop resized to the vision input size.\n"
    "View 2 is the target local patch.\n"
    "Use View 1 only as surrounding context. Predict road geometry only inside View 2.\n"
    "The output coordinate system is still the target local patch coordinate system described below."
)


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def first_int(*values):
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def row_meta(row):
    meta = row.get("meta")
    return meta if isinstance(meta, dict) else {}


def group_key(row):
    meta = row_meta(row)
    image = Path(str(row.get("image", "")))
    return (
        row.get("tile_id")
        or meta.get("tile_id")
        or meta.get("log_id")
        or meta.get("sample_id")
        or str(image.parent)
    )


def patch_size_for(row, image_size, default_patch_size):
    meta = row_meta(row)
    box = row.get("base_patch_box_full") or meta.get("base_patch_box_full")
    if isinstance(box, list) and len(box) == 4:
        width = first_int(box[2]) - first_int(box[0])
        if width and width > 0:
            return width
    return first_int(row.get("patch_size"), meta.get("patch_size"), default_patch_size, image_size[0])


def stride_for(row, patch_size, default_stride):
    meta = row_meta(row)
    return first_int(row.get("stride"), meta.get("stride"), default_stride, patch_size)


def patch_xy_for(row, patch_size, stride):
    meta = row_meta(row)
    box = row.get("base_patch_box_full") or meta.get("base_patch_box_full")
    if isinstance(box, list) and len(box) >= 2:
        x0 = first_int(box[0])
        y0 = first_int(box[1])
        if x0 is not None and y0 is not None:
            return x0, y0

    x0 = first_int(row.get("x0"), meta.get("x0"))
    y0 = first_int(row.get("y0"), meta.get("y0"))
    if x0 is not None and y0 is not None:
        return x0, y0

    col = first_int(row.get("patch_col"), row.get("col"), meta.get("patch_col"), meta.get("col"))
    patch_row = first_int(row.get("patch_row"), row.get("row"), meta.get("patch_row"), meta.get("row"))
    if col is None or patch_row is None:
        raise ValueError(f"Cannot infer patch row/col or x0/y0 for sample {row.get('id')}")
    return col * stride, patch_row * stride


def open_patch(image_folder: Path, image_rel: str):
    image_path = Path(image_rel)
    if not image_path.is_absolute():
        image_path = image_folder / image_path
    return Image.open(image_path).convert("RGB")


def paste_patch(canvas, image_folder: Path, row, default_patch_size, default_stride):
    patch = open_patch(image_folder, row["image"])
    patch_size = patch_size_for(row, patch.size, default_patch_size)
    stride = stride_for(row, patch_size, default_stride)
    x0, y0 = patch_xy_for(row, patch_size, stride)
    if patch.size != (patch_size, patch_size):
        patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
    canvas.paste(patch, (x0, y0))


def group_canvas_size(group_rows, image_folder: Path, default_patch_size, default_stride):
    max_x = 0
    max_y = 0
    mode = "RGB"
    for row in group_rows:
        patch = open_patch(image_folder, row["image"])
        mode = patch.mode
        patch_size = patch_size_for(row, patch.size, default_patch_size)
        stride = stride_for(row, patch_size, default_stride)
        x0, y0 = patch_xy_for(row, patch_size, stride)
        max_x = max(max_x, x0 + patch_size)
        max_y = max(max_y, y0 + patch_size)
        meta = row_meta(row)
        source_size = meta.get("source_image_size") or meta.get("original_source_image_size")
        if isinstance(source_size, list) and len(source_size) >= 2:
            sx = first_int(source_size[0])
            sy = first_int(source_size[1])
            if sx and sy:
                max_x = max(max_x, sx)
                max_y = max(max_y, sy)
    return max_x, max_y, mode


def crop_with_padding(canvas, box, fill=(0, 0, 0)):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    out = Image.new("RGB", (width, height), fill)
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(canvas.width, right)
    src_bottom = min(canvas.height, bottom)
    if src_right > src_left and src_bottom > src_top:
        crop = canvas.crop((src_left, src_top, src_right, src_bottom))
        out.paste(crop, (src_left - left, src_top - top))
    return out


def context_rel_path(context_dir: Path, image_rel: str, suffix: str):
    image_path = Path(image_rel)
    stem = image_path.stem + suffix + image_path.suffix
    return context_dir / image_path.parent / stem


def inject_view_prompt(row):
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return
    for sentence in conversations:
        if not isinstance(sentence, dict) or sentence.get("from") != "human":
            continue
        value = str(sentence.get("value", ""))
        if VIEW_PROMPT in value:
            return
        if "<image>" in value:
            sentence["value"] = value.replace("<image>", "<image>\n" + VIEW_PROMPT, 1)
        else:
            sentence["value"] = VIEW_PROMPT + "\n" + value
        return


def process_rows(rows, args):
    image_folder = args.image_folder
    context_dir = Path(args.context_dir)
    groups = defaultdict(list)
    for row in rows:
        if "image" not in row:
            continue
        groups[group_key(row)].append(row)

    for _, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        width, height, mode = group_canvas_size(group_rows, image_folder, args.patch_size, args.stride)
        canvas = Image.new(mode, (width, height), (0, 0, 0))
        for row in group_rows:
            paste_patch(canvas, image_folder, row, args.patch_size, args.stride)

        for row in group_rows:
            patch = open_patch(image_folder, row["image"])
            patch_size = patch_size_for(row, patch.size, args.patch_size)
            stride = stride_for(row, patch_size, args.stride)
            x0, y0 = patch_xy_for(row, patch_size, stride)
            pad = max(0, (args.context_size - patch_size) // 2)
            context = crop_with_padding(canvas, (x0 - pad, y0 - pad, x0 + patch_size + pad, y0 + patch_size + pad))
            if context.size != (args.output_size, args.output_size):
                context = context.resize((args.output_size, args.output_size), Image.BILINEAR)

            rel_context = context_rel_path(context_dir, row["image"], args.context_suffix)
            out_path = image_folder / rel_context
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not out_path.exists():
                context.save(out_path)
            row[args.context_image_key] = str(rel_context).replace("\\", "/")
            meta = row.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["context_image_key"] = args.context_image_key
                meta["context_size"] = args.context_size
                meta["context_resized_size"] = args.output_size
                meta["context_source"] = "stitched_neighbor_patches"
            if args.update_prompt:
                inject_view_prompt(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--context-dir", default="context_1024_resize512")
    parser.add_argument("--context-image-key", default="context_image")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--context-suffix", default="_ctx1024r512")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-update-prompt", dest="update_prompt", action="store_false")
    parser.set_defaults(update_prompt=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_json_or_jsonl(args.input_jsonl)
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]
    rows = process_rows(rows, args)
    dump_jsonl(args.output_jsonl, rows)
    print(f"wrote {len(rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
