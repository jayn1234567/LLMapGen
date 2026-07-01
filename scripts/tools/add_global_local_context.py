#!/usr/bin/env python3
"""Add a resized global-context view to LLMapGen centerline trainroots."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image


GRID_RE = re.compile(r"(?:^|[_/\\-])r(?P<row>\d+)[_\\-]?c(?P<col>\d+)(?:[_/\\.-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class PatchInfo:
    sample_id: str
    rel_image: str
    abs_image: Path
    scene_key: str
    row: int | None
    col: int | None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def meta_by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id", "")): row for row in rows if str(row.get("id", "")).strip()}


def first_image_rel(row: Dict[str, Any], meta: Dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, list) and images:
        return str(images[0]).strip().replace("\\", "/")
    meta_images = meta.get("images")
    if isinstance(meta_images, list) and meta_images:
        return str(meta_images[0]).strip().replace("\\", "/")
    return str(row.get("image", meta.get("image", ""))).strip().replace("\\", "/")


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_row_col(row: Dict[str, Any], meta: Dict[str, Any], rel_image: str) -> Tuple[int | None, int | None]:
    for source in (row, meta):
        for row_key, col_key in (
            ("patch_row", "patch_col"),
            ("grid_row", "grid_col"),
            ("row", "col"),
            ("r", "c"),
        ):
            r = parse_int(source.get(row_key))
            c = parse_int(source.get(col_key))
            if r is not None and c is not None:
                return r, c
    match = GRID_RE.search(str(rel_image))
    if match:
        return int(match.group("row")), int(match.group("col"))
    return None, None


def infer_scene_key(row: Dict[str, Any], meta: Dict[str, Any], rel_image: str) -> str:
    for source in (row, meta):
        for key in ("tile_id", "log_id", "scene_id", "map_id", "city"):
            value = str(source.get(key, "")).strip()
            if value:
                return value
    rel = Path(str(rel_image).replace("\\", "/"))
    parent = str(rel.parent).replace("\\", "/")
    match = GRID_RE.search(str(rel))
    if match:
        prefix = str(rel.name[: match.start()]).strip("_-.")
        return f"{parent}/{prefix}" if prefix else parent
    return parent


def build_patch_infos(
    rows: Sequence[Dict[str, Any]],
    metas: Sequence[Dict[str, Any]],
    media_dir: Path,
) -> List[PatchInfo]:
    metas_by_id = meta_by_id(metas)
    infos: List[PatchInfo] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", index))
        meta = metas_by_id.get(sample_id, {})
        rel_image = first_image_rel(row, meta)
        if not rel_image:
            continue
        row_idx, col_idx = infer_row_col(row, meta, rel_image)
        infos.append(
            PatchInfo(
                sample_id=sample_id,
                rel_image=rel_image,
                abs_image=(media_dir / rel_image).resolve(),
                scene_key=infer_scene_key(row, meta, rel_image),
                row=row_idx,
                col=col_idx,
            )
        )
    return infos


def open_patch(path: Path, local_patch_size: int) -> Image.Image:
    with Image.open(path) as img:
        patch = img.convert("RGB")
    target = int(local_patch_size)
    if patch.size != (target, target):
        patch = patch.resize((target, target), Image.Resampling.BICUBIC)
    return patch


def paste_clipped(canvas: Image.Image, patch: Image.Image, left: int, top: int) -> None:
    canvas_w, canvas_h = canvas.size
    patch_w, patch_h = patch.size
    dst_left = max(0, int(left))
    dst_top = max(0, int(top))
    dst_right = min(canvas_w, int(left) + patch_w)
    dst_bottom = min(canvas_h, int(top) + patch_h)
    if dst_right <= dst_left or dst_bottom <= dst_top:
        return
    src_left = dst_left - int(left)
    src_top = dst_top - int(top)
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)
    canvas.paste(patch.crop((src_left, src_top, src_right, src_bottom)), (dst_left, dst_top))


def build_context_image(
    info: PatchInfo,
    lookup: Dict[Tuple[str, int, int], PatchInfo],
    *,
    context_size: int,
    output_size: int,
    local_patch_size: int,
    pad_fill_rgb: Tuple[int, int, int],
) -> Image.Image:
    context = Image.new("RGB", (int(context_size), int(context_size)), tuple(pad_fill_rgb))
    half_offset = (int(context_size) - int(local_patch_size)) // 2
    if info.row is None or info.col is None:
        paste_clipped(context, open_patch(info.abs_image, int(local_patch_size)), half_offset, half_offset)
        return context.resize((int(output_size), int(output_size)), Image.Resampling.BICUBIC)

    radius = int(context_size) // int(local_patch_size) + 2
    for rr in range(int(info.row) - radius, int(info.row) + radius + 1):
        for cc in range(int(info.col) - radius, int(info.col) + radius + 1):
            neighbor = lookup.get((info.scene_key, rr, cc))
            if neighbor is None or not neighbor.abs_image.is_file():
                continue
            left = half_offset + (cc - int(info.col)) * int(local_patch_size)
            top = half_offset + (rr - int(info.row)) * int(local_patch_size)
            paste_clipped(context, open_patch(neighbor.abs_image, int(local_patch_size)), left, top)
    return context.resize((int(output_size), int(output_size)), Image.Resampling.BICUBIC)


def selected_rows(rows: Sequence[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    limit = int(max_samples)
    return [dict(row) for row in (rows[:limit] if limit > 0 else rows)]


def link_media_dirs(rows: Sequence[Dict[str, Any]], metas: Sequence[Dict[str, Any]], media_dir: Path, output_root: Path) -> None:
    top_dirs = set()
    metas_by_id = meta_by_id(metas)
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", index))
        rel_image = first_image_rel(row, metas_by_id.get(sample_id, {}))
        parts = Path(rel_image).parts
        if parts:
            top_dirs.add(parts[0])
    for name in sorted(top_dirs):
        source = (media_dir / name).resolve()
        target = output_root / name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
        except OSError:
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def process_split(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    trainroot = Path(args.trainroot).expanduser().resolve()
    output_root = Path(args.output_trainroot).expanduser().resolve()
    media_dir = Path(args.media_dir).expanduser().resolve() if str(args.media_dir).strip() else trainroot
    rows = load_jsonl(trainroot / f"{split}.jsonl")
    metas = load_jsonl(trainroot / f"meta_{split}.jsonl")
    if not rows:
        return {"split": split, "rows": 0, "written": 0, "missing_images": 0}

    all_infos = build_patch_infos(rows, metas, media_dir)
    info_by_id = {info.sample_id: info for info in all_infos}
    lookup = {
        (info.scene_key, int(info.row), int(info.col)): info
        for info in all_infos
        if info.row is not None and info.col is not None
    }
    max_samples = int(args.max_train_samples if split == "train" else args.max_eval_samples)
    if max_samples <= 0:
        max_samples = int(args.max_samples)
    out_rows = selected_rows(rows, max_samples)
    selected_ids = {str(row.get("id", index)) for index, row in enumerate(out_rows)}
    out_metas = [dict(meta) for meta in metas if str(meta.get("id", "")) in selected_ids]
    out_meta_by_id = meta_by_id(out_metas)

    context_dir = output_root / f"context_{int(args.context_size)}_resize{int(args.output_size)}" / split
    context_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_images = 0
    for index, row in enumerate(out_rows):
        sample_id = str(row.get("id", index))
        info = info_by_id.get(sample_id)
        if info is None or not info.abs_image.is_file():
            missing_images += 1
            continue
        context = build_context_image(
            info,
            lookup,
            context_size=int(args.context_size),
            output_size=int(args.output_size),
            local_patch_size=int(args.local_patch_size),
            pad_fill_rgb=tuple(int(x) for x in args.pad_fill_rgb),
        )
        context_rel = (context_dir / f"{sample_id}.png").relative_to(output_root).as_posix()
        context.save(output_root / context_rel)
        row[str(args.context_image_key)] = context_rel
        row["global_local_context"] = {
            "context_size": int(args.context_size),
            "context_output_size": int(args.output_size),
            "local_patch_size": int(args.local_patch_size),
            "view_order": ["global_context", "local_patch"],
        }
        meta = out_meta_by_id.get(sample_id)
        if meta is not None:
            meta[str(args.context_image_key)] = context_rel
            meta["global_local_context"] = dict(row["global_local_context"])
        written += 1

    link_media_dirs(rows, metas, media_dir, output_root)
    write_jsonl(output_root / f"{split}.jsonl", out_rows)
    if out_metas:
        write_jsonl(output_root / f"meta_{split}.jsonl", out_metas)
    return {"split": split, "rows": len(rows), "written": written, "missing_images": missing_images}


def parse_pad_fill(text: str) -> Tuple[int, int, int]:
    parts = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--pad-fill-rgb must contain three comma-separated integers.")
    return tuple(max(0, min(255, value)) for value in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", type=str, required=True)
    parser.add_argument("--output-trainroot", type=str, required=True)
    parser.add_argument("--media-dir", type=str, default="")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--context-image-key", type=str, default="context_image")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--local-patch-size", type=int, default=512)
    parser.add_argument("--pad-fill-rgb", type=parse_pad_fill, default=parse_pad_fill("0,0,0"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_trainroot).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [process_split(args, split) for split in args.splits]
    info_path = output_root / "global_local_context_info.json"
    info_path.write_text(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_trainroot": str(output_root), "summaries": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
