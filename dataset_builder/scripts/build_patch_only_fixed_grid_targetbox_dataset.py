"""Build a fixed-grid target-box dataset from an existing patch-only root.

This script keeps the full patch image but changes supervision from
"full-patch map" to "predict only inside the current target box".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, IO, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `common` 里是通用数据集 helper。
# - `load_jsonl`: 读取 patch-only 的主 jsonl 和 meta jsonl
# - `extract_message_content`: 从源 ShareGPT 样本里提取 system prompt
# - `link_or_copy_images`: 在输出根下暴露 images/ 树
# - `build_sharegpt_dataset_info`: 生成 LLaMAFactory 可读的 dataset_info.json
from unimapgen.dataset_build_refactor.common import (
    build_sharegpt_dataset_info,
    ensure_dir,
    extract_message_content,
    link_or_copy_images,
    load_jsonl,
    make_sharegpt_record,
)
# `fixed16` 里是 Stage A box-level 目标构建 helper。
# - `build_grid_boxes`: 把一张 896 patch 切成固定 4x4 box
# - `build_prompt_endpoints`: 为当前 box 选 prompt anchor
# - `build_target_lines_for_box`: 把 patch GT 裁到当前 box 内
from unimapgen.dataset_build_refactor.fixed16 import (
    build_grid_boxes,
    build_prompt_endpoints,
    build_target_lines_for_box,
    format_fixed16_prompt,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage A fixed-grid dataset building.

    Important parameter groups:
    - roots and splits: `--input-root`, `--output-root`, `--splits`
    - task geometry: `--grid-size`
    - sampling: `--target-empty-ratio`, `--seed`
    - target-line construction: `--resample-step-px`, `--boundary-tol-px`
    - output exposure: `--image-root-mode`
    """
    parser = argparse.ArgumentParser(
        description="Refactored v2: build a fixed-grid target-box patch-only dataset from an existing full-patch patch-only dataset."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Existing patch-only dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Splits to process.")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size per side. 4 means 16 fixed target boxes.")
    parser.add_argument("--target-empty-ratio", type=float, default=0.10, help="Maximum empty-sample ratio after filtering.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for empty-sample downsampling.")
    parser.add_argument("--resample-step-px", type=float, default=12.0, help="Resample clipped target lines. Use <=0 to keep original point spacing.")
    parser.add_argument("--boundary-tol-px", type=float, default=2.5, help="Boundary tolerance for cut endpoint detection.")
    parser.add_argument("--use-system-prompt-from-source", action="store_true", help="Reuse the source system prompt if present.")
    parser.add_argument("--image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"], help="How to expose images under the output root.")
    return parser.parse_args()


def write_jsonl_line(handle: IO[str], row: Dict) -> None:
    """Append a single json object as one jsonl line."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_empty_indices(empty_count: int, keep_empty: int, rng: random.Random) -> Optional[Set[int]]:
    """Choose which empty samples to keep after ratio-based downsampling."""
    if keep_empty <= 0:
        return set()
    if keep_empty >= empty_count:
        return None
    return set(rng.sample(range(empty_count), keep_empty))


def append_sampled_jsonl_pairs(
    *,
    rows_src: Path,
    meta_src: Path,
    rows_dst: Path,
    meta_dst: Path,
    keep_indices: Optional[Set[int]],
) -> int:
    """Append the selected empty rows from temp jsonl files into the final split."""
    written = 0
    with rows_dst.open("a", encoding="utf-8") as row_out, meta_dst.open("a", encoding="utf-8") as meta_out:
        with rows_src.open("r", encoding="utf-8") as row_in, meta_src.open("r", encoding="utf-8") as meta_in:
            for idx, (row_line, meta_line) in enumerate(zip(row_in, meta_in)):
                if keep_indices is not None and idx not in keep_indices:
                    continue
                row_out.write(row_line)
                meta_out.write(meta_line)
                written += 1
    return written


def report_split_progress(
    *,
    split: str,
    used_source: int,
    total_source: int,
    generated_total: int,
    generated_non_empty: int,
    generated_empty: int,
) -> None:
    """Print coarse progress so long fixed16 builds remain inspectable in logs."""
    total_text = str(total_source) if total_source > 0 else "?"
    print(
        (
            f"[fixed16-build] split={split} source={used_source}/{total_text} "
            f"generated={generated_total} non_empty={generated_non_empty} empty={generated_empty}"
        ),
        flush=True,
    )


def build_fixed16_record(
    *,
    sample_id: str,
    image_rel_path: str,
    prompt_text: str,
    target_lines: Sequence[Dict],
    system_prompt: str,
) -> Dict:
    """Build one ShareGPT row for a fixed16 target-box sample."""
    return make_sharegpt_record(
        sample_id=sample_id,
        image_rel_path=image_rel_path,
        user_text=prompt_text,
        assistant_payload={"lines": list(target_lines)},
        system_prompt=system_prompt,
    )


def build_fixed16_meta_row(
    *,
    row_id: str,
    source_meta: Dict,
    box: Dict[str, int],
    prompt_info: Dict[str, object],
    prompt_text: str,
    target_lines: Sequence[Dict],
    grid_size: int,
    resample_step_px: float,
    system_prompt: str,
) -> Dict:
    """Build one `meta_*.jsonl` row for a fixed16 sample.

    Parameters:
    - `source_meta`: patch-only meta row for the parent patch.
    - `box`: one grid box inside the patch.
    - `prompt_info`: chosen anchor endpoints for the user prompt.
    - `target_lines`: GT lines clipped to the current box.
    """
    return {
        "id": row_id,
        "source_id": str(source_meta.get("id")),
        "split": source_meta.get("split"),
        "family_id": source_meta.get("family_id"),
        "source_image": source_meta.get("source_image"),
        "patch_id": source_meta.get("patch_id"),
        "row": source_meta.get("row"),
        "col": source_meta.get("col"),
        "scan_index": source_meta.get("scan_index"),
        "image": source_meta.get("image"),
        "crop_box": source_meta.get("crop_box"),
        "target_mode": "fixed_grid_target_box_map",
        "coord_system": source_meta.get("coord_system", "patch_local_896"),
        "serialization_mode": source_meta.get("serialization_mode", "paper_structured"),
        "line_direction_mode": source_meta.get("line_direction_mode", "canonical_cut_then_origin"),
        "line_sort_mode": source_meta.get("line_sort_mode", "first_point_distance_to_patch_origin"),
        "resample_mode": "equal_distance" if float(resample_step_px) > 0 else "inherit_source_spacing",
        "resample_step_px": float(resample_step_px),
        "has_system_prompt": bool(system_prompt.strip()),
        "grid_size": int(grid_size),
        "grid_row": int(box["grid_row"]),
        "grid_col": int(box["grid_col"]),
        "target_box": {
            "x_min": int(box["x_min"]),
            "y_min": int(box["y_min"]),
            "x_max": int(box["x_max"]),
            "y_max": int(box["y_max"]),
        },
        "target_box_area": int(
            (int(box["x_max"]) - int(box["x_min"]) + 1)
            * (int(box["y_max"]) - int(box["y_min"]) + 1)
        ),
        "anchor_source": str(prompt_info["anchor_source"]),
        "anchor_start_xy": [int(prompt_info["start_x"]), int(prompt_info["start_y"])],
        "anchor_end_xy": [int(prompt_info["end_x"]), int(prompt_info["end_y"])],
        "anchor_piece_points": prompt_info["anchor_piece_points"],
        "num_target_lines": len(target_lines),
        "num_target_points": int(sum(len(item.get("points", [])) for item in target_lines)),
        "prompt_text": prompt_text,
        "target_lines": list(target_lines),
    }


def build_split(
    *,
    split: str,
    input_root: Path,
    output_root: Path,
    grid_size: int,
    target_empty_ratio: float,
    rng: random.Random,
    boundary_tol_px: float,
    resample_step_px: float,
    reuse_system_prompt: bool,
) -> Dict[str, object]:
    """Expand one split from patch-level samples into fixed-grid box samples.

    Workflow:
    - load patch-only rows/meta
    - generate `grid_size x grid_size` boxes per patch
    - clip GT into each box
    - stream non-empty and empty samples separately
    - downsample empties during finalize
    """
    split_jsonl = input_root / f"{split}.jsonl"
    split_meta_jsonl = input_root / f"meta_{split}.jsonl"
    if not split_jsonl.exists() or not split_meta_jsonl.exists():
        return {
            "missing_split": True,
            "split_jsonl": str(split_jsonl),
            "split_meta_jsonl": str(split_meta_jsonl),
        }

    print(f"[fixed16-build] split={split} loading rows from {split_jsonl}", flush=True)
    # `load_jsonl` 读取 patch-only 的 ShareGPT 样本。
    rows = load_jsonl(split_jsonl)
    print(f"[fixed16-build] split={split} loading meta from {split_meta_jsonl}", flush=True)
    # `load_jsonl` 读取 patch-only 的 meta；fixed16 真正依赖的是这里面的 target_lines / crop_box / image。
    meta_rows = load_jsonl(split_meta_jsonl)
    row_by_id = {str(row.get("id")): row for row in rows}

    total_source = len(meta_rows)
    print(
        f"[fixed16-build] split={split} loaded rows={len(rows)} meta_rows={len(meta_rows)}",
        flush=True,
    )
    temp_root = output_root / ".tmp_fixed16_build" / split
    if temp_root.exists():
        shutil.rmtree(temp_root)
    ensure_dir(temp_root)

    non_empty_rows_path = temp_root / "non_empty_rows.jsonl"
    non_empty_meta_path = temp_root / "non_empty_meta.jsonl"
    empty_rows_path = temp_root / "empty_rows.jsonl"
    empty_meta_path = temp_root / "empty_meta.jsonl"

    generated_total = 0
    generated_non_empty = 0
    generated_empty = 0
    used_source = 0
    progress_interval = 250

    try:
        with (
            non_empty_rows_path.open("w", encoding="utf-8") as non_empty_rows_out,
            non_empty_meta_path.open("w", encoding="utf-8") as non_empty_meta_out,
            empty_rows_path.open("w", encoding="utf-8") as empty_rows_out,
            empty_meta_path.open("w", encoding="utf-8") as empty_meta_out,
        ):
            for src_meta in meta_rows:
                row_id = str(src_meta.get("id", ""))
                src_row = row_by_id.get(row_id)
                if src_row is None:
                    continue

                image_rel_path = str(src_meta.get("image") or src_row.get("images", [""])[0])
                crop_box = dict(src_meta.get("crop_box", {}) or {})
                patch_size = int(crop_box.get("x_max", 0)) - int(crop_box.get("x_min", 0))
                if patch_size <= 1:
                    continue
                full_patch_target_lines = list(src_meta.get("target_lines", []))
                # `extract_message_content` 只是在需要时把源样本里的 system prompt 原样继承过来。
                system_prompt = extract_message_content(src_row, "system") if reuse_system_prompt else ""
                # `build_grid_boxes` 根据 patch_size 和 grid_size 生成 4x4 固定 box。
                boxes = build_grid_boxes(patch_size=patch_size, grid_size=int(grid_size))
                for box in boxes:
                    prompt_info = build_prompt_endpoints(
                        target_lines=full_patch_target_lines,
                        target_box=box,
                        patch_size=patch_size,
                    )
                    target_lines = build_target_lines_for_box(
                        full_patch_target_lines=full_patch_target_lines,
                        target_box=box,
                        patch_size=patch_size,
                        boundary_tol_px=boundary_tol_px,
                        resample_step_px=resample_step_px,
                    )
                    prompt_text = format_fixed16_prompt(
                        {
                            "start_x": int(prompt_info["start_x"]),
                            "start_y": int(prompt_info["start_y"]),
                            "end_x": int(prompt_info["end_x"]),
                            "end_y": int(prompt_info["end_y"]),
                            "box_x_min": int(box["x_min"]),
                            "box_y_min": int(box["y_min"]),
                            "box_x_max": int(box["x_max"]),
                            "box_y_max": int(box["y_max"]),
                        }
                    )
                    new_row_id = f"{row_id}_g{int(box['grid_row'])}{int(box['grid_col'])}"
                    row = build_fixed16_record(
                        sample_id=new_row_id,
                        image_rel_path=image_rel_path,
                        prompt_text=prompt_text,
                        target_lines=target_lines,
                        system_prompt=system_prompt,
                    )
                    meta = build_fixed16_meta_row(
                        row_id=new_row_id,
                        source_meta=src_meta,
                        box=box,
                        prompt_info=prompt_info,
                        prompt_text=prompt_text,
                        target_lines=target_lines,
                        grid_size=grid_size,
                        resample_step_px=resample_step_px,
                        system_prompt=system_prompt,
                    )
                    if meta["num_target_lines"] > 0:
                        write_jsonl_line(non_empty_rows_out, row)
                        write_jsonl_line(non_empty_meta_out, meta)
                        generated_non_empty += 1
                    else:
                        write_jsonl_line(empty_rows_out, row)
                        write_jsonl_line(empty_meta_out, meta)
                        generated_empty += 1
                    generated_total += 1

                used_source += 1
                if used_source == 1 or used_source % progress_interval == 0 or used_source == total_source:
                    report_split_progress(
                        split=split,
                        used_source=used_source,
                        total_source=total_source,
                        generated_total=generated_total,
                        generated_non_empty=generated_non_empty,
                        generated_empty=generated_empty,
                    )

        if generated_non_empty <= 0 and float(target_empty_ratio) < 1.0:
            raise ValueError("Split contains no non-empty samples; cannot enforce empty-ratio target.")

        keep_empty = (
            generated_empty
            if float(target_empty_ratio) >= 1.0
            else min(generated_empty, math.floor(generated_non_empty * target_empty_ratio / (1.0 - target_empty_ratio)))
        )
        keep_empty_indices = choose_empty_indices(empty_count=generated_empty, keep_empty=keep_empty, rng=rng)

        final_rows_path = output_root / f"{split}.jsonl"
        final_meta_path = output_root / f"meta_{split}.jsonl"
        shutil.copyfile(non_empty_rows_path, final_rows_path)
        shutil.copyfile(non_empty_meta_path, final_meta_path)
        kept_empty = append_sampled_jsonl_pairs(
            rows_src=empty_rows_path,
            meta_src=empty_meta_path,
            rows_dst=final_rows_path,
            meta_dst=final_meta_path,
            keep_indices=keep_empty_indices,
        )
        kept_total = generated_non_empty + kept_empty
        print(
            (
                f"[fixed16-build] split={split} finalize kept_total={kept_total} "
                f"kept_non_empty={generated_non_empty} kept_empty={kept_empty}"
            ),
            flush=True,
        )
        return {
            "generated_total": generated_total,
            "generated_non_empty": generated_non_empty,
            "generated_empty": generated_empty,
            "kept_total": kept_total,
            "kept_non_empty": generated_non_empty,
            "kept_empty": kept_empty,
            "kept_empty_ratio": (kept_empty / kept_total if kept_total else 0.0),
            "used_source_samples": used_source,
            "written_rows": kept_total,
            "written_meta_rows": kept_total,
        }
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    """Validate args, build requested splits, and write summary metadata."""
    args = parse_args()
    if int(args.grid_size) <= 0:
        raise ValueError("--grid-size must be positive.")
    if not (0.0 <= float(args.target_empty_ratio) <= 1.0):
        raise ValueError("--target-empty-ratio must be in [0, 1]. Use 1.0 to keep all empty boxes.")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    image_mode = link_or_copy_images(input_root=input_root, output_root=output_root, mode=str(args.image_root_mode))
    rng = random.Random(int(args.seed))

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "grid_size": int(args.grid_size),
        "num_boxes_per_patch": int(args.grid_size) * int(args.grid_size),
        "target_empty_ratio": float(args.target_empty_ratio),
        "seed": int(args.seed),
        "image_root_mode": image_mode,
        "splits": {},
    }

    splits = [str(x) for x in args.splits]
    for split in splits:
        summary["splits"][split] = build_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            grid_size=int(args.grid_size),
            target_empty_ratio=float(args.target_empty_ratio),
            rng=rng,
            boundary_tol_px=float(args.boundary_tol_px),
            resample_step_px=float(args.resample_step_px),
            reuse_system_prompt=bool(args.use_system_prompt_from_source),
        )

    dataset_info = build_sharegpt_dataset_info(output_root=output_root, splits=splits)
    (output_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
