"""Build the Stage B dataset from an existing fixed16 Stage A root.

Despite the legacy filename, the current behavior is trace-based: each target
box receives incoming short traces extracted from left/top neighbor boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `common` 里是通用 IO / prompt helper。
# - `load_jsonl`: 读取 Stage A fixed16 的主 jsonl 和 meta jsonl
# - `extract_message_content`: 在需要时复用源 system prompt
# - `resolve_optional_text`: 统一处理 system prompt override 逻辑
# - `link_or_copy_images`: 在输出根下暴露 images/ 树
from unimapgen.dataset_build_refactor.common import (
    build_sharegpt_dataset_info,
    ensure_dir,
    extract_message_content,
    link_or_copy_images,
    load_jsonl,
    make_sharegpt_record,
    resolve_optional_text,
    write_jsonl,
)
# `stageb` 里是 Stage B 特有 helper。
# - `extract_state_points`: 从左邻 / 上邻 box 的 target_lines 提取 incoming trace
# - `format_stageb_trace_prompt`: 把 state_points 序列化进当前 prompt
# - `safe_int`: 防御式整数转换，避免 meta 中个别字段为空时报错
from unimapgen.dataset_build_refactor.stageb import (
    STAGEB_TRACE_PROMPT_TEMPLATE,
    extract_state_points,
    format_stageb_trace_prompt,
    safe_int,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage B dataset building.

    Main parameter groups:
    - roots/splits: `--input-root`, `--output-root`, `--splits`
    - neighbor-trace extraction: `--grid-size`, `--boundary-tol-px`,
      `--trace-points-per-hint`
    - prompt control: `--use-system-prompt-from-source`, `--system-prompt*`
    - output exposure: `--image-root-mode`
    """
    parser = argparse.ArgumentParser(
        description="Refactored v2: build a Stage B fixed16 dataset with GT neighbor traces serialized as incoming points."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Existing fixed16 Stage A dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output Stage B dataset root.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Splits to process.")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size per side. 4 means 16 target boxes.")
    parser.add_argument("--boundary-tol-px", type=float, default=2.5, help="Tolerance for identifying left/top cut points.")
    parser.add_argument("--trace-points-per-hint", type=int, default=3, help="Maximum ordered points kept for each incoming trace hint.")
    parser.add_argument("--use-system-prompt-from-source", action="store_true", help="Reuse the source system prompt if present and no explicit system prompt is given.")
    parser.add_argument("--system-prompt", type=str, default="", help="Explicit system prompt override.")
    parser.add_argument("--system-prompt-file", type=str, default="", help="Path to a text file containing the system prompt.")
    parser.add_argument("--image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"], help="How to expose images under the output root.")
    return parser.parse_args()


def build_split(
    *,
    split: str,
    input_root: Path,
    output_root: Path,
    default_grid_size: int,
    boundary_tol_px: float,
    trace_points_per_hint: int,
    explicit_system_prompt: str,
    reuse_system_prompt: bool,
) -> Dict[str, object]:
    """Build one Stage B split from a Stage A fixed16 dataset.

    The function groups rows by `source_id`, reconstructs neighbor ownership in
    box order, extracts left/top incoming traces, and writes new ShareGPT rows
    whose assistant target stays equal to the current box GT.
    """
    split_jsonl = input_root / f"{split}.jsonl"
    split_meta_jsonl = input_root / f"meta_{split}.jsonl"
    if not split_jsonl.exists() or not split_meta_jsonl.exists():
        return {
            "missing_split": True,
            "split_jsonl": str(split_jsonl),
            "split_meta_jsonl": str(split_meta_jsonl),
        }

    # `load_jsonl` 读取 Stage A fixed16 的训练样本和元信息。
    rows = load_jsonl(split_jsonl)
    meta_rows = load_jsonl(split_meta_jsonl)
    row_by_id = {str(row.get("id")): row for row in rows}

    source_group_meta: Dict[str, Dict[int, Dict]] = {}
    for src_meta in meta_rows:
        source_group_id = str(src_meta.get("source_id") or src_meta.get("id"))
        grid_size = safe_int(src_meta.get("grid_size", default_grid_size), default=default_grid_size)
        grid_row = safe_int(src_meta.get("grid_row", -1), default=-1)
        grid_col = safe_int(src_meta.get("grid_col", -1), default=-1)
        if grid_row < 0 or grid_col < 0:
            continue
        subpatch_id = int(grid_row) * int(grid_size) + int(grid_col)
        source_group_meta.setdefault(source_group_id, {})[int(subpatch_id)] = src_meta

    out_rows: List[Dict] = []
    out_meta: List[Dict] = []
    with_state = 0
    without_state = 0
    total_state_traces = 0
    total_state_trace_points = 0

    for src_meta in meta_rows:
        row_id = str(src_meta.get("id"))
        src_row = row_by_id.get(row_id)
        if src_row is None:
            continue

        source_group_id = str(src_meta.get("source_id") or row_id)

        crop_box = src_meta.get("crop_box", {})
        patch_size = safe_int(crop_box.get("x_max", 0)) - safe_int(crop_box.get("x_min", 0))
        if patch_size <= 1:
            continue

        target_box = src_meta.get("target_box", {})
        if not isinstance(target_box, dict) or not target_box:
            continue

        target_lines = list(src_meta.get("target_lines", []))
        grid_size = safe_int(src_meta.get("grid_size", default_grid_size), default=default_grid_size)
        grid_row = safe_int(src_meta.get("grid_row", -1), default=-1)
        grid_col = safe_int(src_meta.get("grid_col", -1), default=-1)
        if grid_row < 0 or grid_col < 0:
            continue

        image_rel_path = str(src_meta.get("image") or src_row.get("images", [""])[0])
        # system prompt 的优先级是：
        # 显式 override > 从源样本复用 > 空字符串。
        system_prompt = (
            explicit_system_prompt
            if explicit_system_prompt
            else (extract_message_content(src_row, "system") if reuse_system_prompt else "")
        )

        # `extract_state_points` 是 Stage B 的核心：
        # 它会根据当前 box 的 grid_row / grid_col，只从左邻和上邻提取 incoming trace。
        state_points = extract_state_points(
            source_group_meta=source_group_meta.get(source_group_id, {}),
            grid_size=grid_size,
            grid_row=grid_row,
            grid_col=grid_col,
            patch_size=patch_size,
            boundary_tol_px=boundary_tol_px,
            trace_points_per_hint=trace_points_per_hint,
        )
        prompt_text = format_stageb_trace_prompt(target_box=target_box, state_points=state_points)

        row = make_sharegpt_record(
            sample_id=row_id,
            image_rel_path=image_rel_path,
            user_text=prompt_text,
            assistant_payload={"lines": list(target_lines)},
            system_prompt=system_prompt,
        )

        subpatch_id = int(grid_row) * int(grid_size) + int(grid_col)
        state_source_patch_ids = sorted({int(item["source_patch"]) for item in state_points})
        meta = {
            "id": row_id,
            "source_stagea_row_id": row_id,
            "source_id": src_meta.get("source_id"),
            "split": split,
            "family_id": src_meta.get("family_id"),
            "source_image": src_meta.get("source_image"),
            "patch_id": src_meta.get("patch_id"),
            "row": src_meta.get("row"),
            "col": src_meta.get("col"),
            "scan_index": src_meta.get("scan_index"),
            "image": image_rel_path,
            "crop_box": crop_box,
            "target_mode": str(src_meta.get("target_mode", "fixed_grid_target_box_map")),
            "state_mode": "gt_neighbor_handoff_trace_points",
            "coord_system": src_meta.get("coord_system", "patch_local_896"),
            "serialization_mode": src_meta.get("serialization_mode", "paper_structured"),
            "line_direction_mode": src_meta.get("line_direction_mode", "canonical_cut_then_origin"),
            "line_sort_mode": src_meta.get("line_sort_mode", "first_point_distance_to_patch_origin"),
            "has_system_prompt": bool(system_prompt.strip()),
            "grid_size": int(grid_size),
            "grid_row": int(grid_row),
            "grid_col": int(grid_col),
            "subpatch_id": int(subpatch_id),
            "history_subpatch_ids": list(range(subpatch_id)),
            "state_source_patch_ids": state_source_patch_ids,
            "num_state_traces": len(state_points),
            "num_state_trace_points": int(sum(len(item.get("points", [])) for item in state_points)),
            "trace_points_per_hint": int(trace_points_per_hint),
            "target_box": {
                "x_min": safe_int(target_box["x_min"]),
                "y_min": safe_int(target_box["y_min"]),
                "x_max": safe_int(target_box["x_max"]),
                "y_max": safe_int(target_box["y_max"]),
            },
            "target_box_area": safe_int(src_meta.get("target_box_area", 0))
            or int((safe_int(target_box["x_max"]) - safe_int(target_box["x_min"]) + 1) * (safe_int(target_box["y_max"]) - safe_int(target_box["y_min"]) + 1)),
            "anchor_source": src_meta.get("anchor_source"),
            "anchor_start_xy": src_meta.get("anchor_start_xy"),
            "anchor_end_xy": src_meta.get("anchor_end_xy"),
            "anchor_piece_points": src_meta.get("anchor_piece_points"),
            "num_target_lines": len(target_lines),
            "num_target_points": int(sum(len(x.get("points", [])) for x in target_lines)),
            "prompt_text": prompt_text,
            "state_points": state_points,
            "target_lines": list(target_lines),
        }

        out_rows.append(row)
        out_meta.append(meta)
        total_state_traces += len(state_points)
        total_state_trace_points += int(sum(len(item.get("points", [])) for item in state_points))
        if state_points:
            with_state += 1
        else:
            without_state += 1

    count_rows = write_jsonl(output_root / f"{split}.jsonl", out_rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", out_meta)
    return {
        "source_groups": len(source_group_meta),
        "used_stagea_rows": len(out_rows),
        "written_rows": count_rows,
        "written_meta_rows": count_meta,
        "samples_with_state": int(with_state),
        "samples_without_state": int(without_state),
        "total_state_traces": int(total_state_traces),
        "avg_state_traces_per_sample": (float(total_state_traces) / float(len(out_rows)) if out_rows else 0.0),
        "total_state_trace_points": int(total_state_trace_points),
        "avg_state_trace_points_per_sample": (float(total_state_trace_points) / float(len(out_rows)) if out_rows else 0.0),
    }


def main() -> None:
    """Validate args, build the Stage B root, and save dataset summaries."""
    args = parse_args()
    if int(args.grid_size) <= 0:
        raise ValueError("--grid-size must be positive.")
    if int(args.trace_points_per_hint) < 2:
        raise ValueError("--trace-points-per-hint must be >= 2.")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    # `resolve_optional_text` 统一处理 system prompt 的 override 逻辑。
    explicit_system_prompt = resolve_optional_text(
        inline_text=str(args.system_prompt),
        file_path=str(args.system_prompt_file),
    )
    image_mode = link_or_copy_images(input_root=input_root, output_root=output_root, mode=str(args.image_root_mode))

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "grid_size": int(args.grid_size),
        "num_boxes_per_patch": int(args.grid_size) * int(args.grid_size),
        "state_mode": "gt_neighbor_handoff_trace_points",
        "state_neighbors": ["left", "top"],
        "trace_points_per_hint": int(args.trace_points_per_hint),
        "boundary_tol_px": float(args.boundary_tol_px),
        "prompt_template": STAGEB_TRACE_PROMPT_TEMPLATE,
        "image_root_mode": image_mode,
        "splits": {},
    }

    splits = [str(x) for x in args.splits]
    for split in splits:
        summary["splits"][split] = build_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            default_grid_size=int(args.grid_size),
            boundary_tol_px=float(args.boundary_tol_px),
            trace_points_per_hint=int(args.trace_points_per_hint),
            explicit_system_prompt=explicit_system_prompt,
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
