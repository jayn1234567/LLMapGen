"""Export patch-only ShareGPT rows from a raw family manifest.

This is the first script in the chain that actually crops image patches and
serializes patch-local road targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `common` 里是通用 IO / 文本解析 helper。
# - `load_json`: 读取原始 ann json
# - `load_jsonl`: 读取 family manifest
# - `resolve_optional_text`: 统一处理系统提示词的“内联文本 / 文件 / 默认值”优先级
# - `write_jsonl`: 写出 train/val 和 meta jsonl
from unimapgen.dataset_build_refactor.common import ensure_dir, load_json, load_jsonl, resolve_optional_text, write_jsonl
# `patch_only` 里是 patch 级目标构建 helper。
# - `collect_global_lines`: 从原始 ann 中筛出需要保留的全局线
# - `build_full_patch_segments_global`: 先把全局线裁到 patch 对应的全局窗口
# - `build_full_patch_target_lines`: 再把坐标改成 patch-local 并写成 target_lines
# - `make_patch_only_record`: 组装 ShareGPT 样本
from unimapgen.dataset_build_refactor.patch_only import (
    PATCH_ONLY_SYSTEM_PROMPT,
    build_full_patch_segments_global,
    build_full_patch_target_lines,
    collect_global_lines,
    make_patch_only_record,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for patch-only export.

    The main knobs are:
    - raw sources: `--ann-json`, `--family-manifest`
    - output location: `--output-root`
    - geometry/target filtering: `--accepted-categories`, `--output-category`
    - polyline processing: `--resample-step-px`, `--boundary-tol-px`
    - prompt controls: system-prompt args
    """
    parser = argparse.ArgumentParser(
        description="Refactored v2: export LLaMAFactory patch-only SFT data from raw OpenSatMap family manifests."
    )
    parser.add_argument("--ann-json", type=str, required=True, help="Raw annotation json. Keys should match source image file names referenced by the family manifest.")
    parser.add_argument("--family-manifest", type=str, required=True, help="Family manifest jsonl produced by build_opensatmap_paper16_family_manifest.py.")
    parser.add_argument("--output-root", type=str, required=True, help="Output dataset root. The script writes jsonl files, meta files, and cropped patch images under this root.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Manifest splits to export. Only family rows whose `split` is in this list are processed.")
    parser.add_argument("--accepted-categories", type=str, nargs="+", default=["lane_line", "virtual_line", "curb"], help="Raw OpenSatMap categories to keep before clipping lines into patches.")
    parser.add_argument("--output-category", type=str, default="road", help="Unified category name written into exported `target_lines`.")
    parser.add_argument("--resample-step-px", type=float, default=12.0, help="Equal-distance resampling step for clipped patch polylines. Smaller means denser points.")
    parser.add_argument("--boundary-tol-px", type=float, default=2.5, help="Tolerance used to decide whether a patch endpoint touches the crop boundary and should be marked as `cut`.")
    parser.add_argument("--use-system-prompt", action="store_true", help="Use the built-in patch-only system prompt unless an explicit override is provided.")
    parser.add_argument("--system-prompt", type=str, default="", help="Inline system prompt override. Takes priority over --use-system-prompt.")
    parser.add_argument("--system-prompt-file", type=str, default="", help="Path to a text file containing the system prompt. Highest priority among prompt options.")
    return parser.parse_args()


def build_patch_only_meta_row(
    *,
    sample_id: str,
    split: str,
    family: Dict,
    patch: Dict,
    image_rel_path: str,
    target_lines: Sequence[Dict],
    resample_step_px: float,
    system_prompt: str,
) -> Dict:
    """Build one `meta_*.jsonl` row for a patch-only training sample."""
    return {
        "id": sample_id,
        "split": split,
        "family_id": family["family_id"],
        "source_image": family["source_image"],
        "patch_id": int(patch["patch_id"]),
        "row": int(patch["row"]),
        "col": int(patch["col"]),
        "scan_index": int(patch["patch_id"]),
        "image": image_rel_path,
        "crop_box": patch["crop_box"],
        "num_target_lines": len(target_lines),
        "target_mode": "full_patch_map",
        "coord_system": "patch_local_896",
        "serialization_mode": "paper_structured",
        "line_direction_mode": "canonical_cut_then_origin",
        "line_sort_mode": "first_point_distance_to_patch_origin",
        "resample_mode": "equal_distance",
        "resample_step_px": float(resample_step_px),
        "has_system_prompt": bool(str(system_prompt).strip()),
        "target_lines": list(target_lines),
    }


def export_split(
    *,
    split: str,
    families: Sequence[Dict],
    annotations: Dict,
    output_root: Path,
    accepted_categories: Sequence[str],
    output_category: str,
    resample_step_px: float,
    boundary_tol_px: float,
    system_prompt: str,
) -> Dict[str, int]:
    """Export one split from family records into patch-only jsonl files.

    For each family patch, this function crops the image, clips global lines
    into the patch, writes the ShareGPT row, and records the patch-local meta.
    """
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    family_count = 0

    for family in families:
        if str(family.get("split")) != split:
            continue

        image_name = str(family["source_image"])
        ann = annotations.get(image_name)
        if not isinstance(ann, dict):
            continue
        family_count += 1

        global_lines = collect_global_lines(ann=ann, accepted_categories=accepted_categories)
        with Image.open(str(family["source_image_path"])) as raw_img:
            raw_image = raw_img.convert("RGB")
            patches = sorted(list(family["patches"]), key=lambda item: int(item["patch_id"]))
            for patch in patches:
                patch_id = int(patch["patch_id"])
                crop_box = patch["crop_box"]
                patch_rect_global = (
                    float(crop_box["x_min"]),
                    float(crop_box["y_min"]),
                    float(crop_box["x_max"]),
                    float(crop_box["y_max"]),
                )
                patch_image = raw_image.crop(
                    (
                        int(crop_box["x_min"]),
                        int(crop_box["y_min"]),
                        int(crop_box["x_max"]),
                        int(crop_box["y_max"]),
                    )
                )

                full_segments_global = build_full_patch_segments_global(
                    global_lines=global_lines,
                    patch_rect_global=patch_rect_global,
                    resample_step_px=resample_step_px,
                    boundary_tol_px=boundary_tol_px,
                )
                target_lines = build_full_patch_target_lines(
                    full_segments_global=full_segments_global,
                    patch=patch,
                    output_category=output_category,
                )

                image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:02d}.png"
                out_image = output_root / image_rel
                ensure_dir(out_image.parent)
                patch_image.save(out_image)

                sample_id = f"{family['family_id']}_p{patch_id:02d}"
                rows.append(
                    make_patch_only_record(
                        sample_id=sample_id,
                        image_rel_path=image_rel.as_posix(),
                        target_lines=target_lines,
                        system_prompt=system_prompt,
                    )
                )
                meta_rows.append(
                    build_patch_only_meta_row(
                        sample_id=sample_id,
                        split=split,
                        family=family,
                        patch=patch,
                        image_rel_path=image_rel.as_posix(),
                        target_lines=target_lines,
                        resample_step_px=resample_step_px,
                        system_prompt=system_prompt,
                    )
                )

    count_main = write_jsonl(output_root / f"{split}.jsonl", rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", meta_rows)
    return {
        "families": family_count,
        "samples": count_main,
        "meta_samples": count_meta,
    }


def main() -> None:
    """Load raw sources, export requested splits, and save dataset metadata."""
    args = parse_args()
    # `load_json` 读原始 annotation json；键通常是原始图片文件名。
    annotations = load_json(Path(args.ann_json).resolve())
    # `load_jsonl` 读上一阶段产出的 family manifest，每一行对应一个 family。
    families = load_jsonl(Path(args.family_manifest).resolve())
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    # `resolve_optional_text` 统一处理 system prompt 的来源优先级：
    # 文件 > 命令行内联文本 > 内置默认 prompt（如果启用）。
    system_prompt = resolve_optional_text(
        inline_text=str(args.system_prompt),
        file_path=str(args.system_prompt_file),
        fallback=PATCH_ONLY_SYSTEM_PROMPT if bool(args.use_system_prompt) else "",
    )

    summary: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        summary[str(split)] = export_split(
            split=str(split),
            families=families,
            annotations=annotations,
            output_root=output_root,
            accepted_categories=[str(x) for x in args.accepted_categories],
            output_category=str(args.output_category),
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
            system_prompt=system_prompt,
        )

    dataset_info = {
        "dataset_name": "unimapgen_paper16_patch_only_sft",
        "source_ann_json": str(Path(args.ann_json).resolve()),
        "source_family_manifest": str(Path(args.family_manifest).resolve()),
        "target_mode": "full_patch_map",
        "coord_system": "patch_local_896",
        "serialization_mode": "paper_structured",
        "line_direction_mode": "canonical_cut_then_origin",
        "line_sort_mode": "first_point_distance_to_patch_origin",
        "resample_mode": "equal_distance",
        "resample_step_px": float(args.resample_step_px),
        "use_system_prompt": bool(system_prompt),
        "system_prompt": system_prompt,
        "summary": summary,
    }
    (output_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for split, info in summary.items():
        print(f"[{split}] families={info['families']} samples={info['samples']} meta={info['meta_samples']}")
    print(f"Saved dataset to {output_root}")


if __name__ == "__main__":
    main()
