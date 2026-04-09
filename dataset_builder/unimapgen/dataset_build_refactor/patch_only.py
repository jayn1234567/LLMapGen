"""Patch-only 阶段 helper。

这个文件负责把原始全局标注转换成 patch 级目标：
- 过滤原始类别
- 把全局线裁到 patch
- 转成 patch-local 坐标
- 组装 patch-only ShareGPT 样本
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .common import make_sharegpt_record
from .geometry import (
    canonicalize_line_direction,
    clamp_points,
    clip_polyline_to_rect,
    dedup_points,
    point_boundary_side,
    resample_polyline,
    simplify_for_json,
    sort_lines,
)
from .viz import save_patch_lines_visualization


PATCH_ONLY_PROMPT_TEMPLATE = """<image>
Please construct the complete road map in the current satellite patch."""

PATCH_ONLY_SYSTEM_PROMPT = (
    "You are a road-map reconstruction assistant for satellite-image patches.\n"
    "Predict the complete road map in the current patch from the satellite image.\n"
    "Return only valid JSON in the required schema.\n"
    "Do not output markdown fences or extra explanation.\n"
    "Keep all coordinates in the patch-local coordinate system."
)


def normalize_opensatmap_category(name: str) -> str:
    """把 OpenSatMap 原始类别名规范成统一写法。"""
    value = str(name).strip().lower()
    if value == "lane line":
        return "lane_line"
    if value == "virtual line":
        return "virtual_line"
    if value == "curb":
        return "curb"
    return value.replace(" ", "_")


def collect_global_lines(ann: Dict, accepted_categories: Sequence[str]) -> List[np.ndarray]:
    """从原始 annotation 里提取需要保留的全局折线。"""
    cat_set = set(str(x) for x in accepted_categories)
    out: List[np.ndarray] = []
    for rec in ann.get("lines", []):
        cat = normalize_opensatmap_category(rec.get("category", ""))
        if cat_set and cat not in cat_set:
            continue
        arr = np.asarray(rec.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        out.append(dedup_points(arr))
    return out


def build_full_patch_segments_global(
    global_lines: Sequence[np.ndarray],
    patch_rect_global: Tuple[float, float, float, float],
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    """把全局折线先裁到 patch 对应的全局窗口中。"""
    out: List[Dict] = []
    for line in global_lines:
        pieces = clip_polyline_to_rect(line, patch_rect_global)
        for piece in pieces:
            piece = resample_polyline(piece, step_px=resample_step_px)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], patch_rect_global, boundary_tol_px)
            end_side = point_boundary_side(piece[-1], patch_rect_global, boundary_tol_px)
            start_type = "cut" if start_side is not None else "start"
            end_type = "cut" if end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            out.append(
                {
                    "points_global": piece,
                    "start_type": start_type,
                    "end_type": end_type,
                }
            )
    return out


def build_full_patch_target_lines(
    full_segments_global: Sequence[Dict],
    patch: Dict,
    output_category: str,
) -> List[Dict]:
    """把裁好的全局片段转成 patch-local `target_lines`。"""
    crop_box = patch["crop_box"]
    patch_size = int(crop_box["x_max"] - crop_box["x_min"])
    out: List[Dict] = []
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    for segment in full_segments_global:
        local = np.asarray(segment["points_global"], dtype=np.float32) - offset
        local = clamp_points(local, patch_size=patch_size)
        if local.ndim != 2 or local.shape[0] < 2:
            continue
        points_json = simplify_for_json(local, patch_size=patch_size)
        if len(points_json) < 2:
            continue
        out.append(
            {
                "category": output_category,
                "start_type": str(segment["start_type"]),
                "end_type": str(segment["end_type"]),
                "points": points_json,
            }
        )
    return sort_lines([item for item in out if len(item.get("points", [])) >= 2])


def make_patch_only_record(
    *,
    sample_id: str,
    image_rel_path: str,
    target_lines: Sequence[Dict],
    system_prompt: str,
) -> Dict:
    """组装一条 patch-only ShareGPT 样本。"""
    return make_sharegpt_record(
        sample_id=sample_id,
        image_rel_path=image_rel_path,
        user_text=PATCH_ONLY_PROMPT_TEMPLATE,
        assistant_payload={"lines": list(target_lines)},
        system_prompt=system_prompt,
    )


def save_patch_only_visualization(*, patch_image: Image.Image, target_lines: Sequence[Dict], out_path) -> None:
    """保存 patch-only 的可视化图。

    这在当前主链里已不再被入口脚本调用，但 helper 仍保留。
    """
    save_patch_lines_visualization(image=patch_image, lines=target_lines, out_path=out_path)
