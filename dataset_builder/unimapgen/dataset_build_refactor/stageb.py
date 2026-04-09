"""Stage B 阶段 helper。

这个文件负责从 Stage A fixed16 的相邻 box 中提取 handoff trace：
- 找到左邻 / 上邻
- 检查 cut 端点是否真的打到共享边界
- 截取短 trace
- 序列化成 Stage B prompt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .geometry import clamp_points
from .viz import draw_endpoint, draw_polyline


STAGEB_TRACE_PROMPT_TEMPLATE = """<image>
Please construct the target road map in the satellite image inside target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}],
Incoming trace points: {trace_points_json}
Keep all coordinates in the patch-local coordinate system."""


def safe_int(value, default: int = 0) -> int:
    """防御式整数转换，失败时回退到默认值。"""
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def simplify_points(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    """把点序列裁到 patch 范围内，并转成 json 友好的整数点。"""
    arr = clamp_points(np.asarray(points_xy, dtype=np.float32), patch_size=patch_size)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    rounded = np.rint(arr).astype(np.int32)
    out: List[List[int]] = []
    for point_xy in rounded.tolist():
        cur = [int(point_xy[0]), int(point_xy[1])]
        if not out or out[-1] != cur:
            out.append(cur)
    return out


def point_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    """给一个点生成用于排序的 key。"""
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def extract_trace_points_for_endpoint(
    pts: np.ndarray,
    endpoint_idx: int,
    patch_size: int,
    max_points: int,
) -> List[List[int]]:
    """从某个端点向线内部截取一小段 trace 点。"""
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return []
    keep = max(2, int(max_points))
    trace = arr[:keep] if int(endpoint_idx) == 0 else arr[::-1][:keep]
    return simplify_points(trace, patch_size=patch_size)


def endpoint_matches_neighbor_boundary(
    point_xy: np.ndarray,
    neighbor_box: Dict[str, int],
    side_to_current: str,
    tol_px: float,
) -> bool:
    """判断邻居 box 的 cut 端点是否落在与当前 box 相接的那条边上。"""
    x = float(point_xy[0])
    y = float(point_xy[1])
    if str(side_to_current) == "left":
        return abs(x - float(neighbor_box["x_max"])) <= float(tol_px)
    if str(side_to_current) == "top":
        return abs(y - float(neighbor_box["y_max"])) <= float(tol_px)
    return False


def extract_state_points_from_neighbor(
    *,
    neighbor_meta: Dict,
    side_to_current: str,
    patch_size: int,
    boundary_tol_px: float,
    trace_points_per_hint: int,
) -> List[Dict]:
    """从一个邻居 box 的 `target_lines` 中提取可用的 incoming trace。"""
    neighbor_box = dict(neighbor_meta.get("target_box", {}) or {})
    if not neighbor_box:
        return []
    neighbor_subpatch_id = safe_int(neighbor_meta.get("subpatch_id", -1), default=-1)
    out: List[Dict] = []
    for line_index, line in enumerate(neighbor_meta.get("target_lines", [])):
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        endpoints = [
            (0, str(line.get("start_type", "start")).strip().lower()),
            (-1, str(line.get("end_type", "end")).strip().lower()),
        ]
        for endpoint_idx, endpoint_type in endpoints:
            if endpoint_type != "cut":
                continue
            point_xy = pts[0] if int(endpoint_idx) == 0 else pts[-1]
            if not endpoint_matches_neighbor_boundary(
                point_xy=point_xy,
                neighbor_box=neighbor_box,
                side_to_current=side_to_current,
                tol_px=float(boundary_tol_px),
            ):
                continue
            trace_points = extract_trace_points_for_endpoint(
                pts=pts,
                endpoint_idx=int(endpoint_idx),
                patch_size=patch_size,
                max_points=int(trace_points_per_hint),
            )
            if len(trace_points) < 2:
                continue
            out.append(
                {
                    "source_patch": int(neighbor_subpatch_id),
                    "points": trace_points,
                    "boundary_side": str(side_to_current),
                    "line_index": int(line_index),
                }
            )
    return out


def sort_state_points(points: List[Dict]) -> List[Dict]:
    """对多条 state trace 做稳定排序。"""
    return sorted(
        points,
        key=lambda item: (
            int(item.get("source_patch", 1_000_000_000)),
            *point_sort_key((item.get("points") or [[1e9, 1e9]])[0]),
        ),
    )


def extract_state_points(
    *,
    source_group_meta: Dict[int, Dict],
    grid_size: int,
    grid_row: int,
    grid_col: int,
    patch_size: int,
    boundary_tol_px: float,
    trace_points_per_hint: int,
) -> List[Dict]:
    """为当前 box 汇总来自左邻 / 上邻的全部 state trace。"""
    subpatch_id = int(grid_row) * int(grid_size) + int(grid_col)
    left_neighbor = subpatch_id - 1 if int(grid_col) > 0 else None
    top_neighbor = subpatch_id - int(grid_size) if int(grid_row) > 0 else None

    state_points: List[Dict] = []
    if left_neighbor is not None:
        state_points.extend(
            extract_state_points_from_neighbor(
                neighbor_meta=source_group_meta.get(int(left_neighbor), {}),
                side_to_current="left",
                patch_size=patch_size,
                boundary_tol_px=float(boundary_tol_px),
                trace_points_per_hint=int(trace_points_per_hint),
            )
        )
    if top_neighbor is not None:
        state_points.extend(
            extract_state_points_from_neighbor(
                neighbor_meta=source_group_meta.get(int(top_neighbor), {}),
                side_to_current="top",
                patch_size=patch_size,
                boundary_tol_px=float(boundary_tol_px),
                trace_points_per_hint=int(trace_points_per_hint),
            )
        )

    seen = set()
    deduped: List[Dict] = []
    for item in sort_state_points(state_points):
        key = (
            int(item["source_patch"]),
            tuple((int(pt[0]), int(pt[1])) for pt in item["points"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_stageb_trace_prompt(*, target_box: Dict[str, int], state_points: Sequence[Dict]) -> str:
    """把 state trace 序列化进 Stage B prompt 文本。"""
    trace_points_json = json.dumps(
        [{"points": [list(point) for point in item["points"]]} for item in state_points],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return STAGEB_TRACE_PROMPT_TEMPLATE.format(
        box_x_min=int(target_box["x_min"]),
        box_y_min=int(target_box["y_min"]),
        box_x_max=int(target_box["x_max"]),
        box_y_max=int(target_box["y_max"]),
        trace_points_json=trace_points_json,
    )


def save_stageb_visualization(
    *,
    patch_image: Image.Image,
    target_lines: Sequence[Dict],
    target_box: Dict[str, int],
    state_points: Sequence[Dict],
    out_path: Path,
) -> None:
    """保存 Stage B 样本可视化图。

    这在当前主链里已不再被入口脚本调用，但 helper 仍保留。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = patch_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    patch_w, patch_h = image.size
    draw.rectangle((0, 0, patch_w - 1, patch_h - 1), outline=(255, 0, 180), width=2)
    draw.rectangle(
        (
            int(target_box["x_min"]),
            int(target_box["y_min"]),
            int(target_box["x_max"]),
            int(target_box["y_max"]),
        ),
        outline=(255, 210, 0),
        width=3,
    )
    for line in target_lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), 3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), 3)
    for item in state_points:
        trace_points = [list(map(int, point[:2])) for point in item.get("points", [])]
        if len(trace_points) < 2:
            continue
        color = (255, 140, 0) if str(item.get("boundary_side")) == "left" else (120, 255, 80)
        draw_polyline(draw, trace_points, color=color, width=3)
        draw_endpoint(draw, trace_points[0], color, 4)
        draw_endpoint(draw, trace_points[-1], color, 3)
    image.save(out_path)
