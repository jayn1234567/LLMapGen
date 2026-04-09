"""几何 helper。

这个文件放的是折线处理相关的基础函数：
- 去重点
- 坐标裁剪
- 重采样
- 矩形裁剪
- 边界判定
- 方向规范化和排序
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def dedup_points(points: Sequence[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    """去掉折线上相邻的重复点或极近点。"""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - out[-1])) > float(eps):
            out.append(arr[idx])
    return np.asarray(out, dtype=np.float32)


def clamp_points(points_xy: np.ndarray, patch_size: int) -> np.ndarray:
    """把点坐标裁到 patch 合法范围内。"""
    arr = np.asarray(points_xy, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    max_coord = float(max(1, int(patch_size) - 1))
    arr[:, 0] = np.clip(arr[:, 0], 0.0, max_coord)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, max_coord)
    return arr


def simplify_for_json(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    """把浮点坐标整理成适合写入 json 的整数点序列。"""
    arr = clamp_points(points_xy, patch_size=patch_size)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    arr = np.rint(arr).astype(np.int32)
    arr = dedup_points(arr.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in arr.tolist()]


def resample_polyline(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    """按固定距离对折线重采样。"""
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return pts[:1]
    step = max(float(step_px), 1.0)
    count = max(2, int(math.floor(total / step)) + 1)
    if max_points is not None:
        count = min(count, int(max_points))
    target = np.linspace(0.0, total, count, dtype=np.float32)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    sampled: List[np.ndarray] = []
    for dist in target:
        idx = int(np.searchsorted(cum, dist, side="right") - 1)
        idx = min(max(idx, 0), len(seg) - 1)
        t0 = float(cum[idx])
        t1 = float(cum[idx + 1])
        ratio = 0.0 if t1 <= t0 else (float(dist) - t0) / (t1 - t0)
        sampled.append(pts[idx] * (1.0 - ratio) + pts[idx + 1] * ratio)
    return dedup_points(sampled)


def line_length_xy(points_xy: Sequence[Sequence[float]]) -> float:
    """计算一条折线的总长度。"""
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum())


def point_in_rect(point_xy: np.ndarray, rect: Tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    """判断一个点是否落在给定矩形内。"""
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps)


def clip_segment_liang_barsky(
    p0: np.ndarray,
    p1: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """用 Liang-Barsky 算法裁剪单条线段到矩形内。"""
    x_min, y_min, x_max, y_max = rect
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    p = [-dx, dx, -dy, dy]
    q = [float(p0[0] - x_min), float(x_max - p0[0]), float(p0[1] - y_min), float(y_max - p0[1])]
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-8:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t
    c0 = np.asarray([p0[0] + u1 * dx, p0[1] + u1 * dy], dtype=np.float32)
    c1 = np.asarray([p0[0] + u2 * dx, p0[1] + u2 * dy], dtype=np.float32)
    return c0, c1


def clip_polyline_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    """把一条折线裁到矩形里，可能返回多段折线片段。"""
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return []
    pieces: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for idx in range(pts.shape[0] - 1):
        clipped = clip_segment_liang_barsky(pts[idx], pts[idx + 1], rect)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
            continue
        c0, c1 = clipped
        if not current:
            current = [c0, c1]
        elif float(np.linalg.norm(current[-1] - c0)) <= 1e-3:
            current.append(c1)
        else:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = [c0, c1]
        if not point_in_rect(pts[idx + 1], rect):
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
    if len(current) >= 2:
        pieces.append(dedup_points(current))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
    """判断点是否贴近矩形某一条边，并返回边名。"""
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    if abs(x - x_min) <= tol_px:
        return "left"
    if abs(y - y_min) <= tol_px:
        return "top"
    if abs(x - x_max) <= tol_px:
        return "right"
    if abs(y - y_max) <= tol_px:
        return "bottom"
    return None


def point_origin_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    """给一个点生成“离原点远近优先”的排序 key。"""
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def canonicalize_line_direction(
    points_xy: np.ndarray,
    start_type: str,
    end_type: str,
) -> Tuple[np.ndarray, str, str]:
    """统一折线方向。

    优先让 `cut -> 非cut` 的方向保持一致；
    如果两端类型相同，则按离原点更近的一端作为起点。
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts, start_type, end_type
    start_is_cut = str(start_type) == "cut"
    end_is_cut = str(end_type) == "cut"
    reverse = False
    if start_is_cut and not end_is_cut:
        reverse = False
    elif end_is_cut and not start_is_cut:
        reverse = True
    elif point_origin_sort_key(pts[-1]) < point_origin_sort_key(pts[0]):
        reverse = True
    if not reverse:
        return pts, start_type, end_type
    return pts[::-1].copy(), end_type, start_type


def sort_lines(lines: List[Dict]) -> List[Dict]:
    """按首点的原点距离顺序对多条线排序。"""
    return sorted(
        lines,
        key=lambda item: (*point_origin_sort_key(item.get("points", [[1e9, 1e9]])[0]),),
    )
