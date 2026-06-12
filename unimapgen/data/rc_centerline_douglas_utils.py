from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dedup_points(points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - out[-1])) > float(eps):
            out.append(arr[idx])
    return np.asarray(out, dtype=np.float32)


def point_origin_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
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


def canonicalize_line_direction(
    points_xy: np.ndarray,
    start_type: str,
    end_type: str,
) -> Tuple[np.ndarray, str, str]:
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


def sort_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(lines, key=lambda item: point_origin_sort_key(item.get("points", [[1e9, 1e9]])[0]))


def simplify_for_json(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return []
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(patch_size - 1))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(patch_size - 1))
    arr = np.rint(arr).astype(np.int32)
    arr = dedup_points(arr.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in arr.tolist()]


def scale_points_inclusive(points_xy: np.ndarray, source_patch_size: int, target_patch_size: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    if int(source_patch_size) <= 1 or int(target_patch_size) <= 1:
        return pts
    scale = float(target_patch_size - 1) / float(source_patch_size - 1)
    pts[:, 0] *= scale
    pts[:, 1] *= scale
    return pts


def point_to_segment_distance(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    seg = end - start
    seg_norm_sq = float(np.dot(seg, seg))
    if seg_norm_sq <= 1e-8:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, seg) / seg_norm_sq)
    t = max(0.0, min(1.0, t))
    proj = start + (t * seg)
    return float(np.linalg.norm(point - proj))


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def heading_at_start(points_xy: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return (0.0, 0.0)
    vec = pts[1] - pts[0]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return (0.0, 0.0)
    return (float(vec[0] / norm), float(vec[1] / norm))


def heading_at_end(points_xy: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return (0.0, 0.0)
    vec = pts[-1] - pts[-2]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return (0.0, 0.0)
    return (float(vec[0] / norm), float(vec[1] / norm))


def angle_between_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    na = float(math.hypot(a[0], a[1]))
    nb = float(math.hypot(b[0], b[1]))
    if na <= 1e-6 or nb <= 1e-6:
        return 180.0
    dot = max(-1.0, min(1.0, float(a[0] * b[0] + a[1] * b[1])))
    return abs(float(math.degrees(math.acos(dot))))


def midpoint_xy(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [
            (float(a[0]) + float(b[0])) * 0.5,
            (float(a[1]) + float(b[1])) * 0.5,
        ],
        dtype=np.float32,
    )


def ramer_douglas_peucker(points_xy: np.ndarray, epsilon_px: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or float(epsilon_px) <= 0.0:
        return pts
    start = pts[0]
    end = pts[-1]
    max_dist = -1.0
    split_idx = -1
    for idx in range(1, pts.shape[0] - 1):
        dist = point_to_segment_distance(pts[idx], start, end)
        if dist > max_dist:
            max_dist = dist
            split_idx = idx
    if max_dist <= float(epsilon_px) or split_idx <= 0:
        return np.asarray([start, end], dtype=np.float32)
    left = ramer_douglas_peucker(pts[: split_idx + 1], epsilon_px=epsilon_px)
    right = ramer_douglas_peucker(pts[split_idx:], epsilon_px=epsilon_px)
    if left.shape[0] == 0:
        return right
    if right.shape[0] == 0:
        return left
    return np.concatenate([left[:-1], right], axis=0)


def reverse_line_record(line: Dict[str, Any]) -> Dict[str, Any]:
    points = list(reversed(line.get("points_xy", [])))
    return {
        "category": str(line.get("category", "centerline")),
        "start_type": str(line.get("end_type", "")),
        "end_type": str(line.get("start_type", "")),
        "points_xy": [point.copy() for point in points],
        "source_indices": list(line.get("source_indices", [])),
        "merged_fragment_count": int(line.get("merged_fragment_count", 1)),
    }


def init_line_records(lines: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        pts = dedup_points(pts)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        out.append(
            {
                "category": str(line.get("category", "centerline")),
                "start_type": str(line.get("start_type", "")),
                "end_type": str(line.get("end_type", "")),
                "points_xy": pts,
                "source_indices": [int(idx)],
                "merged_fragment_count": 1,
            }
        )
    return out


def can_merge_line_pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    endpoint_tol_px: float,
    heading_tol_deg: float,
) -> Tuple[bool, float, float, Dict[str, Any], Dict[str, Any]]:
    best: Tuple[bool, float, float, Dict[str, Any], Dict[str, Any]] = (False, 1e9, 1e9, left, right)
    for left_rev in (False, True):
        left_line = reverse_line_record(left) if left_rev else left
        pts_left = np.asarray(left_line.get("points_xy", []), dtype=np.float32)
        if pts_left.ndim != 2 or pts_left.shape[0] < 2:
            continue
        for right_rev in (False, True):
            right_line = reverse_line_record(right) if right_rev else right
            pts_right = np.asarray(right_line.get("points_xy", []), dtype=np.float32)
            if pts_right.ndim != 2 or pts_right.shape[0] < 2:
                continue
            dist = point_distance(pts_left[-1], pts_right[0])
            if dist > float(endpoint_tol_px):
                continue
            angle = angle_between_deg(heading_at_end(pts_left), heading_at_start(pts_right))
            if angle > float(heading_tol_deg):
                continue
            if not best[0] or (float(dist), float(angle)) < (best[1], best[2]):
                best = (True, float(dist), float(angle), left_line, right_line)
    return best


def merge_two_line_records(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    pts_left = np.asarray(left.get("points_xy", []), dtype=np.float32)
    pts_right = np.asarray(right.get("points_xy", []), dtype=np.float32)
    if pts_left.ndim != 2 or pts_left.shape[0] < 2:
        return right
    if pts_right.ndim != 2 or pts_right.shape[0] < 2:
        return left
    join_point = midpoint_xy(pts_left[-1], pts_right[0])
    merged = np.concatenate(
        [
            pts_left[:-1],
            join_point.reshape(1, 2),
            pts_right[1:],
        ],
        axis=0,
    )
    merged = dedup_points(merged)
    return {
        "category": str(left.get("category", "centerline")),
        "start_type": str(left.get("start_type", "")),
        "end_type": str(right.get("end_type", "")),
        "points_xy": merged,
        "source_indices": list(left.get("source_indices", [])) + list(right.get("source_indices", [])),
        "merged_fragment_count": int(left.get("merged_fragment_count", 1)) + int(right.get("merged_fragment_count", 1)),
    }


def merge_target_line_fragments(
    lines: Sequence[Dict[str, Any]],
    *,
    patch_size: int,
    endpoint_tol_px: float = 6.0,
    heading_tol_deg: float = 22.5,
) -> List[Dict[str, Any]]:
    active = init_line_records(lines)
    while True:
        outgoing_best: Dict[int, Tuple[int, float, float, Dict[str, Any], Dict[str, Any]]] = {}
        incoming_best: Dict[int, Tuple[int, float, float, Dict[str, Any], Dict[str, Any]]] = {}
        for i, left in enumerate(active):
            best: Tuple[int, float, float, Dict[str, Any], Dict[str, Any]] | None = None
            for j, right in enumerate(active):
                if i == j:
                    continue
                ok, dist, angle, left_oriented, right_oriented = can_merge_line_pair(
                    left,
                    right,
                    endpoint_tol_px=float(endpoint_tol_px),
                    heading_tol_deg=float(heading_tol_deg),
                )
                if not ok:
                    continue
                cand = (int(j), float(dist), float(angle), left_oriented, right_oriented)
                if best is None or (cand[1], cand[2], cand[0]) < (best[1], best[2], best[0]):
                    best = cand
            if best is not None:
                outgoing_best[i] = best
        for i, cand in outgoing_best.items():
            j = int(cand[0])
            current = incoming_best.get(j)
            if current is None or (cand[1], cand[2], i) < (current[1], current[2], current[0]):
                incoming_best[j] = (int(i), float(cand[1]), float(cand[2]), cand[3], cand[4])

        matched: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
        used: set[int] = set()
        for i, cand in outgoing_best.items():
            j = int(cand[0])
            incoming = incoming_best.get(j)
            if incoming is None or int(incoming[0]) != int(i):
                continue
            if i in used or j in used:
                continue
            used.add(int(i))
            used.add(int(j))
            matched.append((int(i), int(j), cand[3], cand[4]))
        if not matched:
            break

        pair_lookup = {int(i): (int(j), left_oriented, right_oriented) for i, j, left_oriented, right_oriented in matched}
        next_active: List[Dict[str, Any]] = []
        consumed: set[int] = set()
        for idx, line in enumerate(active):
            if idx in consumed:
                continue
            if idx in pair_lookup:
                j, left_oriented, right_oriented = pair_lookup[idx]
                consumed.add(int(idx))
                consumed.add(int(j))
                next_active.append(merge_two_line_records(left_oriented, right_oriented))
            elif idx not in used:
                next_active.append(line)
        active = next_active

    out: List[Dict[str, Any]] = []
    for line in active:
        pts = np.asarray(line.get("points_xy", []), dtype=np.float32)
        pts = dedup_points(pts)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        points_json = simplify_for_json(pts, patch_size=int(patch_size))
        if len(points_json) < 2:
            continue
        out.append(
            {
                "category": str(line.get("category", "centerline")),
                "start_type": str(line.get("start_type", "")),
                "end_type": str(line.get("end_type", "")),
                "points": points_json,
            }
        )
    return sort_lines(out)


def build_douglas_target_lines(
    centerline_json_path: Path,
    *,
    source_patch_size: int = 512,
    target_patch_size: int = 256,
    douglas_epsilon_px: float = 2.5,
    boundary_tol_px: float | None = None,
) -> List[Dict[str, Any]]:
    obj = load_json(centerline_json_path)
    if boundary_tol_px is None:
        boundary_tol_px = 2.5 * float(max(target_patch_size - 1, 1)) / float(max(source_patch_size - 1, 1))
    rect = (0.0, 0.0, float(target_patch_size - 1), float(target_patch_size - 1))
    out: List[Dict[str, Any]] = []
    for rec in obj.get("lines", []):
        pts = np.asarray(rec.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        pts = dedup_points(pts)
        if pts.shape[0] < 2:
            continue
        pts = scale_points_inclusive(pts, source_patch_size=source_patch_size, target_patch_size=target_patch_size)
        pts = ramer_douglas_peucker(pts, epsilon_px=float(douglas_epsilon_px))
        pts = dedup_points(pts)
        if pts.shape[0] < 2:
            continue
        start_side = point_boundary_side(pts[0], rect, float(boundary_tol_px))
        end_side = point_boundary_side(pts[-1], rect, float(boundary_tol_px))
        start_type = "cut" if start_side is not None else "start"
        end_type = "cut" if end_side is not None else "end"
        pts, start_type, end_type = canonicalize_line_direction(pts, start_type=start_type, end_type=end_type)
        points_json = simplify_for_json(pts, patch_size=target_patch_size)
        if len(points_json) < 2:
            continue
        out.append(
            {
                "category": "centerline",
                "start_type": start_type,
                "end_type": end_type,
                "points": points_json,
            }
        )
    return sort_lines(out)


__all__ = [
    "build_douglas_target_lines",
    "merge_target_line_fragments",
    "can_merge_line_pair",
    "canonicalize_line_direction",
    "dedup_points",
    "load_json",
    "point_boundary_side",
    "point_origin_sort_key",
    "ramer_douglas_peucker",
    "scale_points_inclusive",
    "simplify_for_json",
    "sort_lines",
]
