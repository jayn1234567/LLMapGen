import json
import math
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


EARTH_RADIUS_M = 6378137.0


def token_key(name: str) -> str:
    name = str(name)
    if name.endswith("_satellite.png"):
        return name[: -len("_satellite.png")]
    if name.endswith(".png"):
        return name.rsplit(".", 1)[0]
    return name


@lru_cache(maxsize=8)
def _load_patch_geometry_map_cached(path: str, mtime_ns: int, size_bytes: int) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Geometry json must be a dict: {path}")

    entries: Dict[str, Dict] = {}
    lons: List[float] = []
    lats: List[float] = []
    for key, rec in raw.items():
        if not isinstance(rec, dict):
            continue
        tok = token_key(rec.get("sample_token", key))
        gps_center = rec.get("gps_center", None)
        if not isinstance(gps_center, (list, tuple)) or len(gps_center) < 2:
            continue
        lon = float(gps_center[0])
        lat = float(gps_center[1])
        lons.append(lon)
        lats.append(lat)
        entries[tok] = {
            "token": tok,
            "gps_center": [lon, lat],
            "image_width": int(rec.get("image_width", 512)),
            "image_height": int(rec.get("image_height", 512)),
            "crop_region": rec.get("crop_region", {}),
        }

    if not entries:
        return {}
    ref_lon = float(np.mean(lons))
    ref_lat = float(np.mean(lats))
    cos_lat = math.cos(math.radians(ref_lat))
    for tok, rec in entries.items():
        lon, lat = rec["gps_center"]
        x = math.radians(lon - ref_lon) * EARTH_RADIUS_M * cos_lat
        y = math.radians(lat - ref_lat) * EARTH_RADIUS_M
        rec["center_xy_m"] = [float(x), float(y)]
        rec["ref_lon_lat"] = [ref_lon, ref_lat]
    return entries


def load_patch_geometry_map(path: str) -> Dict[str, Dict]:
    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    return _load_patch_geometry_map_cached(abs_path, int(stat.st_mtime_ns), int(stat.st_size))


def build_patch_scan_order(tokens: Sequence[str], geom_map: Dict[str, Dict]) -> List[str]:
    keyed = []
    missing = []
    for name in tokens:
        tok = token_key(name)
        if tok in geom_map:
            cx, cy = geom_map[tok]["center_xy_m"]
            keyed.append((tok, float(cx), float(cy)))
        else:
            missing.append(tok)
    keyed.sort(key=lambda x: (-x[2], x[1]))
    out = [x[0] for x in keyed]
    out.extend(sorted(set(missing)))
    return out


def select_adjacent_global_lines(
    global_lines: Sequence[Dict],
    geom_rec: Dict,
    image_size: int,
    meter_per_pixel: float,
    overlap_margin_px: float,
    source_margin_px: float = 96.0,
    center_margin_m: float = 96.0,
) -> List[Dict]:
    out = []
    crop = geom_rec.get("crop_region", {}) or {}
    crop_x_min = float(crop.get("x_min", 0.0))
    crop_y_min = float(crop.get("y_min", 0.0))
    crop_x_max = float(crop.get("x_max", crop_x_min + float(geom_rec.get("image_width", image_size))))
    crop_y_max = float(crop.get("y_max", crop_y_min + float(geom_rec.get("image_height", image_size))))
    crop_original_image = str(crop.get("original_image", ""))
    center_xy = np.asarray(geom_rec.get("center_xy_m", [0.0, 0.0]), dtype=np.float32)
    for line in global_lines:
        keep = False
        pts_src = np.asarray(line.get("points_source_px", []), dtype=np.float32)
        if (
            pts_src.ndim == 2
            and pts_src.shape[0] > 0
            and crop_original_image
            and str(line.get("source_image", "")) == crop_original_image
        ):
            mask = (
                ((pts_src[:, 0] >= crop_x_min - float(source_margin_px)) & (pts_src[:, 0] <= crop_x_max + float(source_margin_px)))
                & ((pts_src[:, 1] >= crop_y_min - float(source_margin_px)) & (pts_src[:, 1] <= crop_y_max + float(source_margin_px)))
            )
            keep = bool(mask.any())
        if not keep:
            pts_g = np.asarray(line.get("points_global_xy", []), dtype=np.float32)
            if pts_g.ndim == 2 and pts_g.shape[0] > 0:
                dist = np.linalg.norm(pts_g - center_xy[None, :], axis=1)
                keep = bool((dist <= float(center_margin_m)).any())
        if keep:
            out.append(line)
    return out


def pixel_to_patch_local_m(points_px: np.ndarray, image_size: int, src_w: int, src_h: int, meter_per_pixel: float) -> np.ndarray:
    arr = np.asarray(points_px, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    sx = float(max(1, src_w - 1)) / float(max(1, image_size - 1))
    sy = float(max(1, src_h - 1)) / float(max(1, image_size - 1))
    px = arr[:, 0] * sx
    py = arr[:, 1] * sy
    cx = 0.5 * float(max(0, src_w - 1))
    cy = 0.5 * float(max(0, src_h - 1))
    x_m = (px - cx) * float(meter_per_pixel)
    y_m = (cy - py) * float(meter_per_pixel)
    return np.stack([x_m, y_m], axis=-1)


def patch_local_m_to_pixel(points_local_m: np.ndarray, image_size: int, src_w: int, src_h: int, meter_per_pixel: float) -> np.ndarray:
    arr = np.asarray(points_local_m, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    cx = 0.5 * float(max(0, src_w - 1))
    cy = 0.5 * float(max(0, src_h - 1))
    px = arr[:, 0] / max(float(meter_per_pixel), 1e-6) + cx
    py = cy - arr[:, 1] / max(float(meter_per_pixel), 1e-6)
    sx = float(max(1, image_size - 1)) / float(max(1, src_w - 1))
    sy = float(max(1, image_size - 1)) / float(max(1, src_h - 1))
    return np.stack([px * sx, py * sy], axis=-1)


def patch_lines_to_global(lines: Sequence[Dict], geom_rec: Dict, image_size: int, meter_per_pixel: float) -> List[Dict]:
    out = []
    center_xy = np.asarray(geom_rec["center_xy_m"], dtype=np.float32)
    src_w = int(geom_rec.get("image_width", image_size))
    src_h = int(geom_rec.get("image_height", image_size))
    crop = geom_rec.get("crop_region", {}) or {}
    crop_x_min = float(crop.get("x_min", 0.0))
    crop_y_min = float(crop.get("y_min", 0.0))
    original_image = str(crop.get("original_image", ""))
    for line in lines:
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        pts_local = pixel_to_patch_local_m(
            points_px=pts,
            image_size=image_size,
            src_w=src_w,
            src_h=src_h,
            meter_per_pixel=meter_per_pixel,
        )
        pts_global = pts_local + center_xy[None, :]
        sx = float(max(1, src_w - 1)) / float(max(1, image_size - 1))
        sy = float(max(1, src_h - 1)) / float(max(1, image_size - 1))
        pts_src = np.stack([pts[:, 0] * sx + crop_x_min, pts[:, 1] * sy + crop_y_min], axis=-1)
        out.append(
            {
                "category": line.get("category", "lane_line"),
                "line_type": line.get("line_type", ""),
                "start_type": line.get("start_type", "start"),
                "end_type": line.get("end_type", "end"),
                "points_global_xy": pts_global.tolist(),
                "points_source_px": pts_src.tolist(),
                "source_image": original_image,
            }
        )
    return out


def _densify_polyline(points_xy: np.ndarray, step_m: float) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return arr
    out = [arr[0]]
    max_step = max(float(step_m), 1e-3)
    for i in range(arr.shape[0] - 1):
        p0 = arr[i]
        p1 = arr[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(seg_len / max_step)))
        for k in range(1, n + 1):
            t = float(k) / float(n)
            out.append(p0 * (1.0 - t) + p1 * t)
    return np.asarray(out, dtype=np.float32)


def _is_border_point(p: np.ndarray, image_size: int, tol_px: float) -> bool:
    x = float(p[0])
    y = float(p[1])
    lo = float(tol_px)
    hi = float(image_size - 1) - float(tol_px)
    return x <= lo or x >= hi or y <= lo or y >= hi


def project_global_lines_to_patch(
    global_lines: Sequence[Dict],
    geom_rec: Dict,
    image_size: int,
    meter_per_pixel: float,
    max_lines: int,
    border_tol_px: float = 3.0,
    overlap_margin_px: float = 24.0,
    densify_step_m: float = 1.0,
) -> List[Dict]:
    out = []
    center_xy = np.asarray(geom_rec["center_xy_m"], dtype=np.float32)
    src_w = int(geom_rec.get("image_width", image_size))
    src_h = int(geom_rec.get("image_height", image_size))
    for line in global_lines:
        crop = geom_rec.get("crop_region", {}) or {}
        crop_x_min = float(crop.get("x_min", 0.0))
        crop_y_min = float(crop.get("y_min", 0.0))
        crop_x_max = float(crop.get("x_max", crop_x_min + src_w))
        crop_y_max = float(crop.get("y_max", crop_y_min + src_h))
        crop_original_image = str(crop.get("original_image", ""))
        pts_src = np.asarray(line.get("points_source_px", []), dtype=np.float32)
        if (
            pts_src.ndim == 2
            and pts_src.shape[0] > 0
            and str(line.get("source_image", "")) == crop_original_image
            and crop_original_image
        ):
            pts_src = _densify_polyline(pts_src, step_m=max(1.0, 1.0 / max(float(meter_per_pixel), 1e-6)))
            sx = float(max(1, image_size - 1)) / float(max(1, src_w - 1))
            sy = float(max(1, image_size - 1)) / float(max(1, src_h - 1))
            pix = np.stack([(pts_src[:, 0] - crop_x_min) * sx, (pts_src[:, 1] - crop_y_min) * sy], axis=-1)
            inside = (
                (pts_src[:, 0] >= crop_x_min - float(overlap_margin_px))
                & (pts_src[:, 0] <= crop_x_max + float(overlap_margin_px))
                & (pts_src[:, 1] >= crop_y_min - float(overlap_margin_px))
                & (pts_src[:, 1] <= crop_y_max + float(overlap_margin_px))
            )
        else:
            pts_g = np.asarray(line.get("points_global_xy", []), dtype=np.float32)
            if pts_g.ndim != 2 or pts_g.shape[0] == 0:
                continue
            pts_g = _densify_polyline(pts_g, step_m=densify_step_m)
            pts_local = pts_g - center_xy[None, :]
            pix = patch_local_m_to_pixel(
                points_local_m=pts_local,
                image_size=image_size,
                src_w=src_w,
                src_h=src_h,
                meter_per_pixel=meter_per_pixel,
            )
            inside = (
                (pix[:, 0] >= -float(overlap_margin_px))
                & (pix[:, 0] <= float(image_size - 1) + float(overlap_margin_px))
                & (pix[:, 1] >= -float(overlap_margin_px))
                & (pix[:, 1] <= float(image_size - 1) + float(overlap_margin_px))
            )
        if int(inside.sum()) <= 0:
            continue
        inside_idx = np.nonzero(inside)[0]
        start_i = max(0, int(inside_idx[0]) - 2)
        end_i = min(int(pix.shape[0] - 1), int(inside_idx[-1]) + 2)
        seg = np.clip(pix[start_i : end_i + 1], 0.0, float(image_size - 1))
        if seg.ndim != 2 or seg.shape[0] == 0:
            continue
        dedup = [seg[0]]
        for p in seg[1:]:
            if float(np.linalg.norm(p - dedup[-1])) > 1e-3:
                dedup.append(p)
        seg = np.asarray(dedup, dtype=np.float32)
        if seg.shape[0] == 0:
            continue
        start_cut = start_i > 0 or _is_border_point(seg[0], image_size=image_size, tol_px=border_tol_px)
        end_cut = end_i < int(pix.shape[0] - 1) or _is_border_point(seg[-1], image_size=image_size, tol_px=border_tol_px)
        out.append(
            {
                "category": line.get("category", "lane_line"),
                "line_type": line.get("line_type", ""),
                "start_type": "cut" if start_cut else "start",
                "end_type": "cut" if end_cut else "end",
                "points": seg,
            }
        )
        if len(out) >= int(max_lines):
            break
    return out


def project_global_cut_points_to_patch(
    global_lines: Sequence[Dict],
    geom_rec: Dict,
    image_size: int,
    meter_per_pixel: float,
    max_points: int,
    border_tol_px: float = 8.0,
    endpoint_margin_px: float = 40.0,
) -> List[Dict]:
    out = []
    center_xy = np.asarray(geom_rec["center_xy_m"], dtype=np.float32)
    src_w = int(geom_rec.get("image_width", image_size))
    src_h = int(geom_rec.get("image_height", image_size))
    crop = geom_rec.get("crop_region", {}) or {}
    crop_x_min = float(crop.get("x_min", 0.0))
    crop_y_min = float(crop.get("y_min", 0.0))
    crop_x_max = float(crop.get("x_max", crop_x_min + src_w))
    crop_y_max = float(crop.get("y_max", crop_y_min + src_h))
    crop_original_image = str(crop.get("original_image", ""))
    for line in global_lines:
        pts_g = np.asarray(line.get("points_global_xy", []), dtype=np.float32)
        if pts_g.ndim != 2 or pts_g.shape[0] == 0:
            continue
        pts_src = np.asarray(line.get("points_source_px", []), dtype=np.float32)
        candidates = []
        if line.get("start_type", "start") == "cut":
            candidates.append(("src", pts_src[0] if pts_src.ndim == 2 and pts_src.shape[0] > 0 else None, pts_g[0]))
        if line.get("end_type", "end") == "cut":
            candidates.append(("src", pts_src[-1] if pts_src.ndim == 2 and pts_src.shape[0] > 0 else None, pts_g[-1]))
        for _, pt_src, pt_g in candidates:
            if pt_src is not None and str(line.get("source_image", "")) == crop_original_image and crop_original_image:
                sx = float(max(1, image_size - 1)) / float(max(1, src_w - 1))
                sy = float(max(1, image_size - 1)) / float(max(1, src_h - 1))
                pix = np.asarray([(pt_src[0] - crop_x_min) * sx, (pt_src[1] - crop_y_min) * sy], dtype=np.float32)
                dx_gap = 0.0
                dy_gap = 0.0
                if float(pt_src[0]) < crop_x_min:
                    dx_gap = crop_x_min - float(pt_src[0])
                elif float(pt_src[0]) > crop_x_max:
                    dx_gap = float(pt_src[0]) - crop_x_max
                if float(pt_src[1]) < crop_y_min:
                    dy_gap = crop_y_min - float(pt_src[1])
                elif float(pt_src[1]) > crop_y_max:
                    dy_gap = float(pt_src[1]) - crop_y_max
                near_window = dx_gap <= float(endpoint_margin_px) and dy_gap <= float(endpoint_margin_px)
            else:
                pix = patch_local_m_to_pixel(
                    points_local_m=(pt_g[None, :] - center_xy[None, :]),
                    image_size=image_size,
                    src_w=src_w,
                    src_h=src_h,
                    meter_per_pixel=meter_per_pixel,
                )[0]
                near_window = (
                    -float(endpoint_margin_px) <= float(pix[0]) <= float(image_size - 1) + float(endpoint_margin_px)
                    and -float(endpoint_margin_px) <= float(pix[1]) <= float(image_size - 1) + float(endpoint_margin_px)
                )
            if not near_window:
                continue
            clipped = np.clip(pix, 0.0, float(image_size - 1))
            if not _is_border_point(clipped, image_size=image_size, tol_px=border_tol_px):
                continue
            out.append(
                {
                    "category": line.get("category", "lane_line"),
                    "line_type": line.get("line_type", ""),
                    "start_type": "cut",
                    "end_type": "cut",
                    "points": clipped[None, :].astype(np.float32),
                }
            )
            if len(out) >= int(max_points):
                return out
    return out


def project_global_cut_traces_to_patch(
    global_lines: Sequence[Dict],
    geom_rec: Dict,
    image_size: int,
    meter_per_pixel: float,
    max_traces: int,
    border_tol_px: float = 8.0,
    endpoint_margin_px: float = 40.0,
    trace_num_points: int = 3,
) -> List[Dict]:
    out = []
    center_xy = np.asarray(geom_rec["center_xy_m"], dtype=np.float32)
    src_w = int(geom_rec.get("image_width", image_size))
    src_h = int(geom_rec.get("image_height", image_size))
    crop = geom_rec.get("crop_region", {}) or {}
    crop_x_min = float(crop.get("x_min", 0.0))
    crop_y_min = float(crop.get("y_min", 0.0))
    crop_x_max = float(crop.get("x_max", crop_x_min + src_w))
    crop_y_max = float(crop.get("y_max", crop_y_min + src_h))
    crop_original_image = str(crop.get("original_image", ""))
    trace_num_points = max(2, int(trace_num_points))
    for line in global_lines:
        pts_g = np.asarray(line.get("points_global_xy", []), dtype=np.float32)
        if pts_g.ndim != 2 or pts_g.shape[0] < 2:
            continue
        pts_src = np.asarray(line.get("points_source_px", []), dtype=np.float32)
        candidates = []
        if line.get("start_type", "start") == "cut":
            idxs = list(range(min(trace_num_points, pts_g.shape[0])))
            candidates.append(("start", idxs))
        if line.get("end_type", "end") == "cut":
            idxs = list(range(max(0, pts_g.shape[0] - trace_num_points), pts_g.shape[0]))
            candidates.append(("end", idxs))
        for side, idxs in candidates:
            seg_g = pts_g[idxs]
            if side == "end":
                seg_g = seg_g[::-1].copy()
            if pts_src.ndim == 2 and pts_src.shape[0] == pts_g.shape[0] and crop_original_image and str(line.get("source_image", "")) == crop_original_image:
                seg_src = pts_src[idxs]
                if side == "end":
                    seg_src = seg_src[::-1].copy()
                near = (
                    np.min(np.abs(seg_src[:, 0] - crop_x_min)) <= float(endpoint_margin_px)
                    or np.min(np.abs(seg_src[:, 0] - crop_x_max)) <= float(endpoint_margin_px)
                    or np.min(np.abs(seg_src[:, 1] - crop_y_min)) <= float(endpoint_margin_px)
                    or np.min(np.abs(seg_src[:, 1] - crop_y_max)) <= float(endpoint_margin_px)
                )
                if not near:
                    continue
                sx = float(max(1, image_size - 1)) / float(max(1, src_w - 1))
                sy = float(max(1, image_size - 1)) / float(max(1, src_h - 1))
                seg_pix = np.stack([(seg_src[:, 0] - crop_x_min) * sx, (seg_src[:, 1] - crop_y_min) * sy], axis=-1)
            else:
                seg_local = seg_g - center_xy[None, :]
                seg_pix = patch_local_m_to_pixel(
                    points_local_m=seg_local,
                    image_size=image_size,
                    src_w=src_w,
                    src_h=src_h,
                    meter_per_pixel=meter_per_pixel,
                )
                near = np.any(
                    (seg_pix[:, 0] >= -float(endpoint_margin_px))
                    & (seg_pix[:, 0] <= float(image_size - 1) + float(endpoint_margin_px))
                    & (seg_pix[:, 1] >= -float(endpoint_margin_px))
                    & (seg_pix[:, 1] <= float(image_size - 1) + float(endpoint_margin_px))
                )
                if not near:
                    continue
            seg_pix = np.clip(seg_pix, 0.0, float(image_size - 1))
            if not _is_border_point(seg_pix[0], image_size=image_size, tol_px=border_tol_px):
                continue
            out.append(
                {
                    "category": line.get("category", "lane_line"),
                    "line_type": line.get("line_type", ""),
                    "start_type": "cut",
                    "end_type": "cut",
                    "points": seg_pix.astype(np.float32),
                }
            )
            if len(out) >= int(max_traces):
                return out
    return out


def build_state_lines_from_global(
    global_lines: Sequence[Dict],
    geom_rec: Dict,
    image_size: int,
    meter_per_pixel: float,
    max_lines: int,
    border_tol_px: float = 4.0,
    overlap_margin_px: float = 24.0,
    endpoint_margin_px: float = 40.0,
    densify_step_m: float = 1.0,
    trace_num_points: int = 3,
    adjacent_source_margin_px: float = 96.0,
    adjacent_center_margin_m: float = 96.0,
) -> Tuple[List[Dict], Dict[str, int]]:
    candidate_lines = select_adjacent_global_lines(
        global_lines=global_lines,
        geom_rec=geom_rec,
        image_size=image_size,
        meter_per_pixel=meter_per_pixel,
        overlap_margin_px=overlap_margin_px,
        source_margin_px=adjacent_source_margin_px,
        center_margin_m=adjacent_center_margin_m,
    )
    projected = project_global_lines_to_patch(
        global_lines=candidate_lines,
        geom_rec=geom_rec,
        image_size=image_size,
        meter_per_pixel=meter_per_pixel,
        max_lines=max_lines,
        border_tol_px=border_tol_px,
        overlap_margin_px=overlap_margin_px,
        densify_step_m=densify_step_m,
    )
    cut_traces = project_global_cut_traces_to_patch(
        global_lines=candidate_lines,
        geom_rec=geom_rec,
        image_size=image_size,
        meter_per_pixel=meter_per_pixel,
        max_traces=max_lines,
        border_tol_px=max(border_tol_px, 6.0),
        endpoint_margin_px=endpoint_margin_px,
        trace_num_points=trace_num_points,
    )
    out = list(projected)
    seen = set()
    for line in out:
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] > 0:
            key = (
                line.get("category", ""),
                line.get("line_type", ""),
                tuple(np.round(pts[0]).astype(int).tolist()),
                tuple(np.round(pts[-1]).astype(int).tolist()),
            )
            seen.add(key)
    for line in cut_traces:
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        key = (
            line.get("category", ""),
            line.get("line_type", ""),
            tuple(np.round(pts[0]).astype(int).tolist()),
            tuple(np.round(pts[-1]).astype(int).tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= int(max_lines):
            break
    stats = {
        "num_candidate_lines": len(candidate_lines),
        "num_projected_lines": len(projected),
        "num_endpoint_primitives": len(cut_traces),
        "num_state_lines": len(out),
    }
    return out[: int(max_lines)], stats


def _line_signature(points_global_xy: Sequence[Sequence[float]], cell_m: float = 1.0) -> Tuple:
    arr = np.asarray(points_global_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return tuple()
    p0 = np.round(arr[0] / float(cell_m)).astype(np.int32)
    p1 = np.round(arr[-1] / float(cell_m)).astype(np.int32)
    a = tuple(p0.tolist())
    b = tuple(p1.tolist())
    if a > b:
        a, b = b, a
    return (a, b, int(arr.shape[0]))


def merge_global_lines(
    global_lines: List[Dict],
    new_lines: Iterable[Dict],
    cell_m: float = 1.0,
    connect_radius_m: float = 3.0,
) -> List[Dict]:
    return merge_global_lines_with_endpoints(
        global_lines=global_lines,
        new_lines=new_lines,
        cell_m=cell_m,
        connect_radius_m=connect_radius_m,
    )


def merge_global_lines_with_endpoints(
    global_lines: List[Dict],
    new_lines: Iterable[Dict],
    cell_m: float = 1.0,
    connect_radius_m: float = 3.0,
) -> List[Dict]:
    existing = set()
    for line in global_lines:
        existing.add(
            (
                line.get("category", "unknown"),
                line.get("line_type", ""),
                _line_signature(line.get("points_global_xy", []), cell_m=cell_m),
            )
        )
    for line in new_lines:
        line = _attach_or_merge_line(global_lines=global_lines, line=line, connect_radius_m=connect_radius_m)
        sig = (
            line.get("category", "unknown"),
            line.get("line_type", ""),
            _line_signature(line.get("points_global_xy", []), cell_m=cell_m),
        )
        if sig in existing:
            continue
        existing.add(sig)
        global_lines.append(line)
    return global_lines


def _attach_or_merge_line(global_lines: List[Dict], line: Dict, connect_radius_m: float) -> Dict:
    new_pts = np.asarray(line.get("points_global_xy", []), dtype=np.float32)
    if new_pts.ndim != 2 or new_pts.shape[0] < 2:
        return line
    best = None
    best_idx = -1
    for i, base in enumerate(global_lines):
        if str(base.get("category", "")) != str(line.get("category", "")):
            continue
        if str(base.get("line_type", "")) != str(line.get("line_type", "")):
            continue
        base_pts = np.asarray(base.get("points_global_xy", []), dtype=np.float32)
        if base_pts.ndim != 2 or base_pts.shape[0] < 2:
            continue
        cand = [
            ("base_end_new_start", float(np.linalg.norm(base_pts[-1] - new_pts[0]))),
            ("base_start_new_end", float(np.linalg.norm(base_pts[0] - new_pts[-1]))),
            ("base_end_new_end", float(np.linalg.norm(base_pts[-1] - new_pts[-1]))),
            ("base_start_new_start", float(np.linalg.norm(base_pts[0] - new_pts[0]))),
        ]
        mode, dist = min(cand, key=lambda x: x[1])
        if dist > float(connect_radius_m):
            continue
        if best is None or dist < best[1]:
            best = (mode, dist)
            best_idx = i
    if best is None or best_idx < 0:
        return line

    base = global_lines.pop(best_idx)
    base_pts = np.asarray(base.get("points_global_xy", []), dtype=np.float32)
    mode = best[0]
    if mode == "base_end_new_start":
        merged_pts = _concat_lines(base_pts, new_pts)
        start_type = base.get("start_type", "start")
        end_type = line.get("end_type", "end")
    elif mode == "base_start_new_end":
        merged_pts = _concat_lines(new_pts, base_pts)
        start_type = line.get("start_type", "start")
        end_type = base.get("end_type", "end")
    elif mode == "base_end_new_end":
        merged_pts = _concat_lines(base_pts, new_pts[::-1].copy())
        start_type = base.get("start_type", "start")
        end_type = line.get("start_type", "start")
    else:
        merged_pts = _concat_lines(new_pts[::-1].copy(), base_pts)
        start_type = line.get("end_type", "end")
        end_type = base.get("end_type", "end")
    return {
        "category": line.get("category", "lane_line"),
        "line_type": line.get("line_type", ""),
        "start_type": start_type,
        "end_type": end_type,
        "points_global_xy": merged_pts.tolist(),
    }


def _concat_lines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0:
        return b
    if b.shape[0] == 0:
        return a
    out = [a[0]]
    for p in a[1:]:
        if float(np.linalg.norm(p - out[-1])) > 1e-3:
            out.append(p)
    start_j = 0
    if float(np.linalg.norm(np.asarray(out[-1]) - b[0])) <= 1.5:
        start_j = 1
    for p in b[start_j:]:
        if float(np.linalg.norm(p - out[-1])) > 1e-3:
            out.append(p)
    return np.asarray(out, dtype=np.float32)
