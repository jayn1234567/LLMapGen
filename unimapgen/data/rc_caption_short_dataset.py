from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from unimapgen.data.rc_centerline_cnn_prefix_dataset import (
    index_rows_by_id,
    load_jsonl,
    load_segmentation_label_map_from_path,
    pil_to_tensor,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a structured BEV road-scene captioning assistant.\n"
    "The image is a black-background BEV road-structure image with two visible road classes.\n"
    "The downstream task is road-centerline prediction.\n"
    "The image itself does not show centerlines directly; use only visible road structure.\n\n"
    "You must output:\n"
    "1. one global scene label\n"
    "2. one grid-state sequence for the fixed 8x8 full-coverage grid\n\n"
    "Allowed Scene labels:\n"
    "straight, curved, branching, intersection-approach, complex\n\n"
    "Allowed GridStates labels:\n"
    "background, lane_boundary, lane_divider, mix\n\n"
    "The 512x512 patch is partitioned into a fixed 8x8 full-coverage grid.\n"
    "Each GridStates entry corresponds to one 64x64 cell.\n"
    "Cell order is row-major from top-left to bottom-right.\n"
    "For each grid cell:\n"
    "- output background if the cell contains no road-class pixels\n"
    "- output lane_boundary if it contains only lane_boundary pixels\n"
    "- output lane_divider if it contains only lane_divider pixels\n"
    "- output mix if it contains both lane_boundary and lane_divider pixels\n\n"
    "Output format must be exactly:\n"
    "Scene=<scene_label>\n"
    "GridStates=[state_1,state_2,...,state_64]\n\n"
    "Do not add explanations or extra text."
)

DEFAULT_USER_PROMPT = (
    "Predict:\n"
    "1. Scene\n"
    "2. GridStates for the fixed 8x8 full-coverage grid in row-major order"
)

GRID_SCHEMA_VERSION = "scene_grid_states_v1"
DEFAULT_GRID_ROWS = 8
DEFAULT_GRID_COLS = 8
DEFAULT_GRID_COUNT = DEFAULT_GRID_ROWS * DEFAULT_GRID_COLS
GRID_LABEL_BACKGROUND = "background"
GRID_LABEL_LANE_BOUNDARY = "lane_boundary"
GRID_LABEL_LANE_DIVIDER = "lane_divider"
GRID_LABEL_MIX = "mix"

VISUAL_TOKENS = [
    "<vis_start>",
    "<vis_patch>",
    "<vis_end>",
]

SIDE_ORDER = ["top", "bottom", "left", "right"]
OPPOSITE_SIDE_MAP = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_visual_placeholder(num_visual_tokens: int) -> str:
    return "<vis_start> " + " ".join(["<vis_patch>"] * int(num_visual_tokens)) + " <vis_end>"


def build_grid_user_prompt(
    *,
    user_prompt_prefix: str = DEFAULT_USER_PROMPT,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> str:
    prefix = str(user_prompt_prefix).strip()
    grid_text = f"GridSize={int(grid_rows)}x{int(grid_cols)}"
    if not prefix:
        return grid_text
    if "GridSize=" in prefix:
        return prefix
    return f"{prefix}\n\n{grid_text}"


def build_grid_caption_text(scene_label: str, grid_states: Sequence[str]) -> str:
    state_text = ",".join(str(state).strip().lower() for state in grid_states)
    return f"Scene={str(scene_label).strip().lower()}\nGridStates=[{state_text}]"


def is_grid_schema_caption_text(text: str) -> bool:
    normalized = str(text).strip()
    if not normalized:
        return False
    return normalized.startswith("Scene=") and "\nGridStates=[" in normalized


def normalize_grid_states(raw_states: Sequence[Any]) -> List[str]:
    states: List[str] = []
    for raw in raw_states:
        state = str(raw).strip().lower()
        if state:
            states.append(state)
    return states


def count_non_background_grid_cells(grid_states: Sequence[Any]) -> int:
    return sum(
        1
        for state in normalize_grid_states(grid_states)
        if str(state).strip().lower() != GRID_LABEL_BACKGROUND
    )


def extract_message_content(sample: Dict[str, Any], role: str) -> str:
    wanted = str(role).strip().lower()
    for message in sample.get("messages", []):
        if str(message.get("role", "")).strip().lower() != wanted:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return ""


def extract_prompt_texts(sample: Dict[str, Any], meta: Dict[str, Any]) -> tuple[str, str]:
    system_text = extract_message_content(sample, "system") or str(meta.get("caption_system_prompt", "")).strip()
    user_text = extract_message_content(sample, "user") or str(meta.get("caption_user_prompt", "")).strip()
    return system_text, user_text


def resolve_structure_multiclass_path(
    meta: Dict[str, Any],
    media_dir: Path,
    *,
    fallback_image_path: Path | None = None,
) -> Path:
    rel_mask = str(meta.get("seg_structure_multiclass", "")).strip()
    if rel_mask:
        mask_path = (media_dir / rel_mask).resolve()
        if mask_path.is_file():
            return mask_path
    if fallback_image_path is not None:
        return Path(fallback_image_path).resolve()
    raise FileNotFoundError("seg_structure_multiclass path is missing and no fallback image path was provided.")


def grid_cell_bounds(
    *,
    row_idx: int,
    col_idx: int,
    patch_size: int,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> tuple[int, int, int, int]:
    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    patch = max(1, int(patch_size))
    x0 = (int(col_idx) * patch) // cols
    x1 = (((int(col_idx) + 1) * patch) // cols) - 1
    y0 = (int(row_idx) * patch) // rows
    y1 = (((int(row_idx) + 1) * patch) // rows) - 1
    return x0, x1, y0, y1


def classify_label_patch(label_patch: np.ndarray) -> str:
    present = {int(value) for value in np.unique(label_patch) if int(value) > 0}
    if not present:
        return GRID_LABEL_BACKGROUND
    if present == {1}:
        return GRID_LABEL_LANE_DIVIDER
    if present == {2}:
        return GRID_LABEL_LANE_BOUNDARY
    return GRID_LABEL_MIX


def classify_grid_states_from_label_map(
    *,
    label_map_path: Path,
    patch_size: int,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> List[str]:
    # Stage 2 采用固定 8x8 全覆盖网格，每个格子的标签直接从 structure_multiclass 掩码统计得到。
    label_map = load_segmentation_label_map_from_path(
        label_map_path,
        image_size=int(patch_size),
        supervision_mode="structure_multiclass",
    )
    label_arr = label_map.cpu().numpy() if torch.is_tensor(label_map) else np.asarray(label_map)
    states: List[str] = []
    for row_idx in range(max(1, int(grid_rows))):
        for col_idx in range(max(1, int(grid_cols))):
            x0, x1, y0, y1 = grid_cell_bounds(
                row_idx=row_idx,
                col_idx=col_idx,
                patch_size=int(patch_size),
                grid_rows=int(grid_rows),
                grid_cols=int(grid_cols),
            )
            states.append(classify_label_patch(label_arr[y0 : y1 + 1, x0 : x1 + 1]))
    return states


def format_side_list(sides: Sequence[str]) -> str:
    ordered = [side for side in SIDE_ORDER if side in set(str(s).strip().lower() for s in sides)]
    if not ordered:
        return "center"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


def normalize_line_records(raw_lines: Sequence[Any]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for raw in raw_lines:
        if isinstance(raw, dict):
            points = raw.get("points", [])
            category = str(raw.get("category", "")).strip().lower() or "structure"
        else:
            points = raw
            category = "structure"
        if not isinstance(points, list) or len(points) < 2:
            continue
        coords: List[List[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            coords.append([float(point[0]), float(point[1])])
        if len(coords) < 2:
            continue
        valid.append({"category": category, "points": coords})
    return valid


def point_to_side(point_xy: Sequence[float], patch_size: int, border_tol_px: float) -> str | None:
    x = float(point_xy[0])
    y = float(point_xy[1])
    max_xy = float(max(0, int(patch_size) - 1))
    tol = float(max(1.0, border_tol_px))
    if abs(y - 0.0) <= tol:
        return "top"
    if abs(y - max_xy) <= tol:
        return "bottom"
    if abs(x - 0.0) <= tol:
        return "left"
    if abs(x - max_xy) <= tol:
        return "right"
    return None


def side_pair_is_opposite(side_a: str, side_b: str) -> bool:
    return OPPOSITE_SIDE_MAP.get(str(side_a).strip().lower(), "") == str(side_b).strip().lower()


def path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return float(np.sum(seg))


def line_directness(points: np.ndarray) -> float:
    total = path_length(points)
    if total <= 1e-6:
        return 0.0
    chord = float(np.linalg.norm(points[-1] - points[0]))
    return chord / total


def line_turning(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    vec = points[1:] - points[:-1]
    norms = np.linalg.norm(vec, axis=1)
    keep = norms > 1e-6
    vec = vec[keep]
    norms = norms[keep]
    if vec.shape[0] < 2:
        return 0.0
    unit = vec / norms[:, None]
    dots = np.sum(unit[1:] * unit[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    return float(np.mean(np.abs(angles)))


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    w = np.asarray(list(weights), dtype=np.float32)
    if w.size != arr.size or float(np.sum(w)) <= 1e-6:
        return float(np.mean(arr))
    return float(np.sum(arr * w) / np.sum(w))


def line_orientation(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    vec = points[-1] - points[0]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return 0.0
    return float(np.mod(np.arctan2(float(vec[1]), float(vec[0])), np.pi))


def orientation_spread(angles: Sequence[float], weights: Sequence[float]) -> float:
    if not angles:
        return 1.0
    angle_arr = np.asarray(list(angles), dtype=np.float32)
    weight_arr = np.asarray(list(weights), dtype=np.float32)
    if angle_arr.size == 0:
        return 1.0
    if weight_arr.size != angle_arr.size or float(np.sum(weight_arr)) <= 1e-6:
        weight_arr = np.ones_like(angle_arr, dtype=np.float32)
    doubled = angle_arr * 2.0
    vec_x = float(np.sum(weight_arr * np.cos(doubled)))
    vec_y = float(np.sum(weight_arr * np.sin(doubled)))
    concentration = float(np.hypot(vec_x, vec_y) / max(1e-6, float(np.sum(weight_arr))))
    return float(max(0.0, min(1.0, 1.0 - concentration)))


def dominant_orientation(angles: Sequence[float], weights: Sequence[float]) -> float:
    if not angles:
        return 0.0
    angle_arr = np.asarray(list(angles), dtype=np.float32)
    weight_arr = np.asarray(list(weights), dtype=np.float32)
    if angle_arr.size == 0:
        return 0.0
    if weight_arr.size != angle_arr.size or float(np.sum(weight_arr)) <= 1e-6:
        weight_arr = np.ones_like(angle_arr, dtype=np.float32)
    doubled = angle_arr * 2.0
    vec_x = float(np.sum(weight_arr * np.cos(doubled)))
    vec_y = float(np.sum(weight_arr * np.sin(doubled)))
    if abs(vec_x) <= 1e-6 and abs(vec_y) <= 1e-6:
        return float(angle_arr[0])
    return float(np.mod(0.5 * np.arctan2(vec_y, vec_x), np.pi))


def orientation_distance(angle: float, reference: float) -> float:
    raw = abs(float(angle) - float(reference))
    raw = float(np.mod(raw, np.pi))
    return float(min(raw, np.pi - raw))


def line_side_support(points: np.ndarray, patch_size: int, border_tol_px: float) -> List[str]:
    if points.shape[0] == 0:
        return []
    max_xy = float(max(0, int(patch_size) - 1))
    tol = float(max(1.0, border_tol_px, float(patch_size) * 0.05))
    sides: List[str] = []
    if float(np.min(points[:, 1])) <= tol:
        sides.append("top")
    if float(np.max(points[:, 1])) >= max_xy - tol:
        sides.append("bottom")
    if float(np.min(points[:, 0])) <= tol:
        sides.append("left")
    if float(np.max(points[:, 0])) >= max_xy - tol:
        sides.append("right")
    return sides


def pick_opposite_side_pair(visible_sides: Sequence[str], side_scores: Dict[str, float]) -> tuple[str, str] | None:
    visible_set = set(str(side).strip().lower() for side in visible_sides)
    best_pair: tuple[str, str] | None = None
    best_score = -1.0
    for side_a, side_b in (("top", "bottom"), ("left", "right")):
        if side_a not in visible_set or side_b not in visible_set:
            continue
        score = float(side_scores.get(side_a, 0.0)) + float(side_scores.get(side_b, 0.0))
        if score > best_score:
            best_pair = (side_a, side_b)
            best_score = score
    return best_pair


def extract_lines_from_meta(meta: Dict[str, Any], media_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    structure_lines: List[Dict[str, Any]] = []
    centerline_lines: List[Dict[str, Any]] = []

    if isinstance(meta.get("target_lines"), list):
        centerline_lines = normalize_line_records(meta.get("target_lines", []))

    centerline_rel = str(meta.get("centerline_json", "")).strip()
    if not centerline_lines and centerline_rel:
        centerline_path = (media_dir / centerline_rel).resolve()
        if centerline_path.is_file():
            centerline_lines = normalize_line_records(load_json(centerline_path).get("lines", []))

    structure_rel = str(meta.get("structure_json", "")).strip()
    if structure_rel:
        structure_path = (media_dir / structure_rel).resolve()
        if structure_path.is_file():
            structure_lines = normalize_line_records(load_json(structure_path).get("lines", []))

    return structure_lines, centerline_lines


def extract_caption_from_sample(sample: Dict[str, Any], meta: Dict[str, Any]) -> tuple[str, str]:
    meta_caption = str(meta.get("caption_short", "")).strip()
    meta_label = str(meta.get("caption_label", "")).strip().lower()
    if meta_caption:
        return meta_label or "provided", meta_caption
    for message in sample.get("messages", []):
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        return meta_label or "provided", content
    return "", ""


def analyze_scene_caption_short(
    *,
    structure_lines: Sequence[Dict[str, Any]],
    centerline_lines: Sequence[Dict[str, Any]],
    patch_size: int,
    border_tol_px: float,
) -> Dict[str, Any]:
    # Caption semantics must come only from visible RC structure.
    # Centerlines are target-side reference geometry and are not visible in the RC input.
    _ = centerline_lines
    visible_lines = list(structure_lines)
    if not visible_lines:
        return {
            "scene_label": "complex",
            "caption_text": "A complex road-structure patch with sparse internal road structures.",
            "num_visible_lines": 0,
            "num_analysis_lines": 0,
            "visible_sides": [],
            "side_count": 0,
            "major_side_set": [],
            "major_side_count": 0,
            "side_scores": {side: 0.0 for side in SIDE_ORDER},
            "side_counts": {side: 0 for side in SIDE_ORDER},
            "mean_directness": 0.0,
            "mean_turning": 0.0,
            "straight_candidate_count": 0,
            "parallel_spread": float(np.pi),
            "parallel_corridor": False,
            "dominant_axis": None,
            "off_axis_center_lines": 0,
            "off_axis_center_support": 0.0,
            "off_axis_total_support": 0.0,
            "off_axis_major_lines": 0,
            "axis_aligned_support": 0.0,
            "off_axis_side_set": [],
            "center_connector_lines": 0,
            "center_connector_support": 0.0,
            "center_connector_internal_lines": 0,
            "center_bridge_lines": 0,
            "center_bridge_support": 0.0,
            "center_crossing_lines": 0,
            "loop_like_lines": 0,
            "central_internal_endpoints": 0,
            "connector_veto": False,
            "center_bridge_veto": False,
            "straight_veto": False,
            "opposite_pair": None,
            "min_support_length": float(max(48.0, float(patch_size) * 0.12)),
            "min_major_length": float(max(72.0, float(patch_size) * 0.18)),
        }

    min_support_length = float(max(48.0, float(patch_size) * 0.12))
    min_major_length = float(max(72.0, float(patch_size) * 0.18))
    center_min = float(patch_size) * 0.30
    center_max = float(patch_size) * 0.70
    center_tight_min = float(patch_size) * 0.42
    center_tight_max = float(patch_size) * 0.58

    line_stats: List[Dict[str, Any]] = []
    central_internal_endpoints = 0
    for line in visible_lines:
        points = np.asarray(line["points"], dtype=np.float32)
        length = path_length(points)
        directness = line_directness(points)
        turning = line_turning(points)
        sides = line_side_support(points, patch_size=patch_size, border_tol_px=border_tol_px)
        touches_tight_center = bool(
            np.any(
                (points[:, 0] >= center_tight_min)
                & (points[:, 0] <= center_tight_max)
                & (points[:, 1] >= center_tight_min)
                & (points[:, 1] <= center_tight_max)
            )
        )
        touches_center = bool(
            np.any(
                (points[:, 0] >= center_min)
                & (points[:, 0] <= center_max)
                & (points[:, 1] >= center_min)
                & (points[:, 1] <= center_max)
            )
        )
        loop_like = bool(
            length >= min_major_length * 0.70
            and len(sides) == 0
            and directness <= 0.82
            and turning >= 0.18
        )
        line_stats.append(
            {
                "points": points,
                "length": float(length),
                "directness": float(directness),
                "turning": float(turning),
                "sides": list(sides),
                "touches_center": touches_center,
                "touches_tight_center": touches_tight_center,
                "loop_like": loop_like,
            }
        )
        for endpoint in (points[0], points[-1]):
            side = point_to_side(endpoint, patch_size=patch_size, border_tol_px=border_tol_px)
            if side is not None:
                continue
            x = float(endpoint[0])
            y = float(endpoint[1])
            if center_min <= x <= center_max and center_min <= y <= center_max:
                central_internal_endpoints += 1

    analysis_lines = [stat for stat in line_stats if float(stat["length"]) >= min_support_length]
    if len(analysis_lines) < 2:
        analysis_lines = list(line_stats)

    side_scores = {side: 0.0 for side in SIDE_ORDER}
    side_counts = {side: 0 for side in SIDE_ORDER}
    for stat in analysis_lines:
        for side in stat["sides"]:
            side_scores[side] += float(stat["length"])
            side_counts[side] += 1

    major_side_set = {
        side
        for stat in analysis_lines
        if float(stat["length"]) >= min_major_length * 0.80
        for side in stat["sides"]
    }

    max_side_score = float(max(side_scores.values())) if side_scores else 0.0
    min_visible_side_score = float(max(min_support_length * 0.90, max_side_score * 0.35))
    side_rank = sorted(SIDE_ORDER, key=lambda side: (-side_scores[side], SIDE_ORDER.index(side)))
    visible_sides = [
        side
        for side in side_rank
        if side_scores[side] >= min_visible_side_score and side_counts[side] > 0
    ]
    side_set = set(visible_sides)

    lengths = [float(stat["length"]) for stat in analysis_lines]
    directness_values = [float(stat["directness"]) for stat in analysis_lines]
    turning_values = [float(stat["turning"]) for stat in analysis_lines]
    mean_directness = weighted_average(directness_values, lengths)
    mean_turning = weighted_average(turning_values, lengths)

    straight_candidates = [
        stat
        for stat in analysis_lines
        if float(stat["length"]) >= min_major_length
        and float(stat["directness"]) >= 0.93
        and float(stat["turning"]) <= 0.16
    ]
    orientation_angles = [line_orientation(stat["points"]) for stat in straight_candidates]
    orientation_weights = [float(stat["length"]) for stat in straight_candidates]
    parallel_spread = orientation_spread(orientation_angles, orientation_weights)
    parallel_corridor = bool(len(straight_candidates) >= 2 and parallel_spread <= 0.10)
    dominant_axis = dominant_orientation(orientation_angles, orientation_weights)

    off_axis_center_lines = 0
    off_axis_center_support = 0.0
    off_axis_total_support = 0.0
    off_axis_major_lines = 0
    axis_aligned_support = 0.0
    off_axis_side_set = set()
    center_connector_lines = 0
    center_connector_support = 0.0
    center_connector_internal_lines = 0
    center_bridge_lines = 0
    center_bridge_support = 0.0
    if len(straight_candidates) >= 2:
        for stat in analysis_lines:
            length = float(stat["length"])
            if length < min_support_length * 0.80:
                continue
            angle = line_orientation(stat["points"])
            axis_gap = orientation_distance(angle, dominant_axis)
            if axis_gap <= 0.20:
                axis_aligned_support += length
            non_opposite_span = bool(len(stat["sides"]) >= 2 and not any(
                side_pair_is_opposite(side_a, side_b)
                for side_a in stat["sides"]
                for side_b in stat["sides"]
                if side_a != side_b
            ))
            connector_like = bool(
                bool(stat["touches_center"])
                and length >= min_support_length * 0.75
                and (
                    axis_gap >= 0.24
                    or float(stat["turning"]) >= 0.10
                    or float(stat["directness"]) <= 0.92
                )
                and (
                    len(stat["sides"]) <= 1
                    or non_opposite_span
                    or axis_gap >= 0.42
                )
            )
            if connector_like:
                center_connector_lines += 1
                center_connector_support += length
                if len(stat["sides"]) <= 1:
                    center_connector_internal_lines += 1
            center_bridge = bool(
                bool(stat["touches_center"])
                and len(stat["sides"]) == 0
                and length >= min_support_length
                and axis_gap >= 0.16
                and float(stat["turning"]) >= 0.18
            )
            if center_bridge:
                center_bridge_lines += 1
                center_bridge_support += length
            off_axis = bool(
                axis_gap >= 0.42
                or (
                    bool(stat["touches_tight_center"])
                    and non_opposite_span
                    and (float(stat["turning"]) >= 0.08 or float(stat["directness"]) <= 0.92)
                )
            )
            if not off_axis:
                continue
            off_axis_total_support += length
            off_axis_side_set.update(str(side) for side in stat["sides"])
            if length >= min_major_length * 0.85:
                off_axis_major_lines += 1
            if bool(stat["touches_tight_center"]):
                off_axis_center_lines += 1
                off_axis_center_support += length

    center_crossing_lines = sum(
        1
        for stat in analysis_lines
        if bool(stat["touches_tight_center"]) and (len(stat["sides"]) >= 2 or float(stat["length"]) >= min_major_length)
    )
    loop_like_lines = sum(1 for stat in analysis_lines if bool(stat["loop_like"]))
    opposite_pair = pick_opposite_side_pair(visible_sides, side_scores)
    connector_veto = bool(
        len(straight_candidates) >= 2
        and center_connector_lines >= 1
        and center_connector_support >= max(min_support_length * 0.95, axis_aligned_support * 0.04)
        and (
            center_connector_internal_lines >= 1
            or central_internal_endpoints >= 1
            or off_axis_center_lines >= 1
            or center_crossing_lines >= 3
        )
    )
    center_bridge_veto = bool(
        len(straight_candidates) >= 2
        and center_bridge_lines >= 1
        and center_bridge_support >= min_support_length
        and parallel_spread <= 0.08
    )
    straight_veto = bool(
        len(straight_candidates) >= 2
        and (
            (
                off_axis_major_lines >= 2
                and off_axis_total_support >= max(min_major_length * 1.35, axis_aligned_support * 0.30)
            )
            or
            off_axis_center_lines >= 1
            and off_axis_center_support >= min_support_length * 0.90
            and len(off_axis_side_set) >= 2
            or connector_veto
            or center_bridge_veto
        )
    )

    num_lines = len(visible_lines)
    side_count = len(side_set)
    major_side_count = len(major_side_set)

    def finish(scene_label: str, caption_text: str) -> Dict[str, Any]:
        return {
            "scene_label": str(scene_label).strip().lower(),
            "caption_text": str(caption_text),
            "num_visible_lines": int(num_lines),
            "num_analysis_lines": int(len(analysis_lines)),
            "visible_sides": list(visible_sides),
            "side_count": int(side_count),
            "major_side_set": sorted(str(side) for side in major_side_set),
            "major_side_count": int(major_side_count),
            "side_scores": {str(side): float(score) for side, score in side_scores.items()},
            "side_counts": {str(side): int(count) for side, count in side_counts.items()},
            "mean_directness": float(mean_directness),
            "mean_turning": float(mean_turning),
            "straight_candidate_count": int(len(straight_candidates)),
            "parallel_spread": float(parallel_spread),
            "parallel_corridor": bool(parallel_corridor),
            "dominant_axis": None if dominant_axis is None else float(dominant_axis),
            "off_axis_center_lines": int(off_axis_center_lines),
            "off_axis_center_support": float(off_axis_center_support),
            "off_axis_total_support": float(off_axis_total_support),
            "off_axis_major_lines": int(off_axis_major_lines),
            "axis_aligned_support": float(axis_aligned_support),
            "off_axis_side_set": sorted(str(side) for side in off_axis_side_set),
            "center_connector_lines": int(center_connector_lines),
            "center_connector_support": float(center_connector_support),
            "center_connector_internal_lines": int(center_connector_internal_lines),
            "center_bridge_lines": int(center_bridge_lines),
            "center_bridge_support": float(center_bridge_support),
            "center_crossing_lines": int(center_crossing_lines),
            "loop_like_lines": int(loop_like_lines),
            "central_internal_endpoints": int(central_internal_endpoints),
            "connector_veto": bool(connector_veto),
            "center_bridge_veto": bool(center_bridge_veto),
            "straight_veto": bool(straight_veto),
            "opposite_pair": None if opposite_pair is None else [str(opposite_pair[0]), str(opposite_pair[1])],
            "min_support_length": float(min_support_length),
            "min_major_length": float(min_major_length),
        }

    if side_count >= 3 and central_internal_endpoints >= 2 and center_crossing_lines == 0:
        return finish(
            "intersection-approach",
            "An intersection-approach road-structure patch with visible branches on the "
            f"{format_side_list(visible_sides)}. The structures stop before the central gap.",
        )

    if side_count >= 3:
        return finish(
            "branching",
            f"A branching road-structure patch with visible branches on the {format_side_list(visible_sides)}.",
        )

    if connector_veto:
        return finish(
            "branching",
            "A branching road-structure patch with connector branches between the main corridors.",
        )

    if center_bridge_veto:
        return finish(
            "branching",
            "A branching road-structure patch with a central connector between the main corridors.",
        )

    if straight_veto and major_side_count >= 3:
        return finish(
            "branching",
            f"A branching road-structure patch with visible branches on the {format_side_list(major_side_set)}.",
        )

    top_two = visible_sides[:2]
    if loop_like_lines >= 2:
        return finish(
            "complex",
            "A complex road-structure patch with loop-like road structures.",
        )

    if parallel_corridor and not straight_veto:
        if opposite_pair is not None:
            side_a, side_b = opposite_pair
            return finish(
                "straight",
                f"A straight road-structure patch with parallel branches running between {side_a} and {side_b}.",
            )
        if visible_sides:
            return finish(
                "straight",
                f"A straight road-structure patch with parallel corridor-like branches near the {format_side_list(visible_sides[:2])}.",
            )
        return finish(
            "straight",
            "A straight road-structure patch with parallel corridor-like branches.",
        )

    if len(top_two) == 2 and not side_pair_is_opposite(top_two[0], top_two[1]) and mean_turning >= 0.10:
        return finish(
            "curved",
            f"A curved road-structure patch with branches curving between {top_two[0]} and {top_two[1]}.",
        )

    if mean_directness >= 0.90 and num_lines >= 2 and side_count <= 2 and not straight_veto:
        if opposite_pair is not None:
            return finish(
                "straight",
                f"A straight road-structure patch with parallel branches running between {opposite_pair[0]} and {opposite_pair[1]}.",
            )
        if visible_sides:
            return finish(
                "straight",
                f"A straight road-structure patch with parallel corridor-like branches near the {format_side_list(visible_sides[:2])}.",
            )

    if loop_like_lines > 0:
        return finish(
            "complex",
            "A complex road-structure patch with loop-like road structures.",
        )

    if visible_sides:
        return finish(
            "complex",
            f"A complex road-structure patch with road structures near the {format_side_list(visible_sides)}.",
        )
    return finish(
        "complex",
        "A complex road-structure patch with internal road structures.",
    )


def build_caption_short(
    *,
    structure_lines: Sequence[Dict[str, Any]],
    centerline_lines: Sequence[Dict[str, Any]],
    patch_size: int,
    border_tol_px: float,
) -> tuple[str, str]:
    analysis = analyze_scene_caption_short(
        structure_lines=structure_lines,
        centerline_lines=centerline_lines,
        patch_size=int(patch_size),
        border_tol_px=float(border_tol_px),
    )
    return (
        str(analysis["scene_label"]).strip().lower(),
        str(analysis["caption_text"]),
    )


def build_scene_grid_caption(
    *,
    sample_id: str,
    label_map_path: Path,
    structure_lines: Sequence[Dict[str, Any]],
    centerline_lines: Sequence[Dict[str, Any]],
    patch_size: int,
    border_tol_px: float,
    user_prompt_prefix: str = DEFAULT_USER_PROMPT,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> Dict[str, Any]:
    scene_label, _ = build_caption_short(
        structure_lines=structure_lines,
        centerline_lines=centerline_lines,
        patch_size=int(patch_size),
        border_tol_px=float(border_tol_px),
    )
    _ = sample_id
    normalized_grid_states = classify_grid_states_from_label_map(
        label_map_path=Path(label_map_path).resolve(),
        patch_size=int(patch_size),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    user_prompt = build_grid_user_prompt(
        user_prompt_prefix=str(user_prompt_prefix),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    caption_text = build_grid_caption_text(scene_label, normalized_grid_states)
    return {
        "scene_label": str(scene_label).strip().lower(),
        "caption_text": str(caption_text),
        "user_prompt": str(user_prompt),
        "grid_states": list(normalized_grid_states),
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "schema_version": GRID_SCHEMA_VERSION,
    }


class RCCaptionShortFormatter:
    def __init__(
        self,
        *,
        image_size: int,
        num_visual_tokens: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> None:
        self.image_size = int(image_size)
        self.num_visual_tokens = int(num_visual_tokens)
        self.system_prompt = str(system_prompt).strip()
        self.user_prompt = str(user_prompt).strip()
        self.special_tokens = list(VISUAL_TOKENS)

    def register_tokens(self, tokenizer: Any) -> int:
        vocab = tokenizer.get_vocab()
        new_tokens = [tok for tok in self.special_tokens if tok not in vocab]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return len(new_tokens)

    def build_user_text(self, user_prompt: str | None = None) -> str:
        prompt_body = str(user_prompt).strip() if user_prompt is not None else str(self.user_prompt).strip()
        # 真正的视觉输入仍由 <vis_patch> 序列承接，user prompt 只负责声明任务。
        return f"{build_visual_placeholder(self.num_visual_tokens)}\n{prompt_body}"

    def apply_chat_template(
        self,
        tokenizer: Any,
        *,
        system_text: str,
        user_text: str,
        assistant_text: str | None,
        add_generation_prompt: bool,
    ) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            conv = [
                {"role": "system", "content": str(system_text)},
                {"role": "user", "content": str(user_text)},
            ]
            if assistant_text is not None:
                conv.append({"role": "assistant", "content": str(assistant_text)})
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": bool(add_generation_prompt),
            }
            for extra_kwargs in ({"enable_thinking": False}, {"thinking": False}, {}):
                try:
                    # Qwen3 "non-thinking" chat templates intentionally inject an empty
                    # <think>...</think> scaffold before the visible assistant answer.
                    # Treat that as the preferred render instead of falling back to the
                    # plain text format, otherwise train/infer prompts drift apart.
                    return tokenizer.apply_chat_template(conv, **template_kwargs, **extra_kwargs)
                except TypeError:
                    continue
        pieces = [
            f"System:\n{system_text}",
            f"User:\n{user_text}",
        ]
        if assistant_text is not None:
            pieces.append(f"Assistant:\n{assistant_text}")
        elif add_generation_prompt:
            pieces.append("Assistant:\n")
        return "\n\n".join(pieces)


@dataclass
class RawRCCaptionSample:
    sample_id: str
    image_path: Path
    pixel_values: torch.Tensor
    prompt_text: str
    full_text: str
    caption_short: str
    caption_label: str


class RCCaptionShortDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        meta_rows: Sequence[Dict[str, Any]] | None,
        media_dir: Path,
        tokenizer: Any,
        formatter: RCCaptionShortFormatter,
        image_size: int,
        border_tol_px: float = 18.0,
    ) -> None:
        self.rows = list(rows)
        self.meta_by_id = index_rows_by_id(meta_rows)
        self.media_dir = Path(media_dir)
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.image_size = int(image_size)
        self.border_tol_px = float(border_tol_px)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RawRCCaptionSample:
        sample = self.rows[index]
        sample_id = str(sample.get("id", index))
        meta = self.meta_by_id.get(sample_id, {})
        rel_image = str(sample.get("images", [""])[0] if sample.get("images") else meta.get("image", ""))
        image_path = (self.media_dir / rel_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for sample {sample_id}: {image_path}")

        with Image.open(image_path) as img:
            pixel_values = pil_to_tensor(img, image_size=self.image_size)

        system_text, user_prompt = extract_prompt_texts(sample, meta)
        caption_label, caption_short = extract_caption_from_sample(sample, meta)
        schema_version = str(meta.get("caption_schema_version", "")).strip()
        uses_grid_schema = bool(
            schema_version == GRID_SCHEMA_VERSION
            and is_grid_schema_caption_text(caption_short)
        )
        if not uses_grid_schema:
            # 如果 jsonl 里没有提前导好的 caption，就从分割标签在线重建 Scene + GridStates。
            structure_lines, centerline_lines = extract_lines_from_meta(meta, media_dir=self.media_dir)
            label_map_path = resolve_structure_multiclass_path(
                meta,
                self.media_dir,
                fallback_image_path=image_path,
            )
            grid_package = build_scene_grid_caption(
                sample_id=str(sample_id),
                label_map_path=label_map_path,
                structure_lines=structure_lines,
                centerline_lines=centerline_lines,
                patch_size=self.image_size,
                border_tol_px=self.border_tol_px,
                user_prompt_prefix=str(user_prompt or self.formatter.user_prompt),
                grid_rows=int(meta.get("caption_grid_rows", DEFAULT_GRID_ROWS)),
                grid_cols=int(meta.get("caption_grid_cols", DEFAULT_GRID_COLS)),
            )
            caption_label = str(grid_package["scene_label"])
            caption_short = str(grid_package["caption_text"])
            user_prompt = str(grid_package["user_prompt"])
            system_text = str(self.formatter.system_prompt)

        user_text = self.formatter.build_user_text(user_prompt or self.formatter.user_prompt)
        prompt_text = self.formatter.apply_chat_template(
            self.tokenizer,
            system_text=system_text or self.formatter.system_prompt,
            user_text=user_text,
            assistant_text=None,
            add_generation_prompt=True,
        )
        full_text = self.formatter.apply_chat_template(
            self.tokenizer,
            system_text=system_text or self.formatter.system_prompt,
            user_text=user_text,
            assistant_text=caption_short,
            add_generation_prompt=False,
        )
        return RawRCCaptionSample(
            sample_id=sample_id,
            image_path=image_path,
            pixel_values=pixel_values,
            prompt_text=str(prompt_text),
            full_text=str(full_text),
            caption_short=str(caption_short),
            caption_label=str(caption_label),
        )


class RCCaptionShortCollator:
    def __init__(
        self,
        *,
        tokenizer: Any,
        cutoff_len: int,
        num_visual_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.cutoff_len = int(cutoff_len)
        self.num_visual_tokens = int(num_visual_tokens)
        self.vis_patch_token_id = int(tokenizer.convert_tokens_to_ids("<vis_patch>"))
        if self.vis_patch_token_id < 0:
            raise ValueError("Tokenizer is missing <vis_patch>.")

    def __call__(self, features: Sequence[RawRCCaptionSample]) -> Dict[str, torch.Tensor]:
        prompt_texts = [item.prompt_text for item in features]
        full_texts = [item.full_text for item in features]
        pixel_values = torch.stack([item.pixel_values for item in features], dim=0)

        full_batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )
        prompt_batch = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )

        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        # 只让 assistant 的结构化 caption 参与 CE；system/user 以及视觉占位 token 全部屏蔽。
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        vis_patch_mask = input_ids.eq(self.vis_patch_token_id)

        for batch_idx, _item in enumerate(features):
            full_len = int(attention_mask[batch_idx].sum().item())
            prompt_text_len = int(prompt_batch["attention_mask"][batch_idx].sum().item())
            full_text_len = int(full_batch["attention_mask"][batch_idx].sum().item())
            prompt_len = max(0, full_len - max(0, full_text_len - prompt_text_len))
            labels[batch_idx, :prompt_len] = -100
            num_vis = int(vis_patch_mask[batch_idx].sum().item())
            if num_vis != self.num_visual_tokens:
                raise ValueError(
                    f"Visual token mismatch for sample={features[batch_idx].sample_id}: "
                    f"expected={self.num_visual_tokens} actual={num_vis}"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "vis_patch_mask": vis_patch_mask,
        }
