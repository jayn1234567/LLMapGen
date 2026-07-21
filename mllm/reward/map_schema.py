from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from mllm.coord_utils import COORD_MODE_NORM1000, COORD_MODE_PIXEL, DEFAULT_COORD_RANGE, normalize_coord_mode
from mllm.coordinate_tokens import decode_coordinate_tokens


@dataclass
class MapParseResult:
    ok: bool
    items: list[dict[str, Any]]
    payload_text: str
    error: str | None = None


def extract_json_payload(text: str) -> str:
    text = str(text or "").strip()
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return text
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[start:idx + 1].strip()
    return text[start:].strip()


def _validate_points(points: Any, patch_size: int, coord_mode: str = COORD_MODE_PIXEL, coord_range: int = DEFAULT_COORD_RANGE):
    if not isinstance(points, list) or not points:
        raise ValueError("missing points")
    clean_points = []
    mode = normalize_coord_mode(coord_mode)
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid point format")
        if not all(isinstance(v, (int, float)) for v in point):
            raise ValueError("point coordinates must be numeric")
        x = int(round(point[0]))
        y = int(round(point[1]))
        if mode == COORD_MODE_NORM1000:
            if x < 0 or y < 0 or x > coord_range or y > coord_range:
                raise ValueError(f"point out of normalized patch bounds: {[x, y]}")
        else:
            if x < 0 or y < 0 or x >= patch_size or y >= patch_size:
                raise ValueError(f"point out of patch bounds: {[x, y]}")
        clean_points.append([x, y])
    return clean_points


def _optional_semantic_type(item: dict[str, Any], primary_key: str) -> str | None:
    value = item.get(primary_key)
    if value is None:
        value = item.get("type")
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def parse_map_json(
    prediction_text: str,
    map_task: str = "lane",
    patch_size: int = 256,
    coord_mode: str = COORD_MODE_PIXEL,
    coord_range: int = DEFAULT_COORD_RANGE,
) -> MapParseResult:
    payload_text = extract_json_payload(
        decode_coordinate_tokens(prediction_text, max_coordinate=coord_range)
    )
    try:
        parsed = json.loads(payload_text)
        if isinstance(parsed, list):
            parsed_items = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("lines"), list):
            parsed_items = parsed["lines"]
        else:
            raise ValueError("prediction must be a JSON list or an object with lines")

        normalized = []
        allow_intersection = map_task in {"lane_intersection", "intersection", "all"}
        for item in parsed_items:
            if not isinstance(item, dict):
                raise ValueError("prediction item is not an object")
            category = str(item.get("category", "centerline")).strip()
            category_lower = "centerline" if category == "CenterLine" else category.lower()
            if category_lower not in {"centerline", "intersection"}:
                raise ValueError(f"unsupported category: {category}")
            if category_lower == "intersection" and not allow_intersection:
                raise ValueError("intersection output is not allowed for lane task")

            clean_points = _validate_points(item.get("points"), patch_size, coord_mode=coord_mode, coord_range=coord_range)
            if category_lower == "centerline":
                start_type = item.get("start_type", "inside")
                end_type = item.get("end_type", "inside")
                if start_type not in {"cut", "inside"} or end_type not in {"cut", "inside"}:
                    raise ValueError("centerline start_type/end_type must be cut or inside")
                normalized_item = {
                    "category": "centerline",
                    "start_type": start_type,
                    "end_type": end_type,
                    "points": clean_points,
                }
                lane_type = _optional_semantic_type(item, "lane_type")
                if lane_type is not None:
                    normalized_item["lane_type"] = lane_type
                normalized.append(normalized_item)
            else:
                is_cut = item.get("is_cut", False)
                if not isinstance(is_cut, bool):
                    raise ValueError("intersection is_cut must be a boolean")
                normalized_item = {
                    "category": "intersection",
                    "is_cut": is_cut,
                    "points": clean_points,
                }
                intersection_type = _optional_semantic_type(item, "intersection_type")
                if intersection_type is not None:
                    normalized_item["intersection_type"] = intersection_type
                normalized.append(normalized_item)
        return MapParseResult(ok=True, items=normalized, payload_text=json.dumps({"lines": normalized}, ensure_ascii=False))
    except Exception as exc:
        return MapParseResult(ok=False, items=[], payload_text=payload_text, error=str(exc))
