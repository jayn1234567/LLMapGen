from __future__ import annotations

import json
from typing import Any


COORD_MODE_PIXEL = "pixel"
COORD_MODE_NORM1000 = "norm1000"
DEFAULT_COORD_RANGE = 1000


def normalize_coord_mode(coord_mode: str | None) -> str:
    mode = str(coord_mode or COORD_MODE_PIXEL).strip().lower()
    if mode in {"pixel", "pixels", "patch_pixel", "patch_local", "patch_local_pixel"}:
        return COORD_MODE_PIXEL
    if mode in {"norm", "normalized", "norm1000", "normalized_1000", "patch_norm1000"}:
        return COORD_MODE_NORM1000
    if "norm1000" in mode or "normalized_1000" in mode:
        return COORD_MODE_NORM1000
    return mode


def coord_system_name(coord_mode: str, patch_size: int, coord_range: int = DEFAULT_COORD_RANGE) -> str:
    mode = normalize_coord_mode(coord_mode)
    if mode == COORD_MODE_NORM1000:
        return f"patch_norm{coord_range}"
    return f"patch_local_{patch_size}"


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def pixel_point_to_coord(
    point: list | tuple,
    patch_width: int,
    patch_height: int | None = None,
    coord_mode: str = COORD_MODE_NORM1000,
    coord_range: int = DEFAULT_COORD_RANGE,
    clamp: bool = False,
) -> list[int]:
    mode = normalize_coord_mode(coord_mode)
    height = patch_height if patch_height is not None else patch_width
    x, y = float(point[0]), float(point[1])
    if mode == COORD_MODE_PIXEL:
        if clamp:
            return [clamp_int(x, 0, patch_width - 1), clamp_int(y, 0, height - 1)]
        return [int(round(x)), int(round(y))]

    if mode != COORD_MODE_NORM1000:
        raise ValueError(f"unsupported coord_mode: {coord_mode}")
    x_out = int(round(x / max(patch_width - 1, 1) * coord_range))
    y_out = int(round(y / max(height - 1, 1) * coord_range))
    if clamp:
        x_out = max(0, min(coord_range, x_out))
        y_out = max(0, min(coord_range, y_out))
    return [x_out, y_out]


def coord_point_to_pixel(
    point: list | tuple,
    patch_width: int,
    patch_height: int | None = None,
    coord_mode: str = COORD_MODE_NORM1000,
    coord_range: int = DEFAULT_COORD_RANGE,
    clamp: bool = True,
) -> list[int]:
    mode = normalize_coord_mode(coord_mode)
    height = patch_height if patch_height is not None else patch_width
    x, y = float(point[0]), float(point[1])
    if mode == COORD_MODE_PIXEL:
        if clamp:
            return [clamp_int(x, 0, patch_width - 1), clamp_int(y, 0, height - 1)]
        return [int(round(x)), int(round(y))]

    if mode != COORD_MODE_NORM1000:
        raise ValueError(f"unsupported coord_mode: {coord_mode}")
    x_out = int(round(x / max(coord_range, 1) * max(patch_width - 1, 1)))
    y_out = int(round(y / max(coord_range, 1) * max(height - 1, 1)))
    if clamp:
        x_out = max(0, min(patch_width - 1, x_out))
        y_out = max(0, min(height - 1, y_out))
    return [x_out, y_out]


def convert_points(
    points: list,
    from_mode: str,
    to_mode: str,
    patch_width: int,
    patch_height: int | None = None,
    coord_range: int = DEFAULT_COORD_RANGE,
    clamp: bool = True,
) -> list[list[int]]:
    from_mode = normalize_coord_mode(from_mode)
    to_mode = normalize_coord_mode(to_mode)
    if from_mode == to_mode:
        converted = [[int(round(point[0])), int(round(point[1]))] for point in points]
        if clamp and to_mode == COORD_MODE_PIXEL:
            height = patch_height if patch_height is not None else patch_width
            return [
                [max(0, min(patch_width - 1, x)), max(0, min(height - 1, y))]
                for x, y in converted
            ]
        if clamp and to_mode == COORD_MODE_NORM1000:
            return [
                [max(0, min(coord_range, x)), max(0, min(coord_range, y))]
                for x, y in converted
            ]
        return converted
    if from_mode == COORD_MODE_PIXEL and to_mode == COORD_MODE_NORM1000:
        return [
            pixel_point_to_coord(point, patch_width, patch_height, to_mode, coord_range, clamp=clamp)
            for point in points
        ]
    if from_mode == COORD_MODE_NORM1000 and to_mode == COORD_MODE_PIXEL:
        return [
            coord_point_to_pixel(point, patch_width, patch_height, from_mode, coord_range, clamp=clamp)
            for point in points
        ]
    raise ValueError(f"unsupported coord conversion: {from_mode} -> {to_mode}")


def convert_items(
    items: list[dict[str, Any]],
    from_mode: str,
    to_mode: str,
    patch_width: int,
    patch_height: int | None = None,
    coord_range: int = DEFAULT_COORD_RANGE,
    clamp: bool = True,
) -> list[dict[str, Any]]:
    converted = []
    for item in items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        new_item = dict(item)
        points = new_item.get("points")
        if isinstance(points, list):
            new_item["points"] = convert_points(
                points,
                from_mode,
                to_mode,
                patch_width,
                patch_height,
                coord_range=coord_range,
                clamp=clamp,
            )
        converted.append(new_item)
    return converted


def normalize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        return {"lines": payload["lines"]}
    if isinstance(payload, list):
        return {"lines": payload}
    return {"lines": []}


def load_json_payload(text_or_payload: Any) -> dict[str, Any]:
    if isinstance(text_or_payload, (dict, list)):
        return normalize_payload(text_or_payload)
    text = str(text_or_payload or "").strip()
    payload = json.loads(text)
    return normalize_payload(payload)


def convert_payload(
    payload: Any,
    from_mode: str,
    to_mode: str,
    patch_width: int,
    patch_height: int | None = None,
    coord_range: int = DEFAULT_COORD_RANGE,
    clamp: bool = True,
) -> dict[str, Any]:
    normalized = load_json_payload(payload)
    return {
        "lines": convert_items(
            normalized["lines"],
            from_mode,
            to_mode,
            patch_width,
            patch_height,
            coord_range=coord_range,
            clamp=clamp,
        )
    }


def payload_to_text(payload: Any) -> str:
    return json.dumps(normalize_payload(payload), ensure_ascii=False, separators=(",", ":"))


def convert_payload_text(
    text: str,
    from_mode: str,
    to_mode: str,
    patch_width: int,
    patch_height: int | None = None,
    coord_range: int = DEFAULT_COORD_RANGE,
    clamp: bool = True,
) -> str:
    payload = convert_payload(
        text,
        from_mode,
        to_mode,
        patch_width,
        patch_height,
        coord_range=coord_range,
        clamp=clamp,
    )
    return payload_to_text(payload)


def record_coord_config(
    record: dict[str, Any] | None,
    default_mode: str = COORD_MODE_PIXEL,
    default_patch_size: int = 256,
    default_coord_range: int = DEFAULT_COORD_RANGE,
) -> dict[str, Any]:
    record = record or {}
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    coord_mode = record.get("coord_mode") or meta.get("coord_mode")
    if coord_mode is None:
        coord_mode = COORD_MODE_NORM1000 if "norm" in str(meta.get("coord_system", "")).lower() else default_mode
    patch_size = int(record.get("patch_size") or meta.get("patch_size") or meta.get("pixel_patch_size") or default_patch_size)
    patch_width = int(record.get("patch_width") or meta.get("patch_width") or patch_size)
    patch_height = int(record.get("patch_height") or meta.get("patch_height") or patch_size)
    coord_range = int(record.get("coord_range") or meta.get("coord_range") or default_coord_range)
    return {
        "coord_mode": normalize_coord_mode(coord_mode),
        "coord_range": coord_range,
        "patch_size": patch_size,
        "patch_width": patch_width,
        "patch_height": patch_height,
    }
