#!/usr/bin/env python3
"""Convert MLLM patch predictions to the RC inter512 patch-result layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


TYPE_TO_LABEL = {
    "common": "1_1",
    "t_intersection": "1_2",
    "small_untyped": "3_0",
    "t_lane_change_area": "4_1",
    "other": "0_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--e2e-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument(
        "--result-subdir",
        default="inter512/tif_512_256",
        help="Path below rc_one_patch_release/center_line_v2.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--collapse-type-to-one",
        action="store_true",
        help=(
            "Set every predicted IntersectionType to 1 for geometry-only original-engine "
            "evaluation. Explicit T intersections retain subtype 2; all others use subtype 1."
        ),
    )
    return parser.parse_args()


def load_json_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {"lines": []}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("prediction payload is not an object")
    return payload


def finite_points(value: Any) -> list[list[float]]:
    points: list[list[float]] = []
    if not isinstance(value, list):
        return points
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append([x, y])
    return points


def prediction_lines(record: dict[str, Any], window_size: int, coord_range: int) -> list[dict[str, Any]]:
    normalized_text = record.get("prediction_json") or record.get("prediction")
    if isinstance(normalized_text, (str, dict)) and normalized_text:
        normalized = load_json_value(normalized_text)
        scale = float(window_size) / float(coord_range)
        converted: list[dict[str, Any]] = []
        for item in normalized.get("lines", []):
            if not isinstance(item, dict):
                continue
            clone = dict(item)
            clone["points"] = [[x * scale, y * scale] for x, y in finite_points(item.get("points"))]
            converted.append(clone)
        return converted

    pixel_text = record.get("prediction_json_pixel")
    return list(load_json_value(pixel_text).get("lines", []))


def record_identity(record: dict[str, Any]) -> tuple[str, str, int, int, int, int]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    scene_id = str(meta.get("scene_id") or "").strip()
    tif_prefix = str(meta.get("tif_prefix") or "").strip()
    row = record.get("row", meta.get("row", meta.get("patch_row")))
    col = record.get("col", meta.get("col", meta.get("patch_col")))
    x0 = record.get("x0", meta.get("x0"))
    y0 = record.get("y0", meta.get("y0"))

    image = Path(str(record.get("image") or "").replace("\\", "/"))
    parts = image.parts
    if not scene_id and len(parts) >= 4 and "images" in parts:
        image_index = parts.index("images")
        if len(parts) > image_index + 1:
            scene_id = parts[image_index + 1]
    if not tif_prefix and len(parts) >= 2:
        tif_prefix = parts[-2].split("_", 1)[0]
    if (row is None or col is None) and image.stem:
        pieces = image.stem.split("_")
        if len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
            row, col = int(pieces[0]), int(pieces[1])

    if not scene_id or not tif_prefix or row is None or col is None:
        raise ValueError(
            f"unable to resolve scene/tif/row/col for record {record.get('record_id') or record.get('id')}"
        )
    row = int(row)
    col = int(col)
    return scene_id, tif_prefix, row, col, int(x0) if x0 is not None else -1, int(y0) if y0 is not None else -1


def empty_or_reset_result_dirs(e2e_root: Path, result_subdir: Path) -> int:
    removed = 0
    for scene in e2e_root.iterdir():
        target = scene / "rc_one_patch_release" / "center_line_v2" / result_subdir
        if target.is_dir():
            shutil.rmtree(target)
            removed += 1
    return removed


def validate_relative_subdir(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"result subdirectory must stay below center_line_v2: {path}")
    return path


def evaluation_label(intersection_type: str, *, collapse_type_to_one: bool) -> str:
    if collapse_type_to_one:
        return "1_2" if intersection_type == "t_intersection" else "1_1"
    return TYPE_TO_LABEL.get(intersection_type, TYPE_TO_LABEL["other"])


def format_predictions(
    prediction_dir: Path,
    e2e_root: Path,
    report_json: Path,
    *,
    window_size: int,
    stride: int,
    coord_range: int,
    result_subdir: Path,
    reset: bool,
    strict: bool,
    collapse_type_to_one: bool = False,
) -> dict[str, Any]:
    prediction_dir = prediction_dir.resolve()
    e2e_root = e2e_root.resolve()
    if not prediction_dir.is_dir():
        raise FileNotFoundError(prediction_dir)
    if not e2e_root.is_dir():
        raise FileNotFoundError(e2e_root)
    if window_size <= 0 or stride <= 0 or coord_range <= 0:
        raise ValueError("window size, stride, and coordinate range must be positive")
    result_subdir = validate_relative_subdir(result_subdir)

    removed_dirs = empty_or_reset_result_dirs(e2e_root, result_subdir) if reset else 0
    counters: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    output_keys: set[tuple[str, str, int, int]] = set()

    for path in sorted(prediction_dir.glob("*.json")):
        counters["prediction_files"] += 1
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError("prediction file does not contain an object")
            scene_id, tif_prefix, row, col, x0, y0 = record_identity(record)
            expected_x0 = col * stride
            expected_y0 = row * stride
            if x0 >= 0 and x0 != expected_x0:
                raise ValueError(f"x0={x0}, expected col*stride={expected_x0}")
            if y0 >= 0 and y0 != expected_y0:
                raise ValueError(f"y0={y0}, expected row*stride={expected_y0}")
            scene = e2e_root / scene_id
            if not (scene / "rc_one_patch_release").is_dir():
                raise FileNotFoundError(f"scene not found below E2E root: {scene_id}")

            output_key = (scene_id, tif_prefix, row, col)
            if output_key in output_keys:
                raise ValueError(f"duplicate patch identity: {output_key}")
            output_keys.add(output_key)

            intersections: list[dict[str, Any]] = []
            if record.get("parse_ok") is False:
                lines: list[dict[str, Any]] = []
                counters["prediction_parse_failures"] += 1
            else:
                try:
                    lines = prediction_lines(record, window_size, coord_range)
                except (TypeError, ValueError, json.JSONDecodeError):
                    lines = []
                    counters["prediction_parse_failures"] += 1
            for item in lines:
                if not isinstance(item, dict) or str(item.get("category", "")).lower() != "intersection":
                    continue
                counters["intersection_items"] += 1
                points = finite_points(item.get("points"))
                if len(points) < 3:
                    counters["dropped_short_intersections"] += 1
                    continue
                if points[0] != points[-1]:
                    points.append(list(points[0]))
                raw_intersection_type = item.get("intersection_type")
                if raw_intersection_type is None or not str(raw_intersection_type).strip():
                    counters["missing_intersection_types"] += 1
                    intersection_type = "other"
                else:
                    intersection_type = str(raw_intersection_type).strip().lower()
                    counters[f"intersection_type:{intersection_type}"] += 1
                label = evaluation_label(
                    intersection_type,
                    collapse_type_to_one=collapse_type_to_one,
                )
                if intersection_type not in TYPE_TO_LABEL:
                    counters["unknown_intersection_types"] += 1
                if collapse_type_to_one:
                    counters["type_collapsed_to_one"] += 1
                intersections.append({"coords": points, "score": 1.0, "label": label})
                counters[f"label:{label}"] += 1

            output_dir = (
                scene
                / "rc_one_patch_release"
                / "center_line_v2"
                / result_subdir
                / f"{tif_prefix}_tif_res"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{row}_{col}.json"
            output_path.write_text(
                json.dumps({"intersection": intersections}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            counters["formatted_patches"] += 1
            counters["formatted_intersections"] += len(intersections)
        except Exception as exc:
            errors.append({"file": str(path), "error": repr(exc)})

    report = {
        "policy": (
            "Geometry uses original norm1000 predictions scaled by window_size/coord_range. "
            "A model parse failure becomes an empty intersection patch; identity, scene, duplicate, "
            "and stride-offset errors remain structural failures. "
            + (
                "All predicted IntersectionType values are collapsed to 1; explicit T intersections "
                "retain subtype 2 and every other prediction uses subtype 1."
                if collapse_type_to_one
                else "Intersection types retain their Dataset V2 semantic mapping."
            )
        ),
        "prediction_dir": str(prediction_dir),
        "e2e_root": str(e2e_root),
        "result_subdir": result_subdir.as_posix(),
        "window_size": window_size,
        "stride": stride,
        "coord_range": coord_range,
        "collapse_type_to_one": collapse_type_to_one,
        "removed_result_dirs": removed_dirs,
        "counts": dict(sorted(counters.items())),
        "errors": errors,
        "complete": not errors and counters["formatted_patches"] == counters["prediction_files"],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if strict and errors:
        raise RuntimeError(f"Failed to format {len(errors)} prediction files; inspect {report_json}")
    if counters["formatted_patches"] <= 0:
        raise RuntimeError("No intersection patch results were formatted")
    if counters["formatted_intersections"] <= 0:
        raise RuntimeError(
            f"No predicted intersections were formatted; inspect raw predictions and {report_json}"
        )
    return report


def main() -> None:
    args = parse_args()
    format_predictions(
        args.prediction_dir,
        args.e2e_root,
        args.report_json,
        window_size=args.window_size,
        stride=args.stride,
        coord_range=args.coord_range,
        result_subdir=Path(args.result_subdir),
        reset=args.reset,
        strict=args.strict,
        collapse_type_to_one=args.collapse_type_to_one,
    )


if __name__ == "__main__":
    main()
