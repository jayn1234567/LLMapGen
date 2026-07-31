#!/usr/bin/env python3
"""Audit E2E scene outputs and represent missing predictions as empty GeoJSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=0)
    parser.add_argument("--baseline-suffix", default="gt")
    parser.add_argument("--query-suffix", default="output_base")
    parser.add_argument("--fill-missing-predictions", action="store_true")
    return parser.parse_args()


def result_dirs(scene: Path, suffix: str) -> list[Path]:
    return sorted(
        (path for path in scene.iterdir() if path.is_dir() and path.name.endswith(suffix)),
        key=lambda path: path.name,
        reverse=True,
    )


def first_result_file(scene: Path, suffix: str, name: str) -> Path | None:
    return next(
        (directory / name for directory in result_dirs(scene, suffix) if (directory / name).is_file()),
        None,
    )


def query_directory(scene: Path, suffix: str) -> Path:
    candidates = result_dirs(scene, suffix)
    exact = scene / suffix
    if exact in candidates or exact.is_dir():
        return exact
    return candidates[0] if candidates else exact


def empty_feature_collection(template: Path | None) -> dict[str, Any]:
    payload: Any = None
    if template is not None and template.is_file():
        try:
            payload = json.loads(template.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            payload = None
    if not isinstance(payload, dict):
        payload = {"type": "FeatureCollection"}
    payload = dict(payload)
    payload["type"] = "FeatureCollection"
    payload["features"] = []
    return payload


def write_empty_geojson(path: Path, template: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(empty_feature_collection(template), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def audit_and_fill(
    root: Path,
    report_path: Path,
    *,
    expected_scenes: int,
    baseline_suffix: str,
    query_suffix: str,
    fill_missing_predictions: bool,
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    scenes = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "rc_one_patch_release").is_dir()
    )
    if expected_scenes > 0 and len(scenes) != expected_scenes:
        raise RuntimeError(f"Expected {expected_scenes} E2E scenes, found {len(scenes)} below {root}")

    missing_ground_truth: list[str] = []
    missing_prediction_before: list[str] = []
    created_empty_predictions: list[str] = []
    created_empty_intersections: list[str] = []

    for scene in scenes:
        gt_lane = first_result_file(scene, baseline_suffix, "Lane.geojson")
        gt_intersection = first_result_file(scene, baseline_suffix, "Intersection.geojson")
        pred_lane = first_result_file(scene, query_suffix, "Lane.geojson")
        if gt_lane is None:
            missing_ground_truth.append(scene.name)
            continue
        query_dir = pred_lane.parent if pred_lane is not None else query_directory(scene, query_suffix)
        if pred_lane is None:
            missing_prediction_before.append(scene.name)
            if fill_missing_predictions:
                write_empty_geojson(query_dir / "Lane.geojson", gt_lane)
                created_empty_predictions.append(scene.name)
                pred_lane = query_dir / "Lane.geojson"
        pred_intersection = query_dir / "Intersection.geojson"
        if pred_lane is not None and not pred_intersection.is_file() and fill_missing_predictions:
            write_empty_geojson(query_dir / "Intersection.geojson", gt_intersection)
            created_empty_intersections.append(scene.name)

    missing_prediction_after = [
        scene.name
        for scene in scenes
        if first_result_file(scene, query_suffix, "Lane.geojson") is None
    ]
    report = {
        "policy": (
            "The original evaluator silently skips scenes without Lane.geojson. Missing prediction "
            "outputs are represented as empty GeoJSON so they count as zero predictions; missing "
            "ground truth is never synthesized."
        ),
        "e2e_root": str(root),
        "scene_count": len(scenes),
        "expected_scenes": expected_scenes,
        "baseline_suffix": baseline_suffix,
        "query_suffix": query_suffix,
        "fill_missing_predictions": fill_missing_predictions,
        "missing_ground_truth": missing_ground_truth,
        "missing_prediction_before": missing_prediction_before,
        "created_empty_predictions": created_empty_predictions,
        "created_empty_intersections": created_empty_intersections,
        "missing_prediction_after": missing_prediction_after,
        "evaluable_scenes_after": len(scenes) - len(set(missing_ground_truth) | set(missing_prediction_after)),
    }
    report["complete"] = not missing_ground_truth and not missing_prediction_after
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if missing_ground_truth:
        raise RuntimeError(f"Missing lane ground truth for scenes: {missing_ground_truth}")
    if missing_prediction_after:
        raise RuntimeError(f"Missing prediction Lane.geojson for scenes: {missing_prediction_after}")
    return report


def main() -> None:
    args = parse_args()
    audit_and_fill(
        args.e2e_root,
        args.report_json,
        expected_scenes=args.expected_scenes,
        baseline_suffix=args.baseline_suffix,
        query_suffix=args.query_suffix,
        fill_missing_predictions=args.fill_missing_predictions,
    )


if __name__ == "__main__":
    main()
