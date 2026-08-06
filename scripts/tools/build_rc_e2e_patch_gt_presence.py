#!/usr/bin/env python3
"""Build the minimal per-patch GT-presence reference needed by E2E filtering."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.evaluate_rc_e2e_patch_metrics import (
    SceneLaneIndex,
    build_inter_tif_index,
    expected_lane_tif,
    find_lane_gt,
    load_prediction_records,
    parse_int_set,
    patch_ground_truth,
    record_row_col,
    record_scene_and_tif,
    scene_root_for_inter_tif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-e2e-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--baseline-name", default="gt")
    parser.add_argument("--gt-crs", default="EPSG:4326")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--ignore-lane-types", default="3,4,22")
    parser.add_argument(
        "--require-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any prediction record cannot be paired with E2E ground truth.",
    )
    return parser.parse_args()


def build_gt_presence(
    raw_e2e_root: Path,
    prediction_dir: Path,
    output_jsonl: Path,
    report_json: Path,
    *,
    baseline_name: str,
    gt_crs: str,
    patch_size: int,
    ignored_lane_types: set[int],
    require_all: bool,
) -> dict[str, Any]:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    predictions, json_files_seen = load_prediction_records(prediction_dir)
    if not predictions:
        raise FileNotFoundError(f"No raw per-patch prediction JSON found below {prediction_dir}")

    inter_index = build_inter_tif_index(raw_e2e_root)
    raster_cache: dict[Path, dict[str, Any]] = {}
    scene_lane_cache: dict[tuple[Path, str], SceneLaneIndex] = {}
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scene_counts: dict[str, int] = defaultdict(int)
    positive_patches = 0

    import rasterio

    for index, record in enumerate(predictions, start=1):
        record_id = str(record.get("record_id") or record.get("id") or f"sample_{index}")
        stage = "resolve_record"
        try:
            scene_id, tif_stem = record_scene_and_tif(record)
            stage = "resolve_inter_tif"
            inter_tif = inter_index[(scene_id, tif_stem)]
            lane_tif = expected_lane_tif(inter_tif)
            if not lane_tif.is_file():
                raise FileNotFoundError(f"Lane TIF not found: {lane_tif}")

            stage = "read_lane_tif"
            if lane_tif not in raster_cache:
                with rasterio.open(lane_tif) as source:
                    if source.crs is None:
                        raise ValueError(f"Lane TIF has no CRS: {lane_tif}")
                    raster_cache[lane_tif] = {
                        "transform": source.transform,
                        "crs": source.crs.to_string(),
                        "width": int(source.width),
                        "height": int(source.height),
                    }
            raster = raster_cache[lane_tif]

            stage = "resolve_lane_gt"
            scene_root = scene_root_for_inter_tif(inter_tif)
            gt_path = find_lane_gt(scene_root, baseline_name)
            lane_cache_key = (gt_path, raster["crs"])
            stage = "load_lane_gt"
            if lane_cache_key not in scene_lane_cache:
                scene_lane_cache[lane_cache_key] = SceneLaneIndex(
                    gt_path,
                    target_crs=raster["crs"],
                    source_crs=gt_crs,
                    ignored_types=ignored_lane_types,
                )

            row, col = record_row_col(record)
            stage = "clip_patch_ground_truth"
            gt_items = patch_ground_truth(
                scene_lane_cache[lane_cache_key],
                transform=raster["transform"],
                raster_width=raster["width"],
                raster_height=raster["height"],
                row=row,
                col=col,
                patch_size=patch_size,
            )
            gt_count = len(gt_items)
            positive_patches += int(gt_count > 0)
            scene_counts[scene_id] += 1
            records.append(
                {
                    "record_id": record_id,
                    "scene_id": scene_id,
                    "ground_truth_centerline_count": gt_count,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "record_id": record_id,
                    "prediction_file": str(record.get("_prediction_file") or ""),
                    "stage": stage,
                    "error": repr(exc),
                }
            )

        if index % 1000 == 0 or index == len(predictions):
            print(
                f"[e2e-gt-presence] processed={index}/{len(predictions)} "
                f"paired={len(records)} errors={len(errors)}",
                flush=True,
            )

    if not records:
        raise RuntimeError("No prediction records could be paired with E2E lane ground truth.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "scope": "GT centerline presence only; no patch geometry metrics are calculated.",
        "raw_e2e_root": str(raw_e2e_root),
        "prediction_dir": str(prediction_dir),
        "output_jsonl": str(output_jsonl),
        "patch_size": patch_size,
        "ignored_lane_types": sorted(ignored_lane_types),
        "json_files_seen": json_files_seen,
        "prediction_records": len(predictions),
        "paired_records": len(records),
        "positive_patches": positive_patches,
        "empty_patches": len(records) - positive_patches,
        "scene_count": len(scene_counts),
        "records_by_scene": dict(sorted(scene_counts.items())),
        "errors": errors,
        "complete": len(records) == len(predictions) and not errors,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if require_all and not report["complete"]:
        preview = json.dumps(errors[:10], ensure_ascii=False, indent=2)
        raise RuntimeError(f"Unable to pair all prediction records; errors={len(errors)}\n{preview}")
    return report


def main() -> None:
    args = parse_args()
    build_gt_presence(
        args.raw_e2e_root.resolve(),
        args.prediction_dir.resolve(),
        args.output_jsonl.resolve(),
        args.report_json.resolve(),
        baseline_name=args.baseline_name,
        gt_crs=args.gt_crs,
        patch_size=args.patch_size,
        ignored_lane_types=parse_int_set(args.ignore_lane_types),
        require_all=args.require_all,
    )


if __name__ == "__main__":
    main()
