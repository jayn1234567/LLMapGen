#!/usr/bin/env python3
"""Validate pixel-grid alignment between RC inter and lane TIF rasters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Extracted E2E dataset root.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 validates every pair.")
    parser.add_argument("--output-json", default="", help="Optional machine-readable report path.")
    return parser.parse_args()


def discover_inter_tifs(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*_inter.tif")
        if path.parent.name == "inter_patch_tif"
    )


def expected_lane_tif(inter_tif: Path) -> Path:
    prefix = inter_tif.stem.removesuffix("_inter")
    return inter_tif.parent.parent / "lane_patch_tif" / f"{prefix}_lane.tif"


def close_values(left: tuple[float, ...], right: tuple[float, ...], *, atol: float, rtol: float) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
        for a, b in zip(left, right)
    )


def raster_metadata(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as source:
        return {
            "width": int(source.width),
            "height": int(source.height),
            "count": int(source.count),
            "dtypes": list(source.dtypes),
            "crs": source.crs.to_string() if source.crs is not None else None,
            "transform": [float(value) for value in tuple(source.transform)],
            "bounds": [float(value) for value in tuple(source.bounds)],
            "resolution": [float(value) for value in source.res],
        }


def compare_pair(
    inter_tif: Path,
    lane_tif: Path,
    *,
    patch_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    inter = raster_metadata(inter_tif)
    lane = raster_metadata(lane_tif)
    errors: list[str] = []
    warnings: list[str] = []

    if (inter["width"], inter["height"]) != (lane["width"], lane["height"]):
        errors.append(
            "size mismatch: "
            f"inter={inter['width']}x{inter['height']} lane={lane['width']}x{lane['height']}"
        )
    if inter["crs"] != lane["crs"]:
        errors.append(f"CRS mismatch: inter={inter['crs']!r} lane={lane['crs']!r}")
    elif inter["crs"] is None:
        warnings.append("both rasters have no CRS")
    if not close_values(tuple(inter["transform"]), tuple(lane["transform"]), atol=atol, rtol=rtol):
        errors.append(f"transform mismatch: inter={inter['transform']} lane={lane['transform']}")
    if not close_values(tuple(inter["bounds"]), tuple(lane["bounds"]), atol=atol, rtol=rtol):
        errors.append(f"bounds mismatch: inter={inter['bounds']} lane={lane['bounds']}")
    if not close_values(tuple(inter["resolution"]), tuple(lane["resolution"]), atol=atol, rtol=rtol):
        errors.append(
            f"resolution mismatch: inter={inter['resolution']} lane={lane['resolution']}"
        )

    inter_grid = [
        math.ceil(inter["height"] / patch_size),
        math.ceil(inter["width"] / patch_size),
    ]
    lane_grid = [
        math.ceil(lane["height"] / patch_size),
        math.ceil(lane["width"] / patch_size),
    ]
    if inter_grid != lane_grid:
        errors.append(f"patch grid mismatch: inter={inter_grid} lane={lane_grid}")

    return {
        "inter_tif": str(inter_tif),
        "lane_tif": str(lane_tif),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "inter": inter,
        "lane": lane,
        "patch_grid_rows_cols": inter_grid,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root).resolve()
    inter_tifs = discover_inter_tifs(input_root)
    if args.max_pairs > 0:
        inter_tifs = inter_tifs[: args.max_pairs]
    if not inter_tifs:
        raise FileNotFoundError(f"No inter_patch_tif/*_inter.tif found below {input_root}")

    pair_reports: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, str]] = []
    for inter_tif in inter_tifs:
        lane_tif = expected_lane_tif(inter_tif)
        if not lane_tif.is_file():
            missing_pairs.append({"inter_tif": str(inter_tif), "expected_lane_tif": str(lane_tif)})
            continue
        pair_reports.append(
            compare_pair(
                inter_tif,
                lane_tif,
                patch_size=args.patch_size,
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    mismatches = [report for report in pair_reports if not report["ok"]]
    warnings = sum(len(report["warnings"]) for report in pair_reports)
    report = {
        "input_root": str(input_root),
        "patch_size": int(args.patch_size),
        "inter_tifs_found": len(inter_tifs),
        "pairs_checked": len(pair_reports),
        "missing_lane_pairs": len(missing_pairs),
        "mismatched_pairs": len(mismatches),
        "warning_count": warnings,
        "ok": not missing_pairs and not mismatches,
        "missing": missing_pairs,
        "mismatches": mismatches,
        "pairs": pair_reports,
    }
    return report


def main() -> None:
    args = parse_args()
    report = validate(args)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "[e2e-raster-check] "
        f"found={report['inter_tifs_found']} checked={report['pairs_checked']} "
        f"missing={report['missing_lane_pairs']} mismatched={report['mismatched_pairs']} "
        f"warnings={report['warning_count']} ok={report['ok']}"
    )
    for item in report["missing"][:10]:
        print(
            "[e2e-raster-check] MISSING "
            f"{item['inter_tif']} -> {item['expected_lane_tif']}"
        )
    for item in report["mismatches"][:10]:
        print(f"[e2e-raster-check] MISMATCH {item['inter_tif']}")
        for error in item["errors"]:
            print(f"  - {error}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
