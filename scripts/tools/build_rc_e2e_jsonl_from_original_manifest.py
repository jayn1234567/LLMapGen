#!/usr/bin/env python3
"""Build model inference JSONL from the original RC E2E crop manifest.

The image patches must already have been produced by the archived project's
``split_inter_tif_for_inference.py``. This tool only supplies the prompt and
coordinate metadata expected by the selected MLLM checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.tools.prepare_rc_e2e_inference_dataset import (
    PROMPT_PROFILE_LOCAL256_550K_V1,
    VIEW_LOCAL256,
    VIEW_LOCAL512,
    dataset_v2_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--prompt-profile", default=PROMPT_PROFILE_LOCAL256_550K_V1)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--black-ratio-threshold", type=float, default=1.0)
    return parser.parse_args()


def _relative_image_path(image_path: str, output_root: Path) -> Path:
    path = Path(image_path)
    resolved = path.resolve() if path.is_absolute() else (output_root / path).resolve()
    try:
        relative = resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Original crop image is outside output root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Original crop image not found: {resolved}")
    return relative


def convert_manifest(
    manifest_json: str | Path,
    output_root: str | Path,
    *,
    output_jsonl: str | Path | None = None,
    prompt_profile: str = PROMPT_PROFILE_LOCAL256_550K_V1,
    patch_size: int = 256,
    coord_range: int = 1000,
    black_ratio_threshold: float = 1.0,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json).resolve()
    root = Path(output_root).resolve()
    if patch_size not in {256, 512}:
        raise ValueError("The original RC E2E crop experiment supports patch_size 256 or 512.")
    if coord_range != 1000:
        raise ValueError("checkpoint-34376 expects coord_range=1000.")
    if not 0.0 <= black_ratio_threshold <= 1.0:
        raise ValueError("black_ratio_threshold must be within [0, 1].")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Original crop manifest is empty or invalid: {manifest_path}")

    view_mode = VIEW_LOCAL256 if patch_size == 256 else VIEW_LOCAL512
    prompt = dataset_v2_prompt(
        view_mode=view_mode,
        target_size=patch_size,
        context_size=patch_size,
        coord_range=coord_range,
        include_intersections=True,
        prompt_profile=prompt_profile,
    )
    jsonl_path = Path(output_jsonl).resolve() if output_jsonl else root / "infer.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(manifest):
            if not isinstance(item, dict):
                raise TypeError(f"Manifest record {index} is not an object.")
            scene_id = str(item["id"])
            tif_stem = str(item["tif"])
            row = int(item["row"])
            col = int(item["col"])
            relative_image = _relative_image_path(str(item["image_path"]), root)
            tif_prefix = tif_stem.split("_", 1)[0]
            sample_id = f"{scene_id}_{tif_prefix}_{row}_{col}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate original-crop sample ID: {sample_id}")
            seen_ids.add(sample_id)

            record = {
                "id": sample_id,
                "image": relative_image.as_posix(),
                "meta": {
                    "scene_id": scene_id,
                    "tile_id": f"{scene_id}_{tif_prefix}",
                    "tif_stem": tif_stem,
                    "tif_prefix": tif_prefix,
                    "row": row,
                    "col": col,
                    "patch_row": row,
                    "patch_col": col,
                    "x0": col * patch_size,
                    "y0": row * patch_size,
                    "patch_size": patch_size,
                    "pixel_patch_size": patch_size,
                    "patch_width": patch_size,
                    "patch_height": patch_size,
                    "context_size": patch_size,
                    "input_image_size": patch_size,
                    "target_roi_in_image": [0, 0, patch_size, patch_size],
                    "view_mode": view_mode,
                    "crop_backend": "original_rc_e2e_split_inter_tif_for_inference",
                    "crop_black_ratio_threshold": black_ratio_threshold,
                    "crop_black_ratio_comparison": ">=",
                    "coord_mode": "norm1000",
                    "coord_system": f"patch_norm{coord_range}",
                    "coord_range": coord_range,
                },
                "conversations": [{"from": "human", "value": prompt}],
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "manifest_json": str(manifest_path),
        "output_root": str(root),
        "infer_jsonl": str(jsonl_path),
        "patch_count": len(seen_ids),
        "patch_size": patch_size,
        "stride": patch_size,
        "view_mode": view_mode,
        "crop_backend": "original_rc_e2e_split_inter_tif_for_inference",
        "crop_black_ratio_threshold": black_ratio_threshold,
        "crop_black_ratio_comparison": ">=",
        "prompt_profile": prompt_profile,
        "coord_mode": "norm1000",
        "coord_range": coord_range,
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    convert_manifest(
        args.manifest_json,
        args.output_root,
        output_jsonl=args.output_jsonl or None,
        prompt_profile=args.prompt_profile,
        patch_size=args.patch_size,
        coord_range=args.coord_range,
        black_ratio_threshold=args.black_ratio_threshold,
    )


if __name__ == "__main__":
    main()
