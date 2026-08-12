#!/usr/bin/env python3
"""Build the three-image local256 native-Qwen3-VL E2E inference dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import rasterio

from data_process.state_update_dataset_common import make_prompt
from scripts.tools.prepare_rc_e2e_inference_dataset import (
    discover_inter_tifs,
    scene_id_for_tif,
)


PROMPT_CONTRACT_VERSION = "three_image_roles_concise_v2"
IMAGE_ROLES = (
    "bev_road_structure",
    "pv_camera_raw_lane",
    "historical_vehicle_trajectory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=None,
        help=(
            "Optional GT-bearing E2E root. Every inference raster must have an aligned "
            "scene/prefix raster here before records are generated."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--black-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--max-tifs", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=0)
    return parser.parse_args()


def _resolve_auxiliary_tif(
    inter_tif: Path,
    suffix: str,
    *,
    directory: str,
) -> Path:
    prefix = inter_tif.stem.removesuffix("_inter")
    centerline_root = inter_tif.parent.parent
    candidates = tuple(
        centerline_root / directory / filename
        for filename in (f"{prefix}_{suffix}.tif", f"0_{suffix}.tif")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_optional_mask_tif(inter_tif: Path) -> Path | None:
    prefix = inter_tif.stem.removesuffix("_inter")
    centerline_root = inter_tif.parent.parent
    candidates = (
        centerline_root / "patch_tif" / f"{prefix}_edit_poly.tif",
        centerline_root / "patch_tif" / "0_edit_poly.tif",
        centerline_root / "lane_patch_tif" / f"{prefix}_edit_poly.tif",
        centerline_root / "lane_patch_tif" / "0_edit_poly.tif",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def required_auxiliary_tifs(inter_tif: Path) -> dict[str, Path | None]:
    return {
        "mask": _resolve_optional_mask_tif(inter_tif),
        "raw_lane": _resolve_auxiliary_tif(
            inter_tif, "lane", directory="patch_tif"
        ),
        "pose": _resolve_auxiliary_tif(
            inter_tif, "pose", directory="lane_patch_tif"
        ),
    }


def _pad_chw(array: np.ndarray, patch_size: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape={array.shape}")
    channels, height, width = array.shape
    padded_height = int(math.ceil(height / patch_size) * patch_size)
    padded_width = int(math.ceil(width / patch_size) * patch_size)
    padded = np.zeros((channels, padded_height, padded_width), dtype=array.dtype)
    padded[:, :height, :width] = array
    return padded


def _black_ratio(chw: np.ndarray) -> float:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape={chw.shape}")
    return float(np.all(chw == 0, axis=0).mean())


def _read_chw(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read()


def _read_masked_clean(image_path: Path, mask_path: Path | None) -> np.ndarray:
    image = _read_chw(image_path)
    if mask_path is None:
        return image
    mask = _read_chw(mask_path)
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"Clean/mask shape mismatch: image={image.shape[-2:]} mask={mask.shape[-2:]}"
        )
    return np.where((mask > 0).any(axis=0, keepdims=True), image, 0)


def _read_masked_binary(image_path: Path, mask_path: Path | None) -> np.ndarray:
    image = _read_chw(image_path)
    positive = (image > 0).any(axis=0)
    if mask_path is None:
        valid = np.ones(image.shape[-2:], dtype=bool)
    else:
        mask = _read_chw(mask_path)
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"Auxiliary/mask shape mismatch: image={image.shape[-2:]} mask={mask.shape[-2:]}"
            )
        valid = (mask > 0).any(axis=0)
    output = np.zeros((3, image.shape[-2], image.shape[-1]), dtype=np.uint8)
    output[:, positive & valid] = 255
    return output


def _image_chunk_to_pil(chunk: np.ndarray) -> Image.Image:
    array = chunk[:3]
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def three_image_prompt(patch_size: int, coord_range: int) -> str:
    return make_prompt(
        True,
        [],
        [],
        phase="a",
        coord_mode="norm1000",
        coord_range=coord_range,
        patch_size=patch_size,
        context_size=patch_size,
        raw_lane_overlay=False,
        raw_lane_separate_image=True,
        pose_second_image=True,
    )


def _relative_image(role_root: str, scene_id: str, tif_stem: str, row: int, col: int) -> Path:
    return Path(role_root) / scene_id / tif_stem / f"{row}_{col}.png"


def _source_path_label(path: Path | None) -> str | None:
    if path is None:
        return None
    return f"{path.parent.name}/{path.name}"


def _tif_key(path: Path) -> tuple[str, str]:
    return scene_id_for_tif(path), path.stem.removesuffix("_inter")


def _raster_grid_metadata(path: Path, patch_size: int) -> dict[str, Any]:
    with rasterio.open(path) as source:
        return {
            "width": int(source.width),
            "height": int(source.height),
            "crs": source.crs.to_string() if source.crs is not None else None,
            "transform": [float(value) for value in tuple(source.transform)],
            "bounds": [float(value) for value in tuple(source.bounds)],
            "grid_rows": int(math.ceil(source.height / patch_size)),
            "grid_cols": int(math.ceil(source.width / patch_size)),
        }


def _close_sequence(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
        for a, b in zip(left, right)
    )


def validate_evaluation_alignment(
    inference_tifs: list[Path],
    evaluation_root: Path,
    *,
    patch_size: int,
    require_exact_keys: bool,
) -> dict[str, Any]:
    evaluation_tifs = discover_inter_tifs(evaluation_root)
    inference_by_key = {_tif_key(path): path for path in inference_tifs}
    evaluation_by_key = {_tif_key(path): path for path in evaluation_tifs}
    duplicate_inference = len(inference_by_key) != len(inference_tifs)
    duplicate_evaluation = len(evaluation_by_key) != len(evaluation_tifs)
    missing_keys = sorted(set(inference_by_key) - set(evaluation_by_key))
    unexpected_keys = (
        sorted(set(evaluation_by_key) - set(inference_by_key)) if require_exact_keys else []
    )
    mismatches: list[dict[str, Any]] = []

    for key in sorted(set(inference_by_key) & set(evaluation_by_key)):
        inference_meta = _raster_grid_metadata(inference_by_key[key], patch_size)
        evaluation_meta = _raster_grid_metadata(evaluation_by_key[key], patch_size)
        errors: list[str] = []
        for field in ("width", "height", "crs", "grid_rows", "grid_cols"):
            if inference_meta[field] != evaluation_meta[field]:
                errors.append(
                    f"{field} mismatch: inference={inference_meta[field]!r} "
                    f"evaluation={evaluation_meta[field]!r}"
                )
        for field in ("transform", "bounds"):
            if not _close_sequence(inference_meta[field], evaluation_meta[field]):
                errors.append(
                    f"{field} mismatch: inference={inference_meta[field]} "
                    f"evaluation={evaluation_meta[field]}"
                )
        if errors:
            mismatches.append(
                {
                    "scene_id": key[0],
                    "tif_prefix": key[1],
                    "inference_tif": str(inference_by_key[key]),
                    "evaluation_tif": str(evaluation_by_key[key]),
                    "errors": errors,
                }
            )

    ok = not (
        duplicate_inference
        or duplicate_evaluation
        or missing_keys
        or unexpected_keys
        or mismatches
    )
    report = {
        "ok": ok,
        "evaluation_root": str(evaluation_root),
        "inference_tif_count": len(inference_tifs),
        "evaluation_tif_count": len(evaluation_tifs),
        "matched_tif_count": len(set(inference_by_key) & set(evaluation_by_key)),
        "require_exact_keys": bool(require_exact_keys),
        "duplicate_inference_keys": duplicate_inference,
        "duplicate_evaluation_keys": duplicate_evaluation,
        "missing_in_evaluation": [
            {"scene_id": scene_id, "tif_prefix": prefix}
            for scene_id, prefix in missing_keys
        ],
        "unexpected_in_evaluation": [
            {"scene_id": scene_id, "tif_prefix": prefix}
            for scene_id, prefix in unexpected_keys
        ],
        "mismatches": mismatches,
    }
    if not ok:
        raise ValueError(
            "Inference/evaluation E2E raster alignment failed: "
            + json.dumps(report, ensure_ascii=False)[:8000]
        )
    return report


def _build_record(
    *,
    scene_id: str,
    inter_tif: Path,
    aux: dict[str, Path | None],
    image_paths: list[Path],
    row: int,
    col: int,
    patch_size: int,
    coord_range: int,
    source_width: int,
    source_height: int,
    padded_width: int,
    padded_height: int,
    prompt: str,
) -> dict[str, Any]:
    prefix = inter_tif.stem.removesuffix("_inter")
    sample_id = f"{scene_id}_{prefix}_{row}_{col}"
    return {
        "id": sample_id,
        "image": image_paths[0].as_posix(),
        "images": [path.as_posix() for path in image_paths],
        "raw_lane_image": image_paths[1].as_posix(),
        "pose_image": image_paths[2].as_posix(),
        "meta": {
            "scene_id": scene_id,
            "tile_id": f"{scene_id}_{prefix}",
            "tif_stem": inter_tif.stem,
            "tif_prefix": prefix,
            "source_tif": str(inter_tif),
            "mask_tif": str(aux["mask"]) if aux["mask"] is not None else None,
            "raw_lane_tif": str(aux["raw_lane"]),
            "pose_tif": str(aux["pose"]),
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
            "target_size": patch_size,
            "context_image_size": patch_size,
            "input_image_size": patch_size,
            "target_roi_in_image": [0, 0, patch_size, patch_size],
            "view_mode": "local256",
            "coord_mode": "norm1000",
            "coord_system": f"patch_norm{coord_range}",
            "coord_range": coord_range,
            "source_image_size": [source_width, source_height],
            "padded_source_image_size": [padded_width, padded_height],
            "raw_lane_overlay": False,
            "raw_lane_auxiliary_image": True,
            "raw_lane_image_source": _source_path_label(aux["raw_lane"]),
            "raw_lane_image_role": "pv_camera_raw_lane",
            "raw_lane_active_model_input": True,
            "raw_lane_separate_image": True,
            "pose_image_source": _source_path_label(aux["pose"]),
            "input_image_roles": list(IMAGE_ROLES),
            "three_image_prompt_contract_version": PROMPT_CONTRACT_VERSION,
        },
        "conversations": [{"from": "human", "value": prompt}],
    }


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    patch_size = int(args.patch_size)
    if patch_size != 256 or int(args.stride) != patch_size:
        raise ValueError("The 800k three-image local256 recipe requires patch-size=stride=256.")
    if int(args.coord_range) != 1000:
        raise ValueError("The 800k three-image local256 recipe requires coord-range=1000.")
    if not 0.0 <= float(args.black_ratio_threshold) <= 1.0:
        raise ValueError("--black-ratio-threshold must be within [0, 1].")

    inter_tifs = discover_inter_tifs(input_root)
    if args.max_tifs > 0:
        inter_tifs = inter_tifs[: args.max_tifs]
    if not inter_tifs:
        raise FileNotFoundError(f"No source *_inter.tif files found below {input_root}")

    evaluation_root_value = getattr(args, "evaluation_root", None)
    evaluation_root = (
        Path(evaluation_root_value).resolve() if evaluation_root_value is not None else None
    )
    alignment_report = None
    if evaluation_root is not None:
        alignment_report = validate_evaluation_alignment(
            inter_tifs,
            evaluation_root,
            patch_size=patch_size,
            require_exact_keys=int(args.max_tifs) <= 0,
        )

    missing: list[dict[str, str]] = []
    for inter_tif in inter_tifs:
        for role, path in required_auxiliary_tifs(inter_tif).items():
            if role != "mask" and (path is None or not path.is_file()):
                missing.append({"inter_tif": str(inter_tif), "role": role, "expected": str(path)})
    if missing:
        raise FileNotFoundError(
            "The E2E source cannot reproduce the 800k three-image training contract; "
            f"missing auxiliary TIFs={len(missing)}, examples={missing[:10]}"
        )

    prompt = three_image_prompt(patch_size, int(args.coord_range))
    if prompt.count("<image>") != 3:
        raise AssertionError("Three-image inference prompt must contain exactly three <image> tokens.")

    output_root.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_root / "infer.jsonl"
    manifest_path = output_root / "patch_manifest.json"
    manifest: list[dict[str, Any]] = []
    patch_count = 0
    skipped_black = 0
    stop = False

    with output_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for tif_index, inter_tif in enumerate(inter_tifs, start=1):
            aux = required_auxiliary_tifs(inter_tif)
            clean = _read_masked_clean(inter_tif, aux["mask"])
            if aux["raw_lane"] is None or aux["pose"] is None:
                raise AssertionError(f"Required auxiliary TIF resolution failed for {inter_tif}")
            raw_lane = _read_masked_binary(aux["raw_lane"], aux["mask"])
            pose = _read_masked_binary(aux["pose"], aux["mask"])
            shapes = {tuple(array.shape[-2:]) for array in (clean, raw_lane, pose)}
            if len(shapes) != 1:
                raise ValueError(
                    f"Aligned three-image source shape mismatch for {inter_tif}: "
                    f"clean={clean.shape}, raw_lane={raw_lane.shape}, pose={pose.shape}"
                )

            source_height, source_width = clean.shape[-2:]
            clean = _pad_chw(clean, patch_size)
            raw_lane = _pad_chw(raw_lane, patch_size)
            pose = _pad_chw(pose, patch_size)
            padded_height, padded_width = clean.shape[-2:]
            rows = padded_height // patch_size
            cols = padded_width // patch_size
            scene_id = scene_id_for_tif(inter_tif)
            print(
                f"[three-image-e2e-data] tif {tif_index}/{len(inter_tifs)}: {inter_tif} "
                f"source={source_width}x{source_height} grid={rows}x{cols}",
                flush=True,
            )

            for row in range(rows):
                for col in range(cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    clean_patch = clean[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
                    ratio = _black_ratio(clean_patch)
                    if ratio >= float(args.black_ratio_threshold):
                        skipped_black += 1
                        continue
                    chunks = (
                        clean_patch,
                        raw_lane[:, y0 : y0 + patch_size, x0 : x0 + patch_size],
                        pose[:, y0 : y0 + patch_size, x0 : x0 + patch_size],
                    )
                    relative_paths = [
                        _relative_image("images", scene_id, inter_tif.stem, row, col),
                        _relative_image("raw_lane_images", scene_id, inter_tif.stem, row, col),
                        _relative_image("pose_images", scene_id, inter_tif.stem, row, col),
                    ]
                    for relative_path, chunk in zip(relative_paths, chunks):
                        destination = output_root / relative_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        _image_chunk_to_pil(chunk).save(destination)

                    record = _build_record(
                        scene_id=scene_id,
                        inter_tif=inter_tif,
                        aux=aux,
                        image_paths=relative_paths,
                        row=row,
                        col=col,
                        patch_size=patch_size,
                        coord_range=int(args.coord_range),
                        source_width=source_width,
                        source_height=source_height,
                        padded_width=padded_width,
                        padded_height=padded_height,
                        prompt=prompt,
                    )
                    jsonl_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    manifest.append(
                        {
                            "id": record["id"],
                            "scene_id": scene_id,
                            "tif": inter_tif.stem,
                            "row": row,
                            "col": col,
                            "x0": x0,
                            "y0": y0,
                            "black_ratio": ratio,
                            "images": [path.as_posix() for path in relative_paths],
                            "input_image_roles": list(IMAGE_ROLES),
                        }
                    )
                    patch_count += 1
                    if args.max_patches > 0 and patch_count >= args.max_patches:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "input_root": str(input_root),
        "evaluation_root": str(evaluation_root) if evaluation_root is not None else None,
        "output_root": str(output_root),
        "view_mode": "local256",
        "patch_size": patch_size,
        "stride": patch_size,
        "coord_mode": "norm1000",
        "coord_range": int(args.coord_range),
        "num_images_per_sample": 3,
        "input_image_roles": list(IMAGE_ROLES),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "black_ratio_threshold": float(args.black_ratio_threshold),
        "black_ratio_comparison": ">=",
        "tif_count": len(inter_tifs),
        "patch_count": patch_count,
        "skipped_black": skipped_black,
        "infer_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_path),
        "evaluation_alignment": alignment_report,
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
