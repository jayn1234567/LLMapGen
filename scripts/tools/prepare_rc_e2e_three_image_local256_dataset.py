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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--black-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--max-tifs", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=0)
    return parser.parse_args()


def _resolve_auxiliary_tif(inter_tif: Path, suffix: str) -> Path:
    prefix = inter_tif.stem.removesuffix("_inter")
    patch_root = inter_tif.parent.parent / "patch_tif"
    candidates = (patch_root / f"{prefix}_{suffix}.tif", patch_root / f"0_{suffix}.tif")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def required_auxiliary_tifs(inter_tif: Path) -> dict[str, Path]:
    return {
        "mask": _resolve_auxiliary_tif(inter_tif, "edit_poly"),
        "raw_lane": _resolve_auxiliary_tif(inter_tif, "lane"),
        "pose": _resolve_auxiliary_tif(inter_tif, "pose"),
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


def _read_masked_clean(image_path: Path, mask_path: Path) -> np.ndarray:
    image = _read_chw(image_path)
    mask = _read_chw(mask_path)
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"Clean/mask shape mismatch: image={image.shape[-2:]} mask={mask.shape[-2:]}"
        )
    return np.where((mask > 0).any(axis=0, keepdims=True), image, 0)


def _read_masked_binary(image_path: Path, mask_path: Path) -> np.ndarray:
    image = _read_chw(image_path)
    mask = _read_chw(mask_path)
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"Auxiliary/mask shape mismatch: image={image.shape[-2:]} mask={mask.shape[-2:]}"
        )
    positive = (image > 0).any(axis=0)
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


def _build_record(
    *,
    scene_id: str,
    inter_tif: Path,
    aux: dict[str, Path],
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
            "mask_tif": str(aux["mask"]),
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
            "raw_lane_image_source": "patch_tif/0_lane.tif",
            "raw_lane_image_role": "pv_camera_raw_lane",
            "raw_lane_active_model_input": True,
            "raw_lane_separate_image": True,
            "pose_image_source": "patch_tif/0_pose.tif",
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

    missing: list[dict[str, str]] = []
    for inter_tif in inter_tifs:
        for role, path in required_auxiliary_tifs(inter_tif).items():
            if not path.is_file():
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
