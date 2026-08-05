#!/usr/bin/env python3
"""Build 800k three-image Stage A datasets without redownloading OBS data.

The clean staging supplies image 1. An existing raw-lane-overlay + pose staging
supplies the selected records, raw-lane mask, pose mask, labels, and metadata.
The finalized records are rewritten to use this strict input order:

1. clean BEV image
2. black/white raw-lane image from patch_tif/0_lane.tif
3. black/white pose image from patch_tif/0_pose.tif
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path, PurePosixPath

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import allocate_quotas, parse_ratio_spec
from data_process.build_dataset_v2_staged import (
    STAGE_MARKER,
    build_sample_owners,
    discover_stage_roots,
)
from data_process.fixed_source_splits import load_fixed_source_split_manifest
from data_process.state_update_dataset_common import make_prompt
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar
from scripts.tools.build_rc_dataset_v2_rawlane_256_context_windows import relabel_metadata


TARGET_SAMPLES = 800_000
DIFFICULTY_RATIOS = "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10"
INTERSECTION_RATIO = 0.28
SOURCE_VARIANTS = ("local256", "context512_roi256")
VARIANT_NAMES = {
    "local256": "rawlane_pose_three_image_local256_800k",
    "context512_roi256": "rawlane_pose_three_image_context512_roi256_800k",
}
IMAGE_ROLES = [
    "bev_road_structure",
    "pv_camera_raw_lane",
    "historical_vehicle_trajectory",
]
INCOMPLETE_MARKER = "_THREE_IMAGE_CONVERSION_INCOMPLETE_DO_NOT_USE.txt"
PROMPT_CONTRACT_VERSION = "three_image_roles_concise_v2"
OBSOLETE_PROMPT_FRAGMENTS = (
    "white lines are predicted lanes on a black background",
    "do not copy it blindly when it conflicts with the visible bev evidence",
    "white lines are historical vehicle trajectories on a black background",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-staging-root",
        default=r"D:\data\fulldata\staging",
        help=(
            "Existing staging whose primary BEV images have no raw-lane overlay. It may "
            "contain local256, context512_roi256, or both, according to --views."
        ),
    )
    parser.add_argument(
        "--clean-context-staging-root",
        default="",
        help=(
            "Optional existing context512_roi256 staging with clean primary BEV images. "
            "If omitted or incomplete, context images are reconstructed from clean local256 tiles."
        ),
    )
    parser.add_argument(
        "--aux-staging-root",
        default=r"D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context",
        help="Existing staging containing overlay BEV, saved raw-lane masks, and pose images.",
    )
    parser.add_argument(
        "--views",
        choices=["local", "context", "both"],
        default="both",
        help=(
            "Generate local256, context512_roi256, or both. Context-only mode can use a "
            "clean context staging directly and does not require clean local256 tiles."
        ),
    )
    parser.add_argument(
        "--direct-context-overlap-only",
        action="store_true",
        help=(
            "For context output, constrain train selection to sample ids that have a direct "
            "clean context image and a matching raw-lane/pose staged record. This disables "
            "clean-image reconstruction for selected train samples."
        ),
    )
    parser.add_argument(
        "--work-root",
        default=r"D:\data\fulldata_rawlane_pose_three_image_800k",
    )
    parser.add_argument(
        "--fixed-source-split-manifest",
        default=r"D:\data\fixed_splits\rc_fixed_large_maps_v1.json",
    )
    parser.add_argument("--target-samples", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--difficulty-ratios", default=DIFFICULTY_RATIOS)
    parser.add_argument("--intersection-target-ratio", type=float, default=INTERSECTION_RATIO)
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--visualize-per-difficulty", type=int, default=0)
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--skip-generic-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def source_variants_for_views(views: str) -> tuple[str, ...]:
    if views == "local":
        return ("local256",)
    if views == "context":
        return ("context512_roi256",)
    if views == "both":
        return SOURCE_VARIANTS
    raise ValueError(f"unsupported views: {views!r}")


def run(command: list[object]) -> None:
    command = [str(item) for item in command]
    print("[three-image-dataset] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".three_image.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def stage_map(staging_root: Path, role: str) -> dict[int, Path]:
    result = {}
    for stage_root in discover_stage_roots(staging_root):
        marker = read_json(stage_root / STAGE_MARKER)
        source_index = int(marker["source_index"])
        if source_index in result:
            raise ValueError(f"duplicate {role} source_index={source_index}: {stage_root}")
        result[source_index] = stage_root
    return result


def build_direct_context_overlap_filter(
    clean_stages: dict[int, dict[str, Path]],
    aux_root: Path,
    destination: Path,
) -> dict:
    """Write ids whose direct context image exists in both staged datasets."""

    auxiliary = stage_map(aux_root, "auxiliary-overlap")
    sample_owner, collisions = build_sample_owners(
        list(auxiliary.values()),
        "last",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    selected_ids = set()
    source_reports = []
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for source_index, aux_stage in sorted(auxiliary.items()):
            clean_stage = clean_stages[source_index]["context512_roi256"]
            clean_ids = set()
            missing_clean_files = 0
            for split in ("train", "eval", "test"):
                clean_records = (
                    clean_stage
                    / "records"
                    / "context512_roi256"
                    / f"{split}.jsonl"
                )
                if not clean_records.is_file():
                    raise FileNotFoundError(
                        f"clean context staged records not found: {clean_records}"
                    )
                with clean_records.open("r", encoding="utf-8-sig") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        sample_id = str(record.get("id") or "").strip()
                        relative = str(record.get("image") or "").strip()
                        if not sample_id or not relative:
                            raise ValueError(
                                f"invalid clean context record at {clean_records}:{line_number}"
                            )
                        image_path = (
                            clean_stage
                            / "variants"
                            / "context512_roi256"
                            / Path(relative)
                        )
                        if not image_path.is_file():
                            missing_clean_files += 1
                            continue
                        if sample_id in clean_ids:
                            raise ValueError(
                                f"duplicate clean context sample id={sample_id}: {clean_stage}"
                            )
                        clean_ids.add(sample_id)

            aux_records = 0
            common_records = 0
            for split in ("train", "eval", "test"):
                aux_index = aux_stage / "records" / f"{split}.index.jsonl"
                if not aux_index.is_file():
                    raise FileNotFoundError(f"auxiliary staged index not found: {aux_index}")
                with aux_index.open("r", encoding="utf-8-sig") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        index_row = json.loads(line)
                        sample_id = str(index_row.get("id") or "").strip()
                        if not sample_id:
                            raise ValueError(
                                f"invalid auxiliary index at {aux_index}:{line_number}"
                            )
                        aux_records += 1
                        raw_sample_id = str(index_row.get("raw_sample_id") or "")
                        if raw_sample_id and sample_owner.get(raw_sample_id) != source_index:
                            continue
                        if sample_id not in clean_ids or sample_id in selected_ids:
                            continue
                        selected_ids.add(sample_id)
                        output.write(
                            json.dumps({"id": sample_id}, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        common_records += 1
            source_reports.append({
                "source_index": source_index,
                "clean_stage": str(clean_stage),
                "aux_stage": str(aux_stage),
                "clean_context_ids_with_files": len(clean_ids),
                "missing_clean_context_files": missing_clean_files,
                "aux_candidate_records": aux_records,
                "common_candidate_records": common_records,
            })
            print(
                f"[three-image-dataset] direct context overlap source={source_index}: "
                f"clean={len(clean_ids)} aux={aux_records} common={common_records}",
                flush=True,
            )
    temporary.replace(destination)
    report = {
        "status": "passed",
        "filter_path": str(destination),
        "unique_common_candidate_ids": len(selected_ids),
        "duplicate_raw_sample_events": len(collisions),
        "sources": source_reports,
    }
    write_json(destination.with_suffix(destination.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def validate_stage_compatibility(
    clean_root: Path,
    aux_root: Path,
    clean_context_root: Path | None = None,
    selected_variants: tuple[str, ...] = SOURCE_VARIANTS,
) -> dict[int, dict[str, Path]]:
    clean = stage_map(clean_root, "clean")
    clean_context = (
        stage_map(clean_context_root, "clean-context")
        if clean_context_root is not None
        else {}
    )
    auxiliary = stage_map(aux_root, "auxiliary")
    resolved = {}
    for source_index, aux_stage in auxiliary.items():
        aux_marker = read_json(aux_stage / STAGE_MARKER)
        candidates = []
        for stage in (clean.get(source_index), clean_context.get(source_index)):
            if stage is not None and stage not in candidates:
                candidates.append(stage)
        if not candidates:
            raise ValueError(
                "clean staging does not cover auxiliary source_index="
                f"{source_index}; clean_root={clean_root}, clean_context_root={clean_context_root}"
            )
        candidate_markers = {}
        for stage in candidates:
            marker = read_json(stage / STAGE_MARKER)
            if bool(marker.get("raw_lane_overlay", False)):
                raise ValueError(f"clean staging is itself overlaid: {stage}")
            candidate_markers[stage] = marker

        local_stage = next(
            (stage for stage in candidates if "local256" in candidate_markers[stage].get("variants", [])),
            None,
        )
        context_stage = next(
            (
                stage
                for stage in candidates
                if "context512_roi256" in candidate_markers[stage].get("variants", [])
            ),
            None,
        )
        if "local256" in selected_variants and local_stage is None and context_stage is None:
            raise ValueError(
                f"source_index={source_index} has neither clean local256 nor clean context images"
            )
        if (
            "context512_roi256" in selected_variants
            and context_stage is None
            and local_stage is None
        ):
            raise ValueError(
                f"source_index={source_index} has neither clean context nor clean local256 images"
            )
        local_mode = "direct" if local_stage is not None else "crop_from_context"
        context_mode = "direct" if context_stage is not None else "mosaic_from_local256"
        print(
            f"[three-image-dataset] clean source {source_index}: "
            f"local256={local_mode}, context512_roi256={context_mode}",
            flush=True,
        )
        if not bool(aux_marker.get("raw_lane_overlay", False)):
            raise ValueError(f"auxiliary staging does not contain the expected overlay data: {aux_stage}")
        if not bool(aux_marker.get("save_raw_lane_image", False)):
            raise ValueError(f"auxiliary staging did not save raw-lane masks: {aux_stage}")
        if not bool(aux_marker.get("pose_second_image", False)):
            raise ValueError(f"auxiliary staging did not save pose inputs: {aux_stage}")
        selected_clean_stages = {
            stage
            for variant, stage in (
                ("local256", local_stage),
                ("context512_roi256", context_stage),
            )
            if variant in selected_variants and stage is not None
        }
        for selected_stage in selected_clean_stages:
            selected_marker = candidate_markers[selected_stage]
            for field in (
                "target_patch_size",
                "context_size",
                "train_stride",
                "eval_test_stride",
            ):
                context_value = selected_marker.get(field)
                aux_value = aux_marker.get(field)
                if context_value is not None and aux_value is not None and context_value != aux_value:
                    raise ValueError(
                        f"clean/aux staging geometry mismatch at source {source_index}: "
                        f"{field} clean={context_value}, auxiliary={aux_value}"
                    )
        resolved[source_index] = {
            "local256": local_stage or context_stage,
            "context512_roi256": context_stage or local_stage,
        }
    print(
        f"[three-image-dataset] compatible staging sources: {len(auxiliary)}; "
        "OBS download and TIFF extraction are not required",
        flush=True,
    )
    return resolved


def finalization_command(args: argparse.Namespace, output_root: Path) -> list[object]:
    command: list[object] = [
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "finalize",
        "--staging-root", Path(args.aux_staging_root).expanduser().resolve(),
        "--output-root", output_root,
        "--views", args.views,
        "--patch-size", 256,
        "--context-size", 512,
        "--train-target-samples", args.target_samples,
        "--difficulty-ratios", args.difficulty_ratios,
        "--intersection-target-ratio", args.intersection_target_ratio,
        "--strict-difficulty-quotas",
        "--difficulty-seed", args.difficulty_seed,
        "--duplicate-policy", "last",
        "--copy-mode", args.copy_mode,
        "--fixed-source-split-manifest", args.fixed_source_split_manifest,
        "--repartition-existing-stages-by-fixed-manifest",
    ]
    candidate_filter = str(getattr(args, "train_candidate_jsonl", "") or "").strip()
    if candidate_filter:
        command.extend(["--train-candidate-jsonl", candidate_filter])
    if args.resume:
        command.append("--resume")
    return command


def find_pose_image(record: dict, split: str) -> str:
    if record.get("pose_image"):
        return str(record["pose_image"])
    for relative in record.get("images") or []:
        if str(relative).startswith(f"pose_images/{split}/"):
            return str(relative)
    raise ValueError(f"sample {record.get('id')} has no pose image")


def find_clean_source(
    clean_stage: Path,
    variant: str,
    final_relative: str,
) -> Path:
    relative = PurePosixPath(final_relative)
    parts = list(relative.parts)
    if len(parts) < 3 or parts[0] != "images":
        raise ValueError(f"invalid primary image path: {final_relative!r}")
    matches = []
    for split in ("train", "eval", "test"):
        parts[1] = split
        candidate = clean_stage / "variants" / variant / Path(*parts)
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one clean staged image for {final_relative!r} under "
            f"{clean_stage / 'variants' / variant}; found={matches}"
        )
    return matches[0]


def _clean_local_patch_path(clean_stage: Path, tile_id: str, x0: int, y0: int) -> Path | None:
    patch_id = f"{tile_id}_x{x0:05d}_y{y0:05d}"
    relative = Path("images")
    matches = []
    for split in ("train", "eval", "test"):
        candidate = (
            clean_stage
            / "variants"
            / "local256"
            / relative
            / split
            / tile_id
            / f"{patch_id}.png"
        )
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) > 1:
        raise ValueError(
            f"duplicate clean local256 patch for tile={tile_id}, x0={x0}, y0={y0}: "
            f"{matches}"
        )
    return matches[0] if matches else None


def _source_hw(meta: dict) -> tuple[int, int] | None:
    value = meta.get("source_image_size")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        return None
    return height, width


def _synthesize_clean_crop_from_local256(
    clean_stage: Path,
    record: dict,
    destination: Path,
    crop_box: tuple[int, int, int, int],
    output_size: int,
    patch_size: int = 256,
) -> str:
    """Reconstruct a clean crop from any overlapping clean local256 patches."""

    meta = record.get("meta") or {}
    tile_id = str(meta.get("tile_id") or "").strip()
    if not tile_id:
        raise ValueError(f"sample {record.get('id')} has no tile_id for clean-image reconstruction")
    crop_x0, crop_y0, crop_x1, crop_y1 = map(int, crop_box)
    if crop_x1 - crop_x0 != output_size or crop_y1 - crop_y0 != output_size:
        raise ValueError(
            f"sample {record.get('id')} has inconsistent clean crop box {crop_box!r}"
        )

    canvas = Image.new("RGB", (output_size, output_size), (0, 0, 0))
    coverage = Image.new("L", (output_size, output_size), 0)
    candidate_step = patch_size // 2
    first_tile_x = ((crop_x0 - patch_size + 1) // candidate_step) * candidate_step
    first_tile_y = ((crop_y0 - patch_size + 1) // candidate_step) * candidate_step
    used_tiles = 0
    for tile_y0 in range(first_tile_y, crop_y1, candidate_step):
        if tile_y0 < 0:
            continue
        for tile_x0 in range(first_tile_x, crop_x1, candidate_step):
            if tile_x0 < 0:
                continue
            source = _clean_local_patch_path(clean_stage, tile_id, tile_x0, tile_y0)
            if source is None:
                continue
            left = max(crop_x0, tile_x0)
            top = max(crop_y0, tile_y0)
            right = min(crop_x1, tile_x0 + patch_size)
            bottom = min(crop_y1, tile_y0 + patch_size)
            if left >= right or top >= bottom:
                continue
            with Image.open(source) as image:
                image = image.convert("RGB")
                if image.size != (patch_size, patch_size):
                    raise ValueError(
                        f"clean local patch must be {patch_size}x{patch_size}, got "
                        f"{image.size}: {source}"
                    )
                crop = image.crop((
                    left - tile_x0,
                    top - tile_y0,
                    right - tile_x0,
                    bottom - tile_y0,
                ))
            destination_box = (left - crop_x0, top - crop_y0)
            canvas.paste(crop, destination_box)
            coverage.paste(
                255,
                (
                    destination_box[0],
                    destination_box[1],
                    destination_box[0] + crop.width,
                    destination_box[1] + crop.height,
                ),
            )
            used_tiles += 1
    if used_tiles == 0:
        raise FileNotFoundError(
            f"no clean local256 tiles found for sample={record.get('id')} tile={tile_id} "
            f"crop_box={[crop_x0, crop_y0, crop_x1, crop_y1]}"
        )

    source_hw = _source_hw(meta)
    if source_hw is not None:
        source_height, source_width = source_hw
        valid_left = max(0, crop_x0) - crop_x0
        valid_top = max(0, crop_y0) - crop_y0
        valid_right = min(source_width, crop_x1) - crop_x0
        valid_bottom = min(source_height, crop_y1) - crop_y0
        if valid_left < valid_right and valid_top < valid_bottom:
            extrema = coverage.crop(
                (valid_left, valid_top, valid_right, valid_bottom)
            ).getextrema()
            if extrema != (255, 255):
                raise FileNotFoundError(
                    "clean local256 staging does not fully cover the valid crop region for "
                    f"sample={record.get('id')} tile={tile_id}; coverage_extrema={extrema}"
                )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".clean_context.partial")
    temporary.unlink(missing_ok=True)
    try:
        canvas.save(temporary, format="PNG", compress_level=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "mosaic_from_overlapping_local256"


def synthesize_clean_context_from_local256(
    clean_stage: Path,
    record: dict,
    destination: Path,
    patch_size: int = 256,
) -> str:
    meta = record.get("meta") or {}
    context_size = int(meta.get("context_image_size", 0))
    if context_size <= patch_size:
        raise ValueError(
            f"sample {record.get('id')} is not a context sample: context_size={context_size}"
        )
    context_box = meta.get("context_box_full")
    if isinstance(context_box, (list, tuple)) and len(context_box) == 4:
        crop_box = tuple(map(int, context_box))
    else:
        roi = meta.get("target_roi_in_image") or [
            (context_size - patch_size) // 2,
            (context_size - patch_size) // 2,
            (context_size + patch_size) // 2,
            (context_size + patch_size) // 2,
        ]
        crop_x0 = int(meta["x0"]) - int(roi[0])
        crop_y0 = int(meta["y0"]) - int(roi[1])
        crop_box = (crop_x0, crop_y0, crop_x0 + context_size, crop_y0 + context_size)
    _synthesize_clean_crop_from_local256(
        clean_stage,
        record,
        destination,
        crop_box,
        context_size,
        patch_size,
    )
    return "mosaic_context_from_local256"


def synthesize_clean_local_from_local256(
    clean_stage: Path,
    record: dict,
    destination: Path,
    patch_size: int = 256,
) -> str:
    meta = record.get("meta") or {}
    target_box = meta.get("target_box_full")
    if isinstance(target_box, (list, tuple)) and len(target_box) == 4:
        crop_box = tuple(map(int, target_box))
    else:
        x0, y0 = int(meta["x0"]), int(meta["y0"])
        crop_box = (x0, y0, x0 + patch_size, y0 + patch_size)
    _synthesize_clean_crop_from_local256(
        clean_stage,
        record,
        destination,
        crop_box,
        patch_size,
        patch_size,
    )
    return "mosaic_local_from_overlapping_local256"


def crop_clean_local_from_context(
    context_stage: Path,
    record: dict,
    destination: Path,
    patch_size: int = 256,
) -> str:
    source = find_clean_source(
        context_stage,
        "context512_roi256",
        str(record["image"]),
    )
    context_marker = read_json(context_stage / STAGE_MARKER)
    context_size = int(context_marker.get("context_size", 512))
    roi = [
        (context_size - patch_size) // 2,
        (context_size - patch_size) // 2,
        (context_size + patch_size) // 2,
        (context_size + patch_size) // 2,
    ]
    roi = tuple(map(int, roi))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".clean_local.partial")
    temporary.unlink(missing_ok=True)
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.size != (context_size, context_size):
                raise ValueError(
                    f"clean context image must be {context_size}x{context_size}, got "
                    f"{image.size}: {source}"
                )
            crop = image.crop(roi)
        if crop.size != (patch_size, patch_size):
            raise ValueError(f"clean context ROI has size={crop.size}, expected {patch_size}x{patch_size}")
        crop.save(temporary, format="PNG", compress_level=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "crop_local256_from_context512_roi256"


def materialize_clean_primary(
    clean_sources: dict[str, Path],
    source_variant: str,
    record: dict,
    destination: Path,
    copy_mode: str,
) -> str:
    local_stage = clean_sources["local256"]
    context_stage = clean_sources["context512_roi256"]
    direct_stage = clean_sources[source_variant]
    marker = read_json(direct_stage / STAGE_MARKER)
    if source_variant in marker.get("variants", []):
        try:
            clean_source = find_clean_source(
                direct_stage,
                source_variant,
                str(record["image"]),
            )
            return replace_file(clean_source, destination, copy_mode)
        except FileNotFoundError:
            pass
    if source_variant == "local256":
        context_marker = read_json(context_stage / STAGE_MARKER)
        if "context512_roi256" in context_marker.get("variants", []):
            try:
                return crop_clean_local_from_context(context_stage, record, destination)
            except FileNotFoundError:
                pass
        return synthesize_clean_local_from_local256(local_stage, record, destination)
    if source_variant == "context512_roi256":
        return synthesize_clean_context_from_local256(local_stage, record, destination)
    raise ValueError(f"unsupported clean source variant: {source_variant}")


def replace_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".clean.partial")
    temporary.unlink(missing_ok=True)
    used_mode = mode
    try:
        if mode == "hardlink":
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
                used_mode = "copy_fallback"
        else:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return used_mode


def three_image_prompt(record: dict) -> str:
    meta = record.get("meta") or {}
    patch_size = int(meta.get("pixel_patch_size", meta.get("patch_width", 256)))
    context_size = int(meta.get("context_image_size", patch_size))
    return make_prompt(
        True,
        [],
        [],
        phase="a",
        coord_mode=str(meta.get("coord_mode", "norm1000")),
        coord_range=int(meta.get("coord_range", 1000)),
        patch_size=patch_size,
        context_size=context_size,
        raw_lane_overlay=False,
        raw_lane_separate_image=True,
        pose_second_image=True,
    )


def transform_record(record: dict, split: str) -> dict:
    primary = str(record["image"])
    raw_lane = str(record.get("raw_lane_image") or "")
    if not raw_lane.startswith(f"raw_lane_images/{split}/"):
        raise ValueError(f"sample {record.get('id')} has invalid raw-lane image: {raw_lane!r}")
    pose = find_pose_image(record, split)
    if not pose.startswith(f"pose_images/{split}/"):
        raise ValueError(f"sample {record.get('id')} has invalid pose image: {pose!r}")
    conversations = list(record.get("conversations") or [])
    if len(conversations) < 2:
        raise ValueError(f"sample {record.get('id')} has invalid conversations")
    conversations[0] = dict(conversations[0])
    conversations[0]["value"] = three_image_prompt(record)
    meta = dict(record.get("meta") or {})
    meta.pop("raw_lane_overlay_source", None)
    meta.update({
        "raw_lane_overlay": False,
        "raw_lane_auxiliary_image": True,
        "raw_lane_image_source": "patch_tif/0_lane.tif",
        "raw_lane_image_role": "pv_camera_raw_lane",
        "raw_lane_active_model_input": True,
        "raw_lane_separate_image": True,
        "pose_image_source": "patch_tif/0_pose.tif",
        "input_image_roles": list(IMAGE_ROLES),
    })
    transformed = dict(record)
    transformed.update({
        "image": primary,
        "images": [primary, raw_lane, pose],
        "raw_lane_image": raw_lane,
        "pose_image": pose,
        "meta": meta,
        "conversations": conversations,
    })
    return transformed


def transform_meta_record(meta_record: dict, transformed: dict) -> dict:
    result = dict(meta_record)
    result["image"] = transformed["image"]
    result["images"] = list(transformed["images"])
    result["raw_lane_image"] = transformed["raw_lane_image"]
    result["pose_image"] = transformed["pose_image"]
    result["meta"] = dict(transformed["meta"])
    return result


def convert_variant_to_three_images(
    variant_root: Path,
    source_variant: str,
    clean_stages: dict[int, Path],
    copy_mode: str,
) -> dict:
    counts = Counter()
    link_modes = Counter()
    for split in ("train", "eval", "test"):
        phase_path = variant_root / "phase_a" / f"{split}.jsonl"
        meta_path = variant_root / "phase_a" / f"meta_{split}.jsonl"
        phase_tmp = phase_path.with_name(phase_path.name + ".three_image.partial")
        meta_tmp = meta_path.with_name(meta_path.name + ".three_image.partial")
        with (
            phase_path.open("r", encoding="utf-8-sig") as phase_handle,
            meta_path.open("r", encoding="utf-8-sig") as meta_handle,
            phase_tmp.open("w", encoding="utf-8") as phase_output,
            meta_tmp.open("w", encoding="utf-8") as meta_output,
        ):
            for line_number, (phase_line, meta_line) in enumerate(
                zip_longest(phase_handle, meta_handle),
                start=1,
            ):
                if phase_line is None or meta_line is None:
                    raise ValueError(f"phase/meta length mismatch for {variant_root} {split}")
                if not phase_line.strip() and not meta_line.strip():
                    continue
                record = json.loads(phase_line)
                meta_record = json.loads(meta_line)
                if record.get("id") != meta_record.get("id"):
                    raise ValueError(
                        f"phase/meta id mismatch at {phase_path}:{line_number}: "
                        f"{record.get('id')!r} != {meta_record.get('id')!r}"
                    )
                source_index = int((record.get("meta") or {}).get("source_index", -1))
                if source_index not in clean_stages:
                    raise ValueError(
                        f"sample {record.get('id')} has unavailable clean source_index={source_index}"
                    )
                destination = variant_root / str(record["image"])
                used_mode = materialize_clean_primary(
                    clean_stages[source_index],
                    source_variant,
                    record,
                    destination,
                    copy_mode,
                )
                link_modes[used_mode] += 1
                transformed = transform_record(record, split)
                phase_output.write(json.dumps(transformed, ensure_ascii=False) + "\n")
                meta_output.write(
                    json.dumps(
                        transform_meta_record(meta_record, transformed),
                        ensure_ascii=False,
                    ) + "\n"
                )
                counts[split] += 1
                if counts[split] % 100_000 == 0:
                    print(
                        f"[three-image-dataset] converted {source_variant}/{split}: "
                        f"{counts[split]}",
                        flush=True,
                    )
        phase_tmp.replace(phase_path)
        meta_tmp.replace(meta_path)
    update_dataset_metadata(variant_root, source_variant, counts)
    return {"counts": dict(counts), "clean_primary_link_modes": dict(link_modes)}


def update_dataset_metadata(variant_root: Path, source_variant: str, counts: Counter) -> None:
    info_path = variant_root / "dataset_info.json"
    info = read_json(info_path)
    input_overlay = dict(info.get("input_overlay") or {})
    input_overlay.update({
        "raw_lane_overlay": False,
        "raw_lane_separate_image": True,
        "raw_lane_overlay_source": "none",
    })
    auxiliary = dict(info.get("auxiliary_image_assets") or {})
    raw_lane = dict(auxiliary.get("raw_lane") or {})
    raw_lane.update({
        "saved": True,
        "active_model_input": True,
        "source": "patch_tif/0_lane.tif",
        "directory": "raw_lane_images",
    })
    auxiliary["raw_lane"] = raw_lane
    info.update({
        "input_overlay": input_overlay,
        "auxiliary_image_assets": auxiliary,
        "multi_image_input": {
            "enabled": True,
            "num_images_per_sample": 3,
            "image_roles": list(IMAGE_ROLES),
            "image_order": list(IMAGE_ROLES),
            "raw_lane_image_source": "patch_tif/0_lane.tif",
            "pose_image_source": "patch_tif/0_pose.tif",
        },
        "three_image_input": True,
        "three_image_prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "clean_bev_source": "existing_non_overlay_staging",
        "record_counts_after_three_image_conversion": dict(counts),
    })
    write_json(info_path, info)


def refresh_existing_three_image_prompts(root: Path) -> bool:
    """Rewrite only user prompts when resuming a dataset built with an older contract."""
    info_path = root / "dataset_info.json"
    if not info_path.is_file():
        return False
    info = read_json(info_path)
    if info.get("three_image_prompt_contract_version") == PROMPT_CONTRACT_VERSION:
        return False

    rewritten = 0
    for split in ("train", "eval", "test"):
        path = root / "phase_a" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot refresh prompt; split is missing: {path}")
        temporary = path.with_name(path.name + ".prompt_refresh.partial")
        with (
            path.open("r", encoding="utf-8-sig") as source,
            temporary.open("w", encoding="utf-8") as output,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                conversations = list(record.get("conversations") or [])
                if not conversations:
                    raise ValueError(f"{path}:{line_number} has no conversations")
                conversations[0] = dict(conversations[0])
                conversations[0]["value"] = three_image_prompt(record)
                record["conversations"] = conversations
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                rewritten += 1
        temporary.replace(path)

    info["three_image_prompt_contract_version"] = PROMPT_CONTRACT_VERSION
    write_json(info_path, info)
    (root / "three_image_validation.json").unlink(missing_ok=True)
    print(
        f"[three-image-dataset] refreshed concise prompts: root={root} records={rewritten}",
        flush=True,
    )
    return True


def validate_three_image_variant(
    root: Path,
    expected_train_samples: int,
    sample_limit: int,
) -> dict:
    info = read_json(root / "dataset_info.json")
    multi = info.get("multi_image_input") or {}
    overlay = info.get("input_overlay") or {}
    raw_asset = ((info.get("auxiliary_image_assets") or {}).get("raw_lane") or {})
    if overlay.get("raw_lane_overlay") is not False:
        raise ValueError(f"raw_lane_overlay must be false: {root}")
    if overlay.get("raw_lane_separate_image") is not True:
        raise ValueError(f"raw_lane_separate_image must be true: {root}")
    if raw_asset.get("active_model_input") is not True:
        raise ValueError(f"raw lane is not an active input: {root}")
    if int(multi.get("num_images_per_sample", 0)) != 3:
        raise ValueError(f"expected three images in dataset_info: {multi!r}")
    if list(multi.get("image_order") or []) != IMAGE_ROLES:
        raise ValueError(f"invalid image order: {multi!r}")
    if info.get("three_image_prompt_contract_version") != PROMPT_CONTRACT_VERSION:
        raise ValueError(
            "invalid three-image prompt contract: "
            f"{info.get('three_image_prompt_contract_version')!r}"
        )
    counts = Counter()
    for split in ("train", "eval", "test"):
        path = root / "phase_a" / f"{split}.jsonl"
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                images = record.get("images")
                if not isinstance(images, list) or len(images) != 3:
                    raise ValueError(f"{path}:{line_number} expected three images, got {images!r}")
                expected_prefixes = (
                    f"images/{split}/",
                    f"raw_lane_images/{split}/",
                    f"pose_images/{split}/",
                )
                for relative, prefix in zip(images, expected_prefixes):
                    if not str(relative).startswith(prefix):
                        raise ValueError(
                            f"{path}:{line_number} image order/path mismatch: {images!r}"
                        )
                    if not (root / str(relative)).is_file():
                        raise FileNotFoundError(f"{path}:{line_number} missing {root / str(relative)}")
                if record.get("image") != images[0]:
                    raise ValueError(f"{path}:{line_number} primary image mismatch")
                if record.get("raw_lane_image") != images[1]:
                    raise ValueError(f"{path}:{line_number} raw-lane image mismatch")
                if record.get("pose_image") != images[2]:
                    raise ValueError(f"{path}:{line_number} pose image mismatch")
                prompt = str((record.get("conversations") or [{}])[0].get("value", ""))
                if prompt.count("<image>") != 3:
                    raise ValueError(f"{path}:{line_number} prompt does not contain three images")
                for text in (
                    "first image is the clean BEV road-structure image",
                    "second image is a lane image predicted by a PV camera model",
                    "third image is a historical vehicle-trajectory image",
                ):
                    if text not in prompt:
                        raise ValueError(f"{path}:{line_number} prompt lacks {text!r}")
                prompt_lower = prompt.lower()
                obsolete = [
                    text for text in OBSOLETE_PROMPT_FRAGMENTS if text in prompt_lower
                ]
                if obsolete:
                    raise ValueError(
                        f"{path}:{line_number} contains obsolete prompt text: {obsolete!r}"
                    )
                if "white lane overlay" in prompt:
                    raise ValueError(f"{path}:{line_number} still describes an overlaid raw lane")
                counts[split] += 1
                if sample_limit > 0 and counts[split] >= sample_limit:
                    break
    actual_train_count = count_jsonl(root / "phase_a" / "train.jsonl")
    if actual_train_count != expected_train_samples:
        raise ValueError(
            f"train count={actual_train_count}, expected={expected_train_samples}: {root}"
        )
    result = {
        "status": "passed",
        "dataset_root": str(root),
        "actual_train_count": actual_train_count,
        "validated_records": dict(counts),
        "validation_sample_limit_per_split": sample_limit,
        "image_order": list(IMAGE_ROLES),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
    }
    write_json(root / "three_image_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def validate_variant_pairing(roots: dict[str, Path]) -> None:
    local_root = roots[VARIANT_NAMES["local256"]]
    context_root = roots[VARIANT_NAMES["context512_roi256"]]
    for split in ("train", "eval", "test"):
        local_path = local_root / "phase_a" / f"{split}.jsonl"
        context_path = context_root / "phase_a" / f"{split}.jsonl"
        with (
            local_path.open("r", encoding="utf-8-sig") as local_handle,
            context_path.open("r", encoding="utf-8-sig") as context_handle,
        ):
            for line_number, (local_line, context_line) in enumerate(
                zip_longest(local_handle, context_handle),
                start=1,
            ):
                if local_line is None or context_line is None:
                    raise ValueError(f"paired variant length mismatch for {split}:{line_number}")
                if json.loads(local_line)["id"] != json.loads(context_line)["id"]:
                    raise ValueError(f"paired variant id mismatch for {split}:{line_number}")
        print(f"[three-image-dataset] paired IDs passed: {split}", flush=True)


def dataset_is_complete(root: Path, target_samples: int) -> bool:
    if not (root / "three_image_validation.json").is_file():
        return False
    try:
        report = read_json(root / "three_image_validation.json")
        return (
            report.get("status") == "passed"
            and int(report.get("actual_train_count", -1)) == target_samples
            and report.get("prompt_contract_version") == PROMPT_CONTRACT_VERSION
            and dataset_has_three_image_contract(root, target_samples)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def dataset_has_three_image_contract(root: Path, target_samples: int) -> bool:
    if not (root / "dataset_info.json").is_file():
        return False
    try:
        info = read_json(root / "dataset_info.json")
        multi = info.get("multi_image_input") or {}
        return (
            count_jsonl(root / "phase_a" / "train.jsonl") == target_samples
            and int(multi.get("num_images_per_sample", 0)) == 3
            and (info.get("input_overlay") or {}).get("raw_lane_overlay") is False
            and info.get("three_image_prompt_contract_version") == PROMPT_CONTRACT_VERSION
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def finalized_overlay_source_is_ready(
    root: Path,
    target_samples: int,
    require_direct_overlap: bool = False,
) -> bool:
    try:
        ready = (
            (root / "dataset_info.json").is_file()
            and count_jsonl(root / "phase_a" / "train.jsonl") == target_samples
        )
        if not ready or not require_direct_overlap:
            return ready
        info = read_json(root / "dataset_info.json")
        return bool(info.get("train_candidate_filter")) and bool(
            (info.get("balance") or {}).get("strict_difficulty_quotas", False)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main(argv=None) -> None:
    args = parse_args(argv)
    clean_root = Path(args.clean_staging_root).expanduser().resolve()
    clean_context_text = str(args.clean_context_staging_root or "").strip()
    clean_context_root = (
        Path(clean_context_text).expanduser().resolve()
        if clean_context_text
        else None
    )
    aux_root = Path(args.aux_staging_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    output_root = work_root / "output_rawlane_pose_three_image_800k"
    package_root = work_root / "packages_rawlane_pose_three_image_800k"
    fixed_manifest = load_fixed_source_split_manifest(args.fixed_source_split_manifest)
    args.fixed_source_split_manifest = str(fixed_manifest["path"])
    if not 0 <= args.intersection_target_ratio <= 1:
        raise ValueError("--intersection-target-ratio must be in [0, 1]")
    if args.target_samples <= 0:
        raise ValueError("--target-samples must be positive")
    expected_difficulty = allocate_quotas(
        args.target_samples,
        parse_ratio_spec(args.difficulty_ratios),
    )
    selected_source_variants = source_variants_for_views(args.views)
    clean_stages = validate_stage_compatibility(
        clean_root,
        aux_root,
        clean_context_root,
        selected_source_variants,
    )
    overlap_report = None
    args.train_candidate_jsonl = ""
    if args.direct_context_overlap_only:
        if selected_source_variants != ("context512_roi256",):
            raise ValueError(
                "--direct-context-overlap-only currently requires --views context"
            )
        overlap_filter = work_root / "manifests" / "direct_context_overlap_ids.jsonl"
        overlap_report = build_direct_context_overlap_filter(
            clean_stages,
            aux_root,
            overlap_filter,
        )
        if int(overlap_report["unique_common_candidate_ids"]) < args.target_samples:
            raise ValueError(
                "the direct clean-context/raw-lane/pose staging intersection is too small: "
                f"common={overlap_report['unique_common_candidate_ids']}, "
                f"target={args.target_samples}."
            )
        args.train_candidate_jsonl = str(overlap_filter)
    output_root.mkdir(parents=True, exist_ok=True)
    package_root.mkdir(parents=True, exist_ok=True)
    final_roots = {
        VARIANT_NAMES[source]: output_root / VARIANT_NAMES[source]
        for source in selected_source_variants
    }
    if args.resume:
        for root in final_roots.values():
            if root.exists():
                refresh_existing_three_image_prompts(root)
    reuse_complete = args.resume and all(
        dataset_is_complete(root, args.target_samples)
        for root in final_roots.values()
    )
    conversion_reports = {}
    if not reuse_complete:
        source_roots = {
            source: output_root / source
            for source in selected_source_variants
        }
        target_roots = {
            source: output_root / VARIANT_NAMES[source]
            for source in selected_source_variants
        }
        incompatible_targets = [
            target
            for target in target_roots.values()
            if target.exists() and not dataset_has_three_image_contract(
                target, args.target_samples
            )
        ]
        if incompatible_targets:
            raise ValueError(
                "an incomplete three-image target already exists. Use a new --work-root, or "
                "remove only the generated output after reviewing three_image_validation.json: "
                f"{incompatible_targets}"
            )
        missing_overlay_sources = [
            source
            for source, source_root in source_roots.items()
            if not target_roots[source].exists()
            and not finalized_overlay_source_is_ready(
                source_root,
                args.target_samples,
                require_direct_overlap=args.direct_context_overlap_only,
            )
        ]
        if missing_overlay_sources:
            run(finalization_command(args, output_root))
        for source_variant in selected_source_variants:
            source_root = source_roots[source_variant]
            target_root = target_roots[source_variant]
            if target_root.exists():
                print(
                    f"[three-image-dataset] reuse converted variant: {target_root}",
                    flush=True,
                )
                continue
            marker_path = source_root / INCOMPLETE_MARKER
            marker_path.write_text(
                "This is an intermediate overlay dataset. Do not train from or package this "
                "directory. Wait until conversion finishes and the directory is renamed to "
                f"{target_root.name}.\n",
                encoding="utf-8",
            )
            print(
                f"[three-image-dataset] WARNING: {source_root} is an incomplete overlay "
                f"intermediate; final target={target_root}",
                flush=True,
            )
            conversion_reports[source_variant] = convert_variant_to_three_images(
                source_root,
                source_variant,
                clean_stages,
                args.copy_mode,
            )
            source_root.rename(target_root)
            (target_root / INCOMPLETE_MARKER).unlink(missing_ok=True)
            relabel_metadata(
                target_root / "dataset_info.json",
                source_variant,
                VARIANT_NAMES[source_variant],
            )
        final_roots = {
            VARIANT_NAMES[source]: output_root / VARIANT_NAMES[source]
            for source in selected_source_variants
        }
        for source_variant in selected_source_variants:
            target_variant = VARIANT_NAMES[source_variant]
            relabel_metadata(
                output_root / "build_summary.json",
                source_variant,
                target_variant,
            )
    else:
        print("[three-image-dataset] reuse completed three-image outputs", flush=True)

    validation_reports = {}
    for variant, root in final_roots.items():
        validation_reports[variant] = validate_three_image_variant(
            root,
            args.target_samples,
            args.validation_sample_limit,
        )
        info = read_json(root / "dataset_info.json")
        actual_difficulty = ((info.get("balance") or {}).get("final_bucket_counts") or {})
        for bucket, expected in expected_difficulty.items():
            if int(actual_difficulty.get(bucket, -1)) != expected:
                raise ValueError(
                    f"{variant} difficulty {bucket}={actual_difficulty.get(bucket)}, "
                    f"expected={expected}"
                )
        actual_intersection = float((info.get("balance") or {}).get("actual_intersection_ratio", -1))
        if abs(actual_intersection - args.intersection_target_ratio) > 1e-8:
            raise ValueError(
                f"{variant} intersection ratio={actual_intersection}, "
                f"expected={args.intersection_target_ratio}"
            )
        if not args.skip_generic_validation:
            run([
                sys.executable,
                "scripts/tools/validate_visualize_rc_dataset_v2.py",
                "--dataset-root", root,
                "--variant", variant,
                "--expected-train-samples", args.target_samples,
                "--difficulty-ratios", args.difficulty_ratios,
                "--expected-intersection-ratio", args.intersection_target_ratio,
                "--output-dir", output_root / f"{variant}_validation",
                "--visualize-per-difficulty", args.visualize_per_difficulty,
                "--image-decode-mode", args.image_decode_mode,
                "--skip-distribution-check",
            ])
    if len(selected_source_variants) == 2:
        validate_variant_pairing(final_roots)

    packages = []
    if not args.skip_package:
        for variant, root in final_roots.items():
            package = package_root / f"{variant}.tar"
            create_variant_tar(root, package, args.resume)
            packages.append(str(package))
    summary = {
        "status": "passed",
        "clean_staging_root": str(clean_root),
        "clean_context_staging_root": str(clean_context_root) if clean_context_root else "",
        "aux_staging_root": str(aux_root),
        "obs_download_performed": False,
        "fixed_source_split_manifest": args.fixed_source_split_manifest,
        "fixed_source_split_sha256": fixed_manifest["file_sha256"],
        "target_samples": args.target_samples,
        "views": args.views,
        "direct_context_overlap_only": args.direct_context_overlap_only,
        "direct_context_overlap_report": overlap_report,
        "difficulty_ratios": args.difficulty_ratios,
        "intersection_target_ratio": args.intersection_target_ratio,
        "image_order": list(IMAGE_ROLES),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "variants": {name: str(root) for name, root in final_roots.items()},
        "packages": packages,
        "conversion_reports": conversion_reports,
        "validation_reports": validation_reports,
    }
    write_json(output_root / "three_image_800k_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
