#!/usr/bin/env python3
"""Build RC Dataset V2 through disk-bounded source shards.

The ``stage`` command turns one raw source into self-contained candidate SFT
records and PNGs, then may delete that verified raw source. The ``finalize``
command globally de-duplicates all completed shards and applies the requested
difficulty/intersection distribution without reopening TIFF or GeoJSON files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import (
    DIFFICULTY_ORDER,
    annotate_translation_grid,
    classify_row,
    dataset_variant_specs,
    empty_candidate_pools,
    parse_ratio_spec,
    select_balanced_candidates,
    variant_row,
    write_images_for_rows,
    write_jsonl_item,
)
from data_process.state_update_dataset_common import (
    COORD_MODE_NORM1000,
    DEFAULT_COORD_RANGE,
    ALLOWED_LANE_TYPES,
    IGNORED_LANE_TYPE_CODES,
    SEMANTIC_SCHEMA_VERSION,
    build_sft_record,
    discover_samples,
    process_sample,
    require_geo_dependencies,
    semantic_sft_record_counts,
    semantic_target_counts,
    validate_rows,
    write_json,
)
from data_process.fixed_source_splits import (
    SPLIT_POLICY as FIXED_SPLIT_POLICY,
    fixed_split_descriptor,
    load_fixed_source_split_manifest,
    split_for_raw_sample,
    validate_fixed_holdout_coverage,
)
from scripts.tools.tag_hard_map_samples import DIFFICULTY_RULE_VERSION


STAGE_MARKER = "stage_complete.json"
STAGE_VERSION = "rc_dataset_v2_source_stage_v4_lane_4_18_25_ignore_3_22"


def add_geometry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--coord-mode", choices=[COORD_MODE_NORM1000], default=COORD_MODE_NORM1000)
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--max-patches-per-sample", type=int, default=None)
    parser.add_argument("--boundary-tol", type=float, default=1.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--line-sample-distance-px", type=float, default=0.0)
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument("--intersection-hint-points", type=int, default=3)
    parser.add_argument("--max-traces-per-side", type=int, default=8)
    parser.add_argument("--max-intersections-per-side", type=int, default=8)
    parser.add_argument("--png-compress-level", type=int, choices=range(0, 10), default=4)
    parser.add_argument(
        "--raw-lane-overlay",
        action="store_true",
        help="Overlay patch_tif/0_lane.tif as white raw-lane pixels on top of every input image.",
    )
    parser.add_argument(
        "--require-raw-lane",
        action="store_true",
        help="Fail if --raw-lane-overlay is enabled and patch_tif/0_lane.tif is missing.",
    )
    parser.add_argument("--raw-lane-threshold", type=float, default=0.0)
    parser.add_argument(
        "--save-raw-lane-image",
        action="store_true",
        help="Save raw lane as a separate black/white auxiliary PNG without activating it as input.",
    )
    parser.add_argument(
        "--raw-lane-separate-image",
        action="store_true",
        help="Activate the saved raw-lane PNG as a separate model input after the clean BEV.",
    )
    parser.add_argument(
        "--pose-second-image",
        action="store_true",
        help="Add patch_tif/0_pose.tif as a separate second image for every sample.",
    )
    parser.add_argument("--pose-threshold", type=float, default=0.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="Convert one raw source into a verified candidate shard.")
    stage.add_argument("--input-root", required=True)
    stage.add_argument("--stage-root", required=True)
    stage.add_argument("--source-index", type=int, required=True)
    stage.add_argument("--source-uri", default="")
    stage.add_argument("--views", choices=["local", "context", "both"], default="local")
    stage.add_argument("--split-seed", type=int, default=42)
    stage.add_argument("--train-ratio", type=float, default=0.90)
    stage.add_argument("--eval-ratio", type=float, default=0.05)
    stage.add_argument(
        "--fixed-source-split-manifest",
        default=os.environ.get("RC_FIXED_SOURCE_SPLIT_MANIFEST", ""),
        help="Explicit raw_sample_id eval/test manifest; all unlisted large maps become train.",
    )
    stage.add_argument("--archive-workers", type=int, default=1)
    stage.add_argument("--selective-archive-extract", action="store_true")
    stage.add_argument("--keep-archives", action="store_true")
    stage.add_argument("--limit-samples", type=int, default=None)
    stage.add_argument(
        "--train-candidate-jsonl",
        default="",
        help="Optional completed train JSONL whose ids limit train rows rendered into this stage.",
    )
    stage.add_argument("--resume", action="store_true")
    stage.add_argument("--delete-input-root-after-stage", action="store_true")
    stage.add_argument(
        "--delete-root-parent",
        default="",
        help="Required safety boundary when --delete-input-root-after-stage is used.",
    )
    add_geometry_args(stage)

    finalize = subparsers.add_parser("finalize", help="Globally balance completed source shards.")
    finalize.add_argument("--staging-root", required=True)
    finalize.add_argument("--output-root", required=True)
    finalize.add_argument("--views", choices=["local", "context", "both"], default="local")
    finalize.add_argument("--train-target-samples", type=int, default=550000)
    finalize.add_argument(
        "--difficulty-ratios",
        default="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10",
    )
    finalize.add_argument("--intersection-target-ratio", type=float, default=0.30)
    finalize.add_argument(
        "--strict-difficulty-quotas",
        action="store_true",
        help=(
            "Keep the requested difficulty counts exact and use translated-grid candidates "
            "only to fill per-bucket base-grid shortages. Never redistributes a shortage to "
            "another difficulty bucket."
        ),
    )
    finalize.add_argument("--difficulty-seed", type=int, default=20260713)
    finalize.add_argument("--duplicate-policy", choices=["last", "first", "error"], default="last")
    finalize.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    finalize.add_argument(
        "--train-candidate-jsonl",
        default="",
        help="Optional completed train JSONL whose ids constrain the new train selection to a subset.",
    )
    finalize.add_argument(
        "--difficulty-override-jsonl",
        default="",
        help="Optional train difficulty sidecar with id, difficulty/stratum, and difficulty_score.",
    )
    finalize.add_argument(
        "--difficulty-rule-version",
        default="",
        help="Rule version recorded when --difficulty-override-jsonl is used.",
    )
    finalize.add_argument("--resume", action="store_true")
    finalize.add_argument("--patch-size", type=int, default=256)
    finalize.add_argument("--context-size", type=int, default=512)
    finalize.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    finalize.add_argument(
        "--fixed-source-split-manifest",
        default=os.environ.get("RC_FIXED_SOURCE_SPLIT_MANIFEST", ""),
    )
    finalize.add_argument(
        "--allow-missing-fixed-holdouts",
        action="store_true",
        help="Allow a source subset to omit fixed eval/test maps. Training leakage still fails.",
    )
    finalize.add_argument(
        "--repartition-existing-stages-by-fixed-manifest",
        action="store_true",
        help=(
            "Reuse hash-split staging with --fixed-source-split-manifest instead of rebuilding raw "
            "sources. Fixed eval/test keep base-grid patches only; available train candidates are reused."
        ),
    )
    return parser.parse_args(argv)


def selected_variants(views: str, patch_size: int = 256, context_size: int = 512) -> list[str]:
    return list(dataset_variant_specs(Path("."), views, patch_size, context_size))


def stable_sample_split(sample_id: str, seed: int, train_ratio: float, eval_ratio: float) -> str:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be in (0, 1)")
    if eval_ratio < 0 or train_ratio + eval_ratio >= 1:
        raise ValueError("--eval-ratio must be >= 0 and train_ratio + eval_ratio must be < 1")
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + eval_ratio:
        return "eval"
    return "test"


def validate_geometry_args(args) -> None:
    if args.patch_size <= 0 or args.context_size < args.patch_size:
        raise ValueError("--patch-size must be positive and --context-size must be >= --patch-size")
    if (args.context_size - args.patch_size) % 2:
        raise ValueError("--context-size minus --patch-size must be even for a centered ROI")
    if args.stride != args.patch_size:
        raise ValueError("--stride must equal --patch-size for eval/test")
    if not 0 < args.train_stride <= args.patch_size or args.patch_size % args.train_stride:
        raise ValueError("--train-stride must be a positive divisor of --patch-size")
    if args.coord_range != DEFAULT_COORD_RANGE:
        raise ValueError(f"--coord-range must be {DEFAULT_COORD_RANGE}")
    if args.raw_lane_overlay and args.raw_lane_separate_image:
        raise ValueError("--raw-lane-overlay and --raw-lane-separate-image are mutually exclusive")
    if args.raw_lane_separate_image and not args.save_raw_lane_image:
        raise ValueError("--raw-lane-separate-image requires --save-raw-lane-image")


def stage_variant_specs(stage_root: Path, views: str, patch_size: int, context_size: int) -> dict:
    return dataset_variant_specs(
        stage_root / "variants",
        views,
        patch_size,
        context_size,
    )


def open_stage_writers(stage_root: Path, variants: list[str]):
    records_dir = stage_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    index_writers = {
        split: (records_dir / f"{split}.index.jsonl").open("w", encoding="utf-8")
        for split in ("train", "eval", "test")
    }
    sft_writers = {}
    for variant in variants:
        variant_dir = records_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        sft_writers[variant] = {
            split: (variant_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in ("train", "eval", "test")
        }
    return index_writers, sft_writers


def close_writers(*writer_groups) -> None:
    for group in writer_groups:
        for value in group.values():
            if isinstance(value, dict):
                for handle in value.values():
                    handle.close()
            else:
                value.close()


def safe_remove_completed_input(input_root: Path, allowed_parent: Path, stage_marker: Path) -> None:
    root = input_root.resolve()
    parent = allowed_parent.resolve()
    if not stage_marker.is_file():
        raise RuntimeError(f"refusing to delete input before stage completion: {stage_marker}")
    if root == parent:
        raise ValueError("input root cannot equal the allowed delete parent")
    try:
        root.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"input root {root} is outside delete boundary {parent}") from exc
    if len(root.parts) <= len(parent.parts):
        raise ValueError(f"unsafe input root for deletion: {root}")
    shutil.rmtree(root)
    print(f"[dataset-v2-stage] removed verified raw source: {root}", flush=True)


def stage_source(args) -> None:
    require_geo_dependencies()
    validate_geometry_args(args)
    input_root = Path(args.input_root)
    stage_root = Path(args.stage_root)
    marker_path = stage_root / STAGE_MARKER
    candidate_jsonl_text = str(getattr(args, "train_candidate_jsonl", "") or "").strip()
    candidate_jsonl = Path(candidate_jsonl_text) if candidate_jsonl_text else None
    allowed_train_ids = load_train_candidate_ids(candidate_jsonl) if candidate_jsonl else None
    candidate_filter_sha256 = file_sha256(candidate_jsonl) if candidate_jsonl else ""
    fixed_manifest_text = str(getattr(args, "fixed_source_split_manifest", "") or "").strip()
    fixed_manifest = (
        load_fixed_source_split_manifest(fixed_manifest_text) if fixed_manifest_text else None
    )
    fixed_descriptor = fixed_split_descriptor(fixed_manifest)
    if args.resume and marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("stage_version") != STAGE_VERSION or not marker.get("semantic_validation_passed"):
            raise ValueError(
                f"stale stage cannot be resumed: {marker_path}. Expected stage_version={STAGE_VERSION}; "
                "rebuild this source shard from its raw source before allowing raw-source deletion."
            )
        expected_variants = selected_variants(args.views, args.patch_size, args.context_size)
        missing_variants = [name for name in expected_variants if name not in marker.get("variants", [])]
        if missing_variants:
            raise ValueError(f"completed stage is missing requested variants {missing_variants}: {marker_path}")
        marker_patch_size = marker.get("target_patch_size")
        if marker_patch_size is not None and int(marker_patch_size) != args.patch_size:
            raise ValueError(f"completed stage uses target_patch_size={marker_patch_size}, expected {args.patch_size}")
        if int(marker.get("train_stride", args.train_stride)) != args.train_stride:
            raise ValueError(
                f"completed stage uses train_stride={marker.get('train_stride')}, expected {args.train_stride}"
            )
        if int(marker.get("eval_test_stride", args.stride)) != args.stride:
            raise ValueError(
                f"completed stage uses eval_test_stride={marker.get('eval_test_stride')}, expected {args.stride}"
            )
        if bool(marker.get("raw_lane_overlay", False)) != bool(args.raw_lane_overlay):
            raise ValueError(
                f"completed stage raw_lane_overlay={marker.get('raw_lane_overlay')}, expected {args.raw_lane_overlay}"
            )
        if bool(marker.get("require_raw_lane", False)) != bool(args.require_raw_lane):
            raise ValueError(
                f"completed stage require_raw_lane={marker.get('require_raw_lane')}, expected {args.require_raw_lane}"
            )
        if bool(marker.get("save_raw_lane_image", False)) != bool(args.save_raw_lane_image):
            raise ValueError(
                f"completed stage save_raw_lane_image={marker.get('save_raw_lane_image')}, "
                f"expected {args.save_raw_lane_image}"
            )
        if bool(marker.get("raw_lane_separate_image", False)) != bool(
            args.raw_lane_separate_image
        ):
            raise ValueError(
                "completed stage raw_lane_separate_image="
                f"{marker.get('raw_lane_separate_image')}, expected {args.raw_lane_separate_image}"
            )
        if bool(marker.get("pose_second_image", False)) != bool(args.pose_second_image):
            raise ValueError(
                f"completed stage pose_second_image={marker.get('pose_second_image')}, "
                f"expected {args.pose_second_image}"
            )
        if abs(float(marker.get("pose_threshold", 0.0)) - float(args.pose_threshold)) > 1e-12:
            raise ValueError(
                f"completed stage pose_threshold={marker.get('pose_threshold')}, expected {args.pose_threshold}"
            )
        marker_filter = marker.get("train_candidate_filter") or {}
        if str(marker_filter.get("sha256") or "") != candidate_filter_sha256:
            raise ValueError(
                f"completed stage uses a different train candidate filter: {marker_path}"
            )
        marker_fixed = marker.get("fixed_source_split") or {}
        expected_fixed_sha = str((fixed_descriptor or {}).get("file_sha256") or "")
        if str(marker_fixed.get("file_sha256") or "") != expected_fixed_sha:
            raise ValueError(
                f"completed stage uses a different fixed source split manifest: {marker_path}"
            )
        print(f"[dataset-v2-stage] completed shard already exists: {marker_path}", flush=True)
        if args.delete_input_root_after_stage and input_root.exists():
            if not args.delete_root_parent:
                raise ValueError("--delete-root-parent is required when deleting the input root")
            safe_remove_completed_input(input_root, Path(args.delete_root_parent), marker_path)
        return

    stage_root.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(
        input_root,
        include_intersections=True,
        delete_archives=not args.keep_archives,
        limit_samples=args.limit_samples,
        require_intersection_features=False,
        archive_workers=args.archive_workers,
        selective_archive_extract=args.selective_archive_extract,
    )
    if not samples:
        raise FileNotFoundError(f"no valid raw RC samples found under {input_root}")

    variants = selected_variants(args.views, args.patch_size, args.context_size)
    specs = stage_variant_specs(stage_root, args.views, args.patch_size, args.context_size)
    index_writers, sft_writers = open_stage_writers(stage_root, variants)
    split_counts = Counter()
    difficulty_counts = Counter()
    intersection_counts = Counter()
    image_counts = Counter()
    semantic_counts = Counter()
    try:
        for sample in tqdm(samples, desc=f"stage source {args.source_index}", unit="sample"):
            split = split_for_raw_sample(sample.sample_id, fixed_manifest)
            if split is None:
                split = stable_sample_split(sample.sample_id, args.split_seed, args.train_ratio, args.eval_ratio)
            process_args = copy.copy(args)
            process_args.stride = args.train_stride if split == "train" else args.stride
            rows = process_sample(
                sample,
                stage_root,
                split,
                True,
                process_args,
                write_images=False,
                max_empty_ratio=-1.0,
            )
            validate_rows(rows, True, args.patch_size, require_semantic_types=True)
            rows = [annotate_translation_grid(row, args.patch_size) for row in rows]
            if split == "train" and allowed_train_ids is not None:
                rows = [row for row in rows if str(row["id"]) in allowed_train_ids]
            for row in rows:
                row["meta"] = dict(row.get("meta", {}))
                row["meta"].update({
                    "source_index": args.source_index,
                    "source_uri": args.source_uri or str(input_root),
                    "staged_build": True,
                })
            write_images_for_rows(
                sample,
                rows,
                specs,
                args.patch_size,
                args.png_compress_level,
                skip_existing=args.resume,
                args=args,
            )
            for row in rows:
                metrics = classify_row(row, args.patch_size, args.coord_range)
                semantic_counts.update(
                    semantic_target_counts(row.get("target_lines", []), sample_id=row["id"], strict=True)
                )
                index_row = {
                    "id": row["id"],
                    "raw_sample_id": sample.sample_id,
                    "source_index": args.source_index,
                    "source_uri": args.source_uri or str(input_root),
                    "split": split,
                    "stratum": metrics["stratum"],
                    "difficulty": metrics["difficulty"],
                    "difficulty_score": metrics["difficulty_score"],
                    "has_intersection": metrics["has_intersection"],
                    "grid_kind": row["meta"]["grid_kind"],
                    "translation_offset": row["meta"]["translation_offset"],
                    "image": row["image"],
                }
                write_jsonl_item(index_writers[split], index_row)
                for variant, spec in specs.items():
                    rendered = variant_row(
                        row,
                        args.patch_size,
                        spec["context_size"],
                        spec["view_mode"],
                    )
                    sft = build_sft_record(
                        rendered,
                        args.patch_size,
                        True,
                        "a",
                        coord_mode=args.coord_mode,
                        coord_range=args.coord_range,
                        context_size=spec["context_size"],
                        view_mode=spec["view_mode"],
                        raw_lane_overlay=bool(getattr(args, "raw_lane_overlay", False)),
                        pose_second_image=bool(getattr(args, "pose_second_image", False)),
                        save_raw_lane_image=bool(getattr(args, "save_raw_lane_image", False)),
                        raw_lane_separate_image=bool(
                            getattr(args, "raw_lane_separate_image", False)
                        ),
                    )
                    semantic_sft_record_counts(sft, strict=True, require_prompt=True)
                    write_jsonl_item(sft_writers[variant][split], sft)
                    image_counts[variant] += 1
                split_counts[split] += 1
                difficulty_counts[metrics["stratum"]] += 1
                if metrics["has_intersection"]:
                    intersection_counts[split] += 1
    finally:
        close_writers(index_writers, sft_writers)

    summary = {
        "stage_version": STAGE_VERSION,
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
        "semantic_validation_passed": True,
        "difficulty_rule_version": DIFFICULTY_RULE_VERSION,
        "source_index": args.source_index,
        "source_uri": args.source_uri or str(input_root),
        "input_root": str(input_root),
        "raw_sample_count": len(samples),
        "split_policy": FIXED_SPLIT_POLICY if fixed_manifest else "sha256_sample_id_seed_threshold",
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "train_stride": args.train_stride,
        "eval_test_stride": args.stride,
        "target_patch_size": args.patch_size,
        "context_size": args.context_size,
        "raw_lane_overlay": bool(args.raw_lane_overlay),
        "raw_lane_overlay_source": "patch_tif/0_lane.tif" if args.raw_lane_overlay else "none",
        "raw_lane_threshold": args.raw_lane_threshold,
        "require_raw_lane": bool(args.require_raw_lane),
        "save_raw_lane_image": bool(args.save_raw_lane_image),
        "raw_lane_separate_image": bool(args.raw_lane_separate_image),
        "raw_lane_auxiliary_directory": "raw_lane_images" if args.save_raw_lane_image else "none",
        "pose_second_image": bool(args.pose_second_image),
        "pose_image_source": "patch_tif/0_pose.tif" if args.pose_second_image else "none",
        "pose_threshold": args.pose_threshold,
        "variants": variants,
        "split_record_counts": dict(split_counts),
        "difficulty_counts": dict(difficulty_counts),
        "intersection_counts": dict(intersection_counts),
        "semantic_target_counts": dict(semantic_counts),
        "image_counts": dict(image_counts),
        "train_candidate_filter": {
            "path": str(candidate_jsonl),
            "sha256": candidate_filter_sha256,
            "unique_ids": len(allowed_train_ids),
        } if candidate_jsonl else None,
        "fixed_source_split": fixed_descriptor,
        "selective_archive_extract": bool(args.selective_archive_extract),
    }
    if sum(split_counts.values()) <= 0:
        raise ValueError("source stage produced no candidate records")
    write_json(marker_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.delete_input_root_after_stage:
        if not args.delete_root_parent:
            raise ValueError("--delete-root-parent is required when deleting the input root")
        safe_remove_completed_input(input_root, Path(args.delete_root_parent), marker_path)


def discover_stage_roots(staging_root: Path) -> list[Path]:
    roots = sorted({path.parent for path in staging_root.rglob(STAGE_MARKER)})
    if not roots:
        raise FileNotFoundError(f"no completed source stages found under {staging_root}")
    return sorted(
        roots,
        key=lambda root: json.loads((root / STAGE_MARKER).read_text(encoding="utf-8"))["source_index"],
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_sample_owners(stage_roots: list[Path], duplicate_policy: str):
    owner = {}
    collisions = []
    for stage_root in stage_roots:
        summary = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        source_index = int(summary["source_index"])
        for split in ("train", "eval", "test"):
            index_path = stage_root / "records" / f"{split}.index.jsonl"
            for _, item in iter_jsonl(index_path):
                sample_id = str(item["raw_sample_id"])
                previous = owner.get(sample_id)
                if previous is not None and previous != source_index:
                    collisions.append({
                        "raw_sample_id": sample_id,
                        "previous_source_index": previous,
                        "new_source_index": source_index,
                    })
                    if duplicate_policy == "error":
                        raise ValueError(f"duplicate raw sample id across source stages: {sample_id}")
                    if duplicate_policy == "first":
                        continue
                owner[sample_id] = source_index
    return owner, collisions


def effective_record_split(
    item: dict,
    source_split: str,
    fixed_manifest: dict | None = None,
    repartition_fixed: bool = False,
) -> str:
    if not repartition_fixed:
        return source_split
    if fixed_manifest is None:
        raise ValueError("fixed manifest is required when repartitioning existing stages")
    return str(split_for_raw_sample(str(item["raw_sample_id"]), fixed_manifest))


def keep_repartitioned_record(item: dict, target_split: str, repartition_fixed: bool) -> bool:
    if not repartition_fixed or target_split == "train":
        return True
    # Bootstrap train uses stride=128 and therefore contains translated-grid
    # augmentation. Evaluation must remain on the canonical stride=256 grid.
    return str(item.get("grid_kind") or "") == "base"


def remap_split_asset_path(relative: str, source_split: str, target_split: str) -> str:
    path = PurePosixPath(str(relative))
    if source_split == target_split:
        return path.as_posix()
    parts = list(path.parts)
    if len(parts) < 3 or parts[1] != source_split:
        raise ValueError(
            f"cannot remap staged asset {relative!r} from {source_split} to {target_split}"
        )
    parts[1] = target_split
    return PurePosixPath(*parts).as_posix()


def remap_record_split_assets(record: dict, source_split: str, target_split: str) -> dict:
    if source_split == target_split:
        return record
    remapped = dict(record)
    for field in ("image", "pose_image", "raw_lane_image"):
        if remapped.get(field):
            remapped[field] = remap_split_asset_path(
                str(remapped[field]), source_split, target_split
            )
    if remapped.get("images"):
        remapped["images"] = [
            remap_split_asset_path(str(item), source_split, target_split)
            for item in remapped["images"]
        ]
    return remapped


def collect_owned_raw_sample_splits(
    stage_roots: list[Path],
    sample_owner: dict[str, int],
    fixed_manifest: dict | None = None,
    repartition_fixed: bool = False,
) -> dict[str, list[str]]:
    split_ids = {split: set() for split in ("train", "eval", "test")}
    for stage_root in stage_roots:
        summary = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        source_index = int(summary["source_index"])
        for source_split in split_ids:
            for _, item in iter_jsonl(stage_root / "records" / f"{source_split}.index.jsonl"):
                raw_sample_id = str(item["raw_sample_id"])
                if sample_owner.get(raw_sample_id) != source_index:
                    continue
                target_split = effective_record_split(
                    item, source_split, fixed_manifest, repartition_fixed
                )
                if keep_repartitioned_record(item, target_split, repartition_fixed):
                    split_ids[target_split].add(raw_sample_id)
    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"raw sample split leakage between {left}/{right}: {sorted(overlap)[:20]}")
    return {split: sorted(values) for split, values in split_ids.items()}


def load_train_candidate_ids(path: Path) -> set[str]:
    candidate_ids = set()
    for line_number, item in iter_jsonl(path):
        patch_id = str(item.get("id", "")).strip()
        if not patch_id:
            raise ValueError(f"candidate train JSONL has no id at {path}:{line_number}")
        if patch_id in candidate_ids:
            raise ValueError(f"duplicate id in candidate train JSONL: {patch_id}")
        candidate_ids.add(patch_id)
    if not candidate_ids:
        raise ValueError(f"candidate train JSONL is empty: {path}")
    return candidate_ids


def load_difficulty_overrides(path: Path | None) -> tuple[dict[str, dict], set[str]]:
    if path is None:
        return {}, set()
    overrides = {}
    versions = set()
    for line_number, item in iter_jsonl(path):
        patch_id = str(item.get("id", item.get("sample_id", ""))).strip()
        difficulty = str(item.get("stratum", item.get("difficulty", ""))).strip()
        if not patch_id:
            raise ValueError(f"difficulty override has no id at {path}:{line_number}")
        if difficulty not in DIFFICULTY_ORDER:
            raise ValueError(
                f"difficulty override has invalid bucket {difficulty!r} at {path}:{line_number}; "
                f"expected one of {DIFFICULTY_ORDER}"
            )
        if patch_id in overrides:
            raise ValueError(f"duplicate difficulty override id at {path}:{line_number}: {patch_id}")
        score = item.get("difficulty_score")
        overrides[patch_id] = {
            "difficulty": difficulty,
            "stratum": difficulty,
            "difficulty_score": float(score) if score is not None else None,
        }
        version = str(item.get("difficulty_rule_version", "")).strip()
        if version:
            versions.add(version)
    if not overrides:
        raise ValueError(f"difficulty override JSONL is empty: {path}")
    return overrides, versions


def load_candidate_pools(
    stage_roots,
    sample_owner,
    allowed_train_ids=None,
    difficulty_overrides=None,
    require_complete_override=False,
    fixed_manifest=None,
    repartition_fixed=False,
):
    base_pools = empty_candidate_pools()
    translated_pools = empty_candidate_pools()
    seen_patch_ids = set()
    candidate_counts = Counter()
    for stage_root in stage_roots:
        source_splits = ("train", "eval", "test") if repartition_fixed else ("train",)
        for source_split in source_splits:
            for _, item in iter_jsonl(stage_root / "records" / f"{source_split}.index.jsonl"):
                if sample_owner.get(str(item["raw_sample_id"])) != int(item["source_index"]):
                    continue
                target_split = effective_record_split(
                    item, source_split, fixed_manifest, repartition_fixed
                )
                if target_split != "train":
                    continue
                patch_id = str(item["id"])
                if allowed_train_ids is not None and patch_id not in allowed_train_ids:
                    continue
                if patch_id in seen_patch_ids:
                    continue
                seen_patch_ids.add(patch_id)
                override = (difficulty_overrides or {}).get(patch_id)
                if override is None and require_complete_override and str(item.get("stratum")) != "empty":
                    raise ValueError(f"difficulty override is missing non-empty train id: {patch_id}")
                stratum = str((override or item)["stratum"])
                pool_type = "intersection" if item.get("has_intersection") else "plain"
                pools = base_pools if item.get("grid_kind") == "base" else translated_pools
                pools[stratum][pool_type].append(patch_id)
                candidate_counts[stratum] += 1
    return base_pools, translated_pools, candidate_counts


def link_or_copy(source: Path, destination: Path, mode: str, resume: bool) -> str:
    if resume and destination.is_file():
        return "reused"
    if not source.is_file():
        raise FileNotFoundError(f"staged candidate image not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
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


def materialize_from_stages(
    stage_roots,
    variants,
    output_root,
    sample_owner,
    selected_train_ids,
    copy_mode,
    resume,
    difficulty_overrides=None,
    fixed_manifest=None,
    repartition_fixed=False,
):
    output_handles = {}
    meta_handles = {}
    for variant in variants:
        phase_root = output_root / variant / "phase_a"
        phase_root.mkdir(parents=True, exist_ok=True)
        output_handles[variant] = {
            split: (phase_root / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in ("train", "eval", "test")
        }
        meta_handles[variant] = {
            split: (phase_root / f"meta_{split}.jsonl").open("w", encoding="utf-8")
            for split in ("train", "eval", "test")
        }

    counts = Counter()
    semantic_counts = Counter()
    link_modes = Counter()
    seen_by_split = {split: set() for split in ("train", "eval", "test")}
    try:
        for stage_root in stage_roots:
            for source_split in ("train", "eval", "test"):
                index_path = stage_root / "records" / f"{source_split}.index.jsonl"
                indexes = iter_jsonl(index_path)
                sft_iters = {
                    variant: iter_jsonl(
                        stage_root / "records" / variant / f"{source_split}.jsonl"
                    )
                    for variant in variants
                }
                for _, index_item in indexes:
                    sft_items = {variant: next(sft_iters[variant])[1] for variant in variants}
                    patch_id = str(index_item["id"])
                    if any(str(item.get("id")) != patch_id for item in sft_items.values()):
                        raise ValueError(f"stage index/SFT order mismatch for {patch_id}")
                    if sample_owner.get(str(index_item["raw_sample_id"])) != int(index_item["source_index"]):
                        continue
                    target_split = effective_record_split(
                        index_item, source_split, fixed_manifest, repartition_fixed
                    )
                    if not keep_repartitioned_record(
                        index_item, target_split, repartition_fixed
                    ):
                        continue
                    if patch_id in seen_by_split[target_split]:
                        continue
                    if target_split == "train" and patch_id not in selected_train_ids:
                        continue
                    seen_by_split[target_split].add(patch_id)
                    override = (
                        (difficulty_overrides or {}).get(patch_id)
                        if target_split == "train" else None
                    )
                    effective_difficulty = (override or index_item).get("difficulty")
                    effective_score = (override or index_item).get("difficulty_score")
                    effective_stratum = (override or index_item).get("stratum")
                    for variant, record in sft_items.items():
                        source_record = record
                        record = remap_record_split_assets(
                            source_record, source_split, target_split
                        )
                        record_semantic_counts = semantic_sft_record_counts(
                            record, strict=True, require_prompt=True
                        )
                        semantic_counts.update(
                            {f"{variant}:{key}": value for key, value in record_semantic_counts.items()}
                        )
                        source_relative_images = source_record.get("images") or [
                            source_record["image"]
                        ]
                        relative_images = record.get("images") or [record["image"]]
                        if not isinstance(relative_images, list) or not relative_images:
                            raise ValueError(f"record {patch_id} has invalid images={relative_images!r}")
                        source_assets = [str(item) for item in source_relative_images]
                        target_assets = [str(item) for item in relative_images]
                        if source_record.get("raw_lane_image"):
                            source_assets.append(str(source_record["raw_lane_image"]))
                            target_assets.append(str(record["raw_lane_image"]))
                        asset_pairs = dict.fromkeys(zip(source_assets, target_assets))
                        for source_relative, target_relative in asset_pairs:
                            source_image = (
                                stage_root / "variants" / variant / Path(source_relative)
                            )
                            destination_image = (
                                output_root / variant / Path(target_relative)
                            )
                            used_mode = link_or_copy(source_image, destination_image, copy_mode, resume)
                            link_modes[used_mode] += 1
                        write_jsonl_item(output_handles[variant][target_split], record)
                        write_jsonl_item(meta_handles[variant][target_split], {
                            "id": patch_id,
                            "image": record["image"],
                            "images": relative_images,
                            "raw_lane_image": record.get("raw_lane_image"),
                            "meta": record.get("meta", {}),
                            "difficulty": effective_difficulty,
                            "difficulty_score": effective_score,
                            "stratum": effective_stratum,
                            "has_intersection": index_item.get("has_intersection"),
                            "grid_kind": index_item.get("grid_kind"),
                            "source_index": index_item.get("source_index"),
                        })
                        counts[f"{variant}:{target_split}"] += 1
                for variant, iterator in sft_iters.items():
                    try:
                        next(iterator)
                    except StopIteration:
                        continue
                    raise ValueError(
                        f"extra staged SFT rows in {stage_root} {variant}/{source_split}"
                    )
    finally:
        close_writers(output_handles, meta_handles)
    return dict(counts), dict(link_modes), dict(semantic_counts)


def finalize_stages(args) -> None:
    patch_size = int(getattr(args, "patch_size", 256))
    context_size = int(getattr(args, "context_size", 512))
    if patch_size <= 0 or context_size < patch_size:
        raise ValueError("--patch-size must be positive and --context-size must be >= --patch-size")
    if (context_size - patch_size) % 2:
        raise ValueError("--context-size minus --patch-size must be even for a centered ROI")
    if args.coord_range != DEFAULT_COORD_RANGE:
        raise ValueError(f"--coord-range must be {DEFAULT_COORD_RANGE}")
    if not 0 <= args.intersection_target_ratio <= 1:
        raise ValueError("--intersection-target-ratio must be in [0, 1]")
    ratios = parse_ratio_spec(args.difficulty_ratios)
    fixed_manifest_text = str(getattr(args, "fixed_source_split_manifest", "") or "").strip()
    fixed_manifest = (
        load_fixed_source_split_manifest(fixed_manifest_text) if fixed_manifest_text else None
    )
    repartition_fixed = bool(
        getattr(args, "repartition_existing_stages_by_fixed_manifest", False)
    )
    if repartition_fixed and fixed_manifest is None:
        raise ValueError(
            "--repartition-existing-stages-by-fixed-manifest requires "
            "--fixed-source-split-manifest"
        )
    fixed_descriptor = fixed_split_descriptor(fixed_manifest)
    expected_fixed_sha = str((fixed_descriptor or {}).get("file_sha256") or "")
    staging_root = Path(args.staging_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_roots = discover_stage_roots(staging_root)
    variants = selected_variants(args.views, patch_size, context_size)
    raw_lane_overlay_values = set()
    require_raw_lane_values = set()
    raw_lane_threshold_values = set()
    save_raw_lane_image_values = set()
    raw_lane_separate_image_values = set()
    pose_second_image_values = set()
    pose_threshold_values = set()
    for stage_root in stage_roots:
        summary = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        if summary.get("stage_version") != STAGE_VERSION or not summary.get("semantic_validation_passed"):
            raise ValueError(
                f"stage {stage_root} uses stale or unverified schema; expected {STAGE_VERSION}. "
                "Rebuild the source stage before finalization."
            )
        missing = [variant for variant in variants if variant not in summary.get("variants", [])]
        if missing:
            raise ValueError(f"stage {stage_root} does not contain requested variants: {missing}")
        stage_patch_size = summary.get("target_patch_size")
        if stage_patch_size is None and patch_size != 256:
            raise ValueError(f"legacy stage has no target patch size and cannot be used for {patch_size}: {stage_root}")
        if stage_patch_size is not None and int(stage_patch_size) != patch_size:
            raise ValueError(
                f"stage target_patch_size={stage_patch_size}, expected {patch_size}: {stage_root}"
            )
        stage_fixed = summary.get("fixed_source_split") or {}
        stage_fixed_sha = str(stage_fixed.get("file_sha256") or "")
        fixed_mismatch = stage_fixed_sha != expected_fixed_sha
        reusable_bootstrap_stage = repartition_fixed and not stage_fixed_sha
        if fixed_mismatch and not reusable_bootstrap_stage:
            raise ValueError(
                f"stage fixed split manifest does not match finalization: {stage_root}; "
                f"stage_sha={stage_fixed.get('file_sha256')!r}, expected_sha={expected_fixed_sha!r}"
            )
        raw_lane_overlay_values.add(bool(summary.get("raw_lane_overlay", False)))
        require_raw_lane_values.add(bool(summary.get("require_raw_lane", False)))
        raw_lane_threshold_values.add(float(summary.get("raw_lane_threshold", 0.0)))
        save_raw_lane_image_values.add(bool(summary.get("save_raw_lane_image", False)))
        raw_lane_separate_image_values.add(
            bool(summary.get("raw_lane_separate_image", False))
        )
        pose_second_image_values.add(bool(summary.get("pose_second_image", False)))
        pose_threshold_values.add(float(summary.get("pose_threshold", 0.0)))
    if (
        len(raw_lane_overlay_values) > 1
        or len(require_raw_lane_values) > 1
        or len(raw_lane_threshold_values) > 1
        or len(save_raw_lane_image_values) > 1
        or len(raw_lane_separate_image_values) > 1
        or len(pose_second_image_values) > 1
        or len(pose_threshold_values) > 1
    ):
        raise ValueError("cannot finalize source stages with mixed raw-lane/pose image settings")
    raw_lane_overlay = next(iter(raw_lane_overlay_values), False)
    require_raw_lane = next(iter(require_raw_lane_values), False)
    raw_lane_threshold = next(iter(raw_lane_threshold_values), 0.0)
    save_raw_lane_image = next(iter(save_raw_lane_image_values), False)
    raw_lane_separate_image = next(iter(raw_lane_separate_image_values), False)
    pose_second_image = next(iter(pose_second_image_values), False)
    pose_threshold = next(iter(pose_threshold_values), 0.0)

    sample_owner, collisions = build_sample_owners(stage_roots, args.duplicate_policy)
    raw_sample_ids_by_split = collect_owned_raw_sample_splits(
        stage_roots,
        sample_owner,
        fixed_manifest=fixed_manifest,
        repartition_fixed=repartition_fixed,
    )
    fixed_coverage = None
    if fixed_manifest is not None:
        fixed_coverage = validate_fixed_holdout_coverage(
            raw_sample_ids_by_split,
            fixed_manifest,
            allow_missing=bool(getattr(args, "allow_missing_fixed_holdouts", False)),
        )
    candidate_jsonl_text = str(getattr(args, "train_candidate_jsonl", "") or "").strip()
    candidate_jsonl = Path(candidate_jsonl_text) if candidate_jsonl_text else None
    allowed_train_ids = load_train_candidate_ids(candidate_jsonl) if candidate_jsonl else None
    override_jsonl_text = str(getattr(args, "difficulty_override_jsonl", "") or "").strip()
    override_jsonl = Path(override_jsonl_text) if override_jsonl_text else None
    difficulty_overrides, override_versions = load_difficulty_overrides(override_jsonl)
    explicit_rule_version = str(getattr(args, "difficulty_rule_version", "") or "").strip()
    if explicit_rule_version:
        difficulty_rule_version = explicit_rule_version
    elif len(override_versions) == 1:
        difficulty_rule_version = next(iter(override_versions))
    elif not override_versions:
        difficulty_rule_version = DIFFICULTY_RULE_VERSION
    else:
        raise ValueError(f"difficulty override mixes rule versions: {sorted(override_versions)}")
    base_pools, translated_pools, candidate_counts = load_candidate_pools(
        stage_roots,
        sample_owner,
        allowed_train_ids,
        difficulty_overrides,
        require_complete_override=override_jsonl is not None,
        fixed_manifest=fixed_manifest,
        repartition_fixed=repartition_fixed,
    )
    selected_counts, balance_report = select_balanced_candidates(
        base_pools,
        translated_pools,
        args.train_target_samples,
        ratios,
        args.intersection_target_ratio,
        args.difficulty_seed,
        strict_difficulty_quotas=bool(
            getattr(args, "strict_difficulty_quotas", False)
        ),
    )
    if balance_report["selected_total"] != args.train_target_samples:
        raise ValueError(
            f"unable to select {args.train_target_samples} unique records; "
            f"selected {balance_report['selected_total']}"
        )
    if abs(balance_report["actual_intersection_ratio"] - args.intersection_target_ratio) > 1e-8:
        raise ValueError(
            "unable to satisfy exact global intersection ratio: "
            f"target={args.intersection_target_ratio}, actual={balance_report['actual_intersection_ratio']}"
        )

    counts, link_modes, semantic_counts = materialize_from_stages(
        stage_roots,
        variants,
        output_root,
        sample_owner,
        set(selected_counts),
        args.copy_mode,
        args.resume,
        difficulty_overrides,
        fixed_manifest=fixed_manifest,
        repartition_fixed=repartition_fixed,
    )
    for variant in variants:
        if counts.get(f"{variant}:train", 0) != args.train_target_samples:
            raise ValueError(f"final train count mismatch for {variant}: {counts}")

    balance_report["difficulty_rule_version"] = difficulty_rule_version
    balance_report["cut_affects_difficulty"] = False
    summary = {
        "dataset_version": "rc_dataset_v2_staged_stage_a_semantic_v1",
        "stage_version": STAGE_VERSION,
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
        "semantic_validation_passed": True,
        "difficulty_rule_version": difficulty_rule_version,
        "staging_root": str(staging_root),
        "source_stage_count": len(stage_roots),
        "source_stages": [str(path) for path in stage_roots],
        "raw_sample_owner_count": len(sample_owner),
        "duplicate_policy": args.duplicate_policy,
        "duplicate_raw_sample_events": collisions,
        "candidate_train_counts": dict(candidate_counts),
        "train_candidate_filter": {
            "path": str(candidate_jsonl),
            "unique_ids": len(allowed_train_ids),
            "selection_is_subset": True,
        } if candidate_jsonl else None,
        "difficulty_override": {
            "path": str(override_jsonl),
            "records": len(difficulty_overrides),
            "rule_version": difficulty_rule_version,
        } if override_jsonl else None,
        "balance": balance_report,
        "variants": variants,
        "record_counts": counts,
        "image_materialization_modes": link_modes,
        "semantic_target_counts": semantic_counts,
        "split_policy": FIXED_SPLIT_POLICY if fixed_manifest else "sha256_sample_id_seed_threshold",
        "fixed_source_split": fixed_descriptor,
        "fixed_source_split_coverage": fixed_coverage,
        "source_stage_split_repartition": {
            "enabled": repartition_fixed,
            "source_policy": "bootstrap_hash_split" if repartition_fixed else "native_stage_split",
            "fixed_holdout_grid_policy": "base_only" if repartition_fixed else "native_stage_grid",
            "train_candidate_policy": "reuse_all_available_grids",
        },
        "coord_mode": COORD_MODE_NORM1000,
        "coord_range": args.coord_range,
        "target_patch_size": patch_size,
        "context_size": context_size,
        "input_overlay": {
            "raw_lane_overlay": raw_lane_overlay,
            "raw_lane_overlay_source": "patch_tif/0_lane.tif" if raw_lane_overlay else "none",
            "raw_lane_threshold": raw_lane_threshold,
            "require_raw_lane": require_raw_lane,
            "overlay_style": "white_pixels_on_rgb_channels",
            "raw_lane_auxiliary_saved": save_raw_lane_image,
            "raw_lane_auxiliary_directory": "raw_lane_images" if save_raw_lane_image else "none",
            "raw_lane_separate_image": raw_lane_separate_image,
        },
        "auxiliary_image_assets": {
            "raw_lane": {
                "saved": save_raw_lane_image,
                "active_model_input": raw_lane_separate_image,
                "source": "patch_tif/0_lane.tif" if save_raw_lane_image else "none",
                "rendering": "white_positive_pixels_on_black_rgb",
                "record_field": "raw_lane_image" if save_raw_lane_image else "none",
            }
        },
        "multi_image_input": {
            "enabled": raw_lane_separate_image or pose_second_image,
            "num_images_per_sample": (
                1 + int(raw_lane_separate_image) + int(pose_second_image)
            ),
            "image_roles": [
                "bev_road_structure",
                *(["pv_camera_raw_lane"] if raw_lane_separate_image else []),
                *(["historical_vehicle_trajectory"] if pose_second_image else []),
            ],
            "raw_lane_image_source": (
                "patch_tif/0_lane.tif" if raw_lane_separate_image else "none"
            ),
            "pose_image_source": "patch_tif/0_pose.tif" if pose_second_image else "none",
            "pose_threshold": pose_threshold,
            "pose_rendering": "white_positive_pixels_on_black_rgb",
        },
    }
    write_json(output_root / "build_summary.json", summary)
    write_json(
        output_root / "semantic_schema_report.json",
        {
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "validation_passed": True,
            "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
            "allowed_lane_types": sorted(ALLOWED_LANE_TYPES),
            "allowed_intersection_types": [
                "common",
                "t_intersection",
                "small_untyped",
                "t_lane_change_area",
                "other",
            ],
            "target_counts": semantic_counts,
        },
    )
    write_json(output_root / "balance_report.json", balance_report)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_root / "balance_report.json", balance_report)
    with (manifest_root / "train_selection.jsonl").open("w", encoding="utf-8") as handle:
        for difficulty in DIFFICULTY_ORDER:
            for grid_kind, pools in (("base", base_pools), ("translated", translated_pools)):
                for pool_type in ("intersection", "plain"):
                    for patch_id in pools[difficulty][pool_type]:
                        if patch_id in selected_counts:
                            write_jsonl_item(handle, {
                                "id": patch_id,
                                "stratum": difficulty,
                                "has_intersection": pool_type == "intersection",
                                "grid_kind": grid_kind,
                                "exact_repeat": False,
                            })
    split_manifest = {
        "dataset_version": summary["dataset_version"],
        "split_unit": "raw_sample_id",
        "split_policy": summary["split_policy"],
        "fixed_source_split": fixed_descriptor,
        "fixed_source_split_coverage": fixed_coverage,
        "raw_sample_owner_count": len(sample_owner),
        "duplicate_policy": args.duplicate_policy,
        "duplicate_raw_sample_events": collisions,
        "raw_sample_ids_by_split": raw_sample_ids_by_split,
        "raw_sample_counts_by_split": {
            split: len(values) for split, values in raw_sample_ids_by_split.items()
        },
        "source_stages": [
            json.loads((path / STAGE_MARKER).read_text(encoding="utf-8"))
            for path in stage_roots
        ],
    }
    write_json(output_root / "split_manifest.json", split_manifest)
    for variant in variants:
        write_json(output_root / variant / "dataset_info.json", summary)
        write_json(output_root / variant / "balance_report.json", balance_report)
        write_json(output_root / variant / "split_manifest.json", split_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "stage":
        stage_source(args)
    else:
        finalize_stages(args)


if __name__ == "__main__":
    main()
