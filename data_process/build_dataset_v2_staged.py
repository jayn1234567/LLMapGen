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
from pathlib import Path

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
        marker_filter = marker.get("train_candidate_filter") or {}
        if str(marker_filter.get("sha256") or "") != candidate_filter_sha256:
            raise ValueError(
                f"completed stage uses a different train candidate filter: {marker_path}"
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
        "split_policy": "sha256_sample_id_seed_threshold",
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


def collect_owned_raw_sample_splits(stage_roots: list[Path], sample_owner: dict[str, int]) -> dict[str, list[str]]:
    split_ids = {split: set() for split in ("train", "eval", "test")}
    for stage_root in stage_roots:
        summary = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        source_index = int(summary["source_index"])
        for split in split_ids:
            for _, item in iter_jsonl(stage_root / "records" / f"{split}.index.jsonl"):
                raw_sample_id = str(item["raw_sample_id"])
                if sample_owner.get(raw_sample_id) == source_index:
                    split_ids[split].add(raw_sample_id)
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
):
    base_pools = empty_candidate_pools()
    translated_pools = empty_candidate_pools()
    seen_patch_ids = set()
    candidate_counts = Counter()
    for stage_root in stage_roots:
        for _, item in iter_jsonl(stage_root / "records" / "train.index.jsonl"):
            if sample_owner.get(str(item["raw_sample_id"])) != int(item["source_index"]):
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
            for split in ("train", "eval", "test"):
                index_path = stage_root / "records" / f"{split}.index.jsonl"
                indexes = iter_jsonl(index_path)
                sft_iters = {
                    variant: iter_jsonl(stage_root / "records" / variant / f"{split}.jsonl")
                    for variant in variants
                }
                for _, index_item in indexes:
                    sft_items = {variant: next(sft_iters[variant])[1] for variant in variants}
                    patch_id = str(index_item["id"])
                    if any(str(item.get("id")) != patch_id for item in sft_items.values()):
                        raise ValueError(f"stage index/SFT order mismatch for {patch_id}")
                    if sample_owner.get(str(index_item["raw_sample_id"])) != int(index_item["source_index"]):
                        continue
                    if patch_id in seen_by_split[split]:
                        continue
                    if split == "train" and patch_id not in selected_train_ids:
                        continue
                    seen_by_split[split].add(patch_id)
                    override = (difficulty_overrides or {}).get(patch_id) if split == "train" else None
                    effective_difficulty = (override or index_item).get("difficulty")
                    effective_score = (override or index_item).get("difficulty_score")
                    effective_stratum = (override or index_item).get("stratum")
                    for variant, record in sft_items.items():
                        record_semantic_counts = semantic_sft_record_counts(
                            record, strict=True, require_prompt=True
                        )
                        semantic_counts.update(
                            {f"{variant}:{key}": value for key, value in record_semantic_counts.items()}
                        )
                        relative_image = Path(str(record["image"]))
                        source_image = stage_root / "variants" / variant / relative_image
                        destination_image = output_root / variant / relative_image
                        used_mode = link_or_copy(source_image, destination_image, copy_mode, resume)
                        link_modes[used_mode] += 1
                        write_jsonl_item(output_handles[variant][split], record)
                        write_jsonl_item(meta_handles[variant][split], {
                            "id": patch_id,
                            "image": record["image"],
                            "meta": record.get("meta", {}),
                            "difficulty": effective_difficulty,
                            "difficulty_score": effective_score,
                            "stratum": effective_stratum,
                            "has_intersection": index_item.get("has_intersection"),
                            "grid_kind": index_item.get("grid_kind"),
                            "source_index": index_item.get("source_index"),
                        })
                        counts[f"{variant}:{split}"] += 1
                for variant, iterator in sft_iters.items():
                    try:
                        next(iterator)
                    except StopIteration:
                        continue
                    raise ValueError(f"extra staged SFT rows in {stage_root} {variant}/{split}")
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
    staging_root = Path(args.staging_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_roots = discover_stage_roots(staging_root)
    variants = selected_variants(args.views, patch_size, context_size)
    raw_lane_overlay_values = set()
    require_raw_lane_values = set()
    raw_lane_threshold_values = set()
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
        raw_lane_overlay_values.add(bool(summary.get("raw_lane_overlay", False)))
        require_raw_lane_values.add(bool(summary.get("require_raw_lane", False)))
        raw_lane_threshold_values.add(float(summary.get("raw_lane_threshold", 0.0)))
    if len(raw_lane_overlay_values) > 1 or len(require_raw_lane_values) > 1 or len(raw_lane_threshold_values) > 1:
        raise ValueError("cannot finalize a mixture of raw-lane and non-raw-lane source stages")
    raw_lane_overlay = next(iter(raw_lane_overlay_values), False)
    require_raw_lane = next(iter(require_raw_lane_values), False)
    raw_lane_threshold = next(iter(raw_lane_threshold_values), 0.0)

    sample_owner, collisions = build_sample_owners(stage_roots, args.duplicate_policy)
    raw_sample_ids_by_split = collect_owned_raw_sample_splits(stage_roots, sample_owner)
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
    )
    selected_counts, balance_report = select_balanced_candidates(
        base_pools,
        translated_pools,
        args.train_target_samples,
        ratios,
        args.intersection_target_ratio,
        args.difficulty_seed,
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
        "split_policy": "sha256_sample_id_seed_threshold",
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
