#!/usr/bin/env python3
"""Validate a completed RC Dataset V2 view and visualize difficulty buckets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Iterator

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import (
    DIFFICULTY_ARGS,
    DIFFICULTY_ORDER,
    allocate_quotas,
    parse_ratio_spec,
)
from data_process.difficulty_profiles import (
    DIFFICULTY_PROFILE_VERSION,
    classify_metrics,
    resolution_aware_score,
    resolve_difficulty_profile,
)
from data_process.state_update_dataset_common import (
    ALLOWED_INTERSECTION_TYPES,
    ALLOWED_LANE_TYPES,
    IGNORED_LANE_TYPE_CODES,
    SEMANTIC_SCHEMA_VERSION,
)
from scripts.tools.tag_hard_map_samples import (
    DIFFICULTY_RULE_VERSION,
    draw_overlay,
    make_contact_sheet,
    safe_name,
    sample_metrics,
)
from scripts.tools.derive_intersection_prompt_dataset import (
    PROMPT_MARKER,
    TASK_MODE as INTERSECTION_PROMPT_TASK_MODE,
    extract_prompt_intersections,
)


DEFAULT_DIFFICULTY_RATIOS = "empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10"
SPLITS = ("train", "eval", "test")
VARIANT_SPECS = {
    "local256": {
        "image_size": (256, 256),
        "target_size": 256,
        "context_image_size": 256,
        "target_roi_in_image": [0, 0, 256, 256],
        "view_mode": "local256",
        "task_mode": "lane_intersection",
    },
    "local256v2": {
        "image_size": (256, 256),
        "target_size": 256,
        "context_image_size": 256,
        "target_roi_in_image": [0, 0, 256, 256],
        "view_mode": "local256",
        "task_mode": "lane_intersection",
    },
    "context512_roi256": {
        "image_size": (512, 512),
        "target_size": 256,
        "context_image_size": 512,
        "target_roi_in_image": [128, 128, 384, 384],
        "view_mode": "context512_roi256",
        "task_mode": "lane_intersection",
    },
    "rawlane_local256_550k": {
        "image_size": (256, 256),
        "target_size": 256,
        "context_image_size": 256,
        "target_roi_in_image": [0, 0, 256, 256],
        "view_mode": "local256",
        "task_mode": "lane_intersection",
    },
    "rawlane_context512_roi256_550k": {
        "image_size": (512, 512),
        "target_size": 256,
        "context_image_size": 512,
        "target_roi_in_image": [128, 128, 384, 384],
        "view_mode": "context512_roi256",
        "task_mode": "lane_intersection",
    },
    "rawlane_pose_local256_800k": {
        "image_size": (256, 256),
        "target_size": 256,
        "context_image_size": 256,
        "target_roi_in_image": [0, 0, 256, 256],
        "view_mode": "local256",
        "task_mode": "lane_intersection",
    },
    "context512_roi256v3": {
        "image_size": (512, 512),
        "target_size": 256,
        "context_image_size": 512,
        "target_roi_in_image": [128, 128, 384, 384],
        "view_mode": "context512_roi256",
        "task_mode": "lane_intersection",
    },
    "local512": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": "lane_intersection",
    },
    "local512v2": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": "lane_intersection",
    },
    "local512v3": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": "lane_intersection",
    },
    "local512v3_1000k": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": "lane_intersection",
    },
    "local512v3_550k_stageab": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": "lane_intersection",
    },
    "local512v3_rot45_135_800k": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": "lane_intersection",
    },
    "local512_intersection_prompt": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": INTERSECTION_PROMPT_TASK_MODE,
    },
    "local512v2_intersection_prompt": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": INTERSECTION_PROMPT_TASK_MODE,
    },
    "local512v3_intersection_prompt": {
        "image_size": (512, 512),
        "target_size": 512,
        "context_image_size": 512,
        "target_roi_in_image": [0, 0, 512, 512],
        "view_mode": "local512",
        "task_mode": INTERSECTION_PROMPT_TASK_MODE,
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Completed Dataset V2 variant root.")
    parser.add_argument(
        "--variant",
        choices=["auto", *VARIANT_SPECS],
        default="auto",
        help="Dataset view; auto uses the dataset-root directory name.",
    )
    parser.add_argument("--output-dir", default="", help="Audit output; defaults beside dataset-root.")
    parser.add_argument("--phase", default="phase_a")
    parser.add_argument("--expected-train-samples", type=int, default=550000)
    parser.add_argument("--difficulty-ratios", default=DEFAULT_DIFFICULTY_RATIOS)
    parser.add_argument("--expected-intersection-ratio", type=float, default=0.30)
    parser.add_argument("--count-tolerance", type=int, default=0)
    parser.add_argument("--skip-distribution-check", action="store_true")
    parser.add_argument("--visualize-per-difficulty", type=int, default=50)
    parser.add_argument("--allow-short-visual-buckets", action="store_true")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--image-decode-mode",
        choices=["sampled", "all", "none"],
        default="sampled",
        help="All image paths are checked; this controls PNG decoding/verification.",
    )
    parser.add_argument("--image-decode-samples-per-split", type=int, default=5000)
    parser.add_argument("--skip-extra-image-scan", action="store_true")
    parser.add_argument("--max-error-examples", type=int, default=1000)
    return parser.parse_args(argv)


@dataclass
class ErrorCollector:
    max_examples: int
    total: int = 0
    counts: Counter = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, message: str, *, split="", line_number=0, sample_id="") -> None:
        self.total += 1
        self.counts[code] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append({
                "code": code,
                "message": message,
                "split": split,
                "line_number": line_number,
                "sample_id": sample_id,
            })


def resolve_split_path(dataset_root: Path, phase: str, split: str) -> Path | None:
    phase_root = dataset_root / phase
    candidates = [phase_root / f"{split}.jsonl"]
    if split == "eval":
        candidates.append(phase_root / "val.jsonl")
    return next((path for path in candidates if path.is_file()), None)


def resolve_variant(dataset_root: Path, requested: str) -> tuple[str, dict[str, Any]]:
    variant = str(requested or "auto")
    if variant == "auto":
        variant = dataset_root.name
    if variant not in VARIANT_SPECS:
        raise ValueError(
            f"Unable to infer Dataset V2 variant from {dataset_root}; pass --variant explicitly."
        )
    return variant, VARIANT_SPECS[variant]


def iter_jsonl(path: Path, split: str, errors: ErrorCollector) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.add("invalid_jsonl", str(exc), split=split, line_number=line_number)
                continue
            if not isinstance(record, dict):
                errors.add(
                    "record_not_object",
                    f"expected object, got {type(record).__name__}",
                    split=split,
                    line_number=line_number,
                )
                continue
            yield line_number, record


def conversation_value(record: dict[str, Any], roles: set[str]) -> Any:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from", item.get("role", ""))).strip().lower()
        if role in roles:
            return item.get("value", item.get("content"))
    return None


def parse_target(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    value = conversation_value(record, {"gpt", "assistant"})
    if not isinstance(value, str):
        return None, "assistant value must be a JSON string"
    stripped = value.strip()
    if stripped.startswith("```"):
        return None, "assistant target contains a markdown fence"
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"assistant JSON is invalid: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        return None, 'assistant JSON must be an object containing a "lines" list'
    return payload, None


def record_with_target_lines(record: dict[str, Any], lines: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(record)
    conversations = []
    replaced = False
    for message in record.get("conversations", []):
        item = dict(message)
        role = str(item.get("from", item.get("role", ""))).strip().lower()
        if role in {"gpt", "assistant"}:
            key = "value" if "value" in item or "content" not in item else "content"
            item[key] = json.dumps({"lines": lines}, ensure_ascii=False, separators=(",", ":"))
            replaced = True
        conversations.append(item)
    if not replaced:
        conversations.append({"from": "gpt", "value": json.dumps({"lines": lines}, separators=(",", ":"))})
    result["conversations"] = conversations
    return result


def safe_image_path(dataset_root: Path, relative: Any) -> tuple[Path | None, str | None, str]:
    if not isinstance(relative, str) or not relative.strip():
        return None, "record.image must be a non-empty string", ""
    normalized = relative.replace("\\", "/").lstrip("/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        return None, f"unsafe image path: {relative!r}", normalized
    path = dataset_root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return None, f"image path escapes dataset root: {relative!r}", normalized
    return path, None, normalized


def valid_point(point: Any) -> bool:
    return (
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in point)
        and all(0 <= value <= 1000 for value in point)
    )


def validate_target_lines(
    lines: list[Any], errors: ErrorCollector, split: str, line_number: int, sample_id: str
) -> tuple[Counter, bool]:
    semantic_counts = Counter()
    has_intersection = False
    for index, item in enumerate(lines):
        if not isinstance(item, dict):
            errors.add(
                "target_line_not_object",
                f"target lines[{index}] is not an object",
                split=split,
                line_number=line_number,
                sample_id=sample_id,
            )
            continue
        category = str(item.get("category", "")).strip().lower()
        points = item.get("points")
        if not isinstance(points, list) or not all(valid_point(point) for point in points):
            errors.add(
                "invalid_points",
                f"target lines[{index}] has invalid norm1000 points",
                split=split,
                line_number=line_number,
                sample_id=sample_id,
            )
        if category == "centerline":
            lane_type = item.get("lane_type")
            semantic_counts[f"lane_type:{lane_type}"] += 1
            if lane_type not in ALLOWED_LANE_TYPES:
                errors.add(
                    "invalid_lane_type",
                    f"target lines[{index}] lane_type={lane_type!r}",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            if item.get("start_type") not in {"cut", "inside"} or item.get("end_type") not in {"cut", "inside"}:
                errors.add(
                    "invalid_endpoint_type",
                    f"target lines[{index}] has invalid start_type/end_type",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            if not isinstance(points, list) or len(points) < 2:
                errors.add(
                    "short_centerline",
                    f"target lines[{index}] has fewer than two points",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
        elif category == "intersection":
            has_intersection = True
            intersection_type = item.get("intersection_type")
            semantic_counts[f"intersection_type:{intersection_type}"] += 1
            if intersection_type not in ALLOWED_INTERSECTION_TYPES:
                errors.add(
                    "invalid_intersection_type",
                    f"target lines[{index}] intersection_type={intersection_type!r}",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            normalized_keys = {
                "".join(character for character in str(key).lower() if character.isalnum())
                for key in item
            }
            if "intersectionsubtype" in normalized_keys:
                errors.add(
                    "unexpected_intersection_subtype",
                    f"target lines[{index}] still contains intersection subtype",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            if not isinstance(item.get("is_cut"), bool):
                errors.add(
                    "invalid_intersection_is_cut",
                    f"target lines[{index}] is_cut must be boolean",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            if not isinstance(points, list) or len(points) < 4 or points[0] != points[-1]:
                errors.add(
                    "open_intersection_polygon",
                    f"target lines[{index}] polygon is not closed",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
        else:
            errors.add(
                "unsupported_category",
                f"target lines[{index}] category={category!r}",
                split=split,
                line_number=line_number,
                sample_id=sample_id,
            )
    return semantic_counts, has_intersection


def validate_metadata(
    record: dict[str, Any],
    errors: ErrorCollector,
    split: str,
    line_number: int,
    sample_id: str,
    variant: str,
    variant_spec: dict[str, Any],
) -> str:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        errors.add("missing_meta", "record.meta is missing", split=split, line_number=line_number, sample_id=sample_id)
        return ""
    target_size = int(variant_spec.get("target_size", 256))
    expected = {
        "coord_mode": "norm1000",
        "coord_range": 1000,
        "pixel_patch_size": target_size,
        "patch_width": target_size,
        "patch_height": target_size,
        "target_size": target_size,
        "context_image_size": variant_spec["context_image_size"],
        "view_mode": variant_spec.get("view_mode", variant),
        "target_roi_in_image": variant_spec["target_roi_in_image"],
    }
    if variant_spec.get("task_mode", "lane_intersection") == INTERSECTION_PROMPT_TASK_MODE:
        expected.update({
            "dataset_variant": variant,
            "task_mode": INTERSECTION_PROMPT_TASK_MODE,
            "oracle_intersection_conditioning": True,
        })
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            errors.add(
                "invalid_meta",
                f"meta.{key}={meta.get(key)!r}, expected {expected_value!r}",
                split=split,
                line_number=line_number,
                sample_id=sample_id,
            )
    return str(meta.get("tile_id", ""))


def update_reservoir(
    reservoirs: dict[str, list[dict[str, Any]]],
    seen: Counter,
    difficulty: str,
    item: dict[str, Any],
    limit: int,
    rng: random.Random,
) -> None:
    if difficulty not in reservoirs or limit <= 0:
        return
    seen[difficulty] += 1
    bucket = reservoirs[difficulty]
    if len(bucket) < limit:
        bucket.append(item)
        return
    replacement = rng.randrange(seen[difficulty])
    if replacement < limit:
        bucket[replacement] = item


def check_build_metadata(dataset_root: Path, errors: ErrorCollector) -> dict[str, Any]:
    candidates = [dataset_root / "dataset_info.json", dataset_root.parent / "build_summary.json"]
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        errors.add("missing_build_metadata", "dataset_info.json/build_summary.json was not found")
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.add("invalid_build_metadata", f"unable to parse {metadata_path}: {exc}")
        return {}
    if metadata.get("semantic_schema_version") != SEMANTIC_SCHEMA_VERSION:
        errors.add(
            "stale_semantic_schema",
            f"semantic_schema_version={metadata.get('semantic_schema_version')!r}, expected {SEMANTIC_SCHEMA_VERSION}",
        )
    if metadata.get("semantic_validation_passed") is not True:
        errors.add("unverified_build", "semantic_validation_passed is not true")
    ignored_codes = metadata.get("ignored_source_lane_type_codes")
    if ignored_codes != sorted(IGNORED_LANE_TYPE_CODES):
        errors.add(
            "stale_lane_type_filter",
            f"ignored_source_lane_type_codes={ignored_codes!r}, expected {sorted(IGNORED_LANE_TYPE_CODES)}",
        )
    balance = metadata.get("balance") if isinstance(metadata.get("balance"), dict) else {}
    compact_balance = {
        "target_total": balance.get("target_total"),
        "selected_total": balance.get("selected_total"),
        "selected_unique": balance.get("selected_unique"),
        "exact_repeated_records": balance.get("exact_repeated_records"),
        "target_ratios": balance.get("target_ratios"),
        "target_quotas": balance.get("target_quotas"),
        "final_bucket_counts": balance.get("final_bucket_counts"),
        "target_intersection_ratio": balance.get("target_intersection_ratio"),
        "actual_intersection_ratio": balance.get("actual_intersection_ratio"),
        "selection_policy": balance.get("selection_policy"),
    }
    for grid_name in ("base_grid", "translation_grid"):
        grid = balance.get(grid_name)
        if isinstance(grid, dict) and isinstance(grid.get("difficulty_plan"), dict):
            compact_balance[grid_name] = {"difficulty_plan": grid["difficulty_plan"]}
    return {
        "path": str(metadata_path),
        "dataset_version": metadata.get("dataset_version"),
        "active_variant": metadata.get("active_variant"),
        "semantic_schema_version": metadata.get("semantic_schema_version"),
        "ignored_source_lane_type_codes": metadata.get("ignored_source_lane_type_codes"),
        "semantic_validation_passed": metadata.get("semantic_validation_passed"),
        "difficulty_rule_version": metadata.get("difficulty_rule_version"),
        "coord_mode": metadata.get("coord_mode"),
        "coord_range": metadata.get("coord_range"),
        "target_patch_size": metadata.get("target_patch_size"),
        "train_stride": metadata.get("train_stride"),
        "balance": compact_balance,
    }


def valid_difficulty_counts(value: Any, expected_total: int) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    counts = {}
    for difficulty in DIFFICULTY_ORDER:
        count = value.get(difficulty, 0 if difficulty == "very_easy" else None)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return None
        counts[difficulty] = count
    if sum(counts.values()) != expected_total:
        return None
    return counts


def resolve_expected_difficulty_counts(
    train_total: int, ratios: dict[str, float], build_metadata: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int], str]:
    requested = allocate_quotas(train_total, ratios)
    balance = build_metadata.get("balance")
    if isinstance(balance, dict):
        final_counts = valid_difficulty_counts(balance.get("final_bucket_counts"), train_total)
        if final_counts is not None:
            return final_counts, requested, "dataset_info.balance.final_bucket_counts"
    return requested, requested, "requested_difficulty_ratios"


def resolve_validation_difficulty_profile(build_metadata: dict[str, Any], target_size: int):
    if build_metadata.get("difficulty_rule_version") != DIFFICULTY_PROFILE_VERSION:
        return None
    return resolve_difficulty_profile(patch_size=target_size)


def decode_image(path: Path, expected_size: tuple[int, int]) -> str | None:
    try:
        with Image.open(path) as image:
            if image.size != expected_size:
                return f"image size={image.size}, expected={expected_size}"
            if image.format != "PNG":
                return f"image format={image.format!r}, expected PNG"
            image.verify()
    except Exception as exc:
        return f"unable to decode PNG: {exc}"
    return None


def render_visualizations(
    reservoirs: dict[str, list[dict[str, Any]]], output_dir: Path, visualize_per_difficulty: int
) -> dict[str, int]:
    viz_args = SimpleNamespace(**vars(DIFFICULTY_ARGS))
    viz_args.coord_mode = "norm1000"
    viz_args.coord_range = 1000.0
    manifest_path = output_dir / "visualization_samples.jsonl"
    rendered_counts = Counter()
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for difficulty in ("easy", "medium", "hard", "very_hard"):
            paths = []
            bucket = sorted(reservoirs[difficulty], key=lambda item: item["line_number"])
            for rank, item in enumerate(bucket[:visualize_per_difficulty]):
                metrics = item["metrics"]
                score = str(metrics.get("difficulty_score", 0)).replace(".", "p")
                output_path = output_dir / "visualizations" / difficulty / (
                    f"{rank:03d}_line-{item['line_number']:07d}_score-{score}_"
                    f"{safe_name(item['sample_id'])}.png"
                )
                draw_overlay(item["record"], metrics, item["image_path"], output_path, viz_args)
                paths.append(output_path)
                rendered_counts[difficulty] += 1
                manifest.write(json.dumps({
                    "difficulty": difficulty,
                    "sample_id": item["sample_id"],
                    "jsonl_line_number": item["line_number"],
                    "source_image": str(item["image_path"]),
                    "visualization": str(output_path),
                    "difficulty_score": metrics.get("difficulty_score"),
                    "tags": metrics.get("tags"),
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
            make_contact_sheet(paths, output_dir / f"contact_sheet_{difficulty}.png")
    return dict(rendered_counts)


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], ErrorCollector]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_root.parent / f"{dataset_root.name}_validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variant, variant_spec = resolve_variant(dataset_root, getattr(args, "variant", "auto"))
    expected_image_size = tuple(variant_spec["image_size"])
    target_size = int(variant_spec["target_size"])
    intersection_prompt_task = variant_spec["task_mode"] == INTERSECTION_PROMPT_TASK_MODE
    errors = ErrorCollector(max_examples=args.max_error_examples)
    build_metadata = check_build_metadata(dataset_root, errors)
    ratios = parse_ratio_spec(args.difficulty_ratios)
    difficulty_profile = resolve_validation_difficulty_profile(build_metadata, target_size)
    rng = random.Random(args.seed)
    reservoirs = {name: [] for name in ("easy", "medium", "hard", "very_hard")}
    reservoir_seen = Counter()
    split_stats: dict[str, dict[str, Any]] = {}
    seen_ids: dict[str, str] = {}
    seen_images: dict[str, str] = {}
    tile_splits: dict[str, str] = {}
    referenced_images: set[str] = set()
    decode_reservoirs: dict[str, list[tuple[str, Path, int]]] = {split: [] for split in SPLITS}
    decode_seen = Counter()

    for split in SPLITS:
        split_path = resolve_split_path(dataset_root, args.phase, split)
        if split_path is None:
            errors.add("missing_split", f"missing {args.phase}/{split}.jsonl", split=split)
            continue
        print(f"[dataset-v2-audit] scan {split}: {split_path}", flush=True)
        stats = {
            "samples": 0,
            "empty_samples": 0,
            "intersection_samples": 0,
            "difficulty_counts": Counter(),
            "semantic_counts": Counter(),
            "referenced_images": 0,
            "decoded_images": 0,
        }
        for line_number, record in iter_jsonl(split_path, split, errors):
            stats["samples"] += 1
            sample_id = str(record.get("id", "")).strip()
            if not sample_id:
                errors.add("missing_id", "record.id is missing", split=split, line_number=line_number)
                sample_id = f"<missing:{split}:{line_number}>"
            previous_split = seen_ids.get(sample_id)
            if previous_split is not None:
                errors.add(
                    "duplicate_id",
                    f"id already appeared in {previous_split}",
                    split=split,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            else:
                seen_ids[sample_id] = split

            tile_id = validate_metadata(
                record,
                errors,
                split,
                line_number,
                sample_id,
                variant,
                variant_spec,
            )
            if tile_id:
                previous_tile_split = tile_splits.get(tile_id)
                if previous_tile_split is not None and previous_tile_split != split:
                    errors.add(
                        "raw_tile_split_leakage",
                        f"tile_id={tile_id!r} appears in {previous_tile_split} and {split}",
                        split=split,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                else:
                    tile_splits[tile_id] = split

            prompt = conversation_value(record, {"human", "user"})
            prompt_intersections = []
            if not isinstance(prompt, str):
                errors.add("missing_prompt", "human prompt is missing", split=split, line_number=line_number, sample_id=sample_id)
            else:
                required_texts = [
                    "lane_type",
                    *sorted(ALLOWED_LANE_TYPES),
                    "intersection_type",
                    "normalized 0-1000",
                    f"{target_size}x{target_size}",
                ]
                input_overlay = build_metadata.get("input_overlay") or {}
                if input_overlay.get("raw_lane_overlay") or input_overlay.get("raw_lane_overlay_source") == "patch_tif/0_lane.tif":
                    required_texts.append("white lane overlay")
                if intersection_prompt_task:
                    required_texts.extend([PROMPT_MARKER, "centerlines only"])
                for required_text in required_texts:
                    if required_text not in prompt:
                        errors.add(
                            "invalid_prompt",
                            f"prompt does not mention {required_text!r}",
                            split=split,
                            line_number=line_number,
                            sample_id=sample_id,
                        )
                if intersection_prompt_task:
                    try:
                        prompt_intersections = extract_prompt_intersections(prompt)
                    except (ValueError, json.JSONDecodeError) as exc:
                        errors.add(
                            "invalid_prompt_intersections",
                            str(exc),
                            split=split,
                            line_number=line_number,
                            sample_id=sample_id,
                        )

            payload, target_error = parse_target(record)
            if target_error:
                errors.add("invalid_assistant_target", target_error, split=split, line_number=line_number, sample_id=sample_id)
                payload = None
            has_intersection = False
            effective_record = record
            if payload is not None:
                lines = payload["lines"]
                semantic_counts, assistant_has_intersection = validate_target_lines(
                    lines, errors, split, line_number, sample_id
                )
                if intersection_prompt_task and assistant_has_intersection:
                    errors.add(
                        "intersection_leaked_to_assistant",
                        "assistant target must contain centerlines only",
                        split=split,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                prompt_semantic_counts = Counter()
                if intersection_prompt_task:
                    prompt_semantic_counts, has_intersection = validate_target_lines(
                        prompt_intersections,
                        errors,
                        split,
                        line_number,
                        sample_id,
                    )
                    effective_lines = list(lines) + list(prompt_intersections)
                    effective_record = record_with_target_lines(record, effective_lines)
                else:
                    has_intersection = assistant_has_intersection
                    effective_lines = lines
                semantic_counts.update(prompt_semantic_counts)
                stats["semantic_counts"].update(semantic_counts)
                if not effective_lines:
                    stats["empty_samples"] += 1
                if has_intersection:
                    stats["intersection_samples"] += 1

            image_path, image_error, normalized_image = safe_image_path(dataset_root, record.get("image"))
            if image_error:
                errors.add("invalid_image_path", image_error, split=split, line_number=line_number, sample_id=sample_id)
            elif image_path is not None:
                expected_prefix = f"images/{split}/"
                if not normalized_image.startswith(expected_prefix) or image_path.suffix.lower() != ".png":
                    errors.add(
                        "unexpected_image_layout",
                        f"image={normalized_image!r}, expected prefix {expected_prefix!r} and .png",
                        split=split,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                previous_image_split = seen_images.get(normalized_image)
                if previous_image_split is not None:
                    errors.add(
                        "duplicate_image_reference",
                        f"image already appeared in {previous_image_split}",
                        split=split,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                else:
                    seen_images[normalized_image] = split
                referenced_images.add(normalized_image)
                stats["referenced_images"] += 1
                if not image_path.is_file():
                    errors.add(
                        "missing_image",
                        f"image not found: {image_path}",
                        split=split,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                elif args.image_decode_mode == "all":
                    image_decode_error = decode_image(image_path, expected_image_size)
                    stats["decoded_images"] += 1
                    if image_decode_error:
                        errors.add("invalid_image", image_decode_error, split=split, line_number=line_number, sample_id=sample_id)
                elif args.image_decode_mode == "sampled" and args.image_decode_samples_per_split > 0:
                    decode_seen[split] += 1
                    bucket = decode_reservoirs[split]
                    item = (sample_id, image_path, line_number)
                    if len(bucket) < args.image_decode_samples_per_split:
                        bucket.append(item)
                    else:
                        replacement = rng.randrange(decode_seen[split])
                        if replacement < args.image_decode_samples_per_split:
                            bucket[replacement] = item

            if split == "train" and payload is not None:
                metrics = sample_metrics(effective_record, (target_size, target_size), DIFFICULTY_ARGS)
                if difficulty_profile is not None:
                    resolution_aware_score(metrics, difficulty_profile)
                    difficulty = classify_metrics(metrics, difficulty_profile)
                else:
                    difficulty = "empty" if not effective_lines else str(metrics["difficulty"])
                stats["difficulty_counts"][difficulty] += 1
                if difficulty in reservoirs and image_path is not None and image_path.is_file():
                    update_reservoir(
                        reservoirs,
                        reservoir_seen,
                        difficulty,
                        {
                            "sample_id": sample_id,
                            "line_number": line_number,
                            "record": effective_record,
                            "metrics": metrics,
                            "image_path": image_path,
                        },
                        args.visualize_per_difficulty,
                        rng,
                    )

            if args.progress_every and stats["samples"] % args.progress_every == 0:
                print(
                    f"[dataset-v2-audit] {split}: {stats['samples']} records, errors={errors.total}",
                    flush=True,
                )
        split_stats[split] = stats

    if args.image_decode_mode == "sampled":
        for split, bucket in decode_reservoirs.items():
            for sample_id, image_path, line_number in bucket:
                image_decode_error = decode_image(image_path, expected_image_size)
                if split in split_stats:
                    split_stats[split]["decoded_images"] += 1
                if image_decode_error:
                    errors.add("invalid_image", image_decode_error, split=split, line_number=line_number, sample_id=sample_id)

    train_stats = split_stats.get("train", {})
    train_total = int(train_stats.get("samples", 0))
    if args.expected_train_samples > 0:
        if train_total != args.expected_train_samples:
            errors.add(
                "train_count_mismatch",
                f"train samples={train_total}, expected={args.expected_train_samples}",
                split="train",
            )
    for split in ("eval", "test"):
        if int(split_stats.get(split, {}).get("samples", 0)) <= 0:
            errors.add("empty_split", f"{split} contains no samples", split=split)

    if not args.skip_distribution_check and train_total > 0:
        expected_quotas, requested_quotas, quota_source = resolve_expected_difficulty_counts(
            train_total, ratios, build_metadata
        )
        actual_quotas = train_stats["difficulty_counts"]
        for difficulty in DIFFICULTY_ORDER:
            delta = abs(int(actual_quotas.get(difficulty, 0)) - expected_quotas[difficulty])
            if delta > args.count_tolerance:
                errors.add(
                    "difficulty_distribution_mismatch",
                    f"{difficulty}={actual_quotas.get(difficulty, 0)}, expected={expected_quotas[difficulty]}",
                    split="train",
                )
    else:
        requested_quotas = {}
        expected_quotas = {}
        quota_source = "distribution_check_skipped"
    if train_total > 0 and args.expected_intersection_ratio >= 0:
        expected_intersections = int(round(train_total * args.expected_intersection_ratio))
        actual_intersections = int(train_stats.get("intersection_samples", 0))
        if abs(actual_intersections - expected_intersections) > args.count_tolerance:
            errors.add(
                "intersection_distribution_mismatch",
                f"intersection samples={actual_intersections}, expected={expected_intersections}",
                split="train",
            )

    if not args.allow_short_visual_buckets:
        for difficulty, bucket in reservoirs.items():
            if len(bucket) < args.visualize_per_difficulty:
                errors.add(
                    "insufficient_visual_samples",
                    f"{difficulty} has only {len(bucket)} visualizable samples, expected {args.visualize_per_difficulty}",
                    split="train",
                )

    actual_images: set[str] = set()
    if not args.skip_extra_image_scan:
        images_root = dataset_root / "images"
        if not images_root.is_dir():
            errors.add("missing_images_root", f"images root not found: {images_root}")
        else:
            print(f"[dataset-v2-audit] scan actual PNG files: {images_root}", flush=True)
            for path in images_root.rglob("*.png"):
                actual_images.add(path.relative_to(dataset_root).as_posix())
            missing_references = referenced_images - actual_images
            extra_images = actual_images - referenced_images
            if missing_references:
                errors.add(
                    "missing_referenced_images",
                    f"{len(missing_references)} referenced PNGs are absent; examples={sorted(missing_references)[:5]}",
                )
            if extra_images:
                errors.add(
                    "unreferenced_extra_images",
                    f"{len(extra_images)} extra PNGs are not referenced; examples={sorted(extra_images)[:5]}",
                )

    rendered_counts = render_visualizations(
        reservoirs, output_dir, args.visualize_per_difficulty
    )
    serializable_splits = {}
    for split, stats in split_stats.items():
        serializable_splits[split] = {
            **stats,
            "difficulty_counts": dict(stats["difficulty_counts"]),
            "semantic_counts": dict(stats["semantic_counts"]),
            "intersection_ratio": (
                stats["intersection_samples"] / stats["samples"] if stats["samples"] else 0.0
            ),
        }
    report = {
        "status": "passed" if errors.total == 0 else "failed",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "variant": variant,
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "difficulty_rule_version": DIFFICULTY_RULE_VERSION,
        "constraints": {
            "expected_train_samples": args.expected_train_samples,
            "difficulty_ratios": ratios,
            "expected_intersection_ratio": args.expected_intersection_ratio,
            "allowed_lane_types": sorted(ALLOWED_LANE_TYPES),
            "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
            "allowed_intersection_types": sorted(ALLOWED_INTERSECTION_TYPES),
            "expected_image_size": list(expected_image_size),
            "image_decode_mode": args.image_decode_mode,
            "image_decode_samples_per_split": args.image_decode_samples_per_split,
        },
        "difficulty_distribution": {
            "quota_source": quota_source,
            "requested_quotas": requested_quotas,
            "expected_final_quotas": expected_quotas,
            "actual_counts": dict(train_stats.get("difficulty_counts", {})),
            "redistribution_from_requested": {
                difficulty: int(expected_quotas.get(difficulty, 0)) - int(requested_quotas.get(difficulty, 0))
                for difficulty in DIFFICULTY_ORDER
            } if requested_quotas else {},
        },
        "build_metadata": build_metadata,
        "splits": serializable_splits,
        "unique_ids": len(seen_ids),
        "unique_tiles": len(tile_splits),
        "referenced_pngs": len(referenced_images),
        "actual_pngs": len(actual_images) if not args.skip_extra_image_scan else None,
        "visualization_counts": rendered_counts,
        "error_total": errors.total,
        "error_counts": dict(errors.counts),
        "error_examples": errors.examples,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "validation_errors.jsonl").open("w", encoding="utf-8") as handle:
        for item in errors.examples:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    return report, errors


def main(argv=None) -> None:
    args = parse_args(argv)
    report, errors = audit(args)
    summary = {
        "status": report["status"],
        "dataset_root": report["dataset_root"],
        "split_samples": {
            split: stats["samples"] for split, stats in report["splits"].items()
        },
        "train_difficulty_counts": report["splits"].get("train", {}).get("difficulty_counts", {}),
        "train_intersection_ratio": report["splits"].get("train", {}).get("intersection_ratio", 0.0),
        "visualization_counts": report["visualization_counts"],
        "error_total": errors.total,
        "error_counts": dict(errors.counts),
        "report": str(Path(report["output_dir"]) / "validation_report.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors.total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
