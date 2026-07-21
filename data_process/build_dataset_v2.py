#!/usr/bin/env python3
"""Build Stage-A RC Dataset V2 views from one or more raw sources.

All requested views share raw-sample splits, target patch ids, geometry labels,
and balanced train selection. The geometry arguments determine stable names:

* local{patch_size}: the complete target patch, for example local256/local512.
* context{context_size}_roi{patch_size}: a padded context crop whose centered
  target ROI is the only supervised region, for example context512_roi256.

Geometry extraction intentionally delegates to state_update_dataset_common so
the TIFF mask, GeoJSON clipping, endpoint types, intersection polygons, and
norm1000 conversion stay aligned with the Jiangjihua data pipeline.
"""

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.state_update_dataset_common import (
    COORD_MODE_NORM1000,
    DEFAULT_COORD_RANGE,
    IGNORED_LANE_TYPE_CODES,
    SEMANTIC_SCHEMA_VERSION,
    build_sft_record,
    centered_target_roi,
    discover_samples,
    extract_centered_context,
    image_chunk_to_pil,
    pad_image_to_patch_grid,
    process_sample,
    public_line_in_model_coord,
    read_masked_image,
    require_geo_dependencies,
    semantic_sft_record_counts,
    split_samples,
    validate_rows,
    write_json,
)
from scripts.tools.tag_hard_map_samples import DIFFICULTY_RULE_VERSION, sample_metrics


DEFAULT_DIFFICULTY_RATIOS = {
    "empty": 0.00,
    "easy": 0.30,
    "medium": 0.33,
    "hard": 0.27,
    "very_hard": 0.10,
}
DIFFICULTY_ORDER = tuple(DEFAULT_DIFFICULTY_RATIOS)
DIFFICULTY_REDISTRIBUTION_GROUPS = (("medium", "hard"), ("easy",), ("very_hard",))
DIFFICULTY_ARGS = SimpleNamespace(
    coord_mode=COORD_MODE_NORM1000,
    coord_range=float(DEFAULT_COORD_RANGE),
    junction_tol=36.0,
    intersection_tol=16.0,
    dense_line_threshold=8,
    dense_point_threshold=34,
    long_total_length_threshold=3600.0,
    many_cut_threshold=6,
    short_line_threshold=90.0,
    curved_line_turn_threshold=45.0,
    sharp_turn_threshold=60.0,
    easy_max_centerlines=3,
    easy_max_points=16,
    easy_max_total_turn=120.0,
    easy_max_single_turn=60.0,
    hard_score_threshold=2.5,
    very_hard_score_threshold=5.5,
)


def parse_ratio_spec(raw):
    ratios = {}
    for item in str(raw).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid difficulty ratio item: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in DIFFICULTY_ORDER:
            raise ValueError(f"unknown difficulty bucket: {name}")
        ratios[name] = float(value)
    missing = [name for name in DIFFICULTY_ORDER if name not in ratios]
    if missing:
        raise ValueError(f"difficulty ratios are missing buckets: {missing}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("difficulty ratios must be non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("difficulty ratios must have a positive sum")
    return {name: ratios[name] / total for name in DIFFICULTY_ORDER}


def allocate_quotas(total, ratios):
    exact = {name: total * ratios[name] for name in DIFFICULTY_ORDER}
    quotas = {name: int(exact[name]) for name in DIFFICULTY_ORDER}
    remainder = total - sum(quotas.values())
    order = sorted(DIFFICULTY_ORDER, key=lambda name: (exact[name] - quotas[name], -DIFFICULTY_ORDER.index(name)), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def take_without_replacement(pool, count, rng):
    if count <= 0 or not pool:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:count]


def empty_candidate_pools():
    return {
        name: {"intersection": [], "plain": []}
        for name in DIFFICULTY_ORDER
    }


def available_counts(pools):
    return {
        name: len(pools[name]["intersection"]) + len(pools[name]["plain"])
        for name in DIFFICULTY_ORDER
    }


def resolve_unique_difficulty_quotas(pools, target_total, ratios, use_target_ratios=True):
    available = available_counts(pools)
    requested = allocate_quotas(target_total, ratios) if use_target_ratios else {
        name: 0 for name in DIFFICULTY_ORDER
    }
    resolved = {
        name: min(requested[name], available[name])
        for name in DIFFICULTY_ORDER
    }
    shifted_in = {name: 0 for name in DIFFICULTY_ORDER}
    shortage = target_total - sum(resolved.values())

    for group in DIFFICULTY_REDISTRIBUTION_GROUPS:
        while shortage > 0:
            candidates = [name for name in group if resolved[name] < available[name]]
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda name: (
                    (shifted_in[name] + 1) / max(ratios[name], 1e-12),
                    DIFFICULTY_ORDER.index(name),
                ),
            )
            resolved[chosen] += 1
            shifted_in[chosen] += 1
            shortage -= 1

    return resolved, {
        "requested_quotas": requested,
        "resolved_quotas": resolved,
        "available_unique": available,
        "shifted_in": shifted_in,
        "unfilled": shortage,
        "redistribution_groups": [list(group) for group in DIFFICULTY_REDISTRIBUTION_GROUPS],
    }


def allocate_global_intersection_quotas(
    pools,
    quotas,
    requested_intersections,
    rng,
):
    """Allocate one global intersection target while preserving difficulty quotas.

    The allocation follows each bucket's natural intersection density and favors
    allocations that do not require repeated records. It intentionally does not
    impose the same intersection percentage on every difficulty bucket.
    """
    requested = int(requested_intersections)
    lower = {}
    upper = {}
    natural_ratios = {}
    weights = {}

    for name in DIFFICULTY_ORDER:
        quota = quotas[name]
        intersection_unique = len(pools[name]["intersection"])
        plain_unique = len(pools[name]["plain"])
        available = intersection_unique + plain_unique
        natural_ratio = intersection_unique / available if available else 0.0
        natural_ratios[name] = natural_ratio
        weights[name] = quota * natural_ratio

        lower[name] = max(0, quota - plain_unique)
        upper[name] = min(quota, intersection_unique)

    minimum = sum(lower.values())
    maximum = sum(upper.values())
    planned_target = min(max(requested, minimum), maximum)
    planned = dict(lower)
    tie_breakers = {name: rng.random() for name in DIFFICULTY_ORDER}

    while sum(planned.values()) < planned_target:
        candidates = [name for name in DIFFICULTY_ORDER if planned[name] < upper[name]]
        if not candidates:
            break

        def allocation_priority(name):
            current = planned[name]
            weight = max(weights[name], 1e-12)
            weighted_progress = (current + 1) / weight
            return weighted_progress, tie_breakers[name]

        chosen_name = min(candidates, key=allocation_priority)
        planned[chosen_name] += 1

    return planned, {
        "requested_records": requested,
        "planned_records": sum(planned.values()),
        "minimum_feasible_records": minimum,
        "maximum_feasible_records": maximum,
        "natural_ratios_by_bucket": natural_ratios,
        "constraint_scope": "global",
    }


def intersection_feasible_range(pools, quotas):
    minimum = 0
    maximum = 0
    for name in DIFFICULTY_ORDER:
        quota = quotas[name]
        intersection_unique = len(pools[name]["intersection"])
        plain_unique = len(pools[name]["plain"])
        minimum += max(0, quota - plain_unique)
        maximum += min(quota, intersection_unique)
    return minimum, maximum


def select_unique_pool_records(pools, quotas, requested_intersections, rng):
    planned_intersections, intersection_plan = allocate_global_intersection_quotas(
        pools,
        quotas,
        requested_intersections,
        rng,
    )
    selected = []
    bucket_report = {}

    for name in DIFFICULTY_ORDER:
        quota = quotas[name]
        intersection_pool = pools[name]["intersection"]
        plain_pool = pools[name]["plain"]
        desired_intersection = planned_intersections[name]
        desired_plain = quota - desired_intersection

        chosen_intersection = take_without_replacement(intersection_pool, desired_intersection, rng)
        chosen_plain = take_without_replacement(plain_pool, desired_plain, rng)
        chosen = chosen_intersection + chosen_plain
        if len(chosen) != quota:
            raise ValueError(
                f"unique candidate selection failed for {name}: selected={len(chosen)}, quota={quota}"
            )
        rng.shuffle(chosen)
        selected.extend(chosen)
        intersection_id_set = set(intersection_pool)
        bucket_report[name] = {
            "quota": quota,
            "available_unique": len(intersection_pool) + len(plain_pool),
            "available_intersection_unique": len(intersection_pool),
            "available_plain_unique": len(plain_pool),
            "available_intersection_ratio": (
                len(intersection_pool) / max(1, len(intersection_pool) + len(plain_pool))
            ),
            "planned_intersection": desired_intersection,
            "planned_intersection_ratio": desired_intersection / max(1, quota),
            "selected": len(chosen),
            "selected_intersection": sum(item in intersection_id_set for item in chosen),
        }
    return selected, intersection_plan, bucket_report


def select_balanced_candidates(
    base_pools,
    translated_pools,
    target_total,
    ratios,
    intersection_target_ratio,
    seed,
):
    rng = random.Random(seed)
    base_quotas, base_difficulty_plan = resolve_unique_difficulty_quotas(
        base_pools,
        target_total,
        ratios,
        use_target_ratios=True,
    )
    base_total = sum(base_quotas.values())
    translated_needed = target_total - base_total
    translated_quotas, translated_difficulty_plan = resolve_unique_difficulty_quotas(
        translated_pools,
        translated_needed,
        ratios,
        use_target_ratios=False,
    )
    final_intersection_target = int(round(target_total * intersection_target_ratio))
    base_min, base_max = intersection_feasible_range(base_pools, base_quotas)
    translated_min, translated_max = intersection_feasible_range(translated_pools, translated_quotas)
    preferred_base_target = int(round(base_total * intersection_target_ratio))
    compatible_base_min = max(base_min, final_intersection_target - translated_max)
    compatible_base_max = min(base_max, final_intersection_target - translated_min)
    if compatible_base_min <= compatible_base_max:
        base_intersection_target = min(
            max(preferred_base_target, compatible_base_min),
            compatible_base_max,
        )
    else:
        base_intersection_target = preferred_base_target

    base_selected, base_intersection_plan, base_bucket_report = select_unique_pool_records(
        base_pools,
        base_quotas,
        base_intersection_target,
        rng,
    )
    base_intersection_ids = {
        item
        for name in DIFFICULTY_ORDER
        for item in base_pools[name]["intersection"]
    }
    base_intersections = sum(item in base_intersection_ids for item in base_selected)
    translated_intersection_target = final_intersection_target - base_intersections
    translated_selected, translated_intersection_plan, translated_bucket_report = select_unique_pool_records(
        translated_pools,
        translated_quotas,
        translated_intersection_target,
        rng,
    )

    selected = base_selected + translated_selected
    rng.shuffle(selected)
    counts = Counter(selected)
    if len(counts) != len(selected):
        raise ValueError("duplicate patch ids detected across base and translated grids")
    intersection_ids = base_intersection_ids | {
        item
        for name in DIFFICULTY_ORDER
        for item in translated_pools[name]["intersection"]
    }
    selected_intersections = sum(item in intersection_ids for item in selected)
    final_bucket_counts = Counter()
    for name in DIFFICULTY_ORDER:
        selected_in_bucket = set(base_pools[name]["intersection"] + base_pools[name]["plain"])
        selected_in_bucket.update(translated_pools[name]["intersection"])
        selected_in_bucket.update(translated_pools[name]["plain"])
        final_bucket_counts[name] = sum(item in selected_in_bucket for item in selected)

    report = {
        "target_total": target_total,
        "selected_total": len(selected),
        "selected_unique": len(counts),
        "exact_repeated_records": 0,
        "base_grid_records": len(base_selected),
        "translated_grid_records": len(translated_selected),
        "target_ratios": ratios,
        "target_quotas": allocate_quotas(target_total, ratios),
        "final_bucket_counts": dict(final_bucket_counts),
        "target_intersection_ratio": intersection_target_ratio,
        "actual_intersection_ratio": selected_intersections / max(1, len(selected)),
        "intersection_constraint_scope": "global",
        "combined_intersection_feasible_records": {
            "minimum": base_min + translated_min,
            "maximum": base_max + translated_max,
            "target": final_intersection_target,
        },
        "base_grid": {
            "difficulty_plan": base_difficulty_plan,
            "intersection_plan": base_intersection_plan,
            "buckets": base_bucket_report,
        },
        "translation_grid": {
            "difficulty_plan": translated_difficulty_plan,
            "intersection_plan": translated_intersection_plan,
            "buckets": translated_bucket_report,
        },
        "selection_policy": "unique_base_then_medium_hard_then_translated_grid",
        "seed": seed,
    }
    return counts, report


def classify_row(row, patch_size, coord_range):
    model_lines = [
        public_line_in_model_coord(line, patch_size, COORD_MODE_NORM1000, coord_range)
        for line in row.get("target_lines", [])
    ]
    record = {
        "id": row["id"],
        "image": row["image"],
        "conversations": [{"from": "gpt", "value": {"lines": model_lines}}],
    }
    metrics = sample_metrics(record, (patch_size, patch_size), DIFFICULTY_ARGS)
    metrics.pop("_payload", None)
    is_empty = not model_lines
    stratum = "empty" if is_empty else metrics["difficulty"]
    has_intersection = any(line.get("category") == "intersection" for line in model_lines)
    metrics.update({
        "stratum": stratum,
        "is_empty": is_empty,
        "has_intersection": has_intersection,
        "tile_id": row.get("tile_id"),
    })
    return metrics


def discover_multi_source_samples(
    input_roots,
    source_uris,
    keep_archives,
    duplicate_policy,
    limit_samples,
    archive_workers,
    selective_archive_extract,
):
    chosen = {}
    chosen_source = {}
    collisions = []
    source_reports = []
    for index, root_text in enumerate(input_roots):
        root = Path(root_text)
        samples = discover_samples(
            root,
            include_intersections=True,
            delete_archives=not keep_archives,
            limit_samples=None,
            require_intersection_features=False,
            archive_workers=archive_workers,
            selective_archive_extract=selective_archive_extract,
        )
        source_uri = source_uris[index] if index < len(source_uris) else str(root)
        if not samples:
            raise FileNotFoundError(
                f"source produced no valid TIFF/GeoJSON raw samples: source={source_uri}, local_root={root}"
            )
        source_reports.append({
            "index": index,
            "source_uri": source_uri,
            "local_root": str(root),
            "discovered_samples": len(samples),
        })
        for sample in samples:
            if sample.sample_id in chosen:
                collision = {
                    "sample_id": sample.sample_id,
                    "previous_root": str(chosen[sample.sample_id].root),
                    "new_root": str(sample.root),
                    "previous_source": chosen_source[sample.sample_id],
                    "new_source": source_uri,
                }
                collisions.append(collision)
                if duplicate_policy == "error":
                    raise ValueError(f"duplicate raw sample id: {sample.sample_id}")
                if duplicate_policy == "first":
                    continue
            chosen[sample.sample_id] = sample
            chosen_source[sample.sample_id] = source_uri
    samples = sorted(chosen.values(), key=lambda item: item.sample_id)
    if limit_samples is not None:
        samples = samples[:limit_samples]
    return samples, chosen_source, collisions, source_reports


def annotate_translation_grid(row, patch_size):
    row = dict(row)
    row["meta"] = dict(row.get("meta", {}))
    x0 = int(row["meta"]["x0"])
    y0 = int(row["meta"]["y0"])
    offset = [
        x0 % patch_size,
        y0 % patch_size,
    ]
    grid_patch_id = str(row["id"])
    tile_id = str(row.get("tile_id") or row["meta"].get("tile_id") or "patch")
    stable_patch_id = f"{tile_id}_x{x0:05d}_y{y0:05d}"
    row["id"] = stable_patch_id
    row["image"] = Path(row["image"]).with_name(f"{stable_patch_id}.png").as_posix()
    row["meta"]["grid_patch_id"] = grid_patch_id
    row["meta"]["stable_patch_id"] = stable_patch_id
    row["meta"]["translation_offset"] = offset
    row["meta"]["grid_kind"] = "base" if offset == [0, 0] else "translated"
    return row


def variant_row(row, target_size, context_size, view_mode):
    result = dict(row)
    result["meta"] = dict(row.get("meta", {}))
    x0 = int(result["meta"]["x0"])
    y0 = int(result["meta"]["y0"])
    roi = centered_target_roi(target_size, context_size)
    context_x0 = x0 - roi[0]
    context_y0 = y0 - roi[1]
    base_id = str(row["id"])
    result["meta"].update({
        "base_patch_id": base_id,
        "exact_repeat": False,
        "view_mode": view_mode,
        "target_size": target_size,
        "context_image_size": context_size,
        "target_roi_in_image": roi,
        "target_box_full": [x0, y0, x0 + target_size, y0 + target_size],
        "context_box_full": [
            context_x0,
            context_y0,
            context_x0 + context_size,
            context_y0 + context_size,
        ],
        "context_padding_mode": "constant_black",
    })
    return result


def dataset_variant_specs(output_root: Path, views: str, patch_size: int, context_size: int) -> dict:
    """Return stable variant names and image geometry for a Dataset V2 build."""
    specs = {}
    if views in {"both", "local"}:
        name = f"local{patch_size}"
        specs[name] = {
            "root": output_root / name,
            "context_size": patch_size,
            "view_mode": name,
        }
    if views in {"both", "context"}:
        name = f"context{context_size}_roi{patch_size}"
        specs[name] = {
            "root": output_root / name,
            "context_size": context_size,
            "view_mode": name,
        }
    return specs


def open_variant_writers(variant_roots, split_name):
    writers = {}
    for name, root in variant_roots.items():
        phase_dir = root / "phase_a"
        phase_dir.mkdir(parents=True, exist_ok=True)
        writers[name] = {
            "sft": (phase_dir / f"{split_name}.jsonl").open("w", encoding="utf-8"),
            "meta": (phase_dir / f"meta_{split_name}.jsonl").open("w", encoding="utf-8"),
        }
    return writers


def close_variant_writers(writers):
    for handles in writers.values():
        for handle in handles.values():
            handle.close()


def write_jsonl_item(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_images_for_rows(sample, rows, variant_specs, patch_size, png_compress_level, skip_existing):
    if not rows:
        return
    image_arr, _, _, _ = read_masked_image(sample.image_tiff, sample.mask_tiff)
    image_arr, _ = pad_image_to_patch_grid(image_arr, patch_size)
    for row in rows:
        x0 = int(row["meta"]["x0"])
        y0 = int(row["meta"]["y0"])
        for spec in variant_specs.values():
            output_path = spec["root"] / row["image"]
            if skip_existing and output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            chunk = extract_centered_context(
                image_arr,
                x0,
                y0,
                patch_size,
                spec["context_size"],
            )
            image_chunk_to_pil(chunk).save(output_path, compress_level=png_compress_level)


def materialize_split(samples, split_name, selected_counts, variant_specs, source_by_id, args):
    writers = open_variant_writers({name: spec["root"] for name, spec in variant_specs.items()}, split_name)
    record_counts = Counter()
    semantic_counts = Counter()
    unique_image_count = 0
    dropped_samples = []
    try:
        for sample in tqdm(samples, desc=f"materialize {split_name}", unit="sample"):
            rows = process_sample(
                sample,
                Path(args.output_root),
                split_name,
                True,
                args,
                write_images=False,
                max_empty_ratio=-1.0,
            )
            validate_rows(rows, True, args.patch_size, require_semantic_types=True)
            rows = [annotate_translation_grid(row, args.patch_size) for row in rows]
            for row in rows:
                row["meta"] = dict(row.get("meta", {}))
                row["meta"]["source_uri"] = source_by_id.get(sample.sample_id, str(sample.root))
            if selected_counts is not None:
                rows = [row for row in rows if selected_counts.get(row["id"], 0) > 0]
            if not rows:
                dropped_samples.append(sample.sample_id)
                continue
            write_images_for_rows(
                sample,
                rows,
                variant_specs,
                args.patch_size,
                args.png_compress_level,
                args.skip_existing_images,
            )
            unique_image_count += len(rows)
            for row in rows:
                if selected_counts is not None and selected_counts[row["id"]] != 1:
                    raise ValueError(f"selected patch must appear exactly once: {row['id']}")
                for name, spec in variant_specs.items():
                    rendered_row = variant_row(
                        row,
                        args.patch_size,
                        spec["context_size"],
                        spec["view_mode"],
                    )
                    sft = build_sft_record(
                        rendered_row,
                        args.patch_size,
                        True,
                        "a",
                        coord_mode=args.coord_mode,
                        coord_range=args.coord_range,
                        context_size=spec["context_size"],
                        view_mode=spec["view_mode"],
                    )
                    record_semantic_counts = semantic_sft_record_counts(
                        sft, strict=True, require_prompt=True
                    )
                    semantic_counts.update(
                        {f"{name}:{key}": value for key, value in record_semantic_counts.items()}
                    )
                    write_jsonl_item(writers[name]["sft"], sft)
                    write_jsonl_item(writers[name]["meta"], rendered_row)
                    record_counts[name] += 1
    finally:
        close_variant_writers(writers)
    return {
        "record_counts": dict(record_counts),
        "semantic_target_counts": dict(semantic_counts),
        "unique_image_count_per_variant": unique_image_count,
        "raw_samples_without_selected_rows": dropped_samples,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, help="Repeat once per downloaded raw source root.")
    parser.add_argument("--source-uri", action="append", default=[], help="Optional source URI paired with each input root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--views", choices=["both", "local", "context"], default="both")
    parser.add_argument("--patch-size", type=int, default=256, help="Supervised target patch size.")
    parser.add_argument("--context-size", type=int, default=512, help="Context-view image size.")
    parser.add_argument("--stride", type=int, default=256, help="Eval/test and base-grid stride.")
    parser.add_argument(
        "--train-stride",
        type=int,
        default=128,
        help="Train candidate stride. 128 adds half-patch translation grids without synthetic padding.",
    )
    parser.add_argument("--coord-mode", choices=[COORD_MODE_NORM1000], default=COORD_MODE_NORM1000)
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--train-target-samples", type=int, default=550000)
    parser.add_argument(
        "--difficulty-ratios",
        default="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10",
    )
    parser.add_argument("--intersection-target-ratio", type=float, default=0.30)
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--eval-count", type=int, default=-1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--duplicate-policy", choices=["last", "first", "error"], default="last")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--max-patches-per-sample", type=int, default=None)
    parser.add_argument("--boundary-tol", type=float, default=1.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--line-sample-distance-px", type=float, default=0.0)
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument("--intersection-hint-points", type=int, default=3)
    parser.add_argument("--max-traces-per-side", type=int, default=8)
    parser.add_argument("--max-intersections-per-side", type=int, default=8)
    parser.add_argument("--png-compress-level", type=int, choices=range(0, 10), default=4)
    parser.add_argument("--skip-existing-images", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--archive-workers",
        type=int,
        default=16,
        help="Number of independent .tar.gz archives to extract concurrently.",
    )
    parser.add_argument(
        "--selective-archive-extract",
        action="store_true",
        help=(
            "Extract only 0_inter.tif, 0_edit_poly.tif, and label_check_crop/*.geojson "
            "from each .tar.gz before deleting the verified source archive."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require_geo_dependencies()
    if args.source_uri and len(args.source_uri) != len(args.input_root):
        raise ValueError("--source-uri must be omitted or repeated exactly once per --input-root")
    if args.patch_size <= 0 or args.context_size < args.patch_size:
        raise ValueError("--patch-size must be positive and --context-size must be >= --patch-size")
    if args.patch_size != args.stride:
        raise ValueError("--stride must equal --patch-size so eval/test retain the base grid")
    if not 0 < args.train_stride <= args.patch_size or args.patch_size % args.train_stride:
        raise ValueError("--train-stride must be a positive divisor of --patch-size")
    centered_target_roi(args.patch_size, args.context_size)
    if args.coord_range != DEFAULT_COORD_RANGE:
        raise ValueError(
            f"Dataset V2 difficulty thresholds require --coord-range {DEFAULT_COORD_RANGE}; "
            f"got {args.coord_range}."
        )
    if not 0 <= args.intersection_target_ratio <= 1:
        raise ValueError("--intersection-target-ratio must be in [0, 1]")
    if args.train_target_samples <= 0:
        raise ValueError("--train-target-samples must be positive")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ratios = parse_ratio_spec(args.difficulty_ratios)
    samples, source_by_id, collisions, source_reports = discover_multi_source_samples(
        args.input_root,
        args.source_uri,
        args.keep_archives,
        args.duplicate_policy,
        args.limit_samples,
        args.archive_workers,
        args.selective_archive_extract,
    )
    if not samples:
        raise FileNotFoundError("no valid raw samples found across the supplied input roots")
    train_samples, eval_samples, test_samples = split_samples(
        samples,
        args.train_ratio,
        args.eval_ratio,
        args.eval_count,
        args.split_seed,
    )
    train_process_args = copy.copy(args)
    train_process_args.stride = args.train_stride

    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    base_pools = empty_candidate_pools()
    translated_pools = empty_candidate_pools()
    candidate_counts = Counter()
    candidate_path = manifest_dir / "train_candidates.jsonl"
    train_samples_without_rows = []
    with candidate_path.open("w", encoding="utf-8") as candidate_writer:
        for sample in tqdm(train_samples, desc="classify train candidates", unit="sample"):
            rows = process_sample(
                sample,
                output_root,
                "train",
                True,
                train_process_args,
                write_images=False,
                max_empty_ratio=-1.0,
            )
            validate_rows(rows, True, args.patch_size, require_semantic_types=True)
            if not rows:
                train_samples_without_rows.append(sample.sample_id)
            for row in rows:
                row = annotate_translation_grid(row, args.patch_size)
                metrics = classify_row(row, args.patch_size, args.coord_range)
                metrics["source_uri"] = source_by_id.get(sample.sample_id, str(sample.root))
                metrics["translation_offset"] = row["meta"]["translation_offset"]
                metrics["grid_kind"] = row["meta"]["grid_kind"]
                pool_type = "intersection" if metrics["has_intersection"] else "plain"
                target_pools = base_pools if metrics["grid_kind"] == "base" else translated_pools
                target_pools[metrics["stratum"]][pool_type].append(row["id"])
                candidate_counts[metrics["stratum"]] += 1
                candidate_writer.write(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")) + "\n")

    selected_counts, balance_report = select_balanced_candidates(
        base_pools,
        translated_pools,
        args.train_target_samples,
        ratios,
        args.intersection_target_ratio,
        args.difficulty_seed,
    )
    balance_report["difficulty_rule_version"] = DIFFICULTY_RULE_VERSION
    balance_report["cut_affects_difficulty"] = False
    if balance_report["selected_total"] != args.train_target_samples:
        raise ValueError(
            f"unable to select {args.train_target_samples} train records; got {balance_report['selected_total']}. "
            "Lower --train-stride to add more translated crop windows or lower --train-target-samples."
        )
    if abs(balance_report["actual_intersection_ratio"] - args.intersection_target_ratio) > 1e-8:
        raise ValueError(
            "unable to satisfy the global intersection target with unique crop windows: "
            f"target={args.intersection_target_ratio}, actual={balance_report['actual_intersection_ratio']}"
        )

    selection_path = manifest_dir / "train_selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as selection_writer:
        for name in DIFFICULTY_ORDER:
            for grid_kind, pools in (("base", base_pools), ("translated", translated_pools)):
                for pool_type in ("intersection", "plain"):
                    for patch_id in pools[name][pool_type]:
                        if selected_counts.get(patch_id, 0):
                            write_jsonl_item(selection_writer, {
                                "id": patch_id,
                                "stratum": name,
                                "has_intersection": pool_type == "intersection",
                                "grid_kind": grid_kind,
                                "exact_repeat": False,
                            })

    variant_specs = dataset_variant_specs(
        output_root,
        args.views,
        args.patch_size,
        args.context_size,
    )

    split_results = {
        "train": materialize_split(
            train_samples,
            "train",
            selected_counts,
            variant_specs,
            source_by_id,
            train_process_args,
        ),
        "eval": materialize_split(eval_samples, "eval", None, variant_specs, source_by_id, args),
        "test": materialize_split(test_samples, "test", None, variant_specs, source_by_id, args),
    }

    split_manifest = {
        "dataset_version": "rc_dataset_v2_stage_a_semantic_v1",
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
        "semantic_validation_passed": True,
        "split_unit": "raw_sample_folder",
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "eval_count": args.eval_count,
        "duplicate_policy": args.duplicate_policy,
        "num_sources": len(args.input_root),
        "num_discovered_unique_raw_samples": len(samples),
        "duplicate_sample_count": len(collisions),
        "duplicate_samples": collisions,
        "sources": source_reports,
        "train_ids": [sample.sample_id for sample in train_samples],
        "eval_ids": [sample.sample_id for sample in eval_samples],
        "test_ids": [sample.sample_id for sample in test_samples],
        "train_samples_without_rows": train_samples_without_rows,
    }
    write_json(output_root / "split_manifest.json", split_manifest)
    write_json(manifest_dir / "balance_report.json", balance_report)

    build_summary = {
        "dataset_version": "rc_dataset_v2_stage_a_semantic_v1",
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
        "semantic_validation_passed": True,
        "difficulty_rule_version": DIFFICULTY_RULE_VERSION,
        "difficulty_rule": {
            "cut_affects_difficulty": False,
            "strict_easy_required": True,
            "easy_max_centerlines": DIFFICULTY_ARGS.easy_max_centerlines,
            "easy_max_points": DIFFICULTY_ARGS.easy_max_points,
            "easy_max_total_turn": DIFFICULTY_ARGS.easy_max_total_turn,
            "easy_max_single_turn": DIFFICULTY_ARGS.easy_max_single_turn,
            "hard_score_threshold": DIFFICULTY_ARGS.hard_score_threshold,
            "very_hard_score_threshold": DIFFICULTY_ARGS.very_hard_score_threshold,
        },
        "baseline": "jiangjihua_tiff_geojson_mask_and_clipping",
        "task": "lane_intersection",
        "phase": "phase_a",
        "coord_mode": args.coord_mode,
        "coord_range": args.coord_range,
        "target_patch_size": args.patch_size,
        "stride": args.stride,
        "train_stride": args.train_stride,
        "candidate_train_counts": dict(candidate_counts),
        "balance": balance_report,
        "splits": split_results,
        "variants": {
            name: {
                "root": str(spec["root"]),
                "image_size_on_disk": spec["context_size"],
                "target_roi_in_image": centered_target_roi(args.patch_size, spec["context_size"]),
                "target_coordinate_frame": f"norm0_{args.coord_range}_relative_to_center_{args.patch_size}",
                "processor_note": "DINOv2 training may resize the complete input view to 518x518.",
            }
            for name, spec in variant_specs.items()
        },
        "controlled_ablation": {
            "same_raw_split": True,
            "same_patch_ids": True,
            "same_labels": True,
            "same_train_selection": True,
            "rotation_enabled": False,
            "offset_grid_enabled": args.train_stride < args.patch_size,
            "translation_offsets": [
                [x_offset, y_offset]
                for y_offset in range(0, args.patch_size, args.train_stride)
                for x_offset in range(0, args.patch_size, args.train_stride)
            ],
            "visible_roi_border_enabled": False,
            "exact_repeat_enabled": False,
            "only_ab_difference": "visible_context_window",
        },
    }
    write_json(output_root / "build_summary.json", build_summary)
    write_json(
        output_root / "semantic_schema_report.json",
        {
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "validation_passed": True,
            "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
            "allowed_lane_types": ["common", "right_turn", "other"],
            "allowed_intersection_types": [
                "common",
                "t_intersection",
                "small_untyped",
                "t_lane_change_area",
                "other",
            ],
            "split_target_counts": {
                split: result["semantic_target_counts"]
                for split, result in split_results.items()
            },
        },
    )
    for name, spec in variant_specs.items():
        write_json(spec["root"] / "split_manifest.json", split_manifest)
        write_json(spec["root"] / "balance_report.json", balance_report)
        variant_info = dict(build_summary)
        variant_info["active_variant"] = name
        write_json(spec["root"] / "dataset_info.json", variant_info)

    print(json.dumps(build_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
