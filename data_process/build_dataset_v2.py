#!/usr/bin/env python3
"""Build paired Stage-A RC Dataset V2 views from one or more raw sources.

The two views share raw-sample splits, target patch ids, geometry labels, and
balanced train selection.  Only the visible image window changes:

* local256: the original 256x256 target patch.
* context512_roi256: a black-padded 512x512 context crop whose central
  256x256 ROI is the only supervised region.

Geometry extraction intentionally delegates to state_update_dataset_common so
the TIFF mask, GeoJSON clipping, endpoint types, intersection polygons, and
norm1000 conversion stay aligned with the Jiangjihua data pipeline.
"""

import argparse
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
    split_samples,
    validate_rows,
    write_json,
)
from scripts.tools.tag_hard_map_samples import sample_metrics


DEFAULT_DIFFICULTY_RATIOS = {
    "empty": 0.00,
    "easy": 0.30,
    "medium": 0.33,
    "hard": 0.27,
    "very_hard": 0.10,
}
DIFFICULTY_ORDER = tuple(DEFAULT_DIFFICULTY_RATIOS)
DIFFICULTY_ARGS = SimpleNamespace(
    coord_mode=COORD_MODE_NORM1000,
    coord_range=float(DEFAULT_COORD_RANGE),
    junction_tol=36.0,
    intersection_tol=16.0,
    dense_line_threshold=8,
    dense_point_threshold=34,
    long_total_length_threshold=3600.0,
    many_cut_threshold=6,
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


def take_with_optional_replacement(pool, count, rng, allow_oversample):
    if count <= 0 or not pool:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    if count <= len(shuffled):
        return shuffled[:count]
    if not allow_oversample:
        return shuffled
    result = list(shuffled)
    while len(result) < count:
        cycle = list(shuffled)
        rng.shuffle(cycle)
        result.extend(cycle[:count - len(result)])
    return result


def _repeat_cost(quota, intersection_count, intersection_unique, plain_unique):
    plain_count = quota - intersection_count
    return (
        max(0, intersection_count - intersection_unique)
        + max(0, plain_count - plain_unique)
    )


def allocate_global_intersection_quotas(
    pools,
    quotas,
    target_total,
    intersection_target_ratio,
    rng,
    allow_oversample,
):
    """Allocate one global intersection target while preserving difficulty quotas.

    The allocation follows each bucket's natural intersection density and favors
    allocations that do not require repeated records. It intentionally does not
    impose the same intersection percentage on every difficulty bucket.
    """
    requested = int(round(target_total * intersection_target_ratio))
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

        if quota <= 0 or intersection_unique == 0:
            lower[name] = 0
            upper[name] = 0
            continue
        if allow_oversample:
            lower[name] = quota if plain_unique == 0 else 0
            upper[name] = quota
        else:
            upper[name] = min(quota, intersection_unique)
            lower[name] = upper[name] if plain_unique == 0 else 0

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
            quota = quotas[name]
            current = planned[name]
            intersection_unique = len(pools[name]["intersection"])
            plain_unique = len(pools[name]["plain"])
            repeat_delta = (
                _repeat_cost(quota, current + 1, intersection_unique, plain_unique)
                - _repeat_cost(quota, current, intersection_unique, plain_unique)
            )
            weight = max(weights[name], 1e-12)
            weighted_progress = (current + 1) / weight
            return repeat_delta, weighted_progress, tie_breakers[name]

        chosen_name = min(candidates, key=allocation_priority)
        planned[chosen_name] += 1

    return planned, {
        "requested_records": requested,
        "planned_records": sum(planned.values()),
        "minimum_feasible_records": minimum,
        "maximum_feasible_records": maximum,
        "requested_ratio": intersection_target_ratio,
        "natural_ratios_by_bucket": natural_ratios,
        "constraint_scope": "global",
    }


def select_balanced_candidates(pools, target_total, ratios, intersection_target_ratio, seed, allow_oversample):
    quotas = allocate_quotas(target_total, ratios)
    rng = random.Random(seed)
    planned_intersections, intersection_plan = allocate_global_intersection_quotas(
        pools,
        quotas,
        target_total,
        intersection_target_ratio,
        rng,
        allow_oversample,
    )
    selected = []
    bucket_report = {}

    for name in DIFFICULTY_ORDER:
        quota = quotas[name]
        intersection_pool = pools[name]["intersection"]
        plain_pool = pools[name]["plain"]
        desired_intersection = planned_intersections[name]
        desired_plain = quota - desired_intersection

        chosen_intersection = take_with_optional_replacement(
            intersection_pool, desired_intersection, rng, allow_oversample
        )
        chosen_plain = take_with_optional_replacement(plain_pool, desired_plain, rng, allow_oversample)
        chosen = chosen_intersection + chosen_plain

        if len(chosen) < quota:
            chosen_ids = set(chosen)
            leftovers = [item for item in intersection_pool + plain_pool if item not in chosen_ids]
            chosen.extend(take_with_optional_replacement(
                leftovers, quota - len(chosen), rng, allow_oversample
            ))
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
            "oversampled_records": max(0, len(chosen) - len(set(chosen))),
        }

    if len(selected) < target_total and allow_oversample:
        selected_set = set(selected)
        global_pool = []
        for name in DIFFICULTY_ORDER:
            global_pool.extend(pools[name]["intersection"])
            global_pool.extend(pools[name]["plain"])
        leftovers = [item for item in global_pool if item not in selected_set]
        selected.extend(take_with_optional_replacement(
            leftovers or global_pool,
            target_total - len(selected),
            rng,
            allow_oversample,
        ))

    rng.shuffle(selected)
    counts = Counter(selected)
    selected_intersections = 0
    intersection_ids = {
        item
        for name in DIFFICULTY_ORDER
        for item in pools[name]["intersection"]
    }
    for item, count in counts.items():
        if item in intersection_ids:
            selected_intersections += count
    report = {
        "target_total": target_total,
        "selected_total": len(selected),
        "selected_unique": len(counts),
        "oversampled_records": len(selected) - len(counts),
        "target_ratios": ratios,
        "target_quotas": quotas,
        "target_intersection_ratio": intersection_target_ratio,
        "actual_intersection_ratio": selected_intersections / max(1, len(selected)),
        "intersection_constraint_scope": "global",
        "intersection_plan": intersection_plan,
        "allow_oversample": allow_oversample,
        "seed": seed,
        "buckets": bucket_report,
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


def discover_multi_source_samples(input_roots, source_uris, keep_archives, duplicate_policy, limit_samples):
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


def variant_row(row, target_size, context_size, view_mode, repeat_index=0):
    result = dict(row)
    result["meta"] = dict(row.get("meta", {}))
    x0 = int(result["meta"]["x0"])
    y0 = int(result["meta"]["y0"])
    roi = centered_target_roi(target_size, context_size)
    context_x0 = x0 - roi[0]
    context_y0 = y0 - roi[1]
    base_id = str(row["id"])
    result["id"] = base_id if repeat_index == 0 else f"{base_id}__repeat{repeat_index:03d}"
    result["meta"].update({
        "base_sample_id": base_id,
        "oversample_copy_index": repeat_index,
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
            validate_rows(rows, True, args.patch_size)
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
                repeat_count = selected_counts[row["id"]] if selected_counts is not None else 1
                for repeat_index in range(repeat_count):
                    for name, spec in variant_specs.items():
                        rendered_row = variant_row(
                            row,
                            args.patch_size,
                            spec["context_size"],
                            spec["view_mode"],
                            repeat_index=repeat_index,
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
                        write_jsonl_item(writers[name]["sft"], sft)
                        write_jsonl_item(writers[name]["meta"], rendered_row)
                        record_counts[name] += 1
    finally:
        close_variant_writers(writers)
    return {
        "record_counts": dict(record_counts),
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
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--coord-mode", choices=[COORD_MODE_NORM1000], default=COORD_MODE_NORM1000)
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--train-target-samples", type=int, default=550000)
    parser.add_argument(
        "--difficulty-ratios",
        default="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10",
    )
    parser.add_argument("--intersection-target-ratio", type=float, default=0.30)
    parser.add_argument("--no-oversample-short-buckets", action="store_true")
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require_geo_dependencies()
    if args.source_uri and len(args.source_uri) != len(args.input_root):
        raise ValueError("--source-uri must be omitted or repeated exactly once per --input-root")
    if args.patch_size != 256 or args.context_size != 512:
        raise ValueError(
            "The controlled Dataset V2 baseline is fixed to target patch 256 and context image 512."
        )
    if args.patch_size != args.stride:
        raise ValueError("Dataset V2 baseline requires stride == patch_size; offset grids are a later ablation.")
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

    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pools = {
        name: {"intersection": [], "plain": []}
        for name in DIFFICULTY_ORDER
    }
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
                args,
                write_images=False,
                max_empty_ratio=-1.0,
            )
            validate_rows(rows, True, args.patch_size)
            if not rows:
                train_samples_without_rows.append(sample.sample_id)
            for row in rows:
                metrics = classify_row(row, args.patch_size, args.coord_range)
                metrics["source_uri"] = source_by_id.get(sample.sample_id, str(sample.root))
                pool_type = "intersection" if metrics["has_intersection"] else "plain"
                pools[metrics["stratum"]][pool_type].append(row["id"])
                candidate_counts[metrics["stratum"]] += 1
                candidate_writer.write(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")) + "\n")

    selected_counts, balance_report = select_balanced_candidates(
        pools,
        args.train_target_samples,
        ratios,
        args.intersection_target_ratio,
        args.difficulty_seed,
        not args.no_oversample_short_buckets,
    )
    if balance_report["selected_total"] != args.train_target_samples:
        raise ValueError(
            f"unable to select {args.train_target_samples} train records; got {balance_report['selected_total']}. "
            "Keep oversampling enabled or lower --train-target-samples."
        )

    selection_path = manifest_dir / "train_selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as selection_writer:
        for name in DIFFICULTY_ORDER:
            for pool_type in ("intersection", "plain"):
                for patch_id in pools[name][pool_type]:
                    repeat_count = selected_counts.get(patch_id, 0)
                    if repeat_count:
                        write_jsonl_item(selection_writer, {
                            "id": patch_id,
                            "stratum": name,
                            "has_intersection": pool_type == "intersection",
                            "repeat_count": repeat_count,
                        })

    variant_specs = {}
    if args.views in {"both", "local"}:
        variant_specs["local256"] = {
            "root": output_root / "local256",
            "context_size": args.patch_size,
            "view_mode": "local256",
        }
    if args.views in {"both", "context"}:
        variant_specs["context512_roi256"] = {
            "root": output_root / "context512_roi256",
            "context_size": args.context_size,
            "view_mode": "context512_roi256",
        }

    split_results = {
        "train": materialize_split(train_samples, "train", selected_counts, variant_specs, source_by_id, args),
        "eval": materialize_split(eval_samples, "eval", None, variant_specs, source_by_id, args),
        "test": materialize_split(test_samples, "test", None, variant_specs, source_by_id, args),
    }

    split_manifest = {
        "dataset_version": "rc_dataset_v2_stage_a",
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
        "dataset_version": "rc_dataset_v2_stage_a",
        "baseline": "jiangjihua_tiff_geojson_mask_and_clipping",
        "task": "lane_intersection",
        "phase": "phase_a",
        "coord_mode": args.coord_mode,
        "coord_range": args.coord_range,
        "target_patch_size": args.patch_size,
        "stride": args.stride,
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
            "offset_grid_enabled": False,
            "visible_roi_border_enabled": False,
            "only_changed_variable": "visible_context_window",
        },
    }
    write_json(output_root / "build_summary.json", build_summary)
    for name, spec in variant_specs.items():
        write_json(spec["root"] / "split_manifest.json", split_manifest)
        write_json(spec["root"] / "balance_report.json", balance_report)
        variant_info = dict(build_summary)
        variant_info["active_variant"] = name
        write_json(spec["root"] / "dataset_info.json", variant_info)

    print(json.dumps(build_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
