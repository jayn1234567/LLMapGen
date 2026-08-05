#!/usr/bin/env python3
"""Build an exact-ratio three-image Dataset V2 subset from a converted pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.state_update_dataset_common import semantic_sft_record_counts
from data_process.build_dataset_v2_staged import STAGE_MARKER, discover_stage_roots
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar
from scripts.tools.convert_context512_roi_triplet_gt_to_dataset_v2 import materialize_file
from scripts.tools.build_rc_dataset_v2_rawlane_pose_three_image_800k_from_staging_windows import (
    find_clean_source,
    find_pose_image,
    transform_record,
)


FORMAT_VERSION = "context512_roi256_three_image_exact_balance_v1"
SPLITS = ("train", "eval", "test")
DIFFICULTIES = ("empty", "easy", "medium", "hard", "very_hard")
DEFAULT_RATIOS = "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Converted full Dataset V2 pool.")
    parser.add_argument("--output-root", required=True, help="Balanced Dataset V2 output root.")
    parser.add_argument("--train-target-samples", type=int, default=800_000)
    parser.add_argument("--difficulty-ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--empty-donor-clean-staging-root",
        default="",
        help="Optional clean context512 staging used only when the converted pool lacks empty samples.",
    )
    parser.add_argument(
        "--empty-donor-aux-staging-root",
        default="",
        help="Optional Raw-Lane/Pose staging paired by source_index with the clean staging.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--package-path", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ratios(spec: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid difficulty ratio item: {item!r}")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty bucket: {name!r}")
        if name in ratios:
            raise ValueError(f"duplicate difficulty bucket: {name!r}")
        ratios[name] = float(raw_value)
    missing = [name for name in DIFFICULTIES if name not in ratios]
    if missing:
        raise ValueError(f"difficulty ratios are missing buckets: {missing}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("difficulty ratios must be non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("difficulty ratios must sum to a positive value")
    return {name: ratios[name] / total for name in DIFFICULTIES}


def allocate_quotas(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("train target samples must be positive")
    exact = {name: total * ratios[name] for name in DIFFICULTIES}
    quotas = {name: int(exact[name]) for name in DIFFICULTIES}
    remainder = total - sum(quotas.values())
    order = sorted(
        DIFFICULTIES,
        key=lambda name: (exact[name] - quotas[name], -DIFFICULTIES.index(name)),
        reverse=True,
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def record_stratum(record: dict, path: Path, line_number: int) -> str:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{path}:{line_number} has no meta object")
    stratum = str(meta.get("stratum") or "").strip()
    if not stratum:
        raise ValueError(f"{path}:{line_number} has no meta.stratum")
    return stratum


def select_train_indices(
    train_jsonl: Path,
    quotas: dict[str, int],
    seed: int,
    progress_every: int,
) -> tuple[list[int], dict[str, int], int]:
    reservoirs = {name: [] for name in DIFFICULTIES}
    available = Counter()
    randomizers = {
        name: random.Random(seed + 104729 * (index + 1))
        for index, name in enumerate(DIFFICULTIES)
    }
    record_index = 0
    with train_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stratum = record_stratum(record, train_jsonl, line_number)
            available[stratum] += 1
            if stratum in reservoirs:
                seen = available[stratum]
                quota = quotas[stratum]
                bucket = reservoirs[stratum]
                if len(bucket) < quota:
                    bucket.append(record_index)
                elif quota > 0:
                    replacement = randomizers[stratum].randrange(seen)
                    if replacement < quota:
                        bucket[replacement] = record_index
            record_index += 1
            if progress_every > 0 and record_index % progress_every == 0:
                print(
                    f"[three-image-balance] scanned train records={record_index} "
                    f"available={dict(available)}",
                    flush=True,
                )
    selected = sorted(index for bucket in reservoirs.values() for index in bucket)
    return selected, dict(available), record_index


def collect_reserved_ids(input_root: Path, selected_indices: list[int]) -> set[str]:
    train_jsonl = input_root / "phase_a" / "train.jsonl"
    selected_ids: set[str] = set()
    cursor = 0
    record_index = 0
    with train_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            include = (
                cursor < len(selected_indices)
                and selected_indices[cursor] == record_index
            )
            record_index += 1
            if not include:
                continue
            cursor += 1
            record = json.loads(line)
            sample_id = str(record.get("id") or "").strip()
            if not sample_id:
                raise ValueError(f"{train_jsonl}:{line_number} has no id")
            if sample_id in selected_ids:
                raise ValueError(f"duplicate selected sample id in source pool: {sample_id}")
            selected_ids.add(sample_id)
    if cursor != len(selected_indices):
        raise AssertionError(f"resolved {cursor}/{len(selected_indices)} selected source ids")
    for split in ("eval", "test"):
        path = input_root / "phase_a" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"input split JSONL not found: {path}")
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("id") or "").strip()
                if not sample_id:
                    raise ValueError(f"{path}:{line_number} has no id")
                if sample_id in selected_ids:
                    raise ValueError(f"duplicate selected sample id across splits: {sample_id}")
                selected_ids.add(sample_id)
    return selected_ids


def stage_map(staging_root: Path, role: str) -> dict[int, Path]:
    result = {}
    for stage_root in discover_stage_roots(staging_root):
        marker = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        source_index = int(marker["source_index"])
        if source_index in result:
            raise ValueError(f"duplicate {role} source_index={source_index}: {stage_root}")
        result[source_index] = stage_root
    return result


def donor_record_sources(
    clean_stage: Path,
    aux_stage: Path,
    record: dict,
    split: str,
) -> tuple[Path, Path, Path]:
    marker = json.loads((clean_stage / STAGE_MARKER).read_text(encoding="utf-8"))
    if bool(marker.get("raw_lane_overlay", False)):
        raise ValueError(f"empty donor clean staging is itself overlaid: {clean_stage}")
    if "context512_roi256" not in set(marker.get("variants") or []):
        raise ValueError(
            f"empty donor clean staging has no context512_roi256 variant: {clean_stage}"
        )
    clean = find_clean_source(clean_stage, "context512_roi256", str(record["image"]))
    raw_relative = str(record.get("raw_lane_image") or "")
    pose_relative = find_pose_image(record, split)
    if not raw_relative:
        raise ValueError(f"empty donor sample {record.get('id')} has no raw_lane_image")
    aux_variant = aux_stage / "variants" / "context512_roi256"
    raw_lane = aux_variant / Path(raw_relative)
    pose = aux_variant / Path(pose_relative)
    for role, path in (("raw_lane", raw_lane), ("pose", pose)):
        if not path.is_file():
            raise FileNotFoundError(
                f"empty donor sample={record.get('id')} {role} image not found: {path}"
            )
    return clean, raw_lane, pose


def select_empty_donor_records(
    clean_staging_root: Path,
    aux_staging_root: Path,
    required: int,
    excluded_ids: set[str],
    seed: int,
    progress_every: int,
) -> tuple[list[dict], dict]:
    if required <= 0:
        return [], {"available_unique_empty": 0, "selected": 0}
    clean_stages = stage_map(clean_staging_root, "empty-donor-clean")
    aux_stages = stage_map(aux_staging_root, "empty-donor-aux")
    missing_clean = sorted(set(aux_stages) - set(clean_stages))
    rng = random.Random(seed + 900_001)
    reservoir: list[dict] = []
    seen_ids = excluded_ids
    scanned = 0
    eligible = 0
    incomplete = 0
    for source_index in sorted(aux_stages):
        if source_index not in clean_stages:
            continue
        aux_stage = aux_stages[source_index]
        clean_stage = clean_stages[source_index]
        index_path = aux_stage / "records" / "train.index.jsonl"
        record_path = aux_stage / "records" / "context512_roi256" / "train.jsonl"
        if not index_path.is_file() or not record_path.is_file():
            raise FileNotFoundError(
                f"empty donor stage record pair missing: {index_path} / {record_path}"
            )
        with (
            index_path.open("r", encoding="utf-8-sig") as index_handle,
            record_path.open("r", encoding="utf-8-sig") as record_handle,
        ):
            for line_number, pair in enumerate(
                zip_longest(index_handle, record_handle),
                start=1,
            ):
                index_line, record_line = pair
                if index_line is None or record_line is None:
                    raise ValueError(f"empty donor index/SFT length mismatch: {aux_stage}")
                if not index_line.strip() and not record_line.strip():
                    continue
                index_row = json.loads(index_line)
                record = json.loads(record_line)
                if index_row.get("id") != record.get("id"):
                    raise ValueError(
                        f"empty donor index/SFT id mismatch at {aux_stage}:{line_number}"
                    )
                scanned += 1
                if str(index_row.get("stratum") or "") != "empty":
                    continue
                sample_id = str(record.get("id") or "").strip()
                if not sample_id or sample_id in seen_ids:
                    continue
                record = dict(record)
                record["meta"] = dict(record.get("meta") or {})
                record["meta"].update({
                    "difficulty": index_row.get("difficulty"),
                    "difficulty_score": index_row.get("difficulty_score"),
                    "stratum": "empty",
                    "has_intersection": False,
                    "empty_donor_source_index": source_index,
                })
                transformed = transform_record(record, "train")
                try:
                    sources = donor_record_sources(
                        clean_stage,
                        aux_stage,
                        transformed,
                        "train",
                    )
                except FileNotFoundError:
                    incomplete += 1
                    continue
                seen_ids.add(sample_id)
                descriptor = {
                    "record": transformed,
                    "clean_stage": clean_stage,
                    "aux_stage": aux_stage,
                    "sources": sources,
                }
                eligible += 1
                if len(reservoir) < required:
                    reservoir.append(descriptor)
                else:
                    replacement = rng.randrange(eligible)
                    if replacement < required:
                        reservoir[replacement] = descriptor
                if progress_every > 0 and scanned % progress_every == 0:
                    print(
                        f"[three-image-balance] scanned empty donor records={scanned} "
                        f"eligible_unique_empty={eligible}",
                        flush=True,
                    )
    report = {
        "clean_staging_root": str(clean_staging_root),
        "aux_staging_root": str(aux_staging_root),
        "scanned_records": scanned,
        "available_unique_empty": eligible,
        "incomplete_empty_candidates": incomplete,
        "aux_source_indexes_without_clean_stage": missing_clean,
        "selected": len(reservoir),
        "required": required,
    }
    return reservoir, report


def materialize_donor_record_images(
    descriptor: dict,
    output_root: Path,
    copy_mode: str,
    resume: bool,
    counters: Counter,
) -> None:
    record = descriptor["record"]
    sources = descriptor["sources"]
    for source, relative in zip(sources, record["images"]):
        status = materialize_file(
            source,
            output_root / Path(str(relative)),
            copy_mode,
            resume,
        )
        counters[f"empty_donor:{status}"] += 1


def materialize_record_images(
    record: dict,
    input_root: Path,
    output_root: Path,
    copy_mode: str,
    resume: bool,
    counters: Counter,
) -> None:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(f"sample={record.get('id')} must reference exactly three images")
    for relative in images:
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"sample={record.get('id')} has unsafe image path: {relative!r}")
        source = input_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"sample={record.get('id')} image not found: {source}")
        status = materialize_file(
            source,
            output_root / relative_path,
            copy_mode,
            resume,
        )
        counters[status] += 1


def meta_record(record: dict) -> dict:
    images = list(record["images"])
    meta = dict(record.get("meta") or {})
    return {
        "id": record.get("id"),
        "image": images[0],
        "images": images,
        "raw_lane_image": images[1],
        "pose_image": images[2],
        "difficulty": meta.get("difficulty"),
        "difficulty_score": meta.get("difficulty_score"),
        "stratum": meta.get("stratum"),
        "has_intersection": meta.get("has_intersection"),
        "meta": meta,
    }


def write_jsonl_item(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_output(output_root: Path, config: dict, resume: bool) -> None:
    marker = output_root / ".balance_config.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not resume:
            raise FileExistsError(
                f"balanced output already exists: {output_root}; use --resume for the same recipe"
            )
        if not marker.is_file():
            raise ValueError(
                f"balanced output has no recipe marker and cannot be resumed safely: {output_root}"
            )
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"balanced output recipe differs from the requested recipe: {output_root}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(marker, config)


def copy_source_metadata(input_root: Path, output_root: Path) -> dict:
    info_path = input_root / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"input dataset_info.json not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def build_balanced_dataset(args: argparse.Namespace) -> dict:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if input_root == output_root:
        raise ValueError("input and output roots must differ")
    train_jsonl = input_root / "phase_a" / "train.jsonl"
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"input train JSONL not found: {train_jsonl}")
    ratios = parse_ratios(args.difficulty_ratios)
    quotas = allocate_quotas(args.train_target_samples, ratios)
    donor_clean_text = str(args.empty_donor_clean_staging_root or "").strip()
    donor_aux_text = str(args.empty_donor_aux_staging_root or "").strip()
    if bool(donor_clean_text) != bool(donor_aux_text):
        raise ValueError(
            "empty donor staging requires both --empty-donor-clean-staging-root and "
            "--empty-donor-aux-staging-root"
        )
    donor_clean_root = Path(donor_clean_text).expanduser().resolve() if donor_clean_text else None
    donor_aux_root = Path(donor_aux_text).expanduser().resolve() if donor_aux_text else None
    source_manifest = input_root / "split_manifest.json"
    config = {
        "format_version": FORMAT_VERSION,
        "input_root": str(input_root),
        "input_split_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest.is_file() else ""
        ),
        "train_target_samples": args.train_target_samples,
        "difficulty_ratios": ratios,
        "difficulty_quotas": quotas,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "shortage_policy": "error",
        "empty_donor_clean_staging_root": str(donor_clean_root or ""),
        "empty_donor_aux_staging_root": str(donor_aux_root or ""),
    }
    prepare_output(output_root, config, args.resume)

    selected_indices, available, source_train_count = select_train_indices(
        train_jsonl,
        quotas,
        args.seed,
        args.progress_every,
    )
    source_deficits = {
        name: quotas[name] - available.get(name, 0)
        for name in DIFFICULTIES
        if available.get(name, 0) < quotas[name]
    }
    non_empty_deficits = {
        name: count for name, count in source_deficits.items() if name != "empty"
    }
    selected_source_ids = collect_reserved_ids(input_root, selected_indices)
    empty_donor_records = []
    empty_donor_report = None
    empty_deficit = source_deficits.get("empty", 0)
    if empty_deficit > 0 and donor_clean_root is not None and donor_aux_root is not None:
        if not donor_clean_root.is_dir() or not donor_aux_root.is_dir():
            raise FileNotFoundError(
                "empty donor staging root is missing: "
                f"clean={donor_clean_root}, aux={donor_aux_root}"
            )
        empty_donor_records, empty_donor_report = select_empty_donor_records(
            donor_clean_root,
            donor_aux_root,
            empty_deficit,
            selected_source_ids,
            args.seed,
            args.progress_every,
        )
    unresolved_empty = max(0, empty_deficit - len(empty_donor_records))
    deficits = dict(non_empty_deficits)
    if unresolved_empty:
        deficits["empty"] = unresolved_empty
    combined_available = dict(available)
    combined_available["empty"] = available.get("empty", 0) + int(
        (empty_donor_report or {}).get("available_unique_empty", 0)
    )
    preflight = {
        "status": "failed" if deficits else "passed",
        "format_version": FORMAT_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_train_records": source_train_count,
        "target_train_records": args.train_target_samples,
        "difficulty_ratios": ratios,
        "required": quotas,
        "available_in_converted_pool": available,
        "available_with_empty_donor": combined_available,
        "deficits": deficits,
        "unselected_buckets": sorted(set(available) - set(DIFFICULTIES)),
        "shortage_policy": "error",
        "empty_donor": empty_donor_report,
    }
    write_json(output_root / "balance_preflight.json", preflight)
    if deficits:
        raise ValueError(
            "strict difficulty quotas cannot be satisfied; "
            f"deficits={deficits}; see {output_root / 'balance_preflight.json'}"
        )
    if len(selected_indices) + len(empty_donor_records) != args.train_target_samples:
        raise AssertionError(
            f"selected source={len(selected_indices)} donor={len(empty_donor_records)}, "
            f"expected {args.train_target_samples}"
        )

    selected_cursor = 0
    selected_ids: set[str] = set()
    output_counts = Counter()
    output_difficulties = Counter()
    semantic_counts = Counter()
    image_counts = Counter()
    phase_root = output_root / "phase_a"
    phase_root.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        for split in SPLITS:
            record_temp = phase_root / f"{split}.jsonl.partial"
            meta_temp = phase_root / f"meta_{split}.jsonl.partial"
            record_temp.unlink(missing_ok=True)
            meta_temp.unlink(missing_ok=True)
            writers[split] = (
                record_temp.open("w", encoding="utf-8", newline="\n"),
                meta_temp.open("w", encoding="utf-8", newline="\n"),
            )

        for split in SPLITS:
            source_jsonl = input_root / "phase_a" / f"{split}.jsonl"
            if not source_jsonl.is_file():
                raise FileNotFoundError(f"input split JSONL not found: {source_jsonl}")
            source_index = 0
            with source_jsonl.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    include = split != "train"
                    if split == "train":
                        include = (
                            selected_cursor < len(selected_indices)
                            and selected_indices[selected_cursor] == source_index
                        )
                        if include:
                            selected_cursor += 1
                    source_index += 1
                    if not include:
                        continue
                    record = json.loads(line)
                    sample_id = str(record.get("id") or "").strip()
                    if not sample_id:
                        raise ValueError(f"{source_jsonl}:{line_number} has no id")
                    if sample_id in selected_ids:
                        raise ValueError(f"duplicate selected sample id: {sample_id}")
                    selected_ids.add(sample_id)
                    stratum = record_stratum(record, source_jsonl, line_number)
                    materialize_record_images(
                        record,
                        input_root,
                        output_root,
                        args.copy_mode,
                        args.resume,
                        image_counts,
                    )
                    write_jsonl_item(writers[split][0], record)
                    write_jsonl_item(writers[split][1], meta_record(record))
                    output_counts[split] += 1
                    output_difficulties[f"{split}:{stratum}"] += 1
                    semantic_counts.update(
                        semantic_sft_record_counts(record, strict=True, require_prompt=True)
                    )
                    if args.progress_every > 0 and sum(output_counts.values()) % args.progress_every == 0:
                        print(
                            f"[three-image-balance] materialized records={sum(output_counts.values())} "
                            f"images={sum(image_counts.values())}",
                            flush=True,
                        )

        for descriptor in empty_donor_records:
            record = descriptor["record"]
            sample_id = str(record.get("id") or "").strip()
            if not sample_id or sample_id in selected_ids:
                raise ValueError(f"duplicate or empty selected donor sample id: {sample_id!r}")
            selected_ids.add(sample_id)
            materialize_donor_record_images(
                descriptor,
                output_root,
                args.copy_mode,
                args.resume,
                image_counts,
            )
            write_jsonl_item(writers["train"][0], record)
            write_jsonl_item(writers["train"][1], meta_record(record))
            output_counts["train"] += 1
            output_difficulties["train:empty"] += 1
            semantic_counts.update(
                semantic_sft_record_counts(record, strict=True, require_prompt=True)
            )
    finally:
        for pair in writers.values():
            for handle in pair:
                handle.close()

    if selected_cursor != len(selected_indices):
        raise AssertionError(
            f"only wrote {selected_cursor}/{len(selected_indices)} selected train records"
        )
    actual_train = {
        name: output_difficulties[f"train:{name}"]
        for name in DIFFICULTIES
    }
    if actual_train != quotas:
        raise ValueError(f"actual train quotas={actual_train} differ from required={quotas}")
    for split in SPLITS:
        (phase_root / f"{split}.jsonl.partial").replace(phase_root / f"{split}.jsonl")
        (phase_root / f"meta_{split}.jsonl.partial").replace(
            phase_root / f"meta_{split}.jsonl"
        )

    source_info = copy_source_metadata(input_root, output_root)
    counts = {split: output_counts[split] for split in SPLITS}
    balance_report = {
        **preflight,
        "status": "passed",
        "selected": actual_train,
        "selected_total": sum(actual_train.values()),
        "record_counts": counts,
        "image_materialization_counts": dict(image_counts),
    }
    write_json(output_root / "balance_report.json", balance_report)
    dataset_info = {
        **source_info,
        "dataset_version": FORMAT_VERSION,
        "variant": output_root.name,
        "source_pool": str(input_root),
        "record_counts": counts,
        "difficulty_counts": {
            split: {
                name: output_difficulties[f"{split}:{name}"]
                for name in sorted({key.split(":", 1)[1] for key in output_difficulties if key.startswith(f"{split}:")})
            }
            for split in SPLITS
        },
        "balance": {
            "strict": True,
            "shortage_policy": "error",
            "seed": args.seed,
            "target_train_samples": args.train_target_samples,
            "ratios": ratios,
            "quotas": quotas,
        },
        "semantic_counts": dict(semantic_counts),
    }
    write_json(output_root / "dataset_info.json", dataset_info)
    split_files = {split: phase_root / f"{split}.jsonl" for split in SPLITS}
    split_manifest = {
        "format_version": FORMAT_VERSION,
        "split_policy": "preserve_eval_test_and_exact_stratified_train_without_replacement",
        "source_pool": str(input_root),
        "source_split_manifest_sha256": config["input_split_manifest_sha256"],
        "counts": counts,
        "jsonl_sha256": {split: sha256_file(path) for split, path in split_files.items()},
        "balance": dataset_info["balance"],
    }
    write_json(output_root / "split_manifest.json", split_manifest)
    source_semantic = input_root / "semantic_schema_report.json"
    semantic_report = (
        json.loads(source_semantic.read_text(encoding="utf-8"))
        if source_semantic.is_file()
        else {}
    )
    semantic_report.update({
        "status": "passed",
        "semantic_counts": dict(semantic_counts),
        "balanced_subset": True,
    })
    write_json(output_root / "semantic_schema_report.json", semantic_report)
    validation = {
        "status": "passed",
        "dataset_root": str(output_root),
        "record_counts": counts,
        "difficulty_counts": actual_train,
        "required_difficulty_counts": quotas,
        "unique_sample_ids": len(selected_ids),
        "three_images_per_sample": True,
        "strict_quota_match": True,
    }
    write_json(output_root / "conversion_validation.json", validation)
    summary = {
        "status": "passed",
        "format_version": FORMAT_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "record_counts": counts,
        "difficulty_counts": actual_train,
        "semantic_counts": dict(semantic_counts),
        "balance_report": str(output_root / "balance_report.json"),
        "validation": validation,
    }
    write_json(output_root / "build_summary.json", summary)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = build_balanced_dataset(args)
    package_path = ""
    if args.package:
        output_root = Path(summary["output_root"])
        package = (
            Path(args.package_path).expanduser().resolve()
            if args.package_path
            else output_root.parent / f"{output_root.name}.tar"
        )
        create_variant_tar(output_root, package, args.resume)
        package_path = str(package)
    print(json.dumps({
        "status": "passed",
        "summary": summary,
        "package": package_path,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
