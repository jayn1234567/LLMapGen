#!/usr/bin/env python3
"""Remap a fixed evaluation set to records from a newer dataset release.

The reference set supplies image identities and difficulty bucket membership.
The target dataset supplies the complete record, including its current prompt,
semantic labels, and image path. Matching prefers preserved patch identifiers
and falls back to the raw tile plus absolute crop origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DIFFICULTIES = ("easy", "medium", "hard", "very_hard")
SPLITS = ("train", "eval", "test")
RC_PATTERN = re.compile(r"^(?P<tile>.+)_r(?P<row>\d+)_c(?P<col>\d+)$")
XY_PATTERN = re.compile(r"^(?P<tile>.+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)$")
ALIAS_META_FIELDS = (
    "grid_patch_id",
    "stable_patch_id",
    "base_patch_id",
    "patch_id",
    "sample_id",
)
TILE_META_FIELDS = ("tile_id", "log_id", "raw_sample_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, help="Directory containing fixed easy/medium/hard/very_hard JSONL files.")
    parser.add_argument("--target-dataset-root", required=True, help="New dataset root containing phase_a and images.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--target-phase", default="phase_a")
    parser.add_argument("--scan-target-splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--allowed-target-splits",
        nargs="+",
        choices=SPLITS,
        default=["eval", "test"],
        help="Only matched records in these splits are emitted. Keep train excluded for fair evaluation.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--min-output-match-ratio", type=float, default=0.0)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--reference-image-root", default="")
    parser.add_argument("--target-image-root", default="")
    parser.add_argument("--verify-pixels", action="store_true")
    parser.add_argument("--require-pixel-match", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(payload)
    return records


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("id", record.get("sample_id", ""))).strip()


def image_value(record: dict[str, Any]) -> str:
    value = record.get("image", record.get("images", ""))
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def normalized_alias(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else ""


def record_aliases(record: dict[str, Any]) -> set[str]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    aliases = set()
    for value in (record_id(record), image_value(record), *(meta.get(name) for name in ALIAS_META_FIELDS)):
        alias = normalized_alias(value)
        if not alias:
            continue
        aliases.add(alias)
        stem = Path(alias).stem
        if stem:
            aliases.add(stem)
    return aliases


def int_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def patch_size_for(record: dict[str, Any], default: int) -> int:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    for name in ("target_size", "pixel_patch_size", "patch_size", "patch_width"):
        value = int_value(meta.get(name))
        if value and value > 0:
            return value
    return default


def coordinate_keys(record: dict[str, Any], default_patch_size: int) -> set[tuple[str, int, int]]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    patch_size = patch_size_for(record, default_patch_size)
    tiles = {
        str(meta.get(name)).strip()
        for name in TILE_META_FIELDS
        if str(meta.get(name, "")).strip()
    }
    aliases = record_aliases(record)
    keys: set[tuple[str, int, int]] = set()

    x0 = int_value(meta.get("x0"))
    y0 = int_value(meta.get("y0"))
    if x0 is not None and y0 is not None:
        for tile in tiles:
            keys.add((tile, x0, y0))

    row = int_value(meta.get("patch_row", meta.get("row")))
    col = int_value(meta.get("patch_col", meta.get("col")))
    stride = int_value(meta.get("stride")) or patch_size
    if row is not None and col is not None:
        for tile in tiles:
            keys.add((tile, col * stride, row * stride))

    for alias in aliases:
        match = XY_PATTERN.match(alias)
        if match:
            keys.add((match.group("tile"), int(match.group("x")), int(match.group("y"))))
            continue
        match = RC_PATTERN.match(alias)
        if match:
            keys.add(
                (
                    match.group("tile"),
                    int(match.group("col")) * patch_size,
                    int(match.group("row")) * patch_size,
                )
            )
    return keys


def target_split_path(root: Path, phase: str, split: str) -> Path:
    candidates = (root / phase / f"{split}.jsonl", root / f"{split}.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Target {split} JSONL not found; checked: {', '.join(map(str, candidates))}")


def load_reference_items(reference_dir: Path, difficulties: list[str], patch_size: int):
    items = []
    alias_index: dict[str, set[str]] = defaultdict(set)
    coord_index: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    seen_ids = Counter()
    for difficulty in difficulties:
        path = reference_dir / f"{difficulty}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Reference difficulty split not found: {path}")
        for order, record in enumerate(read_jsonl(path)):
            uid = f"{difficulty}:{order}"
            rid = record_id(record)
            seen_ids[rid] += 1
            item = {
                "uid": uid,
                "difficulty": difficulty,
                "order": order,
                "record": record,
                "reference_id": rid,
                "reference_image": image_value(record),
                "aliases": record_aliases(record),
                "coordinates": coordinate_keys(record, patch_size),
            }
            items.append(item)
            for alias in item["aliases"]:
                alias_index[alias].add(uid)
            for key in item["coordinates"]:
                coord_index[key].add(uid)
    duplicate_ids = sorted(key for key, count in seen_ids.items() if key and count > 1)
    return items, alias_index, coord_index, duplicate_ids


def scan_target_candidates(
    target_root: Path,
    phase: str,
    splits: list[str],
    items_by_uid: dict[str, dict[str, Any]],
    alias_index: dict[str, set[str]],
    coord_index: dict[tuple[str, int, int], set[str]],
    patch_size: int,
    progress_every: int,
):
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scan_counts = Counter()
    for split in splits:
        path = target_split_path(target_root, phase, split)
        for source_index, record in enumerate(iter_jsonl(path)):
            scan_counts[split] += 1
            if progress_every and scan_counts[split] % progress_every == 0:
                print(f"[fixed-eval-remap] scanned {scan_counts[split]} target {split} records", flush=True)
            aliases = record_aliases(record)
            coordinates = coordinate_keys(record, patch_size)
            candidate_uids = set()
            for alias in aliases:
                candidate_uids.update(alias_index.get(alias, ()))
            for key in coordinates:
                candidate_uids.update(coord_index.get(key, ()))
            if not candidate_uids:
                continue
            target_id = record_id(record)
            for uid in candidate_uids:
                reference = items_by_uid[uid]
                shared_aliases = sorted(reference["aliases"] & aliases)
                shared_coordinates = sorted(reference["coordinates"] & coordinates)
                if shared_aliases:
                    score = 100
                    method = "preserved_patch_id"
                elif shared_coordinates:
                    score = 90
                    method = "tile_xy"
                else:
                    continue
                candidates[uid].append(
                    {
                        "score": score,
                        "method": method,
                        "target_split": split,
                        "target_id": target_id,
                        "target_image": image_value(record),
                        "shared_aliases": shared_aliases,
                        "shared_coordinates": [list(item) for item in shared_coordinates],
                        "record": record,
                        "source_index": source_index,
                    }
                )
    return candidates, dict(scan_counts)


def choose_candidates(items, candidates, allowed_splits):
    split_preference = {"test": 0, "eval": 1, "train": 2}
    chosen = {}
    mapping_rows = {}
    for item in items:
        uid = item["uid"]
        options = candidates.get(uid, [])
        base = {
            "uid": uid,
            "difficulty": item["difficulty"],
            "reference_order": item["order"],
            "reference_id": item["reference_id"],
            "reference_image": item["reference_image"],
            "candidate_count": len(options),
        }
        if not options:
            mapping_rows[uid] = {**base, "status": "unmatched"}
            continue
        options.sort(
            key=lambda value: (
                -value["score"],
                split_preference.get(value["target_split"], 99),
                value["target_id"],
            )
        )
        best_score = options[0]["score"]
        best = [option for option in options if option["score"] == best_score]
        unique_targets = {(option["target_split"], option["target_id"]) for option in best}
        if len(unique_targets) != 1:
            mapping_rows[uid] = {
                **base,
                "status": "ambiguous",
                "top_score": best_score,
                "top_targets": [list(value) for value in sorted(unique_targets)],
            }
            continue
        selected = best[0]
        status = "matched" if selected["target_split"] in allowed_splits else "disallowed_target_split"
        mapping_rows[uid] = {
            **base,
            "status": status,
            "method": selected["method"],
            "score": selected["score"],
            "target_split": selected["target_split"],
            "target_id": selected["target_id"],
            "target_image": selected["target_image"],
            "shared_aliases": selected["shared_aliases"],
            "shared_coordinates": selected["shared_coordinates"],
        }
        if status == "matched":
            chosen[uid] = selected

    reverse = defaultdict(list)
    for uid, selected in chosen.items():
        reverse[(selected["target_split"], selected["target_id"])].append(uid)
    for target, uids in reverse.items():
        if len(uids) <= 1:
            continue
        for uid in uids:
            chosen.pop(uid, None)
            mapping_rows[uid]["status"] = "duplicate_target"
            mapping_rows[uid]["duplicate_target"] = list(target)
            mapping_rows[uid]["duplicate_reference_uids"] = sorted(uids)
    return chosen, mapping_rows


def build_basename_index(root: Path, wanted: set[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not root.is_dir() or not wanted:
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.name in wanted:
            index[path.name].append(path)
    return index


def resolve_image_path(record: dict[str, Any], root: Path, index: dict[str, list[Path]]) -> Path | None:
    relative = image_value(record)
    if not relative:
        return None
    direct = Path(relative)
    if not direct.is_absolute():
        direct = root / direct
    if direct.is_file():
        return direct
    matches = index.get(Path(relative).name, [])
    return matches[0] if len(matches) == 1 else None


def pixel_digest(path: Path) -> str:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for --verify-pixels") from exc
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{image.width}x{image.height}:RGB\0".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def verify_chosen_pixels(
    items_by_uid,
    chosen,
    mapping_rows,
    reference_root: Path,
    target_root: Path,
    require_match: bool,
):
    reference_names = {Path(item["reference_image"]).name for item in items_by_uid.values() if item["reference_image"]}
    target_names = {Path(value["target_image"]).name for value in chosen.values() if value["target_image"]}
    reference_index = build_basename_index(reference_root, reference_names)
    target_index = build_basename_index(target_root, target_names)
    verification_counts = Counter()
    for uid in list(chosen):
        reference_record = items_by_uid[uid]["record"]
        target_record = chosen[uid]["record"]
        reference_path = resolve_image_path(reference_record, reference_root, reference_index)
        target_path = resolve_image_path(target_record, target_root, target_index)
        row = mapping_rows[uid]
        row["reference_image_resolved"] = str(reference_path) if reference_path else ""
        row["target_image_resolved"] = str(target_path) if target_path else ""
        if reference_path is None or target_path is None:
            result = "missing_image"
        else:
            result = "equal" if pixel_digest(reference_path) == pixel_digest(target_path) else "different"
        row["pixel_verification"] = result
        verification_counts[result] += 1
        if require_match and result != "equal":
            chosen.pop(uid, None)
            row["status"] = "pixel_verification_failed"
    return dict(verification_counts)


def annotate_target_record(item, selected):
    record = dict(selected["record"])
    meta = dict(record.get("meta", {}))
    meta["fixed_eval_reference"] = {
        "reference_id": item["reference_id"],
        "reference_image": item["reference_image"],
        "reference_difficulty": item["difficulty"],
        "target_split": selected["target_split"],
        "match_method": selected["method"],
    }
    record["meta"] = meta
    return record


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if not 0.0 <= args.min_output_match_ratio <= 1.0:
        raise ValueError("--min-output-match-ratio must be in [0, 1]")
    if args.require_pixel_match:
        args.verify_pixels = True
    if args.verify_pixels and not args.reference_image_root:
        raise ValueError("--reference-image-root is required with --verify-pixels")

    reference_dir = Path(args.reference_dir).resolve()
    target_root = Path(args.target_dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    difficulties = list(dict.fromkeys(args.reference_difficulties))
    allowed_splits = set(args.allowed_target_splits)

    items, alias_index, coord_index, duplicate_reference_ids = load_reference_items(
        reference_dir,
        difficulties,
        args.patch_size,
    )
    items_by_uid = {item["uid"]: item for item in items}
    print(
        f"[fixed-eval-remap] references={len(items)} aliases={len(alias_index)} coordinates={len(coord_index)}",
        flush=True,
    )
    candidates, scan_counts = scan_target_candidates(
        target_root,
        args.target_phase,
        list(dict.fromkeys(args.scan_target_splits)),
        items_by_uid,
        alias_index,
        coord_index,
        args.patch_size,
        args.progress_every,
    )
    chosen, mapping_rows = choose_candidates(items, candidates, allowed_splits)

    pixel_counts = {}
    if args.verify_pixels:
        target_image_root = Path(args.target_image_root or args.target_dataset_root).resolve()
        pixel_counts = verify_chosen_pixels(
            items_by_uid,
            chosen,
            mapping_rows,
            Path(args.reference_image_root).resolve(),
            target_image_root,
            args.require_pixel_match,
        )

    selected_by_bucket = {difficulty: [] for difficulty in difficulties}
    for item in items:
        selected = chosen.get(item["uid"])
        if selected is None:
            continue
        selected_by_bucket[item["difficulty"]].append(annotate_target_record(item, selected))

    all_selected = []
    for difficulty in difficulties:
        records = selected_by_bucket[difficulty]
        write_jsonl(output_dir / f"{difficulty}.jsonl", records)
        all_selected.extend(records)
    write_jsonl(output_dir / "all_selected.jsonl", all_selected)
    ordered_mapping = [mapping_rows[item["uid"]] for item in items]
    write_jsonl(output_dir / "mapping.jsonl", ordered_mapping)

    status_counts = Counter(row["status"] for row in ordered_mapping)
    target_split_counts = Counter(
        row.get("target_split")
        for row in ordered_mapping
        if row.get("target_split")
    )
    method_counts = Counter(
        row.get("method")
        for row in ordered_mapping
        if row.get("method")
    )
    reference_counts = Counter(item["difficulty"] for item in items)
    selected_counts = {key: len(value) for key, value in selected_by_bucket.items()}
    selected_ratio = len(all_selected) / max(1, len(items))
    report = {
        "reference_dir": str(reference_dir),
        "target_dataset_root": str(target_root),
        "output_dir": str(output_dir),
        "reference_count": len(items),
        "reference_counts": dict(reference_counts),
        "selected_count": len(all_selected),
        "selected_counts": selected_counts,
        "selected_ratio": selected_ratio,
        "status_counts": dict(status_counts),
        "target_split_counts_before_filter": dict(target_split_counts),
        "match_method_counts": dict(method_counts),
        "scan_target_splits": list(args.scan_target_splits),
        "allowed_target_splits": list(args.allowed_target_splits),
        "target_scan_counts": scan_counts,
        "duplicate_reference_ids": duplicate_reference_ids,
        "pixel_verification_counts": pixel_counts,
        "fair_holdout_only": "train" not in allowed_splits,
        "leakage_warning": (
            "Records mapped to target train are excluded. Allowing train makes the input set larger but leaks training samples."
            if "train" not in allowed_splits
            else "Target train records are allowed; metrics are not a clean holdout evaluation."
        ),
        "all_selected_jsonl": str(output_dir / "all_selected.jsonl"),
        "mapping_jsonl": str(output_dir / "mapping.jsonl"),
    }
    write_json(output_dir / "mapping_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if args.require_all and len(all_selected) != len(items):
        raise SystemExit(f"Not every reference was mapped: selected={len(all_selected)} reference={len(items)}")
    if selected_ratio < args.min_output_match_ratio:
        raise SystemExit(
            f"Selected match ratio {selected_ratio:.6f} is below required {args.min_output_match_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
