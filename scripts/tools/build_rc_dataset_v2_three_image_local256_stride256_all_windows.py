#!/usr/bin/env python3
"""Build a three-image local256 Dataset V2 from completed staging shards.

This builder is intentionally different from the older balanced 800k wrapper:

* it reuses completed staging only; no OBS download or TIFF extraction;
* train rows are restricted to the 256-pixel base grid;
* every non-empty train row is kept;
* empty train rows are capped at ``--empty-ratio`` of the final train set;
* the model receives three separate PNGs: clean BEV, RawLane, and Pose;
* the final dataset is packaged as a tar archive by default.

The staging contract is the one emitted by ``build_dataset_v2_staged.py``:
``records/{split}.index.jsonl``, ``records/local256/{split}.jsonl`` and
``variants/local256/...``.  The auxiliary staging must additionally expose
``raw_lane_image`` and ``pose_image`` fields (or matching paths in ``images``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path, PurePosixPath

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("Pillow is required: install it in the dataset environment") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANT = "local256"
STAGE_MARKER = "stage_complete.json"
SPLITS = ("train", "eval", "test")
IMAGE_ROLES = ("bev_road_structure", "pv_camera_raw_lane", "historical_vehicle_trajectory")
PROMPT_CONTRACT_VERSION = "three_image_local256_stride256_v1"
DEFAULT_CLEAN_STAGING = r"D:\data\fulldata\staging"
DEFAULT_AUX_STAGING = r"D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context"
DEFAULT_WORK_ROOT = r"D:\data\rc_dataset_v2_three_image_local256_stride256_all"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-staging-root", default=DEFAULT_CLEAN_STAGING)
    parser.add_argument("--aux-staging-root", default=DEFAULT_AUX_STAGING)
    parser.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    parser.add_argument("--empty-ratio", type=float, default=0.05)
    parser.add_argument("--selection-seed", type=int, default=20260824)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--package-path", default="")
    parser.add_argument("--validation-sample-limit", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def discover_stage_roots(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"staging root does not exist: {root}")
    roots = []
    for marker in root.rglob(STAGE_MARKER):
        try:
            marker_data = read_json(marker)
            if "source_index" in marker_data:
                roots.append(marker.parent)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not roots:
        raise FileNotFoundError(f"no completed source stages found under: {root}")
    unique = {path.resolve() for path in roots}
    return sorted(unique, key=lambda path: int(read_json(path / STAGE_MARKER)["source_index"]))


def stage_source_index(stage_root: Path) -> int:
    return int(read_json(stage_root / STAGE_MARKER)["source_index"])


def stage_record_paths(stage_root: Path, split: str) -> tuple[Path, Path]:
    index_path = stage_root / "records" / f"{split}.index.jsonl"
    record_path = stage_root / "records" / VARIANT / f"{split}.jsonl"
    if not index_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(
            "staging record pair is missing; expected index and local256 SFT files: "
            f"{index_path}, {record_path}"
        )
    return index_path, record_path


def iter_stage_rows(stage_root: Path, split: str):
    index_path, record_path = stage_record_paths(stage_root, split)
    with (
        index_path.open("r", encoding="utf-8-sig") as index_handle,
        record_path.open("r", encoding="utf-8-sig") as record_handle,
    ):
        for line_number, pair in enumerate(zip_longest(index_handle, record_handle), start=1):
            index_line, record_line = pair
            if index_line is None or record_line is None:
                raise ValueError(f"index/SFT length mismatch at {index_path}:{line_number}")
            if not index_line.strip() and not record_line.strip():
                continue
            if not index_line.strip() or not record_line.strip():
                raise ValueError(f"blank index/SFT mismatch at {index_path}:{line_number}")
            index_item = json.loads(index_line)
            record = json.loads(record_line)
            if str(index_item.get("id")) != str(record.get("id")):
                raise ValueError(
                    f"index/SFT id mismatch at {index_path}:{line_number}: "
                    f"{index_item.get('id')!r} != {record.get('id')!r}"
                )
            yield index_item, record


def is_base_grid(index_item: dict, record: dict) -> bool:
    grid_kind = str(index_item.get("grid_kind") or "").strip()
    if grid_kind:
        return grid_kind == "base"
    offset = index_item.get("translation_offset")
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        return int(offset[0]) == 0 and int(offset[1]) == 0
    meta = record.get("meta") or {}
    offset = meta.get("translation_offset")
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        return int(offset[0]) == 0 and int(offset[1]) == 0
    try:
        return int(meta["x0"]) % 256 == 0 and int(meta["y0"]) % 256 == 0
    except (KeyError, TypeError, ValueError):
        return False


def record_key_candidates(index_item: dict, record: dict) -> list[str]:
    keys = []
    for value in (
        index_item.get("id"),
        record.get("id"),
        (record.get("meta") or {}).get("stable_patch_id"),
        (record.get("meta") or {}).get("grid_patch_id"),
    ):
        value = str(value or "").strip()
        if value and value not in keys:
            keys.append(value)
    meta = record.get("meta") or {}
    tile_id = str(meta.get("tile_id") or "").strip()
    try:
        x0, y0 = int(meta["x0"]), int(meta["y0"])
    except (KeyError, TypeError, ValueError):
        x0 = y0 = None
    if tile_id and x0 is not None and y0 is not None:
        key = f"{tile_id}_x{x0:05d}_y{y0:05d}"
        if key not in keys:
            keys.append(key)
    return keys


def build_stage_row_map(stage_root: Path) -> dict[str, Path]:
    """Map stable patch ids to clean local256 PNGs in one source stage."""

    result: dict[str, Path] = {}
    for split in SPLITS:
        for index_item, record in iter_stage_rows(stage_root, split):
            if not is_base_grid(index_item, record):
                continue
            relative = str(record.get("image") or "").strip()
            if not relative:
                raise ValueError(f"clean record {record.get('id')} has no image path")
            source = stage_root / "variants" / VARIANT / Path(*PurePosixPath(relative).parts)
            if not source.is_file():
                continue
            for key in record_key_candidates(index_item, record):
                previous = result.get(key)
                if previous is not None and previous != source:
                    raise ValueError(f"duplicate clean image key {key}: {previous}, {source}")
                result[key] = source
    if not result:
        raise FileNotFoundError(f"clean stage contains no base-grid local256 images: {stage_root}")
    return result


def normalize_relpath(value: str, prefix: str, split: str, primary: str) -> str:
    value = str(value or "").replace("\\", "/").strip()
    if value.startswith(prefix + "/"):
        return value
    primary_parts = list(PurePosixPath(primary).parts)
    if len(primary_parts) >= 3 and primary_parts[0] == "images":
        return "/".join((prefix, split, *primary_parts[2:]))
    name = PurePosixPath(value).name or (PurePosixPath(primary).name or "sample.png")
    return "/".join((prefix, split, name))


def auxiliary_value_candidates(record: dict, kind: str, split: str) -> list[str]:
    field = "raw_lane_image" if kind == "raw_lane" else "pose_image"
    candidates = []
    value = record.get(field)
    if value:
        candidates.append(str(value).replace("\\", "/"))
    for value in record.get("images") or []:
        value = str(value).replace("\\", "/")
        if value.startswith((f"{kind}_images/", "raw_lane_images/" if kind == "raw_lane" else "pose_images/")):
            candidates.append(value)
    primary = str(record.get("image") or "").replace("\\", "/")
    prefix = "raw_lane_images" if kind == "raw_lane" else "pose_images"
    candidates.append(normalize_relpath("", prefix, split, primary))
    if primary:
        stem = PurePosixPath(primary).stem
        suffix = PurePosixPath(primary).suffix or ".png"
        primary_parts = list(PurePosixPath(primary).parts)
        if len(primary_parts) >= 3 and primary_parts[0] == "images":
            group = "/".join(primary_parts[2:-1])
            name = primary_parts[-1]
            names = [name, f"{stem}_{kind}{suffix}", f"{stem}_{'raw_lane' if kind == 'raw_lane' else 'pose'}{suffix}"]
            for name_value in names:
                candidates.append("/".join(item for item in (prefix, split, group, name_value) if item))
    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def resolve_auxiliary_image(stage_root: Path, record: dict, kind: str, split: str) -> tuple[Path, str]:
    candidates = auxiliary_value_candidates(record, kind, split)
    for relative in candidates:
        source = stage_root / "variants" / VARIANT / Path(*PurePosixPath(relative).parts)
        if source.is_file():
            prefix = "raw_lane_images" if kind == "raw_lane" else "pose_images"
            output_relative = normalize_relpath(relative, prefix, split, str(record.get("image") or ""))
            return source, output_relative
    raise FileNotFoundError(
        f"sample={record.get('id')} cannot resolve {kind} image under {stage_root}; "
        f"tried={candidates}"
    )


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
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


def target_lines(record: dict) -> list:
    conversations = record.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError(f"sample={record.get('id')} has invalid conversations")
    value = conversations[-1].get("value", "")
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"sample={record.get('id')} assistant target is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"sample={record.get('id')} assistant target is not an object")
    lines = payload.get("lines", [])
    if not isinstance(lines, list):
        raise ValueError(f"sample={record.get('id')} target lines is not a list")
    return lines


def is_empty_target(record: dict) -> bool:
    lines = target_lines(record)
    return not any(isinstance(line, dict) and str(line.get("category", "")).strip() for line in lines)


def choose_train_rows(rows: list[dict], empty_ratio: float, seed: int) -> tuple[list[dict], dict]:
    if not 0 <= empty_ratio < 1:
        raise ValueError("--empty-ratio must be in [0, 1)")
    nonempty = [row for row in rows if not row["is_empty"]]
    empty = [row for row in rows if row["is_empty"]]
    if nonempty:
        max_empty = math.floor(len(nonempty) * empty_ratio / (1.0 - empty_ratio))
    else:
        max_empty = 0
    keep_empty = min(len(empty), max_empty)
    rng = random.Random(seed)
    empty_sorted = sorted(empty, key=lambda row: str(row["index"].get("id", "")))
    rng.shuffle(empty_sorted)
    selected = nonempty + empty_sorted[:keep_empty]
    selected.sort(key=lambda row: str(row["index"].get("id", "")))
    return selected, {
        "candidate_total": len(rows),
        "candidate_nonempty": len(nonempty),
        "candidate_empty": len(empty),
        "empty_ratio_target": empty_ratio,
        "empty_max_kept": max_empty,
        "empty_kept": keep_empty,
        "final_train_samples": len(selected),
        "final_empty_ratio": keep_empty / len(selected) if selected else 0.0,
    }


def three_image_prompt() -> str:
    return "\n".join([
        "<image>",
        "<image>",
        "<image>",
        "You are a professional self-driving agent reconstructing road structure from BEV data.",
        "The first image is the clean BEV road-structure image.",
        "The second image is a lane image predicted by a PV camera model; treat it as a noisy hint only.",
        "The third image is a historical vehicle-trajectory image; use it only as supporting evidence.",
        "Construct the complete road map visible in the current 256x256 image patch.",
        "Coordinates use a normalized 0-1000 grid over the 256x256 image patch.",
        "Do not copy an auxiliary image blindly when it conflicts with the clean BEV evidence.",
        "Return only valid JSON in the form {\"lines\":[...]} with no markdown or explanation.",
        "For every centerline, include category, lane_type, start_type, end_type, and points.",
        "Use lane_type common, right_turn, waiting_area, bus_lane, main_auxiliary_connector, or other.",
        "For every intersection, include category, intersection_type, is_cut, and polygon points.",
        "Use intersection_type common, t_intersection, small_untyped, t_lane_change_area, or other.",
    ])


def transform_record(row: dict, split: str, clean_path: Path, aux_stage: Path, output_root: Path, copy_mode: str) -> tuple[dict, str, str]:
    index_item = row["index"]
    record = dict(row["record"])
    primary = str(record.get("image") or "").replace("\\", "/")
    if not primary.startswith("images/"):
        primary = normalize_relpath(primary, "images", split, primary)
    raw_source, raw_relative = resolve_auxiliary_image(aux_stage, record, "raw_lane", split)
    pose_source, pose_relative = resolve_auxiliary_image(aux_stage, record, "pose", split)
    clean_source = clean_path
    primary_destination = output_root / Path(*PurePosixPath(primary).parts)
    raw_destination = output_root / Path(*PurePosixPath(raw_relative).parts)
    pose_destination = output_root / Path(*PurePosixPath(pose_relative).parts)
    modes = [
        link_or_copy(clean_source, primary_destination, copy_mode),
        link_or_copy(raw_source, raw_destination, copy_mode),
        link_or_copy(pose_source, pose_destination, copy_mode),
    ]
    meta = dict(record.get("meta") or {})
    meta.update({
        "source_index": int(index_item.get("source_index", -1)),
        "raw_sample_id": str(index_item.get("raw_sample_id", "")),
        "stride": 256,
        "train_stride": 256,
        "grid_kind": "base",
        "translation_offset": [0, 0],
        "raw_lane_overlay": False,
        "raw_lane_separate_image": True,
        "raw_lane_image_source": "patch_tif/0_lane.tif",
        "pose_image_source": "patch_tif/0_pose.tif",
        "input_image_roles": list(IMAGE_ROLES),
        "three_image_input": True,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
    })
    conversations = list(record.get("conversations") or [])
    if len(conversations) < 2:
        raise ValueError(f"sample={record.get('id')} has invalid conversations")
    conversations[0] = dict(conversations[0])
    conversations[0]["value"] = three_image_prompt()
    record.update({
        "image": primary,
        "images": [primary, raw_relative, pose_relative],
        "raw_lane_image": raw_relative,
        "pose_image": pose_relative,
        "meta": meta,
        "conversations": conversations,
    })
    return record, ",".join(modes), str(index_item.get("id"))


def collect_rows(aux_stages: list[Path]) -> tuple[list[dict], dict]:
    owner: dict[str, int] = {}
    collisions = []
    for stage in aux_stages:
        source_index = stage_source_index(stage)
        for split in SPLITS:
            for index_item, _record in iter_stage_rows(stage, split):
                raw_id = str(index_item.get("raw_sample_id") or "").strip()
                if not raw_id:
                    raise ValueError(f"aux index has no raw_sample_id: {stage} {split}")
                if raw_id in owner and owner[raw_id] != source_index:
                    collisions.append({"raw_sample_id": raw_id, "previous": owner[raw_id], "new": source_index})
                owner[raw_id] = source_index

    rows = []
    seen_ids: set[str] = set()
    for stage in aux_stages:
        source_index = stage_source_index(stage)
        for split in SPLITS:
            for index_item, record in iter_stage_rows(stage, split):
                raw_id = str(index_item.get("raw_sample_id") or "")
                if owner.get(raw_id) != source_index or not is_base_grid(index_item, record):
                    continue
                patch_id = str(index_item.get("id") or record.get("id") or "").strip()
                if not patch_id:
                    raise ValueError(f"aux row has no patch id: {stage} {split}")
                if patch_id in seen_ids:
                    raise ValueError(f"duplicate base-grid patch id across staging: {patch_id}")
                seen_ids.add(patch_id)
                rows.append({
                    "index": dict(index_item),
                    "record": record,
                    "stage": stage,
                    "split": split,
                    "is_empty": is_empty_target(record),
                })
    if not rows:
        raise ValueError("no base-grid rows found in auxiliary staging")
    return rows, {"raw_sample_owner_collisions": collisions, "unique_base_rows": len(rows)}


def build_clean_maps(clean_stages: list[Path]) -> dict[int, dict[str, Path]]:
    result = {}
    for stage in clean_stages:
        source_index = stage_source_index(stage)
        if source_index in result:
            raise ValueError(f"duplicate clean source_index={source_index}: {stage}")
        result[source_index] = build_stage_row_map(stage)
        print(f"[three-image-local256] clean source {source_index}: {len(result[source_index])} base image keys", flush=True)
    return result


def resolve_clean_image(clean_maps: dict[int, dict[str, Path]], row: dict) -> Path:
    source_index = int(row["index"].get("source_index", -1))
    mapping = clean_maps.get(source_index)
    if mapping is None:
        raise FileNotFoundError(f"no clean stage for source_index={source_index}, sample={row['index'].get('id')}")
    matches = [mapping[key] for key in record_key_candidates(row["index"], row["record"]) if key in mapping]
    unique = []
    for match in matches:
        if match not in unique:
            unique.append(match)
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected one clean local256 image for sample={row['index'].get('id')} "
            f"source_index={source_index}; matches={unique}"
        )
    return unique[0]


def make_meta(record: dict, row: dict, split: str, is_empty: bool) -> dict:
    index_item = row["index"]
    return {
        "id": str(index_item.get("id") or record.get("id")),
        "image": record["image"],
        "images": list(record["images"]),
        "raw_lane_image": record["raw_lane_image"],
        "pose_image": record["pose_image"],
        "split": split,
        "source_index": int(index_item.get("source_index", -1)),
        "raw_sample_id": str(index_item.get("raw_sample_id", "")),
        "difficulty": index_item.get("difficulty"),
        "difficulty_score": index_item.get("difficulty_score"),
        "has_intersection": bool(index_item.get("has_intersection", False)),
        "is_empty": is_empty,
        "grid_kind": "base",
        "translation_offset": [0, 0],
        "meta": dict(record.get("meta") or {}),
    }


def sha256_ids(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update((str(record["id"]) + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_output(root: Path, sample_limit: int) -> dict:
    counts = Counter()
    empty_counts = Counter()
    decoded = 0
    for split in SPLITS:
        path = root / "phase_a" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, record in iter_jsonl(path):
            images = record.get("images")
            if not isinstance(images, list) or len(images) != 3:
                raise ValueError(f"{path}:{line_number} must contain exactly three images")
            expected = ("images/", "raw_lane_images/", "pose_images/")
            for relative, prefix in zip(images, expected):
                relative = str(relative)
                if not relative.startswith(prefix) or not (root / Path(*PurePosixPath(relative).parts)).is_file():
                    raise FileNotFoundError(f"{path}:{line_number} missing image {relative}")
            prompt = str((record.get("conversations") or [{}])[0].get("value", ""))
            if prompt.count("<image>") != 3:
                raise ValueError(f"{path}:{line_number} prompt must contain three image tokens")
            empty = is_empty_target(record)
            counts[split] += 1
            empty_counts[split] += int(empty)
            if sample_limit > 0 and decoded < sample_limit:
                for relative in images:
                    with Image.open(root / Path(*PurePosixPath(str(relative)).parts)) as image:
                        if image.size != (256, 256):
                            raise ValueError(f"{path}:{line_number} image size is {image.size}, expected (256,256)")
                decoded += 1
    train_empty_ratio = empty_counts["train"] / counts["train"] if counts["train"] else 0.0
    result = {
        "status": "passed",
        "dataset_root": str(root),
        "split_counts": dict(counts),
        "empty_counts": dict(empty_counts),
        "train_empty_ratio": train_empty_ratio,
        "decoded_sample_triplets": decoded,
        "image_roles": list(IMAGE_ROLES),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
    }
    if train_empty_ratio > 0.050000001:
        raise ValueError(f"train empty ratio exceeds 5%: {train_empty_ratio}")
    write_json(root / "three_image_validation.json", result)
    return result


def package_dataset(root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = package_path.with_name(package_path.name + ".partial")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, mode="w") as archive:
        archive.add(root, arcname=root.name, recursive=True)
    temporary.replace(package_path)


def main(argv=None) -> None:
    args = parse_args(argv)
    clean_root = Path(args.clean_staging_root).expanduser().resolve()
    aux_root = Path(args.aux_staging_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    output_root = work_root / "output" / "rawlane_pose_three_image_local256_stride256_all"
    package_path = Path(args.package_path).expanduser().resolve() if args.package_path else work_root / "packages" / "rawlane_pose_three_image_local256_stride256_all.tar"
    complete_marker = output_root / "build_complete.json"
    if args.resume and complete_marker.is_file():
        print(f"[three-image-local256] reuse completed output: {output_root}", flush=True)
        print(json.dumps(validate_output(output_root, args.validation_sample_limit), ensure_ascii=False, indent=2), flush=True)
        if not args.skip_package and not package_path.is_file():
            package_dataset(output_root, package_path)
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output exists but is not marked complete: {output_root}. "
            "Use a new --work-root or remove only this generated output after inspection."
        )

    clean_stages = discover_stage_roots(clean_root)
    aux_stages = discover_stage_roots(aux_root)
    clean_by_index = build_clean_maps(clean_stages)
    aux_by_index = {stage_source_index(stage): stage for stage in aux_stages}
    missing_clean = sorted(set(aux_by_index) - set(clean_by_index))
    if missing_clean:
        raise FileNotFoundError(f"auxiliary source stages have no clean counterpart: {missing_clean}")
    rows, collection_report = collect_rows(aux_stages)
    train_candidates = [row for row in rows if row["split"] == "train"]
    selected_train, selection_report = choose_train_rows(train_candidates, args.empty_ratio, args.selection_seed)
    selected_by_id = {str(row["index"]["id"]): row for row in selected_train}
    split_rows = {
        "train": selected_train,
        "eval": [row for row in rows if row["split"] == "eval"],
        "test": [row for row in rows if row["split"] == "test"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[three-image-local256] auxiliary base rows={len(rows)}", flush=True)
    print(json.dumps(selection_report, ensure_ascii=False, indent=2), flush=True)
    phase_counts = Counter()
    difficulty_counts = Counter()
    intersection_counts = Counter()
    link_modes = Counter()
    for split in SPLITS:
        record_rows = []
        meta_rows = []
        for ordinal, row in enumerate(split_rows[split], start=1):
            clean_source = resolve_clean_image(clean_by_index, row)
            transformed, modes, patch_id = transform_record(row, split, clean_source, row["stage"], output_root, args.copy_mode)
            record_rows.append(transformed)
            meta_rows.append(make_meta(transformed, row, split, row["is_empty"]))
            phase_counts[split] += 1
            difficulty_counts[f"{split}:{row['index'].get('stratum', row['index'].get('difficulty', 'unknown'))}"] += 1
            intersection_counts[split] += int(bool(row["index"].get("has_intersection", False)))
            link_modes.update(modes.split(","))
            if args.progress_every > 0 and ordinal % args.progress_every == 0:
                print(f"[three-image-local256] materialized {split}={ordinal}", flush=True)
        write_jsonl(output_root / "phase_a" / f"{split}.jsonl", record_rows)
        write_jsonl(output_root / "phase_a" / f"meta_{split}.jsonl", meta_rows)

    info = {
        "dataset_version": "rc_dataset_v2_three_image_local256_stride256_all_v1",
        "task": "lane_intersection",
        "input_root": "completed_staging_only",
        "patch_size": 256,
        "context_size": 256,
        "stride": 256,
        "train_stride": 256,
        "eval_test_stride": 256,
        "coord_mode": "norm1000",
        "coord_range": 1000,
        "three_image_input": True,
        "multi_image_input": {
            "enabled": True,
            "num_images_per_sample": 3,
            "image_roles": list(IMAGE_ROLES),
            "image_order": list(IMAGE_ROLES),
        },
        "input_overlay": {
            "raw_lane_overlay": False,
            "raw_lane_separate_image": True,
            "raw_lane_image_source": "patch_tif/0_lane.tif",
            "pose_image_source": "patch_tif/0_pose.tif",
        },
        "train_policy": {
            "keep_all_nonempty": True,
            "empty_ratio_max": args.empty_ratio,
            "selection_seed": args.selection_seed,
            "difficulty_stratification": False,
            "target_sample_count": None,
        },
        "source_staging": {
            "clean_staging_root": str(clean_root),
            "aux_staging_root": str(aux_root),
            "source_indices": sorted(aux_by_index),
        },
        "split_counts": dict(phase_counts),
        "intersection_counts": dict(intersection_counts),
        "difficulty_counts": dict(difficulty_counts),
        "link_modes": dict(link_modes),
    }
    write_json(output_root / "dataset_info.json", info)
    build_summary = {
        "status": "built",
        "output_root": str(output_root),
        "package_path": str(package_path),
        "collection": collection_report,
        "selection": selection_report,
        "split_counts": dict(phase_counts),
        "source_indices": sorted(aux_by_index),
    }
    write_json(output_root / "balance_report.json", selection_report)
    write_json(output_root / "build_summary.json", build_summary)
    write_json(output_root / "split_manifest.json", {
        "split_unit": "staging_patch_record",
        "stride": 256,
        "split_counts": dict(phase_counts),
        "split_id_sha256": {
            split: sha256_ids([
                {"id": str(row["index"]["id"])} for row in split_rows[split]
            ]) for split in SPLITS
        },
        "fixed_eval_source": False,
    })
    validation = validate_output(output_root, args.validation_sample_limit)
    build_summary["validation"] = validation
    write_json(output_root / "build_summary.json", build_summary)
    if not args.skip_package:
        print(f"[three-image-local256] packaging: {package_path}", flush=True)
        package_dataset(output_root, package_path)
    write_json(output_root / "build_complete.json", {
        "status": "passed",
        "dataset_root": str(output_root),
        "package_path": str(package_path) if not args.skip_package else "",
        "train_samples": phase_counts["train"],
        "train_empty_ratio": validation["train_empty_ratio"],
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
    })
    print(json.dumps(build_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
