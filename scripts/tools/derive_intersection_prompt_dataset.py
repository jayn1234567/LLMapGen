#!/usr/bin/env python3
"""Derive a centerline-only SFT dataset conditioned on current-patch intersection GT."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from itertools import zip_longest
from pathlib import Path, PurePosixPath


PROMPT_MARKER = "Current-patch intersection ground truth JSON:"
TASK_MODE = "centerline_conditioned_on_gt_intersections"
SPLITS = ("train", "eval", "test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Completed local512 lane+intersection dataset root.")
    parser.add_argument("--output-root", required=True, help="Self-contained derived dataset root.")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


def compact_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_target(record: dict) -> dict:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"sample={record.get('id')} has no conversations list")
    for message in conversations:
        if str(message.get("from", message.get("role", ""))).lower() in {"gpt", "assistant"}:
            value = message.get("value", message.get("content"))
            payload = json.loads(value)
            if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
                raise ValueError(f"sample={record.get('id')} assistant target is not a lines object")
            return payload
    raise ValueError(f"sample={record.get('id')} has no assistant target")


def build_prompt(patch_size: int, coord_range: int, intersections: list[dict]) -> str:
    intersection_payload = compact_json({"lines": intersections})
    return "\n".join([
        "<image>",
        "Please construct all road centerlines in the current BEV (Bird's Eye View) image patch.",
        f"Coordinates use a normalized 0-{coord_range} grid over the original {patch_size}x{patch_size} image patch.",
        "The exact current-patch intersection ground truth is provided below as an oracle structural constraint.",
        "Use these intersection polygons and types to understand road connectivity, but do not reproduce intersections in the answer.",
        'Each supplied intersection includes "intersection_type", "is_cut", and closed polygon "points".',
        "",
        PROMPT_MARKER,
        intersection_payload,
        "",
        'Return only valid JSON in the form {"lines":[...]} with no extra explanation.',
        'Output centerlines only. Do not output any object whose "category" is "intersection".',
        (
            'For every centerline, include "lane_type" with exactly one of: '
            '"common" for a regular centerline, "right_turn" for a right-turn-only '
            'centerline, "waiting_area" for a waiting-area centerline, "bus_lane" '
            'for a bus-lane centerline, "main_auxiliary_connector" for a connector '
            'between main and auxiliary roads, or "other" for any remaining lane class.'
        ),
    ])


def extract_prompt_intersections(prompt: str) -> list[dict]:
    if not isinstance(prompt, str) or PROMPT_MARKER not in prompt:
        raise ValueError(f"prompt does not contain {PROMPT_MARKER!r}")
    remainder = prompt.split(PROMPT_MARKER, 1)[1].lstrip()
    first_line = remainder.splitlines()[0] if remainder else ""
    payload = json.loads(first_line)
    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        raise ValueError("intersection ground truth prompt payload must contain a lines list")
    intersections = payload["lines"]
    if any(str(item.get("category", "")).lower() != "intersection" for item in intersections):
        raise ValueError("intersection ground truth prompt payload contains a non-intersection object")
    return intersections


def transform_record(
    record: dict,
    dataset_variant: str | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    payload = parse_target(record)
    centerlines = []
    intersections = []
    for item in payload["lines"]:
        category = str(item.get("category", "")).strip().lower()
        if category == "centerline":
            centerlines.append(item)
        elif category == "intersection":
            intersections.append(item)
        else:
            raise ValueError(f"sample={record.get('id')} has unsupported category={category!r}")

    result = dict(record)
    meta = dict(record.get("meta") or {})
    patch_size = int(meta.get("target_size", meta.get("pixel_patch_size", 0)))
    coord_range = int(meta.get("coord_range", 0))
    if patch_size <= 0 or coord_range <= 0:
        raise ValueError(f"sample={record.get('id')} has invalid target_size/coord_range metadata")
    variant = str(dataset_variant or f"local{patch_size}_intersection_prompt")
    meta.update({
        "dataset_variant": variant,
        "task_mode": TASK_MODE,
        "oracle_intersection_conditioning": True,
        "current_patch_intersection_gt_count": len(intersections),
        "intersection_hint_source_train": "current_patch_ground_truth",
        "intersection_hint_source_infer": "current_patch_ground_truth_required",
    })
    result["meta"] = meta
    result["conversations"] = [
        {"from": "human", "value": build_prompt(patch_size, coord_range, intersections)},
        {"from": "gpt", "value": compact_json({"lines": centerlines})},
    ]
    return result, centerlines, intersections


def safe_relative_image(value: str) -> Path:
    normalized = str(value).replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe image path: {value!r}")
    return Path(*path.parts)


def link_or_copy(source: Path, destination: Path, mode: str, resume: bool) -> str:
    if resume and destination.is_file():
        return "reused"
    if not source.is_file():
        raise FileNotFoundError(f"source image not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    used_mode = mode
    try:
        if mode == "hardlink":
            try:
                temporary.hardlink_to(source)
            except OSError:
                shutil.copy2(source, temporary)
                used_mode = "copy_fallback"
        else:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return used_mode


def transform_split(input_root: Path, output_root: Path, split: str, args) -> dict:
    source_path = input_root / "phase_a" / f"{split}.jsonl"
    source_meta_path = input_root / "phase_a" / f"meta_{split}.jsonl"
    if not source_path.is_file() or not source_meta_path.is_file():
        raise FileNotFoundError(f"missing source split or metadata: {source_path}, {source_meta_path}")
    output_phase = output_root / "phase_a"
    output_phase.mkdir(parents=True, exist_ok=True)
    destination_path = output_phase / f"{split}.jsonl"
    destination_meta_path = output_phase / f"meta_{split}.jsonl"

    counts = Counter()
    link_modes = Counter()
    with (
        source_path.open("r", encoding="utf-8-sig") as source,
        source_meta_path.open("r", encoding="utf-8-sig") as source_meta,
        destination_path.open("w", encoding="utf-8") as destination,
        destination_meta_path.open("w", encoding="utf-8") as destination_meta,
    ):
        source_rows = (line for line in source if line.strip())
        meta_rows = (line for line in source_meta if line.strip())
        for index, pair in enumerate(zip_longest(source_rows, meta_rows), start=1):
            line, meta_line = pair
            if line is None or meta_line is None:
                raise ValueError(f"source SFT/meta row count mismatch in split={split}")
            record = json.loads(line)
            meta_record = json.loads(meta_line)
            if str(record.get("id")) != str(meta_record.get("id")):
                raise ValueError(f"source SFT/meta id mismatch in split={split} row={index}")
            transformed, centerlines, intersections = transform_record(
                record,
                dataset_variant=output_root.name,
            )
            parsed_intersections = extract_prompt_intersections(transformed["conversations"][0]["value"])
            if parsed_intersections != intersections:
                raise ValueError(f"sample={record.get('id')} prompt intersection round-trip mismatch")

            relative_image = safe_relative_image(transformed["image"])
            used_mode = link_or_copy(
                input_root / relative_image,
                output_root / relative_image,
                args.copy_mode,
                args.resume,
            )
            link_modes[used_mode] += 1
            destination.write(compact_json(transformed) + "\n")
            meta_record = dict(meta_record)
            meta_record["meta"] = transformed["meta"]
            destination_meta.write(compact_json(meta_record) + "\n")

            counts["samples"] += 1
            counts["centerlines"] += len(centerlines)
            counts["intersections_in_prompt"] += len(intersections)
            if intersections:
                counts["samples_with_intersections"] += 1
            for item in centerlines:
                counts[f"assistant_lane_type:{item.get('lane_type')}"] += 1
            for item in intersections:
                counts[f"prompt_intersection_type:{item.get('intersection_type')}"] += 1
            if args.progress_every and index % args.progress_every == 0:
                print(f"[intersection-prompt] {split}: {index} records", flush=True)
    return {"counts": dict(counts), "image_materialization_modes": dict(link_modes)}


def derive_dataset(input_root: Path, output_root: Path, args) -> dict:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root:
        raise ValueError("input and output dataset roots must differ")
    output_root.mkdir(parents=True, exist_ok=True)
    split_reports = {
        split: transform_split(input_root, output_root, split, args)
        for split in SPLITS
    }

    for name in ("balance_report.json", "split_manifest.json", "pairing_report.json"):
        source = input_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)

    source_info_path = input_root / "dataset_info.json"
    source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    source_info.update({
        "dataset_variant": output_root.name,
        "task": TASK_MODE,
        "base_dataset_root": str(input_root),
        "oracle_intersection_conditioning": True,
        "prompt_condition_categories": ["intersection"],
        "assistant_target_categories": ["centerline"],
        "derived_split_reports": split_reports,
    })
    (output_root / "dataset_info.json").write_text(
        json.dumps(source_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "passed",
        "task_mode": TASK_MODE,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": split_reports,
    }
    (output_root / "derivation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None):
    args = parse_args(argv)
    report = derive_dataset(Path(args.input_root), Path(args.output_root), args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
