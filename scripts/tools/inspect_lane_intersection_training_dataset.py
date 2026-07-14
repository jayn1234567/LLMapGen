#!/usr/bin/env python3
"""Inspect a prepared lane/intersection SFT dataset before expensive training."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


ASSISTANT_ROLES = {"assistant", "gpt"}
HUMAN_ROLES = {"human", "user"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Dataset root or extraction root.")
    parser.add_argument("--phase", default="phase_a", help="Preferred phase directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--image-checks-per-split", type=int, default=64)
    parser.add_argument("--expected-image-size", type=int, default=0)
    parser.add_argument("--coord-min", type=float, default=0.0)
    parser.add_argument("--coord-max", type=float, default=1000.0)
    parser.add_argument("--forbid-lane-type", action="append", default=[])
    parser.add_argument("--require-intersection-type-fields", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", default="")
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def stable_value(value: Any) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)


def numeric_equal(value: Any, expected: str) -> bool:
    try:
        return float(value) == float(expected)
    except (TypeError, ValueError):
        return stable_value(value).lower() == str(expected).strip().lower()


def resolve_dataset_root(input_root: Path, phase: str) -> tuple[Path, str]:
    root = input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    direct_phase = root / phase / "train.jsonl"
    if direct_phase.is_file():
        return root, "phase"
    if (root / "train.jsonl").is_file():
        return root, "flat"

    phase_candidates = sorted(root.rglob(f"{phase}/train.jsonl"))
    phase_candidates = [path for path in phase_candidates if "__MACOSX" not in path.parts]
    if len(phase_candidates) == 1:
        return phase_candidates[0].parent.parent, "phase"
    if len(phase_candidates) > 1:
        roots = "\n".join(str(path.parent.parent) for path in phase_candidates[:20])
        raise RuntimeError(f"multiple {phase}/train.jsonl dataset roots found:\n{roots}")

    flat_candidates = sorted(root.rglob("train.jsonl"))
    flat_candidates = [
        path
        for path in flat_candidates
        if path.parent.name not in {"phase_a", "phase_b"} and "__MACOSX" not in path.parts
    ]
    if len(flat_candidates) == 1:
        return flat_candidates[0].parent, "flat"
    candidates = phase_candidates + flat_candidates
    preview = "\n".join(str(path) for path in candidates[:20]) or "<none>"
    raise FileNotFoundError(
        f"unable to resolve one prepared dataset root below {root}; train candidates:\n{preview}"
    )


def resolve_split_path(dataset_root: Path, phase: str, layout: str, split: str) -> Path | None:
    base = dataset_root / phase if layout == "phase" else dataset_root
    names = [f"{split}.jsonl"]
    if split == "eval":
        names.append("val.jsonl")
    for name in names:
        path = base / name
        if path.is_file():
            return path
    return None


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"expected JSON array in {path}")
            for record in payload:
                if isinstance(record, dict):
                    yield record
            return

        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            yield record


def conversation_text(record: dict[str, Any], roles: set[str]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    chunks = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        if role in roles:
            value = message.get("value", message.get("content", ""))
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def parse_assistant_json(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return json.loads(value)


def extract_lines(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("lines", "target_lines", "map", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "points" in payload:
            return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def semantic_fields(value: Any, result: dict[str, list[Any]]) -> None:
    if isinstance(value, dict):
        normalized = {normalized_key(key): item for key, item in value.items()}
        for key, item in normalized.items():
            if key == "intersectiontype":
                result["intersection_type"].append(item)
            elif key == "intersectionsubtype":
                result["intersection_subtype"].append(item)
            elif key.endswith("lanetype") or key.endswith("lanetypecode"):
                result["lane_type"].append(item)
        if "intersectiontype" in normalized:
            main_type = stable_value(normalized["intersectiontype"])
            sub_type = stable_value(normalized.get("intersectionsubtype"))
            result["intersection_pair"].append(f"{main_type}|{sub_type}")
        for item in value.values():
            semantic_fields(item, result)
    elif isinstance(value, list):
        for item in value:
            semantic_fields(item, result)


def image_relpath(record: dict[str, Any]) -> tuple[str, str]:
    value = record.get("image")
    if isinstance(value, str) and value.strip():
        return value.replace("\\", "/").lstrip("/"), "image"
    values = record.get("images")
    if isinstance(values, str) and values.strip():
        return values.replace("\\", "/").lstrip("/"), "images_string"
    if isinstance(values, list) and values and isinstance(values[0], str):
        return values[0].replace("\\", "/").lstrip("/"), "images_list"
    return "", "missing"


def update_reservoir(
    reservoir: list[str], value: str, seen: int, limit: int, rng: random.Random
) -> None:
    if not value or limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(value)
        return
    position = rng.randrange(seen)
    if position < limit:
        reservoir[position] = value


def update_coordinates(
    line: dict[str, Any], split_stats: dict[str, Any], coord_min: float, coord_max: float
) -> None:
    points = line.get("points")
    if not isinstance(points, list):
        split_stats["lines_without_points"] += 1
        return
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            split_stats["invalid_points"] += 1
            continue
        x, y = point[0], point[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            split_stats["invalid_points"] += 1
            continue
        x_value = float(x)
        y_value = float(y)
        split_stats["coordinate_pairs"] += 1
        split_stats["coordinate_min"] = min(split_stats["coordinate_min"], x_value, y_value)
        split_stats["coordinate_max"] = max(split_stats["coordinate_max"], x_value, y_value)
        if x_value < coord_min or x_value > coord_max or y_value < coord_min or y_value > coord_max:
            split_stats["out_of_range_points"] += 1


def inspect_split(
    path: Path,
    dataset_root: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "samples": 0,
        "top_level_keys": Counter(),
        "image_field": Counter(),
        "conversation_role_sequences": Counter(),
        "missing_conversations": 0,
        "missing_assistant": 0,
        "invalid_assistant_json": 0,
        "target_categories": Counter(),
        "centerline_type_values": Counter(),
        "intersection_type_values": Counter(),
        "intersection_subtype_values": Counter(),
        "intersection_type_pairs": Counter(),
        "target_intersection_type_values": Counter(),
        "target_intersection_subtype_values": Counter(),
        "target_intersection_type_pairs": Counter(),
        "meta_intersection_type_values": Counter(),
        "meta_intersection_subtype_values": Counter(),
        "meta_intersection_type_pairs": Counter(),
        "lane_type_values": Counter(),
        "forbidden_lane_type_samples": 0,
        "prompt_mentions_intersection_type": 0,
        "prompt_mentions_intersection_subtype": 0,
        "coordinate_pairs": 0,
        "coordinate_min": math.inf,
        "coordinate_max": -math.inf,
        "out_of_range_points": 0,
        "invalid_points": 0,
        "lines_without_points": 0,
        "sample_image_paths": [],
        "sample_ids_with_errors": [],
    }
    forbidden_types = [str(value) for value in args.forbid_lane_type]

    for index, record in enumerate(iter_records(path), start=1):
        if args.max_samples_per_split > 0 and index > args.max_samples_per_split:
            break
        stats["samples"] += 1
        sample_id = str(record.get("id", record.get("sample_id", index)))
        stats["top_level_keys"].update(record.keys())

        rel_image, image_field = image_relpath(record)
        stats["image_field"][image_field] += 1
        update_reservoir(
            stats["sample_image_paths"],
            rel_image,
            stats["samples"],
            args.image_checks_per_split,
            rng,
        )

        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            stats["missing_conversations"] += 1
            if len(stats["sample_ids_with_errors"]) < 20:
                stats["sample_ids_with_errors"].append(sample_id)
            conversations = []
        roles = []
        for message in conversations:
            if isinstance(message, dict):
                roles.append(str(message.get("from", message.get("role", ""))).lower())
        stats["conversation_role_sequences"]["->".join(roles) or "<none>"] += 1

        human_text = conversation_text(record, HUMAN_ROLES).lower()
        if "intersectiontype" in normalized_key(human_text):
            stats["prompt_mentions_intersection_type"] += 1
        if "intersectionsubtype" in normalized_key(human_text):
            stats["prompt_mentions_intersection_subtype"] += 1

        assistant_text = conversation_text(record, ASSISTANT_ROLES)
        payload: Any = None
        if not assistant_text:
            stats["missing_assistant"] += 1
        else:
            try:
                payload = parse_assistant_json(assistant_text)
            except (json.JSONDecodeError, TypeError, ValueError):
                stats["invalid_assistant_json"] += 1
                if len(stats["sample_ids_with_errors"]) < 20:
                    stats["sample_ids_with_errors"].append(sample_id)

        lines = record.get("target_lines")
        if not isinstance(lines, list):
            lines = extract_lines(payload)
        lines = [item for item in lines if isinstance(item, dict)]

        meta_fields: dict[str, list[Any]] = {
            "intersection_type": [],
            "intersection_subtype": [],
            "intersection_pair": [],
            "lane_type": [],
        }
        target_fields: dict[str, list[Any]] = {
            "intersection_type": [],
            "intersection_subtype": [],
            "intersection_pair": [],
            "lane_type": [],
        }
        semantic_fields(record.get("meta", {}), meta_fields)
        if payload is not None:
            semantic_fields(payload, target_fields)

        forbidden_found = False
        for line in lines:
            category = str(line.get("category", line.get("class", "unknown"))).strip().lower()
            stats["target_categories"][category or "<empty>"] += 1
            update_coordinates(line, stats, args.coord_min, args.coord_max)
            normalized = {normalized_key(key): value for key, value in line.items()}
            if category == "centerline":
                lane_type = normalized.get("lanetype", normalized.get("type"))
                if lane_type is not None:
                    stats["centerline_type_values"][stable_value(lane_type)] += 1
                    if any(numeric_equal(lane_type, value) for value in forbidden_types):
                        forbidden_found = True

        for value in target_fields["intersection_type"]:
            stats["target_intersection_type_values"][stable_value(value)] += 1
            stats["intersection_type_values"][stable_value(value)] += 1
        for value in target_fields["intersection_subtype"]:
            stats["target_intersection_subtype_values"][stable_value(value)] += 1
            stats["intersection_subtype_values"][stable_value(value)] += 1
        for value in target_fields["intersection_pair"]:
            stats["target_intersection_type_pairs"][stable_value(value)] += 1
            stats["intersection_type_pairs"][stable_value(value)] += 1
        for value in meta_fields["intersection_type"]:
            stats["meta_intersection_type_values"][stable_value(value)] += 1
            stats["intersection_type_values"][stable_value(value)] += 1
        for value in meta_fields["intersection_subtype"]:
            stats["meta_intersection_subtype_values"][stable_value(value)] += 1
            stats["intersection_subtype_values"][stable_value(value)] += 1
        for value in meta_fields["intersection_pair"]:
            stats["meta_intersection_type_pairs"][stable_value(value)] += 1
            stats["intersection_type_pairs"][stable_value(value)] += 1
        for value in target_fields["lane_type"] + meta_fields["lane_type"]:
            stats["lane_type_values"][stable_value(value)] += 1
            if any(numeric_equal(value, forbidden) for forbidden in forbidden_types):
                forbidden_found = True
        if forbidden_found:
            stats["forbidden_lane_type_samples"] += 1
            if len(stats["sample_ids_with_errors"]) < 20:
                stats["sample_ids_with_errors"].append(sample_id)

        if args.progress_every > 0 and stats["samples"] % args.progress_every == 0:
            print(f"[dataset-inspect] {path.name}: scanned {stats['samples']} samples", flush=True)

    checked_sizes: Counter[str] = Counter()
    missing_images = []
    image_errors = []
    if stats["sample_image_paths"]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image-size inspection") from exc
        for rel_path in stats["sample_image_paths"]:
            full_path = dataset_root / rel_path
            if not full_path.is_file():
                missing_images.append(rel_path)
                continue
            try:
                with Image.open(full_path) as image:
                    checked_sizes[f"{image.width}x{image.height}"] += 1
            except Exception as exc:  # Pillow raises format-specific errors.
                image_errors.append(f"{rel_path}: {exc!r}")

    stats["checked_image_sizes"] = checked_sizes
    stats["missing_checked_images"] = missing_images[:20]
    stats["missing_checked_image_count"] = len(missing_images)
    stats["image_read_errors"] = image_errors[:20]
    stats["image_read_error_count"] = len(image_errors)
    stats["coordinate_min"] = None if math.isinf(stats["coordinate_min"]) else stats["coordinate_min"]
    stats["coordinate_max"] = None if math.isinf(stats["coordinate_max"]) else stats["coordinate_max"]
    del stats["sample_image_paths"]

    for key, value in list(stats.items()):
        if isinstance(value, Counter):
            stats[key] = dict(value.most_common())
    return stats


def build_failures(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures = []
    splits = report["splits"]
    for required_split in ("train", "eval"):
        if required_split in args.splits and required_split not in splits:
            failures.append(f"required split is missing: {required_split}")
    for split, stats in splits.items():
        if stats["samples"] == 0:
            failures.append(f"{split}: no samples")
        if stats["missing_conversations"]:
            failures.append(f"{split}: {stats['missing_conversations']} samples have no conversations")
        if stats["missing_assistant"]:
            failures.append(f"{split}: {stats['missing_assistant']} samples have no assistant target")
        if stats["invalid_assistant_json"]:
            failures.append(f"{split}: {stats['invalid_assistant_json']} assistant targets are invalid JSON")
        incompatible_images = sum(
            value for key, value in stats["image_field"].items() if key != "image"
        )
        if incompatible_images:
            failures.append(
                f"{split}: {incompatible_images} samples do not use the MLLM-compatible singular image field"
            )
        if stats["missing_checked_image_count"]:
            failures.append(f"{split}: {stats['missing_checked_image_count']} checked images are missing")
        if stats["image_read_error_count"]:
            failures.append(f"{split}: {stats['image_read_error_count']} checked images cannot be read")
        if args.expected_image_size:
            expected = f"{args.expected_image_size}x{args.expected_image_size}"
            unexpected = {
                size: count
                for size, count in stats["checked_image_sizes"].items()
                if size != expected
            }
            if unexpected:
                failures.append(f"{split}: unexpected checked image sizes: {unexpected}")
        if stats["out_of_range_points"]:
            failures.append(
                f"{split}: {stats['out_of_range_points']} target points fall outside "
                f"[{args.coord_min}, {args.coord_max}]"
            )
        if stats["invalid_points"]:
            failures.append(f"{split}: {stats['invalid_points']} malformed target points")
        if stats["forbidden_lane_type_samples"]:
            failures.append(
                f"{split}: {stats['forbidden_lane_type_samples']} samples contain forbidden lane type(s) "
                f"{args.forbid_lane_type}"
            )

    if args.require_intersection_type_fields:
        train_report = splits.get("train", {})
        type_total = sum(train_report.get("target_intersection_type_values", {}).values())
        subtype_total = sum(train_report.get("target_intersection_subtype_values", {}).values())
        if type_total == 0:
            failures.append("intersectiontype was not found in train assistant target JSON")
        if subtype_total == 0:
            failures.append("intersectionsubtype was not found in train assistant target JSON")
    return failures


def main() -> None:
    args = parse_args()
    input_root = Path(args.dataset_root)
    dataset_root, layout = resolve_dataset_root(input_root, args.phase)
    rng = random.Random(args.seed)
    report: dict[str, Any] = {
        "input_root": str(input_root.expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "layout": layout,
        "phase": args.phase,
        "constraints": {
            "expected_image_size": args.expected_image_size,
            "coordinate_range": [args.coord_min, args.coord_max],
            "forbidden_lane_types": args.forbid_lane_type,
            "require_intersection_type_fields": args.require_intersection_type_fields,
        },
        "splits": {},
    }

    for split in args.splits:
        split_path = resolve_split_path(dataset_root, args.phase, layout, split)
        if split_path is None:
            print(f"[dataset-inspect] split not found: {split}", flush=True)
            continue
        print(f"[dataset-inspect] scanning {split}: {split_path}", flush=True)
        report["splits"][split] = inspect_split(split_path, dataset_root, args, rng)

    failures = build_failures(report, args)
    report["status"] = "failed" if failures else "passed"
    report["failures"] = failures

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[dataset-inspect] report: {report_path}", flush=True)

    summary = {
        "status": report["status"],
        "dataset_root": report["dataset_root"],
        "layout": layout,
        "split_samples": {
            split: value["samples"] for split, value in report["splits"].items()
        },
        "intersection_type_values": {
            split: value["intersection_type_values"]
            for split, value in report["splits"].items()
        },
        "intersection_subtype_values": {
            split: value["intersection_subtype_values"]
            for split, value in report["splits"].items()
        },
        "intersection_type_pairs": {
            split: value["intersection_type_pairs"]
            for split, value in report["splits"].items()
        },
        "target_intersection_type_values": {
            split: value["target_intersection_type_values"]
            for split, value in report["splits"].items()
        },
        "target_intersection_subtype_values": {
            split: value["target_intersection_subtype_values"]
            for split, value in report["splits"].items()
        },
        "target_intersection_type_pairs": {
            split: value["target_intersection_type_pairs"]
            for split, value in report["splits"].items()
        },
        "meta_intersection_type_values": {
            split: value["meta_intersection_type_values"]
            for split, value in report["splits"].items()
        },
        "meta_intersection_subtype_values": {
            split: value["meta_intersection_subtype_values"]
            for split, value in report["splits"].items()
        },
        "lane_type_values": {
            split: value["lane_type_values"] for split, value in report["splits"].items()
        },
        "image_sizes": {
            split: value["checked_image_sizes"] for split, value in report["splits"].items()
        },
        "coordinate_ranges": {
            split: [value["coordinate_min"], value["coordinate_max"]]
            for split, value in report["splits"].items()
        },
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
