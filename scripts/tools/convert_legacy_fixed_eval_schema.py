#!/usr/bin/env python3
"""Convert a fixed legacy evaluation set to the current prompt/JSON schema.

Images, sample IDs, difficulty buckets, coordinates, and geometry targets are
kept from the legacy records. The user prompt is copied verbatim from a current
Dataset V2 record. Missing semantic type fields remain missing so type metrics
can skip them instead of inventing labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DIFFICULTIES = ("easy", "medium", "hard", "very_hard")
DEFAULT_EXPECTED_COUNTS = "easy=300,medium=300,hard=300,very_hard=100"
DATASET_V2_LOCAL256_PROMPT = """<image>
Please construct the complete road map in the current BEV (Bird's Eye View) image patch.
Coordinates use a normalized 0-1000 grid over the original 256x256 image patch.

Return only valid JSON in the form {"lines":[...]} with no extra explanation.
For every centerline, include "lane_type" with exactly one of: "common" for a regular centerline, "right_turn" for a right-turn-only centerline, or "other" for any remaining lane class. Do not output U-turn reference lines.
For every intersection, include "intersection_type" with exactly one of: "common" for a common intersection, "t_intersection" for a T-intersection, "small_untyped" for a small untyped intersection, or "t_lane_change_area" for a T-shaped lane-change area, or "other" for any remaining or unknown intersection class.

Incoming traces JSON:
[]

Incoming intersections JSON:
[]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt-template-jsonl",
        default="",
        help=(
            "Optional current Dataset V2 JSONL whose first human message supplies the prompt. "
            "When omitted, use the built-in local256 prompt from the immutable 550k release."
        ),
    )
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--expected-counts", default=DEFAULT_EXPECTED_COUNTS)
    parser.add_argument("--require-norm1000", action="store_true")
    parser.add_argument(
        "--image-source-root",
        default="",
        help="Legacy dataset root used to resolve the image bytes referenced by fixed JSONL records.",
    )
    parser.add_argument(
        "--materialize-images",
        choices=["none", "copy"],
        default="none",
        help="Copy resolved legacy images below output-dir/images and rewrite records to that self-contained tree.",
    )
    parser.add_argument("--require-images", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_expected_counts(text: str) -> dict[str, int]:
    result = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid expected-count item: {item!r}")
        result[name.strip()] = int(value)
    return result


def message_role(message: dict[str, Any]) -> str:
    return str(message.get("from", message.get("role", ""))).strip().lower()


def message_value(message: dict[str, Any]) -> Any:
    return message.get("value", message.get("content", ""))


def set_message_value(message: dict[str, Any], value: str) -> None:
    if "value" in message or "content" not in message:
        message["value"] = value
    else:
        message["content"] = value


def find_message(conversations: Any, roles: set[str], reverse: bool = False) -> dict[str, Any] | None:
    if not isinstance(conversations, list):
        return None
    sequence = reversed(conversations) if reverse else conversations
    for message in sequence:
        if isinstance(message, dict) and message_role(message) in roles:
            return message
    return None


def load_prompt_template(path: Path) -> tuple[str, str]:
    for record in read_jsonl(path):
        human = find_message(record.get("conversations"), {"human", "user"})
        if human is None:
            continue
        value = str(message_value(human) or "")
        if value.strip():
            return value, str(record.get("id", record.get("sample_id", "")))
    raise ValueError(f"No non-empty human prompt found in current template JSONL: {path}")


def image_value(record: dict[str, Any]) -> str:
    value = record.get("image", record.get("images", ""))
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def set_image_value(record: dict[str, Any], value: str) -> None:
    if "image" in record or "images" not in record:
        record["image"] = value
        return
    current = record.get("images")
    record["images"] = [value] if isinstance(current, list) else value


def resolve_legacy_image(record: dict[str, Any], image_root: Path) -> tuple[Path | None, str]:
    raw = image_value(record).replace("\\", "/")
    if not raw:
        return None, "missing_record_path"
    relative = Path(raw)
    direct = relative if relative.is_absolute() else image_root / relative
    if direct.is_file():
        return direct.resolve(), "original_path"

    basename = relative.name
    tile_name = relative.parent.name
    candidates = []
    for split in ("test", "eval", "val", "train"):
        candidate = image_root / "images" / split / tile_name / basename
        if candidate.is_file():
            candidates.append(candidate.resolve())
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0], "alternate_split"
    if len(candidates) > 1:
        return None, "ambiguous_alternate_split"

    image_tree = image_root / "images"
    fallback = list(image_tree.rglob(basename)) if image_tree.is_dir() else []
    fallback = list(dict.fromkeys(path.resolve() for path in fallback if path.is_file()))
    if len(fallback) == 1:
        return fallback[0], "basename_search"
    if len(fallback) > 1:
        return None, "ambiguous_basename"
    return None, "not_found"


def materialize_legacy_image(
    record: dict[str, Any],
    difficulty: str,
    image_root: Path,
    output_dir: Path,
    stats: Counter,
) -> tuple[bool, dict[str, str]]:
    original = image_value(record)
    source, method = resolve_legacy_image(record, image_root)
    stats[f"image_resolution:{method}"] += 1
    if source is None:
        return False, {"sample_id": str(record.get("id", "")), "image": original, "reason": method}

    tile_name = source.parent.name
    destination = output_dir / "images" / difficulty / tile_name / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        shutil.copy2(source, destination)
    relative = destination.relative_to(output_dir).as_posix()
    set_image_value(record, relative)
    stats["images_materialized"] += 1
    if original.replace("\\", "/") != relative:
        stats["image_paths_rewritten"] += 1
    meta = dict(record.get("meta", {}))
    conversion = dict(meta.get("legacy_fixed_eval_conversion", {}))
    conversion.update({
        "original_image": original,
        "resolved_source_image": str(source),
        "materialized_image": relative,
        "image_resolution_method": method,
    })
    meta["legacy_fixed_eval_conversion"] = conversion
    record["meta"] = meta
    return True, {}


def parse_target_payload(raw_value: Any, sample_id: str) -> dict[str, Any]:
    if isinstance(raw_value, str):
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Legacy assistant target is not valid JSON for {sample_id}: {exc}") from exc
    else:
        payload = copy.deepcopy(raw_value)
    if isinstance(payload, list):
        payload = {"lines": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        raise ValueError(f"Legacy assistant target must contain a lines list for {sample_id}")
    return payload


def validate_norm1000(record: dict[str, Any], payload: dict[str, Any], sample_id: str) -> None:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    coord_mode = str(meta.get("coord_mode", meta.get("coord_system", ""))).lower()
    if coord_mode and "norm1000" not in coord_mode:
        raise ValueError(f"Legacy record is not norm1000 for {sample_id}: coord_mode={coord_mode!r}")
    for line_index, item in enumerate(payload["lines"]):
        if not isinstance(item, dict):
            raise ValueError(f"Non-object line at {sample_id}:lines[{line_index}]")
        points = item.get("points")
        if not isinstance(points, list):
            continue
        for point_index, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(f"Invalid point at {sample_id}:lines[{line_index}].points[{point_index}]")
            x, y = point[:2]
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 0 <= x <= 1000 and 0 <= y <= 1000):
                raise ValueError(f"Coordinate outside norm1000 at {sample_id}: {(x, y)}")


def convert_record(
    source: dict[str, Any],
    difficulty: str,
    prompt: str,
    args: argparse.Namespace,
    stats: Counter,
) -> dict[str, Any]:
    record = copy.deepcopy(source)
    sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
    if not sample_id:
        raise ValueError("Legacy record has no id/sample_id")
    conversations = record.get("conversations")
    human = find_message(conversations, {"human", "user"})
    assistant = find_message(conversations, {"gpt", "assistant"}, reverse=True)
    if human is None or assistant is None:
        raise ValueError(f"Legacy conversations are incomplete for {sample_id}")
    set_message_value(human, prompt)

    payload = parse_target_payload(message_value(assistant), sample_id)
    if args.require_norm1000:
        validate_norm1000(record, payload, sample_id)
    for item in payload["lines"]:
        if not isinstance(item, dict):
            raise ValueError(f"Non-object geometry item for {sample_id}")
        category = str(item.get("category", "")).strip().lower()
        if category in {"centerline", "lane", "lane_centerline"}:
            if str(item.get("lane_type", "")).strip():
                stats["lane_type_preserved"] += 1
            else:
                item.pop("lane_type", None)
                stats["lane_type_missing_skipped_by_eval"] += 1
        elif category == "intersection":
            if str(item.get("intersection_type", "")).strip():
                stats["intersection_type_preserved"] += 1
            else:
                item.pop("intersection_type", None)
                stats["intersection_type_missing_skipped_by_eval"] += 1
    set_message_value(assistant, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    meta = dict(record.get("meta", {}))
    meta["legacy_fixed_eval_conversion"] = {
        "difficulty": difficulty,
        "prompt_source": str(args.prompt_template_jsonl),
        "geometry_source": "legacy_reference",
        "image_source": "legacy_reference",
        "missing_semantic_type_policy": "preserve_missing_and_skip_type_evaluation",
    }
    record["meta"] = meta
    stats["records"] += 1
    stats[f"difficulty:{difficulty}"] += 1
    return record


def main() -> None:
    args = parse_args()
    reference_dir = Path(args.reference_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if str(args.prompt_template_jsonl).strip():
        prompt_path = Path(args.prompt_template_jsonl).resolve()
        prompt, prompt_sample_id = load_prompt_template(prompt_path)
        prompt_source = str(prompt_path)
    else:
        prompt_path = None
        prompt = DATASET_V2_LOCAL256_PROMPT
        prompt_sample_id = "<built-in>"
        prompt_source = "built-in:dataset_v2_local256_550k_v1"
    args.prompt_template_jsonl = prompt_source
    expected_counts = parse_expected_counts(args.expected_counts)
    image_root = Path(args.image_source_root).resolve() if str(args.image_source_root).strip() else None
    if args.materialize_images != "none" and image_root is None:
        raise ValueError("--image-source-root is required with --materialize-images")
    if image_root is not None and not (image_root / "images").is_dir():
        raise FileNotFoundError(f"Legacy image root is missing its images directory: {image_root}")
    stats: Counter = Counter()
    all_records: list[dict[str, Any]] = []
    seen_ids = set()
    missing_images: list[dict[str, str]] = []

    for difficulty in args.difficulties:
        input_path = reference_dir / f"{difficulty}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"Legacy fixed split not found: {input_path}")
        converted = []
        for source in read_jsonl(input_path):
            record = convert_record(source, difficulty, prompt, args, stats)
            sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID across fixed splits: {sample_id}")
            seen_ids.add(sample_id)
            if image_root is not None:
                if args.materialize_images == "copy":
                    resolved, detail = materialize_legacy_image(
                        record, difficulty, image_root, output_dir, stats
                    )
                else:
                    source, method = resolve_legacy_image(record, image_root)
                    stats[f"image_resolution:{method}"] += 1
                    resolved = source is not None
                    detail = {} if resolved else {
                        "sample_id": sample_id,
                        "image": image_value(record),
                        "reason": method,
                    }
                if not resolved:
                    missing_images.append(detail)
            converted.append(record)
        expected = expected_counts.get(difficulty)
        if expected is not None and len(converted) != expected:
            raise ValueError(f"Expected {expected} records in {input_path}, found {len(converted)}")
        write_jsonl(output_dir / f"{difficulty}.jsonl", converted)
        all_records.extend(converted)

    if args.require_images and missing_images:
        raise FileNotFoundError(
            f"Unable to resolve {len(missing_images)} legacy images; examples={missing_images[:5]}"
        )

    expected_total = sum(expected_counts.get(name, 0) for name in args.difficulties)
    if expected_total and len(all_records) != expected_total:
        raise ValueError(f"Expected {expected_total} converted records, found {len(all_records)}")
    write_jsonl(output_dir / "all_selected.jsonl", all_records)
    report = {
        "reference_dir": str(reference_dir),
        "output_dir": str(output_dir),
        "prompt_template_jsonl": str(prompt_path) if prompt_path else "",
        "prompt_source": prompt_source,
        "prompt_template_sample_id": prompt_sample_id,
        "prompt": prompt,
        "num_records": len(all_records),
        "expected_counts": expected_counts,
        "stats": dict(sorted(stats.items())),
        "image_source_root": str(image_root) if image_root else "",
        "materialize_images": args.materialize_images,
        "missing_images": missing_images,
        "self_contained_images": args.materialize_images == "copy" and not missing_images,
        "geometry_and_images_preserved_from_legacy": True,
        "type_metric_policy": (
            "Missing legacy lane_type/intersection_type fields remain missing and do not "
            "participate in matched semantic-type accuracy. Geometry metrics use every target."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
