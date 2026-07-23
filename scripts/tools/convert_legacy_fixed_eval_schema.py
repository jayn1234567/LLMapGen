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
    stats: Counter = Counter()
    all_records: list[dict[str, Any]] = []
    seen_ids = set()

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
            converted.append(record)
        expected = expected_counts.get(difficulty)
        if expected is not None and len(converted) != expected:
            raise ValueError(f"Expected {expected} records in {input_path}, found {len(converted)}")
        write_jsonl(output_dir / f"{difficulty}.jsonl", converted)
        all_records.extend(converted)

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
