#!/usr/bin/env python3
"""Normalize the type-clean 512 lane/intersection dataset for MLLM SFT."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


HUMAN_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"assistant", "gpt"}
PROMPT_MARKER = "Output taxonomy JSON schema:"
TAXONOMY_VERSION = "typeclean512_lane_intersection_v1"
VALID_INTERSECTION_PAIRS = {
    (1, 1): "common intersection",
    (1, 2): "T-intersection",
    (1, 3): "small untyped intersection",
    (4, 1): "T-shaped lane-change area",
}
TAXONOMY_PROMPT = """Output taxonomy JSON schema:
Return only valid JSON in the form {\"lines\":[...]} with no extra explanation.
For every centerline, include \"lane_type\" with exactly one of: \"common\" for a regular centerline, \"right_turn\" for a right-turn-only centerline, or \"other\" for any remaining lane class. Do not output U-turn reference lines.
For every intersection, include integer \"intersection_type\" and integer \"intersection_subtype\".
The only valid intersection pairs are: [1,1] common intersection; [1,2] T-intersection; [1,3] small untyped intersection; [4,1] T-shaped lane-change area.
Keep all coordinates in the normalized 0-1000 patch-local coordinate system."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Raw extracted dataset root.")
    parser.add_argument("--output-root", required=True, help="Root for normalized JSONL files.")
    parser.add_argument("--phase", default="phase_a")
    parser.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    parser.add_argument("--summary-report", default="")
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def stable_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    return str(value).strip()


def integer_code(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def resolve_dataset_root(input_root: Path, phase: str) -> tuple[Path, str]:
    root = input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if (root / phase / "train.jsonl").is_file():
        return root, "phase"
    if (root / "train.jsonl").is_file():
        return root, "flat"

    phase_candidates = [
        path for path in root.rglob(f"{phase}/train.jsonl") if "__MACOSX" not in path.parts
    ]
    if len(phase_candidates) == 1:
        return phase_candidates[0].parent.parent, "phase"
    flat_candidates = [
        path
        for path in root.rglob("train.jsonl")
        if path.parent.name not in {"phase_a", "phase_b"} and "__MACOSX" not in path.parts
    ]
    if len(flat_candidates) == 1:
        return flat_candidates[0].parent, "flat"
    candidates = phase_candidates + flat_candidates
    preview = "\n".join(str(path) for path in candidates[:20]) or "<none>"
    raise RuntimeError(f"unable to resolve exactly one dataset root below {root}:\n{preview}")


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
                raise ValueError(f"expected a JSON array in {path}")
            for record in payload:
                if not isinstance(record, dict):
                    raise ValueError(f"non-object record in {path}")
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


def semantic_value(line: dict[str, Any], normalized_names: set[str]) -> tuple[bool, Any]:
    for key, value in line.items():
        if normalized_key(key) in normalized_names:
            return True, value
    return False, None


def remove_semantic_keys(line: dict[str, Any], normalized_names: set[str]) -> None:
    for key in list(line):
        if normalized_key(key) in normalized_names:
            del line[key]


def normalize_centerline(
    line: dict[str, Any], stats: dict[str, Any]
) -> dict[str, Any] | None:
    found, source_value = semantic_value(line, {"lanetype", "lanetypecode", "type"})
    source_key = stable_value(source_value) if found else "<missing>"
    stats["source_lane_types"][source_key] += 1

    source_code = integer_code(source_value)
    source_name = str(source_value).strip().lower() if found else ""
    if source_code == 3:
        stats["dropped_u_turn_centerlines"] += 1
        return None
    if source_code == 1 or source_name == "common":
        target_type = "common"
    elif source_code == 2 or source_name in {"right_turn", "rightturn"}:
        target_type = "right_turn"
    else:
        target_type = "other"
        if not found:
            stats["missing_lane_types_mapped_to_other"] += 1

    remove_semantic_keys(line, {"lanetype", "lanetypecode", "type"})
    line["lane_type"] = target_type
    stats["target_lane_types"][target_type] += 1
    return line


def normalize_intersection(
    line: dict[str, Any], sample_id: str, stats: dict[str, Any]
) -> dict[str, Any]:
    has_type, raw_type = semantic_value(line, {"intersectiontype"})
    has_subtype, raw_subtype = semantic_value(line, {"intersectionsubtype"})
    main_type = integer_code(raw_type) if has_type else None
    subtype = integer_code(raw_subtype) if has_subtype else None

    raw_pair = f"{stable_value(raw_type)}|{stable_value(raw_subtype) if has_subtype else '<missing>'}"
    stats["source_intersection_pairs"][raw_pair] += 1

    # The producer omitted subtype=1 from type-4 records, although the source
    # taxonomy defines that class as 4-1. Restore the canonical pair here.
    if main_type == 4 and subtype is None:
        subtype = 1
        stats["restored_type4_subtype1"] += 1

    pair = (main_type, subtype)
    if pair not in VALID_INTERSECTION_PAIRS:
        raise ValueError(
            f"sample={sample_id} has unsupported intersection pair "
            f"{stable_value(raw_type)}|{stable_value(raw_subtype)}"
        )

    remove_semantic_keys(line, {"intersectiontype", "intersectionsubtype"})
    line["intersection_type"] = main_type
    line["intersection_subtype"] = subtype
    stats["target_intersection_pairs"][f"{main_type}|{subtype}"] += 1
    return line


def normalize_payload(payload: Any, sample_id: str, stats: dict[str, Any]) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        lines = payload["lines"]
    elif isinstance(payload, list):
        lines = payload
    else:
        raise ValueError(f"sample={sample_id} assistant target has no lines array")

    normalized_lines = []
    dropped_in_sample = 0
    for item in lines:
        if not isinstance(item, dict):
            raise ValueError(f"sample={sample_id} has a non-object target line")
        line = dict(item)
        category = str(line.get("category", line.get("class", ""))).strip().lower()
        if category == "centerline":
            line = normalize_centerline(line, stats)
            if line is None:
                dropped_in_sample += 1
                continue
        elif category == "intersection":
            line = normalize_intersection(line, sample_id, stats)
        normalized_lines.append(line)

    if dropped_in_sample:
        stats["samples_with_dropped_u_turn_centerlines"] += 1
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["lines"] = normalized_lines
        return payload
    return normalized_lines


def append_taxonomy_prompt(text: str) -> str:
    base = text
    marker_position = base.find(PROMPT_MARKER)
    if marker_position >= 0:
        base = base[:marker_position]
    return f"{base.rstrip()}\n\n{TAXONOMY_PROMPT}"


def normalize_record(record: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(record.get("id", record.get("sample_id", stats["records"] + 1)))
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"sample={sample_id} has no conversations list")

    human_messages = 0
    assistant_messages = 0
    normalized_messages = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError(f"sample={sample_id} contains a non-object conversation message")
        message = dict(message)
        role_key = "from" if "from" in message else "role"
        value_key = "value" if "value" in message else "content"
        role = str(message.get(role_key, "")).strip().lower()
        value = message.get(value_key, "")
        if role in HUMAN_ROLES:
            if not isinstance(value, str):
                raise ValueError(f"sample={sample_id} human prompt is not text")
            message[value_key] = append_taxonomy_prompt(value)
            human_messages += 1
        elif role in ASSISTANT_ROLES:
            if not isinstance(value, str):
                raise ValueError(f"sample={sample_id} assistant target is not text")
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"sample={sample_id} assistant target is invalid JSON: {exc}") from exc
            payload = normalize_payload(payload, sample_id, stats)
            message[value_key] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            assistant_messages += 1
        normalized_messages.append(message)

    if human_messages == 0 or assistant_messages == 0:
        raise ValueError(
            f"sample={sample_id} requires human and assistant messages; "
            f"found human={human_messages}, assistant={assistant_messages}"
        )

    result = dict(record)
    result["conversations"] = normalized_messages
    meta = dict(result.get("meta", {})) if isinstance(result.get("meta"), dict) else {}
    meta["sft_taxonomy_version"] = TAXONOMY_VERSION
    result["meta"] = meta
    stats["records"] += 1
    return result


def empty_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "source_lane_types": Counter(),
        "target_lane_types": Counter(),
        "dropped_u_turn_centerlines": 0,
        "samples_with_dropped_u_turn_centerlines": 0,
        "missing_lane_types_mapped_to_other": 0,
        "source_intersection_pairs": Counter(),
        "target_intersection_pairs": Counter(),
        "restored_type4_subtype1": 0,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def convert_split(
    input_path: Path,
    output_path: Path,
    overwrite: bool,
    progress_every: int,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    stats = empty_stats()
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in iter_records(input_path):
                normalized = normalize_record(record, stats)
                handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                if progress_every > 0 and stats["records"] % progress_every == 0:
                    print(
                        f"[typeclean-prepare] {input_path.name}: "
                        f"converted {stats['records']} samples",
                        flush=True,
                    )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    result = json_ready(stats)
    result["input_path"] = str(input_path)
    result["output_path"] = str(output_path)
    result["output_size_mb"] = round(output_path.stat().st_size / 1024 / 1024, 3)
    return result


def convert_dataset(
    input_root: Path,
    output_root: Path,
    phase: str,
    splits: list[str],
    overwrite: bool,
    progress_every: int,
) -> dict[str, Any]:
    dataset_root, layout = resolve_dataset_root(input_root, phase)
    output_root = output_root.expanduser().resolve()
    if output_root == dataset_root:
        raise ValueError("output root must differ from the input dataset root")
    output_base = output_root / phase if layout == "phase" else output_root
    report: dict[str, Any] = {
        "taxonomy_version": TAXONOMY_VERSION,
        "input_root": str(dataset_root),
        "output_root": str(output_root),
        "layout": layout,
        "phase": phase,
        "lane_policy": {
            "1": "common",
            "2": "right_turn",
            "3": "drop_u_turn_reference_line",
            "missing_or_any_other": "other",
        },
        "intersection_policy": {
            f"{main}|{sub}": name
            for (main, sub), name in VALID_INTERSECTION_PAIRS.items()
        },
        "splits": {},
    }
    for split in splits:
        input_path = resolve_split_path(dataset_root, phase, layout, split)
        if input_path is None:
            if split in {"train", "eval"}:
                raise FileNotFoundError(f"required split is missing: {split}")
            print(f"[typeclean-prepare] optional split not found: {split}", flush=True)
            continue
        output_path = output_base / f"{split}.jsonl"
        report["splits"][split] = convert_split(
            input_path,
            output_path,
            overwrite,
            progress_every,
        )
    return report


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        Path(args.input_root),
        Path(args.output_root),
        str(args.phase),
        list(args.splits),
        bool(args.overwrite),
        int(args.progress_every),
    )
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    if args.summary_report:
        report_path = Path(args.summary_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"{report_json}\n", encoding="utf-8")
        print(f"[typeclean-prepare] summary: {report_path}", flush=True)
    print(report_json, flush=True)


if __name__ == "__main__":
    main()
