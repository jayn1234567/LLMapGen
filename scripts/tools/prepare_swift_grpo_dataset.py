#!/usr/bin/env python3
"""Convert UniMapGen Stage-A records to an ms-swift GRPO prompt dataset.

The converter deliberately keeps the assistant prediction out of the training
target.  A Stage-A inference JSONL is useful here because it already contains
the exact prompt, image path, coordinate contract, and ground truth, but the
``prediction`` fields are audit data only and must never become ``solution``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from mllm.coord_utils import record_coord_config


IMAGE_MARKER = "<image>"
USER_START = "<|im_start|>user\n"
TURN_END = "<|im_end|>"


def _read_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a JSONL file or a JSON array."""
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        for raw in handle:
            if raw.strip():
                first = raw.lstrip()
                break
        if not first:
            return
        if first.startswith("["):
            text = first + handle.read()
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for row in payload:
                if not isinstance(row, dict):
                    raise ValueError(f"Every record in {path} must be an object")
                yield row
            return

        yield json.loads(first)
        for line_number, raw in enumerate(handle, start=2):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"Record {line_number} in {path} must be an object")
            yield row


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("from") or "").strip().lower()


def _message_content(message: dict[str, Any]) -> str:
    value = message.get("content")
    if value is None:
        value = message.get("value")
    if isinstance(value, list):
        # Swift accepts text plus an explicit images column.  Keep text parts
        # here and let the image column carry the actual pixels.
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
                if part.get("type") == "image" or "image" in part:
                    parts.append(IMAGE_MARKER)
        return "".join(parts)
    return _as_text(value)


def _extract_chatml_user(text: str) -> str:
    """Extract the last ChatML user body, leaving plain prompts untouched."""
    text = str(text or "").strip()
    if USER_START not in text:
        return text
    start = text.rfind(USER_START) + len(USER_START)
    body = text[start:]
    end = body.find(TURN_END)
    if end >= 0:
        body = body[:end]
    return body.strip()


def _find_user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and _message_role(message) in {"user", "human"}:
                return _message_content(message)

    conversations = row.get("conversations") or row.get("conversation")
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, dict) and _message_role(message) in {"user", "human"}:
                return _message_content(message)

    for key in ("user_prompt", "prompt_text", "query", "prompt"):
        if row.get(key):
            return _extract_chatml_user(_as_text(row[key]))
    return ""


def _find_ground_truth(row: dict[str, Any]) -> tuple[Any, str]:
    for key in ("ground_truth", "ground_truth_json", "gt_json", "labels", "label"):
        if key in row and row[key] is not None:
            return row[key], key

    conversations = row.get("conversations") or row.get("conversation")
    if isinstance(conversations, list):
        for message in reversed(conversations):
            if isinstance(message, dict) and _message_role(message) in {"assistant", "gpt", "model"}:
                value = message.get("content")
                if value is None:
                    value = message.get("value")
                return value, "conversations.assistant"
    return None, ""


def _find_images(row: dict[str, Any]) -> list[str]:
    value = row.get("images")
    if value is None:
        # Train-set inference outputs keep both an absolute ``image`` path
        # and a portable ``image_relpath``.  Prefer the latter when present so
        # the JSONL can be moved to another host/DI mount.
        value = row.get("image_relpath")
    if value is None:
        value = row.get("image")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                path = item.get("path") or item.get("image") or item.get("url")
                if path:
                    result.append(str(path))
        return result
    if isinstance(value, dict):
        path = value.get("path") or value.get("image") or value.get("url")
        return [str(path)] if path else []
    return []


def _normalise_image_path(path: str, image_root: Path | None) -> tuple[str, Path | None]:
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_absolute():
        resolved = raw.resolve()
    elif image_root is not None:
        resolved = (image_root / raw).resolve()
    else:
        resolved = raw.resolve()

    if image_root is not None:
        root = image_root.resolve()
        try:
            relative = resolved.relative_to(root)
            return relative.as_posix(), resolved
        except ValueError:
            # An absolute path outside ROOT_IMAGE_DIR is still valid for Swift,
            # but retaining it makes the mismatch visible in the summary.
            return str(resolved), resolved
    return (str(resolved) if raw.is_absolute() else raw.as_posix()), resolved


def _coord_config(row: dict[str, Any]) -> dict[str, Any]:
    explicit = row.get("coord_config")
    if isinstance(explicit, str):
        try:
            explicit = json.loads(explicit)
        except json.JSONDecodeError:
            explicit = None
    config = record_coord_config(
        row,
        # Dataset V2 Context512/ROI256 uses norm1000 relative to the 256 ROI.
        default_mode="norm1000",
        default_patch_size=256,
        default_coord_range=1000,
    )
    if isinstance(explicit, dict):
        config.update(explicit)
    # Retain useful ROI/input metadata without allowing arbitrary objects to
    # replace the fields used by the reward matcher.
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for key in ("input_width", "input_height", "roi", "roi_box", "supervision_region"):
        if key in row:
            config[key] = row[key]
        elif key in meta:
            config[key] = meta[key]
    return config


def _jsonable_ground_truth(value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text, text
        # Validate JSON when possible, but preserve the original text so the
        # reward can deliberately assign an invalid/zero score to bad labels.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text, text
        return text, parsed
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")), value
    text = _as_text(value)
    return text, text


def convert_record(row: dict[str, Any], image_root: Path | None, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(row.get("id") or row.get("sample_id") or row.get("record_id") or "").strip()
    if not sample_id:
        raise ValueError("record has no id/sample_id/record_id")

    prompt = _extract_chatml_user(_find_user_prompt(row))
    if IMAGE_MARKER not in prompt:
        prompt = f"{IMAGE_MARKER}\n{prompt}".strip()
    images = _find_images(row)
    if not images:
        raise ValueError(f"{sample_id}: no image/images field")
    converted_images = []
    missing_images = []
    for image in images:
        converted, resolved = _normalise_image_path(image, image_root)
        converted_images.append(converted)
        if resolved is not None and not resolved.is_file():
            missing_images.append(converted)
    if strict and missing_images:
        raise FileNotFoundError(f"{sample_id}: image(s) not found: {missing_images[:3]}")

    gt_value, gt_source = _find_ground_truth(row)
    if gt_value is None:
        raise ValueError(f"{sample_id}: no ground truth field or assistant conversation")
    gt_text, _ = _jsonable_ground_truth(gt_value)
    coord_config = _coord_config(row)
    map_task = str(row.get("map_task") or row.get("task") or "lane_intersection")

    output = {
        "messages": [{"role": "user", "content": prompt}],
        "images": converted_images,
        # Swift's GRPO data contract uses solution as the reference answer.
        "solution": gt_text,
        # Keep explicit copies for custom reward functions and auditability.
        "ground_truth": gt_text,
        "coord_config": coord_config,
        "map_task": map_task,
        "sample_id": sample_id,
    }
    if row.get("difficulty") is not None:
        output["difficulty"] = row["difficulty"]
    if isinstance(row.get("meta"), dict):
        output["source_meta"] = row["meta"]

    audit = {
        "sample_id": sample_id,
        "ground_truth_source": gt_source,
        "num_images": len(converted_images),
        "missing_images": missing_images,
        "prompt_has_image_marker": IMAGE_MARKER in prompt,
        "coord_config": coord_config,
        "map_task": map_task,
        # This is intentionally only a presence flag, never the prediction.
        "prediction_present_in_source": any(
            key in row for key in ("prediction", "prediction_json", "raw_prediction", "pred_json")
        ),
    }
    return output, audit


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_dataset(
    input_jsonl: Path,
    output_jsonl: Path,
    image_root: Path | None,
    limit: int,
    strict: bool,
    source_kind: str,
    summary_json: Path | None,
) -> dict[str, Any]:
    seen: set[str] = set()
    source_records = 0
    output_records = 0
    missing_image_records = 0
    source_rows_with_prediction_audit_flag = 0
    errors: list[dict[str, Any]] = []
    errors_skipped_count = 0

    # A 550K inference JSONL can be several gigabytes.  Keep only the output
    # path and compact counters in memory; write each converted row atomically
    # to a sibling temporary file and publish it after the scan succeeds.
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_jsonl.name}.", suffix=".tmp", dir=str(output_jsonl.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in _read_records(input_jsonl):
                if limit > 0 and output_records >= limit:
                    break
                source_index = source_records
                source_records += 1
                try:
                    result, audit = convert_record(row, image_root=image_root, strict=strict)
                except Exception as exc:
                    if strict:
                        raise
                    errors_skipped_count += 1
                    if len(errors) < 100:
                        errors.append({"source_index": source_index, "error": str(exc)})
                    continue
                sample_id = result["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                seen.add(sample_id)
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_records += 1
                missing_image_records += bool(audit["missing_images"])
                source_rows_with_prediction_audit_flag += bool(audit["prediction_present_in_source"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_jsonl)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

    summary = {
        "input_jsonl": str(input_jsonl.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "image_root": str(image_root.resolve()) if image_root else None,
        "source_kind": source_kind,
        "source_records_scanned": source_records,
        "output_records": output_records,
        "limit": limit,
        "strict": strict,
        "prediction_fields_never_used_as_solution": True,
        "missing_image_records": missing_image_records,
        "source_rows_with_prediction_audit_flag": source_rows_with_prediction_audit_flag,
        "errors_skipped": errors,
        "errors_skipped_count": errors_skipped_count,
        "errors_truncated": errors_skipped_count > len(errors),
        "sample_ids_unique": len(seen) == output_records,
    }
    if summary_json is None:
        summary_json = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    _atomic_write_json(summary_json, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--strict", action="store_true", help="fail on the first malformed/missing record")
    parser.add_argument("--source-kind", choices=("auto", "inference", "dataset"), default="auto")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    summary = build_dataset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        image_root=args.image_root,
        limit=args.limit,
        strict=args.strict,
        source_kind=args.source_kind,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
