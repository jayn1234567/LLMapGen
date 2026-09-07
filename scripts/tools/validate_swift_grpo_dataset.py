#!/usr/bin/env python3
"""Fail-closed validation for the ms-swift UniMapGen GRPO JSONL contract.

The validator is intentionally independent of ms-swift.  It can therefore be
run in the DI image before Swift starts, and it reports the first few bad rows
without loading a model or keeping a large dataset in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from mllm.coord_utils import record_coord_config
from mllm.reward.map_schema import parse_map_json


def _records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            yield line_number, row


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages")
    if not isinstance(value, list):
        raise ValueError("messages must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("messages must contain objects")
    return value


def _image_paths(row: dict[str, Any]) -> list[str]:
    value = row.get("images")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("images must be a non-empty string list")


def _resolve_image(image: str, image_root: Path) -> Path:
    raw = Path(os.path.expanduser(image))
    candidate = raw if raw.is_absolute() else image_root / raw
    return candidate.resolve()


def _solution_text(row: dict[str, Any], coord_config: dict[str, Any]) -> str:
    value = row.get("solution")
    if value is None:
        value = row.get("ground_truth")
    if value is None:
        raise ValueError("solution/ground_truth is missing")
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if not text:
        raise ValueError("solution/ground_truth is empty")
    parsed = parse_map_json(
        text,
        map_task=str(row.get("map_task") or row.get("task") or "lane_intersection"),
        patch_size=int(coord_config.get("patch_size", 256)),
        coord_mode=str(coord_config.get("coord_mode", "norm1000")),
        coord_range=int(coord_config.get("coord_range", 1000)),
    )
    if not parsed.ok:
        raise ValueError(f"solution is not parseable map JSON: {parsed.error}")
    return text


def _validate_coord_config(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("coord_config")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"coord_config is not JSON: {exc}") from exc
    if value is None:
        value = record_coord_config(
            row,
            default_mode="norm1000",
            default_patch_size=256,
            default_coord_range=1000,
        )
    if not isinstance(value, dict):
        raise ValueError("coord_config must be an object")
    mode = str(value.get("coord_mode", value.get("mode", "norm1000"))).lower()
    if mode not in {"norm1000", "pixel", "normalized", "norm"}:
        raise ValueError(f"unsupported coord_mode: {mode}")
    patch_size = value.get("patch_size", value.get("pixel_patch_size", 256))
    try:
        patch_size = int(patch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid patch_size: {patch_size!r}") from exc
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    coord_range = value.get("coord_range", 1000)
    try:
        coord_range = int(coord_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid coord_range: {coord_range!r}") from exc
    if coord_range <= 0:
        raise ValueError(f"coord_range must be positive, got {coord_range}")
    return dict(value, coord_mode=mode, patch_size=patch_size, coord_range=coord_range)


def validate_dataset(
    dataset_jsonl: Path,
    image_root: Path,
    *,
    expected_images: int,
    limit: int,
    strict: bool,
    summary_json: Path | None,
) -> dict[str, Any]:
    if not dataset_jsonl.is_file():
        raise FileNotFoundError(f"dataset JSONL not found: {dataset_jsonl}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root not found: {image_root}")

    seen: set[str] = set()
    errors: list[dict[str, Any]] = []
    duplicate_count = 0
    checked = 0
    image_count = 0
    parseable_solutions = 0
    for line_number, row in _records(dataset_jsonl):
        if limit > 0 and checked >= limit:
            break
        checked += 1
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        try:
            if not sample_id:
                raise ValueError("sample_id is missing")
            if sample_id in seen:
                duplicate_count += 1
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            messages = _messages(row)
            users = [m for m in messages if str(m.get("role", "")).lower() == "user"]
            if len(users) != 1:
                raise ValueError(f"expected exactly one user message, got {len(users)}")
            content = users[0].get("content", "")
            if not isinstance(content, str) or "<image>" not in content:
                raise ValueError("user message must contain <image>")
            images = _image_paths(row)
            if not images:
                raise ValueError("images is empty")
            if expected_images > 0 and len(images) != expected_images:
                raise ValueError(f"expected {expected_images} image(s), got {len(images)}")
            for image in images:
                path = _resolve_image(image, image_root)
                if not path.is_file():
                    raise FileNotFoundError(f"image not found: {image} -> {path}")
            image_count += len(images)
            coord_config = _validate_coord_config(row)
            _solution_text(row, coord_config)
            parseable_solutions += 1
            if not str(row.get("map_task") or "lane_intersection").strip():
                raise ValueError("map_task is empty")
            forbidden = {"prediction", "prediction_json", "raw_prediction", "pred_json"}
            present = sorted(forbidden.intersection(row))
            if present:
                raise ValueError(f"prediction fields must not be in Swift rows: {present}")
        except Exception as exc:  # Keep scanning so the report is actionable.
            if len(errors) < 100:
                errors.append({"line": line_number, "sample_id": sample_id, "error": str(exc)})
            if strict:
                break

    summary = {
        "dataset_jsonl": str(dataset_jsonl.resolve()),
        "image_root": str(image_root.resolve()),
        "records_checked": checked,
        "sample_ids_unique": duplicate_count == 0,
        "images_checked": image_count,
        "parseable_solutions": parseable_solutions,
        "expected_images_per_row": expected_images,
        "limit": limit,
        "strict": strict,
        "error_count": len(errors),
        "errors": errors,
        "status": "ok" if not errors else "failed",
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"Swift GRPO dataset validation failed with {len(errors)} error(s)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--expected-images", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--strict", action="store_true", help="stop after the first invalid row")
    parser.add_argument("--summary-json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = validate_dataset(
            args.dataset_jsonl.resolve(),
            args.image_root.resolve(),
            expected_images=max(0, args.expected_images),
            limit=max(0, args.limit),
            strict=args.strict,
            summary_json=args.summary_json.resolve() if args.summary_json else None,
        )
    except Exception as exc:
        print(f"[swift-dataset] ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "[swift-dataset] OK "
        f"rows={summary['records_checked']} images={summary['images_checked']} "
        f"solutions={summary['parseable_solutions']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
