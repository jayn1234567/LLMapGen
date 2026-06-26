#!/usr/bin/env python3
"""Convert DI QA data into the DINOv2 centerline JSON trainroot format.

Input layout:

    dataset/
      img/<group_id>/*.png
      img/<group_id>/output.json
      train.jsonl
      test.jsonl

Input rows are expected to use:

    {
      "id": "...",
      "images": ["img/<group_id>/<image>.png"],
      "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": {"point": [...], "category": "CenterLine"}}
      ]
    }

Output layout:

    prepared_trainroot/
      train.jsonl
      val.jsonl
      meta_train.jsonl
      meta_val.jsonl
      img -> ../dataset/img   # symlink by default
      dataset_info.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="DI dataset root, or the dataset extract parent when --dataset-dir-name is set.",
    )
    parser.add_argument(
        "--dataset-dir-name",
        default="",
        help="Optional dataset directory name under --input-root after OBS zip extraction.",
    )
    parser.add_argument("--output-root", required=True, help="Output trainroot directory.")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--eval-file", default="test.jsonl", help="Usually test.jsonl in the private dataset.")
    parser.add_argument("--eval-output-name", default="val.jsonl")
    parser.add_argument("--image-root", default="img")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--coord-max", type=int, default=512)
    parser.add_argument("--media-mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--allow-empty-lines", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_records(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input split not found: {path}")
    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise TypeError(f"Expected dict at {path}:{line_no}, got {type(payload)!r}")
                records.append(payload)
                if max_samples > 0 and len(records) >= max_samples:
                    break
        return records

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for key in ("data", "records", "samples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                records = [item for item in value if isinstance(item, dict)]
                break
        if not records:
            records = [payload]
    else:
        raise TypeError(f"Expected list or dict JSON at {path}, got {type(payload)!r}")
    return records[:max_samples] if max_samples > 0 else records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def is_xy(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and is_number(value[0]) and is_number(value[1])


def normalize_xy(value: Sequence[Any], coord_max: int) -> List[int]:
    x = int(round(float(value[0])))
    y = int(round(float(value[1])))
    upper = int(coord_max)
    return [max(0, min(upper, x)), max(0, min(upper, y))]


def normalize_category(value: Any) -> str:
    category = str(value or "centerline").strip().lower()
    category = category.replace("-", "_").replace(" ", "_")
    if category in {"centerline", "center_line", "centerlines", "center_lines", "line"}:
        return "centerline"
    if category in {"centerlane", "center_lane"}:
        return "centerline"
    return category or "centerline"


def parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return text


def points_from_sequence(raw: Any, coord_max: int) -> List[List[int]]:
    if not isinstance(raw, (list, tuple)):
        return []
    points: List[List[int]] = []
    for item in raw:
        if is_xy(item):
            xy = normalize_xy(item, coord_max)
            if not points or points[-1] != xy:
                points.append(xy)
    return points


def lines_from_points_payload(raw: Any, category: str, coord_max: int) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if is_xy(raw):
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    if not raw:
        return []

    if all(is_xy(item) for item in raw):
        points = points_from_sequence(raw, coord_max)
        return [{"category": category, "points": points}] if len(points) >= 2 else []

    lines: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            lines.extend(extract_lines(item, coord_max=coord_max, default_category=category))
            continue
        points = points_from_sequence(item, coord_max)
        if len(points) >= 2:
            lines.append({"category": category, "points": points})
    return lines


def extract_lines(payload: Any, *, coord_max: int, default_category: str = "centerline") -> List[Dict[str, Any]]:
    payload = parse_json_like(payload)
    category = normalize_category(default_category)

    if isinstance(payload, list):
        return lines_from_points_payload(payload, category, coord_max)

    if not isinstance(payload, dict):
        return []

    category = normalize_category(payload.get("category", payload.get("type", default_category)))
    raw_lines = payload.get("lines")
    if isinstance(raw_lines, list):
        lines: List[Dict[str, Any]] = []
        for raw_line in raw_lines:
            if isinstance(raw_line, dict):
                line_category = normalize_category(raw_line.get("category", category))
                raw_points = raw_line.get("points", raw_line.get("point"))
                points = points_from_sequence(raw_points, coord_max)
                if len(points) >= 2:
                    out: Dict[str, Any] = {"category": line_category, "points": points}
                    for key in ("start_type", "end_type"):
                        value = str(raw_line.get(key, "")).strip()
                        if value:
                            out[key] = value
                    lines.append(out)
            else:
                lines.extend(lines_from_points_payload(raw_line, category, coord_max))
        return lines

    for key in ("points", "point", "polyline", "polylines", "centerline", "centerlines"):
        if key in payload:
            return lines_from_points_payload(payload.get(key), category, coord_max)
    return []


def normalize_assistant_payload(value: Any, *, coord_max: int) -> Tuple[str, List[Dict[str, Any]]]:
    lines = extract_lines(value, coord_max=coord_max)
    payload = {"lines": lines}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), lines


def get_images(record: Dict[str, Any]) -> List[str]:
    raw = record.get("images", record.get("image", record.get("Image", [])))
    if isinstance(raw, str):
        images = [raw]
    elif isinstance(raw, list):
        images = [str(item) for item in raw if str(item).strip()]
    else:
        images = []
    return [item.replace("\\", "/").lstrip("/") for item in images if item.strip()]


def get_conversations(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("conversations", record.get("Conversations", record.get("messages", [])))
    return raw if isinstance(raw, list) else []


def role_from(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def convert_record(
    record: Dict[str, Any],
    *,
    index: int,
    input_root: Path,
    coord_max: int,
    patch_size: int,
    allow_missing_images: bool,
    allow_empty_lines: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    sample_id = str(record.get("id", record.get("sample_id", index))).strip() or str(index)
    images = get_images(record)
    stats = {"missing_image": 0, "empty_lines": 0, "missing_assistant": 0}
    if not images:
        raise ValueError(f"sample={sample_id} has no images field.")
    image_path = input_root / images[0]
    if not image_path.is_file():
        stats["missing_image"] = 1
        if not allow_missing_images:
            raise FileNotFoundError(f"sample={sample_id} image not found: {image_path}")

    messages: List[Dict[str, str]] = []
    assistant_value: Any = None
    for turn in get_conversations(record):
        if not isinstance(turn, dict):
            continue
        role = role_from(turn.get("from", turn.get("role")))
        value = turn.get("value", turn.get("content", ""))
        if role == "assistant":
            assistant_value = value
            assistant_json, lines = normalize_assistant_payload(value, coord_max=coord_max)
            messages.append({"role": "assistant", "content": assistant_json})
        elif role in {"system", "user"}:
            content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            messages.append({"role": role, "content": str(content)})

    if assistant_value is None:
        stats["missing_assistant"] = 1
        assistant_json, lines = normalize_assistant_payload(record, coord_max=coord_max)
        messages.append({"role": "assistant", "content": assistant_json})
    else:
        _, lines = normalize_assistant_payload(assistant_value, coord_max=coord_max)

    if not lines:
        stats["empty_lines"] = 1
        if not allow_empty_lines:
            raise ValueError(f"sample={sample_id} has no valid centerline with at least two points.")

    if not any(item.get("role") == "user" for item in messages):
        messages.insert(0, {"role": "user", "content": "Predict the road centerlines for this patch."})

    out_record = {
        "id": sample_id,
        "images": images,
        "messages": messages,
    }
    meta = {
        "id": sample_id,
        "image": images[0],
        "target_lines": lines,
        "num_target_lines": len(lines),
        "patch_size": int(patch_size),
        "source_format": "di_qa",
    }
    return out_record, meta, stats


def prepare_media(input_root: Path, output_root: Path, image_root: str, mode: str) -> str:
    if mode == "none":
        return "none"
    src = input_root / image_root
    dst = output_root / image_root
    if not src.exists():
        raise FileNotFoundError(f"Image root not found: {src}")
    if dst.exists() or dst.is_symlink():
        return "exists"
    output_root.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(src, dst)
        return "copy"
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except OSError:
        shutil.copytree(src, dst)
        return "copy_fallback"


def convert_split(
    *,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    input_root: Path,
    coord_max: int,
    patch_size: int,
    allow_missing_images: bool,
    allow_empty_lines: bool,
    max_samples: int,
    dry_run: bool,
) -> Dict[str, Any]:
    records = load_records(input_path, max_samples=max_samples)
    converted: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    totals = {"missing_image": 0, "empty_lines": 0, "missing_assistant": 0}
    for index, record in enumerate(records):
        out_record, meta, stats = convert_record(
            record,
            index=index,
            input_root=input_root,
            coord_max=coord_max,
            patch_size=patch_size,
            allow_missing_images=allow_missing_images,
            allow_empty_lines=allow_empty_lines,
        )
        converted.append(out_record)
        metas.append(meta)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + int(value)

    if not dry_run:
        write_jsonl(output_path, converted)
        write_jsonl(meta_path, metas)

    preview = converted[0] if converted else {}
    return {
        "input": str(input_path),
        "output": str(output_path),
        "meta": str(meta_path),
        "num_records": len(converted),
        **totals,
        "preview": preview,
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    dataset_dir_name = str(args.dataset_dir_name).strip().strip("/\\")
    if dataset_dir_name:
        input_root = (input_root / dataset_dir_name).resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if input_root == output_root:
        raise ValueError("--input-root and --output-root must be different.")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    train_input = input_root / str(args.train_file)
    eval_input = input_root / str(args.eval_file)
    train_output = output_root / "train.jsonl"
    eval_output = output_root / str(args.eval_output_name)
    if eval_output.name != "val.jsonl":
        print(f"[prepare-di-qa] warning: eval output is {eval_output.name}; train.py --trainroot expects val.jsonl.")
    train_meta = output_root / "meta_train.jsonl"
    eval_meta = output_root / f"meta_{eval_output.stem}.jsonl"

    summary: Dict[str, Any] = {
        "input_root": str(input_root),
        "dataset_dir_name": dataset_dir_name,
        "output_root": str(output_root),
        "image_root": str(args.image_root),
        "patch_size": int(args.patch_size),
        "coord_max": int(args.coord_max),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary["media_mode_result"] = prepare_media(input_root, output_root, str(args.image_root), str(args.media_mode))
    else:
        summary["media_mode_result"] = "dry_run"

    summary["train"] = convert_split(
        input_path=train_input,
        output_path=train_output,
        meta_path=train_meta,
        input_root=input_root,
        coord_max=int(args.coord_max),
        patch_size=int(args.patch_size),
        allow_missing_images=bool(args.allow_missing_images),
        allow_empty_lines=bool(args.allow_empty_lines),
        max_samples=int(args.max_train_samples),
        dry_run=bool(args.dry_run),
    )
    summary["eval"] = convert_split(
        input_path=eval_input,
        output_path=eval_output,
        meta_path=eval_meta,
        input_root=input_root,
        coord_max=int(args.coord_max),
        patch_size=int(args.patch_size),
        allow_missing_images=bool(args.allow_missing_images),
        allow_empty_lines=bool(args.allow_empty_lines),
        max_samples=int(args.max_eval_samples),
        dry_run=bool(args.dry_run),
    )

    if not args.dry_run:
        info_path = output_root / "dataset_info.json"
        info_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["dataset_info"] = str(info_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
