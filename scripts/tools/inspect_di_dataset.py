#!/usr/bin/env python3
"""Print a compact structure report for private DI/QA centerline datasets.

The script is read-only. It is intended for quickly sharing enough dataset
schema information to adapt the converter/training pipeline without copying a
large or sensitive dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="Dataset root, or extraction parent when --dataset-dir-name is set.",
    )
    parser.add_argument(
        "--dataset-dir-name",
        default="",
        help="Optional dataset directory under --input-root, e.g. data_line_samples_33w.",
    )
    parser.add_argument("--phase", default="phase_a", help="Phase directory to inspect. Default: phase_a.")
    parser.add_argument("--image-root", default="images", help="Image root under dataset root. Default: images.")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample rows to inspect per JSONL file.")
    parser.add_argument("--tree-depth", type=int, default=3, help="Maximum directory tree depth to print.")
    parser.add_argument("--tree-limit", type=int, default=40, help="Maximum entries printed per directory.")
    parser.add_argument(
        "--string-preview",
        type=int,
        default=80,
        help="Max characters shown for strings. Use 0 to hide string text completely.",
    )
    parser.add_argument("--skip-image-count", action="store_true", help="Skip recursive image counting.")
    parser.add_argument("--show-raw-sample", action="store_true", help="Print compact raw sample JSON snippets.")
    return parser.parse_args()


def resolve_root(input_root: str, dataset_dir_name: str) -> Path:
    root = Path(input_root).expanduser().resolve()
    name = dataset_dir_name.strip().strip("/\\")
    if name:
        root = (root / name).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{num_bytes}B"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def rel_or_str(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def iter_tree(path: Path, *, max_depth: int, entry_limit: int, depth: int = 0) -> Iterator[str]:
    if depth > max_depth:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        yield "  " * depth + "[permission denied]"
        return
    except FileNotFoundError:
        yield "  " * depth + "[missing]"
        return

    shown = entries[:entry_limit]
    hidden = max(0, len(entries) - len(shown))
    for item in shown:
        prefix = "  " * depth
        if item.is_dir():
            yield f"{prefix}{item.name}/"
            if depth + 1 <= max_depth:
                yield from iter_tree(item, max_depth=max_depth, entry_limit=entry_limit, depth=depth + 1)
        else:
            try:
                size = human_size(item.stat().st_size)
            except OSError:
                size = "?"
            yield f"{prefix}{item.name} ({size})"
    if hidden:
        yield "  " * depth + f"... {hidden} more entries"


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as f:
        for _ in f:
            total += 1
    return total


def load_jsonl_samples(path: Path, limit: int) -> Tuple[List[Dict[str, Any]], int, int]:
    samples: List[Dict[str, Any]] = []
    bad_json = 0
    nonempty = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            nonempty += 1
            if len(samples) >= limit:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if isinstance(payload, dict):
                samples.append(payload)
            else:
                samples.append({"__non_dict__": type(payload).__name__, "value": payload})
    return samples, nonempty, bad_json


def short_value(value: Any, string_preview: int) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool({value})"
    if isinstance(value, (int, float)):
        return f"{type(value).__name__}({value})"
    if isinstance(value, str):
        if string_preview <= 0:
            return f"str(len={len(value)})"
        text = value.replace("\n", "\\n").replace("\r", "\\r")
        if len(text) > string_preview:
            text = text[:string_preview] + "..."
        return f"str(len={len(value)}, preview={text!r})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict(keys={keys[:12]}, total_keys={len(keys)})"
    return type(value).__name__


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


def is_xy(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def summarize_points(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, (list, tuple)):
        return {"type": type(raw).__name__, "num_points": 0}
    if raw and all(is_xy(item) for item in raw):
        xs = [float(item[0]) for item in raw]
        ys = [float(item[1]) for item in raw]
        return {
            "type": "xy_list",
            "num_points": len(raw),
            "first_point": list(raw[0][:2]),
            "x_range": [min(xs), max(xs)],
            "y_range": [min(ys), max(ys)],
        }
    return {"type": "list", "len": len(raw), "first_item": short_value(raw[0], 40) if raw else "empty"}


def summarize_line_payload(payload: Any) -> Dict[str, Any]:
    payload = parse_json_like(payload)
    if isinstance(payload, dict):
        for key in ("lines", "target_lines", "gt_lines", "lines_gt", "centerlines", "centerline"):
            value = payload.get(key)
            if isinstance(value, list):
                return summarize_lines(value)
        for key in ("points", "point", "polyline", "polylines"):
            if key in payload:
                return {"format": f"dict.{key}", "lines": [summarize_points(payload.get(key))]}
        return {"format": "dict", "keys": list(payload.keys())[:20]}
    if isinstance(payload, list):
        if payload and all(is_xy(item) for item in payload):
            return {"format": "point_list", "lines": [summarize_points(payload)]}
        return summarize_lines(payload)
    return {"format": type(payload).__name__}


def summarize_lines(lines: Sequence[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"format": "lines_list", "num_lines": len(lines), "first_lines": []}
    for item in list(lines)[:3]:
        if isinstance(item, dict):
            points = item.get("points", item.get("point", item.get("polyline")))
            out["first_lines"].append(
                {
                    "keys": list(item.keys())[:12],
                    "category": item.get("category", item.get("type")),
                    "points": summarize_points(points),
                }
            )
        else:
            out["first_lines"].append({"points": summarize_points(item)})
    return out


def get_record_id(record: Dict[str, Any], fallback: int) -> str:
    return str(record.get("id", record.get("sample_id", fallback)))


def get_record_images(record: Dict[str, Any]) -> List[str]:
    raw = record.get("images", record.get("image", record.get("Image", [])))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def get_conversations(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("conversations", record.get("Conversations", record.get("messages", [])))
    return raw if isinstance(raw, list) else []


def role_from(turn: Dict[str, Any]) -> str:
    return str(turn.get("from", turn.get("role", ""))).lower()


def content_from(turn: Dict[str, Any]) -> Any:
    return turn.get("value", turn.get("content", ""))


def resolve_image_path(dataset_root: Path, image_root: str, split: str, image: str) -> Tuple[str, Path, bool]:
    normalized = str(image).replace("\\", "/").strip().lstrip("/")
    raw_path = Path(normalized)
    if raw_path.is_absolute():
        return normalized, raw_path, raw_path.is_file()

    root = image_root.strip().strip("/\\")
    split = split.strip().strip("/\\")
    candidates = [normalized]
    if root:
        candidates.append(f"{root}/{normalized}")
        if split:
            candidates.append(f"{root}/{split}/{normalized}")
            if normalized.startswith(f"{split}/"):
                candidates.append(f"{root}/{normalized}")
            candidates.append(f"{root}/{split}/{Path(normalized).name}")

    seen = set()
    for rel in candidates:
        rel = rel.replace("\\", "/").lstrip("/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        path = dataset_root / rel
        if path.is_file():
            return rel, path, True
    rel = candidates[0].replace("\\", "/").lstrip("/")
    return rel, dataset_root / rel, False


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with path.open("rb") as f:
            header = f.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker_start = f.read(1)
                    if not marker_start:
                        break
                    if marker_start != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                        length = struct.unpack(">H", f.read(2))[0]
                        data = f.read(length - 2)
                        height, width = struct.unpack(">HH", data[1:5])
                        return int(width), int(height)
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    length_bytes = f.read(2)
                    if len(length_bytes) != 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]
                    f.seek(length - 2, os.SEEK_CUR)
    except OSError:
        return None
    return None


def inspect_jsonl_file(
    *,
    dataset_root: Path,
    path: Path,
    image_root: str,
    split: str,
    samples: int,
    string_preview: int,
    show_raw_sample: bool,
) -> None:
    rel_path = rel_or_str(path, dataset_root)
    print_header(f"JSONL: {rel_path}")
    if not path.is_file():
        print("missing")
        return

    size = path.stat().st_size
    rows, nonempty, bad_json = load_jsonl_samples(path, samples)
    print(f"path: {path}")
    print(f"size: {human_size(size)}")
    print(f"nonempty_lines: {nonempty}")
    print(f"bad_json_in_sample_window: {bad_json}")
    if not rows:
        print("samples: none")
        return

    key_counter: Counter[str] = Counter()
    type_by_key: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for key, value in row.items():
            key_counter[key] += 1
            type_by_key[key][type(value).__name__] += 1
    print("sampled_keys:")
    for key, count in key_counter.most_common():
        print(f"  - {key}: seen={count}, types={dict(type_by_key[key])}")

    for idx, row in enumerate(rows):
        sample_id = get_record_id(row, idx)
        print()
        print(f"sample[{idx}] id={sample_id}")
        print(f"  keys={list(row.keys())}")
        images = get_record_images(row)
        print(f"  images={images[:3]} total={len(images)}")
        if images:
            rel, img_path, exists = resolve_image_path(dataset_root, image_root, split, images[0])
            dims = image_size(img_path) if exists else None
            size_text = human_size(img_path.stat().st_size) if exists else "missing"
            print(f"  first_image_resolved={rel}")
            print(f"  first_image_exists={exists}, size={size_text}, dims={dims}")

        convs = get_conversations(row)
        if convs:
            roles = [role_from(turn) for turn in convs if isinstance(turn, dict)]
            print(f"  conversations_len={len(convs)}, roles={roles}")
            for turn_idx, turn in enumerate(convs[:4]):
                if not isinstance(turn, dict):
                    print(f"    turn[{turn_idx}] non_dict={type(turn).__name__}")
                    continue
                role = role_from(turn)
                content = content_from(turn)
                print(f"    turn[{turn_idx}] role={role}, content={short_value(content, string_preview)}")
                if role in {"gpt", "assistant"}:
                    print(f"      assistant_payload={json.dumps(summarize_line_payload(content), ensure_ascii=False)}")
        else:
            print("  conversations/messages: none")

        for key in ("target_lines", "gt_lines", "lines_gt", "lines", "point", "points", "centerline", "centerlines"):
            if key in row:
                print(f"  {key}_summary={json.dumps(summarize_line_payload({key: row[key]}), ensure_ascii=False)}")

        if show_raw_sample:
            raw = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            if len(raw) > 1200:
                raw = raw[:1200] + "..."
            print(f"  raw={raw}")


def inspect_json_file(path: Path, root: Path, string_preview: int) -> None:
    rel_path = rel_or_str(path, root)
    print_header(f"JSON: {rel_path}")
    if not path.is_file():
        print("missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"parse_error: {type(exc).__name__}: {exc}")
        return
    print(f"size: {human_size(path.stat().st_size)}")
    print(f"summary: {short_value(payload, string_preview)}")
    if isinstance(payload, dict):
        for key in list(payload.keys())[:30]:
            print(f"  {key}: {short_value(payload[key], string_preview)}")
    elif isinstance(payload, list):
        print(f"  len: {len(payload)}")
        if payload:
            print(f"  first: {short_value(payload[0], string_preview)}")


def count_images(path: Path) -> Dict[str, Any]:
    total = 0
    ext_counts: Counter[str] = Counter()
    sample_paths: List[str] = []
    if not path.is_dir():
        return {"exists": False, "total": 0, "extensions": {}, "samples": []}
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        ext = item.suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        total += 1
        ext_counts[ext] += 1
        if len(sample_paths) < 5:
            try:
                sample_paths.append(item.relative_to(path).as_posix())
            except ValueError:
                sample_paths.append(str(item))
    return {"exists": True, "total": total, "extensions": dict(ext_counts), "samples": sample_paths}


def main() -> None:
    args = parse_args()
    root = resolve_root(args.input_root, args.dataset_dir_name)
    phase = args.phase.strip().strip("/\\")
    image_root = args.image_root.strip().strip("/\\")

    print_header("DATASET INSPECTION REPORT")
    print(f"resolved_root: {root}")
    print(f"dataset_dir_name: {args.dataset_dir_name!r}")
    print(f"phase: {phase!r}")
    print(f"image_root: {image_root!r}")
    print(f"samples_per_jsonl: {args.samples}")

    print_header("DIRECTORY TREE")
    for line in iter_tree(root, max_depth=args.tree_depth, entry_limit=args.tree_limit):
        print(line)

    print_header("KNOWN PATHS")
    known_paths = [
        "dataset_info.json",
        "split_manifest.json",
        f"{phase}/train.jsonl" if phase else "train.jsonl",
        f"{phase}/eval.jsonl" if phase else "eval.jsonl",
        f"{phase}/test.jsonl" if phase else "test.jsonl",
        f"{phase}/meta_train.jsonl" if phase else "meta_train.jsonl",
        f"{phase}/meta_eval.jsonl" if phase else "meta_eval.jsonl",
        f"{phase}/meta_test.jsonl" if phase else "meta_test.jsonl",
        image_root,
        f"{image_root}/train",
        f"{image_root}/eval",
        f"{image_root}/test",
    ]
    for rel in known_paths:
        path = root / rel
        if path.exists():
            kind = "dir" if path.is_dir() else "file"
            size = "" if path.is_dir() else f", size={human_size(path.stat().st_size)}"
            print(f"[OK] {rel} ({kind}{size})")
        else:
            print(f"[MISS] {rel}")

    if not args.skip_image_count:
        print_header("IMAGE COUNTS")
        for split in ("train", "eval", "test"):
            path = root / image_root / split
            print(f"{image_root}/{split}: {json.dumps(count_images(path), ensure_ascii=False)}")

    for rel in ("dataset_info.json", "split_manifest.json"):
        inspect_json_file(root / rel, root, args.string_preview)

    jsonl_targets = [
        (f"{phase}/train.jsonl" if phase else "train.jsonl", "train"),
        (f"{phase}/eval.jsonl" if phase else "eval.jsonl", "eval"),
        (f"{phase}/test.jsonl" if phase else "test.jsonl", "test"),
        (f"{phase}/meta_train.jsonl" if phase else "meta_train.jsonl", "train"),
        (f"{phase}/meta_eval.jsonl" if phase else "meta_eval.jsonl", "eval"),
        (f"{phase}/meta_test.jsonl" if phase else "meta_test.jsonl", "test"),
    ]
    for rel, split in jsonl_targets:
        inspect_jsonl_file(
            dataset_root=root,
            path=root / rel,
            image_root=image_root,
            split=split,
            samples=max(0, args.samples),
            string_preview=args.string_preview,
            show_raw_sample=bool(args.show_raw_sample),
        )

    print_header("DONE")
    print("Copy this report back to the assistant. If prompts are sensitive, rerun with --string-preview 0.")


if __name__ == "__main__":
    main()
