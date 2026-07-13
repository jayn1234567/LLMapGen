#!/usr/bin/env python3
"""Inspect CapRL-Qwen3VL-4B and the v9 best SFT dataset on Ascend/ModelArts.

The script is intentionally read-only for OBS assets. By default it copies only
small model metadata files, not weight shards. The dataset is a zip file, so it
must be downloaded before JSONL samples can be inspected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_CAPRL_OBS = (
    "obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/"
    "checkpoints/CapRL-Qwen3VL-4B"
)
DEFAULT_DATASET_OBS = (
    "obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/"
    "data_lane_intersection_samples_norm_33w_empty_patch.zip"
)
DEFAULT_DATASET_DIR = "data_lane_intersection_samples_norm_33w_empty_patch"
DEFAULT_WORK_DIR = "/cache/jn/inspect_caprl_v9_best_assets"

WEIGHT_SUFFIXES = (
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".onnx",
)
SMALL_MODEL_SUFFIXES = (
    ".json",
    ".txt",
    ".model",
    ".jinja",
    ".yaml",
    ".yml",
    ".md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caprl-obs-path", default=DEFAULT_CAPRL_OBS)
    parser.add_argument("--dataset-obs-path", default=DEFAULT_DATASET_OBS)
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--sample-lines", type=int, default=3)
    parser.add_argument("--string-preview", type=int, default=240)
    parser.add_argument("--max-model-list", type=int, default=600)
    parser.add_argument("--count-lines", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--download-model-all",
        action="store_true",
        help="Copy the whole CapRL directory, including weight shards. Default only copies metadata.",
    )
    return parser.parse_args()


def import_moxing():
    try:
        import moxing as mox  # type: ignore

        return mox
    except Exception as exc:  # pragma: no cover - runs on Ascend.
        print("ERROR: failed to import moxing.", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        print(
            "Hint: on Ascend/ModelArts, install setuptools if pkg_resources is missing, "
            "then install the pinned moxing_framework wheel used by the training scripts.",
            file=sys.stderr,
        )
        raise


def obs_join(root: str, child: str) -> str:
    return root.rstrip("/") + "/" + child.strip("/")


def normalize_list_item(parent: str, item: str) -> str:
    if item.startswith("obs://"):
        return item.rstrip("/")
    return obs_join(parent, item).rstrip("/")


def rel_from_obs(root: str, path: str) -> str:
    root = root.rstrip("/") + "/"
    if path.startswith(root):
        return path[len(root) :]
    return path.rsplit("/", 1)[-1]


def try_list_dir(mox: Any, path: str) -> list[str] | None:
    try:
        items = mox.file.list_directory(path)
    except Exception:
        return None
    if not items:
        return []
    return [normalize_list_item(path, str(item)) for item in items]


def list_obs_tree(mox: Any, root: str, max_items: int) -> tuple[list[str], list[str]]:
    """Return (dirs, files) under an OBS root using best-effort recursion."""

    dirs: list[str] = []
    files: list[str] = []
    queue = [root.rstrip("/")]
    seen: set[str] = set()
    while queue and len(dirs) + len(files) < max_items:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        children = try_list_dir(mox, current)
        if children is None:
            files.append(current)
            continue
        dirs.append(current)
        for child in children:
            if len(dirs) + len(files) >= max_items:
                break
            if child in seen:
                continue
            if child.lower().endswith(WEIGHT_SUFFIXES):
                files.append(child)
            else:
                nested = try_list_dir(mox, child)
                if nested is None:
                    files.append(child)
                else:
                    dirs.append(child)
                    queue.extend(nested)
                    seen.add(child)
    return dirs, files


def copy_model_metadata(
    mox: Any,
    obs_root: str,
    local_root: Path,
    max_items: int,
    download_all: bool,
    force: bool,
) -> dict[str, Any]:
    if force and local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    if download_all:
        print(f"[model] copy full model dir: {obs_root} -> {local_root}")
        mox.file.copy_parallel(obs_root, str(local_root))
        return {"mode": "full", "local_root": str(local_root)}

    print(f"[model] list OBS model dir: {obs_root}")
    dirs, files = list_obs_tree(mox, obs_root, max_items=max_items)
    copied: list[str] = []
    skipped_weights: list[str] = []
    for obs_file in files:
        rel = rel_from_obs(obs_root, obs_file)
        suffix = Path(rel).suffix.lower()
        if obs_file.lower().endswith(WEIGHT_SUFFIXES):
            skipped_weights.append(rel)
            continue
        if suffix not in SMALL_MODEL_SUFFIXES:
            continue
        local_file = local_root / rel
        local_file.parent.mkdir(parents=True, exist_ok=True)
        if force or not local_file.exists():
            print(f"[model] copy metadata {obs_file} -> {local_file}")
            mox.file.copy(obs_file, str(local_file))
        copied.append(rel)
    return {
        "mode": "metadata",
        "local_root": str(local_root),
        "obs_dirs": [rel_from_obs(obs_root, item) for item in dirs],
        "obs_files": [rel_from_obs(obs_root, item) for item in files],
        "copied_metadata": copied,
        "skipped_weight_files": skipped_weights[:80],
        "skipped_weight_file_count": len(skipped_weights),
    }


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": repr(exc)}


def summarize_config(model_root: Path) -> dict[str, Any]:
    config = read_json(model_root / "config.json")
    tokenizer_config = read_json(model_root / "tokenizer_config.json")
    generation_config = read_json(model_root / "generation_config.json")
    index_files = sorted(str(p.relative_to(model_root)) for p in model_root.rglob("*.index.json"))
    summary: dict[str, Any] = {
        "config_json_exists": isinstance(config, dict),
        "tokenizer_config_exists": isinstance(tokenizer_config, dict),
        "generation_config_exists": isinstance(generation_config, dict),
        "index_files": index_files,
    }
    if isinstance(config, dict):
        for key in (
            "model_type",
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
            "torch_dtype",
            "transformers_version",
        ):
            if key in config:
                summary[key] = config[key]
        for nested_key in ("text_config", "vision_config"):
            nested = config.get(nested_key)
            if isinstance(nested, dict):
                summary[nested_key] = {
                    key: nested.get(key)
                    for key in (
                        "model_type",
                        "architectures",
                        "hidden_size",
                        "num_hidden_layers",
                        "num_attention_heads",
                        "vocab_size",
                        "torch_dtype",
                    )
                    if key in nested
                }
    if isinstance(tokenizer_config, dict):
        summary["tokenizer_class"] = tokenizer_config.get("tokenizer_class")
        summary["chat_template_exists"] = bool(tokenizer_config.get("chat_template"))
        summary["model_max_length"] = tokenizer_config.get("model_max_length")
    return summary


def download_and_extract_dataset(
    mox: Any,
    dataset_obs: str,
    zip_path: Path,
    extract_root: Path,
    force: bool,
) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if force and zip_path.exists():
        zip_path.unlink()
    if force and extract_root.exists():
        shutil.rmtree(extract_root)

    if not zip_path.exists():
        print(f"[dataset] copy zip: {dataset_obs} -> {zip_path}")
        mox.file.copy(dataset_obs, str(zip_path))
    else:
        print(f"[dataset] reuse zip: {zip_path}")

    if not extract_root.exists() or not any(extract_root.iterdir()):
        print(f"[dataset] unzip: {zip_path} -> {extract_root}")
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
    else:
        print(f"[dataset] reuse extracted root: {extract_root}")
    return extract_root


def find_dataset_root(extract_root: Path, dataset_dir_name: str) -> Path:
    direct = extract_root / dataset_dir_name
    if direct.exists():
        return direct
    candidates = []
    for path in extract_root.rglob("phase_a"):
        if (path / "train.jsonl").exists() or (path / "eval.jsonl").exists():
            candidates.append(path.parent)
    if candidates:
        return candidates[0]
    return direct


def compact_string(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if limit <= 0:
        return ""
    if len(value) > limit:
        return value[:limit] + "..."
    return value


def try_parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except Exception:
        pass
    start = min([idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0], default=-1)
    if start >= 0:
        maybe = stripped[start:]
        try:
            return json.loads(maybe)
        except Exception:
            return value
    return value


def collect_xy_pairs(obj: Any, out: list[tuple[float, float]]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            collect_xy_pairs(value, out)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            out.append((float(obj[0]), float(obj[1])))
            return
        for value in obj:
            collect_xy_pairs(value, out)


def extract_conversations(record: dict[str, Any]) -> list[dict[str, Any]]:
    conv = record.get("conversations", record.get("conversation", record.get("messages", [])))
    if isinstance(conv, dict):
        return [conv]
    if isinstance(conv, list):
        return [item for item in conv if isinstance(item, dict)]
    return []


def role_of(message: dict[str, Any]) -> str:
    return str(message.get("from", message.get("role", ""))).lower()


def value_of(message: dict[str, Any]) -> Any:
    return message.get("value", message.get("content", ""))


def resolve_image_path(dataset_root: Path, record: dict[str, Any]) -> Path | None:
    image_value = record.get("images", record.get("image", record.get("img", None)))
    if isinstance(image_value, list):
        image_value = image_value[0] if image_value else None
    if not isinstance(image_value, str) or not image_value:
        return None
    path = Path(image_value)
    if path.is_absolute():
        return path
    return dataset_root / path


def image_size(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def jsonl_samples(path: Path, sample_lines: int) -> list[dict[str, Any]]:
    samples = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except Exception as exc:
                samples.append({"_parse_error": repr(exc), "_raw": line[:500]})
            if len(samples) >= sample_lines:
                break
    return samples


def count_jsonl_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def inspect_jsonl(path: Path, dataset_root: Path, sample_lines: int, preview: int, count_lines: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
    }
    if count_lines:
        info["line_count"] = count_jsonl_lines(path)
    samples = jsonl_samples(path, sample_lines)
    sample_infos = []
    all_points: list[tuple[float, float]] = []
    for idx, record in enumerate(samples):
        conversations = extract_conversations(record) if isinstance(record, dict) else []
        assistant_values = [
            value_of(message)
            for message in conversations
            if role_of(message) in {"gpt", "assistant"}
        ]
        human_values = [
            value_of(message)
            for message in conversations
            if role_of(message) in {"human", "user"}
        ]
        points: list[tuple[float, float]] = []
        for value in assistant_values:
            parsed = try_parse_jsonish(value)
            collect_xy_pairs(parsed, points)
        all_points.extend(points)
        img_path = resolve_image_path(dataset_root, record) if isinstance(record, dict) else None
        sample_infos.append(
            {
                "index": idx,
                "top_keys": sorted(record.keys()) if isinstance(record, dict) else [],
                "id": record.get("id", record.get("sample_id")) if isinstance(record, dict) else None,
                "image_field": record.get("images", record.get("image")) if isinstance(record, dict) else None,
                "image_exists": bool(img_path and img_path.exists()),
                "image_size": image_size(img_path),
                "conversation_roles": [role_of(message) for message in conversations],
                "human_preview": compact_string(str(human_values[0]), preview) if human_values else "",
                "assistant_preview": compact_string(str(assistant_values[0]), preview) if assistant_values else "",
                "assistant_xy_pairs": len(points),
                "assistant_xy_range": coord_range(points),
            }
        )
    info["samples"] = sample_infos
    info["sample_xy_pair_count"] = len(all_points)
    info["sample_xy_range"] = coord_range(all_points)
    return info


def coord_range(points: list[tuple[float, float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }


def inspect_dataset(dataset_root: Path, sample_lines: int, preview: int, count_lines: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.exists(),
    }
    if not dataset_root.exists():
        return info
    top_entries = []
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir():
            top_entries.append(path.name + "/")
        else:
            top_entries.append(f"{path.name} ({round(path.stat().st_size / (1024 * 1024), 3)} MB)")
    info["top_entries"] = top_entries
    for phase in ("phase_a", "phase_b"):
        phase_dir = dataset_root / phase
        phase_info: dict[str, Any] = {"exists": phase_dir.exists()}
        if phase_dir.exists():
            for split in ("train", "eval", "test"):
                phase_info[f"{split}_jsonl"] = inspect_jsonl(
                    phase_dir / f"{split}.jsonl",
                    dataset_root,
                    sample_lines,
                    preview,
                    count_lines,
                )
        info[phase] = phase_info
    return info


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    model_local = work_dir / "checkpoints" / "CapRL-Qwen3VL-4B"
    dataset_zip = work_dir / "data_lane_intersection_samples_norm_33w_empty_patch.zip"
    extract_root = work_dir / "dataset_extract"

    mox = import_moxing()
    report: dict[str, Any] = {
        "caprl_obs_path": args.caprl_obs_path,
        "dataset_obs_path": args.dataset_obs_path,
        "work_dir": str(work_dir),
    }

    print_section("DOWNLOAD / LIST MODEL")
    report["model_copy"] = copy_model_metadata(
        mox,
        args.caprl_obs_path,
        model_local,
        max_items=args.max_model_list,
        download_all=args.download_model_all,
        force=args.force,
    )
    report["model_config_summary"] = summarize_config(model_local)
    print(json.dumps(report["model_config_summary"], ensure_ascii=False, indent=2))

    print_section("DOWNLOAD / EXTRACT DATASET")
    download_and_extract_dataset(mox, args.dataset_obs_path, dataset_zip, extract_root, force=args.force)
    dataset_root = find_dataset_root(extract_root, args.dataset_dir_name)
    report["dataset"] = inspect_dataset(
        dataset_root,
        sample_lines=args.sample_lines,
        preview=args.string_preview,
        count_lines=args.count_lines,
    )

    print_section("MODEL OBS FILES")
    print(json.dumps(report["model_copy"], ensure_ascii=False, indent=2)[:12000])
    print_section("DATASET REPORT")
    print(json.dumps(report["dataset"], ensure_ascii=False, indent=2)[:24000])

    report_path = work_dir / "caprl_v9_best_assets_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_section("DONE")
    print(f"Report saved to: {report_path}")
    print("Copy the MODEL CONFIG SUMMARY and DATASET REPORT back to the assistant.")


if __name__ == "__main__":
    main()
