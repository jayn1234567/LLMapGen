#!/usr/bin/env python3
"""Visualize empty Dataset V2 samples from completed staging shards or a final dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STAGE_MARKER = "stage_complete.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--staging-root", default="")
    source.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--variant",
        choices=["local256", "context512_roi256"],
        default="local256",
    )
    parser.add_argument("--split", choices=["train", "eval", "test"], default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--thumbnail-width", type=int, default=900)
    return parser.parse_args(argv)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def record_images(record: dict) -> list[str]:
    images = record.get("images")
    if isinstance(images, list):
        return [str(item) for item in images]
    image = record.get("image")
    if isinstance(image, list):
        return [str(item) for item in image]
    if isinstance(image, str) and image:
        return [image]
    raise ValueError(f"sample={record.get('id')} has no image paths")


def target_is_empty(record: dict) -> bool:
    conversations = record.get("conversations") or []
    assistant = next(
        (item for item in conversations if str(item.get("from", "")).lower() in {"gpt", "assistant"}),
        None,
    )
    if not assistant:
        return False
    try:
        payload = json.loads(str(assistant.get("value", "")))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("lines") == []


def reservoir_add(bucket: list[dict], item: dict, seen: int, limit: int, rng: random.Random) -> None:
    if len(bucket) < limit:
        bucket.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        bucket[replacement] = item


def collect_from_staging(
    staging_root: Path,
    variant: str,
    split: str,
    limit: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    completed_roots = sorted(path.parent for path in staging_root.rglob(STAGE_MARKER))
    if not completed_roots:
        raise FileNotFoundError(f"no completed staging shards under: {staging_root}")
    selected: list[dict] = []
    seen_empty = 0
    scanned = 0
    used_sources = []
    for stage_root in completed_roots:
        marker = json.loads((stage_root / STAGE_MARKER).read_text(encoding="utf-8"))
        if variant not in marker.get("variants", []):
            continue
        index_path = stage_root / "records" / f"{split}.index.jsonl"
        record_path = stage_root / "records" / variant / f"{split}.jsonl"
        if not index_path.is_file() or not record_path.is_file():
            continue
        used_sources.append(int(marker.get("source_index", -1)))
        index_iter = iter_jsonl(index_path)
        record_iter = iter_jsonl(record_path)
        for (index_line, index_item), (record_line, record) in zip(index_iter, record_iter):
            if index_line != record_line:
                raise ValueError(f"staging JSONL alignment failed: {stage_root}")
            scanned += 1
            if str(index_item.get("stratum")) != "empty":
                continue
            if not target_is_empty(record):
                raise ValueError(
                    f"index says empty but assistant target is not empty: sample={record.get('id')}"
                )
            seen_empty += 1
            reservoir_add(
                selected,
                {
                    "record": record,
                    "asset_root": stage_root / "variants" / variant,
                    "source_index": int(marker.get("source_index", -1)),
                    "line_number": record_line,
                },
                seen_empty,
                limit,
                rng,
            )
    return selected, {
        "completed_source_indices": sorted(set(used_sources)),
        "scanned_records": scanned,
        "empty_candidates": seen_empty,
    }


def collect_from_dataset(
    dataset_root: Path,
    split: str,
    limit: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    candidates = [
        dataset_root / "phase_a" / f"{split}.jsonl",
        dataset_root / "phasea" / f"{split}.jsonl",
        dataset_root / f"{split}.jsonl",
    ]
    record_path = next((path for path in candidates if path.is_file()), None)
    if record_path is None:
        raise FileNotFoundError(f"cannot find {split}.jsonl under: {dataset_root}")
    selected: list[dict] = []
    seen_empty = 0
    scanned = 0
    for line_number, record in iter_jsonl(record_path):
        scanned += 1
        if not target_is_empty(record):
            continue
        seen_empty += 1
        reservoir_add(
            selected,
            {
                "record": record,
                "asset_root": dataset_root,
                "source_index": (record.get("meta") or {}).get("source_index", -1),
                "line_number": line_number,
            },
            seen_empty,
            limit,
            rng,
        )
    return selected, {
        "completed_source_indices": [],
        "scanned_records": scanned,
        "empty_candidates": seen_empty,
    }


def fit_text(draw: ImageDraw.ImageDraw, text: str, width: int) -> str:
    font = ImageFont.load_default()
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


def render_triplet(item: dict, output_path: Path) -> None:
    record = item["record"]
    image_paths = [item["asset_root"] / relative for relative in record_images(record)]
    if len(image_paths) != 3:
        raise ValueError(f"sample={record.get('id')} expected 3 images, found {len(image_paths)}")
    images = []
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing image for sample={record.get('id')}: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    panel_width = sum(image.width for image in images)
    panel_height = max(image.height for image in images)
    header_height = 44
    canvas = Image.new("RGB", (panel_width, panel_height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"empty | source={item['source_index']} | line={item['line_number']} | "
        f"id={record.get('id')}"
    )
    draw.text((8, 4), fit_text(draw, title, panel_width - 16), fill="black")
    labels = ["clean BEV", "Raw-Lane", "Pose"]
    x = 0
    for image, label in zip(images, labels):
        draw.text((x + 8, 24), label, fill="black")
        canvas.paste(image, (x, header_height))
        x += image.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def make_contact_sheet(paths: list[Path], output_path: Path, columns: int, thumbnail_width: int) -> None:
    if not paths:
        return
    thumbnails = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            height = max(1, round(thumb.height * thumbnail_width / thumb.width))
            thumbnails.append(thumb.resize((thumbnail_width, height), Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumbnail_width, rows * cell_height), "white")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * thumbnail_width, (index // columns) * cell_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.staging_root:
        source_root = Path(args.staging_root).expanduser().resolve()
        selected, scan = collect_from_staging(
            source_root, args.variant, args.split, args.num_samples, rng
        )
        source_mode = "completed_staging"
    else:
        source_root = Path(args.dataset_root).expanduser().resolve()
        selected, scan = collect_from_dataset(source_root, args.split, args.num_samples, rng)
        source_mode = "final_dataset"
    if not selected:
        raise ValueError(f"no empty samples found under: {source_root}")
    selected.sort(key=lambda item: (item["source_index"], item["line_number"]))
    rendered = []
    manifest_path = output_dir / "empty_samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rank, item in enumerate(selected):
            sample_id = str(item["record"].get("id", f"sample_{rank}"))
            safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in sample_id)
            output_path = output_dir / "samples" / f"{rank:03d}_{safe_id}.png"
            render_triplet(item, output_path)
            rendered.append(output_path)
            manifest.write(json.dumps({
                "rank": rank,
                "id": sample_id,
                "source_index": item["source_index"],
                "line_number": item["line_number"],
                "images": record_images(item["record"]),
                "visualization": str(output_path),
            }, ensure_ascii=False) + "\n")
    contact_sheet = output_dir / "empty_contact_sheet.jpg"
    make_contact_sheet(
        rendered,
        contact_sheet,
        max(1, args.contact_sheet_columns),
        max(120, args.thumbnail_width),
    )
    summary = {
        "status": "passed",
        "source_mode": source_mode,
        "source_root": str(source_root),
        "variant": args.variant,
        "split": args.split,
        **scan,
        "rendered_samples": len(rendered),
        "output_dir": str(output_dir),
        "contact_sheet": str(contact_sheet),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
