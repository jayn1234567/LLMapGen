#!/usr/bin/env python3
"""Build a centerline-only SFT phase with oracle intersection hints in prompts.

The source phase is expected to contain lane+intersection records. For every
sample, this script moves ground-truth intersection polygons into the user
prompt and keeps only centerline records in the assistant target. The resulting
JSONL remains compatible with the existing ``mllm.train.train_qwen`` dataset
loader: ``image`` plus ``conversations`` with ``human`` and ``gpt`` turns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Dataset root containing the source phase directory.")
    parser.add_argument(
        "--output-root",
        default="",
        help="Dataset root that will receive the output phase. Defaults to --input-root.",
    )
    parser.add_argument("--source-phase", default="phase_a", help="Source phase directory, e.g. phase_a.")
    parser.add_argument(
        "--output-phase",
        default="phase_a_oracle_intersection",
        help="Output phase directory to create.",
    )
    parser.add_argument(
        "--splits",
        default="train,eval,test",
        help="Comma-separated split stems to convert. Missing splits are skipped.",
    )
    parser.add_argument("--coord-max", type=int, default=512, help="Coordinate upper bound described in the prompt.")
    parser.add_argument("--max-records-per-split", type=int, default=0, help="Optional smoke-test cap per split.")
    parser.add_argument("--drop-empty-centerline", action="store_true", help="Drop samples with no centerline target.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output phase.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, max_records: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected object at {path}:{line_no}, got {type(payload)!r}")
            records.append(payload)
            if max_records > 0 and len(records) >= max_records:
                break
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def normalize_category(value: Any, default: str = "centerline") -> str:
    category = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
    if category in {"centerline", "center_line", "centerlines", "center_lines", "centerlane", "center_lane", "line"}:
        return "centerline"
    if category in {"intersection", "junction", "road_intersection", "crossing_region"}:
        return "intersection"
    return category or default


def clean_points(value: Any) -> List[List[int]]:
    points: List[List[int]] = []
    if not isinstance(value, list):
        return points
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xy = [int(round(float(point[0]))), int(round(float(point[1])))]
        except (TypeError, ValueError):
            continue
        if not points or points[-1] != xy:
            points.append(xy)
    return points


def clean_line(raw_line: Dict[str, Any], *, default_category: str) -> Dict[str, Any] | None:
    category = normalize_category(raw_line.get("category", raw_line.get("type", default_category)), default_category)
    points = clean_points(raw_line.get("points", raw_line.get("point", [])))
    if category == "intersection" and len(points) >= 3 and points[0] != points[-1]:
        points.append(list(points[0]))
    min_points = 3 if category == "intersection" else 2
    if len(points) < min_points:
        return None

    output: Dict[str, Any] = {"category": category}
    if category == "centerline":
        start_type = str(raw_line.get("start_type", "")).strip()
        end_type = str(raw_line.get("end_type", "")).strip()
        if start_type:
            output["start_type"] = start_type
        if end_type:
            output["end_type"] = end_type
    else:
        is_cut = raw_line.get("is_cut")
        if isinstance(is_cut, bool):
            output["is_cut"] = is_cut
        raw_type = str(raw_line.get("intersection_type", raw_line.get("type", ""))).strip()
        if raw_type and normalize_category(raw_type, "intersection") != "intersection":
            output["type"] = raw_type
    output["points"] = points
    return output


def extract_conversations(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        return conversations
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    converted: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            role = "human"
        elif role == "assistant":
            role = "gpt"
        converted.append({"from": role, "value": message.get("content", "")})
    return converted


def assistant_payload(record: Dict[str, Any]) -> Any:
    for turn in extract_conversations(record):
        role = str(turn.get("from", turn.get("role", ""))).strip().lower()
        if role in {"gpt", "assistant"}:
            return parse_json_like(turn.get("value", turn.get("content", "")))
    if isinstance(record.get("target_lines"), list):
        return {"lines": record["target_lines"]}
    return {}


def split_payload_lines(payload: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    centerlines: List[Dict[str, Any]] = []
    intersections: List[Dict[str, Any]] = []

    def add(raw: Any, default_category: str) -> None:
        if not isinstance(raw, dict):
            return
        item = clean_line(raw, default_category=default_category)
        if item is None:
            return
        if item["category"] == "intersection":
            intersections.append(item)
        elif item["category"] == "centerline":
            centerlines.append(item)

    if isinstance(payload, list):
        for item in payload:
            add(item, "centerline")
        return centerlines, intersections

    if not isinstance(payload, dict):
        return centerlines, intersections

    raw_lines = payload.get("lines", [])
    if isinstance(raw_lines, list):
        for item in raw_lines:
            add(item, "centerline")

    raw_intersections = payload.get("intersections", [])
    if isinstance(raw_intersections, list):
        for item in raw_intersections:
            if isinstance(item, dict):
                merged = dict(item)
                merged["category"] = "intersection"
                add(merged, "intersection")
    return centerlines, intersections


def image_field(record: Dict[str, Any]) -> str:
    image = str(record.get("image", "")).strip()
    if image:
        return image
    images = record.get("images")
    if isinstance(images, list) and images:
        return str(images[0]).strip()
    return ""


def coord_note(record: Dict[str, Any], coord_max: int) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    coord_mode = str(meta.get("coord_mode", meta.get("coord_system", ""))).lower()
    coord_range = meta.get("coord_range", coord_max)
    patch_size = meta.get("pixel_patch_size", meta.get("patch_size", 512))
    if "norm" in coord_mode:
        return f"Coordinates use a normalized 0-{coord_range} grid over the original {patch_size}x{patch_size} patch."
    return f"Coordinates use patch-local integer coordinates in [0,{coord_max}]."


def centerline_schema(centerlines: Sequence[Dict[str, Any]]) -> str:
    endpoint_values = sorted(
        {
            str(line.get(key)).strip()
            for line in centerlines
            for key in ("start_type", "end_type")
            if str(line.get(key, "")).strip()
        }
    )
    if endpoint_values:
        endpoint_hint = "|".join(endpoint_values)
        return (
            '{"lines":[{"category":"centerline",'
            f'"start_type":"{endpoint_hint}","end_type":"{endpoint_hint}",'
            '"points":[[x,y],[x,y]]}]}'
        )
    return '{"lines":[{"category":"centerline","points":[[x,y],[x,y]]}]}'


def build_prompt(
    record: Dict[str, Any],
    centerlines: Sequence[Dict[str, Any]],
    intersections: Sequence[Dict[str, Any]],
    coord_max: int,
) -> str:
    intersections_json = json.dumps(list(intersections), ensure_ascii=False, separators=(",", ":"))
    schema = (
        '{"lines":[]}\n'
        "or\n"
        f"{centerline_schema(centerlines)}"
    )
    return "\n".join(
        [
            "<image>",
            "Known intersection regions JSON:",
            intersections_json,
            "",
        "The known intersection regions are ground-truth oracle context for this patch.",
        "Predict only the road centerline records inside the current patch.",
        "Do not output intersection records and do not copy the known intersections into the answer.",
        "Match the centerline JSON field set used by the training labels.",
        coord_note(record, coord_max),
        "Return only valid JSON with this schema:",
        schema,
        ]
    )


def convert_record(record: Dict[str, Any], coord_max: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    centerlines, intersections = split_payload_lines(assistant_payload(record))
    sample_id = str(record.get("id", record.get("sample_id", ""))).strip()
    image = image_field(record)
    target_text = json.dumps({"lines": centerlines}, ensure_ascii=False, separators=(",", ":"))
    out_record: Dict[str, Any] = {
        "id": sample_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": build_prompt(record, centerlines, intersections, coord_max)},
            {"from": "gpt", "value": target_text},
        ],
    }
    for key in ("context_image", "global_local_context", "tile_id", "patch_row", "patch_col", "base_patch_box_full"):
        if key in record:
            out_record[key] = record[key]
    meta = dict(record.get("meta", {})) if isinstance(record.get("meta"), dict) else {}
    meta.update(
        {
            "id": sample_id,
            "image": image,
            "task_mode": "oracle_intersection_centerline",
            "oracle_intersections": intersections,
            "oracle_intersection_count": len(intersections),
            "target_lines": centerlines,
            "target_centerline_count": len(centerlines),
        }
    )
    return out_record, meta


def split_names(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def convert_split(
    *,
    source_path: Path,
    output_path: Path,
    meta_output_path: Path,
    coord_max: int,
    max_records: int,
    drop_empty_centerline: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    records = load_jsonl(source_path, max_records=max_records)
    converted: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    dropped = 0
    centerline_count = 0
    intersection_count = 0
    records_with_intersections = 0
    for record in records:
        out_record, meta = convert_record(record, coord_max)
        num_centerlines = int(meta["target_centerline_count"])
        num_intersections = int(meta["oracle_intersection_count"])
        if drop_empty_centerline and num_centerlines <= 0:
            dropped += 1
            continue
        converted.append(out_record)
        metas.append(meta)
        centerline_count += num_centerlines
        intersection_count += num_intersections
        records_with_intersections += int(num_intersections > 0)

    if not dry_run:
        write_jsonl(output_path, converted)
        write_jsonl(meta_output_path, metas)

    return {
        "source": str(source_path),
        "output": str(output_path),
        "meta_output": str(meta_output_path),
        "read_records": len(records),
        "written_records": len(converted),
        "dropped_records": dropped,
        "centerline_count": centerline_count,
        "oracle_intersection_count": intersection_count,
        "records_with_oracle_intersections": records_with_intersections,
        "preview": converted[0] if converted else {},
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else input_root
    source_phase = str(args.source_phase).strip().strip("/\\")
    output_phase = str(args.output_phase).strip().strip("/\\")
    source_dir = input_root / source_phase
    output_dir = output_root / output_phase
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source phase not found: {source_dir}")
    if output_dir.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output phase already exists: {output_dir}. Pass --overwrite to replace files.")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_phase": source_phase,
        "output_phase": output_phase,
        "coord_max": int(args.coord_max),
        "splits": {},
        "dry_run": bool(args.dry_run),
    }
    for split in split_names(args.splits):
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.is_file():
            summary["splits"][split] = {"skipped": True, "reason": f"missing {source_path}"}
            continue
        summary["splits"][split] = convert_split(
            source_path=source_path,
            output_path=output_dir / f"{split}.jsonl",
            meta_output_path=output_dir / f"meta_{split}.jsonl",
            coord_max=int(args.coord_max),
            max_records=int(args.max_records_per_split),
            drop_empty_centerline=bool(args.drop_empty_centerline),
            dry_run=bool(args.dry_run),
        )

    if not args.dry_run:
        (output_dir / "oracle_intersection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
