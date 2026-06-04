#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_TRAINROOT = (
    "/mingli01/data/outputs/"
    "rc_clean_dataset_xiuzheng_trainroot_patch256_douglas_merge6h22_plus_rot45_135_centerline_rcstyle_20260428"
)
DEFAULT_MEDIA_ROOT = "/mingli01/data/outputs/rc_centerline_dpo_media_patch256_offset_rot45_135_20260528"
DEFAULT_JOB_FAMILY = (
    "stage3_rc_qwen3vl4b_direct_centerline_patch256_douglas_merge6h22_plus_rot45_135_json_"
    "purelora_colorprompt_4gpu_e6_lr1e4_clean_dataset_xiuzheng_20260429"
)
DEFAULT_BEST_CHECKPOINT = (
    "/mingli01/data/outputs/"
    "stage3_rc_qwen3vl4b_direct_centerline_patch256_douglas_merge6h22_plus_rot45_135_json_"
    "purelora_colorprompt_4gpu_e6_lr1e4_clean_dataset_xiuzheng_rcstyle_20260429/"
    "manual_best_step167500_eval_loss0p234529_20260501"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact, real-sample JSON describing the RC centerline SFT training data format."
    )
    parser.add_argument("--trainroot", type=Path, default=Path(DEFAULT_TRAINROOT))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data_samples/rc_centerline_sft_train_samples_20260604.json"),
    )
    parser.add_argument("--media-root", type=str, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--model", type=str, default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--training-job-family", type=str, default=DEFAULT_JOB_FAMILY)
    parser.add_argument("--best-checkpoint", type=str, default=DEFAULT_BEST_CHECKPOINT)
    parser.add_argument(
        "--sample-kinds",
        nargs="+",
        default=["offset_patch256", "rotated_patch256_045deg", "rotated_patch256_135deg"],
        help="Sample kinds to include. The first matching training row is exported for each kind.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in load_jsonl(path))


def sample_kind(row: Dict[str, Any]) -> str:
    image = str((row.get("images") or [""])[0])
    if "patches256_offset/" in image:
        return "offset_patch256"
    if "patches256_rotcrop/" in image and "rot045" in image:
        return "rotated_patch256_045deg"
    if "patches256_rotcrop/" in image and "rot135" in image:
        return "rotated_patch256_135deg"
    if "patches256_rotcrop/" in image:
        return "rotated_patch256_other"
    return "other"


def message_content(row: Dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def select_samples(train_path: Path, wanted_kinds: List[str]) -> "OrderedDict[str, tuple[int, Dict[str, Any]]]":
    wanted = set(wanted_kinds)
    selected: "OrderedDict[str, tuple[int, Dict[str, Any]]]" = OrderedDict()
    for index, row in enumerate(load_jsonl(train_path)):
        kind = sample_kind(row)
        if kind in wanted and kind not in selected:
            selected[kind] = (index, row)
        if len(selected) == len(wanted):
            break
    missing = [kind for kind in wanted_kinds if kind not in selected]
    if missing:
        raise ValueError(f"Could not find requested sample kinds: {missing}")
    return selected


def sample_type_counts(train_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in load_jsonl(train_path):
        kind = sample_kind(row)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def selected_meta(meta_path: Path, sample_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    keep_keys = [
        "id",
        "split",
        "tile_id",
        "city",
        "source_mode",
        "image",
        "centerline_gt",
        "centerline_json",
        "structure_json",
        "patch_row",
        "patch_col",
        "base_patch_box_full4096",
        "coord_system",
        "sampling_mode",
        "resample_step_px",
        "douglas_epsilon_px",
        "target_fragment_merge_enabled",
        "target_merge_endpoint_tol_px",
        "target_merge_heading_tol_deg",
        "num_target_lines_before_merge",
        "num_target_lines_after_merge",
        "num_target_lines",
        "num_target_intersections",
        "line_direction_mode",
        "line_sort_mode",
        "serialization_mode",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for meta in load_jsonl(meta_path):
        sample_id = str(meta.get("id", ""))
        if sample_id in sample_ids:
            out[sample_id] = {key: meta[key] for key in keep_keys if key in meta}
            if len(out) == len(sample_ids):
                break
    return out


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    trainroot = args.trainroot
    train_path = trainroot / "train.jsonl"
    val_path = trainroot / "val.jsonl"
    meta_train_path = trainroot / "meta_train.jsonl"
    for path in (train_path, val_path, meta_train_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = select_samples(train_path, list(args.sample_kinds))
    ids = {str(row.get("id", "")) for _, row in selected.values()}
    meta_by_id = selected_meta(meta_train_path, ids)

    samples: List[Dict[str, Any]] = []
    for kind, (index, row) in selected.items():
        assistant_text = message_content(row, "assistant")
        try:
            assistant_json: Any = json.loads(assistant_text)
        except json.JSONDecodeError:
            assistant_json = None
        samples.append(
            {
                "sample_kind": kind,
                "source_row_index_in_train_jsonl": index,
                "id": row.get("id"),
                "image": (row.get("images") or [""])[0],
                "raw_train_jsonl_record": row,
                "parsed_assistant_json": assistant_json,
                "meta_train_subset": meta_by_id.get(str(row.get("id", "")), {}),
            }
        )

    return {
        "name": args.output_json.stem,
        "purpose": "Concrete examples of the ShareGPT-style samples used by the Qwen3-VL-4B direct RC centerline SFT run.",
        "task": "Given a 256x256 RC-style road-structure patch image, predict road centerlines as patch-local JSON.",
        "sft_run": {
            "model": str(args.model),
            "training_job_family": str(args.training_job_family),
            "best_checkpoint_used_later": str(args.best_checkpoint),
        },
        "source_dataset": {
            "trainroot": str(trainroot),
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "meta_train_jsonl": str(meta_train_path),
            "train_rows": count_jsonl(train_path),
            "val_rows": count_jsonl(val_path),
            "media_root_used_for_training_and_eval": str(args.media_root),
            "sample_type_counts_in_train": sample_type_counts(train_path),
        },
        "format_notes": [
            "Each training row has id, images, and messages fields.",
            "messages contains system, user, and assistant turns in ShareGPT/LLaMAFactory style.",
            "The user turn contains an <image> placeholder; images[0] is the relative path under the media root.",
            "The assistant turn is a JSON string with lines and intersections fields.",
            "Coordinates are patch-local integer pixels in the 0..255 range.",
            "This centerline-only SFT dataset keeps intersections empty; later joint datasets add intersection targets.",
            "Centerline targets use Douglas sampling and relaxed fragment merging before serialization.",
        ],
        "assistant_schema": {
            "lines": [
                {
                    "category": "centerline",
                    "start_type": "cut|start|end",
                    "end_type": "cut|start|end",
                    "points": [[0, 0], [255, 255]],
                }
            ],
            "intersections": [],
        },
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "samples": len(payload["samples"])}, indent=2))


if __name__ == "__main__":
    main()
