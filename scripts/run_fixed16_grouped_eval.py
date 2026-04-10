from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_qwen2_5vl_lora_small_eval import (
    append_jsonl,
    generate_with_custom_engine,
    generate_with_llamafactory_engine,
    load_jsonl,
    parse_generated_json,
    sanitize_lines,
    stack_panels,
)
from unimapgen.compare_metrics import _sample_metrics
from unimapgen.paper_metrics import evaluate_prediction_items


def parse_categories_arg(raw: str) -> List[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or ["road"]


def parse_prediction_text(
    text: str,
    prediction_format: str = "json",
    discrete_categories: Optional[List[str]] = None,
    discrete_coord_num_bins: int = 896,
    discrete_image_size: int = 896,
    discrete_token_schema: str = "legacy_xy",
    discrete_include_text_prompt_tokens: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if str(prediction_format).strip().lower() != "json":
        raise NotImplementedError("This minimal StageA fixed16 bundle only supports prediction_format=json.")
    return parse_generated_json(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-16 grouped inference, merge 16 local predictions back to 896x896, and compute merged metrics."
    )
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--meta-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--engine", type=str, default="custom", choices=["custom", "llamafactory"])
    parser.add_argument("--template", type=str, default="qwen2_vl")
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-ids", type=str, default="")
    parser.add_argument("--max-source-patches", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--dynamic-schedule", action="store_true")
    parser.add_argument("--dynamic-chunk-size", type=int, default=0)
    parser.add_argument("--prepare-dynamic-tasks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--meter-per-pixel", type=float, default=0.15)
    parser.add_argument("--line-width-px", type=int, default=6)
    parser.add_argument("--paper-categories", type=str, default="")
    parser.add_argument("--prediction-format", type=str, default="json", choices=["json"])
    parser.add_argument("--discrete-categories", type=str, default="road")
    parser.add_argument("--discrete-coord-num-bins", type=int, default=896)
    parser.add_argument("--discrete-token-schema", type=str, default="legacy_xy", choices=["legacy_xy", "shared_numbers"])
    parser.add_argument("--disable-legacy-text-prompt-tokens", action="store_true")
    parser.add_argument("--user-prompt-style", type=str, default="dataset")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=896)
    parser.add_argument("--merge-endpoint-dist-px", type=float, default=4.0)
    parser.add_argument("--internal-boundary-tol-px", type=float, default=3.0)
    parser.add_argument("--allow-incomplete-groups", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dedup_points(points: Sequence[Sequence[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for point in points:
        cur = [int(point[0]), int(point[1])]
        if not out or out[-1] != cur:
            out.append(cur)
    return out


def line_length(points: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for idx in range(len(points) - 1):
        dx = float(points[idx + 1][0] - points[idx][0])
        dy = float(points[idx + 1][1] - points[idx][1])
        total += math.hypot(dx, dy)
    return total


def parse_source_ids(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_polyline(draw: ImageDraw.ImageDraw, points: Sequence[Sequence[int]], color: Tuple[int, int, int], width: int) -> None:
    pts = [tuple(int(v) for v in point[:2]) for point in points]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)


def boundary_key(point: Sequence[int], patch_size: int, grid_size: int, tol_px: float) -> Optional[Tuple[str, int]]:
    x = float(point[0])
    y = float(point[1])
    step = float(patch_size) / float(grid_size)
    for idx in range(1, int(grid_size)):
        x_edge = step * idx
        if abs(x - x_edge) <= float(tol_px):
            return ("v", idx)
        y_edge = step * idx
        if abs(y - y_edge) <= float(tol_px):
            return ("h", idx)
    return None


def maybe_merge_lines(
    line_a: Dict[str, Any],
    line_b: Dict[str, Any],
    patch_size: int,
    grid_size: int,
    merge_endpoint_dist_px: float,
    internal_boundary_tol_px: float,
) -> Optional[Dict[str, Any]]:
    a_points = [list(map(int, pt[:2])) for pt in line_a.get("points", [])]
    b_points = [list(map(int, pt[:2])) for pt in line_b.get("points", [])]
    if len(a_points) < 2 or len(b_points) < 2:
        return None

    candidates = [
        ("end", "start", a_points, b_points, str(line_a.get("start_type", "start")), str(line_b.get("end_type", "end"))),
        ("end", "end", a_points, list(reversed(b_points)), str(line_a.get("start_type", "start")), str(line_b.get("start_type", "start"))),
        ("start", "start", list(reversed(a_points)), b_points, str(line_a.get("end_type", "end")), str(line_b.get("end_type", "end"))),
        ("start", "end", list(reversed(a_points)), list(reversed(b_points)), str(line_a.get("end_type", "end")), str(line_b.get("start_type", "start"))),
    ]

    for _, _, left_points, right_points, merged_start_type, merged_end_type in candidates:
        left_end = left_points[-1]
        right_start = right_points[0]
        if math.hypot(float(left_end[0] - right_start[0]), float(left_end[1] - right_start[1])) > float(merge_endpoint_dist_px):
            continue
        left_boundary = boundary_key(left_end, patch_size=patch_size, grid_size=grid_size, tol_px=internal_boundary_tol_px)
        right_boundary = boundary_key(right_start, patch_size=patch_size, grid_size=grid_size, tol_px=internal_boundary_tol_px)
        if left_boundary is None or right_boundary is None or left_boundary != right_boundary:
            continue
        merged_points = dedup_points(left_points + right_points[1:])
        if len(merged_points) < 2:
            continue
        return {
            "category": line_a.get("category", line_b.get("category", "road")),
            "start_type": merged_start_type,
            "end_type": merged_end_type,
            "points": merged_points,
        }
    return None


def stitch_lines(
    lines: Sequence[Dict[str, Any]],
    patch_size: int,
    grid_size: int,
    merge_endpoint_dist_px: float,
    internal_boundary_tol_px: float,
) -> List[Dict[str, Any]]:
    active = [
        {
            "category": line.get("category", "road"),
            "start_type": str(line.get("start_type", "start")),
            "end_type": str(line.get("end_type", "end")),
            "points": dedup_points(line.get("points", [])),
        }
        for line in lines
        if len(line.get("points", [])) >= 2
    ]
    changed = True
    while changed:
        changed = False
        next_lines: List[Dict[str, Any]] = []
        used = [False] * len(active)
        for idx, line_a in enumerate(active):
            if used[idx]:
                continue
            merged_any = False
            for jdx in range(idx + 1, len(active)):
                if used[jdx]:
                    continue
                merged = maybe_merge_lines(
                    line_a=line_a,
                    line_b=active[jdx],
                    patch_size=patch_size,
                    grid_size=grid_size,
                    merge_endpoint_dist_px=merge_endpoint_dist_px,
                    internal_boundary_tol_px=internal_boundary_tol_px,
                )
                if merged is None:
                    continue
                used[idx] = True
                used[jdx] = True
                next_lines.append(merged)
                changed = True
                merged_any = True
                break
            if not merged_any and not used[idx]:
                used[idx] = True
                next_lines.append(line_a)
        active = next_lines

    active = [line for line in active if len(line.get("points", [])) >= 2]
    active.sort(key=lambda item: (-line_length(item.get("points", [])), item.get("points", [[10**9, 10**9]])[0][1], item.get("points", [[10**9, 10**9]])[0][0]))
    return active


def build_overlay(
    image: Image.Image,
    lines: Sequence[Dict[str, Any]],
    title: str,
    line_color: Tuple[int, int, int],
    patch_size: int,
    grid_size: int,
) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for idx in range(1, int(grid_size)):
        edge = int(round(float(patch_size) * idx / float(grid_size)))
        draw.line([(edge, 0), (edge, patch_size - 1)], fill=(255, 210, 0), width=1)
        draw.line([(0, edge), (patch_size - 1, edge)], fill=(255, 210, 0), width=1)
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=(255, 0, 180), width=2)
    for line in lines:
        pts = [list(map(int, point[:2])) for point in line.get("points", [])]
        draw_polyline(draw, pts, color=line_color, width=4)
        if pts:
            draw_endpoint(draw, pts[0], line_color, radius=3)
            draw_endpoint(draw, pts[-1], line_color, radius=3)
    text = title[:120]
    box_w = min(patch_size - 8, 12 + 8 * len(text))
    draw.rectangle((8, 8, box_w, 34), fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255))
    return panel


def mean_dict(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def validate_shard_args(args: argparse.Namespace) -> None:
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0:
        raise ValueError("--shard-index must be >= 0")
    if not bool(args.merge_shards) and int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be smaller than --num-shards")
    if bool(args.dynamic_schedule) and int(args.dynamic_chunk_size) <= 0:
        raise ValueError("--dynamic-chunk-size must be >= 1 when --dynamic-schedule is enabled")
    if bool(args.prepare_dynamic_tasks) and not bool(args.dynamic_schedule):
        raise ValueError("--prepare-dynamic-tasks requires --dynamic-schedule")


def shard_tag(shard_index: int, num_shards: int) -> str:
    return f"shard{int(shard_index):02d}of{int(num_shards):02d}"


def shard_output_paths(
    output_dir: Path,
    num_shards: int,
    shard_index: int,
    merge_shards: bool,
) -> Dict[str, Path]:
    if int(num_shards) > 1 and not bool(merge_shards):
        tag = shard_tag(shard_index=shard_index, num_shards=num_shards)
        return {
            "box_predictions": output_dir / f"box_predictions.{tag}.jsonl",
            "merged_predictions": output_dir / f"merged_patch_predictions.{tag}.jsonl",
            "summary": output_dir / f"merged_summary.{tag}.json",
            "manifest": output_dir / f"run_manifest.{tag}.json",
            "viz_dir": output_dir / "viz" / tag,
        }
    return {
        "box_predictions": output_dir / "box_predictions.jsonl",
        "merged_predictions": output_dir / "merged_patch_predictions.jsonl",
        "summary": output_dir / "merged_summary.json",
        "manifest": output_dir / "run_manifest.json",
        "viz_dir": output_dir / "viz",
    }


def select_groups_for_shard(
    groups: Sequence[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]],
    num_shards: int,
    shard_index: int,
) -> List[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]]:
    if int(num_shards) <= 1:
        return list(groups)
    return [item for idx, item in enumerate(groups) if idx % int(num_shards) == int(shard_index)]


def dynamic_task_pool_dir(output_dir: Path, num_shards: int, dynamic_chunk_size: int) -> Path:
    return output_dir / f"task_pool.dynamic_n{int(num_shards):02d}_c{int(dynamic_chunk_size):03d}"


def split_groups_into_tasks(
    groups: Sequence[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]],
    dynamic_chunk_size: int,
) -> List[List[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]]]:
    chunk_size = max(1, int(dynamic_chunk_size))
    tasks: List[List[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]]] = []
    for start in range(0, len(groups), chunk_size):
        tasks.append(list(groups[start : start + chunk_size]))
    return tasks


def prepare_dynamic_task_pool(
    *,
    output_dir: Path,
    groups: Sequence[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]],
    num_shards: int,
    dynamic_chunk_size: int,
) -> Path:
    task_dir = dynamic_task_pool_dir(
        output_dir=output_dir,
        num_shards=int(num_shards),
        dynamic_chunk_size=int(dynamic_chunk_size),
    )
    ensure_dir(task_dir)
    for stale_path in sorted(task_dir.glob("task.*.json")):
        stale_path.unlink()
    queue_manifest = task_dir / "queue_manifest.json"
    if queue_manifest.exists():
        queue_manifest.unlink()

    tasks = split_groups_into_tasks(groups=groups, dynamic_chunk_size=int(dynamic_chunk_size))
    for task_index, task_groups in enumerate(tasks):
        task_payload = {
            "task_index": int(task_index),
            "num_tasks": int(len(tasks)),
            "source_ids": [source_id for source_id, _ in task_groups],
            "num_source_patches": len(task_groups),
        }
        task_path = task_dir / f"task.{task_index:04d}.pending.json"
        write_json(task_path, task_payload)

    write_json(
        queue_manifest,
        {
            "num_shards": int(num_shards),
            "dynamic_chunk_size": int(dynamic_chunk_size),
            "num_tasks": int(len(tasks)),
            "num_source_patches": len(groups),
            "selected_source_ids": [source_id for source_id, _ in groups],
        },
    )
    return task_dir


def claim_next_dynamic_task(task_dir: Path, shard_index: int) -> Optional[Tuple[Dict[str, Any], Path]]:
    for pending_path in sorted(task_dir.glob("task.*.pending.json")):
        running_path = pending_path.with_name(
            pending_path.name.replace(".pending.json", f".worker{int(shard_index):02d}.running.json")
        )
        try:
            pending_path.rename(running_path)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        payload = json.loads(running_path.read_text(encoding="utf-8"))
        return payload, running_path
    return None


def finalize_dynamic_task(running_path: Path) -> None:
    done_path = running_path.with_name(running_path.name.replace(".running.json", ".done.json"))
    running_path.rename(done_path)


def build_patch_items(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "id": str(row.get("id", "")),
                "image": str(row.get("image", "")),
                "pred_lines": sanitize_lines(list(row.get("pred_lines", []))),
                "gt_lines": sanitize_lines(list(row.get("gt_lines", []))),
                "state_lines": [],
            }
        )
    return items


def _looks_like_peft_adapter(path: Union[str, Path]) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def load_generation_engine(
    args: argparse.Namespace,
) -> Tuple[Optional[AutoProcessor], Optional[torch.nn.Module], Optional[Any]]:
    processor = None
    model = None
    chat_model = None
    if args.engine == "custom":
        model_source = str(args.adapter)
        processor_path = str(args.processor_path or model_source)
        is_peft_adapter = _looks_like_peft_adapter(model_source)
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        if is_peft_adapter:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(args.base_model),
                torch_dtype=torch.bfloat16,
                device_map="auto" if str(args.device).startswith("cuda") else None,
                trust_remote_code=True,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("Processor does not expose a tokenizer.")
            current_vocab_size = int(base_model.get_input_embeddings().weight.shape[0])
            target_vocab_size = int(len(tokenizer))
            if target_vocab_size != current_vocab_size:
                base_model.resize_token_embeddings(target_vocab_size)
            base_model.config.vocab_size = int(base_model.get_input_embeddings().weight.shape[0])
            if hasattr(base_model, "vocab_size"):
                base_model.vocab_size = int(base_model.get_input_embeddings().weight.shape[0])
            model = PeftModel.from_pretrained(base_model, model_source)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=torch.bfloat16,
                device_map="auto" if str(args.device).startswith("cuda") else None,
                trust_remote_code=True,
            )
        if not str(args.device).startswith("cuda"):
            model = model.to(str(args.device))
        model.eval()
    else:
        from llamafactory.chat import ChatModel

        infer_args = {
            "model_name_or_path": str(args.base_model),
            "adapter_name_or_path": str(args.adapter),
            "finetuning_type": "lora",
            "stage": "sft",
            "template": str(args.template),
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
            "trust_remote_code": True,
            "image_max_pixels": int(args.image_max_pixels),
        }
        chat_model = ChatModel(infer_args)
    return processor, model, chat_model


def process_source_group(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    source_id: str,
    items: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    processor: Optional[AutoProcessor],
    model: Optional[torch.nn.Module],
    chat_model: Optional[Any],
    viz_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    merged_pred_lines_raw: List[Dict[str, Any]] = []
    merged_gt_lines_raw: List[Dict[str, Any]] = []
    image_rel = str(items[0][0].get("images", [""])[0])
    image_path = (dataset_root / image_rel).resolve()
    image = Image.open(image_path).convert("RGB")
    discrete_categories = parse_categories_arg(args.discrete_categories)

    box_results: List[Dict[str, Any]] = []
    for row, meta in items:
        sample_id = str(row.get("id"))
        if args.engine == "custom":
            assert processor is not None and model is not None
            pred_text, sample_score = generate_with_custom_engine(
                sample=row,
                image=image,
                image_path=image_path,
                processor=processor,
                model=model,
                max_new_tokens=int(args.max_new_tokens),
                prediction_format=str(args.prediction_format),
                discrete_categories=discrete_categories,
                discrete_coord_num_bins=int(args.discrete_coord_num_bins),
                discrete_image_size=int(args.patch_size),
                discrete_token_schema=str(args.discrete_token_schema),
                discrete_include_text_prompt_tokens=not bool(args.disable_legacy_text_prompt_tokens),
                user_prompt_style=str(args.user_prompt_style),
            )
        else:
            pred_text, sample_score = generate_with_llamafactory_engine(
                sample=row,
                image=image,
                chat_model=chat_model,
                max_new_tokens=int(args.max_new_tokens),
                prediction_format=str(args.prediction_format),
                discrete_categories=discrete_categories,
                discrete_coord_num_bins=int(args.discrete_coord_num_bins),
                discrete_image_size=int(args.patch_size),
                discrete_token_schema=str(args.discrete_token_schema),
                discrete_include_text_prompt_tokens=not bool(args.disable_legacy_text_prompt_tokens),
                user_prompt_style=str(args.user_prompt_style),
            )
        pred_obj, cleaned_pred_text = parse_prediction_text(
            pred_text,
            prediction_format=str(args.prediction_format),
            discrete_categories=discrete_categories,
            discrete_coord_num_bins=int(args.discrete_coord_num_bins),
            discrete_image_size=int(args.patch_size),
            discrete_token_schema=str(args.discrete_token_schema),
            discrete_include_text_prompt_tokens=not bool(args.disable_legacy_text_prompt_tokens),
        )
        parse_ok = pred_obj is not None
        pred_lines = sanitize_lines(list((pred_obj or {"lines": []}).get("lines", [])))
        gt_lines = sanitize_lines(list(meta.get("target_lines", [])))
        merged_pred_lines_raw.extend(pred_lines)
        merged_gt_lines_raw.extend(gt_lines)

        box_results.append(
            {
                "id": sample_id,
                "source_id": source_id,
                "image": image_rel,
                "grid_row": int(meta.get("grid_row", -1)),
                "grid_col": int(meta.get("grid_col", -1)),
                "target_box": meta.get("target_box", {}),
                "parse_ok": parse_ok,
                "sample_score": sample_score,
                "pred_text": pred_text,
                "pred_json_text": cleaned_pred_text,
                "pred_lines": pred_lines,
                "gt_lines": gt_lines,
            }
        )

    merged_pred_lines = stitch_lines(
        lines=merged_pred_lines_raw,
        patch_size=int(args.patch_size),
        grid_size=int(args.grid_size),
        merge_endpoint_dist_px=float(args.merge_endpoint_dist_px),
        internal_boundary_tol_px=float(args.internal_boundary_tol_px),
    )
    merged_gt_lines = stitch_lines(
        lines=merged_gt_lines_raw,
        patch_size=int(args.patch_size),
        grid_size=int(args.grid_size),
        merge_endpoint_dist_px=float(args.merge_endpoint_dist_px),
        internal_boundary_tol_px=float(args.internal_boundary_tol_px),
    )

    metrics = _sample_metrics(pred_lines=merged_pred_lines, gt_lines=merged_gt_lines, thresholds=[2.0, 4.0, 8.0])
    merged_row = {
        "id": source_id,
        "image": image_rel,
        "group_size": len(items),
        "pred_lines": merged_pred_lines,
        "gt_lines": merged_gt_lines,
        "metrics": metrics,
    }
    patch_item = {
        "id": source_id,
        "image": image_rel,
        "pred_lines": merged_pred_lines,
        "gt_lines": merged_gt_lines,
        "state_lines": [],
    }

    if viz_dir is not None:
        gt_panel = build_overlay(
            image=image,
            lines=merged_gt_lines,
            title=f"{source_id} | GT merged",
            line_color=(0, 120, 255),
            patch_size=int(args.patch_size),
            grid_size=int(args.grid_size),
        )
        pred_panel = build_overlay(
            image=image,
            lines=merged_pred_lines,
            title=f"{source_id} | Pred merged",
            line_color=(255, 60, 60),
            patch_size=int(args.patch_size),
            grid_size=int(args.grid_size),
        )
        compare = stack_panels(gt_panel, pred_panel)
        gt_panel.save(viz_dir / f"{source_id}_gt.png")
        pred_panel.save(viz_dir / f"{source_id}_pred.png")
        compare.save(viz_dir / f"{source_id}_compare.png")

    return box_results, merged_row, {key: float(value) for key, value in metrics.items()}, patch_item


def build_summary_payload(
    *,
    num_source_patches: int,
    expected_group_size: int,
    allow_incomplete_groups: bool,
    skipped_incomplete_groups: Sequence[Dict[str, Any]],
    patch_metrics_rows: Sequence[Dict[str, float]],
    patch_items_for_paper: Sequence[Dict[str, Any]],
    meter_per_pixel: float,
    line_width_px: int,
    categories: Optional[Sequence[str]],
    patch_size: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "num_source_patches": int(num_source_patches),
        "expected_group_size": int(expected_group_size),
        "allow_incomplete_groups": bool(allow_incomplete_groups),
        "skipped_incomplete_groups": list(skipped_incomplete_groups),
        "mean_patch_metrics": mean_dict(patch_metrics_rows),
        "merged_paper_metrics": evaluate_prediction_items(
            items=list(patch_items_for_paper),
            meter_per_pixel=float(meter_per_pixel),
            line_width_px=int(line_width_px),
            categories=list(categories) if categories else None,
            default_image_size=int(patch_size),
        ),
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def merge_shard_outputs(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    selected_groups: Sequence[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]],
    skipped_incomplete_groups: Sequence[Dict[str, Any]],
    expected_group_size: int,
    paper_categories: Sequence[str],
) -> None:
    if int(args.num_shards) <= 1:
        raise ValueError("--merge-shards requires --num-shards > 1")

    expected_source_ids = [source_id for source_id, _ in selected_groups]
    source_order = {source_id: idx for idx, source_id in enumerate(expected_source_ids)}
    merged_rows_all: List[Dict[str, Any]] = []
    box_rows_all: List[Dict[str, Any]] = []
    shard_manifests: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    for shard_index in range(int(args.num_shards)):
        paths = shard_output_paths(
            output_dir=output_dir,
            num_shards=int(args.num_shards),
            shard_index=shard_index,
            merge_shards=False,
        )
        missing_for_shard = [
            str(path)
            for path in (paths["box_predictions"], paths["merged_predictions"], paths["manifest"], paths["summary"])
            if not path.exists()
        ]
        if missing_for_shard:
            missing_files.extend(missing_for_shard)
            continue

        shard_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        # Older shard manifests may contain each source id twice because worker
        # bookkeeping pre-seeded the list before appending processed ids again.
        # Normalize here so completed shard outputs remain mergeable.
        shard_source_ids = list(dict.fromkeys(str(item) for item in shard_manifest.get("selected_source_ids", [])))
        if not bool(args.dynamic_schedule):
            expected_ids_for_shard = [
                source_id for idx, source_id in enumerate(expected_source_ids) if idx % int(args.num_shards) == shard_index
            ]
            if shard_source_ids != expected_ids_for_shard:
                raise ValueError(
                    f"Shard {shard_index} source-id mismatch. expected={expected_ids_for_shard} got={shard_source_ids}"
                )

        shard_merged_rows = load_jsonl(paths["merged_predictions"])
        shard_box_rows = load_jsonl(paths["box_predictions"])
        merged_rows_all.extend(shard_merged_rows)
        box_rows_all.extend(shard_box_rows)
        shard_manifests.append(
            {
                "shard_index": shard_index,
                "tag": shard_tag(shard_index=shard_index, num_shards=int(args.num_shards)),
                "summary_path": str(paths["summary"]),
                "manifest_path": str(paths["manifest"]),
                "box_predictions_path": str(paths["box_predictions"]),
                "merged_predictions_path": str(paths["merged_predictions"]),
                "num_source_patches": len(shard_source_ids),
                "selected_source_ids": shard_source_ids,
            }
        )

    if missing_files:
        raise ValueError(f"Missing shard outputs before merge: {missing_files}")

    merged_ids = [str(row.get("id", "")) for row in merged_rows_all]
    missing_source_ids = [source_id for source_id in expected_source_ids if source_id not in set(merged_ids)]
    extra_source_ids = sorted({source_id for source_id in merged_ids if source_id not in source_order})
    duplicate_source_ids = sorted({source_id for source_id in merged_ids if merged_ids.count(source_id) > 1})
    if missing_source_ids or extra_source_ids or duplicate_source_ids:
        raise ValueError(
            "Merged shard outputs do not match expected source ids: "
            f"missing={missing_source_ids}, extra={extra_source_ids}, duplicates={duplicate_source_ids}"
        )

    merged_rows_all.sort(key=lambda row: source_order[str(row.get("id", ""))])
    box_rows_all.sort(
        key=lambda row: (
            source_order.get(str(row.get("source_id", "")), 10**9),
            int(row.get("grid_row", 10**9)),
            int(row.get("grid_col", 10**9)),
            str(row.get("id", "")),
        )
    )

    canonical_paths = shard_output_paths(
        output_dir=output_dir,
        num_shards=int(args.num_shards),
        shard_index=0,
        merge_shards=True,
    )
    write_jsonl(canonical_paths["box_predictions"], box_rows_all)
    write_jsonl(canonical_paths["merged_predictions"], merged_rows_all)

    patch_metrics_rows = [
        {str(key): float(value) for key, value in row.get("metrics", {}).items()}
        for row in merged_rows_all
        if isinstance(row.get("metrics"), dict)
    ]
    summary = build_summary_payload(
        num_source_patches=len(expected_source_ids),
        expected_group_size=int(expected_group_size),
        allow_incomplete_groups=bool(args.allow_incomplete_groups),
        skipped_incomplete_groups=skipped_incomplete_groups,
        patch_metrics_rows=patch_metrics_rows,
        patch_items_for_paper=build_patch_items(merged_rows_all),
        meter_per_pixel=float(args.meter_per_pixel),
        line_width_px=int(args.line_width_px),
        categories=paper_categories,
        patch_size=int(args.patch_size),
        extra_fields={
            "num_shards": int(args.num_shards),
            "merged_from_shards": True,
            "scheduling_mode": "dynamic" if bool(args.dynamic_schedule) else "static",
            "dynamic_chunk_size": int(args.dynamic_chunk_size) if bool(args.dynamic_schedule) else 0,
        },
    )
    manifest = {
        "dataset_jsonl": str(args.dataset_jsonl.resolve()),
        "meta_jsonl": str(args.meta_jsonl.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "adapter": str(args.adapter),
        "num_selected_source_patches": len(expected_source_ids),
        "selected_source_ids": expected_source_ids,
        "skipped_incomplete_groups": list(skipped_incomplete_groups),
        "num_shards": int(args.num_shards),
        "merged_from_shards": True,
        "scheduling_mode": "dynamic" if bool(args.dynamic_schedule) else "static",
        "dynamic_chunk_size": int(args.dynamic_chunk_size) if bool(args.dynamic_schedule) else 0,
        "shard_manifests": shard_manifests,
    }
    write_json(canonical_paths["summary"], summary)
    write_json(canonical_paths["manifest"], manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    validate_shard_args(args)
    dataset_rows = load_jsonl(args.dataset_jsonl)
    meta_rows = load_jsonl(args.meta_jsonl)
    row_by_id = {str(row.get("id")): row for row in dataset_rows}

    groups: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for meta in meta_rows:
        sample_id = str(meta.get("id", ""))
        row = row_by_id.get(sample_id)
        if row is None:
            continue
        source_id = str(meta.get("source_id") or "")
        if not source_id:
            source_id = sample_id.rsplit("_g", 1)[0] if "_g" in sample_id else sample_id
        groups.setdefault(source_id, []).append((row, meta))

    wanted_source_ids = parse_source_ids(args.source_ids)
    ordered_source_ids = sorted(groups.keys())
    if wanted_source_ids:
        ordered_source_ids = [sid for sid in wanted_source_ids if sid in groups]
        missing = [sid for sid in wanted_source_ids if sid not in groups]
        if missing:
            raise ValueError(f"source ids not found: {missing}")
    elif int(args.max_source_patches) > 0:
        rng = random.Random(int(args.seed))
        ordered_source_ids = sorted(rng.sample(ordered_source_ids, min(len(ordered_source_ids), int(args.max_source_patches))))

    expected_group_size = int(args.grid_size) * int(args.grid_size)
    selected_groups: List[Tuple[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]] = []
    skipped_incomplete: List[Dict[str, Any]] = []
    for source_id in ordered_source_ids:
        items = sorted(
            groups[source_id],
            key=lambda pair: (
                int(pair[1].get("grid_row", 10**9)),
                int(pair[1].get("grid_col", 10**9)),
                str(pair[0].get("id", "")),
            ),
        )
        if len(items) != expected_group_size and not bool(args.allow_incomplete_groups):
            skipped_incomplete.append({"source_id": source_id, "group_size": len(items)})
            continue
        selected_groups.append((source_id, items))

    if not selected_groups:
        raise ValueError(
            "No complete source groups selected. Use a full-grid fixed16 dataset or pass --allow-incomplete-groups."
        )

    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    paper_categories = [item.strip() for item in str(args.paper_categories).split(",") if item.strip()]
    group_map = {source_id: items for source_id, items in selected_groups}

    if bool(args.merge_shards):
        merge_shard_outputs(
            args=args,
            output_dir=output_dir,
            selected_groups=selected_groups,
            skipped_incomplete_groups=skipped_incomplete,
            expected_group_size=expected_group_size,
            paper_categories=paper_categories,
        )
        return

    total_selected_groups = len(selected_groups)
    if bool(args.prepare_dynamic_tasks):
        task_dir = prepare_dynamic_task_pool(
            output_dir=output_dir,
            groups=selected_groups,
            num_shards=int(args.num_shards),
            dynamic_chunk_size=int(args.dynamic_chunk_size),
        )
        print(
            json.dumps(
                {
                    "event": "prepared_dynamic_tasks",
                    "task_dir": str(task_dir),
                    "num_tasks": len(split_groups_into_tasks(selected_groups, int(args.dynamic_chunk_size))),
                    "num_source_patches": int(total_selected_groups),
                    "dynamic_chunk_size": int(args.dynamic_chunk_size),
                    "num_shards": int(args.num_shards),
                },
                ensure_ascii=False,
            )
        )
        return

    static_selected_groups = select_groups_for_shard(
        groups=selected_groups,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )

    paths = shard_output_paths(
        output_dir=output_dir,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        merge_shards=False,
    )
    if not bool(args.skip_viz):
        ensure_dir(paths["viz_dir"])

    box_predictions_path = paths["box_predictions"]
    merged_predictions_path = paths["merged_predictions"]
    for path in (box_predictions_path, merged_predictions_path):
        if path.exists():
            path.unlink()
        path.write_text("", encoding="utf-8")

    # Record source ids in the actual processing order. Do not pre-seed this
    # list for static sharding; otherwise the manifest duplicates every source
    # once when handle_group() appends the processed ids again.
    worker_selected_source_ids: List[str] = []
    claimed_task_indices: List[int] = []
    total_dynamic_tasks = 0
    if bool(args.dynamic_schedule) and int(args.num_shards) > 1:
        task_dir = dynamic_task_pool_dir(
            output_dir=output_dir,
            num_shards=int(args.num_shards),
            dynamic_chunk_size=int(args.dynamic_chunk_size),
        )
        queue_manifest = task_dir / "queue_manifest.json"
        if not queue_manifest.exists():
            raise ValueError(f"Missing dynamic task manifest: {queue_manifest}")
        queue_payload = json.loads(queue_manifest.read_text(encoding="utf-8"))
        total_dynamic_tasks = int(queue_payload.get("num_tasks", 0))

    if not bool(args.dynamic_schedule) and not static_selected_groups:
        summary = build_summary_payload(
            num_source_patches=0,
            expected_group_size=int(expected_group_size),
            allow_incomplete_groups=bool(args.allow_incomplete_groups),
            skipped_incomplete_groups=skipped_incomplete,
            patch_metrics_rows=[],
            patch_items_for_paper=[],
            meter_per_pixel=float(args.meter_per_pixel),
            line_width_px=int(args.line_width_px),
            categories=paper_categories,
            patch_size=int(args.patch_size),
            extra_fields={
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "total_source_patches_before_sharding": int(total_selected_groups),
                "scheduling_mode": "static",
                "dynamic_chunk_size": 0,
            },
        )
        manifest = {
            "dataset_jsonl": str(args.dataset_jsonl.resolve()),
            "meta_jsonl": str(args.meta_jsonl.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "adapter": str(args.adapter),
            "selected_source_ids": [],
            "num_selected_source_patches": 0,
            "skipped_incomplete_groups": skipped_incomplete,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "total_source_patches_before_sharding": int(total_selected_groups),
            "scheduling_mode": "static",
            "dynamic_chunk_size": 0,
            "claimed_task_indices": [],
        }
        write_json(paths["summary"], summary)
        write_json(paths["manifest"], manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    processor: Optional[AutoProcessor] = None
    model: Optional[torch.nn.Module] = None
    chat_model: Optional[Any] = None

    patch_metrics_rows: List[Dict[str, float]] = []
    patch_items_for_paper: List[Dict[str, Any]] = []
    started_at = time.time()
    processed_group_count = 0
    dynamic_mode = bool(args.dynamic_schedule) and int(args.num_shards) > 1
    viz_dir = None if bool(args.skip_viz) else paths["viz_dir"]

    def maybe_init_engine() -> None:
        nonlocal processor, model, chat_model
        if processor is None and model is None and chat_model is None:
            processor, model, chat_model = load_generation_engine(args)

    def handle_group(source_id: str, items: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
        nonlocal processed_group_count
        maybe_init_engine()
        box_results, merged_row, metric_row, patch_item = process_source_group(
            args=args,
            dataset_root=args.dataset_root,
            source_id=source_id,
            items=items,
            processor=processor,
            model=model,
            chat_model=chat_model,
            viz_dir=viz_dir,
        )
        for box_result in box_results:
            append_jsonl(box_predictions_path, box_result)
        append_jsonl(merged_predictions_path, merged_row)
        patch_metrics_rows.append(metric_row)
        patch_items_for_paper.append(patch_item)
        worker_selected_source_ids.append(source_id)
        processed_group_count += 1

        should_report = (
            processed_group_count == 1
            or (not dynamic_mode and processed_group_count == len(static_selected_groups))
            or (int(args.progress_every) > 0 and processed_group_count % int(args.progress_every) == 0)
        )
        if not should_report:
            return

        elapsed_sec = max(0.0, time.time() - started_at)
        progress_payload: Dict[str, Any] = {
            "event": "progress",
            "processed_groups": processed_group_count,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "source_id": source_id,
            "elapsed_sec": round(elapsed_sec, 1),
            "scheduling_mode": "dynamic" if dynamic_mode else "static",
            "total_source_patches_before_sharding": int(total_selected_groups),
        }
        if dynamic_mode:
            progress_payload["claimed_tasks"] = len(claimed_task_indices)
            progress_payload["total_tasks"] = int(total_dynamic_tasks)
        else:
            total_groups = len(static_selected_groups)
            avg_sec = elapsed_sec / max(1, processed_group_count)
            remaining_sec = avg_sec * max(0, total_groups - processed_group_count)
            progress_payload["total_groups"] = total_groups
            progress_payload["eta_sec"] = round(remaining_sec, 1)
        print(json.dumps(progress_payload, ensure_ascii=False), flush=True)

    if dynamic_mode:
        task_dir = dynamic_task_pool_dir(
            output_dir=output_dir,
            num_shards=int(args.num_shards),
            dynamic_chunk_size=int(args.dynamic_chunk_size),
        )
        while True:
            claimed = claim_next_dynamic_task(task_dir=task_dir, shard_index=int(args.shard_index))
            if claimed is None:
                break
            task_payload, running_path = claimed
            task_index = int(task_payload.get("task_index", -1))
            source_ids = [str(item) for item in task_payload.get("source_ids", [])]
            claimed_task_indices.append(task_index)
            print(
                json.dumps(
                    {
                        "event": "claim_task",
                        "task_index": task_index,
                        "num_tasks": int(task_payload.get("num_tasks", total_dynamic_tasks)),
                        "num_source_patches": len(source_ids),
                        "claimed_tasks": len(claimed_task_indices),
                        "shard_index": int(args.shard_index),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            for source_id in source_ids:
                if source_id not in group_map:
                    raise ValueError(f"Dynamic task source_id not found in selection: {source_id}")
                handle_group(source_id=source_id, items=group_map[source_id])
            finalize_dynamic_task(running_path)
    else:
        for source_id, items in static_selected_groups:
            handle_group(source_id=source_id, items=items)

    summary = build_summary_payload(
        num_source_patches=len(worker_selected_source_ids),
        expected_group_size=int(expected_group_size),
        allow_incomplete_groups=bool(args.allow_incomplete_groups),
        skipped_incomplete_groups=skipped_incomplete,
        patch_metrics_rows=patch_metrics_rows,
        patch_items_for_paper=patch_items_for_paper,
        meter_per_pixel=float(args.meter_per_pixel),
        line_width_px=int(args.line_width_px),
        categories=paper_categories,
        patch_size=int(args.patch_size),
        extra_fields={
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "total_source_patches_before_sharding": int(total_selected_groups),
            "scheduling_mode": "dynamic" if dynamic_mode else "static",
            "dynamic_chunk_size": int(args.dynamic_chunk_size) if dynamic_mode else 0,
            "claimed_task_indices": claimed_task_indices,
            "num_claimed_tasks": len(claimed_task_indices),
        },
    )
    manifest = {
        "dataset_jsonl": str(args.dataset_jsonl.resolve()),
        "meta_jsonl": str(args.meta_jsonl.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "adapter": str(args.adapter),
        "selected_source_ids": worker_selected_source_ids,
        "num_selected_source_patches": len(worker_selected_source_ids),
        "skipped_incomplete_groups": skipped_incomplete,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "total_source_patches_before_sharding": int(total_selected_groups),
        "scheduling_mode": "dynamic" if dynamic_mode else "static",
        "dynamic_chunk_size": int(args.dynamic_chunk_size) if dynamic_mode else 0,
        "claimed_task_indices": claimed_task_indices,
    }
    write_json(paths["summary"], summary)
    write_json(paths["manifest"], manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
