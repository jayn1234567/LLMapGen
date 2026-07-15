#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.torch_runtime import maybe_disable_cudnn_from_env

maybe_disable_cudnn_from_env(torch)

from mllm.constants import IMAGE_TOKEN_INDEX
from mllm.coord_utils import (
    COORD_MODE_PIXEL,
    COORD_MODE_NORM1000,
    DEFAULT_COORD_RANGE,
    convert_items,
    convert_payload_text,
    payload_to_text,
    record_coord_config,
)
from mllm.coordinate_tokens import (
    COORDINATE_TOKEN_MODE_ANGLE,
    append_coordinate_token_instruction,
)
from mllm.mm_utils import process_images, tokenizer_image_token
from scripts.tools.infer_centerline_checkpoint import (
    build_prompt,
    completion_token_ids,
    _config_overrides_from_args,
    extract_json_payload,
    load_model_components,
    normalize_prediction_text,
    parse_map_json,
    read_manifest,
    resolve_coordinate_token_settings,
)
from scripts.tools.map_visualization import (
    record_tile_id,
    render_whole_map_visualizations as render_common_whole_map_visualizations,
    sanitize_points,
)


TASK_TEXT = "Please construct the complete road map in the current BEV (Bird's Eye View) image patch."


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("patch_results"), list):
                return payload["patch_results"]
            if isinstance(payload, dict):
                return [payload]
            return payload
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_image_path(raw_path: str, image_folder: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return image_folder / image_path


def get_meta_int(record, key, fallback=None):
    meta = record.get("meta") or {}
    value = meta.get(key, record.get(key, fallback))
    if value is None:
        return fallback
    return int(value)


def sort_patch_records(records):
    def key_fn(record):
        meta = record.get("meta") or {}
        tile_id = record_tile_id(record)
        row = get_meta_int(record, "row", get_meta_int(record, "patch_row", 0))
        col = get_meta_int(record, "col", get_meta_int(record, "patch_col", 0))
        return str(tile_id), row, col, str(record.get("id", ""))

    return sorted(records, key=key_fn)


def distributed_info(args):
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", "0"))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def resolve_runtime_device(device: str, local_rank: int) -> str:
    if device and device not in {"auto", "cuda", "npu"}:
        return device
    if device in {"", "auto", "cuda"} and torch.cuda.is_available():
        return f"cuda:{local_rank}" if device in {"", "auto", "cuda"} else device
    if device in {"", "auto", "npu"}:
        try:
            import torch_npu  # noqa: F401

            if hasattr(torch, "npu") and torch.npu.is_available():
                return f"npu:{local_rank}"
        except Exception:
            pass
    if device == "cuda":
        return f"cuda:{local_rank}"
    if device == "npu":
        return f"npu:{local_rank}"
    return "cpu"


def shard_records_by_tile(records, rank: int, world_size: int):
    grouped = {}
    for record in records:
        grouped.setdefault(record_tile_id(record), []).append(record)
    tile_ids = sorted(grouped)
    assigned_tile_ids = tile_ids[rank::world_size]
    assigned = []
    for tile_id in assigned_tile_ids:
        assigned.extend(grouped[tile_id])
    return assigned, assigned_tile_ids, tile_ids


def rank_json_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}_rank{rank:05d}{path.suffix}")


def wait_for_rank_summaries(output_json: Path, world_size: int, timeout_seconds: int):
    deadline = time.time() + timeout_seconds
    paths = [rank_json_path(output_json, rank) for rank in range(world_size)]
    missing = [path for path in paths if not path.is_file()]
    while missing and time.time() < deadline:
        time.sleep(5)
        missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise TimeoutError(f"timed out waiting for rank summaries: {missing_text}")
    return paths


def merge_rank_summaries(rank_paths):
    patch_results = []
    merged_global_lines = []
    base_summary = None
    for path in rank_paths:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        if base_summary is None:
            base_summary = {key: value for key, value in summary.items() if key not in {"patch_results", "merged_global"}}
        patch_results.extend(summary.get("patch_results", []))
        merged_global_lines.extend((summary.get("merged_global") or {}).get("lines", []))
    patch_results.sort(key=lambda item: (item.get("idx", 0), item.get("record_id", "")))
    merged_global_lines.sort(key=lambda item: (
        str(item.get("category", "")),
        item.get("points", [[0, 0]])[0][1] if item.get("points") else 0,
        item.get("points", [[0, 0]])[0][0] if item.get("points") else 0,
    ))
    summary = base_summary or {}
    summary.update({
        "num_patches": len(patch_results),
        "merged_global": {"lines": merged_global_lines},
        "patch_results": patch_results,
    })
    return summary


def build_eval_payload(records, args, evaluate_records_fn, evaluate_lane_intersection_records_fn):
    eval_kwargs = dict(
        meter_per_pixel=args.eval_meter_per_pixel,
        buffer_size=args.eval_buffer_size,
        match_threshold=args.eval_match_threshold,
    )
    if args.include_intersections:
        map_eval = evaluate_lane_intersection_records_fn(records, **eval_kwargs)
        return {
            "centerline_eval": map_eval["lane"],
            "intersection_eval": map_eval["intersection"],
            "lane_intersection_eval": map_eval["lane_intersection"],
            "map_eval": map_eval,
        }
    return evaluate_records_fn(records, **eval_kwargs)


def print_eval_payload(eval_payload, args, print_eval_table_fn, print_lane_intersection_eval_tables_fn):
    if args.include_intersections:
        print_lane_intersection_eval_tables_fn(eval_payload["map_eval"])
    else:
        print_eval_table_fn(eval_payload)


def eval_console_payload(eval_path, eval_payload, args):
    payload = {"eval_json": str(eval_path), "centerline_eval_json": str(eval_path)}
    if args.include_intersections:
        payload.update({
            "centerline_eval": eval_payload["centerline_eval"],
            "intersection_eval": eval_payload["intersection_eval"],
            "lane_intersection_eval": eval_payload["lane_intersection_eval"],
        })
    else:
        payload["centerline_eval"] = eval_payload
    return payload


def coord_description(coord_mode: str, coord_range: int, patch_size: int) -> str:
    if coord_mode == COORD_MODE_NORM1000:
        return f"Coordinates use a normalized 0-{coord_range} grid over the original {patch_size}x{patch_size} image patch."
    return f"Coordinates use original patch pixel coordinates in [0,{patch_size - 1}]."


def make_user_prompt(patch_size: int, incoming_traces, incoming_intersections=None,
                     include_intersections=False, coord_mode=COORD_MODE_PIXEL, coord_range=DEFAULT_COORD_RANGE):
    traces_json = json.dumps(incoming_traces, ensure_ascii=False, separators=(",", ":"))
    parts = [
        "<image>",
        TASK_TEXT,
        coord_description(coord_mode, coord_range, patch_size),
        "",
        "Incoming traces JSON:",
        traces_json,
    ]
    if include_intersections:
        inter_json = json.dumps(incoming_intersections or [], ensure_ascii=False, separators=(",", ":"))
        parts.extend(["", "Incoming intersections JSON:", inter_json])
    parts.extend([
        "",
        "Each incoming trace has 1 to 3 points. If multiple points are present, they are ordered from the previous patch interior toward the current patch boundary.",
        "Incoming traces are continuity hints only; they may be incomplete or absent.",
    ])
    if include_intersections:
        parts.append("Each incoming intersection has 1 to 3 boundary points from neighboring patches.")
    return "\n".join(parts)


def resolve_coord_config(record, args):
    cfg = record_coord_config(
        record,
        default_mode=COORD_MODE_PIXEL,
        default_patch_size=args.patch_size,
        default_coord_range=args.coord_range,
    )
    if args.coord_mode != "auto":
        cfg["coord_mode"] = args.coord_mode
        cfg["coord_range"] = args.coord_range
    return cfg


def near(value, target, tol):
    return abs(value - target) <= tol


def endpoint_trace(points, endpoint_index, max_points):
    if endpoint_index == 0:
        return list(reversed(points[:max_points]))
    return points[-max_points:]


def point_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def dedupe_consecutive_points(points):
    result = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def resample_polyline_by_distance(points, sample_distance_px):
    points = dedupe_consecutive_points(sanitize_points(points))
    if sample_distance_px <= 0 or len(points) < 2:
        return points

    cumulative = [0.0]
    for prev, cur in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + point_distance(prev, cur))
    total_length = cumulative[-1]
    if total_length <= 0:
        return points

    targets = []
    distance = 0.0
    while distance < total_length:
        targets.append(distance)
        distance += sample_distance_px
    if not targets or not math.isclose(targets[-1], total_length):
        targets.append(total_length)

    sampled = []
    seg_idx = 0
    for target in targets:
        while seg_idx < len(cumulative) - 2 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        start = points[seg_idx]
        end = points[seg_idx + 1]
        seg_len = cumulative[seg_idx + 1] - cumulative[seg_idx]
        ratio = 0.0 if seg_len <= 0 else (target - cumulative[seg_idx]) / seg_len
        sampled.append([
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        ])
    return sampled


def sampled_endpoint_trace(points, endpoint_index, max_points, sample_distance_px):
    if sample_distance_px <= 0:
        return endpoint_trace(points, endpoint_index, max_points)

    sampled = resample_polyline_by_distance(points, sample_distance_px)
    if not sampled:
        return []

    boundary_point = points[0] if endpoint_index == 0 else points[-1]
    nearest_idx = min(range(len(sampled)), key=lambda idx: point_distance(sampled[idx], boundary_point))
    if endpoint_index == 0:
        end_idx = min(len(sampled), nearest_idx + max_points)
        return list(reversed(sampled[nearest_idx:end_idx]))

    start_idx = max(0, nearest_idx - max_points + 1)
    return sampled[start_idx:nearest_idx + 1]


def transform_trace_points(points, dx, dy):
    return [[int(round(x + dx)), int(round(y + dy))] for x, y in sanitize_points(points)]


def extract_neighbor_traces(neighbor_lines, side, patch_size, boundary_tol, max_points, sample_distance_px):
    traces = []
    for line in neighbor_lines:
        if not isinstance(line, dict):
            continue
        if line.get("category") != "centerline":
            continue
        points = sanitize_points(line.get("points") or [])
        if not points:
            continue

        candidates = []
        if side == "left":
            if line.get("end_type") == "cut" and near(points[-1][0], patch_size - 1, boundary_tol):
                candidates.append(sampled_endpoint_trace(points, -1, max_points, sample_distance_px))
            if line.get("start_type") == "cut" and near(points[0][0], patch_size - 1, boundary_tol):
                candidates.append(sampled_endpoint_trace(points, 0, max_points, sample_distance_px))
            dx, dy = -patch_size, 0
            trace_side = "left"
        else:
            if line.get("end_type") == "cut" and near(points[-1][1], patch_size - 1, boundary_tol):
                candidates.append(sampled_endpoint_trace(points, -1, max_points, sample_distance_px))
            if line.get("start_type") == "cut" and near(points[0][1], patch_size - 1, boundary_tol):
                candidates.append(sampled_endpoint_trace(points, 0, max_points, sample_distance_px))
            dx, dy = 0, -patch_size
            trace_side = "top"

        for trace_points in candidates:
            if trace_points:
                traces.append({
                    "side": trace_side,
                    "points": transform_trace_points(trace_points, dx, dy),
                })
    return traces


def assign_trace_ids(traces):
    counts = {"left": 0, "top": 0}
    result = []
    for trace in traces:
        side = trace["side"]
        prefix = "L" if side == "left" else "T"
        result.append({
            "id": f"{prefix}{counts[side]}",
            "side": side,
            "points": trace["points"],
        })
        counts[side] += 1
    return result


def assign_intersection_ids(hints):
    counts = {"left": 0, "top": 0}
    result = []
    for hint in hints:
        side = hint["side"]
        prefix = "IL" if side == "left" else "IT"
        result.append({
            "id": f"{prefix}{counts[side]}",
            "side": side,
            "points": hint["points"],
        })
        counts[side] += 1
    return result


def build_incoming_traces(state_by_pos, tile_id, row, col, patch_size, boundary_tol, max_points, sample_distance_px):
    traces = []
    left_lines = state_by_pos.get((tile_id, row, col - 1), [])
    top_lines = state_by_pos.get((tile_id, row - 1, col), [])
    traces.extend(extract_neighbor_traces(left_lines, "left", patch_size, boundary_tol, max_points, sample_distance_px))
    traces.extend(extract_neighbor_traces(top_lines, "top", patch_size, boundary_tol, max_points, sample_distance_px))
    return assign_trace_ids(traces)


def sample_points(points, max_points):
    points = [point for idx, point in enumerate(points) if idx == 0 or point != points[idx - 1]]
    if len(points) <= max_points:
        return points
    if max_points <= 1:
        return [points[len(points) // 2]]
    step = (len(points) - 1) / (max_points - 1)
    return [points[round(i * step)] for i in range(max_points)]


def extract_neighbor_intersections(neighbor_lines, side, patch_size, boundary_tol, max_points):
    hints = []
    for line in neighbor_lines:
        if not isinstance(line, dict):
            continue
        if line.get("category") != "intersection" or not line.get("is_cut"):
            continue
        points = sanitize_points(line.get("points") or [])
        if side == "left":
            boundary_points = [point for point in points if near(point[0], patch_size - 1, boundary_tol)]
            dx, dy = -patch_size, 0
        else:
            boundary_points = [point for point in points if near(point[1], patch_size - 1, boundary_tol)]
            dx, dy = 0, -patch_size
        boundary_points = sample_points(boundary_points, max_points)
        if not boundary_points:
            continue
        hints.append({
            "side": side,
            "points": transform_trace_points(boundary_points, dx, dy),
        })
    return hints


def build_incoming_intersections(state_by_pos, tile_id, row, col, patch_size, boundary_tol, max_points):
    hints = []
    left_lines = state_by_pos.get((tile_id, row, col - 1), [])
    top_lines = state_by_pos.get((tile_id, row - 1, col), [])
    hints.extend(extract_neighbor_intersections(left_lines, "left", patch_size, boundary_tol, max_points))
    hints.extend(extract_neighbor_intersections(top_lines, "top", patch_size, boundary_tol, max_points))
    return assign_intersection_ids(hints)


def local_to_global_lines(lines, x0, y0):
    global_lines = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        xy_points = sanitize_points(line.get("points", []))
        if not xy_points:
            continue
        converted = dict(line)
        converted["points"] = [[int(x + x0), int(y + y0)] for x, y in xy_points]
        global_lines.append(converted)
    return global_lines


def load_json_maybe(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {}


def normalize_lines(payload):
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        return payload["lines"]
    if isinstance(payload, list):
        return payload
    return []


def offset_lines(lines, dx, dy):
    shifted = []
    for line in normalize_lines(lines):
        if not isinstance(line, dict):
            continue
        xy_points = sanitize_points(line.get("points") or [])
        if not xy_points:
            continue
        out = dict(line)
        out["points"] = [[int(round(x + dx)), int(round(y + dy))] for x, y in xy_points]
        shifted.append(out)
    return shifted


def draw_map_lines(image: Image.Image, payload, centerline_color: tuple, intersection_color: tuple, width: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for item in normalize_lines(payload):
        if not isinstance(item, dict):
            continue
        xy_points = sanitize_points(item.get("points") or [])
        if not xy_points:
            continue
        category = str(item.get("category", "centerline")).lower()
        color = intersection_color if category == "intersection" else centerline_color
        for idx in range(len(xy_points) - 1):
            draw.line([xy_points[idx], xy_points[idx + 1]], fill=color, width=width)
        for x, y in xy_points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    return image


def add_title(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 40), "black")
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill="white", font=font)
    return canvas


def render_whole_map_visualizations(patch_results, image_folder: Path, output_dir: Path):
    """Render one stitched BEV canvas per tile for B-stage state-update outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for record in patch_results:
        tile_id = record.get("tile_id") or (record.get("meta") or {}).get("tile_id") or "tile"
        grouped.setdefault(str(tile_id), []).append(record)

    rendered = []
    for tile_id, records in sorted(grouped.items()):
        if not records:
            continue
        origin_x = min(int(record.get("x0", 0)) for record in records)
        origin_y = min(int(record.get("y0", 0)) for record in records)
        max_x = max(int(record.get("x0", 0)) + int(record.get("patch_width", record.get("patch_size", 256))) for record in records)
        max_y = max(int(record.get("y0", 0)) + int(record.get("patch_height", record.get("patch_size", 256))) for record in records)
        width = max(1, max_x - origin_x)
        height = max(1, max_y - origin_y)

        background = Image.new("RGB", (width, height), "black")
        for record in records:
            image_path = resolve_image_path(record.get("image", ""), image_folder)
            if not image_path.exists():
                continue
            patch_width = int(record.get("patch_width", record.get("patch_size", 256)))
            patch_height = int(record.get("patch_height", record.get("patch_size", 256)))
            patch = Image.open(image_path).convert("RGB")
            if patch.size != (patch_width, patch_height):
                patch = patch.resize((patch_width, patch_height))
            x = int(record.get("x0", 0)) - origin_x
            y = int(record.get("y0", 0)) - origin_y
            background.paste(patch, (x, y))

        gt_lines = []
        pred_lines = []
        for record in records:
            x0 = int(record.get("x0", 0))
            y0 = int(record.get("y0", 0))
            gt_text = record.get("ground_truth_pixel") or record.get("labels_pixel") or ""
            if not gt_text:
                gt_text = record.get("ground_truth") or record.get("labels") or "{}"
            gt_lines.extend(offset_lines(normalize_lines(load_json_maybe(gt_text)), x0 - origin_x, y0 - origin_y))
            pred_lines.extend(offset_lines(record.get("lines_global") or [], -origin_x, -origin_y))

        gt_canvas = draw_map_lines(background.copy(), {"lines": gt_lines}, (0, 255, 0), (255, 255, 0))
        pred_canvas = draw_map_lines(background.copy(), {"lines": pred_lines}, (255, 0, 0), (0, 128, 255))
        gt_panel = add_title(gt_canvas, f"{tile_id} Ground Truth")
        pred_panel = add_title(pred_canvas, f"{tile_id} Prediction")
        compare = Image.new("RGB", (gt_panel.width + pred_panel.width + 10, gt_panel.height), "black")
        compare.paste(gt_panel, (0, 0))
        compare.paste(pred_panel, (gt_panel.width + 10, 0))

        safe_tile = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tile_id)
        gt_path = output_dir / f"{safe_tile}_ground_truth.png"
        pred_path = output_dir / f"{safe_tile}_prediction.png"
        compare_path = output_dir / f"{safe_tile}_compare.png"
        gt_canvas.save(gt_path)
        pred_canvas.save(pred_path)
        compare.save(compare_path)
        rendered.append({
            "tile_id": tile_id,
            "origin": [origin_x, origin_y],
            "size": [width, height],
            "ground_truth": str(gt_path),
            "prediction": str(pred_path),
            "compare": str(compare_path),
        })
    return rendered


def run_patch_inference(record, prompt_text, image_folder, tokenizer, model, image_processor,
                        conv_template, max_new_tokens, temperature):
    image_path = resolve_image_path(record["image"], image_folder).resolve()
    image = Image.open(image_path).convert("RGB")
    images_tensor = process_images([image], image_processor, model.config)
    vision_tower = model.get_vision_tower()
    dtype = vision_tower.dtype if vision_tower is not None else next(model.parameters()).dtype
    image_device = vision_tower.device if vision_tower is not None else model.device
    if isinstance(images_tensor, list):
        images_tensor = [img.to(dtype=dtype, device=image_device) for img in images_tensor]
    else:
        images_tensor = images_tensor.to(dtype=dtype, device=image_device)

    coordinate_token_mode, coordinate_token_max = resolve_coordinate_token_settings(
        tokenizer,
        model.config,
    )
    if coordinate_token_mode == COORDINATE_TOKEN_MODE_ANGLE:
        prompt_text = append_coordinate_token_instruction(
            prompt_text,
            max_coordinate=coordinate_token_max,
        )
    prompt = build_prompt(prompt_text, conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)
    generation_config = getattr(model, "generation_config", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None) or tokenizer.pad_token_id
    eos_token_id = getattr(generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    attention_mask = input_ids.ne(pad_token_id).to(model.device)

    generate_kwargs = {
        "attention_mask": attention_mask,
        "images": images_tensor,
        "image_sizes": [image.size],
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": temperature > 0,
        "num_beams": 1,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generate_kwargs)

    decoded_ids, decoded_mode = completion_token_ids(output_ids, input_ids)
    raw_prediction = tokenizer.batch_decode(decoded_ids.unsqueeze(0), skip_special_tokens=False)[0].strip()
    prediction = normalize_prediction_text(
        raw_prediction,
        max_coordinate=coordinate_token_max,
    )
    return image_path, prompt, raw_prediction, prediction, int(input_ids.shape[1]), int(output_ids.shape[1]), int(decoded_ids.numel()), decoded_mode


def main():
    parser = argparse.ArgumentParser(description="Patch-by-patch state-update inference for centerline/intersection maps.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--patch-json", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vision_tower", default="", help="Override external DINO vision tower path when checkpoint metadata points elsewhere.")
    parser.add_argument("--mm_vision_tower_type", default="")
    parser.add_argument("--multi_vision_towers", default="")
    parser.add_argument("--multi_vision_tower_types", default="")
    parser.add_argument("--multi_vision_input_image_sizes", default="")
    parser.add_argument("--multi_vision_primary_index", type=int, default=None)
    parser.add_argument("--multi_vision_hidden_size", type=int, default=None)
    parser.add_argument("--multi_vision_target_grid", type=int, default=None)
    parser.add_argument("--multi_vision_fusion", default="")
    parser.add_argument("--multi_vision_router_temperature", type=float, default=None)
    parser.add_argument("--multi_vision_router_hidden_ratio", type=float, default=None)
    parser.add_argument("--multi_vision_router_use_diff", type=lambda x: str(x).lower() in ("1", "true", "yes", "on"), default=None)
    parser.add_argument("--multi_vision_dropout", type=float, default=None)
    parser.add_argument("--vision_layer_fusion_indexes", nargs="*", type=int, default=None)
    parser.add_argument("--vision_layer_fusion_type", default="")
    parser.add_argument("--input_image_size", type=int, default=None)
    parser.add_argument("--disable_deepstack", action="store_true")
    parser.add_argument("--deepstack_visual_indexes", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-json-dir", default="", help="Directory for per-patch JSON files. Defaults to output-dir.")
    parser.add_argument("--merged-output-json", default="", help="Optional path for merged global map JSON.")
    parser.add_argument("--whole-map-viz-dir", default="", help="Directory for stitched whole-map visualizations. Defaults to output-json sibling whole_map_viz/.")
    parser.add_argument("--skip-whole-map-viz", action="store_true", help="Disable stitched whole-map visualization output.")
    parser.add_argument("--conv-template", default="conv_qwen_3_state_update_centerline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--coord-mode", choices=["auto", COORD_MODE_PIXEL, COORD_MODE_NORM1000], default="auto")
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--boundary-tol", type=float, default=2.0)
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument(
        "--trace-sample-distance-px",
        type=float,
        default=5.0,
        help="Stage-B incoming trace resampling stride in patch pixels. Set 0 to keep the original endpoint vertices.",
    )
    parser.add_argument("--intersection-hint-points", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-intersections", action="store_true")
    parser.add_argument("--eval-centerline", action="store_true")
    parser.add_argument("--eval-output-json", default="", help="Path for aggregate centerline metrics JSON.")
    parser.add_argument("--eval-meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--eval-buffer-size", type=float, default=1.0)
    parser.add_argument("--eval-match-threshold", type=float, default=0.33)
    parser.add_argument("--dry-run-prompts", action="store_true")
    parser.add_argument("--distributed-by-tile", action="store_true", help="Shard complete tile_id groups across torchrun ranks.")
    parser.add_argument("--rank", type=int, default=None, help="Override global rank. Defaults to RANK env.")
    parser.add_argument("--local-rank", type=int, default=None, help="Override local rank. Defaults to LOCAL_RANK env.")
    parser.add_argument("--world-size", type=int, default=None, help="Override world size. Defaults to WORLD_SIZE env.")
    parser.add_argument("--distributed-merge-timeout", type=int, default=7200, help="Seconds rank0 waits for rank summaries.")
    args = parser.parse_args()

    evaluate_records = print_eval_table = None
    evaluate_lane_intersection_records = print_lane_intersection_eval_tables = None
    if args.eval_centerline:
        from infer_index.line_eval import (
            evaluate_records,
            evaluate_lane_intersection_records,
            print_eval_table,
            print_lane_intersection_eval_tables,
        )

    rank, local_rank, world_size = distributed_info(args)
    distributed = args.distributed_by_tile and world_size > 1
    args.device = resolve_runtime_device(args.device, local_rank)

    all_records = sort_patch_records(load_json_or_jsonl(Path(args.patch_json)))
    for global_idx, record in enumerate(all_records):
        record["_mllm_global_idx"] = global_idx
    if distributed:
        records, assigned_tile_ids, all_tile_ids = shard_records_by_tile(all_records, rank, world_size)
    else:
        records = all_records
        assigned_tile_ids = sorted({record_tile_id(record) for record in records})
        all_tile_ids = assigned_tile_ids

    image_folder = Path(args.image_folder)
    checkpoint_dir = Path(args.checkpoint_dir)
    manifest = read_manifest(checkpoint_dir)
    output_json_path = Path(args.output_json)

    print(json.dumps({
        "distributed_by_tile": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": args.device,
        "num_total_records": len(all_records),
        "num_total_tiles": len(all_tile_ids),
        "num_rank_records": len(records),
        "num_rank_tiles": len(assigned_tile_ids),
        "rank_tile_preview": assigned_tile_ids[:10],
        "trace_sample_distance_px": args.trace_sample_distance_px,
    }, ensure_ascii=False))

    tokenizer = model = image_processor = None
    if not args.dry_run_prompts and records:
        config_overrides = _config_overrides_from_args(args)
        tokenizer, model, image_processor = load_model_components(checkpoint_dir, manifest, args.device, config_overrides=config_overrides)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    sample_json_dir = Path(args.sample_json_dir) if args.sample_json_dir else output_dir
    if sample_json_dir is not None:
        sample_json_dir.mkdir(parents=True, exist_ok=True)

    state_by_pos = {}
    patch_results = []
    merged_global_lines = []

    for local_idx, record in enumerate(records):
        idx = int(record.get("_mllm_global_idx", local_idx))
        row = get_meta_int(record, "row", get_meta_int(record, "patch_row", 0))
        col = get_meta_int(record, "col", get_meta_int(record, "patch_col", 0))
        meta = record.get("meta") or {}
        coord_cfg = resolve_coord_config(record, args)
        patch_size_px = coord_cfg["patch_size"]
        tile_id = record_tile_id(record)
        x0 = int(meta.get("x0", col * patch_size_px))
        y0 = int(meta.get("y0", row * patch_size_px))

        incoming_traces_pixel = build_incoming_traces(
            state_by_pos,
            tile_id,
            row,
            col,
            patch_size_px,
            args.boundary_tol,
            args.trace_points,
            args.trace_sample_distance_px,
        )
        incoming_intersections_pixel = []
        if args.include_intersections:
            incoming_intersections_pixel = build_incoming_intersections(
                state_by_pos,
                tile_id,
                row,
                col,
                patch_size_px,
                args.boundary_tol,
                args.intersection_hint_points,
            )
        incoming_traces = convert_items(
            incoming_traces_pixel,
            COORD_MODE_PIXEL,
            coord_cfg["coord_mode"],
            coord_cfg["patch_width"],
            coord_cfg["patch_height"],
            coord_range=coord_cfg["coord_range"],
            clamp=False,
        )
        incoming_intersections = convert_items(
            incoming_intersections_pixel,
            COORD_MODE_PIXEL,
            coord_cfg["coord_mode"],
            coord_cfg["patch_width"],
            coord_cfg["patch_height"],
            coord_range=coord_cfg["coord_range"],
            clamp=False,
        )
        prompt_text = make_user_prompt(
            patch_size_px,
            incoming_traces,
            incoming_intersections,
            include_intersections=args.include_intersections,
            coord_mode=coord_cfg["coord_mode"],
            coord_range=coord_cfg["coord_range"],
        )

        parse_ok = False
        parsed_lines_model = []
        parsed_lines = []
        parse_error = ""
        raw_prediction = ""
        prediction = ""
        prompt = ""
        input_token_len = 0
        output_token_len = 0
        decoded_token_len = 0
        decoded_mode = ""
        image_path = resolve_image_path(record["image"], image_folder).resolve()

        if args.dry_run_prompts:
            prompt = prompt_text
            prediction = record["conversations"][1]["value"] if len(record.get("conversations", [])) > 1 else "{}"
        else:
            image_path, prompt, raw_prediction, prediction, input_token_len, output_token_len, decoded_token_len, decoded_mode = run_patch_inference(
                record,
                prompt_text,
                image_folder,
                tokenizer,
                model,
                image_processor,
                args.conv_template,
                args.max_new_tokens,
                args.temperature,
            )

        try:
            parsed_lines_model = parse_map_json(
                prediction,
                map_task="lane_intersection" if args.include_intersections else "lane",
                patch_size=coord_cfg["patch_size"],
                coord_mode=coord_cfg["coord_mode"],
                coord_range=coord_cfg["coord_range"],
            )
            parsed_lines = convert_items(
                parsed_lines_model,
                coord_cfg["coord_mode"],
                COORD_MODE_PIXEL,
                coord_cfg["patch_width"],
                coord_cfg["patch_height"],
                coord_range=coord_cfg["coord_range"],
                clamp=True,
            )
            parse_ok = True
        except Exception as exc:
            parse_error = str(exc)
        prediction_json = payload_to_text({"lines": parsed_lines_model}) if parse_ok else extract_json_payload(prediction)
        prediction_json_pixel = payload_to_text({"lines": parsed_lines}) if parse_ok else ""

        state_by_pos[(tile_id, row, col)] = parsed_lines if parse_ok else []
        global_lines = local_to_global_lines(parsed_lines, x0, y0) if parse_ok else []
        merged_global_lines.extend(global_lines)

        result = {
            "idx": idx,
            "record_id": record.get("id", f"patch_{idx}"),
            "image": str(image_path),
            "tile_id": tile_id,
            "row": row,
            "col": col,
            "x0": x0,
            "y0": y0,
            "patch_size": coord_cfg["patch_size"],
            "patch_width": coord_cfg["patch_width"],
            "patch_height": coord_cfg["patch_height"],
            "coord_mode": coord_cfg["coord_mode"],
            "coord_range": coord_cfg["coord_range"],
            "meta": meta,
            "incoming_traces": incoming_traces,
            "incoming_traces_pixel": incoming_traces_pixel,
            "incoming_trace_sample_distance_px": args.trace_sample_distance_px,
            "incoming_intersections": incoming_intersections,
            "incoming_intersections_pixel": incoming_intersections_pixel,
            "prompt": prompt,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "prediction_json": prediction_json,
            "prediction_json_pixel": prediction_json_pixel,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "lines_local": parsed_lines,
            "lines_local_model": parsed_lines_model,
            "lines_global": global_lines,
            "input_token_len": input_token_len,
            "output_token_len": output_token_len,
            "decoded_token_len": decoded_token_len,
            "decoded_mode": decoded_mode,
        }
        if len(record.get("conversations", [])) > 1:
            result["ground_truth"] = record["conversations"][1]["value"]
            try:
                result["ground_truth_pixel"] = convert_payload_text(
                    result["ground_truth"],
                    coord_cfg["coord_mode"],
                    COORD_MODE_PIXEL,
                    coord_cfg["patch_width"],
                    coord_cfg["patch_height"],
                    coord_range=coord_cfg["coord_range"],
                    clamp=True,
                )
            except Exception:
                result["ground_truth_pixel"] = result["ground_truth"]
        patch_results.append(result)

        if sample_json_dir is not None:
            if distributed:
                out_path = sample_json_dir / f"rank{rank:05d}_{idx:06d}_{row:03d}_{col:03d}.json"
            else:
                out_path = sample_json_dir / f"{idx:04d}_{row:03d}_{col:03d}.json"
            dump_json(out_path, result)

        print(json.dumps({
            "idx": idx,
            "record_id": result["record_id"],
            "tile_id": tile_id,
            "row": row,
            "col": col,
            "num_incoming_traces": len(incoming_traces),
            "num_incoming_intersections": len(incoming_intersections),
            "parse_ok": parse_ok,
            "num_lines": len(parsed_lines),
            "parse_error": parse_error,
        }, ensure_ascii=False))

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "patch_json": args.patch_json,
        "conv_template": args.conv_template,
        "dry_run_prompts": args.dry_run_prompts,
        "include_intersections": args.include_intersections,
        "trace_sample_distance_px": args.trace_sample_distance_px,
        "distributed_by_tile": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": args.device,
        "assigned_tile_ids": assigned_tile_ids,
        "num_total_tiles": len(all_tile_ids),
        "num_total_records": len(all_records),
        "num_patches": len(patch_results),
        "merged_global": {"lines": merged_global_lines},
        "patch_results": patch_results,
    }

    if distributed:
        rank_summary_path = rank_json_path(output_json_path, rank)
        dump_json(rank_summary_path, summary)
        print(json.dumps({
            "rank": rank,
            "rank_summary_json": str(rank_summary_path),
            "num_rank_patches": len(patch_results),
        }, ensure_ascii=False))
        if rank != 0:
            return

        rank_summary_paths = wait_for_rank_summaries(
            output_json_path,
            world_size,
            args.distributed_merge_timeout,
        )
        summary = merge_rank_summaries(rank_summary_paths)
        for key in ("rank", "local_rank", "device", "assigned_tile_ids"):
            summary.pop(key, None)
        summary.update({
            "checkpoint_dir": str(checkpoint_dir),
            "patch_json": args.patch_json,
            "conv_template": args.conv_template,
            "dry_run_prompts": args.dry_run_prompts,
            "include_intersections": args.include_intersections,
            "distributed_by_tile": True,
            "world_size": world_size,
            "rank_summaries": [str(path) for path in rank_summary_paths],
            "num_total_tiles": len(all_tile_ids),
            "num_total_records": len(all_records),
        })
        patch_results = summary["patch_results"]
        merged_global_lines = summary["merged_global"]["lines"]

    if not args.skip_whole_map_viz:
        whole_map_viz_dir = Path(args.whole_map_viz_dir) if args.whole_map_viz_dir else Path(args.output_json).parent / "whole_map_viz"
        summary["whole_map_viz_dir"] = str(whole_map_viz_dir)
        summary["whole_map_visualizations"] = render_common_whole_map_visualizations(patch_results, image_folder, whole_map_viz_dir)
    if args.eval_centerline:
        eval_payload = build_eval_payload(
            patch_results,
            args,
            evaluate_records,
            evaluate_lane_intersection_records,
        )
        if args.include_intersections:
            summary.update(eval_payload)
        else:
            summary["centerline_eval"] = eval_payload
        eval_path = Path(args.eval_output_json) if args.eval_output_json else Path(args.output_json).with_name("eval.json")
        dump_json(eval_path, eval_payload)
        print_eval_payload(eval_payload, args, print_eval_table, print_lane_intersection_eval_tables)
        print(json.dumps(eval_console_payload(eval_path, eval_payload, args), ensure_ascii=False))
    if args.merged_output_json:
        dump_json(Path(args.merged_output_json), summary["merged_global"])
    dump_json(output_json_path, summary)


if __name__ == "__main__":
    main()
