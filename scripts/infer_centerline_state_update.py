#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from scripts.infer_centerline_checkpoint import (
    build_prompt,
    load_model_components,
    normalize_prediction_text,
    parse_map_json,
    read_manifest,
)


TASK_TEXT = "Please construct the complete road map in the current BEV (Bird's Eye View) image patch."


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return json.loads(text)
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
        tile_id = meta.get("tile_id", record.get("tile_id", ""))
        row = get_meta_int(record, "row", get_meta_int(record, "patch_row", 0))
        col = get_meta_int(record, "col", get_meta_int(record, "patch_col", 0))
        return str(tile_id), row, col, str(record.get("id", ""))

    return sorted(records, key=key_fn)


def make_user_prompt(patch_size: int, incoming_traces, incoming_intersections=None, include_intersections=False):
    traces_json = json.dumps(incoming_traces, ensure_ascii=False, separators=(",", ":"))
    parts = [
        "<image>",
        TASK_TEXT,
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


def near(value, target, tol):
    return abs(value - target) <= tol


def endpoint_trace(points, endpoint_index, max_points):
    if endpoint_index == 0:
        return list(reversed(points[:max_points]))
    return points[-max_points:]


def transform_trace_points(points, dx, dy):
    return [[int(round(x + dx)), int(round(y + dy))] for x, y in points]


def extract_neighbor_traces(neighbor_lines, side, patch_size, boundary_tol, max_points):
    traces = []
    for line in neighbor_lines:
        if line.get("category") != "centerline":
            continue
        points = line.get("points") or []
        if not points:
            continue

        candidates = []
        if side == "left":
            if line.get("end_type") == "cut" and near(points[-1][0], patch_size - 1, boundary_tol):
                candidates.append(endpoint_trace(points, -1, max_points))
            if line.get("start_type") == "cut" and near(points[0][0], patch_size - 1, boundary_tol):
                candidates.append(endpoint_trace(points, 0, max_points))
            dx, dy = -patch_size, 0
            trace_side = "left"
        else:
            if line.get("end_type") == "cut" and near(points[-1][1], patch_size - 1, boundary_tol):
                candidates.append(endpoint_trace(points, -1, max_points))
            if line.get("start_type") == "cut" and near(points[0][1], patch_size - 1, boundary_tol):
                candidates.append(endpoint_trace(points, 0, max_points))
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


def build_incoming_traces(state_by_pos, tile_id, row, col, patch_size, boundary_tol, max_points):
    traces = []
    left_lines = state_by_pos.get((tile_id, row, col - 1), [])
    top_lines = state_by_pos.get((tile_id, row - 1, col), [])
    traces.extend(extract_neighbor_traces(left_lines, "left", patch_size, boundary_tol, max_points))
    traces.extend(extract_neighbor_traces(top_lines, "top", patch_size, boundary_tol, max_points))
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
        if line.get("category") != "intersection" or not line.get("is_cut"):
            continue
        points = line.get("points") or []
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
        converted = dict(line)
        converted["points"] = [[int(x + x0), int(y + y0)] for x, y in line.get("points", [])]
        global_lines.append(converted)
    return global_lines


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

    prompt = build_prompt(prompt_text, conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=images_tensor,
            image_sizes=[image.size],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw_prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    prediction = normalize_prediction_text(raw_prediction)
    return image_path, prompt, raw_prediction, prediction, int(input_ids.shape[1]), int(output_ids.shape[1])


def main():
    parser = argparse.ArgumentParser(description="Patch-by-patch state-update inference for centerline/intersection maps.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--patch-json", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--conv-template", default="conv_qwen_3_state_update_centerline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--boundary-tol", type=float, default=2.0)
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument("--intersection-hint-points", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-intersections", action="store_true")
    parser.add_argument("--eval-centerline", action="store_true")
    parser.add_argument("--eval-meter-per-pixel", type=float, default=1.0)
    parser.add_argument("--eval-buffer-size", type=float, default=1.0)
    parser.add_argument("--eval-match-threshold", type=float, default=0.33)
    parser.add_argument("--dry-run-prompts", action="store_true")
    args = parser.parse_args()

    evaluate_records = None
    if args.eval_centerline:
        from scripts.centerline_eval_metrics import evaluate_records

    records = sort_patch_records(load_json_or_jsonl(Path(args.patch_json)))
    image_folder = Path(args.image_folder)
    checkpoint_dir = Path(args.checkpoint_dir)
    manifest = read_manifest(checkpoint_dir)

    tokenizer = model = image_processor = None
    if not args.dry_run_prompts:
        tokenizer, model, image_processor = load_model_components(checkpoint_dir, manifest, args.device)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    state_by_pos = {}
    patch_results = []
    merged_global_lines = []

    for idx, record in enumerate(records):
        row = get_meta_int(record, "row", get_meta_int(record, "patch_row", 0))
        col = get_meta_int(record, "col", get_meta_int(record, "patch_col", 0))
        meta = record.get("meta") or {}
        tile_id = meta.get("tile_id", record.get("tile_id", ""))
        x0 = int(meta.get("x0", col * args.patch_size))
        y0 = int(meta.get("y0", row * args.patch_size))

        incoming_traces = build_incoming_traces(
            state_by_pos,
            tile_id,
            row,
            col,
            args.patch_size,
            args.boundary_tol,
            args.trace_points,
        )
        incoming_intersections = []
        if args.include_intersections:
            incoming_intersections = build_incoming_intersections(
                state_by_pos,
                tile_id,
                row,
                col,
                args.patch_size,
                args.boundary_tol,
                args.intersection_hint_points,
            )
        prompt_text = make_user_prompt(
            args.patch_size,
            incoming_traces,
            incoming_intersections,
            include_intersections=args.include_intersections,
        )

        parse_ok = False
        parsed_lines = []
        parse_error = ""
        raw_prediction = ""
        prediction = ""
        prompt = ""
        input_token_len = 0
        output_token_len = 0
        image_path = resolve_image_path(record["image"], image_folder).resolve()

        if args.dry_run_prompts:
            prompt = prompt_text
            prediction = record["conversations"][1]["value"] if len(record.get("conversations", [])) > 1 else "{}"
        else:
            image_path, prompt, raw_prediction, prediction, input_token_len, output_token_len = run_patch_inference(
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
            parsed_lines = parse_map_json(prediction)
            parse_ok = True
        except Exception as exc:
            parse_error = str(exc)

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
            "patch_size": args.patch_size,
            "incoming_traces": incoming_traces,
            "incoming_intersections": incoming_intersections,
            "prompt": prompt,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "lines_local": parsed_lines,
            "lines_global": global_lines,
            "input_token_len": input_token_len,
            "output_token_len": output_token_len,
        }
        if len(record.get("conversations", [])) > 1:
            result["ground_truth"] = record["conversations"][1]["value"]
        patch_results.append(result)

        if output_dir is not None:
            out_path = output_dir / f"{idx:04d}_{row:03d}_{col:03d}.json"
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
        "num_patches": len(patch_results),
        "merged_global": {"lines": merged_global_lines},
        "patch_results": patch_results,
    }
    if args.eval_centerline:
        summary["centerline_eval"] = evaluate_records(
            patch_results,
            meter_per_pixel=args.eval_meter_per_pixel,
            buffer_size=args.eval_buffer_size,
            match_threshold=args.eval_match_threshold,
        )
    dump_json(Path(args.output_json), summary)


if __name__ == "__main__":
    main()
