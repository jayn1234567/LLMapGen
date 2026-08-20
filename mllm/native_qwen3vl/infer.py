from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from PIL import Image
import torch

from infer_index.line_eval import evaluate_lane_intersection_records, evaluate_records
from mllm.coord_utils import (
    COORD_MODE_NORM1000,
    COORD_MODE_PIXEL,
    DEFAULT_COORD_RANGE,
    convert_items,
    convert_payload_text,
    payload_to_text,
    record_coord_config,
)
from mllm.reward.map_schema import extract_json_payload, parse_map_json
from mllm.torch_runtime import maybe_disable_cudnn_from_env

from .data import build_qwen3vl_messages, load_json_or_jsonl, strip_image_token
from .modeling import (
    generate_one,
    load_native_model,
    load_processor,
    move_inputs_to_device,
    select_device,
)


maybe_disable_cudnn_from_env(torch)


def normalize_prediction_text(raw: str) -> str:
    text = str(raw or "").strip()
    for token in ("<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"):
        text = text.replace(token, "")
    return text.strip()


def resolve_image_path(raw_path: str, image_folder: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    return (Path(image_folder) / path).resolve()


def resolve_coord(record: dict[str, Any], args) -> dict[str, Any]:
    cfg = record_coord_config(
        record,
        default_mode=COORD_MODE_NORM1000,
        default_patch_size=args.default_patch_size,
        default_coord_range=args.coord_range,
    )
    if args.coord_mode != "auto":
        cfg["coord_mode"] = args.coord_mode
    return cfg


def record_origin(record: dict[str, Any]) -> tuple[int, int]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    x0 = int(record.get("x0") or meta.get("x0") or 0)
    y0 = int(record.get("y0") or meta.get("y0") or 0)
    return x0, y0


def record_row_col(record: dict[str, Any], coord_cfg: dict[str, Any]) -> tuple[int, int]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    x0, y0 = record_origin(record)
    row = int(record.get("row", meta.get("row", meta.get("patch_row", y0 // max(coord_cfg["patch_height"], 1)))))
    col = int(record.get("col", meta.get("col", meta.get("patch_col", x0 // max(coord_cfg["patch_width"], 1)))))
    return row, col


def sort_records(records: list[dict[str, Any]], args) -> list[tuple[int, dict[str, Any]]]:
    def key(pair):
        idx, record = pair
        coord_cfg = resolve_coord(record, args)
        row, col = record_row_col(record, coord_cfg)
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        return str(meta.get("tile_id", record.get("tile_id", ""))), row, col, idx

    return sorted(enumerate(records), key=key)


def line_category(item: dict[str, Any]) -> str:
    category = str(item.get("category", "centerline")).strip().lower()
    return "centerline" if category in {"centerline", "center_line", "lane", ""} else category


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def resample_polyline(points: list[list[int]], distance_px: float) -> list[list[int]]:
    if distance_px <= 0 or len(points) <= 2:
        return [[int(round(x)), int(round(y))] for x, y in points]
    lengths = [_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = sum(lengths)
    if total <= 0:
        return [[int(round(points[-1][0])), int(round(points[-1][1]))]]
    targets = [0.0]
    current = distance_px
    while current < total:
        targets.append(current)
        current += distance_px
    targets.append(total)

    sampled = []
    seg_idx = 0
    seg_start_dist = 0.0
    for target in targets:
        while seg_idx < len(lengths) - 1 and seg_start_dist + lengths[seg_idx] < target:
            seg_start_dist += lengths[seg_idx]
            seg_idx += 1
        p0 = points[seg_idx]
        p1 = points[seg_idx + 1]
        seg_len = max(lengths[seg_idx], 1e-6)
        t = (target - seg_start_dist) / seg_len
        x = float(p0[0]) + (float(p1[0]) - float(p0[0])) * t
        y = float(p0[1]) + (float(p1[1]) - float(p0[1])) * t
        point = [int(round(x)), int(round(y))]
        if not sampled or sampled[-1] != point:
            sampled.append(point)
    return sampled


def boundary_match(point: list[int], side: str, width: int, height: int, tol: float) -> bool:
    x, y = float(point[0]), float(point[1])
    if side == "left":
        return x >= width - 1 - tol
    if side == "top":
        return y >= height - 1 - tol
    return False


def transform_neighbor_point(point: list[int], side: str, width: int, height: int) -> list[int]:
    x, y = int(round(point[0])), int(round(point[1]))
    if side == "left":
        return [x - width, y]
    if side == "top":
        return [x, y - height]
    return [x, y]


def trace_from_line(line: dict[str, Any], side: str, width: int, height: int, tol: float) -> list[list[int]] | None:
    points = line.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    start_cut = line.get("start_type") == "cut"
    end_cut = line.get("end_type") == "cut"
    first_is_boundary = boundary_match(points[0], side, width, height, tol)
    last_is_boundary = boundary_match(points[-1], side, width, height, tol)
    if first_is_boundary and start_cut:
        ordered = list(reversed(points))
    elif last_is_boundary and end_cut:
        ordered = list(points)
    elif first_is_boundary:
        ordered = list(reversed(points))
    elif last_is_boundary:
        ordered = list(points)
    else:
        return None
    return [transform_neighbor_point(point, side, width, height) for point in ordered]


def convert_hint_items(items: list[dict[str, Any]], coord_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if coord_cfg["coord_mode"] == COORD_MODE_PIXEL:
        return items
    return convert_items(
        items,
        COORD_MODE_PIXEL,
        coord_cfg["coord_mode"],
        coord_cfg["patch_width"],
        coord_cfg["patch_height"],
        coord_range=coord_cfg["coord_range"],
        clamp=False,
    )


def extract_neighbor_traces(
    neighbor_lines: list[dict[str, Any]],
    side: str,
    coord_cfg: dict[str, Any],
    *,
    boundary_tol: float,
    max_points: int,
    sample_distance_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = coord_cfg["patch_width"], coord_cfg["patch_height"]
    traces_pixel = []
    for line in neighbor_lines:
        if line_category(line) != "centerline":
            continue
        trace_points = trace_from_line(line, side, width, height, boundary_tol)
        if not trace_points:
            continue
        sampled = resample_polyline(trace_points, sample_distance_px)
        sampled = sampled[-max_points:] if max_points > 0 else sampled
        if len(sampled) >= 1:
            traces_pixel.append({"side": side, "points": sampled, "id": f"T{len(traces_pixel)}"})
    return convert_hint_items(traces_pixel, coord_cfg), traces_pixel


def extract_neighbor_intersections(
    neighbor_lines: list[dict[str, Any]],
    side: str,
    coord_cfg: dict[str, Any],
    *,
    boundary_tol: float,
    max_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = coord_cfg["patch_width"], coord_cfg["patch_height"]
    intersections_pixel = []
    for item in neighbor_lines:
        if line_category(item) != "intersection" or item.get("is_cut") is not True:
            continue
        points = item.get("points")
        if not isinstance(points, list):
            continue
        boundary_points = [
            transform_neighbor_point(point, side, width, height)
            for point in points
            if isinstance(point, list) and len(point) == 2 and boundary_match(point, side, width, height, boundary_tol)
        ]
        if not boundary_points:
            continue
        selected = boundary_points[-max_points:] if max_points > 0 else boundary_points
        intersections_pixel.append({"side": side, "points": selected, "id": f"I{len(intersections_pixel)}"})
    return convert_hint_items(intersections_pixel, coord_cfg), intersections_pixel


def coord_description(coord_cfg: dict[str, Any]) -> str:
    if coord_cfg["coord_mode"] == COORD_MODE_NORM1000:
        return f"Coordinates use a normalized 0-{coord_cfg['coord_range']} grid over the original {coord_cfg['patch_width']}x{coord_cfg['patch_height']} image patch."
    return f"Coordinates use patch-local pixel coordinates over the original {coord_cfg['patch_width']}x{coord_cfg['patch_height']} image patch."


def build_state_update_prompt(
    incoming_traces: list[dict[str, Any]],
    incoming_intersections: list[dict[str, Any]],
    coord_cfg: dict[str, Any],
    include_intersections: bool,
) -> str:
    target = "complete road map" if include_intersections else "road centerlines"
    return (
        "<image>\n"
        f"Please construct the {target} in the current BEV (Bird's Eye View) image patch.\n"
        f"{coord_description(coord_cfg)}\n\n"
        f"Incoming traces JSON:\n{json.dumps(incoming_traces, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"Incoming intersections JSON:\n{json.dumps(incoming_intersections, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Each incoming trace has 1 to 3 points. If multiple points are present, they are ordered from the previous patch interior toward the current patch boundary.\n"
        "Incoming traces are continuity hints only; they may be incomplete or absent.\n"
        "Each incoming intersection has 1 to 3 boundary points from neighboring patches."
    )


def parse_prediction(prediction: str, map_task: str, coord_cfg: dict[str, Any]):
    parsed = parse_map_json(
        prediction,
        map_task=map_task,
        patch_size=coord_cfg["patch_size"],
        coord_mode=coord_cfg["coord_mode"],
        coord_range=coord_cfg["coord_range"],
    )
    if not parsed.ok:
        return False, [], [], extract_json_payload(prediction), "", parsed.error or ""
    parsed_model = parsed.items
    parsed_pixel = convert_items(
        parsed_model,
        coord_cfg["coord_mode"],
        COORD_MODE_PIXEL,
        coord_cfg["patch_width"],
        coord_cfg["patch_height"],
        coord_range=coord_cfg["coord_range"],
        clamp=True,
    )
    return True, parsed_model, parsed_pixel, payload_to_text({"lines": parsed_model}), payload_to_text({"lines": parsed_pixel}), ""


def local_to_global(lines: list[dict[str, Any]], x0: int, y0: int) -> list[dict[str, Any]]:
    global_lines = []
    for line in lines:
        copied = dict(line)
        points = copied.get("points")
        if isinstance(points, list):
            copied["points"] = [[int(x + x0), int(y + y0)] for x, y in points]
        global_lines.append(copied)
    return global_lines


def build_result(
    *,
    record: dict[str, Any],
    idx: int,
    image_path: Path,
    image_paths: list[Path] | None = None,
    prompt: str,
    rendered_prompt: str,
    raw_prediction: str,
    input_token_len: int,
    output_token_len: int,
    decoded_token_len: int,
    coord_cfg: dict[str, Any],
    map_task: str,
    checkpoint_dir: str,
    incoming_traces=None,
    incoming_traces_pixel=None,
    incoming_intersections=None,
    incoming_intersections_pixel=None,
):
    prediction = normalize_prediction_text(raw_prediction)
    parse_ok, parsed_model, parsed_pixel, prediction_json, prediction_json_pixel, parse_error = parse_prediction(
        prediction,
        map_task,
        coord_cfg,
    )
    x0, y0 = record_origin(record)
    row, col = record_row_col(record, coord_cfg)
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    result = {
        "idx": idx,
        "checkpoint_dir": checkpoint_dir,
        "record_id": record.get("id", f"sample_{idx}"),
        "image": str(image_path),
        "images": [str(path) for path in (image_paths or [image_path])],
        "tile_id": meta.get("tile_id", record.get("tile_id", "tile")),
        "row": row,
        "col": col,
        "x0": x0,
        "y0": y0,
        "meta": meta,
        "coord_mode": coord_cfg["coord_mode"],
        "coord_range": coord_cfg["coord_range"],
        "patch_size": coord_cfg["patch_size"],
        "patch_width": coord_cfg["patch_width"],
        "patch_height": coord_cfg["patch_height"],
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "raw_prediction": raw_prediction,
        "prediction": prediction,
        "prediction_json": prediction_json,
        "prediction_json_pixel": prediction_json_pixel,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "num_items": len(parsed_model) if parse_ok else 0,
        "lines_local": parsed_pixel,
        "lines_local_model": parsed_model,
        "lines_global": local_to_global(parsed_pixel, x0, y0) if parse_ok else [],
        "input_token_len": input_token_len,
        "output_token_len": output_token_len,
        "decoded_token_len": decoded_token_len,
        "native_backend": "qwen3vl",
    }
    if incoming_traces is not None:
        result["incoming_traces"] = incoming_traces
        result["incoming_traces_pixel"] = incoming_traces_pixel or []
    if incoming_intersections is not None:
        result["incoming_intersections"] = incoming_intersections
        result["incoming_intersections_pixel"] = incoming_intersections_pixel or []
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
    return result


def run_one(model, processor, record, idx, prompt, args, device):
    image_values = record.get("images")
    if image_values is None:
        image_values = record.get("image")
    if isinstance(image_values, str):
        image_values = [image_values]
    if not isinstance(image_values, list) or not image_values:
        raise ValueError(f"record has no image/images: {record.get('id', idx)}")
    image_paths = [resolve_image_path(value, args.image_folder) for value in image_values]
    generated = generate_one(
        model,
        processor,
        image_paths,
        prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )
    return image_paths, generated


def _merge_native_processor_inputs(processor_inputs, pad_token_id: int):
    """Merge single-example native processor outputs like the training collator."""
    sequence_keys = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids", "position_ids"}
    # Keep this list aligned with NativeQwen3VLDataCollator: image/grid
    # tensors are flattened in sample order, while other tensors with equal
    # shapes retain a batch dimension.
    image_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    if not processor_inputs:
        raise ValueError("Cannot merge an empty native inference batch")

    def squeeze_sequence(value):
        if value.ndim >= 2 and value.shape[0] == 1:
            return value[0]
        return value

    input_lengths = []
    merged = {}
    keys = sorted(set().union(*(item.keys() for item in processor_inputs)))
    for key in keys:
        values = [item[key] for item in processor_inputs if key in item]
        if len(values) != len(processor_inputs):
            raise ValueError(f"Native processor input {key!r} is missing from some samples")
        if not torch.is_tensor(values[0]):
            continue
        if key in sequence_keys:
            sequences = [squeeze_sequence(value) for value in values]
            if any(value.ndim != 1 for value in sequences):
                raise ValueError(f"Native sequence input {key!r} is not one-dimensional")
            width = max(int(value.numel()) for value in sequences)
            padding_value = pad_token_id if key == "input_ids" else 0
            padded = [
                torch.nn.functional.pad(value, (width - int(value.numel()), 0), value=padding_value)
                for value in sequences
            ]
            merged[key] = torch.stack(padded, dim=0)
            if key == "input_ids":
                input_lengths = [int(value.numel()) for value in sequences]
        elif key in image_keys:
            if key.endswith("grid_thw"):
                values = [
                    value[0]
                    if value.ndim == 3 and value.shape[0] == 1
                    else value
                    for value in values
                ]
            merged[key] = torch.cat(values, dim=0)
        elif all(value.shape == values[0].shape for value in values):
            merged[key] = torch.stack(values, dim=0)
        else:
            raise ValueError(
                f"Cannot merge native processor input {key!r}: "
                f"{[tuple(value.shape) for value in values]}"
            )
    if not input_lengths:
        raise ValueError("Native processor output has no input_ids")
    return merged, input_lengths


def _completion_token_ids(output_ids, input_ids, attention_mask, row_index: int):
    """Extract one completion for padded, unpadded, or completion-only outputs."""
    output = output_ids[row_index] if output_ids.ndim == 2 else output_ids
    padded_prompt = input_ids[row_index] if input_ids.ndim == 2 else input_ids
    mask = attention_mask[row_index] if attention_mask is not None and attention_mask.ndim == 2 else attention_mask
    prompt = padded_prompt[mask.bool()] if mask is not None else padded_prompt
    for candidate in (padded_prompt, prompt):
        if output.numel() >= candidate.numel() and torch.equal(output[:candidate.numel()], candidate):
            return output[candidate.numel():]
    return output


def run_batch(model, processor, batch_records, args, device, *, batch_number=1, total_batches=1):
    """Run one native generate call for a phase-A batch."""
    batch_size = len(batch_records)
    image_count = 0
    first_id = batch_records[0][1].get("id", batch_records[0][0]) if batch_records else "none"
    print(
        f"[native-infer] batch={batch_number}/{total_batches} prepare_start "
        f"samples={batch_size} first_id={first_id}",
        flush=True,
    )
    prepare_started = time.perf_counter()
    prompts = []
    image_paths_batch = []
    rendered_prompts = []
    processor_inputs = []
    for idx, record in batch_records:
        image_values = record.get("images")
        if image_values is None:
            image_values = record.get("image")
        if isinstance(image_values, str):
            image_values = [image_values]
        if not isinstance(image_values, list) or not image_values:
            raise ValueError(f"record has no image/images: {record.get('id', idx)}")
        image_count += len(image_values)
        image_paths = [resolve_image_path(value, args.image_folder) for value in image_values]
        prompt = record["conversations"][0]["value"] if record.get("conversations") else args.prompt
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        messages = build_qwen3vl_messages(prompt, image_paths, None, args.system_prompt)
        rendered_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_input = processor(
            text=[rendered_prompt],
            images=images,
            return_tensors="pt",
        )
        prompts.append(prompt)
        image_paths_batch.append(image_paths)
        rendered_prompts.append(rendered_prompt)
        processor_inputs.append(processor_input)

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = int(pad_token_id if pad_token_id is not None else 0)
    inputs, input_lengths = _merge_native_processor_inputs(processor_inputs, pad_token_id)
    padded_input_len = int(inputs["input_ids"].shape[1])
    inputs = move_inputs_to_device(inputs, device)
    prepare_elapsed = time.perf_counter() - prepare_started
    print(
        f"[native-infer] batch={batch_number}/{total_batches} prepare_done "
        f"samples={batch_size} images={image_count} padded_input_tokens={padded_input_len} "
        f"elapsed={prepare_elapsed:.2f}s",
        flush=True,
    )

    generation_config = getattr(model, "generation_config", None)
    generation_pad_id = getattr(generation_config, "pad_token_id", None) or pad_token_id
    eos_token_id = getattr(generation_config, "eos_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": args.temperature > 0,
        "num_beams": 1,
        "pad_token_id": generation_pad_id,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature

    print(
        f"[native-infer] batch={batch_number}/{total_batches} generate_start "
        f"samples={batch_size} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    generate_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
    generate_elapsed = time.perf_counter() - generate_started
    print(
        f"[native-infer] batch={batch_number}/{total_batches} generate_done "
        f"samples={batch_size} elapsed={generate_elapsed:.2f}s",
        flush=True,
    )
    if output_ids.ndim == 1:
        output_ids = output_ids.unsqueeze(0)
    if int(output_ids.shape[0]) != len(batch_records):
        raise RuntimeError(
            "Native generate returned an unexpected batch dimension: "
            f"expected={len(batch_records)}, actual={int(output_ids.shape[0])}"
        )
    results = []
    for row, (idx, record) in enumerate(batch_records):
        coord_cfg = resolve_coord(record, args)
        completion_ids = _completion_token_ids(
            output_ids,
            inputs["input_ids"],
            inputs.get("attention_mask"),
            row,
        )
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(completion_ids.unsqueeze(0), skip_special_tokens=False)[0]
        else:
            decoded = tokenizer.batch_decode(completion_ids.unsqueeze(0), skip_special_tokens=False)[0]
        raw_prediction = normalize_prediction_text(decoded)
        results.append(build_result(
            record=record,
            idx=idx,
            image_path=image_paths_batch[row][0],
            image_paths=image_paths_batch[row],
            prompt=prompts[row],
            rendered_prompt=rendered_prompts[row],
            raw_prediction=raw_prediction,
            input_token_len=input_lengths[row],
            output_token_len=input_lengths[row] + int(completion_ids.numel()),
            decoded_token_len=int(completion_ids.numel()),
            coord_cfg=coord_cfg,
            map_task=args.map_task,
            checkpoint_dir=args.model_name_or_path,
        ))
    return results


def run_phase_a(model, processor, records, args, device):
    results = []
    batch_size = int(args.per_device_infer_batch_size)
    if batch_size < 1:
        raise ValueError("--per-device-infer-batch-size must be >= 1")
    total_records = len(records)
    total_batches = math.ceil(total_records / batch_size) if total_records else 0
    phase_started = time.perf_counter()
    for batch_number, start in enumerate(range(0, total_records, batch_size), start=1):
        batch_results = run_batch(
            model,
            processor,
            records[start:start + batch_size],
            args,
            device,
            batch_number=batch_number,
            total_batches=total_batches,
        )
        results.extend(batch_results)
        elapsed = max(time.perf_counter() - phase_started, 1e-9)
        print(
            f"[native-infer] progress processed={len(results)}/{total_records} "
            f"batch={batch_number}/{total_batches} "
            f"throughput={len(results) / elapsed:.2f} samples/s/npu",
            flush=True,
        )
    return results


def run_phase_b(model, processor, records, args, device):
    results = []
    state_by_pos: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for idx, record in records:
        coord_cfg = resolve_coord(record, args)
        row, col = record_row_col(record, coord_cfg)
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        tile_id = str(meta.get("tile_id", record.get("tile_id", "tile")))
        traces, traces_pixel = [], []
        intersections, intersections_pixel = [], []

        left_lines = state_by_pos.get((tile_id, row, col - 1), [])
        top_lines = state_by_pos.get((tile_id, row - 1, col), [])
        if left_lines:
            one, one_pixel = extract_neighbor_traces(
                left_lines,
                "left",
                coord_cfg,
                boundary_tol=args.boundary_tol,
                max_points=args.trace_points,
                sample_distance_px=args.trace_sample_distance_px,
            )
            traces.extend(one)
            traces_pixel.extend(one_pixel)
            if args.include_intersections:
                one_i, one_i_pixel = extract_neighbor_intersections(
                    left_lines,
                    "left",
                    coord_cfg,
                    boundary_tol=args.boundary_tol,
                    max_points=args.intersection_points,
                )
                intersections.extend(one_i)
                intersections_pixel.extend(one_i_pixel)
        if top_lines:
            one, one_pixel = extract_neighbor_traces(
                top_lines,
                "top",
                coord_cfg,
                boundary_tol=args.boundary_tol,
                max_points=args.trace_points,
                sample_distance_px=args.trace_sample_distance_px,
            )
            traces.extend(one)
            traces_pixel.extend(one_pixel)
            if args.include_intersections:
                one_i, one_i_pixel = extract_neighbor_intersections(
                    top_lines,
                    "top",
                    coord_cfg,
                    boundary_tol=args.boundary_tol,
                    max_points=args.intersection_points,
                )
                intersections.extend(one_i)
                intersections_pixel.extend(one_i_pixel)

        prompt = build_state_update_prompt(
            traces,
            intersections,
            coord_cfg,
            include_intersections=args.include_intersections,
        )
        image_paths, generated = run_one(model, processor, record, idx, prompt, args, device)
        result = build_result(
            record=record,
            idx=idx,
            image_path=image_paths[0],
            image_paths=image_paths,
            prompt=prompt,
            rendered_prompt=generated["rendered_prompt"],
            raw_prediction=generated["raw_prediction"],
            input_token_len=generated["input_token_len"],
            output_token_len=generated["output_token_len"],
            decoded_token_len=generated["decoded_token_len"],
            coord_cfg=coord_cfg,
            map_task=args.map_task,
            checkpoint_dir=args.model_name_or_path,
            incoming_traces=traces,
            incoming_traces_pixel=traces_pixel,
            incoming_intersections=intersections,
            incoming_intersections_pixel=intersections_pixel,
        )
        state_by_pos[(tile_id, row, col)] = result["lines_local"] if result["parse_ok"] else []
        results.append(result)
    return results


def write_outputs(results: list[dict[str, Any]], args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_path = output_dir / "summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(results)
    parse_ok = sum(1 for item in results if item.get("parse_ok"))
    eval_payload = {
        "summary_json": str(summary_path),
        "total": total,
        "parse_ok": parse_ok,
        "parse_ok_rate": parse_ok / total if total else 0.0,
        "native_backend": "qwen3vl",
    }
    if not args.skip_eval:
        if args.map_task == "lane_intersection":
            map_eval = evaluate_lane_intersection_records(
                results,
                meter_per_pixel=args.eval_meter_per_pixel,
                buffer_size=args.eval_buffer_size,
                match_threshold=args.eval_match_threshold,
            )
            eval_payload.update({
                "centerline_eval": map_eval["lane"],
                "intersection_eval": map_eval["intersection"],
                "lane_intersection_eval": map_eval["lane_intersection"],
                "map_eval": map_eval,
            })
        else:
            eval_payload["line_eval"] = evaluate_records(
                results,
                meter_per_pixel=args.eval_meter_per_pixel,
                buffer_size=args.eval_buffer_size,
                match_threshold=args.eval_match_threshold,
            )
    eval_path = output_dir / "eval_summary.json"
    eval_path.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_visualize:
        viz_dir = output_dir / "viz"
        cmd = [
            sys.executable,
            "scripts/tools/visualize_centerline.py",
            "--input-dir",
            str(output_dir),
            "--image-folder",
            str(args.image_folder),
            "--output-dir",
            str(viz_dir),
            "--map-task",
            args.map_task,
            "--eval-output-json",
            str(output_dir / "visualize_eval_summary.json"),
        ]
        if args.skip_whole_map_viz:
            cmd.append("--skip-whole-map-viz")
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(f"[WARN] visualization failed with exit code {proc.returncode}", file=sys.stderr)
    return summary_path, eval_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=["phase_a", "phase_b"], default="phase_a")
    parser.add_argument("--map-task", choices=["lane", "lane_intersection"], default="lane_intersection")
    parser.add_argument("--prompt", default="<image>\nPlease construct the complete road map in the current BEV image patch.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument(
        "--per-device-infer-batch-size",
        type=int,
        default=int(os.environ.get("PER_DEVICE_INFER_BATCH_SIZE", "1")),
        help="Number of samples passed to one generate() call on each device.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm1000"], default="auto")
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    parser.add_argument("--default-patch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--include-intersections", action="store_true")
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument("--trace-sample-distance-px", type=float, default=5.0)
    parser.add_argument("--intersection-points", type=int, default=3)
    parser.add_argument("--boundary-tol", type=float, default=3.0)
    parser.add_argument("--eval-meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--eval-buffer-size", type=float, default=1.0)
    parser.add_argument("--eval-match-threshold", type=float, default=0.33)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-whole-map-viz", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


class _InferenceModelArgs:
    def __init__(self, args):
        self.model_name_or_path = args.model_name_or_path
        self.model_base = args.model_base
        self.trust_remote_code = args.trust_remote_code
        self.attn_implementation = None


class _InferenceTrainingArgs:
    def __init__(self, args):
        self.bf16 = args.bf16
        self.fp16 = not args.bf16


def main():
    args = parse_args()
    records = load_json_or_jsonl(args.test_json)
    if args.num_samples and args.num_samples > 0:
        records = records[:args.num_samples]
    indexed_records = sort_records(records, args) if args.phase == "phase_b" else list(enumerate(records))

    processor_path = args.model_name_or_path
    if not Path(processor_path, "preprocessor_config.json").is_file() and args.model_base:
        processor_path = args.model_base
    processor = load_processor(processor_path, trust_remote_code=args.trust_remote_code)
    device = select_device(args.device)
    model = load_native_model(_InferenceModelArgs(args), _InferenceTrainingArgs(args), device_map=None)
    model.to(device)
    model.eval()
    if args.map_task == "lane_intersection":
        args.include_intersections = True
    if args.per_device_infer_batch_size < 1:
        raise ValueError("--per-device-infer-batch-size must be >= 1")

    print(
        f"[native-infer] model_ready device={device} samples={len(indexed_records)} "
        f"batch_size={args.per_device_infer_batch_size} phase={args.phase}",
        flush=True,
    )

    inference_started = time.perf_counter()
    if args.phase == "phase_b":
        results = run_phase_b(model, processor, indexed_records, args, device)
    else:
        results = run_phase_a(model, processor, indexed_records, args, device)
    inference_elapsed = max(time.perf_counter() - inference_started, 1e-9)
    print(f"[native-infer] per-device batch size: {args.per_device_infer_batch_size}", flush=True)
    print(f"DI_throughput: {len(results) / inference_elapsed:.2f} samples/s/npu", flush=True)
    summary_path, eval_path = write_outputs(results, args)
    print(f"Native Qwen3-VL inference summary: {summary_path}")
    print(f"Native Qwen3-VL eval summary: {eval_path}")


if __name__ == "__main__":
    main()
