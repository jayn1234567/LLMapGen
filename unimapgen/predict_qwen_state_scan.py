import argparse

import torch
from tqdm import tqdm

from unimapgen.data.qwen_map_dataset import build_state_token_ids_from_lines
from unimapgen.qwen_map_pipeline import (
    build_qwen_map_components,
    build_qwen_map_dataset,
    lines_to_jsonable,
    maybe_load_model_checkpoint,
    save_json,
)
from unimapgen.state_geometry import (
    build_patch_scan_order,
    build_state_lines_from_global,
    load_patch_geometry_map,
    merge_global_lines,
    patch_lines_to_global,
    token_key,
)
from unimapgen.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--output", type=str, default="outputs/qwen_state_scan_predictions.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"[StateScan] Loaded config: {args.config}", flush=True)
    print(f"[StateScan] Building dataset split={args.split}...", flush=True)
    ds = build_qwen_map_dataset(
        cfg,
        split=str(args.split),
        max_samples=args.max_samples,
        train_augment=False,
    )
    print(f"[StateScan] Dataset ready: {len(ds)} samples", flush=True)
    print("[StateScan] Building tokenizer/model...", flush=True)
    qwen_map_tokenizer, _, model = build_qwen_map_components(cfg, train_set=ds)
    maybe_load_model_checkpoint(model, args.checkpoint)
    print(f"[StateScan] Checkpoint loaded: {args.checkpoint}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[StateScan] Device={device}", flush=True)

    dec_cfg = cfg.get("decode", {})
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else dec_cfg.get("max_new_tokens", 256))
    min_new_tokens = int(args.min_new_tokens if args.min_new_tokens is not None else dec_cfg.get("min_new_tokens", 0))
    temperature = float(args.temperature if args.temperature is not None else dec_cfg.get("temperature", 1.0))
    top_k = int(args.top_k if args.top_k is not None else dec_cfg.get("top_k", 1))
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty is not None else dec_cfg.get("repetition_penalty", 1.0)
    )
    use_grammar_constraint = bool(dec_cfg.get("use_grammar_constraint", True))
    grammar_min_points_per_line = int(dec_cfg.get("grammar_min_points_per_line", 2))
    grammar_max_lines = int(dec_cfg.get("grammar_max_lines", cfg["serialization"]["max_lines"]))

    prefix_mode = str(cfg["data"].get("state_prefix_mode", "cut_points"))
    max_lines = int(cfg["serialization"]["max_lines"])
    geometry_json = str(cfg["data"].get("patch_geometry_json", "")).strip()
    geom_map = load_patch_geometry_map(geometry_json) if geometry_json else {}
    border_tol_px = float(cfg["data"].get("geometry_border_tol_px", 4.0))
    overlap_margin_px = float(cfg["data"].get("geometry_overlap_margin_px", 24.0))
    endpoint_margin_px = float(cfg["data"].get("geometry_endpoint_margin_px", 40.0))
    densify_step_m = float(cfg["data"].get("geometry_densify_step_m", 1.0))
    connect_radius_m = float(cfg["data"].get("geometry_connect_radius_m", 3.0))
    trace_num_points = int(cfg["data"].get("geometry_trace_num_points", 3))
    adjacent_source_margin_px = float(cfg["data"].get("geometry_adjacent_source_margin_px", endpoint_margin_px))
    adjacent_center_margin_m = float(cfg["data"].get("geometry_adjacent_center_margin_m", 96.0))
    scan_indices = list(range(len(ds)))
    if geom_map:
        token_to_index = {token_key(ds.items[i]["token"]): i for i in range(len(ds))}
        ordered_tokens = build_patch_scan_order([ds.items[i]["token"] for i in range(len(ds))], geom_map)
        scan_indices = [token_to_index[tok] for tok in ordered_tokens if tok in token_to_index]
    print(
        f"[StateScan] split={args.split} scan_items={len(scan_indices)} has_geometry={bool(geom_map)} "
        f"max_new_tokens={max_new_tokens}",
        flush=True,
    )

    prev_pred_lines = []
    global_lines = []
    outputs = []
    with torch.no_grad():
        for idx in tqdm(scan_indices, desc=f"StateScan {args.split}", leave=False):
            sample = ds[idx]
            tok = token_key(sample["token"])
            if tok in geom_map:
                state_lines, state_stats = build_state_lines_from_global(
                    global_lines=global_lines,
                    geom_rec=geom_map[tok],
                    image_size=int(cfg["data"]["image_size"]),
                    meter_per_pixel=float(cfg["data"].get("meter_per_pixel", 0.15)),
                    max_lines=max_lines,
                    border_tol_px=border_tol_px,
                    overlap_margin_px=overlap_margin_px,
                    endpoint_margin_px=endpoint_margin_px,
                    densify_step_m=densify_step_m,
                    trace_num_points=trace_num_points,
                    adjacent_source_margin_px=adjacent_source_margin_px,
                    adjacent_center_margin_m=adjacent_center_margin_m,
                )
                state_source = "geometry_global_merge"
            else:
                state_lines = prev_pred_lines
                state_stats = {
                    "num_projected_lines": len(prev_pred_lines),
                    "num_endpoint_primitives": 0,
                    "num_state_lines": len(prev_pred_lines),
                }
                state_source = "prev_prediction"
            prompt_text = ds._build_prompt_text(prev_lines=state_lines)
            prompt_ids = qwen_map_tokenizer.encode_prompt(prompt_text)
            prompt_input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
            state_custom_ids = build_state_token_ids_from_lines(
                map_tokenizer=ds.map_tokenizer,
                lines=state_lines,
                prefix_mode=prefix_mode,
                max_lines=max_lines,
                trace_num_points=trace_num_points,
            )
            state_qwen_ids = qwen_map_tokenizer.encode_map_token_ids(state_custom_ids)
            state_input_ids = torch.tensor(state_qwen_ids, dtype=torch.long, device=device).unsqueeze(0)
            state_attention_mask = torch.ones_like(state_input_ids, dtype=torch.long)

            pred_qwen_ids = model.generate(
                image=sample["image"].unsqueeze(0).to(device),
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                state_input_ids=state_input_ids,
                state_attention_mask=state_attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                grammar_helper=qwen_map_tokenizer if use_grammar_constraint else None,
                grammar_min_points_per_line=grammar_min_points_per_line,
                grammar_max_lines=grammar_max_lines,
            )[0].detach().cpu().tolist()
            pred_custom_ids = qwen_map_tokenizer.decode_qwen_map_ids_to_custom_ids(pred_qwen_ids)
            pred_lines = ds.map_tokenizer.decode_to_lines(pred_custom_ids)
            gt_custom_ids = [int(x) for x in sample["map_token_ids"].tolist()]
            gt_lines = ds.map_tokenizer.decode_to_lines(gt_custom_ids)
            if tok in geom_map:
                cur_global = patch_lines_to_global(
                    lines=pred_lines,
                    geom_rec=geom_map[tok],
                    image_size=int(cfg["data"]["image_size"]),
                    meter_per_pixel=float(cfg["data"].get("meter_per_pixel", 0.15)),
                )
                global_lines = merge_global_lines(
                    global_lines=global_lines,
                    new_lines=cur_global,
                    cell_m=1.0,
                    connect_radius_m=connect_radius_m,
                )

            outputs.append(
                {
                    "token": sample["token"],
                    "prompt_text": prompt_text,
                    "state_source": state_source,
                    "state_prefix_mode": prefix_mode,
                    "state_custom_ids": state_custom_ids,
                    "num_candidate_lines": int(state_stats.get("num_candidate_lines", 0)),
                    "num_projected_lines": int(state_stats.get("num_projected_lines", len(state_lines))),
                    "num_endpoint_primitives": int(state_stats.get("num_endpoint_primitives", 0)),
                    "num_state_lines": int(state_stats.get("num_state_lines", len(state_lines))),
                    "num_global_lines": len(global_lines),
                    "has_geometry": bool(tok in geom_map),
                    "pred_qwen_ids": pred_qwen_ids,
                    "pred_custom_ids": pred_custom_ids,
                    "gt_custom_ids": gt_custom_ids,
                    "pred_lines": lines_to_jsonable(pred_lines),
                    "gt_lines": lines_to_jsonable(gt_lines),
                }
            )
            prev_pred_lines = pred_lines

    save_json(args.output, outputs)
    print(f"saved {len(outputs)} state-scan samples to {args.output}")


if __name__ == "__main__":
    main()
