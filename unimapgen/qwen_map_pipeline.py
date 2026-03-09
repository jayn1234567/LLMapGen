import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from unimapgen.data.qwen_map_dataset import OpenSatMapQwenDataset, OpenSatMapQwenDatasetConfig, QwenMapCollator


def build_qwen_map_dataset(cfg: Dict, split: str, max_samples, train_augment: bool):
    dcfg = cfg["data"]
    scfg = cfg["serialization"]
    return OpenSatMapQwenDataset(
        OpenSatMapQwenDatasetConfig(
            opensatmap_root=str(dcfg["opensatmap_root"]),
            ann_json_path=str(dcfg["opensatmap_ann_json"]),
            split=str(split),
            image_size=int(dcfg["image_size"]),
            max_samples=max_samples,
            sample_interval_meter=float(scfg["sample_interval_meter"]),
            meter_per_pixel=float(dcfg.get("meter_per_pixel", 0.15)),
            max_lines=int(scfg["max_lines"]),
            max_points_per_line=int(scfg["max_points_per_line"]),
            categories=list(scfg["categories"]),
            line_types=list(scfg.get("line_types", [])),
            max_seq_len=int(scfg["max_seq_len"]),
            coord_num_bins=scfg.get("coord_num_bins"),
            angle_num_bins=int(scfg.get("angle_num_bins", 360)),
            train_augment=bool(train_augment),
            aug_rot90_prob=float(dcfg.get("aug_rot90_prob", 0.0)) if train_augment else 0.0,
            aug_hflip_prob=float(dcfg.get("aug_hflip_prob", 0.0)) if train_augment else 0.0,
            aug_vflip_prob=float(dcfg.get("aug_vflip_prob", 0.0)) if train_augment else 0.0,
            prompt_template=str(dcfg.get("prompt_template", default_prompt_template())),
            use_state_update=bool(dcfg.get("use_state_update", False)),
            state_update_mode=str(dcfg.get("state_update_mode", "sample_prev")),
            state_prefix_mode=str(dcfg.get("state_prefix_mode", "cut_points")),
            splits_meta_path=str(dcfg.get("splits_meta_path")) if dcfg.get("splits_meta_path") else None,
            patch_geometry_json=str(dcfg.get("patch_geometry_json")) if dcfg.get("patch_geometry_json") else None,
            geometry_border_tol_px=float(dcfg.get("geometry_border_tol_px", 4.0)),
            geometry_overlap_margin_px=float(dcfg.get("geometry_overlap_margin_px", 32.0)),
            geometry_endpoint_margin_px=float(dcfg.get("geometry_endpoint_margin_px", 96.0)),
            geometry_densify_step_m=float(dcfg.get("geometry_densify_step_m", 1.0)),
            geometry_connect_radius_m=float(dcfg.get("geometry_connect_radius_m", 3.0)),
            geometry_trace_num_points=int(dcfg.get("geometry_trace_num_points", 3)),
            geometry_adjacent_source_margin_px=float(dcfg.get("geometry_adjacent_source_margin_px", 96.0)),
            geometry_adjacent_center_margin_m=float(dcfg.get("geometry_adjacent_center_margin_m", 96.0)),
            split_dir=str(dcfg.get("opensatmap_split_dir")) if dcfg.get("opensatmap_split_dir") else None,
        )
    )


def build_qwen_map_components(cfg: Dict, train_set: OpenSatMapQwenDataset):
    from unimapgen.data.qwen_map_tokenizer import QwenMapTokenizer
    from unimapgen.models.qwen_map_generator import QwenSatelliteMapGenerator

    model_cfg = cfg["model"]
    print("[Init] Loading Qwen tokenizer...", flush=True)
    print(
        f"[Init] model_cfg dino_model_path={model_cfg['dino_model_path']} "
        f"qwen_model_path={model_cfg['qwen_model_path']}",
        flush=True,
    )
    qwen_map_tokenizer = QwenMapTokenizer(
        qwen_model_path=str(model_cfg["qwen_model_path"]),
        map_tokenizer=train_set.map_tokenizer,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        trust_remote_code=True,
    )
    print("[Init] Qwen tokenizer ready", flush=True)
    collator = QwenMapCollator(qwen_map_tokenizer=qwen_map_tokenizer)
    print("[Init] Loading Qwen generator model...", flush=True)
    model = QwenSatelliteMapGenerator(
        dino_model_path=str(model_cfg["dino_model_path"]),
        qwen_model_path=str(model_cfg["qwen_model_path"]),
        vocab_size=int(qwen_map_tokenizer.vocab_size),
        allowed_map_token_ids=qwen_map_tokenizer.allowed_map_token_ids,
        map_eos_token_id=int(qwen_map_tokenizer.map_eos_token_id),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        freeze_satellite=bool(model_cfg.get("freeze_satellite", True)),
        freeze_llm=bool(model_cfg.get("freeze_llm", False)),
        sat_token_hw=tuple(model_cfg.get("sat_token_hw", [8, 8])),
        sat_patch_size=int(model_cfg.get("sat_patch_size", 14)),
        sat_drop_cls_token=bool(model_cfg.get("sat_drop_cls_token", True)),
        sat_normalize_input=bool(model_cfg.get("sat_normalize_input", True)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
    )
    if bool(model_cfg.get("semantic_init_new_map_tokens", False)):
        init_stats = model.semantic_initialize_new_embeddings(qwen_map_tokenizer=qwen_map_tokenizer)
        print(
            "[Init] Semantic init for new map tokens "
            f"(initialized={init_stats['initialized']} skipped={init_stats['skipped']})",
            flush=True,
        )
    print("[Init] Qwen generator model ready", flush=True)
    return qwen_map_tokenizer, collator, model


def compute_shift_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    if int(valid.sum().item()) == 0:
        return 0, 0
    pred = shift_logits.argmax(dim=-1)
    correct = int((pred.eq(shift_labels) & valid).sum().item())
    total = int(valid.sum().item())
    return correct, total


def maybe_load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[Init] Loaded checkpoint={checkpoint_path} "
        f"(missing={len(missing)} unexpected={len(unexpected)})"
    )


def default_prompt_template() -> str:
    return (
        "You are given a satellite image embedding. "
        "Generate a serialized vector map using only the reserved map tokens. "
        "Represent {categories}. "
        "Output at most {max_lines} polylines and at most {max_points_per_line} points per polyline."
    )


def lines_to_jsonable(lines: List[Dict]) -> List[Dict]:
    out = []
    for line in lines:
        pts = line.get("points", [])
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        out.append(
            {
                "category": line.get("category", ""),
                "line_type": line.get("line_type", ""),
                "start_type": line.get("start_type", "start"),
                "end_type": line.get("end_type", "end"),
                "points": pts,
            }
        )
    return out


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
