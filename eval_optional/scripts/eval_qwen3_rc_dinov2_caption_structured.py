#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def _resolve_repo_root() -> Path:
    # 兼容最小仓库的新布局，保证 eval_optional 下的脚本也能直接导入核心模块。
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.rc_caption_short_dataset import (  # noqa: E402
    DEFAULT_GRID_COUNT,
    GRID_LABEL_BACKGROUND,
    GRID_LABEL_LANE_BOUNDARY,
    GRID_LABEL_LANE_DIVIDER,
    GRID_LABEL_MIX,
    RCCaptionShortDataset,
    RCCaptionShortFormatter,
    load_jsonl,
)
from unimapgen.models.qwen3_rc_dinov2_caption_llava import Qwen3RCDinoCaptionModel  # noqa: E402
from unimapgen.rc_llm_runtime import load_json_dict, resolve_meta_jsonl, set_random_seed  # noqa: E402

import numpy as np


SCENE_LABELS = {
    "straight",
    "curved",
    "branching",
    "intersection-approach",
    "complex",
}
GRID_LABELS = [
    GRID_LABEL_BACKGROUND,
    GRID_LABEL_LANE_BOUNDARY,
    GRID_LABEL_LANE_DIVIDER,
    GRID_LABEL_MIX,
]
INVALID_GRID_LABEL = "__invalid__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline structured evaluation for Stage-2 RC DINOv2 -> Qwen caption alignment."
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dinov2-model-name-or-path", type=str, default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--border-tol-px", type=float, default=18.0)
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # 同时读取 args.json 和 modules.pt：前者给超参，后者给视觉 token 数和 bridge 元信息。
    args_path = checkpoint_dir / "args.json"
    modules_path = checkpoint_dir / "rc_dinov2_caption_modules.pt"
    saved_args = load_json_dict(args_path) if args_path.is_file() else {}
    modules_state = torch.load(str(modules_path), map_location="cpu", weights_only=False) if modules_path.is_file() else {}
    if not isinstance(saved_args, dict):
        saved_args = {}
    if not isinstance(modules_state, dict):
        modules_state = {}
    return saved_args, modules_state


def strip_reasoning_segments(text: str) -> str:
    cleaned = str(text)
    while True:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>")
        if start < 0 or end < 0 or end < start:
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned


def clean_caption_text(text: str) -> str:
    cleaned = strip_reasoning_segments(str(text))
    for token in ("<vis_start>", "<vis_patch>", "<vis_end>"):
        cleaned = cleaned.replace(token, " ")
    cleaned = " ".join(cleaned.replace("\n", " ").split())
    return cleaned.strip()


def select_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if float(temperature) <= 0.0 or int(top_k) <= 1:
        return torch.argmax(logits, dim=-1)
    scaled = logits / max(float(temperature), 1e-6)
    k = max(1, min(int(top_k), int(scaled.shape[-1])))
    values, indices = torch.topk(scaled, k=k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return indices.gather(-1, sampled).squeeze(-1)


@torch.no_grad()
def generate_caption(
    *,
    model: Qwen3RCDinoCaptionModel,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> Dict[str, Any]:
    # Stage 2 推理时不会走官方 VL 协议，而是先构造 inputs_embeds，再直接替换 <vis_patch> 对应槽位。
    prompt_batch = tokenizer(
        [str(prompt_text)],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = prompt_batch["input_ids"].to(device)
    attention_mask = prompt_batch["attention_mask"].to(device)
    vis_patch_mask = input_ids.eq(int(model.vis_patch_token_id))
    model_dtype = next(model.parameters()).dtype
    pixel_values = pixel_values.unsqueeze(0).to(device=device, dtype=model_dtype)

    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    visual_embeddings = model.build_visual_embeddings(pixel_values)
    inputs_embeds = model.inject_visual_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        visual_embeddings=visual_embeddings.to(dtype=inputs_embeds.dtype),
        vis_patch_mask=vis_patch_mask,
    )
    outputs = model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token = select_next_token(outputs.logits[:, -1, :], temperature=float(temperature), top_k=int(top_k))
    eos_token_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    current_attention_mask = attention_mask
    generated_ids: List[int] = []

    for _ in range(int(max_new_tokens)):
        token_id = int(next_token.item())
        if token_id == eos_token_id:
            break
        generated_ids.append(token_id)
        current_token = next_token.view(1, 1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
            dim=1,
        )
        outputs = model.language_model(
            input_ids=current_token,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = select_next_token(outputs.logits[:, -1, :], temperature=float(temperature), top_k=int(top_k))

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "generated_ids": generated_ids,
        "caption_text": clean_caption_text(decoded),
    }


def parse_structured_caption(text: str) -> Dict[str, Any]:
    # 把自由生成文本压回固定的 Scene + 64 个 GridStates 结构，便于做严格分类指标。
    raw_text = str(text).strip()
    scene_match = re.search(r"Scene\s*=\s*([A-Za-z\-]+)", raw_text, flags=re.IGNORECASE)
    grid_match = re.search(r"GridStates\s*=\s*\[([^\]]*)\]", raw_text, flags=re.IGNORECASE | re.DOTALL)

    scene_label = str(scene_match.group(1)).strip().lower() if scene_match else ""
    scene_valid = scene_label in SCENE_LABELS

    raw_states: List[str] = []
    if grid_match:
        raw_states = [
            str(state).strip().lower()
            for state in str(grid_match.group(1)).replace("\n", " ").split(",")
            if str(state).strip()
        ]
    valid_states = [state for state in raw_states if state in GRID_LABELS]
    grid_length_ok = len(raw_states) == int(DEFAULT_GRID_COUNT)
    grid_values_ok = len(valid_states) == len(raw_states)
    padded_states = list(raw_states[: int(DEFAULT_GRID_COUNT)])
    if len(padded_states) < int(DEFAULT_GRID_COUNT):
        padded_states.extend([INVALID_GRID_LABEL] * (int(DEFAULT_GRID_COUNT) - len(padded_states)))
    normalized_states = [
        state if state in GRID_LABELS else INVALID_GRID_LABEL
        for state in padded_states[: int(DEFAULT_GRID_COUNT)]
    ]

    return {
        "raw_text": raw_text,
        "scene_label": scene_label if scene_valid else "",
        "scene_valid": bool(scene_valid),
        "grid_states": normalized_states,
        "grid_length_ok": bool(grid_length_ok),
        "grid_values_ok": bool(grid_values_ok),
        "parse_ok": bool(scene_valid and grid_length_ok and grid_values_ok),
    }


def compute_macro_f1(gt_states: Sequence[str], pred_states: Sequence[str]) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    for label in GRID_LABELS:
        tp = sum(1 for gt, pred in zip(gt_states, pred_states) if gt == label and pred == label)
        fp = sum(1 for gt, pred in zip(gt_states, pred_states) if gt != label and pred == label)
        fn = sum(1 for gt, pred in zip(gt_states, pred_states) if gt == label and pred != label)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        f1_values.append(f1)
    return {
        "macro_f1": float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    set_random_seed(int(args.seed))

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    modules_state_path = checkpoint_dir / "rc_dinov2_caption_modules.pt"
    if not modules_state_path.is_file():
        raise FileNotFoundError(f"Missing modules state: {modules_state_path}")

    saved_args, modules_state = load_checkpoint_metadata(checkpoint_dir)
    model_name_or_path = str(saved_args.get("model_name_or_path", "")).strip() or str(checkpoint_dir)
    dinov2_model_name_or_path = str(args.dinov2_model_name_or_path).strip() or str(
        saved_args.get("dinov2_model_name_or_path", "")
    ).strip()
    if not dinov2_model_name_or_path:
        raise ValueError(
            "dinov2_model_name_or_path is required. "
            "Either pass it explicitly or provide a checkpoint dir with args.json."
        )

    image_size = int(args.image_size) if int(args.image_size) > 0 else int(saved_args.get("image_size", 512))
    encoder_input_pad_size = int(modules_state.get("encoder_input_pad_size", saved_args.get("encoder_input_pad_size", 0)))
    visual_grid_size = int(modules_state.get("visual_grid_size", 0))
    num_visual_tokens = int(modules_state.get("num_visual_tokens", 0))
    if visual_grid_size <= 0 and num_visual_tokens > 0:
        visual_grid_size = int(round(num_visual_tokens ** 0.5))
    if visual_grid_size <= 0:
        effective = max(int(image_size), int(encoder_input_pad_size))
        visual_grid_size = max(1, int(effective) // 14)
    if num_visual_tokens <= 0:
        num_visual_tokens = int(visual_grid_size * visual_grid_size)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    formatter = RCCaptionShortFormatter(
        image_size=int(image_size),
        num_visual_tokens=int(num_visual_tokens),
    )
    formatter.register_tokens(tokenizer)

    dataset_path = Path(args.dataset_jsonl).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_path}")
    dataset_meta_path = resolve_meta_jsonl(dataset_path, args.dataset_meta_jsonl)
    rows = load_jsonl(dataset_path, max_samples=int(args.max_samples))
    meta_rows = load_jsonl(dataset_meta_path) if dataset_meta_path is not None else []
    dataset = RCCaptionShortDataset(
        rows=rows,
        meta_rows=meta_rows,
        media_dir=Path(args.media_dir).resolve(),
        tokenizer=tokenizer,
        formatter=formatter,
        image_size=int(image_size),
        border_tol_px=float(args.border_tol_px),
    )

    model = Qwen3RCDinoCaptionModel(
        model_name_or_path=str(model_name_or_path),
        tokenizer=tokenizer,
        dinov2_model_name_or_path=str(dinov2_model_name_or_path),
        modules_state_path=str(modules_state_path),
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        visual_projector_hidden_dim=int(saved_args.get("visual_projector_hidden_dim", 4096)),
        geometric_mlp_hidden_dim=int(saved_args.get("geometric_mlp_hidden_dim", 512)),
        token_alignment_hidden_dim=int(saved_args.get("token_alignment_hidden_dim", 4096)),
        token_alignment_num_layers=int(saved_args.get("token_alignment_num_layers", 2)),
        token_alignment_dropout=float(saved_args.get("token_alignment_dropout", 0.0)),
        language_model_dtype=str(saved_args.get("model_dtype", "auto")),
        local_files_only=True,
        freeze_language_model=True,
        freeze_vision_encoder=True,
        encoder_input_pad_size=int(encoder_input_pad_size),
        train_full_token_embeddings=bool(saved_args.get("train_full_token_embeddings", False)),
    )
    device = str(args.device)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.bfloat16 if bool(saved_args.get("bf16", False)) else None)
    else:
        model = model.to(device=device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    num_samples = 0
    scene_correct = 0
    exact_match = 0
    parse_ok_count = 0
    gt_all_states: List[str] = []
    pred_all_states: List[str] = []
    prediction_rows: List[Dict[str, Any]] = []

    # 这里按样本逐条生成并解析，再把 scene / grid 两层指标统一汇总。
    with predictions_path.open("w", encoding="utf-8") as f:
        for sample in tqdm(dataset, desc="stage2-eval", dynamic_ncols=True):
            pred = generate_caption(
                model=model,
                tokenizer=tokenizer,
                pixel_values=sample.pixel_values,
                prompt_text=sample.prompt_text,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                device=device,
            )

            gt_struct = parse_structured_caption(sample.caption_short)
            pred_struct = parse_structured_caption(pred["caption_text"])
            gt_scene = str(sample.caption_label).strip().lower() or str(gt_struct["scene_label"])
            pred_scene = str(pred_struct["scene_label"])
            gt_states = list(gt_struct["grid_states"])
            pred_states = list(pred_struct["grid_states"])

            sample_scene_correct = bool(gt_scene == pred_scene)
            sample_cell_correct = int(sum(1 for gt, pred_state in zip(gt_states, pred_states) if gt == pred_state))
            sample_exact_match = bool(sample_scene_correct and all(gt == pred_state for gt, pred_state in zip(gt_states, pred_states)))

            num_samples += 1
            parse_ok_count += int(pred_struct["parse_ok"])
            scene_correct += int(sample_scene_correct)
            exact_match += int(sample_exact_match)
            gt_all_states.extend(gt_states)
            pred_all_states.extend(pred_states)

            row = {
                "id": sample.sample_id,
                "image_path": str(sample.image_path),
                "gt_caption_short": sample.caption_short,
                "pred_caption_short": pred["caption_text"],
                "gt_scene": gt_scene,
                "pred_scene": pred_scene,
                "scene_correct": sample_scene_correct,
                "grid_cell_correct": sample_cell_correct,
                "grid_cell_total": int(DEFAULT_GRID_COUNT),
                "exact_match": sample_exact_match,
                "pred_parse_ok": bool(pred_struct["parse_ok"]),
                "pred_grid_length_ok": bool(pred_struct["grid_length_ok"]),
                "pred_grid_values_ok": bool(pred_struct["grid_values_ok"]),
                "gt_grid_states": gt_states,
                "pred_grid_states": pred_states,
                "generated_ids": pred["generated_ids"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            prediction_rows.append(row)

    macro_f1_result = compute_macro_f1(gt_all_states, pred_all_states)
    total_cells = int(len(gt_all_states))
    grid_cell_correct = int(sum(1 for gt, pred_state in zip(gt_all_states, pred_all_states) if gt == pred_state))
    metrics = {
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_jsonl": str(dataset_path),
        "dataset_meta_jsonl": str(dataset_meta_path) if dataset_meta_path is not None else "",
        "media_dir": str(Path(args.media_dir).resolve()),
        "output_dir": str(output_dir),
        "num_samples": int(num_samples),
        "scene_acc": float(scene_correct / num_samples) if num_samples > 0 else 0.0,
        "grid_cell_acc": float(grid_cell_correct / total_cells) if total_cells > 0 else 0.0,
        "macro_f1": float(macro_f1_result["macro_f1"]),
        "exact_match": float(exact_match / num_samples) if num_samples > 0 else 0.0,
        "parse_ok_rate": float(parse_ok_count / num_samples) if num_samples > 0 else 0.0,
        "grid_total_cells": int(total_cells),
        "scene_correct": int(scene_correct),
        "grid_cell_correct": int(grid_cell_correct),
        "exact_match_count": int(exact_match),
        "per_class": macro_f1_result["per_class"],
        "predictions_jsonl": str(predictions_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
