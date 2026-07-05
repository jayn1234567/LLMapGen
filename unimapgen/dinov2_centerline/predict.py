"""Inference entry for the cleaned DINOv2 centerline JSON SFT route."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from unimapgen.data.rc_centerline_json_sft_dataset import (
    RCCenterlineJSONSFTDataset,
    RCCenterlineJSONSFTFormatter,
    default_system_prompt_for_task,
    default_user_prompt_for_task,
    load_jsonl,
    normalize_centerline_json_text,
)
from unimapgen.dinov2_centerline.model import Qwen3RCDinoCenterlineJSONSFTModel
from unimapgen.rc_llm_runtime import infer_visual_layout, load_json_dict, resolve_meta_jsonl, set_random_seed
from unimapgen.runtime.device import is_accelerator_device, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict centerline JSON with the minimal DINOv2 -> Qwen route.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--run-root", type=str, default="", help="Training output root containing args.json/tokenizer/modules.")
    parser.add_argument("--run-args-json", type=str, default="", help="Explicit args.json saved by training.")
    parser.add_argument("--model-name-or-path", type=str, default=os.environ.get("MODEL_NAME_OR_PATH", ""))
    parser.add_argument("--tokenizer-name-or-path", type=str, default=os.environ.get("TOKENIZER_NAME_OR_PATH", ""))
    parser.add_argument("--dinov2-model-name-or-path", type=str, default=os.environ.get("DINOV2_MODEL_NAME_OR_PATH", ""))
    parser.add_argument("--visual-encoder-checkpoint-path", type=str, default=os.environ.get("VISUAL_ENCODER_CHECKPOINT_PATH", ""))
    parser.add_argument("--bridge-modules-state-path", type=str, default=os.environ.get("BRIDGE_MODULES_STATE_PATH", ""))
    parser.add_argument("--map-task", type=str, default=os.environ.get("MAP_TASK", "lane_intersection"))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("IMAGE_SIZE", "512")))
    parser.add_argument("--encoder-input-pad-size", type=int, default=int(os.environ.get("ENCODER_INPUT_PAD_SIZE", "518")))
    parser.add_argument("--visual-token-compressor", type=str, default=os.environ.get("VISUAL_TOKEN_COMPRESSOR", "none"))
    parser.add_argument("--visual-token-compressor-grid-size", type=int, default=int(os.environ.get("VISUAL_TOKEN_COMPRESSOR_GRID_SIZE", "0")))
    parser.add_argument("--visual-token-compressor-hidden-dim", type=int, default=int(os.environ.get("VISUAL_TOKEN_COMPRESSOR_HIDDEN_DIM", "512")))
    parser.add_argument("--visual-token-compressor-depth", type=int, default=int(os.environ.get("VISUAL_TOKEN_COMPRESSOR_DEPTH", "2")))
    parser.add_argument("--visual-token-compressor-dropout", type=float, default=float(os.environ.get("VISUAL_TOKEN_COMPRESSOR_DROPOUT", "0.0")))
    parser.add_argument("--visual-projector-hidden-dim", type=int, default=int(os.environ.get("VISUAL_PROJECTOR_HIDDEN_DIM", "4096")))
    parser.add_argument("--geometric-mlp-hidden-dim", type=int, default=int(os.environ.get("GEOMETRIC_MLP_HIDDEN_DIM", "512")))
    parser.add_argument("--token-alignment-hidden-dim", type=int, default=int(os.environ.get("TOKEN_ALIGNMENT_HIDDEN_DIM", "4096")))
    parser.add_argument("--token-alignment-num-layers", type=int, default=int(os.environ.get("TOKEN_ALIGNMENT_NUM_LAYERS", "2")))
    parser.add_argument("--token-alignment-dropout", type=float, default=float(os.environ.get("TOKEN_ALIGNMENT_DROPOUT", "0.0")))
    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "32")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "64")))
    parser.add_argument("--lora-dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    parser.add_argument("--trainroot", type=str, default="", help="Root with split jsonl files; usually the prepared trainroot.")
    parser.add_argument("--split", type=str, default="val", help="Used with --trainroot.")
    parser.add_argument("--dataset-jsonl", type=str, default="", help="Alternative explicit dataset jsonl.")
    parser.add_argument("--dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, default="", help="Defaults to --trainroot when possible.")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Total dataset shards for parallel inference.")
    parser.add_argument("--shard-index", type=int, default=0, help="This process shard index, 0-based.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda[:id], or npu[:id].")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--context-image-key", type=str, default="")
    parser.add_argument("--require-context-image", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def has_tokenizer_files(path: Path) -> bool:
    return any((path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"))


def has_model_artifact(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "adapter_model.safetensors",
            "adapter_config.json",
        )
    )


def resolve_checkpoint_dir(path_str: str) -> Tuple[Path, Path]:
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint-dir not found: {path}")
    if has_model_artifact(path):
        run_root = path if (path / "args.json").is_file() else path.parent
        return path, run_root
    candidates = sorted(
        [item for item in path.iterdir() if item.is_dir() and item.name.startswith("checkpoint-") and has_model_artifact(item)],
        key=lambda item: int(item.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint artifacts found under: {path}")
    return candidates[-1], path


def load_training_args_output_dir(checkpoint_dir: Path) -> Path | None:
    training_args_path = checkpoint_dir / "training_args.bin"
    if not training_args_path.is_file():
        return None
    try:
        training_args = torch.load(str(training_args_path), map_location="cpu", weights_only=False)
        output_dir = getattr(training_args, "output_dir", "")
    except Exception:
        return None
    if not str(output_dir).strip():
        return None
    return Path(str(output_dir)).expanduser().resolve()


def fallback_run_args_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    model_name_or_path = str(args.model_name_or_path).strip()
    dinov2_model_name_or_path = str(args.dinov2_model_name_or_path).strip()
    if not model_name_or_path or not dinov2_model_name_or_path:
        return {}
    return {
        "model_name_or_path": model_name_or_path,
        "tokenizer_name_or_path": str(args.tokenizer_name_or_path).strip() or model_name_or_path,
        "dinov2_model_name_or_path": dinov2_model_name_or_path,
        "visual_encoder_checkpoint_path": str(args.visual_encoder_checkpoint_path).strip(),
        "bridge_modules_state_path": str(args.bridge_modules_state_path).strip(),
        "image_size": int(args.image_size),
        "encoder_input_pad_size": int(args.encoder_input_pad_size),
        "visual_projector_hidden_dim": int(args.visual_projector_hidden_dim),
        "geometric_mlp_hidden_dim": int(args.geometric_mlp_hidden_dim),
        "token_alignment_hidden_dim": int(args.token_alignment_hidden_dim),
        "token_alignment_num_layers": int(args.token_alignment_num_layers),
        "token_alignment_dropout": float(args.token_alignment_dropout),
        "visual_token_compressor": str(args.visual_token_compressor).strip() or "none",
        "visual_token_compressor_grid_size": int(args.visual_token_compressor_grid_size),
        "visual_token_compressor_hidden_dim": int(args.visual_token_compressor_hidden_dim),
        "visual_token_compressor_depth": int(args.visual_token_compressor_depth),
        "visual_token_compressor_dropout": float(args.visual_token_compressor_dropout),
        "map_task": str(args.map_task).strip() or "lane_intersection",
        "use_lora": True,
        "no_lora": False,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "freeze_language_model": False,
        "freeze_vision_encoder": True,
        "num_visual_views": 1,
        "use_global_local_views": False,
        "use_view_type_embedding": False,
        "view_type_embedding_count": 2,
        "view_type_embedding_init_std": 0.02,
        "model_dtype": "auto",
        "fallback_args_source": "cli_or_environment",
    }


def load_run_args(run_root: Path, checkpoint_dir: Path, explicit_args_json: str = "", fallback_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    args_path = (
        Path(str(explicit_args_json)).expanduser().resolve()
        if str(explicit_args_json).strip()
        else run_root / "args.json"
    )
    candidates = [args_path]
    training_output_dir = load_training_args_output_dir(checkpoint_dir)
    if training_output_dir is not None:
        candidates.append(training_output_dir / "args.json")
    for candidate in candidates:
        if candidate.is_file():
            payload = load_json_dict(candidate)
            payload.setdefault("args_json_path", str(candidate))
            return payload
    if fallback_args:
        print(
            "[dinov2-centerline-predict] args.json not found; using CLI/environment fallback args. "
            f"looked_for={[str(item) for item in candidates]}",
            flush=True,
        )
        return dict(fallback_args)
    raise FileNotFoundError(
        "Missing args.json and no fallback model paths were provided. "
        f"looked_for={[str(item) for item in candidates]}. "
        "Set MODEL_NAME_OR_PATH and DINOV2_MODEL_NAME_OR_PATH, or pass --run-root/--run-args-json."
    )


def apply_runtime_path_overrides(saved_args: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "dinov2_model_name_or_path": args.dinov2_model_name_or_path,
        "visual_encoder_checkpoint_path": args.visual_encoder_checkpoint_path,
        "bridge_modules_state_path": args.bridge_modules_state_path,
    }
    changed: Dict[str, Dict[str, str]] = {}
    for key, raw_value in overrides.items():
        value = str(raw_value).strip()
        if not value:
            continue
        old_value = str(saved_args.get(key, "")).strip()
        if old_value != value:
            saved_args[key] = value
            changed[key] = {"old": old_value, "new": value}
    if changed:
        print(
            "[dinov2-centerline-predict] runtime path overrides="
            + json.dumps(changed, ensure_ascii=False),
            flush=True,
        )
    return saved_args


def resolve_dataset_paths(args: argparse.Namespace) -> Tuple[Path, Path | None, Path]:
    if str(args.trainroot).strip():
        trainroot = Path(str(args.trainroot)).expanduser().resolve()
        split = str(args.split).strip()
        dataset_path = trainroot / f"{split}.jsonl"
        meta_path = trainroot / f"meta_{split}.jsonl"
        media_dir = Path(str(args.media_dir)).expanduser().resolve() if str(args.media_dir).strip() else trainroot
        return dataset_path, (meta_path if meta_path.is_file() else None), media_dir

    if not str(args.dataset_jsonl).strip():
        raise ValueError("Pass either --trainroot or --dataset-jsonl.")
    if not str(args.media_dir).strip():
        raise ValueError("--media-dir is required when --trainroot is not used.")
    dataset_path = Path(str(args.dataset_jsonl)).expanduser().resolve()
    meta_path = resolve_meta_jsonl(dataset_path, str(args.dataset_meta_jsonl))
    media_dir = Path(str(args.media_dir)).expanduser().resolve()
    return dataset_path, meta_path, media_dir


def _coerce_xy_point(raw: Any) -> List[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        x = float(raw[0])
        y = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not torch.isfinite(torch.tensor([x, y], dtype=torch.float32)).all():
        return None
    return [x, y]


def sanitize_lines(lines: Any) -> List[Dict[str, Any]]:
    if not isinstance(lines, list):
        return []
    sanitized: List[Dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        raw_points = line.get("points", [])
        points: List[List[int]] = []
        if isinstance(raw_points, (list, tuple)):
            for raw in raw_points:
                pt = _coerce_xy_point(raw)
                if pt is not None:
                    points.append([int(round(pt[0])), int(round(pt[1]))])
        if len(points) < 2:
            continue
        out = {
            "category": str(line.get("category", "centerline") or "centerline"),
            "start_type": str(line.get("start_type", "")),
            "end_type": str(line.get("end_type", "")),
            "points": points,
        }
        sanitized.append(out)
    return sanitized


def normalize_lines_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_lines = payload.get("lines", [])
    if not isinstance(raw_lines, list):
        return []
    return sanitize_lines(
        [
            {
                "category": raw_line.get("category", "centerline") if isinstance(raw_line, dict) else "centerline",
                "start_type": raw_line.get("start_type", "") if isinstance(raw_line, dict) else "",
                "end_type": raw_line.get("end_type", "") if isinstance(raw_line, dict) else "",
                "points": raw_line.get("points", []) if isinstance(raw_line, dict) else [],
            }
            for raw_line in raw_lines
        ]
    )


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", str(text), flags=re.DOTALL).strip()


def extract_first_json_object(text: str) -> str:
    source = str(text)
    start = source.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(source)):
        char = source[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    return ""


def parse_prediction_text(raw_text: str) -> Dict[str, Any]:
    cleaned = strip_think_blocks(str(raw_text))
    json_candidate = extract_first_json_object(cleaned) or cleaned.strip()
    if not json_candidate:
        return {"parse_ok": False, "normalized_json": "", "pred_lines": [], "parse_error": "empty_prediction"}
    try:
        normalized_json = normalize_centerline_json_text(json_candidate)
        payload = json.loads(normalized_json)
        return {
            "parse_ok": True,
            "normalized_json": normalized_json,
            "pred_lines": normalize_lines_payload(payload if isinstance(payload, dict) else {}),
            "parse_error": "",
        }
    except Exception as exc:
        return {"parse_ok": False, "normalized_json": "", "pred_lines": [], "parse_error": repr(exc)}


def decode_ground_truth_lines(sample: Any, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta_lines = meta.get("target_lines", [])
    if isinstance(meta_lines, list) and meta_lines:
        return sanitize_lines(meta_lines)
    try:
        payload = json.loads(str(sample.assistant_json))
    except Exception:
        payload = {}
    return normalize_lines_payload(payload if isinstance(payload, dict) else {})


def has_complete_valid_json(text: str) -> bool:
    candidate = extract_first_json_object(strip_think_blocks(str(text)))
    if not candidate:
        return False
    try:
        normalize_centerline_json_text(candidate)
        return True
    except Exception:
        return False


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
def generate_json_text(
    *,
    model: Qwen3RCDinoCenterlineJSONSFTModel,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> Dict[str, Any]:
    prompt_batch = tokenizer([str(prompt_text)], padding=True, truncation=True, return_tensors="pt")
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
        if generated_ids and len(generated_ids) % 8 == 0:
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            if has_complete_valid_json(partial_text):
                break

    return {"generated_ids": generated_ids, "raw_text": tokenizer.decode(generated_ids, skip_special_tokens=False).strip()}


def strip_common_state_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.", "model.")
    updated = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if updated and all(str(key).startswith(prefix) for key in updated.keys()):
                updated = {str(key)[len(prefix) :]: value for key, value in updated.items()}
                changed = True
    return updated


def load_checkpoint_state(model: torch.nn.Module, checkpoint_dir: Path) -> bool:
    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        return False
    raw_state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unexpected checkpoint state type: {type(raw_state)!r}")
    state_dict = raw_state.get("state_dict", raw_state) if "state_dict" in raw_state else raw_state
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected state_dict type: {type(state_dict)!r}")
    cleaned_state_dict = strip_common_state_prefixes({str(key): value for key, value in state_dict.items()})
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    missing = [str(key) for key in missing]
    unexpected = [str(key) for key in unexpected]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state_dict mismatch.\n"
            f"missing_keys={missing[:20]}\n"
            f"unexpected_keys={unexpected[:20]}"
        )
    return True


def tokenizer_source_for(checkpoint_dir: Path, run_root: Path, saved_args: Dict[str, Any]) -> str:
    if has_tokenizer_files(checkpoint_dir):
        return str(checkpoint_dir)
    if has_tokenizer_files(run_root):
        return str(run_root)
    return str(saved_args.get("tokenizer_name_or_path") or saved_args.get("model_name_or_path"))


def model_source_for(checkpoint_dir: Path, saved_args: Dict[str, Any]) -> str:
    if (checkpoint_dir / "adapter_config.json").is_file() and not (checkpoint_dir / "pytorch_model.bin").is_file():
        return str(checkpoint_dir)
    return str(saved_args["model_name_or_path"])


def modules_state_for(checkpoint_dir: Path, run_root: Path, saved_args: Dict[str, Any]) -> str:
    for filename in ("rc_dinov2_centerline_json_modules.pt", "rc_dinov2_centerline_json_modules.pth"):
        modules_path = checkpoint_dir / filename
        if modules_path.is_file():
            return str(modules_path)
        run_modules_path = run_root / filename
        if run_modules_path.is_file():
            return str(run_modules_path)
    return str(saved_args.get("bridge_modules_state_path", ""))


def main() -> None:
    args = parse_args()
    args.device = resolve_torch_device(str(args.device))
    set_random_seed(int(args.seed))

    checkpoint_dir, inferred_run_root = resolve_checkpoint_dir(str(args.checkpoint_dir))
    run_root = (
        Path(str(args.run_root)).expanduser().resolve()
        if str(args.run_root).strip()
        else inferred_run_root
    )
    if str(args.run_args_json).strip() and not str(args.run_root).strip():
        run_root = Path(str(args.run_args_json)).expanduser().resolve().parent
    saved_args = load_run_args(
        run_root,
        checkpoint_dir,
        str(args.run_args_json),
        fallback_run_args_from_cli(args),
    )
    saved_args = apply_runtime_path_overrides(saved_args, args)
    dataset_path, dataset_meta_path, media_dir = resolve_dataset_paths(args)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_path}")
    if not media_dir.exists():
        raise FileNotFoundError(f"media_dir not found: {media_dir}")

    image_size = int(saved_args.get("image_size", 512))
    encoder_input_pad_size = int(saved_args.get("encoder_input_pad_size", 518))
    encoder_visual_grid_size, encoder_tokens_per_view = infer_visual_layout(
        image_size=image_size,
        encoder_input_pad_size=encoder_input_pad_size,
        patch_size=14,
    )
    visual_token_compressor = str(saved_args.get("visual_token_compressor", "none")).strip().lower()
    if visual_token_compressor in {"", "none", "identity"}:
        visual_grid_size = int(encoder_visual_grid_size)
    else:
        visual_grid_size = int(
            saved_args.get("effective_visual_grid_size")
            or saved_args.get("visual_token_compressor_grid_size")
            or 0
        )
        if visual_grid_size <= 0:
            raise ValueError(
                "Compressed checkpoint is missing effective_visual_grid_size/visual_token_compressor_grid_size."
            )
    tokens_per_view = int(visual_grid_size) * int(visual_grid_size)
    num_visual_views = int(saved_args.get("num_visual_views") or (2 if saved_args.get("use_global_local_views") else 1))
    num_visual_tokens = int(tokens_per_view) * int(num_visual_views)
    context_image_key = str(args.context_image_key).strip() or str(saved_args.get("context_image_key", "context_image"))
    require_context_image = (
        bool(args.require_context_image)
        if args.require_context_image is not None
        else bool(saved_args.get("require_context_image", num_visual_views > 1))
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source_for(checkpoint_dir, run_root, saved_args),
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        use_fast=False,
    )
    formatter = RCCenterlineJSONSFTFormatter(
        image_size=image_size,
        num_visual_tokens=num_visual_tokens,
        system_prompt=str(saved_args.get("system_prompt") or default_system_prompt_for_task(saved_args.get("map_task", "lane"))),
        user_prompt=str(saved_args.get("user_prompt") or default_user_prompt_for_task(saved_args.get("map_task", "lane"))),
    )
    formatter.register_tokens(tokenizer)

    rows = load_jsonl(dataset_path)
    if bool(args.shuffle):
        random.Random(int(args.seed)).shuffle(rows)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}.")
    total_rows_before_shard = len(rows)
    if num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]
    meta_rows = load_jsonl(dataset_meta_path) if dataset_meta_path is not None else []
    meta_by_id = {str(item.get("id", "")): item for item in meta_rows if str(item.get("id", "")).strip()}

    dataset = RCCenterlineJSONSFTDataset(
        rows=rows,
        meta_rows=meta_rows,
        media_dir=media_dir,
        tokenizer=tokenizer,
        formatter=formatter,
        image_size=image_size,
        context_image_key=context_image_key,
        require_context_image=bool(require_context_image),
    )

    model = Qwen3RCDinoCenterlineJSONSFTModel(
        model_name_or_path=model_source_for(checkpoint_dir, saved_args),
        tokenizer=tokenizer,
        dinov2_model_name_or_path=str(saved_args["dinov2_model_name_or_path"]),
        visual_encoder_checkpoint_path=str(saved_args.get("visual_encoder_checkpoint_path", "")),
        modules_state_path=modules_state_for(checkpoint_dir, run_root, saved_args),
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        encoder_visual_grid_size=int(encoder_visual_grid_size),
        num_visual_views=int(num_visual_views),
        visual_projector_hidden_dim=int(saved_args.get("visual_projector_hidden_dim", 4096)),
        geometric_mlp_hidden_dim=int(saved_args.get("geometric_mlp_hidden_dim", 512)),
        token_alignment_hidden_dim=int(saved_args.get("token_alignment_hidden_dim", 4096)),
        token_alignment_num_layers=int(saved_args.get("token_alignment_num_layers", 2)),
        token_alignment_dropout=float(saved_args.get("token_alignment_dropout", 0.0)),
        visual_token_compressor=visual_token_compressor,
        visual_token_compressor_hidden_dim=int(saved_args.get("visual_token_compressor_hidden_dim", 512)),
        visual_token_compressor_depth=int(saved_args.get("visual_token_compressor_depth", 2)),
        visual_token_compressor_dropout=float(saved_args.get("visual_token_compressor_dropout", 0.0)),
        use_view_type_embedding=bool(saved_args.get("use_view_type_embedding", False)),
        view_type_embedding_count=int(saved_args.get("view_type_embedding_count", max(2, num_visual_views))),
        view_type_embedding_init_std=float(saved_args.get("view_type_embedding_init_std", 0.02)),
        language_model_dtype=str(saved_args.get("model_dtype", "auto")),
        local_files_only=bool(args.local_files_only),
        freeze_language_model=bool(saved_args.get("freeze_language_model", False)),
        freeze_vision_encoder=bool(saved_args.get("freeze_vision_encoder", True)),
        encoder_input_pad_size=int(encoder_input_pad_size),
        use_lora=not bool(saved_args.get("no_lora", False)),
        lora_rank=int(saved_args.get("lora_rank", 32)),
        lora_alpha=int(saved_args.get("lora_alpha", 64)),
        lora_dropout=float(saved_args.get("lora_dropout", 0.05)),
        gradient_checkpointing=False,
    )
    loaded_trainer_state = load_checkpoint_state(model, checkpoint_dir)

    device = str(args.device)
    model.eval()
    if is_accelerator_device(device):
        if bool(saved_args.get("bf16", False)):
            model = model.to(device=device, dtype=torch.bfloat16)
        else:
            model = model.to(device=device)
    else:
        model = model.to(device=device)

    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = (
        Path(str(args.summary_json)).expanduser().resolve()
        if str(args.summary_json).strip()
        else output_jsonl.with_suffix(".summary.json")
    )

    total = 0
    parse_ok = 0
    total_gt_lines = 0
    total_pred_lines = 0
    with output_jsonl.open("w", encoding="utf-8", buffering=1) as f:
        for sample in tqdm(dataset, desc="dinov2-centerline-predict", dynamic_ncols=True):
            total += 1
            meta = meta_by_id.get(str(sample.sample_id), {})
            generated = generate_json_text(
                model=model,
                tokenizer=tokenizer,
                pixel_values=sample.pixel_values,
                prompt_text=sample.prompt_text,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                device=device,
            )
            parsed = parse_prediction_text(str(generated["raw_text"]))
            gt_lines = decode_ground_truth_lines(sample, meta)
            pred_lines = sanitize_lines(parsed["pred_lines"])
            rel_image = str(meta.get("image", "")).strip()
            if not rel_image:
                try:
                    rel_image = str(sample.image_path.resolve().relative_to(media_dir))
                except Exception:
                    rel_image = str(sample.image_path.resolve())

            record = {
                "id": sample.sample_id,
                "image": rel_image.replace("\\", "/"),
                "gt_lines": gt_lines,
                "pred_lines": pred_lines,
                "state_lines": [],
                "gt_json": sample.assistant_json,
                "pred_json": parsed["normalized_json"],
                "raw_prediction_text": generated["raw_text"],
                "parse_ok": bool(parsed["parse_ok"]),
                "parse_error": str(parsed["parse_error"]),
                "num_gt_lines": len(gt_lines),
                "num_pred_lines": len(pred_lines),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            parse_ok += int(bool(parsed["parse_ok"]))
            total_gt_lines += int(len(gt_lines))
            total_pred_lines += int(len(pred_lines))

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_root": str(run_root),
        "loaded_trainer_state": bool(loaded_trainer_state),
        "dataset_jsonl": str(dataset_path),
        "dataset_meta_jsonl": str(dataset_meta_path or ""),
        "media_dir": str(media_dir),
        "output_jsonl": str(output_jsonl),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "total_rows_before_shard": int(total_rows_before_shard),
        "num_rows": int(total),
        "parse_ok": int(parse_ok),
        "parse_ok_rate": float(parse_ok) / float(total) if total > 0 else 0.0,
        "avg_gt_lines": float(total_gt_lines) / float(total) if total > 0 else 0.0,
        "avg_pred_lines": float(total_pred_lines) / float(total) if total > 0 else 0.0,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
