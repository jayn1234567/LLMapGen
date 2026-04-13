from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def _resolve_repo_root() -> Path:
    # 兼容最小仓库目录拆分，避免从 eval_optional 运行时出现导入路径问题。
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.rc_centerline_json_sft_dataset import (  # noqa: E402
    RCCenterlineJSONSFTDataset,
    RCCenterlineJSONSFTFormatter,
    load_jsonl,
    normalize_centerline_json_text,
)
from unimapgen.models.qwen3_rc_dinov2_centerline_json_sft import (  # noqa: E402
    Qwen3RCDinoCenterlineJSONSFTModel,
)
from unimapgen.rc_llm_runtime import (  # noqa: E402
    infer_visual_layout,
    load_json_dict,
    resolve_meta_jsonl,
    set_random_seed,
)


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
        cleaned_points: List[List[int]] = []
        if isinstance(raw_points, (list, tuple)):
            if len(raw_points) >= 2 and not isinstance(raw_points[0], (list, tuple)):
                pt = _coerce_xy_point(raw_points)
                if pt is not None:
                    cleaned_points.append([int(round(pt[0])), int(round(pt[1]))])
            else:
                for raw in raw_points:
                    pt = _coerce_xy_point(raw)
                    if pt is not None:
                        cleaned_points.append([int(round(pt[0])), int(round(pt[1]))])
        if len(cleaned_points) < 2:
            continue
        sanitized.append(
            {
                "category": line.get("category", "centerline"),
                "start_type": line.get("start_type", "start"),
                "end_type": line.get("end_type", "end"),
                "points": cleaned_points,
            }
        )
        for key in ("score", "confidence", "line_score"):
            if key in line:
                sanitized[-1][key] = line[key]
    return sanitized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict JSON centerlines with the DINOv2 -> Qwen3 RC SFT model.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_checkpoint_dir(path_str: str) -> tuple[Path, Path]:
    path = Path(path_str).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint-dir not found: {path}")
    if path.name.startswith("checkpoint-") and (path / "pytorch_model.bin").is_file():
        return path, path.parent
    candidates = sorted(
        [item for item in path.iterdir() if item.is_dir() and item.name.startswith("checkpoint-")],
        key=lambda item: int(item.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under: {path}")
    return candidates[-1], path


def load_run_args(run_root: Path) -> Dict[str, Any]:
    args_path = run_root / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing args.json under run root: {args_path}")
    return load_json_dict(args_path)


def select_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if float(temperature) <= 0.0 or int(top_k) <= 1:
        return torch.argmax(logits, dim=-1)
    scaled = logits / max(float(temperature), 1e-6)
    k = max(1, min(int(top_k), int(scaled.shape[-1])))
    values, indices = torch.topk(scaled, k=k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return indices.gather(-1, sampled).squeeze(-1)


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


def load_checkpoint_state(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    # 这里只加载某个 checkpoint-* 下的语言模型权重，而 bridge / 视觉配置由 run_root 的 args 驱动。
    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint weights: {checkpoint_path}")
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


def normalize_lines_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_lines = payload.get("lines", [])
    if not isinstance(raw_lines, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue
        points = raw_line.get("points", [])
        normalized.append(
            {
                "category": "centerline",
                "points": points,
            }
        )
    return sanitize_lines(normalized)


def strip_think_blocks(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL)
    return cleaned.strip()


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
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    return ""


def parse_prediction_text(raw_text: str) -> Dict[str, Any]:
    # 生成结果里可能混入 think 残留或不完整片段，这里统一裁出第一个 JSON 对象再做规范化。
    cleaned = strip_think_blocks(str(raw_text))
    json_candidate = extract_first_json_object(cleaned) or cleaned.strip()
    if not json_candidate:
        return {
            "parse_ok": False,
            "normalized_json": "",
            "pred_lines": [],
            "parse_error": "empty_prediction",
        }
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
        return {
            "parse_ok": False,
            "normalized_json": "",
            "pred_lines": [],
            "parse_error": repr(exc),
        }


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
    # 推理时沿用 Stage 2/3 的视觉注入方式：先替换 <vis_patch> embedding，再自回归解码 JSON。
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
        if generated_ids and (len(generated_ids) % 8 == 0):
            # 只要已经形成完整合法 JSON，就提前停，避免为了等 eos 继续生成无关尾巴。
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            if has_complete_valid_json(partial_text):
                break

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return {
        "generated_ids": generated_ids,
        "raw_text": str(decoded).strip(),
    }


def main() -> None:
    args = parse_args()
    set_random_seed(int(args.seed))

    # 先恢复 run_root 的训练配置，再加载最新或指定 checkpoint，保证推理和训练的 prompt/bridge 完全一致。
    checkpoint_dir, run_root = resolve_checkpoint_dir(str(args.checkpoint_dir))
    saved_args = load_run_args(run_root)

    image_size = int(saved_args.get("image_size", 512))
    encoder_input_pad_size = int(saved_args.get("encoder_input_pad_size", 518))
    visual_grid_size, num_visual_tokens = infer_visual_layout(
        image_size=image_size,
        encoder_input_pad_size=encoder_input_pad_size,
        patch_size=14,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        use_fast=False,
    )
    system_prompt = str(saved_args.get("system_prompt", "")).strip()
    user_prompt = str(saved_args.get("user_prompt", "")).strip()
    formatter = RCCenterlineJSONSFTFormatter(
        image_size=int(image_size),
        num_visual_tokens=int(num_visual_tokens),
        **({"system_prompt": system_prompt} if system_prompt else {}),
        **({"user_prompt": user_prompt} if user_prompt else {}),
    )
    formatter.register_tokens(tokenizer)

    dataset_path = Path(args.dataset_jsonl).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_path}")
    dataset_meta_path = resolve_meta_jsonl(dataset_path, args.dataset_meta_jsonl)
    rows = load_jsonl(dataset_path)
    if bool(args.shuffle):
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    meta_rows = load_jsonl(dataset_meta_path) if dataset_meta_path is not None else []
    meta_by_id = {str(item.get("id", "")): item for item in meta_rows if str(item.get("id", "")).strip()}
    dataset = RCCenterlineJSONSFTDataset(
        rows=rows,
        meta_rows=meta_rows,
        media_dir=Path(args.media_dir).resolve(),
        tokenizer=tokenizer,
        formatter=formatter,
        image_size=int(image_size),
    )

    model = Qwen3RCDinoCenterlineJSONSFTModel(
        model_name_or_path=str(saved_args["model_name_or_path"]),
        tokenizer=tokenizer,
        dinov2_model_name_or_path=str(saved_args["dinov2_model_name_or_path"]),
        visual_encoder_checkpoint_path=str(saved_args.get("visual_encoder_checkpoint_path", "")),
        modules_state_path=str(saved_args.get("bridge_modules_state_path", "")),
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        visual_projector_hidden_dim=int(saved_args.get("visual_projector_hidden_dim", 4096)),
        geometric_mlp_hidden_dim=int(saved_args.get("geometric_mlp_hidden_dim", 512)),
        token_alignment_hidden_dim=int(saved_args.get("token_alignment_hidden_dim", 4096)),
        token_alignment_num_layers=int(saved_args.get("token_alignment_num_layers", 2)),
        token_alignment_dropout=float(saved_args.get("token_alignment_dropout", 0.0)),
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
    load_checkpoint_state(model, checkpoint_dir)

    device = str(args.device)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.bfloat16 if bool(saved_args.get("bf16", False)) else None)
    else:
        model = model.to(device=device)

    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json).resolve() if str(args.summary_json).strip() else output_jsonl.with_suffix(".summary.json")
    total = 0
    parse_ok = 0
    total_gt_lines = 0
    total_pred_lines = 0
    records: List[Dict[str, Any]] = []

    with output_jsonl.open("w", encoding="utf-8", buffering=1) as f:
        for sample in tqdm(dataset, desc="json-sft-predict", dynamic_ncols=True):
            total += 1
            meta = meta_by_id.get(str(sample.sample_id), {})
            pred = generate_json_text(
                model=model,
                tokenizer=tokenizer,
                pixel_values=sample.pixel_values,
                prompt_text=sample.prompt_text,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                device=device,
            )
            parsed = parse_prediction_text(str(pred["raw_text"]))
            gt_lines = decode_ground_truth_lines(sample, meta)
            pred_lines = sanitize_lines(parsed["pred_lines"])
            rel_image = str(meta.get("image", "")).strip()
            if not rel_image:
                try:
                    rel_image = str(sample.image_path.resolve().relative_to(Path(args.media_dir).resolve()))
                except Exception:
                    rel_image = str(sample.image_path.resolve())
            record = {
                "id": sample.sample_id,
                "image": rel_image,
                "gt_lines": gt_lines,
                "pred_lines": pred_lines,
                "state_lines": [],
                "gt_json": sample.assistant_json,
                "pred_json": parsed["normalized_json"],
                "raw_prediction_text": pred["raw_text"],
                "parse_ok": bool(parsed["parse_ok"]),
                "parse_error": str(parsed["parse_error"]),
                "num_gt_lines": len(gt_lines),
                "num_pred_lines": len(pred_lines),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            parse_ok += int(bool(parsed["parse_ok"]))
            total_gt_lines += int(len(gt_lines))
            total_pred_lines += int(len(pred_lines))

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_root": str(run_root),
        "dataset_jsonl": str(dataset_path),
        "dataset_meta_jsonl": (str(dataset_meta_path) if dataset_meta_path is not None else ""),
        "media_dir": str(Path(args.media_dir).resolve()),
        "output_jsonl": str(output_jsonl),
        "num_rows": int(total),
        "parse_ok": int(parse_ok),
        "parse_ok_rate": (float(parse_ok) / float(total) if total > 0 else 0.0),
        "avg_gt_lines": (float(total_gt_lines) / float(total) if total > 0 else 0.0),
        "avg_pred_lines": (float(total_pred_lines) / float(total) if total > 0 else 0.0),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
