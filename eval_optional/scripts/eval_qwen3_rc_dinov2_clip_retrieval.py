#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer


def _resolve_repo_root() -> Path:
    # 评估脚本放在 eval_optional 下，这里仍然通过向上查找 unimapgen 定位仓库根。
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.rc_semantic_align_dataset import (  # noqa: E402
    RCSemanticAlignCollator,
    RCSemanticAlignDataset,
    load_jsonl,
)
from unimapgen.models.qwen3_rc_dinov2_clip_align import (  # noqa: E402
    Qwen3RCDinoClipAlignModel,
)
from unimapgen.rc_llm_runtime import (  # noqa: E402
    infer_visual_layout,
    load_json_dict,
    resolve_meta_jsonl,
    set_random_seed,
)

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation for Stage-1 RC DINOv2 <-> Qwen CLIP alignment.")
    parser.add_argument("--train-output-dir", type=str, required=True)
    parser.add_argument("--dataset-jsonl", type=str, default="")
    parser.add_argument("--dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_training_args(train_output_dir: Path) -> Dict[str, Any]:
    # Retrieval eval 复用训练时保存的 args.json，避免手工重复填写模型和视觉配置。
    args_path = train_output_dir / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Training args.json not found: {args_path}")
    return load_json_dict(args_path)


def resolve_tokenizer_path(train_output_dir: Path, training_args: Dict[str, Any]) -> str:
    required = ("tokenizer_config.json", "special_tokens_map.json")
    if all((train_output_dir / name).is_file() for name in required):
        return str(train_output_dir)
    return str(training_args.get("tokenizer_name_or_path") or training_args.get("model_name_or_path") or "")


def resolve_path(explicit: str, fallback: str, what: str) -> Path:
    raw = str(explicit).strip() or str(fallback).strip()
    if not raw:
        raise ValueError(f"Missing required path for {what}")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def normalize_side_key(raw_sides: Sequence[Any]) -> str:
    cleaned = [str(side).strip().lower() for side in raw_sides if str(side).strip()]
    return ",".join(cleaned)


def batched_dataset_samples(
    dataset: RCSemanticAlignDataset,
    collator: RCSemanticAlignCollator,
    batch_size: int,
) -> Sequence[tuple[List[Any], Dict[str, torch.Tensor]]]:
    total = len(dataset)
    for start in range(0, total, max(1, int(batch_size))):
        samples = [dataset[idx] for idx in range(start, min(total, start + int(batch_size)))]
        yield samples, collator(samples)


def topk_recall(similarity: torch.Tensor, positive_mask: torch.Tensor, k: int) -> float:
    topk = min(max(1, int(k)), int(similarity.shape[1]))
    indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=False).indices
    hits = positive_mask.gather(1, indices).any(dim=1)
    return float(hits.float().mean().item())


def margin_stats(similarity: torch.Tensor, positive_mask: torch.Tensor) -> Dict[str, float]:
    masked_positive = similarity.masked_fill(~positive_mask, float("-inf"))
    masked_negative = similarity.masked_fill(positive_mask, float("-inf"))
    best_positive = masked_positive.max(dim=1).values
    hardest_negative = masked_negative.max(dim=1).values
    has_negative = torch.isfinite(hardest_negative)
    safest_negative = torch.where(has_negative, hardest_negative, torch.zeros_like(hardest_negative))
    margin = best_positive - safest_negative
    return {
        "best_positive_mean": float(best_positive.mean().item()),
        "best_positive_median": float(best_positive.median().item()),
        "hard_negative_mean": float(safest_negative.mean().item()),
        "hard_negative_median": float(safest_negative.median().item()),
        "margin_mean": float(margin.mean().item()),
        "margin_median": float(margin.median().item()),
        "margin_positive_rate": float((margin > 0).float().mean().item()),
    }


def build_top1_predictions(
    *,
    query_ids: Sequence[str],
    query_scene_labels: Sequence[str],
    query_side_keys: Sequence[str],
    query_texts: Sequence[str],
    query_group_ids: np.ndarray,
    cand_ids: Sequence[str],
    cand_scene_labels: Sequence[str],
    cand_side_keys: Sequence[str],
    cand_texts: Sequence[str],
    cand_group_ids: np.ndarray,
    similarity: torch.Tensor,
) -> List[Dict[str, Any]]:
    top1_idx = similarity.argmax(dim=1).cpu().numpy().astype(np.int64)
    predictions: List[Dict[str, Any]] = []
    similarity_np = similarity.cpu().numpy()
    for query_index, cand_index in enumerate(top1_idx.tolist()):
        positive_mask = (cand_group_ids == query_group_ids[query_index])
        positive_scores = similarity_np[query_index][positive_mask]
        best_positive = float(np.max(positive_scores)) if positive_scores.size > 0 else float("-inf")
        margin = float(best_positive - similarity_np[query_index][cand_index])
        predictions.append(
            {
                "query_id": str(query_ids[query_index]),
                "query_scene_label": str(query_scene_labels[query_index]),
                "query_visible_sides": str(query_side_keys[query_index]),
                "query_group_id": int(query_group_ids[query_index]),
                "query_text": str(query_texts[query_index]),
                "top1_id": str(cand_ids[cand_index]),
                "top1_scene_label": str(cand_scene_labels[cand_index]),
                "top1_visible_sides": str(cand_side_keys[cand_index]),
                "top1_group_id": int(cand_group_ids[cand_index]),
                "top1_text": str(cand_texts[cand_index]),
                "top1_score": float(similarity_np[query_index][cand_index]),
                "best_positive_score": best_positive,
                "top1_is_group_match": bool(cand_group_ids[cand_index] == query_group_ids[query_index]),
                "top1_is_scene_match": bool(cand_scene_labels[cand_index] == query_scene_labels[query_index]),
                "top1_is_side_match": bool(cand_side_keys[cand_index] == query_side_keys[query_index]),
                "top1_gap_to_best_positive": margin,
            }
        )
    return predictions


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_random_seed(int(args.seed))

    train_output_dir = Path(args.train_output_dir).expanduser().resolve()
    if not train_output_dir.is_dir():
        raise FileNotFoundError(f"train_output_dir not found: {train_output_dir}")
    training_args = load_training_args(train_output_dir)

    dataset_jsonl = resolve_path(
        explicit=str(args.dataset_jsonl),
        fallback=str(training_args.get("eval_dataset_jsonl") or training_args.get("dataset_jsonl") or ""),
        what="dataset_jsonl",
    )
    dataset_meta_jsonl = resolve_meta_jsonl(dataset_jsonl, explicit_meta_jsonl=str(args.dataset_meta_jsonl)) or resolve_path(
        explicit="",
        fallback=str(training_args.get("eval_dataset_meta_jsonl") or training_args.get("dataset_meta_jsonl") or ""),
        what="dataset_meta_jsonl",
    )
    media_dir = resolve_path(
        explicit=str(args.media_dir),
        fallback=str(training_args.get("media_dir") or ""),
        what="media_dir",
    )
    modules_state_path = resolve_path(
        explicit="",
        fallback=str(train_output_dir / "rc_dinov2_clip_align_modules.pt"),
        what="modules_state_path",
    )
    output_dir = (
        Path(str(args.output_dir).strip()).expanduser().resolve()
        if str(args.output_dir).strip()
        else (train_output_dir / "retrieval_eval_val")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = (
        Path(str(args.output_json).strip()).expanduser().resolve()
        if str(args.output_json).strip()
        else (output_dir / "metrics.json")
    )

    # 使用训练输出目录里的 tokenizer / bridge / args，保证离线评估和正式训练严格同口径。
    tokenizer_name_or_path = resolve_tokenizer_path(train_output_dir, training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_rows = load_jsonl(dataset_jsonl, max_samples=int(args.max_samples))
    meta_rows = load_jsonl(dataset_meta_jsonl)
    dataset = RCSemanticAlignDataset(
        rows=dataset_rows,
        meta_rows=meta_rows,
        media_dir=media_dir,
        image_size=int(training_args.get("image_size", 512)),
        tokenizer=tokenizer,
        cutoff_len=int(training_args.get("cutoff_len", 256)),
        border_tol_px=float(training_args.get("border_tol_px", 18.0)),
    )
    collator = RCSemanticAlignCollator(
        tokenizer=tokenizer,
        cutoff_len=int(training_args.get("cutoff_len", 256)),
    )

    device = str(args.device).strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    modules_state = torch.load(str(modules_state_path), map_location="cpu", weights_only=False)
    visual_grid_size = int(modules_state.get("visual_grid_size", 0)) if isinstance(modules_state, dict) else 0
    num_visual_tokens = int(modules_state.get("num_visual_tokens", 0)) if isinstance(modules_state, dict) else 0
    if visual_grid_size <= 0 or num_visual_tokens <= 0:
        visual_grid_size, num_visual_tokens = infer_visual_layout(
            image_size=int(training_args.get("image_size", 512)),
            encoder_input_pad_size=int(training_args.get("encoder_input_pad_size", 0)),
            patch_size=14,
        )

    model = Qwen3RCDinoClipAlignModel(
        model_name_or_path=str(training_args.get("model_name_or_path")),
        dinov2_model_name_or_path=str(training_args.get("dinov2_model_name_or_path")),
        visual_encoder_checkpoint_path=str(training_args.get("visual_encoder_checkpoint_path", "")),
        modules_state_path=str(modules_state_path),
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        contrastive_dim=int(training_args.get("contrastive_dim", 1024)),
        visual_projector_hidden_dim=int(training_args.get("visual_projector_hidden_dim", 4096)),
        geometric_mlp_hidden_dim=int(training_args.get("geometric_mlp_hidden_dim", 512)),
        token_alignment_hidden_dim=int(training_args.get("token_alignment_hidden_dim", 4096)),
        token_alignment_num_layers=int(training_args.get("token_alignment_num_layers", 2)),
        token_alignment_dropout=float(training_args.get("token_alignment_dropout", 0.0)),
        language_model_dtype=str(training_args.get("model_dtype", "auto")),
        local_files_only=bool(args.local_files_only),
        freeze_language_model=bool(training_args.get("freeze_language_model", True)),
        freeze_vision_encoder=bool(training_args.get("freeze_vision_encoder", True)),
        encoder_input_pad_size=int(training_args.get("encoder_input_pad_size", 0)),
    )
    model.eval()
    model.to(device)

    # 先分别编码图像和文本，再构造全量相似度矩阵，最后统一计算 recall / margin / top1 语义一致性。
    image_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []
    sample_ids: List[str] = []
    scene_labels: List[str] = []
    side_keys: List[str] = []
    texts: List[str] = []
    group_ids: List[int] = []

    use_bf16 = device.startswith("cuda")
    with torch.inference_mode():
        for samples, batch in batched_dataset_samples(dataset, collator, batch_size=int(args.batch_size)):
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast_enabled = bool(use_bf16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                batch_image_embeddings, _ = model.encode_image(batch["pixel_values"])
                batch_text_embeddings = model.encode_text(batch["input_ids"], batch["attention_mask"])
            image_embeddings.append(batch_image_embeddings.detach().cpu().to(dtype=torch.float32))
            text_embeddings.append(batch_text_embeddings.detach().cpu().to(dtype=torch.float32))
            for sample in samples:
                sample_ids.append(str(sample.sample_id))
                scene_labels.append(str(sample.scene_label))
                side_keys.append(normalize_side_key(sample.visible_sides))
                texts.append(str(sample.text))
                group_ids.append(int(sample.group_id))

    image_embeddings_tensor = torch.cat(image_embeddings, dim=0)
    text_embeddings_tensor = torch.cat(text_embeddings, dim=0)
    group_ids_np = np.asarray(group_ids, dtype=np.int64)
    similarity = image_embeddings_tensor @ text_embeddings_tensor.t()
    positive_mask = torch.from_numpy(group_ids_np.reshape(-1, 1) == group_ids_np.reshape(1, -1))

    i2t_top1_idx = similarity.argmax(dim=1).cpu().numpy().astype(np.int64)
    t2i_similarity = similarity.t().contiguous()
    t2i_top1_idx = t2i_similarity.argmax(dim=1).cpu().numpy().astype(np.int64)

    i2t_scene_acc = float(np.mean([scene_labels[q] == scene_labels[i2t_top1_idx[q]] for q in range(len(scene_labels))]))
    i2t_side_acc = float(np.mean([side_keys[q] == side_keys[i2t_top1_idx[q]] for q in range(len(side_keys))]))
    t2i_scene_acc = float(np.mean([scene_labels[q] == scene_labels[t2i_top1_idx[q]] for q in range(len(scene_labels))]))
    t2i_side_acc = float(np.mean([side_keys[q] == side_keys[t2i_top1_idx[q]] for q in range(len(side_keys))]))

    metrics = {
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_meta_jsonl": str(dataset_meta_jsonl),
        "media_dir": str(media_dir),
        "train_output_dir": str(train_output_dir),
        "modules_state_path": str(modules_state_path),
        "num_samples": int(len(sample_ids)),
        "num_unique_groups": int(len(set(group_ids))),
        "batch_size": int(args.batch_size),
        "model": {
            "model_name_or_path": str(training_args.get("model_name_or_path")),
            "dinov2_model_name_or_path": str(training_args.get("dinov2_model_name_or_path")),
            "visual_encoder_checkpoint_path": str(training_args.get("visual_encoder_checkpoint_path", "")),
            "contrastive_dim": int(training_args.get("contrastive_dim", 1024)),
            "image_size": int(training_args.get("image_size", 512)),
            "encoder_input_pad_size": int(training_args.get("encoder_input_pad_size", 0)),
            "visual_grid_size": int(visual_grid_size),
            "num_visual_tokens": int(num_visual_tokens),
            "freeze_language_model": bool(training_args.get("freeze_language_model", True)),
            "freeze_vision_encoder": bool(training_args.get("freeze_vision_encoder", True)),
            "logit_scale_exp": float(model.logit_scale.detach().exp().cpu().item()),
        },
        "retrieval": {
            "image_to_text_group_r1": topk_recall(similarity, positive_mask, 1),
            "image_to_text_group_r5": topk_recall(similarity, positive_mask, 5),
            "image_to_text_group_r10": topk_recall(similarity, positive_mask, 10),
            "text_to_image_group_r1": topk_recall(t2i_similarity, positive_mask.t(), 1),
            "text_to_image_group_r5": topk_recall(t2i_similarity, positive_mask.t(), 5),
            "text_to_image_group_r10": topk_recall(t2i_similarity, positive_mask.t(), 10),
        },
        "top1_semantics": {
            "image_to_text_scene_acc": i2t_scene_acc,
            "image_to_text_side_set_acc": i2t_side_acc,
            "text_to_image_scene_acc": t2i_scene_acc,
            "text_to_image_side_set_acc": t2i_side_acc,
        },
        "similarity": {
            "image_to_text": margin_stats(similarity, positive_mask),
            "text_to_image": margin_stats(t2i_similarity, positive_mask.t()),
        },
    }

    i2t_predictions = build_top1_predictions(
        query_ids=sample_ids,
        query_scene_labels=scene_labels,
        query_side_keys=side_keys,
        query_texts=texts,
        query_group_ids=group_ids_np,
        cand_ids=sample_ids,
        cand_scene_labels=scene_labels,
        cand_side_keys=side_keys,
        cand_texts=texts,
        cand_group_ids=group_ids_np,
        similarity=similarity,
    )
    t2i_predictions = build_top1_predictions(
        query_ids=sample_ids,
        query_scene_labels=scene_labels,
        query_side_keys=side_keys,
        query_texts=texts,
        query_group_ids=group_ids_np,
        cand_ids=sample_ids,
        cand_scene_labels=scene_labels,
        cand_side_keys=side_keys,
        cand_texts=texts,
        cand_group_ids=group_ids_np,
        similarity=t2i_similarity,
    )
    write_json(output_json, metrics)
    write_jsonl(output_dir / "image_to_text_top1.jsonl", i2t_predictions)
    write_jsonl(output_dir / "text_to_image_top1.jsonl", t2i_predictions)

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
