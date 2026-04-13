from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, Trainer


def _resolve_repo_root() -> Path:
    # 最小仓库把脚本拆到了 train_required/eval_optional 下，这里动态向上找到真正的仓库根。
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
    save_qwen3_rc_dinov2_clip_align_modules,
)
from unimapgen.rc_llm_runtime import (  # noqa: E402
    create_training_arguments,
    infer_visual_layout,
    inspect_visual_encoder_checkpoint,
    resolve_meta_jsonl,
    save_run_args,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 DINOv2 -> Qwen3 CLIP-style RC semantic alignment.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default="")
    parser.add_argument("--dinov2-model-name-or-path", type=str, default="")
    parser.add_argument("--visual-encoder-checkpoint-path", type=str, default="")
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--eval-dataset-jsonl", type=str, default="")
    parser.add_argument("--eval-dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cutoff-len", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--encoder-input-pad-size", type=int, default=0)
    parser.add_argument("--contrastive-dim", type=int, default=1024)
    parser.add_argument("--visual-projector-hidden-dim", type=int, default=4096)
    parser.add_argument("--geometric-mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--token-alignment-hidden-dim", type=int, default=4096)
    parser.add_argument("--token-alignment-num-layers", type=int, default=2)
    parser.add_argument("--token-alignment-dropout", type=float, default=0.0)
    parser.add_argument("--model-dtype", type=str, default="auto")
    parser.add_argument("--freeze-language-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--border-tol-px", type=float, default=18.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(int(args.seed))

    # 先把训练/验证数据入口和输出目录校验好，避免起训练后才因为路径问题中断。
    dataset_path = Path(args.dataset_jsonl).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_path}")
    dataset_meta_path = resolve_meta_jsonl(dataset_path, args.dataset_meta_jsonl)
    eval_dataset_path = Path(args.eval_dataset_jsonl).resolve() if str(args.eval_dataset_jsonl).strip() else None
    if eval_dataset_path is not None and not eval_dataset_path.is_file():
        raise FileNotFoundError(f"eval_dataset_jsonl not found: {eval_dataset_path}")
    eval_dataset_meta_path = (
        resolve_meta_jsonl(eval_dataset_path, args.eval_dataset_meta_jsonl) if eval_dataset_path is not None else None
    )
    media_dir = Path(args.media_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    save_run_args(output_dir, args)

    inferred_ckpt_args: Dict[str, Any] = {}
    visual_encoder_checkpoint_path = str(args.visual_encoder_checkpoint_path).strip()
    if visual_encoder_checkpoint_path:
        # 复用 Stage A 保存下来的参数，保证 DINO 输入尺寸和 token 数推断与训练时一致。
        inferred_ckpt_args = inspect_visual_encoder_checkpoint(visual_encoder_checkpoint_path)

    dinov2_model_name_or_path = str(args.dinov2_model_name_or_path).strip() or str(
        inferred_ckpt_args.get("dinov2_model_name_or_path", "")
    ).strip()
    if not dinov2_model_name_or_path:
        raise ValueError(
            "dinov2_model_name_or_path is required. "
            "Either pass it explicitly or provide visual_encoder_checkpoint_path with saved training args."
        )

    encoder_input_pad_size = int(args.encoder_input_pad_size)
    if encoder_input_pad_size <= 0:
        encoder_input_pad_size = int(inferred_ckpt_args.get("encoder_input_pad_size", 0))

    visual_grid_size, num_visual_tokens = infer_visual_layout(
        image_size=int(args.image_size),
        encoder_input_pad_size=int(encoder_input_pad_size),
        patch_size=14,
    )

    print(f"[rc-dino-clip] model={args.model_name_or_path}", flush=True)
    print(f"[rc-dino-clip] tokenizer={args.tokenizer_name_or_path or args.model_name_or_path}", flush=True)
    print(f"[rc-dino-clip] dinov2_model_name_or_path={dinov2_model_name_or_path}", flush=True)
    print(f"[rc-dino-clip] visual_encoder_checkpoint_path={visual_encoder_checkpoint_path}", flush=True)
    print(f"[rc-dino-clip] dataset_jsonl={dataset_path}", flush=True)
    print(f"[rc-dino-clip] dataset_meta_jsonl={dataset_meta_path}", flush=True)
    print(f"[rc-dino-clip] eval_dataset_jsonl={eval_dataset_path}", flush=True)
    print(f"[rc-dino-clip] eval_dataset_meta_jsonl={eval_dataset_meta_path}", flush=True)
    print(f"[rc-dino-clip] media_dir={media_dir}", flush=True)
    print(f"[rc-dino-clip] output_dir={output_dir}", flush=True)
    print(f"[rc-dino-clip] image_size={int(args.image_size)}", flush=True)
    print(f"[rc-dino-clip] encoder_input_pad_size={int(encoder_input_pad_size)}", flush=True)
    print(f"[rc-dino-clip] visual_grid_size={visual_grid_size}", flush=True)
    print(f"[rc-dino-clip] num_visual_tokens={num_visual_tokens}", flush=True)
    print(f"[rc-dino-clip] contrastive_dim={int(args.contrastive_dim)}", flush=True)
    print(f"[rc-dino-clip] freeze_language_model={bool(args.freeze_language_model)}", flush=True)
    print(f"[rc-dino-clip] freeze_vision_encoder={bool(args.freeze_vision_encoder)}", flush=True)

    tokenizer_name_or_path = str(args.tokenizer_name_or_path).strip() or str(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(dataset_path, max_samples=int(args.max_samples))
    meta_rows = load_jsonl(dataset_meta_path) if dataset_meta_path is not None else []
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    eval_rows: List[Dict[str, Any]] = []
    eval_meta_rows: List[Dict[str, Any]] = []
    if eval_dataset_path is not None:
        eval_rows = load_jsonl(eval_dataset_path, max_samples=int(args.max_eval_samples))
        eval_meta_rows = load_jsonl(eval_dataset_meta_path) if eval_dataset_meta_path is not None else []

    # Stage 1 是双塔式粗对齐：图像侧提 embedding，文本侧提 embedding，再做 contrastive 学习。
    train_dataset = RCSemanticAlignDataset(
        rows=rows,
        meta_rows=meta_rows,
        media_dir=media_dir,
        image_size=int(args.image_size),
        tokenizer=tokenizer,
        cutoff_len=int(args.cutoff_len),
        border_tol_px=float(args.border_tol_px),
    )
    eval_dataset = (
        RCSemanticAlignDataset(
            rows=eval_rows,
            meta_rows=eval_meta_rows,
            media_dir=media_dir,
            image_size=int(args.image_size),
            tokenizer=tokenizer,
            cutoff_len=int(args.cutoff_len),
            border_tol_px=float(args.border_tol_px),
        )
        if eval_rows
        else None
    )
    collator = RCSemanticAlignCollator(
        tokenizer=tokenizer,
        cutoff_len=int(args.cutoff_len),
    )

    model = Qwen3RCDinoClipAlignModel(
        model_name_or_path=str(args.model_name_or_path),
        dinov2_model_name_or_path=dinov2_model_name_or_path,
        visual_encoder_checkpoint_path=visual_encoder_checkpoint_path,
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        contrastive_dim=int(args.contrastive_dim),
        visual_projector_hidden_dim=int(args.visual_projector_hidden_dim),
        geometric_mlp_hidden_dim=int(args.geometric_mlp_hidden_dim),
        token_alignment_hidden_dim=int(args.token_alignment_hidden_dim),
        token_alignment_num_layers=int(args.token_alignment_num_layers),
        token_alignment_dropout=float(args.token_alignment_dropout),
        language_model_dtype=str(args.model_dtype),
        local_files_only=bool(args.local_files_only),
        freeze_language_model=bool(args.freeze_language_model),
        freeze_vision_encoder=bool(args.freeze_vision_encoder),
        encoder_input_pad_size=int(encoder_input_pad_size),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rc-dino-clip] trainable_params={trainable_params} total_params={total_params}", flush=True)

    effective_eval_strategy = str(args.evaluation_strategy)
    if eval_dataset is None:
        effective_eval_strategy = "no"

    # 这里统一走兼容层，兼顾不同 transformers 版本对 TrainingArguments 字段名的差异。
    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_strategy=str(args.save_strategy),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        eval_steps=(int(args.eval_steps) if int(args.eval_steps) > 0 else None),
        dataloader_num_workers=int(args.dataloader_num_workers),
        bf16=bool(args.bf16),
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
        remove_unused_columns=False,
        report_to=[],
    )
    training_args = create_training_arguments(
        base_kwargs=training_kwargs,
        evaluation_strategy=effective_eval_strategy,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(str(args.resume_from_checkpoint).strip() or None)
    )
    print(json.dumps({"train_result": train_result.metrics}, ensure_ascii=False), flush=True)
    save_qwen3_rc_dinov2_clip_align_modules(model, output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
