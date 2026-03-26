from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import Trainer, TrainingArguments

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.discrete_stagea_dataset import (  # noqa: E402
    PatchOnlyDiscreteCollator,
    PatchOnlyDiscreteDataset,
    extract_assistant_lines,
    load_jsonl,
)
from unimapgen.discrete_map_token_format import DiscreteMapTokenFormatter  # noqa: E402
from unimapgen.models.qwen2_5vl_discrete import (  # noqa: E402
    build_training_model,
    load_processor,
    save_runtime_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable Stage A discrete-token train entry.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--eval-dataset-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--processor-output-dir", type=str, default="")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--evaluation-strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--coord-num-bins", type=int, default=896)
    parser.add_argument("--token-schema", type=str, default="shared_numbers", choices=["legacy_xy", "shared_numbers"])
    parser.add_argument("--disable-legacy-text-prompt-tokens", action="store_true")
    parser.add_argument("--categories", type=str, default="road")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_formatter(args: argparse.Namespace) -> DiscreteMapTokenFormatter:
    categories = [item.strip() for item in str(args.categories).split(",") if item.strip()]
    return DiscreteMapTokenFormatter(
        image_size=int(args.image_size),
        categories=categories,
        max_seq_len=int(args.cutoff_len),
        coord_num_bins=int(args.coord_num_bins),
        coordinate_token_style=str(args.token_schema),
        include_text_prompt_tokens=not bool(args.disable_legacy_text_prompt_tokens),
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    dataset_path = Path(args.dataset_jsonl).resolve()
    eval_dataset_path = Path(args.eval_dataset_jsonl).resolve() if str(args.eval_dataset_jsonl).strip() else None
    media_dir = Path(args.media_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processor_output_dir = (
        Path(args.processor_output_dir).resolve()
        if str(args.processor_output_dir).strip()
        else (output_dir / "processor").resolve()
    )

    formatter = build_formatter(args)

    print(f"[portable-dtok-train] model={args.model_name_or_path}", flush=True)
    print(f"[portable-dtok-train] dataset_jsonl={dataset_path}", flush=True)
    print(f"[portable-dtok-train] eval_dataset_jsonl={eval_dataset_path}", flush=True)
    print(f"[portable-dtok-train] media_dir={media_dir}", flush=True)
    print(f"[portable-dtok-train] output_dir={output_dir}", flush=True)
    print(f"[portable-dtok-train] processor_output_dir={processor_output_dir}", flush=True)
    print(f"[portable-dtok-train] token_schema={args.token_schema}", flush=True)
    print(f"[portable-dtok-train] coord_num_bins={args.coord_num_bins}", flush=True)
    print(f"[portable-dtok-train] finetune_mode={'full' if bool(args.no_lora) else 'lora'}", flush=True)

    processor = load_processor(args.model_name_or_path, formatter=formatter)
    save_runtime_assets(processor=processor, processor_output_dir=processor_output_dir, formatter=formatter)

    rows = load_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    eval_rows: List[Dict[str, Any]] = []
    if eval_dataset_path is not None and str(args.evaluation_strategy) != "no":
        eval_rows = load_jsonl(eval_dataset_path)
        if not eval_rows:
            raise ValueError(f"No eval rows loaded from {eval_dataset_path}")

    print(f"[portable-dtok-train] train_rows={len(rows)}", flush=True)
    if eval_rows:
        print(f"[portable-dtok-train] eval_rows={len(eval_rows)}", flush=True)
    sample_preview = formatter.lines_to_text(extract_assistant_lines(rows[0]))
    print(f"[portable-dtok-train] sample_target={sample_preview[:220]}", flush=True)

    train_dataset = PatchOnlyDiscreteDataset(rows=rows, media_dir=media_dir, processor=processor, formatter=formatter)
    eval_dataset = (
        PatchOnlyDiscreteDataset(rows=eval_rows, media_dir=media_dir, processor=processor, formatter=formatter)
        if eval_rows
        else None
    )
    collator = PatchOnlyDiscreteCollator(processor=processor, cutoff_len=int(args.cutoff_len))
    model = build_training_model(args=args, processor=processor)

    training_kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=not bool(str(args.resume_from_checkpoint).strip()),
        remove_unused_columns=False,
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        dataloader_num_workers=int(args.dataloader_num_workers),
        bf16=bool(args.bf16),
        report_to=[],
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
    )
    if str(args.evaluation_strategy) == "steps" and int(args.eval_steps) > 0:
        training_kwargs["eval_steps"] = int(args.eval_steps)
    try:
        training_args = TrainingArguments(evaluation_strategy=str(args.evaluation_strategy), **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(eval_strategy=str(args.evaluation_strategy), **training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    try:
        trainer = Trainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=getattr(processor, "tokenizer", None), **trainer_kwargs)

    resume_from_checkpoint = str(args.resume_from_checkpoint).strip() or None
    if resume_from_checkpoint is not None:
        print(f"[portable-dtok-train] resume_from_checkpoint={resume_from_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    save_runtime_assets(processor=processor, processor_output_dir=processor_output_dir, formatter=formatter)
    print("[portable-dtok-train] training finished", flush=True)


if __name__ == "__main__":
    main()
