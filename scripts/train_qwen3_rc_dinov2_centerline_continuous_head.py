#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.rc_centerline_continuous_head_dataset import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    RCCenterlineContinuousHeadCollator,
    RCCenterlineContinuousHeadDataset,
    RCCenterlineContinuousHeadFormatter,
    load_jsonl,
)
from unimapgen.models.qwen3_rc_centerline_16745style import unwrap_model  # noqa: E402
from unimapgen.models.qwen3_rc_dinov2_centerline_continuous_head import (  # noqa: E402
    Qwen3RCDinoCenterlineContinuousHeadModel,
    save_qwen3_rc_dinov2_centerline_continuous_head_modules,
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
    parser = argparse.ArgumentParser(
        description="Train Qwen3 DINOv2 RC centerline model with a minimal continuous coordinate regression head."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default="")
    parser.add_argument("--dinov2-model-name-or-path", type=str, default="")
    parser.add_argument("--visual-encoder-checkpoint-path", type=str, default="")
    parser.add_argument("--bridge-modules-state-path", type=str, default="")
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
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cutoff-len", type=int, default=7168)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--encoder-input-pad-size", type=int, default=518)
    parser.add_argument("--visual-projector-hidden-dim", type=int, default=4096)
    parser.add_argument("--geometric-mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--token-alignment-hidden-dim", type=int, default=4096)
    parser.add_argument("--token-alignment-num-layers", type=int, default=2)
    parser.add_argument("--token-alignment-dropout", type=float, default=0.0)
    parser.add_argument("--coord-head-hidden-dim", type=int, default=1024)
    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--coord-loss-beta", type=float, default=0.02)
    parser.add_argument("--coord-use-sigmoid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-dtype", type=str, default="auto")
    parser.add_argument("--freeze-language-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", type=str, default=DEFAULT_USER_PROMPT)
    return parser.parse_args()


class RCDinoContinuousHeadTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        coord_loss_weight: float = 1.0,
        coord_loss_beta: float = 0.02,
        tokenizer_for_save: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.coord_loss_weight = float(coord_loss_weight)
        self.coord_loss_beta = float(coord_loss_beta)
        self.tokenizer_for_save = tokenizer_for_save
        self._coord_reg_loss_sum = 0.0
        self._coord_reg_loss_count = 0
        self._coord_reg_mae_sum = 0.0
        self._coord_reg_mae_count = 0

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:
        merged_logs = dict(logs)
        if self._coord_reg_loss_count > 0:
            merged_logs["coord_reg_loss"] = self._coord_reg_loss_sum / float(self._coord_reg_loss_count)
            self._coord_reg_loss_sum = 0.0
            self._coord_reg_loss_count = 0
        if self._coord_reg_mae_count > 0:
            merged_logs["coord_reg_mae"] = self._coord_reg_mae_sum / float(self._coord_reg_mae_count)
            self._coord_reg_mae_sum = 0.0
            self._coord_reg_mae_count = 0
        super().log(merged_logs, *args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        target_dir = output_dir or self.args.output_dir
        if not self.args.should_save:
            return
        save_qwen3_rc_dinov2_centerline_continuous_head_modules(
            self.model,
            target_dir,
            tokenizer=self.tokenizer_for_save,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        checkpoint_dir = Path(str(resume_from_checkpoint))
        has_standard_model = any(
            (checkpoint_dir / name).is_file()
            for name in (
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
            )
        )
        has_rc_adapter = (checkpoint_dir / "adapter_config.json").is_file() and (
            checkpoint_dir / "rc_dinov2_centerline_continuous_head_modules.pt"
        ).is_file()
        if has_rc_adapter and not has_standard_model:
            print(
                (
                    "[rc-cont-head] skip Trainer model reload for custom adapter checkpoint; "
                    f"weights were already restored from {checkpoint_dir}"
                ),
                flush=True,
            )
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _compute_coord_loss(
        self,
        model: torch.nn.Module,
        coord_pred: torch.Tensor,
        coord_target_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        shift_pred = coord_pred[:, :-1, :].contiguous().float()
        shift_target = coord_target_values[:, 1:, :].contiguous().float()
        valid_mask = shift_target[..., 0].ge(0.0) & shift_target[..., 1].ge(0.0)
        unwrapped = unwrap_model(model)
        coord_head = getattr(unwrapped, "coord_head", None)
        if coord_head is None:
            raise AttributeError("Continuous-head model is missing coord_head.")
        if not bool(valid_mask.any()):
            # Keep the regression head attached to the graph even when this
            # micro-batch has no coordinate targets on the current rank.
            zero = sum(param.float().sum() * 0.0 for param in coord_head.parameters())
            return zero, zero, 0

        masked_pred = shift_pred[valid_mask]
        masked_target = shift_target[valid_mask]
        reg_loss = F.smooth_l1_loss(
            masked_pred,
            masked_target,
            beta=float(self.coord_loss_beta),
            reduction="mean",
        )
        mae = torch.abs(masked_pred - masked_target).mean()
        return reg_loss, mae, int(valid_mask.sum().item())

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        del num_items_in_batch
        coord_target_values = inputs.pop("coord_target_values")
        outputs = model(**inputs)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if self.coord_loss_weight <= 0.0:
            return (base_loss, outputs) if return_outputs else base_loss

        coord_pred = outputs["coord_pred"] if isinstance(outputs, dict) else getattr(outputs, "coord_pred")
        coord_reg_loss, coord_reg_mae, coord_count = self._compute_coord_loss(
            model=model,
            coord_pred=coord_pred,
            coord_target_values=coord_target_values,
        )
        loss = base_loss + coord_reg_loss.to(base_loss.dtype) * self.coord_loss_weight
        if model.training and coord_count > 0:
            self._coord_reg_loss_sum += float(coord_reg_loss.detach().item())
            self._coord_reg_loss_count += 1
            self._coord_reg_mae_sum += float(coord_reg_mae.detach().item())
            self._coord_reg_mae_count += 1
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_random_seed(int(args.seed))

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

    print(f"[rc-cont-head] model={args.model_name_or_path}", flush=True)
    print(f"[rc-cont-head] tokenizer={args.tokenizer_name_or_path or args.model_name_or_path}", flush=True)
    print(f"[rc-cont-head] dinov2_model_name_or_path={dinov2_model_name_or_path}", flush=True)
    print(f"[rc-cont-head] visual_encoder_checkpoint_path={visual_encoder_checkpoint_path}", flush=True)
    print(f"[rc-cont-head] bridge_modules_state_path={args.bridge_modules_state_path}", flush=True)
    print(f"[rc-cont-head] dataset_jsonl={dataset_path}", flush=True)
    print(f"[rc-cont-head] dataset_meta_jsonl={dataset_meta_path}", flush=True)
    print(f"[rc-cont-head] eval_dataset_jsonl={eval_dataset_path}", flush=True)
    print(f"[rc-cont-head] eval_dataset_meta_jsonl={eval_dataset_meta_path}", flush=True)
    print(f"[rc-cont-head] media_dir={media_dir}", flush=True)
    print(f"[rc-cont-head] output_dir={output_dir}", flush=True)
    print(f"[rc-cont-head] image_size={int(args.image_size)}", flush=True)
    print(f"[rc-cont-head] encoder_input_pad_size={int(encoder_input_pad_size)}", flush=True)
    print(f"[rc-cont-head] visual_grid_size={visual_grid_size}", flush=True)
    print(f"[rc-cont-head] num_visual_tokens={num_visual_tokens}", flush=True)
    print(f"[rc-cont-head] cutoff_len={int(args.cutoff_len)}", flush=True)
    print(f"[rc-cont-head] coord_head_hidden_dim={int(args.coord_head_hidden_dim)}", flush=True)
    print(f"[rc-cont-head] coord_loss_weight={float(args.coord_loss_weight)}", flush=True)
    print(f"[rc-cont-head] coord_loss_beta={float(args.coord_loss_beta)}", flush=True)
    print(f"[rc-cont-head] coord_use_sigmoid={bool(args.coord_use_sigmoid)}", flush=True)

    tokenizer_name_or_path = str(args.tokenizer_name_or_path).strip() or str(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        use_fast=False,
    )
    formatter = RCCenterlineContinuousHeadFormatter(
        image_size=int(args.image_size),
        num_visual_tokens=int(num_visual_tokens),
        system_prompt=str(args.system_prompt),
        user_prompt=str(args.user_prompt),
    )
    num_added = formatter.register_tokens(tokenizer)
    print(f"[rc-cont-head] added_tokens={num_added}", flush=True)

    rows = load_jsonl(dataset_path, max_samples=int(args.max_samples))
    meta_rows = load_jsonl(dataset_meta_path) if dataset_meta_path is not None else []
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    eval_rows: List[Dict[str, Any]] = []
    eval_meta_rows: List[Dict[str, Any]] = []
    if eval_dataset_path is not None:
        eval_rows = load_jsonl(eval_dataset_path, max_samples=int(args.max_eval_samples))
        eval_meta_rows = load_jsonl(eval_dataset_meta_path) if eval_dataset_meta_path is not None else []
        if not eval_rows:
            raise ValueError(f"No eval rows loaded from {eval_dataset_path}")

    train_dataset = RCCenterlineContinuousHeadDataset(
        rows=rows,
        meta_rows=meta_rows,
        media_dir=media_dir,
        tokenizer=tokenizer,
        formatter=formatter,
        image_size=int(args.image_size),
    )
    eval_dataset = (
        RCCenterlineContinuousHeadDataset(
            rows=eval_rows,
            meta_rows=eval_meta_rows,
            media_dir=media_dir,
            tokenizer=tokenizer,
            formatter=formatter,
            image_size=int(args.image_size),
        )
        if eval_rows
        else None
    )
    collator = RCCenterlineContinuousHeadCollator(
        tokenizer=tokenizer,
        cutoff_len=int(args.cutoff_len),
        num_visual_tokens=int(num_visual_tokens),
    )

    model = Qwen3RCDinoCenterlineContinuousHeadModel(
        model_name_or_path=str(args.model_name_or_path),
        tokenizer=tokenizer,
        dinov2_model_name_or_path=dinov2_model_name_or_path,
        visual_encoder_checkpoint_path=visual_encoder_checkpoint_path,
        modules_state_path=str(args.bridge_modules_state_path).strip(),
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        visual_projector_hidden_dim=int(args.visual_projector_hidden_dim),
        geometric_mlp_hidden_dim=int(args.geometric_mlp_hidden_dim),
        token_alignment_hidden_dim=int(args.token_alignment_hidden_dim),
        token_alignment_num_layers=int(args.token_alignment_num_layers),
        token_alignment_dropout=float(args.token_alignment_dropout),
        coord_head_hidden_dim=int(args.coord_head_hidden_dim),
        coord_use_sigmoid=bool(args.coord_use_sigmoid),
        language_model_dtype=str(args.model_dtype),
        local_files_only=bool(args.local_files_only),
        freeze_language_model=bool(args.freeze_language_model),
        freeze_vision_encoder=bool(args.freeze_vision_encoder),
        encoder_input_pad_size=int(encoder_input_pad_size),
        use_lora=not bool(args.no_lora),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rc-cont-head] trainable_params={trainable_params} total_params={total_params}", flush=True)

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
        gradient_checkpointing=bool(args.gradient_checkpointing),
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
        remove_unused_columns=False,
        report_to=[],
        label_names=["labels", "coord_target_values"],
    )
    training_args = create_training_arguments(
        base_kwargs=training_kwargs,
        evaluation_strategy=str(args.evaluation_strategy),
    )

    trainer = RCDinoContinuousHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer_for_save=tokenizer,
        coord_loss_weight=float(args.coord_loss_weight),
        coord_loss_beta=float(args.coord_loss_beta),
    )
    train_result = trainer.train(
        resume_from_checkpoint=(str(args.resume_from_checkpoint).strip() or None)
    )
    print(json.dumps({"train_result": train_result.metrics}, ensure_ascii=False), flush=True)
    save_qwen3_rc_dinov2_centerline_continuous_head_modules(model, output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
