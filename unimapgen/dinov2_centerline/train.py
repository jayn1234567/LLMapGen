"""Training entry for the cleaned DINOv2 centerline JSON SFT route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, Trainer

from unimapgen.data.rc_centerline_json_sft_dataset import (
    RCCenterlineJSONSFTCollator,
    RCCenterlineJSONSFTDataset,
    RCCenterlineJSONSFTFormatter,
    default_system_prompt_for_task,
    default_user_prompt_for_task,
    load_jsonl,
    normalize_map_task,
)
from unimapgen.dinov2_centerline.data import prepare_trainroot
from unimapgen.dinov2_centerline.model import (
    Qwen3RCDinoCenterlineJSONSFTModel,
    save_qwen3_rc_dinov2_centerline_json_modules,
)
from unimapgen.rc_llm_runtime import (
    create_training_arguments,
    infer_visual_layout,
    inspect_visual_encoder_checkpoint,
    resolve_meta_jsonl,
    save_run_args,
    set_random_seed,
)
from unimapgen.runtime.device import maybe_enable_npu_runtime, resolve_ddp_backend


GLOBAL_LOCAL_VIEW_PROMPT = (
    "Two visual views are provided in order. View 1 is a wider surrounding context crop "
    "resized to the model input size. View 2 is the target local patch. Use View 1 only "
    "as spatial context. Predict only the road geometry inside View 2, and output all "
    "coordinates in the View 2 patch-local coordinate system."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the minimal DINO-family -> Qwen centerline JSON SFT model. "
            "Use --trainroot for the common train.jsonl/meta_train.jsonl layout, "
            "or pass --dataset-jsonl/--media-dir explicitly."
        )
    )

    # Model and checkpoint inputs.
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Base Qwen/Qwen3 language model path.")
    parser.add_argument("--tokenizer-name-or-path", type=str, default="", help="Defaults to --model-name-or-path.")
    parser.add_argument("--dinov2-model-name-or-path", type=str, default="", help="DINOv2 ViT-L/14 checkpoint path.")
    parser.add_argument("--vision-model-name-or-path", type=str, default="", help="Generic DINO-family vision checkpoint path. Defaults to --dinov2-model-name-or-path.")
    parser.add_argument("--vision-patch-size", type=int, default=14, help="Vision encoder patch size. Use 14 for DINOv2-L/14 and 16 for DINOv3 ViT/16.")
    parser.add_argument("--vision-num-prefix-tokens", type=int, default=-1, help="Number of non-patch tokens to drop before patch tokens. -1 auto-detects CLS/register tokens.")
    parser.add_argument("--visual-encoder-checkpoint-path", type=str, default="")
    parser.add_argument("--bridge-modules-state-path", type=str, default="")
    parser.add_argument("--local-files-only", action="store_true")

    # Dataset inputs.
    parser.add_argument("--trainroot", type=str, default="", help="Root with train.jsonl, meta_train.jsonl, val.jsonl.")
    parser.add_argument("--dataset-jsonl", type=str, default="", help="Alternative explicit training jsonl.")
    parser.add_argument("--dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--eval-dataset-jsonl", type=str, default="")
    parser.add_argument("--eval-dataset-meta-jsonl", type=str, default="")
    parser.add_argument("--media-dir", type=str, default="", help="Defaults to trainroot/prepared trainroot when possible.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--use-global-local-views", action="store_true")
    parser.add_argument("--global-local-view-count", type=int, default=2)
    parser.add_argument("--context-image-key", type=str, default="context_image")
    parser.add_argument("--require-context-image", action="store_true")
    parser.add_argument("--global-local-prompt", action=argparse.BooleanOptionalAction, default=True)

    # Optional built-in target preparation.
    parser.add_argument("--prepare-trainroot", action="store_true")
    parser.add_argument("--prepared-trainroot", type=str, default="")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--douglas-epsilon-px", type=float, default=2.5)
    parser.add_argument("--merge-endpoint-tol-px", type=float, default=6.0)
    parser.add_argument("--merge-heading-tol-deg", type=float, default=22.5)
    parser.add_argument("--no-link-media-dirs", action="store_true")

    # Output and training schedule.
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--language-model-lr", type=float, default=0.0, help="Optional LR for Qwen/Qwen3 language_model params. 0 uses --learning-rate.")
    parser.add_argument("--vision-encoder-lr", type=float, default=0.0, help="Optional LR for DINO-family vision_encoder params. 0 uses --learning-rate.")
    parser.add_argument("--alignment-lr", type=float, default=0.0, help="Optional LR for visual bridge/alignment params. 0 uses --learning-rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--evaluation-strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--metric-for-best-model", type=str, default="eval_loss")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--optim", type=str, default="", help="Optional Hugging Face TrainingArguments optim, e.g. adafactor for full-parameter smoke tests.")
    parser.add_argument("--deepspeed", type=str, default="", help="Optional DeepSpeed config path for ZeRO/FSDP-style full-parameter training.")
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--device-backend",
        type=str,
        default="auto",
        choices=["auto", "cuda", "npu", "cpu"],
        help="Runtime backend. Use npu on Ascend/DI training jobs; auto picks npu, cuda, then cpu.",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="",
        help="Distributed backend override. Defaults to hccl on NPU, nccl on CUDA.",
    )

    # Model architecture knobs kept because checkpoints depend on them.
    parser.add_argument("--cutoff-len", type=int, default=7168)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--encoder-input-pad-size", type=int, default=518)
    parser.add_argument("--visual-projector-hidden-dim", type=int, default=4096)
    parser.add_argument("--geometric-mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--token-alignment-hidden-dim", type=int, default=4096)
    parser.add_argument("--token-alignment-num-layers", type=int, default=2)
    parser.add_argument("--token-alignment-dropout", type=float, default=0.0)
    parser.add_argument("--use-view-type-embedding", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--view-type-embedding-count", type=int, default=2)
    parser.add_argument("--view-type-embedding-init-std", type=float, default=0.02)
    parser.add_argument(
        "--visual-token-compressor",
        type=str,
        default="none",
        choices=["none", "learned_conv"],
        help="Optional learnable visual-token compressor before the projector.",
    )
    parser.add_argument(
        "--visual-token-compressor-grid-size",
        type=int,
        default=0,
        help="Compressed visual grid size per view. 0 keeps the raw DINOv2 grid.",
    )
    parser.add_argument("--visual-token-compressor-hidden-dim", type=int, default=512)
    parser.add_argument("--visual-token-compressor-depth", type=int, default=2)
    parser.add_argument("--visual-token-compressor-dropout", type=float, default=0.0)
    parser.add_argument("--model-dtype", type=str, default="auto")
    parser.add_argument("--freeze-language-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vision-train-last-n-layers",
        type=int,
        default=0,
        help="When --freeze-vision-encoder is true, unfreeze only the last N DINO-family transformer layers.",
    )
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Prompt contract.
    parser.add_argument("--map-task", type=str, default="lane", choices=["lane", "lane_intersection"])
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--user-prompt", type=str, default="")
    return parser.parse_args()


def resolve_prompt_contract(args: argparse.Namespace) -> None:
    args.map_task = normalize_map_task(args.map_task)
    if not str(args.system_prompt).strip():
        args.system_prompt = default_system_prompt_for_task(args.map_task)
    if not str(args.user_prompt).strip():
        args.user_prompt = default_user_prompt_for_task(args.map_task)
    if bool(args.use_global_local_views) and bool(args.global_local_prompt):
        if GLOBAL_LOCAL_VIEW_PROMPT not in str(args.user_prompt):
            args.user_prompt = f"{GLOBAL_LOCAL_VIEW_PROMPT}\n\n{args.user_prompt}"


def resolve_dataset_paths(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path | None, Path | None, Path | None, Path]:
    trainroot_arg = str(args.trainroot).strip()
    if trainroot_arg:
        trainroot = Path(trainroot_arg).expanduser().resolve()
        if bool(args.prepare_trainroot):
            prepared = (
                Path(str(args.prepared_trainroot)).expanduser().resolve()
                if str(args.prepared_trainroot).strip()
                else (output_dir / "prepared_douglas_merge_trainroot").resolve()
            )
            print(
                json.dumps(
                    {
                        "stage": "prepare_trainroot",
                        "input_root": str(trainroot),
                        "output_root": str(prepared),
                        "douglas_epsilon_px": float(args.douglas_epsilon_px),
                        "merge_endpoint_tol_px": float(args.merge_endpoint_tol_px),
                        "merge_heading_tol_deg": float(args.merge_heading_tol_deg),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            prepare_trainroot(
                input_root=trainroot,
                output_root=prepared,
                splits=("train", "val"),
                patch_size=int(args.patch_size),
                douglas_epsilon_px=float(args.douglas_epsilon_px),
                merge_endpoint_tol_px=float(args.merge_endpoint_tol_px),
                merge_heading_tol_deg=float(args.merge_heading_tol_deg),
                link_media_dirs=not bool(args.no_link_media_dirs),
            )
            trainroot = prepared

        dataset_path = trainroot / "train.jsonl"
        dataset_meta_path = trainroot / "meta_train.jsonl"
        eval_dataset_path = trainroot / "val.jsonl"
        eval_meta_path = trainroot / "meta_val.jsonl"
        media_dir = Path(str(args.media_dir)).expanduser().resolve() if str(args.media_dir).strip() else trainroot
        return (
            dataset_path,
            dataset_meta_path if dataset_meta_path.is_file() else None,
            eval_dataset_path if eval_dataset_path.is_file() else None,
            eval_meta_path if eval_meta_path.is_file() else None,
            media_dir,
        )

    if bool(args.prepare_trainroot):
        raise ValueError("--prepare-trainroot requires --trainroot.")
    if not str(args.dataset_jsonl).strip():
        raise ValueError("Pass either --trainroot or --dataset-jsonl.")
    if not str(args.media_dir).strip():
        raise ValueError("--media-dir is required when --trainroot is not used.")

    dataset_path = Path(str(args.dataset_jsonl)).expanduser().resolve()
    dataset_meta_path = resolve_meta_jsonl(dataset_path, str(args.dataset_meta_jsonl))
    eval_dataset_path = (
        Path(str(args.eval_dataset_jsonl)).expanduser().resolve() if str(args.eval_dataset_jsonl).strip() else None
    )
    eval_meta_path = (
        resolve_meta_jsonl(eval_dataset_path, str(args.eval_dataset_meta_jsonl))
        if eval_dataset_path is not None
        else None
    )
    media_dir = Path(str(args.media_dir)).expanduser().resolve()
    return dataset_path, dataset_meta_path, eval_dataset_path, eval_meta_path, media_dir



def _positive_or_default(value: float, default: float) -> float:
    value = float(value)
    return value if value > 0 else float(default)


def _parameter_lr_group(name: str, args: argparse.Namespace) -> tuple[str, float]:
    base_lr = float(args.learning_rate)
    language_lr = _positive_or_default(float(args.language_model_lr), base_lr)
    vision_lr = _positive_or_default(float(args.vision_encoder_lr), base_lr)
    alignment_lr = _positive_or_default(float(args.alignment_lr), base_lr)

    if name.startswith("vision_encoder."):
        return "vision_encoder", vision_lr
    if name.startswith("language_model."):
        return "language_model", language_lr
    if name.startswith(
        (
            "visual_norm.",
            "visual_token_compressor.",
            "visual_projector.",
            "geometric_position_mlp.",
            "token_alignment.",
            "view_type_embeddings.",
            "special_token_adapter.",
        )
    ):
        return "alignment", alignment_lr
    return "other_trainable", alignment_lr


def _uses_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    lname = name.lower()
    if param.ndim < 2:
        return False
    if name.endswith(".bias"):
        return False
    if any(token in lname for token in ("norm", "layernorm", "layer_norm", "embedding")):
        return False
    return True


def build_grouped_lr_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> Optional[torch.optim.Optimizer]:
    if not any(float(value) > 0 for value in (args.language_model_lr, args.vision_encoder_lr, args.alignment_lr)):
        return None

    grouped: Dict[tuple[str, float, float], Dict[str, Any]] = {}
    summaries: Dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name, lr = _parameter_lr_group(name, args)
        weight_decay = float(args.weight_decay) if _uses_weight_decay(name, param) else 0.0
        key = (group_name, float(lr), float(weight_decay))
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "name": group_name,
            }
        grouped[key]["params"].append(param)
        summaries[group_name] = summaries.get(group_name, 0) + int(param.numel())

    param_groups = list(grouped.values())
    print(
        json.dumps(
            {
                "stage": "grouped_lr_optimizer",
                "base_lr": float(args.learning_rate),
                "language_model_lr": _positive_or_default(float(args.language_model_lr), float(args.learning_rate)),
                "vision_encoder_lr": _positive_or_default(float(args.vision_encoder_lr), float(args.learning_rate)),
                "alignment_lr": _positive_or_default(float(args.alignment_lr), float(args.learning_rate)),
                "group_param_counts": summaries,
                "num_optimizer_groups": len(param_groups),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return torch.optim.AdamW(param_groups, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

def main() -> None:
    args = parse_args()
    if bool(args.use_global_local_views):
        args.require_context_image = True
    if args.use_view_type_embedding is None:
        args.use_view_type_embedding = bool(args.use_global_local_views)
    resolve_prompt_contract(args)
    resolved_backend = maybe_enable_npu_runtime(str(args.device_backend))
    args.resolved_device_backend = str(resolved_backend)
    args.ddp_backend = resolve_ddp_backend(str(resolved_backend), str(args.ddp_backend)) or ""
    set_random_seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()

    dataset_path, dataset_meta_path, eval_dataset_path, eval_meta_path, media_dir = resolve_dataset_paths(args, output_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_path}")
    if eval_dataset_path is not None and not eval_dataset_path.is_file():
        raise FileNotFoundError(f"eval_dataset_jsonl not found: {eval_dataset_path}")
    if not media_dir.exists():
        raise FileNotFoundError(f"media_dir not found: {media_dir}")

    inferred_ckpt_args: Dict[str, Any] = {}
    visual_encoder_checkpoint_path = str(args.visual_encoder_checkpoint_path).strip()
    if visual_encoder_checkpoint_path:
        inferred_ckpt_args = inspect_visual_encoder_checkpoint(visual_encoder_checkpoint_path)

    vision_model_name_or_path = str(args.vision_model_name_or_path).strip() or str(
        args.dinov2_model_name_or_path
    ).strip() or str(
        inferred_ckpt_args.get("dinov2_model_name_or_path", "")
    ).strip()
    if not vision_model_name_or_path:
        raise ValueError(
            "vision_model_name_or_path is required. Pass --vision-model-name-or-path or --dinov2-model-name-or-path, "
            "or provide --visual-encoder-checkpoint-path with saved args."
        )
    args.vision_model_name_or_path = vision_model_name_or_path
    args.dinov2_model_name_or_path = str(args.dinov2_model_name_or_path).strip() or vision_model_name_or_path
    args.vision_patch_size = int(args.vision_patch_size)
    args.vision_num_prefix_tokens = int(args.vision_num_prefix_tokens)

    encoder_input_pad_size = int(args.encoder_input_pad_size)
    if encoder_input_pad_size <= 0:
        encoder_input_pad_size = int(inferred_ckpt_args.get("encoder_input_pad_size", 0))
    args.encoder_input_pad_size = int(encoder_input_pad_size)
    encoder_visual_grid_size, encoder_tokens_per_view = infer_visual_layout(
        image_size=int(args.image_size),
        encoder_input_pad_size=int(encoder_input_pad_size),
        patch_size=int(args.vision_patch_size),
    )
    visual_token_compressor = str(args.visual_token_compressor).strip().lower()
    if visual_token_compressor == "none":
        visual_grid_size = int(encoder_visual_grid_size)
    else:
        visual_grid_size = int(args.visual_token_compressor_grid_size)
        if visual_grid_size <= 0:
            raise ValueError("--visual-token-compressor-grid-size must be > 0 when compression is enabled.")
        if visual_grid_size > int(encoder_visual_grid_size):
            raise ValueError(
                f"visual_token_compressor_grid_size={visual_grid_size} cannot exceed raw encoder grid "
                f"{encoder_visual_grid_size}."
            )
    tokens_per_view = int(visual_grid_size) * int(visual_grid_size)
    num_visual_views = int(args.global_local_view_count) if bool(args.use_global_local_views) else 1
    if num_visual_views <= 0:
        raise ValueError("--global-local-view-count must be positive.")
    num_visual_tokens = int(tokens_per_view) * int(num_visual_views)
    args.encoder_visual_grid_size = int(encoder_visual_grid_size)
    args.encoder_tokens_per_view = int(encoder_tokens_per_view)
    args.effective_visual_grid_size = int(visual_grid_size)
    args.effective_tokens_per_view = int(tokens_per_view)
    args.effective_num_visual_tokens = int(num_visual_tokens)
    args.effective_dataset_jsonl = str(dataset_path)
    args.effective_dataset_meta_jsonl = str(dataset_meta_path or "")
    args.effective_eval_dataset_jsonl = str(eval_dataset_path or "")
    args.effective_eval_dataset_meta_jsonl = str(eval_meta_path or "")
    args.effective_media_dir = str(media_dir)
    args.tokens_per_view = int(tokens_per_view)
    args.num_visual_views = int(num_visual_views)
    args.num_visual_tokens = int(num_visual_tokens)
    save_run_args(output_dir, args)

    print(
        json.dumps(
            {
                "stage": "dinov2_centerline_train_setup",
                "model": str(args.model_name_or_path),
                "dinov2_model": args.dinov2_model_name_or_path,
                "vision_model": vision_model_name_or_path,
                "vision_patch_size": int(args.vision_patch_size),
                "vision_num_prefix_tokens": int(args.vision_num_prefix_tokens),
                "dataset_jsonl": str(dataset_path),
                "dataset_meta_jsonl": str(dataset_meta_path or ""),
                "eval_dataset_jsonl": str(eval_dataset_path or ""),
                "eval_dataset_meta_jsonl": str(eval_meta_path or ""),
                "media_dir": str(media_dir),
                "output_dir": str(output_dir),
                "image_size": int(args.image_size),
                "encoder_input_pad_size": int(encoder_input_pad_size),
                "encoder_visual_grid_size": int(encoder_visual_grid_size),
                "encoder_tokens_per_view": int(encoder_tokens_per_view),
                "visual_grid_size": int(visual_grid_size),
                "tokens_per_view": int(tokens_per_view),
                "num_visual_views": int(num_visual_views),
                "num_visual_tokens": int(num_visual_tokens),
                "use_global_local_views": bool(args.use_global_local_views),
                "context_image_key": str(args.context_image_key),
                "use_view_type_embedding": bool(args.use_view_type_embedding),
                "visual_token_compressor": str(args.visual_token_compressor),
                "visual_token_compressor_grid_size": int(args.visual_token_compressor_grid_size),
                "map_task": str(args.map_task),
                "use_lora": not bool(args.no_lora),
                "language_model_lr": float(args.language_model_lr),
                "vision_encoder_lr": float(args.vision_encoder_lr),
                "alignment_lr": float(args.alignment_lr),
                "freeze_vision_encoder": bool(args.freeze_vision_encoder),
                "vision_train_last_n_layers": int(args.vision_train_last_n_layers),
                "device_backend": str(args.resolved_device_backend),
                "ddp_backend": str(args.ddp_backend),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_name_or_path).strip() or str(args.model_name_or_path),
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        use_fast=False,
    )
    formatter = RCCenterlineJSONSFTFormatter(
        image_size=int(args.image_size),
        num_visual_tokens=int(num_visual_tokens),
        system_prompt=str(args.system_prompt),
        user_prompt=str(args.user_prompt),
    )
    print(f"[dinov2-centerline] added_tokens={formatter.register_tokens(tokenizer)}", flush=True)

    rows = load_jsonl(dataset_path, max_samples=int(args.max_samples))
    if not rows:
        raise ValueError(f"No training rows loaded from {dataset_path}")
    meta_rows = load_jsonl(dataset_meta_path) if dataset_meta_path is not None else []

    eval_rows: List[Dict[str, Any]] = []
    eval_meta_rows: List[Dict[str, Any]] = []
    if eval_dataset_path is not None:
        eval_rows = load_jsonl(eval_dataset_path, max_samples=int(args.max_eval_samples))
        eval_meta_rows = load_jsonl(eval_meta_path) if eval_meta_path is not None else []
        if not eval_rows:
            raise ValueError(f"No eval rows loaded from {eval_dataset_path}")

    train_dataset = RCCenterlineJSONSFTDataset(
        rows=rows,
        meta_rows=meta_rows,
        media_dir=media_dir,
        tokenizer=tokenizer,
        formatter=formatter,
        image_size=int(args.image_size),
        context_image_key=str(args.context_image_key),
        require_context_image=bool(args.require_context_image),
    )
    eval_dataset = (
        RCCenterlineJSONSFTDataset(
            rows=eval_rows,
            meta_rows=eval_meta_rows,
            media_dir=media_dir,
            tokenizer=tokenizer,
            formatter=formatter,
            image_size=int(args.image_size),
            context_image_key=str(args.context_image_key),
            require_context_image=bool(args.require_context_image),
        )
        if eval_rows
        else None
    )
    collator = RCCenterlineJSONSFTCollator(
        tokenizer=tokenizer,
        cutoff_len=int(args.cutoff_len),
        num_visual_tokens=int(num_visual_tokens),
    )

    model = Qwen3RCDinoCenterlineJSONSFTModel(
        model_name_or_path=str(args.model_name_or_path),
        tokenizer=tokenizer,
        dinov2_model_name_or_path=str(args.dinov2_model_name_or_path),
        vision_model_name_or_path=str(args.vision_model_name_or_path),
        vision_patch_size=int(args.vision_patch_size),
        vision_num_prefix_tokens=int(args.vision_num_prefix_tokens),
        visual_encoder_checkpoint_path=visual_encoder_checkpoint_path,
        modules_state_path=str(args.bridge_modules_state_path).strip(),
        num_visual_tokens=int(num_visual_tokens),
        visual_grid_size=int(visual_grid_size),
        encoder_visual_grid_size=int(encoder_visual_grid_size),
        num_visual_views=int(num_visual_views),
        visual_projector_hidden_dim=int(args.visual_projector_hidden_dim),
        geometric_mlp_hidden_dim=int(args.geometric_mlp_hidden_dim),
        token_alignment_hidden_dim=int(args.token_alignment_hidden_dim),
        token_alignment_num_layers=int(args.token_alignment_num_layers),
        token_alignment_dropout=float(args.token_alignment_dropout),
        visual_token_compressor=str(args.visual_token_compressor),
        visual_token_compressor_hidden_dim=int(args.visual_token_compressor_hidden_dim),
        visual_token_compressor_depth=int(args.visual_token_compressor_depth),
        visual_token_compressor_dropout=float(args.visual_token_compressor_dropout),
        use_view_type_embedding=bool(args.use_view_type_embedding),
        view_type_embedding_count=int(args.view_type_embedding_count),
        view_type_embedding_init_std=float(args.view_type_embedding_init_std),
        language_model_dtype=str(args.model_dtype),
        local_files_only=bool(args.local_files_only),
        freeze_language_model=bool(args.freeze_language_model),
        freeze_vision_encoder=bool(args.freeze_vision_encoder),
        vision_train_last_n_layers=int(args.vision_train_last_n_layers),
        encoder_input_pad_size=int(encoder_input_pad_size),
        use_lora=not bool(args.no_lora),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[dinov2-centerline] trainable_params={trainable_params} total_params={total_params}", flush=True)

    if bool(args.load_best_model_at_end) and eval_dataset is None:
        raise ValueError("--load-best-model-at-end requires an eval dataset.")
    eval_strategy = str(args.evaluation_strategy) if eval_dataset is not None else "no"
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "num_train_epochs": float(args.num_train_epochs),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "logging_steps": int(args.logging_steps),
        "save_strategy": str(args.save_strategy),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "eval_steps": (int(args.eval_steps) if int(args.eval_steps) > 0 else None),
        "load_best_model_at_end": bool(args.load_best_model_at_end),
        "metric_for_best_model": (str(args.metric_for_best_model).strip() or None),
        "greater_is_better": args.greater_is_better,
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "bf16": bool(args.bf16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
        "ddp_backend": (str(args.ddp_backend).strip() or None),
        "remove_unused_columns": False,
        "report_to": [],
        "label_names": ["labels"],
    }
    if str(args.optim).strip():
        training_kwargs["optim"] = str(args.optim).strip()
    if str(args.deepspeed).strip():
        training_kwargs["deepspeed"] = str(args.deepspeed).strip()

    training_args = create_training_arguments(
        base_kwargs=training_kwargs,
        evaluation_strategy=eval_strategy,
    )

    optimizer = build_grouped_lr_optimizer(model, args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        optimizers=((optimizer, None) if optimizer is not None else (None, None)),
    )
    result = trainer.train(resume_from_checkpoint=(str(args.resume_from_checkpoint).strip() or None))
    print(json.dumps({"train_result": result.metrics}, ensure_ascii=False), flush=True)
    save_qwen3_rc_dinov2_centerline_json_modules(model, output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
