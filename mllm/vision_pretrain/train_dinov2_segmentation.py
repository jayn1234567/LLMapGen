from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from .data import (
    RoadLaneSegmentationDataset,
    discover_segmentation_samples,
    save_split_manifest,
    seed_worker,
)
from .dinov2_segmentation import Dinov2RoadSegmentationModel
from .metrics import confusion_matrix, metrics_from_confusion, segmentation_loss


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hugging Face DINOv2 road-lane segmentation pretraining."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_roots", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=[6, 12, 18, 24])
    parser.add_argument("--projection_channels", type=int, default=256)
    parser.add_argument(
        "--decoder_type",
        choices=("multilayer_weighted", "legacy_single_layer", "dinov3_style_fpn"),
        default="multilayer_weighted",
    )
    parser.add_argument(
        "--normalization_mode",
        choices=("processor", "minus_half"),
        default="processor",
        help="processor uses the HF image processor stats; minus_half applies image / 255 - 0.5.",
    )
    parser.add_argument(
        "--split_strategy",
        choices=("hash_group", "ordered_per_root"),
        default="hash_group",
        help="hash_group is the existing group split; ordered_per_root mirrors the pasted DINOv3 recipe.",
    )
    parser.add_argument(
        "--vision_unfreeze_last_n_blocks",
        type=int,
        default=-1,
        help="Negative trains the full backbone; otherwise train only the final N blocks and final norm.",
    )
    parser.add_argument("--vision_lora_enable", type=parse_bool, default=False)
    parser.add_argument("--vision_lora_r", type=int, default=8)
    parser.add_argument("--vision_lora_alpha", type=float, default=16.0)
    parser.add_argument("--vision_lora_dropout", type=float, default=0.0)
    parser.add_argument("--vision_lora_target_modules", default="query,value")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        help="Non-negative overrides warmup_ratio with a fixed optimizer-step count.",
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--foreground_ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.5)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--ignore_mask_value", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument(
        "--best_metric",
        choices=("loss", "mean_iou", "lane_iou", "lane_f1"),
        default="mean_iou",
    )
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--augment", type=parse_bool, default=True)
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    return parser


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def initialize_runtime(requested_device: str) -> tuple[torch.device, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device_type = requested_device
    if device_type == "auto":
        if _npu_available():
            device_type = "npu"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

    if device_type == "npu":
        if not _npu_available():
            raise RuntimeError("NPU was requested, but torch_npu/NPU is not available.")
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
        backend = "hccl"
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return device, rank, local_rank, world_size


def set_seed(seed: int, rank: int) -> None:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def rank0_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def _parameter_groups(
    module: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    prefix: str,
) -> list[dict]:
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    groups = []
    if decay_parameters:
        groups.append(
            {
                "params": decay_parameters,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "group_name": f"{prefix}_decay",
            }
        )
    if no_decay_parameters:
        groups.append(
            {
                "params": no_decay_parameters,
                "lr": float(learning_rate),
                "weight_decay": 0.0,
                "group_name": f"{prefix}_no_decay",
            }
        )
    return groups


def build_optimizer(model: Dinov2RoadSegmentationModel, args: argparse.Namespace) -> AdamW:
    groups = _parameter_groups(
        model.vision_encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        prefix="vision_encoder",
    )
    groups.extend(
        _parameter_groups(
            model.decoder,
            learning_rate=args.decoder_learning_rate,
            weight_decay=args.weight_decay,
            prefix="segmentation_decoder",
        )
    )
    return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
    warmup_steps: int = -1,
    min_lr_ratio: float,
) -> LambdaLR:
    resolved_warmup_steps = (
        int(warmup_steps)
        if int(warmup_steps) >= 0
        else max(0, round(total_steps * float(warmup_ratio)))
    )

    def lr_lambda(step: int) -> float:
        if resolved_warmup_steps > 0 and step < resolved_warmup_steps:
            return max(1e-8, float(step + 1) / float(resolved_warmup_steps))
        progress = (step - resolved_warmup_steps) / max(
            1,
            total_steps - resolved_warmup_steps,
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def is_better_metric(metric_name: str, candidate: float, incumbent: float) -> bool:
    if metric_name == "loss":
        return candidate < incumbent
    return candidate > incumbent


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    world_size: int,
) -> dict:
    model.eval()
    # HCCL supports float32 reduction but not torch.float64.
    matrix = torch.zeros((2, 2), dtype=torch.float32, device=device)
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = torch.zeros((), dtype=torch.float32, device=device)
    local_samples = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with _autocast_context(device, args.bf16):
                logits = model(pixel_values)
                loss, _ = segmentation_loss(
                    logits,
                    labels,
                    foreground_weight=args.foreground_ce_weight,
                    dice_weight=args.dice_loss_weight,
                )
            batch_size = labels.shape[0]
            loss_sum += loss.detach().float() * batch_size
            sample_count += batch_size
            matrix += confusion_matrix(logits, labels).to(device=device)
            local_samples += batch_size

    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = float((loss_sum / sample_count.clamp_min(1.0)).item())
    elapsed = max(time.perf_counter() - started, 1e-6)
    metrics["throughput_samples_per_second_per_npu"] = local_samples / elapsed
    return metrics


def save_best_artifacts(
    model: Dinov2RoadSegmentationModel,
    image_processor: object,
    output_dir: Path,
    *,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    metrics: dict,
    dataset_report: dict,
) -> None:
    best_dir = output_dir / "best"
    vision_dir = best_dir / "vision_tower"
    best_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": args.model_name_or_path,
        "architecture": "Dinov2RoadSegmentationModel",
        "input_size": args.input_size,
        "patch_size": model.patch_size,
        "visual_grid_size": args.input_size // model.patch_size,
        "hidden_state_indices": list(model.hidden_state_indices),
        "decoder_type": model.decoder_type,
        "normalization_mode": args.normalization_mode,
        "image_mean": list(image_processor.image_mean),
        "image_std": list(image_processor.image_std),
        "split_strategy": args.split_strategy,
        "vision_unfreeze_last_n_blocks": model.vision_unfreeze_last_n_blocks,
        "vision_lora_enable": model.vision_lora_enable,
        "vision_lora_r": model.vision_lora_r,
        "vision_lora_alpha": model.vision_lora_alpha,
        "vision_lora_dropout": model.vision_lora_dropout,
        "vision_lora_target_modules": model.vision_lora_target_modules,
        "vision_lora_modules": list(model.vision_lora_modules),
        "gradient_checkpointing_mode": model.gradient_checkpointing_mode,
        "trainable_vision_block_indices": (
            None
            if model.trainable_vision_block_indices is None
            else list(model.trainable_vision_block_indices)
        ),
        "vision_training": (
            "lora_merged_on_export"
            if model.vision_lora_enable
            else (
            "full_parameter_except_unused_mask_token"
            if model.vision_unfreeze_last_n_blocks < 0
            else f"last_{model.vision_unfreeze_last_n_blocks}_blocks_plus_final_norm"
            )
        ),
        "best_metric": args.best_metric,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_metrics": metrics,
        "dataset_report": dataset_report,
    }
    model.save_vision_tower(vision_dir, image_processor, metadata)
    torch.save(
        {
            "decoder": model.head_state_dict(),
            "metadata": metadata,
        },
        best_dir / "segmentation_head.pt",
    )
    (best_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    device, rank, local_rank, world_size = initialize_runtime(args.device)
    set_seed(args.split_seed, rank)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        use_fast=False,
    )
    image_processor.size = {"shortest_edge": int(args.input_size)}
    image_processor.crop_size = {"height": int(args.input_size), "width": int(args.input_size)}
    image_mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
    image_std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])
    if args.normalization_mode == "minus_half":
        image_mean = [0.5, 0.5, 0.5]
        image_std = [1.0, 1.0, 1.0]
        image_processor.image_mean = list(image_mean)
        image_processor.image_std = list(image_std)

    train_samples, val_samples, discovery_report = discover_segmentation_samples(
        args.dataset_roots,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        split_strategy=args.split_strategy,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    if rank == 0:
        save_split_manifest(
            output_dir / "split_manifest.json",
            train_samples,
            val_samples,
            discovery_report,
        )
    dataset_report = {
        "roots": list(discovery_report.roots),
        "total_samples": discovery_report.total_samples,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "groups": discovery_report.groups,
        "missing_images": discovery_report.missing_images,
    }
    rank0_print(rank, f"[dinov2-seg] dataset={json.dumps(dataset_report, ensure_ascii=True)}")

    train_dataset = RoadLaneSegmentationDataset(
        train_samples,
        input_size=args.input_size,
        image_mean=image_mean,
        image_std=image_std,
        augment=args.augment,
        ignore_mask_value=args.ignore_mask_value,
    )
    val_dataset = RoadLaneSegmentationDataset(
        val_samples,
        input_size=args.input_size,
        image_mean=image_mean,
        image_std=image_std,
        augment=False,
        ignore_mask_value=args.ignore_mask_value,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.split_seed,
        )
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )
    loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type != "cpu",
        "worker_init_fn": seed_worker,
        "persistent_workers": int(args.num_workers) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    if not len(train_loader):
        raise RuntimeError("Training dataloader is empty. Reduce batch size or provide more samples.")

    raw_model = Dinov2RoadSegmentationModel.from_pretrained(
        args.model_name_or_path,
        input_size=args.input_size,
        hidden_state_indices=args.hidden_state_indices,
        projection_channels=args.projection_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        decoder_type=args.decoder_type,
        vision_unfreeze_last_n_blocks=args.vision_unfreeze_last_n_blocks,
        vision_lora_enable=args.vision_lora_enable,
        vision_lora_r=args.vision_lora_r,
        vision_lora_alpha=args.vision_lora_alpha,
        vision_lora_dropout=args.vision_lora_dropout,
        vision_lora_target_modules=args.vision_lora_target_modules,
    )
    raw_model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    frozen_parameters = [
        name for name, parameter in raw_model.named_parameters() if not parameter.requires_grad
    ]
    rank0_print(
        rank,
        f"[dinov2-seg] parameters trainable={trainable_parameters:,} total={total_parameters:,} "
        f"decoder_type={raw_model.decoder_type} "
        f"vision_unfreeze_last_n_blocks={raw_model.vision_unfreeze_last_n_blocks} "
        f"vision_lora_enable={raw_model.vision_lora_enable} "
        f"vision_lora_modules={raw_model.vision_lora_modules[:12]} "
        f"gradient_checkpointing_mode={raw_model.gradient_checkpointing_mode} "
        f"trainable_vision_block_indices={raw_model.trainable_vision_block_indices} "
        f"frozen_parameter_tensors={len(frozen_parameters)} "
        f"frozen_preview={json.dumps(frozen_parameters[:20])}",
    )
    optimizer = build_optimizer(raw_model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = (
        int(args.max_steps)
        if int(args.max_steps) > 0
        else int(args.num_train_epochs) * updates_per_epoch
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    model: torch.nn.Module = raw_model
    if world_size > 1:
        device_ids = [local_rank] if device.type in {"npu", "cuda"} else None
        model = DistributedDataParallel(
            raw_model,
            device_ids=device_ids,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    resolved = vars(args).copy()
    resolved.update(
        {
            "device": str(device),
            "rank": rank,
            "world_size": world_size,
            "total_steps": total_steps,
            "effective_global_batch_size": (
                args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
            ),
        }
    )
    rank0_print(rank, f"[dinov2-seg] config={json.dumps(resolved, ensure_ascii=True, default=str)}")
    if rank == 0:
        (output_dir / "training_args.json").write_text(
            json.dumps(resolved, indent=2, ensure_ascii=True, default=str),
            encoding="utf-8",
        )

    global_step = 0
    best_metric_value = math.inf if args.best_metric == "loss" else -math.inf
    best_metrics: dict | None = None
    stop_training = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(args.num_train_epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        group_loss = 0.0
        group_ce = 0.0
        group_dice = 0.0
        group_samples = 0
        group_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(train_loader) - group_start)
            should_update = (batch_index + 1) % args.gradient_accumulation_steps == 0 or (
                batch_index + 1 == len(train_loader)
            )
            sync_context = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not should_update:
                sync_context = model.no_sync()

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with sync_context:
                with _autocast_context(device, args.bf16):
                    logits = model(pixel_values)
                    loss, components = segmentation_loss(
                        logits,
                        labels,
                        foreground_weight=args.foreground_ce_weight,
                        dice_weight=args.dice_loss_weight,
                    )
                (loss / group_size).backward()

            if epoch == 1 and batch_index == 0:
                missing_gradients = [
                    name
                    for name, parameter in raw_model.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                if missing_gradients:
                    raise RuntimeError(
                        "Trainable parameters did not receive gradients on the first backward pass: "
                        f"count={len(missing_gradients)}, preview={missing_gradients[:32]}. "
                        "This usually indicates an invalid gradient-checkpointing/adapter configuration."
                    )
                rank0_print(
                    rank,
                    "[dinov2-seg] first-backward gradient audit passed: "
                    f"trainable_tensors={sum(parameter.requires_grad for parameter in raw_model.parameters())}",
                )

            group_loss += float(loss.detach().item())
            group_ce += float(components["cross_entropy"].item())
            group_dice += float(components["dice_loss"].item())
            group_samples += int(labels.shape[0])
            if not should_update:
                continue

            if float(args.max_grad_norm) > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    args.max_grad_norm,
                )
            else:
                gradient_norm = torch.tensor(float("nan"), device=device)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                averaged_loss = reduce_mean(
                    torch.tensor(group_loss / group_size, device=device),
                    world_size,
                )
                elapsed = max(time.perf_counter() - group_started, 1e-6)
                throughput = group_samples / elapsed
                if rank == 0:
                    valid_pixels = labels != 255
                    target_foreground_ratio = (
                        ((labels == 1) & valid_pixels).sum().float()
                        / valid_pixels.sum().clamp_min(1).float()
                    )
                    predicted_foreground_ratio = (
                        ((logits.argmax(dim=1) == 1) & valid_pixels).sum().float()
                        / valid_pixels.sum().clamp_min(1).float()
                    )
                    log_payload = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(averaged_loss.item()),
                        "cross_entropy": group_ce / group_size,
                        "dice_loss": group_dice / group_size,
                        "gradient_norm": float(gradient_norm.detach().float().item()),
                        "target_foreground_ratio": float(target_foreground_ratio.item()),
                        "predicted_foreground_ratio": float(predicted_foreground_ratio.item()),
                        "vision_lr": optimizer.param_groups[0]["lr"],
                        "DI_throughput": f"{throughput:.2f} samples/s/npu",
                    }
                    print(json.dumps(log_payload, ensure_ascii=True), flush=True)
                    print(f"DI_throughput: {throughput:.2f} samples/s/npu", flush=True)
            group_loss = 0.0
            group_ce = 0.0
            group_dice = 0.0
            group_samples = 0
            group_started = time.perf_counter()
            if global_step >= total_steps:
                stop_training = True
                break

        should_evaluate = epoch % args.eval_every_epochs == 0 or stop_training
        if should_evaluate:
            metrics = evaluate(
                model,
                val_loader,
                device=device,
                args=args,
                world_size=world_size,
            )
            if rank == 0:
                print(f"[dinov2-seg] eval={json.dumps(metrics, ensure_ascii=True)}", flush=True)
                print(
                    "DI_throughput: "
                    f"{metrics['throughput_samples_per_second_per_npu']:.2f} samples/s/npu",
                    flush=True,
                )
                candidate_metric = float(metrics[args.best_metric])
                if is_better_metric(args.best_metric, candidate_metric, best_metric_value):
                    best_metric_value = candidate_metric
                    best_metrics = metrics
                    save_best_artifacts(
                        raw_model,
                        image_processor,
                        output_dir,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        metrics=metrics,
                        dataset_report=dataset_report,
                    )
                    print(
                        f"[dinov2-seg] saved best vision tower: "
                        f"metric={args.best_metric} value={candidate_metric:.8f} "
                        f"path={output_dir / 'best' / 'vision_tower'}",
                        flush=True,
                    )
            if world_size > 1:
                dist.barrier()
        if stop_training:
            break

    if rank == 0:
        summary = {
            "global_step": global_step,
            "best_metric": args.best_metric,
            "best_metric_value": best_metric_value,
            "best_mean_iou": None if best_metrics is None else best_metrics.get("mean_iou"),
            "best_metrics": best_metrics,
            "vision_tower": str(output_dir / "best" / "vision_tower"),
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"[dinov2-seg] complete={json.dumps(summary, ensure_ascii=True)}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
