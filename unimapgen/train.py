import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimapgen.data import build_dataset_from_cfg, collate_fn
from unimapgen.models import build_model_from_cfg
from unimapgen.utils import cosine_lr, ensure_dir, load_yaml, set_seed


def maybe_load_init_checkpoint(model: torch.nn.Module, init_checkpoint: str) -> None:
    if not init_checkpoint:
        return
    ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[Init] Loaded checkpoint={init_checkpoint} "
        f"(missing={len(missing)} unexpected={len(unexpected)})"
    )


def freeze_parameters_by_prefix(model: torch.nn.Module, prefixes) -> None:
    if not prefixes:
        return
    prefixes = [str(p) for p in prefixes if str(p).strip()]
    if not prefixes:
        return
    frozen = 0
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = False
            frozen += p.numel()
    ratio = 100.0 * float(frozen) / float(max(1, total))
    print(f"[Init] Freeze prefixes={prefixes} frozen={frozen}/{total} ({ratio:.2f}%)")


def apply_modality_mask(
    image: torch.Tensor,
    pv_images: torch.Tensor,
    bev_drop_prob: float,
    pv_drop_prob: float,
):
    """
    Randomly masks BEV/PV modalities during training for robustness.
    Ensures each sample keeps at least one visual modality.
    """
    if pv_images is None:
        return image, pv_images, 0.0, 0.0
    b = image.shape[0]
    device = image.device
    keep_bev = torch.rand(b, device=device) >= float(bev_drop_prob)
    keep_pv = torch.rand(b, device=device) >= float(pv_drop_prob)
    both_drop = (~keep_bev) & (~keep_pv)
    keep_bev = keep_bev | both_drop

    bev_mask = keep_bev.view(b, 1, 1, 1).to(dtype=image.dtype)
    pv_mask = keep_pv.view(b, 1, 1, 1, 1).to(dtype=pv_images.dtype)
    image = image * bev_mask
    pv_images = pv_images * pv_mask
    bev_drop_ratio = float((~keep_bev).float().mean().item())
    pv_drop_ratio = float((~keep_pv).float().mean().item())
    return image, pv_images, bev_drop_ratio, pv_drop_ratio


def build_datasets(cfg):
    dcfg = cfg["data"]
    train_set = build_dataset_from_cfg(
        cfg,
        split=dcfg["train_split"],
        max_samples=dcfg.get("max_train_samples"),
        train_augment=bool(dcfg.get("train_augment", False)),
    )
    val_set = build_dataset_from_cfg(
        cfg,
        split=dcfg["val_split"],
        max_samples=dcfg.get("max_val_samples"),
        train_augment=False,
    )
    return train_set, val_set


def run_val(model, loader, device, pad_id, eos_id, label_smoothing, eos_loss_weight):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_tok = 0
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            pv_images = batch.get("pv_images")
            if pv_images is not None:
                pv_images = pv_images.to(device, non_blocking=True)
            prompt_types = batch.get("prompt_type")
            if prompt_types is not None:
                prompt_types = prompt_types.to(device, non_blocking=True)
            prompt_tokens = batch.get("prompt_tokens")
            if prompt_tokens is not None:
                prompt_tokens = prompt_tokens.to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            logits = model(
                image,
                inp,
                pv_images=pv_images,
                prompt_types=prompt_types,
                prompt_tokens=prompt_tokens,
            )
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
                label_smoothing=float(label_smoothing),
            ).view_as(tgt)
            if float(eos_loss_weight) != 1.0:
                per_tok = per_tok * torch.where(
                    tgt.eq(int(eos_id)),
                    torch.full_like(per_tok, float(eos_loss_weight)),
                    torch.ones_like(per_tok),
                )
            valid = loss_mask & tgt.ne(pad_id)
            loss = (per_tok * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            total_loss += float(loss.item()) * image.shape[0]
            total_count += image.shape[0]

            pred = logits.argmax(dim=-1)
            mask = tgt.ne(pad_id)
            total_tok += int(mask.sum().item())
            total_correct += int((pred.eq(tgt) & mask).sum().item())
    mean_loss = total_loss / max(total_count, 1)
    tok_acc = total_correct / max(total_tok, 1)
    return mean_loss, tok_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)
    cfg_snapshot_path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(cfg_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
    run_meta_path = os.path.join(out_dir, "run_meta.txt")
    with open(run_meta_path, "w", encoding="utf-8") as f:
        f.write("command: " + " ".join(sys.argv) + "\n")
        f.write(f"seed: {cfg['seed']}\n")
        f.write(f"init_checkpoint: {cfg['train'].get('init_checkpoint', '')}\n")
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    train_set, val_set = build_datasets(cfg)
    pad_id = train_set.tokenizer.pad_id
    eos_id = train_set.tokenizer.eos_id

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )
    val_workers = 0 if num_workers <= 0 else max(1, num_workers // 2)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )

    model = build_model_from_cfg(cfg, vocab_size=train_set.tokenizer.vocab_size, pad_id=pad_id)
    maybe_load_init_checkpoint(model, str(cfg["train"].get("init_checkpoint", "")).strip())
    freeze_parameters_by_prefix(model, cfg["train"].get("freeze_module_prefixes", []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters left after freeze_module_prefixes.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")
    epochs = int(cfg["train"]["epochs"])
    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    best_val = 1e9
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    eos_loss_weight = float(cfg["train"].get("eos_loss_weight", 1.0))
    mm_cfg = cfg["train"].get("modality_mask", {})
    mm_enable = bool(mm_cfg.get("enable", False))
    bev_drop_prob = float(mm_cfg.get("bev_drop_prob", 0.0))
    pv_drop_prob = float(mm_cfg.get("pv_drop_prob", 0.0))

    print(f"[Info] Train samples={len(train_set)} Val samples={len(val_set)} Vocab={train_set.tokenizer.vocab_size}")
    print(f"[Info] Device={device}")
    if mm_enable:
        print(
            "[Info] Modality mask enabled "
            f"(bev_drop_prob={bev_drop_prob:.2f}, pv_drop_prob={pv_drop_prob:.2f})"
        )

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_count = 0
        ep_bev_drop = 0.0
        ep_pv_drop = 0.0
        ep_mask_steps = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for it, batch in enumerate(pbar, start=1):
            lr = cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                base_lr=float(cfg["train"]["lr"]),
                warmup_steps=int(cfg["train"]["warmup_steps"]),
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            image = batch["image"].to(device, non_blocking=True)
            pv_images = batch.get("pv_images")
            if pv_images is not None:
                pv_images = pv_images.to(device, non_blocking=True)
            prompt_types = batch.get("prompt_type")
            if prompt_types is not None:
                prompt_types = prompt_types.to(device, non_blocking=True)
            prompt_tokens = batch.get("prompt_tokens")
            if prompt_tokens is not None:
                prompt_tokens = prompt_tokens.to(device, non_blocking=True)
            if mm_enable and pv_images is not None:
                image, pv_images, bev_dr, pv_dr = apply_modality_mask(
                    image=image,
                    pv_images=pv_images,
                    bev_drop_prob=bev_drop_prob,
                    pv_drop_prob=pv_drop_prob,
                )
                ep_bev_drop += bev_dr
                ep_pv_drop += pv_dr
                ep_mask_steps += 1
            tokens = batch["tokens"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                logits = model(
                    image,
                    inp,
                    pv_images=pv_images,
                    prompt_types=prompt_types,
                    prompt_tokens=prompt_tokens,
                )
                per_tok = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt.reshape(-1),
                    ignore_index=pad_id,
                    reduction="none",
                    label_smoothing=label_smoothing,
                ).view_as(tgt)
                if float(eos_loss_weight) != 1.0:
                    per_tok = per_tok * torch.where(
                        tgt.eq(int(eos_id)),
                        torch.full_like(per_tok, float(eos_loss_weight)),
                        torch.ones_like(per_tok),
                    )
                valid = loss_mask & tgt.ne(pad_id)
                loss = (per_tok * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

            scaler.scale(loss).backward()
            if float(cfg["train"]["grad_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.item()) * image.shape[0]
            ep_count += image.shape[0]
            global_step += 1

            if it % int(cfg["train"]["log_every"]) == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        train_loss = ep_loss / max(ep_count, 1)
        val_loss, val_tok_acc = run_val(
            model,
            val_loader,
            device=device,
            pad_id=pad_id,
            eos_id=eos_id,
            label_smoothing=label_smoothing,
            eos_loss_weight=eos_loss_weight,
        )
        dt = time.time() - t0
        mean_bev_drop = (ep_bev_drop / ep_mask_steps) if (mm_enable and ep_mask_steps > 0) else 0.0
        mean_pv_drop = (ep_pv_drop / ep_mask_steps) if (mm_enable and ep_mask_steps > 0) else 0.0
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_tok_acc={val_tok_acc:.4f} time={dt:.1f}s"
        )
        if mm_enable and ep_mask_steps > 0:
            print(
                f"[Epoch {epoch}] modality_drop_ratio "
                f"bev={mean_bev_drop:.3f} pv={mean_pv_drop:.3f}"
            )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_tok_acc": float(val_tok_acc),
                        "epoch_seconds": float(dt),
                        "modality_drop_bev": float(mean_bev_drop),
                        "modality_drop_pv": float(mean_pv_drop),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        latest_path = os.path.join(out_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "vocab": train_set.tokenizer.itos,
            },
            latest_path,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "vocab": train_set.tokenizer.itos,
                },
                os.path.join(out_dir, "best.pt"),
            )


if __name__ == "__main__":
    main()
