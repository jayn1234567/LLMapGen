import argparse
import os
import shutil
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimapgen.qwen_map_pipeline import (
    build_qwen_map_components,
    build_qwen_map_dataset,
    compute_shift_metrics,
    maybe_load_model_checkpoint,
)
from unimapgen.utils import cosine_lr, ensure_dir, load_yaml, set_seed


def run_val(model, loader, device, desc: str = ""):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    total_tok = 0
    iterator = loader
    if desc:
        iterator = tqdm(loader, desc=desc, leave=False)
    with torch.inference_mode():
        for batch in iterator:
            image = batch["image"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            state_input_ids = batch["state_input_ids"].to(device)
            state_attention_mask = batch["state_attention_mask"].to(device)
            map_input_ids = batch["map_input_ids"].to(device)
            map_attention_mask = batch["map_attention_mask"].to(device)

            out = model(
                image=image,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                state_input_ids=state_input_ids,
                state_attention_mask=state_attention_mask,
                map_input_ids=map_input_ids,
                map_attention_mask=map_attention_mask,
            )
            total_loss += float(out["loss"].item()) * image.shape[0]
            total_count += image.shape[0]
            correct, total = compute_shift_metrics(out["logits"], out["labels"])
            total_correct += correct
            total_tok += total
    return total_loss / max(total_count, 1), float(total_correct) / float(max(total_tok, 1))


def atomic_torch_save(obj, path: str) -> None:
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_link_or_copy(src: str, dst: str) -> str:
    tmp_path = dst + ".tmp"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.link(src, tmp_path)
        os.replace(tmp_path, dst)
        return "hardlink"
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = str(cfg["output_dir"])
    ensure_dir(out_dir)
    print(f"[Init] Loaded config: {args.config}", flush=True)
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
    with open(os.path.join(out_dir, "run_meta.txt"), "w", encoding="utf-8") as f:
        f.write("command: " + " ".join(sys.argv) + "\n")
        f.write(f"seed: {cfg['seed']}\n")
        f.write(f"init_checkpoint: {cfg['train'].get('init_checkpoint', '')}\n")

    print("[Init] Building train dataset...", flush=True)
    train_set = build_qwen_map_dataset(
        cfg,
        split=str(cfg["data"]["train_split"]),
        max_samples=cfg["data"].get("max_train_samples"),
        train_augment=bool(cfg["data"].get("train_augment", False)),
    )
    print(f"[Init] Train dataset ready: {len(train_set)} samples", flush=True)
    print("[Init] Building val dataset...", flush=True)
    val_set = build_qwen_map_dataset(
        cfg,
        split=str(cfg["data"]["val_split"]),
        max_samples=cfg["data"].get("max_val_samples"),
        train_augment=False,
    )
    print(f"[Init] Val dataset ready: {len(val_set)} samples", flush=True)
    print("[Init] Building tokenizer/collator/model...", flush=True)
    qwen_map_tokenizer, collator, model = build_qwen_map_components(cfg, train_set=train_set)
    print("[Init] Model components ready", flush=True)
    maybe_load_model_checkpoint(model, str(cfg["train"].get("init_checkpoint", "")).strip())

    batch_size = int(cfg["train"]["batch_size"])
    val_batch_size = int(cfg["train"].get("val_batch_size", batch_size))
    num_workers = int(cfg["data"].get("num_workers", 0))
    val_num_workers = int(cfg["data"].get("val_num_workers", 0 if num_workers <= 0 else max(1, num_workers // 2)))
    persistent_workers = bool(cfg["data"].get("persistent_workers", True)) and num_workers > 0
    val_persistent_workers = bool(cfg["data"].get("persistent_workers", True)) and val_num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
        persistent_workers=val_persistent_workers,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters in Qwen map branch.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda")
    epochs = int(cfg["train"]["epochs"])
    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    best_val = 1e9
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    save_latest = bool(cfg["train"].get("save_latest", True))
    hardlink_best_to_latest = bool(cfg["train"].get("hardlink_best_to_latest", True))

    print(f"[Info] Train samples={len(train_set)} Val samples={len(val_set)}", flush=True)
    print(f"[Info] Qwen vocab after map extension={qwen_map_tokenizer.vocab_size}", flush=True)
    print(f"[Info] Device={device}", flush=True)
    print(
        f"[Info] Train batch_size={batch_size} Val batch_size={val_batch_size} "
        f"num_workers(train/val)={num_workers}/{val_num_workers} "
        f"save_latest={save_latest}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_count = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            lr = cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                base_lr=float(cfg["train"]["lr"]),
                warmup_steps=int(cfg["train"].get("warmup_steps", 0)),
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            image = batch["image"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            state_input_ids = batch["state_input_ids"].to(device)
            state_attention_mask = batch["state_attention_mask"].to(device)
            map_input_ids = batch["map_input_ids"].to(device)
            map_attention_mask = batch["map_attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda"):
                out = model(
                    image=image,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    state_input_ids=state_input_ids,
                    state_attention_mask=state_attention_mask,
                    map_input_ids=map_input_ids,
                    map_attention_mask=map_attention_mask,
                )
                loss = out["loss"]
            scaler.scale(loss).backward()
            if float(cfg["train"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.item()) * image.shape[0]
            ep_count += image.shape[0]
            global_step += 1
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{lr:.2e}")

        train_loss = ep_loss / max(ep_count, 1)
        train_elapsed = time.time() - t0
        print(f"[Epoch {epoch}] Train loop finished in {train_elapsed:.1f}s. Running validation...", flush=True)
        val_t0 = time.time()
        val_loss, val_tok_acc = run_val(model, val_loader, device, desc=f"Val {epoch}/{epochs}")
        val_elapsed = time.time() - val_t0
        print(f"[Epoch {epoch}] Validation finished in {val_elapsed:.1f}s. Saving checkpoints...", flush=True)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_token_acc": val_tok_acc,
            "train_sec": train_elapsed,
            "val_sec": val_elapsed,
        }
        latest_path = os.path.join(out_dir, "latest.pt")
        best_path = os.path.join(out_dir, "best.pt")
        ckpt_obj = {"model": model.state_dict(), "epoch": epoch, "cfg": cfg}
        save_t0 = time.time()
        latest_saved = False
        if save_latest:
            atomic_torch_save(ckpt_obj, latest_path)
            latest_saved = True
        if val_loss < best_val:
            best_val = val_loss
            if latest_saved and hardlink_best_to_latest:
                best_save_mode = atomic_link_or_copy(latest_path, best_path)
            else:
                atomic_torch_save(ckpt_obj, best_path)
                best_save_mode = "save"
            record["best_updated"] = True
            record["best_save_mode"] = best_save_mode
        else:
            record["best_updated"] = False
        save_elapsed = time.time() - save_t0
        record["checkpoint_sec"] = save_elapsed
        record["elapsed_sec"] = train_elapsed + val_elapsed + save_elapsed
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json_dumps(record) + "\n")
        print(record, flush=True)


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()
