from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from mllm.torch_runtime import maybe_disable_cudnn_from_env
from mllm.model.builder import load_pretrained_model
from mllm.train.checkpoint_metadata import sync_qwen_multimodal_config, write_qwen_multimodal_checkpoint_metadata
from mllm.train.train_qwen import print_trainable_parameters, resolve_lora_target_modules

maybe_disable_cudnn_from_env(torch)


def is_peft_checkpoint(path: str | Path) -> bool:
    return Path(path, "adapter_config.json").is_file()


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def get_base_policy_model(model):
    model = unwrap_model(model)
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_zero_or_cpu(param, name: str):
    if hasattr(param, "ds_id"):
        try:
            from deepspeed import zero
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        except Exception as exc:  # pragma: no cover - only relevant for ZeRO checkpoints
            raise ImportError(f"Saving ZeRO-partitioned parameter {name} requires deepspeed") from exc
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            pass
        with zero.GatheredParameters([param]):
            return param.data.detach().cpu().clone()
    return param.detach().cpu().clone()


def _peft_lora_state(named_params, bias: str):
    if bias == "none":
        selected = {name: param for name, param in named_params if "lora_" in name}
    elif bias == "all":
        selected = {name: param for name, param in named_params if "lora_" in name or "bias" in name}
    elif bias == "lora_only":
        selected = {}
        maybe_bias = {}
        lora_bias_names = set()
        for name, param in named_params:
            if "lora_" in name:
                selected[name] = param
                lora_bias_names.add(name.split("lora_")[0] + "bias")
            elif "bias" in name:
                maybe_bias[name] = param
        for name, param in maybe_bias.items():
            if name in lora_bias_names:
                selected[name] = param
    else:
        raise ValueError(f"Unsupported LoRA bias setting: {bias}")
    return {name: _maybe_zero_or_cpu(param, name) for name, param in selected.items()}


def _non_lora_trainables_state(named_params):
    selected = {
        name: param
        for name, param in named_params
        if "lora_" not in name and param.requires_grad
    }
    return {name: _maybe_zero_or_cpu(param, name) for name, param in selected.items()}


def _load_non_lora_trainables(model, checkpoint_path: str | Path) -> int:
    path = Path(checkpoint_path) / "non_lora_trainables.bin"
    if not path.is_file():
        return 0
    state = torch.load(path, map_location="cpu")
    normalized = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("base_model.model.", "base_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    if not normalized:
        return 0
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if unexpected:
        print(f"[mllm-rl] Unexpected non-LoRA trainable keys: {unexpected[:20]}")
    if missing:
        print(f"[mllm-rl] Missing keys after non-LoRA load: {len(missing)}")
    return len(normalized)


def _model_config_overrides(model_args, training_args) -> dict[str, Any]:
    vision_tower = model_args.multi_vision_towers or model_args.vision_tower
    overrides = {
        "mm_vision_tower": vision_tower,
        "vision_tower": vision_tower,
        "mm_vision_tower_type": model_args.mm_vision_tower_type,
        "input_image_size": model_args.input_image_size,
        "deepstack_visual_indexes": None if model_args.disable_deepstack else model_args.deepstack_visual_indexes,
        "disable_deepstack": model_args.disable_deepstack,
        "mm_projector_type": model_args.mm_projector_type,
        "mm_vision_select_layer": model_args.mm_vision_select_layer,
        "mm_vision_select_feature": model_args.mm_vision_select_feature,
        "mm_patch_merge_type": model_args.mm_patch_merge_type,
        "multi_vision_towers": model_args.multi_vision_towers,
        "multi_vision_tower_types": model_args.multi_vision_tower_types,
        "multi_vision_input_image_sizes": model_args.multi_vision_input_image_sizes,
        "multi_vision_primary_index": model_args.multi_vision_primary_index,
        "multi_vision_hidden_size": model_args.multi_vision_hidden_size,
        "multi_vision_target_grid": model_args.multi_vision_target_grid,
        "multi_vision_fusion": model_args.multi_vision_fusion,
        "multi_vision_router_temperature": model_args.multi_vision_router_temperature,
        "multi_vision_router_hidden_ratio": model_args.multi_vision_router_hidden_ratio,
        "multi_vision_router_use_diff": model_args.multi_vision_router_use_diff,
        "multi_vision_dropout": model_args.multi_vision_dropout,
        "tokenizer_model_max_length": training_args.model_max_length,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _load_policy_from_full_checkpoint(model_args, training_args, device: str):
    dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_args.model_name_or_path,
        model_base=None,
        model_name=os.path.basename(str(model_args.model_name_or_path)),
        device_map=None,
        device=device,
        tokenizer_use_fast=model_args.tokenizer_use_fast,
        model_config_overrides=_model_config_overrides(model_args, training_args),
        torch_dtype=dtype,
    )
    return tokenizer, model, image_processor


def _load_policy_from_peft_checkpoint(model_args, training_args, device: str):
    from peft import PeftModel

    adapter_config = _load_json(Path(model_args.model_name_or_path) / "adapter_config.json")
    base_path = model_args.model_base or adapter_config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError("PEFT policy checkpoint requires --model_base or adapter_config.base_model_name_or_path")

    dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base_path,
        model_base=None,
        model_name=os.path.basename(str(base_path)),
        device_map=None,
        device=device,
        tokenizer_use_fast=model_args.tokenizer_use_fast,
        model_config_overrides=_model_config_overrides(model_args, training_args),
        torch_dtype=dtype,
    )
    loaded = _load_non_lora_trainables(model, model_args.model_name_or_path)
    if loaded:
        print(f"[mllm-rl] Loaded {loaded} non-LoRA trainable tensors from {model_args.model_name_or_path}")
    model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=True)
    return tokenizer, model, image_processor


def load_policy_model(model_args, training_args, device: str):
    if model_args.disable_deepstack:
        model_args.deepstack_visual_indexes = None
    if is_peft_checkpoint(model_args.model_name_or_path):
        tokenizer, model, image_processor = _load_policy_from_peft_checkpoint(model_args, training_args, device)
    else:
        tokenizer, model, image_processor = _load_policy_from_full_checkpoint(model_args, training_args, device)

    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = training_args.model_max_length
    return tokenizer, model, image_processor


def apply_trainable_policy(model, training_args):
    from peft import LoraConfig, PeftModel, get_peft_model

    if isinstance(unwrap_model(model), PeftModel):
        return model

    if training_args.lora_enable:
        targets = resolve_lora_target_modules(
            model,
            target_scope=training_args.lora_target_scope,
            target_modules=training_args.lora_target_modules,
            exclude_modules=training_args.lora_exclude_modules,
        )
        if not targets:
            raise ValueError(
                "No LoRA target modules were resolved for GRPO. "
                f"scope={training_args.lora_target_scope!r}, manual={training_args.lora_target_modules!r}"
            )
        config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=targets,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        print(f"[mllm-rl] Adding GRPO LoRA adapters: scope={training_args.lora_target_scope}, targets={len(targets)}")
        model = get_peft_model(model, config)
        return model

    scopes = {item.strip() for item in str(training_args.full_train_scope or "all").split(",") if item.strip()}
    if "all" in scopes:
        model.requires_grad_(True)
        return model

    model.requires_grad_(False)
    for name, param in model.named_parameters():
        is_projector = "mm_projector" in name
        is_vision = "vision_tower" in name
        is_deepstack = "deepstack" in name or "deepstack_mergers" in name
        is_llm = not (is_projector or is_vision or is_deepstack)
        if (
            ("llm" in scopes and is_llm)
            or ("projector" in scopes and is_projector)
            or ("vision" in scopes and is_vision and not is_deepstack)
            or ("deepstack" in scopes and is_deepstack)
        ):
            param.requires_grad = True
    return model


def create_optimizer(model, training_args):
    decay = []
    no_decay = []
    special_projector_decay = []
    special_projector_no_decay = []
    special_fusion_decay = []
    special_fusion_no_decay = []
    special_vision_decay = []
    special_vision_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_decay = param.ndim >= 2 and "bias" not in name and "norm" not in name.lower()
        is_projector = "mm_projector" in name
        is_vision = "vision_tower" in name
        is_vision_fusion = any(
            keyword in name
            for keyword in (
                "vision_tower.expert_adapters",
                "vision_tower.router",
                "vision_tower.post_fusion",
                "vision_tower.out_norm",
            )
        )
        if is_projector and training_args.mm_projector_lr is not None:
            (special_projector_decay if is_decay else special_projector_no_decay).append(param)
        elif is_vision_fusion and training_args.mm_vision_fusion_lr is not None:
            (special_fusion_decay if is_decay else special_fusion_no_decay).append(param)
        elif is_vision and training_args.mm_vision_tower_lr is not None:
            (special_vision_decay if is_decay else special_vision_no_decay).append(param)
        else:
            (decay if is_decay else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": training_args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if training_args.mm_projector_lr is not None:
        groups.extend([
            {"params": special_projector_decay, "weight_decay": training_args.weight_decay, "lr": training_args.mm_projector_lr},
            {"params": special_projector_no_decay, "weight_decay": 0.0, "lr": training_args.mm_projector_lr},
        ])
    if training_args.mm_vision_fusion_lr is not None:
        groups.extend([
            {"params": special_fusion_decay, "weight_decay": training_args.weight_decay, "lr": training_args.mm_vision_fusion_lr},
            {"params": special_fusion_no_decay, "weight_decay": 0.0, "lr": training_args.mm_vision_fusion_lr},
        ])
    if training_args.mm_vision_tower_lr is not None:
        groups.extend([
            {"params": special_vision_decay, "weight_decay": training_args.weight_decay, "lr": training_args.mm_vision_tower_lr},
            {"params": special_vision_no_decay, "weight_decay": 0.0, "lr": training_args.mm_vision_tower_lr},
        ])
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(
        groups,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )


def save_policy_checkpoint(model, tokenizer, output_dir: str | Path, training_args, optimizer=None, scheduler=None, state: dict[str, Any] | None = None):
    model_to_save = unwrap_model(model)
    base_model = get_base_policy_model(model_to_save)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sync_qwen_multimodal_config(base_model)
    except Exception:
        sync_qwen_multimodal_config(model_to_save)

    if hasattr(model_to_save, "peft_config"):
        lora_state = _peft_lora_state(model_to_save.named_parameters(), training_args.lora_bias)
        non_lora_state = _non_lora_trainables_state(model_to_save.named_parameters())
        model_to_save.save_pretrained(output_dir, state_dict=lora_state)
        base_model.config.save_pretrained(output_dir)
        torch.save(non_lora_state, output_dir / "non_lora_trainables.bin")
        write_qwen_multimodal_checkpoint_metadata(base_model, output_dir)
    else:
        model_to_save.save_pretrained(output_dir)
        write_qwen_multimodal_checkpoint_metadata(model_to_save, output_dir)

    tokenizer.save_pretrained(output_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), output_dir / "scheduler.pt")
    if state is not None:
        (output_dir / "rl_trainer_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def print_policy_parameters(model):
    print_trainable_parameters(unwrap_model(model))
