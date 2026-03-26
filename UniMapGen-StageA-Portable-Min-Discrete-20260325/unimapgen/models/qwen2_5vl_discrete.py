from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from unimapgen.discrete_map_token_format import DiscreteMapTokenFormatter


def initialize_new_token_embeddings(model: torch.nn.Module, old_vocab_size: int) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return
    new_vocab_size = int(input_embeddings.weight.shape[0])
    if new_vocab_size <= int(old_vocab_size):
        return
    with torch.no_grad():
        avg_input = input_embeddings.weight[: int(old_vocab_size)].mean(dim=0, keepdim=True)
        input_embeddings.weight[int(old_vocab_size) : new_vocab_size] = avg_input
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.shape[0] == new_vocab_size:
            avg_output = output_embeddings.weight[: int(old_vocab_size)].mean(dim=0, keepdim=True)
            output_embeddings.weight[int(old_vocab_size) : new_vocab_size] = avg_output


def load_processor(model_name_or_path: str, formatter: DiscreteMapTokenFormatter) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    formatter.register_tokens_with_processor(processor)
    return processor


def save_runtime_assets(processor: AutoProcessor, processor_output_dir: Path, formatter: DiscreteMapTokenFormatter) -> None:
    processor_output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(processor_output_dir)
    runtime_info = {
        "image_size": int(formatter.image_size),
        "coord_num_bins": int(formatter.coord_num_bins),
        "token_schema": str(formatter.coordinate_token_style),
        "categories": list(formatter.categories),
        "include_text_prompt_tokens": bool(formatter.include_text_prompt_tokens),
    }
    (processor_output_dir / "discrete_token_runtime.json").write_text(
        json.dumps(runtime_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_training_model(args: Any, processor: AutoProcessor) -> torch.nn.Module:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    old_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer.")
    new_vocab_size = int(len(tokenizer))
    if new_vocab_size != old_vocab_size:
        model.resize_token_embeddings(new_vocab_size)
        initialize_new_token_embeddings(model, old_vocab_size=old_vocab_size)
    final_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    model.config.vocab_size = final_vocab_size
    if hasattr(model, "vocab_size"):
        model.vocab_size = final_vocab_size
    if not bool(args.no_lora):
        peft_cfg = LoraConfig(
            r=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, peft_cfg)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def load_inference_model(model_or_checkpoint: str, device: str) -> torch.nn.Module:
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_or_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto" if str(device).startswith("cuda") else None,
        trust_remote_code=True,
    )
    if not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model
