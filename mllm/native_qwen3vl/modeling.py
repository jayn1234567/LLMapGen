from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch
from transformers import AutoProcessor

from .data import build_qwen3vl_messages, normalize_image_paths


def resolve_native_model_class():
    import transformers

    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        "Cannot find a native Qwen3-VL model class in transformers. "
        "Install a Qwen3-VL-capable transformers build."
    )


def load_processor(model_path: str | Path, trust_remote_code: bool = True):
    return AutoProcessor.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)


def _from_pretrained_kwargs(model_args: Any, dtype):
    kwargs = {
        "trust_remote_code": getattr(model_args, "trust_remote_code", True),
    }
    attn = getattr(model_args, "attn_implementation", None)
    if attn:
        kwargs["attn_implementation"] = attn
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def load_native_model(model_args: Any, training_args: Any | None = None, device_map: str | dict | None = None):
    model_cls = resolve_native_model_class()
    model_path = Path(str(model_args.model_name_or_path))
    adapter_config_path = model_path / "adapter_config.json"
    adapter_path = model_path if adapter_config_path.is_file() else None
    base_model_path = str(model_path)
    if adapter_path is not None:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_path = str(
            getattr(model_args, "model_base", None)
            or adapter_config.get("base_model_name_or_path")
            or ""
        )
        if not base_model_path:
            raise ValueError(
                f"Native Qwen3-VL adapter requires model_base: {adapter_config_path}"
            )
    dtype = None
    if training_args is not None:
        if getattr(training_args, "bf16", False):
            dtype = torch.bfloat16
        elif getattr(training_args, "fp16", False):
            dtype = torch.float16
    elif torch.cuda.is_available():
        dtype = torch.float16

    kwargs = _from_pretrained_kwargs(model_args, dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        model = model_cls.from_pretrained(base_model_path, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(base_model_path, **kwargs)

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading a native Qwen3-VL LoRA checkpoint requires peft") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    return model


def select_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def processor_text_inputs(
    processor: Any,
    prompt: str,
    image_paths: str | Path | Sequence[str | Path],
    system_prompt: str | None = None,
):
    normalized_paths = normalize_image_paths(image_paths)
    messages = build_qwen3vl_messages(prompt, normalized_paths, None, system_prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images = [Image.open(image_path).convert("RGB") for image_path in normalized_paths]
    return processor(text=[text], images=images, return_tensors="pt"), text


def decode_generated_tokens(processor: Any, output_ids: torch.Tensor, input_len: int) -> tuple[str, int]:
    completion_ids = output_ids[:, input_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(completion_ids, skip_special_tokens=False)[0]
    else:
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)[0]
    return decoded.strip(), int(completion_ids.numel())


def generate_one(
    model: Any,
    processor: Any,
    image_path: str | Path | Sequence[str | Path],
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    inputs, rendered_prompt = processor_text_inputs(
        processor, prompt, image_path, system_prompt=system_prompt
    )
    input_ids = inputs["input_ids"]
    input_token_len = int(input_ids.shape[1])
    inputs = move_inputs_to_device(inputs, device)

    generation_config = getattr(model, "generation_config", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(generation_config, "pad_token_id", None) or getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None) or getattr(tokenizer, "eos_token_id", None)

    kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": temperature > 0,
        "num_beams": 1,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if temperature > 0:
        kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
    raw_prediction, decoded_len = decode_generated_tokens(processor, output_ids, input_token_len)
    return {
        "rendered_prompt": rendered_prompt,
        "raw_prediction": raw_prediction,
        "input_token_len": input_token_len,
        "output_token_len": int(output_ids.shape[1]),
        "decoded_token_len": decoded_len,
    }


def trainer_processor_kwarg(processor: Any) -> dict[str, Any]:
    import transformers

    signature = inspect.signature(transformers.Trainer.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": processor}
    return {"tokenizer": getattr(processor, "tokenizer", processor)}
