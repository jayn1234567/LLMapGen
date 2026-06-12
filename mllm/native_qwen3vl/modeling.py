from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import nn
from transformers import AutoProcessor

from .data import build_qwen3vl_messages


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


class LinearizedConv3dProjection(nn.Module):
    """Run Qwen3-VL patch embedding with matmul while keeping Conv3d-shaped weights."""

    def __init__(self, conv3d: nn.Conv3d):
        super().__init__()
        if conv3d.groups != 1 or conv3d.padding != (0, 0, 0) or conv3d.dilation != (1, 1, 1):
            raise ValueError("Only Qwen3-VL patch-embed Conv3d with groups=1, padding=0, dilation=1 is supported.")
        if conv3d.kernel_size != conv3d.stride:
            raise ValueError("Only Qwen3-VL patch-embed Conv3d with kernel_size == stride is supported.")

        self.in_channels = conv3d.in_channels
        self.out_channels = conv3d.out_channels
        self.kernel_size = conv3d.kernel_size
        self.stride = conv3d.stride
        self.padding = conv3d.padding
        self.dilation = conv3d.dilation
        self.groups = conv3d.groups
        self.weight = nn.Parameter(conv3d.weight.detach().clone())
        self.bias = nn.Parameter(conv3d.bias.detach().clone()) if conv3d.bias is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(batch, -1)
        weight = self.weight.reshape(self.out_channels, -1)
        hidden_states = torch.nn.functional.linear(hidden_states, weight, self.bias)
        return hidden_states.reshape(batch, self.out_channels, 1, 1, 1)


def replace_qwen3vl_patch_embed_conv3d_with_linear(model: nn.Module) -> int:
    replaced = 0
    for module in model.modules():
        patch_embed = getattr(module, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        if isinstance(proj, nn.Conv3d):
            patch_embed.proj = LinearizedConv3dProjection(proj)
            replaced += 1
    if replaced == 0:
        raise RuntimeError("No Qwen3-VL patch_embed.proj Conv3d module was found to replace.")
    return replaced


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
        model = model_cls.from_pretrained(str(model_args.model_name_or_path), **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(str(model_args.model_name_or_path), **kwargs)

    if getattr(model_args, "replace_patch_embed_conv3d_with_linear", False):
        replaced = replace_qwen3vl_patch_embed_conv3d_with_linear(model)
        print(f"Replaced {replaced} Qwen3-VL patch_embed Conv3d module(s) with linearized projection.")
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


def processor_text_inputs(processor: Any, prompt: str, image_path: str | Path):
    messages = build_qwen3vl_messages(prompt, image_path, None)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = Image.open(image_path).convert("RGB")
    return processor(text=[text], images=[image], return_tensors="pt"), text


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
    image_path: str | Path,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict[str, Any]:
    inputs, rendered_prompt = processor_text_inputs(processor, prompt, image_path)
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
