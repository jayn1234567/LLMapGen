from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn


def _tensor_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if hasattr(payload, "state_dict") and not isinstance(payload, dict):
        try:
            state = payload.state_dict()
        except Exception:
            state = None
        if isinstance(state, dict):
            return {str(key): value for key, value in state.items() if torch.is_tensor(value)}

    if not isinstance(payload, dict):
        return {}

    direct = {str(key): value for key, value in payload.items() if torch.is_tensor(value)}
    if direct:
        return direct

    for key in ("vision_encoder", "encoder", "backbone", "model", "state_dict", "module", "net"):
        nested_state = _tensor_state_dict(payload.get(key))
        if nested_state:
            return nested_state
    return {}


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "_orig_mod.",
        "base_model.model.",
        "base_model.",
        "vision_encoder.",
        "encoder.",
        "backbone.",
        "net.",
    )
    state = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if state and all(str(key).startswith(prefix) for key in state):
                state = {str(key)[len(prefix) :]: value for key, value in state.items()}
                changed = True
                break
    return state


def _suffix_aligned_state(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    target_state = module.state_dict()
    source_by_suffix: Dict[str, Tuple[str, torch.Tensor]] = {}
    for source_key, value in state_dict.items():
        parts = str(source_key).split(".")
        for start in range(len(parts)):
            suffix = ".".join(parts[start:])
            target = target_state.get(suffix)
            if target is None or tuple(target.shape) != tuple(value.shape):
                continue
            existing = source_by_suffix.get(suffix)
            if existing is None or len(str(source_key)) < len(existing[0]):
                source_by_suffix[suffix] = (str(source_key), value)
    return {target_key: value for target_key, (_, value) in source_by_suffix.items()}


def _find_target_key(
    target_state: Dict[str, torch.Tensor],
    value: torch.Tensor,
    suffixes: Sequence[str],
) -> str | None:
    shape = tuple(value.shape)
    for suffix in suffixes:
        matches = [
            key
            for key, target_value in target_state.items()
            if str(key).endswith(str(suffix))
            and (tuple(target_value.shape) == shape or int(target_value.numel()) == 0)
        ]
        if matches:
            exact = [
                key
                for key in matches
                if tuple(target_state[key].shape) == shape
            ]
            return sorted(exact or matches, key=len)[0]
    return None


def _candidate_tensor_values(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    values = [value]
    if value.ndim >= 1 and int(value.shape[0]) == 1:
        values.append(value.squeeze(0))
    if value.ndim == 2:
        values.append(value.t().contiguous())
    if value.ndim == 4:
        values.append(value.flatten(1))

    deduped = []
    seen = set()
    for candidate in values:
        signature = tuple(candidate.shape)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return tuple(deduped)


def _format_state_examples(state_dict: Dict[str, torch.Tensor], limit: int = 8) -> str:
    examples = []
    for key, value in list(state_dict.items())[:limit]:
        if torch.is_tensor(value):
            examples.append(f"{key}:{tuple(value.shape)}")
        else:
            examples.append(f"{key}:{type(value).__name__}")
    return ", ".join(examples)


def _assign_by_suffix(
    mapped: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    value: torch.Tensor | None,
    suffixes: Sequence[str],
) -> bool:
    if value is None or not torch.is_tensor(value):
        return False
    for candidate in _candidate_tensor_values(value):
        target_key = _find_target_key(target_state, candidate, suffixes)
        if target_key is None:
            continue
        mapped[target_key] = candidate
        return True
    return False


def _scripted_encoder_view(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(str(key).startswith("encoder.blocks.") for key in raw_state):
        return raw_state

    stripped = _strip_common_prefixes(raw_state)
    if any(str(key).startswith("encoder.blocks.") for key in stripped):
        return stripped

    encoder_state: Dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        key_str = str(key)
        marker = ".encoder."
        marker_pos = key_str.find(marker)
        if marker_pos < 0:
            continue
        encoder_state[key_str[marker_pos + 1 :]] = value
    if any(str(key).startswith("encoder.blocks.") for key in encoder_state):
        return encoder_state

    return raw_state


def _scripted_dinov3_to_hf_state(
    vision_encoder: nn.Module,
    raw_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    raw_state = _scripted_encoder_view(raw_state)
    if not any(str(key).startswith("encoder.blocks.") for key in raw_state):
        return {}

    target_state = vision_encoder.state_dict()
    mapped: Dict[str, torch.Tensor] = {}

    _assign_by_suffix(mapped, target_state, raw_state.get("encoder.cls_token"), ("embeddings.cls_token", "cls_token"))
    _assign_by_suffix(
        mapped,
        target_state,
        raw_state.get("encoder.storage_tokens"),
        ("embeddings.register_tokens", "register_tokens", "storage_tokens"),
    )
    _assign_by_suffix(mapped, target_state, raw_state.get("encoder.mask_token"), ("embeddings.mask_token", "mask_token"))
    _assign_by_suffix(
        mapped,
        target_state,
        raw_state.get("encoder.patch_embed.proj.weight"),
        (
            "embeddings.patch_embeddings.projection.weight",
            "embeddings.patch_embeddings.weight",
            "patch_embeddings.projection.weight",
            "patch_embeddings.weight",
            "patch_embed.proj.weight",
        ),
    )
    _assign_by_suffix(
        mapped,
        target_state,
        raw_state.get("encoder.patch_embed.proj.bias"),
        (
            "embeddings.patch_embeddings.projection.bias",
            "embeddings.patch_embeddings.bias",
            "patch_embeddings.projection.bias",
            "patch_embeddings.bias",
            "patch_embed.proj.bias",
        ),
    )

    block_indexes = sorted(
        {
            int(str(key).split(".")[2])
            for key in raw_state
            if str(key).startswith("encoder.blocks.")
            and len(str(key).split(".")) > 3
            and str(key).split(".")[2].isdigit()
        }
    )
    for idx in block_indexes:
        src = f"encoder.blocks.{idx}"
        target_layers = (
            f"layer.{idx}",
            f"model.layer.{idx}",
            f"model.layers.{idx}",
            f"model.blocks.{idx}",
            f"encoder.layer.{idx}",
            f"encoder.layers.{idx}",
            f"encoder.blocks.{idx}",
            f"blocks.{idx}",
            f"layers.{idx}",
        )

        for norm_name in ("norm1", "norm2"):
            for attr in ("weight", "bias"):
                _assign_by_suffix(
                    mapped,
                    target_state,
                    raw_state.get(f"{src}.{norm_name}.{attr}"),
                    tuple(f"{layer}.{norm_name}.{attr}" for layer in target_layers),
                )

        for mlp_name in ("fc1", "fc2"):
            for attr in ("weight", "bias"):
                if mlp_name == "fc1":
                    projector_names = ("up_proj", "gate_proj")
                else:
                    projector_names = ("down_proj",)
                _assign_by_suffix(
                    mapped,
                    target_state,
                    raw_state.get(f"{src}.mlp.{mlp_name}.{attr}"),
                    tuple(f"{layer}.mlp.{mlp_name}.{attr}" for layer in target_layers)
                    + tuple(
                        f"{layer}.mlp.{target_name}.{attr}"
                        for layer in target_layers
                        for target_name in projector_names
                    ),
                )

        for source_name, target_names in (
            ("ls1.gamma", ("layer_scale1.lambda1", "layer_scale1.gamma", "ls1.gamma")),
            ("ls2.gamma", ("layer_scale2.lambda1", "layer_scale2.gamma", "ls2.gamma")),
        ):
            _assign_by_suffix(
                mapped,
                target_state,
                raw_state.get(f"{src}.{source_name}"),
                tuple(f"{layer}.{target}" for layer in target_layers for target in target_names),
            )

        for attr in ("weight", "bias"):
            _assign_by_suffix(
                mapped,
                target_state,
                raw_state.get(f"{src}.attn.proj.{attr}"),
                tuple(
                    f"{layer}.{target}.{attr}"
                    for layer in target_layers
                    for target in (
                        "attention.output.dense",
                        "attention.output.projection",
                        "attention.o_proj",
                        "self_attn.o_proj",
                        "attn.o_proj",
                        "attn.proj",
                    )
                ),
            )

        qkv_weight = raw_state.get(f"{src}.attn.qkv.qkv.weight")
        qkv_bias = raw_state.get(f"{src}.attn.qkv.qkv.bias")
        if torch.is_tensor(qkv_weight) and qkv_weight.ndim == 2 and qkv_weight.shape[0] % 3 == 0:
            q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
            q_delta_a = raw_state.get(f"{src}.attn.qkv.linear_a_q.weight")
            q_delta_b = raw_state.get(f"{src}.attn.qkv.linear_b_q.weight")
            v_delta_a = raw_state.get(f"{src}.attn.qkv.linear_a_v.weight")
            v_delta_b = raw_state.get(f"{src}.attn.qkv.linear_b_v.weight")
            if torch.is_tensor(q_delta_a) and torch.is_tensor(q_delta_b):
                q_delta = torch.matmul(q_delta_b.to(dtype=q_weight.dtype), q_delta_a.to(dtype=q_weight.dtype))
                if tuple(q_delta.shape) == tuple(q_weight.shape):
                    q_weight = q_weight + q_delta
            if torch.is_tensor(v_delta_a) and torch.is_tensor(v_delta_b):
                v_delta = torch.matmul(v_delta_b.to(dtype=v_weight.dtype), v_delta_a.to(dtype=v_weight.dtype))
                if tuple(v_delta.shape) == tuple(v_weight.shape):
                    v_weight = v_weight + v_delta
            for name, value in (("query", q_weight), ("key", k_weight), ("value", v_weight)):
                proj_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[name]
                _assign_by_suffix(
                    mapped,
                    target_state,
                    value,
                    tuple(
                        f"{layer}.{target}.{name}.weight"
                        for layer in target_layers
                        for target in ("attention.attention", "attention.self", "attention")
                    )
                    + tuple(
                        f"{layer}.{target}.{proj_name}.weight"
                        for layer in target_layers
                        for target in ("attention", "self_attn", "attn")
                    )
                    + tuple(f"{layer}.attn.{name}.weight" for layer in target_layers),
                )

        if torch.is_tensor(qkv_bias) and qkv_bias.ndim == 1 and qkv_bias.shape[0] % 3 == 0:
            q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
            for name, value in (("query", q_bias), ("key", k_bias), ("value", v_bias)):
                proj_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[name]
                _assign_by_suffix(
                    mapped,
                    target_state,
                    value,
                    tuple(
                        f"{layer}.{target}.{name}.bias"
                        for layer in target_layers
                        for target in ("attention.attention", "attention.self", "attention")
                    )
                    + tuple(
                        f"{layer}.{target}.{proj_name}.bias"
                        for layer in target_layers
                        for target in ("attention", "self_attn", "attn")
                    )
                    + tuple(f"{layer}.attn.{name}.bias" for layer in target_layers),
                )

    for source_key, suffixes in (
        ("encoder.norm.weight", ("layernorm.weight", "norm.weight")),
        ("encoder.norm.bias", ("layernorm.bias", "norm.bias")),
    ):
        _assign_by_suffix(mapped, target_state, raw_state.get(source_key), suffixes)
    return mapped


def _candidate_encoder_states(
    vision_encoder: nn.Module,
    raw_state: Dict[str, torch.Tensor],
) -> list[tuple[str, Dict[str, torch.Tensor]]]:
    stripped = _strip_common_prefixes(raw_state)
    candidates: list[tuple[str, Dict[str, torch.Tensor]]] = [("raw", raw_state), ("stripped", stripped)]
    if stripped and not all(str(key).startswith("model.") for key in stripped):
        candidates.append(("stripped_plus_model", {f"model.{key}": value for key, value in stripped.items()}))
    if raw_state and not all(str(key).startswith("model.") for key in raw_state):
        candidates.append(("raw_plus_model", {f"model.{key}": value for key, value in raw_state.items()}))
    suffix_aligned = _suffix_aligned_state(vision_encoder, raw_state)
    if suffix_aligned:
        candidates.append(("suffix_aligned", suffix_aligned))
    scripted_dinov3 = _scripted_dinov3_to_hf_state(vision_encoder, raw_state)
    if scripted_dinov3:
        candidates.append(("scripted_dinov3_hf", scripted_dinov3))

    deduped: list[tuple[str, Dict[str, torch.Tensor]]] = []
    seen = set()
    for name, state in candidates:
        signature = tuple(sorted(state.keys()))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append((name, state))
    return deduped


def _shape_match_score(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> tuple[int, int]:
    target_state = module.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in target_state
        and (
            tuple(target_state[key].shape) == tuple(value.shape)
            or int(target_state[key].numel()) == 0
        )
    }
    return len(matched), sum(int(value.numel()) for value in matched.values())


def _resolve_tensor_parent(module: nn.Module, key: str) -> tuple[Any, str]:
    obj: Any = module
    parts = str(key).split(".")
    for part in parts[:-1]:
        if part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential, list, tuple)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj, parts[-1]


def _assign_tensor_to_module(module: nn.Module, key: str, value: torch.Tensor) -> bool:
    parent, name = _resolve_tensor_parent(module, key)
    tensor = value.detach().clone()
    parameters = getattr(parent, "_parameters", {})
    if name in parameters:
        old = parameters.get(name)
        requires_grad = True if old is None else bool(old.requires_grad)
        parameters[name] = nn.Parameter(tensor, requires_grad=requires_grad)
        return True
    buffers = getattr(parent, "_buffers", {})
    if name in buffers:
        buffers[name] = tensor
        return True
    current = getattr(parent, name, None)
    if torch.is_tensor(current):
        setattr(parent, name, tensor)
        return True
    return False


def _load_state_dict_allow_zero_shape(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> tuple[list[str], list[str], int]:
    target_state = module.state_dict()
    zero_shape_targets = {
        key
        for key, value in target_state.items()
        if int(value.numel()) == 0 and key in state_dict
    }
    if not zero_shape_targets:
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        return list(missing), list(unexpected), 0

    assigned = 0
    unexpected: list[str] = []
    for key, value in state_dict.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(target_state[key].shape) == tuple(value.shape):
            continue
        if int(target_state[key].numel()) != 0:
            unexpected.append(key)
            continue
        if _assign_tensor_to_module(module, key, value):
            assigned += 1
        else:
            unexpected.append(key)

    exact_state = {
        key: value
        for key, value in state_dict.items()
        if key in module.state_dict()
        and tuple(module.state_dict()[key].shape) == tuple(value.shape)
        and key not in unexpected
    }
    missing, load_unexpected = module.load_state_dict(exact_state, strict=False)
    return list(missing), list(unexpected) + list(load_unexpected), assigned


def _select_encoder_state(
    vision_encoder: nn.Module,
    raw_state: Dict[str, torch.Tensor],
) -> tuple[str, Dict[str, torch.Tensor]]:
    best_name = ""
    best_state: Dict[str, torch.Tensor] = {}
    best_score = (0, 0)
    for name, candidate in _candidate_encoder_states(vision_encoder, raw_state):
        score = _shape_match_score(vision_encoder, candidate)
        if score > best_score:
            best_name = name
            best_state = candidate
            best_score = score
    if best_score[0] <= 0:
        lora_keys = [key for key in raw_state if "lora" in str(key).lower()]
        source_examples = _format_state_examples(raw_state)
        target_examples = _format_state_examples(vision_encoder.state_dict())
        hint = ""
        if lora_keys:
            hint = (
                " The checkpoint looks like a LoRA-only adapter. Merge the LoRA adapter "
                "into the base DINOv3 weights first, or provide a scripted DINOv3 checkpoint "
                "that includes encoder blocks."
            )
        raise ValueError(
            "Unable to match any DINOv3 visual encoder weights by shape."
            f"{hint} source_key_examples=[{source_examples}] target_key_examples=[{target_examples}]"
        )
    return best_name, best_state


def load_dinov3_vision_checkpoint(
    vision_encoder: nn.Module,
    checkpoint_path: str | Path,
) -> str:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 visual checkpoint not found: {path}")

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    raw_state = _tensor_state_dict(state)
    if not raw_state:
        raise ValueError(f"Unable to extract DINOv3 visual encoder weights from checkpoint: {path}")

    selected_name, encoder_state = _select_encoder_state(vision_encoder, raw_state)
    missing, unexpected, zero_shape_assigned = _load_state_dict_allow_zero_shape(vision_encoder, encoder_state)
    print(
        "[dinov3-checkpoint] "
        f"selected_state={selected_name} tensors={len(encoder_state)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"zero_shape_assigned={zero_shape_assigned} path={path}",
        flush=True,
    )
    return selected_name


def apply_scripted_dinov3_processor_stats(image_processor: Any) -> None:
    if image_processor is None:
        return
    image_processor.image_mean = [0.5, 0.5, 0.5]
    image_processor.image_std = [1.0, 1.0, 1.0]
    print("[dinov3-checkpoint] using scripted DINOv3 normalization: mean=0.5 std=1.0", flush=True)
