import importlib

import torch


def _get_qwen_modeling_module(model):
    for cls in type(model).mro():
        module_name = getattr(cls, "__module__", "")
        if module_name.startswith("transformers.models.qwen"):
            return importlib.import_module(module_name)
    raise RuntimeError(f"Unsupported Qwen model class for DeepStack forward: {type(model).__name__}")


def clear_deepstack_context(model):
    model._active_visual_pos_mask = None
    model._active_deepstack_visual_embeds = None


def densify_deepstack_visual_embeds(visual_pos_mask, deepstack_visual_embeds, reference_embeds):
    """Build full-sequence residual tensors before entering checkpointed LLM layers."""
    if visual_pos_mask is None or deepstack_visual_embeds is None:
        return None

    mask = visual_pos_mask.to(device=reference_embeds.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[:2] != reference_embeds.shape[:2]:
        raise RuntimeError(
            "DeepStack visual mask shape mismatch: "
            f"mask {tuple(mask.shape)}, inputs {tuple(reference_embeds.shape[:2])}"
        )

    dense_embeds = []
    for layer_idx, visual_embeds in enumerate(deepstack_visual_embeds):
        if visual_embeds is None:
            dense_embeds.append(None)
            continue

        src = visual_embeds.to(device=reference_embeds.device, dtype=reference_embeds.dtype)
        if src.shape[-1] != reference_embeds.shape[-1]:
            raise RuntimeError(
                f"DeepStack dim mismatch at LLM layer {layer_idx}: "
                f"visual dim {src.shape[-1]}, hidden dim {reference_embeds.shape[-1]}"
            )

        dense = reference_embeds.new_zeros(reference_embeds.shape)
        if src.ndim == 2:
            num_visual_tokens = src.shape[0]
            if int(mask.sum().item()) != num_visual_tokens:
                raise RuntimeError(
                    f"DeepStack token count mismatch at LLM layer {layer_idx}: "
                    f"visual tokens {num_visual_tokens}, image positions {int(mask.sum().item())}"
                )
            dense[mask] = src
        elif src.ndim == 3:
            bsz = mask.shape[0]
            src_bsz, num_visual_tokens, _ = src.shape
            for b in range(bsz):
                pos = mask[b].nonzero(as_tuple=True)[0]
                if pos.numel() != num_visual_tokens:
                    raise RuntimeError(
                        f"DeepStack token count mismatch at LLM layer {layer_idx}: "
                        f"visual tokens {num_visual_tokens}, image positions {pos.numel()}"
                    )
                dense[b, pos] = src[min(b, src_bsz - 1)]
        else:
            raise RuntimeError(
                f"DeepStack expected visual embeddings with 2 or 3 dims at LLM layer {layer_idx}, "
                f"got shape {tuple(src.shape)}"
            )
        dense_embeds.append(dense)

    return dense_embeds


def add_deepstack_visual_features(hidden_states, visual_pos_mask, visual_embeds, layer_idx):
    src = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if src.ndim == 3 and src.shape == hidden_states.shape:
        return hidden_states + src

    visual_pos_mask = visual_pos_mask.to(device=hidden_states.device, dtype=torch.bool)
    if visual_pos_mask.ndim == 1:
        visual_pos_mask = visual_pos_mask.unsqueeze(0)

    updated = hidden_states.clone()
    if src.ndim == 2:
        num_visual_tokens, src_dim = src.shape
        if src_dim != hidden_states.shape[-1]:
            raise RuntimeError(
                f"DeepStack dim mismatch at LLM layer {layer_idx}: "
                f"visual dim {src_dim}, hidden dim {hidden_states.shape[-1]}"
            )
        if visual_pos_mask.sum().item() != num_visual_tokens:
            raise RuntimeError(
                f"DeepStack token count mismatch at LLM layer {layer_idx}: "
                f"visual tokens {num_visual_tokens}, image positions {visual_pos_mask.sum().item()}"
            )
        updated[visual_pos_mask] = updated[visual_pos_mask] + src
        return updated

    if src.ndim != 3:
        raise RuntimeError(
            f"DeepStack expected visual embeddings with 2 or 3 dims at LLM layer {layer_idx}, "
            f"got shape {tuple(src.shape)}"
        )

    bsz, _, hidden_dim = hidden_states.shape
    src_bsz, num_visual_tokens, src_dim = src.shape
    if src_dim != hidden_dim:
        raise RuntimeError(
            f"DeepStack dim mismatch at LLM layer {layer_idx}: "
            f"visual dim {src_dim}, hidden dim {hidden_dim}"
        )

    for b in range(bsz):
        pos = visual_pos_mask[b].nonzero(as_tuple=True)[0]
        if pos.numel() != num_visual_tokens:
            raise RuntimeError(
                f"DeepStack token count mismatch at LLM layer {layer_idx}: "
                f"visual tokens {num_visual_tokens}, image positions {pos.numel()}"
            )
        updated[b, pos] = updated[b, pos] + src[min(b, src_bsz - 1)]
    return updated


def _make_deepstack_hook(model, layer_idx):
    def hook(module, args, output):
        deepstack_visual_embeds = getattr(model, "_active_deepstack_visual_embeds", None)
        visual_pos_mask = getattr(model, "_active_visual_pos_mask", None)
        if deepstack_visual_embeds is None or visual_pos_mask is None:
            return None
        if layer_idx >= len(deepstack_visual_embeds):
            return None

        ds_feat = deepstack_visual_embeds[layer_idx]
        if ds_feat is None:
            return None

        if isinstance(output, tuple):
            hidden_states = output[0]
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 3:
                return None
            hidden_states = add_deepstack_visual_features(
                hidden_states, visual_pos_mask, ds_feat, layer_idx
            )
            return (hidden_states,) + output[1:]

        if not isinstance(output, torch.Tensor) or output.ndim < 3:
            return None
        return add_deepstack_visual_features(output, visual_pos_mask, ds_feat, layer_idx)

    return hook


def ensure_deepstack_hooks(model):
    if getattr(model, "_deepstack_hooks_installed", False):
        return
    layers = getattr(model.get_model(), "layers", None)
    if layers is None:
        raise RuntimeError("DeepStack injection requires the language model to expose .layers")
    model._deepstack_hook_handles = [
        layer.register_forward_hook(_make_deepstack_hook(model, layer_idx))
        for layer_idx, layer in enumerate(layers)
    ]
    model._deepstack_hooks_installed = True
    clear_deepstack_context(model)


def set_deepstack_context(model, visual_pos_mask, deepstack_visual_embeds):
    ensure_deepstack_hooks(model)
    model._active_visual_pos_mask = visual_pos_mask
    model._active_deepstack_visual_embeds = deepstack_visual_embeds


def deepstack_decoder_forward(
    model,
    inputs_embeds,
    attention_mask,
    position_ids,
    past_key_values,
    visual_pos_mask,
    deepstack_visual_embeds,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    cache_position=None,
):
    """Run Qwen decoder layers with explicit post-layer DeepStack residuals."""
    if inputs_embeds is None:
        raise ValueError("DeepStack forward requires inputs_embeds")

    modeling_module = _get_qwen_modeling_module(model)
    DynamicCache = getattr(modeling_module, "DynamicCache")
    create_causal_mask = getattr(modeling_module, "create_causal_mask")
    create_sliding_window_causal_mask = getattr(modeling_module, "create_sliding_window_causal_mask")

    use_cache = getattr(model.config, "use_cache", False) if use_cache is None else use_cache
    output_hidden_states = (
        getattr(model.config, "output_hidden_states", False)
        if output_hidden_states is None else output_hidden_states
    )
    output_attentions = (
        getattr(model.config, "output_attentions", False)
        if output_attentions is None else output_attentions
    )

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=model.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if cache_position is not None:
            mask_kwargs["cache_position"] = cache_position
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if getattr(model, "has_sliding_layers", False):
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    dense_deepstack_visual_embeds = densify_deepstack_visual_embeds(
        visual_pos_mask, deepstack_visual_embeds, inputs_embeds
    )
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    layer_types = getattr(model.config, "layer_types", None)
    for layer_idx, decoder_layer in enumerate(model.layers[: model.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_kwargs = {
            "attention_mask": causal_mask_mapping[layer_types[layer_idx]] if layer_types else causal_mask_mapping["full_attention"],
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if cache_position is not None:
            layer_kwargs["cache_position"] = cache_position

        layer_outputs = decoder_layer(hidden_states, **layer_kwargs)
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_attentions += (layer_outputs[1],)
        else:
            hidden_states = layer_outputs

        if dense_deepstack_visual_embeds is not None and layer_idx < len(dense_deepstack_visual_embeds):
            ds_feat = dense_deepstack_visual_embeds[layer_idx]
            if ds_feat is not None:
                hidden_states = add_deepstack_visual_features(
                    hidden_states, visual_pos_mask, ds_feat, layer_idx
                )

    hidden_states = model.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return hidden_states, all_hidden_states, all_attentions, past_key_values if use_cache else None
