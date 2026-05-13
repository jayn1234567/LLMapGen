import torch


def clear_deepstack_context(model):
    model._active_visual_pos_mask = None
    model._active_deepstack_visual_embeds = None


def _add_deepstack_visual_features(hidden_states, visual_pos_mask, visual_embeds, layer_idx):
    visual_pos_mask = visual_pos_mask.to(device=hidden_states.device, dtype=torch.bool)
    if visual_pos_mask.ndim == 1:
        visual_pos_mask = visual_pos_mask.unsqueeze(0)

    src = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
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
            hidden_states = _add_deepstack_visual_features(
                hidden_states, visual_pos_mask, ds_feat, layer_idx
            )
            return (hidden_states,) + output[1:]

        if not isinstance(output, torch.Tensor) or output.ndim < 3:
            return None
        return _add_deepstack_visual_features(output, visual_pos_mask, ds_feat, layer_idx)

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
