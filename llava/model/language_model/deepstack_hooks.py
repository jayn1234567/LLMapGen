import torch


def clear_deepstack_context(model):
    model._active_visual_pos_mask = None
    model._active_deepstack_visual_embeds = None


def _make_deepstack_hook(model, layer_idx):
    def hook(module, args):
        deepstack_visual_embeds = getattr(model, "_active_deepstack_visual_embeds", None)
        visual_pos_mask = getattr(model, "_active_visual_pos_mask", None)
        if deepstack_visual_embeds is None or visual_pos_mask is None:
            return None
        if layer_idx >= len(deepstack_visual_embeds):
            return None

        hs, *rest = args
        visual_pos_mask = visual_pos_mask.to(device=hs.device, dtype=torch.bool)
        if visual_pos_mask.ndim == 1:
            visual_pos_mask = visual_pos_mask.unsqueeze(0)
        ds_feat = deepstack_visual_embeds[layer_idx]
        if ds_feat is None:
            return None

        src = ds_feat.to(device=hs.device, dtype=hs.dtype)
        if src.ndim == 2:
            src = src.unsqueeze(0)
        bsz, seq_len, hidden_dim = hs.shape
        src_bsz, num_visual_tokens, src_dim = src.shape
        if src_dim != hidden_dim:
            raise RuntimeError(
                f"DeepStack dim mismatch at LLM layer {layer_idx}: "
                f"visual dim {src_dim}, hidden dim {hidden_dim}"
            )

        scattered = torch.zeros(bsz, seq_len, hidden_dim, device=hs.device, dtype=hs.dtype)
        for b in range(bsz):
            pos = visual_pos_mask[b].nonzero(as_tuple=True)[0]
            if pos.numel() != num_visual_tokens:
                raise RuntimeError(
                    f"DeepStack token count mismatch at LLM layer {layer_idx}: "
                    f"visual tokens {num_visual_tokens}, image positions {pos.numel()}"
                )
            scattered[b, pos] = src[min(b, src_bsz - 1)]
        return (hs + scattered,) + tuple(rest)

    return hook


def ensure_deepstack_hooks(model):
    if getattr(model, "_deepstack_hooks_installed", False):
        return
    layers = getattr(model.get_model(), "layers", None)
    if layers is None:
        raise RuntimeError("DeepStack injection requires the language model to expose .layers")
    model._deepstack_hook_handles = [
        layer.register_forward_pre_hook(_make_deepstack_hook(model, layer_idx))
        for layer_idx, layer in enumerate(layers)
    ]
    model._deepstack_hooks_installed = True
    clear_deepstack_context(model)


def set_deepstack_context(model, visual_pos_mask, deepstack_visual_embeds):
    ensure_deepstack_hooks(model)
    model._active_visual_pos_mask = visual_pos_mask
    model._active_deepstack_visual_embeds = deepstack_visual_embeds
