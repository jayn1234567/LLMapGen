from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


def _get_qwen3_classes():
    try:
        from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
        return Qwen3Config, Qwen3Model, Qwen3ForCausalLM
    except ImportError:
        pass
    try:
        from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
        return Qwen2Config, Qwen2Model, Qwen2ForCausalLM
    except ImportError:
        raise ImportError(
            "Neither Qwen3ForCausalLM nor Qwen2ForCausalLM found. "
            "Please upgrade transformers to >= 4.51.0 for Qwen3, or >= 4.46.0 for Qwen2."
        )


_Qwen3Config, _Qwen3Model, _Qwen3ForCausalLM = _get_qwen3_classes()


class LlavaQwen3ConfigWrapper(_Qwen3Config):
    model_type = "llava_qwen3"


class LlavaQwen3Model(LlavaMetaModel, _Qwen3Model):
    config_class = LlavaQwen3ConfigWrapper

    def __init__(self, config):
        super().__init__(config)


class LlavaQwen3ForCausalLM(_Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen3ConfigWrapper

    def __init__(self, config):
        super(_Qwen3ForCausalLM, self).__init__(config)
        self.model = LlavaQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def _deepstack_forward(self, inputs_embeds, attention_mask, position_ids,
                           past_key_values, visual_pos_mask, deepstack_visual_embeds,
                           use_cache, output_attentions, output_hidden_states, cache_position):
        n_ds = len(deepstack_visual_embeds) if deepstack_visual_embeds else 0
        handles = []

        if n_ds > 0:
            def make_hook(layer_idx):
                def hook(module, args):
                    hs, *rest = args
                    ds_feat = deepstack_visual_embeds[layer_idx]
                    if ds_feat is not None:
                        src = ds_feat.to(device=hs.device, dtype=hs.dtype)
                        if src.ndim == 2:
                            src = src.unsqueeze(0)
                        B, S, D = hs.shape
                        src_bsz, N, src_dim = src.shape
                        if src_dim != D:
                            raise RuntimeError(
                                f"DeepStack dim mismatch at LLM layer {layer_idx}: "
                                f"visual dim {src_dim}, hidden dim {D}"
                            )
                        scattered = torch.zeros(B, S, D, device=hs.device, dtype=hs.dtype)
                        for b in range(B):
                            pos = visual_pos_mask[b].nonzero(as_tuple=True)[0]
                            if pos.numel() != N:
                                raise RuntimeError(
                                    f"DeepStack token count mismatch at LLM layer {layer_idx}: "
                                    f"visual tokens {N}, image positions {pos.numel()}"
                                )
                            scattered[b, pos] = src[min(b, src_bsz - 1)]
                        hs = hs + scattered
                    return (hs,) + tuple(rest)
                return hook

            for i in range(n_ds):
                handles.append(self.model.layers[i].register_forward_pre_hook(make_hook(i)))

        try:
            outputs = self.model(
                input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, cache_position=cache_position,
            )
        finally:
            for h in handles:
                h.remove()

        return outputs.last_hidden_state, outputs.hidden_states, outputs.attentions, outputs.past_key_values

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, images=None,
                image_sizes=None, return_dict=None, cache_position=None,
                visual_pos_mask=None, deepstack_visual_embeds=None):

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values,
             inputs_embeds, labels, visual_pos_mask, deepstack_visual_embeds,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask,
                past_key_values, labels, images, image_sizes,
            )
        else:
            if visual_pos_mask is None:
                visual_pos_mask = getattr(self, "_generation_visual_pos_mask", None)
            if deepstack_visual_embeds is None:
                deepstack_visual_embeds = getattr(self, "_generation_deepstack_visual_embeds", None)
            if past_key_values is not None:
                visual_pos_mask = None
                deepstack_visual_embeds = None

        if deepstack_visual_embeds is not None and visual_pos_mask is not None and inputs_embeds is not None:
            if inputs_embeds.ndim >= 3:
                if position_ids is None:
                    position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long,
                                                device=inputs_embeds.device).unsqueeze(0)
                    position_ids = position_ids.expand(inputs_embeds.shape[0], -1)
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                                device=inputs_embeds.device)
                hidden_states, all_hs, all_attn, next_cache = self._deepstack_forward(
                    inputs_embeds, attention_mask, position_ids, past_key_values,
                    visual_pos_mask, deepstack_visual_embeds,
                    use_cache, output_attentions, output_hidden_states, cache_position,
                )
                logits = self.lm_head(hidden_states)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.CrossEntropyLoss()(
                        shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                return CausalLMOutputWithPast(
                    loss=loss, logits=logits, past_key_values=next_cache,
                    hidden_states=all_hs, attentions=all_attn)

        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict)

    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs: raise NotImplementedError
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, visual_pos_mask, deepstack_visual_embeds,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_pos_mask = None
            deepstack_visual_embeds = None
        self._generation_visual_pos_mask = visual_pos_mask
        self._generation_deepstack_visual_embeds = deepstack_visual_embeds
        try:
            return super().generate(position_ids=position_ids, attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds, **kwargs)
        finally:
            self._generation_visual_pos_mask = None
            self._generation_deepstack_visual_embeds = None

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None: inputs['images'] = images
        if image_sizes is not None: inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_qwen3", LlavaQwen3ConfigWrapper)
AutoModelForCausalLM.register(LlavaQwen3ConfigWrapper, LlavaQwen3ForCausalLM)
