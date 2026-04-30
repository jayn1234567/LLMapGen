
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def _deepstack_forward(self, inputs_embeds, attention_mask, position_ids,
                           past_key_values, visual_pos_mask, deepstack_visual_embeds,
                           use_cache, output_attentions, output_hidden_states, cache_position):
        """Use standard model forward with pre-hooks on first N layers for deepstack injection."""
        n_ds = len(deepstack_visual_embeds) if deepstack_visual_embeds else 0
        handles = []

        if n_ds > 0:
            def make_hook(layer_idx):
                def hook(module, args):
                    hs, *rest = args
                    ds_feat = deepstack_visual_embeds[layer_idx]
                    if ds_feat is not None:
                        try:
                            src = ds_feat.float()
                            if src.ndim == 2: src = src.unsqueeze(0)
                            B, S, D = hs.shape
                            _, N, _ = src.shape
                            mask_count = visual_pos_mask.sum().item() // B
                            if N == mask_count or N == S:
                                actual_N = min(N, mask_count)
                                scattered = torch.zeros(B, S, D, device=hs.device, dtype=src.dtype)
                                for b in range(B):
                                    pos = visual_pos_mask[b].nonzero(as_tuple=True)[0][:actual_N]
                                    scattered[b, pos] = src[min(b, src.shape[0] - 1)][:actual_N]
                                hs = hs + scattered
                        except (RuntimeError, IndexError, ValueError):
                            pass
                    return (hs,) + tuple(rest)
                return hook

            for i in range(n_ds):
                handles.append(self.model.layers[i].register_forward_pre_hook(make_hook(i)))

        outputs = self.model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, cache_position=cache_position,
        )

        for h in handles:
            h.remove()

        return outputs.last_hidden_state, outputs.hidden_states, outputs.attentions, outputs.past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        visual_pos_mask = None
        deepstack_visual_embeds = None

        if inputs_embeds is None:
            (
                input_ids, position_ids, attention_mask, past_key_values,
                inputs_embeds, labels, visual_pos_mask, deepstack_visual_embeds,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask,
                past_key_values, labels, images, image_sizes,
            )

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
                    hidden_states=all_hs, attentions=all_attn,
                )

        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs, position_ids, attention_mask, _,
                inputs_embeds, _, _, _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask,
                None, None, images, image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs,
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaQwen2ForCausalLM)
