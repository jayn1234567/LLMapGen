import torch
import torch.nn as nn
from copy import copy

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .deepstack_hooks import clear_deepstack_context, deepstack_decoder_forward


def _missing_qwen_import(family: str):
    raise ImportError(
        f"{family} support requires a Transformers build that exports the corresponding "
        "Qwen classes. Use the project fastvlm environment or upgrade Transformers."
    )


try:
    from transformers import Qwen3MoeConfig as _Qwen3MoeConfig
    from transformers import Qwen3MoeModel as _Qwen3MoeModel
    from transformers import Qwen3MoeForCausalLM as _Qwen3MoeForCausalLM
except ImportError:  # pragma: no cover - depends on installed Transformers
    _Qwen3MoeConfig = _Qwen3MoeModel = _Qwen3MoeForCausalLM = None

try:
    from transformers import Qwen3_5TextConfig as _Qwen3_5Config
    from transformers import Qwen3_5TextModel as _Qwen3_5Model
    from transformers import Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM
except ImportError:  # pragma: no cover - depends on installed Transformers
    _Qwen3_5Config = _Qwen3_5Model = _Qwen3_5ForCausalLM = None

try:
    from transformers import Qwen3_5MoeTextConfig as _Qwen3_5MoeConfig
    from transformers import Qwen3_5MoeTextModel as _Qwen3_5MoeModel
    from transformers import Qwen3_5MoeForCausalLM as _Qwen3_5MoeForCausalLM
except ImportError:  # pragma: no cover - depends on installed Transformers
    _Qwen3_5MoeConfig = _Qwen3_5MoeModel = _Qwen3_5MoeForCausalLM = None


def _qwen_get_model(self):
    return self.model


def _qwen_deepstack_forward(self, inputs_embeds, attention_mask, position_ids,
                            past_key_values, visual_pos_mask, deepstack_visual_embeds,
                            use_cache, output_attentions, output_hidden_states, cache_position):
    return deepstack_decoder_forward(
        self.model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        visual_pos_mask=visual_pos_mask,
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )


def _qwen_forward(self, input_ids=None, attention_mask=None, position_ids=None,
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
                position_ids = torch.arange(
                    0,
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ).unsqueeze(0)
                position_ids = position_ids.expand(inputs_embeds.shape[0], -1)
            if attention_mask is None:
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
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
                loss=loss,
                logits=logits,
                past_key_values=next_cache,
                hidden_states=all_hs,
                attentions=all_attn,
            )

    clear_deepstack_context(self)
    return super(type(self), self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


@torch.no_grad()
def _qwen_generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError
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
    layer_types = getattr(self.config, "layer_types", None) or []
    generation_config = getattr(self, "generation_config", None)
    original_generation_use_cache = getattr(generation_config, "use_cache", None) if generation_config is not None else None
    original_config_use_cache = getattr(self.config, "use_cache", None)
    has_linear_attention = any(str(layer_type) == "linear_attention" for layer_type in layer_types)
    if has_linear_attention:
        kwargs["use_cache"] = False
        self.config.use_cache = False
        if generation_config is not None:
            local_generation_config = copy(generation_config)
            local_generation_config.use_cache = False
            kwargs["generation_config"] = local_generation_config
        try:
            return _qwen_generate_without_cache(
                self,
                inputs,
                inputs_embeds,
                attention_mask,
                visual_pos_mask,
                deepstack_visual_embeds,
                **kwargs,
            )
        finally:
            self._generation_visual_pos_mask = None
            self._generation_deepstack_visual_embeds = None
            if generation_config is not None and original_generation_use_cache is not None:
                generation_config.use_cache = original_generation_use_cache
            if original_config_use_cache is not None:
                self.config.use_cache = original_config_use_cache
    try:
        return super(type(self), self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
    finally:
        self._generation_visual_pos_mask = None
        self._generation_deepstack_visual_embeds = None
        if generation_config is not None and original_generation_use_cache is not None:
            generation_config.use_cache = original_generation_use_cache
        if original_config_use_cache is not None:
            self.config.use_cache = original_config_use_cache


def _qwen_prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    images = kwargs.pop("images", None)
    image_sizes = kwargs.pop("image_sizes", None)
    inputs = super(type(self), self).prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    if images is not None:
        inputs["images"] = images
    if image_sizes is not None:
        inputs["image_sizes"] = image_sizes
    return inputs


def _as_eos_list(eos_token_id):
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, (list, tuple, set)):
        return [int(item) for item in eos_token_id]
    return [int(eos_token_id)]


def _qwen_generate_without_cache(self, input_ids, inputs_embeds, attention_mask,
                                 visual_pos_mask, deepstack_visual_embeds, **kwargs):
    max_new_tokens = int(kwargs.get("max_new_tokens", 20))
    do_sample = bool(kwargs.get("do_sample", False))
    temperature = float(kwargs.get("temperature", 1.0) or 1.0)
    generation_config = kwargs.get("generation_config", None) or getattr(self, "generation_config", None)
    eos_token_id = kwargs.get("eos_token_id", None)
    if eos_token_id is None and generation_config is not None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)
    eos_ids = set(_as_eos_list(eos_token_id))

    if attention_mask is None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    if input_ids is None:
        input_ids = torch.empty((inputs_embeds.shape[0], 0), dtype=torch.long, device=inputs_embeds.device)

    cur_embeds = inputs_embeds
    cur_attention_mask = attention_mask
    cur_visual_pos_mask = visual_pos_mask
    generated = []

    for _ in range(max_new_tokens):
        seq_len = cur_embeds.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=cur_embeds.device).unsqueeze(0)
        position_ids = position_ids.expand(cur_embeds.shape[0], -1)
        outputs = self.forward(
            inputs_embeds=cur_embeds,
            attention_mask=cur_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            visual_pos_mask=cur_visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = outputs.logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        next_embed = self.get_model().embed_tokens(next_token)
        cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones_like(next_token, dtype=cur_attention_mask.dtype)],
            dim=1,
        )
        if cur_visual_pos_mask is not None:
            false_mask = torch.zeros_like(next_token, dtype=torch.bool)
            cur_visual_pos_mask = torch.cat([cur_visual_pos_mask.to(torch.bool), false_mask], dim=1)
        if eos_ids and all(int(token.item()) in eos_ids for token in next_token.flatten()):
            break

    if generated:
        generated_ids = torch.cat(generated, dim=1)
        return torch.cat([input_ids.to(generated_ids.device), generated_ids], dim=1)
    return input_ids


def _attach_qwen_methods(cls):
    cls.get_model = _qwen_get_model
    cls._deepstack_forward = _qwen_deepstack_forward
    cls.forward = _qwen_forward
    cls.generate = _qwen_generate
    cls.prepare_inputs_for_generation = _qwen_prepare_inputs_for_generation
    cls.__abstractmethods__ = frozenset()
    return cls


if _Qwen3MoeConfig is not None:
    class Qwen3MoeMultimodalConfig(_Qwen3MoeConfig):
        model_type = "qwen3_moe"


    class LegacyLlavaQwen3MoeConfigWrapper(Qwen3MoeMultimodalConfig):
        model_type = "llava_qwen3_moe"


    class Qwen3MoeMultimodalModel(LlavaMetaModel, _Qwen3MoeModel):
        config_class = Qwen3MoeMultimodalConfig

        def __init__(self, config):
            super().__init__(config)


    @_attach_qwen_methods
    class Qwen3MoeMultimodalForCausalLM(_Qwen3MoeForCausalLM, LlavaMetaForCausalLM):
        config_class = Qwen3MoeMultimodalConfig

        def __init__(self, config):
            super(_Qwen3MoeForCausalLM, self).__init__(config)
            self.model = Qwen3MoeMultimodalModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()
else:
    class Qwen3MoeMultimodalConfig:
        model_type = "qwen3_moe"

        def __init__(self, *args, **kwargs):
            _missing_qwen_import("Qwen3-MoE")


    LegacyLlavaQwen3MoeConfigWrapper = Qwen3MoeMultimodalConfig

    class Qwen3MoeMultimodalForCausalLM:
        def __init__(self, *args, **kwargs):
            _missing_qwen_import("Qwen3-MoE")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _missing_qwen_import("Qwen3-MoE")


if _Qwen3_5Config is not None:
    class Qwen3_5MultimodalConfig(_Qwen3_5Config):
        model_type = "qwen3_5_text"


    class LegacyLlavaQwen3_5ConfigWrapper(Qwen3_5MultimodalConfig):
        model_type = "llava_qwen3_5"


    class Qwen3_5MultimodalModel(LlavaMetaModel, _Qwen3_5Model):
        config_class = Qwen3_5MultimodalConfig

        def __init__(self, config):
            super().__init__(config)


    @_attach_qwen_methods
    class Qwen3_5MultimodalForCausalLM(_Qwen3_5ForCausalLM, LlavaMetaForCausalLM):
        config_class = Qwen3_5MultimodalConfig

        def __init__(self, config):
            super(_Qwen3_5ForCausalLM, self).__init__(config)
            self.model = Qwen3_5MultimodalModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()
else:
    class Qwen3_5MultimodalConfig:
        model_type = "qwen3_5_text"

        def __init__(self, *args, **kwargs):
            _missing_qwen_import("Qwen3.5")


    LegacyLlavaQwen3_5ConfigWrapper = Qwen3_5MultimodalConfig

    class Qwen3_5MultimodalForCausalLM:
        def __init__(self, *args, **kwargs):
            _missing_qwen_import("Qwen3.5")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _missing_qwen_import("Qwen3.5")


if _Qwen3_5MoeConfig is not None:
    class Qwen3_5MoeMultimodalConfig(_Qwen3_5MoeConfig):
        model_type = "qwen3_5_moe_text"


    class LegacyLlavaQwen3_5MoeConfigWrapper(Qwen3_5MoeMultimodalConfig):
        model_type = "llava_qwen3_5_moe"


    class Qwen3_5MoeMultimodalModel(LlavaMetaModel, _Qwen3_5MoeModel):
        config_class = Qwen3_5MoeMultimodalConfig

        def __init__(self, config):
            super().__init__(config)


    @_attach_qwen_methods
    class Qwen3_5MoeMultimodalForCausalLM(_Qwen3_5MoeForCausalLM, LlavaMetaForCausalLM):
        config_class = Qwen3_5MoeMultimodalConfig

        def __init__(self, config):
            super(_Qwen3_5MoeForCausalLM, self).__init__(config)
            self.model = Qwen3_5MoeMultimodalModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()
else:
    class Qwen3_5MoeMultimodalConfig:
        model_type = "qwen3_5_moe_text"

        def __init__(self, *args, **kwargs):
            _missing_qwen_import("Qwen3.5-MoE")


    LegacyLlavaQwen3_5MoeConfigWrapper = Qwen3_5MoeMultimodalConfig

    class Qwen3_5MoeMultimodalForCausalLM:
        def __init__(self, *args, **kwargs):
            _missing_qwen_import("Qwen3.5-MoE")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _missing_qwen_import("Qwen3.5-MoE")


LlavaQwen3MoeForCausalLM = Qwen3MoeMultimodalForCausalLM
LlavaQwen3_5ForCausalLM = Qwen3_5MultimodalForCausalLM
LlavaQwen3_5MoeForCausalLM = Qwen3_5MoeMultimodalForCausalLM

if _Qwen3MoeConfig is not None:
    AutoConfig.register("llava_qwen3_moe", LegacyLlavaQwen3MoeConfigWrapper)
    AutoModelForCausalLM.register(Qwen3MoeMultimodalConfig, Qwen3MoeMultimodalForCausalLM)
if _Qwen3_5Config is not None:
    AutoConfig.register("llava_qwen3_5", LegacyLlavaQwen3_5ConfigWrapper)
    AutoModelForCausalLM.register(Qwen3_5MultimodalConfig, Qwen3_5MultimodalForCausalLM)
if _Qwen3_5MoeConfig is not None:
    AutoConfig.register("llava_qwen3_5_moe", LegacyLlavaQwen3_5MoeConfigWrapper)
    AutoModelForCausalLM.register(Qwen3_5MoeMultimodalConfig, Qwen3_5MoeMultimodalForCausalLM)
