__all__ = [
    "LlavaLlamaForCausalLM",
    "LlavaQwen2ForCausalLM",
    "LlavaQwen3ForCausalLM",
    "LlavaQwen3MoeForCausalLM",
    "LlavaQwen3_5ForCausalLM",
    "LlavaQwen3_5MoeForCausalLM",
    "Qwen2MultimodalForCausalLM",
    "Qwen3MultimodalForCausalLM",
    "Qwen3MoeMultimodalForCausalLM",
    "Qwen3_5MultimodalForCausalLM",
    "Qwen3_5MoeMultimodalForCausalLM",
]


def __getattr__(name):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
