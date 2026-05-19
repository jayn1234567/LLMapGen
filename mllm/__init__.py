__all__ = [
    "LlavaLlamaForCausalLM",
    "LlavaQwen2ForCausalLM",
    "LlavaQwen3ForCausalLM",
    "Qwen2MultimodalForCausalLM",
    "Qwen3MultimodalForCausalLM",
]


def __getattr__(name):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
