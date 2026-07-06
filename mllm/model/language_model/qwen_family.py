def _norm_text(value) -> str:
    return str(value or "").lower().replace("-", "_").replace(".", "_")


def qwen_family_from_text(*values) -> str | None:
    text = " ".join(_norm_text(value) for value in values if value is not None)
    compact = text.replace("_", "")
    if "qwen35moe" in compact or "qwen3_5_moe" in text or "qwen3_5moe" in text:
        return "qwen3_5_moe"
    if "qwen35" in compact or "qwen3_5" in text:
        return "qwen3_5"
    if "qwen3moe" in compact or "qwen3_moe" in text:
        return "qwen3_moe"
    if "qwen3" in compact:
        return "qwen3"
    if "qwen2" in compact:
        return "qwen2"
    return None


def qwen_family_from_config(config) -> str | None:
    if config is None:
        return None
    architectures = getattr(config, "architectures", None) or []
    return qwen_family_from_text(
        getattr(config, "model_type", ""),
        config.__class__.__name__,
        *architectures,
    )


def qwen_family_from_config_dict(config_dict: dict | None, metadata: dict | None = None) -> str | None:
    config_dict = config_dict or {}
    metadata = metadata or {}
    return qwen_family_from_text(
        config_dict.get("model_type"),
        metadata.get("model_type"),
        *(config_dict.get("architectures") or []),
        *(metadata.get("architectures") or []),
    )


def is_qwen3_or_newer_family(family: str | None) -> bool:
    return family in {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}


def qwen_family_architecture(family: str) -> str:
    if family == "qwen3_5":
        return "Qwen3_5MultimodalForCausalLM"
    if family == "qwen3_5_moe":
        return "Qwen3_5MoeMultimodalForCausalLM"
    if family == "qwen3_moe":
        return "Qwen3MoeMultimodalForCausalLM"
    if family == "qwen3":
        return "Qwen3MultimodalForCausalLM"
    if family == "qwen2":
        return "Qwen2MultimodalForCausalLM"
    raise ValueError(f"Unsupported Qwen family: {family}")


def qwen_text_architecture(family: str) -> str:
    if family == "qwen3_5":
        return "Qwen3_5ForCausalLM"
    if family == "qwen3_5_moe":
        return "Qwen3_5MoeForCausalLM"
    if family == "qwen3_moe":
        return "Qwen3MoeForCausalLM"
    if family == "qwen3":
        return "Qwen3ForCausalLM"
    if family == "qwen2":
        return "Qwen2ForCausalLM"
    raise ValueError(f"Unsupported Qwen family: {family}")


def qwen_multimodal_config_class(family: str):
    if family == "qwen3":
        from mllm.model.language_model.llava_qwen3 import Qwen3MultimodalConfig

        return Qwen3MultimodalConfig
    if family == "qwen3_moe":
        from mllm.model.language_model.llava_qwen3_5 import Qwen3MoeMultimodalConfig

        return Qwen3MoeMultimodalConfig
    if family == "qwen3_5":
        from mllm.model.language_model.llava_qwen3_5 import Qwen3_5MultimodalConfig

        return Qwen3_5MultimodalConfig
    if family == "qwen3_5_moe":
        from mllm.model.language_model.llava_qwen3_5 import Qwen3_5MoeMultimodalConfig

        return Qwen3_5MoeMultimodalConfig
    if family == "qwen2":
        from mllm.model.language_model.llava_qwen import Qwen2MultimodalConfig

        return Qwen2MultimodalConfig
    raise ValueError(f"Unsupported Qwen family: {family}")


def qwen_multimodal_model_class(family: str):
    if family == "qwen3":
        from mllm.model.language_model.llava_qwen3 import Qwen3MultimodalForCausalLM

        return Qwen3MultimodalForCausalLM
    if family == "qwen3_moe":
        from mllm.model.language_model.llava_qwen3_5 import Qwen3MoeMultimodalForCausalLM

        return Qwen3MoeMultimodalForCausalLM
    if family == "qwen3_5":
        from mllm.model.language_model.llava_qwen3_5 import Qwen3_5MultimodalForCausalLM

        return Qwen3_5MultimodalForCausalLM
    if family == "qwen3_5_moe":
        from mllm.model.language_model.llava_qwen3_5 import Qwen3_5MoeMultimodalForCausalLM

        return Qwen3_5MoeMultimodalForCausalLM
    if family == "qwen2":
        from mllm.model.language_model.llava_qwen import Qwen2MultimodalForCausalLM

        return Qwen2MultimodalForCausalLM
    raise ValueError(f"Unsupported Qwen family: {family}")


def as_qwen_multimodal_config(config, family: str | None = None):
    family = family or qwen_family_from_config(config)
    if family is None:
        raise ValueError("Cannot infer Qwen family from config")
    config_class = qwen_multimodal_config_class(family)
    if isinstance(config, config_class):
        return config
    config_dict = config.to_dict()
    config_dict.pop("model_type", None)
    return config_class(**config_dict)
