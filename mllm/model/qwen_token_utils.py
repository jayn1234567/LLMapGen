import json
import os

QWEN_BOS_TOKEN_ID = 151643
QWEN_PAD_TOKEN_ID = 151643
QWEN_EOS_TOKEN_ID = 151645
QWEN_GENERATION_EOS_TOKEN_IDS = [QWEN_EOS_TOKEN_ID, QWEN_PAD_TOKEN_ID]


def _as_lower(value):
    return str(value or "").lower()


def _looks_like_qwen(config=None, tokenizer=None, model_name_or_path=None):
    model_type = _as_lower(getattr(config, "model_type", ""))
    if "qwen" in model_type:
        return True

    for arch in getattr(config, "architectures", []) or []:
        if "qwen" in _as_lower(arch):
            return True

    if tokenizer is not None and "qwen" in tokenizer.__class__.__name__.lower():
        return True

    return "qwen" in _as_lower(model_name_or_path)


def _looks_like_qwen_config_file(model_name_or_path=None):
    if model_name_or_path is None:
        return False
    config_path = os.path.join(str(model_name_or_path), "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except Exception:
        return False
    model_type = _as_lower(config_dict.get("model_type"))
    if "qwen" in model_type:
        return True
    return any("qwen" in _as_lower(arch) for arch in config_dict.get("architectures", []) or [])


def _token_from_id(tokenizer, token_id):
    if tokenizer is None:
        return None
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        return None
    if token is None or token == tokenizer.unk_token:
        return None
    return token


def _set_token_if_missing(tokenizer, attr_name, token_id):
    if tokenizer is None or getattr(tokenizer, f"{attr_name}_id", None) is not None:
        return
    token = _token_from_id(tokenizer, token_id)
    if token is not None:
        setattr(tokenizer, attr_name, token)


def sync_qwen_token_config(tokenizer=None, model=None, config=None, generation_config=None, model_name_or_path=None):
    """Keep Qwen tokenizer/model generation ids aligned with official checkpoints."""

    if config is None and model is not None:
        config = getattr(model, "config", None)
    if generation_config is None and model is not None:
        generation_config = getattr(model, "generation_config", None)

    if not _looks_like_qwen(config=config, tokenizer=tokenizer, model_name_or_path=model_name_or_path):
        return False

    _set_token_if_missing(tokenizer, "bos_token", QWEN_BOS_TOKEN_ID)
    _set_token_if_missing(tokenizer, "pad_token", QWEN_PAD_TOKEN_ID)
    _set_token_if_missing(tokenizer, "eos_token", QWEN_EOS_TOKEN_ID)

    if config is not None:
        if getattr(config, "bos_token_id", None) is None:
            config.bos_token_id = QWEN_BOS_TOKEN_ID
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = QWEN_PAD_TOKEN_ID
        if getattr(config, "eos_token_id", None) is None:
            config.eos_token_id = QWEN_EOS_TOKEN_ID

    if generation_config is not None:
        if getattr(generation_config, "bos_token_id", None) is None:
            generation_config.bos_token_id = QWEN_BOS_TOKEN_ID
        if getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = QWEN_PAD_TOKEN_ID
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None or eos_token_id == QWEN_EOS_TOKEN_ID:
            generation_config.eos_token_id = list(QWEN_GENERATION_EOS_TOKEN_IDS)

    return True


def qwen_tokenizer_kwargs(model_name_or_path=None, config=None):
    if _looks_like_qwen(config=config, model_name_or_path=model_name_or_path) or _looks_like_qwen_config_file(model_name_or_path):
        return {
            "fix_mistral_regex": True,
            # Some cloud Transformers/Qwen3-VL combinations still require the
            # remote tokenizer/config code path even when local SFT checkpoints
            # can be loaded by the supervised training entry.
            "trust_remote_code": True,
        }
    return {}


def normalize_qwen_config_dict(config_dict, generation_config_dict=None):
    model_type = _as_lower(config_dict.get("model_type")) if config_dict else ""
    architectures = config_dict.get("architectures", []) if config_dict else []
    if "qwen" not in model_type and not any("qwen" in _as_lower(arch) for arch in architectures):
        return config_dict, generation_config_dict

    if config_dict.get("bos_token_id") is None:
        config_dict["bos_token_id"] = QWEN_BOS_TOKEN_ID
    if config_dict.get("pad_token_id") is None:
        config_dict["pad_token_id"] = QWEN_PAD_TOKEN_ID
    if config_dict.get("eos_token_id") is None:
        config_dict["eos_token_id"] = QWEN_EOS_TOKEN_ID

    rope_scaling = config_dict.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        # Qwen3-VL text_config carries vision mRoPE metadata. The extracted
        # text-only Qwen3 LLM does not consume these fields, and recent
        # transformers versions warn about them under rope_type=default.
        rope_scaling.pop("mrope_interleaved", None)
        rope_scaling.pop("mrope_section", None)
        if not rope_scaling:
            config_dict.pop("rope_scaling", None)

    if generation_config_dict is not None:
        if generation_config_dict.get("bos_token_id") is None:
            generation_config_dict["bos_token_id"] = QWEN_BOS_TOKEN_ID
        if generation_config_dict.get("pad_token_id") is None:
            generation_config_dict["pad_token_id"] = QWEN_PAD_TOKEN_ID
        eos_token_id = generation_config_dict.get("eos_token_id")
        if eos_token_id is None or eos_token_id == QWEN_EOS_TOKEN_ID:
            generation_config_dict["eos_token_id"] = list(QWEN_GENERATION_EOS_TOKEN_IDS)

    return config_dict, generation_config_dict
